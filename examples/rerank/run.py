from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "data" / "cases.json"
SOURCES_PATH = ROOT / "data" / "sources.json"
MODEL = "Qwen/Qwen3-Reranker-4B"
ARTIFACT_NAMES = {
    "sec_filing_amendment": "sec-restatement",
    "cms_lower_limb_orthosis": "cms-orthosis-documentation",
    "ntsb_detector_alert": "ntsb-bearing-alert",
    "scotus_two_contracts": "supreme-court-arbitration",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    return value


def load_and_verify_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    cases = read_json(CASES_PATH)
    sources = read_json(SOURCES_PATH)
    policy = cases.get("integrity_policy", {})
    if policy.get("synthetic_or_paraphrased_evidence") is not False:
        raise ValueError("Cases must reject synthetic or paraphrased evidence")
    if sources.get("synthetic_or_paraphrased_evidence") is not False:
        raise ValueError("Source manifest must reject synthetic evidence")
    if cases.get("model") != MODEL:
        raise ValueError(f"Expected model {MODEL}")
    if set(cases["cases"]) != set(ARTIFACT_NAMES):
        raise ValueError("Case set changed")

    known_sources = set(sources["sources"])
    for case_id, case in cases["cases"].items():
        if not case["query_provenance"].startswith("authored evaluation query"):
            raise ValueError(f"Missing query provenance for {case_id}")
        candidate_ids: set[str] = set()
        for candidate in case["candidates"]:
            candidate_id = candidate["id"]
            if candidate_id in candidate_ids:
                raise ValueError(f"Duplicate candidate {candidate_id}")
            candidate_ids.add(candidate_id)
            if candidate["source_id"] not in known_sources:
                raise ValueError(f"Unknown source for {candidate_id}")
            actual = sha256_bytes(candidate["text"].encode("utf-8"))
            if actual != candidate["sha256"]:
                raise ValueError(f"Source excerpt changed for {candidate_id}")
        if case["expected_top_candidate_id"] not in candidate_ids:
            raise ValueError(f"Expected top candidate missing for {case_id}")
    return cases, sources


def build_audit_envelope(case_id: str, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "SIEClient.score",
        "endpoint": f"/v1/score/{MODEL}",
        "model": MODEL,
        "query": {
            "id": f"{case_id}-query",
            "text": case["query"],
        },
        "items": [
            {
                "id": candidate["id"],
                "text": candidate["text"],
                "source_id": candidate["source_id"],
                "source_excerpt_sha256": candidate["sha256"],
            }
            for candidate in case["candidates"]
        ],
        "wait_for_capacity": True,
        "provision_timeout_s": 900,
    }


def validate_response(
    case_id: str,
    case: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    if response.get("model") != MODEL:
        raise ValueError(f"Unexpected model in response for {case_id}")
    if response.get("query_id") != f"{case_id}-query":
        raise ValueError(f"Unexpected query ID for {case_id}")
    score_rows = response.get("scores")
    if not isinstance(score_rows, list):
        raise TypeError(f"Missing score list for {case_id}")
    scores: list[dict[Any, Any]] = []
    for index, row in enumerate(score_rows):
        if not isinstance(row, dict):
            raise TypeError(f"Invalid score row at index {index}")
        scores.append(row)

    expected_ids = {candidate["id"] for candidate in case["candidates"]}
    observed_ids = {row.get("item_id") for row in scores}
    if observed_ids != expected_ids or len(scores) != len(expected_ids):
        raise ValueError(f"Candidate coverage changed for {case_id}")
    ranks: list[int] = []
    for row in scores:
        rank = row.get("rank")
        if type(rank) is not int:
            raise ValueError(f"Invalid rank for {row.get('item_id')}")
        ranks.append(rank)
    ranks.sort()
    if ranks != list(range(len(scores))):
        raise ValueError(f"Ranks are incomplete for {case_id}")
    for row in scores:
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Invalid score for {row.get('item_id')}")
    top = min(scores, key=lambda row: row["rank"])
    expected_top = case["expected_top_candidate_id"]
    if top["item_id"] != expected_top:
        raise ValueError(f"{case_id}: expected {expected_top}, received {top['item_id']}")
    return {
        "passed": True,
        "expected_top_candidate_id": expected_top,
        "observed_top_candidate_id": top["item_id"],
        "candidate_count": len(scores),
    }


def run_cases(selected_case: str | None) -> dict[str, Any]:
    from sie_sdk import Item, SIEClient

    cases, _ = load_and_verify_inputs()
    base_url = os.environ.get("SIE_BASE_URL", "http://127.0.0.1:8080")
    api_key = os.environ.get("SIE_API_KEY") or None
    client = SIEClient(base_url, api_key=api_key, timeout_s=900)
    results: dict[str, Any] = {}

    for case_id, case in cases["cases"].items():
        if selected_case and case_id != selected_case:
            continue
        response = client.score(
            MODEL,
            Item(id=f"{case_id}-query", text=case["query"]),
            [Item(id=candidate["id"], text=candidate["text"]) for candidate in case["candidates"]],
            wait_for_capacity=True,
            provision_timeout_s=900,
        )
        raw = to_jsonable(response)
        evaluation = validate_response(case_id, case, raw)
        results[case_id] = {
            "audit_envelope": build_audit_envelope(case_id, case),
            "raw_response": raw,
            "evaluation": evaluation,
        }
    if not results:
        raise ValueError(f"Unknown case: {selected_case}")
    return {
        "completed_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "endpoint": base_url,
        "model": MODEL,
        "cases": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank exact primary-source passages with public SIE")
    parser.add_argument("--case", choices=sorted(ARTIFACT_NAMES))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cases(args.case)
    if args.output:
        write_json(args.output, result)
        print(f"Wrote {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
