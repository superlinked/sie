from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "data" / "cases.json"
SOURCES_PATH = ROOT / "data" / "sources.json"
MODEL = "urchade/gliner_multi-v2.1"
ANCHOR_FIELDS = ("text", "label", "start", "end")
ARTIFACT_NAMES = {
    "sec_filing_amendment": "sec-restatement",
    "cms_lower_limb_orthosis": "cms-orthosis-documentation",
    "ntsb_detector_alert": "ntsb-bearing-alert",
    "scotus_two_contracts": "supreme-court-caption",
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


def anchor_key(value: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        value["text"],
        value["label"],
        value["start"],
        value["end"],
    )


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
        source = case["source"]
        if source["source_id"] not in known_sources:
            raise ValueError(f"Unknown source for {case_id}")
        if source["text"] != case["text"]:
            raise ValueError(f"Source text changed for {case_id}")
        actual = sha256_bytes(case["text"].encode("utf-8"))
        if actual != source["sha256"]:
            raise ValueError(f"Source excerpt changed for {case_id}")
        labels = case.get("labels")
        if not isinstance(labels, list) or not labels or len(set(labels)) != len(labels):
            raise ValueError(f"Invalid labels for {case_id}")
        required_anchors = case.get("required_anchors")
        if not isinstance(required_anchors, list) or not required_anchors:
            raise ValueError(f"Missing required anchors for {case_id}")
        seen_anchors: set[tuple[str, str, int, int]] = set()
        for index, anchor in enumerate(required_anchors):
            if not isinstance(anchor, dict) or set(anchor) != set(ANCHOR_FIELDS):
                raise ValueError(f"Invalid required anchor at index {index} for {case_id}")
            anchor_data = cast(dict[str, Any], anchor)
            if not isinstance(anchor_data["text"], str) or not anchor_data["text"]:
                raise ValueError(f"Invalid required anchor text at index {index} for {case_id}")
            if anchor_data["label"] not in labels:
                raise ValueError(f"Invalid required anchor label at index {index} for {case_id}")
            start = anchor_data["start"]
            end = anchor_data["end"]
            if type(start) is not int or type(end) is not int:
                raise ValueError(f"Invalid required anchor offsets at index {index} for {case_id}")
            if start < 0 or end <= start or end > len(case["text"]):
                raise ValueError(f"Out-of-range required anchor at index {index} for {case_id}")
            if case["text"][start:end] != anchor_data["text"]:
                raise ValueError(f"Required anchor text mismatch at index {index} for {case_id}")
            key = anchor_key(anchor_data)
            if key in seen_anchors:
                raise ValueError(f"Duplicate required anchor at index {index} for {case_id}")
            seen_anchors.add(key)
    return cases, sources


def build_audit_envelope(case_id: str, case: dict[str, Any]) -> dict[str, Any]:
    source = case["source"]
    return {
        "method": "SIEClient.extract",
        "endpoint": f"/v1/extract/{MODEL}",
        "model": MODEL,
        "item": {
            "id": f"{case_id}-source",
            "text": case["text"],
            "source_id": source["source_id"],
            "source_excerpt_sha256": source["sha256"],
        },
        "labels": case["labels"],
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
    if response.get("id") != f"{case_id}-source":
        raise ValueError(f"Unexpected item ID for {case_id}")
    entity_rows = response.get("entities")
    if not isinstance(entity_rows, list):
        raise TypeError(f"Missing entity list for {case_id}")
    entities: list[dict[Any, Any]] = []
    for index, entity in enumerate(entity_rows):
        if not isinstance(entity, dict):
            raise TypeError(f"Invalid entity row at index {index}")
        entities.append(entity)

    allowed_labels = set(case["labels"])
    text = case["text"]
    for index, entity in enumerate(entities):
        label = entity.get("label")
        if label not in allowed_labels:
            raise ValueError(f"Unrequested label at entity {index}: {label}")
        start = entity.get("start")
        end = entity.get("end")
        if type(start) is not int or type(end) is not int:
            raise TypeError(f"Non-integer offsets at entity {index}")
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"Invalid offsets at entity {index}")
        if text[start:end] != entity.get("text"):
            raise ValueError(f"Offset text mismatch at entity {index}")
        score = entity.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Invalid score at entity {index}")
        if score < 0 or score > 1:
            raise ValueError(f"Out-of-range score at entity {index}")
    observed_anchors = {anchor_key(entity) for entity in entities}
    anchor_checks = [
        {
            **anchor,
            "passed": anchor_key(anchor) in observed_anchors,
        }
        for anchor in case["required_anchors"]
    ]
    missing_anchors = [check for check in anchor_checks if not check["passed"]]
    if missing_anchors:
        missing = ", ".join(
            f"{anchor['text']!r} ({anchor['label']} at {anchor['start']}:{anchor['end']})" for anchor in missing_anchors
        )
        raise ValueError(f"{case_id}: missing required anchors: {missing}")
    return {
        "passed": True,
        "entity_count": len(entities),
        "returned_labels": sorted({entity["label"] for entity in entities}),
        "required_anchor_count": len(anchor_checks),
        "matched_anchor_count": len(anchor_checks) - len(missing_anchors),
        "anchor_checks": anchor_checks,
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
        response = client.extract(
            MODEL,
            Item(id=f"{case_id}-source", text=case["text"]),
            labels=case["labels"],
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
    parser = argparse.ArgumentParser(description="Extract custom entities from exact primary-source text")
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
