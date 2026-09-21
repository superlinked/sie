#!/usr/bin/env python3
"""Grade a product photo against your own labels with SIE Cloud, and record it.

    uv run python run.py --images /path/to/VisA          # both runs
    uv run python run.py --images /path/to/VisA --run 02-pass-reject
    python3 run.py --show cashew-damaged-014             # print a request, no network
    python3 run.py --check-requests                      # rebuild every recorded request, no network

Two calls per photograph, the calls the /image-classify task page shows:

    POST https://api.superlinked.com/v1/score/Qwen%2FQwen3-VL-Reranker-2B
    {"query": {"images": [{"data": "<base64 jpeg>", "format": "jpeg"}]},
     "items": [{"id": "0", "text": "whole undamaged cashew"},
               {"id": "1", "text": "broken or damaged cashew"}]}

    POST https://api.superlinked.com/v1/encode/google%2Fsiglip2-base-patch16-224
    {"items": [{"images": [...]}, {"text": "..."}, {"text": "..."}],
     "params": {"output_types": ["dense"]}}

No training set and no fine-tune: the labels are strings, and swapping them
changes what the model grades for.

The photographs are VisA (Amazon Science, CC BY 4.0) and are not redistributed
with this example, because the source is a 20 GB tar. `--images` points at your
own extracted copy; every file is checked against the SHA-256 the pre-registered
inputs file pins before it is sent, so a wrong or altered copy stops the run.
The stored request carries `sha256:<digest>` in place of the base64 bytes,
which is why `--show` and `--check-requests` need no photographs at all.

Results go to --output as a manifest.json and a calls.json in the dataset's own
shape. The key comes from SIE_API_KEY and is never written out.

You do not need a key, and you do not need to run this. The recorded calls are
already published. `python3 fetch.py` downloads them and `python3 score.py`
scores them offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sie_sdk import SIEClient

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
INPUTS_DIR = EVIDENCE / "inputs"

TASK = "image-classify"
ENDPOINT = "https://api.superlinked.com"
RERANKER = "Qwen/Qwen3-VL-Reranker-2B"
SIGLIP = "google/siglip2-base-patch16-224"
RUNS = ("01-grades", "02-pass-reject")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    """SHA-256 of sorted-key, compact JSON: stable across file formatting."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def load_run(run: str) -> dict[str, Any]:
    path = INPUTS_DIR / f"{run}.json"
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: python3 fetch.py")
    document = json.loads(path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for case in document["cases"]:
        if case["id"] in seen:
            raise SystemExit(f"{run}: duplicate case {case['id']}")
        seen.add(case["id"])
        for field in ("product", "expected", "image", "sha256"):
            if not case.get(field):
                raise SystemExit(f"{run}/{case['id']} is missing {field}")
        labels = document["label_sets"].get(case["product"])
        if not labels or sorted(labels) != sorted(document["classes"]):
            raise SystemExit(f"{run}/{case['id']}: no label set covering every registered class")
        if case["expected"] not in document["classes"]:
            raise SystemExit(f"{run}/{case['id']}: expected class {case['expected']!r} is not registered")
    return document


def labels_for(document: dict[str, Any], case: dict[str, Any]) -> list[str]:
    """The label strings this photo is scored against, in registered order."""
    labels = document["label_sets"][case["product"]]
    return [labels[name] for name in document["classes"]]


def image_for(case: dict[str, Any], images: Path | None) -> bytes:
    """The VisA photograph this case sends, checked against its pinned digest."""
    if images is None:
        raise SystemExit("Pass --images with your extracted VisA copy to send these calls")
    path = images / case["image"]
    if not path.is_file():
        raise SystemExit(f"{case['id']}: missing {path}; --images should point at the extracted VisA root")
    data = path.read_bytes()
    if sha256_bytes(data) != case["sha256"]:
        raise SystemExit(f"{case['id']}: {path} is not the photograph the inputs file pins")
    return data


def request_record(document: dict[str, Any], case: dict[str, Any], call: str) -> dict[str, Any]:
    """The request as it is stored: the photograph by digest, not inline.

    The digest comes from the pre-registered inputs file, so this builds the
    exact stored record without reading a single image.
    """
    labels = labels_for(document, case)
    image = {"data": f"sha256:{case['sha256']}", "format": "jpeg"}
    if call == "reranker-score":
        return {
            "method": "POST",
            "endpoint": ENDPOINT,
            "path": f"/v1/score/{RERANKER.replace('/', '%2F')}",
            "model": RERANKER,
            "headers": {"Content-Type": "application/json", "Accept": "application/json"},
            "image_data": "base64 of the pre-registered JPEG, replaced here by its sha256",
            "body": {
                "query": {"images": [image]},
                "items": [{"id": str(index), "text": label} for index, label in enumerate(labels)],
            },
        }
    return {
        "method": "POST",
        "endpoint": ENDPOINT,
        "path": f"/v1/encode/{SIGLIP.replace('/', '%2F')}",
        "model": SIGLIP,
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "image_data": "base64 of the pre-registered JPEG, replaced here by its sha256",
        "body": {
            "items": [{"images": [image]}, *({"text": label} for label in labels)],
            "params": {"output_types": ["dense"]},
        },
    }


def record(
    client: SIEClient, document: dict[str, Any], case: dict[str, Any], call: str, run: str, data: bytes
) -> dict[str, Any]:
    """Send one call and return its calls.json entry."""
    labels = labels_for(document, case)
    stored = request_record(document, case, call)
    image = {"data": data, "format": "jpeg"}
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    if call == "reranker-score":
        model = RERANKER
        result = client.score(
            model,
            {"images": [image]},
            [{"id": str(index), "text": label} for index, label in enumerate(labels)],
        )
        envelope = {key: value for key, value in result.items() if key != "request"}
    else:
        model = SIGLIP
        results = client.encode(
            model,
            [{"images": [image]}, *({"text": label} for label in labels)],
            output_types=["dense"],
        )
        envelope = {
            "items": [{key: value for key, value in item.items() if key != "request"} for item in results],
            "model": model,
        }
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    response = {"status": 200, "headers": {}, "body": envelope}
    return {
        "slug": f"{run}__{case['id']}__{call}",
        "run": run,
        "case": case["id"],
        "call": call,
        "model": model,
        "model_revision": client.last_model_revision,
        "requested_at": requested_at,
        "status": "ok",
        "http_status": 200,
        "attempts": client.last_retry_count + 1,
        "request": stored,
        "request_sha256": canonical_sha256(stored),
        "response": response,
        "response_sha256": canonical_sha256(response),
        "timing": {"duration_ms": elapsed_ms},
        "source_image_sha256": case["sha256"],
    }


def check_requests(documents: dict[str, dict[str, Any]]) -> int:
    """Rebuild every recorded request with this runner and compare the digests.

    This is what makes the runner worth shipping: it establishes that the file
    in this repository sends what the recording says was sent, rather than
    merely resembling it. It needs no photographs, because the stored request
    carries the digest rather than the bytes.
    """
    calls_path = EVIDENCE / "calls.json"
    if not calls_path.is_file():
        raise SystemExit(f"Missing {calls_path}. Run: python3 fetch.py")
    recorded = {entry["slug"]: entry for entry in json.loads(calls_path.read_text(encoding="utf-8"))["calls"]}
    matched, problems = 0, []
    expected: set[str] = set()
    for run, document in documents.items():
        for case in document["cases"]:
            for call in ("reranker-score", "siglip-encode"):
                slug = f"{run}__{case['id']}__{call}"
                expected.add(slug)
                entry = recorded.get(slug)
                if entry is None:
                    problems.append(f"{slug}: this runner would send it, but nothing recorded it")
                    continue
                mine = canonical_sha256(request_record(document, case, call))
                if mine != entry["request_sha256"]:
                    problems.append(f"{slug}: this runner builds a different request than the one recorded")
                elif mine != canonical_sha256(entry["request"]):
                    problems.append(f"{slug}: the stored request does not match its own recorded digest")
                else:
                    matched += 1
    for slug in recorded:
        if slug not in expected:
            problems.append(f"{slug}: recorded, but this runner would never send it")
    print(f"requests: {matched} of {len(expected)} rebuilt identically by this runner", file=sys.stderr)
    for line in problems:
        print(f"  {line}", file=sys.stderr)
    return 1 if problems else 0


def write_json(path: Path, value: Any, secret: str | None) -> None:
    text = json.dumps(value, indent="\t", ensure_ascii=False) + "\n"
    # Check the serialized bytes before they reach the disk, so a key echoed
    # back inside a response body never lands even if a later call fails.
    if secret and secret in text:
        raise SystemExit(f"Refusing to write {path.name}: it contains the API key")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=RUNS, help="run one of the two recorded runs")
    parser.add_argument("--case", help="run one case id")
    parser.add_argument("--show", metavar="CASE", help="print one request and exit, sending nothing")
    parser.add_argument(
        "--check-requests",
        action="store_true",
        help="rebuild every recorded request with this runner and compare the digests, sending nothing",
    )
    parser.add_argument("--images", type=Path, help="root of your extracted VisA copy")
    parser.add_argument(
        "--output", type=Path, default=HERE / "run-output", help="where to write calls.json and manifest.json"
    )
    args = parser.parse_args()

    documents = {run: load_run(run) for run in (RUNS if not args.run else (args.run,))}

    if args.check_requests:
        return check_requests({run: load_run(run) for run in RUNS})

    if args.show:
        for run, document in documents.items():
            case = next((item for item in document["cases"] if item["id"] == args.show), None)
            if case is not None:
                print(json.dumps(request_record(document, case, "reranker-score"), indent=2, ensure_ascii=False))
                return 0
        raise SystemExit(f"Unknown case: {args.show}")

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY to send these calls, or run score.py on the recorded ones instead")
    base_url = os.environ.get("SIE_BASE_URL", ENDPOINT)
    # Imported here rather than at module scope so --show and --check-requests
    # run on a bare `python3` with nothing installed. Sending the calls needs
    # the SDK: `uv sync`, then `uv run python run.py`.
    from sie_sdk import SIEClient

    client = SIEClient(base_url, api_key=api_key, timeout_s=900)

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    entries: list[dict[str, Any]] = []
    for run, document in documents.items():
        cases = [case for case in document["cases"] if not args.case or case["id"] == args.case]
        if args.case and not cases:
            continue
        for case in cases:
            data = image_for(case, args.images)
            for call in ("reranker-score", "siglip-encode"):
                print(f"{run}__{case['id']}__{call}", file=sys.stderr)
                entry = record(client, document, case, call, run, data)
                entries.append(entry)
                print(f"  {entry['timing']['duration_ms']:.0f}ms", file=sys.stderr)
    if not entries:
        raise SystemExit(f"Unknown case: {args.case}")

    slugs = [entry["slug"] for entry in entries]
    if len(slugs) != len(set(slugs)):
        raise SystemExit("duplicate call slugs; refusing to write a calls.json two checks could read differently")

    write_json(
        args.output / "calls.json",
        {
            "schema_version": "1.0",
            "task": TASK,
            "endpoint": base_url,
            "runner": "run.py",
            "runs": sorted(documents),
            "calls": entries,
        },
        api_key,
    )
    write_json(
        args.output / "manifest.json",
        {
            "task": TASK,
            "endpoint": base_url,
            "models": [RERANKER, SIGLIP],
            "model_revision": next((e["model_revision"] for e in entries if e["model_revision"]), None),
            "run_started_utc": started_at,
            "run_completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "complete_run": not args.case and not args.run,
            "calls_recorded": len(entries),
            "inputs_sha256": {
                run: sha256_bytes((INPUTS_DIR / f"{run}.json").read_bytes()) for run in documents
            },
            "inputs_sha256_rule": "sha256 of the bytes of each inputs file",
        },
        api_key,
    )
    print(f"{len(entries)} calls written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
