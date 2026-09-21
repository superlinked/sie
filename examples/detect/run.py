#!/usr/bin/env python3
"""Find named objects in a photograph with SIE Cloud, and record the call.

    uv run python run.py                            # every pinned photo, both models
    uv run python run.py --case javits-pallet-jacks # one photo
    python3 run.py --show javits-pallet-jacks       # print a request, no network
    python3 run.py --verify-images                  # hash the photos, no network
    python3 run.py --check-requests                 # rebuild every recorded request, no network

Two calls per photograph, the calls the /detect task page shows:

    POST https://api.superlinked.com/v1/extract/IDEA-Research__grounding-dino-base
    POST https://api.superlinked.com/v1/extract/google__owlv2-base-patch16-ensemble
    {"items": [{"images": [{"data": "<base64 jpeg>", "format": "jpeg"}]}],
     "params": {"labels": ["pallet jack", "person"]}}

The labels come from evidence/inputs/inputs.json and were registered, with a
hand count of every instance visible in each photo, before the first model
call. This runner cannot change either.

Results go to --output as a manifest.json and a calls.json in the dataset's own
shape, one entry per call holding the request, the response, the HTTP status,
the served model revision and the round-trip time. The key comes from
SIE_API_KEY and is never written out.

You do not need a key, and you do not need to run this. The recorded calls are
already published. `python3 fetch.py` downloads them and `python3 score.py`
scores them offline.

Only Grounding DINO's boxes carry a published figure. The OWLv2 calls are
recorded for comparison and appear in no total on the page.
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
INPUTS_PATH = EVIDENCE / "inputs" / "inputs.json"
IMAGES_DIR = EVIDENCE / "inputs" / "images"

TASK = "detect"
ENDPOINT = "https://api.superlinked.com"
MODELS = {
    "grounding-dino": "IDEA-Research/grounding-dino-base",
    "owlv2": "google/owlv2-base-patch16-ensemble",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    """SHA-256 of sorted-key, compact JSON: stable across file formatting."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def load_inputs() -> dict[str, Any]:
    if not INPUTS_PATH.is_file():
        raise SystemExit(f"Missing {INPUTS_PATH}. Run: python3 fetch.py")
    inputs = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for case in inputs["cases"]:
        if case["id"] in seen:
            raise SystemExit(f"Duplicate case {case['id']}")
        seen.add(case["id"])
        for field in ("file", "image_sha256", "labels", "expected_counts", "source"):
            if not case.get(field):
                raise SystemExit(f"{case['id']} is missing {field}")
        registered = {label.casefold() for label in case["labels"]}
        counted = {label.casefold() for label in case["expected_counts"]}
        if registered != counted:
            raise SystemExit(f"{case['id']}: the labels sent and the labels hand-counted are not the same set")
    return inputs


def image_for(case: dict[str, Any]) -> bytes:
    """The photograph this case sends, checked against the digest it registers.

    A file that is present but does not match stops the run rather than being
    quietly replaced: the interesting case is a changed input, not a stale one.
    """
    path = IMAGES_DIR / case["file"]
    if not path.is_file():
        raise SystemExit(f"{case['id']}: missing {path}. Run: python3 fetch.py")
    data = path.read_bytes()
    if sha256_bytes(data) != case["image_sha256"]:
        raise SystemExit(f"{case['id']}: {path} is not the image inputs.json pins; re-run fetch.py")
    return data


def payload_reference(case: dict[str, Any], data: bytes) -> dict[str, Any]:
    """A machine-readable stand-in for an image that is not stored inline.

    A reader can read the file from the dataset and hash it; a script can see
    the payload is by-reference rather than inline, which a prose placeholder
    made impossible to tell apart from real data.
    """
    return {
        "data": {
            "file": case["file"],
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "omitted": "base64 of the image file below",
        },
        "format": "jpeg",
    }


def request_record(case: dict[str, Any], call: str, data: bytes) -> dict[str, Any]:
    """The request as it is stored: the image by reference, not inline."""
    model = MODELS[call]
    return {
        "method": "POST",
        "endpoint": ENDPOINT,
        "path": f"/v1/extract/{model.replace('/', '%2F')}",
        "model": model,
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "body": {
            "items": [{"images": [payload_reference(case, data)]}],
            "params": {"labels": list(case["labels"])},
        },
    }


def record(client: SIEClient, case: dict[str, Any], call: str, data: bytes) -> dict[str, Any]:
    """Send one call and return its calls.json entry."""
    model = MODELS[call]
    stored = request_record(case, call, data)
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    result = client.extract(
        model,
        {"images": [{"data": data, "format": "jpeg"}]},
        labels=list(case["labels"]),
    )
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    # The SDK attaches request-scoped metadata to the dict it returns. Drop it,
    # so the recorded body stays the server's own envelope and nothing else.
    item = {key: value for key, value in result.items() if key != "request"}
    body = {"items": [item], "model": model}
    response = {"status": 200, "headers": {}, "body": body}
    return {
        "slug": f"{case['id']}__{call}",
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
        "images": [{"path": f"inputs/images/{case['file']}", "sha256": sha256_bytes(data)}],
    }


def verify_images(inputs: dict[str, Any]) -> int:
    """Hash every fetched photograph against the SHA-256 inputs.json pins.

    A file that is present but does not match is a failure. A file that was
    never fetched is NOT counted as passing: it is reported separately and
    prominently, because a check that silently skips what it cannot read is
    worse than no check.
    """
    matched, mismatched, absent = 0, [], []
    for case in inputs["cases"]:
        path = IMAGES_DIR / case["file"]
        if not path.is_file():
            absent.append(f"{case['id']}: {case['file']} ({case['source']['original_url']})")
            continue
        if sha256_bytes(path.read_bytes()) != case["image_sha256"]:
            mismatched.append(f"{case['id']}: {case['file']} does not match the SHA-256 inputs.json pins")
        else:
            matched += 1
    total = matched + len(mismatched) + len(absent)
    print(f"images: {matched} of {total} photographs verified", file=sys.stderr)
    for line in mismatched:
        print(f"  MISMATCH {line}", file=sys.stderr)
    if absent:
        print(f"  {len(absent)} NOT CHECKED, run fetch.py to download them:", file=sys.stderr)
        for line in absent:
            print(f"    {line}", file=sys.stderr)
    return 1 if (mismatched or absent) else 0


def check_requests(inputs: dict[str, Any]) -> int:
    """Rebuild every recorded request with this runner and compare the digests.

    This is what makes the runner worth shipping: it establishes that the file
    in this repository sends what the recording says was sent, rather than
    merely resembling it. A call in the recording that this runner cannot
    rebuild is a failure, and so is a case it would send that nothing recorded.
    """
    calls_path = EVIDENCE / "calls.json"
    if not calls_path.is_file():
        raise SystemExit(f"Missing {calls_path}. Run: python3 fetch.py")
    recorded = {
        entry["slug"]: entry for entry in json.loads(calls_path.read_text(encoding="utf-8"))["calls"]
    }
    matched, problems = 0, []
    expected: set[str] = set()
    for case in inputs["cases"]:
        data = image_for(case)
        for call in MODELS:
            slug = f"{case['id']}__{call}"
            expected.add(slug)
            entry = recorded.get(slug)
            if entry is None:
                problems.append(f"{slug}: this runner would send it, but nothing recorded it")
                continue
            mine = canonical_sha256(request_record(case, call, data))
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
    parser.add_argument("--case", help="run one case id")
    parser.add_argument("--show", metavar="CASE", help="print one request and exit, sending nothing")
    parser.add_argument("--verify-images", action="store_true", help="hash every fetched photograph and exit")
    parser.add_argument(
        "--check-requests",
        action="store_true",
        help="rebuild every recorded request with this runner and compare the digests, sending nothing",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "run-output", help="where to write calls.json and manifest.json"
    )
    args = parser.parse_args()

    inputs = load_inputs()
    by_id = {case["id"]: case for case in inputs["cases"]}

    if args.verify_images:
        return verify_images(inputs)

    if args.check_requests:
        return check_requests(inputs)

    if args.show:
        case = by_id.get(args.show)
        if case is None:
            raise SystemExit(f"Unknown case: {args.show}")
        print(json.dumps(request_record(case, "grounding-dino", image_for(case)), indent=2, ensure_ascii=False))
        return 0

    cases = [case for case in inputs["cases"] if not args.case or case["id"] == args.case]
    if not cases:
        raise SystemExit(f"Unknown case: {args.case}")

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY to send these calls, or run score.py on the recorded ones instead")
    base_url = os.environ.get("SIE_BASE_URL", ENDPOINT)
    # Imported here rather than at module scope so --show and the offline
    # checks above run on a bare `python3` with nothing installed. Sending the
    # calls needs the SDK: `uv sync`, then `uv run python run.py`.
    from sie_sdk import SIEClient

    client = SIEClient(base_url, api_key=api_key, timeout_s=900)

    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    entries: list[dict[str, Any]] = []
    for case in cases:
        data = image_for(case)
        for call in MODELS:
            print(f"{case['id']}__{call}: POST /v1/extract", file=sys.stderr)
            entry = record(client, case, call, data)
            entries.append(entry)
            boxes = len(entry["response"]["body"]["items"][0].get("objects") or [])
            print(f"  {entry['timing']['duration_ms']:.0f}ms, {boxes} boxes", file=sys.stderr)

    slugs = [entry["slug"] for entry in entries]
    if len(slugs) != len(set(slugs)):
        raise SystemExit("duplicate call slugs; refusing to write a calls.json two checks could read differently")

    write_json(
        args.output / "calls.json",
        {"schema_version": "1.0", "task": TASK, "endpoint": base_url, "runner": "run.py", "calls": entries},
        api_key,
    )
    write_json(
        args.output / "manifest.json",
        {
            "task": TASK,
            "endpoint": base_url,
            "models": sorted(MODELS.values()),
            "model_revision": next((e["model_revision"] for e in entries if e["model_revision"]), None),
            "run_started_utc": started_at,
            "run_completed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "complete_run": not args.case,
            "calls_recorded": len(entries),
            "inputs_sha256": sha256_bytes(INPUTS_PATH.read_bytes()),
            "inputs_sha256_rule": "sha256 of the bytes of inputs/inputs.json",
        },
        api_key,
    )
    print(f"{len(entries)} calls written to {args.output}", file=sys.stderr)
    print(
        "No box-review.json is written: a verdict per box is a hand judgement, not something a rerun produces.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
