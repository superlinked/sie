#!/usr/bin/env python3
"""Read the values off an application screenshot with SIE Cloud, and record it.

    uv run python run.py                             # every pinned case
    uv run python run.py --case gitlab-runner-fleet  # one case
    python3 run.py --show gitlab-runner-fleet        # print a request, no network
    python3 run.py --verify-images                   # hash the images, no network

One call per screen, the call the /screenshot-mining task page shows:

    POST https://api.superlinked.com/v1/generate/Qwen__Qwen3.8-27B-FP8
    {"prompt": "<what to read off this screen>",
     "max_new_tokens": <per case>,
     "images": [{"data": "<base64 png or jpeg>", "format": "png"}],
     "grammar": {"json_schema": {...}, "label": "<name>", "strict": true}}

Only cases that register a temperature send one; otherwise no sampling fields
are sent and the reply is whatever the model profile defaults produce under the
grammar. Results go to --output as a manifest.json and a calls.json in the
dataset's own shape, one entry per call holding the request, the response, the
HTTP status, the served model revision and the round-trip time. The key comes
from SIE_API_KEY and is never written out.

The screenshots are the projects' own published images. The stored request
carries a `$payload` descriptor in place of the base64 bytes, giving the media
type, byte length, SHA-256 and the commit-pinned URL the file came from. Images
are read from evidence/inputs/images/; anything absent is fetched from that URL
and checked against the SHA-256 inputs.json pins before it is used.

You do not need a key, and you do not need to run this. The recorded calls are
already published. `python3 fetch.py` downloads them and `python3 score.py`
scores them offline.

The expected values and the comparison rules come from
evidence/inputs/inputs.json and were read off each screenshot at full
resolution before the first model call. This runner cannot change them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sie_sdk import SIEClient

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
INPUTS_PATH = EVIDENCE / "inputs" / "inputs.json"
IMAGES_DIR = EVIDENCE / "inputs" / "images"

TASK = "screenshot-mining"
ENDPOINT = "https://api.superlinked.com"
MODEL = "Qwen/Qwen3.8-27B-FP8"
PATH = f"/v1/generate/{MODEL.replace('/', '__')}"
MEDIA_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    """SHA-256 of sorted-key, compact JSON: stable across file formatting."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


def image_format(file_name: str) -> str:
    """The `format` the page's snippet sends for this file extension."""
    extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if not extension:
        return "png"
    return "jpeg" if extension == "jpg" else extension


def load_inputs() -> dict[str, Any]:
    if not INPUTS_PATH.is_file():
        raise SystemExit(f"Missing {INPUTS_PATH}. Run: python3 fetch.py")
    inputs = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    seen: set[str] = set()
    for case in inputs["cases"]:
        if case["id"] in seen:
            raise SystemExit(f"Duplicate case {case['id']}")
        seen.add(case["id"])
        for field in ("image_url", "image_sha256", "file_name", "app", "publisher", "license", "source_page", "calls"):
            if not case.get(field):
                raise SystemExit(f"{case['id']} is missing {field}")
        for call in case["calls"]:
            # This runner builds one body shape. A case registering another
            # kind stops the run rather than being sent the wrong request.
            if call.get("kind") != "qwen-schema":
                raise SystemExit(
                    f"{case['id']}/{call['call']} is kind {call.get('kind')!r}, which run.py does not send"
                )
            if "expected" not in call:
                raise SystemExit(f"{case['id']}/{call['call']} has no pre-registered expected values")
    return inputs


def image_for(case: dict[str, Any]) -> bytes:
    """The screenshot this case sends: from the dataset, else from the source.

    A file that is present but does not match its registered SHA-256 stops the
    run rather than being re-downloaded over, because the interesting case is a
    changed input, not a corrupt cache.
    """
    path = IMAGES_DIR / case["file_name"]
    if path.is_file():
        data = path.read_bytes()
        if sha256_bytes(data) != case["image_sha256"]:
            raise SystemExit(f"{case['id']}: {path} is not the image inputs.json pins; re-run fetch.py")
        return data
    print(f"  {case['file_name']} is not in the fetched copy, reading {case['image_url']}", file=sys.stderr)
    request = urllib.request.Request(case["image_url"], headers={"User-Agent": f"sie-examples/{TASK}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if sha256_bytes(data) != case["image_sha256"]:
        raise SystemExit(
            f"{case['id']}: the source image bytes changed; review the source before touching image_sha256"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def payload_reference(case: dict[str, Any], data: bytes) -> dict[str, Any]:
    """A machine-readable stand-in for an image that is not stored inline.

    A reader can read the file from the dataset or fetch `url` and hash it; a
    script can see the payload is by-reference rather than inline, which a
    prose placeholder made impossible to tell apart from real data.
    """
    fmt = image_format(case["file_name"])
    return {
        "format": fmt,
        "$payload": {
            "media_type": MEDIA_TYPES.get(fmt, f"image/{fmt}"),
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "url": case["image_url"],
            "file_name": case["file_name"],
            "stored": False,
            "note": (
                "By reference, not inline. The bytes are in the sie-task-evidence dataset at "
                "inputs/images/`file_name`, and at `url`; either way, check `sha256`. "
                "`run.py --verify-images` does that, failing on a mismatch and reporting anything "
                "it could not read rather than passing over it."
            ),
        },
    }


def body_for(call: dict[str, Any], image: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prompt": call["prompt"],
        "max_new_tokens": call["max_new_tokens"],
        "images": [image],
        "grammar": {"json_schema": call["schema"], "label": call["schema_name"], "strict": True},
    }
    if "temperature" in call:
        body["temperature"] = call["temperature"]
    return body


def request_record(case: dict[str, Any], call: dict[str, Any], data: bytes) -> dict[str, Any]:
    """The request as it is stored: the image by reference, not inline."""
    return {
        "method": "POST",
        "url": ENDPOINT + PATH,
        "model": MODEL,
        "body": body_for(call, payload_reference(case, data)),
    }


def record(client: SIEClient, case: dict[str, Any], call: dict[str, Any], data: bytes) -> dict[str, Any]:
    """Send one call and return its calls.json entry."""
    stored = request_record(case, call, data)
    extra: dict[str, Any] = {}
    if "temperature" in call:
        extra["temperature"] = call["temperature"]
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    result = client.generate(
        MODEL,
        call["prompt"],
        max_new_tokens=call["max_new_tokens"],
        images=[{"data": data, "format": image_format(case["file_name"])}],
        grammar={"json_schema": call["schema"], "label": call["schema_name"], "strict": True},
        **extra,
    )
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    # The SDK attaches request-scoped metadata to the dict it returns. Drop it,
    # so `value` stays the server's own envelope and nothing else.
    envelope = {key: value for key, value in result.items() if key != "request"}
    # The entry shape the published calls.json uses, so score.py reads a fresh
    # run the same way it reads the recorded one. The one field a fresh run
    # cannot carry is `entry_sha256`, the RFC 8785 canonical digest the
    # sie-web recorder wrote over each whole entry; score.py does not check it
    # either, and the README says so.
    return {
        "slug": f"{case['id']}__{call['call']}",
        "case": case["id"],
        "call": call["call"],
        "requested_at": requested_at,
        "status": "ok",
        "http_status": 200,
        "request": {
            **stored,
            "recorded_sha256": canonical_sha256(stored),
            "recorded_sha256_scope": "the request record as written here, with the $payload descriptor",
        },
        "response": {
            "value": envelope,
            # The SDK surfaces no response headers, so a fresh run records none.
            "http_headers": {},
            "recorded_sha256": canonical_sha256({"status": 200, "headers": {}, "body": envelope}),
        },
        "model": MODEL,
        "model_revision": client.last_model_revision,
        "retry_count": client.last_retry_count,
        "timing": {"duration_ms": elapsed_ms},
        "images": [{"path": f"inputs/images/{case['file_name']}", "sha256": sha256_bytes(data)}],
    }


def verify_images(inputs: dict[str, Any]) -> int:
    """Hash every fetched screenshot against the SHA-256 inputs.json pins.

    A file that is present but does not match is a failure. A file that was
    never fetched is NOT counted as passing: it is reported separately and
    prominently, because a check that silently skips what it cannot read is
    worse than no check.
    """
    matched, mismatched, absent = 0, [], []
    files: set[str] = set()
    for case in inputs["cases"]:
        files.add(case["file_name"])
        path = IMAGES_DIR / case["file_name"]
        if not path.is_file():
            absent.append(f"{case['id']}: {case['file_name']} ({case['image_url']})")
            continue
        if sha256_bytes(path.read_bytes()) != case["image_sha256"]:
            mismatched.append(f"{case['id']}: {case['file_name']} does not match the SHA-256 inputs.json pins")
        else:
            matched += 1
    total = matched + len(mismatched) + len(absent)
    # One check per case, not per file: two cases can register the same
    # screenshot, and each is checked against its own pinned digest.
    print(f"images: {matched} of {total} case images verified, across {len(files)} distinct files", file=sys.stderr)
    for line in mismatched:
        print(f"  MISMATCH {line}", file=sys.stderr)
    if absent:
        print(f"  {len(absent)} NOT CHECKED, run fetch.py to download them:", file=sys.stderr)
        for line in absent:
            print(f"    {line}", file=sys.stderr)
    return 1 if (mismatched or absent) else 0


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
    parser.add_argument("--verify-images", action="store_true", help="hash every fetched screenshot and exit")
    parser.add_argument(
        "--output", type=Path, default=HERE / "run-output", help="where to write calls.json and manifest.json"
    )
    args = parser.parse_args()

    inputs = load_inputs()
    by_id = {case["id"]: case for case in inputs["cases"]}

    if args.verify_images:
        return verify_images(inputs)

    if args.show:
        case = by_id.get(args.show)
        if case is None:
            raise SystemExit(f"Unknown case: {args.show}")
        data = image_for(case)
        print(json.dumps(request_record(case, case["calls"][0], data), indent=2, ensure_ascii=False))
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
        for call in case["calls"]:
            print(f"{case['id']}__{call['call']}: POST {PATH}", file=sys.stderr)
            entry = record(client, case, call, data)
            entries.append(entry)
            value = entry["response"]["value"]
            print(f"  {entry['timing']['duration_ms']:.0f}ms, finish={value.get('finish_reason')}", file=sys.stderr)

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
            "path": PATH,
            "model": MODEL,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
