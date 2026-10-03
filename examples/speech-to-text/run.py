#!/usr/bin/env python3
"""Transcribe a meeting, call or hearing clip with SIE Cloud, and record it.

    uv run python run.py                                  # every pinned clip
    uv run python run.py --case primock-uti-antibiotics   # one clip
    python3 run.py --show primock-uti-antibiotics         # print a request, no network
    python3 run.py --verify-audio                         # hash the clips, no network
    python3 run.py --check-requests                       # rebuild every recorded request, no network

Two calls per clip, the first of which is the call the /speech-to-text task
page shows:

    POST https://api.superlinked.com/v1/extract/openai%2Fwhisper-large-v3-turbo
    {"items": [{"audio": {"data": "<base64 mp3>", "format": "mp3"}}],
     "params": {"instruction": "Transcribe the audio verbatim."}}

The second sends the same clip with no params at all, as a control. It is
recorded, scored and never displayed as the result.

The human transcripts and the key terms come from evidence/inputs/inputs.json.
Each clip is cut at the published human timing, with no padding, and each
reference is that publisher's own transcript for the span. Both were fixed
before the first model call and this runner cannot change them.

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
INPUTS_PATH = EVIDENCE / "inputs" / "inputs.json"
AUDIO_DIR = EVIDENCE / "inputs" / "audio"

TASK = "speech-to-text"
ENDPOINT = "https://api.superlinked.com"
MODEL = "openai/whisper-large-v3-turbo"
PATH = f"/v1/extract/{MODEL.replace('/', '%2F')}"
INSTRUCTION = "Transcribe the audio verbatim."
CALLS = ("snippet", "control")


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
        for field in ("reference", "key_terms", "start", "end", "license"):
            if not case.get(field):
                raise SystemExit(f"{case['id']} is missing {field}")
        if case["end"] <= case["start"]:
            raise SystemExit(f"{case['id']}: the clip ends before it starts")
    return inputs


def audio_for(case: dict[str, Any], expected: str | None = None) -> bytes:
    """The clip this case sends, checked against the digest the run pinned."""
    path = AUDIO_DIR / f"{case['id']}.mp3"
    if not path.is_file():
        raise SystemExit(f"{case['id']}: missing {path}. Run: python3 fetch.py")
    data = path.read_bytes()
    if expected and sha256_bytes(data) != expected:
        raise SystemExit(f"{case['id']}: {path} is not the clip the run was sent; re-run fetch.py")
    return data


def payload_reference(case: dict[str, Any], digest: str, size: int) -> dict[str, Any]:
    """A machine-readable stand-in for audio that is not stored inline."""
    return {
        "data": {
            "replaced_with": "served clip bytes, base64-encoded",
            "clip_url": f"/reference/speech-to-text/{case['id']}.mp3",
            "sha256": digest,
            "bytes": size,
        },
        "format": "mp3",
    }


def request_record(case: dict[str, Any], call: str, digest: str, size: int) -> dict[str, Any]:
    """The request as it is stored: the clip by reference, not inline."""
    body: dict[str, Any] = {"items": [{"audio": payload_reference(case, digest, size)}]}
    if call == "snippet":
        body["params"] = {"instruction": INSTRUCTION}
    return {
        "method": "POST",
        "endpoint": ENDPOINT,
        "path": PATH,
        "model": MODEL,
        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
        "body": body,
    }


def record(client: SIEClient, case: dict[str, Any], call: str, data: bytes) -> dict[str, Any]:
    """Send one call and return its calls.json entry."""
    digest = sha256_bytes(data)
    stored = request_record(case, call, digest, len(data))
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    result = client.extract(
        MODEL,
        {"audio": {"data": data, "format": "mp3"}},
        **({"instruction": INSTRUCTION} if call == "snippet" else {}),
    )
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    # The SDK attaches request-scoped metadata to the dict it returns. Drop it,
    # so the recorded body stays the server's own envelope and nothing else.
    item = {key: value for key, value in result.items() if key != "request"}
    response = {"status": 200, "headers": {}, "body": {"items": [item], "model": MODEL}}
    return {
        "slug": f"{case['id']}__{call}",
        "case": case["id"],
        "call": call,
        "model": MODEL,
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
        "audio": {"path": f"inputs/audio/{case['id']}.mp3", "sha256": digest},
    }


def verify_audio(inputs: dict[str, Any]) -> int:
    """Hash every fetched clip against the digest its recorded call pins.

    A file that is present but does not match is a failure. A file that was
    never fetched is NOT counted as passing: it is reported separately and
    prominently, because a check that silently skips what it cannot read is
    worse than no check.
    """
    calls_path = EVIDENCE / "calls.json"
    if not calls_path.is_file():
        raise SystemExit(f"Missing {calls_path}. Run: python3 fetch.py")
    pinned = {
        entry["case"]: entry["audio"]["sha256"]
        for entry in json.loads(calls_path.read_text(encoding="utf-8"))["calls"]
    }
    matched, mismatched, absent = 0, [], []
    for case in inputs["cases"]:
        path = AUDIO_DIR / f"{case['id']}.mp3"
        if not path.is_file():
            absent.append(f"{case['id']}: {case['id']}.mp3")
            continue
        if case["id"] not in pinned:
            mismatched.append(f"{case['id']}: no recorded call pins a digest for this clip")
        elif sha256_bytes(path.read_bytes()) != pinned[case["id"]]:
            mismatched.append(f"{case['id']}: {case['id']}.mp3 is not the clip the run was sent")
        else:
            matched += 1
    total = matched + len(mismatched) + len(absent)
    print(f"audio: {matched} of {total} clips verified", file=sys.stderr)
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
    merely resembling it.
    """
    calls_path = EVIDENCE / "calls.json"
    if not calls_path.is_file():
        raise SystemExit(f"Missing {calls_path}. Run: python3 fetch.py")
    recorded = {entry["slug"]: entry for entry in json.loads(calls_path.read_text(encoding="utf-8"))["calls"]}
    matched, problems = 0, []
    expected: set[str] = set()
    for case in inputs["cases"]:
        for call in CALLS:
            slug = f"{case['id']}__{call}"
            expected.add(slug)
            entry = recorded.get(slug)
            if entry is None:
                problems.append(f"{slug}: this runner would send it, but nothing recorded it")
                continue
            sent = entry["request"]["body"]["items"][0]["audio"]["data"]
            data = audio_for(case, sent["sha256"])
            mine = canonical_sha256(request_record(case, call, sha256_bytes(data), len(data)))
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
    parser.add_argument("--verify-audio", action="store_true", help="hash every fetched clip and exit")
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

    if args.verify_audio:
        return verify_audio(inputs)

    if args.check_requests:
        return check_requests(inputs)

    if args.show:
        case = by_id.get(args.show)
        if case is None:
            raise SystemExit(f"Unknown case: {args.show}")
        data = audio_for(case)
        print(
            json.dumps(
                request_record(case, "snippet", sha256_bytes(data), len(data)), indent=2, ensure_ascii=False
            )
        )
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
        data = audio_for(case)
        for call in CALLS:
            print(f"{case['id']}__{call}: POST {PATH}", file=sys.stderr)
            entry = record(client, case, call, data)
            entries.append(entry)
            words = len(entry["response"]["body"]["items"][0].get("data", {}).get("text", "").split())
            print(f"  {entry['timing']['duration_ms']:.0f}ms, {words} words", file=sys.stderr)

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
            "instruction": INSTRUCTION,
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
