#!/usr/bin/env python3
"""Ask a question about an image with SIE Cloud, and record the call.

    uv run python run.py                        # every pinned case
    uv run python run.py --case fire-hose-gauge # one case
    python3 run.py --show fire-hose-gauge       # print a request, no network

One call per case, the call the /caption-vqa task page shows:

    POST https://api.superlinked.com/v1/generate/Qwen__Qwen3.8-27B-FP8
    {"prompt": "<question> Start your reply with the answer in one sentence.",
     "max_new_tokens": 256,
     "images": [{"data": "<base64 jpeg>", "format": "jpeg"}]}

No sampling fields are sent, so the answer is whatever the model profile
defaults produce. Results go to --output as a manifest.json and a calls.json in
the dataset's own shape, one entry per call holding the request, the response,
the HTTP status, the served model revision and the round-trip time. The stored
request names the image file and its SHA-256 in place of the base64 bytes. The
key comes from SIE_API_KEY and is never written out.

You do not need a key, and you do not need to run this. The recorded calls are
already published. `python3 fetch.py` downloads them and `python3 score.py`
scores them offline.

The questions, the expected answers and the scoring patterns come from
evidence/inputs/inputs.json and were fixed before the first model call. This
runner cannot change them.
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

TASK = "caption-vqa"
ENDPOINT = "https://api.superlinked.com"
MODEL = "Qwen/Qwen3.8-27B-FP8"
# Native generate takes the SIE-safe id; the SDK normalizes it.
PATH = f"/v1/generate/{MODEL.replace('/', '__')}"
MAX_NEW_TOKENS = 256
IMAGE_FORMAT = "jpeg"


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
    if not inputs.get("answer_instruction"):
        raise SystemExit("inputs.json is missing answer_instruction")
    seen: set[str] = set()
    for case in inputs["cases"]:
        if case["id"] in seen:
            raise SystemExit(f"Duplicate case {case['id']}")
        seen.add(case["id"])
    return inputs


def image_for(case: dict[str, Any]) -> tuple[bytes, str]:
    """The image this case sends, refused if it is not the one inputs.json pins."""
    path = IMAGES_DIR / Path(case["image"]).name
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: python3 fetch.py")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != case["image_sha256"]:
        raise SystemExit(f"{case['id']}: {path.name} is not the image inputs.json pins")
    return data, digest


def prompt_for(case: dict[str, Any], inputs: dict[str, Any]) -> str:
    """The question plus the pre-registered answer-format sentence."""
    return f"{case['question']} {inputs['answer_instruction']}"


def request_record(case: dict[str, Any], inputs: dict[str, Any], digest: str) -> dict[str, Any]:
    """The request as it is stored: the image by reference, not inline."""
    file_name = Path(case["image"]).name
    return {
        "method": "POST",
        "url": ENDPOINT + PATH,
        "model": MODEL,
        "body": {
            "prompt": prompt_for(case, inputs),
            "max_new_tokens": MAX_NEW_TOKENS,
            "images": [{"data": f"<base64 of inputs/images/{file_name}, sha256 {digest}>", "format": IMAGE_FORMAT}],
        },
    }


def record(client: SIEClient, case: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    """Send one case and return its calls.json entry."""
    raw, digest = image_for(case)
    stored = request_record(case, inputs, digest)
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    result = client.generate(
        MODEL,
        prompt_for(case, inputs),
        max_new_tokens=MAX_NEW_TOKENS,
        images=[{"data": raw, "format": IMAGE_FORMAT}],
    )
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    # The SDK attaches request-scoped metadata to the dict it returns. Drop it,
    # so `response` stays the server's own envelope and nothing else.
    envelope = {key: value for key, value in result.items() if key != "request"}
    response = {"status": 200, "headers": {}, "body": envelope}
    return {
        "slug": case["id"],
        "case": case["id"],
        "call": "proof",
        "requested_at": requested_at,
        "request": stored,
        "status": 200,
        "response": response,
        "request_sha256": canonical_sha256(stored),
        "response_sha256": canonical_sha256(response),
        "model": MODEL,
        "model_revision": client.last_model_revision,
        "attempts": client.last_retry_count + 1,
        "timing": {"duration_ms": elapsed_ms},
        "images": [{"path": f"inputs/images/{Path(case['image']).name}", "sha256": digest}],
    }


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
    parser.add_argument(
        "--output", type=Path, default=HERE / "run-output", help="where to write calls.json and manifest.json"
    )
    args = parser.parse_args()

    inputs = load_inputs()
    by_id = {case["id"]: case for case in inputs["cases"]}

    if args.show:
        case = by_id.get(args.show)
        if case is None:
            raise SystemExit(f"Unknown case: {args.show}")
        _, digest = image_for(case)
        print(json.dumps(request_record(case, inputs, digest), indent=2, ensure_ascii=False))
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
        print(f"{case['id']}: POST {PATH}", file=sys.stderr)
        entry = record(client, case, inputs)
        entries.append(entry)
        text = entry["response"]["body"].get("text") or ""
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        print(f"  {entry['timing']['duration_ms']:.0f}ms  {first}", file=sys.stderr)

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
            "max_new_tokens": MAX_NEW_TOKENS,
            "answer_instruction": inputs["answer_instruction"],
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
