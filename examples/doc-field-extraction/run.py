#!/usr/bin/env python3
"""Turn a document image into typed fields with SIE Cloud, and record the call.

    uv run python run.py                                # every pinned call
    uv run python run.py --case faa-8130-3-export-flap  # one document
    uv run python run.py --call playground              # one call type
    python3 run.py --show faa-8130-3-export-flap        # print a request, no network

One call per document, the call the /doc-field-extraction task page shows:

    POST https://api.superlinked.com/v1/generate/Qwen__Qwen3.8-27B-FP8
    {"prompt": "Extract the document fields described by the JSON schema. ...",
     "max_new_tokens": 2048,
     "images": [{"data": "<base64 page image>", "format": "jpeg"}],
     "grammar": {"json_schema": {...}, "strict": true}}

The proof calls raise max_new_tokens to 2048 so long tables are not cut; the
playground call keeps the snippet's 512. No sampling fields are sent, so the
reply is whatever the model profile defaults produce under the grammar. Results
go to --output as a manifest.json and a calls.json in the dataset's own shape,
one entry per call holding the request, the response, the HTTP status, the
served model revision and the round-trip time. The stored request names the
image file and its SHA-256 in place of the base64 bytes. The key comes from
SIE_API_KEY and is never written out.

You do not need a key, and you do not need to run this. The recorded calls are
already published. `python3 fetch.py` downloads them and `python3 score.py`
scores them offline.

The schemas and the expected values come from evidence/inputs/inputs.json and
were fixed before the first model call. This runner cannot change them.
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

TASK = "doc-field-extraction"
ENDPOINT = "https://api.superlinked.com"
MODEL = "Qwen/Qwen3.8-27B-FP8"
PATH = f"/v1/generate/{MODEL.replace('/', '__')}"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    """SHA-256 of sorted-key, compact JSON: stable across file formatting."""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(text.encode("utf-8"))


# Free-text metadata that documents the run without shaping a single score.
INPUTS_METADATA_KEYS = ("note", "registered_utc")


def inputs_digest(inputs: dict[str, Any]) -> str:
    """Hash what decides the score: documents, schemas, expectations, rules.

    Not the file bytes, which a formatter rewrites, and not the prose fields,
    which are edited to describe the run more accurately. score.py computes it
    the same way.
    """
    scored = {key: value for key, value in inputs.items() if key not in INPUTS_METADATA_KEYS}
    return canonical_sha256(scored)


def load_inputs() -> dict[str, Any]:
    if not INPUTS_PATH.is_file():
        raise SystemExit(f"Missing {INPUTS_PATH}. Run: python3 fetch.py")
    inputs = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    # main() indexes these by id, and a dict keeps the LAST row sharing a key.
    # Reject duplicates outright rather than let a second row shadow the case
    # the run is about to send, which nothing downstream would notice.
    seen: set[str] = set()
    for case in inputs["cases"]:
        if case["id"] in seen:
            raise SystemExit(f"Duplicate case {case['id']}")
        seen.add(case["id"])
    return inputs


def image_for(case: dict[str, Any]) -> tuple[bytes, str]:
    """The page image this case sends, refused if it is not the pinned one."""
    path = IMAGES_DIR / case["image"]
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run: python3 fetch.py")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != case["image_sha256"]:
        raise SystemExit(f"{case['id']}: {path.name} is not the image inputs.json pins")
    return data, digest


def calls_for(inputs: dict[str, Any], case: dict[str, Any], only: str | None) -> list[dict[str, Any]]:
    calls = []
    if only in (None, "proof"):
        calls.append(
            {
                "call": "proof",
                "schema": inputs["schemas"][case["schema"]],
                "max_new_tokens": inputs["proof_max_new_tokens"],
            }
        )
    playground = inputs["playground"]
    if only in (None, "playground") and playground["case"] == case["id"]:
        calls.append(
            {
                "call": "playground",
                "schema": json.loads(playground["schema_text"]),
                "max_new_tokens": playground["max_new_tokens"],
            }
        )
    return calls


def body_for(inputs: dict[str, Any], call: dict[str, Any], image: Any, fmt: str) -> dict[str, Any]:
    # Key order matches the curl body the page's snippet builds.
    return {
        "prompt": inputs["prompt"],
        "max_new_tokens": call["max_new_tokens"],
        "images": [{"data": image, "format": fmt}],
        "grammar": {"json_schema": call["schema"], "strict": True},
    }


def request_record(inputs: dict[str, Any], case: dict[str, Any], call: dict[str, Any], digest: str) -> dict[str, Any]:
    """The request as it is stored: the image by reference, not inline."""
    placeholder = f"<base64 of inputs/images/{case['image']}, sha256 {digest}>"
    return {
        "method": "POST",
        "url": ENDPOINT + PATH,
        "model": MODEL,
        "body": body_for(inputs, call, placeholder, case["format"]),
    }


def record(client: SIEClient, inputs: dict[str, Any], case: dict[str, Any], call: dict[str, Any]) -> dict[str, Any]:
    """Send one call and return its calls.json entry."""
    raw, digest = image_for(case)
    stored = request_record(inputs, case, call, digest)
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    result = client.generate(
        MODEL,
        inputs["prompt"],
        max_new_tokens=call["max_new_tokens"],
        images=[{"data": raw, "format": case["format"]}],
        grammar={"json_schema": call["schema"], "strict": True},
    )
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    # The SDK attaches request-scoped metadata to the dict it returns. Drop it,
    # so `response` stays the server's own envelope and nothing else.
    envelope = {key: value for key, value in result.items() if key != "request"}
    response = {"status": 200, "headers": {}, "body": envelope}
    return {
        "slug": f"{case['id']}__{call['call']}",
        "case": case["id"],
        "call": call["call"],
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
        "images": [{"path": f"inputs/images/{case['image']}", "sha256": digest}],
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
    parser.add_argument("--call", choices=["proof", "playground"], help="run one call type")
    parser.add_argument("--show", metavar="CASE", help="print one proof request and exit, sending nothing")
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
        call = calls_for(inputs, case, "proof")[0]
        _, digest = image_for(case)
        print(json.dumps(request_record(inputs, case, call, digest), indent=2, ensure_ascii=False))
        return 0

    cases = [case for case in inputs["cases"] if not args.case or case["id"] == args.case]
    if not cases:
        raise SystemExit(f"Unknown case: {args.case}")
    # An impossible selection, say --call playground on a case that has none,
    # must fail loudly rather than write a run with zero calls.
    selected = [(case, call) for case in cases for call in calls_for(inputs, case, args.call)]
    if not selected:
        raise SystemExit(f"No calls match case={args.case!r} call={args.call!r}; nothing was run")

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
    for case, call in selected:
        print(f"{case['id']}__{call['call']}: POST {PATH}", file=sys.stderr)
        entry = record(client, inputs, case, call)
        entries.append(entry)
        body = entry["response"]["body"]
        usage = body.get("usage") or {}
        print(
            f"  {entry['timing']['duration_ms']:.0f}ms, finish={body.get('finish_reason')},"
            f" tokens out={usage.get('completion_tokens')}",
            file=sys.stderr,
        )

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
            "complete_run": not args.case and not args.call,
            "prompt": inputs["prompt"],
            "calls_recorded": len(entries),
            "inputs_sha256": inputs_digest(inputs),
            "inputs_sha256_rule": (
                "sha256 over json.dumps(inputs minus the 'note' and 'registered_utc' keys, "
                "sort_keys=True, separators=(',', ':'), ensure_ascii=False)"
            ),
        },
        api_key,
    )
    print(f"{len(entries)} calls written to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
