#!/usr/bin/env python3
"""Send the /structured-output calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check              # offline, no key, nothing installed
    python3 run.py --show <case-id>     # offline, prints one request
    uv sync && uv run python run.py --record    # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Path          /v1/chat/completions
Model         Qwen/Qwen3.8-27B-FP8
Revision      the server returns it in X-Sie-Model-Revision

Calls go through `sie_sdk.SIEClient`, per AGENTS.md. The import is deferred
into main() so `--check` and `--show` run on a bare `python3` with nothing
installed; that is the property worth protecting, because it lets a reader
confirm this runner is the one that produced the evidence for free.

`--check` rebuilds every page-evidence request body from `data/inputs/cases.json`
and compares it with the request recorded in `data/calls.json`. The bodies in
this file were confirmed against the SDK by intercepting the client transport:
`client.chat_completions` puts exactly these 13 bodies on the wire.

The `diagnostics/*` sets in calls.json came from earlier revisions of the
sie-web runner and are NOT rebuilt here; `--check` reports them as not checked
rather than counting them as passes.

Migrated from apps/site/tests/fixtures/reference/structured-output/run.py in
superlinked/sie-web@b07b6d73. The request bodies are unchanged; the inputs now
come from the fetched dataset and the output is one calls.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sie_sdk import SIEClient

ENDPOINT = "https://api.superlinked.com"
PATH = "/v1/chat/completions"
MODEL = "Qwen/Qwen3.8-27B-FP8"
MAX_COMPLETION_TOKENS = 512


def build_body(case: dict[str, Any]) -> dict[str, Any]:
    """The exact body the playground snippet sends for this task."""
    schema = case["schema"]
    system = case["instruction"] + "\n\nRequired JSON schema:\n" + json.dumps(schema, indent=2)
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": case["text"]},
        ],
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "structured_output", "strict": True, "schema": schema},
        },
    }


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def check(data_dir: Path) -> int:
    cases = {case["id"]: case for case in load(data_dir / "inputs/cases.json")["cases"]}
    calls = load(data_dir / "calls.json")["calls"]

    rebuilt = 0
    mismatched: list[str] = []
    not_checked: list[str] = []
    for call in calls:
        if call["set"] != "page":
            not_checked.append(call["id"])
            continue
        case = cases.get(call["case"])
        if case is None:
            mismatched.append(f"{call['id']}: no case in inputs/cases.json")
            continue
        if build_body(case) != call["request"]["body"]:
            mismatched.append(f"{call['id']}: rebuilt body differs from the recorded body")
        elif call["path"] != PATH:
            mismatched.append(f"{call['id']}: path {call['path']} differs from {PATH}")
        else:
            rebuilt += 1

    print(f"{rebuilt} page requests rebuilt from inputs and matched the recorded request")
    if not_checked:
        print(
            f"{len(not_checked)} calls NOT checked: archived diagnostics sweeps "
            "written by earlier runner revisions, which this script does not rebuild"
        )
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    return 1 if mismatched else 0


def record(client: SIEClient, case: dict[str, Any]) -> dict[str, Any]:
    """Send one case through the SDK and return its calls.json entry."""
    body = build_body(case)
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    completion = client.chat_completions(
        body["model"],
        body["messages"],
        max_completion_tokens=body["max_completion_tokens"],
        response_format=body["response_format"],
    )
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    return {
        "id": f"page/{case['id']}",
        "set": "page",
        "case": case["id"],
        "model": MODEL,
        "endpoint": ENDPOINT,
        "path": PATH,
        "status": 200,
        "timing": {"at": requested_at, "latency_ms": latency_ms, "attempts": 1},
        "request": {"method": "POST", "endpoint": ENDPOINT, "path": PATH, "model": MODEL, "body": body},
        # The SDK surfaces no response headers, so a fresh run records none and
        # says so rather than leaving an empty field to be read as "none sent".
        "response": {"status": 200, "body": completion},
        "recorded": {
            "model_revision": client.last_model_revision,
            "retry_count": client.last_retry_count,
            "response_headers": "not surfaced by sie_sdk; the archived run recorded them from raw HTTP",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument("--check", action="store_true", help="offline check, the default")
    parser.add_argument("--show", metavar="CASE", help="print one request and exit, sending nothing")
    parser.add_argument("--record", action="store_true", help="make live calls (needs SIE_API_KEY)")
    parser.add_argument("--out", default="run-output/calls.json", help="where --record writes")
    args = parser.parse_args()

    data_dir = Path(args.data)
    cases = load(data_dir / "inputs/cases.json")["cases"]
    by_id = {case["id"]: case for case in cases}

    if args.show:
        case = by_id.get(args.show)
        if case is None:
            raise SystemExit(f"Unknown case: {args.show}")
        print(json.dumps({"method": "POST", "path": PATH, "body": build_body(case)}, indent=2, ensure_ascii=False))
        return 0

    if not args.record:
        return check(data_dir)

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY to send these calls, or run score.py on the recorded ones instead")
    base_url = os.environ.get("SIE_CLUSTER_URL") or os.environ.get("SIE_BASE_URL") or ENDPOINT
    # Imported here rather than at module scope so --check and --show run on a
    # bare `python3` with nothing installed. Sending needs the SDK:
    # `uv sync`, then `uv run python run.py --record`.
    from sie_sdk import SIEClient  # noqa: PLC0415

    client = SIEClient(base_url, api_key=api_key, timeout_s=900)

    calls = []
    for case in cases:
        entry = record(client, case)
        calls.append(entry)
        print(f"{case['id']}: {entry['timing']['latency_ms']:.0f}ms", file=sys.stderr)

    ids = [entry["id"] for entry in calls]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate call ids; refusing to write a calls.json two checks could read differently")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"task": "structured-output", "call_count": len(calls), "calls": calls}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
