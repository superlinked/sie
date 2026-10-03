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

It is a bijection, not a walk over what is there: the expected call ids come
from the inputs, so a page call that is missing, recorded twice, or implied by
no case fails the check.

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


class CallFailedError(Exception):
    """A call that did not produce a usable result.

    The SDK raises for a transport or HTTP failure. This covers the other
    half: a 200 whose item carries an `error`, which must never be recorded
    as though it were a result.
    """


def failure_entry(
    call_id: str,
    set_name: str,
    case_id: str,
    model: str,
    path: str,
    body: dict[str, Any],
    error: BaseException,
    call_name: str | None = None,
) -> dict[str, Any]:
    """What a failed call records.

    Every field the success path writes, so that `check` and `score.py` can
    read a calls.json holding failures instead of raising KeyError on it. Only
    the values differ: the status never reads as success, the response is null
    and `error` says what went wrong. A recorder and a reader that disagree
    about shape is how a failed run gets mistaken for a missing one.
    """
    entry: dict[str, Any] = {
        "id": call_id,
        "set": set_name,
        "case": case_id,
    }
    if call_name is not None:
        entry["call"] = call_name
    entry.update(
        {
            "model": model,
            "endpoint": ENDPOINT,
            "path": path,
            "status": "error",
            "error": {"type": type(error).__name__, "message": str(error)},
            "timing": {"at": datetime.now(UTC).isoformat(timespec="seconds"), "latency_ms": None, "attempts": 1},
            "request": {"method": "POST", "endpoint": ENDPOINT, "path": path, "model": model, "body": body},
            "response": None,
            "recorded": {},
        }
    )
    return entry


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
    """Compare the recorded page calls with the ones the inputs imply.

    A bijection, not a walk over what happens to be there: the expected call
    ids come from inputs/cases.json, so a call that is missing, recorded twice
    or not derivable from any case all fail. Checking only the calls present
    would pass a calls.json with one of them deleted.
    """
    cases = {case["id"]: case for case in load(data_dir / "inputs/cases.json")["cases"]}
    calls = load(data_dir / "calls.json")["calls"]

    # Keyed by call id, so the case a call is checked against comes from the
    # id it was recorded under, never from a field inside the call itself.
    expected = {f"page/{case_id}": case for case_id, case in cases.items()}

    recorded: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    not_checked: list[str] = []
    for call in calls:
        if call["set"] != "page":
            not_checked.append(call["id"])
            continue
        if call["id"] in recorded:
            duplicates.append(call["id"])
            continue
        recorded[call["id"]] = call

    missing = sorted(set(expected) - set(recorded))
    unexpected = sorted(set(recorded) - set(expected))

    rebuilt = 0
    mismatched: list[str] = []
    for call_id in sorted(set(expected) & set(recorded)):
        call = recorded[call_id]
        case = expected[call_id]
        if call["case"] != case["id"]:
            mismatched.append(f"{call_id}: case {call['case']} differs from {case['id']}")
        elif build_body(case) != call["request"]["body"]:
            mismatched.append(f"{call_id}: rebuilt body differs from the recorded body")
        elif call["path"] != PATH:
            mismatched.append(f"{call_id}: path {call['path']} differs from {PATH}")
        else:
            rebuilt += 1

    print(f"{rebuilt} page requests rebuilt from inputs and matched the recorded request")
    print(f"{len(expected)} calls expected from inputs/cases.json, {len(recorded)} recorded in the page set")
    if not_checked:
        print(
            f"{len(not_checked)} calls NOT checked: archived diagnostics sweeps "
            "written by earlier runner revisions, which this script does not rebuild"
        )
    for call_id in missing:
        print(f"MISSING {call_id}: expected from the inputs, absent from calls.json", file=sys.stderr)
    for call_id in duplicates:
        print(f"DUPLICATE {call_id}: recorded more than once", file=sys.stderr)
    for call_id in unexpected:
        print(f"UNEXPECTED {call_id}: recorded but no case in inputs/cases.json implies it", file=sys.stderr)
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    if missing or duplicates or unexpected or mismatched:
        return 1
    return 0


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
    # A 200 is not a result. Refuse anything without a usable message.
    choices = completion.get("choices") or []
    if not choices or not (choices[0].get("message") or {}).get("content"):
        raise CallFailedError(f"{case['id']}: response carried no assistant message")
    # chat_completions returns the server's own envelope, unlike extract and
    # encode, which return a per-item result. The one thing the SDK adds is
    # request-scoped metadata under `request`, which the archived body has no
    # counterpart for; dropping it leaves the envelope the gateway sent.
    envelope = {key: value for key, value in completion.items() if key != "request"}
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
        "response": {
            "status": 200,
            "body": envelope,
            "shape": (
                "the server's own chat completion envelope as sie_sdk returns it, "
                "with the SDK's request-scoped metadata dropped; the SDK surfaces "
                "no response headers, which the archived run recorded from raw HTTP"
            ),
        },
        "recorded": {
            "model_revision": client.last_model_revision,
            "retry_count": client.last_retry_count,
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
    failed: list[str] = []
    for case in cases:
        try:
            entry = record(client, case)
        except Exception as error:  # noqa: BLE001
            failed.append(f"page/{case['id']}: {type(error).__name__}: {error}")
            calls.append(failure_entry(f"page/{case['id']}", "page", case["id"], MODEL, PATH, build_body(case), error))
            print(f"{case['id']}: FAILED {type(error).__name__}", file=sys.stderr)
            continue
        calls.append(entry)
        print(f"{case['id']}: {entry['timing']['latency_ms']:.0f}ms", file=sys.stderr)

    ids = [entry["id"] for entry in calls]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate call ids; refusing to write a calls.json two checks could read differently")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "task": "structured-output",
                "call_count": len(calls),
                "failed_calls": len(failed),
                "complete": not failed,
                "calls": calls,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    if failed:
        # A run that failed must not look like a run that succeeded.
        print(f"{len(failed)} of {len(calls)} calls FAILED:", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        print(f'{out_path} records them with status "error" and is not a complete run', file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
