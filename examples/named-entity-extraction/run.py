#!/usr/bin/env python3
"""Send the /named-entities calls to a SIE deployment, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check            # offline, no key, nothing installed
    python3 run.py --show <case-id>   # offline, prints one call
    uv sync && uv run python run.py --record   # live, needs a GLiNER endpoint

Model         urchade/gliner_multi-v2.1
Path          /v1/extract/urchade/gliner_multi-v2.1
Recorded run  public SIE v0.6.23 on an L4 in Modal, 2026-07-24. The run did not
              save its base URL, so calls.json and manifest.json record the
              runtime, the server commit and the hardware instead of an endpoint.

Calls go through `sie_sdk.SIEClient`, per AGENTS.md. The import is deferred into
main() so `--check` and `--show` run on a bare `python3` with nothing installed.

`--check` rebuilds all four recorded call envelopes from `data/inputs/cases.json`
and compares each with the recorded one. It is a bijection, not a walk over what
is there: the expected call ids come from the inputs, one per case, so a call
that is missing, recorded twice or implied by no case fails the check.

What the envelope records is the SDK call, not the wire body: `client.extract`
takes an Item and a label list, and `source_id` and `source_excerpt_sha256` are
provenance fields the SIE payload does not carry.

The CMS response keeps the model's incorrect `proof of delivery` span under the
`missing documentation` label. It is not a required anchor, and leaving it in
makes label and threshold tradeoffs easier to inspect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sie_sdk import SIEClient

MODEL = "urchade/gliner_multi-v2.1"
PATH = f"/v1/extract/{MODEL}"
DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
REQUEST_SHAPE = (
    "the SDK call envelope run.py builds, not the wire body. "
    "source_id and source_excerpt_sha256 are provenance fields "
    "the SIE payload does not carry."
)
RESPONSE_SHAPE = (
    "the per-item result sie_sdk returned, unmodified. The SDK surfaces no server envelope and no response headers."
)


class CallFailedError(Exception):
    """A call that did not produce a usable result."""


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def to_jsonable(value: Any) -> Any:
    """What the SDK returned, as plain JSON types and nothing else."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    return value


def build_envelope(case_id: str, case: dict[str, Any]) -> dict[str, Any]:
    """The SDK call this example makes for one case, as it is recorded."""
    source = case["source"]
    return {
        "method": "SIEClient.extract",
        "endpoint": PATH,
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


def failure_entry(case_id: str, endpoint: str, envelope: dict[str, Any], error: BaseException) -> dict[str, Any]:
    """What a failed call records.

    Every field the success path writes, so that `check` and `score.py` read a
    calls.json holding failures instead of raising KeyError on it. Only the
    values differ: the status never reads as success, the response is null and
    `error` says what went wrong.
    """
    return {
        "id": f"page/{case_id}",
        "set": "page",
        "case": case_id,
        "model": MODEL,
        "endpoint": endpoint,
        "path": PATH,
        "status": "error",
        "error": {"type": type(error).__name__, "message": str(error)},
        "timing": {"latency_ms": None, "attempts": 1},
        "request": {"method": "SIEClient.extract", "shape": REQUEST_SHAPE, "body": envelope},
        "response": None,
        "recorded": {},
    }


def check(data_dir: Path) -> int:
    cases = load(data_dir / "inputs/cases.json")["cases"]
    calls = load(data_dir / "calls.json")["calls"]

    expected = {f"page/{case_id}": (case_id, case) for case_id, case in cases.items()}
    recorded: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for call in calls:
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
        case_id, case = expected[call_id]
        if call["case"] != case_id:
            mismatched.append(f"{call_id}: case {call['case']} differs from {case_id}")
        elif call["model"] != MODEL:
            mismatched.append(f"{call_id}: model {call['model']} differs from {MODEL}")
        elif call["path"] != PATH:
            mismatched.append(f"{call_id}: path {call['path']} differs from {PATH}")
        elif build_envelope(case_id, case) != call["request"]["body"]:
            mismatched.append(f"{call_id}: rebuilt envelope differs from the recorded one")
        else:
            rebuilt += 1

    print(f"{rebuilt} of {len(calls)} recorded calls rebuilt from the inputs and matched")
    print(f"{len(expected)} calls expected from {len(cases)} cases, {len(recorded)} recorded")
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


def record(client: SIEClient, endpoint: str, case_id: str, case: dict[str, Any]) -> dict[str, Any]:
    """Send one case through the SDK and return its calls.json entry."""
    from sie_sdk import Item  # noqa: PLC0415

    envelope = build_envelope(case_id, case)
    started = time.monotonic()
    result = client.extract(
        MODEL,
        Item(id=f"{case_id}-source", text=case["text"]),
        labels=case["labels"],
        wait_for_capacity=True,
        provision_timeout_s=900,
    )
    latency_ms = round((time.monotonic() - started) * 1000, 3)
    body = to_jsonable(result)
    # A 200 can still carry a per-item failure. Never record one as a result.
    if not isinstance(body, dict) or body.get("error"):
        raise CallFailedError(f"{case_id}: item error {body.get('error') if isinstance(body, dict) else body!r}")
    if "entities" not in body:
        raise CallFailedError(f"{case_id}: response carried no entities field")
    return {
        "id": f"page/{case_id}",
        "set": "page",
        "case": case_id,
        "model": MODEL,
        "endpoint": endpoint,
        "path": PATH,
        "status": 200,
        "timing": {"latency_ms": latency_ms, "attempts": 1},
        "request": {"method": "SIEClient.extract", "shape": REQUEST_SHAPE, "body": envelope},
        "response": {"status": 200, "shape": RESPONSE_SHAPE, "body": body},
        "recorded": {"model_revision": client.last_model_revision, "retry_count": client.last_retry_count},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument("--check", action="store_true", help="offline check, the default")
    parser.add_argument("--show", metavar="CASE", help="print one call and exit, sending nothing")
    parser.add_argument("--record", action="store_true", help="make live calls")
    parser.add_argument("--out", default="run-output/calls.json", help="where --record writes")
    args = parser.parse_args()

    data_dir = Path(args.data)
    cases = load(data_dir / "inputs/cases.json")["cases"]

    if args.show:
        if args.show not in cases:
            raise SystemExit(f"Unknown case: {args.show}. Known: {', '.join(cases)}")
        print(json.dumps(build_envelope(args.show, cases[args.show]), indent=2, ensure_ascii=False))
        return 0

    if not args.record:
        return check(data_dir)

    # Sanity-check the excerpts before spending anything on them: a rewritten
    # excerpt must not reach the model as though it were a source quotation.
    excerpts = load(data_dir / "inputs/sources.json")["excerpts"]
    for case_id, case in cases.items():
        canonical = excerpts.get(case_id)
        if canonical is None or sha256_text(case["text"]) != canonical["sha256"]:
            raise SystemExit(f"{case_id}: excerpt does not match inputs/sources.json")

    endpoint = os.environ.get("SIE_CLUSTER_URL") or os.environ.get("SIE_BASE_URL") or DEFAULT_ENDPOINT
    api_key = os.environ.get("SIE_API_KEY") or None
    # Deferred so --check and --show run on a bare `python3` with nothing installed.
    from sie_sdk import SIEClient  # noqa: PLC0415

    client = SIEClient(endpoint, api_key=api_key, timeout_s=900)

    calls = []
    failed: list[str] = []
    for case_id, case in cases.items():
        try:
            entry = record(client, endpoint, case_id, case)
        except Exception as error:  # noqa: BLE001
            failed.append(f"page/{case_id}: {type(error).__name__}: {error}")
            calls.append(failure_entry(case_id, endpoint, build_envelope(case_id, case), error))
            print(f"{case_id}: FAILED {type(error).__name__}", file=sys.stderr)
            continue
        calls.append(entry)
        print(f"{case_id}: {entry['timing']['latency_ms']:.0f}ms", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "task": "named-entity-extraction",
                "call_count": len(calls),
                "failed_calls": len(failed),
                "complete": not failed,
                "calls": calls,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    if failed:
        # A run that failed must not look like a run that succeeded.
        print(f"{len(failed)} of {len(cases)} calls FAILED:", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
