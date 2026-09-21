#!/usr/bin/env python3
"""Send the /redact calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check             # offline, no key, nothing installed
    python3 run.py --show <case-id>    # offline, prints one request
    uv sync && uv run python run.py --record    # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Path          /v1/extract/<model>
Models        urchade/gliner_multi_pii-v1   (the model the page shows)
              numind/NuNER_Zero             (recorded over the same documents)
Revision      every response carried x-sie-model-revision
              10333b84de80b402376b626eb25366fb081d3faeb893eb4b01cf32e8c27e4aff

Calls go through `sie_sdk.SIEClient`, per AGENTS.md. The import is deferred into
main() so `--check` and `--show` run on a bare `python3` with nothing installed.

Twelve documents go to both models, so 24 calls. Each request names the ID
types that document uses, as plain strings, and the response gives back exact
character offsets.

Two of the twelve documents are tails: the part of a long document past word
384, sent as a second request. `parent_case` and `parent_offset` in
inputs/cases.json say which document they came from and where they start.

`--check` rebuilds all 24 recorded request bodies from `data/inputs/cases.json`
and compares each with the recorded request. Those bodies were confirmed against
the SDK by intercepting the client transport: `client.extract` puts exactly these
24 bodies on the wire, at exactly these paths.

Migrated from apps/site/tests/fixtures/reference/redact/run.py in
superlinked/sie-web@b07b6d73.
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
MODELS = ["urchade/gliner_multi_pii-v1", "numind/NuNER_Zero"]


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def build_body(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": [{"id": case["id"], "text": case["text"]}],
        "params": {"labels": case["labels"]},
    }


def check(data_dir: Path) -> int:
    cases = {case["id"]: case for case in load(data_dir / "inputs/cases.json")["cases"]}
    calls = load(data_dir / "calls.json")["calls"]

    rebuilt = 0
    mismatched: list[str] = []
    for call in calls:
        case = cases.get(call["case"])
        if case is None:
            mismatched.append(f"{call['id']}: no case with that id in inputs/cases.json")
            continue
        if call["model"] not in MODELS:
            mismatched.append(f"{call['id']}: unexpected model {call['model']}")
            continue
        if build_body(case) != call["request"]["body"]:
            mismatched.append(f"{call['id']}: rebuilt body differs from the recorded body")
            continue
        expected_path = f"/v1/extract/{call['model']}"
        if call["path"] != expected_path:
            mismatched.append(f"{call['id']}: path {call['path']} differs from {expected_path}")
            continue
        rebuilt += 1

    print(f"{rebuilt} of {len(calls)} recorded requests rebuilt from the inputs and matched")
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    return 1 if mismatched else 0


def record(client: SIEClient, model: str, case: dict[str, Any]) -> dict[str, Any]:
    """Send one document through the SDK and return its calls.json entry."""
    body = build_body(case)
    path = f"/v1/extract/{model}"
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    result = client.extract(model, body["items"][0], labels=body["params"]["labels"])
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    # Rebuild the envelope the archived run recorded, so score.py reads a fresh
    # run the same way. The SDK returns the per-item result, not the server's
    # envelope, and surfaces no response headers; both are said so below.
    item = {key: value for key, value in result.items() if key not in ("model", "request")}
    return {
        "id": f"{model.replace('/', '__')}/{case['id']}",
        "set": model.replace("/", "__"),
        "case": case["id"],
        "model": model,
        "endpoint": ENDPOINT,
        "path": path,
        "status": 200,
        "timing": {"at": requested_at, "latency_ms": latency_ms, "attempts": 1},
        "request": {"method": "POST", "endpoint": ENDPOINT, "path": path, "model": model, "body": body},
        "response": {
            "status": 200,
            "body": {"items": [item], "model": result.get("model", model)},
            "shape": "rebuilt from the sie_sdk per-item result; the SDK returns no server envelope and no headers",
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
        body = build_body(case)
        shown = [{"model": model, "method": "POST", "path": f"/v1/extract/{model}", "body": body} for model in MODELS]
        print(json.dumps(shown, indent=2, ensure_ascii=False))
        return 0

    if not args.record:
        return check(data_dir)

    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY to send these calls, or run score.py on the recorded ones instead")
    base_url = os.environ.get("SIE_CLUSTER_URL") or os.environ.get("SIE_BASE_URL") or ENDPOINT
    # Deferred so --check and --show run on a bare `python3` with nothing installed.
    from sie_sdk import SIEClient  # noqa: PLC0415

    client = SIEClient(base_url, api_key=api_key, timeout_s=900)

    calls = []
    for model in MODELS:
        for case in cases:
            entry = record(client, model, case)
            calls.append(entry)
            print(f"{model} {case['id']}: {entry['timing']['latency_ms']:.0f}ms", file=sys.stderr)

    ids = [entry["id"] for entry in calls]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate call ids; refusing to write a calls.json two checks could read differently")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"task": "redact", "call_count": len(calls), "calls": calls}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
