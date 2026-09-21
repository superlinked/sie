#!/usr/bin/env python3
"""Send the /structured-output calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check        # offline, no key, the default
    python3 run.py --record       # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Path          /v1/chat/completions
Model         Qwen/Qwen3.8-27B-FP8
Revision      the server returns it in X-Sie-Model-Revision; --record stores it

Standard library only.

`--check` makes no network call. It rebuilds every page-evidence request body
from `data/inputs/cases.json` and compares it with the request recorded in
`data/calls.json`. A reader who wants to know whether this script is really the
script that produced the evidence can run it with no API key and no spend.

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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

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
        if build_body(case) == call["request"]["body"]:
            rebuilt += 1
        else:
            mismatched.append(f"{call['id']}: rebuilt body differs from the recorded body")

    print(f"{rebuilt} page requests rebuilt from inputs and matched the recorded request")
    if not_checked:
        print(
            f"{len(not_checked)} calls NOT checked: archived diagnostics sweeps "
            "written by earlier runner revisions, which this script does not rebuild"
        )
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    return 1 if mismatched else 0


def api_key() -> str:
    key = os.environ.get("SIE_API_KEY", "").strip()
    if not key:
        raise SystemExit("--record needs SIE_API_KEY. Use --check for the offline check.")
    return key


def post(url: str, key: str, body: dict[str, Any]) -> tuple[int, dict[str, str], Any, float]:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
            raw = response.read()
            status = response.status
            headers = dict(response.headers)
    except urllib.error.HTTPError as error:
        raw = error.read()
        status = error.code
        headers = dict(error.headers)
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    return status, headers, json.loads(raw), latency_ms


def record(data_dir: Path, out_path: Path) -> int:
    key = api_key()
    endpoint = (os.environ.get("SIE_CLUSTER_URL") or os.environ.get("SIE_BASE_URL") or ENDPOINT).rstrip("/")
    cases = load(data_dir / "inputs/cases.json")["cases"]
    calls = []
    for case in cases:
        body = build_body(case)
        status, headers, response, latency_ms = post(f"{endpoint}{PATH}", key, body)
        calls.append(
            {
                "id": f"page/{case['id']}",
                "set": "page",
                "case": case["id"],
                "model": MODEL,
                "endpoint": endpoint,
                "path": PATH,
                "status": status,
                "timing": {"latency_ms": latency_ms, "attempts": 1},
                "request": {
                    "method": "POST",
                    "endpoint": endpoint,
                    "path": PATH,
                    "model": MODEL,
                    "body": body,
                },
                "response": {"status": status, "headers": headers, "body": response},
                "recorded": {
                    "model_revision": headers.get("X-Sie-Model-Revision"),
                    "server_version": headers.get("X-Sie-Server-Version"),
                },
            }
        )
        print(f"{case['id']}: HTTP {status} in {latency_ms:.0f}ms")
    out_path.write_text(
        json.dumps({"task": "structured-output", "call_count": len(calls), "calls": calls}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument("--check", action="store_true", help="offline check, the default")
    parser.add_argument("--record", action="store_true", help="make live calls (needs SIE_API_KEY)")
    parser.add_argument("--out", default="run-output/calls.json", help="where --record writes")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not args.record:
        return check(data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return record(data_dir, out_path)


if __name__ == "__main__":
    sys.exit(main())
