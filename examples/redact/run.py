#!/usr/bin/env python3
"""Send the /redact calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check      # offline, no key, the default
    python3 run.py --record     # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Path          /v1/extract/<model>
Models        urchade/gliner_multi_pii-v1   (the model the page shows)
              numind/NuNER_Zero             (recorded over the same documents)
Revision      every response carried x-sie-model-revision
              10333b84de80b402376b626eb25366fb081d3faeb893eb4b01cf32e8c27e4aff

Standard library only.

Twelve documents go to both models, so 24 calls. Each request names the ID
types that document uses, as plain strings, and the response gives back exact
character offsets.

Two of the twelve documents are tails: the part of a long document past word
384, sent as a second request. `parent_case` and `parent_offset` in
inputs/cases.json say which document they came from and where they start.

`--check` makes no network call. It rebuilds all 24 recorded request bodies
from `data/inputs/cases.json` and compares each with the recorded request.

Migrated from apps/site/tests/fixtures/reference/redact/run.py in
superlinked/sie-web@b07b6d73.
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
        if build_body(case) != call["request"]["body"]:
            mismatched.append(f"{call['id']}: rebuilt body differs from the recorded body")
            continue
        if call["model"] not in MODELS:
            mismatched.append(f"{call['id']}: unexpected model {call['model']}")
            continue
        rebuilt += 1

    print(f"{rebuilt} of {len(calls)} recorded requests rebuilt from the inputs and matched")
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    return 1 if mismatched else 0


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
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
            raw, status, headers = response.read(), response.status, dict(response.headers)
    except urllib.error.HTTPError as error:
        raw, status, headers = error.read(), error.code, dict(error.headers)
    return status, headers, json.loads(raw), round((time.monotonic() - started) * 1000, 1)


def record(data_dir: Path, out_path: Path) -> int:
    key = os.environ.get("SIE_API_KEY", "").strip()
    if not key:
        raise SystemExit("--record needs SIE_API_KEY. Use --check for the offline check.")
    endpoint = (os.environ.get("SIE_CLUSTER_URL") or os.environ.get("SIE_BASE_URL") or ENDPOINT).rstrip("/")
    cases = load(data_dir / "inputs/cases.json")["cases"]

    calls = []
    for model in MODELS:
        path = f"/v1/extract/{model}"
        for case in cases:
            body = build_body(case)
            status, headers, response, latency_ms = post(f"{endpoint}{path}", key, body)
            calls.append(
                {
                    "id": f"{model.replace('/', '__')}/{case['id']}",
                    "set": model.replace("/", "__"),
                    "case": case["id"],
                    "model": model,
                    "endpoint": endpoint,
                    "path": path,
                    "status": status,
                    "timing": {"latency_ms": latency_ms, "attempts": 1},
                    "request": {
                        "method": "POST",
                        "endpoint": endpoint,
                        "path": path,
                        "model": model,
                        "headers": {"Accept": "application/json", "Content-Type": "application/json"},
                        "body": body,
                    },
                    "response": {"status": status, "headers": headers, "body": response},
                    "recorded": {"model_revision": headers.get("x-sie-model-revision")},
                }
            )
            print(f"{model} {case['id']}: HTTP {status} in {latency_ms:.0f}ms")

    out_path.write_text(
        json.dumps({"task": "redact", "call_count": len(calls), "calls": calls}, indent=2) + "\n",
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
