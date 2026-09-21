#!/usr/bin/env python3
"""Send the /classify calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check                 # offline, no key, the default
    python3 run.py --record --set snips    # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Path          /v1/extract/knowledgator%2Fgliclass-large-v3.0
Model         knowledgator/gliclass-large-v3.0
Revision      the served revision came back in x-sie-model-revision as
              10333b84de80b402376b626eb25366fb081d3faeb893eb4b01cf32e8c27e4aff
              on every recorded response. SIE does not document how that value
              maps to a Hugging Face commit, so no mapping is claimed here.

Standard library only.

`--check` makes no network call. It rebuilds all 62 recorded request bodies
from the case files in `data/inputs/` and compares each with the request
recorded in `data/calls.json`.

The five sets:

    snips                 14 SNIPS validation requests, 7 action labels. The page.
    clinc150              12 CLINC150 messages, 8 domain queue names.
    clinc150-definitions  the same 12 against intent-list label strings.
    cfpb                  12 CFPB complaint narratives, 10 product queues.
    cfpb-definitions      the same 12 against CFPB sub-product strings.

Migrated from apps/site/tests/fixtures/reference/classify/run.py in
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
MODEL = "knowledgator/gliclass-large-v3.0"
PATH = "/v1/extract/knowledgator%2Fgliclass-large-v3.0"

SETS = {
    "snips": ("snips", False),
    "clinc150": ("clinc150", False),
    "clinc150-definitions": ("clinc150", True),
    "cfpb": ("cfpb", False),
    "cfpb-definitions": ("cfpb", True),
}


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def labels_for(case_file: dict[str, Any], definitions: bool) -> list[str]:
    if definitions:
        return [entry["label"] for entry in case_file["definition_labels"]]
    return list(case_file["labels"])


def build_body(text: str, labels: list[str]) -> dict[str, Any]:
    return {"items": [{"text": text}], "params": {"labels": labels}}


def check(data_dir: Path) -> int:
    calls = load(data_dir / "calls.json")["calls"]
    rebuilt = 0
    mismatched: list[str] = []
    unknown: list[str] = []

    for set_name, (folder, definitions) in SETS.items():
        case_file = load(data_dir / f"inputs/{folder}/cases.json")
        by_slug = {case["slug"]: case for case in case_file["cases"]}
        labels = labels_for(case_file, definitions)
        for call in calls:
            if call["set"] != set_name:
                continue
            case = by_slug.get(call["case"])
            if case is None:
                mismatched.append(f"{call['id']}: no case with that slug in inputs/{folder}/cases.json")
                continue
            if build_body(case["text"], labels) == call["request"]["body"]:
                rebuilt += 1
            else:
                mismatched.append(f"{call['id']}: rebuilt body differs from the recorded body")

    for call in calls:
        if call["set"] not in SETS:
            unknown.append(call["id"])

    print(f"{rebuilt} of {len(calls)} recorded requests rebuilt from the inputs and matched")
    for line in unknown:
        print(f"NOT CHECKED {line}: unknown set", file=sys.stderr)
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    return 1 if mismatched or unknown or rebuilt != len(calls) else 0


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


def record(data_dir: Path, set_name: str, out_path: Path) -> int:
    key = os.environ.get("SIE_API_KEY", "").strip()
    if not key:
        raise SystemExit("--record needs SIE_API_KEY. Use --check for the offline check.")
    folder, definitions = SETS[set_name]
    endpoint = (os.environ.get("SIE_CLUSTER_URL") or os.environ.get("SIE_BASE_URL") or ENDPOINT).rstrip("/")
    case_file = load(data_dir / f"inputs/{folder}/cases.json")
    labels = labels_for(case_file, definitions)

    calls = []
    for case in case_file["cases"]:
        body = build_body(case["text"], labels)
        status, headers, response, latency_ms = post(f"{endpoint}{PATH}", key, body)
        calls.append(
            {
                "id": f"{set_name}/{case['slug']}",
                "set": set_name,
                "case": case["slug"],
                "model": MODEL,
                "endpoint": endpoint,
                "path": PATH,
                "status": status,
                "timing": {"latency_ms": latency_ms, "attempts": 1},
                "request": {"method": "POST", "endpoint": endpoint, "path": PATH, "model": MODEL, "body": body},
                "response": {"status": status, "headers": headers, "body": response},
                "recorded": {"served_model_revision_header": headers.get("x-sie-model-revision")},
            }
        )
        print(f"{case['slug']}: HTTP {status} in {latency_ms:.0f}ms")

    out_path.write_text(
        json.dumps({"task": "classify", "call_count": len(calls), "calls": calls}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument("--check", action="store_true", help="offline check, the default")
    parser.add_argument("--record", action="store_true", help="make live calls (needs SIE_API_KEY)")
    parser.add_argument("--set", choices=sorted(SETS), default="snips", help="which set --record sends")
    parser.add_argument("--out", default="run-output/calls.json", help="where --record writes")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not args.record:
        return check(data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return record(data_dir, args.set, out_path)


if __name__ == "__main__":
    sys.exit(main())
