#!/usr/bin/env python3
"""Send the /classify calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check                  # offline, no key, nothing installed
    python3 run.py --show <case-slug>       # offline, prints one request
    uv sync && uv run python run.py --record --set snips     # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Path          /v1/extract/knowledgator/gliclass-large-v3.0
Model         knowledgator/gliclass-large-v3.0
Revision      the served revision came back in x-sie-model-revision as
              10333b84de80b402376b626eb25366fb081d3faeb893eb4b01cf32e8c27e4aff
              on every recorded response. SIE does not document how that value
              maps to a Hugging Face commit, so no mapping is claimed here.

Calls go through `sie_sdk.SIEClient`, per AGENTS.md. The import is deferred into
main() so `--check` and `--show` run on a bare `python3` with nothing installed.

`--check` rebuilds all 62 recorded request bodies from the case files in
`data/inputs/` and compares each with the recorded request. Those bodies were
confirmed against the SDK by intercepting the client transport: `client.extract`
puts exactly these 62 bodies on the wire.

One recorded detail differs from what the SDK sends, and it is a path spelling,
not a body: the 2026-09-15 runner percent-encoded the model id into
`/v1/extract/knowledgator%2Fgliclass-large-v3.0`, while the SDK sends the slash
unencoded. The two unquote to the same path and reach the same endpoint.
`--check` asserts that equivalence rather than hiding it. The other four tasks
in this batch recorded the unencoded form.

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
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sie_sdk import SIEClient

ENDPOINT = "https://api.superlinked.com"
MODEL = "knowledgator/gliclass-large-v3.0"
# What the SDK sends. The archived run recorded the percent-encoded spelling.
PATH = f"/v1/extract/{MODEL}"
RECORDED_PATH = "/v1/extract/knowledgator%2Fgliclass-large-v3.0"

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

    if urllib.parse.unquote(RECORDED_PATH) != PATH:
        mismatched.append(f"recorded path {RECORDED_PATH} does not unquote to {PATH}")

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
            if build_body(case["text"], labels) != call["request"]["body"]:
                mismatched.append(f"{call['id']}: rebuilt body differs from the recorded body")
            elif urllib.parse.unquote(call["path"]) != PATH:
                mismatched.append(f"{call['id']}: path {call['path']} is not {PATH}")
            else:
                rebuilt += 1

    unknown = [call["id"] for call in calls if call["set"] not in SETS]

    print(f"{rebuilt} of {len(calls)} recorded requests rebuilt from the inputs and matched")
    print(f"recorded path {RECORDED_PATH} unquotes to {PATH}, which is what the SDK sends")
    for line in unknown:
        print(f"NOT CHECKED {line}: unknown set", file=sys.stderr)
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    return 1 if mismatched or unknown or rebuilt != len(calls) else 0


def record(client: SIEClient, set_name: str, slug: str, text: str, labels: list[str]) -> dict[str, Any]:
    """Send one case through the SDK and return its calls.json entry."""
    body = build_body(text, labels)
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    result = client.extract(MODEL, body["items"][0], labels=labels)
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    # Rebuild the envelope the archived run recorded, so score.py reads a fresh
    # run the same way. The SDK returns the per-item result, not the server's
    # envelope, and surfaces no response headers; both are said so below.
    item = {key: value for key, value in result.items() if key not in ("model", "request")}
    return {
        "id": f"{set_name}/{slug}",
        "set": set_name,
        "case": slug,
        "model": MODEL,
        "endpoint": ENDPOINT,
        "path": PATH,
        "status": 200,
        "timing": {"at": requested_at, "latency_ms": latency_ms, "attempts": 1},
        "request": {"method": "POST", "endpoint": ENDPOINT, "path": PATH, "model": MODEL, "body": body},
        "response": {
            "status": 200,
            "body": {"items": [item], "model": result.get("model", MODEL)},
            "shape": "rebuilt from the sie_sdk per-item result; the SDK returns no server envelope and no headers",
        },
        "recorded": {
            "served_model_revision_header": client.last_model_revision,
            "retry_count": client.last_retry_count,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument("--check", action="store_true", help="offline check, the default")
    parser.add_argument("--show", metavar="SLUG", help="print one request and exit, sending nothing")
    parser.add_argument("--record", action="store_true", help="make live calls (needs SIE_API_KEY)")
    parser.add_argument("--set", choices=sorted(SETS), default="snips", help="which set --show and --record use")
    parser.add_argument("--out", default="run-output/calls.json", help="where --record writes")
    args = parser.parse_args()

    data_dir = Path(args.data)
    folder, definitions = SETS[args.set]

    if args.show:
        case_file = load(data_dir / f"inputs/{folder}/cases.json")
        by_slug = {case["slug"]: case for case in case_file["cases"]}
        case = by_slug.get(args.show)
        if case is None:
            raise SystemExit(f"Unknown case in set {args.set}: {args.show}")
        body = build_body(case["text"], labels_for(case_file, definitions))
        print(json.dumps({"method": "POST", "path": PATH, "body": body}, indent=2, ensure_ascii=False))
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

    case_file = load(data_dir / f"inputs/{folder}/cases.json")
    labels = labels_for(case_file, definitions)
    calls = []
    for case in case_file["cases"]:
        entry = record(client, args.set, case["slug"], case["text"], labels)
        calls.append(entry)
        print(f"{case['slug']}: {entry['timing']['latency_ms']:.0f}ms", file=sys.stderr)

    ids = [entry["id"] for entry in calls]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate call ids; refusing to write a calls.json two checks could read differently")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"task": "classify", "call_count": len(calls), "calls": calls}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
