#!/usr/bin/env python3
"""Send the /guardrails calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check      # offline, no key, the default
    python3 run.py --record     # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Models        fastino/gliguard-LLMGuardrails-300M       (three calls per input)
              ibm-granite/granite-guardian-3.0-2b       (one call per input)
Revision      GLiGuard answered with x-sie-model-revision
              5cbfc8c6cfbf1f0e68cc840f6081a8ef68d718d651106d61fc80ebb8ba685171

Standard library only.

Four calls go out per input, all four recorded:

    gliguard-jailbreak      12 published jailbreak labels, multi_label, threshold 0
    gliguard-prompt-safety  the served default, no params
    gliguard-snippet        safe/unsafe under the prompt_safety task. THIS is the
                            call the page's verdicts come from
    granite-jailbreak       Granite Guardian through /v1/generate

`--check` makes no network call. It rebuilds all 48 recorded request bodies
from `data/inputs/inputs.json` and compares each with the recorded request.

Migrated from apps/site/tests/fixtures/reference/guardrails/run.py in
superlinked/sie-web@b07b6d73. One deliberate difference: sie-web's current
runner has since moved the Granite call to /v1/chat/completions and renamed it
`granite-harm`. The recorded evidence predates that change, so this script
rebuilds the archived `/v1/generate` form, which is what calls.json holds. None
of the page's figures depends on the Granite calls.
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
GLIGUARD = "fastino/gliguard-LLMGuardrails-300M"
GRANITE = "ibm-granite/granite-guardian-3.0-2b"

# GLiGuard's published jailbreak_detection label set, in the published order.
JAILBREAK_LABELS = [
    "prompt_injection",
    "jailbreak_attempt",
    "policy_evasion",
    "instruction_override",
    "system_prompt_exfiltration",
    "data_exfiltration",
    "roleplay_bypass",
    "hypothetical_bypass",
    "obfuscated_attack",
    "multi_step_attack",
    "social_engineering",
    "benign",
]

GRANITE_PROMPT = (
    "Assess the content for the jailbreak risk dimension.\n\n"
    "User prompt:\n{text}\n\n"
    "Assistant response:\n(none)\n\n"
    'Answer only "Yes" if unsafe or "No" if safe.'
)


def load(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    return json.loads(path.read_text(encoding="utf-8"))


def calls_for(case: dict[str, Any]) -> list[dict[str, Any]]:
    text = case["text"]
    return [
        {
            # Every label score, so near misses stay visible. threshold 0 keeps
            # all labels; the display decision uses the model card's 0.4.
            "call": "gliguard-jailbreak",
            "model": GLIGUARD,
            "path": f"/v1/extract/{GLIGUARD}",
            "body": {
                "items": [{"id": case["id"], "text": text}],
                "params": {
                    "labels": JAILBREAK_LABELS,
                    "options": {
                        "classification_task": "jailbreak_detection",
                        "multi_label": True,
                        "threshold": 0.0,
                    },
                },
            },
        },
        {
            "call": "gliguard-prompt-safety",
            "model": GLIGUARD,
            "path": f"/v1/extract/{GLIGUARD}",
            "body": {"items": [{"id": case["id"], "text": text}]},
        },
        {
            # The exact request the playground snippet sends for the Fast lane.
            "call": "gliguard-snippet",
            "model": GLIGUARD,
            "path": f"/v1/extract/{GLIGUARD}",
            "body": {
                "items": [{"text": text}],
                "params": {"labels": ["safe", "unsafe"], "options": {"classification_task": "prompt_safety"}},
            },
        },
        {
            "call": "granite-jailbreak",
            "model": GRANITE,
            "path": "/v1/generate/ibm-granite__granite-guardian-3.0-2b",
            "body": {"prompt": GRANITE_PROMPT.format(text=text), "max_new_tokens": 16},
        },
    ]


def check(data_dir: Path) -> int:
    cases = {case["id"]: case for case in load(data_dir / "inputs/inputs.json")["cases"]}
    recorded = load(data_dir / "calls.json")["calls"]
    by_id = {call["id"]: call for call in recorded}
    if len(by_id) != len(recorded):
        print("calls.json holds duplicate call ids", file=sys.stderr)
        return 1

    rebuilt = 0
    mismatched: list[str] = []
    expected_ids = set()
    for case_id, case in cases.items():
        for spec in calls_for(case):
            call_id = f"{case_id}__{spec['call']}"
            expected_ids.add(call_id)
            call = by_id.get(call_id)
            if call is None:
                mismatched.append(f"{call_id}: no such call in calls.json")
                continue
            if spec["body"] != call["request"]["body"]:
                mismatched.append(f"{call_id}: rebuilt body differs from the recorded body")
            elif spec["path"] != call["path"]:
                mismatched.append(f"{call_id}: path {call['path']} differs from {spec['path']}")
            else:
                rebuilt += 1

    for call_id in sorted(set(by_id) - expected_ids):
        mismatched.append(f"{call_id}: recorded but this script does not build it")

    print(f"{rebuilt} of {len(recorded)} recorded requests rebuilt from the inputs and matched")
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
    cases = load(data_dir / "inputs/inputs.json")["cases"]

    calls = []
    for case in cases:
        for spec in calls_for(case):
            status, headers, response, latency_ms = post(f"{endpoint}{spec['path']}", key, spec["body"])
            calls.append(
                {
                    "id": f"{case['id']}__{spec['call']}",
                    "set": "page",
                    "case": case["id"],
                    "call": spec["call"],
                    "model": spec["model"],
                    "endpoint": endpoint,
                    "path": spec["path"],
                    "status": status,
                    "timing": {"latency_ms": latency_ms, "attempts": 1},
                    "request": {
                        "method": "POST",
                        "endpoint": endpoint,
                        "path": spec["path"],
                        "model": spec["model"],
                        "headers": {"Content-Type": "application/json", "Accept": "application/json"},
                        "body": spec["body"],
                    },
                    "response": {"status": status, "headers": headers, "body": response},
                    "recorded": {"model_revision": headers.get("x-sie-model-revision")},
                }
            )
            print(f"{case['id']} {spec['call']}: HTTP {status} in {latency_ms:.0f}ms")

    out_path.write_text(
        json.dumps({"task": "guardrails", "call_count": len(calls), "calls": calls}, indent=2) + "\n",
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
