#!/usr/bin/env python3
"""Send the /guardrails calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check             # offline, no key, nothing installed
    python3 run.py --show <case-id>    # offline, prints one input's 4 requests
    uv sync && uv run python run.py --record    # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Models        fastino/gliguard-LLMGuardrails-300M       (three calls per input)
              ibm-granite/granite-guardian-3.0-2b       (one call per input)
Revision      GLiGuard answered with x-sie-model-revision
              5cbfc8c6cfbf1f0e68cc840f6081a8ef68d718d651106d61fc80ebb8ba685171

Calls go through `sie_sdk.SIEClient`, per AGENTS.md. The import is deferred into
main() so `--check` and `--show` run on a bare `python3` with nothing installed.

Four calls go out per input, all four recorded:

    gliguard-jailbreak      12 published jailbreak labels, multi_label, threshold 0
    gliguard-prompt-safety  the served default, no params
    gliguard-snippet        safe/unsafe under the prompt_safety task. THIS is the
                            call the page's verdicts come from
    granite-jailbreak       Granite Guardian through /v1/generate

`--check` rebuilds all 48 recorded request bodies from `data/inputs/inputs.json`
and compares each with the recorded request. Those bodies were confirmed against
the SDK by intercepting the client transport: `client.extract` and
`client.generate` put exactly these 48 bodies on the wire, at exactly these
paths, including the `__` model spelling `client.generate` normalises to.

It is a bijection, not a walk over what is there: the expected call ids come
from the inputs, four per input, so a call that is missing, recorded twice, or
implied by no input fails the check.

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
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sie_sdk import SIEClient

ENDPOINT = "https://api.superlinked.com"
GLIGUARD = "fastino/gliguard-LLMGuardrails-300M"
GRANITE = "ibm-granite/granite-guardian-3.0-2b"
# client.generate normalises the HF-style id to this spelling, which is what the
# 2026-09-15 run recorded.
GRANITE_PATH = "/v1/generate/ibm-granite__granite-guardian-3.0-2b"

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


class CallFailedError(Exception):
    """A call that did not produce a usable result.

    The SDK raises for a transport or HTTP failure. This covers the other
    half: a 200 whose item carries an `error`, which must never be recorded
    as though it were a result.
    """


def failure_entry(
    call_id: str, case_id: str, model: str, path: str, body: dict[str, Any], error: BaseException
) -> dict[str, Any]:
    """What a failed call records. Never a status that reads as success."""
    return {
        "id": call_id,
        "case": case_id,
        "model": model,
        "endpoint": ENDPOINT,
        "path": path,
        "status": "error",
        "error": {"type": type(error).__name__, "message": str(error)},
        "request": {"method": "POST", "endpoint": ENDPOINT, "path": path, "model": model, "body": body},
        "response": None,
    }


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
            "path": GRANITE_PATH,
            "body": {"prompt": GRANITE_PROMPT.format(text=text), "max_new_tokens": 16},
        },
    ]


def check(data_dir: Path) -> int:
    """Compare the recorded calls with the ones the inputs imply.

    A bijection, not a walk over what happens to be there: the expected call
    ids come from inputs/inputs.json, four per input, so a call that is
    missing, recorded twice or not derivable from any input all fail. Checking
    only the calls present would pass a calls.json with one of them deleted.
    """
    cases = load(data_dir / "inputs/inputs.json")["cases"]
    calls = load(data_dir / "calls.json")["calls"]

    expected: dict[str, dict[str, Any]] = {}
    for case in cases:
        for spec in calls_for(case):
            expected[f"{case['id']}__{spec['call']}"] = spec

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
        spec = expected[call_id]
        case_id, call_name = call_id.split("__", 1)
        if call["case"] != case_id:
            mismatched.append(f"{call_id}: case {call['case']} differs from {case_id}")
        elif call["call"] != call_name:
            mismatched.append(f"{call_id}: call {call['call']} differs from {call_name}")
        elif call["model"] != spec["model"]:
            mismatched.append(f"{call_id}: model {call['model']} differs from {spec['model']}")
        elif spec["body"] != call["request"]["body"]:
            mismatched.append(f"{call_id}: rebuilt body differs from the recorded body")
        elif spec["path"] != call["path"]:
            mismatched.append(f"{call_id}: path {call['path']} differs from {spec['path']}")
        else:
            rebuilt += 1

    print(f"{rebuilt} of {len(calls)} recorded requests rebuilt from the inputs and matched")
    print(f"{len(expected)} calls expected from {len(cases)} inputs at 4 calls each, {len(recorded)} recorded")
    for call_id in missing:
        print(f"MISSING {call_id}: expected from the inputs, absent from calls.json", file=sys.stderr)
    for call_id in duplicates:
        print(f"DUPLICATE {call_id}: recorded more than once", file=sys.stderr)
    for call_id in unexpected:
        print(f"UNEXPECTED {call_id}: recorded but no input in inputs/inputs.json implies it", file=sys.stderr)
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    if missing or duplicates or unexpected or mismatched:
        return 1
    return 0


def record(client: SIEClient, case: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Send one call through the SDK and return its calls.json entry."""
    body = spec["body"]
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    if spec["call"] == "granite-jailbreak":
        result = client.generate(GRANITE, body["prompt"], max_new_tokens=body["max_new_tokens"])
        if result.get("finish_reason") == "error" or result.get("text") is None:
            raise CallFailedError(f"{case['id']} {spec['call']}: generate returned no text")
        response_body = {key: value for key, value in result.items() if key != "request"}
        shape = "the sie_sdk generate result; the SDK surfaces no response headers"
    else:
        params = body.get("params") or {}
        result = client.extract(
            spec["model"], body["items"][0], labels=params.get("labels"), options=params.get("options")
        )
        # A 200 can still carry a per-item failure. Never record one as a result.
        if result.get("error"):
            raise CallFailedError(f"{case['id']} {spec['call']}: item error {result['error']}")
        if not result.get("classifications"):
            raise CallFailedError(f"{case['id']} {spec['call']}: response carried no classifications")
        item = {key: value for key, value in result.items() if key not in ("model", "request")}
        response_body = {"items": [item], "model": result.get("model", spec["model"])}
        shape = "rebuilt from the sie_sdk per-item result; the SDK returns no server envelope and no headers"
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    return {
        "id": f"{case['id']}__{spec['call']}",
        "set": "page",
        "case": case["id"],
        "call": spec["call"],
        "model": spec["model"],
        "endpoint": ENDPOINT,
        "path": spec["path"],
        "status": 200,
        "timing": {"at": requested_at, "latency_ms": latency_ms, "attempts": 1},
        "request": {
            "method": "POST",
            "endpoint": ENDPOINT,
            "path": spec["path"],
            "model": spec["model"],
            "body": body,
        },
        "response": {"status": 200, "body": response_body, "shape": shape},
        "recorded": {
            "model_revision": client.last_model_revision,
            "retry_count": client.last_retry_count,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument("--check", action="store_true", help="offline check, the default")
    parser.add_argument("--show", metavar="CASE", help="print one input's four requests and exit, sending nothing")
    parser.add_argument("--record", action="store_true", help="make live calls (needs SIE_API_KEY)")
    parser.add_argument("--out", default="run-output/calls.json", help="where --record writes")
    args = parser.parse_args()

    data_dir = Path(args.data)
    cases = load(data_dir / "inputs/inputs.json")["cases"]
    by_id = {case["id"]: case for case in cases}

    if args.show:
        case = by_id.get(args.show)
        if case is None:
            raise SystemExit(f"Unknown case: {args.show}")
        shown = [
            {"call": spec["call"], "method": "POST", "path": spec["path"], "body": spec["body"]}
            for spec in calls_for(case)
        ]
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
    failed: list[str] = []
    for case in cases:
        for spec in calls_for(case):
            call_id = f"{case['id']}__{spec['call']}"
            try:
                entry = record(client, case, spec)
            except Exception as error:  # noqa: BLE001
                failed.append(f"{call_id}: {type(error).__name__}: {error}")
                calls.append(failure_entry(call_id, case["id"], spec["model"], spec["path"], spec["body"], error))
                print(f"{case['id']} {spec['call']}: FAILED {type(error).__name__}", file=sys.stderr)
                continue
            calls.append(entry)
            print(f"{case['id']} {spec['call']}: {entry['timing']['latency_ms']:.0f}ms", file=sys.stderr)

    ids = [entry["id"] for entry in calls]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate call ids; refusing to write a calls.json two checks could read differently")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "task": "guardrails",
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
