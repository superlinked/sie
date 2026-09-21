#!/usr/bin/env python3
"""Send the /rerank calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check                 # offline, no key, nothing installed
    python3 run.py --show <case-id>        # offline, prints one call
    uv sync && uv run python run.py --record

Endpoint  https://api.superlinked.com
Model     Qwen/Qwen3-Reranker-4B
Path      /v1/score/Qwen/Qwen3-Reranker-4B

Each case is sent once per arm against the same four candidates: with no
instruction, and then under each relevance rule in `inputs/cases.json`. There
are four of those, so five calls per case and 120 for the 24 cases. The page
publishes two of the rules and the recording keeps all four. The query names
the subject and never the status, so only the instruction can separate a
regulation already adopted from a proposal on the same subject.

Calls go through `sie_sdk.SIEClient`, per AGENTS.md. The import is deferred into
main() so `--check` and `--show` run on a bare `python3` with nothing installed.

`--check` rebuilds every recorded call envelope from `data/inputs/cases.json`
and compares it with the recorded one. It is a bijection, not a walk over what
is there: the expected call ids come from the inputs, one per case per arm, so a
call that is missing, recorded twice or implied by no case fails the check.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sie_sdk import SIEClient

MODEL = "Qwen/Qwen3-Reranker-4B"
PATH = f"/v1/score/{MODEL}"
ENDPOINT = "https://api.superlinked.com"

# The revision GET /v1/models reports for these weights. run.py refuses to
# record against another one, and score.py refuses to score a recording made
# against another one, because different weights produce different scores.
MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"

BASELINE = "none"
REQUEST_SHAPE = (
    "the SDK call envelope run.py builds. `instruction` is a real request field; "
    "`case` and `rule` are provenance fields the SIE payload does not carry."
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


def arms(payload: dict[str, Any]) -> dict[str, str | None]:
    """Every arm a case is sent under: the baseline first, then each rule."""
    return {BASELINE: None, **payload["rules"]}


def call_id(case_id: str, rule: str) -> str:
    return f"{case_id}/{rule}"


def build_envelope(case: dict[str, Any], rule: str, instruction: str | None) -> dict[str, Any]:
    """The SDK call this example makes for one case under one rule."""
    envelope: dict[str, Any] = {
        "method": "SIEClient.score",
        "endpoint": PATH,
        "model": MODEL,
        "case": case["id"],
        "rule": rule,
        "query": {"id": f"{case['id']}-query", "text": case["query"]},
        "items": [{"id": candidate["id"], "text": candidate["text"]} for candidate in case["candidates"]],
    }
    if instruction is not None:
        envelope["instruction"] = instruction
    return envelope


def expected_calls(payload: dict[str, Any]) -> dict[str, tuple[dict[str, Any], str, str | None]]:
    out = {}
    for case in payload["cases"]:
        for rule, instruction in arms(payload).items():
            out[call_id(case["id"], rule)] = (case, rule, instruction)
    return out


def check(data_dir: Path) -> int:
    payload = load(data_dir / "inputs/cases.json")
    calls = load(data_dir / "calls.json")["calls"]

    expected = expected_calls(payload)
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
    for identifier in sorted(set(expected) & set(recorded)):
        call = recorded[identifier]
        case, rule, instruction = expected[identifier]
        if call["model"] != MODEL:
            mismatched.append(f"{identifier}: model {call['model']} differs from {MODEL}")
        elif call["path"] != PATH:
            mismatched.append(f"{identifier}: path {call['path']} differs from {PATH}")
        elif build_envelope(case, rule, instruction) != call["request"]["body"]:
            mismatched.append(f"{identifier}: rebuilt envelope differs from the recorded one")
        else:
            rebuilt += 1

    print(f"{rebuilt} of {len(calls)} recorded calls rebuilt from the inputs and matched")
    print(f"{len(expected)} calls expected from {len(payload['cases'])} cases, {len(recorded)} recorded")
    for identifier in missing:
        print(f"MISSING {identifier}: expected from the inputs, absent from calls.json", file=sys.stderr)
    for identifier in duplicates:
        print(f"DUPLICATE {identifier}: recorded more than once", file=sys.stderr)
    for identifier in unexpected:
        print(f"UNEXPECTED {identifier}: recorded but no case in inputs/cases.json implies it", file=sys.stderr)
    for line in mismatched:
        print(f"MISMATCH {line}", file=sys.stderr)
    return 1 if (missing or duplicates or unexpected or mismatched) else 0


def served_revision(client: Any) -> str:
    """The model revision the endpoint reports for itself, via GET /v1/models.

    Raises rather than returning a placeholder. A run that cannot establish
    which weights answered has nothing to record.
    """
    try:
        listed = client.list_models()
    except Exception as error:
        raise SystemExit(f"Could not read the model revision from /v1/models: {error}") from error
    for model in listed if isinstance(listed, list) else listed.get("models", []):
        if isinstance(model, dict) and model.get("name") == MODEL:
            revision = model.get("revision") or ""
            if not revision:
                raise SystemExit(f"/v1/models lists {MODEL} with no revision")
            return revision
    raise SystemExit(f"/v1/models does not list {MODEL}")


def record(
    client: SIEClient, base_url: str, case: dict[str, Any], rule: str, instruction: str | None
) -> dict[str, Any]:
    """Send one case under one rule and return its calls.json entry."""
    from sie_sdk import Item  # noqa: PLC0415

    envelope = build_envelope(case, rule, instruction)
    kwargs = {"wait_for_capacity": True, "provision_timeout_s": 900}
    if instruction is not None:
        kwargs["instruction"] = instruction
    started = time.monotonic()
    result = client.score(
        MODEL,
        Item(id=f"{case['id']}-query", text=case["query"]),
        [Item(id=candidate["id"], text=candidate["text"]) for candidate in case["candidates"]],
        **kwargs,
    )
    latency_ms = round((time.monotonic() - started) * 1000, 3)
    body = to_jsonable(result)
    # A 200 is not a result. Refuse anything without scores to read.
    if not isinstance(body, dict) or not body.get("scores"):
        raise CallFailedError(f"{case['id']}/{rule}: response carried no scores")
    returned = {entry["item_id"] for entry in body["scores"]}
    sent = {candidate["id"] for candidate in case["candidates"]}
    if returned != sent:
        raise CallFailedError(f"{case['id']}/{rule}: scored {sorted(returned)}, sent {sorted(sent)}")
    return {
        "id": call_id(case["id"], rule),
        "case": case["id"],
        "rule": rule,
        "model": MODEL,
        "endpoint": base_url,
        "path": PATH,
        "status": 200,
        "timing": {"latency_ms": latency_ms, "attempts": 1},
        "request": {"method": "SIEClient.score", "shape": REQUEST_SHAPE, "body": envelope},
        "response": {"status": 200, "shape": RESPONSE_SHAPE, "body": body},
        "recorded": {"model_revision": client.last_model_revision, "retry_count": client.last_retry_count},
    }


def failure_entry(
    case: dict[str, Any], rule: str, base_url: str, envelope: dict[str, Any], error: BaseException
) -> dict[str, Any]:
    """What a failed call records: every field the success path writes."""
    return {
        "id": call_id(case["id"], rule),
        "case": case["id"],
        "rule": rule,
        "model": MODEL,
        "endpoint": base_url,
        "path": PATH,
        "status": "error",
        "error": {"type": type(error).__name__, "message": str(error)},
        "timing": {"latency_ms": None, "attempts": 1},
        "request": {"method": "SIEClient.score", "shape": REQUEST_SHAPE, "body": envelope},
        "response": None,
        "recorded": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data", help="fetched evidence directory")
    parser.add_argument("--check", action="store_true", help="offline check, the default")
    parser.add_argument("--show", metavar="CASE", help="print one case's calls and exit, sending nothing")
    parser.add_argument("--record", action="store_true", help="make live calls")
    parser.add_argument("--out", default="run-output", help="directory --record writes")
    parser.add_argument(
        "--allow-revision-mismatch",
        action="store_true",
        help="record even if the endpoint serves a revision other than the published one",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    payload = load(data_dir / "inputs/cases.json")
    by_id = {case["id"]: case for case in payload["cases"]}

    if args.show:
        case = by_id.get(args.show)
        if case is None:
            raise SystemExit(f"Unknown case: {args.show}. Known: {', '.join(by_id)}")
        for rule, instruction in arms(payload).items():
            print(json.dumps(build_envelope(case, rule, instruction), indent=2, ensure_ascii=False))
        return 0

    if not args.record:
        return check(data_dir)

    base_url = os.environ.get("SIE_BASE_URL") or os.environ.get("SIE_CLUSTER_URL") or ENDPOINT
    api_key = os.environ.get("SIE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set SIE_API_KEY. To check the published figures without a key, run score.py instead.")

    from sie_sdk import SIEClient  # noqa: PLC0415

    client = SIEClient(base_url, api_key=api_key, timeout_s=900)
    base_url = client.base_url.rstrip("/")
    print(f"endpoint {base_url}{PATH}", file=sys.stderr)

    # Checked before anything is scored, so a mismatch costs no credits.
    model_revision = served_revision(client)
    print(f"revision {model_revision}", file=sys.stderr)
    if model_revision != MODEL_REVISION and not args.allow_revision_mismatch:
        raise SystemExit(
            f"This endpoint serves {model_revision!r} and this example publishes {MODEL_REVISION!r}.\n"
            "Different weights produce different scores, and score.py rejects a recording made against "
            "another revision.\nRe-run with --allow-revision-mismatch to record anyway, for your own "
            "comparison rather than to reproduce the published figures."
        )

    calls: list[dict[str, Any]] = []
    failed: list[str] = []
    for case in payload["cases"]:
        for rule, instruction in arms(payload).items():
            try:
                entry = record(client, base_url, case, rule, instruction)
            except Exception as error:  # noqa: BLE001
                failed.append(f"{call_id(case['id'], rule)}: {type(error).__name__}: {error}")
                calls.append(failure_entry(case, rule, base_url, build_envelope(case, rule, instruction), error))
                print(f"{call_id(case['id'], rule)}: FAILED {type(error).__name__}", file=sys.stderr)
                continue
            calls.append(entry)
            print(f"{entry['id']}: {entry['timing']['latency_ms']:.0f}ms", file=sys.stderr)

    # Re-read after the last call. The preflight proves the weights were right
    # when the run started; this proves they did not roll over while it was in
    # flight, which would leave the manifest attributing scores to a checkpoint
    # that did not produce all of them.
    final_revision = served_revision(client)
    if final_revision != model_revision:
        raise SystemExit(
            f"The endpoint served {model_revision!r} before these calls and {final_revision!r} after them. "
            "The recording spans two checkpoints, so it is not written out. Re-run it."
        )

    deployment = sorted({c["recorded"].get("model_revision") for c in calls if c["recorded"].get("model_revision")})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calls.json").write_text(
        json.dumps(
            {
                "task": "rerank",
                "call_count": len(calls),
                "failed_calls": len(failed),
                "complete": not failed,
                "calls": calls,
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "manifest-partial.json").write_text(
        json.dumps(
            {
                "task": "rerank",
                "endpoint": base_url,
                "path": PATH,
                "model": MODEL,
                "model_revision": model_revision,
                "deployment_revision": deployment[0] if len(deployment) == 1 else deployment,
                "run_date": time.strftime("%Y-%m-%d", time.gmtime()),
                "recorded_by": "examples/rerank/run.py",
                "cases": len(payload["cases"]),
                "arms": list(arms(payload)),
                "calls_recorded": len(calls),
            },
            indent=1,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_dir}/calls.json", file=sys.stderr)
    if failed:
        print(f"{len(failed)} of {len(calls)} calls FAILED:", file=sys.stderr)
        for line in failed:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
