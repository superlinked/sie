#!/usr/bin/env python3
"""Send the /sparse-embeddings calls to SIE Cloud, or check the recorded ones.

    python3 fetch.py
    python3 run.py --check             # offline, no key, nothing installed
    python3 run.py --show <input-id>   # offline, prints one request
    uv sync && uv run python run.py --record    # live, needs SIE_API_KEY

Endpoint      https://api.superlinked.com
Path          /v1/encode/<model>
Models        prithivida/Splade_PP_en_v2   (Hugging Face f0d4aa214dcb60c274052a52c0497535e3aec64c)
              BAAI/bge-m3:sparse           (Hugging Face 5617a9f61b028005a4858fdac845db406aefb181)

Calls go through `sie_sdk.SIEClient`, per AGENTS.md. The import is deferred into
main() so `--check` and `--show` run on a bare `python3` with nothing installed.

Thirteen texts go to both models, so 26 calls. Every call asks for
`output_types: ["sparse"]` and gets back token indices with weights.

`--check` rebuilds all 26 recorded request bodies from `data/inputs/inputs.json`
and compares each with the recorded request. Those bodies were confirmed against
the SDK by intercepting the client transport: `client.encode` puts exactly these
26 bodies on the wire, at exactly these paths.

It is a bijection, not a walk over what is there: the expected call ids come
from the inputs, one per model per text, so a call that is missing, recorded
twice, or implied by no input fails the check.

`--record` writes calls.json only. It does NOT rewrite `derived/decoded/`, which
maps token indices back to strings using each model's own tokenizer. That needs
the tokenizer files, and this example is built to run without downloading model
weights.

Migrated from apps/site/tests/fixtures/reference/sparse/run.py in
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
MODELS = {
    "splade": "prithivida/Splade_PP_en_v2",
    "bge": "BAAI/bge-m3:sparse",
}
# The folder each model's decoded terms live under in the dataset.
DECODED_DIR = {"splade": "splade", "bge": "bge-m3-sparse"}


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


def build_body(input_id: str, text: str) -> dict[str, Any]:
    return {"items": [{"id": input_id, "text": text}], "params": {"output_types": ["sparse"]}}


def check(data_dir: Path) -> int:
    """Compare the recorded calls with the ones the inputs imply.

    A bijection, not a walk over what happens to be there: the expected call
    ids come from inputs/inputs.json, one per model per text, so a call that is
    missing, recorded twice or not derivable from any input all fail. Checking
    only the calls present would pass a calls.json with one of them deleted.
    """
    inputs = load(data_dir / "inputs/inputs.json")["inputs"]
    calls = load(data_dir / "calls.json")["calls"]

    expected: dict[str, tuple[str, dict[str, Any]]] = {}
    for set_name, model in MODELS.items():
        for item in inputs:
            expected[f"{DECODED_DIR[set_name]}/{item['id']}"] = (model, item)

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
        model, item = expected[call_id]
        if call["case"] != item["id"]:
            mismatched.append(f"{call_id}: case {call['case']} differs from {item['id']}")
        elif call["model"] != model:
            mismatched.append(f"{call_id}: model {call['model']} differs from {model}")
        elif build_body(item["id"], item["text"]) != call["request"]["body"]:
            mismatched.append(f"{call_id}: rebuilt body differs from the recorded body")
        elif call["path"] != f"/v1/encode/{model}":
            mismatched.append(f"{call_id}: path {call['path']} differs from /v1/encode/{model}")
        else:
            rebuilt += 1

    print(f"{rebuilt} of {len(calls)} recorded requests rebuilt from the inputs and matched")
    print(
        f"{len(expected)} calls expected from {len(inputs)} texts across {len(MODELS)} models, {len(recorded)} recorded"
    )
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


def record(client: SIEClient, set_name: str, model: str, item: dict[str, Any]) -> dict[str, Any]:
    """Send one text through the SDK and return its calls.json entry."""
    body = build_body(item["id"], item["text"])
    path = f"/v1/encode/{model}"
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    started = time.monotonic()
    result = client.encode(model, body["items"][0], output_types=body["params"]["output_types"])
    latency_ms = round((time.monotonic() - started) * 1000, 1)
    # A 200 can still carry a per-item failure. Never record one as a result.
    if result.get("error"):
        raise CallFailedError(f"{item['id']}: item error {result['error']}")
    if "sparse" not in result:
        raise CallFailedError(f"{item['id']}: response carried no sparse vector")
    # The SDK hands back indices and values as numpy arrays. Convert to lists so
    # calls.json holds the same JSON types the archived run recorded, and so
    # score.py reads a fresh run the same way.
    sparse = result["sparse"]
    wire_sparse = {
        "indices": [int(index) for index in sparse["indices"]],
        "values": [float(value) for value in sparse["values"]],
    }
    return {
        "id": f"{DECODED_DIR[set_name]}/{item['id']}",
        "set": set_name,
        "case": item["id"],
        "model": model,
        "endpoint": ENDPOINT,
        "path": path,
        "status": 200,
        "timing": {"at": requested_at, "latency_ms": latency_ms, "attempts": 1},
        "request": {"method": "POST", "endpoint": ENDPOINT, "path": path, "model": model, "body": body},
        "response": {
            "status": 200,
            "body": {"items": [{"id": result.get("id", item["id"]), "sparse": wire_sparse}], "model": model},
            "shape": (
                "rebuilt from the sie_sdk per-item result, with the numpy arrays "
                "converted to lists; the SDK returns no server envelope, no dims "
                "or dtype, and no response headers"
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
    parser.add_argument("--show", metavar="INPUT", help="print one request and exit, sending nothing")
    parser.add_argument("--record", action="store_true", help="make live calls (needs SIE_API_KEY)")
    parser.add_argument("--out", default="run-output/calls.json", help="where --record writes")
    args = parser.parse_args()

    data_dir = Path(args.data)
    inputs = load(data_dir / "inputs/inputs.json")["inputs"]
    by_id = {item["id"]: item for item in inputs}

    if args.show:
        item = by_id.get(args.show)
        if item is None:
            raise SystemExit(f"Unknown input: {args.show}")
        body = build_body(item["id"], item["text"])
        shown = [
            {"model": model, "method": "POST", "path": f"/v1/encode/{model}", "body": body} for model in MODELS.values()
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
    for set_name, model in MODELS.items():
        for item in inputs:
            call_id = f"{DECODED_DIR[set_name]}/{item['id']}"
            try:
                entry = record(client, set_name, model, item)
            except Exception as error:  # noqa: BLE001
                failed.append(f"{call_id}: {type(error).__name__}: {error}")
                calls.append(
                    failure_entry(
                        call_id,
                        set_name,
                        item["id"],
                        model,
                        f"/v1/encode/{model}",
                        build_body(item["id"], item["text"]),
                        error,
                    )
                )
                print(f"{model} {item['id']}: FAILED {type(error).__name__}", file=sys.stderr)
                continue
            calls.append(entry)
            print(f"{model} {item['id']}: {entry['timing']['latency_ms']:.0f}ms", file=sys.stderr)

    ids = [entry["id"] for entry in calls]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate call ids; refusing to write a calls.json two checks could read differently")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "task": "sparse",
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
    print("derived/decoded/ is NOT rewritten by --record; it needs each model's tokenizer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
