"""Collect a new paired reproduction with the unchanged question-only recipe."""

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path

import collect
import openai_population as acquisition
from reader_io import durable_writes, exclusive_run, read, require, sha


PHASES = {"learn": ("train", "development"), "calibration": ("calibration",), "final": ("final",)}


def checked_final(path, inputs, source_hash, pair_hash):
    require(path is not None, "Final acquisition requires the raw frozen policy bundle")
    frozen = read(path)
    require(
        frozen["format"] == "bbeh-frozen-frontiers-v1"
        and frozen["source_plan_sha256"] == source_hash
        and frozen["pair_sha256"] == pair_hash
        and frozen["ids"] == [row["id"] for row in inputs]
        and frozen["final_outcomes_accessed"] is False
        and frozen["frozen_before_final_outcomes"] is True,
        "Frozen policies do not match this final population and pair",
    )
    methods = frozen["methods"]
    require({"base", "lora"} <= set(methods), "Freeze both native variants before final collection")
    kinds = [method["scores"]["router_kind"] for method in methods.values()]
    require(kinds.count("embedding") >= 2 and "cpu" in kinds, "Freeze embedding methods and controls")
    for method in methods.values():
        scores = method["scores"]
        require(method["candidates"] and method["calibration_frontier_candidate_ids"], "Missing frozen policies")
        for name in ("probabilities", "expected_usd"):
            values = scores[name]
            require(len(values) == len(inputs), "Missing final input scores")
            require(all(isinstance(row, list) and len(row) == 2 for row in values), "Invalid paired scores")
            require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                        and (name != "probabilities" or v <= 1) for row in values for v in row), "Invalid score value")
    return {"sha256": sha(path), "methods": sorted(methods), "n": len(inputs)}


def prepare(source, pair_path, phase, budget, dispatch_seconds, frozen=None):
    source, pair_path = Path(source), Path(pair_path)
    cap = Decimal(str(budget))
    require(cap.is_finite() and cap > 0, "Budget must be a positive finite USD amount")
    require(type(dispatch_seconds) is int and dispatch_seconds > 0, "Invalid dispatch deadline")
    pair = read(pair_path)
    require(pair["models"] == {"cheap": collect.ARMS["luna"]["model"], "strong": collect.ARMS["sol"]["model"]}
            and pair["settings"] == collect.OPENAI_SETTINGS and pair["system_prompt"] == collect.SYSTEM
            and pair["prices_usd_per_million"] == collect.PRICES, "Pair settings differ from the collection recipe")
    source_plan = read(source / "source-plan.json")
    require(sha(source / "meta.json") == source_plan["files_sha256"]["meta.json"], "Source metadata changed")
    metadata = read(source / "meta.json")
    by_id = {row["id"]: row for row in metadata}
    require(len(by_id) == len(metadata), "Duplicate source ID")
    items, files = {}, {}
    for role in PHASES[phase]:
        path = source / f"question-only/{role}-inputs.json"
        require(sha(path) == source_plan["files_sha256"][f"question-only/{role}-inputs.json"], "Question inputs changed")
        rows = read(path)
        expected = [row["id"] for row in metadata if row["retained"] and row["role"] == role]
        require(rows and [row["id"] for row in rows] == expected, "Role population/order changed")
        for row in rows:
            require(hashlib.sha256(row["request"].encode()).hexdigest() == by_id[row["id"]]["request_sha256"], "Request changed")
            for arm in collect.ARMS:
                collect.body_for(row, arm)  # Rejects references and every extra payload field.
        items[role], files[f"{role}-inputs.json"] = rows, sha(path)
    here = Path(__file__).parent
    files.update({name: sha(here / name) for name in ("collect.py", "openai_population.py", "collect_answers.py", "reader_io.py")})
    plan = {
        "format": "bbeh-public-acquisition-v1", "reproduction": True,
        "phase": phase, "roles": list(items), "counts": {role: len(rows) for role, rows in items.items()},
        "source_plan_sha256": sha(source / "source-plan.json"), "pair_sha256": sha(pair_path),
        "files_sha256": files, "budget_usd": str(cap), "dispatch_seconds": dispatch_seconds,
        "workers_per_arm": acquisition.WORKERS_PER_ARM, "max_physical_attempts": acquisition.MAX_ATTEMPTS,
        "selection": "first terminal provider response; no content/format/quality retries",
        "budget_scope": "All known physical-attempt charges plus unresolved execution reservations and in-flight reservations",
        "collector_inputs_question_only": True, "references_loaded_by_collector": False,
        "final_gate": None,
    }
    if phase == "final":
        plan["final_gate"] = checked_final(frozen, items["final"], plan["source_plan_sha256"], plan["pair_sha256"])
    else:
        require(frozen is None, "A final gate is only applicable to final collection")
    return items, plan


def execute(items, plan, output, *, call=None):
    output = Path(output).resolve()
    require(call is not None or bool(os.environ.get("OPENAI_API_KEY")), "Set OPENAI_API_KEY for the OpenAI collector")
    with exclusive_run(output), durable_writes(acquisition, output) as commit:
        return acquisition.run(items, output, plan, commit, call=call or acquisition.acquire,
                               dispatch_seconds=plan["dispatch_seconds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pair", type=Path, default=Path(__file__).with_name("pair.json"))
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-usd", required=True)
    parser.add_argument("--dispatch-seconds", type=int, default=20900)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--execute", action="store_true", help="Dispatch paid calls; otherwise print the immutable plan")
    args = parser.parse_args()
    items, plan = prepare(args.source, args.pair, args.phase, args.budget_usd, args.dispatch_seconds, args.frozen)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    result = execute(items, plan, args.output)
    print(json.dumps({key: result[key] for key in (
        "status", "physical_attempts", "selected_responses", "known_list_usd", "unknown_reservation_usd", "stop"
    )}, indent=2))


if __name__ == "__main__":
    main()
