"""Unchanged-weight Laya baseline with matched train/development cost estimates."""

import argparse
import json
from pathlib import Path

import numpy as np

import bbeh_native as native
import recipe as core


def prepare(run, training_grades, development_grades, initial_checkpoint, output):
    run, output = Path(run), Path(output)
    plan, rows = core.checked_plan(run)
    core.require(not output.exists(), "Preserve the previous base export")
    y, costs, hashes = [], [], {}
    for role, path in (("train", training_grades), ("development", development_grades)):
        _, labels, usd = core.grade_arrays(plan, rows, core.read(path), role)
        y.append(labels)
        costs.append(usd)
        hashes[role] = core.sha(path)
    stats = core.conditional_usd(np.concatenate(y), np.concatenate(costs))
    output.mkdir(parents=True)
    export = native.copy_base_checkpoint(initial_checkpoint, output / "checkpoint", plan["native"])
    result = {
        "format": "bbeh-unchanged-base-v1", "plan_sha256": core.sha(run / "plan.json"),
        "source_plan_sha256": plan["source_plan_sha256"], "pair_sha256": plan["pair_sha256"],
        "adapter_sha256": core.sha(__file__), "grade_files_sha256": hashes,
        "native_source_sha256": core.sources(), "checkpoint": export,
        "conditional_cost_usd": stats,
        "weights_fitted": False, "calibration_or_final_outcomes_accessed": False,
        "cost_scope": "The same train+development conditional USD means as the native refit; base weights are unchanged.",
    }
    core.write(output / "base.json", result)
    return result


def checked_base(run, base):
    run, base = Path(run), Path(base)
    plan, rows = core.checked_plan(run)
    model = core.read(base / "base.json")
    core.require(model["format"] == "bbeh-unchanged-base-v1" and model["plan_sha256"] == core.sha(run / "plan.json")
                 and model["source_plan_sha256"] == plan["source_plan_sha256"] and model["pair_sha256"] == plan["pair_sha256"]
                 and model["adapter_sha256"] == core.sha(__file__) and model["native_source_sha256"] == core.sources()
                 and model["weights_fitted"] is False and model["calibration_or_final_outcomes_accessed"] is False,
                 "Base source/plan/weight contract changed")
    actual = native.checkpoint_files(base / "checkpoint")
    core.require(actual == model["checkpoint"]["files_sha256"] and model["checkpoint"]["weights_byte_identical"] is True
                 and actual.get("model.safetensors") == native.INITIAL_WEIGHTS_SHA256, "Base checkpoint bytes changed")
    return plan, rows, model


def score(run, base, role, output, device):
    core.require(role in ("calibration", "final") and device in ("cpu", "cuda"), "Invalid input-only scoring role/device")
    core.require(not Path(output).exists(), "Preserve previous scores")
    plan, rows, model = checked_base(run, base)
    selected = [row for row in rows if row["role"] == role]
    probabilities = native.checkpoint_probabilities(Path(base) / "checkpoint", selected, plan["pair"],
                                                    plan["native"], device=device)
    expected = core.expected_usd(probabilities, model["conditional_cost_usd"])
    result = {
        "format": "bbeh-input-only-scores-v1", "source_plan_sha256": plan["source_plan_sha256"],
        "pair_sha256": plan["pair_sha256"], "model_sha256": core.sha(Path(base) / "base.json"),
        "outcomes_accessed": False, "role": role, "ids": [row["id"] for row in selected],
        "methods": {"base": {"probabilities": probabilities.tolist(), "expected_usd": expected.tolist(),
                              "router_kind": "native", "metadata_assisted": False}},
        "device": device, "router_hosting_usd": None,
    }
    core.write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--training-grades", required=True, type=Path)
    p.add_argument("--development-grades", required=True, type=Path)
    p.add_argument("--initial-checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p = sub.add_parser("score")
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--base", required=True, type=Path)
    p.add_argument("--split", choices=("calibration", "final"), required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.run, args.training_grades, args.development_grades, args.initial_checkpoint, args.output)
    else:
        result = score(args.run, args.base, args.split, args.output, args.device)
    print(json.dumps({"format": result["format"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
