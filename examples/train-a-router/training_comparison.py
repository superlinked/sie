"""Compare a fast LoRA refit with the published recipe using the same cost models."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import workflow
from embedding import recipes as core
from embedding.cost_iteration import refit
from evaluation import frontier
from evaluation import utility_router as utility
from reader_runtime import ensure_utf8

METHODS = {"fast_lora": "lora", "fast_lora__textcost": "lora__textcost"}
CAPS = (0.2, 0.4, 0.6, 0.8)


def choose(candidates, cap):
    """Choose from a calibration frontier, never from final measurements."""
    eligible = [p for p in candidates if p["calibration"]["answer_cost_usd_per10000"] <= cap]
    return min(eligible, key=lambda p: (
        -p["calibration"]["answer_accuracy_percent"],
        p["calibration"]["answer_cost_usd_per10000"], p["candidate_id"],
    )) if eligible else None


def checked_native(data, native_run):
    reference = data / "reference/native"
    plan = core.read(reference / "plan.json")
    core.require(core.sha(native_run / "plan.json") == core.sha(reference / "plan.json"),
                 "Fast run must use the published populations and scientific input plan")
    trained = core.read(native_run / "native/trained.json")
    recipe_path = native_run / "native/fast-recipe.json"
    recipe = core.read(recipe_path)
    implementations = {name: core.sha(workflow.ROOT / "native" / name)
                       for name in ("fast_protocol.py", "fast_runtime.py", "fast_train.py")}
    core.require(
        trained["plan_sha256"] == core.sha(reference / "plan.json")
        and trained["training_ids"] == plan["populations"]["train"] + plan["populations"]["development"]
        and trained["selection_train_ids"] == plan["populations"]["train"]
        and trained["selection_development_ids"] == plan["populations"]["development"]
        and trained["training_grades_sha256"] == core.sha(data / "grades/grades-train.json")
        and trained["development_grades_sha256"] == core.sha(data / "grades/grades-development.json")
        and trained["fast_recipe_sha256"] == core.sha(recipe_path)
        and trained["fast_runtime_sha256"] == core.sha(native_run / "native/fast-runtime.json")
        and recipe["reference_plan_sha256"] == trained["plan_sha256"]
        and recipe["format"] == "laya-fast-runtime-recipe-v1"
        and recipe["recipe_id"] == "laya-fa2-grouped-v1"
        and recipe["implementation_sources_sha256"] == implementations
        and trained["fast_implementation_sources_sha256"] == implementations
        and recipe["final_outcomes_accessed"] is False,
        "Fast training provenance changed",
    )
    for role in ("calibration", "final"):
        score = core.read(native_run / f"native/scores-{role}.json")
        core.require(score["model_sha256"] == core.sha(native_run / "native/trained.json")
                     and set(score["methods"]) == {"base", "lora"}, "Native scores belong to another model")
    return trained


def calibrate(data, native_run, output, comparison_plan=None):
    """Freeze the comparison and its illustrated policies without final answer labels."""
    data, native_run, output = workflow.checked_data(data), Path(native_run).resolve(), Path(output).resolve()
    core.require(not output.exists(), "Use a new comparison directory")
    if comparison_plan is not None:
        declaration = core.read(comparison_plan)
        core.require(declaration["format"] == "laya-fast-post-publication-comparison-v1"
                     and declaration["reference"]["manifest_sha256"] == core.sha(data / "manifest.json")
                     and tuple(declaration["comparison"]["illustrations"][
                         "fractions_of_always_strong_calibration_answer_cost"]) == CAPS,
                     "Comparison plan uses another study or budget choices")
    trained = checked_native(data, native_run)
    reference = data / "reference"
    presentation = core.read(data / "metadata/presentation-policies.json")
    core.require(presentation["calibration_sha256"] == core.sha(reference / "calibration-frontiers.json")
                 and presentation["frozen_sha256"] == core.sha(reference / "frozen-frontiers.json")
                 and tuple(presentation["illustrative_caps_fraction_of_always_strong_answer_cost"]) == CAPS,
                 "Published comparison choices changed")
    output.mkdir(parents=True)
    paths, base_drift = {}, {}
    for role in ("calibration", "final"):
        incoming = native_run / f"native/scores-{role}.json"
        score = core.read(incoming)
        old = core.read(reference / f"native/native/scores-{role}.json")
        core.require(score["ids"] == old["ids"], "Native score order changed")
        base_drift[role] = float(np.max(np.abs(
            np.asarray(score["methods"]["base"]["probabilities"])
            - np.asarray(old["methods"]["base"]["probabilities"]))))
        selected = {**score, "methods": {"fast_lora": score["methods"]["lora"]},
                    "parent_scores_sha256": core.sha(incoming)}
        path = output / f"native-{role}.json"
        core.write(path, selected)
        augmented = output / f"cost-{role}.json"
        refit.augment_files(path, reference / f"cost/costs-{role}.json", augmented,
                            output / "combined-model.json")
        paths[role] = [reference / f"cost/scores-{role}.json",
                       reference / f"cost/native-scores-{role}.json", augmented]
    calibration_path, frozen_path = output / "calibration-frontiers.json", output / "frozen-frontiers.json"
    calibration = frontier.calibrate(reference / "embedding", data / "grades/grades-calibration.json",
                                      paths["calibration"], data / "fees.json", calibration_path)
    frontier.freeze(reference / "embedding", calibration_path, paths["final"], data / "fees.json", frozen_path)
    prior = core.read(reference / "calibration-frontiers.json")
    for method, value in prior["methods"].items():
        core.require(workflow.same_values(calibration["methods"][method], value), "Reference calibration changed")
    strong = next(p["calibration"]["answer_cost_usd_per10000"]
                  for p in calibration["methods"]["lora"]["candidates"]
                  if p["policy"]["mode"] == "always_strong")
    choices = []
    for method, original in METHODS.items():
        for fraction in CAPS:
            selected = choose(calibration["methods"][method]["frontier"], strong * fraction)
            choices.append({"method": method, "reference_method": original, "cap_fraction": fraction,
                            "candidate": selected,
                            "reference": presentation["choices"][original][str(fraction)]})
    core.write(output / "comparison-freeze.json", {
        "format": "laya-training-comparison-v1", "new_blind_evaluation": False,
        "final_outcomes_read_by_comparison": False, "choices": choices,
        "calibration_sha256": core.sha(calibration_path), "frozen_sha256": core.sha(frozen_path),
        "comparison_source_sha256": core.sha(__file__),
        "comparison_plan_sha256": core.sha(comparison_plan) if comparison_plan is not None else None,
        "reference_presentation_sha256": core.sha(data / "metadata/presentation-policies.json"),
        "dataset_manifest_sha256": core.sha(data / "manifest.json"),
        "fast_trained_sha256": core.sha(native_run / "native/trained.json"),
        "fast_recipe_sha256": trained["fast_recipe_sha256"],
        "reference_trained_sha256": core.sha(reference / "native/native/trained.json"),
        "base_max_probability_difference": base_drift,
        "timing": {"fast_fit_seconds": trained["elapsed_seconds"],
                   "reference_fit_seconds": core.read(reference / "native/native/trained.json")["elapsed_seconds"],
                   "fast_selected_epoch": trained["selected_epoch"],
                   "reference_selected_epoch": core.read(reference / "native/native/trained.json")["selected_epoch"],
                   "scope": "Full fit calls; dependency setup and input-only scoring are separate. Historical reference."},
    })
    return {"comparison": str(output), "frozen": True, "final_outcomes_read": False}


def paired_differences(rows, y, costs, reference_routes, candidate_routes):
    """Resample both routes jointly, keeping families together and fixed task weights."""
    index = np.arange(len(y))
    reference_routes, candidate_routes = reference_routes.astype(int), candidate_routes.astype(int)
    values = np.column_stack((
        y[index, reference_routes], y[index, candidate_routes],
        costs[index, reference_routes], costs[index, candidate_routes],
        reference_routes, candidate_routes, costs[:, 1],
    ))

    def differences(means):
        return {
            "accuracy_pp": 100 * (means[..., 1] - means[..., 0]),
            "answer_usd_per_10000": 10000 * (means[..., 3] - means[..., 2]),
            "answer_savings_pp": 100 * (means[..., 2] - means[..., 3]) / means[..., 6],
            "strong_share_pp": 100 * (means[..., 5] - means[..., 4]),
        }

    core.require(np.all(costs[:, 1] >= 0) and costs[:, 1].mean() > 0, "Undefined always-Sol cost")
    point = differences(values.mean(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        draws = differences(frontier.Bootstrap(rows, 2000).means(values))
    return {name: {"estimate": float(value), "ci95": frontier.interval(draws[name])}
            for name, value in point.items()}


def evaluate(data, output):
    """Replay frozen policies against already published answers; never select on them."""
    data, output = workflow.checked_data(data), Path(output).resolve()
    frozen_path = output / "frozen-frontiers.json"
    declaration = core.read(output / "comparison-freeze.json")
    core.require(declaration["format"] == "laya-training-comparison-v1"
                 and declaration["comparison_source_sha256"] == core.sha(__file__)
                 and declaration["dataset_manifest_sha256"] == core.sha(data / "manifest.json")
                 and declaration["reference_presentation_sha256"]
                 == core.sha(data / "metadata/presentation-policies.json")
                 and declaration["frozen_sha256"] == core.sha(frozen_path)
                 and declaration["calibration_sha256"] == core.sha(output / "calibration-frontiers.json")
                 and declaration["new_blind_evaluation"] is False, "Frozen comparison changed")
    reference = data / "reference"
    final = frontier.evaluate(reference / "embedding", frozen_path, data / "grades/grades-final.json",
                               output / "final-report.json")
    frozen = core.read(frozen_path)
    plan = core.checked_plan(reference / "embedding")
    rows = core.role_rows(reference / "embedding", plan, "final")
    y, costs, _ = core.grades(plan, rows, data / "grades/grades-final.json", "final")
    comparisons = []
    for choice in declaration["choices"]:
        if choice["candidate"] is None:
            comparisons.append({**choice, "infeasible": True})
            continue
        policies, measured = [], {}
        for kind, method in (("reference", choice["reference_method"]), ("candidate", choice["method"])):
            selected = choice[kind]
            method_data = frozen["methods"][method]
            saved = next(p for p in method_data["candidates"] if p["candidate_id"] == selected["candidate_id"])
            core.require(saved["policy"] == selected["policy"], "Illustrated policy changed")
            scores = method_data["scores"]
            policies.append(utility.replay(saved["policy"], scores["probabilities"], scores["expected_usd"]))
            measured[kind] = next(p for p in final["methods"][method]["points"]
                                  if p["candidate_id"] == selected["candidate_id"])
        comparisons.append({"method": choice["method"], "cap_fraction": choice["cap_fraction"],
                            **measured, "candidate_minus_reference": paired_differences(rows, y, costs, *policies)})
    summary = {"format": "laya-training-comparison-report-v1", "new_blind_evaluation": False,
               "comparison_freeze_sha256": core.sha(output / "comparison-freeze.json"),
               "comparison_plan_sha256": declaration["comparison_plan_sha256"],
               "reference_presentation_sha256": declaration["reference_presentation_sha256"],
               "final_report_sha256": core.sha(output / "final-report.json"), "n": len(rows),
               "timing": declaration["timing"], "comparisons": comparisons,
               "interval_scope": "Pointwise paired family intervals conditional on tasks; not training-seed uncertainty.",
               "selection": "All illustrated budget choices were frozen on calibration before final replay.",
               "cost_scope": "Native answer-cost savings; hosting and one-time training are separate."}
    core.write(output / "comparison-report.json", summary)
    workflow.invoke("evaluation/report.py", "--calibration", output / "calibration-frontiers.json",
                    "--frozen", frozen_path, "--final-report", output / "final-report.json",
                    "--output", output / "explorer", "--reproduction")
    workflow.invoke("evaluation/source_sensitivity.py", "--run", reference / "embedding", "--frozen", frozen_path,
                    "--grades", data / "grades/grades-final.json", "--primary", output / "final-report.json",
                    "--specification", workflow.ROOT / "source-format-sensitivity.json",
                    "--output", output / "source-format-sensitivity.json")
    return {"report": str(output / "comparison-report.json"), "open": str(output / "explorer/index.html")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "evaluate"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-run", type=Path, help="The native directory inside your prepared run")
    parser.add_argument("--plan", type=Path, help="Optional predeclared validation plan to bind into the comparison")
    args = parser.parse_args()
    if args.phase == "calibrate":
        core.require(args.native_run is not None, "Calibration needs --native-run")
        calibrate(args.data, args.native_run, args.output, args.plan)
    else:
        evaluate(args.data, args.output)


if __name__ == "__main__":
    ensure_utf8()
    main()
