"""Development utility curves from saved train-only epoch margins; no inference."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import bbeh_native as native
import recipe as core
from evaluation import frontier, utility_router as utility


def epoch_history(run, plan, train_path, development_path, development_rows):
    run = Path(run)
    start = core.read(run / "native/training-start.json")
    core.require(
        start["plan_sha256"] == core.sha(run / "plan.json")
        and start["training_grades_sha256"] == core.sha(train_path)
        and start["development_grades_sha256"] == core.sha(development_path)
        and start["config"] == plan["native"]
        and start["selection_fit_rows"] == len(plan["populations"]["train"])
        and start["selection_validation_rows"] == len(development_rows)
        and start["initial_files_sha256"]["model.safetensors"] == core.INITIAL_WEIGHTS_SHA256,
        "Selection source/grade/configuration binding changed",
    )
    selected = core.read(run / "native/selected-epoch.json")
    history = [core.read(run / f"native/epoch-{epoch}.json") for epoch in range(9)]
    core.require(
        selected["history"] == history
        and selected["validation_ids"] == [row["id"] for row in development_rows]
        and selected["criterion"] == plan["native"]["selection"]
        and selected["epoch"] == native.select_epoch(history),
        "Selection history/order/selected epoch changed",
    )
    return history, selected["epoch"]


def curve_for_epoch(record, rows, labels, observed_costs, train_costs, bootstrap):
    margins = np.asarray(record["margins"], dtype=float)
    nll = native.summed_nll(margins, labels)
    core.require(
        math.isclose(nll, record["summed_binary_nll"], rel_tol=0, abs_tol=1e-12),
        "Saved development NLL does not match its epoch margins",
    )
    probabilities = np.exp(-np.logaddexp(0, -margins))
    expected = core.expected_usd(probabilities, train_costs)
    curve = frontier.enumerate_frontier(
        {"probabilities": probabilities, "expected_usd": expected}, labels, observed_costs, None
    )
    frontier_ids = {node["candidate_id"] for node in curve["frontier"]}
    points = []
    for point in curve["candidates"]:
        node = {
            "candidate_id": point["candidate_id"],
            "rule": point["policy"],
            "development": point["calibration"],
            "development_matched_random_strong_probability": point["calibration_matched_random_strong_probability"],
        }
        if point["candidate_id"] in frontier_ids:
            routes = utility.replay(point["policy"], probabilities, expected)
            fees = np.zeros(len(rows)) if point["policy"]["mode"].startswith("always_") else None
            intervals = frontier.uncertainty(
                bootstrap,
                routes,
                labels,
                observed_costs,
                fees,
                point["calibration_matched_random_strong_probability"],
            )
            node["paired_ci95"] = {
                name.replace("calibration_budget", "development_budget"): value for name, value in intervals.items()
            }
        points.append(node)
    return {
        "setting": record["epoch"],
        "label": "Unadapted Laya (epoch0)" if record["epoch"] == 0 else f"LoRA epoch{record['epoch']}",
        "nll": nll,
        "train_loss": record["train_loss"],
        "probabilities": probabilities.tolist(),
        "expected_usd": expected.tolist(),
        "points": points,
        "frontier_candidate_ids": sorted(frontier_ids),
        "randomized_hull_candidate_ids": curve["randomized_hull_candidate_ids"],
        "randomized_hull_interpretation": curve["randomized_hull_interpretation"],
    }


def build(run, training_grades, development_grades, *, draws=2000):
    run = Path(run)
    plan, metadata = core.checked_plan(run)
    training, development = core.read(training_grades), core.read(development_grades)
    train_rows, train_y, train_cost = core.grade_arrays(plan, metadata, training, "train")
    rows, y, costs = core.grade_arrays(plan, metadata, development, "development")
    core.require(training["cost_scope"] == development["cost_scope"], "Mixed answer-cost accounting")
    history, selected = epoch_history(run, plan, training_grades, development_grades, rows)
    cost_model = core.conditional_usd(train_y, train_cost)
    bootstrap = frontier.Bootstrap(rows, draws)
    settings = []
    for record in history:
        curve = curve_for_epoch(record, rows, y, costs, cost_model, bootstrap)
        curve["selected_by_development_nll"] = record["epoch"] == selected
        settings.append(curve)
    task_summary = []
    for task in sorted({row["task"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["task"] == task]
        task_summary.append(
            {
                "task": task,
                "n": len(indices),
                "families": len({rows[i]["family"] for i in indices}),
                "accuracy_percent": (100 * y[indices].mean(axis=0)).tolist(),
                "answer_usd_per_10000": (10000 * costs[indices].mean(axis=0)).tolist(),
            }
        )
    return {
        "format": "bbeh-development-candidate-frontiers-v1",
        "role": "development",
        "plan_sha256": core.sha(run / "plan.json"),
        "model_sha256": core.sha(run / "native/selected-epoch.json"),
        "source_plan_sha256": plan["source_plan_sha256"],
        "pair_sha256": plan["pair_sha256"],
        "grades_sha256": core.sha(development_grades),
        "training_grades_sha256": core.sha(training_grades),
        "selection_receipts_sha256": {
            name: core.sha(run / "native" / name)
            for name in ("training-start.json", "selected-epoch.json", *(f"epoch-{i}.json" for i in range(9)))
        },
        "ids": [row["id"] for row in rows],
        "n": len(rows),
        "families": len({row["family"] for row in rows}),
        "tasks": task_summary,
        "methods": {"lora": {"router_kind": "native", "metadata_assisted": False, "settings": settings}},
        "unadapted_base_setting": 0,
        "selected_settings": {"lora": selected},
        "conditional_cost_usd": cost_model,
        "cost_estimation_ids": [row["id"] for row in train_rows],
        "cost_estimation_role": "train",
        "random": {
            "analytic_expectation": True,
            "strong_probability_range": [0, 1],
            "accuracy_endpoints_percent": (100 * y.mean(axis=0)).tolist(),
            "answer_usd_per_10000_endpoints": (10000 * costs.mean(axis=0)).tolist(),
        },
        "cost_scope": development["cost_scope"],
        "router_cost_scope": "Native hosting unpriced in these curves; answer costs only. Always-arm bypasses have no router cost.",
        "interpretation": "Train-only selection models scored on development. This tuning population selects the declared epoch by NLL; curves and pointwise family intervals are diagnostics, not unbiased held-out estimates or simultaneous guarantees. No frontier point is selected here.",
        "refit_used_for_development_predictions": False,
        "additional_model_inference": False,
        "calibration_or_final_outcomes_accessed": False,
        "no_preferred_quality_target": True,
        "bootstrap": {
            "draws": draws,
            "seed": 89,
            "unit": "paired whole families within task; fixed observed task weights",
        },
        "implementation_sha256": {
            "development_report": core.sha(__file__),
            "native": core.sha(native.__file__),
            "recipe": core.sha(core.__file__),
            "frontier": core.sha(frontier.__file__),
            "utility": core.sha(utility.__file__),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--training-grades", type=Path, required=True)
    parser.add_argument("--development-grades", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    result = build(args.run, args.training_grades, args.development_grades, draws=args.draws)
    core.write(args.output, result)
    print({"output": str(args.output), "selected_epoch": result["selected_settings"]["lora"], "n": result["n"]})


if __name__ == "__main__":
    main()
