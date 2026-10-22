"""Inspect fixed train-only candidates on development; never recalibrate or refit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embedding import recipes as core
from evaluation import frontier, utility_router as utility


def build(run, grades_path, fees_path, *, draws=2000):
    run = Path(run)
    plan, model = core.checked_plan(run), core.read(run / "model.json")
    core.require(
        model["plan_sha256"] == core.sha(run / "plan.json")
        and model["source_plan_sha256"] == plan["source_plan_sha256"]
        and model["pair_sha256"] == plan["pair_sha256"]
        and model["development_grades_sha256"] == core.sha(grades_path)
        and model["calibration_or_final_outcomes_accessed"] is False,
        "Development model/grade lineage changed",
    )
    rows = core.role_rows(run, plan, "development")
    y, costs, grade = core.grades(plan, rows, grades_path, "development")
    fees = frontier.fee_rows(plan, rows, fees_path)
    bootstrap = frontier.Bootstrap(rows, draws)
    methods, seen = {}, set()
    for candidate in model["tuning"]:
        name, setting = candidate["method"], candidate["setting"]
        key = (name, setting)
        core.require(key not in seen, "Duplicate development candidate")
        seen.add(key)
        p = utility._matrix(
            candidate["probabilities"], "development probabilities", probability=True
        )
        expected = utility._matrix(
            candidate["expected_usd"], "development expected cost"
        )
        core.require(
            len(p) == len(expected) == len(rows), "Development score population changed"
        )
        nll = core.nll(y, p)
        core.require(
            abs(nll - candidate["nll"]) <= 1e-12,
            "Development NLL differs from fitted receipt",
        )
        priced = name in {"knn", "embedding"}
        overhead = fees if priced else None
        curve = frontier.enumerate_frontier(candidate, y, costs, overhead)
        frontier_ids = {node["candidate_id"] for node in curve["frontier"]}
        points = []
        for point in curve["candidates"]:
            node = {
                "candidate_id": point["candidate_id"],
                "rule": point["policy"],
                "development": point["calibration"],
                "development_matched_random_strong_probability": point[
                    "calibration_matched_random_strong_probability"
                ],
            }
            if point["candidate_id"] in frontier_ids:
                routes = utility.replay(point["policy"], p, expected)
                charged = (
                    np.zeros(len(rows))
                    if point["policy"]["mode"].startswith("always_")
                    else overhead
                )
                intervals = frontier.uncertainty(
                    bootstrap,
                    routes,
                    y,
                    costs,
                    charged,
                    point["calibration_matched_random_strong_probability"],
                )
                node["paired_ci95"] = {
                    key.replace("calibration_budget", "development_budget"): value
                    for key, value in intervals.items()
                }
            points.append(node)
        method = methods.setdefault(
            name,
            {
                "router_kind": "embedding" if priced else "cpu",
                "metadata_assisted": name == "task_metadata",
                "settings": [],
            },
        )
        method["settings"].append(
            {
                "setting": setting,
                "nll": nll,
                "selected_by_development_nll": model["selected"].get(name, "fixed")
                == setting,
                "probabilities": candidate["probabilities"],
                "expected_usd": candidate["expected_usd"],
                "points": points,
                "frontier_candidate_ids": sorted(frontier_ids),
                "randomized_hull_candidate_ids": curve["randomized_hull_candidate_ids"],
            }
        )
    expected_settings = {("knn", value) for value in core.PROTOCOL["neighbors"]}
    expected_settings |= {("embedding", value) for value in core.PROTOCOL["penalties"]}
    expected_settings |= {
        (name, "fixed") for name in ("text_length", "text_topic", "task_metadata")
    }
    core.require(seen == expected_settings, "Fixed development candidate set changed")
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
        "model_sha256": core.sha(run / "model.json"),
        "source_plan_sha256": plan["source_plan_sha256"],
        "pair_sha256": plan["pair_sha256"],
        "grades_sha256": core.sha(grades_path),
        "fees_sha256": core.sha(fees_path),
        "ids": [row["id"] for row in rows],
        "n": len(rows),
        "families": len({row["family"] for row in rows}),
        "tasks": task_summary,
        "methods": methods,
        "selected_settings": model["selected"],
        "random": {
            "analytic_expectation": True,
            "strong_probability_range": [0, 1],
            "accuracy_endpoints_percent": (100 * y.mean(axis=0)).tolist(),
            "answer_usd_per_10000_endpoints": (10000 * costs.mean(axis=0)).tolist(),
        },
        "cost_scope": grade["cost_scope"],
        "router_cost_scope": "Encoder singleton API fee for kNN/embedding; head/CPU hosting unpriced. Always-arm bypasses incur no router fee.",
        "interpretation": "Train-only predictions evaluated on the tuning population. Full curves and pointwise family intervals are development diagnostics, not unbiased held-out estimates or simultaneous guarantees. NLL alone chooses the declared k/penalty; no frontier point is selected here.",
        "refit_used_for_development_predictions": False,
        "calibration_or_final_outcomes_accessed": False,
        "bootstrap": {
            "draws": draws,
            "seed": 89,
            "unit": "paired whole families within task; fixed observed task weights",
        },
        "implementation_sha256": {
            "development_report": core.sha(__file__),
            "frontier": core.sha(frontier.__file__),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--grades", required=True)
    parser.add_argument("--fees", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    core.write(args.output, build(args.run, args.grades, args.fees))


if __name__ == "__main__":
    main()
