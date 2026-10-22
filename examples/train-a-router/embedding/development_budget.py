"""Compact descriptive development comparisons at fixed answer-cost budgets."""

from __future__ import annotations

from embedding import recipes as core

FRACTIONS = (0.2, 0.4, 0.6, 0.8)


def summarize(report):
    core.require(
        report["role"] == "development"
        and report["refit_used_for_development_predictions"] is False
        and report["calibration_or_final_outcomes_accessed"] is False,
        "Only train-fitted development diagnostics are allowed",
    )
    low_cost, high_cost = report["random"]["answer_usd_per_10000_endpoints"]
    low_quality, high_quality = report["random"]["accuracy_endpoints_percent"]
    core.require(
        high_cost > low_cost, "Budget interpretation needs higher mean Sol cost"
    )
    methods = {}
    for name, method in report["methods"].items():
        selected = [s for s in method["settings"] if s["selected_by_development_nll"]]
        core.require(
            len(selected) == 1, "Exactly one NLL-selected or fixed recipe required"
        )
        setting = selected[0]
        points = []
        for fraction in FRACTIONS:
            budget = fraction * high_cost
            eligible = [
                p
                for p in setting["points"]
                if p["development"]["answer_cost_usd_per10000"] <= budget
            ]
            random_p = (budget - low_cost) / (high_cost - low_cost)
            random_quality = (
                low_quality + random_p * (high_quality - low_quality)
                if 0 <= random_p <= 1
                else None
            )
            row = {
                "fraction_of_always_sol_answer_cost": fraction,
                "answer_budget_usd_per_10000": budget,
                "analytic_random_accuracy_at_budget_percent": random_quality,
                "analytic_random_strong_probability": random_p
                if 0 <= random_p <= 1
                else None,
                "available": bool(eligible),
            }
            if eligible:
                best = min(
                    eligible,
                    key=lambda p: (
                        -p["development"]["answer_accuracy_percent"],
                        p["development"]["answer_cost_usd_per10000"],
                        p["candidate_id"],
                    ),
                )
                metric = best["development"]
                spend = metric["answer_cost_usd_per10000"]
                matched_p = (spend - low_cost) / (high_cost - low_cost)
                matched_quality = (
                    low_quality + matched_p * (high_quality - low_quality)
                    if 0 <= matched_p <= 1
                    else None
                )
                row.update(
                    candidate_id=best["candidate_id"],
                    rule=best["rule"],
                    metrics=metric,
                    paired_ci95=best.get("paired_ci95"),
                    accuracy_gain_over_budget_random_pp=(
                        metric["answer_accuracy_percent"] - random_quality
                        if random_quality is not None
                        else None
                    ),
                    accuracy_gain_over_actual_spend_random_pp=(
                        metric["answer_accuracy_percent"] - matched_quality
                        if matched_quality is not None
                        else None
                    ),
                )
            points.append(row)
        methods[name] = {
            "metadata_assisted": method["metadata_assisted"],
            "selected_setting": setting["setting"],
            "selected_nll": setting["nll"],
            "candidate_nlls": [
                {"setting": s["setting"], "nll": s["nll"]} for s in method["settings"]
            ],
            "budget_points": points,
        }
    return {
        "format": "bbeh-development-answer-budget-diagnostic-v1",
        "role": "development",
        "n": report["n"],
        "families": report["families"],
        "model_sha256": report["model_sha256"],
        "grades_sha256": report["grades_sha256"],
        "baselines": report["random"],
        "methods": methods,
        "interpretation": (
            "Exploratory development-only accuracy maxima under four declared answer-cost budgets; "
            "the selected k/penalty remains the NLL choice. These operating points are not frozen "
            "calibration/final choices. Pointwise intervals ignore this curve/threshold selection. "
            "Random comparators are analytic mixtures, not observed runs. Encoder and hosting "
            "costs are unpriced here; no net-savings claim. Metadata-assisted task control is diagnostic."
        ),
        "calibration_or_final_outcomes_accessed": False,
    }
