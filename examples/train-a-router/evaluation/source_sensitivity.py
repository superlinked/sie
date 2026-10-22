"""Replay unchanged frozen policies after a source-only, prespecified exclusion."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from embedding import recipes as core
from evaluation import frontier, sweep, utility_router as utility


def evaluate(run, frozen_path, grades_path, primary_path, specification_path, output):
    run = Path(run)
    plan = core.checked_plan(run)
    frozen, primary, specification = map(core.read, (frozen_path, primary_path, specification_path))
    core.require(
        primary["frozen_sha256"] == core.sha(frozen_path)
        and primary["grades_sha256"] == core.sha(grades_path)
        and primary["no_final_selection"] is True
        and frozen["plan_sha256"] == core.sha(run / "plan.json")
        and frozen["frozen_before_final_outcomes"] is True
        and specification["fixed_before_final_outcomes"] is True
        and specification["source_plan_sha256"] == plan["source_plan_sha256"],
        "Sensitivity requires the verified primary replay and original source-only specification",
    )
    rows = core.role_rows(run, plan, "final")
    y, costs, grade = core.grades(plan, rows, grades_path, "final")
    setting = specification["final_sensitivity"]
    excluded = set(setting["excluded_ids"])
    core.require(len(excluded) == len(setting["excluded_ids"]), "Duplicate sensitivity ID")
    declared = {r["id"]: r for r in specification["source_ambiguous_rows"] if r["role"] == "final"}
    core.require(excluded == set(declared), "Sensitivity differs from predeclared source ambiguities")
    for row in rows:
        if row["id"] in excluded:
            evidence = declared[row["id"]]
            core.require(
                row["family"] == evidence["family"]
                and hashlib.sha256(row["request"].encode()).hexdigest() == evidence["request_sha256"],
                "Excluded source input changed",
            )
    keep = np.array([row["id"] not in excluded for row in rows], dtype=bool)
    core.require(len(rows) == setting["n_primary"] and int(keep.sum()) == setting["n_sensitivity"],
                 "Sensitivity population changed")
    subset = [row for row, retained in zip(rows, keep, strict=True) if retained]
    y, costs = y[keep], costs[keep]
    bootstrap = frontier.Bootstrap(subset, primary["bootstrap_draws"])
    methods = {}
    for name, saved in frozen["methods"].items():
        score = saved["scores"]
        probabilities = np.asarray(score["probabilities"])[keep]
        expected = np.asarray(score["expected_usd"])[keep]
        fees = np.asarray(frozen["encoder_fees_usd"])[keep] if score["router_kind"] == "embedding" else None
        frontier_ids = set(saved["calibration_frontier_candidate_ids"])
        candidates = [row for row in saved["candidates"] if row["candidate_id"] in frontier_ids]
        metrics = sweep.sweep_metrics([row["policy"] for row in candidates], probabilities, expected, y, costs, fees)
        points = []
        for candidate, metric in zip(candidates, metrics, strict=True):
            policy = candidate["policy"]
            charged = np.zeros(len(subset)) if policy["mode"].startswith("always_") else fees
            routes = utility.replay(policy, probabilities, expected)
            points.append({
                "candidate_id": candidate["candidate_id"], "rule": policy, **metric,
                "paired_ci95": frontier.uncertainty(bootstrap, routes, y, costs, charged,
                    candidate["calibration_matched_random_strong_probability"]),
            })
        methods[name] = {"points": points, "router_kind": score["router_kind"],
                         "metadata_assisted": score["metadata_assisted"]}
    result = {
        "format": "bbeh-source-format-sensitivity-result-v1",
        "specification_sha256": core.sha(specification_path), "primary_report_sha256": core.sha(primary_path),
        "frozen_sha256": core.sha(frozen_path), "grades_sha256": core.sha(grades_path),
        "source_plan_sha256": plan["source_plan_sha256"], "excluded_ids": sorted(excluded),
        "n_primary": len(rows), "n": len(subset), "families": len({row["family"] for row in subset}),
        "random": {"accuracy_endpoints_percent": (100 * y.mean(axis=0)).tolist(),
                   "answer_usd_per_10000_endpoints": (10000 * costs.mean(axis=0)).tolist()},
        "methods": methods, "policies_unchanged": True, "no_refit_recalibration_or_selection": True,
        "cost_scope": grade["cost_scope"], "implementation_sha256": core.sha(__file__),
        "interpretation": "Supplementary source-quality sensitivity. Primary fixed-population results remain authoritative; no threshold or method is selected using this subset.",
    }
    core.write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("run", "frozen", "grades", "primary", "specification", "output"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    evaluate(args.run, args.frozen, args.grades, args.primary, args.specification, args.output)


if __name__ == "__main__":
    main()
