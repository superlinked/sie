"""Whole calibration frontier and unchanged final replay; no model fitting."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np

from embedding import recipes as core
from evaluation import sweep, utility_router as utility

PROTOCOL = {
    "format": "bbeh-frontier-evaluation-v1",
    "population": "Uniform per retained question; exact original source roles/families",
    "thresholds": "All calibration crossings/intervals, cost limit and always-arm bypasses; no preferred gap target",
    "final": "Replay saved calibration candidates only; never select on final quality",
    "bootstrap": {"draws": 2000, "seed": 89, "unit": "paired whole family within task, fixed observed task weights"},
    "interval_scope": (
        "Pointwise for calibration-frontier rules, conditional on observed tasks; "
        "no simultaneous curve guarantee"
    ),
    "random": "Analytic expected mixtures; p for each matched calibration-budget comparator is frozen on calibration",
    "cost": (
        "Selected answer USD plus validated singleton encoder fee when applicable; "
        "other hosting unpriced, no zero imputation"
    ),
}


def checked_scores(plan, rows, paths, role):
    methods, bindings = {}, {}
    for path in paths:
        doc = core.read(path)
        core.require(
            doc["format"] == "bbeh-input-only-scores-v1"
            and doc["role"] == role
            and doc["source_plan_sha256"] == plan["source_plan_sha256"]
            and doc["pair_sha256"] == plan["pair_sha256"]
            and doc["outcomes_accessed"] is False
            and doc["ids"] == [r["id"] for r in rows],
            "Score source/role/order mismatch",
        )
        core.require_hash(doc["model_sha256"])
        for name, value in doc["methods"].items():
            core.require(
                name not in methods
                and value["router_kind"] in {"embedding", "cpu", "native"}
                and type(value["metadata_assisted"]) is bool,
                "Duplicate/invalid score method",
            )
            p = utility._matrix(value["probabilities"], "probabilities", probability=True)
            c = utility._matrix(value["expected_usd"], "expected USD")
            core.require(len(p) == len(c) == len(rows), "Score count mismatch")
            methods[name] = value
            bindings[name] = {"score_sha256": core.sha(path), "model_sha256": doc["model_sha256"]}
    core.require(bool(methods), "No methods supplied")
    return methods, bindings


def fee_rows(plan, rows, path):
    info = core.read(path)
    core.require(
        info["source_plan_sha256"] == plan["source_plan_sha256"]
        and info["model"] == core.MODEL
        and info["hf_revision"] == core.REVISION
        and info["instruction"] == core.INSTRUCTION
        and info["cost_basis"] in {"observed_singleton", "validated_singleton_projection"},
        "Need a verified per-request encoder fee basis",
    )
    by_id = {row["id"]: row for row in info["rows"]}
    core.require(len(by_id) == len(info["rows"]), "Duplicate fee ID")
    values = [utility._number(float(by_id[row["id"]]["projected_singleton_usd"]), "fee USD") for row in rows]
    return values


def random_probability(spend, cheap, strong):
    if strong == cheap:
        return None
    value = (spend - cheap) / (strong - cheap)
    return float(value) if 0 <= value <= 1 else None


def randomized_hull(candidates):
    """Upper quality hull over already nondominated cost-ordered candidates."""
    hull = []
    for row in candidates:
        metric = row["calibration"]
        x = Fraction.from_float(metric["selection_cost_usd_per10000"])
        y = Fraction.from_float(metric["answer_accuracy_percent"])
        while len(hull) >= 2:
            _, ax, ay = hull[-2]
            _, bx, by = hull[-1]
            if (by - ay) * (x - bx) > (y - by) * (bx - ax):
                break
            hull.pop()
        hull.append((row["candidate_id"], x, y))
    return [row[0] for row in hull]


def enumerate_frontier(value, y, costs, fees):
    # The reused implementation requires a nonempty target list. Only its
    # target-independent enumeration is retained; no target choice is exposed.
    result = utility.calibrate(
        value["probabilities"], value["expected_usd"], y, costs, router_usd=fees, targets_pp=(0,)
    )
    result.pop("selected")
    result.pop("targets_pp")
    result["no_preferred_quality_target"] = True
    result["randomized_hull_candidate_ids"] = randomized_hull(result["frontier"])
    result["randomized_hull_interpretation"] = "Mixtures between saved rules; not additional deterministic policies"
    cheap, strong = costs.mean(axis=0)
    for row in result["candidates"]:
        spend = row["calibration"]["selection_cost_usd_per10000"] / 10000
        row["calibration_matched_random_strong_probability"] = random_probability(spend, cheap, strong)
    return result


def calibrate(run, grade_path, score_paths, fees_path, output):
    run = Path(run)
    plan = core.checked_plan(run)
    rows = core.role_rows(run, plan, "calibration")
    y, costs, grade = core.grades(plan, rows, grade_path, "calibration")
    scores, bindings = checked_scores(plan, rows, score_paths, "calibration")
    fees = fee_rows(plan, rows, fees_path)
    methods = {}
    for name, value in scores.items():
        result = enumerate_frontier(value, y, costs, fees if value["router_kind"] == "embedding" else None)
        methods[name] = {
            **result,
            "metadata_assisted": value["metadata_assisted"],
            "router_kind": value["router_kind"],
            **bindings[name],
        }
    result = {
        "format": "bbeh-calibration-frontiers-v1",
        "protocol": PROTOCOL,
        "plan_sha256": core.sha(run / "plan.json"),
        "source_plan_sha256": plan["source_plan_sha256"],
        "pair_sha256": plan["pair_sha256"],
        "grades_sha256": core.sha(grade_path),
        "grader_sha256": grade["grader_sha256"],
        "cost_scope": grade["cost_scope"],
        "fees_sha256": core.sha(fees_path),
        "ids": [r["id"] for r in rows],
        "methods": methods,
        "random": {"strong_probability_range": [0, 1], "analytic_expectation": True},
        "implementation_sha256": {
            "frontier": core.sha(__file__),
            "utility": core.sha(utility.__file__),
            "sweep": core.sha(sweep.__file__),
        },
        "final_outcomes_accessed": False,
    }
    core.write(output, result)
    return result


def freeze(run, calibration_path, score_paths, fees_path, output):
    run = Path(run)
    plan, calibration = core.checked_plan(run), core.read(calibration_path)
    core.require(
        calibration["format"] == "bbeh-calibration-frontiers-v1"
        and calibration["protocol"] == PROTOCOL
        and calibration["plan_sha256"] == core.sha(run / "plan.json")
        and calibration["fees_sha256"] == core.sha(fees_path)
        and calibration["implementation_sha256"]["frontier"] == core.sha(__file__),
        "Calibration source/protocol changed",
    )
    rows = core.role_rows(run, plan, "final")
    methods, bindings = checked_scores(plan, rows, score_paths, "final")
    core.require(set(methods) == set(calibration["methods"]), "All calibrated methods need final input scores")
    fees = fee_rows(plan, rows, fees_path)
    saved = {}
    for name, scores in methods.items():
        prior = calibration["methods"][name]
        core.require(
            prior["model_sha256"] == bindings[name]["model_sha256"]
            and prior["router_kind"] == scores["router_kind"]
            and prior["metadata_assisted"] == scores["metadata_assisted"],
            "Model changed after calibration",
        )
        saved[name] = {
            "scores": scores,
            **bindings[name],
            "candidates": prior["candidates"],
            "calibration_frontier_candidate_ids": [r["candidate_id"] for r in prior["frontier"]],
            "calibration_randomized_hull_candidate_ids": prior["randomized_hull_candidate_ids"],
        }
    result = {
        "format": "bbeh-frozen-frontiers-v1",
        "protocol": PROTOCOL,
        "plan_sha256": core.sha(run / "plan.json"),
        "source_plan_sha256": plan["source_plan_sha256"],
        "pair_sha256": plan["pair_sha256"],
        "calibration_sha256": core.sha(calibration_path),
        "calibration_grader_sha256": calibration["grader_sha256"],
        "cost_scope": calibration["cost_scope"],
        "source_rows": [{k: r[k] for k in ("id", "dataset", "task", "family", "role")} for r in rows],
        "ids": [r["id"] for r in rows],
        "methods": saved,
        "encoder_fees_usd": fees,
        "fees_sha256": core.sha(fees_path),
        "final_outcomes_accessed": False,
        "frozen_before_final_outcomes": True,
        "implementation_sha256": calibration["implementation_sha256"],
    }
    core.write(output, result)
    return result


class Bootstrap:
    def __init__(self, rows, draws=2000):
        self.parts, self.draws = [], draws
        families = {}
        for row in rows:
            core.require(
                families.setdefault(row["family"], row["task"]) == row["task"],
                "A family spans tasks; requires a joint resampling design",
            )
        rng = np.random.default_rng(89)
        for task in sorted({r["task"] for r in rows}):
            ids = [i for i, r in enumerate(rows) if r["task"] == task]
            families = sorted({rows[i]["family"] for i in ids})
            group = np.array([families.index(rows[i]["family"]) for i in ids])
            weights = rng.multinomial(len(families), [1 / len(families)] * len(families), size=draws)
            denominator = weights @ np.bincount(group, minlength=len(families))
            self.parts.append((ids, group, weights, denominator, len(ids) / len(rows)))

    def means(self, values):
        values = np.asarray(values)
        result = np.zeros((self.draws, values.shape[1]))
        for ids, group, weights, denominator, proportion in self.parts:
            sums = np.array(
                [
                    np.bincount(group, weights=values[ids, col], minlength=weights.shape[1])
                    for col in range(values.shape[1])
                ]
            ).T
            result += proportion * (weights @ sums) / denominator[:, None]
        return result


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return {
        "lower": float(np.quantile(finite, 0.025)) if len(finite) else None,
        "upper": float(np.quantile(finite, 0.975)) if len(finite) else None,
        "defined_draws": len(finite),
        "total_draws": len(values),
    }


def uncertainty(bootstrap, routes, y, costs, fees, random_p):
    index = np.arange(len(y))
    chosen_y, chosen_cost = y[index, routes.astype(int)], costs[index, routes.astype(int)]
    fee_array = np.zeros(len(y)) if fees is None else np.asarray(fees)
    matrix = np.column_stack((y[:, 0], y[:, 1], costs[:, 0], costs[:, 1], chosen_y, chosen_cost, fee_array))
    means = bootstrap.means(matrix)
    baseline = means[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        values = {
            "accuracy_percent": 100 * means[:, 4],
            "quality_gap_pp": 100 * (means[:, 1] - means[:, 4]),
            "accuracy_gain_over_luna_pp": 100 * (means[:, 4] - means[:, 0]),
            "answer_usd_per_10000": 10000 * means[:, 5],
            "answer_savings_percent": 100 * (1 - means[:, 5] / baseline),
        }
        if fees is not None:
            values["total_usd_per_10000"] = 10000 * (means[:, 5] + means[:, 6])
            values["net_savings_percent"] = 100 * (1 - (means[:, 5] + means[:, 6]) / baseline)
        if random_p is not None:
            random_y = (1 - random_p) * means[:, 0] + random_p * means[:, 1]
            random_cost = (1 - random_p) * means[:, 2] + random_p * means[:, 3]
            values["accuracy_gain_over_calibration_budget_random_pp"] = 100 * (means[:, 4] - random_y)
            values["cost_difference_from_calibration_budget_random_usd_per_10000"] = 10000 * (
                means[:, 5] + means[:, 6] - random_cost
            )
    return {name: interval(value) for name, value in values.items()}


def evaluate(run, frozen_path, grades_path, output, *, draws=2000):
    run = Path(run)
    plan, frozen = core.checked_plan(run), core.read(frozen_path)
    core.require(
        frozen["format"] == "bbeh-frozen-frontiers-v1"
        and frozen["protocol"] == PROTOCOL
        and frozen["plan_sha256"] == core.sha(run / "plan.json")
        and frozen["implementation_sha256"]["frontier"] == core.sha(__file__)
        and frozen["frozen_before_final_outcomes"] is True,
        "Missing original frozen lineage",
    )
    rows = core.role_rows(run, plan, "final")
    metadata = [{k: r[k] for k in ("id", "dataset", "task", "family", "role")} for r in rows]
    core.require(
        frozen["source_rows"] == metadata and frozen["ids"] == [r["id"] for r in rows],
        "Final source population changed",
    )
    y, costs, grade = core.grades(plan, rows, grades_path, "final")
    core.require(
        grade["grader_sha256"] == frozen["calibration_grader_sha256"]
        or grade.get("parent_grader_sha256") == frozen["calibration_grader_sha256"],
        "Final grading semantics changed",
    )
    bootstrap = Bootstrap(rows, draws)
    methods = {}
    for name, saved in frozen["methods"].items():
        scores = saved["scores"]
        fees = frozen["encoder_fees_usd"] if scores["router_kind"] == "embedding" else None
        candidates = saved["candidates"]
        points = sweep.sweep_metrics(
            [r["policy"] for r in candidates], scores["probabilities"], scores["expected_usd"], y, costs, fees
        )
        result = []
        for candidate, metrics in zip(candidates, points, strict=True):
            node = {"candidate_id": candidate["candidate_id"], "rule": candidate["policy"], **metrics}
            if candidate["candidate_id"] in saved["calibration_frontier_candidate_ids"]:
                routes = utility.replay(candidate["policy"], scores["probabilities"], scores["expected_usd"])
                charged = np.zeros(len(rows)) if candidate["policy"]["mode"].startswith("always_") else fees
                p = candidate["calibration_matched_random_strong_probability"]
                node["calibration_matched_random_strong_probability"] = p
                node["paired_ci95"] = uncertainty(bootstrap, routes, y, costs, charged, p)
            result.append(node)
        methods[name] = {
            "points": result,
            "metadata_assisted": scores["metadata_assisted"],
            "router_kind": scores["router_kind"],
            "calibration_frontier_candidate_ids": saved["calibration_frontier_candidate_ids"],
            "native_or_cpu_hosting_priced": False,
        }
    result = {
        "format": "bbeh-frontier-final-report-v1",
        "protocol": PROTOCOL,
        "frozen_sha256": core.sha(frozen_path),
        "grades_sha256": core.sha(grades_path),
        "source_plan_sha256": plan["source_plan_sha256"],
        "pair_sha256": plan["pair_sha256"],
        "n": len(rows),
        "families": len({r["family"] for r in rows}),
        "methods": methods,
        "random": {
            "analytic_expectation": True,
            "strong_probability_range": [0, 1],
            "accuracy_endpoints_percent": (100 * y.mean(axis=0)).tolist(),
            "answer_usd_per_10000_endpoints": (10000 * costs.mean(axis=0)).tolist(),
        },
        "bootstrap_draws": draws,
        "no_final_selection": True,
        "cost_scope": grade["cost_scope"],
        "router_cost_scope": (
            "Embedding answer+encoder API fees; local head hosting unpriced. "
            "Native/text/task hosting unpriced: answer-only curves."
        ),
        "latency": None,
        "latency_note": (
            "Attach independently measured resident inference receipts; "
            "batch-fit timing is not serving latency."
        ),
    }
    core.write(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "freeze", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--output", required=True)
        if name in {"calibrate", "evaluate"}:
            p.add_argument("--grades", required=True)
        if name in {"calibrate", "freeze"}:
            p.add_argument("--scores", action="append", required=True)
            p.add_argument("--fees", required=True)
        if name == "freeze":
            p.add_argument("--calibration", required=True)
        if name == "evaluate":
            p.add_argument("--frozen", required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args.run, args.grades, args.scores, args.fees, args.output)
    elif args.command == "freeze":
        freeze(args.run, args.calibration, args.scores, args.fees, args.output)
    else:
        evaluate(args.run, args.frozen, args.grades, args.output)


if __name__ == "__main__":
    main()
