"""Inspect saved calibration candidates; optionally replay them on published tests.

No fitting, model calls, new thresholds or final-outcome policy selection occur.
The saved calibration grid need not exhaust every final-input routing pattern.
"""

from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import paired_calibration
import paired_report
import paired_training as core
import reproduction
import utility_router as utility

FORMAT = "train-a-router-calibration-frontier-v1"
LEARNED = ("base", "lora", "embedding", "topic")
METHODS = (*LEARNED, "always_strong", "always_cheap", "random")
PROVENANCE_KEYS = (
    "plan_sha256",
    "source_plan_sha256",
    "pair_spec_sha256",
    "acquisition_protocol_sha256",
    "grader_sha256",
    "calibration_grades_sha256",
    "training_grades_sha256",
    "fee_projection_sha256",
    "scores_sha256",
    "cost_scope",
    "calibration_source_sha256",
    "utility_source_sha256",
)


def load_calibration(run):
    """Return unchanged, hash-checked routes and their actual calibration file."""
    run = Path(run)
    plan, rows = core.checked_plan(run)
    bundle, envelope = reproduction.checked_routes(
        run / "frozen/frozen-final-routes.json", rows
    )
    core.require(
        bundle["plan_sha256"] == core.sha(run / "plan.json"),
        "Calibration run plan changed",
    )
    for key in (
        "source_plan_sha256",
        "pair_spec_sha256",
        "acquisition_protocol_sha256",
    ):
        core.require(
            bundle[key] == plan[key], "Calibration study binding changed: " + key
        )
    core.require(
        bundle["cost_scope"] == plan["pair"]["cost_scope"],
        "Calibration cost scope changed",
    )
    parent = run / "frozen" / ("internal" if envelope is not None else "")
    path = parent / "calibration.json"
    core.require(
        core.sha(path) == bundle["calibration_sha256"], "Saved calibration file changed"
    )
    calibration = core.read(path)
    core.require(
        calibration["format"] == "routerbench-paired-calibration-v1"
        and calibration["protocol"] == paired_calibration.PROTOCOL
        and calibration["ids"] == plan["populations"]["calibration"],
        "Calibration protocol or population changed",
    )
    for key in PROVENANCE_KEYS:
        core.require(
            calibration[key] == bundle[key], "Calibration provenance differs: " + key
        )
    core.require(
        bundle["calibration_source_sha256"] == core.sha(paired_calibration.__file__)
        and bundle["utility_source_sha256"] == core.sha(utility.__file__),
        "Saved calibration implementation changed",
    )
    for method in (*LEARNED, "topic_metadata_diagnostic"):
        result = calibration["methods"][method]
        candidates = result["candidates"]
        core.require(
            result["protocol"] == utility.PROTOCOL
            and result["n"] == len(calibration["ids"])
            and [row["candidate_id"] for row in candidates]
            == list(range(len(candidates)))
            and bool(candidates),
            "Invalid saved candidate population",
        )
        for row in candidates:
            utility._read_policy(row["policy"])
        for target, selected in result["selected"].items():
            candidate = candidates[selected["candidate_id"]]
            core.require(
                candidate["policy"] == selected["policy"],
                "Selected candidate rule changed",
            )
            frozen = bundle["policies"][f"{method}@{target}pp"]
            core.require(
                frozen["rule"] == candidate["policy"]
                and frozen["calibration_candidate_id"] == candidate["candidate_id"],
                "Selected route/candidate binding changed",
            )
    return bundle, calibration


def _fraction(value):
    return Fraction.from_float(float(value))


def _metrics(
    correct_count,
    answer_cost,
    strong_count,
    n,
    baseline_correct,
    baseline_cost,
    router_cost,
):
    total = None if router_cost is None else answer_cost + router_cost
    return {
        "answer_accuracy_percent": float(100 * correct_count / n),
        "quality_gap_pp": float(100 * (baseline_correct - correct_count) / n),
        "answer_usd_per_10000": float(10000 * answer_cost / n),
        "router_usd_per_10000": None
        if router_cost is None
        else float(10000 * router_cost / n),
        "total_usd_per_10000": None if total is None else float(10000 * total / n),
        "strong_share": float(strong_count / n),
        "answer_savings_percent": None
        if not baseline_cost
        else float(100 * (1 - answer_cost / baseline_cost)),
        "net_savings_percent": None
        if total is None or not baseline_cost
        else float(100 * (1 - total / baseline_cost)),
    }


def sweep_metrics(
    policies, probabilities, expected_usd, correct, observed_usd, router_usd=None
):
    """Evaluate K saved rules in O((N+K) log(N+K)) with exact crossing decisions.

    Observed costs only contribute to reporting. The rule uses the frozen
    predicted probabilities/costs; it never sees a selected answer's length.
    """
    deltas = utility._deltas(probabilities, expected_usd)
    labels = utility._matrix(correct, "correct", binary=True)
    costs = [
        [_fraction(v) for v in row]
        for row in utility._matrix(observed_usd, "observed_usd")
    ]
    n = len(deltas)
    core.require(len(labels) == len(costs) == n, "Sweep rows do not align")
    if router_usd is None:
        overhead = None
    else:
        core.require(len(router_usd) == n, "Sweep router fees do not align")
        overhead = sum(
            (_fraction(utility._number(v, "router_usd")) for v in router_usd),
            Fraction(),
        )
    parsed = [utility._read_policy(policy) for policy in policies]
    queries = sorted({penalty for mode, penalty in parsed if mode == "utility"})
    baseline_correct = sum(row[1] for row in labels)
    baseline_cost = sum((row[1] for row in costs), Fraction())
    decisions = [quality >= 0 for quality, _ in deltas]

    def counts(routes):
        return (
            sum(row[int(route)] for row, route in zip(labels, routes, strict=True)),
            sum(
                (row[int(route)] for row, route in zip(costs, routes, strict=True)),
                Fraction(),
            ),
            sum(routes),
        )

    state = list(counts(decisions))

    def flip(index, use_strong):
        if decisions[index] != use_strong:
            sign = 1 if use_strong else -1
            state[0] += sign * (labels[index][1] - labels[index][0])
            state[1] += sign * (costs[index][1] - costs[index][0])
            state[2] += sign
            decisions[index] = use_strong

    events = {}
    for index, (quality, cost) in enumerate(deltas):
        if cost and quality / cost >= 0:
            events.setdefault(quality / cost, []).append(index)
    events = sorted(events.items())
    position, snapshots = 0, {}
    for query in queries:
        while position < len(events) and events[position][0] < query:
            for index in events[position][1]:
                flip(index, deltas[index][1] < 0)
            position += 1
        equal = position < len(events) and events[position][0] == query
        if equal:
            # Both directions choose strong exactly on their shared crossing.
            for index in events[position][1]:
                if deltas[index][1] < 0:
                    flip(index, True)
        snapshots[query] = tuple(state)
        if equal:
            for index in events[position][1]:
                if deltas[index][1] > 0:
                    flip(index, False)
            position += 1
    limit = counts(
        [cost < 0 or (cost == 0 and quality >= 0) for quality, cost in deltas]
    )
    bypass = {
        "always_strong": (baseline_correct, baseline_cost, n),
        "always_cheap": counts([False] * n),
    }
    result = []
    for mode, penalty in parsed:
        values = (
            bypass[mode]
            if mode in bypass
            else limit
            if mode == "cost_limit"
            else snapshots[penalty]
        )
        fee = Fraction() if mode in bypass else overhead
        result.append(_metrics(*values, n, baseline_correct, baseline_cost, fee))
    return result


def _saved_metrics(values):
    return {
        "answer_accuracy_percent": values["answer_accuracy_percent"],
        "quality_gap_pp": values["gap_vs_strong_pp"],
        "answer_usd_per_10000": values["answer_cost_usd_per10000"],
        "router_usd_per_10000": values["router_cost_usd_per10000"],
        "total_usd_per_10000": values["total_cost_usd_per10000"],
        "strong_share": 1 - values["cheap_share"],
        "answer_savings_percent": values["gross_answer_savings_percent"],
        "net_savings_percent": values["net_savings_percent"],
    }


def _point(method, candidate_id, rule, metrics, **extra):
    return {
        "method": method,
        "candidate_id": candidate_id,
        "point_id": f"{method}:{candidate_id}",
        "rule": deepcopy(rule),
        **metrics,
        "calibration_pareto": False,
        "selected_targets_pp": [],
        "analytic_random": False,
        **extra,
    }


def _method_points(method, saved, metrics):
    frontier = {row["candidate_id"] for row in saved["frontier"]}
    targets = {}
    for target, selected in saved["selected"].items():
        targets.setdefault(selected["candidate_id"], []).append(float(target))
    return [
        _point(
            method,
            row["candidate_id"],
            row["policy"],
            metric,
            calibration_pareto=row["candidate_id"] in frontier,
            selected_targets_pp=sorted(targets.get(row["candidate_id"], [])),
        )
        for row, metric in zip(saved["candidates"], metrics, strict=True)
    ]


def _controls(baselines, random):
    points = []
    for arm in ("strong", "cheap"):
        method = "always_" + arm
        points.append(
            _point(
                method,
                None,
                {"format": utility.FORMAT, "mode": method},
                baselines[arm],
                point_id=method,
            )
        )
    for name, p, rule, targets in (
        ("cheap-endpoint", 0.0, {"strong_probability": 0.0}, []),
        ("strong-endpoint", 1.0, {"strong_probability": 1.0}, []),
        *(
            ("target-" + key, row["strong_probability"], row, [float(key)])
            for key, row in random.items()
        ),
    ):
        metrics = {
            key: (1 - p) * baselines["cheap"][key] + p * baselines["strong"][key]
            for key in baselines["cheap"]
            if baselines["cheap"][key] is not None
            and baselines["strong"][key] is not None
        }
        metrics["strong_share"] = p
        # When always-strong itself costs zero, percentage savings is undefined.
        for key in baselines["cheap"]:
            metrics.setdefault(key, None)
        points.append(
            _point(
                "random",
                None,
                rule,
                metrics,
                point_id="random:" + name,
                selected_targets_pp=targets,
                analytic_random=True,
            )
        )
    return points


def _checked_test_inputs(data, run, bundle, plan, rows):
    original_path = data / "reference/frozen/frozen-final-routes.json"
    original, original_envelope = reproduction.checked_routes(original_path, rows)
    core.require(
        original_envelope is None,
        "Original acquisition needs its original frozen routes",
    )
    outer = core.read(run / "frozen/frozen-final-routes.json")
    if outer["format"] == reproduction.FORMAT:
        core.require(
            outer["original_frozen_routing_sha256"] == core.sha(original_path),
            "Original acquisition reference changed",
        )
    else:
        core.require(
            core.sha(run / "frozen/frozen-final-routes.json")
            == core.sha(original_path),
            "Original route binding changed",
        )
    for key in (
        "source_plan_sha256",
        "pair_spec_sha256",
        "acquisition_protocol_sha256",
        "grader_sha256",
        "cost_scope",
    ):
        core.require(
            bundle[key] == original[key], "Sweep/original lineage differs: " + key
        )
    core.require(
        bundle["source_rows"] == original["source_rows"],
        "Sweep final population changed",
    )
    grades_path = data / "recordings/grades-final.json"
    grades = core.read(grades_path)
    paired_report.check_lineage(original, grades, core.sha(original_path))
    core.require(
        grades["arms"] == plan["pair"]["arms"]
        and grades["pair"] == plan["pair"]["pair"],
        "Final model identities changed",
    )
    core.require(
        [r["id"] for r in grades["rows"]] == plan["populations"]["final"]
        and all(r["role"] == "final" for r in grades["rows"]),
        "Final grade order/role changed",
    )
    labels = [[r["cheap_correct"], r["strong_correct"]] for r in grades["rows"]]
    costs = [[r["cheap_cost_usd"], r["strong_cost_usd"]] for r in grades["rows"]]
    fees_path = data / "fees/projection.json"
    core.require(
        core.sha(fees_path) == bundle["fee_projection_sha256"],
        "Embedding fee projection changed",
    )
    fees = paired_calibration.checked_fees(core.read(fees_path), rows)["final"]
    scores = {}
    for method in ("embedding", "native", "topic"):
        path = run / f"{method}-scores.json"
        core.require(
            core.sha(path) == bundle["scores_sha256"][method],
            "Input-only scores changed: " + method,
        )
        scores[method] = core.read(path)
        core.require(
            scores[method]["plan_sha256"] == bundle["plan_sha256"]
            and scores[method]["outcomes_accessed"] is False,
            "Score plan or input-only boundary changed",
        )
    variants = {
        "embedding": scores["embedding"],
        "topic": scores["topic"],
        **scores["native"]["variants"],
    }
    methods = {
        name: paired_calibration.checked_score_roles(value, plan)["final"]
        for name, value in variants.items()
    }
    return grades_path, labels, costs, fees, methods


def _verify_selected(bundle, calibration, methods):
    """Check every original target mask, including the excluded metadata diagnostic."""
    count = 0
    for name, policy in bundle["policies"].items():
        method, mode = policy["method"], policy["kind"]
        saved = policy["strong_probabilities"]
        if method in (*LEARNED, "topic_metadata_diagnostic"):
            score = (
                methods["topic"]["source_topic_diagnostic"]
                if method == "topic_metadata_diagnostic"
                else methods[method]
            )
            actual = (
                utility.replay(
                    policy["rule"], score["probabilities"], score["expected_usd"]
                )
                .astype(int)
                .tolist()
            )
        elif mode == "random":
            target = str(int(policy["target_gap_pp"]))
            actual = [calibration["random"][target]["strong_probability"]] * len(saved)
        else:
            actual = [int(method == "always_strong")] * len(saved)
        core.require(
            actual == saved, "Saved target decisions differ from score replay: " + name
        )
        count += 1
    return count


def build(data, run=None, include_test=False):
    """Build calibration-first plot data without modifying any input artifact."""
    core.require(type(include_test) is bool, "include_test must be an explicit boolean")
    data = Path(data)
    run = Path(run) if run is not None else data / "reference"
    bundle, calibration = load_calibration(run)
    points, baselines = [], {}
    for method in LEARNED:
        saved = calibration["methods"][method]
        metrics = [_saved_metrics(row["calibration"]) for row in saved["candidates"]]
        points.extend(_method_points(method, saved, metrics))
        if method == LEARNED[0]:
            for row, metric in zip(saved["candidates"], metrics, strict=True):
                if row["policy"]["mode"].startswith("always_"):
                    baselines[row["policy"]["mode"].removeprefix("always_")] = metric
    points.extend(_controls(baselines, calibration["random"]))
    result = {
        "format": FORMAT,
        "methods": list(METHODS),
        "calibration": points,
        "provenance": {
            **{key: bundle[key] for key in PROVENANCE_KEYS},
            "calibration_sha256": bundle["calibration_sha256"],
            "routing_sha256": core.sha(run / "frozen/frozen-final-routes.json"),
            "analysis_source_sha256": core.sha(__file__),
        },
        "interpretation": {
            "analysis_mode": "saved_calibration_candidates",
            "test_outcomes_accessed": include_test,
            "policy_selection_split": "calibration",
            "new_blind_evaluation": False,
            "grid": "Every saved calibration candidate; not every possible final-input crossing.",
            "test": "Opt-in retrospective evaluation of these candidates; a test Pareto envelope is descriptive hindsight.",
            "random": "Analytical expected mixtures of always-cheap and always-strong; connecting endpoints is not an observed randomized run.",
            "cost": "Embedding includes validated singleton encoder fees but excludes application hosting. Native/topic hosting is unpriced; null totals are not free hosting. Always-arm bypasses avoid router fees.",
            "uncertainty": "Sweep points have no simultaneous confidence band. Original fixed-target intervals remain in the published report.",
            "weighting": "Equal weight per question within each split; related families remain grouped.",
        },
    }
    if not include_test:
        return result
    plan, rows = core.checked_plan(run)
    grade_path, labels, costs, fees, methods = _checked_test_inputs(
        data, run, bundle, plan, rows
    )
    result["provenance"]["verified_saved_target_masks"] = _verify_selected(
        bundle, calibration, methods
    )
    result["provenance"]["final_grades_sha256"] = core.sha(grade_path)
    result["test"] = []
    for method in LEARNED:
        saved, score = calibration["methods"][method], methods[method]
        metrics = sweep_metrics(
            [row["policy"] for row in saved["candidates"]],
            score["probabilities"],
            score["expected_usd"],
            labels,
            costs,
            fees if method == "embedding" else None,
        )
        result["test"].extend(_method_points(method, saved, metrics))
        if method == LEARNED[0]:
            baselines = {
                row["policy"]["mode"].removeprefix("always_"): metric
                for row, metric in zip(saved["candidates"], metrics, strict=True)
                if row["policy"]["mode"].startswith("always_")
            }
    result["test"].extend(_controls(baselines, calibration["random"]))
    return result
