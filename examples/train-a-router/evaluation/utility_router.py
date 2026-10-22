"""Calibrate a frozen two-arm expected-utility rule; no model fitting.

Every matrix is N x 2 in cheap/strong order. Those arm names do not imply a
per-request cost or quality ordering. Costs use the caller's declared USD
basis; they may be measured charges or an explicitly labeled projection.

The rule selects strong when delta_probability - lambda * delta_USD >= 0.
Nonnegative lambda is calibrated against paired answers, never at dispatch.
Saved rational lambdas preserve exact crossings and intervals, including when
two crossings have no representable float between them. Arithmetic treats the
finite float64 inputs as exact values; no tolerance changes tie decisions.
"""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Real

import numpy as np

FORMAT = "expected-utility-policy-v1"
PROTOCOL = {
    "rule": "strong iff (p_strong-p_cheap) - lambda*(expected_USD_strong-expected_USD_cheap) >= 0",
    "lambda": "nonnegative; exact crossings, intervening intervals and the cost-minimizing limit",
    "tie": "strong on equal utility; at the cost limit, lower expected cost then higher probability then strong",
    "selection": "least calibration cost, then smaller signed quality loss, router bypass, smaller lambda, always-strong bypass tie",
    "cost": "Supplied USD basis; router overhead applies to every utility-routed request and to neither always-arm bypass.",
    "unknown_router_cost": "None means unpriced: select by answer costs and report no native net-cost claim.",
    "quality": "Empirical signed gap versus always strong; rescues offset regressions. Not a future-risk guarantee.",
    "arithmetic": "Float64 inputs are exact for utility/cost comparison; gap targets use their decimal spelling. No rounding tolerance changes ties.",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, *, binary=False):
    if binary:
        _require(isinstance(value, (Real, bool, np.bool_)) and value in (0, 1), f"{name} must be binary")
        return int(value)
    _require(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_)), f"{name} must be numeric, not boolean"
    )
    value = float(value)
    _require(math.isfinite(value) and value >= 0, f"{name} must be finite and nonnegative")
    return value


def _matrix(value, name, *, probability=False, binary=False):
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a nonempty N x 2 matrix") from error
    _require(array.ndim == 2 and array.shape[0] > 0 and array.shape[1] == 2, f"{name} must be a nonempty N x 2 matrix")
    rows = [[_number(item, name, binary=binary) for item in row] for row in array.tolist()]
    _require(not probability or all(item <= 1 for row in rows for item in row), "Probabilities must lie in [0,1]")
    return rows


def _fraction(value):
    return Fraction.from_float(float(value))


def _deltas(probabilities, expected_usd):
    probabilities = _matrix(probabilities, "probabilities", probability=True)
    expected_usd = _matrix(expected_usd, "expected_usd")
    _require(len(probabilities) == len(expected_usd), "Probability and expected-cost rows differ")
    return [
        (_fraction(p[1]) - _fraction(p[0]), _fraction(c[1]) - _fraction(c[0]))
        for p, c in zip(probabilities, expected_usd, strict=True)
    ]


def _policy(mode, penalty=None):
    result = {"format": FORMAT, "mode": mode}
    if mode == "utility":
        result["lambda"] = {"numerator": str(penalty.numerator), "denominator": str(penalty.denominator)}
    return result


def _read_policy(policy):
    _require(isinstance(policy, dict) and policy.get("format") == FORMAT, "Unknown policy format")
    mode = policy.get("mode")
    _require(
        isinstance(mode, str) and mode in {"always_strong", "always_cheap", "utility", "cost_limit"},
        "Unknown policy mode",
    )
    expected = {"format", "mode", "lambda"} if mode == "utility" else {"format", "mode"}
    _require(set(policy) == expected, "Unexpected policy fields")
    if mode != "utility":
        return mode, None
    ratio = policy["lambda"]
    _require(isinstance(ratio, dict) and set(ratio) == {"numerator", "denominator"}, "Invalid saved lambda")
    try:
        n, d = int(ratio["numerator"]), int(ratio["denominator"])
        _require(n >= 0 and d > 0, "Lambda must be nonnegative")
        penalty = Fraction(n, d)
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("Invalid saved lambda") from error
    _require(
        ratio == {"numerator": str(penalty.numerator), "denominator": str(penalty.denominator)},
        "Saved lambda must be a canonical exact fraction",
    )
    return mode, penalty


def replay(policy, probabilities, expected_usd):
    """Return strong-arm booleans, using only input-derived predictions.

    A serving caller must inspect an always-arm mode before running its router:
    that saved mode is a direct bypass and needs no probabilities or embedding.
    This array replay interface still validates the supplied evaluation inputs.
    """
    mode, penalty = _read_policy(policy)
    deltas = _deltas(probabilities, expected_usd)
    if mode.startswith("always_"):
        return np.full(len(deltas), mode == "always_strong", dtype=bool)
    if mode == "cost_limit":
        return np.array([cost < 0 or (cost == 0 and quality >= 0) for quality, cost in deltas], dtype=bool)
    return np.array([quality - penalty * cost >= 0 for quality, cost in deltas], dtype=bool)


def _float(value):
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError("Metric exceeds the finite reporting range") from error
    _require(math.isfinite(result), "Metric exceeds the finite reporting range")
    return result


def calibrate(probabilities, expected_usd, correct, observed_usd, *, targets_pp=(1, 2, 0, 5), router_usd=None):
    """Enumerate all utility decisions and choose cheapest feasible policies.

    ``correct`` contains both paired binary outcomes. ``observed_usd`` contains
    their costs under the declared accounting basis, including a common price
    projection when appropriate. ``router_usd`` is a scalar or N-vector charged
    on every utility-routed request. None leaves router pricing unknown.

    The sweep is O(N log N), apart from output size. It records crossings AND
    open intervals: simultaneous opposite-direction crossings can produce a
    boundary routing pattern different from both neighboring patterns.
    Full candidates and the nondominated calibration frontier are returned.
    Freeze the selected policies before accessing final outcomes.
    """
    deltas = _deltas(probabilities, expected_usd)
    y = _matrix(correct, "correct", binary=True)
    costs = _matrix(observed_usd, "observed_usd")
    n = len(deltas)
    _require(len(y) == len(costs) == n, "Calibration rows must align exactly")
    costs = [[_fraction(item) for item in row] for row in costs]
    targets = [_number(value, "targets_pp") for value in targets_pp]
    _require(
        bool(targets) and len(set(targets)) == len(targets) and max(targets) <= 100, "Invalid or duplicate gap targets"
    )
    if router_usd is None:
        overhead = None
    elif isinstance(router_usd, Real):
        overhead = n * _fraction(_number(router_usd, "router_usd"))
    else:
        values = np.asarray(router_usd, dtype=object)
        _require(values.shape == (n,), "Router costs must be a scalar or aligned N-vector")
        overhead = sum((_fraction(_number(value, "router_usd")) for value in values.tolist()), Fraction())

    strong_correct = sum(row[1] for row in y)
    strong_cost = sum((row[1] for row in costs), Fraction())
    candidates, internals = [], []

    def counts(decisions):
        return {
            "correct": sum(row[int(route)] for row, route in zip(y, decisions, strict=True)),
            "cost": sum((row[int(route)] for row, route in zip(costs, decisions, strict=True)), Fraction()),
            "cheap": sum(not route for route in decisions),
            "regressions": sum(not route and row[1] > row[0] for row, route in zip(y, decisions, strict=True)),
            "rescues": sum(not route and row[0] > row[1] for row, route in zip(y, decisions, strict=True)),
        }

    def add(policy, state):
        mode, penalty = _read_policy(policy)
        bypass = mode.startswith("always_")
        router = Fraction() if bypass else overhead
        selection_cost = state["cost"] + (router if router is not None else 0)
        loss = strong_correct - state["correct"]
        metric = {
            "answer_accuracy_percent": 100 * state["correct"] / n,
            "gap_vs_strong_pp": 100 * loss / n,
            "cheap_share": state["cheap"] / n,
            "regressions": int(state["regressions"]),
            "rescues": int(state["rescues"]),
            "answer_cost_usd_per10000": _float(10000 * state["cost"] / n),
            "router_cost_usd_per10000": None if router is None else _float(10000 * router / n),
            "total_cost_usd_per10000": None if router is None else _float(10000 * selection_cost / n),
            "selection_cost_usd_per10000": _float(10000 * selection_cost / n),
            "gross_answer_savings_percent": None
            if not strong_cost
            else _float(100 * (1 - state["cost"] / strong_cost)),
            "net_savings_percent": None
            if router is None or not strong_cost
            else _float(100 * (1 - selection_cost / strong_cost)),
        }
        record = {"candidate_id": len(candidates), "policy": policy, "calibration": metric}
        candidates.append(record)
        # Infinity follows every finite penalty; exact bypass ties prefer strong.
        tie = (not bypass, mode == "cost_limit", penalty or Fraction(), mode != "always_strong")
        internals.append((selection_cost, loss, tie, record))

    add(_policy("always_strong"), counts([True] * n))
    add(_policy("always_cheap"), counts([False] * n))
    decisions = [quality >= 0 for quality, _ in deltas]
    state = counts(decisions)
    add(_policy("utility", Fraction()), state)

    def flip(index, use_strong):
        if decisions[index] == use_strong:
            return
        sign = 1 if use_strong else -1
        state["correct"] += sign * (y[index][1] - y[index][0])
        state["cost"] += sign * (costs[index][1] - costs[index][0])
        state["cheap"] -= sign
        state["regressions"] -= sign * (y[index][1] > y[index][0])
        state["rescues"] -= sign * (y[index][0] > y[index][1])
        decisions[index] = use_strong

    crossings = {}
    for index, (quality, cost) in enumerate(deltas):
        if cost and quality / cost >= 0:
            crossings.setdefault(quality / cost, []).append(index)
    previous = Fraction()
    for crossing in sorted(crossings):
        if crossing > previous:
            add(_policy("utility", (previous + crossing) / 2), state)
        for index in crossings[crossing]:
            if deltas[index][1] < 0:
                flip(index, True)
        if crossing:
            add(_policy("utility", crossing), state)
        for index in crossings[crossing]:
            if deltas[index][1] > 0:
                flip(index, False)
        previous = crossing
    if crossings:
        add(_policy("utility", previous + 1), state)
    add(_policy("cost_limit"), state)

    ordered = sorted(internals, key=lambda item: (item[0], item[1], item[2], item[3]["candidate_id"]))
    frontier, best_loss = [], None
    for _, loss, _, record in ordered:
        if best_loss is None or loss < best_loss:
            frontier.append(record)
            best_loss = loss
    selected = {}
    for target in targets:
        winner = next(record for _, loss, _, record in ordered if 100 * loss <= n * Fraction(str(target)))
        key = str(int(target)) if target.is_integer() else repr(target)
        selected[key] = {"target_gap_pp": target, **winner}
    return {
        "format": "expected-utility-calibration-v1",
        "protocol": PROTOCOL,
        "n": n,
        "arm_order": ["cheap", "strong"],
        "selection_cost_basis": "answer USD plus router USD"
        if overhead is not None
        else "answer USD only; router unpriced",
        "targets_pp": targets,
        "candidates": candidates,
        "frontier": frontier,
        "selected": selected,
    }
