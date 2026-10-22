"""Exact metric sweep copied from the validated public frontier implementation."""

from fractions import Fraction
from embedding import recipes as core
from evaluation import utility_router as utility

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
