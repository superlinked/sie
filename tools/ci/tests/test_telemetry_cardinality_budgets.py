"""Every telemetry cardinality budget's limit equals the product of its formula's factors."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = yaml.safe_load((ROOT / "telemetry/contract.yaml").read_text())
BUDGETS = CONTRACT["cardinality_budgets"]
LIMITS = [
    pytest.param(budget_name, group, id=f"{budget_name}.{group}")
    for budget_name, budget in BUDGETS.items()
    for group in budget.get("instrument_limits") or {}
]


def factor(budget: dict, token: str) -> int:
    """Resolve one formula factor to its series count.

    ``<name>_series_pairs`` is the ``total_series_pairs`` of the budget's
    ``<name>_model_profile_pairs`` block. Any other factor is a fixed domain:
    an integer, or the name of a contract enum whose size is the factor.
    """
    if token.endswith("_series_pairs"):
        pairs = budget.get(token.removesuffix("_series_pairs") + "_model_profile_pairs")
        assert pairs is not None, f"formula factor {token!r} has no model/profile pair block"
        return pairs["total_series_pairs"]
    domains = budget.get("fixed_domains") or {}
    assert token in domains, f"formula factor {token!r} is not a fixed domain"
    domain = domains[token]
    if isinstance(domain, int):
        return domain
    assert domain in CONTRACT["enums"], f"fixed domain {token!r} names unknown enum {domain!r}"
    return len(CONTRACT["enums"][domain])


def test_budgets_declare_instrument_limits() -> None:
    assert LIMITS


@pytest.mark.parametrize(("budget_name", "group"), LIMITS)
def test_instrument_limit_equals_its_formula(budget_name: str, group: str) -> None:
    budget = BUDGETS[budget_name]
    declared = budget["instrument_limits"][group]
    tokens = [token.strip() for token in declared["formula"].split("*")]
    assert declared["limit"] == math.prod(factor(budget, token) for token in tokens), declared["formula"]
