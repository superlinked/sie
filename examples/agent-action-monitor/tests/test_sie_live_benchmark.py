"""Live SIE benchmark: skipped until DUSK_SIE_ENDPOINT reaches an SIE cluster.

Once SIE is actually reachable (self-hosted container or the Superlinked-hosted
tester endpoint), this proves the primitives are load-bearing rather than a
no-op: precision/recall on the labelled fixture stay at least as good as the
deterministic-only baseline, and at least one refused action's reasons show a
SIE-sourced signal (rerank or extract), not just the static rule-based checks.
"""

from __future__ import annotations

import os
import sys

import pytest

LAB_DIR = os.path.join(os.path.dirname(__file__), "..", "lab", "actions")
sys.path.insert(0, os.path.abspath(LAB_DIR))

import generate_actions  # noqa: E402

from dusk.actions.event import AgentAction  # noqa: E402
from dusk.actions.normaliser import normalise_record  # noqa: E402
from dusk.actions.verdict import ActionGate  # noqa: E402
from dusk.config import Config  # noqa: E402
from dusk.trace.vector import sie_encode  # noqa: E402


def _normal() -> list[AgentAction]:
    return [normalise_record("generic", r) for r in generate_actions.normal_actions()]


def _attacks() -> list[AgentAction]:
    return [normalise_record("generic", r) for r in generate_actions.out_of_pattern_actions()]


@pytest.fixture(autouse=True)
def _skip_unless_sie_reachable() -> None:
    if sie_encode("connectivity check") is None:
        pytest.skip(
            "SIE not installed/reachable; set DUSK_SIE_ENDPOINT and, when required, "
            "SIE_API_KEY to run this"
        )


def test_live_sie_precision_recall_matches_or_beats_deterministic_baseline() -> None:
    config = Config()
    gate = ActionGate(config=config)
    gate.learn(_normal())

    labelled = [(a, False) for a in _normal()] + [(a, True) for a in _attacks()]
    tp = fp = fn = tn = 0
    for action, is_attack in labelled:
        refused = gate.evaluate(action).refused
        if is_attack and refused:
            tp += 1
        elif is_attack and not refused:
            fn += 1
        elif not is_attack and refused:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0

    assert (precision, recall) == (1.0, 1.0), (
        f"precision={precision:.2f} recall={recall:.2f} (tp={tp} fp={fp} fn={fn} tn={tn}) "
        "with live SIE -- must not regress versus the deterministic-only baseline"
    )


def test_live_sie_primitives_actually_fire_for_at_least_one_attack() -> None:
    gate = ActionGate(config=Config())
    gate.learn(_normal())

    fired_markers = ("SIE rerank", "SIE extract")
    saw_sie_signal = any(
        any(marker in reason for reason in gate.evaluate(attack).analysis.reasons)
        for marker in fired_markers
        for attack in _attacks()
    )
    assert saw_sie_signal, (
        "expected at least one attack's reasons to carry a SIE rerank/extract marker; "
        "if none do, SIE may be configured but not actually contributing a signal"
    )
