"""Live SIE benchmark: skipped until DUSK_SIE_ENDPOINT reaches an SIE cluster.

Once SIE is actually reachable (self-hosted container or the Superlinked-hosted
tester endpoint), this proves the primitives are load-bearing rather than a
no-op. Two things the deterministic-only gate cannot claim on its own:

- Precision holds on held-out negatives (legitimate actions the baseline
  never trained on), not just on the training set replayed as its own
  negatives, which any gate would trivially get right.
- At least one attack (``generate_actions.sie_only_attacks``) is
  constructed to evade every deterministic check, and is scored twice --
  once with real SIE calls, once with them forced off -- to show its score
  genuinely lands on opposite sides of the block threshold, not just that a
  reason string happens to mention SIE.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

LAB_DIR = os.path.join(os.path.dirname(__file__), "..", "lab", "actions")
sys.path.insert(0, os.path.abspath(LAB_DIR))

import generate_actions  # noqa: E402

from dusk.actions.analyse import analyse  # noqa: E402
from dusk.actions.baseline import Baseline  # noqa: E402
from dusk.actions.event import AgentAction  # noqa: E402
from dusk.actions.normaliser import normalise_record  # noqa: E402
from dusk.actions.verdict import ActionGate  # noqa: E402
from dusk.config import Config  # noqa: E402
from dusk.trace.vector import sie_encode  # noqa: E402


def _normal() -> list[AgentAction]:
    return [normalise_record("generic", r) for r in generate_actions.normal_actions()]


def _held_out_normal() -> list[AgentAction]:
    return [normalise_record("generic", r) for r in generate_actions.held_out_normal_actions()]


def _attacks() -> list[AgentAction]:
    return [normalise_record("generic", r) for r in generate_actions.out_of_pattern_actions()]


def _sie_only_attacks() -> list[AgentAction]:
    return [normalise_record("generic", r) for r in generate_actions.sie_only_attacks()]


@pytest.fixture(autouse=True)
def _skip_unless_sie_reachable() -> None:
    if sie_encode("connectivity check") is None:
        pytest.skip(
            "SIE not installed/reachable; set DUSK_SIE_ENDPOINT and, when required, "
            "SIE_API_KEY to run this"
        )


def test_live_sie_precision_recall_matches_or_beats_deterministic_baseline() -> None:
    """Precision/recall with live SIE, on a fixture the gate could not memorise its way through.

    Negatives include held-out actions never folded into the baseline (not
    just the training set replayed as its own test), and attacks include
    one constructed to evade every deterministic check (see
    generate_actions.sie_only_attacks) -- so a perfect score here reflects
    SIE actually contributing, not the fixture being trivial by
    construction.
    """
    config = Config()
    gate = ActionGate(config=config)
    gate.learn(_normal())

    labelled = (
        [(a, False) for a in _normal()]
        + [(a, False) for a in _held_out_normal()]
        + [(a, True) for a in _attacks()]
        + [(a, True) for a in _sie_only_attacks()]
    )
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
        for attack in _attacks() + _sie_only_attacks()
    )
    assert saw_sie_signal, (
        "expected at least one attack's reasons to carry a SIE rerank/extract marker; "
        "if none do, SIE may be configured but not actually contributing a signal"
    )


def test_disabling_sie_lets_the_evasive_attack_slip_through() -> None:
    """The property the tests above depend on: SIE is genuinely load-bearing here.

    generate_actions.sie_only_attacks() is built so its deterministic-only
    score lands under gate_block_threshold -- it reuses a known action type
    and target class, and its new tokens/values are deliberately outside
    the static sensitive frozenset. The only thing that can still refuse it
    is SIE's extract signal recognising "superuser" as a role/privilege
    term. Scoring the same action with real SIE calls versus with them
    forced off proves that directly, instead of asserting a reason string
    happens to be present.
    """
    baseline = Baseline.learn(_normal())
    attack = _sie_only_attacks()[0]
    threshold = Config().gate_block_threshold

    with_sie = analyse(baseline, attack)
    with (
        patch("dusk.actions.analyse.sie_extract", return_value=[]),
        patch("dusk.actions.analyse.sie_score", return_value=None),
    ):
        without_sie = analyse(baseline, attack)

    assert without_sie.score < threshold, (
        f"expected this attack to evade the deterministic-only score (got "
        f"{without_sie.score:.2f} >= {threshold}) -- if it doesn't, it isn't "
        f"actually testing what SIE adds; see generate_actions.sie_only_attacks"
    )
    assert with_sie.score >= threshold, (
        f"expected live SIE to push this attack's score over the block "
        f"threshold (got {with_sie.score:.2f} < {threshold}); if it doesn't, "
        f"SIE's extract signal isn't currently catching 'superuser' as a "
        f"privileged term -- the load-bearing claim in docs/sie-primitives.md "
        f"and the README does not hold for this run"
    )
