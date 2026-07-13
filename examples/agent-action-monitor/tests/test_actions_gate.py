"""Tests for the agent action gate: baseline, analyse, verdict, and benchmark."""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime

LAB_DIR = os.path.join(os.path.dirname(__file__), "..", "lab", "actions")
sys.path.insert(0, os.path.abspath(LAB_DIR))

import generate_actions  # noqa: E402

from dusk.actions import baseline as baseline_module  # noqa: E402
from dusk.actions.analyse import analyse  # noqa: E402
from dusk.actions.baseline import Baseline, target_class  # noqa: E402
from dusk.actions.event import AgentAction  # noqa: E402
from dusk.actions.normaliser import normalise_record  # noqa: E402
from dusk.actions.offense_memory import OffenseMemory, OffenseRecord  # noqa: E402
from dusk.actions.verdict import ALLOW, BLOCK, WOULD_BLOCK, ActionGate  # noqa: E402
from dusk.config import Config  # noqa: E402
from dusk.trace.vector import ExtractedTerm  # noqa: E402

CONFIG = Config()
_TS = datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)


def _action(agent_id: str, action_type: str, target: str, **change: object) -> AgentAction:
    return AgentAction(
        agent_id=agent_id,
        timestamp=_TS,
        action_type=action_type,
        target=target,
        change={"before": None, "after": dict(change) if change else None},
        source="generic",
    )


def _normal() -> list[AgentAction]:
    return [normalise_record("generic", r) for r in generate_actions.normal_actions()]


def _attacks() -> list[AgentAction]:
    return [normalise_record("generic", r) for r in generate_actions.out_of_pattern_actions()]


# --- baseline ----------------------------------------------------------------


def test_baseline_learns_per_agent() -> None:
    """Learning records one profile per distinct agent."""
    baseline = Baseline.learn(_normal())
    assert set(baseline.agents) == {"netops-agent", "segment-agent", "iam-agent"}
    netops = baseline.profile_for("netops-agent")
    assert netops is not None
    assert "firewall_rule_change" in netops.action_types
    assert "fw" in netops.target_classes


def test_target_class_groups_by_first_token() -> None:
    """Targets that share a prefix share a class."""
    assert target_class("fw-corp-https") == "fw"
    assert target_class("fw-guest-to-restricted") == "fw"
    assert target_class("seg-corporate") == "seg"


def test_change_values_flattens_nested_dicts() -> None:
    """A value buried in a nested dict must not be invisible to scoring."""
    change = {"before": None, "after": {"rules": {"cidr": "0.0.0.0/0", "port": 22}}}
    values = baseline_module._change_values(change)
    assert "0.0.0.0/0" in values
    assert "22" in values


def test_change_values_flattens_nested_lists() -> None:
    """A value buried inside a list of dicts must not be invisible to scoring."""
    change = {
        "before": None,
        "after": {"rules": [{"cidr": "10.0.0.0/8"}, {"cidr": "0.0.0.0/0", "port": 22}]},
    }
    values = baseline_module._change_values(change)
    assert "0.0.0.0/0" in values
    assert "10.0.0.0/8" in values
    assert "22" in values


def test_change_values_still_flattens_top_level() -> None:
    """The original flat-dict behaviour is unchanged."""
    change = {"before": None, "after": {"port": 443}}
    assert baseline_module._change_values(change) == {"443"}


def test_change_values_depth_is_bounded() -> None:
    """An adversarially deep payload does not make flattening unbounded."""
    nested: dict[str, object] = {"leaf": "0.0.0.0/0"}
    for _ in range(20):
        nested = {"wrapper": nested}
    change = {"before": None, "after": nested}
    # Should not raise (e.g. RecursionError) and should not necessarily find
    # the deeply buried leaf, since depth is capped defensively.
    baseline_module._change_values(change)


# --- analyse -----------------------------------------------------------------


def test_known_good_action_scores_zero() -> None:
    """An action matching the agent's baseline is not anomalous."""
    baseline = Baseline.learn(_normal())
    result = analyse(baseline, _action("netops-agent", "firewall_rule_change", "fw-corp-https"))
    assert result.score == 0.0
    assert result.reasons == ["action matches the agent's established pattern"]


def test_new_action_type_is_flagged() -> None:
    """An agent doing a verb it never does scores high and maps to ATT&CK."""
    baseline = Baseline.learn(_normal())
    result = analyse(baseline, _action("segment-agent", "firewall_rule_change", "fw-x"))
    assert result.score >= CONFIG.gate_block_threshold
    assert "T1562" in result.mitre_attack
    assert result.mitre_atlas.startswith("AML.")


def test_privilege_escalation_is_flagged() -> None:
    """Granting a sensitive role is caught even when the verb is familiar."""
    baseline = Baseline.learn(_normal())
    result = analyse(baseline, _action("iam-agent", "role_assignment", "ra-self", role="owner"))
    assert result.score >= CONFIG.gate_block_threshold
    assert result.blast_radius == "high"
    assert any("sensitive" in r for r in result.reasons)


def test_nested_privilege_escalation_is_flagged() -> None:
    """A sensitive value buried in a nested change payload is not invisible.

    Same scenario as test_privilege_escalation_is_flagged, but the sensitive
    value sits inside a nested structure -- realistic for a control-plane
    payload shaped like {"after": {"rules": [{"role": "owner"}]}}. Before the
    nested-flatten fix, this would score 0.0 and pass through silently.
    """
    baseline = Baseline.learn(_normal())
    action = AgentAction(
        agent_id="iam-agent",
        timestamp=_TS,
        action_type="role_assignment",
        target="ra-self",
        change={"before": None, "after": {"grants": [{"role": "owner", "scope": "global"}]}},
        source="generic",
    )
    result = analyse(baseline, action)
    assert result.score >= CONFIG.gate_block_threshold
    assert result.blast_radius == "high"
    assert any("sensitive" in r for r in result.reasons)


def test_unknown_agent_is_noted() -> None:
    """An agent with no baseline is called out."""
    baseline = Baseline.learn(_normal())
    result = analyse(baseline, _action("ghost-agent", "route_change", "rt-x"))
    assert result.score > 0.0
    assert any("no established baseline" in r for r in result.reasons)


def test_agent_history_without_sie_does_not_change_score() -> None:
    """Passing history with SIE unavailable is a no-op (the default, no-SIE case)."""
    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-corp-https", port=443)
    with_history = analyse(baseline, action, agent_history=_normal())
    without_history = analyse(baseline, action)
    assert with_history.score == without_history.score
    assert with_history.reasons == without_history.reasons


def test_agent_history_low_rerank_similarity_adds_reason() -> None:
    """A low SIE rerank score adds a reason and raises the score, on top of rule-based checks."""
    from unittest.mock import patch

    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-corp-https", port=443)
    baseline_only = analyse(baseline, action)

    with patch("dusk.actions.analyse.sie_score", return_value=[0.05, 0.05]):
        reranked = analyse(baseline, action, agent_history=_normal()[:2])

    assert reranked.score > baseline_only.score
    assert any("SIE rerank" in r for r in reranked.reasons)


def test_agent_history_high_rerank_similarity_is_unchanged() -> None:
    """A confident rerank match does not add the low-similarity reason."""
    from unittest.mock import patch

    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-corp-https", port=443)
    baseline_only = analyse(baseline, action)

    with patch("dusk.actions.analyse.sie_score", return_value=[0.9, 0.9]):
        reranked = analyse(baseline, action, agent_history=_normal()[:2])

    assert reranked.score == baseline_only.score
    assert not any("SIE rerank" in r for r in reranked.reasons)


def test_sie_extract_flags_terms_missed_by_the_static_frozenset() -> None:
    """A GLiNER-extracted term outside the hardcoded sensitive set adds a reason."""
    from unittest.mock import patch

    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-corp-https", port=443)
    baseline_only = analyse(baseline, action)

    term = ExtractedTerm(text="superuser", label="role", score=0.95)
    with patch("dusk.actions.analyse.sie_extract", return_value=[term]):
        extracted = analyse(baseline, action)

    assert extracted.score > baseline_only.score
    assert any("SIE extract" in r and "superuser" in r for r in extracted.reasons)


def test_sie_extract_terms_already_in_static_set_are_not_duplicated() -> None:
    """A term the static frozenset already catches doesn't add a second reason."""
    from unittest.mock import patch

    baseline = Baseline.learn(_normal())
    action = _action("iam-agent", "role_assignment", "ra-self", role="owner")
    baseline_only = analyse(baseline, action)

    term = ExtractedTerm(text="owner", label="role", score=0.95)
    with patch("dusk.actions.analyse.sie_extract", return_value=[term]):
        extracted = analyse(baseline, action)

    assert extracted.score == baseline_only.score
    assert not any("SIE extract" in r for r in extracted.reasons)


def test_sie_extract_low_confidence_term_is_ignored() -> None:
    """A GLiNER hit below the confidence floor is dropped, not counted as evidence."""
    from unittest.mock import patch

    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-corp-https", port=443)
    baseline_only = analyse(baseline, action)

    term = ExtractedTerm(text="superuser", label="role", score=0.2)
    with patch("dusk.actions.analyse.sie_extract", return_value=[term]):
        extracted = analyse(baseline, action)

    assert extracted.score == baseline_only.score


def test_sie_extract_high_confidence_non_privileged_label_is_ignored() -> None:
    """A confident but ordinary resource/port extraction is not privilege escalation."""
    from unittest.mock import patch

    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-corp-https", port=443)
    baseline_only = analyse(baseline, action)

    terms = [
        ExtractedTerm(text="eth0", label="resource", score=0.95),
        ExtractedTerm(text="8080", label="port", score=0.95),
    ]
    with patch("dusk.actions.analyse.sie_extract", return_value=terms):
        extracted = analyse(baseline, action)

    assert extracted.score == baseline_only.score
    assert not any("SIE extract" in r for r in extracted.reasons)
    assert not any("SIE extract" in r for r in extracted.reasons)


def test_sie_extract_unavailable_does_not_change_score() -> None:
    """The default (no-SIE) case: sie_extract returns [] and nothing changes."""
    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-corp-https", port=443)
    result = analyse(baseline, action)
    assert not any("SIE extract" in r for r in result.reasons)


# --- repeat-offense signal -----------------------------------------------------


def _offense(**overrides: object) -> OffenseRecord:
    defaults: dict[str, object] = {
        "trace_id": "trace-offense-1",
        "agent_id": "netops-agent",
        "action_type": "firewall_rule_change",
        "target_class": "fw",
        "tokens": ("fw", "restricted"),
        "verdict": "BLOCK",
        "timestamp": datetime.now(UTC),
    }
    defaults.update(overrides)
    return OffenseRecord(**defaults)  # type: ignore[arg-type]


def test_no_offenses_does_not_change_score() -> None:
    """A clean-history agent is completely unaffected by the repeat-offense signal."""
    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-restricted")
    without = analyse(baseline, action)
    with_empty = analyse(baseline, action, offenses=[])
    assert without.score == with_empty.score
    assert without.reasons == with_empty.reasons


def test_matching_offense_raises_score_and_cites_trace_id() -> None:
    """A same-type, same-target-class repeat past offense adds score and names the prior trace."""
    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-restricted")
    baseline_only = analyse(baseline, action)
    offense = _offense(trace_id="trace-xyz")

    with_offense = analyse(baseline, action, offenses=[offense], config=CONFIG)

    assert with_offense.score > baseline_only.score
    assert any("trace-xyz" in r for r in with_offense.reasons)


def test_different_action_type_offense_does_not_match() -> None:
    """An offense for a different action type must not contribute -- type match is required."""
    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "role_assignment", "ra-self", role="owner")
    baseline_only = analyse(baseline, action)
    offense = _offense(action_type="firewall_rule_change")

    with_offense = analyse(baseline, action, offenses=[offense], config=CONFIG)

    assert with_offense.score == baseline_only.score


def test_repeat_offense_contribution_is_capped() -> None:
    """The signal alone cannot exceed repeat_offense_max_contribution, however strong the match."""
    config = Config(repeat_offense_max_contribution=0.05)
    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-restricted")
    baseline_only = analyse(baseline, action)
    offense = _offense()

    with_offense = analyse(baseline, action, offenses=[offense], config=config)

    assert with_offense.score - baseline_only.score <= 0.05 + 1e-9


def test_old_offense_contributes_less_than_a_recent_one() -> None:
    """Decay: an offense from long ago must weigh less than one from moments ago."""
    from datetime import timedelta

    config = Config(repeat_offense_half_life_days=10.0)
    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-restricted")

    recent = analyse(
        baseline, action, offenses=[_offense(timestamp=datetime.now(UTC))], config=config
    )
    old = analyse(
        baseline,
        action,
        offenses=[_offense(timestamp=datetime.now(UTC) - timedelta(days=100))],
        config=config,
    )

    assert recent.score > old.score


def test_multiple_offenses_use_the_single_best_match_not_the_sum() -> None:
    """Anti-gaming: flooding with many weak matches must not out-score one strong match."""
    config = Config(repeat_offense_max_contribution=1.0)
    baseline = Baseline.learn(_normal())
    # A known action/target for this agent, so the deterministic checks below
    # contribute 0 and only the repeat-offense signal moves the score --
    # otherwise both cases would saturate at the 1.0 clamp and be indistinguishable.
    action = _action("netops-agent", "firewall_rule_change", "fw-corp-https", port=443)

    single_strong = analyse(
        baseline,
        action,
        offenses=[_offense(target_class="fw", tokens=("fw", "corp", "https"))],
        config=config,
    )
    many_weak = analyse(
        baseline,
        action,
        offenses=[_offense(target_class="seg", tokens=("seg",)) for _ in range(20)],
        config=config,
    )

    # The weak matches don't even share a target class or token, so they
    # contribute nothing at all -- confirming there is no additive stacking.
    assert many_weak.score < single_strong.score


def test_offense_reason_names_the_verdict_and_date() -> None:
    baseline = Baseline.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-restricted")
    offense = _offense(verdict="WOULD-BLOCK")

    result = analyse(baseline, action, offenses=[offense], config=CONFIG)

    assert any("WOULD-BLOCK" in r for r in result.reasons)


# --- verdict -----------------------------------------------------------------


def test_gate_allows_routine_refuses_attacks() -> None:
    """Watch mode allows routine actions and WOULD-BLOCKs the attacks."""
    gate = ActionGate(config=CONFIG)
    gate.learn(_normal())
    for action in _normal():
        assert gate.evaluate(action).verdict == ALLOW
    for attack in _attacks():
        v = gate.evaluate(attack)
        assert v.verdict == WOULD_BLOCK
        assert v.refused is True


def test_gate_learn_tracks_raw_history_per_agent() -> None:
    """learn() keeps the raw actions ActionGate.evaluate() feeds into analyse()."""
    gate = ActionGate(config=CONFIG)
    gate.learn(_normal())
    netops_history = gate._history.get("netops-agent", [])
    assert netops_history
    assert all(a.agent_id == "netops-agent" for a in netops_history)


def test_gate_evaluate_passes_history_to_sie_rerank() -> None:
    """A gate evaluation surfaces the SIE rerank reason when the mocked score is low."""
    from unittest.mock import patch

    gate = ActionGate(config=CONFIG)
    gate.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-corp-https", port=443)

    with patch("dusk.actions.analyse.sie_score", return_value=[0.05] * 20):
        verdict = gate.evaluate(action)

    assert any("SIE rerank" in r for r in verdict.analysis.reasons)


def test_enforce_mode_blocks() -> None:
    """Enforce mode renders BLOCK instead of WOULD-BLOCK."""
    gate = ActionGate(config=CONFIG, enforce=True)
    gate.learn(_normal())
    assert gate.evaluate(_attacks()[0]).verdict == BLOCK


def test_empty_baseline_evaluation_does_not_crash() -> None:
    """A gate with no baseline still renders a verdict for a single action."""
    gate = ActionGate(config=CONFIG)
    verdict = gate.evaluate(_action("a", "route_change", "rt-1"))
    assert verdict.verdict in (ALLOW, WOULD_BLOCK)


# --- offense memory wiring -----------------------------------------------------


def test_gate_without_offense_memory_behaves_as_before() -> None:
    """No offense_memory passed -- the repeat-offense signal is a complete no-op."""
    gate = ActionGate(config=CONFIG)
    gate.learn(_normal())
    attack = _attacks()[0]
    first = gate.evaluate(attack)
    second = gate.evaluate(attack)
    assert first.analysis.score == second.analysis.score


def test_refused_verdict_is_recorded_in_offense_memory() -> None:
    memory = OffenseMemory(storage_path=None)
    gate = ActionGate(config=CONFIG, offense_memory=memory)
    gate.learn(_normal())
    attack = _attacks()[0]

    verdict = gate.evaluate(attack)

    offenses = memory.offenses_for(attack.agent_id)
    assert len(offenses) == 1
    assert offenses[0].trace_id == verdict.trace_id
    assert offenses[0].verdict == verdict.verdict


def test_allowed_verdict_is_not_recorded_in_offense_memory() -> None:
    memory = OffenseMemory(storage_path=None)
    gate = ActionGate(config=CONFIG, offense_memory=memory)
    gate.learn(_normal())

    for action in _normal():
        verdict = gate.evaluate(action)
        assert verdict.verdict == ALLOW

    assert all(memory.offenses_for(a.agent_id) == [] for a in _normal())


def test_repeated_attack_scores_higher_the_second_time() -> None:
    """The end-to-end point of this feature: a repeat offender is judged more harshly."""
    memory = OffenseMemory(storage_path=None)
    gate = ActionGate(config=CONFIG, offense_memory=memory)
    gate.learn(_normal())
    attack = _attacks()[0]

    first = gate.evaluate(attack)
    second = gate.evaluate(attack)

    assert second.analysis.score >= first.analysis.score
    assert any(first.trace_id in r for r in second.analysis.reasons)


def test_gate_verdict_trace_id_is_unique_per_evaluation() -> None:
    gate = ActionGate(config=CONFIG)
    gate.learn(_normal())
    action = _action("netops-agent", "firewall_rule_change", "fw-corp-https", port=443)
    first = gate.evaluate(action)
    second = gate.evaluate(action)
    assert first.trace_id != second.trace_id


# --- labelled benchmark ------------------------------------------------------


def test_benchmark_precision_recall() -> None:
    """On the labelled fixture the gate catches every attack with no false alarms."""
    gate = ActionGate(config=CONFIG)
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
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0

    # Surface the numbers in the assertion message for the demo record.
    assert (precision, recall, fp_rate) == (1.0, 1.0, 0.0), (
        f"precision={precision:.2f} recall={recall:.2f} fp_rate={fp_rate:.2f} "
        f"(tp={tp} fp={fp} fn={fn} tn={tn})"
    )
