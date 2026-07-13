"""Score agent behavior against its baseline and explain each signal."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from dusk.actions.baseline import Baseline, action_features
from dusk.actions.event import AgentAction
from dusk.trace.vector import sie_extract, sie_score

if TYPE_CHECKING:
    from dusk.actions.offense_memory import OffenseRecord
    from dusk.config import Config

logger = logging.getLogger("dusk.actions.analyse")

#: Weight each novelty signal contributes to the anomaly score.
_W_NEW_ACTION_TYPE = 0.55
_W_NEW_TARGET_CLASS = 0.4
_W_NEW_TOKENS = 0.2
_W_NEW_CHANGE_VALUES = 0.25
_W_UNKNOWN_AGENT = 0.5
_W_SENSITIVE = 0.35
#: Extra contribution when SIE's reranker finds no close match in the
#: agent's raw history, even though the deterministic feature checks above
#: found nothing new. Only fires when SIE is configured and reachable; the
#: rule-based score above is unchanged otherwise.
_W_LOW_SEMANTIC_SIMILARITY = 0.2
#: Rerank score below this is "no close match". sie_score() bounds the raw logit into [0, 1] via
#: sigmoid, but this floor is a heuristic, not an empirically calibrated cutoff.
_SEMANTIC_SIMILARITY_FLOOR = 0.3
#: Extra contribution when SIE's zero-shot extractor (GLiNER) surfaces a
#: privileged term the static frozenset below doesn't already cover. Slightly
#: below _W_SENSITIVE since it's a probabilistic match, not an exact one.
_W_EXTRACTED_SENSITIVE = 0.3
#: Below this confidence, a zero-shot extraction is treated as noise, not evidence.
_EXTRACT_CONFIDENCE_FLOOR = 0.5
#: Only these GLiNER labels indicate privilege escalation; "resource"/"segment"/"port" (also in
#: DEFAULT_EXTRACT_LABELS) are neutral descriptors and shouldn't add score on their own.
_PRIVILEGED_LABELS = frozenset({"role", "privilege"})

#: MITRE ATT&CK technique per normalised action type.
_ATTCK: dict[str, str] = {
    "firewall_rule_change": "T1562.004 Impair Defenses: Disable or Modify System Firewall",
    "port_change": "T1562.004 Impair Defenses: Disable or Modify System Firewall",
    "route_change": "T1599 Network Boundary Bridging",
    "segment_change": "T1599 Network Boundary Bridging",
    "role_assignment": "T1098 Account Manipulation",
    "unknown": "T1078 Valid Accounts",
}

#: MITRE ATLAS technique describing the agent-level cause.
_ATLAS = "AML.T0051 LLM Prompt Injection"

#: Sensitive change values that signal privilege escalation when newly introduced.
_SENSITIVE_VALUES = frozenset({"owner", "admin", "root", "0.0.0.0/0"})
#: Target tokens that indicate a cross-boundary or restricted reach.
_SENSITIVE_TOKENS = frozenset({"restricted", "guest", "self", "owner", "global", "all"})
#: The union, used to test whether a term is sensitive.
_SENSITIVE = _SENSITIVE_VALUES | _SENSITIVE_TOKENS


@dataclass
class AnalysisResult:
    """The outcome of analysing one action against a baseline.

    Attributes:
        agent_id: The acting agent.
        action_type: The action's normalised verb.
        target: What was acted on.
        score: Anomaly score in ``0..1``; higher is more anomalous.
        reasons: Human-readable explanations of the score.
        mitre_attack: The mapped MITRE ATT&CK technique.
        mitre_atlas: The mapped MITRE ATLAS technique.
        blast_radius: Coarse impact estimate, ``"low"``, ``"medium"`` or
            ``"high"``.
        predicted_next: What an attacker would likely do next.
    """

    agent_id: str
    action_type: str
    target: str
    score: float
    reasons: list[str] = field(default_factory=list)
    mitre_attack: str = ""
    mitre_atlas: str = ""
    blast_radius: str = "low"
    predicted_next: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the result."""
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "target": self.target,
            "score": round(self.score, 4),
            "reasons": self.reasons,
            "mitre_attack": self.mitre_attack,
            "mitre_atlas": self.mitre_atlas,
            "blast_radius": self.blast_radius,
            "predicted_next": self.predicted_next,
        }


def _blast_radius(action: AgentAction, features: dict[str, Any]) -> str:
    """Estimate how much damage the action could do."""
    sensitive = (features["tokens"] | features["change_values"]) & (
        _SENSITIVE_TOKENS | _SENSITIVE_VALUES
    )
    if sensitive:
        return "high"
    if action.action_type in ("firewall_rule_change", "role_assignment", "route_change"):
        return "medium"
    return "low"


def _predicted_next(action: AgentAction) -> str:
    """Predict the attacker's likely next move after this action."""
    mapping = {
        "firewall_rule_change": (
            "expect lateral movement into the newly reachable segment; watch for "
            "east-west connections from this agent"
        ),
        "route_change": (
            "expect traffic redirection or interception; watch for new flows toward "
            "the changed next hop"
        ),
        "segment_change": (
            "expect access into the resized or new segment; watch for first-time connections there"
        ),
        "role_assignment": (
            "expect privilege use; watch for actions that the newly granted role "
            "permits but this agent never took before"
        ),
    }
    return mapping.get(
        action.action_type,
        "watch this agent for further actions outside its established pattern",
    )


def _extracted_sensitive_terms(features: dict[str, Any]) -> set[str]:
    """Pull privileged terms from the action's target/change text via SIE extract.

    Returns an empty set whenever SIE is not configured/reachable, or when
    every extraction is low-confidence or labeled as a neutral descriptor
    rather than role/privilege (see _EXTRACT_CONFIDENCE_FLOOR, _PRIVILEGED_LABELS).
    """
    text = " ".join(sorted(features["tokens"] | features["change_values"]))
    if not text:
        return set()
    return {
        term.text.lower()
        for term in sie_extract(text)
        if term.score >= _EXTRACT_CONFIDENCE_FLOOR and term.label.lower() in _PRIVILEGED_LABELS
    }


def _semantic_novelty(
    action: AgentAction, agent_history: list[AgentAction]
) -> tuple[float, str | None]:
    """Rerank ``action`` against the agent's raw history via SIE's cross-encoder.

    Returns ``(0.0, None)`` whenever there is no history to compare against
    or SIE is not configured/reachable, so the deterministic score above is
    unchanged in the default (no-SIE) case.
    """
    if not agent_history:
        return 0.0, None

    query = f"{action.action_type} {action.target}"
    candidates = [f"{a.action_type} {a.target}" for a in agent_history]
    scores = sie_score(query, candidates)
    if not scores:
        return 0.0, None

    best = max(scores)
    if best < _SEMANTIC_SIMILARITY_FLOOR:
        return (
            _W_LOW_SEMANTIC_SIMILARITY,
            f"SIE rerank finds no close match in this agent's recorded history (best={best:.2f})",
        )
    return 0.0, None


def _extra_sie_signals(
    action: AgentAction, features: dict[str, Any], agent_history: list[AgentAction]
) -> tuple[float, list[str]]:
    """Optional SIE-backed signals: extracted privileged terms and rerank novelty.

    Both are no-ops (contribute nothing) when SIE is not configured/reachable,
    so the deterministic score above is unchanged in the default case.
    """
    extra_score = 0.0
    reasons: list[str] = []

    extracted = _extracted_sensitive_terms(features) - _SENSITIVE
    if extracted:
        extra_score += _W_EXTRACTED_SENSITIVE
        reasons.append(f"SIE extract flags additional privileged terms {sorted(extracted)}")

    novelty_score, novelty_reason = _semantic_novelty(action, agent_history)
    if novelty_reason:
        extra_score += novelty_score
        reasons.append(novelty_reason)

    return extra_score, reasons


def _offense_similarity(
    action: AgentAction, features: dict[str, Any], offense: OffenseRecord
) -> float:
    """How closely ``action`` matches a single past offense, in ``0..1``.

    Requires the same action type to count at all -- a firewall change
    doesn't make a later role assignment suspicious just because both were
    once blocked. Beyond that, target class and shared tokens each add
    partial credit, so an exact repeat of the same target scores highest
    while a same-type action against a related-but-different target still
    contributes something.
    """
    if action.action_type != offense.action_type:
        return 0.0
    similarity = 0.5
    if features["target_class"] and features["target_class"] == offense.target_class:
        similarity += 0.3
    if features["tokens"] & set(offense.tokens):
        similarity += 0.2
    return min(1.0, similarity)


def _decay(offense: OffenseRecord, half_life_days: float) -> float:
    """Exponential decay: 1.0 for a brand-new offense, 0.5 at one half-life, etc."""
    age_days = max(0.0, (datetime.now(UTC) - offense.timestamp).total_seconds() / 86400)
    return float(math.pow(0.5, age_days / half_life_days))


def _repeat_offense_signal(
    action: AgentAction,
    features: dict[str, Any],
    offenses: list[OffenseRecord] | None,
    config: Config | None,
) -> tuple[float, str | None]:
    """Score how much ``action`` resembles this agent's own past refusals.

    Takes the single best-matching prior offense rather than summing across
    all of them, so an attacker cannot inflate the contribution by
    triggering many weak matches -- only the strongest, most relevant prior
    refusal counts, and its weight decays with age. Returns ``(0.0, None)``
    when there is no meaningful match, so a clean-history agent is
    completely unaffected by this signal.
    """
    if not offenses:
        return 0.0, None

    from dusk.config import get_config

    cfg = config or get_config()
    best_offense: OffenseRecord | None = None
    best_weight = 0.0
    for offense in offenses:
        similarity = _offense_similarity(action, features, offense)
        if similarity <= 0.0:
            continue
        weight = similarity * _decay(offense, cfg.repeat_offense_half_life_days)
        if weight > best_weight:
            best_weight = weight
            best_offense = offense

    if best_offense is None or best_weight <= 0.0:
        return 0.0, None

    contribution = min(cfg.repeat_offense_max_contribution, best_weight)
    reason = (
        f"resembles a prior {best_offense.verdict} action by this agent "
        f"(trace {best_offense.trace_id}, {best_offense.timestamp.date().isoformat()})"
    )
    return contribution, reason


def analyse(
    baseline: Baseline,
    action: AgentAction,
    agent_history: list[AgentAction] | None = None,
    offenses: list[OffenseRecord] | None = None,
    config: Config | None = None,
) -> AnalysisResult:
    """Score and explain ``action`` against ``baseline``.

    Args:
        baseline: The learned per-agent baseline.
        action: The action to evaluate.
        agent_history: The agent's raw known-good actions, used for an
            optional SIE-reranked semantic novelty check on top of the
            deterministic feature checks below. Omit or pass an empty list
            to skip this signal entirely.
        offenses: The agent's past refused verdicts, used for the
            repeat-offense signal. Omit or pass an empty list to skip this
            signal entirely -- a clean-history agent is unaffected.
        config: Configuration providing ``repeat_offense_max_contribution``
            and ``repeat_offense_half_life_days``. Defaults to the
            process-wide singleton; only consulted when ``offenses`` is
            non-empty.

    Returns:
        An :class:`AnalysisResult` with score, reasons, and mappings.
    """
    features = action_features(action)
    profile = baseline.profile_for(action.agent_id)
    reasons: list[str] = []
    score = 0.0

    if profile is None or profile.count == 0:
        score += _W_UNKNOWN_AGENT
        reasons.append(
            f"agent '{action.agent_id}' has no established baseline; "
            f"its behaviour cannot be vouched for"
        )
        sensitive = (features["tokens"] | features["change_values"]) & _SENSITIVE
        if sensitive:
            score += _W_SENSITIVE
            reasons.append(f"action touches sensitive terms {sorted(sensitive)}")
    else:
        if features["action_type"] not in profile.action_types:
            score += _W_NEW_ACTION_TYPE
            reasons.append(
                f"action type '{features['action_type']}' is new for this agent, "
                f"which normally does {sorted(profile.action_types)}"
            )
        if features["target_class"] and features["target_class"] not in profile.target_classes:
            score += _W_NEW_TARGET_CLASS
            reasons.append(
                f"target class '{features['target_class']}' is new for this agent, "
                f"which normally touches {sorted(profile.target_classes)}"
            )
        new_tokens = features["tokens"] - profile.tokens
        if new_tokens:
            score += _W_NEW_TOKENS
            reasons.append(f"target introduces unseen terms {sorted(new_tokens)}")
        new_values = features["change_values"] - profile.change_values
        if new_values:
            score += _W_NEW_CHANGE_VALUES
            reasons.append(f"change introduces unseen values {sorted(new_values)}")
        sensitive_new = (new_tokens | new_values) & _SENSITIVE
        if sensitive_new:
            score += _W_SENSITIVE
            reasons.append(
                f"newly introduces sensitive or privileged terms {sorted(sensitive_new)}"
            )

    extra_score, extra_reasons = _extra_sie_signals(action, features, agent_history or [])
    score += extra_score
    reasons.extend(extra_reasons)

    repeat_score, repeat_reason = _repeat_offense_signal(action, features, offenses, config)
    if repeat_reason:
        score += repeat_score
        reasons.append(repeat_reason)

    score = min(1.0, score)
    blast = _blast_radius(action, features)
    if not reasons:
        reasons.append("action matches the agent's established pattern")

    result = AnalysisResult(
        agent_id=action.agent_id,
        action_type=action.action_type,
        target=action.target,
        score=score,
        reasons=reasons,
        mitre_attack=_ATTCK.get(action.action_type, _ATTCK["unknown"]),
        mitre_atlas=_ATLAS,
        blast_radius=blast,
        predicted_next=_predicted_next(action),
    )
    logger.debug(
        "analysed agent=%s action_type=%s score=%.2f blast=%s",
        action.agent_id,
        action.action_type,
        score,
        blast,
    )
    return result
