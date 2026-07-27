"""Deterministic per-agent behavioral baselines and feature extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from dusk.actions.event import AgentAction

#: Split a target identifier into lowercase word tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Defensive width bounds for request-driven feature extraction. Flask's
#: MAX_CONTENT_LENGTH bounds the request body in bytes, but a pathologically
#: wide or repetitive payload can still sit well under that cap while
#: materializing far more tokens/nodes than any real target or change delta
#: would -- these bound width, on top of the existing recursion-depth guard.
_MAX_TOKENS = 200
_MAX_FLATTEN_NODES = 500


def target_class(target: str) -> str:
    """Return the coarse class of a target, its first token.

    For example ``"fw-corp-https"`` and ``"fw-guest-to-restricted"`` share the
    class ``"fw"``, while ``"seg-corporate"`` is ``"seg"``. The class captures
    the kind of resource an agent touches without overfitting to exact names.
    """
    tokens = _TOKEN_RE.findall(target.lower())
    return tokens[0] if tokens else ""


def target_tokens(target: str) -> set[str]:
    """Return the set of lowercase word tokens in a target identifier, capped at _MAX_TOKENS."""
    return set(_TOKEN_RE.findall(target.lower())[:_MAX_TOKENS])


def _flatten_scalars(
    payload: Any,  # noqa: ANN401
    values: set[str],
    budget: list[int],
    *,
    _depth: int = 0,
) -> None:
    """Collect nested scalar values, bounded by recursion depth and node count.

    ``budget`` is a shared single-element counter (a list so recursive calls
    mutate the same cell) decremented once per node visited; traversal stops
    once it hits zero, independent of how deep or wide the payload is.
    """
    if _depth > 10 or budget[0] <= 0:
        return
    budget[0] -= 1
    if isinstance(payload, dict):
        for value in payload.values():
            _flatten_scalars(value, values, budget, _depth=_depth + 1)
    elif isinstance(payload, list):
        for value in payload:
            _flatten_scalars(value, values, budget, _depth=_depth + 1)
    elif isinstance(payload, (str, int, float, bool)):
        values.add(str(payload).lower())


def _change_values(change: dict[str, Any]) -> set[str]:
    """Flatten a change delta into a set of stringified scalar values, at any nesting depth."""
    values: set[str] = set()
    budget = [_MAX_FLATTEN_NODES]
    for side in ("before", "after"):
        _flatten_scalars(change.get(side), values, budget)
    return values


@dataclass
class AgentProfile:
    """The learned normal behaviour of a single agent."""

    agent_id: str
    action_types: set[str] = field(default_factory=set)
    target_classes: set[str] = field(default_factory=set)
    tokens: set[str] = field(default_factory=set)
    change_values: set[str] = field(default_factory=set)
    count: int = 0

    def observe(self, action: AgentAction) -> None:
        """Fold one known-good action into the profile."""
        self.action_types.add(action.action_type)
        self.target_classes.add(target_class(action.target))
        self.tokens |= target_tokens(action.target)
        self.change_values |= _change_values(action.change)
        self.count += 1


def action_features(action: AgentAction) -> dict[str, Any]:
    """Extract the comparable features of an action.

    Returns the action's action type, target class, target tokens, and change
    values. This is the single seam a vector backend would replace.
    """
    return {
        "action_type": action.action_type,
        "target_class": target_class(action.target),
        "tokens": target_tokens(action.target),
        "change_values": _change_values(action.change),
    }


class Baseline:
    """A collection of per-agent profiles learned from known-good actions."""

    def __init__(self) -> None:
        self._profiles: dict[str, AgentProfile] = {}

    @classmethod
    def learn(cls, actions: list[AgentAction]) -> Baseline:
        """Build a baseline from a history of known-good actions."""
        baseline = cls()
        for action in actions:
            baseline.observe(action)
        return baseline

    def observe(self, action: AgentAction) -> None:
        """Fold one known-good action into its agent's profile."""
        profile = self._profiles.get(action.agent_id)
        if profile is None:
            profile = AgentProfile(agent_id=action.agent_id)
            self._profiles[action.agent_id] = profile
        profile.observe(action)

    def profile_for(self, agent_id: str) -> AgentProfile | None:
        """Return the profile for an agent, or ``None`` if unseen."""
        return self._profiles.get(agent_id)

    @property
    def agents(self) -> list[str]:
        """The sorted list of agents the baseline has profiles for."""
        return sorted(self._profiles)

    def __len__(self) -> int:
        return len(self._profiles)
