"""Deterministic per-agent behavioral baselines and feature extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from dusk.actions.event import AgentAction

#: Split a target identifier into lowercase word tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def target_class(target: str) -> str:
    """Return the coarse class of a target, its first token.

    For example ``"fw-corp-https"`` and ``"fw-guest-to-restricted"`` share the
    class ``"fw"``, while ``"seg-corporate"`` is ``"seg"``. The class captures
    the kind of resource an agent touches without overfitting to exact names.
    """
    tokens = _TOKEN_RE.findall(target.lower())
    return tokens[0] if tokens else ""


def target_tokens(target: str) -> set[str]:
    """Return the set of lowercase word tokens in a target identifier."""
    return set(_TOKEN_RE.findall(target.lower()))


def _flatten_scalars(payload: Any, values: set[str], *, _depth: int = 0) -> None:  # noqa: ANN401
    """Collect nested scalar values with a defensive recursion limit."""
    if _depth > 10:
        return
    if isinstance(payload, dict):
        for value in payload.values():
            _flatten_scalars(value, values, _depth=_depth + 1)
    elif isinstance(payload, list):
        for value in payload:
            _flatten_scalars(value, values, _depth=_depth + 1)
    elif isinstance(payload, (str, int, float, bool)):
        values.add(str(payload).lower())


def _change_values(change: dict[str, Any]) -> set[str]:
    """Flatten a change delta into a set of stringified scalar values, at any nesting depth."""
    values: set[str] = set()
    for side in ("before", "after"):
        _flatten_scalars(change.get(side), values)
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
