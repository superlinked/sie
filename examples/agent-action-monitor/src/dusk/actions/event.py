"""The AgentAction event: one normalised control-plane action.

An :class:`AgentAction` is controller-agnostic by design. Whatever system an
agent used to change the network (a cloud API, an SDN controller, a policy
endpoint), the event records the same canonical facts: who acted, when, what
kind of change, on what target, and the before/after delta. Source-specific
shapes are mapped onto this schema by adapters.

This object describes what happened, not how bad it is. It carries no
severity, score, blast radius, or verdict; those belong to later layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

#: The normalised set of action verbs. Anything an adapter cannot classify
#: maps to ``"unknown"`` rather than being invented.
KNOWN_ACTION_TYPES: frozenset[str] = frozenset(
    {
        "firewall_rule_change",
        "route_change",
        "segment_change",
        "role_assignment",
        "port_change",
        "unknown",
    }
)


@dataclass(frozen=True)
class AgentAction:
    """One normalised action an agent performed on the network control plane.

    Attributes:
        agent_id: Identity of the acting agent (service account, role, or
            agent name as the controller reports it). Must be non-empty.
        timestamp: When the action occurred. Must be timezone-aware.
        action_type: Normalised verb, one of :data:`KNOWN_ACTION_TYPES`.
        target: What was acted on (resource id, rule name, segment). Must be
            non-empty.
        change: Structured delta with keys ``"before"`` and ``"after"``;
            either may be ``None`` for a create or delete.
        source: Originating controller, for example ``"azure"`` or
            ``"generic"``.
        raw_ref: Opaque reference back to the original record (an id), never
            the full payload. May be ``None``.
    """

    agent_id: str
    timestamp: datetime
    action_type: str
    target: str
    change: dict[str, Any]
    source: str
    raw_ref: str | None = None

    def __post_init__(self) -> None:
        """Validate the event, raising ValueError on invalid input.

        Raises:
            ValueError: If ``agent_id`` or ``target`` is empty, ``timestamp``
                is not timezone-aware, ``action_type`` is not a known verb, or
                ``change`` is not a mapping. Values are never silently coerced.
        """
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("target must be a non-empty string")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self.action_type not in KNOWN_ACTION_TYPES:
            raise ValueError(
                f"action_type '{self.action_type}' is not one of the known "
                f"types: {', '.join(sorted(KNOWN_ACTION_TYPES))}"
            )
        if not isinstance(self.change, dict):
            raise ValueError("change must be a dict with 'before' and 'after' keys")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict, with the timestamp as an ISO 8601 string."""
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
            "action_type": self.action_type,
            "target": self.target,
            "change": self.change,
            "source": self.source,
            "raw_ref": self.raw_ref,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentAction:
        """Build an :class:`AgentAction` from a :meth:`to_dict` mapping.

        Args:
            data: A mapping as produced by :meth:`to_dict`, with the timestamp
                as an ISO 8601 string.

        Returns:
            The reconstructed :class:`AgentAction`.

        Raises:
            ValueError: If a required key is missing or a value is invalid.
        """
        try:
            timestamp = datetime.fromisoformat(data["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid or missing timestamp: {exc}") from exc
        try:
            return cls(
                agent_id=data["agent_id"],
                timestamp=timestamp,
                action_type=data["action_type"],
                target=data["target"],
                change=data["change"],
                source=data["source"],
                raw_ref=data.get("raw_ref"),
            )
        except KeyError as exc:
            raise ValueError(f"Missing required field: {exc}") from exc
