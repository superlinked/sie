"""Normalize proposed Bedrock Converse tool calls before execution.

The harness supplies agent identity and timestamp because Bedrock's
``toolUse`` block contains only the tool name, arguments, and call ID.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dusk.actions.adapters.base import AdapterError, SourceAdapter
from dusk.actions.event import AgentAction

#: Substrings of the tool name mapped to a canonical action_type.
_ACTION_TYPE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("firewall", "securitygroup"), "firewall_rule_change"),
    (("route",), "route_change"),
    (("segment", "subnet", "vpc"), "segment_change"),
    (("role", "permission"), "role_assignment"),
    (("port",), "port_change"),
)


def _action_type(tool_name: str | None) -> str:
    """Classify the canonical action_type from the tool name."""
    if not tool_name:
        return "unknown"
    lowered = tool_name.lower()
    for needles, action_type in _ACTION_TYPE_RULES:
        if any(needle in lowered for needle in needles):
            return action_type
    return "unknown"


class BedrockAdapter(SourceAdapter):
    """Adapter for a proposed Bedrock Converse API tool-call."""

    source = "bedrock"

    def parse(self, raw: dict[str, Any]) -> AgentAction:
        """Map a raw record already carrying agent_id/timestamp into an AgentAction.

        Satisfies the :class:`SourceAdapter` contract for registry lookup
        and the generic ingest path. ``raw`` must additionally carry
        ``agent_id`` and ``timestamp`` (the harness's job to attach --
        see :func:`parse_tool_use` for the primary entry point used by
        agent-demo).

        Args:
            raw: A dict with ``tool_use`` (Bedrock's toolUse block) plus
                ``agent_id`` and ``timestamp``.

        Returns:
            The canonical :class:`AgentAction`.

        Raises:
            AdapterError: If the tool-call, identity, or timestamp is
                missing or malformed.
        """
        tool_use = raw.get("tool_use")
        if not isinstance(tool_use, dict):
            raise AdapterError("Bedrock record missing 'tool_use' block")
        try:
            timestamp = raw["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            return self.parse_tool_use(tool_use, agent_id=raw["agent_id"], timestamp=timestamp)
        except KeyError as exc:
            raise AdapterError(f"Bedrock record missing required field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"Malformed Bedrock record: {exc}") from exc

    def parse_tool_use(
        self, tool_use: dict[str, Any], *, agent_id: str, timestamp: datetime
    ) -> AgentAction:
        """Map one Bedrock toolUse block into an :class:`AgentAction`.

        Args:
            tool_use: The ``toolUse`` content block from a Converse API
                response (``name``, ``input``, ``toolUseId``).
            agent_id: Identity of the agent that made the call. Supplied
                by the harness, not present in the toolUse block itself.
            timestamp: When the call was made. Must be timezone-aware.

        Returns:
            The canonical :class:`AgentAction`.

        Raises:
            AdapterError: If the tool-call lacks the name or target input
                needed to form a valid AgentAction.
        """
        tool_name = tool_use.get("name")
        action_type = _action_type(tool_name if isinstance(tool_name, str) else None)

        tool_input = tool_use.get("input")
        if not isinstance(tool_input, dict):
            raise AdapterError("Bedrock toolUse block missing 'input'")

        target = tool_input.get("target")
        if not target:
            raise AdapterError("Bedrock toolUse input missing 'target'")

        change = {
            "before": tool_input.get("before"),
            "after": tool_input.get("after"),
        }

        try:
            return AgentAction(
                agent_id=agent_id,
                timestamp=timestamp,
                action_type=action_type,
                target=target,
                change=change,
                source=self.source,
                raw_ref=tool_use.get("toolUseId"),
            )
        except ValueError as exc:
            raise AdapterError(f"Malformed Bedrock tool-call: {exc}") from exc
