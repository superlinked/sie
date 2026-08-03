"""Azure activity-log adapter.

Maps an Azure Monitor activity-log record into the canonical AgentAction.

Field assumptions (documented because Azure's activity-log shape varies by
API version, and these are the fields this adapter relies on):

- ``operationName``: the operation. In the activity-log schema this can be a
  plain string or an object ``{"value": ..., "localizedValue": ...}``. This
  adapter accepts both and falls back to ``authorization.action`` (the RBAC
  action, for example ``"Microsoft.Network/networkSecurityGroups/write"``).
- ``caller``: the acting identity (a user principal name or object id). Used
  as ``agent_id``.
- ``resourceId``: the affected resource. Used as ``target``.
- ``eventTimestamp``: ISO 8601 time of the event. A trailing ``Z`` is
  accepted. Used as ``timestamp``.
- ``properties``: optional, may carry ``before``/``after`` deltas. Used to
  build ``change``.
- ``correlationId`` / ``eventDataId``: opaque reference to the record. Used as
  ``raw_ref``.

Operation-name to action_type mapping (substring, case-insensitive):

- networkSecurityGroups or securityRules -> ``firewall_rule_change``
- routeTables or routes                  -> ``route_change``
- virtualNetworks or subnets             -> ``segment_change``
- roleAssignments                        -> ``role_assignment``
- anything else                          -> ``unknown``

This adapter normalises shape only. It assigns no severity.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dusk.actions.adapters.base import AdapterError, SourceAdapter
from dusk.actions.event import AgentAction

#: Substrings of the operation name mapped to a canonical action_type.
_ACTION_TYPE_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("networksecuritygroups", "securityrules"), "firewall_rule_change"),
    (("routetables", "routes"), "route_change"),
    (("virtualnetworks", "subnets"), "segment_change"),
    (("roleassignments",), "role_assignment"),
)


def _operation_name(raw: dict[str, Any]) -> str | None:
    """Extract the operation name as a string, accepting both Azure shapes."""
    op = raw.get("operationName")
    if isinstance(op, dict):
        op = op.get("value") or op.get("localizedValue")
    if not op:
        authorization = raw.get("authorization")
        if isinstance(authorization, dict):
            op = authorization.get("action")
    return op if isinstance(op, str) and op else None


def _action_type(operation: str | None) -> str:
    """Classify the canonical action_type from the operation name."""
    if not operation:
        return "unknown"
    lowered = operation.lower()
    for needles, action_type in _ACTION_TYPE_RULES:
        if any(needle in lowered for needle in needles):
            return action_type
    return "unknown"


class AzureAdapter(SourceAdapter):
    """Adapter for Azure Monitor activity-log records."""

    source = "azure"

    def parse(self, raw: dict[str, Any]) -> AgentAction:
        """Map one Azure activity-log record into an :class:`AgentAction`.

        Args:
            raw: One Azure activity-log record.

        Returns:
            The canonical :class:`AgentAction`.

        Raises:
            AdapterError: If the record lacks the identity, target, or
                timestamp needed to form a valid AgentAction.
        """
        operation = _operation_name(raw)
        action_type = _action_type(operation)

        properties = raw.get("properties")
        if isinstance(properties, dict):
            change: dict[str, Any] = {
                "before": properties.get("before"),
                "after": properties.get("after"),
            }
        else:
            change = {"before": None, "after": None}

        event_timestamp = raw.get("eventTimestamp")
        try:
            if not isinstance(event_timestamp, str):
                raise AdapterError("Azure record has no string eventTimestamp")
            # Azure may emit a trailing 'Z'; normalise it to an explicit offset.
            timestamp = datetime.fromisoformat(event_timestamp.replace("Z", "+00:00"))
            return AgentAction(
                agent_id=raw.get("caller") or "",
                timestamp=timestamp,
                action_type=action_type,
                target=raw.get("resourceId") or "",
                change=change,
                source=self.source,
                raw_ref=raw.get("correlationId") or raw.get("eventDataId"),
            )
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"Malformed Azure record: {exc}") from exc
