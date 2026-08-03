"""Generic adapter for vendor-neutral action records.

The generic record already uses the canonical field names (``agent_id``,
``timestamp``, ``action_type``, ``target``, ``change``, ``source``, optional
``raw_ref``). This is the path for any source we have not written a dedicated
adapter for yet, and the path the synthetic generator uses. The adapter
validates and builds an :class:`~dusk.actions.event.AgentAction`, surfacing
any problem as an :class:`AdapterError`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from dusk.actions.adapters.base import AdapterError, SourceAdapter
from dusk.actions.event import AgentAction


class GenericAdapter(SourceAdapter):
    """Adapter for records already shaped like the canonical schema."""

    source = "generic"

    def parse(self, raw: dict[str, Any]) -> AgentAction:
        """Validate and build an :class:`AgentAction` from a generic record.

        Args:
            raw: A record whose keys match the canonical AgentAction fields.
                ``timestamp`` may be a datetime or an ISO 8601 string.

        Returns:
            The canonical :class:`AgentAction`.

        Raises:
            AdapterError: If a required field is missing or any value is
                invalid (empty identity or target, naive timestamp, unknown
                action_type).
        """
        try:
            raw_timestamp = raw["timestamp"]
            timestamp = (
                datetime.fromisoformat(raw_timestamp)
                if isinstance(raw_timestamp, str)
                else raw_timestamp
            )
            change = raw.get("change") or {"before": None, "after": None}
            return AgentAction(
                agent_id=raw["agent_id"],
                timestamp=timestamp,
                action_type=raw["action_type"],
                target=raw["target"],
                change=change,
                source=raw.get("source", self.source),
                raw_ref=raw.get("raw_ref"),
            )
        except KeyError as exc:
            raise AdapterError(f"Generic record missing required field: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise AdapterError(f"Invalid generic record: {exc}") from exc
