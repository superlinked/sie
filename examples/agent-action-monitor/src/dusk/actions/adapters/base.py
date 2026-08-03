"""Base contract for source adapters.

A :class:`SourceAdapter` maps one source's raw record into the canonical
:class:`~dusk.actions.event.AgentAction`. The contract is deliberate:

- Map source-specific operation names onto the canonical fields. Where a
  field is optional and absent, apply a sensible default (an empty
  ``details`` mapping, for example) rather than failing.
- Raise :class:`AdapterError` only when the record is malformed enough that
  it cannot yield a valid AgentAction (missing identity, missing target,
  unparseable timestamp). One bad record is the caller's to skip; a
  structurally impossible record is the adapter's to reject.

Adapters normalise shape only. They assign no severity, score, or verdict;
those belong to later layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dusk.actions.event import AgentAction


class AdapterError(ValueError):
    """Raised when a raw source record cannot yield a valid AgentAction."""


class SourceAdapter(ABC):
    """Abstract base for a single control-plane source adapter."""

    #: Canonical source name this adapter handles (for example ``"azure"``).
    source: str = "base"

    @abstractmethod
    def parse(self, raw: dict[str, Any]) -> AgentAction:
        """Map one raw source record into an :class:`AgentAction`.

        Args:
            raw: One record in the source's native shape.

        Returns:
            The canonical :class:`AgentAction`.

        Raises:
            AdapterError: If the record cannot yield a valid AgentAction.
        """
        raise NotImplementedError
