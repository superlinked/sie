"""Route source records through registered AgentAction adapters."""

from __future__ import annotations

import logging
from typing import Any

from dusk.actions.adapters.azure import AzureAdapter
from dusk.actions.adapters.base import AdapterError, SourceAdapter
from dusk.actions.adapters.bedrock import BedrockAdapter
from dusk.actions.adapters.generic import GenericAdapter
from dusk.actions.event import AgentAction

logger = logging.getLogger("dusk.actions.normaliser")

#: Registry of source adapters, keyed by canonical source name.
_REGISTRY: dict[str, SourceAdapter] = {
    GenericAdapter.source: GenericAdapter(),
    AzureAdapter.source: AzureAdapter(),
    BedrockAdapter.source: BedrockAdapter(),
}


def known_sources() -> list[str]:
    """Return the sorted list of registered source names."""
    return sorted(_REGISTRY)


def get_adapter(source: str) -> SourceAdapter:
    """Return the adapter registered for ``source``.

    Args:
        source: Canonical source name, for example ``"azure"`` or
            ``"generic"``.

    Returns:
        The registered :class:`SourceAdapter`.

    Raises:
        AdapterError: If no adapter is registered for ``source``.
    """
    adapter = _REGISTRY.get(source)
    if adapter is None:
        raise AdapterError(
            f"No adapter registered for source '{source}'. Known sources: "
            f"{', '.join(known_sources())}"
        )
    return adapter


def normalise_record(source: str, raw: dict[str, Any]) -> AgentAction:
    """Normalise one raw record from ``source`` into an :class:`AgentAction`.

    Args:
        source: Canonical source name selecting the adapter.
        raw: One raw record in that source's native shape.

    Returns:
        The canonical :class:`AgentAction`.

    Raises:
        AdapterError: If the source is unknown or the record is malformed.
    """
    return get_adapter(source).parse(raw)
