"""LoRA capability seam for adapters.

Groups the LoRA lifecycle an adapter exposes when it supports LoRA into one
typed capability, so callers ask ``adapter.lora_capability()`` once and use the
returned handle instead of flag-checking ``supports_lora()`` and then calling
the individual hooks. PEFT and SGLang generation/embedding implement this
capability. SGLang additionally describes engine-owned aliases and pinned
startup residency; adapters without runtime loading return ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LoraRuntimeConfig:
    """Engine-owned aliases and startup residency, keyed by declared source id."""

    aliases: dict[str, str]
    preloaded: frozenset[str]
    pinned: frozenset[str]
    max_loaded: int


class LoraStateUncertainError(RuntimeError):
    """A mutation could not be rolled back; retain its capacity until model reload."""


class LoraUnavailableError(ValueError):
    """The declared adapter is absent or is draining for eviction."""


class LoraCapability(Protocol):
    """The LoRA lifecycle exposed by a LoRA-capable adapter.

    An adapter that returns non-``None`` from ``lora_capability()`` implements
    this whole protocol, so callers no longer branch on ``supports_lora()`` and
    then hope the individual hooks are present.
    """

    def supports_hot_lora_reload(self) -> bool:
        """True when fetch can run in the background before a serialized attach.

        Attach may still briefly pause an engine's scheduler. This flag enables
        the immediate LORA_LOADING response; it does not promise zero impact
        on concurrent inference.
        """

    def fetch_lora(self, lora_path: str, revision: str | None = None) -> str:
        """Bring a LoRA adapter's files to one local directory and return it.

        Hot-reload adapters do their network and disk I/O here, outside inference.
        """

    def discard_lora_source(self, lora_path: str) -> None:
        """Release temporary fetched files only if no attached adapter can use them."""

    def load_lora(self, lora_path: str, revision: str | None = None, *, source: str | None = None) -> int:
        """Load a LoRA adapter (optionally at a pinned 40-hex commit SHA), from
        ``source`` when :meth:`fetch_lora` already brought it local; return its
        memory usage in bytes.
        """

    def unload_lora(self, lora_name: str) -> None:
        """Unload a previously loaded LoRA adapter."""

    def set_active_lora(self, lora_name: str | None) -> None:
        """Select the active LoRA for the next inference call (``None`` = base)."""
