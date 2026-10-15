"""LoRA capability seam for adapters.

Groups the LoRA lifecycle an adapter exposes when it supports LoRA into one
typed capability, so callers ask ``adapter.lora_capability()`` once and use the
returned handle instead of flag-checking ``supports_lora()`` and then calling
the individual hooks. The PEFT mixin (in-process, hot-reloadable) implements
this capability. An adapter whose engine loads its LoRAs at launch, such as the
SGLang embedding adapter, selects them by served name and returns ``None``.
"""

from __future__ import annotations

from typing import Protocol


class LoraCapability(Protocol):
    """The LoRA lifecycle exposed by a LoRA-capable adapter.

    An adapter that returns non-``None`` from ``lora_capability()`` implements
    this whole protocol, so callers no longer branch on ``supports_lora()`` and
    then hope the individual hooks are present.
    """

    def supports_hot_lora_reload(self) -> bool:
        """True if LoRAs can be loaded without blocking inference (PEFT), False
        if the load blocks all requests (SGLang HTTP).
        """

    def fetch_lora(self, lora_path: str, revision: str | None = None) -> str:
        """Bring a LoRA adapter's files to one local directory and return it.

        Hot-reload adapters do their network and disk I/O here, outside inference.
        """

    def load_lora(self, lora_path: str, revision: str | None = None, *, source: str | None = None) -> int:
        """Load a LoRA adapter (optionally at a pinned 40-hex commit SHA), from
        ``source`` when :meth:`fetch_lora` already brought it local; return its
        memory usage in bytes.
        """

    def unload_lora(self, lora_name: str) -> None:
        """Unload a previously loaded LoRA adapter."""

    def set_active_lora(self, lora_name: str | None) -> None:
        """Select the active LoRA for the next inference call (``None`` = base)."""
