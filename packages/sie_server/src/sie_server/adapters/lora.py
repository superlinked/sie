"""LoRA capability seam for adapters.

Groups the LoRA lifecycle an adapter exposes when it supports LoRA into one
typed capability, so callers ask ``adapter.lora_capability()`` once and use the
returned handle instead of flag-checking ``supports_lora()`` and then calling
the individual hooks. The two implementations behind this capability are the
PEFT mixin (in-process, hot-reloadable) and the SGLang adapter (HTTP, blocking).
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

    def load_lora(self, lora_path: str, revision: str | None = None) -> int:
        """Load a LoRA adapter (optionally at a pinned 40-hex commit SHA);
        return its memory usage in bytes.
        """

    def unload_lora(self, lora_name: str) -> None:
        """Unload a previously loaded LoRA adapter."""

    def set_active_lora(self, lora_name: str | None) -> None:
        """Select the active LoRA for the next inference call (``None`` = base)."""
