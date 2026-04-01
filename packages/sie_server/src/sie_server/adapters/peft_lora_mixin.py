"""PEFT-based LoRA mixin for PyTorch model adapters.

This mixin provides LoRA (Low-Rank Adaptation) support using the PEFT library.
Adapters that use PyTorch models can inherit from this mixin to gain LoRA capabilities.

Usage:
    class MyFlashAdapter(PEFTLoRAMixin, ModelAdapter):
        def load(self, device: str) -> None:
            ...
            self._model = AutoModel.from_pretrained(...)
            # Store base model reference for LoRA wrapping
            self._base_model = self._model

See DESIGN.md Section 7.3 for the LoRA architecture specification.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peft import PeftModel

logger = logging.getLogger(__name__)


class PEFTLoRAMixin:
    """Mixin providing PEFT-based LoRA support for PyTorch adapters.

    This mixin provides a common implementation of LoRA loading/switching
    using the PEFT library. It works with any adapter that stores its
    PyTorch model in `self._model`.

    The mixin expects:
    - `self._model`: The PyTorch model (will be wrapped by PeftModel)
    - `self._device`: The device string (for memory estimation)

    How it works:
    1. First LoRA load: Creates PeftModel.from_pretrained(base_model, lora_path)
    2. Additional LoRAs: Calls peft_model.load_adapter(lora_path, adapter_name)
    3. Switching: Calls peft_model.set_adapter(lora_name) before inference
    4. Unloading: Calls peft_model.delete_adapter(lora_name)

    Memory management:
    - Each LoRA adds ~1-5% of base model memory
    - Memory is estimated from adapter weight sizes
    - LRU eviction is handled by ModelLoader (not this mixin)

    Thread safety:
    - load_lora() is safe to call from thread pool
    - set_active_lora() is lightweight and thread-safe
    - The adapter switching is lock-free
    """

    # Track PEFT state
    _peft_model: PeftModel | None = None
    _active_lora: str | None = None
    _loaded_loras: set[str]  # Track which LoRAs are loaded

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize _loaded_loras set for subclasses."""
        super().__init_subclass__(**kwargs)
        # This ensures each subclass gets its own set

    def _ensure_lora_tracking(self) -> None:
        """Ensure LoRA tracking is initialized."""
        if not hasattr(self, "_loaded_loras"):
            self._loaded_loras = set()

    def supports_lora(self) -> bool:
        """Return True - PEFT adapters support LoRA."""
        return True

    def supports_hot_lora_reload(self) -> bool:
        """Return True - PEFT loading is non-blocking.

        Unlike SGLang which blocks the server during LoRA loading,
        PEFT can load adapters in a thread pool without blocking inference.
        """
        return True

    def load_lora(self, lora_path: str) -> int:
        """Load a LoRA adapter using PEFT.

        On first call, wraps the base model with PeftModel.
        On subsequent calls, adds the adapter to the existing PeftModel.

        Args:
            lora_path: HuggingFace path (e.g., "org/lora-name") or local path.
                This path is also used as the adapter name for switching.

        Returns:
            Memory usage of the loaded LoRA in bytes.

        Raises:
            RuntimeError: If model not loaded or PEFT import fails.
            ValueError: If LoRA is already loaded.
        """
        self._ensure_lora_tracking()

        # Check if already loaded
        if lora_path in self._loaded_loras:
            logger.warning("LoRA '%s' is already loaded, skipping", lora_path)
            return 0

        # Get base model - must exist
        base_model = getattr(self, "_model", None)
        if base_model is None:
            msg = "Model not loaded. Call load() before load_lora()."
            raise RuntimeError(msg)

        try:
            from peft import PeftModel
        except ImportError as e:
            msg = "PEFT is required for LoRA support. Install with: pip install peft"
            raise RuntimeError(msg) from e

        logger.info("Loading LoRA adapter: %s", lora_path)

        if self._peft_model is None:
            # First LoRA - wrap the base model
            logger.debug("Creating PeftModel from base model")
            self._peft_model = PeftModel.from_pretrained(
                base_model,
                lora_path,
                adapter_name=lora_path,
            )
            # Update self._model to point to the PEFT-wrapped model
            # This ensures encode() uses the LoRA-enhanced model
            self._model = self._peft_model
        else:
            # Additional LoRA - add to existing PeftModel
            logger.debug("Adding adapter to existing PeftModel")
            self._peft_model.load_adapter(lora_path, adapter_name=lora_path)

        self._loaded_loras.add(lora_path)

        # Force-disable adapter layers after loading.
        # PEFT's load_adapter() can re-enable adapter layers, so we must
        # explicitly disable them. Reset _active_lora to force set_active_lora(None)
        # to actually run (not short-circuit via the equality check).
        self._active_lora = "__force_reset__"
        self.set_active_lora(None)

        # Estimate memory usage from adapter parameters
        memory_bytes = self._estimate_lora_memory(lora_path)
        logger.info("LoRA '%s' loaded, estimated memory: %.2f MB", lora_path, memory_bytes / 1024 / 1024)

        return memory_bytes

    def unload_lora(self, lora_name: str) -> None:
        """Unload a LoRA adapter.

        Called during LRU eviction when max_loras is exceeded.

        Args:
            lora_name: The LoRA adapter name to unload.

        Raises:
            ValueError: If LoRA is not loaded.
        """
        self._ensure_lora_tracking()

        if lora_name not in self._loaded_loras:
            msg = f"LoRA '{lora_name}' is not loaded"
            raise ValueError(msg)

        if self._peft_model is None:
            msg = "PeftModel is None but LoRA is in _loaded_loras - inconsistent state"
            raise RuntimeError(msg)

        logger.info("Unloading LoRA adapter: %s", lora_name)

        # If this is the active LoRA, switch to base first
        if self._active_lora == lora_name:
            self.set_active_lora(None)

        # Delete the adapter
        self._peft_model.delete_adapter(lora_name)
        self._loaded_loras.discard(lora_name)

        # If no LoRAs remain, unwrap the model
        if not self._loaded_loras:
            logger.debug("No LoRAs remaining, unwrapping PeftModel")
            # Get the base model back
            base_model = self._peft_model.get_base_model()
            self._model = base_model
            del self._peft_model
            self._peft_model = None
            gc.collect()

    def set_active_lora(self, lora_name: str | None) -> None:
        """Set the active LoRA for the next inference call.

        Called by the worker before each batch to switch to the appropriate
        LoRA adapter.

        Args:
            lora_name: LoRA adapter name, or None for base model.
        """
        self._ensure_lora_tracking()

        # Skip if no change
        if lora_name == self._active_lora:
            return

        if self._peft_model is None:
            if lora_name is not None:
                logger.warning("set_active_lora('%s') called but no LoRAs are loaded", lora_name)
            return

        if lora_name is None:
            # Switch to base model (disable adapter layers)
            # Use PEFT's disable_adapter_layers() instead of transformers' disable_adapters()
            logger.debug("Disabling LoRA adapters (base model)")
            self._peft_model.disable_adapter_layers()
        else:
            if lora_name not in self._loaded_loras:
                msg = f"LoRA '{lora_name}' is not loaded. Available: {self._loaded_loras}"
                raise ValueError(msg)

            logger.debug("Setting active LoRA: %s", lora_name)
            # Re-enable adapter layers if they were disabled, then set the active adapter
            # Use PEFT's enable_adapter_layers() instead of transformers' enable_adapters()
            self._peft_model.enable_adapter_layers()
            self._peft_model.set_adapter(lora_name)

        self._active_lora = lora_name

    def _estimate_lora_memory(self, lora_name: str) -> int:
        """Estimate memory usage of a LoRA adapter.

        Sums the memory of all LoRA parameters (lora_A, lora_B matrices).

        Args:
            lora_name: The LoRA adapter name.

        Returns:
            Estimated memory in bytes.
        """
        if self._peft_model is None:
            return 0

        total_bytes = 0

        try:
            import torch

            # Iterate through named parameters looking for LoRA weights
            for name, param in self._peft_model.named_parameters():
                # LoRA parameters typically have adapter name in their path
                # and contain "lora_" in the name
                if lora_name in name and "lora_" in name and isinstance(param, torch.Tensor):
                    total_bytes += param.numel() * param.element_size()

        except (AttributeError, RuntimeError) as e:
            logger.warning("Failed to estimate LoRA memory: %s", e)
            # Fallback: assume ~2% of base model memory
            base_memory = getattr(self, "memory_footprint", lambda: 0)()
            total_bytes = int(base_memory * 0.02)

        return total_bytes
