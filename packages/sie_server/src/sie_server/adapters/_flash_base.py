from __future__ import annotations

import gc
import importlib
import logging
from typing import TYPE_CHECKING, Any, ClassVar

from sie_server.adapters.base import ModelAdapter

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


def _import_adapter_class(adapter_path: str) -> type[ModelAdapter]:
    """Import an adapter class from a 'module:ClassName' string.

    Uses the same resolution logic as ``loader._import_builtin_adapter`` but
    accepts the **short form** used in the fallback table (e.g.
    ``"sentence_transformer:SentenceTransformerDenseAdapter"``).  The
    ``sie_server.adapters.`` prefix is prepended automatically.

    Args:
        adapter_path: ``"module:ClassName"`` or
            ``"sie_server.adapters.module:ClassName"`` string.

    Returns:
        The adapter class.

    Raises:
        ImportError: If module or class cannot be found.
    """
    if ":" not in adapter_path:
        msg = f"Invalid adapter_path '{adapter_path}': expected 'module:ClassName'"
        raise ImportError(msg)

    module_path, class_name = adapter_path.rsplit(":", 1)

    # Allow both short form ("sentence_transformer:Foo") and full form
    if not module_path.startswith("sie_server."):
        module_path = f"sie_server.adapters.{module_path}"

    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        msg = f"Adapter class '{class_name}' not found in module '{module_path}'"
        raise ImportError(msg)

    return getattr(module, class_name)


class FlashBaseAdapter(ModelAdapter):
    """Thin base class for all flash-attention adapters.

    Provides:
    - Declarative fallback: set ``fallback_adapter_path`` and optionally
      ``fallback_kwargs_overrides`` instead of overriding ``create_for_device``.
    - Common ``unload()`` with gc + cache clearing.
    - ``_resolve_dtype()`` for compute precision mapping.
    - ``get_preprocessor()`` returning ``CharCountPreprocessor``.

    Subclasses with custom fallback logic (e.g. SPLADEFlashAdapter) can still
    override ``create_for_device()`` directly.
    """

    # -- Declarative fallback ------------------------------------------------
    # Subclasses set these to enable automatic flash -> non-flash fallback.
    fallback_adapter_path: ClassVar[str | None] = None
    fallback_kwargs_overrides: ClassVar[dict[str, Any]] = {}

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> ModelAdapter:
        """Factory method for device-aware adapter instantiation.

        When ``fallback_adapter_path`` is ``None`` the adapter is always
        returned as-is (no fallback).  Otherwise the standard CUDA +
        flash-attn check runs and falls back to the declared class on
        incompatible hardware.
        """
        if cls.fallback_adapter_path is None:
            return cls(**kwargs)

        # Import lazily to avoid circular deps (core.inference -> core.loader -> base)
        from sie_server.core.inference import is_flash_attention_available

        if device.startswith("cuda") and is_flash_attention_available():
            return cls(**kwargs)

        # Resolve fallback class from string path
        fallback_class = _import_adapter_class(cls.fallback_adapter_path)
        merged = {**kwargs, **cls.fallback_kwargs_overrides}

        if not device.startswith("cuda"):
            logger.info(
                "%s requires CUDA. Using %s for device '%s'. "
                "For optimal performance, use a Linux system with NVIDIA GPU.",
                cls.__name__,
                fallback_class.__name__,
                device,
            )
        else:
            logger.warning(
                "Flash Attention unavailable (requires Ampere+ GPU and flash-attn package). "
                "Using %s for device '%s'. "
                "To install on Linux: pip install sie-server[flash-attn]",
                fallback_class.__name__,
                device,
            )

        return fallback_class(**merged)

    # -- Common unload -------------------------------------------------------
    def unload(self) -> None:
        """Unload model weights and free GPU memory."""
        import torch as _torch

        device = getattr(self, "_device", None)

        # Clear standard fields
        for attr in ("_model", "_tokenizer"):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)

        # Clear subclass-specific fields
        for attr in self._extra_fields_to_clear():
            if hasattr(self, attr):
                setattr(self, attr, None)

        self._device = None

        gc.collect()
        if device and str(device).startswith("cuda"):
            _torch.cuda.empty_cache()

    def _extra_fields_to_clear(self) -> list[str]:
        """Override to list additional instance attributes to clear on unload."""
        return []

    # -- Shared utilities ----------------------------------------------------
    def _resolve_dtype(self) -> torch.dtype:
        """Map ``self._compute_precision`` to a ``torch.dtype``."""
        import torch as _torch

        dtype_map: dict[str, torch.dtype] = {
            "float16": _torch.float16,
            "bfloat16": _torch.bfloat16,
            "float32": _torch.float32,
        }
        return dtype_map.get(
            getattr(self, "_compute_precision", "float16"),
            _torch.float16,
        )

    def get_preprocessor(self) -> Any:
        """Return ``CharCountPreprocessor`` for cost estimation."""
        from sie_server.core.preprocessor import CharCountPreprocessor

        return CharCountPreprocessor(
            model_name=getattr(self, "_model_name_or_path", ""),
        )
