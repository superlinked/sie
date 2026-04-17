from __future__ import annotations

import importlib
import logging
from typing import Any, ClassVar

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters.base import ModelAdapter

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


class FlashBaseAdapter(BaseAdapter):
    """Thin base class for all flash-attention adapters.

    Provides:
    - Declarative fallback: set ``fallback_adapter_path`` and optionally
      ``fallback_kwargs_overrides`` instead of overriding ``create_for_device``.

    Inherits from ``BaseAdapter``:
    - ``unload()`` driven by ``spec.unload_fields``
    - ``_resolve_dtype()``
    - ``get_preprocessor()`` returning ``CharCountPreprocessor``
    - ``_check_loaded()``

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

        if device.startswith("cuda") and is_flash_attention_available(device):
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
                "To install on Linux: uv add 'sie-server[flash-attn]'",
                fallback_class.__name__,
                device,
            )

        return fallback_class(**merged)
