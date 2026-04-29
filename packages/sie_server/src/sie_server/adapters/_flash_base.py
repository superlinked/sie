from __future__ import annotations

import importlib
import logging
from typing import Any, ClassVar

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters.base import ModelAdapter

logger = logging.getLogger(__name__)

# HuggingFace tokenizers use a very large integer as a sentinel meaning
# "no model_max_length set" (transformers.tokenization_utils_base.VERY_LARGE_INTEGER
# is int(1e30)). Anything at or above this threshold should be treated as "unknown".
_HF_TOKENIZER_MAX_LENGTH_SENTINEL = int(1e29)


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

    # -- Shared helpers ------------------------------------------------------

    def _resolve_tokenizer_ceiling(
        self,
        tokenizer: Any,
        model: Any,
        requested: int,
    ) -> int:
        """Return the largest safe sequence length for this tokenizer+model.

        Clamps ``requested`` to the minimum of ``tokenizer.model_max_length``
        and ``model.config.max_position_embeddings`` when those are set to
        real (non-sentinel) values. Logs a warning if clamping had to occur.

        This guards against bundled YAMLs or runtime overrides asking for a
        sequence length that the model's positional embeddings cannot
        actually support — feeding such inputs through the model triggers a
        device-side index-out-of-bounds in
        ``embedding(position_ids)`` which on CUDA poisons the worker.

        Args:
            tokenizer: A HuggingFace tokenizer (anything with
                ``model_max_length``); may be ``None``.
            model: A HuggingFace model whose ``.config`` may carry
                ``max_position_embeddings``. Pass ``None`` when the adapter
                loads weights without a HF model object (e.g. raw safetensors
                in nomic_flash) — the helper will then fall back to the
                tokenizer cap alone.
            requested: The currently configured ``max_seq_length``.

        Returns:
            ``min(requested, *valid_caps)``; ``requested`` unchanged when
            no informative cap is available.
        """
        caps: list[int] = []

        tok_max = getattr(tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and tok_max > 0 and tok_max < _HF_TOKENIZER_MAX_LENGTH_SENTINEL:
            caps.append(tok_max)

        model_config = getattr(model, "config", None)
        pos_max = getattr(model_config, "max_position_embeddings", None)
        if isinstance(pos_max, int) and pos_max > 0:
            caps.append(pos_max)

        if not caps:
            return requested

        ceiling = min(caps)
        if requested > ceiling:
            logger.warning(
                "%s: configured max_seq_length=%d exceeds model capacity %d "
                "(tokenizer.model_max_length=%s, model.config.max_position_embeddings=%s); "
                "clamping to %d to avoid out-of-bounds position embeddings.",
                type(self).__name__,
                requested,
                ceiling,
                tok_max,
                pos_max,
                ceiling,
            )
            return ceiling
        return requested

    @staticmethod
    def _coerce_runtime_max_length(raw: Any, ceiling: int) -> int:
        """Validate a runtime ``max_seq_length`` override and clamp to ``ceiling``.

        Runtime overrides arrive untyped from request payloads / profile JSON
        and may be ``None``, a string, a float, or a negative number. A
        malformed value must never reach ``min(...)`` (which crashes for
        ``None`` / non-numerics) and must never bypass the load-time ceiling.

        Args:
            raw: The raw ``opts.get("max_seq_length")`` value.
            ceiling: The load-time resolved ceiling (positive int).

        Returns:
            ``min(int(raw), ceiling)`` when ``raw`` is a positive integer
            (or a string/float coercible to one); otherwise ``ceiling``.
        """
        if raw is None:
            return ceiling
        # Reject bools (which are ``int`` subclasses) — they are never a
        # meaningful sequence length.
        if isinstance(raw, bool):
            return ceiling
        if isinstance(raw, int):
            candidate = raw
        else:
            try:
                candidate = int(raw)
            except (TypeError, ValueError):
                return ceiling
        if candidate <= 0:
            return ceiling
        return min(candidate, ceiling)

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
