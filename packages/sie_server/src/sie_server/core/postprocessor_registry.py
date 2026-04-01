"""Postprocessor registry for output transforms.

This module manages postprocessors for loaded models and provides
async postprocessing that runs in a shared thread pool.

Postprocessors are triggered explicitly via runtime options (not inferred
from output types). See DESIGN.md Section 7.3.2.

Example:
    # Register when model loads
    registry.register("colbert-ir/colbertv2.0", {"muvera": MuveraPostprocessor(...)})

    # Transform after inference (checks options for postprocessor keys)
    elapsed_ms = await registry.transform(model, output, options, is_query=False)

    # Unregister when model unloads
    registry.unregister("colbert-ir/colbertv2.0")
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from sie_server.core.postprocessor import QuantizePostprocessor

if TYPE_CHECKING:
    from sie_server.core.inference_output import EncodeOutput
    from sie_server.core.postprocessor import Postprocessor

logger = logging.getLogger(__name__)

# Known postprocessor option keys
POSTPROCESSOR_OPTION_KEYS = frozenset({"muvera", "output_dtype"})

# Global quantize postprocessor (stateless, shared across all models)
_QUANTIZE_POSTPROCESSOR = QuantizePostprocessor()


class PostprocessorRegistry:
    """Registry for model postprocessors with parallel execution.

    Manages postprocessors for all loaded models and applies transforms
    based on runtime options. Shares a thread pool with PreprocessorRegistry
    for controlled CPU contention.

    Postprocessors are keyed by option name (e.g., "muvera", "quantize").
    When transform() is called, it checks the options dict for these keys
    and applies the matching postprocessor if the value is not None.

    Usage:
        # Create with shared executor
        registry = PostprocessorRegistry(cpu_pool)

        # Register postprocessors when model loads
        registry.register(model_name, {"muvera": MuveraPostprocessor(...)})

        # Transform after inference (async, runs in thread pool)
        elapsed_ms = await registry.transform(model, output, options, is_query=False)

        # Unregister when model unloads
        registry.unregister(model_name)
    """

    def __init__(self, cpu_pool: ThreadPoolExecutor) -> None:
        """Initialize the postprocessor registry.

        Args:
            cpu_pool: Shared thread pool (typically from PreprocessorRegistry).
        """
        self._executor = cpu_pool
        # model_name -> option_key -> Postprocessor
        self._postprocessors: dict[str, dict[str, Postprocessor]] = {}
        logger.info("PostprocessorRegistry initialized with shared CPU pool")

    def register(self, model_name: str, postprocessors: dict[str, Postprocessor]) -> None:
        """Register postprocessors for a model.

        Args:
            model_name: Model name.
            postprocessors: Dict mapping option keys to postprocessor instances.
                Example: {"muvera": MuveraPostprocessor(...)}
        """
        if not postprocessors:
            return

        self._postprocessors[model_name] = dict(postprocessors)
        logger.debug(
            "Registered postprocessors for %s: %s",
            model_name,
            list(postprocessors.keys()),
        )

    def unregister(self, model_name: str) -> None:
        """Unregister all postprocessors for a model.

        Args:
            model_name: Model name to unregister.
        """
        if model_name in self._postprocessors:
            del self._postprocessors[model_name]
            logger.debug("Unregistered postprocessors for %s", model_name)

    def has_postprocessor(self, model_name: str, option_key: str) -> bool:
        """Check if a postprocessor is registered for a model and option key.

        Args:
            model_name: Model name.
            option_key: Option key (e.g., "muvera", "quantize").

        Returns:
            True if postprocessor is registered.
        """
        return model_name in self._postprocessors and option_key in self._postprocessors[model_name]

    def get_postprocessor(self, model_name: str, option_key: str) -> Postprocessor | None:
        """Get a postprocessor for a model and option key.

        Args:
            model_name: Model name.
            option_key: Option key (e.g., "muvera", "quantize").

        Returns:
            Postprocessor instance or None if not registered.
        """
        return self._postprocessors.get(model_name, {}).get(option_key)

    @property
    def registered_models(self) -> list[str]:
        """Return list of models with registered postprocessors."""
        return list(self._postprocessors.keys())

    def get_option_keys(self, model_name: str) -> list[str]:
        """Return list of registered option keys for a model.

        Args:
            model_name: Model name.

        Returns:
            List of option keys.
        """
        return list(self._postprocessors.get(model_name, {}).keys())

    async def transform(
        self,
        model_name: str,
        output: EncodeOutput,
        options: dict[str, Any],
        *,
        is_query: bool = False,
    ) -> float:
        """Apply postprocessors based on options.

        Checks options dict for known postprocessor keys. For each key that
        is present and not None, applies the matching postprocessor.

        Postprocessor order:
        1. Model-specific postprocessors (e.g., MUVERA)
        2. Global quantization (output_dtype)

        Args:
            model_name: Model name.
            output: EncodeOutput to transform (modified in-place).
            options: Runtime options dict (effective options after profile merge).
            is_query: Whether items are queries (affects some postprocessors).

        Returns:
            Total elapsed time in milliseconds.
        """
        # Find model-specific postprocessors to apply
        model_postprocessors = self._postprocessors.get(model_name, {})
        to_apply: list[tuple[str, Postprocessor, Any]] = []
        for option_key, postprocessor in model_postprocessors.items():
            if option_key in options:
                config = options[option_key]
                if config is not None:
                    to_apply.append((option_key, postprocessor, config))

        # Check for global quantization
        output_dtype = options.get("output_dtype")
        has_quantize = output_dtype is not None and output_dtype != "float32"

        if not to_apply and not has_quantize:
            return 0.0

        # Run transforms in thread pool
        loop = asyncio.get_running_loop()

        def _run_transforms() -> float:
            start = time.perf_counter()

            # 1. Apply model-specific postprocessors first (e.g., MUVERA)
            for option_key, postprocessor, _config in to_apply:
                logger.debug("Applying postprocessor %s for %s", option_key, model_name)
                postprocessor.transform(output, is_query=is_query)

            # 2. Apply global quantization last (after all transforms)
            if has_quantize:
                if not isinstance(output_dtype, str):
                    raise TypeError(f"output_dtype must be a string, got {type(output_dtype)}")
                logger.debug("Applying quantization %s for %s", output_dtype, model_name)
                _QUANTIZE_POSTPROCESSOR.quantize(output, output_dtype=output_dtype)

            elapsed_ms = (time.perf_counter() - start) * 1000
            return elapsed_ms

        transform_fn = functools.partial(_run_transforms)
        return await loop.run_in_executor(self._executor, transform_fn)

    def transform_sync(
        self,
        model_name: str,
        output: EncodeOutput,
        options: dict[str, Any],
        *,
        is_query: bool = False,
    ) -> float:
        """Apply postprocessors synchronously (for use outside async context).

        Postprocessor order:
        1. Model-specific postprocessors (e.g., MUVERA)
        2. Global quantization (output_dtype)

        Args:
            model_name: Model name.
            output: EncodeOutput to transform (modified in-place).
            options: Runtime options dict.
            is_query: Whether items are queries.

        Returns:
            Total elapsed time in milliseconds.
        """
        # Find model-specific postprocessors to apply
        model_postprocessors = self._postprocessors.get(model_name, {})
        to_apply: list[tuple[str, Postprocessor, Any]] = []
        for option_key, postprocessor in model_postprocessors.items():
            if option_key in options:
                config = options[option_key]
                if config is not None:
                    to_apply.append((option_key, postprocessor, config))

        # Check for global quantization
        output_dtype = options.get("output_dtype")
        has_quantize = output_dtype is not None and output_dtype != "float32"

        if not to_apply and not has_quantize:
            return 0.0

        start = time.perf_counter()

        # 1. Apply model-specific postprocessors first
        for option_key, postprocessor, _config in to_apply:
            logger.debug("Applying postprocessor %s for %s", option_key, model_name)
            postprocessor.transform(output, is_query=is_query)

        # 2. Apply global quantization last
        if has_quantize:
            if not isinstance(output_dtype, str):
                raise TypeError(f"output_dtype must be a string, got {type(output_dtype)}")
            logger.debug("Applying quantization %s for %s", output_dtype, model_name)
            _QUANTIZE_POSTPROCESSOR.quantize(output, output_dtype=output_dtype)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return elapsed_ms
