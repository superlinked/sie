"""Preprocessor registry for parallel preprocessing across modalities.

This module manages preprocessors for all loaded models and provides
async preprocessing that runs in a thread pool to avoid blocking
the event loop.

See roadmap.md Project 10.4 Phase 2 for design details.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from sie_server.core.preprocessor import ImagePreprocessor, Preprocessor, TextPreprocessor

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    from sie_server.config.model import ModelConfig
    from sie_server.core.prepared import PreparedBatch
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# Default number of preprocessing workers
DEFAULT_WORKERS = 8


class PreprocessorRegistry:
    """Registry for model preprocessors with parallel execution.

    Manages preprocessors for all loaded models and dispatches preprocessing
    to the appropriate handler based on item content and model capabilities.

    Thread pool is shared across all models for CPU-bound preprocessing work.
    This includes tokenization, image processing, and future modalities.

    Usage:
        registry = PreprocessorRegistry()

        # Register preprocessors when model loads
        registry.register_text(model_name, tokenizer)
        registry.register_image(model_name, processor)

        # Prepare items for batching (async, runs in thread pool)
        batch = await registry.prepare(model_name, items, config)

        # Unregister when model unloads
        registry.unregister(model_name)
    """

    def __init__(self, max_workers: int | None = None) -> None:
        """Initialize the preprocessor registry.

        Args:
            max_workers: Number of worker threads. Defaults to min(CPU count, 8).
        """
        if max_workers is None:
            max_workers = min(os.cpu_count() or DEFAULT_WORKERS, DEFAULT_WORKERS)

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="preprocess",
        )
        self._max_workers = max_workers

        # model_name -> modality -> Preprocessor
        self._preprocessors: dict[str, dict[str, Preprocessor]] = {}

        logger.info("PreprocessorRegistry initialized with %d workers", max_workers)

    @property
    def max_workers(self) -> int:
        """Return maximum number of worker threads."""
        return self._max_workers

    def register_text(
        self,
        model_name: str,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    ) -> None:
        """Register a text preprocessor (tokenizer) for a model.

        Args:
            model_name: Model name.
            tokenizer: HuggingFace tokenizer instance.
        """
        preprocessor = TextPreprocessor(tokenizer, model_name)
        self._register(model_name, preprocessor)
        logger.debug("Registered text preprocessor for %s", model_name)

    def register_image(
        self,
        model_name: str,
        processor: Any,  # SiglipProcessor, CLIPProcessor, NemoColEmbedPreprocessor, etc.
    ) -> None:
        """Register an image preprocessor for a model.

        Args:
            model_name: Model name.
            processor: HuggingFace processor with image processing capability,
                or a custom Preprocessor instance (e.g., NemoColEmbedPreprocessor).
        """
        # If processor is already a Preprocessor, use it directly
        if isinstance(processor, Preprocessor):
            preprocessor = processor
        else:
            # Wrap HuggingFace processor in ImagePreprocessor
            preprocessor = ImagePreprocessor(processor, model_name)
        self._register(model_name, preprocessor)
        logger.debug("Registered image preprocessor for %s", model_name)

    def _register(self, model_name: str, preprocessor: Preprocessor) -> None:
        """Register a preprocessor for a model and modality.

        Args:
            model_name: Model name.
            preprocessor: Preprocessor instance.
        """
        if model_name not in self._preprocessors:
            self._preprocessors[model_name] = {}
        self._preprocessors[model_name][preprocessor.modality] = preprocessor

    def unregister(self, model_name: str) -> None:
        """Unregister all preprocessors for a model.

        Args:
            model_name: Model name to unregister.
        """
        if model_name in self._preprocessors:
            del self._preprocessors[model_name]
            logger.debug("Unregistered preprocessors for %s", model_name)

    def has_preprocessor(self, model_name: str, modality: str = "text") -> bool:
        """Check if a preprocessor is registered for a model and modality.

        Args:
            model_name: Model name.
            modality: Modality to check ("text", "image", "audio").

        Returns:
            True if preprocessor is registered.
        """
        return model_name in self._preprocessors and modality in self._preprocessors[model_name]

    def get_preprocessor(self, model_name: str, modality: str) -> Preprocessor | None:
        """Get a preprocessor for a model and modality.

        Args:
            model_name: Model name.
            modality: Modality ("text", "image", "audio").

        Returns:
            Preprocessor instance or None if not registered.
        """
        return self._preprocessors.get(model_name, {}).get(modality)

    @property
    def registered_models(self) -> list[str]:
        """Return list of models with registered preprocessors."""
        return list(self._preprocessors.keys())

    def get_modalities(self, model_name: str) -> list[str]:
        """Return list of registered modalities for a model.

        Args:
            model_name: Model name.

        Returns:
            List of modality names.
        """
        return list(self._preprocessors.get(model_name, {}).keys())

    async def prepare(
        self,
        model_name: str,
        items: list[Item],
        config: ModelConfig,
        *,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[Any]:
        """Prepare items for batching.

        Dispatches to appropriate preprocessor based on item content.
        Runs in thread pool to avoid blocking event loop.

        Args:
            model_name: Model name.
            items: Items to prepare.
            config: Model configuration.
            is_query: Whether items are queries (True) or documents (False).
                Passed to preprocessor to allow query-specific handling
                (e.g., ColBERT query expansion, instruction prefixes).
            instruction: Optional instruction text (e.g., DocVQA question).
                Passed to vision preprocessors like Florence2Preprocessor.
            task: Optional task token (e.g., "<DocVQA>", "<OCR_WITH_REGION>").
                Passed to vision preprocessors to override default task.

        Returns:
            PreparedBatch ready for batching and inference.

        Raises:
            KeyError: If no preprocessor is registered for the required modality.
            ValueError: If items contain unsupported modalities.
        """
        # Determine required modality based on items
        has_text = any(item.text is not None for item in items)
        has_images = any(item.images for item in items)  # Truthy if non-empty list

        # Select preprocessor based on content
        # Priority: images > text (matching current adapter behavior for CLIP/SigLIP)
        if has_images:
            modality = "image"
        elif has_text:
            modality = "text"
        else:
            msg = "Items must have text or images"
            raise ValueError(msg)

        preprocessor = self.get_preprocessor(model_name, modality)
        if preprocessor is None:
            msg = f"No {modality} preprocessor registered for model: {model_name}"
            raise KeyError(msg)

        # Trivial preprocessors (e.g. CharCountPreprocessor) run inline to
        # avoid ~1.5ms thread-pool scheduling overhead per request.
        if getattr(preprocessor, "is_trivial", False):
            return preprocessor.prepare(
                items,
                config=config,
                is_query=is_query,
                instruction=instruction,
                task=task,
            )

        # Run preprocessing in thread pool
        # Use functools.partial to handle keyword-only arguments
        # Pass instruction and task for vision models (Florence-2, Donut)
        loop = asyncio.get_running_loop()
        prepare_fn = functools.partial(
            preprocessor.prepare,
            items,
            config=config,
            is_query=is_query,
            instruction=instruction,
            task=task,
        )
        return await loop.run_in_executor(self._executor, prepare_fn)

    def prepare_sync(
        self,
        model_name: str,
        items: list[Item],
        config: ModelConfig,
        *,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[Any]:
        """Prepare items synchronously (for use outside async context).

        Args:
            model_name: Model name.
            items: Items to prepare.
            config: Model configuration.
            is_query: Whether items are queries (True) or documents (False).
            instruction: Optional instruction text (e.g., DocVQA question).
            task: Optional task token (e.g., "<DocVQA>").

        Returns:
            PreparedBatch ready for batching and inference.
        """
        # Determine required modality based on items
        has_text = any(item.text is not None for item in items)
        has_images = any(item.images for item in items)  # Truthy if non-empty list

        if has_images:
            modality = "image"
        elif has_text:
            modality = "text"
        else:
            msg = "Items must have text or images"
            raise ValueError(msg)

        preprocessor = self.get_preprocessor(model_name, modality)
        if preprocessor is None:
            msg = f"No {modality} preprocessor registered for model: {model_name}"
            raise KeyError(msg)

        return preprocessor.prepare(items, config=config, is_query=is_query, instruction=instruction, task=task)

    def shutdown(self, *, wait: bool = True) -> None:
        """Shutdown the thread pool.

        Args:
            wait: Whether to wait for pending work to complete.
        """
        self._executor.shutdown(wait=wait)
        self._preprocessors.clear()
        logger.info("PreprocessorRegistry shutdown")
