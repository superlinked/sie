import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import Item
from sie_server.types.responses import Entity

logger = logging.getLogger(__name__)

_ERR_REQUIRES_LABELS = "GLiNER-bi requires labels parameter for extraction"

# Maximum number of distinct label-set embeddings to cache.
_LABEL_CACHE_MAX_SIZE = 64


class GLiNERBiAdapter(BaseAdapter):
    """Adapter for GLiNER bi-encoder models with pre-computed label embedding caching.

    Uses the standard ``gliner`` package but with bi-encoder architecture.
    The key performance feature is that label embeddings can be pre-computed
    and cached via ``encode_labels()`` + ``batch_predict_with_embeds()``,
    giving near-constant inference time regardless of label count.

    Reference models:
    - knowledgator/gliner-bi-base-v2.0 (Ettin text encoder)
    - knowledgator/modern-gliner-bi-base-v1.0 (ModernBERT text encoder)

    See plan .kilo/plans/1776678677227-glowing-moon.md for design details.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model",),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        precompute_labels: bool = True,
        attn_implementation: str | None = None,
        max_len: int | None = None,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            threshold: Minimum confidence score for entity extraction (0-1).
            flat_ner: If True, enforce non-overlapping entities.
            multi_label: If True, allow same span to have multiple labels.
            precompute_labels: If True, cache label embeddings for reuse across
                requests with the same label set. This is the key bi-encoder
                performance feature.
            attn_implementation: Attention implementation for loading (e.g.,
                ``"flash_attention_2"`` for ModernBERT-based models).
            max_len: Maximum sequence length override for the model.
            compute_precision: Compute precision for inference.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._threshold = threshold
        self._flat_ner = flat_ner
        self._multi_label = multi_label
        self._precompute_labels = precompute_labels
        self._attn_implementation = attn_implementation
        self._max_len = max_len
        self._compute_precision = compute_precision

        self._model: Any = None
        self._device: str | None = None
        # LRU cache: frozenset(labels) -> pre-computed label embeddings.
        # Protected by _cache_lock for thread safety (defensive — the
        # current ModelWorker uses a single-worker executor, but this
        # guards against future architectural changes).
        self._label_cache: OrderedDict[frozenset[str], Any] = OrderedDict()
        self._cache_lock = threading.Lock()

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu", "mps").
        """
        from gliner import GLiNER  # ty:ignore[unresolved-import]

        self._device = device

        dtype = torch.float32
        if device != "cpu":
            if self._compute_precision == "float16":
                dtype = torch.float16
            elif self._compute_precision == "bfloat16":
                dtype = torch.bfloat16

        # Build from_pretrained kwargs
        load_kwargs: dict[str, Any] = {}
        if self._attn_implementation is not None:
            # Fall back to SDPA if Flash Attention 2 is not available
            attn_impl = self._attn_implementation
            if attn_impl == "flash_attention_2":
                from sie_server.core.inference import is_flash_attention_available

                if not is_flash_attention_available(device):
                    logger.info(
                        "Flash Attention 2 not available on %s, falling back to SDPA",
                        device,
                    )
                    attn_impl = "sdpa"
            load_kwargs["_attn_implementation"] = attn_impl
        if self._max_len is not None:
            load_kwargs["max_length"] = self._max_len

        self._model = GLiNER.from_pretrained(
            self._model_name_or_path,
            **load_kwargs,
        )

        if device == "cpu":
            self._model = self._model.to(device)
        else:
            self._model = self._model.to(device, dtype=dtype)

        # Clear any stale label cache from a prior load
        self._label_cache.clear()

    def unload(self) -> None:
        """Unload model and clear label embedding cache."""
        with self._cache_lock:
            self._label_cache.clear()
        super().unload()

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        """Extract entities from items using bi-encoder with optional label caching.

        When ``precompute_labels`` is enabled (default), label embeddings are
        computed once and cached so that subsequent requests with the same label
        set skip the label encoder entirely. This makes inference time
        nearly independent of label count.

        Args:
            items: List of items to extract from (must have text).
            labels: Entity types to extract (e.g., ["person", "organization"]).
            output_schema: Unused (interface compatibility).
            instruction: Unused (interface compatibility).
            options: Adapter options to override defaults.
                Supported: threshold, flat_ner, multi_label, precompute_labels.
            prepared_items: Unused (interface compatibility).

        Returns:
            ExtractOutput with entities per item.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If labels not provided or items lack text.
        """
        self._check_loaded()

        if not labels:
            raise ValueError(_ERR_REQUIRES_LABELS)

        texts = [self._extract_text(item) for item in items]

        opts = options or {}
        effective_threshold = opts.get("threshold", self._threshold)
        effective_flat_ner = opts.get("flat_ner", self._flat_ner)
        effective_multi_label = opts.get("multi_label", self._multi_label)
        use_precompute = opts.get("precompute_labels", self._precompute_labels)

        with torch.inference_mode():
            if use_precompute:
                batch_entities = self._predict_with_cached_embeds(
                    texts,
                    labels,
                    threshold=effective_threshold,
                    flat_ner=effective_flat_ner,
                    multi_label=effective_multi_label,
                )
            else:
                # Fall back to standard GLiNER inference path
                batch_entities = self._model.inference(
                    texts,
                    labels,
                    threshold=effective_threshold,
                    flat_ner=effective_flat_ner,
                    multi_label=effective_multi_label,
                )

        # Convert to SIE Entity format (same as GLiNERAdapter)
        all_entities: list[list[Entity]] = []
        for entities in batch_entities:
            entity_results: list[Entity] = []
            for entity in entities:
                entity_results.append(
                    Entity(
                        text=entity["text"],
                        label=entity["label"],
                        score=float(entity["score"]),
                        start=entity["start"],
                        end=entity["end"],
                    )
                )
            all_entities.append(entity_results)

        return ExtractOutput(entities=all_entities)

    def _predict_with_cached_embeds(
        self,
        texts: list[str],
        labels: list[str],
        *,
        threshold: float,
        flat_ner: bool,
        multi_label: bool,
    ) -> list[list[dict[str, Any]]]:
        """Run prediction using pre-computed label embeddings.

        Caches label embeddings keyed by the label set. When a cache hit
        occurs, the label encoder is skipped entirely — only the text encoder
        and span decoder run.

        Args:
            texts: Batch of text strings.
            labels: Entity type labels.
            threshold: Confidence threshold.
            flat_ner: Non-overlapping entity mode.
            multi_label: Multi-label mode.

        Returns:
            List of entity dicts per text (same format as ``GLiNER.inference``).
        """
        cache_key = frozenset(labels)

        with self._cache_lock:
            if cache_key in self._label_cache:
                # Move to end for LRU ordering
                self._label_cache.move_to_end(cache_key)
                label_embeds = self._label_cache[cache_key]
            else:
                # Compute and cache label embeddings (runs model forward pass
                # inside the lock — acceptable because the lock is only
                # contended during concurrent adapter calls, which the current
                # single-worker executor prevents).
                label_embeds = self._model.encode_labels(labels, batch_size=8)
                self._label_cache[cache_key] = label_embeds

                # Evict oldest if cache exceeds max size
                while len(self._label_cache) > _LABEL_CACHE_MAX_SIZE:
                    evicted_key, _ = self._label_cache.popitem(last=False)
                    logger.debug("Evicted label cache entry: %s", evicted_key)

        return self._model.batch_predict_with_embeds(
            texts,
            label_embeds,
            labels,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="GLiNER-bi adapter"))
        return item.text
