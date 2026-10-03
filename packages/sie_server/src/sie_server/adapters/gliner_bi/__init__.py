import logging
import math
import threading
from collections import OrderedDict
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._word_window import (
    MAX_DOCUMENT_WINDOWS,
    bound_gliner_words,
    gliner_windows,
    merge_window_spans,
    plan_forwards,
    window_item_counts,
    window_rows,
)
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Entity, ErrorCode

logger = logging.getLogger(__name__)

_ERR_REQUIRES_LABELS = "GLiNER-bi requires labels parameter for extraction"
_ERR_NO_RELATIONS = "GLiNER bi-encoder models do not extract relations; options.relation_labels is not supported"
_ERR_TOO_MANY_WINDOWS = (
    f"GLiNER-bi reads a document in at most {MAX_DOCUMENT_WINDOWS} windows of words; "
    "split this document into shorter items"
)

# Maximum number of distinct label-set embeddings to cache.
_LABEL_CACHE_MAX_SIZE = 64
# Rows gliner's inference (and the meter, which mirrors it) puts in one forward pass.
_GLINER_BATCH_SIZE = 8


def _validate_threshold(value: Any) -> None:
    message = "GLiNER-bi threshold must be a finite number between 0 and 1"
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidInputError(message)
    try:
        threshold = float(value)
    except OverflowError as exc:
        raise InvalidInputError(message) from exc
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise InvalidInputError(message)


class GLiNERBiAdapter(BaseAdapter):
    """Adapter for GLiNER bi-encoder models with pre-computed label embedding caching.

    Uses the standard ``gliner`` package but with bi-encoder architecture.
    The key performance feature is that label embeddings can be pre-computed
    and cached via ``encode_labels()`` + ``batch_predict_with_embeds()``,
    giving near-constant inference time regardless of label count.

    A document longer than the model's word window is read whole, as
    overlapping windows, as the GLiNER adapter reads it (see
    ``_word_window.document_windows``).

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
        revision: str | None = None,
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
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts.
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
        self._revision = revision

        self._model: Any = None
        self._device: str | None = None
        # True when the text encoder's attention memory grows with the square of a row.
        self._quadratic_attention = False
        # LRU cache: labels in request order -> pre-computed label embeddings.
        # The embeddings are positional, so the key must keep the order.
        # Protected by _cache_lock for thread safety (defensive — the
        # current ModelWorker uses a single-worker executor, but this
        # guards against future architectural changes).
        self._label_cache: OrderedDict[tuple[str, ...], Any] = OrderedDict()
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
        if self._revision is not None:
            load_kwargs["revision"] = self._revision

        self._model = GLiNER.from_pretrained(
            self._model_name_or_path,
            **load_kwargs,
        )

        if device == "cpu":
            self._model = self._model.to(device)
        else:
            self._model = self._model.to(device, dtype=dtype)
        # gliner's max_len counts words, whatever their subwords: read at most a
        # bounded number of subwords too, with a long word in pieces.
        self._quadratic_attention = bound_gliner_words(self._model)

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
            raise InvalidInputError(_ERR_REQUIRES_LABELS)

        opts = options or {}
        if opts.get("relation_labels"):
            raise InvalidInputError(_ERR_NO_RELATIONS)
        _validate_threshold(opts.get("threshold", self._threshold))

        texts = [self._extract_text(item) for item in items]

        effective_threshold = opts.get("threshold", self._threshold)
        effective_flat_ner = opts.get("flat_ner", self._flat_ner)
        effective_multi_label = opts.get("multi_label", self._multi_label)
        use_precompute = opts.get("precompute_labels", self._precompute_labels)
        # A document longer than the model's word window is read as several
        # overlapping windows, each a row of its own.
        plans = gliner_windows(self._model, texts)
        rows, owners, overlaps = window_rows(texts, plans)
        # A bi-encoder's row is the document alone, so its metered tokens are its row tokens.
        row_counts = self._row_token_counts(rows, labels, overlaps)
        row_tokens = (
            self._row_token_counts(rows, labels) if any(overlap is not None for overlap in overlaps) else row_counts
        )
        input_token_counts = window_item_counts(row_counts, owners, len(texts))

        def predict(batch: list[str]) -> list[Any]:
            if use_precompute:
                return self._predict_with_cached_embeds(
                    batch,
                    labels,
                    threshold=effective_threshold,
                    flat_ner=effective_flat_ner,
                    multi_label=effective_multi_label,
                )
            # Fall back to standard GLiNER inference path
            return self._model.inference(
                batch,
                labels,
                threshold=effective_threshold,
                flat_ner=effective_flat_ner,
                multi_label=effective_multi_label,
            )

        groups = None
        if row_tokens is not None and self._quadratic_attention and rows:
            groups = plan_forwards(row_tokens, rows_per_pass=_GLINER_BATCH_SIZE)
        with torch.inference_mode():
            if not rows:
                row_entities: list[Any] = []
            elif groups is None:
                row_entities = predict(rows)
            else:
                row_entities = [[] for _ in rows]
                for group in groups:
                    group_entities = predict([rows[index] for index in group])
                    if len(group_entities) != len(group):
                        raise ValueError("GLiNER-bi returned predictions for a different number of items")
                    for index, entities in zip(group, group_entities, strict=True):
                        row_entities[index] = entities

        item_rows: list[list[int]] = [[] for _ in texts]
        for position, owner in enumerate(owners):
            item_rows[owner].append(position)
        batch_entities = [
            []
            if windows is None
            else merge_window_spans(
                windows,
                [row_entities[position] for position in positions],
                text,
                flat_ner=bool(effective_flat_ner),
                multi_label=bool(effective_multi_label),
            )
            for text, windows, positions in zip(texts, plans, item_rows, strict=True)
        ]

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

        errors = None
        if any(windows is None for windows in plans):
            errors = [
                None
                if windows is not None
                else ExtractItemError(code=ErrorCode.INPUT_TOO_LONG.value, message=_ERR_TOO_MANY_WINDOWS)
                for windows in plans
            ]
        return ExtractOutput(entities=all_entities, errors=errors, input_token_counts=input_token_counts)

    def _doc_input_token_counts(self, texts: list[str], labels: list[str]) -> list[int] | None:
        """Count the document tokens the bi-encoder encodes of each text, over all its windows."""
        plans = gliner_windows(self._model, texts)
        rows, owners, overlaps = window_rows(texts, plans)
        return window_item_counts(self._row_token_counts(rows, labels, overlaps), owners, len(texts))

    def _row_token_counts(
        self, texts: list[str], labels: list[str], overlaps: list[int | None] | None = None
    ) -> list[int] | None:
        """Count the document tokens the bi-encoder actually encodes, per row.

        A bi-encoder encodes labels separately, so its text input is the
        document alone. Delegate word splitting (the bounded splitter installed
        at load) and max-length word truncation
        to the pinned GLiNER processor, then count the attended tokens of the
        retained window, special tokens included, as the GLiNER adapter does.
        Batches match GLiNER inference's default batch size. GLiNER skips
        whitespace-only documents without encoding them, so they count zero.

        ``overlaps[i]`` is None for a row holding the start of a document, or
        the number of words row ``i`` shares with the window before it; such
        a row counts neither its special tokens nor those words' subwords, so
        a document read in several windows counts each of its tokens once.
        """
        processor = getattr(self._model, "data_processor", None)
        prepare_inputs = getattr(self._model, "prepare_inputs", None)
        prepare_base_input = getattr(self._model, "prepare_base_input", None)
        if processor is None or prepare_inputs is None or prepare_base_input is None:
            return None
        encoded_positions = [index for index, text in enumerate(texts) if text.strip()]
        counts = [0] * len(texts)
        if not encoded_positions:
            return counts
        try:
            split_texts, _, _ = prepare_inputs([texts[index] for index in encoded_positions])
            raw_items = prepare_base_input(split_texts)
            encoded_counts: list[int] = []
            for start in range(0, len(raw_items), 8):
                raw_batch = processor.collate_raw_batch(raw_items[start : start + 8], entity_types=labels)
                encoded = processor.tokenize_inputs(raw_batch["tokens"])
                for batch_index, mask in enumerate(encoded["attention_mask"].tolist()):
                    overlap = None if overlaps is None else overlaps[encoded_positions[start + batch_index]]
                    if overlap is None:
                        encoded_counts.append(int(sum(mask)))
                        continue
                    word_ids = encoded.word_ids(batch_index)
                    encoded_counts.append(
                        sum(
                            1
                            for attended, word_id in zip(mask, word_ids, strict=True)
                            if attended and word_id is not None and word_id >= overlap
                        )
                    )
        except Exception:  # noqa: BLE001 -- metering must never fail an extraction
            return None
        if len(encoded_counts) != len(encoded_positions):
            return None
        for index, count in zip(encoded_positions, encoded_counts, strict=True):
            counts[index] = count
        return counts

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

        Caches label embeddings keyed by the ordered label list, because the
        embeddings are matched to labels by position. When a cache hit
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
        cache_key = tuple(labels)

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
            raise InvalidInputError(ERR_REQUIRES_TEXT.format(adapter_name="GLiNER-bi adapter"))
        return item.text
