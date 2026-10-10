from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Iterable
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar

import torch
from huggingface_hub import snapshot_download

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._prompt_limit import DEFAULT_MAX_SCHEMA_PROMPT_TOKENS, PromptLimit, check_label_chars
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._word_window import (
    MAX_WORD_CHARS,
    DocumentWindow,
    SubwordCounter,
    Window,
    WindowedSplitter,
    merge_window_spans,
    plan_forwards,
    quadratic_attention,
    subword_budget,
)
from sie_server.adapters.gliner2.words import linear_equivalent
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Classification, Entity, Relation

_ERR_REQUIRES_LABELS = "GLiNER2 requires labels parameter for extraction"
_STRUCTURE_NAME = "_sie_root"
_STRUCTURE_DELIMITERS = ("::", "|", "[", "]")
# Metering tokenizes a prefix of this many characters per token of the window
# (then 4x more, up to the whole text) and stops once the prefix alone fills
# the window.
_METER_CHARS_PER_TOKEN = 16
_METER_MARGIN_TOKENS = 64
# A space after a non-space character, matched in reversed text: the tokenizers
# of these checkpoints split pre-tokens at spaces, after normalizing whitespace.
_SPACE_AFTER_TEXT = re.compile(r" (?=\S)")
# Words a document is read up to when the model config sets no max_seq_length.
_DEFAULT_MAX_WORDS = 512
# Subword tokens a document may take per word it reads. The GLiNER2 checkpoints
# read English: prose, code, CSV and JSON logs run at up to 2.4 subwords per word,
# other European languages at up to 3.2. GLiNER2 reads a URL as one word, so a
# document made mostly of long links is read only up to the budget.
_SUBWORDS_PER_WORD = 4
# gliner2 keeps the tokens of every word it tokenizes in an LRU of 50,000
# entries. Only words of at most _CACHED_WORD_CHARS characters are kept here, so
# the cache stays within a few tens of MB; a longer word is tokenized each time.
_WORD_CACHE_SIZE = 16384
_CACHED_WORD_CHARS = 32
# Rows gliner2 puts in one forward pass when not told otherwise.
_PACKAGE_BATCH_SIZE = 8
# Task prompt tokens per label or task name beyond its own tokens: gliner2
# marks each entity type or class with one token, and builds one structure of
# about ten tokens around each relation type.
_PROMPT_TOKENS_PER_ENTRY = 2
_PROMPT_TOKENS_PER_RELATION = 12
_SENTENCE_END = (".", "!", "?")
# The prompt's own markers, [SEP_TEXT], and the tokenizer's specials.
_ROW_OVERHEAD_TOKENS = 8
_DEBERTA_CONFIG = {"model_type": "deberta-v2"}
# Classification reads a text longer than one window as overlapping windows of
# the same size (see ``_classification_windows``). A window starts this many
# words before the end of the one before it, so a sentence cut by one window's
# edge is read whole by the next.
_CLASSIFY_OVERLAP_WORDS = 64
# Most windows one text is classified in: at 512 words a window, 57,408 words.
# A longer text is rejected with INVALID_INPUT rather than classified in part,
# since a verdict on part of a text reads as a verdict on all of it.
_MAX_CLASSIFY_WINDOWS = 128
# Most windows one text is extracted in. A longer text is not read in part
# either: that item comes back with the per-item ``INPUT_TOO_LONG`` error and
# no extraction, and the other items of the request are unaffected.
_MAX_EXTRACT_WINDOWS = 128
# Characters of the rest of a text read to find its next window, doubled until
# the window ends at least MAX_WORD_CHARS before them (or the text ends), so a
# window is found without copying the rest of a long text each time.
_WINDOW_SLICE_CHARS = 16384
_NON_SPACE = re.compile(r"\S")


class GLiNER2Adapter(BaseAdapter):
    """Adapter for GLiNER2 zero-shot extraction and classification models.

    GLiNER2 uses the separate ``gliner2`` pip package (NOT ``gliner``). It
    performs named entity recognition, relation extraction, flat structured
    extraction, and schema-conditioned classification.

    Key API differences from GLiNER v1:
    - ``GLiNER2.from_pretrained(name, map_location=device, quantize=True)``
    - ``extract_entities()`` returns nested-by-label dict, not a flat list
    - Batch methods cover entities, relations, structured data, and classification
    - Classification uses ``classify_text()`` / ``batch_classify_text()``

    A request's labels, class labels, relation types or schema fields (field
    names, descriptions and choices), which gliner2 encodes with every document
    and does not bill, may take at most ``max_prompt_tokens`` tokens (default
    2048), and each label, task name, field name or choice at most 128
    characters; a longer prompt is rejected with ``INVALID_INPUT``.

    gliner2 reads at most ``max_seq_length`` words of a document. Every task
    reads all of a longer text, as overlapping windows of that many words: the
    windows' classifications are pooled into one (see
    ``pool_window_classifications``), and the windows' extractions are merged
    into one result at document offsets. A text that fits one window is read
    exactly as before, in the same call. A text that takes more than
    ``_MAX_EXTRACT_WINDOWS`` windows of extraction comes back with a per-item
    ``INPUT_TOO_LONG`` error rather than being read in part.

    Reference models:
    - fastino/gliner2-base-v1
    - fastino/gliner2-large-v1

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
        classification_task: str | None = None,
        default_labels: list[str] | None = None,
        multi_label: bool = False,
        positive_label: str | None = None,
        max_seq_length: int | None = None,
        max_prompt_tokens: int = DEFAULT_MAX_SCHEMA_PROMPT_TOKENS,
        compute_precision: ComputePrecision = "float16",
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            threshold: Minimum confidence score for extraction/classification (0-1).
            classification_task: Optional schema task name. When set, ``extract``
                returns classifications instead of entities. The task may be
                overridden per request through runtime options.
            default_labels: Optional labels used when a request omits ``labels``.
                Request-provided labels take precedence.
            multi_label: Whether the configured classification task may return
                multiple labels.
            positive_label: Optional label of a single-label classification
                that a text takes when any window of it takes that label, such
                as ``unsafe`` for a guard (see ``pool_window_classifications``).
                It applies only when the request's labels include it, and may
                be overridden per request through runtime options.
            max_seq_length: Maximum document and schema input length.
            max_prompt_tokens: Most tokens a request's labels, class labels,
                relation types or schema fields may take in the task prompt
                encoded with each document (see ``_prompt_limit``).
            compute_precision: Compute precision for inference.
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._threshold = threshold
        self._classification_task = classification_task
        self._default_labels = self._validate_labels(default_labels) if default_labels is not None else None
        self._multi_label = multi_label
        self._positive_label = self._validate_positive_label(positive_label, self._default_labels)
        self._max_seq_length = max_seq_length
        self._prompt_limit = PromptLimit("GLiNER2", max_prompt_tokens)
        self._compute_precision = compute_precision
        self._revision = revision

        self._model: Any = None
        # How the loaded gliner2 splits words (see ``_model_text``); None until loaded.
        self._lower_text_first: bool | None = None
        # The splitter gliner2 reads words with, bounded to the window it reads (see ``_window``).
        self._word_splitter: WindowedSplitter | None = None
        self._count_subwords: Callable[[list[str]], list[int]] | None = None
        # Whether the encoder's attention memory grows with the square of a row (see ``_run_planned``).
        self._quadratic_attention = True
        self._device: str | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu", "mps").
        """
        from gliner2 import GLiNER2  # ty:ignore[unresolved-import]

        self._device = device

        # GLiNER2 does not forward arbitrary kwargs to Hugging Face downloads.
        # Resolve a pinned snapshot ourselves so every model file comes from the
        # configured immutable revision. Pre-staged weights remain local paths.
        model_path = self._model_name_or_path
        if self._revision is not None and not Path(model_path).is_dir():
            model_path = snapshot_download(repo_id=model_path, revision=self._revision)

        use_quantize = device != "cpu" and self._compute_precision == "float16"
        model = GLiNER2.from_pretrained(
            model_path,
            map_location=device,
            quantize=use_quantize,
        )
        encoder = getattr(model, "encoder", None)
        self._use_linear_word_splitter(model.processor, getattr(encoder, "config", None))
        self._model = model

    def _use_linear_word_splitter(self, processor: Any, encoder_config: Any = None) -> None:
        """Split words with a linear-time equivalent of gliner2's splitter, bounded to the window gliner2 reads.

        gliner2's word-splitting regex takes time quadratic in the length of a
        run of e-mail address characters (``"...."``, ``"a.a.a."``), on the
        thread that serves every request. gliner2 also reads ``max_len`` words
        whatever their length, so the splitter yields only the window of
        ``_word_window.read_window``: a word longer than 256 characters in
        pieces, and at most ``_SUBWORDS_PER_WORD`` subword tokens per word of
        the window.

        Raises:
            RuntimeError: gliner2's splitter is not the one this adapter has an
                equivalent for.
        """
        splitter = linear_equivalent(processor.word_splitter)
        if splitter is None:
            raise RuntimeError(
                f"GLiNER2 has no linear-time equivalent of {type(processor.word_splitter).__name__}; "
                "gliner2's word splitter changed"
            )
        # An unknown encoder is taken to be DeBERTa, the encoder of every GLiNER2 checkpoint.
        config = encoder_config if encoder_config is not None else _DEBERTA_CONFIG
        tokenize = self._bounded_tokenization(processor)
        count_subwords = SubwordCounter(lambda words: [len(tokenize(word)) for word in words])
        max_words = self._max_seq_length or _DEFAULT_MAX_WORDS
        windowed = WindowedSplitter(
            splitter,
            max_words=max_words,
            max_subwords=subword_budget(max_words, config, per_word=_SUBWORDS_PER_WORD),
            count_subwords=count_subwords,
        )
        processor.word_splitter = windowed
        self._lower_text_first = splitter.lower_text_first
        self._word_splitter = windowed
        self._count_subwords = count_subwords
        self._quadratic_attention = quadratic_attention(config)

    @staticmethod
    def _bounded_tokenization(processor: Any) -> Callable[[str], list[str]]:
        """Tokenize words as gliner2 does, keeping only short words' tokens.

        gliner2 caches the tokens of every word it tokenizes, however long, for
        the adapter's lifetime; its cache is replaced by one that keeps words
        of at most ``_CACHED_WORD_CHARS`` characters.
        """
        tokenize = processor.tokenizer.tokenize
        cached = lru_cache(maxsize=_WORD_CACHE_SIZE)(tokenize)

        def tokenize_word(word: str) -> list[str]:
            return cached(word) if len(word) <= _CACHED_WORD_CHARS else tokenize(word)

        if hasattr(processor, "_tokenize_cached"):
            processor._tokenize_cached = tokenize_word
        return tokenize_word

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
        """Extract entities, relations, classifications, or flat structured data."""
        self._check_loaded()
        texts = [self._extract_text(item) for item in items]
        reads = [self._read(text) for text in texts]
        opts = options or {}
        effective_threshold = self._validate_threshold(opts.get("threshold", self._threshold))
        classification_task = opts.get("classification_task", self._classification_task)
        multi_label = opts.get("multi_label", self._multi_label)
        if not isinstance(multi_label, bool):
            raise InvalidInputError("GLiNER2 multi_label must be boolean")
        input_token_counts = self._doc_input_token_counts(texts)

        if output_schema is not None:
            if labels:
                raise InvalidInputError("GLiNER2 structured extraction does not accept labels")
            if classification_task is not None:
                raise InvalidInputError("GLiNER2 structured extraction does not accept classification_task")
            structures = self._json_schema_to_structures(output_schema)
            specs = [spec for fields in structures.values() for spec in fields]
            # A field's choices are listed in its structure, and each again in a prefix before the document.
            choices = [
                choice for definition in output_schema["properties"].values() for choice in definition.get("enum") or []
            ]
            check_label_chars("GLiNER2", "output_schema property names", output_schema["properties"])
            check_label_chars("GLiNER2", "output_schema enum values", choices)
            prompt = self._prompt_tokens(specs, key=("json", tuple(specs)), extra=len(choices))
            with torch.inference_mode():
                raw_results, errors, input_token_counts = self._run_whole(
                    texts,
                    reads,
                    lambda batch: self._model.batch_extract_json(
                        batch,
                        structures,
                        batch_size=len(batch),
                        threshold=effective_threshold,
                        include_confidence=False,
                        include_spans=False,
                        max_len=self._max_seq_length,
                    ),
                    input_token_counts=input_token_counts,
                    prompt=prompt,
                    rows_per_pass=len(texts),
                    merge=lambda _, __, ___, results: self._merge_structured(results),
                )
            return ExtractOutput(
                entities=[[] for _ in texts],
                data=[
                    {}
                    if errors is not None and errors[index] is not None
                    else self._flatten_structured_result(result, output_schema=output_schema)
                    for index, result in enumerate(raw_results)
                ],
                errors=errors,
                input_token_counts=input_token_counts,
            )

        effective_labels = labels if labels is not None else self._default_labels
        normalized_labels = self._validate_labels(effective_labels)
        relation_entities = [self._extract_relation_entities(item) for item in items]
        if any(entities is not None for entities in relation_entities):
            if not all(entities for entities in relation_entities):
                raise InvalidInputError(
                    "GLiNER2 relation extraction requires non-empty entities in every item metadata"
                )
            normalized_entities = [
                self._normalize_input_entities(item, entities or []) for item, entities in zip(items, relation_entities)
            ]
            check_label_chars("GLiNER2", "labels", normalized_labels)
            prompt = self._prompt_tokens(
                normalized_labels, per_entry=_PROMPT_TOKENS_PER_RELATION, key=("relations", tuple(normalized_labels))
            )
            with torch.inference_mode():
                raw_results, errors, input_token_counts = self._run_whole(
                    texts,
                    reads,
                    lambda batch: self._model.batch_extract_relations(
                        batch,
                        normalized_labels,
                        batch_size=len(batch),
                        threshold=effective_threshold,
                        include_confidence=True,
                        include_spans=True,
                        max_len=self._max_seq_length,
                    ),
                    input_token_counts=input_token_counts,
                    prompt=prompt,
                    rows_per_pass=len(texts),
                    merge=lambda index, text, plan, results: self._merge_relations(
                        text, plan, results, normalized_entities[index]
                    ),
                )
            return ExtractOutput(
                entities=[
                    [] if errors is not None and errors[index] is not None else entities
                    for index, entities in enumerate(normalized_entities)
                ],
                errors=errors,
                relations=[
                    self._flatten_relations(result, entities=entities)
                    for result, entities in zip(raw_results, normalized_entities)
                ],
                input_token_counts=input_token_counts,
            )

        if classification_task is not None:
            if not isinstance(classification_task, str) or not classification_task.strip():
                raise InvalidInputError("GLiNER2 classification_task must be a non-empty string")
            check_label_chars("GLiNER2", "classification_task", [classification_task])
            check_label_chars("GLiNER2", "labels", normalized_labels)
            positive_label = self._effective_positive_label(opts, normalized_labels)
            prompt = self._prompt_tokens(
                [classification_task, *normalized_labels],
                key=("classification", classification_task, tuple(normalized_labels)),
            )
            item_windows = [self._classification_windows(text, read) for text, read in zip(texts, reads, strict=True)]
            if any(len(one) > 1 for one in item_windows):
                input_token_counts = self._windowed_input_token_counts(texts, item_windows)
            return self._classify(
                item_windows,
                normalized_labels,
                task=classification_task,
                multi_label=multi_label,
                positive_label=positive_label,
                threshold=effective_threshold,
                input_token_counts=input_token_counts,
                prompt=prompt,
            )

        def extract_entities(batch: list[str]) -> list[Any]:
            if len(batch) == 1:
                return [
                    self._model.extract_entities(
                        batch[0],
                        normalized_labels,
                        threshold=effective_threshold,
                        include_confidence=True,
                        include_spans=True,
                        max_len=self._max_seq_length,
                    )
                ]
            return self._model.batch_extract_entities(
                batch,
                normalized_labels,
                threshold=effective_threshold,
                include_confidence=True,
                include_spans=True,
                max_len=self._max_seq_length,
            )

        check_label_chars("GLiNER2", "labels", normalized_labels)
        prompt = self._prompt_tokens(normalized_labels, key=("entities", tuple(normalized_labels)))
        with torch.inference_mode():
            raw_results, errors, input_token_counts = self._run_whole(
                texts,
                reads,
                extract_entities,
                input_token_counts=input_token_counts,
                prompt=prompt,
                rows_per_pass=_PACKAGE_BATCH_SIZE,
                merge=lambda _, text, plan, results: self._merge_entities(text, plan, results),
            )

        all_entities = [self._flatten_entities(result, text=text) for text, result in zip(texts, raw_results)]
        return ExtractOutput(entities=all_entities, errors=errors, input_token_counts=input_token_counts)

    def _run_whole(
        self,
        texts: list[str],
        reads: list[tuple[str, int | None, Window | None]],
        run: Callable[[list[str]], list[Any]],
        *,
        prompt: int | None,
        input_token_counts: list[int] | None,
        rows_per_pass: int,
        merge: Callable[[int, str, tuple[list[int], list[int], list[tuple[str, int | None]]], list[Any]], Any],
    ) -> tuple[list[Any], list[ExtractItemError | None] | None, list[int] | None]:
        """Run extraction over every window of each item's text and merge the windows' results.

        Every window of every item runs in the same planned batch as one row
        (see ``_run_planned``), and ``merge`` builds one item's result from its
        windows' (``_merge_entities``, ``_merge_relations`` or
        ``_merge_structured``). A text that fits the model's window is read as
        that one window, exactly as before. A text taking more than
        ``_MAX_EXTRACT_WINDOWS`` windows, or one whose windows' spans cannot be
        moved back to it, comes back with the per-item ``INPUT_TOO_LONG``
        error, is not inferred and is metered at zero; the other items of the
        request are unaffected.
        """
        plans: list[tuple[list[int], list[int], list[tuple[str, int | None]]] | None] = []
        errors: list[ExtractItemError | None] = []
        for text, read in zip(texts, reads, strict=True):
            plan = self._windows(text, read, limit=_MAX_EXTRACT_WINDOWS)
            if plan is None:
                errors.append(
                    ExtractItemError(
                        code="INPUT_TOO_LONG",
                        message=f"GLiNER2 reads a text in at most {_MAX_EXTRACT_WINDOWS} windows of "
                        f"{self._max_seq_length or _DEFAULT_MAX_WORDS} words. Split the text into smaller items.",
                    )
                )
                plans.append(None)
                continue
            source = text.lower() if self._lower_text_first else text
            if len(plan[2]) > 1 and len(source) != len(text):
                errors.append(
                    ExtractItemError(
                        code="INPUT_TOO_LONG",
                        message="GLiNER2 lowercases this text before reading it, and lowercasing changes its "
                        "length, so the spans of its windows cannot be moved back to the text. "
                        "Split the text into smaller items.",
                    )
                )
                plans.append(None)
                continue
            errors.append(None)
            plans.append(plan)
        flat = [window for plan in plans if plan is not None for window in plan[2]]
        results: list[Any] = [{} for _ in texts]
        if flat:
            raw = self._run_planned(
                [model_text for model_text, _ in flat],
                self._row_tokens(flat, prompt),
                run,
                rows_per_pass=1 if len(flat) == 1 else rows_per_pass,
            )
            if len(raw) != len(flat):
                raise ValueError("GLiNER2 returned results for a different number of items")
            offset = 0
            for index, plan in enumerate(plans):
                if plan is None:
                    continue
                window_results = raw[offset : offset + len(plan[2])]
                offset += len(plan[2])
                results[index] = (
                    merge(index, texts[index], plan, window_results) if len(plan[2]) > 1 else window_results[0]
                )
        if any(plan is not None and len(plan[2]) > 1 for plan in plans):
            counts = self._windowed_input_token_counts(texts, [plan[2] if plan is not None else [] for plan in plans])
        else:
            counts = (
                [count if error is None else 0 for count, error in zip(input_token_counts, errors, strict=True)]
                if input_token_counts is not None
                else ([0] * len(texts) if all(plan is None for plan in plans) else None)
            )
        return results, errors if any(error is not None for error in errors) else None, counts

    def _merge_entities(
        self,
        text: str,
        plan: tuple[list[int], list[int], list[tuple[str, int | None]]],
        results: list[Any],
    ) -> dict[str, Any]:
        """One text's entities from the entities found in each of its windows.

        Each window's spans move back to the text by the window's start, and
        the windows' spans merge as the GLiNER adapters merge theirs (see
        ``_word_window.merge_window_spans``): a span cut at a window's edge
        gives way when the window next to it read that region whole and found
        an overlapping span of the same label, the same span keeps its highest
        score, and overlapping spans resolve highest score first. A span's text
        is re-read from the text, so it keeps its case whatever the windows
        were read from.
        """
        starts, read_ends, windows = plan
        found: list[list[dict[str, Any]]] = []
        for result in results:
            spans: list[dict[str, Any]] = []
            for label, label_spans in (result.get("entities") or {}).items():
                if not isinstance(label_spans, list):
                    continue
                for span in label_spans:
                    if isinstance(span, dict):
                        spans.append(
                            {
                                "start": span.get("start"),
                                "end": span.get("end"),
                                "label": label,
                                "score": span.get("confidence", 0.0),
                                "text": span.get("text", ""),
                            }
                        )
            found.append(spans)
        document = [
            DocumentWindow(start, start + len(model_text), read_end, None)
            for start, read_end, (model_text, _) in zip(starts, read_ends, windows, strict=True)
        ]
        merged = merge_window_spans(document, found, text, flat_ner=False, multi_label=True)
        entities: dict[str, Any] = {}
        for span in merged:
            entities.setdefault(span["label"], []).append(
                {"text": span["text"], "start": span["start"], "end": span["end"], "confidence": span["score"]}
            )
        return {"entities": entities}

    def _merge_relations(
        self,
        text: str,
        plan: tuple[list[int], list[int], list[tuple[str, int | None]]],
        results: list[Any],
        entities: list[Entity],
    ) -> dict[str, Any]:
        """One text's relations from the relations found in each of its windows.

        A relation found by several windows is kept once per (head, tail,
        relation) at its highest score, and a relation's head and tail are
        moved back to the text: their spans move by the window's start and the
        text is re-read there, or an endpoint's text matches an entity's text
        case-insensitively when it carries no span. A relation whose head or
        tail is none of ``entities`` is dropped, as in one window. A relation
        whose head and tail no single window holds together is not found.
        """
        starts = plan[0]
        originals = {entity["text"] for entity in entities}
        by_lower: dict[str, str] = {}
        for original_text in originals:
            by_lower.setdefault(original_text.lower(), original_text)

        def original(endpoint: Any, offset: int) -> str | None:
            if not isinstance(endpoint, dict):
                return None
            found = endpoint.get("text")
            if not isinstance(found, str):
                return None
            if found in originals:
                return found
            start, end = endpoint.get("start"), endpoint.get("end")
            if (
                isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
            ):
                moved_start, moved_end = offset + start, offset + end
                if 0 <= moved_start < moved_end <= len(text) and text[moved_start:moved_end].lower() == found.lower():
                    return text[moved_start:moved_end]
            return by_lower.get(found.lower())

        best: dict[tuple[str, str, str], tuple[float, dict[str, Any]]] = {}
        for offset, result in zip(starts, results, strict=True):
            by_type = result.get("relation_extraction") or {}
            if not isinstance(by_type, dict):
                continue
            for relation_type, candidates in by_type.items():
                if not isinstance(relation_type, str) or not isinstance(candidates, list):
                    continue
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    head, tail = candidate.get("head"), candidate.get("tail")
                    resolved_head, resolved_tail = original(head, offset), original(tail, offset)
                    if resolved_head is None or resolved_tail is None:
                        continue
                    head_confidence = head.get("confidence", 0.0) if isinstance(head, dict) else 0.0
                    tail_confidence = tail.get("confidence", 0.0) if isinstance(tail, dict) else 0.0
                    score = min(head_confidence, tail_confidence)
                    key = (resolved_head, resolved_tail, relation_type)
                    if key not in best or score > best[key][0]:
                        best[key] = (
                            score,
                            {
                                "head": {"text": resolved_head, "confidence": head_confidence},
                                "tail": {"text": resolved_tail, "confidence": tail_confidence},
                            },
                        )
        extraction: dict[str, Any] = {}
        for (_, _, relation_type), (_, candidate) in best.items():
            extraction.setdefault(relation_type, []).append(candidate)
        return {"relation_extraction": extraction}

    def _merge_structured(self, results: list[Any]) -> dict[str, Any]:
        """One text's structured data from the data extracted from each of its windows.

        A string field takes the first window that found a value for it, and a
        list field takes every window's values in window order, once each. A
        field no window found stays missing.
        """
        merged: dict[str, Any] = {}
        for result in results:
            values = result.get(_STRUCTURE_NAME) or []
            if not isinstance(values, list):
                continue
            for record in values:
                if not isinstance(record, dict):
                    continue
                for field, value in record.items():
                    if isinstance(value, list):
                        kept = merged.setdefault(field, [])
                        for one in value:
                            if one not in kept:
                                kept.append(one)
                    elif value is not None and value != "" and merged.get(field) is None:
                        merged[field] = value
        return {_STRUCTURE_NAME: [merged]}

    def _classify(
        self,
        item_windows: list[list[tuple[str, int | None]]],
        labels: list[str],
        *,
        task: str,
        multi_label: bool,
        positive_label: str | None,
        threshold: float,
        input_token_counts: list[int] | None,
        prompt: int | None,
    ) -> ExtractOutput:
        """Run one GLiNER2 classification schema over each item's windows and normalize its results.

        ``item_windows`` holds each item's windows as ``(model text, subwords)``
        (one for a text that fits the window gliner2 reads). Every window of
        every item runs in the same planned batch, and each item's window
        results are pooled into one (``pool_window_classifications``).
        """
        tasks = {
            task: {
                "labels": labels,
                "multi_label": multi_label,
                "cls_threshold": threshold,
            }
        }

        def classify(batch: list[str]) -> list[Any]:
            if len(batch) == 1:
                return [
                    self._model.classify_text(
                        batch[0],
                        tasks,
                        threshold=threshold,
                        include_confidence=True,
                        max_len=self._max_seq_length,
                    )
                ]
            return self._model.batch_classify_text(
                batch,
                tasks,
                threshold=threshold,
                include_confidence=True,
                max_len=self._max_seq_length,
            )

        flat = [window for windows in item_windows for window in windows]
        with torch.inference_mode():
            raw_results = self._run_planned(
                [model_text for model_text, _ in flat],
                self._row_tokens(flat, prompt),
                classify,
                rows_per_pass=1 if len(flat) == 1 else _PACKAGE_BATCH_SIZE,
            )
        if len(raw_results) != len(flat):
            raise ValueError("GLiNER2 returned results for a different number of items")

        all_classifications: list[list[Classification]] = []
        offset = 0
        for windows in item_windows:
            window_results = raw_results[offset : offset + len(windows)]
            offset += len(windows)
            if len(window_results) == 1:
                all_classifications.append(
                    self._flatten_classifications(window_results[0], task=task, threshold=threshold)
                )
                continue
            found = [
                [
                    (one["label"], one["score"])
                    for one in self._flatten_classifications(result, task=task, threshold=0.0)
                ]
                for result in window_results
            ]
            pooled = pool_window_classifications(
                found, labels=labels, multi_label=multi_label, positive_label=positive_label
            )
            classifications = [
                Classification(label=label, score=score) for label, score in pooled if score >= threshold
            ]
            classifications.sort(key=lambda classification: classification["score"], reverse=True)
            all_classifications.append(classifications)
        return ExtractOutput(
            entities=[[] for _ in item_windows],
            classifications=all_classifications,
            input_token_counts=input_token_counts,
        )

    def _prompt_tokens(
        self,
        entries: list[str],
        *,
        key: tuple[Any, ...],
        per_entry: int = _PROMPT_TOKENS_PER_ENTRY,
        extra: int = 0,
    ) -> int | None:
        """Estimated tokens of the task prompt gliner2 builds from ``entries``, checked against the limit.

        Each label, class label, relation type and schema field is counted
        with the tokens gliner2 adds around it, plus ``extra`` (a token per
        field choice, which gliner2 lists again before the document). None when words are
        not counted (no bounded splitter is installed); the prompt's
        characters are still checked.

        Raises:
            InvalidInputError: The prompt takes more than ``max_prompt_tokens``.
        """
        count = self._count_subwords
        strings = [entry for entry in entries if isinstance(entry, str)]

        def tokens() -> int:
            if count is None:
                return 0
            return sum(count(strings)) + per_entry * len(strings) + extra + _ROW_OVERHEAD_TOKENS

        prompt = self._prompt_limit.check(strings, tokens, (per_entry, extra, *key))
        return prompt if count is not None else None

    def _row_tokens(self, windows: list[tuple[str, int | None]], prompt: int | None) -> list[int] | None:
        """Estimated tokens of each item's encoder row: the task prompt, then the words it reads.

        None when the words were not counted (no bounded splitter is installed).
        """
        if prompt is None or any(subwords is None for _, subwords in windows):
            return None
        return [prompt + (subwords or 0) for _, subwords in windows]

    def _run_planned(
        self,
        texts: list[str],
        rows: list[int] | None,
        run: Callable[[list[str]], list[Any]],
        *,
        rows_per_pass: int,
    ) -> list[Any]:
        """``run(texts)``, split into several calls when one forward pass would hold too long a batch.

        gliner2 pads a pass to its longest row, and a DeBERTa encoder's
        attention memory grows with rows times the square of that length, so
        rows are grouped by length within ``_word_window.ATTENTION_BUDGET``
        (see ``plan_forwards``). A batch that fits runs exactly as before.
        """
        groups = plan_forwards(rows, rows_per_pass=rows_per_pass) if rows and self._quadratic_attention else None
        if groups is None:
            return list(run(texts))
        results: list[Any] = [None] * len(texts)
        for group in groups:
            group_results = list(run([texts[index] for index in group]))
            if len(group_results) != len(group):
                raise ValueError("GLiNER2 returned results for a different number of items")
            for index, result in zip(group, group_results, strict=True):
                results[index] = result
        return results

    @classmethod
    def _flatten_classifications(
        cls,
        result: dict[str, Any],
        *,
        task: str,
        threshold: float,
    ) -> list[Classification]:
        """Convert confidence-bearing GLiNER2 task output to SIE classifications."""
        task_result = result.get(task)
        if task_result is None:
            return []
        candidates = [task_result] if isinstance(task_result, dict) else task_result
        if not isinstance(candidates, list):
            raise ValueError("GLiNER2 returned malformed classifications")

        classifications: list[Classification] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise ValueError("GLiNER2 returned malformed classification")
            label = candidate.get("label")
            if not isinstance(label, str) or not label.strip() or "confidence" not in candidate:
                raise ValueError("GLiNER2 returned malformed classification")
            score = cls._validate_score(candidate["confidence"], "classification")
            if score >= threshold:
                classifications.append(Classification(label=label.strip(), score=score))
        classifications.sort(key=lambda classification: classification["score"], reverse=True)
        return classifications

    @classmethod
    def _flatten_entities(cls, result: dict[str, Any], *, text: str) -> list[Entity]:
        """Normalize confidence-bearing spans and verify character offsets."""
        entity_list: list[Entity] = []
        entities_dict = result.get("entities", {})
        if not isinstance(entities_dict, dict):
            raise ValueError("GLiNER2 returned malformed entities")
        for label_name, spans in entities_dict.items():
            if not isinstance(label_name, str) or not label_name.strip() or not isinstance(spans, list):
                raise ValueError("GLiNER2 returned malformed entities")
            for span in spans:
                if not isinstance(span, dict):
                    raise ValueError("GLiNER2 returned malformed entity span")
                start, end, span_text = cls._normalize_entity_span(
                    text,
                    span.get("start"),
                    span.get("end"),
                    span.get("text"),
                )
                score = cls._validate_score(span.get("confidence", 0.0), "entity")
                entity_list.append(Entity(text=span_text, label=label_name.strip(), score=score, start=start, end=end))
        entity_list.sort(key=lambda entity: entity.get("start") or 0)
        return entity_list

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise InvalidInputError(ERR_REQUIRES_TEXT.format(adapter_name="GLiNER2 adapter"))
        return item.text

    @staticmethod
    def _validate_threshold(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InvalidInputError("GLiNER2 threshold must be a finite number between 0 and 1")
        threshold = float(value)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise InvalidInputError("GLiNER2 threshold must be a finite number between 0 and 1")
        return threshold

    @staticmethod
    def _validate_score(value: object, output_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"GLiNER2 returned invalid {output_name} confidence")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"GLiNER2 returned invalid {output_name} confidence")
        return score

    @staticmethod
    def _normalize_entity_span(
        text: str,
        start_value: object,
        end_value: object,
        span_text_value: object,
    ) -> tuple[int, int, str]:
        """Return exact source offsets, clipping only verified boundary punctuation."""

        def invalid_offsets() -> ValueError:
            def safe_offset(value: object) -> str:
                if type(value) in {bool, float, int, type(None)}:
                    return repr(value)
                return "<redacted>"

            span_text_length = len(span_text_value) if isinstance(span_text_value, str) else None
            return ValueError(
                "GLiNER2 returned invalid character offsets "
                f"(start={safe_offset(start_value)} [{type(start_value).__name__}], "
                f"end={safe_offset(end_value)} [{type(end_value).__name__}], "
                f"text_length={len(text)}, span_text_length={span_text_length}, "
                f"span_text_type={type(span_text_value).__name__})"
            )

        if (
            not isinstance(start_value, int)
            or isinstance(start_value, bool)
            or not isinstance(end_value, int)
            or isinstance(end_value, bool)
            or not isinstance(span_text_value, str)
        ):
            raise invalid_offsets()

        start = start_value
        end = end_value
        span_text = span_text_value
        if 0 <= start < end <= len(text) and text[start:end] == span_text:
            return start, end, span_text

        if start < 0 or start >= len(text) or end <= start or len(span_text) != end - start or end <= len(text):
            raise invalid_offsets()
        source_prefix = text[start:]
        overflow = span_text[len(source_prefix) :]
        if (
            not overflow
            or span_text[: len(source_prefix)] != source_prefix
            or any(not char.isspace() and not unicodedata.category(char).startswith("P") for char in overflow)
        ):
            raise invalid_offsets()
        return start, len(text), source_prefix

    @staticmethod
    def _validate_labels(labels: list[str] | None) -> list[str]:
        if labels is None:
            raise InvalidInputError(_ERR_REQUIRES_LABELS)
        if not isinstance(labels, list):
            raise InvalidInputError("GLiNER2 labels must be a list")
        if not labels:
            raise InvalidInputError(_ERR_REQUIRES_LABELS)
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise InvalidInputError("GLiNER2 labels must be non-empty strings")
        normalized = [label.strip() for label in labels]
        if len(set(normalized)) != len(normalized):
            raise InvalidInputError("GLiNER2 labels must be unique")
        return normalized

    @staticmethod
    def _extract_relation_entities(item: Item) -> list[dict[str, Any]] | None:
        if item.metadata is None or "entities" not in item.metadata:
            return None
        entities = item.metadata["entities"]
        if not isinstance(entities, list):
            raise InvalidInputError("GLiNER2 item metadata.entities must be a list")
        return entities

    @classmethod
    def _normalize_input_entities(cls, item: Item, entities: list[dict[str, Any]]) -> list[Entity]:
        text = item.text or ""
        normalized: list[Entity] = []
        for entity in entities:
            if not isinstance(entity, dict):
                raise InvalidInputError("GLiNER2 relation entities must be objects")
            start = entity.get("start")
            end = entity.get("end")
            entity_text = entity.get("text")
            label = entity.get("label", "ENTITY")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or not isinstance(entity_text, str)
                or not isinstance(label, str)
                or not label.strip()
                or start < 0
                or end <= start
                or end > len(text)
                or text[start:end] != entity_text
            ):
                raise InvalidInputError("GLiNER2 relation entities require valid character offsets")
            try:
                score = cls._validate_score(entity.get("score", 1.0), "relation entity")
            except ValueError as error:
                raise InvalidInputError(str(error)) from error
            normalized.append(Entity(text=entity_text, label=label.strip(), score=score, start=start, end=end))
        return normalized

    @classmethod
    def _flatten_relations(
        cls,
        result: dict[str, Any],
        *,
        entities: list[Entity],
    ) -> list[Relation]:
        by_type = result.get("relation_extraction", {})
        if not isinstance(by_type, dict):
            raise ValueError("GLiNER2 returned malformed relations")
        relations: list[Relation] = []
        entity_texts = {entity["text"] for entity in entities}
        for relation_type, candidates in by_type.items():
            if not isinstance(relation_type, str) or not relation_type.strip() or not isinstance(candidates, list):
                raise ValueError("GLiNER2 returned malformed relations")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ValueError("GLiNER2 returned malformed relation")
                head = candidate.get("head")
                tail = candidate.get("tail")
                if not isinstance(head, dict) or not isinstance(tail, dict):
                    raise ValueError("GLiNER2 returned malformed relation endpoints")
                head_text = head.get("text")
                tail_text = tail.get("text")
                if not isinstance(head_text, str) or not isinstance(tail_text, str):
                    raise ValueError("GLiNER2 returned malformed relation endpoints")
                # Upstream discovers relation endpoints from the text; it does
                # not accept the caller-supplied entity set as a constraint.
                # Treat out-of-set candidates as negative predictions so they
                # never escape the supplied-entity boundary or fail the whole
                # batch when another candidate is valid.
                if head_text not in entity_texts or tail_text not in entity_texts:
                    continue
                head_score = cls._validate_score(head.get("confidence", 0.0), "relation head")
                tail_score = cls._validate_score(tail.get("confidence", 0.0), "relation tail")
                relations.append(
                    Relation(
                        head=head_text,
                        tail=tail_text,
                        relation=relation_type.strip(),
                        score=min(head_score, tail_score),
                    )
                )
        relations.sort(
            key=lambda relation: (
                -relation["score"],
                relation["relation"],
                relation["head"],
                relation["tail"],
            )
        )
        return relations

    @staticmethod
    def _json_schema_to_structures(output_schema: dict[str, Any]) -> dict[str, list[str]]:
        if output_schema.get("type") != "object":
            raise InvalidInputError("GLiNER2 output_schema root type must be object")
        properties = output_schema.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise InvalidInputError("GLiNER2 output_schema requires non-empty properties")
        allowed_root = {
            "type",
            "properties",
            "required",
            "additionalProperties",
            "description",
            "title",
        }
        unsupported_root = set(output_schema) - allowed_root
        if unsupported_root:
            raise InvalidInputError(f"GLiNER2 output_schema has unsupported root keywords: {sorted(unsupported_root)}")
        required = output_schema.get("required", [])
        if (
            not isinstance(required, list)
            or any(not isinstance(name, str) or name not in properties for name in required)
            or len(set(required)) != len(required)
        ):
            raise InvalidInputError("GLiNER2 output_schema required must contain unique property names")
        additional_properties = output_schema.get("additionalProperties", True)
        if not isinstance(additional_properties, bool):
            raise InvalidInputError("GLiNER2 output_schema additionalProperties must be boolean")

        fields: list[str] = []
        for name, definition in properties.items():
            if not isinstance(name, str) or not name or any(delimiter in name for delimiter in _STRUCTURE_DELIMITERS):
                raise InvalidInputError("GLiNER2 output_schema property names contain unsupported delimiters")
            if not isinstance(definition, dict):
                raise InvalidInputError(f"GLiNER2 output_schema property {name!r} must be an object")
            unsupported = set(definition) - {"type", "description", "enum", "items", "title"}
            if unsupported:
                raise InvalidInputError(
                    f"GLiNER2 output_schema property {name!r} has unsupported keywords: {sorted(unsupported)}"
                )
            description = definition.get("description") or definition.get("title")
            if description is not None and not isinstance(description, str):
                raise InvalidInputError(f"GLiNER2 output_schema property {name!r} description must be a string")
            description = description.replace("::", ":") if description else None

            field_type = definition.get("type")
            enum = definition.get("enum")
            if enum is not None:
                if field_type != "string" or not isinstance(enum, list) or not enum:
                    raise InvalidInputError(f"GLiNER2 output_schema property {name!r} has invalid enum")
                if any(
                    not isinstance(choice, str)
                    or not choice
                    or any(delimiter in choice for delimiter in _STRUCTURE_DELIMITERS)
                    for choice in enum
                ):
                    raise InvalidInputError(
                        f"GLiNER2 output_schema property {name!r} enum values contain unsupported delimiters"
                    )
                choices = "|".join(enum)
                spec = f"{name}::[{choices}]::str"
            elif field_type == "string":
                spec = f"{name}::str"
            elif field_type == "array" and definition.get("items") == {"type": "string"}:
                spec = f"{name}::list"
            else:
                raise InvalidInputError(
                    f"GLiNER2 output_schema property {name!r} supports only string, string enum, or array of strings"
                )
            if description:
                spec = f"{spec}::{description}"
            fields.append(spec)
        return {_STRUCTURE_NAME: fields}

    @classmethod
    def _flatten_structured_result(
        cls,
        result: dict[str, Any],
        *,
        output_schema: dict[str, Any],
    ) -> dict[str, Any]:
        values = result.get(_STRUCTURE_NAME, [])
        if values in (None, []):
            data: dict[str, Any] = {}
        elif not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
            raise ValueError("GLiNER2 structured extraction did not return exactly one root object")
        else:
            data = dict(values[0])
        cls._validate_structured_result(data, output_schema)
        return data

    @staticmethod
    def _validate_structured_result(data: dict[str, Any], output_schema: dict[str, Any]) -> None:
        properties = output_schema["properties"]
        required = output_schema.get("required", [])
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"GLiNER2 structured extraction omitted required properties: {missing}")

        if output_schema.get("additionalProperties", True) is False:
            unexpected = sorted(set(data) - set(properties))
            if unexpected:
                raise ValueError(f"GLiNER2 structured extraction returned unexpected properties: {unexpected}")

        for name, value in data.items():
            definition = properties.get(name)
            if definition is None:
                continue
            expected_type = definition.get("type")
            if expected_type == "string":
                if not isinstance(value, str):
                    raise ValueError(f"GLiNER2 structured extraction property {name!r} must be a string")
                choices = definition.get("enum")
                if choices is not None and value not in choices:
                    raise ValueError(f"GLiNER2 structured extraction property {name!r} is outside its enum")
            elif not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                raise ValueError(f"GLiNER2 structured extraction property {name!r} must be an array of strings")

    def _model_text(self, text: str) -> str:
        """The prefix of ``text`` holding every word gliner2 reads from it, or ``text``."""
        return self._window(text)[0]

    def _window(self, text: str) -> tuple[str, int | None]:
        """``(model text, subwords)`` of ``text``: see ``_read``."""
        model_text, subwords, _ = self._read(text)
        return model_text, subwords

    def _read(self, text: str) -> tuple[str, int | None, Window | None]:
        """``(model text, subwords, window)``: the prefix of ``text`` gliner2 reads the same words from, and their subwords.

        gliner2 splits a text into words (with the bounded splitter installed
        at load, which yields only the window it reads) and keeps the first
        ``max_len``. ``Window.cut`` is where a prefix gives the same window:
        the end of the word holding the last piece read, or the first piece
        not read. No word crosses that point, so the prefix splits into the
        same words at the same offsets. gliner2 1.x splits the lowercased text
        and indexes the original with those offsets, so the prefix ends at the
        lowercased offset; it is used only when lowercasing keeps the text's
        length and the prefix's lowercase starts the lowercased text (a final
        sigma can lowercase differently at the cut). gliner2 2.x splits the
        text as given. The window is read from the text as gliner2 reads it,
        with the "." it appends to a text without a sentence end, and its word
        offsets index the text as the splitter reads it (lowercased first by
        gliner2 1.x). The subwords and the window are None when no bounded
        splitter is installed.
        """
        splitter = self._word_splitter
        if splitter is None:
            return text, None, None
        # gliner2 ends a text without a sentence end with ".", and reads that too.
        window = splitter.window(text if text.endswith(_SENTENCE_END) else text + ".", lower=True)
        cut = window.cut
        if cut is None:
            return text, window.subwords, window
        source = text.lower() if self._lower_text_first else text
        if len(source) != len(text):
            # Offsets into the lowercased text do not index this one.
            return text, window.subwords, window
        if cut < len(source) and source[cut].isspace():
            # Keep the separator: gliner2 ends a text without a sentence end with
            # ".", which a URL word (running to whitespace) would absorb.
            cut += 1
        if cut >= len(text):
            return text, window.subwords, window
        prefix = text[:cut]
        if self._lower_text_first and not source.startswith(prefix.lower()):
            return text, window.subwords, window
        return prefix, window.subwords, window

    def _classification_windows(
        self, text: str, first: tuple[str, int | None, Window | None]
    ) -> list[tuple[str, int | None]]:
        """The windows ``text`` is classified in, as ``(model text, subwords)``.

        ``first`` is ``_read(text)``, the window gliner2 reads of ``text``. A
        text all of whose words it reads is one window, the text as before;
        a longer text is read as the windows of ``_windows``.

        Raises:
            InvalidInputError: The text takes more than ``_MAX_CLASSIFY_WINDOWS`` windows.
        """
        windows = self._windows(text, first, limit=_MAX_CLASSIFY_WINDOWS)
        if windows is None:
            raise InvalidInputError(
                f"GLiNER2 classifies a text in at most {_MAX_CLASSIFY_WINDOWS} windows of "
                f"{self._max_seq_length or _DEFAULT_MAX_WORDS} words; this text needs more. "
                "Split it into several items."
            )
        return windows[2]

    def _windows(
        self, text: str, first: tuple[str, int | None, Window | None], *, limit: int
    ) -> tuple[list[int], list[int], list[tuple[str, int | None]]] | None:
        """``(starts, read ends, windows)``: every window ``text`` is read in.

        Each window is ``(model text, subwords)`` as ``_read`` returns it, so a
        text all of whose words gliner2 reads is one window, the text as
        before. ``first`` is ``_read(text)``. Otherwise each next window starts
        at the start of the last ``_CLASSIFY_OVERLAP_WORDS`` words the one
        before it read (at most half of them, so every window reads new
        words), and is read as gliner2 reads a text starting there: at most
        ``max_seq_length`` words within the subword budget. A window after the
        first is cut from the text as the splitter reads it (lowercased first
        by gliner2 1.x, which lowercases it again to the same text), so that
        the words' offsets index it. ``starts[i]`` is where window ``i`` begins
        in ``text`` and ``read ends[i]`` is where its last word ends there, so
        a span found in window ``i`` moves back to the text by ``starts[i]``.

        Returns ``None`` when the text takes more than ``limit`` windows.
        """
        model_text, subwords, window = first
        windows = [(model_text, subwords)]
        starts = [0]
        read_ends = [window.words[-1][2] if window is not None and window.words else 0]
        if window is None or window.cut is None:
            return starts, read_ends, windows
        source = text.lower() if self._lower_text_first else text
        start = 0
        while True:
            words = window.words
            read_end = start + words[-1][2]
            if _NON_SPACE.search(source, read_end) is None:
                return starts, read_ends, windows
            if len(windows) >= limit:
                return None
            shared = min(_CLASSIFY_OVERLAP_WORDS, len(words) // 2)
            start = start + words[len(words) - shared][1] if shared else read_end
            model_text, subwords, window = self._read_from(source, start)
            windows.append((model_text, subwords))
            starts.append(start)
            read_ends.append(start + (window.words[-1][2] if window is not None and window.words else 0))
            if window is None or window.cut is None:
                return starts, read_ends, windows

    def _read_from(self, source: str, start: int) -> tuple[str, int | None, Window | None]:
        """``_read`` of ``source[start:]``, from a slice of it long enough to give the same window.

        The slice grows until the last word the window reads ends at least
        ``MAX_WORD_CHARS`` characters before the slice does (or the slice
        reaches the end of ``source``), so no word it reads is cut short by the
        slice. A word longer than that is read in pieces from its start, so its
        pieces are the same in the slice as in ``source``.
        """
        size = _WINDOW_SLICE_CHARS
        while True:
            rest = source[start : start + size]
            read = self._read(rest)
            window = read[2]
            if (
                start + size >= len(source)
                or window is None
                or (window.cut is not None and bool(window.words) and window.words[-1][2] + MAX_WORD_CHARS <= len(rest))
            ):
                return read
            size *= 2

    def _windowed_input_token_counts(
        self, texts: list[str], item_windows: list[list[tuple[str, int | None]]]
    ) -> list[int] | None:
        """Input tokens of each item classified in ``item_windows``.

        A text read in one window is metered as before. A text read in several
        is metered as the sum of its windows, each as the same text sent alone
        is metered (its tokens up to ``max_seq_length``), so the words two
        windows share count twice: the model encodes them twice.
        """
        metered = [
            [text] if len(windows) == 1 else [model_text for model_text, _ in windows]
            for text, windows in zip(texts, item_windows, strict=True)
        ]
        counts = self._doc_input_token_counts([one for parts in metered for one in parts])
        if counts is None:
            return None
        totals = []
        offset = 0
        for parts in metered:
            totals.append(sum(counts[offset : offset + len(parts)]))
            offset += len(parts)
        return totals

    def _effective_positive_label(self, options: dict[str, Any], labels: list[str]) -> str | None:
        """The positive label of this request: a runtime override, else the configured one if among ``labels``."""
        if "positive_label" in options:
            return self._validate_positive_label(options["positive_label"], labels)
        if self._positive_label is not None and self._positive_label in labels:
            return self._positive_label
        return None

    @staticmethod
    def _validate_positive_label(value: object, labels: list[str] | None) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise InvalidInputError("GLiNER2 positive_label must be a non-empty string")
        label = value.strip()
        if labels is not None and label not in labels:
            raise InvalidInputError("GLiNER2 positive_label must be one of the labels")
        return label

    def _doc_input_token_counts(self, texts: list[str]) -> list[int] | None:
        processor = getattr(self._model, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            return None
        limit = self._max_seq_length
        try:
            counts: list[int | None] = [None] * len(texts)
            short = [index for index, text in enumerate(texts) if limit is None or len(text) <= _meter_budget(limit)]
            if short:
                encoded = tokenizer(
                    [texts[index] for index in short],
                    add_special_tokens=True,
                    truncation=limit is not None,
                    max_length=limit,
                )
                ids = encoded["input_ids"]
                if len(ids) != len(short):
                    return None
                for index, row in zip(short, ids, strict=True):
                    counts[index] = len(row)
            if limit is not None:
                for index, text in enumerate(texts):
                    if counts[index] is None:
                        counts[index] = self._long_doc_input_tokens(tokenizer, text, limit)
        except Exception:  # noqa: BLE001 -- metering must not fail extraction
            return None
        complete = [count for count in counts if count is not None]
        return complete if len(complete) == len(texts) else None

    @staticmethod
    def _long_doc_input_tokens(tokenizer: Any, text: str, limit: int) -> int:
        """Tokens of a long ``text`` with special tokens, truncated to ``limit``.

        Tokenizing a multi-megabyte text takes about a second, so prefixes are
        tokenized first, from ``_METER_CHARS_PER_TOKEN`` characters per window
        token up. A prefix ending just before a space tokenizes like the start of
        the text (these tokenizers split pre-tokens at spaces), so once it fills
        the window, so does the text. Without a space in the second half of the
        prefix, the prefix must fill the window with ``_METER_MARGIN_TOKENS`` to
        spare, for the tokens at its cut.
        """
        budget = _meter_budget(limit)
        while budget < len(text):
            cut = _last_space(text, budget)
            if cut is not None:
                if _token_count(tokenizer, text[:cut]) >= limit:
                    return limit
            elif _token_count(tokenizer, text[:budget]) >= limit + _METER_MARGIN_TOKENS:
                return limit
            budget *= 4
        encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=limit)
        return len(encoded["input_ids"])


def _meter_budget(limit: int) -> int:
    return limit * _METER_CHARS_PER_TOKEN


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def _last_space(text: str, end: int) -> int | None:
    """Index of the last space after a non-space character in the second half of ``text[:end]``, or None."""
    begin = end // 2
    match = _SPACE_AFTER_TEXT.search(text[begin:end][::-1])
    return None if match is None else end - 1 - match.start()


def pool_window_classifications(
    windows: list[list[tuple[str, float]]],
    *,
    labels: list[str],
    multi_label: bool,
    positive_label: str | None,
) -> list[tuple[str, float]]:
    """One text's classification from the classifications of its windows.

    ``windows`` holds each window's ``(label, confidence)`` pairs as gliner2
    returns them. A text is taken to have a label when any window of it has
    the label, so the pooled confidence of a label is its greatest confidence
    in any window (max pooling), and a label seen only late in a long text is
    not outweighed by the windows before it.

    A multi-label task (sigmoid per label) returns every label whose
    confidence reaches the threshold, so a label's greatest confidence over
    the windows is exact wherever it reaches the threshold; each label is
    returned with it.

    A single-label task (softmax over the labels) returns only the label a
    window takes and its probability, and one label is returned for the
    text, the result of one window:

    * With ``positive_label``: the window taking that label with the highest
      confidence. When no window takes it and there are two labels, the
      positive label's probability in a window is one minus the returned
      one, so the window with the lowest confidence is the one where it is
      highest. The text's probability of the positive label is then exactly
      its greatest probability in any window, whichever label is returned.
    * Otherwise, and with more than two labels when no window takes the
      positive label (whose probability in a window that took another label
      gliner2 does not return): the window with the highest confidence, the
      label with the greatest pooled confidence among those returned.

    Ties go to the earliest window.
    """
    if multi_label:
        best: dict[str, float] = {}
        for found in windows:
            for label, score in found:
                if label not in best or score > best[label]:
                    best[label] = score
        return list(best.items())
    chosen = [found[0] for found in windows if found]
    if not chosen:
        return []
    if positive_label is not None:
        positive = [one for one in chosen if one[0] == positive_label]
        if positive:
            return [max(positive, key=lambda one: one[1])]
        if len(labels) == 2:
            return [min(chosen, key=lambda one: one[1])]
    return [max(chosen, key=lambda one: one[1])]
