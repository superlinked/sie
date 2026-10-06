"""Native entity extraction for GLiNER2 span and boundary checkpoints.

The encoded-row limit includes the processor's entity schema and complete
document. The package's ``max_len`` counts words and truncates them; it stays
``None`` here. Oversized items fail before inference, so callers can choose
their own source windows explicitly.
"""

from __future__ import annotations

import math
from functools import lru_cache
from numbers import Real
from pathlib import Path
from typing import Any, ClassVar

import torch
from huggingface_hub import snapshot_download

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._prompt_limit import MAX_PROMPT_CHARS_PER_TOKEN, check_label_chars
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters._word_window import plan_forwards
from sie_server.adapters.gliner2.decisions import MARKERS
from sie_server.adapters.gliner2.words import linear_equivalent
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Entity, ErrorCode

_DEFAULT_MAX_ROW_TOKENS = 4096
_DEFAULT_MAX_PROMPT_TOKENS = 2048
_DEFAULT_BATCH_SIZE = 8
_WORD_CACHE_SIZE = 16384
_CACHED_WORD_CHARS = 32
_MAX_SOURCE_CHARS_PER_ROW_TOKEN = 64
_MAX_SOURCE_WORDS_PER_ROW_TOKEN = 4
_MAX_SOURCE_WORD_CHARS = 4096
_CHECKPOINT_FILES = (
    "config.json",
    "encoder_config/*",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "model.safetensors",
)
_ERR_RESULT = "GLiNER2 returned malformed entity results"


class GLiNER2EntitiesAdapter(BaseAdapter):
    """Extract exact source spans with ``gliner2==2.0.0``'s AutoExtractor.

    ``max_seq_length`` limits the entire encoded row, including the schema
    prompt; ``max_prompt_tokens`` separately limits that prompt. Successful
    items bill the document-segment subwords that the native processor emits,
    including its appended terminal punctuation. ``data`` also records the
    full encoded-row and prompt counts. Failed items bill zero tokens.

    Entity labels are exact names, preserving case and punctuation, except
    reserved prompt markers, with at most 128 characters per label. Optional
    ``options.entity_descriptions`` maps existing labels to guidance in the
    native schema's separate description channel. Admission also bounds
    source preprocessing to 64 characters and four words per row-budget token,
    with at most 4096 characters per word. The prompt allows at most one label
    and 32 combined label/description characters per prompt-budget token,
    including labels repeated in description prefixes. Exceeding any admission
    bound rejects the complete item before native collation or inference.
    Runtime options also accept the native ``threshold`` in [0, 1]. This
    adapter does not perform classification, relations, JSON extraction or
    automatic source windowing.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model", "_processor", "_tokenizer", "_word_splitter", "_word_cache", "_architecture"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        max_seq_length: int | None = None,
        max_prompt_tokens: int = _DEFAULT_MAX_PROMPT_TOKENS,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        threshold: float = 0.5,
        default_labels: list[str] | None = None,
        compute_precision: ComputePrecision = "float32",
    ) -> None:
        self._model_name_or_path = str(model_name_or_path)
        self._revision = revision
        self._max_seq_length = self._positive_integer(
            _DEFAULT_MAX_ROW_TOKENS if max_seq_length is None else max_seq_length, "max_seq_length"
        )
        self._max_prompt_tokens = self._positive_integer(max_prompt_tokens, "max_prompt_tokens")
        self._batch_size = self._positive_integer(batch_size, "batch_size")
        self._threshold = self._validate_threshold(threshold)
        self._default_labels = self._validate_labels(default_labels) if default_labels is not None else None
        if compute_precision not in {"float16", "bfloat16", "float32"}:
            raise ValueError("GLiNER2 entities compute_precision must be float16, bfloat16 or float32")
        self._compute_precision = compute_precision
        self._model: Any = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._word_splitter: Any = None
        self._word_cache: Any = None
        self._architecture: str | None = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        """Resolve the snapshot, then let the native loader select its architecture."""
        try:
            from gliner2 import AutoExtractor  # ty:ignore[unresolved-import]
        except ImportError as exc:
            raise RuntimeError("GLiNER2 entities requires gliner2==2.0.0 (AutoExtractor)") from exc

        path = self._model_name_or_path
        if not Path(path).is_dir():
            path = snapshot_download(repo_id=path, revision=self._revision, allow_patterns=list(_CHECKPOINT_FILES))
        model = AutoExtractor.from_pretrained(path, map_location=device, quantize=False)
        architecture = getattr(model, "architecture", None)
        if architecture not in {"span", "boundary"}:
            raise RuntimeError("GLiNER2 entities requires a native span or boundary checkpoint")
        processor = model.processor
        splitter = linear_equivalent(processor.word_splitter)
        if splitter is None or splitter.lower_text_first:
            raise RuntimeError("GLiNER2 entities requires the verified source-preserving native word splitter")
        processor.word_splitter = splitter

        # The vendor cache retains long document words and schema strings.
        # Keep only short strings without changing their tokenization.
        tokenize = processor.tokenizer.tokenize
        cached = lru_cache(maxsize=_WORD_CACHE_SIZE)(tokenize)

        def tokenize_word(word: str) -> list[str]:
            return cached(word) if len(word) <= _CACHED_WORD_CHARS else tokenize(word)

        processor._tokenize_cached = tokenize_word
        processor.change_mode(is_training=False)
        model.to(dtype=self._resolve_dtype())
        model.eval()
        self._model = model
        self._processor = processor
        self._tokenizer = processor.tokenizer
        self._word_splitter = splitter
        self._word_cache = cached
        self._architecture = architecture
        self._device = device

    def unload(self) -> None:
        """Drop native objects and cached tokenizations."""
        if self._word_cache is not None:
            self._word_cache.cache_clear()
        super().unload()

    def warmup(self) -> None:
        """Initialize the native extraction path with one small source."""
        self.extract([Item(text="Alice.")], labels=["person"])

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
        """Validate complete native rows, then extract accepted sources in bounded batches."""
        _ = prepared_items
        self._check_loaded()
        if output_schema is not None or instruction is not None:
            raise InvalidInputError("GLiNER2 entities accepts labels, without output_schema or instruction")
        opts = options or {}
        if set(opts) - {"threshold", "entity_descriptions"}:
            raise InvalidInputError(
                "GLiNER2 entities supports only the threshold and entity_descriptions runtime options"
            )
        threshold = self._validate_threshold(opts.get("threshold", self._threshold))
        effective_labels = self._validate_labels(self._default_labels if labels is None else labels)
        descriptions = (
            self._validate_descriptions(opts["entity_descriptions"], effective_labels)
            if "entity_descriptions" in opts
            else {}
        )
        entity_types = (
            {label: descriptions.get(label, {}) for label in effective_labels} if descriptions else effective_labels
        )

        entities: list[list[Entity]] = [[] for _ in items]
        data: list[dict[str, Any]] = [{} for _ in items]
        errors: list[ExtractItemError | None] = [None for _ in items]
        counts = [0] * len(items)
        accepted: list[tuple[int, str, int, int, int]] = []
        with self._tokenizer_guard():
            prompt_failure = self._prompt_admission_failure(effective_labels, descriptions)
            schema = self._model.create_schema().entities(entity_types).build() if prompt_failure is None else None
            for index, item in enumerate(items):
                text = item.text
                if not isinstance(text, str):
                    errors[index] = ExtractItemError(
                        code=ErrorCode.INVALID_INPUT.value, message="GLiNER2 entities requires non-blank text"
                    )
                    continue
                admission_failure = prompt_failure or self._source_admission_failure(text)
                if admission_failure is not None:
                    errors[index] = ExtractItemError(code=ErrorCode.INPUT_TOO_LONG.value, message=admission_failure)
                    continue
                if not text.strip():
                    errors[index] = ExtractItemError(
                        code=ErrorCode.INVALID_INPUT.value, message="GLiNER2 entities requires non-blank text"
                    )
                    continue
                assert schema is not None
                row_tokens, document_tokens = self._measure_row(text, schema)
                prompt_tokens = row_tokens - document_tokens
                if row_tokens > self._max_seq_length or prompt_tokens > self._max_prompt_tokens:
                    errors[index] = ExtractItemError(
                        code=ErrorCode.INPUT_TOO_LONG.value,
                        message=(
                            f"GLiNER2 entities needs {row_tokens} encoded tokens including {prompt_tokens} prompt tokens; "
                            f"limits are {self._max_seq_length} per complete row and {self._max_prompt_tokens} per prompt. "
                            + (
                                "Send a shorter source window or fewer or shorter labels or descriptions."
                                if descriptions
                                else "Send a shorter source window or fewer or shorter labels."
                            )
                        ),
                    )
                    continue
                accepted.append((index, text, row_tokens, document_tokens, prompt_tokens))

            row_sizes = [row_tokens for _, _, row_tokens, _, _ in accepted]
            groups = plan_forwards(row_sizes, rows_per_pass=self._batch_size)
            if groups is None:
                groups = [
                    list(range(start, min(start + self._batch_size, len(accepted))))
                    for start in range(0, len(accepted), self._batch_size)
                ]
            with torch.inference_mode():
                for group in groups:
                    sources = [accepted[row][1] for row in group]
                    results = self._model.batch_extract_entities(
                        sources,
                        entity_types,
                        batch_size=len(sources),
                        threshold=threshold,
                        include_confidence=True,
                        include_spans=True,
                        max_len=None,
                    )
                    if not isinstance(results, list) or len(results) != len(group):
                        for row in group:
                            errors[accepted[row][0]] = ExtractItemError(
                                code=ErrorCode.INFERENCE_ERROR.value, message=_ERR_RESULT
                            )
                        continue
                    for row, result in zip(group, results, strict=True):
                        index, text, row_tokens, document_tokens, prompt_tokens = accepted[row]
                        try:
                            entities[index] = self._flatten_entities(result, text, effective_labels)
                        except ValueError as exc:
                            errors[index] = ExtractItemError(code=ErrorCode.INFERENCE_ERROR.value, message=str(exc))
                            continue
                        counts[index] = document_tokens
                        data[index] = {
                            "encoded_row_token_count": row_tokens,
                            "schema_prompt_token_count": prompt_tokens,
                        }
        return ExtractOutput(
            entities=entities,
            data=data,
            errors=errors if any(error is not None for error in errors) else None,
            input_token_counts=counts,
        )

    def _measure_row(self, text: str, schema: dict[str, Any]) -> tuple[int, int]:
        """Count the exact native row and its document segment, without a forward pass."""
        batch = self._processor.collate_fn_inference(
            [(text, schema)], max_len=None, error_policy="raise", architecture=self._architecture
        )
        model_text = text if text.endswith((".", "!", "?")) else text + "."
        expected_offsets = [(start, end) for _, start, end in self._word_splitter(model_text)]
        if (
            len(batch.original_lengths) != 1
            or batch.original_texts != [model_text]
            or len(batch.mapped_indices) != 1
            or len(batch.start_mappings) != 1
            or len(batch.end_mappings) != 1
            or list(zip(batch.start_mappings[0], batch.end_mappings[0], strict=True)) != expected_offsets
        ):
            raise RuntimeError("GLiNER2 processor did not preserve the complete source and its character mappings")
        row_tokens = batch.original_lengths[0]
        mappings = batch.mapped_indices[0]
        if (
            not isinstance(row_tokens, int)
            or isinstance(row_tokens, bool)
            or row_tokens < 1
            or row_tokens != len(mappings)
            or any(len(mapping) != 3 or mapping[0] not in {"schema", "sep", "text"} for mapping in mappings)
        ):
            raise RuntimeError("GLiNER2 processor returned invalid encoded token mappings")
        return row_tokens, sum(mapping[0] == "text" for mapping in mappings)

    def count_input_tokens(self, items: list[Item]) -> None:
        """Counts come from request-qualified native rows in ``extract``."""
        _ = items

    def _prompt_admission_failure(self, labels: list[str], descriptions: dict[str, str]) -> str | None:
        max_chars = self._max_prompt_tokens * MAX_PROMPT_CHARS_PER_TOKEN
        prompt_chars = sum(len(label) for label in labels)
        if len(labels) > self._max_prompt_tokens or prompt_chars > max_chars:
            return (
                f"GLiNER2 entities permits at most {self._max_prompt_tokens} labels and "
                f"{max_chars} total label characters. "
                "Send fewer or shorter labels."
            )
        if descriptions:
            if any(len(description) > max_chars for description in descriptions.values()):
                return (
                    f"GLiNER2 entities permits at most {max_chars} characters per description. "
                    "Send shorter descriptions."
                )
            # Native description prefixes repeat each described label name.
            prompt_chars += sum(len(label) + len(description) for label, description in descriptions.items())
        if prompt_chars > max_chars:
            return (
                f"GLiNER2 entities permits at most {self._max_prompt_tokens} labels and "
                f"{max_chars} total label and description characters, including repeated label names. "
                "Send fewer or shorter labels or descriptions."
            )
        return None

    def _source_admission_failure(self, text: str) -> str | None:
        max_chars = self._max_seq_length * _MAX_SOURCE_CHARS_PER_ROW_TOKEN
        max_words = self._max_seq_length * _MAX_SOURCE_WORDS_PER_ROW_TOKEN
        failure = (
            f"GLiNER2 entities permits at most {max_chars} source characters, {max_words} source words and "
            f"{_MAX_SOURCE_WORD_CHARS} characters per word. Send a shorter complete source window."
        )
        if len(text) > max_chars:
            return failure
        for index, (_, start, end) in enumerate(self._word_splitter(text), 1):
            if index > max_words or end - start > _MAX_SOURCE_WORD_CHARS:
                return failure
        return None

    @staticmethod
    def _positive_integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"GLiNER2 entities {name} must be a positive integer")
        return value

    @staticmethod
    def _validate_threshold(value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise InvalidInputError("GLiNER2 entities threshold must be a finite number between 0 and 1")
        try:
            threshold = float(value)
        except (OverflowError, ValueError) as exc:
            raise InvalidInputError("GLiNER2 entities threshold must be a finite number between 0 and 1") from exc
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise InvalidInputError("GLiNER2 entities threshold must be a finite number between 0 and 1")
        return threshold

    @staticmethod
    def _validate_labels(value: object) -> list[str]:
        if not isinstance(value, list) or not value:
            raise InvalidInputError("GLiNER2 entities requires a non-empty list of labels")
        labels: list[str] = []
        for label in value:
            if not isinstance(label, str) or not label.strip():
                raise InvalidInputError("GLiNER2 entity labels must be non-blank strings")
            if any(marker in label for marker in MARKERS):
                raise InvalidInputError("GLiNER2 entity labels may not contain structural tokens of the model's prompt")
            labels.append(label)
        if len(set(labels)) != len(labels):
            raise InvalidInputError("GLiNER2 entity labels must be unique")
        check_label_chars("GLiNER2 entities", "labels", labels)
        return labels

    @staticmethod
    def _validate_descriptions(value: object, labels: list[str]) -> dict[str, str]:
        if not isinstance(value, dict):
            raise InvalidInputError("GLiNER2 entity_descriptions must be an object mapping existing labels to strings")
        label_names = set(labels)
        descriptions: dict[str, str] = {}
        for label, description in value.items():
            if not isinstance(label, str) or label not in label_names:
                raise InvalidInputError("GLiNER2 entity_descriptions keys must exactly match existing labels")
            if not isinstance(description, str):
                raise InvalidInputError("GLiNER2 entity_descriptions values must be strings")
            if any(marker in description for marker in MARKERS):
                raise InvalidInputError(
                    "GLiNER2 entity descriptions may not contain structural tokens of the model's prompt"
                )
            descriptions[label] = description
        return descriptions

    @staticmethod
    def _flatten_entities(result: Any, text: str, labels: list[str]) -> list[Entity]:
        if not isinstance(result, dict) or not isinstance(result.get("entities"), dict):
            raise ValueError(_ERR_RESULT)
        entities: list[Entity] = []
        for label, spans in result["entities"].items():
            if label not in labels or not isinstance(spans, list):
                raise ValueError(_ERR_RESULT)
            for span in spans:
                if not isinstance(span, dict):
                    raise ValueError(_ERR_RESULT)
                start, end, surface = span.get("start"), span.get("end"), span.get("text")
                if (
                    isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or not isinstance(surface, str)
                    or not 0 <= start < end <= len(text)
                    or text[start:end] != surface
                ):
                    raise ValueError("GLiNER2 returned an entity span that does not match the original source")
                confidence = span.get("confidence")
                if isinstance(confidence, bool) or not isinstance(confidence, Real):
                    raise ValueError("GLiNER2 returned invalid entity confidence")
                try:
                    score = float(confidence)
                except (OverflowError, ValueError) as exc:
                    raise ValueError("GLiNER2 returned invalid entity confidence") from exc
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("GLiNER2 returned invalid entity confidence")
                entities.append(Entity(text=surface, label=label, score=score, start=start, end=end))
        entities.sort(key=lambda entity: entity["start"])
        return entities
