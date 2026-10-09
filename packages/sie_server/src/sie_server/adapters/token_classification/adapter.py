"""Fixed-head OntoNotes extraction with complete, source-aligned token windows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForTokenClassification, AutoTokenizer, RobertaTokenizerFast

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import Entity, ErrorCode

_WINDOW_TOKENS = 512
_OVERLAP_TOKENS = 128
_MAX_DOCUMENT_TOKENS = 16384
_TYPES = (
    "PERSON",
    "NORP",
    "FAC",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "DATE",
    "TIME",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "ORDINAL",
    "CARDINAL",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
)
_ID2LABEL = dict(enumerate(["O", *[label for tag in _TYPES for label in (f"B-{tag}", f"I-{tag}")]]))
_TYPE_MAP = {"PERSON": "person", "ORG": "organization", "GPE": "location", "LOC": "location", "FAC": "location"}
_LABELS = ("person", "organization", "location")
_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
_MODEL_INPUTS = frozenset({"input_ids", "attention_mask", "token_type_ids"})
_ERR_TOKENIZATION = "OntoNotes tokenizer did not preserve complete source-aligned token windows"
_ERR_INFERENCE = "OntoNotes token classification failed; no partial entities are returned"


class _DocumentTooLongError(ValueError):
    """The complete document exceeds the configured token-work bound."""


@dataclass(frozen=True)
class _Window:
    inputs: dict[str, list[int]]
    offsets: list[tuple[int, int]]
    special: list[int]
    token_count: int


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    tag: str
    score: float


class OntoNotesTokenClassificationAdapter(BaseAdapter):
    """Expose a RoBERTa OntoNotes BIO head as three fixed extraction labels.

    Documents use 512-token windows (special tokens included) with 128-token
    overlap. The full untruncated token sequence is checked before any forward;
    at most 16,384 document tokens are accepted by default. Usage counts every
    forwarded window, including special tokens and repeated overlap, without
    padding. SIMPLE BIO grouping and the Transformers overlap policy preserve
    literal character offsets; entities crossing a window may remain fragments.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model", "_tokenizer"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        max_seq_length: int | None = None,
        window_overlap: int = _OVERLAP_TOKENS,
        max_document_tokens: int = _MAX_DOCUMENT_TOKENS,
        compute_precision: ComputePrecision = "float32",
    ) -> None:
        if max_seq_length is not None and (type(max_seq_length) is not int or max_seq_length != _WINDOW_TOKENS):
            raise ValueError("OntoNotes token classification requires max_seq_length=512")
        if type(window_overlap) is not int or window_overlap != _OVERLAP_TOKENS:
            raise ValueError("OntoNotes token classification requires window_overlap=128")
        if type(max_document_tokens) is not int or not 0 < max_document_tokens <= _MAX_DOCUMENT_TOKENS:
            raise ValueError("max_document_tokens must be an integer from 1 through 16384")
        if compute_precision != "float32":
            raise ValueError("OntoNotes token classification requires float32")
        self._model_name_or_path = str(model_name_or_path)
        self._revision = revision
        self._max_seq_length = _WINDOW_TOKENS
        self._window_overlap = window_overlap
        self._max_document_tokens = max_document_tokens
        self._compute_precision = compute_precision
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        """Load builtin safetensors and fast tokenization from the same snapshot."""
        path = self._model_name_or_path
        if not Path(path).is_dir():
            path = snapshot_download(repo_id=path, revision=self._revision, allow_patterns=list(_CHECKPOINT_FILES))
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, trust_remote_code=False)
        if (
            not isinstance(tokenizer, RobertaTokenizerFast)
            or not tokenizer.is_fast
            or tokenizer.model_max_length != _WINDOW_TOKENS
            or tokenizer.add_prefix_space is not True
            or tokenizer.init_kwargs.get("trim_offsets", True) is not True
            or tokenizer.num_special_tokens_to_add(pair=False) != 2
            or not {"input_ids", "attention_mask"} <= set(tokenizer.model_input_names) <= _MODEL_INPUTS
        ):
            raise RuntimeError("OntoNotes token classification requires the 512-token fast RoBERTa tokenizer")
        model = AutoModelForTokenClassification.from_pretrained(
            path,
            use_safetensors=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
        )
        config = model.config
        if (
            config.model_type != "roberta"
            or config.architectures != ["RobertaForTokenClassification"]
            or config.num_labels != len(_ID2LABEL)
            or config.id2label != _ID2LABEL
            or config.label2id != {label: index for index, label in _ID2LABEL.items()}
            or type(config.max_position_embeddings) is not int
            or config.max_position_embeddings != _WINDOW_TOKENS + 2
        ):
            raise RuntimeError("OntoNotes token classification requires the complete pinned 37-label RoBERTa head")
        model.to(device=device, dtype=torch.float32)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def count_input_tokens(self, items: list[Item]) -> list[int] | None:
        """Only terminal forwarded-window counts are authoritative for this adapter."""
        return None

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
        """Extract full documents, keeping item failures and forwarded usage aligned."""
        self._check_loaded()
        if instruction is not None or output_schema is not None or options:
            raise InvalidInputError("OntoNotes extraction accepts only labels, without instruction, schema or options")
        effective_labels = self._labels(labels)
        entities: list[list[Entity]] = [[] for _ in items]
        errors: list[ExtractItemError | None] = [None for _ in items]
        counts = [0] * len(items)
        for index, item in enumerate(items):
            text = item.text
            if not isinstance(text, str) or not text.strip():
                errors[index] = ExtractItemError(
                    code=ErrorCode.INVALID_INPUT.value,
                    message="OntoNotes extraction requires non-blank text",
                )
                continue
            try:
                with self._tokenizer_guard():
                    windows = self._windows(text)
            except _DocumentTooLongError:
                errors[index] = ExtractItemError(
                    code=ErrorCode.INPUT_TOO_LONG.value,
                    message=f"OntoNotes extraction accepts at most {self._max_document_tokens} complete document tokens",
                )
                continue
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError):
                errors[index] = ExtractItemError(code=ErrorCode.INFERENCE_ERROR.value, message=_ERR_TOKENIZATION)
                continue
            spans: list[_Span] = []
            try:
                with torch.inference_mode():
                    for window in windows:
                        inputs = {
                            name: torch.tensor([values], dtype=torch.long, device=self._device)
                            for name, values in window.inputs.items()
                        }
                        counts[index] += window.token_count
                        logits = self._model(**inputs).logits
                        spans.extend(self._decode(logits, window))
                if len(windows) > 1:
                    spans = self._overlaps(spans)
                entities[index] = [
                    Entity(
                        text=text[span.start : span.end],
                        start=span.start,
                        end=span.end,
                        label=_TYPE_MAP[span.tag],
                        score=span.score,
                    )
                    for span in spans
                    if _TYPE_MAP.get(span.tag) in effective_labels
                ]
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, RuntimeError):
                errors[index] = ExtractItemError(code=ErrorCode.INFERENCE_ERROR.value, message=_ERR_INFERENCE)
        return ExtractOutput(
            entities=entities,
            errors=errors if any(errors) else None,
            input_token_counts=counts,
        )

    @staticmethod
    def _labels(labels: list[str] | None) -> set[str]:
        if labels is None:
            return set(_LABELS)
        if (
            not isinstance(labels, list)
            or not labels
            or any(not isinstance(label, str) or label not in _LABELS for label in labels)
            or len(labels) != len(set(labels))
        ):
            raise InvalidInputError(
                "OntoNotes labels must be a nonempty unique subset of person, organization, location"
            )
        return set(labels)

    @staticmethod
    def _offsets(value: Any, text: str, size: int) -> list[tuple[int, int]]:
        if not isinstance(value, list) or len(value) != size:
            raise ValueError(_ERR_TOKENIZATION)
        result: list[tuple[int, int]] = []
        for offset in value:
            if (
                not isinstance(offset, (list, tuple))
                or len(offset) != 2
                or any(type(part) is not int for part in offset)
                or not 0 <= offset[0] <= offset[1] <= len(text)
            ):
                raise ValueError(_ERR_TOKENIZATION)
            result.append((offset[0], offset[1]))
        return result

    @staticmethod
    def _integers(value: Any) -> list[int]:
        if not isinstance(value, list) or any(type(part) is not int or part < 0 for part in value):
            raise ValueError(_ERR_TOKENIZATION)
        return value

    def _windows(self, text: str) -> list[_Window]:
        """Verify every overflow window against the complete ID/offset sequence."""
        full = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        full_ids = self._integers(full["input_ids"])
        if len(full_ids) > self._max_document_tokens:
            raise _DocumentTooLongError
        full_offsets = self._offsets(full["offset_mapping"], text, len(full_ids))
        if not full_ids or any(
            current[0] < previous[0] or current[1] < previous[1] for previous, current in pairwise(full_offsets)
        ):
            raise ValueError(_ERR_TOKENIZATION)
        encoded = self._tokenizer(
            text,
            truncation=True,
            max_length=self._max_seq_length,
            stride=self._window_overlap,
            padding=False,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_attention_mask=True,
        )
        rows = encoded["input_ids"]
        if not isinstance(rows, list) or not rows or encoded["overflow_to_sample_mapping"] != [0] * len(rows):
            raise ValueError(_ERR_TOKENIZATION)
        names = list(self._tokenizer.model_input_names)
        for name in {*names, "offset_mapping", "special_tokens_mask"}:
            if not isinstance(encoded[name], list) or len(encoded[name]) != len(rows):
                raise ValueError(_ERR_TOKENIZATION)
        windows: list[_Window] = []
        start = 0
        for index, row in enumerate(rows):
            ids = self._integers(row)
            if not 2 < len(ids) <= self._max_seq_length:
                raise ValueError(_ERR_TOKENIZATION)
            offsets = self._offsets(encoded["offset_mapping"][index], text, len(ids))
            special = self._integers(encoded["special_tokens_mask"][index])
            inputs = {name: self._integers(encoded[name][index]) for name in names}
            if len(special) != len(ids) or any(len(values) != len(ids) for values in inputs.values()):
                raise ValueError(_ERR_TOKENIZATION)
            attention = inputs["attention_mask"]
            if any(value not in (0, 1) for value in [*attention, *special]):
                raise ValueError(_ERR_TOKENIZATION)
            if attention != sorted(attention, reverse=True):
                raise ValueError(_ERR_TOKENIZATION)
            for token, active, marker, offset in zip(ids, attention, special, offsets, strict=True):
                if (marker and offset != (0, 0)) or (
                    not active and (not marker or token != self._tokenizer.pad_token_id)
                ):
                    raise ValueError(_ERR_TOKENIZATION)
            positions = [
                i for i, (active, marker) in enumerate(zip(attention, special, strict=True)) if active and not marker
            ]
            end = min(start + self._max_seq_length - 2, len(full_ids))
            if (
                [ids[i] for i in positions] != full_ids[start:end]
                or [offsets[i] for i in positions] != full_offsets[start:end]
                or [token for token, active in zip(ids, attention, strict=True) if active]
                != self._tokenizer.build_inputs_with_special_tokens(full_ids[start:end])
            ):
                raise ValueError(_ERR_TOKENIZATION)
            windows.append(_Window(inputs=inputs, offsets=offsets, special=special, token_count=sum(attention)))
            if end == len(full_ids):
                if index != len(rows) - 1:
                    raise ValueError(_ERR_TOKENIZATION)
                return windows
            start = end - self._window_overlap
        raise ValueError(_ERR_TOKENIZATION)

    @staticmethod
    def _decode(logits: Any, window: _Window) -> list[_Span]:
        """SIMPLE BIO grouping; original offsets are never expanded or repaired."""
        if (
            not isinstance(logits, torch.Tensor)
            or not logits.is_floating_point()
            or tuple(logits.shape) != (1, len(window.offsets), len(_ID2LABEL))
            or not torch.isfinite(logits).all().item()
        ):
            raise ValueError(_ERR_INFERENCE)
        scores, predictions = logits[0].float().softmax(dim=-1).max(dim=-1)
        spans: list[_Span] = []
        group: list[tuple[int, int, float]] = []
        tag = "O"

        def finish() -> None:
            if group and tag != "O":
                score = sum(token[2] for token in group) / len(group)
                if not math.isfinite(score):
                    raise ValueError(_ERR_INFERENCE)
                spans.append(_Span(group[0][0], group[-1][1], tag, score))

        for index, (label_id, score) in enumerate(zip(predictions.tolist(), scores.tolist(), strict=True)):
            start, end = window.offsets[index]
            if window.special[index] or not window.inputs["attention_mask"][index] or start == end:
                continue
            label = _ID2LABEL[label_id]
            prefix, current_tag = ("I", "O") if label == "O" else label.split("-", 1)
            if group and (prefix == "B" or current_tag != tag):
                finish()
                group = []
            tag = current_tag
            group.append((start, end, score))
        finish()
        return spans

    @staticmethod
    def _overlaps(spans: list[_Span]) -> list[_Span]:
        """Transformers SIMPLE window overlap policy: length, then score, then first."""
        if not spans:
            return []
        ordered = sorted(spans, key=lambda span: span.start)
        result: list[_Span] = []
        previous = ordered[0]
        for span in ordered[1:]:
            if span.start < previous.end:
                length, previous_length = span.end - span.start, previous.end - previous.start
                if length > previous_length or (length == previous_length and span.score > previous.score):
                    previous = span
            else:
                result.append(previous)
                previous = span
        result.append(previous)
        return result
