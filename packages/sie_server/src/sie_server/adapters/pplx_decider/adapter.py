"""Text-only native Perplexity decisions through SIE extraction and decision APIs.

Format 1 checkpoints: every (state, question) is a complete chat-template row,
read out at its last token. Format 2 checkpoints (joint protocol v3): an item's
questions share one row and each is read out at its own marker; a row over the
window is bisected by questions, every chunk keeping the full state, exactly
as the reference does. The checkpoint's
readout, temperature, pooling and attention mode are preserved. Rows never
truncate; oversized items fail independently before any of their rows run.
Actual input usage sums the complete, unpadded rows the backbone processed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._checkpoint import resolve_checkpoint
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._transformers_compat import install_transformers_security_fixes
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.adapters.errors import InputTooLongError
from sie_server.adapters.pplx_decider.model import DecisionModel, checkpoint_config, load_model, validate_tokenizer
from sie_server.adapters.pplx_decider.questions import (
    SYSTEM_PROMPT,
    Question,
    answer,
    describe,
    joint_messages,
    messages,
    parse_questions,
)
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import ErrorCode

MAX_WINDOW = 8192
MAX_ITEM_TOKENS = 32_768
MAX_FORWARD_ROWS = 8
_CHECKPOINT_FILES = (
    "config.json",
    "decision_config.json",
    "joint_config.json",
    "readout.safetensors",
    "model*.safetensors",
    "model*.safetensors.index.json",
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template*",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)


@dataclass
class Row:
    """One already-tokenized, unpadded row, its original item index and its questions.

    ``markers`` holds each question's readout position in the unpadded row;
    ``None`` means the single question is read out by the checkpoint's pooling.
    """

    item: int
    questions: tuple[Question, ...]
    inputs: dict[str, torch.Tensor]
    markers: tuple[int, ...] | None = None

    @property
    def question(self) -> Question:
        return self.questions[0]

    @property
    def length(self) -> int:
        return self.inputs["input_ids"].shape[0]


class PerplexityDecisionAdapter(BaseAdapter):
    """Serve the checkpoint's native decision head, with no generation or remote code."""

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text",),
        outputs=("json",),
        unload_fields=("_model", "_processor", "_tokenizer"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        max_seq_length: int | None = None,
        max_forward_tokens: int = MAX_WINDOW,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Set serving bounds; optional architecture dependencies are loaded only in load()."""
        _ = kwargs
        if compute_precision not in {"bfloat16", "float32"}:
            raise ValueError("Perplexity decisions support bfloat16 or float32 compute precision")
        window = MAX_WINDOW if max_seq_length is None else max_seq_length
        if type(window) is not int or not 1 <= window <= MAX_WINDOW:
            raise ValueError("Perplexity max_seq_length must be between 1 and 8192")
        if type(max_forward_tokens) is not int or not window <= max_forward_tokens <= MAX_ITEM_TOKENS:
            raise ValueError("max_forward_tokens must cover one full row and be at most 32768")
        self._model_name_or_path = str(model_name_or_path)
        self._revision = revision
        self._compute_precision = compute_precision
        self._max_seq_length = window
        self._max_forward_tokens = max_forward_tokens
        self._model: DecisionModel | None = None
        self._processor: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None
        self._codes: tuple[str, ...] = ()
        self._marker: str | None = None
        self._marker_id: int | None = None
        self._escapes: tuple[str, ...] = ()

    def load(self, device: str) -> None:
        """Load the locally resolved checkpoint with Transformers 5's native architecture."""
        from transformers import AutoProcessor

        directory = resolve_checkpoint(
            self._model_name_or_path, revision=self._revision, allow_patterns=_CHECKPOINT_FILES
        )
        config = checkpoint_config(directory)
        install_transformers_security_fixes()
        processor = AutoProcessor.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        processor.tokenizer.padding_side = "left"
        validate_tokenizer(processor.tokenizer, config)
        dtype = self._resolve_dtype() if device.startswith("cuda") else torch.float32
        model = load_model(directory, config, device, dtype)
        self._model, self._processor = model, processor
        self._tokenizer, self._device = processor.tokenizer, device
        self._codes = tuple(config["codes"])
        if "joint" in config:
            marker = config["joint"]["marker"]
            self._marker = marker
            self._marker_id = processor.tokenizer.encode(marker, add_special_tokens=False)[0]
            specials = {*processor.tokenizer.all_special_tokens, marker}
            self._escapes = tuple(sorted(specials, key=len, reverse=True))

    def _state(self, item: Item) -> str:
        if item.images or item.audio is not None or item.video is not None or item.document is not None:
            raise InvalidInputError("Perplexity decisions currently accept text only")
        metadata = item.metadata or {}
        has_state = "state" in metadata
        if has_state == (item.text is not None):
            raise InvalidInputError("Supply either item.text or metadata.state")
        return describe(metadata["state"] if has_state else item.text, self._max_seq_length * 64)

    def _encode(self, chat: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        with self._tokenizer_guard():
            text = self._processor.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            return dict(self._processor(text=[text], images=None, padding=True, truncation=False, return_tensors="pt"))

    def _joint_rows(self, index: int, state: str, questions: tuple[Question, ...]) -> list[Row]:
        """One row for all questions, bisected (reference order) only when it exceeds the window."""
        if self._marker is None:
            raise RuntimeError(ERR_NOT_LOADED)
        encoded = self._encode(joint_messages(state, questions, self._codes, self._marker, self._escapes))
        ids = self._checked(encoded)
        if ids.shape[1] > self._max_seq_length:
            if len(questions) == 1:
                raise InputTooLongError("A decision question with its state exceeds the window; no input was truncated")
            middle = len(questions) // 2
            return self._joint_rows(index, state, questions[:middle]) + self._joint_rows(
                index, state, questions[middle:]
            )
        markers = tuple(int(position) for position in (ids[0] == self._marker_id).nonzero().flatten().tolist())
        if len(markers) != len(questions):
            raise ValueError("Readout markers do not match questions after tokenization")
        return [Row(index, questions, {name: tensor[0] for name, tensor in encoded.items()}, markers)]

    @staticmethod
    def _checked(encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        if set(encoded) - {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}:
            raise ValueError("Unexpected text processor inputs")
        if any(
            not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] != 1 for value in encoded.values()
        ):
            raise ValueError("Expected one complete text processor row")
        ids, mask = encoded["input_ids"], encoded["attention_mask"]
        if ids.shape != mask.shape or not mask.bool().all() or ids.shape[1] == 0:
            raise ValueError("Expected a nonempty unpadded text row")
        return ids

    def _item_rows(self, index: int, item: Item, questions: tuple[Question, ...]) -> list[Row]:
        state = self._state(item)
        if self._marker is not None:
            rows = self._joint_rows(index, state, questions)
            if sum(row.length for row in rows) > MAX_ITEM_TOKENS:
                raise InputTooLongError("Decision rows exceed the token budget; no input was truncated")
            return rows
        rows = []
        tokens = 0
        # Prepare one row at a time: a request with many oversized questions
        # cannot materialize an unbounded, heavily padded tokenizer batch.
        for question in questions:
            encoded = self._encode(messages(state, question, self._codes))
            length = self._checked(encoded).shape[1]
            tokens += length
            if length > self._max_seq_length or tokens > MAX_ITEM_TOKENS:
                raise InputTooLongError("Decision rows exceed the token budget; no input was truncated")
            rows.append(Row(index, (question,), {name: tensor[0] for name, tensor in encoded.items()}))
        return rows

    def _forward(self, rows: list[Row]) -> list[list[float]]:
        """Probabilities per question, in row then question order."""
        model = self._model
        if model is None:
            raise RuntimeError(ERR_NOT_LOADED)
        longest = max(row.length for row in rows)
        inputs: dict[str, torch.Tensor] = {}
        for name in rows[0].inputs:
            pad = self._tokenizer.pad_token_id if name == "input_ids" else 0
            inputs[name] = torch.stack(
                [torch.nn.functional.pad(row.inputs[name], (longest - row.length, 0), value=pad) for row in rows]
            ).to(self._device)
        counts = [len(question.keys) for row in rows for question in row.questions]
        positions = None
        if any(row.markers is not None for row in rows):
            if not all(row.markers is not None for row in rows):
                raise ValueError("Joint and single-question rows cannot share a forward")
            row_index = [i for i, row in enumerate(rows) for _ in row.questions]
            col_index = [longest - row.length + m for row in rows for m in row.markers or ()]
            positions = (
                torch.tensor(row_index, device=self._device),
                torch.tensor(col_index, device=self._device),
            )
        with torch.inference_mode():
            probabilities = model(inputs, counts, positions).cpu().tolist()
        return [values[:count] for values, count in zip(probabilities, counts, strict=True)]

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
        """Answer each item's typed questions, preserving per-item errors and order."""
        _ = prepared_items
        if self._model is None or self._processor is None:
            raise RuntimeError(ERR_NOT_LOADED)
        if labels is not None or instruction is not None:
            raise InvalidInputError("Use typed output_schema questions with their own instructions and criteria")
        if options is not None and not isinstance(options, dict):
            raise InvalidInputError("Decision options must be an object")
        opts = options or {}
        if set(opts) - {"overflow_policy"} or opts.get("overflow_policy", "error") != "error":
            raise InvalidInputError(
                "Perplexity decisions accept only overflow_policy='error'; checkpoint calibration is fixed"
            )
        questions = parse_questions(output_schema)
        rows: list[Row] = []
        errors: list[ExtractItemError | None] = [None] * len(items)
        counts = [0] * len(items)
        data: list[dict[str, Any]] = [{} for _ in items]
        for index, item in enumerate(items):
            try:
                rows.extend(self._item_rows(index, item, questions))
            except InputTooLongError as exc:
                errors[index] = ExtractItemError(code=ErrorCode.INPUT_TOO_LONG.value, message=str(exc))
            except InvalidInputError as exc:
                errors[index] = ExtractItemError(code=ErrorCode.INVALID_INPUT.value, message=str(exc))
        # Stable length sort reduces padding while the Row keeps its original
        # item/question mapping. A single item may span several bounded forwards.
        rows.sort(key=lambda row: row.length, reverse=True)
        start = 0
        while start < len(rows):
            end = start + 1
            while (
                end < len(rows)
                and end - start < MAX_FORWARD_ROWS
                and rows[start].length * (end - start + 1) <= self._max_forward_tokens
            ):
                end += 1
            batch = rows[start:end]
            values = iter(self._forward(batch))
            for row in batch:
                for question in row.questions:
                    data[row.item][question.id] = answer(question, next(values))
                counts[row.item] += row.length
            start = end
        # Restore the caller's question order as well as the item order.
        data = [{question.id: item[question.id] for question in questions if question.id in item} for item in data]
        return ExtractOutput(entities=[[] for _ in items], data=data, errors=errors, input_token_counts=counts)

    def extract_item_costs(
        self,
        items: list[Item],
        *,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[int] | None:
        """Estimate complete prompt characters without tokenizing on the event loop."""
        _ = kwargs
        try:
            questions = parse_questions(output_schema)
        except (InvalidInputError, InputTooLongError):
            return None
        # Include the repeated state, question, system prompt, option codes and
        # estimated chat-template overhead. This only groups work: _item_rows
        # and extract still enforce the exact unpadded and padded token limits.
        prompt_chars = sum(
            len(SYSTEM_PROMPT)
            + len(question.instructions)
            + sum(map(len, question.descriptions))
            + 6 * len(question.keys)
            + 256
            for question in questions
        )
        limit = MAX_ITEM_TOKENS * 4
        costs: list[int] = []
        for item in items:
            try:
                costs.append(min(limit, len(questions) * len(self._state(item)) + prompt_chars))
            except (InvalidInputError, InputTooLongError):
                # Let extraction report the original per-item validation error.
                costs.append(limit)
        return costs
