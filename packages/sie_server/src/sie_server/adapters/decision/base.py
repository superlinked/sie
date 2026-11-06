"""Admission and batching shared by native and engine-backed decision adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters.decision.classification import classification_schema, classifications
from sie_server.adapters.decision.questions import Question, answer, describe, parse_questions
from sie_server.adapters.errors import InputTooLongError
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import InvalidInputError, Item
from sie_server.types.responses import ErrorCode

MAX_ITEM_TOKENS = 32_768
MAX_FORWARD_ROWS = 8


def check_token_budget(length: int, total: int, window: int) -> int:
    total += length
    if length > window or total > MAX_ITEM_TOKENS:
        raise InputTooLongError("Decision rows exceed the token budget; no input was truncated")
    return total


@dataclass
class Row:
    """An unpadded forward row and its original item/question coordinates."""

    item: int
    questions: tuple[Question, ...]
    inputs: dict[str, torch.Tensor]
    markers: tuple[int, ...] | None = None
    prefix_length: int | None = None
    temperature: float = 1.0

    @property
    def question(self) -> Question:
        return self.questions[0]

    @property
    def length(self) -> int:
        return self.inputs["input_ids"].shape[0]


class DecisionBatchAdapter(BaseAdapter):
    """Bound complete items before running any of their decision rows."""

    _max_seq_length: int
    _max_forward_tokens: int

    def _check_decision_loaded(self) -> None:
        raise NotImplementedError

    def _item_rows(self, index: int, item: Item, questions: tuple[Question, ...]) -> list[Row]:
        raise NotImplementedError

    def _forward(self, rows: list[Row]) -> list[list[float]]:
        raise NotImplementedError

    def _parse_questions(self, schema: dict[str, Any] | None) -> tuple[Question, ...]:
        return parse_questions(schema)

    def _answer(self, question: Question, probabilities: list[float]) -> dict[str, Any]:
        return answer(question, probabilities)

    def _extract_questions(
        self,
        labels: list[str] | None,
        output_schema: dict[str, Any] | None,
        instruction: str | None,
        options: dict[str, Any] | None,
    ) -> tuple[Question, ...]:
        if instruction is not None:
            raise InvalidInputError("Use classification labels or typed questions with their own instructions")
        if options is not None and not isinstance(options, dict):
            raise InvalidInputError("Decision options must be an object")
        opts = options or {}
        allowed = {"overflow_policy"}
        if labels is not None:
            allowed.update({"hypothesis_template", "multi_label"})
        if set(opts) - allowed or opts.get("overflow_policy", "error") != "error":
            raise InvalidInputError(
                "Unsupported decision options; overflow_policy must be 'error' and calibration is fixed"
            )
        if labels is not None:
            if output_schema is not None:
                raise InvalidInputError("Supply either classification labels or typed output_schema questions")
            output_schema = classification_schema(labels, opts)
        return self._parse_questions(output_schema)

    def _raw_state(self, item: Item) -> Any:
        if item.images or item.audio is not None or item.video is not None or item.document is not None:
            raise InvalidInputError("Decisions currently accept text only")
        metadata = item.metadata or {}
        has_state = "state" in metadata
        if has_state == (item.text is not None):
            raise InvalidInputError("Supply either item.text or metadata.state")
        state = metadata["state"] if has_state else item.text
        describe(state, self._max_seq_length * 64)
        return state

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
        _ = prepared_items
        self._check_decision_loaded()
        questions = self._extract_questions(labels, output_schema, instruction, options)
        rows: list[Row] = []
        errors: list[ExtractItemError | None] = [None] * len(items)
        counts = [0] * len(items)
        data: list[dict[str, Any]] = [{} for _ in items]
        for index, item in enumerate(items):
            try:
                item_rows = self._item_rows(index, item, questions)
                if (
                    any(row.length > self._max_seq_length for row in item_rows)
                    or sum(row.length for row in item_rows) > MAX_ITEM_TOKENS
                ):
                    raise InputTooLongError("Decision rows exceed the token budget; no input was truncated")
                rows.extend(item_rows)
            except InputTooLongError as exc:
                errors[index] = ExtractItemError(code=ErrorCode.INPUT_TOO_LONG.value, message=str(exc))
            except InvalidInputError as exc:
                errors[index] = ExtractItemError(code=ErrorCode.INVALID_INPUT.value, message=str(exc))
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
            expected = [(row, question) for row in batch for question in row.questions]
            for (row, question), probabilities in zip(expected, self._forward(batch), strict=True):
                data[row.item][question.id] = self._answer(question, probabilities)
            for row in batch:
                counts[row.item] += row.length
            start = end
        data = [{question.id: item[question.id] for question in questions if question.id in item} for item in data]
        return ExtractOutput(
            entities=[[] for _ in items],
            classifications=classifications(data, labels, questions) if labels is not None else None,
            data=data if labels is None else None,
            errors=errors,
            input_token_counts=counts,
        )

    def extract_item_costs(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[int] | None:
        _ = kwargs
        try:
            questions = self._extract_questions(labels, output_schema, instruction, options)
        except (InvalidInputError, InputTooLongError):
            return None
        prompt_chars = sum(
            len(question.instructions) + sum(map(len, question.descriptions)) + 8 * len(question.keys) + 512
            for question in questions
        )
        limit = MAX_ITEM_TOKENS * 4
        costs: list[int] = []
        for item in items:
            try:
                state = describe(self._raw_state(item), self._max_seq_length * 64)
                costs.append(min(limit, 2 * (len(questions) * len(state) + prompt_chars)))
            except (InvalidInputError, InputTooLongError):
                costs.append(limit)
        return costs
