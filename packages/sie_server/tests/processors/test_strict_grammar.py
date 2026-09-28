"""Tests for strict structured-output verification."""

from __future__ import annotations

import time
import urllib.request
from collections.abc import AsyncIterator

import pytest
from sie_server.adapters._generation_base import GenerationChunk, ToolCallDelta
from sie_server.processors import strict_grammar
from sie_server.processors.strict_grammar import (
    MODEL_OUTPUT_PARSE_ERROR,
    enforce_strict_grammar,
    output_violation,
    requires_output_validation,
)
from sie_server.types.grammar import GrammarSpec

_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "pattern": "^[A-Z]"},
        "count": {"type": "integer"},
    },
    "required": ["name", "count"],
    "additionalProperties": False,
}
_STRICT_SCHEMA = GrammarSpec(kind="json_schema", value=_SCHEMA, strict=True)
_STRICT_REGEX = GrammarSpec(kind="regex", value=r"[A-Z]{3}-\d{4}", strict=True)


async def _stream(chunks: list[GenerationChunk]) -> AsyncIterator[GenerationChunk]:
    for chunk in chunks:
        yield chunk


async def _drain(chunks: AsyncIterator[GenerationChunk]) -> list[GenerationChunk]:
    return [chunk async for chunk in chunks]


def _completion(*deltas: str, finish_reason: str = "stop") -> list[GenerationChunk]:
    chunks = [GenerationChunk(text_delta=delta) for delta in deltas]
    chunks.append(
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason=finish_reason,  # type: ignore[arg-type]
            prompt_tokens=3,
            completion_tokens=len(deltas),
        )
    )
    return chunks


@pytest.mark.parametrize(
    ("grammar", "expected"),
    [
        (None, False),
        (GrammarSpec(kind="json_schema", value=_SCHEMA), False),
        (GrammarSpec(kind="json_schema", value=_SCHEMA, strict=False), False),
        (_STRICT_SCHEMA, True),
        (_STRICT_REGEX, True),
        (GrammarSpec(kind="ebnf", value='root ::= "a"', strict=True), False),
    ],
)
def test_requires_output_validation(grammar: GrammarSpec | None, expected: bool) -> None:
    assert requires_output_validation(grammar) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"name": "Ada", "count": 2}', None),
        ('  {"name": "Ada", "count": 2}\n', None),
        ("[]", "generated output does not match the requested JSON schema at $ ('type' keyword)"),
        ('{"name": "Ada"}', "generated output does not match the requested JSON schema at $ ('required' keyword)"),
        (
            '{"name": "ada", "count": 2}',
            "generated output does not match the requested JSON schema at $.name ('pattern' keyword)",
        ),
        (
            '{"name": "Ada", "count": 2, "extra": true}',
            "generated output does not match the requested JSON schema at $ ('additionalProperties' keyword)",
        ),
        ('{"name": "Ada", "count": ', "generated output is not valid JSON"),
        ('{"name": "Ada", "count": NaN}', "generated output is not valid JSON"),
        ("", "generated output is not valid JSON"),
    ],
)
def test_json_schema_output_violation(text: str, expected: str | None) -> None:
    assert output_violation(_STRICT_SCHEMA, text) == expected


def test_json_schema_pattern_properties_use_the_bounded_engine() -> None:
    grammar = GrammarSpec(
        kind="json_schema",
        value={
            "type": "object",
            "patternProperties": {"^x-": {"type": "string"}},
            "additionalProperties": False,
        },
        strict=True,
    )

    assert output_violation(grammar, '{"x-trace": "abc"}') is None
    assert output_violation(grammar, '{"x-trace": 1}') == (
        "generated output does not match the requested JSON schema at $.x-trace ('type' keyword)"
    )
    assert output_violation(grammar, '{"other": "abc"}') == (
        "generated output does not match the requested JSON schema at $ ('additionalProperties' keyword)"
    )


def test_json_schema_internal_refs_resolve_and_remote_refs_are_never_fetched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("strict grammar verification must not fetch remote schemas")

    monkeypatch.setattr(urllib.request, "urlopen", refuse_network)
    local = GrammarSpec(
        kind="json_schema",
        value={"$defs": {"item": {"type": "integer"}}, "type": "array", "items": {"$ref": "#/$defs/item"}},
        strict=True,
    )
    remote = GrammarSpec(kind="json_schema", value={"$ref": "https://schemas.example.com/item.json"}, strict=True)

    assert output_violation(local, "[1, 2]") is None
    assert output_violation(local, '[1, "2"]') == (
        "generated output does not match the requested JSON schema at $[1] ('type' keyword)"
    )
    assert output_violation(remote, "{}") == "generated output could not be verified against the requested grammar"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ABC-1234", None),
        ("ABC-1234 ", "generated output does not match the requested regex"),
        ("xABC-1234", "generated output does not match the requested regex"),
    ],
)
def test_regex_output_violation_requires_a_full_match(text: str, expected: str | None) -> None:
    assert output_violation(_STRICT_REGEX, text) == expected


def test_pathological_pattern_is_bounded_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(strict_grammar, "_PATTERN_BUDGET_S", 0.2)
    grammar = GrammarSpec(kind="regex", value=r"(a|aa)+$", strict=True)

    started = time.monotonic()
    violation = output_violation(grammar, "a" * 64 + "!")

    assert violation == "generated output could not be verified against the requested grammar"
    assert time.monotonic() - started < 5


@pytest.mark.asyncio
async def test_strict_violation_replaces_the_stop_terminal() -> None:
    chunks = await _drain(enforce_strict_grammar(_stream(_completion('{"name": ', '"Ada"}')), _STRICT_SCHEMA))

    assert [chunk.text_delta for chunk in chunks[:-1]] == ['{"name": ', '"Ada"}']
    terminal = chunks[-1]
    assert terminal.done is True
    assert terminal.finish_reason == "error"
    assert terminal.error_code == MODEL_OUTPUT_PARSE_ERROR
    assert terminal.error_message == (
        "generated output does not match the requested JSON schema at $ ('required' keyword)"
    )
    assert terminal.prompt_tokens == 3
    assert terminal.completion_tokens == 2


@pytest.mark.asyncio
async def test_conforming_strict_output_keeps_the_stop_terminal() -> None:
    chunks = await _drain(
        enforce_strict_grammar(_stream(_completion('{"name": "Ada", ', '"count": 1}')), _STRICT_SCHEMA)
    )

    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].error_code is None


@pytest.mark.parametrize(
    "grammar",
    [
        None,
        GrammarSpec(kind="json_schema", value=_SCHEMA),
        GrammarSpec(kind="json_schema", value=_SCHEMA, strict=False),
    ],
)
def test_non_strict_grammars_return_the_stream_unchanged(grammar: GrammarSpec | None) -> None:
    upstream = _stream(_completion("not json"))

    assert enforce_strict_grammar(upstream, grammar) is upstream


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["length", "cancelled", "error"])
async def test_non_stop_terminals_are_not_verified(finish_reason: str) -> None:
    chunks = await _drain(
        enforce_strict_grammar(_stream(_completion('{"name": ', finish_reason=finish_reason)), _STRICT_SCHEMA)
    )

    assert chunks[-1].finish_reason == finish_reason
    assert chunks[-1].error_code is None


@pytest.mark.asyncio
async def test_existing_terminal_errors_are_preserved() -> None:
    upstream = [
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="error",
            error_code="empty_model_output",
            error_message="model produced no visible output text",
        )
    ]

    chunks = await _drain(enforce_strict_grammar(_stream(upstream), _STRICT_SCHEMA))

    assert chunks[-1].error_code == "empty_model_output"


@pytest.mark.asyncio
async def test_tool_call_turns_are_not_verified() -> None:
    upstream = [
        GenerationChunk(text_delta="", tool_call_delta=ToolCallDelta(index=0, id="call_1", function_name="lookup")),
        GenerationChunk(text_delta="", tool_call_delta=ToolCallDelta(index=0, arguments_delta='{"q": "x"}')),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    ]

    chunks = await _drain(enforce_strict_grammar(_stream(upstream), _STRICT_SCHEMA))

    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].error_code is None


@pytest.mark.asyncio
async def test_every_candidate_is_verified() -> None:
    terminal = GenerationChunk(
        text_delta="",
        done=True,
        finish_reason="stop",
        candidates=(
            {"text": '{"name": "Ada", "count": 1}', "finish_reason": "stop", "logprobs": None},
            {"text": '{"name": "Ada"}', "finish_reason": "stop", "logprobs": None},
            {"text": '{"name": ', "finish_reason": "length", "logprobs": None},
        ),
    )

    chunks = await _drain(enforce_strict_grammar(_stream([terminal]), _STRICT_SCHEMA))

    assert chunks[-1].error_code == MODEL_OUTPUT_PARSE_ERROR
    assert "'required' keyword" in (chunks[-1].error_message or "")


@pytest.mark.asyncio
async def test_shared_tool_call_terminal_still_verifies_choices_that_stopped() -> None:
    terminal = GenerationChunk(
        text_delta="",
        done=True,
        finish_reason="tool_calls",
        candidates=(
            {
                "text": "",
                "finish_reason": "tool_calls",
                "logprobs": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"text": '{"name": "Ada"}', "finish_reason": "stop", "logprobs": None},
        ),
    )

    chunks = await _drain(enforce_strict_grammar(_stream([terminal]), _STRICT_SCHEMA))

    assert chunks[-1].error_code == MODEL_OUTPUT_PARSE_ERROR
    assert "'required' keyword" in (chunks[-1].error_message or "")


@pytest.mark.asyncio
async def test_streamed_tool_call_terminal_still_verifies_other_choices() -> None:
    upstream = [
        GenerationChunk(
            text_delta="", choice_index=0, tool_call_delta=ToolCallDelta(index=0, id="c", function_name="f")
        ),
        GenerationChunk(text_delta="", choice_index=0, finish_reason="tool_calls"),
        GenerationChunk(text_delta="not a code", choice_index=1),
        GenerationChunk(text_delta="", choice_index=1, finish_reason="stop"),
        GenerationChunk(text_delta="", done=True, finish_reason="tool_calls"),
    ]

    chunks = await _drain(enforce_strict_grammar(_stream(upstream), _STRICT_REGEX))

    assert chunks[-1].error_code == MODEL_OUTPUT_PARSE_ERROR


@pytest.mark.asyncio
async def test_streamed_choices_are_verified_independently() -> None:
    upstream = [
        GenerationChunk(text_delta="ABC-", choice_index=0),
        GenerationChunk(text_delta="XYZ-", choice_index=1),
        GenerationChunk(text_delta="1234", choice_index=0),
        GenerationChunk(text_delta="12", choice_index=1),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    ]

    chunks = await _drain(enforce_strict_grammar(_stream(upstream), _STRICT_REGEX))

    assert chunks[-1].error_code == MODEL_OUTPUT_PARSE_ERROR
    assert chunks[-1].error_message == "generated output does not match the requested regex"


@pytest.mark.asyncio
async def test_streamed_choice_finish_reasons_exclude_truncated_and_tool_call_choices() -> None:
    upstream = [
        GenerationChunk(text_delta="ABC-1234", choice_index=0),
        GenerationChunk(text_delta="XYZ-", choice_index=1),
        GenerationChunk(text_delta="", choice_index=0, finish_reason="stop"),
        GenerationChunk(text_delta="12", choice_index=1, finish_reason="length"),
        GenerationChunk(
            text_delta="",
            choice_index=2,
            tool_call_delta=ToolCallDelta(index=0, id="call_1", function_name="lookup"),
        ),
        GenerationChunk(text_delta="", choice_index=2, finish_reason="stop"),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    ]

    chunks = await _drain(enforce_strict_grammar(_stream(upstream), _STRICT_REGEX))

    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].error_code is None


@pytest.mark.asyncio
async def test_streamed_choice_that_stops_is_still_verified() -> None:
    upstream = [
        GenerationChunk(text_delta="ABC-1234", choice_index=0, finish_reason="length"),
        GenerationChunk(text_delta="not a code", choice_index=1),
        GenerationChunk(text_delta="", choice_index=1, finish_reason="stop"),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    ]

    chunks = await _drain(enforce_strict_grammar(_stream(upstream), _STRICT_REGEX))

    assert chunks[-1].error_code == MODEL_OUTPUT_PARSE_ERROR


@pytest.mark.asyncio
async def test_closing_the_wrapper_closes_the_upstream_iterator() -> None:
    closed = False

    async def upstream() -> AsyncIterator[GenerationChunk]:
        nonlocal closed
        try:
            yield GenerationChunk(text_delta="{")
            yield GenerationChunk(text_delta="}")
        finally:
            closed = True

    wrapped = enforce_strict_grammar(upstream(), _STRICT_SCHEMA)
    assert (await anext(wrapped)).text_delta == "{"
    await wrapped.aclose()  # type: ignore[attr-defined]

    assert closed is True
