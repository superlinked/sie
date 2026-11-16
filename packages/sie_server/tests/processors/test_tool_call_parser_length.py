"""A tool call cut short by the token limit finishes as ``length``, not as a parse error."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from sie_server.adapters._generation_base import GenerationChunk
from sie_server.processors.tool_call_parser import parse_tool_call_stream

_CALL = '<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>'
# Ends on a prefix of the close tag, which the parser holds back in its text buffer.
_CUT_CALL = '<tool_call>{"name":"get_weather","arguments":{"city":"To</tool_c'


async def _chunks(items: list[GenerationChunk]) -> AsyncIterator[GenerationChunk]:
    for item in items:
        yield item


async def _parse(items: list[GenerationChunk], **kwargs: object) -> list[GenerationChunk]:
    return [item async for item in parse_tool_call_stream(_chunks(items), **kwargs)]  # type: ignore[arg-type]


def _text(out: list[GenerationChunk], choice: int = 0) -> str:
    return "".join(chunk.text_delta for chunk in out if chunk.choice_index == choice)


def _names(out: list[GenerationChunk], choice: int = 0) -> list[str]:
    return [
        chunk.tool_call_delta.function_name
        for chunk in out
        if chunk.choice_index == choice and chunk.tool_call_delta is not None and chunk.tool_call_delta.function_name
    ]


def _closures(out: list[GenerationChunk]) -> dict[int, str | None]:
    return {chunk.choice_index: chunk.finish_reason for chunk in out if not chunk.done and chunk.finish_reason}


def _assert_parse_error(out: list[GenerationChunk]) -> None:
    assert out[-1].done is True
    assert out[-1].finish_reason == "error"
    assert out[-1].error_code == "MODEL_OUTPUT_PARSE_ERROR"
    assert out[-1].error_message == "unterminated <tool_call> block"


@pytest.mark.asyncio
async def test_single_choice_cut_by_length_finishes_as_length() -> None:
    out = await _parse(
        [
            GenerationChunk(text_delta="Sure. "),
            GenerationChunk(text_delta=_CUT_CALL),
            GenerationChunk(text_delta="", done=True, finish_reason="length", prompt_tokens=7, completion_tokens=16),
        ]
    )

    assert [chunk.done for chunk in out].count(True) == 1
    terminal = out[-1]
    assert terminal.finish_reason == "length"
    assert terminal.error_code is None
    assert (terminal.prompt_tokens, terminal.completion_tokens) == (7, 16)
    # The partial call is dropped whole: no delta, and none of its text leaks.
    assert all(chunk.tool_call_delta is None for chunk in out)
    assert _text(out) == "Sure. "


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_reason", ["stop", None])
async def test_single_choice_unterminated_for_another_reason_stays_a_parse_error(finish_reason: str | None) -> None:
    out = await _parse(
        [
            GenerationChunk(text_delta="Sure. " + _CUT_CALL),
            GenerationChunk(text_delta="", done=True, finish_reason=finish_reason),  # type: ignore[arg-type]
        ]
    )

    _assert_parse_error(out)


@pytest.mark.asyncio
async def test_stream_ending_without_a_terminal_stays_a_parse_error() -> None:
    out = await _parse([GenerationChunk(text_delta=_CUT_CALL)])

    _assert_parse_error(out)


@pytest.mark.asyncio
async def test_complete_call_then_cut_call_keeps_the_first_and_finishes_as_length() -> None:
    out = await _parse(
        [
            GenerationChunk(text_delta=_CALL + _CUT_CALL),
            GenerationChunk(text_delta="", done=True, finish_reason="length"),
        ]
    )

    assert _names(out) == ["get_weather"]
    assert out[-1].done is True
    assert out[-1].finish_reason == "length"
    assert _text(out) == ""


@pytest.mark.asyncio
async def test_native_format_cut_by_length_finishes_as_length() -> None:
    out = await _parse(
        [
            GenerationChunk(text_delta="Sure. <|tool_call>call:f{a:1"),
            GenerationChunk(text_delta="", done=True, finish_reason="length"),
        ],
        tool_call_format="gemma4",
    )

    assert out[-1].finish_reason == "length"
    assert all(chunk.tool_call_delta is None for chunk in out)
    assert _text(out) == "Sure. "


@pytest.mark.asyncio
async def test_streamed_choices_close_independently_when_one_is_cut_by_length() -> None:
    out = await _parse(
        [
            GenerationChunk(text_delta="A " + _CUT_CALL, choice_index=0),
            GenerationChunk(text_delta="B " + _CALL + _CUT_CALL, choice_index=1),
            GenerationChunk(text_delta="C " + _CALL, choice_index=2),
            GenerationChunk(text_delta="", choice_index=0, finish_reason="length", completion_tokens=5),
            GenerationChunk(text_delta="", choice_index=1, finish_reason="length"),
            GenerationChunk(text_delta="", choice_index=2, finish_reason="length"),
            GenerationChunk(text_delta="", done=True, finish_reason="length", prompt_tokens=3, completion_tokens=30),
        ]
    )

    # Choice 2 was cut after a complete call, so it keeps ``tool_calls``.
    assert _closures(out) == {0: "length", 1: "length", 2: "tool_calls"}
    assert [_names(out, choice) for choice in range(3)] == [[], ["get_weather"], ["get_weather"]]
    assert [_text(out, choice) for choice in range(3)] == ["A ", "B ", "C "]
    closure_0 = next(chunk for chunk in out if chunk.choice_index == 0 and chunk.finish_reason == "length")
    assert closure_0.completion_tokens == 5
    assert out[-1].done is True
    assert out[-1].error_code is None
    assert (out[-1].prompt_tokens, out[-1].completion_tokens) == (3, 30)


@pytest.mark.asyncio
async def test_streamed_choice_unterminated_at_stop_stays_a_parse_error() -> None:
    out = await _parse(
        [
            GenerationChunk(text_delta=_CUT_CALL, choice_index=0),
            GenerationChunk(text_delta="fine", choice_index=1),
            GenerationChunk(text_delta="", choice_index=0, finish_reason="stop"),
            GenerationChunk(text_delta="", choice_index=1, finish_reason="length"),
            GenerationChunk(text_delta="", done=True, finish_reason="length"),
        ]
    )

    _assert_parse_error(out)


@pytest.mark.asyncio
async def test_streamed_choice_without_its_own_finish_uses_the_terminal_length() -> None:
    out = await _parse(
        [
            GenerationChunk(text_delta="A " + _CUT_CALL, choice_index=0),
            GenerationChunk(text_delta="B", choice_index=1),
            GenerationChunk(text_delta="", done=True, finish_reason="length"),
        ]
    )

    assert out[-1].done is True
    assert out[-1].finish_reason == "length"
    assert [_text(out, choice) for choice in range(2)] == ["A ", "B"]
    assert all(chunk.tool_call_delta is None for chunk in out)


@pytest.mark.asyncio
async def test_buffered_candidates_cut_by_length_drop_the_partial_call() -> None:
    candidates = (
        {"text": "A " + _CUT_CALL, "finish_reason": "length", "logprobs": None},
        {"text": "B " + _CALL + _CUT_CALL, "finish_reason": "length", "logprobs": None},
        {"text": "C " + _CUT_CALL, "finish_reason": "stop", "logprobs": None},
        {"text": "D " + _CALL, "finish_reason": "length", "logprobs": None},
    )
    out = await _parse([GenerationChunk(text_delta="", done=True, finish_reason="length", candidates=candidates)])

    assert out[-1].done is True
    assert out[-1].error_code is None
    cut, call_then_cut, unterminated_at_stop, complete = out[-1].candidates or ()
    assert cut == {"text": "A ", "finish_reason": "length", "logprobs": None}
    assert call_then_cut["text"] == "B "
    assert call_then_cut["finish_reason"] == "length"
    assert [call["function"]["name"] for call in call_then_cut["tool_calls"]] == ["get_weather"]
    # Not cut by the token limit: the malformed block stays in the text verbatim.
    assert unterminated_at_stop == {"text": "C " + _CUT_CALL, "finish_reason": "stop", "logprobs": None}
    assert complete["finish_reason"] == "tool_calls"
