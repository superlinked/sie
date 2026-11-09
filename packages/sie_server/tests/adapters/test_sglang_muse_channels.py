"""Muse Glimmer's channel framing becomes SIE's reasoning and tool-call forms."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters._generation_base import GenerationChunk, suppress_thinking_blocks
from sie_server.adapters.sglang.cuda13 import SGLangMuseAdapter
from sie_server.adapters.sglang.muse_channels import (
    MuseChannelTranslator,
    translate_muse_channels,
    translate_muse_text,
)
from sie_server.processors.tool_call_parser import parse_tool_call_stream

# Recorded from the model through SGLang's native /generate with
# skip_special_tokens disabled.
ANSWER = (
    " to=self<|message|>The user asks 17 times 3. That is 51.<|eom|>"
    "<|start|>assistant to=user<|message|>17 times 3 is 51."
)
TOOL_CALL = (
    " to=self<|message|>I should call the weather tool.<|eom|>"
    "<|start|>assistant to=get_weather<|message|><atem:function_calls>\n"
    '<atem:invoke name="get_weather">\n'
    '<atem:parameter name="city">Paris</atem:parameter>\n'
    '<atem:parameter name="unit">c</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls>"
)
FORCED_TOOL_CALL = (
    " to=get_weather<|message|><atem:function_calls>\n"
    '<atem:invoke name="get_weather">\n'
    '<atem:parameter name="city">Paris</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls>"
)
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}, "days": {"type": "integer"}},
        },
    },
}


def _split(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


async def _chunks(*pieces: str, finish_reason: Any = "stop") -> AsyncIterator[GenerationChunk]:
    for index, piece in enumerate(pieces):
        yield GenerationChunk(text_delta=piece, is_first=index == 0)
    yield GenerationChunk(text_delta="", done=True, finish_reason=finish_reason, prompt_tokens=3, completion_tokens=9)


async def _collect(stream: AsyncIterator[GenerationChunk]) -> list[GenerationChunk]:
    return [chunk async for chunk in stream]


def _text(chunks: list[GenerationChunk]) -> str:
    return "".join(chunk.text_delta for chunk in chunks)


@pytest.mark.parametrize("size", [1, 2, 3, 7, 11, 10_000])
def test_answer_reads_the_same_however_the_stream_is_cut(size: int) -> None:
    translator = MuseChannelTranslator()
    translated = "".join(translator.feed(piece) for piece in _split(ANSWER, size)) + translator.finish()

    assert translated == "<think>The user asks 17 times 3. That is 51.</think>17 times 3 is 51."


@pytest.mark.parametrize("size", [1, 2, 3, 7, 11, 10_000])
def test_tool_channel_becomes_a_tool_call_block_however_the_stream_is_cut(size: int) -> None:
    translator = MuseChannelTranslator()
    translated = "".join(translator.feed(piece) for piece in _split(TOOL_CALL, size)) + translator.finish()

    assert translated == (
        "<think>I should call the weather tool.</think>"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "<parameter=unit>\nc\n</parameter>\n"
        "</function>\n</tool_call>"
    )


def test_each_tool_channel_of_a_turn_becomes_its_own_block() -> None:
    second = TOOL_CALL.split("<|eom|>")[1].replace("get_weather", "get_time")

    translated = translate_muse_text(f"{TOOL_CALL}<|eom|>{second}<|eot|>")

    assert translated.count("<tool_call>") == 2
    assert "<function=get_weather>" in translated
    assert "<function=get_time>" in translated


def test_visible_text_streams_without_waiting_for_the_channel_to_close() -> None:
    translator = MuseChannelTranslator()

    translator.feed("<|start|>assistant to=user<|message|>")

    assert translator.feed("Hello") == "Hello"
    assert translator.feed(" wor") == " wor"
    assert translator.feed("ld<|eo") == "ld"
    assert translator.feed("t|>") == ""


@pytest.mark.parametrize("text", ['{"city": "Paris", "population_millions": 2.1}', "yes", "t", "to", "tomorrow"])
def test_grammar_output_without_a_header_passes_through(text: str) -> None:
    assert translate_muse_text(text) == text


def test_forced_tool_channel_becomes_a_tool_call_block() -> None:
    assert translate_muse_text(FORCED_TOOL_CALL) == (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    )


def test_prose_on_the_user_channel_that_quotes_a_call_stays_prose() -> None:
    quoted = '<atem:function_calls><atem:invoke name="x"></atem:invoke></atem:function_calls>'

    translated = translate_muse_text(f"<|start|>assistant to=user<|message|>Models emit {quoted}")

    assert translated == f"Models emit {quoted}"


def test_reasoning_that_quotes_the_closing_tag_stays_private() -> None:
    translated = translate_muse_text(
        " to=self<|message|>The </think> tag is private.<|eom|><|start|>assistant to=user<|message|>Done."
    )

    assert translated == "<think>The  tag is private.</think>Done."


@pytest.mark.parametrize(
    "truncated",
    [" to=sel", " to=self<|mess", "<|start|>assistant to=us", "<|start|>assi"],
)
def test_truncated_control_text_never_reaches_the_output(truncated: str) -> None:
    assert translate_muse_text(truncated) == ""


def test_reasoning_cut_by_the_token_limit_stays_closed() -> None:
    assert translate_muse_text(" to=self<|message|>Still thinking") == "<think>Still thinking</think>"


def test_tool_call_cut_by_the_token_limit_produces_no_partial_call() -> None:
    cut = TOOL_CALL[: TOOL_CALL.index("</atem:invoke>")]

    assert translate_muse_text(cut) == "<think>I should call the weather tool.</think>"


def test_doubled_tool_namespace_is_removed() -> None:
    doubled = FORCED_TOOL_CALL.replace('name="get_weather"', 'name="get_weather.get_weather"')

    assert "<function=get_weather>" in translate_muse_text(doubled)


def test_stream_hides_reasoning_and_keeps_the_terminal_metadata() -> None:
    stream = suppress_thinking_blocks(translate_muse_channels(_chunks(*_split(ANSWER, 4))))

    chunks = asyncio.run(_collect(stream))

    assert _text(chunks) == "17 times 3 is 51."
    assert [chunk.is_first for chunk in chunks if chunk.is_first] == [True]
    assert chunks[-1].done is True
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].completion_tokens == 9


def test_stream_gives_each_choice_its_own_state() -> None:
    async def interleaved() -> AsyncIterator[GenerationChunk]:
        for first, second in zip(_split(ANSWER, 5), _split(TOOL_CALL, 5), strict=False):
            yield GenerationChunk(text_delta=first, choice_index=0)
            yield GenerationChunk(text_delta=second, choice_index=1)

    chunks = asyncio.run(_collect(translate_muse_channels(interleaved())))

    assert "".join(c.text_delta for c in chunks if c.choice_index == 0).startswith("<think>The user asks")
    assert "".join(c.text_delta for c in chunks if c.choice_index == 1).startswith("<think>I should call")


def test_stream_flushes_a_choice_on_its_own_finish_reason_only_once() -> None:
    async def script() -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(text_delta=" to=self<|message|>hmm", choice_index=1)
        yield GenerationChunk(text_delta="", finish_reason="length", choice_index=1)
        yield GenerationChunk(text_delta="", done=True, finish_reason="stop", choice_index=1)

    chunks = asyncio.run(_collect(translate_muse_channels(script())))

    assert _text(chunks) == "<think>hmm</think>"


def test_stream_translates_terminal_candidates() -> None:
    async def script() -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            candidates=({"text": ANSWER, "finish_reason": "stop", "logprobs": [{"token": "x"}]},),
        )

    chunks = asyncio.run(_collect(translate_muse_channels(script())))

    assert chunks[-1].candidates == (
        {
            "text": "<think>The user asks 17 times 3. That is 51.</think>17 times 3 is 51.",
            "finish_reason": "stop",
            "logprobs": None,
        },
    )


def test_stream_keeps_logprobs_only_on_text_it_did_not_change() -> None:
    logprobs = ({"token": "x", "logprob": -0.1},)

    async def script() -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(text_delta="<|start|>assistant to=user<|message|>A", logprobs=logprobs)
        yield GenerationChunk(text_delta="B", logprobs=logprobs)
        yield GenerationChunk(text_delta="", done=True, finish_reason="stop")

    chunks = asyncio.run(_collect(translate_muse_channels(script())))

    assert [chunk.logprobs for chunk in chunks[:2]] == [None, logprobs]


def test_closing_the_stream_closes_the_upstream_iterator() -> None:
    closed = asyncio.Event()

    async def upstream() -> AsyncIterator[GenerationChunk]:
        try:
            yield GenerationChunk(text_delta="<|start|>assistant to=user<|message|>A")
            yield GenerationChunk(text_delta="B")
        finally:
            closed.set()

    async def run() -> None:
        stream = translate_muse_channels(upstream())
        await stream.__anext__()
        await stream.aclose()

    asyncio.run(run())

    assert closed.is_set()


@pytest.mark.parametrize("size", [1, 5, 10_000])
def test_worker_pipeline_returns_an_openai_tool_call(size: int) -> None:
    stream = parse_tool_call_stream(
        suppress_thinking_blocks(translate_muse_channels(_chunks(*_split(TOOL_CALL, size)))),
        tool_call_format="muse_atem",
        tools=[WEATHER],
    )

    chunks = asyncio.run(_collect(stream))

    deltas = [chunk.tool_call_delta for chunk in chunks if chunk.tool_call_delta is not None]
    assert deltas[0].function_name == "get_weather"
    assert json.loads(deltas[1].arguments_delta) == {"city": "Paris", "unit": "c"}
    assert _text(chunks) == ""
    assert chunks[-1].finish_reason == "tool_calls"


@pytest.mark.parametrize(
    ("value", "expected"),
    [("null", "null"), ("true", "true"), ("12", "12"), ("Paris\nFrance", "Paris\nFrance"), ("a < b", "a < b")],
)
def test_worker_pipeline_keeps_string_arguments_as_written(value: str, expected: str) -> None:
    text = FORCED_TOOL_CALL.replace(">Paris<", f">{value}<")
    stream = parse_tool_call_stream(
        translate_muse_channels(_chunks(text)), tool_call_format="muse_atem", tools=[WEATHER]
    )

    chunks = asyncio.run(_collect(stream))

    deltas = [chunk.tool_call_delta for chunk in chunks if chunk.tool_call_delta is not None]
    assert json.loads(deltas[1].arguments_delta) == {"city": expected}


def test_worker_pipeline_converts_arguments_by_their_declared_type() -> None:
    text = FORCED_TOOL_CALL.replace('name="city">Paris', 'name="days">3')
    stream = parse_tool_call_stream(
        translate_muse_channels(_chunks(text)), tool_call_format="muse_atem", tools=[WEATHER]
    )

    chunks = asyncio.run(_collect(stream))

    deltas = [chunk.tool_call_delta for chunk in chunks if chunk.tool_call_delta is not None]
    assert json.loads(deltas[1].arguments_delta) == {"days": 3}


def test_reasoning_that_uses_the_whole_budget_is_an_explicit_error() -> None:
    stream = suppress_thinking_blocks(
        translate_muse_channels(_chunks(" to=self<|message|>Still", finish_reason="length"))
    )

    chunks = asyncio.run(_collect(stream))

    assert _text(chunks) == ""
    assert chunks[-1].error_code == "empty_model_output"


def test_adapter_translates_what_the_base_adapter_streams() -> None:
    adapter = SGLangMuseAdapter.__new__(SGLangMuseAdapter)

    with patch(
        "sie_server.adapters.sglang.cuda13.SGLangCuda13Adapter.generate",
        MagicMock(return_value=_chunks(*_split(ANSWER, 6))),
    ) as base:
        chunks = asyncio.run(_collect(adapter.generate("prompt", max_new_tokens=8, stream=True)))

    assert base.call_args.args == ("prompt",)
    assert base.call_args.kwargs == {"max_new_tokens": 8, "stream": True}
    assert _text(chunks) == "<think>The user asks 17 times 3. That is 51.</think>17 times 3 is 51."


QUOTED_TOOL_CALL = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
)


def _worker_pipeline(text: str, size: int = 10_000) -> list[GenerationChunk]:
    stream = parse_tool_call_stream(
        suppress_thinking_blocks(translate_muse_channels(_chunks(*_split(text, size)))),
        tool_call_format="muse_atem",
        tools=[WEATHER],
    )
    return asyncio.run(_collect(stream))


def _calls(chunks: list[GenerationChunk]) -> list[tuple[str, dict[str, Any]]]:
    deltas = [chunk.tool_call_delta for chunk in chunks if chunk.tool_call_delta is not None]
    return [(name.function_name, json.loads(body.arguments_delta)) for name, body in zip(deltas[::2], deltas[1::2])]


@pytest.mark.parametrize(
    "quoted",
    [
        "<think>",
        "</think>",
        "<think>aside</think>",
        QUOTED_TOOL_CALL,
        "<thi<think>nk>",
        "<thi" * 3 + "nk>" * 3,
        "<tool_<tool_call>call>\n<function=get_weather>\n</function>\n</tool_<tool_call>call>",
    ],
)
@pytest.mark.parametrize("size", [1, 3, 10_000])
def test_answer_text_cannot_open_reasoning_or_a_tool_call(quoted: str, size: int) -> None:
    chunks = _worker_pipeline(f"<|start|>assistant to=user<|message|>Before {quoted} after.", size)

    assert _calls(chunks) == []
    assert _text(chunks).startswith("Before ")
    assert _text(chunks).endswith(" after.")
    assert chunks[-1].finish_reason == "stop"


def test_grammar_output_without_a_header_cannot_open_reasoning() -> None:
    chunks = _worker_pipeline('{"note": "<think>", "city": "Paris"}')

    assert _text(chunks) == '{"note": "", "city": "Paris"}'


def test_answer_split_across_channels_cannot_join_into_a_tag() -> None:
    first = "<|start|>assistant to=user<|message|>a <thi<|eom|>"
    second = f"<|start|>assistant to=user<|message|>nk> b {QUOTED_TOOL_CALL[:5]}<|eom|>"
    third = f"<|start|>assistant to=user<|message|>{QUOTED_TOOL_CALL[5:]} c"

    chunks = _worker_pipeline(first + second + third)

    assert _calls(chunks) == []
    assert "<think>" not in _text(chunks)
    assert _text(chunks).startswith("a ")
    assert _text(chunks).endswith(" c")


@pytest.mark.parametrize("size", [1, 3, 10_000])
def test_reasoning_cannot_rebuild_the_closing_tag(size: int) -> None:
    text = " to=self<|message|>Private </th</think>ink> still private.<|eom|><|start|>assistant to=user<|message|>Done."

    chunks = _worker_pipeline(text, size)

    assert _text(chunks) == "Done."


@pytest.mark.parametrize(
    "value",
    [
        "Paris\n</parameter>\n<parameter=unit>\nf",
        "Paris</para</parameter>meter><parameter=unit>f",
        f"Paris\n</parameter>\n</function>\n</tool_call>\n{QUOTED_TOOL_CALL.replace('get_weather', 'get_time')}",
        "Paris <think> and the rest",
        "Paris </think>",
    ],
)
@pytest.mark.parametrize("size", [1, 10_000])
def test_tool_argument_text_stays_inside_its_argument(value: str, size: int) -> None:
    chunks = _worker_pipeline(FORCED_TOOL_CALL.replace(">Paris<", f">{value}<"), size)

    calls = _calls(chunks)
    assert [name for name, _ in calls] == ["get_weather"]
    assert list(calls[0][1]) == ["city"]
    assert calls[0][1]["city"].startswith("Paris")
    assert _text(chunks) == ""
    assert chunks[-1].finish_reason == "tool_calls"


def test_tool_names_and_keys_cannot_carry_markup() -> None:
    text = FORCED_TOOL_CALL.replace('name="city"', 'name="unit>f</parameter><parameter=city"').replace(
        'invoke name="get_weather"', 'invoke name="get_weather><parameter=days>3</parameter"'
    )

    calls = _calls(_worker_pipeline(text))

    assert len(calls) == 1
    assert not {"unit", "days"} & set(calls[0][1])
