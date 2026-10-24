"""The Messages translation replays the vectors the gateway replays.

Every file under ``conformance/anthropic_messages/`` is read by the gateway's
tests and by these, so a request is accepted, refused and answered the same
way on both topologies. A translated request must also pass this server's own
chat validation, as the gateway checks it against its chat seam.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from sie_server.api.anthropic_translation import (
    MAX_TOOLS,
    ErrorContext,
    FrameParser,
    MessagesError,
    StreamMachine,
    anthropic_error,
    chat_completion_to_message,
    messages_to_chat,
    stream_events,
)
from sie_server.api.openai_local import (
    _validate_chat_message_media,
    _validate_chat_messages,
    _validate_cuda_chat_body,
)

VECTORS = Path(__file__).resolve().parents[4] / "conformance" / "anthropic_messages"


def _vectors(name: str) -> Any:
    return json.loads((VECTORS / name).read_text(encoding="utf-8"))


def _chat_route_accepts(chat: dict[str, Any]) -> None:
    _validate_chat_messages(chat["messages"])
    _validate_chat_message_media(chat["messages"])
    _validate_cuda_chat_body(chat)


def _frames(case: dict[str, Any]) -> list[str]:
    return [frame if isinstance(frame, str) else json.dumps(frame) for frame in case["frames"]]


def _named(events: list[Any]) -> list[dict[str, Any]]:
    return [{"event": event.name, "data": event.data} for event in events]


def _parse_events(body: bytes) -> list[dict[str, Any]]:
    parsed = []
    for frame in body.decode().split("\n\n"):
        if not frame.strip():
            continue
        name, data = "", None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        assert data is not None, frame
        assert data["type"] == name, frame
        parsed.append({"event": name, "data": data})
    return parsed


async def _drain(body: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in body])


@pytest.mark.parametrize("case", _vectors("request_vectors.json")["accepted"], ids=lambda case: case["name"])
def test_an_accepted_request_vector_translates_onto_the_chat_body(case: dict[str, Any]) -> None:
    translation = messages_to_chat(case["body"], case.get("anthropic_version"))

    assert translation.chat == case["chat"]
    assert translation.first_image == case.get("first_image")
    _chat_route_accepts(translation.chat)


@pytest.mark.parametrize("case", _vectors("request_vectors.json")["rejected"], ids=lambda case: case["name"])
def test_a_rejected_request_vector_is_refused_with_its_wording(case: dict[str, Any]) -> None:
    with pytest.raises(MessagesError) as refused:
        messages_to_chat(case["body"], case.get("anthropic_version"))

    assert refused.value.status == 400
    assert refused.value.message == case["message"]


@pytest.mark.parametrize("case", _vectors("claude_code_requests.json")["cases"], ids=lambda case: case["name"])
def test_a_captured_claude_code_request_translates_or_fails_with_the_recovery_wording(case: dict[str, Any]) -> None:
    expect = case["expect"]
    version = case["headers"].get("anthropic-version")
    if "error" in expect:
        with pytest.raises(MessagesError) as refused:
            messages_to_chat(case["body"], version)
        assert refused.value.message == expect["error"]
        return
    translation = messages_to_chat(case["body"], version)
    chat = expect["chat"]
    assert [message["role"] for message in translation.chat["messages"]] == chat["roles"]
    assert translation.chat["stream"] == chat["stream"]
    assert translation.chat["max_tokens"] == chat["max_tokens"]
    assert translation.chat["include_reasoning"] == chat["include_reasoning"]
    assert len(translation.chat.get("tools") or []) == chat["tool_count"]
    assert translation.chat.get("stop") == chat.get("stop")
    _chat_route_accepts(translation.chat)


def test_tool_count_over_the_limit_is_refused() -> None:
    tools = [{"name": f"tool_{index}", "input_schema": {"type": "object"}} for index in range(MAX_TOOLS + 1)]
    body = {"model": "m", "max_tokens": 1, "tools": tools, "messages": [{"role": "user", "content": "x"}]}

    with pytest.raises(MessagesError) as refused:
        messages_to_chat(body, None)

    assert refused.value.message == "tools: at most 64 tools are supported, got 65"


@pytest.mark.parametrize("case", _vectors("response_vectors.json")["translated"], ids=lambda case: case["name"])
def test_a_response_vector_translates_the_chat_completion(case: dict[str, Any]) -> None:
    assert chat_completion_to_message(case["chat"], case["model"], case["request_id"]) == case["message"]


@pytest.mark.parametrize("case", _vectors("response_vectors.json")["faults"], ids=lambda case: case["name"])
def test_a_faulty_chat_completion_is_not_translated(case: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"^(chat completion|tool call)"):
        chat_completion_to_message(case["chat"], "m", "r")


@pytest.mark.parametrize("case", _vectors("stream_vectors.json")["streams"], ids=lambda case: case["name"])
def test_a_stream_vector_drives_the_state_machine(case: dict[str, Any]) -> None:
    machine = StreamMachine(case["message_id"], case["model"])
    events = machine.start()
    for frame in _frames(case):
        events += machine.on_data(frame)
    events += machine.on_end()

    assert _named(events) == case["events"]


@pytest.mark.parametrize("case", _vectors("stream_vectors.json")["streams"], ids=lambda case: case["name"])
async def test_a_stream_vector_survives_the_sse_body_split_at_any_byte(case: dict[str, Any]) -> None:
    sse = "".join(f"data: {frame}\n\n:\n\n" for frame in _frames(case)).encode()

    async def chunks() -> AsyncIterator[bytes]:
        for start in range(0, len(sse), 7):
            yield sse[start : start + 7]

    body = stream_events(chunks(), StreamMachine(case["message_id"], case["model"]), ping_interval=60.0)

    assert _parse_events(await _drain(body)) == case["events"]


def test_the_frame_parser_joins_multi_line_data_and_skips_comments() -> None:
    parser = FrameParser()

    assert parser.push(b': keep-alive\r\n\r\ndata: {"a"') == []
    assert parser.push(b":1}\r\n\r\nevent: x\ndata: one\ndata: two\n\n") == ['{"a":1}', "one\ntwo"]


@pytest.mark.parametrize("case", _vectors("error_vectors.json")["errors"], ids=lambda case: case["name"])
def test_an_error_vector_renders_the_anthropic_envelope(case: dict[str, Any]) -> None:
    body = b"" if case["body"] is None else json.dumps(case["body"]).encode()
    headers, content = anthropic_error(
        case["status"],
        list((case.get("headers") or {}).items()),
        body,
        ErrorContext(first_image=case.get("first_image")),
    )
    expect = case["expect"]
    first = {}
    for name, value in headers:
        first.setdefault(name, value)

    assert case["status"] == expect["status"]
    assert first["content-type"] == "application/json"
    for name, value in (expect.get("headers") or {}).items():
        assert first.get(name) == value, name
    for name in expect.get("absent_headers") or []:
        assert name not in first, name
    rendered = json.loads(content)
    assert rendered["type"] == "error"
    assert rendered["error"]["type"] == expect["type"]
    assert rendered["error"]["message"] == expect["message"]
    assert rendered["request_id"] == expect.get("request_id")


_STEPS: dict[str, str] = {
    "role": json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}),
    "text a": json.dumps({"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}]}),
    "text space": json.dumps({"choices": [{"index": 0, "delta": {"content": " "}, "finish_reason": None}]}),
    "text newline": json.dumps({"choices": [{"index": 0, "delta": {"content": "\n"}, "finish_reason": None}]}),
    "reasoning": json.dumps({"choices": [{"index": 0, "delta": {"reasoning_content": "r"}, "finish_reason": None}]}),
    **{
        f"tool start {index}": json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "id": f"c{index}",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        for index in (0, 1)
    },
    **{
        f"tool args {index}": json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": index, "function": {"arguments": "{}"}}]},
                        "finish_reason": None,
                    }
                ]
            }
        )
        for index in (0, 1)
    },
    **{
        f"finish {reason}": json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})
        for reason in ("stop", "tool_calls", "length")
    },
    "usage": json.dumps({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}),
    "error": json.dumps(
        {
            "choices": [],
            "error": {"message": "m", "type": "server_error", "param": None, "code": "RESOURCE_EXHAUSTED"},
        }
    ),
    "done": "[DONE]",
}


def _assert_event_grammar(events: list[dict[str, Any]], context: str) -> None:
    """Check the event grammar Anthropic clients parse.

    ``message_start``, one block at a time with contiguous indices and matching
    deltas, then ``message_delta`` and ``message_stop`` or one ``error``, and
    nothing after.
    """
    assert events[0]["event"] == "message_start", context
    open_block: tuple[int, str] | None = None
    next_index = 0
    ended = False
    for position, event in enumerate(events[1:], start=1):
        assert not ended, f"{context}: event after the end at {position}"
        data = event["data"]
        name = event["event"]
        if name == "content_block_start":
            assert open_block is None, f"{context}: two open blocks"
            assert data["index"] == next_index, context
            next_index += 1
            open_block = (data["index"], data["content_block"]["type"])
        elif name == "content_block_delta":
            assert open_block is not None, f"{context}: delta without an open block"
            assert data["index"] == open_block[0], context
            allowed = {
                "text": {"text_delta"},
                "thinking": {"thinking_delta", "signature_delta"},
                "tool_use": {"input_json_delta"},
            }[open_block[1]]
            assert data["delta"]["type"] in allowed, context
        elif name == "content_block_stop":
            assert open_block is not None, context
            assert data["index"] == open_block[0], context
            open_block = None
        elif name == "message_delta":
            assert open_block is None, f"{context}: message_delta with an open block"
            assert events[position + 1]["event"] == "message_stop", context
        elif name in ("message_stop", "error"):
            ended = True
        else:
            assert name == "ping", f"{context}: unexpected event {name}"
    assert ended, f"{context}: no terminal event"


def test_any_chat_chunk_sequence_yields_a_valid_event_grammar() -> None:
    alphabet = list(_STEPS)
    seed = 0x9E3779B97F4A7C15
    mask = (1 << 64) - 1
    for round_number in range(4000):
        steps = []
        for _ in range(round_number % 12 + 1):
            seed ^= (seed << 13) & mask
            seed ^= seed >> 7
            seed ^= (seed << 17) & mask
            steps.append(alphabet[seed % len(alphabet)])
        machine = StreamMachine("msg_x", "m")
        events = machine.start()
        for step in steps:
            events += machine.on_data(_STEPS[step])
        events += machine.on_end()
        _assert_event_grammar(_named(events), f"round {round_number}: {steps}")


async def test_a_silent_stream_sends_pings_until_output_arrives() -> None:
    async def slow() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.15)
        yield b'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n'
        yield b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}\n\n'
        yield b"data: [DONE]\n\n"

    events = _parse_events(await _drain(stream_events(slow(), StreamMachine("msg_p", "m"), ping_interval=0.02)))
    names = [event["event"] for event in events]
    first_block = names.index("content_block_start")

    assert names[0] == "message_start"
    assert names.count("ping") >= 2, names
    assert set(names[1:first_block]) == {"ping"}
    assert names[-1] == "message_stop"
    _assert_event_grammar(events, "pings")


class _Source:
    """A chat body that records whether it was closed."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self) -> _Source:
        return self

    async def __anext__(self) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        await asyncio.get_running_loop().create_future()
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        self.closed = True


async def test_an_error_event_ends_the_body_and_closes_the_chat_body() -> None:
    source = _Source([b"data: {not json\n\n"])

    body = await asyncio.wait_for(_drain(stream_events(source, StreamMachine("msg_e", "m"), ping_interval=60.0)), 5)

    assert [event["event"] for event in _parse_events(body)] == ["message_start", "error"]
    assert source.closed


async def test_closing_the_messages_body_closes_the_chat_body() -> None:
    source = _Source([])
    body = stream_events(source, StreamMachine("msg_d", "m"), ping_interval=60.0)

    first = await anext(body)
    assert first.startswith(b"event: message_start")
    assert not source.closed
    await body.aclose()

    assert source.closed
