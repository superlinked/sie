"""The Responses translation replays the vectors the gateway replays.

Every file under ``conformance/openai_responses/`` is read by the gateway's
tests and by these, so a request is accepted, refused and answered the same
way on both topologies. A translated request must also pass this server's own
chat validation. ``created_at`` is the one value the clock decides, so the
vectors carry ``0`` and these tests pass ``0``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from sie_server.api.openai_local import (
    _validate_chat_message_media,
    _validate_chat_messages,
    _validate_cuda_chat_body,
)
from sie_server.api.responses_translation import (
    Event,
    FrameParser,
    ResponsesError,
    StreamMachine,
    chat_completion_to_response,
    responses_error,
    responses_to_chat,
    stream_events,
)

VECTORS = Path(__file__).resolve().parents[4] / "conformance" / "openai_responses"


def _vectors(name: str) -> Any:
    return json.loads((VECTORS / name).read_text(encoding="utf-8"))


def _chat_route_accepts(chat: dict[str, Any]) -> None:
    _validate_chat_messages(chat["messages"])
    _validate_chat_message_media(chat["messages"])
    _validate_cuda_chat_body(chat)


def _frames(case: dict[str, Any]) -> list[str]:
    return [frame if isinstance(frame, str) else json.dumps(frame, ensure_ascii=False) for frame in case["frames"]]


def _named(events: list[Event]) -> list[dict[str, Any]]:
    return [{"event": event.name, "data": event.data} for event in events]


def _machine(case: dict[str, Any]) -> StreamMachine:
    return StreamMachine(responses_to_chat(case["request"]).shape, case["request_id"], 0)


def _parse_body(body: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    """The events of a Responses SSE body, and its comment frames."""
    events: list[dict[str, Any]] = []
    comments: list[str] = []
    for frame in body.decode().split("\n\n"):
        if not frame:
            continue
        if frame.startswith(":"):
            comments.append(frame)
            continue
        name, data = "", None
        for line in frame.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        assert data is not None, frame
        assert data["type"] == name, frame
        events.append({"event": name, "data": data})
    return events, comments


async def _drain(body: AsyncIterator[bytes]) -> bytes:
    return b"".join([chunk async for chunk in body])


@pytest.mark.parametrize("case", _vectors("request_vectors.json")["accepted"], ids=lambda case: case["name"])
def test_an_accepted_request_vector_translates_onto_the_chat_body(case: dict[str, Any]) -> None:
    translation = responses_to_chat(case["body"])

    assert translation.chat == case["chat"]
    assert translation.stream is case["chat"]["stream"]
    _chat_route_accepts(translation.chat)


@pytest.mark.parametrize("case", _vectors("request_vectors.json")["rejected"], ids=lambda case: case["name"])
def test_a_rejected_request_vector_is_refused_with_its_wording(case: dict[str, Any]) -> None:
    with pytest.raises(ResponsesError) as refused:
        responses_to_chat(case["body"])

    error = refused.value
    expect = case["expect"]
    assert error.status == expect["status"]
    assert error.body() == {
        "error": {
            "message": expect["message"],
            "type": expect["type"],
            "param": expect["param"],
            "code": expect["code"],
        }
    }


@pytest.mark.parametrize("case", _vectors("codex_requests.json")["requests"], ids=lambda case: case["name"])
def test_a_captured_codex_request_translates_or_is_refused(case: dict[str, Any]) -> None:
    expect = case["expect"]
    if "refused" in expect:
        with pytest.raises(ResponsesError) as refused:
            responses_to_chat(case["body"])
        error = refused.value
        assert {
            "status": error.status,
            "type": error.body()["error"]["type"],
            "code": error.code,
            "param": error.param,
            "message": error.message,
        } == expect["refused"]
        return
    chat = responses_to_chat(case["body"]).chat
    summary = expect["chat"]
    assert [message["role"] for message in chat["messages"]] == summary["roles"]
    assert chat["stream"] is summary["stream"]
    assert len(chat.get("tools") or []) == summary["tool_count"]
    assert chat.get("tool_choice") == summary["tool_choice"]
    assert chat.get("parallel_tool_calls") == summary["parallel_tool_calls"]
    assert chat["include_reasoning"] is summary["include_reasoning"]
    assert chat.get("reasoning_effort") == summary.get("reasoning_effort")
    images = [
        part
        for message in chat["messages"]
        if isinstance(message["content"], list)
        for part in message["content"]
        if part["type"] == "image_url"
    ]
    assert len(images) == summary["image_parts"]
    _chat_route_accepts(chat)


@pytest.mark.parametrize("case", _vectors("response_vectors.json")["translated"], ids=lambda case: case["name"])
def test_a_response_vector_translates_the_chat_completion(case: dict[str, Any]) -> None:
    shape = responses_to_chat(case["request"]).shape

    assert shape.model == case["model"]
    assert chat_completion_to_response(case["chat"], shape, case["request_id"], 0) == case["response"]


@pytest.mark.parametrize(
    "chat",
    [
        {},
        {"choices": []},
        {"choices": [{"finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "", "tool_calls": [{"function": {"name": "f"}}]}}]},
        {"choices": [{"message": {"content": "", "tool_calls": [{"id": "c", "function": {}}]}}]},
    ],
    ids=["no choices key", "no choices", "no message", "a call without an id", "a call without a name"],
)
def test_a_faulty_chat_completion_is_not_translated(chat: dict[str, Any]) -> None:
    shape = responses_to_chat({"model": "m", "input": "x"}).shape

    with pytest.raises(ValueError, match=r"^(chat completion|tool call)"):
        chat_completion_to_response(chat, shape, "r", 0)


@pytest.mark.parametrize("case", _vectors("stream_vectors.json")["streams"], ids=lambda case: case["name"])
def test_a_stream_vector_drives_the_state_machine(case: dict[str, Any]) -> None:
    machine = _machine(case)
    events = machine.start()
    for frame in _frames(case):
        events += machine.on_data(frame)
    events += machine.on_end()

    assert _named(events) == case["events"]
    assert machine.done


@pytest.mark.parametrize("case", _vectors("stream_vectors.json")["streams"], ids=lambda case: case["name"])
async def test_a_stream_vector_survives_the_sse_body_split_at_any_byte(case: dict[str, Any]) -> None:
    sse = "".join(f": keepalive\n\ndata: {frame}\n\n" for frame in _frames(case)).encode()

    async def chunks() -> AsyncIterator[bytes]:
        for start in range(len(sse)):
            yield sse[start : start + 1]

    events, comments = _parse_body(await _drain(stream_events(chunks(), _machine(case))))

    assert events == case["events"]
    assert comments
    assert set(comments) == {": keepalive"}


def test_the_frame_parser_joins_multi_line_data_and_keeps_comment_frames() -> None:
    parser = FrameParser()

    assert parser.push(b': keep-alive\r\n\r\ndata: {"a"') == [("comment", ": keep-alive\n\n")]
    assert parser.push(b":1}\r\n\r\nevent: x\n: note\ndata: one\ndata: two\n\nid: 1\n\n") == [
        ("data", '{"a":1}'),
        ("data", "one\ntwo"),
    ]


@pytest.mark.parametrize("case", _vectors("error_vectors.json")["errors"], ids=lambda case: case["name"])
def test_an_error_vector_keeps_the_answer_and_renames_the_param(case: dict[str, Any]) -> None:
    body = json.dumps(case["body"]).encode()
    sent = [*(case.get("headers") or {}).items(), ("Content-Length", str(len(body)))]

    headers, content = responses_error(sent, body)

    expect = case["expect"]
    for name, value in (expect.get("headers") or {}).items():
        assert dict(headers)[name] == value, name
    for name in expect.get("absent_headers") or []:
        assert name not in dict(headers), name
    assert "content-length" not in dict(headers)
    assert json.loads(content) == expect["body"]


def test_a_body_that_is_not_an_openai_error_passes_unchanged() -> None:
    for body in (b"", b"not json", b'{"detail": {"code": "X", "param": "max_completion_tokens"}}', b"[]"):
        assert responses_error([("retry-after", "3")], body) == ([("retry-after", "3")], body)


_STEPS: dict[str, str] = {
    "role": json.dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}),
    "text": json.dumps({"choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}]}),
    **{
        f"call {index}": json.dumps(
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
                                    "function": {"name": name, "arguments": ""},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        for index, name in ((0, "f"), (1, "edit"))
    },
    **{
        f"arguments {index}": json.dumps(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": index, "function": {"arguments": '{"input":"x"}'}}]},
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
    "error": json.dumps({"choices": [], "error": {"message": "m", "type": "server_error", "code": "inference_error"}}),
    "malformed": "{not json",
    "done": "[DONE]",
}
_SHAPE = responses_to_chat(
    {
        "model": "m",
        "input": "x",
        "tools": [{"type": "function", "name": "f"}, {"type": "custom", "name": "edit"}],
    }
).shape


def _assert_event_grammar(events: list[dict[str, Any]], context: str) -> None:
    """Check the event grammar Responses clients parse.

    ``response.created`` and ``response.in_progress``, then one item at a time
    with contiguous output indices and the events of its kind, then one
    terminal event whose output is the items closed so far, and nothing after.
    Sequence numbers count from 0.
    """
    assert [event["data"]["sequence_number"] for event in events] == list(range(len(events))), context
    assert [event["event"] for event in events[:2]] == ["response.created", "response.in_progress"], context
    open_item: tuple[int, str] | None = None
    closed: list[dict[str, Any]] = []
    next_index = 0
    allowed = {
        "message": {
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
        },
        "function_call": {"response.function_call_arguments.delta", "response.function_call_arguments.done"},
        "custom_tool_call": {"response.custom_tool_call_input.delta", "response.custom_tool_call_input.done"},
    }
    for position, event in enumerate(events[2:], start=2):
        name, data = event["event"], event["data"]
        if name in ("response.completed", "response.incomplete", "response.failed"):
            assert position == len(events) - 1, f"{context}: event after the end"
            if name != "response.failed":
                assert open_item is None, f"{context}: {name} with an open item"
            assert data["response"]["output"] == closed, context
            assert {item["status"] for item in closed} <= {"completed", "incomplete"}, context
            return
        if name == "response.output_item.added":
            assert open_item is None, f"{context}: two open items"
            assert data["output_index"] == next_index, context
            next_index += 1
            open_item = (data["output_index"], data["item"]["type"])
        elif name == "response.output_item.done":
            assert open_item is not None, context
            assert (data["output_index"], data["item"]["type"]) == open_item, context
            closed.append(data["item"])
            open_item = None
        else:
            assert open_item is not None, f"{context}: {name} without an open item"
            assert name in allowed[open_item[1]], f"{context}: {name} in a {open_item[1]} item"
            assert data["output_index"] == open_item[0], context
    pytest.fail(f"{context}: no terminal event")


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
        machine = StreamMachine(_SHAPE, "r", 0)
        events = machine.start()
        for step in steps:
            events += machine.on_data(_STEPS[step])
        events += machine.on_end()
        _assert_event_grammar(_named(events), f"round {round_number}: {steps}")


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (["{not json"], "the generation stream sent a malformed chunk"),
        (
            [json.dumps({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0}]}, "finish_reason": None}]})],
            "a tool call arrived without its id and name",
        ),
        (
            [_STEPS["call 0"], _STEPS["call 1"], _STEPS["arguments 0"]],
            "tool call arguments arrived after another output item started",
        ),
        (
            [_STEPS["call 0"], _STEPS["text"], _STEPS["arguments 0"]],
            "tool call arguments arrived after another output item started",
        ),
    ],
    ids=["malformed chunk", "call without an id", "arguments after a later call", "arguments after text"],
)
def test_a_chat_stream_the_translation_cannot_follow_fails_the_response(frames: list[str], message: str) -> None:
    machine = StreamMachine(_SHAPE, "r", 0)
    events = machine.start()
    for frame in frames:
        events += machine.on_data(frame)

    assert events[-1].name == "response.failed"
    assert events[-1].data["response"]["error"] == {"code": "server_error", "message": message}
    assert machine.done


class _Source:
    """A chat body that records whether it was closed."""

    def __init__(self, chunks: list[bytes], *, fail: bool = False) -> None:
        self._chunks = list(chunks)
        self._fail = fail
        self.closed = False

    def __aiter__(self) -> _Source:
        return self

    async def __anext__(self) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        if self._fail:
            raise OSError("the chat body broke")
        await asyncio.get_running_loop().create_future()
        raise AssertionError("unreachable")

    async def aclose(self) -> None:
        self.closed = True


async def test_a_chat_body_that_breaks_fails_the_response() -> None:
    source = _Source([b'data: {"choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}\n\n'], fail=True)

    events, _ = _parse_body(await asyncio.wait_for(_drain(stream_events(source, StreamMachine(_SHAPE, "r", 0))), 5))

    assert events[-1]["event"] == "response.failed"
    assert events[-1]["data"]["response"]["error"] == {
        "code": "server_error",
        "message": "the generation stream ended unexpectedly",
    }
    assert source.closed


async def test_the_terminal_event_ends_the_body_and_closes_the_chat_body() -> None:
    source = _Source([f"data: {_STEPS['error']}\n\n".encode()])

    events, _ = _parse_body(await asyncio.wait_for(_drain(stream_events(source, StreamMachine(_SHAPE, "r", 0))), 5))

    assert [event["event"] for event in events] == ["response.created", "response.in_progress", "response.failed"]
    assert source.closed


async def test_closing_the_responses_body_closes_the_chat_body() -> None:
    source = _Source([])
    body = stream_events(source, StreamMachine(_SHAPE, "r", 0))

    first = await anext(body)
    assert first.startswith(b"event: response.created")
    assert not source.closed
    await body.aclose()

    assert source.closed
