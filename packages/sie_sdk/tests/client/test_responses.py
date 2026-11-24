"""Tests for the typed Responses SDK surface (buffered and streaming)."""

from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from typing import Any, Self, get_args, get_type_hints
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import httpx
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.errors import ProvisioningError, RequestError, ServerError, SIEConnectionError
from sie_sdk.types import ResponseInputContentPart, ResponseInputItem, ResponseInputMessage, ResponseTool


def _payload() -> dict[str, Any]:
    return {
        "id": "resp-req-1",
        "object": "response",
        "created_at": 1,
        "model": "m",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg-req-1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    }


def _sync_response(
    status: int,
    body: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.headers = headers or {}
    response.content = json.dumps(body).encode()
    response.json.return_value = body
    return response


def test_responses_input_types_match_public_openapi() -> None:
    part_types = get_args(get_type_hints(ResponseInputContentPart)["type"])
    roles = get_args(get_type_hints(ResponseInputMessage)["role"])

    assert set(part_types) == {"text", "input_text", "output_text"}
    assert set(roles) == {"system", "user", "assistant", "developer"}


def test_responses_sends_complete_typed_body_and_attaches_request_metadata() -> None:
    response = _sync_response(
        200,
        _payload(),
        headers={
            "x-sie-request-id": "req-1",
            "x-sie-units-input-tokens": "2",
            "x-sie-units-output-tokens": "1",
            "x-sie-credits-debited": "9",
        },
    )
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.return_value = response
        client = SIEClient("https://gateway.example.test", base_url_headers={"Modal-Key": "wk"})

        result = client.responses(
            "m",
            [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            max_output_tokens=32,
            temperature=0.2,
            top_p=0.9,
            seed=-7,
            gpu="l4",
        )

        assert result["output"][0]["content"][0]["text"] == "Hello"
        assert result["request"] == {
            "id": "req-1",
            "usage": {"input_tokens": 2, "output_tokens": 1},
            "credits_debited": 9,
        }
        call = client_cls.return_value.post.call_args
        assert call.args[0] == "/v1/responses"
        assert json.loads(call.kwargs["content"]) == {
            "model": "m",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hi"}]}],
            "max_output_tokens": 32,
            "temperature": 0.2,
            "top_p": 0.9,
            "seed": -7,
        }
        assert call.kwargs["headers"] == {
            "content-type": "application/json",
            "accept": "application/json",
            "X-SIE-MACHINE-PROFILE": "l4",
        }
        assert len(client_cls.call_args.kwargs["event_hooks"]["request"]) == 1
        client.close()


def test_responses_retries_pre_execution_provisioning_only() -> None:
    provisioning = _sync_response(
        503,
        {"error": {"code": "PROVISIONING", "message": "warming"}},
        headers={"Retry-After": "0.01"},
    )
    with (
        patch("sie_sdk.client.sync.httpx.Client") as client_cls,
        patch("sie_sdk.client.sync.time.sleep"),
    ):
        client_cls.return_value.post.side_effect = [provisioning, _sync_response(200, _payload())]
        client = SIEClient("http://localhost:8080")

        client.responses("m", "Hi", provision_timeout_s=5)

        assert client_cls.return_value.post.call_count == 2
        assert client.last_retry_count == 1
        client.close()


def test_responses_surfaces_provisioning_without_retry_when_waiting_is_disabled() -> None:
    provisioning = _sync_response(
        503,
        {"error": {"code": "PROVISIONING", "message": "warming"}},
        headers={"Retry-After": "0.01"},
    )
    with (
        patch("sie_sdk.client.sync.httpx.Client") as client_cls,
        patch("sie_sdk.client.sync.time.sleep") as sleep,
    ):
        client_cls.return_value.post.return_value = provisioning
        client = SIEClient("http://localhost:8080")

        with pytest.raises(ProvisioningError):
            client.responses("m", "Hi", wait_for_capacity=False)

        assert client_cls.return_value.post.call_count == 1
        assert client.last_retry_count == 0
        sleep.assert_not_called()
        client.close()


def test_responses_does_not_retry_post_publish_timeout() -> None:
    timeout = _sync_response(
        504,
        {"error": {"code": "GATEWAY_TIMEOUT", "message": "timed out"}},
        headers={
            "x-sie-request-id": "req-timeout",
            "x-sie-units-output-tokens": "4",
            "x-sie-credits-debited": "11",
        },
    )
    with (
        patch("sie_sdk.client.sync.httpx.Client") as client_cls,
        patch("sie_sdk.client.sync.time.sleep") as sleep,
    ):
        client_cls.return_value.post.return_value = timeout
        client = SIEClient("http://localhost:8080")

        with pytest.raises(ServerError) as exc_info:
            client.responses("m", "Hi")

        assert client_cls.return_value.post.call_count == 1
        sleep.assert_not_called()
        assert exc_info.value.request == {
            "id": "req-timeout",
            "usage": {"output_tokens": 4},
            "credits_debited": 11,
        }
        client.close()


def test_responses_rejects_redirect_without_retrying() -> None:
    redirect = _sync_response(
        303,
        {"private": "response-body"},
        headers={
            "content-type": "text/html",
            "location": "https://redirect.example/private-token",
            "x-sie-request-id": "req-responses-redirect",
        },
    )
    with (
        patch("sie_sdk.client.sync.httpx.Client") as client_cls,
        patch("sie_sdk.client.sync.time.sleep") as sleep,
    ):
        client_cls.return_value.post.return_value = redirect
        client = SIEClient("http://localhost:8080")

        with pytest.raises(RequestError) as exc_info:
            client.responses("m", "Hi")

        assert exc_info.value.status_code == 303
        assert exc_info.value.request == {"id": "req-responses-redirect"}
        assert "private-token" not in str(exc_info.value)
        assert "response-body" not in str(exc_info.value)
        assert client_cls.return_value.post.call_count == 1
        assert client.last_retry_count == 0
        sleep.assert_not_called()
        client.close()


def test_responses_consumes_modal_continuation_without_reposting() -> None:
    redirect = _sync_response(
        303,
        {},
        headers={"location": "/v1/responses?__modal_attempt_token=opaque"},
    )
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.return_value = redirect
        client_cls.return_value.get.return_value = _sync_response(200, _payload())
        client = SIEClient("https://gateway.example.test")

        result = client.responses("m", "Hi")

        assert result["status"] == "completed"
        assert client_cls.return_value.post.call_count == 1
        assert client_cls.return_value.get.call_args.args[0] == "/v1/responses?__modal_attempt_token=opaque"
        client.close()


def test_responses_does_not_retry_midflight_transport_failure() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.side_effect = httpx.ReadError("connection reset")
        client = SIEClient("http://localhost:8080")

        with pytest.raises(SIEConnectionError, match="Connection lost mid-request"):
            client.responses("m", "Hi")

        assert client_cls.return_value.post.call_count == 1
        client.close()


_IMAGE = "data:image/png;base64,iVBORw0KGgo="

_TOOLS: list[ResponseTool] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        "strict": False,
    },
    {"type": "custom", "name": "run_python", "description": "Run code", "format": {"type": "text"}},
]

_INPUT: list[ResponseInputItem] = [
    {"role": "developer", "content": "Be brief."},
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "What is this?"},
            {"type": "input_image", "image_url": _IMAGE, "detail": "low"},
        ],
    },
    {"type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": '{"city":"Paris"}'},
    {"type": "function_call_output", "call_id": "call_1", "output": "sunny"},
    {"type": "custom_tool_call", "call_id": "call_2", "name": "run_python", "input": "print(1)"},
    {
        "type": "custom_tool_call_output",
        "call_id": "call_2",
        "output": [{"type": "input_text", "text": "1"}, {"type": "input_image", "image_url": _IMAGE}],
    },
]

_TYPED_FIELDS: dict[str, Any] = {
    "instructions": "You are a helpful assistant.",
    "tools": _TOOLS,
    "tool_choice": {"type": "custom", "name": "run_python"},
    "parallel_tool_calls": False,
    "reasoning": {"effort": "low"},
    "text": {"format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}, "strict": True}},
    "metadata": {"tenant": "t-1"},
    "prompt_cache_key": "cache-1",
    "user": "user-1",
    "safety_identifier": "safety-1",
}

_ABSENT_PARAMETERS = {"store", "previous_response_id", "conversation", "background", "stream", "extra_body"}


@pytest.mark.parametrize("method", ["responses", "stream_responses"])
@pytest.mark.parametrize("client_cls", [SIEClient, SIEAsyncClient])
def test_responses_signatures_cover_typed_fields_and_omit_refused_fields(
    client_cls: type[SIEClient | SIEAsyncClient], method: str
) -> None:
    parameters = inspect.signature(getattr(client_cls, method)).parameters

    assert set(_TYPED_FIELDS) <= set(parameters)
    assert {"max_output_tokens", "temperature", "top_p", "seed"} <= set(parameters)
    assert not _ABSENT_PARAMETERS & set(parameters)


def test_responses_sends_every_typed_field() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.return_value = _sync_response(200, _payload())
        client = SIEClient("http://localhost:8080")

        client.responses("m", _INPUT, max_output_tokens=64, **_TYPED_FIELDS)

        sent = json.loads(client_cls.return_value.post.call_args.kwargs["content"])
        assert sent == {"model": "m", "input": _INPUT, "max_output_tokens": 64, **_TYPED_FIELDS}
        client.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tool_choice", "required"),
        ("tool_choice", {"type": "function", "name": "get_weather"}),
        ("reasoning", {"effort": None}),
        ("text", {"format": {"type": "json_object"}}),
        ("text", {"format": {"type": "text"}}),
    ],
)
def test_responses_forwards_field_values_unchanged(field: str, value: Any) -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.post.return_value = _sync_response(200, _payload())
        client = SIEClient("http://localhost:8080")

        client.responses("m", "Hi", **{field: value})

        sent = json.loads(client_cls.return_value.post.call_args.kwargs["content"])
        assert sent == {"model": "m", "input": "Hi", field: value}
        client.close()


class _FakeStream:
    def __init__(
        self,
        *,
        status_code: int = 200,
        lines: list[str] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> None:
        self.status_code = status_code
        self._lines = lines or []
        self.headers = headers or {"content-type": "text/event-stream"}
        self._json = json_body or {}
        self.content = json.dumps(self._json).encode()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def iter_lines(self) -> Iterator[str]:
        yield from self._lines

    def read(self) -> bytes:
        return self.content

    def json(self) -> dict[str, Any]:
        return self._json


def _sse(*events: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for sequence_number, event in enumerate(events):
        if "type" in event:
            lines.append(f"event: {event['type']}")
        lines.append(f"data: {json.dumps({**event, 'sequence_number': sequence_number})}")
        lines.append("")
    return lines


def _response(status: str, output: list[dict[str, Any]] | None = None, **fields: Any) -> dict[str, Any]:
    terminal = status != "in_progress"
    return {
        "id": "resp-req-1",
        "object": "response",
        "created_at": 0,
        "model": "m",
        "status": status,
        "incomplete_details": None,
        "error": None,
        "store": False,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "output": output or [],
        "usage": (
            {
                "input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 5,
            }
            if terminal
            else None
        ),
        **fields,
    }


def _lifecycle(kind: str, response: dict[str, Any]) -> dict[str, Any]:
    return {"type": f"response.{kind}", "response": response}


def _opening() -> list[dict[str, Any]]:
    return [_lifecycle("created", _response("in_progress")), _lifecycle("in_progress", _response("in_progress"))]


def _message(text: str, status: str) -> dict[str, Any]:
    content = [] if status == "in_progress" else [{"type": "output_text", "text": text, "annotations": []}]
    return {"type": "message", "id": "msg-req-1", "role": "assistant", "status": status, "content": content}


def _text_events(*deltas: str) -> list[dict[str, Any]]:
    text = "".join(deltas)
    part = {"type": "output_text", "text": text, "annotations": []}
    location = {"item_id": "msg-req-1", "output_index": 0, "content_index": 0}
    return [
        {"type": "response.output_item.added", "output_index": 0, "item": _message("", "in_progress")},
        {"type": "response.content_part.added", **location, "part": {**part, "text": ""}},
        *({"type": "response.output_text.delta", **location, "delta": delta} for delta in deltas),
        {"type": "response.output_text.done", **location, "text": text},
        {"type": "response.content_part.done", **location, "part": part},
        {"type": "response.output_item.done", "output_index": 0, "item": _message(text, "completed")},
    ]


def _function_call(status: str, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc-req-1-0",
        "call_id": "call_1",
        "name": "get_weather",
        "arguments": arguments,
        "status": status,
    }


def _function_call_events() -> list[dict[str, Any]]:
    location = {"item_id": "fc-req-1-0", "output_index": 0}
    return [
        {"type": "response.output_item.added", "output_index": 0, "item": _function_call("in_progress", "")},
        {"type": "response.function_call_arguments.delta", **location, "delta": '{"city":'},
        {"type": "response.function_call_arguments.delta", **location, "delta": '"Paris"}'},
        {"type": "response.function_call_arguments.done", **location, "arguments": '{"city":"Paris"}'},
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": _function_call("completed", '{"city":"Paris"}'),
        },
    ]


def _custom_tool_call(status: str, tool_input: str) -> dict[str, Any]:
    return {
        "type": "custom_tool_call",
        "id": "ctc-req-1-0",
        "call_id": "call_2",
        "name": "run_python",
        "input": tool_input,
        "status": status,
    }


def _custom_tool_call_events() -> list[dict[str, Any]]:
    location = {"item_id": "ctc-req-1-0", "output_index": 0}
    return [
        {"type": "response.output_item.added", "output_index": 0, "item": _custom_tool_call("in_progress", "")},
        {"type": "response.custom_tool_call_input.delta", **location, "delta": "print(1)"},
        {"type": "response.custom_tool_call_input.done", **location, "input": "print(1)"},
        {"type": "response.output_item.done", "output_index": 0, "item": _custom_tool_call("completed", "print(1)")},
    ]


def _failed(code: str, message: str) -> dict[str, Any]:
    return _lifecycle("failed", _response("failed", error={"code": code, "message": message}))


def _stream_cases() -> list[Any]:
    text = [
        *_opening(),
        *_text_events("Hel", "lo"),
        _lifecycle("completed", _response("completed", [_message("Hello", "completed")])),
    ]
    function_call = [
        *_opening(),
        *_function_call_events(),
        _lifecycle("completed", _response("completed", [_function_call("completed", '{"city":"Paris"}')])),
    ]
    custom_tool_call = [
        *_opening(),
        *_custom_tool_call_events(),
        _lifecycle("completed", _response("completed", [_custom_tool_call("completed", "print(1)")])),
    ]
    incomplete = [
        *_opening(),
        *_text_events("Hel"),
        _lifecycle(
            "incomplete",
            _response(
                "incomplete",
                [_message("Hel", "completed")],
                incomplete_details={"reason": "max_output_tokens"},
            ),
        ),
    ]
    return [
        pytest.param(text, "response.completed", id="text"),
        pytest.param(function_call, "response.completed", id="function_call"),
        pytest.param(custom_tool_call, "response.completed", id="custom_tool_call"),
        pytest.param(incomplete, "response.incomplete", id="incomplete"),
    ]


@pytest.mark.parametrize(("events", "terminal"), _stream_cases())
def test_stream_responses_yields_typed_events_until_terminal(events: list[dict[str, Any]], terminal: str) -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.stream.return_value = _FakeStream(lines=_sse(*events))
        client = SIEClient("http://localhost:8080")

        out = list(client.stream_responses("m", "Hi", tools=_TOOLS, max_output_tokens=8))

        assert [event["type"] for event in out] == [event["type"] for event in events]
        assert [event["sequence_number"] for event in out] == list(range(len(events)))
        assert out[-1]["type"] == terminal
        assert out == [{**event, "sequence_number": i} for i, event in enumerate(events)]
        call = client_cls.return_value.stream.call_args
        assert call.args == ("POST", "/v1/responses")
        assert json.loads(call.kwargs["content"]) == {
            "model": "m",
            "input": "Hi",
            "stream": True,
            "max_output_tokens": 8,
            "tools": _TOOLS,
        }
        assert call.kwargs["headers"]["accept"] == "text/event-stream"
        client.close()


def test_stream_responses_sends_every_typed_field() -> None:
    events = [*_opening(), _lifecycle("completed", _response("completed"))]
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.stream.return_value = _FakeStream(lines=_sse(*events))
        client = SIEClient("http://localhost:8080")

        list(client.stream_responses("m", _INPUT, seed=5, **_TYPED_FIELDS))

        sent = json.loads(client_cls.return_value.stream.call_args.kwargs["content"])
        assert sent == {"model": "m", "input": _INPUT, "stream": True, "seed": 5, **_TYPED_FIELDS}
        client.close()


def test_stream_responses_raises_failed_event_as_server_error() -> None:
    events = [*_opening(), *_text_events("Hel"), _failed("inference_error", "boom")]
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls, patch("sie_sdk.client.sync.time.sleep") as sleep:
        client_cls.return_value.stream.return_value = _FakeStream(lines=_sse(*events))
        client = SIEClient("http://localhost:8080")
        seen: list[str] = []

        with pytest.raises(ServerError, match="boom") as exc_info:
            seen.extend(event["type"] for event in client.stream_responses("m", "Hi"))

        assert exc_info.value.code == "inference_error"
        assert seen == [event["type"] for event in events[:-1]]
        assert client_cls.return_value.stream.call_count == 1
        sleep.assert_not_called()
        client.close()


def test_stream_responses_raises_failed_event_without_error_object() -> None:
    failed = _lifecycle("failed", _response("failed"))
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.stream.return_value = _FakeStream(lines=_sse(*_opening(), failed))
        client = SIEClient("http://localhost:8080")

        with pytest.raises(ServerError) as exc_info:
            list(client.stream_responses("m", "Hi"))

        assert exc_info.value.code == "error"
        client.close()


def test_stream_responses_raises_error_frame_after_output() -> None:
    error_frame = {"error": {"code": "MODEL_LOADING", "message": "loading"}}
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.stream.return_value = _FakeStream(lines=_sse(*_opening(), error_frame))
        client = SIEClient("http://localhost:8080")

        with pytest.raises(ServerError, match="loading"):
            list(client.stream_responses("m", "Hi"))

        assert client_cls.return_value.stream.call_count == 1
        assert client.last_retry_count == 0
        client.close()


def test_stream_responses_requires_a_terminal_event() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.stream.return_value = _FakeStream(lines=_sse(*_opening(), *_text_events("Hel")))
        client = SIEClient("http://localhost:8080")

        with pytest.raises(SIEConnectionError, match="before generation completed"):
            list(client.stream_responses("m", "Hi"))

        client.close()


def test_stream_responses_retries_503_provisioning_before_the_first_event() -> None:
    provisioning = _FakeStream(
        status_code=503,
        headers={"Retry-After": "0.01", "content-type": "application/json"},
        json_body={"error": {"code": "PROVISIONING", "message": "warming"}},
    )
    events = [*_opening(), _lifecycle("completed", _response("completed"))]
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls, patch("sie_sdk.client.sync.time.sleep"):
        client_cls.return_value.stream.side_effect = [provisioning, _FakeStream(lines=_sse(*events))]
        client = SIEClient("http://localhost:8080")

        out = list(client.stream_responses("m", "Hi", provision_timeout_s=5.0))

        assert out[-1]["type"] == "response.completed"
        assert client_cls.return_value.stream.call_count == 2
        assert client.last_retry_count == 1
        client.close()


def test_stream_responses_retries_first_sse_capacity_error_frame() -> None:
    loading = _FakeStream(lines=_sse({"error": {"code": "MODEL_LOADING", "message": "loading"}}))
    events = [*_opening(), _lifecycle("completed", _response("completed"))]
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls, patch("sie_sdk.client.sync.time.sleep"):
        client_cls.return_value.stream.side_effect = [loading, _FakeStream(lines=_sse(*events))]
        client = SIEClient("http://localhost:8080")

        out = list(client.stream_responses("m", "Hi", provision_timeout_s=5.0))

        assert [event["type"] for event in out] == [event["type"] for event in events]
        assert client_cls.return_value.stream.call_count == 2
        assert client.last_retry_count == 1
        client.close()


def test_stream_responses_surfaces_provisioning_without_retry_when_waiting_is_disabled() -> None:
    provisioning = _FakeStream(
        status_code=503,
        headers={"Retry-After": "0.01", "content-type": "application/json"},
        json_body={"error": {"code": "PROVISIONING", "message": "warming"}},
    )
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls, patch("sie_sdk.client.sync.time.sleep") as sleep:
        client_cls.return_value.stream.return_value = provisioning
        client = SIEClient("http://localhost:8080")

        with pytest.raises(ProvisioningError):
            list(client.stream_responses("m", "Hi", wait_for_capacity=False))

        assert client_cls.return_value.stream.call_count == 1
        sleep.assert_not_called()
        client.close()


class _AsyncRaw:
    def __init__(
        self,
        status: int,
        body: dict[str, Any],
        *,
        headers: dict[str, str] | None = None,
        lines: list[str] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {}
        self._lines = lines or []
        self.content = self._aiter_bytes()

    async def _aiter_bytes(self):
        for line in self._lines:
            yield f"{line}\n".encode()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def read(self) -> bytes:
        return json.dumps(self._body).encode()


def _patch_async_session(
    client: SIEAsyncClient,
    *,
    post_returns: _AsyncRaw | None = None,
    post_side_effect: list[_AsyncRaw] | Exception | None = None,
) -> MagicMock:
    session = MagicMock()
    session.post = (
        MagicMock(side_effect=post_side_effect)
        if post_side_effect is not None
        else MagicMock(return_value=post_returns)
    )
    session.close = AsyncMock()
    client._session = session
    return session


@pytest.mark.asyncio
async def test_async_responses_matches_typed_body_metadata_and_origin_headers() -> None:
    raw = _AsyncRaw(
        200,
        _payload(),
        headers={
            "x-sie-request-id": "req-async",
            "x-sie-units-output-tokens": "1",
            "x-sie-credits-debited": "6",
        },
    )
    client = SIEAsyncClient("https://gateway.example.test", base_url_headers={"Modal-Key": "wk"})
    session = _patch_async_session(client, post_returns=raw)

    result = await client.responses("m", "Hi", max_output_tokens=8, temperature=0, seed=3)

    assert result["request"] == {
        "id": "req-async",
        "usage": {"output_tokens": 1},
        "credits_debited": 6,
    }
    call = session.post.call_args
    assert call.args[0] == "/v1/responses"
    assert json.loads(call.kwargs["data"]) == {
        "model": "m",
        "input": "Hi",
        "max_output_tokens": 8,
        "temperature": 0,
        "seed": 3,
    }
    assert call.kwargs["headers"]["accept"] == "application/json"
    assert call.kwargs["headers"]["Modal-Key"] == "wk"
    assert call.kwargs["allow_redirects"] is False
    await client.close()


@pytest.mark.asyncio
async def test_async_responses_retries_pre_execution_provisioning_only() -> None:
    provisioning = _AsyncRaw(
        503,
        {"error": {"code": "PROVISIONING", "message": "warming"}},
        headers={"Retry-After": "0.01"},
    )
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_async_session(
        client,
        post_side_effect=[provisioning, _AsyncRaw(200, _payload())],
    )

    with patch("sie_sdk.client.async_.asyncio.sleep", new=AsyncMock()):
        await client.responses("m", "Hi", provision_timeout_s=5)

    assert session.post.call_count == 2
    await client.close()


@pytest.mark.asyncio
async def test_async_responses_does_not_retry_midflight_transport_failure() -> None:
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_async_session(
        client,
        post_side_effect=aiohttp.ClientPayloadError("truncated response"),
    )

    with pytest.raises(SIEConnectionError, match="Request failed"):
        await client.responses("m", "Hi")

    assert session.post.call_count == 1
    await client.close()


@pytest.mark.asyncio
async def test_async_responses_does_not_retry_post_publish_timeout() -> None:
    timeout = _AsyncRaw(
        504,
        {"error": {"code": "GATEWAY_TIMEOUT", "message": "timed out"}},
        headers={
            "x-sie-request-id": "req-async-timeout",
            "x-sie-units-output-tokens": "4",
            "x-sie-credits-debited": "12",
        },
    )
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_async_session(client, post_returns=timeout)

    with pytest.raises(ServerError) as exc_info:
        await client.responses("m", "Hi")

    assert session.post.call_count == 1
    assert exc_info.value.request == {
        "id": "req-async-timeout",
        "usage": {"output_tokens": 4},
        "credits_debited": 12,
    }
    await client.close()


@pytest.mark.asyncio
async def test_async_responses_consumes_modal_continuation_without_reposting() -> None:
    client = SIEAsyncClient("https://gateway.example.test")
    session = _patch_async_session(
        client,
        post_returns=_AsyncRaw(
            303,
            {},
            headers={"location": "/v1/responses?__modal_attempt_token=opaque"},
        ),
    )
    session.get = MagicMock(return_value=_AsyncRaw(200, _payload()))

    result = await client.responses("m", "Hi")

    assert result["status"] == "completed"
    assert session.post.call_count == 1
    assert session.get.call_args.args[0] == "/v1/responses?__modal_attempt_token=opaque"
    await client.close()


@pytest.mark.asyncio
async def test_async_responses_sends_every_typed_field() -> None:
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_async_session(client, post_returns=_AsyncRaw(200, _payload()))

    await client.responses("m", _INPUT, max_output_tokens=64, **_TYPED_FIELDS)

    sent = json.loads(session.post.call_args.kwargs["data"])
    assert sent == {"model": "m", "input": _INPUT, "max_output_tokens": 64, **_TYPED_FIELDS}
    await client.close()


@pytest.mark.parametrize(("events", "terminal"), _stream_cases())
@pytest.mark.asyncio
async def test_async_stream_responses_yields_typed_events_until_terminal(
    events: list[dict[str, Any]], terminal: str
) -> None:
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_async_session(client, post_returns=_AsyncRaw(200, {}, lines=_sse(*events)))

    out = [event async for event in client.stream_responses("m", _INPUT, **_TYPED_FIELDS)]

    assert out == [{**event, "sequence_number": i} for i, event in enumerate(events)]
    assert out[-1]["type"] == terminal
    call = session.post.call_args
    assert call.args[0] == "/v1/responses"
    assert json.loads(call.kwargs["data"]) == {"model": "m", "input": _INPUT, "stream": True, **_TYPED_FIELDS}
    assert call.kwargs["headers"]["accept"] == "text/event-stream"
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_responses_raises_failed_event_as_server_error() -> None:
    events = [*_opening(), *_text_events("Hel"), _failed("inference_error", "boom")]
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_async_session(client, post_returns=_AsyncRaw(200, {}, lines=_sse(*events)))
    seen: list[str] = []

    with pytest.raises(ServerError, match="boom") as exc_info:
        async for event in client.stream_responses("m", "Hi"):
            seen.append(event["type"])

    assert exc_info.value.code == "inference_error"
    assert seen == [event["type"] for event in events[:-1]]
    assert session.post.call_count == 1
    await client.close()


@pytest.mark.asyncio
async def test_async_stream_responses_requires_a_terminal_event() -> None:
    client = SIEAsyncClient("http://localhost:8080")
    _patch_async_session(client, post_returns=_AsyncRaw(200, {}, lines=_sse(*_opening())))

    with pytest.raises(SIEConnectionError, match="before generation completed"):
        _ = [event async for event in client.stream_responses("m", "Hi")]

    await client.close()


@pytest.mark.parametrize(
    "first_attempt",
    [
        pytest.param(
            _AsyncRaw(503, {"error": {"code": "PROVISIONING", "message": "warming"}}, headers={"Retry-After": "0.01"}),
            id="http-503",
        ),
        pytest.param(
            _AsyncRaw(200, {}, lines=_sse({"error": {"code": "MODEL_LOADING", "message": "loading"}})),
            id="first-sse-error",
        ),
    ],
)
@pytest.mark.asyncio
async def test_async_stream_responses_retries_capacity_before_the_first_event(first_attempt: _AsyncRaw) -> None:
    events = [*_opening(), _lifecycle("completed", _response("completed"))]
    client = SIEAsyncClient("http://localhost:8080")
    session = _patch_async_session(
        client,
        post_side_effect=[first_attempt, _AsyncRaw(200, {}, lines=_sse(*events))],
    )

    with patch("sie_sdk.client.async_.asyncio.sleep", new=AsyncMock()):
        out = [event async for event in client.stream_responses("m", "Hi", provision_timeout_s=5)]

    assert [event["type"] for event in out] == [event["type"] for event in events]
    assert session.post.call_count == 2
    assert client.last_retry_count == 1
    await client.close()
