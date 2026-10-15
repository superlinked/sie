import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters.errors import UPSTREAM_FAILURE_MESSAGES
from sie_server.adapters.remote._limits import upstream_limiter
from sie_server.adapters.remote.openai import OpenAIUpstreamAdapter
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.api import openai_local
from sie_server.api.generate import router as generate_router
from sie_server.api.openai_completions import router as completions_router
from sie_server.api.openai_local import _remote_chat_response, router
from sie_server.api.openai_responses import router as responses_router
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.loader import expand_profile_variants
from sie_server.core.upstream_client import upstream_client
from sie_server.processors import remote_chat_prompt

MODEL = "caller/model"
SAFE_MODEL = "caller__model"
BODY = {"model": MODEL, "messages": [{"role": "user", "content": "question"}]}
USAGE = {"prompt_tokens": 37, "completion_tokens": 2, "total_tokens": 39}


class ChatStream(httpx.AsyncByteStream):
    def __init__(self, events: list[dict | str], *, disconnect: bool = False) -> None:
        self.events = events
        self.disconnect = disconnect
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self.events:
            data = event if isinstance(event, str) else json.dumps(event)
            yield b"data: " + data.encode() + b"\n\n"
        if self.disconnect:
            raise httpx.ReadError("private upstream detail")

    async def aclose(self) -> None:
        self.closed = True


def completion(**changes: Any) -> dict:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
        "usage": USAGE,
        **changes,
    }


def json_response(status: int, payload: dict, *, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(
        status,
        headers={"content-type": "application/json", **(headers or {})},
        stream=httpx.ByteStream(json.dumps(payload).encode()),
    )


def events() -> list[dict | str]:
    return [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "answer"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": USAGE},
        "[DONE]",
    ]


@pytest.fixture(params=["sie", "openai"])
def remote_chat(
    request: pytest.FixtureRequest,
) -> Iterator[tuple[TestClient, SieUpstreamAdapter | OpenAIUpstreamAdapter, ModelConfig, list[httpx.Request]]]:
    kind = request.param
    adapter_class = SieUpstreamAdapter if kind == "sie" else OpenAIUpstreamAdapter
    upstream = Upstream.model_validate(
        {
            "kind": kind,
            **({"endpoints": ["chat", "completions"]} if kind == "openai" else {}),
            "base_url": "http://127.0.0.1:8088/prefix" + ("/v1" if kind == "openai" else ""),
            "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
        }
    )
    install_upstreams({"chat-api": upstream})
    adapter = adapter_class(upstream="chat-api", upstream_model="operator/model")
    adapter.load("cpu")
    config = ModelConfig.model_validate(
        {
            "sie_id": MODEL,
            "remote_backed": True,
            "tasks": {"generate": {"context_length": 4096, "max_output_tokens": 64}},
            "profiles": {
                "default": {
                    "adapter_path": f"{adapter_class.__module__}:{adapter_class.__name__}",
                    "max_batch_tokens": 8192,
                    "adapter_options": {
                        "loadtime": {"upstream": "chat-api", "upstream_model": "operator/model"},
                        "runtime": {"default_sampling": {"top_k": 5}, "stop_tokens": ["configured-stop"]},
                    },
                }
            },
        }
    )
    registry = MagicMock()
    registry.device = "cpu"
    registry.has_model.return_value = True
    registry.get_config.return_value = config
    registry.is_failed.return_value = False
    registry.is_unloading.return_value = False
    registry.is_loading.return_value = False
    registry.is_loaded.return_value = True
    registry.get.return_value = adapter
    app = FastAPI()
    app.include_router(router)
    app.include_router(generate_router)
    app.include_router(completions_router)
    app.include_router(responses_router)
    app.state.registry = registry
    requests: list[httpx.Request] = []
    try:
        with TestClient(app) as client:
            yield client, adapter, config, requests
            client.portal.call(adapter.aclose_client)
    finally:
        adapter.unload()
        install_upstreams({})


def answer_with(
    remote_chat: tuple[TestClient, SieUpstreamAdapter | OpenAIUpstreamAdapter, ModelConfig, list[httpx.Request]],
    response: httpx.Response | Callable[[], httpx.Response],
) -> None:
    _, adapter, _, requests = remote_chat

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response() if callable(response) else response

    assert adapter._upstream is not None
    adapter._async_client = upstream_client(adapter._upstream, transport=httpx.MockTransport(respond))


def test_cpu_chat_preserves_identity_exact_usage_and_operator_defaults(remote_chat: tuple) -> None:
    client, _, _, requests = remote_chat
    answer_with(remote_chat, json_response(200, completion(model="private", id="private-id")))
    response = client.post("/v1/chat/completions", json={**BODY, "safety_identifier": "private-caller"})
    assert response.status_code == 200
    assert response.json()["model"] == MODEL
    assert response.json()["usage"] == USAGE
    assert "private" not in response.text
    assert response.headers["X-SIE-Served-By"] == "remote"
    assert response.headers["X-SIE-Upstream"] == "chat-api"
    assert len(requests) == 1
    assert requests[0].url.path == "/prefix/v1/chat/completions"
    sent = json.loads(requests[0].content)
    assert sent["model"] == "operator/model"
    assert sent["max_tokens"] == 64
    assert sent["top_k"] == 5
    assert sent["stop"] == ["configured-stop"]
    assert "safety_identifier" not in sent
    assert "separate_reasoning" not in sent


@pytest.mark.parametrize("include_usage", [None, False, True])
def test_stream_usage_is_required_internally_and_optional_for_the_caller(
    remote_chat: tuple, include_usage: bool
) -> None:
    client, _, _, requests = remote_chat
    stream = ChatStream(events())
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    body = {**BODY, "stream": True}
    if include_usage is not None:
        body["stream_options"] = {"include_usage": include_usage}
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200
    payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
    assert all(payload["model"] == MODEL for payload in payloads)
    assert any(payload.get("usage") == USAGE for payload in payloads) is bool(include_usage)
    assert response.text.endswith("data: [DONE]\n\n")
    assert json.loads(requests[0].content)["stream_options"] == {"include_usage": True}
    assert stream.closed
    assert upstream_limiter("chat-api")._in_flight == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_pre_output_failure_retains_503_retry_after_and_does_not_retry(remote_chat: tuple, streaming: bool) -> None:
    client, _, _, requests = remote_chat
    answer_with(remote_chat, json_response(503, {"error": {"message": "private"}}, headers={"retry-after": "19"}))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "19"
    assert response.json()["error"]["code"] == "QUEUE_FULL"
    assert "private" not in response.text
    assert len(requests) == 1
    assert upstream_limiter("chat-api")._in_flight == 0


def test_malformed_first_event_is_502_before_streaming_starts(remote_chat: tuple) -> None:
    client, _, _, _ = remote_chat
    stream = ChatStream(["private malformed body"])
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert response.status_code == 502
    assert "private" not in response.text
    assert stream.closed


def _remote_chat_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == openai_local.__name__]


@pytest.mark.parametrize("streaming", [False, True])
def test_a_502_before_output_logs_its_reason(
    remote_chat: tuple, streaming: bool, caplog: pytest.LogCaptureFixture
) -> None:
    client, _, _, _ = remote_chat
    answer_with(remote_chat, json_response(401, {"error": {"message": "private", "code": 401}}))
    with caplog.at_level(logging.WARNING, logger=openai_local.__name__):
        response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "the upstream rejected the server's credential"
    assert response.json()["error"]["reason"] == "upstream_credential_rejected"
    assert _remote_chat_warnings(caplog) == [f"remote chat for {MODEL} failed: upstream answered 401"]
    assert "private" not in caplog.text


@pytest.mark.parametrize("streaming", [False, True])
def test_a_missing_upstream_key_is_a_502_whose_log_names_it(
    remote_chat: tuple, streaming: bool, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, adapter, _, requests = remote_chat
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    keyed = adapter._upstream.model_copy(update={"api_key_secret": "CHAT_API_KEY"})
    adapter._async_client = upstream_client(keyed, transport=httpx.MockTransport(requests.append))
    with caplog.at_level(logging.WARNING, logger=openai_local.__name__):
        response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
    assert response.status_code == 502
    assert response.json()["error"]["message"] == "the server has no usable credential for its upstream"
    assert response.json()["error"]["reason"] == "upstream_credential_unavailable"
    assert _remote_chat_warnings(caplog) == [f"remote chat for {MODEL} failed: the upstream credential is unavailable"]
    assert not requests


@pytest.mark.parametrize("disconnect", [False, True])
def test_failure_after_output_is_in_band_and_never_retried(
    remote_chat: tuple, disconnect: bool, caplog: pytest.LogCaptureFixture
) -> None:
    client, _, _, requests = remote_chat
    stream = ChatStream(events()[:1], disconnect=disconnect)
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    with caplog.at_level(logging.WARNING, logger=openai_local.__name__):
        response = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert response.status_code == 200
    assert '"content":"answer"' in response.text
    assert '"error"' in response.text
    assert "private" not in response.text
    [warning] = _remote_chat_warnings(caplog)
    assert warning.startswith(f"remote chat for {MODEL} failed after output: ")
    assert "private" not in caplog.text
    assert len(requests) == 1
    assert stream.closed
    assert upstream_limiter("chat-api")._in_flight == 0


@pytest.mark.parametrize(
    "changes", [{"max_tokens": 65}, {"n": 0}, {"stream_options": []}, {"base_url": "http://other"}]
)
def test_invalid_requests_do_not_dispatch(remote_chat: tuple, changes: dict) -> None:
    client, _, _, requests = remote_chat
    answer_with(remote_chat, lambda: pytest.fail("invalid request must not dispatch"))
    response = client.post("/v1/chat/completions", json={**BODY, **changes})
    assert response.status_code == 400
    assert not requests


def test_remote_forbid_does_not_dispatch(remote_chat: tuple) -> None:
    client, _, _, requests = remote_chat
    answer_with(remote_chat, lambda: pytest.fail("forbidden request must not dispatch"))
    response = client.post("/v1/chat/completions", json=BODY, headers={"X-SIE-Remote": "forbid"})
    assert response.status_code == 400
    assert not requests


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {"prompt_tokens": 4097, "completion_tokens": 2, "total_tokens": 4099},
        {"prompt_tokens": 37, "completion_tokens": 65, "total_tokens": 102},
    ],
)
def test_missing_or_out_of_bound_usage_fails_closed(remote_chat: tuple, usage: dict | None) -> None:
    client, _, _, _ = remote_chat
    answer_with(remote_chat, json_response(200, completion(usage=usage)))
    assert client.post("/v1/chat/completions", json=BODY).status_code == 502


async def test_disconnect_before_body_delivery_closes_the_primed_upstream(remote_chat: tuple) -> None:
    _, adapter, config, _ = remote_chat
    stream = ChatStream(events())
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    response = await _remote_chat_response(
        adapter,
        {**BODY, "max_tokens": 64},
        requested_model=MODEL,
        config=config,
        stream=True,
        headers={},
        strict_grammar=None,
    )
    assert upstream_limiter("chat-api")._in_flight == 1

    async def send(_: dict) -> None:
        raise asyncio.CancelledError

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    with pytest.raises(asyncio.CancelledError):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert stream.closed
    assert upstream_limiter("chat-api")._in_flight == 0


@pytest.mark.parametrize("part_type", ["text", "input_text"])
def test_text_content_parts_remain_supported(remote_chat: tuple, part_type: str) -> None:
    client, _, _, requests = remote_chat
    answer_with(remote_chat, json_response(200, completion()))
    messages = [{"role": "user", "content": [{"type": part_type, "text": "question"}]}]
    assert client.post("/v1/chat/completions", json={**BODY, "messages": messages}).status_code == 200
    assert json.loads(requests[0].content)["messages"] == messages


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("hidden", [False, True])
def test_raw_upstream_response_obeys_the_ingress_byte_limit(
    remote_chat: tuple, monkeypatch: pytest.MonkeyPatch, streaming: bool, hidden: bool
) -> None:
    client, _, _, _ = remote_chat
    monkeypatch.setattr(openai_local, "_MAX_CHAT_RESPONSE_BYTES", 512)
    if streaming:
        event = {"choices": [{"index": 0, "delta": {"content": "x" * 2048}, "finish_reason": None}]}
        if hidden:
            event = {"choices": [], "discarded_private_metadata": "x" * 2048}
        stream = ChatStream([event, *events()])
        response = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    else:
        payload = (
            completion(discarded_private_metadata="x" * 2048)
            if hidden
            else completion(
                choices=[{"index": 0, "message": {"role": "assistant", "content": "x" * 2048}, "finish_reason": "stop"}]
            )
        )
        response = json_response(200, payload)
    answer_with(remote_chat, response)
    assert client.post("/v1/chat/completions", json={**BODY, "stream": streaming}).status_code == 502
    assert upstream_limiter("chat-api")._in_flight == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_strict_json_output_is_verified_after_remote_parsing(remote_chat: tuple, streaming: bool) -> None:
    client, _, _, _ = remote_chat
    if streaming:
        response = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ChatStream(events()))
    else:
        response = json_response(200, completion())
    answer_with(remote_chat, response)
    body = {
        **BODY,
        "stream": streaming,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": {"type": "object"}},
        },
    }
    result = client.post("/v1/chat/completions", json=body)
    if streaming:
        assert result.status_code == 200
        assert '"code":"MODEL_OUTPUT_PARSE_ERROR"' in result.text
    else:
        assert result.status_code == 500
        assert result.json()["error"]["code"] == "MODEL_OUTPUT_PARSE_ERROR"


def test_strict_invalid_first_event_preserves_the_buffered_error_before_200(remote_chat: tuple) -> None:
    client, _, _, _ = remote_chat
    stream = ChatStream([{"choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": "stop"}]}])
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    body = {
        **BODY,
        "stream": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": {"type": "object"}},
        },
    }
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "MODEL_OUTPUT_PARSE_ERROR"
    assert stream.closed
    assert upstream_limiter("chat-api")._in_flight == 0


@pytest.mark.parametrize("code", ["INPUT_TOO_LONG", "INVALID_INPUT"])
@pytest.mark.parametrize("streaming", [False, True])
def test_upstream_input_refusal_is_a_sanitized_400(remote_chat: tuple, code: str, streaming: bool) -> None:
    client, _, _, requests = remote_chat
    answer_with(
        remote_chat,
        json_response(400, {"error": {"message": "private upstream detail"}}, headers={"X-SIE-Error-Code": code}),
    )
    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == ("INPUT_TOO_LONG" if code == "INPUT_TOO_LONG" else "invalid_request")
    assert "private" not in response.text
    assert len(requests) == 1


@pytest.mark.parametrize(
    ("status", "caller_error"), [(400, True), (422, True), (401, False), (403, False), (404, False)]
)
@pytest.mark.parametrize("streaming", [False, True])
def test_upstream_refusal_without_a_code_is_the_callers_400_only_for_400_and_422(
    remote_chat: tuple, status: int, caller_error: bool, streaming: bool
) -> None:
    client, _, _, requests = remote_chat
    answer_with(
        remote_chat,
        json_response(status, {"error": {"message": "private upstream detail", "type": "invalid_request_error"}}),
    )
    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
    assert response.status_code == (400 if caller_error else 502)
    assert response.json()["error"]["code"] == ("invalid_request" if caller_error else "upstream_error")
    assert "private" not in response.text
    assert "chat-api" not in response.text
    assert len(requests) == 1


TOOLS = [
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "list_dir", "parameters": {"type": "object"}}},
]
READ_FILE = {"type": "function", "function": {"name": "read_file"}}


def tool_answer(names: list[str], *, streaming: bool) -> httpx.Response:
    calls = [
        {"id": f"call-{index}", "type": "function", "function": {"name": name, "arguments": "{}"}}
        for index, name in enumerate(names)
    ]
    finish = "tool_calls" if names else "stop"
    if not streaming:
        message = (
            {"role": "assistant", "content": None, "tool_calls": calls}
            if names
            else completion()["choices"][0]["message"]
        )
        return json_response(200, completion(choices=[{"index": 0, "message": message, "finish_reason": finish}]))
    delta = (
        {"role": "assistant", "tool_calls": [{"index": index, **call} for index, call in enumerate(calls)]}
        if names
        else {"role": "assistant", "content": "answer"}
    )
    stream = ChatStream(
        [
            {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
            {"choices": [], "usage": USAGE},
            "[DONE]",
        ]
    )
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)


@pytest.mark.parametrize(
    ("names", "changes"),
    [
        pytest.param(["delete_everything"], {}, id="undeclared"),
        pytest.param(["read_file"], {"tool_choice": "none"}, id="choice-none"),
        pytest.param(["read_file", "list_dir"], {"parallel_tool_calls": False}, id="parallel-false"),
        pytest.param(["list_dir"], {"tool_choice": READ_FILE}, id="named-other"),
        pytest.param([], {"tool_choice": READ_FILE}, id="named-missing"),
        pytest.param([], {"tool_choice": "required"}, id="required-missing"),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_direct_remote_chat_refuses_an_answer_outside_the_requested_tools(
    remote_chat: tuple, names: list[str], changes: dict, streaming: bool
) -> None:
    client, _, _, requests = remote_chat
    answer_with(remote_chat, tool_answer(names, streaming=streaming))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming, "tools": TOOLS, **changes})
    if streaming and not names:
        assert response.status_code == 200
        assert '"content":"answer"' in response.text
        assert '"code":"upstream_error"' in response.text
        assert '"finish_reason":"stop"' not in response.text
    else:
        assert response.status_code == 502, response.text
        assert response.json()["error"]["code"] == "upstream_error"
    assert not any(name in response.text for name in names)
    assert len(requests) == 1
    assert upstream_limiter("chat-api")._in_flight == 0


@pytest.mark.parametrize(
    ("names", "changes"),
    [
        pytest.param(["read_file"], {}, id="declared"),
        pytest.param(["read_file", "list_dir"], {}, id="parallel"),
        pytest.param(["read_file"], {"parallel_tool_calls": False}, id="parallel-false"),
        pytest.param(["read_file"], {"tool_choice": READ_FILE}, id="named"),
        pytest.param(["list_dir"], {"tool_choice": "required"}, id="required"),
        pytest.param([], {"tool_choice": "none"}, id="choice-none"),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_direct_remote_chat_serves_an_answer_within_the_requested_tools(
    remote_chat: tuple, names: list[str], changes: dict, streaming: bool
) -> None:
    client, _, _, _ = remote_chat
    answer_with(remote_chat, tool_answer(names, streaming=streaming))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming, "tools": TOOLS, **changes})
    assert response.status_code == 200, response.text
    assert '"error"' not in response.text
    if streaming:
        assert response.text.endswith("data: [DONE]\n\n")
        assert all(f'"name":"{name}"' in response.text for name in names)
    else:
        calls = response.json()["choices"][0]["message"].get("tool_calls") or []
        assert [call["function"]["name"] for call in calls] == names


def tool_fragments(fragments: list[tuple[int, str]]) -> httpx.Response:
    frames: list[dict | str] = []
    started: set[int] = set()
    for index, name in fragments:
        tool: dict[str, Any] = {"index": index, "function": {"name": name}}
        if index not in started:
            started.add(index)
            tool.update(id=f"call-{index}", type="function")
            tool["function"]["arguments"] = "{}"
        frames.append({"choices": [{"index": 0, "delta": {"tool_calls": [tool]}, "finish_reason": None}]})
    frames += [
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": USAGE},
        "[DONE]",
    ]
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ChatStream(frames))


@pytest.mark.parametrize(
    ("fragments", "served", "delivered"),
    [
        pytest.param([(0, "read"), (0, "_file")], True, {0: "read_file"}, id="name-split-across-events"),
        pytest.param(
            [(0, "read_file"), (1, "list_dir")],
            True,
            {0: "read_file", 1: "list_dir"},
            id="second-call-in-a-later-event",
        ),
        pytest.param([(0, "read_file"), (0, "read_file")], False, {0: "read_file"}, id="name-repeated-per-delta"),
    ],
)
def test_direct_remote_chat_accumulates_each_streamed_tool_name_by_call_index(
    remote_chat: tuple, fragments: list[tuple[int, str]], served: bool, delivered: dict[int, str]
) -> None:
    client, _, _, requests = remote_chat
    answer_with(remote_chat, tool_fragments(fragments))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": True, "tools": TOOLS})
    assert response.status_code == 200, response.text
    names: dict[int, str] = {}
    for line in response.text.splitlines():
        if line.startswith("data: {"):
            for choice in json.loads(line[6:]).get("choices") or []:
                for call in choice["delta"].get("tool_calls") or []:
                    names[call["index"]] = names.get(call["index"], "") + call["function"].get("name", "")
    assert names == delivered
    assert ('"finish_reason":"tool_calls"' in response.text) is served
    assert ('"code":"upstream_error"' in response.text) is not served
    assert response.text.endswith("data: [DONE]\n\n")
    assert len(requests) == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_direct_remote_chat_serves_explicit_null_tools(remote_chat: tuple, streaming: bool) -> None:
    client, _, _, requests = remote_chat
    answer_with(remote_chat, tool_answer([], streaming=streaming))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming, "tools": None})
    assert response.status_code == 200, response.text
    assert "answer" in response.text
    assert '"error"' not in response.text
    assert len(requests) == 1


@pytest.mark.parametrize("name", ["math.factorial", "two words", "café", "a/b", "a" * 65])
def test_tool_names_outside_the_openai_pattern_never_dispatch(remote_chat: tuple, name: str) -> None:
    client, _, _, requests = remote_chat
    answer_with(remote_chat, lambda: pytest.fail("an invalid tool name must not dispatch"))
    tools = [TOOLS[0], {"type": "function", "function": {"name": name}}]
    response = client.post("/v1/chat/completions", json={**BODY, "tools": tools})
    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == "tools[1].function.name"
    assert error["type"] == "invalid_request_error"
    unnamed = client.post("/v1/chat/completions", json={**BODY, "tools": [{"type": "function", "function": {}}]})
    assert error["code"] == unnamed.json()["error"]["code"] == "INVALID_INPUT"
    assert not requests


@pytest.mark.parametrize("name", ["get-weather_2", "A" * 64])
def test_tool_names_inside_the_openai_pattern_dispatch(remote_chat: tuple, name: str) -> None:
    client, _, _, requests = remote_chat
    answer_with(remote_chat, json_response(200, completion()))
    tools = [TOOLS[0], {"type": "function", "function": {"name": name}}]
    assert client.post("/v1/chat/completions", json={**BODY, "tools": tools}).status_code == 200
    assert [tool["function"]["name"] for tool in json.loads(requests[0].content)["tools"]] == ["read_file", name]


@pytest.mark.parametrize("streaming", [False, True])
def test_cached_serving_switch_refusal_is_a_sanitized_503(remote_chat: tuple, streaming: bool) -> None:
    client, adapter, _, requests = remote_chat
    answer_with(remote_chat, lambda: pytest.fail("disabled serving must not dispatch"))
    install_upstreams({"chat-api": adapter._upstream}, remote_serving=False)
    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "QUEUE_FULL"
    assert not requests


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
def test_remote_reasoning_is_not_exposed_in_logprobs(remote_chat: tuple, streaming: bool, reasoning_field: str) -> None:
    client, _, _, _ = remote_chat
    raw = completion()
    raw["choices"][0]["message"][reasoning_field] = "PRIVATE_REASONING"
    raw["choices"][0]["logprobs"] = {"content": [{"token": "PRIVATE_REASONING", "logprob": -0.5}]}
    if streaming:
        chunks = events()
        chunks[0]["choices"][0]["delta"][reasoning_field] = "PRIVATE_REASONING"
        chunks[0]["choices"][0]["logprobs"] = raw["choices"][0]["logprobs"]
        answer_with(
            remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ChatStream(chunks))
        )
    else:
        answer_with(remote_chat, json_response(200, raw))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming, "logprobs": True})
    assert response.status_code == 200
    assert "PRIVATE_REASONING" not in response.text


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("declares_reasoning", [False, True])
def test_include_reasoning_publishes_remote_reasoning_only_for_a_reasoning_model(
    remote_chat: tuple, streaming: bool, declares_reasoning: bool
) -> None:
    client, _, config, _ = remote_chat
    if declares_reasoning:
        config.tasks.generate.chat_template_kwargs = {"enable_thinking": True}
    raw = completion()
    raw["choices"][0]["message"]["reasoning_content"] = "PRIVATE_REASONING"
    if streaming:
        chunks = events()
        chunks[0]["choices"][0]["delta"]["reasoning_content"] = "PRIVATE_REASONING"
        answer_with(
            remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ChatStream(chunks))
        )
    else:
        answer_with(remote_chat, json_response(200, raw))

    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming, "include_reasoning": True})

    assert response.status_code == 200
    if streaming:
        payloads = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
        reasoning = "".join(p["choices"][0]["delta"].get("reasoning_content", "") for p in payloads if p["choices"])
    else:
        reasoning = response.json()["choices"][0]["message"].get("reasoning_content", "")
    assert reasoning == ("PRIVATE_REASONING" if declares_reasoning else "")


@pytest.mark.parametrize(("status", "code"), [(400, "INVALID_INPUT"), (400, None), (503, None)])
def test_native_remote_stream_refusal_precedes_http_success(remote_chat: tuple, status: int, code: str | None) -> None:
    client, _, config, requests = remote_chat
    config.profiles["default"].adapter_options.runtime.clear()
    answer_with(
        remote_chat,
        json_response(status, {"error": {"code": code, "message": "private"}}, headers={"retry-after": "19"}),
    )
    response = client.post(
        f"/v1/generate/{SAFE_MODEL}", json={"prompt": "question", "max_new_tokens": 8, "stream": True}
    )
    assert response.status_code == status, response.text
    assert "text/event-stream" not in response.headers["content-type"]
    assert "private" not in response.text
    if status == 503:
        assert response.headers["retry-after"] == "19"
    assert len(requests) == 1
    assert upstream_limiter("chat-api")._in_flight == 0


def test_native_remote_failure_after_output_remains_a_stream_error(remote_chat: tuple) -> None:
    client, adapter, config, requests = remote_chat
    config.profiles["default"].adapter_options.runtime.clear()
    first = (
        {"choices": [{"index": 0, "text": "answer", "finish_reason": None}]}
        if isinstance(adapter, OpenAIUpstreamAdapter)
        else {"request_id": "upstream-id", "seq": 0, "text_delta": "answer", "done": False}
    )
    stream = ChatStream([first], disconnect=True)
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    response = client.post(
        f"/v1/generate/{SAFE_MODEL}", json={"prompt": "question", "max_new_tokens": 8, "stream": True}
    )
    assert response.status_code == 200, response.text
    assert "answer" in response.text
    assert '"finish_reason": "error"' in response.text
    assert "private upstream detail" not in response.text
    assert len(requests) == 1
    assert stream.closed
    assert upstream_limiter("chat-api")._in_flight == 0


CREDENTIAL = "sk-canary-0d3c7a51e9b24f68"
PROMPT = "private prompt 5c2e"
UNAVAILABLE = "upstream 'chat-api' is unavailable: request failed (ConnectError)"


def _render_chat_locally(remote_chat: tuple, monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, config, _ = remote_chat
    data = config.model_dump()
    data.update(remote_backed=False, hf_id="onboarded/model", hf_revision="a" * 40)
    data["profiles"]["default"]["adapter_options"]["runtime"] = {}
    client.app.state.registry.get_config.return_value = ModelConfig.model_validate(data)
    template = SimpleNamespace(
        chat_template="template", apply_chat_template=lambda messages, **_: messages[-1]["content"]
    )
    monkeypatch.setattr(remote_chat_prompt, "_tokenizer", lambda *args: template)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("surface", "failure", "status", "source", "message"),
    [
        ("chat", "404", 502, "route", f"remote chat for {MODEL} failed: upstream answered 404"),
        ("chat", "connect", 503, "route", f"remote chat for {MODEL} failed: {UNAVAILABLE}"),
        ("rendered chat", "401", 502, "adapter", "remote generation failed: upstream answered 401"),
        ("rendered chat", "connect", 503, "adapter", f"remote generation failed: {UNAVAILABLE}"),
        ("native", "connect", 503, "adapter", f"remote generation failed: {UNAVAILABLE}"),
        ("completions", "connect", 503, "adapter", f"remote generation failed: {UNAVAILABLE}"),
    ],
)
def test_a_remote_generation_failure_logs_one_warning_with_its_reason(
    remote_chat: tuple,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    surface: str,
    failure: str,
    status: int,
    source: str,
    message: str,
    streaming: bool,
) -> None:
    client, adapter, config, requests = remote_chat
    config.profiles["default"].adapter_options.runtime.clear()
    if surface == "rendered chat":
        _render_chat_locally(remote_chat, monkeypatch)
    adapter._upstream = adapter._upstream.model_copy(update={"api_key_secret": "CHAT_API_KEY"})
    monkeypatch.setenv("CHAT_API_KEY", CREDENTIAL)

    def respond() -> httpx.Response:
        if failure == "connect":
            raise httpx.ConnectError("private upstream detail")
        return json_response(int(failure), {"error": {"message": "private upstream detail"}})

    answer_with(remote_chat, respond)
    if surface == "native":
        path, body = f"/v1/generate/{SAFE_MODEL}", {"prompt": PROMPT, "max_new_tokens": 8, "stream": streaming}
    elif surface == "completions":
        path, body = "/v1/completions", {"model": MODEL, "prompt": PROMPT, "max_tokens": 8, "stream": streaming}
    else:
        path, body = "/v1/chat/completions", {**BODY, "messages": [{"role": "user", "content": PROMPT}]}
        body["stream"] = streaming
    caplog.set_level(logging.DEBUG)
    response = client.post(path, json=body)

    assert response.status_code == status, response.text
    payload = response.json()
    reason = {"401": "upstream_credential_rejected", "404": "upstream_model_not_found"}.get(
        failure, "upstream_unavailable"
    )
    assert payload.get("error", payload.get("detail"))["reason"] == reason
    logger_name = openai_local.__name__ if source == "route" else type(adapter).__module__
    warnings = [(record.name, record.getMessage()) for record in caplog.records if record.levelno >= logging.WARNING]
    assert warnings == [(logger_name, message)]
    assert [request.headers["authorization"] for request in requests] == [f"Bearer {CREDENTIAL}"]
    assert requests[0].url.path.endswith("/chat/completions") is (surface == "chat")
    for private in (CREDENTIAL, PROMPT, "private upstream detail"):
        assert private not in caplog.text


@pytest.mark.parametrize("streaming", [False, True])
def test_a_native_remote_failure_before_output_logs_one_warning_with_its_reason(
    remote_chat: tuple, caplog: pytest.LogCaptureFixture, streaming: bool
) -> None:
    client, adapter, config, _ = remote_chat
    config.profiles["default"].adapter_options.runtime.clear()
    answer_with(remote_chat, json_response(401, {"error": {"message": "private upstream detail"}}))
    caplog.set_level(logging.DEBUG)
    response = client.post(
        f"/v1/generate/{SAFE_MODEL}", json={"prompt": PROMPT, "max_new_tokens": 8, "stream": streaming}
    )

    assert response.status_code == 500
    [record] = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert (record.name, record.getMessage()) == (
        type(adapter).__module__,
        "remote generation failed: upstream answered 401",
    )
    for private in (PROMPT, "private upstream detail"):
        assert private not in caplog.text


_GENERATION_SURFACES = [
    ("native", False),
    ("native", True),
    ("chat", False),
    ("chat", True),
    ("completions", False),
    ("completions", True),
    ("responses", False),
]


def _cold_generation_bridge(remote_chat: tuple) -> MagicMock:
    client, _, remote_config, _ = remote_chat
    remote_config.profiles["default"].adapter_options.runtime.clear()
    data = remote_config.model_dump(mode="json")
    data.update(
        remote_backed=False, hf_id="weights/model", routing={"policy": "fallback", "fallback_profile": "remote"}
    )
    data["profiles"] = {
        "remote": data["profiles"]["default"],
        "default": {
            "adapter_path": "sie_server.adapters.sglang:SGLangGenerationAdapter",
            "max_batch_tokens": 8192,
            "kv_budget_tokens": 4096,
        },
    }
    config = ModelConfig.model_validate(data)
    variants = expand_profile_variants([config])
    registry = client.app.state.registry
    registry.get_config.side_effect = variants.__getitem__
    registry.is_loading.side_effect = lambda name: name == MODEL
    registry.is_loaded.side_effect = lambda name: name != MODEL
    registry.start_load_async = AsyncMock(return_value=True)
    return registry


def _generation_request(surface: str, *, stream: bool) -> tuple[str, dict[str, Any]]:
    if surface == "native":
        return f"/v1/generate/{SAFE_MODEL}", {"prompt": "question", "max_new_tokens": 8, "stream": stream}
    if surface == "chat":
        return "/v1/chat/completions", {**BODY, "stream": stream, "max_tokens": 8}
    if surface == "responses":
        return "/v1/responses", {"model": MODEL, "input": "question", "max_output_tokens": 8}
    return "/v1/completions", {"model": MODEL, "prompt": "question", "max_tokens": 8, "stream": stream}


@pytest.mark.parametrize(("surface", "streaming"), _GENERATION_SURFACES)
@pytest.mark.parametrize("upstream_status", [400, 503, 500, 200])
def test_generation_bridge_restores_original_local_refusal_across_surfaces(
    remote_chat: tuple, surface: str, streaming: bool, upstream_status: int
) -> None:
    client, _, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat)
    answer_with(
        remote_chat,
        json_response(
            upstream_status, {"error": {"code": "INVALID_INPUT", "message": "private"}}, headers={"retry-after": "19"}
        ),
    )
    path, body = _generation_request(surface, stream=streaming)
    response = client.post(path, json=body)
    assert response.status_code == 503, response.text
    payload = response.json()
    error = payload.get("error", payload.get("detail"))
    assert error["code"] == "MODEL_LOADING"
    assert response.headers["retry-after"] == "5"
    assert response.headers["X-SIE-Served-By"] == "local"
    assert response.headers["X-SIE-Fallback-Reason"] == "model_loading"
    assert "X-SIE-Fallback-Error" in response.headers
    assert "private" not in response.text
    assert len(requests) == 1
    registry.start_load_async.assert_awaited_once_with(MODEL, "cpu")
    assert upstream_limiter("chat-api")._in_flight == 0


@pytest.mark.parametrize(("surface", "streaming"), _GENERATION_SURFACES)
@pytest.mark.parametrize("upstream_status", [400, 422])
def test_uncoded_upstream_refusal_is_a_logged_callers_400_on_every_generation_surface(
    remote_chat: tuple, surface: str, streaming: bool, upstream_status: int, caplog: pytest.LogCaptureFixture
) -> None:
    client, _, config, requests = remote_chat
    config.profiles["default"].adapter_options.runtime.clear()
    answer_with(
        remote_chat,
        json_response(upstream_status, {"error": {"message": "private upstream detail", "param": "top_k"}}),
    )
    path, body = _generation_request(surface, stream=streaming)
    with caplog.at_level("WARNING", logger="sie_server.adapters.remote._http"):
        response = client.post(path, json=body)
    assert response.status_code == 400, response.text
    payload = response.json()
    assert payload.get("error", payload.get("detail"))["code"] == "invalid_request"
    assert "private" not in response.text
    assert "chat-api" not in response.text
    assert "X-SIE-Upstream" not in response.headers
    assert len(requests) == 1
    refusals = [record.getMessage() for record in caplog.records if record.name == "sie_server.adapters.remote._http"]
    assert refusals == [
        f"upstream 'chat-api' refused a generation request: the upstream refused the input ({upstream_status})"
    ]


@pytest.mark.parametrize(("surface", "streaming"), _GENERATION_SURFACES)
def test_remote_forbid_keeps_cold_generation_off_the_bridge(remote_chat: tuple, surface: str, streaming: bool) -> None:
    client, _, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat)
    answer_with(remote_chat, json_response(200, completion()))
    path, body = _generation_request(surface, stream=streaming)
    response = client.post(path, json=body, headers={"X-SIE-Remote": "forbid"})
    assert response.status_code == 503, response.text
    assert "X-SIE-Fallback-Reason" not in response.headers
    assert not requests
    registry.start_load_async.assert_not_awaited()


@pytest.mark.parametrize("surface", ["native", "chat", "completions", "responses"])
def test_buffered_generation_bridge_discloses_success_and_warms_local(remote_chat: tuple, surface: str) -> None:
    client, adapter, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat)
    if surface == "chat":
        answer_with(remote_chat, json_response(200, completion()))
    else:
        if isinstance(adapter, OpenAIUpstreamAdapter):
            upstream_events = [
                {"choices": [{"index": 0, "text": "answer", "finish_reason": None}]},
                {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}]},
                {"choices": [], "usage": USAGE},
                "[DONE]",
            ]
        else:
            upstream_events = [
                {"request_id": "upstream-id", "seq": 0, "text_delta": "answer", "done": False},
                {
                    "request_id": "upstream-id",
                    "seq": 1,
                    "text_delta": "",
                    "done": True,
                    "finish_reason": "stop",
                    "usage": USAGE,
                },
                "[DONE]",
            ]
        answer_with(
            remote_chat,
            httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ChatStream(upstream_events)),
        )
    path, body = _generation_request(surface, stream=False)
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert "answer" in response.text
    assert response.json()["model"] == MODEL
    assert response.headers["X-SIE-Served-By"] == "remote"
    assert response.headers["X-SIE-Fallback-Reason"] == "model_loading"
    assert response.headers["X-SIE-Upstream"] == "chat-api"
    assert len(requests) == 1
    registry.start_load_async.assert_awaited_once_with(MODEL, "cpu")
    assert upstream_limiter("chat-api")._in_flight == 0


@pytest.mark.parametrize("surface", ["native", "chat", "completions"])
def test_generation_bridge_never_restores_a_refusal_after_output(remote_chat: tuple, surface: str) -> None:
    client, adapter, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat)
    if surface == "chat":
        first = events()[0]
    elif isinstance(adapter, OpenAIUpstreamAdapter):
        first = {"choices": [{"index": 0, "text": "answer", "finish_reason": None}]}
    else:
        first = {"request_id": "upstream-id", "seq": 0, "text_delta": "answer", "done": False}
    stream = ChatStream([first], disconnect=True)
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    path, body = _generation_request(surface, stream=True)
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert "answer" in response.text
    assert '"error"' in response.text
    assert "MODEL_LOADING" not in response.text
    assert "private upstream detail" not in response.text
    assert response.headers["X-SIE-Served-By"] == "remote"
    assert len(requests) == 1
    registry.start_load_async.assert_awaited_once_with(MODEL, "cpu")
    assert stream.closed
    assert upstream_limiter("chat-api")._in_flight == 0


@pytest.mark.parametrize(("surface", "streaming"), _GENERATION_SURFACES)
def test_invalid_generation_never_starts_a_bridge(remote_chat: tuple, surface: str, streaming: bool) -> None:
    client, _, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat)
    answer_with(remote_chat, json_response(200, completion()))
    path, body = _generation_request(surface, stream=streaming)
    body[
        "max_new_tokens" if surface == "native" else "max_output_tokens" if surface == "responses" else "max_tokens"
    ] = 1000
    response = client.post(path, json=body)
    assert response.status_code == 400, response.text
    assert "X-SIE-Fallback-Reason" not in response.headers
    assert not requests
    registry.start_load_async.assert_not_awaited()


@pytest.mark.parametrize(
    "options", [{"include_usage": "private-not-bool"}, {"include_usage": 1}, {"extra": True}, [], True]
)
@pytest.mark.parametrize("streaming", [False, True])
def test_malformed_chat_stream_options_never_start_a_bridge(remote_chat: tuple, options: Any, streaming: bool) -> None:
    client, _, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat)
    answer_with(remote_chat, json_response(200, completion()))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming, "stream_options": options})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "stream_options"
    assert not requests
    registry.start_load_async.assert_not_awaited()


def test_chat_bridge_does_not_dispatch_unsupported_media(remote_chat: tuple) -> None:
    client, _, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat)
    answer_with(remote_chat, json_response(200, completion()))
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "MODEL_LOADING"
    assert not requests
    registry.start_load_async.assert_awaited_once_with(MODEL, "cpu")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", "bad"),
        ("temperature", True),
        ("top_p", 2),
        ("top_k", 1.5),
        ("logprobs", "bad"),
        ("top_logprobs", True),
        ("logit_bias", []),
        ("stop", 7),
        ("stop", ["stop", 7]),
        ("tools", True),
        ("tools", [{"type": "function", "function": {"name": "x", "parameters": {"type": 7}}}]),
    ],
)
@pytest.mark.parametrize("streaming", [False, True])
def test_malformed_common_chat_fields_never_start_a_bridge(
    remote_chat: tuple, field: str, value: Any, streaming: bool
) -> None:
    client, _, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat)
    answer_with(remote_chat, json_response(400, {"error": {"message": "private"}}))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": streaming, field: value})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == field
    assert not requests
    registry.start_load_async.assert_not_awaited()


@pytest.mark.parametrize("choice", [None, "auto", "none"])
def test_empty_chat_tools_remain_valid_on_a_bridge(remote_chat: tuple, choice: str | None) -> None:
    client, _, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat)
    registry.device = "cuda"
    answer_with(remote_chat, json_response(200, completion()))
    body = {**BODY, "tools": []}
    if choice is not None:
        body["tool_choice"] = choice
    response = client.post("/v1/chat/completions", json=body)
    assert response.status_code == 200, response.text
    assert len(requests) == 1
    registry.start_load_async.assert_awaited_once_with(MODEL, "cuda")


@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "missing"}}])
def test_empty_chat_tools_cannot_satisfy_a_required_choice(remote_chat: tuple, choice: Any) -> None:
    client, _, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat)
    registry.device = "cuda"
    response = client.post("/v1/chat/completions", json={**BODY, "tools": [], "tool_choice": choice})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "tool_choice"
    assert not requests
    registry.start_load_async.assert_not_awaited()


@pytest.mark.parametrize("tools", [{}, {"tools": None}], ids=["no-tools", "null-tools"])
@pytest.mark.parametrize("stream", [False, True])
def test_onboarded_direct_chat_uses_raw_template_without_chat_dispatch(
    remote_chat, monkeypatch: pytest.MonkeyPatch, stream: bool, tools: dict
) -> None:
    client, adapter, config, requests = remote_chat
    data = config.model_dump()
    data.update(remote_backed=False, hf_id="onboarded/model", hf_revision="a" * 40)
    data["profiles"]["default"]["adapter_options"]["runtime"] = {}
    onboarded = ModelConfig.model_validate(data)
    client.app.state.registry.get_config.return_value = onboarded
    monkeypatch.setattr(
        remote_chat_prompt,
        "_tokenizer",
        lambda *args: SimpleNamespace(
            chat_template="template", apply_chat_template=lambda *args, **kwargs: "owned prompt"
        ),
    )
    if isinstance(adapter, OpenAIUpstreamAdapter):
        raw_events = [
            {"choices": [{"index": 0, "text": "answer", "finish_reason": None}]},
            {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}], "usage": USAGE},
            "[DONE]",
        ]
        path = "/prefix/v1/completions"
    else:
        raw_events = [
            {"text_delta": "answer", "done": False},
            {"text_delta": "", "done": True, "finish_reason": "stop", "usage": USAGE},
        ]
        path = "/prefix/v1/generate/operator__model"
    upstream_stream = ChatStream(raw_events)
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream_stream))
    response = client.post(
        "/v1/chat/completions",
        json={
            **BODY,
            "max_tokens": 32,
            "max_completion_tokens": None,
            "stream": stream,
            "stream_options": {"include_usage": True},
            **tools,
        },
    )
    assert response.status_code == 200
    assert len(requests) == 1
    assert requests[0].url.path == path
    assert json.loads(requests[0].content)["prompt"] == "owned prompt"
    sent = json.loads(requests[0].content)
    assert sent.get("max_tokens", sent.get("max_new_tokens")) == 32
    assert response.headers["X-SIE-Served-By"] == "remote"
    assert "answer" in response.text
    assert "owned prompt" not in response.text
    assert '"error"' not in response.text
    assert upstream_stream.closed
    if not stream:
        assert response.json()["choices"][0]["message"]["content"] == "answer"
        assert response.json()["usage"] == USAGE
    else:
        assert response.text.endswith("data: [DONE]\n\n")


@pytest.mark.parametrize(("surface", "streaming"), [entry for entry in _GENERATION_SURFACES if entry[0] != "chat"])
def test_native_upstream_prompt_reasoning_ownership_across_surfaces(remote_chat, surface, streaming) -> None:
    client, adapter, config, requests = remote_chat
    config.profiles["default"].adapter_options.runtime.clear()
    config.tasks.generate.chat_template_kwargs = {"enable_thinking": False}
    text = "answer<think>PRIVATE_BLOCK</think>tail"
    if isinstance(adapter, OpenAIUpstreamAdapter):
        frames = [
            {"choices": [{"index": 0, "text": "PRIVATE_PROMPT</think>" + text, "finish_reason": None}]},
            {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}], "usage": USAGE},
            "[DONE]",
        ]
    else:
        frames = [{"text_delta": text, "done": False}, {"done": True, "finish_reason": "stop", "usage": USAGE}]
    stream = ChatStream(frames)
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    path, request = _generation_request(surface, stream=streaming)
    request["input" if surface == "responses" else "prompt"] = "owned prompt<think>"
    response = client.post(path, json=request)
    assert response.status_code == 200, response.text
    assert "answer" in response.text
    assert "tail" in response.text
    assert "PRIVATE" not in response.text
    assert '"error"' not in response.text
    assert len(requests) == 1
    assert stream.closed


@pytest.mark.parametrize("response_format", ["json", ["json"], 1, True, False, [], ""])
@pytest.mark.parametrize("cold", [False, True])
def test_invalid_response_format_fails_before_template_load_or_dispatch(remote_chat, response_format, cold) -> None:
    client, _, _, requests = remote_chat
    registry = _cold_generation_bridge(remote_chat) if cold else client.app.state.registry
    response = client.post("/v1/chat/completions", json={**BODY, "response_format": response_format})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "response_format"
    assert not requests
    registry.start_load_async.assert_not_called()


@pytest.mark.parametrize("remote_chat", ["openai"], indirect=True)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("include_reasoning", [False, True])
def test_onboarded_direct_chat_answers_rendered_reasoning_only_when_asked(
    remote_chat, monkeypatch: pytest.MonkeyPatch, stream: bool, include_reasoning: bool
) -> None:
    client, _, config, _ = remote_chat
    data = config.model_dump()
    data.update(remote_backed=False, hf_id="onboarded/model", hf_revision="a" * 40)
    data["profiles"]["default"]["adapter_options"]["runtime"] = {}
    data["tasks"]["generate"]["chat_template_kwargs"] = {"enable_thinking": True}
    client.app.state.registry.get_config.return_value = ModelConfig.model_validate(data)
    monkeypatch.setattr(
        remote_chat_prompt,
        "_tokenizer",
        lambda *args: SimpleNamespace(
            chat_template="template", apply_chat_template=lambda *args, **kwargs: "owned prompt<think>"
        ),
    )
    upstream_stream = ChatStream(
        [
            {"choices": [{"index": 0, "text": "PRIVATE_REASONING</think>answer", "finish_reason": None}]},
            {"choices": [{"index": 0, "text": "", "finish_reason": "stop"}], "usage": USAGE},
            "[DONE]",
        ]
    )
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=upstream_stream))

    response = client.post(
        "/v1/chat/completions",
        json={**BODY, "max_tokens": 32, "stream": stream, "include_reasoning": include_reasoning},
    )

    assert response.status_code == 200
    if stream:
        payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
        deltas = [payload["choices"][0]["delta"] for payload in payloads if payload["choices"]]
        reasoning = "".join(delta.get("reasoning_content", "") for delta in deltas)
        content = "".join(delta.get("content") or "" for delta in deltas)
    else:
        message = response.json()["choices"][0]["message"]
        reasoning, content = message.get("reasoning_content", ""), message["content"]
    assert content == "answer"
    assert reasoning == ("PRIVATE_REASONING" if include_reasoning else "")
    assert "think>" not in response.text
    assert upstream_stream.closed


def test_the_upstream_failure_reasons_are_the_wire_fixture() -> None:
    fixture = json.loads(
        (Path(__file__).resolve().parents[3] / "wire-fixtures" / "upstream_failure_reasons.json").read_text(
            encoding="utf-8"
        )
    )
    assert UPSTREAM_FAILURE_MESSAGES == fixture["reasons"]


ACCOUNT = "acct-7Q2Lw0d9"
_FINAL_FAILURES = {
    "401": "upstream_credential_rejected",
    "402": "upstream_payment_required",
    "403": "upstream_credential_rejected",
    "404": "upstream_model_not_found",
    "409": "upstream_refused",
    "501": "upstream_unavailable",
    "malformed": "upstream_invalid_response",
}
_UPSTREAM_IDENTIFIERS = ("chat-api", "127.0.0.1", "/prefix", "operator/model", ACCOUNT, "private upstream detail")


def _answer_with_failure(remote_chat: tuple, failure: str) -> None:
    _, adapter, _, requests = remote_chat

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        detail = f"private upstream detail: account {ACCOUNT} at {request.url}"
        if failure == "connect":
            raise httpx.ConnectError(detail)
        if failure != "malformed":
            return json_response(int(failure), {"error": {"message": detail, "code": int(failure)}})
        if request.headers["accept"] == "text/event-stream":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=ChatStream([{"detail": detail}])
            )
        return json_response(200, {"detail": detail})

    adapter._async_client = upstream_client(adapter._upstream, transport=httpx.MockTransport(respond))


@pytest.mark.parametrize(("surface", "streaming"), _GENERATION_SURFACES)
@pytest.mark.parametrize("failure", ["401", "402", "403", "404", "409", "501", "connect", "malformed"])
def test_an_upstream_failure_tells_the_caller_its_reason_and_nothing_that_identifies_the_upstream(
    remote_chat: tuple, monkeypatch: pytest.MonkeyPatch, surface: str, streaming: bool, failure: str
) -> None:
    client, adapter, config, requests = remote_chat
    config.profiles["default"].adapter_options.runtime.clear()
    adapter._upstream = adapter._upstream.model_copy(update={"api_key_secret": "CHAT_API_KEY"})
    monkeypatch.setenv("CHAT_API_KEY", CREDENTIAL)
    _answer_with_failure(remote_chat, failure)
    path, body = _generation_request(surface, stream=streaming)

    response = client.post(path, json=body)

    payload = response.json()
    error = payload.get("error", payload.get("detail"))
    if failure == "connect":
        assert response.status_code == 503, response.text
        assert error["code"] == ("QUEUE_FULL" if surface == "chat" else "RESOURCE_EXHAUSTED")
        assert error["reason"] == "upstream_unavailable"
    else:
        reason = _FINAL_FAILURES[failure]
        assert response.status_code == (502 if surface == "chat" else 500), response.text
        assert error["code"] == ("upstream_error" if surface == "chat" else "inference_error")
        assert error["reason"] == reason
        assert error["message"] == UPSTREAM_FAILURE_MESSAGES[reason]
    assert [request.headers["authorization"] for request in requests] == [f"Bearer {CREDENTIAL}"]
    for private in (*_UPSTREAM_IDENTIFIERS, CREDENTIAL):
        assert private not in response.text


@pytest.mark.parametrize("surface", ["native", "chat", "completions"])
def test_an_upstream_lost_after_output_ends_the_stream_with_its_reason(remote_chat: tuple, surface: str) -> None:
    client, adapter, config, requests = remote_chat
    config.profiles["default"].adapter_options.runtime.clear()
    if surface == "chat":
        first = events()[0]
    elif isinstance(adapter, OpenAIUpstreamAdapter):
        first = {"choices": [{"index": 0, "text": "answer", "finish_reason": None}]}
    else:
        first = {"request_id": "upstream-id", "seq": 0, "text_delta": "answer", "done": False}
    stream = ChatStream([first], disconnect=True)
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    path, body = _generation_request(surface, stream=True)

    response = client.post(path, json=body)

    assert response.status_code == 200, response.text
    assert "answer" in response.text
    events_with_errors = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: {") and '"error"' in line
    ]
    assert [event["error"]["reason"] for event in events_with_errors] == ["upstream_unavailable"]
    assert events_with_errors[0]["error"]["message"] == UPSTREAM_FAILURE_MESSAGES["upstream_unavailable"]
    for private in _UPSTREAM_IDENTIFIERS:
        assert private not in response.text
    assert len(requests) == 1
    assert stream.closed
