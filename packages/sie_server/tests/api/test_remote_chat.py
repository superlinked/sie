import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters.remote._limits import upstream_limiter
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.api import openai_local
from sie_server.api.openai_local import _remote_chat_response, router
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.upstream_client import upstream_client

MODEL = "caller/model"
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


@pytest.fixture
def remote_chat() -> Iterator[tuple[TestClient, SieUpstreamAdapter, ModelConfig, list[httpx.Request]]]:
    upstream = Upstream.model_validate(
        {
            "kind": "sie",
            "base_url": "http://127.0.0.1:8088/prefix",
            "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
        }
    )
    install_upstreams({"chat-api": upstream})
    adapter = SieUpstreamAdapter(upstream="chat-api", upstream_model="operator/model")
    adapter.load("cpu")
    config = ModelConfig.model_validate(
        {
            "sie_id": MODEL,
            "remote_backed": True,
            "tasks": {"generate": {"context_length": 4096, "max_output_tokens": 64}},
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
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
    remote_chat: tuple[TestClient, SieUpstreamAdapter, ModelConfig, list[httpx.Request]],
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


@pytest.mark.parametrize("disconnect", [False, True])
def test_failure_after_output_is_in_band_and_never_retried(remote_chat: tuple, disconnect: bool) -> None:
    client, _, _, requests = remote_chat
    stream = ChatStream(events()[:1], disconnect=disconnect)
    answer_with(remote_chat, httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    response = client.post("/v1/chat/completions", json={**BODY, "stream": True})
    assert response.status_code == 200
    assert '"content":"answer"' in response.text
    assert '"error"' in response.text
    assert "private" not in response.text
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
