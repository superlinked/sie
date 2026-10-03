import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
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


@pytest.mark.parametrize("status", [400, 503])
def test_native_remote_stream_refusal_precedes_http_success(remote_chat: tuple, status: int) -> None:
    client, _, config, requests = remote_chat
    config.profiles["default"].adapter_options.runtime.clear()
    answer_with(
        remote_chat,
        json_response(
            status,
            {"error": {"code": "INVALID_INPUT", "message": "private"}}
            if status == 400
            else {"error": {"message": "private"}},
            headers={"retry-after": "19"},
        ),
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
