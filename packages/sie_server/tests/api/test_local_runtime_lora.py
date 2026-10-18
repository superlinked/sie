"""Direct public API LoRA authorization, cold retries, forwarding, and stream leases."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from sie_server.adapters._generation_base import GenerationChunk
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.api import generate, openai_local
from sie_server.api.lora import LeasedStreamingResponse
from sie_server.config.model import ModelConfig
from sie_server.core.registry import ModelRegistry
from starlette.requests import ClientDisconnect


@pytest.fixture
def local_lora(monkeypatch):
    paths = {"support": "org/a", "legal": "org/b"}
    loadtime = {
        "lora_paths": paths,
        "preload_loras": [],
        "mem_fraction_static": 0.5,
        "max_lora_rank": 4,
        "lora_target_modules": ["q_proj"],
        "max_loras_per_batch": 2,
    }
    config = ModelConfig.model_validate(
        {
            "sie_id": "test/model",
            "hf_id": "test/model",
            "inputs": {"text": True},
            "tasks": {"generate": {"context_length": 1024, "max_output_tokens": 64}},
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.sglang.generation:SGLangGenerationAdapter",
                    "max_batch_tokens": 1024,
                    "kv_budget_tokens": 1024,
                    "adapter_options": {"loadtime": loadtime},
                }
            },
        }
    )
    adapter = SGLangGenerationAdapter("test/model", **loadtime)
    adapter._server_url = "http://127.0.0.1:30400"
    adapter._lora.started()
    calls = []

    async def native(prompt, **kwargs):
        calls.append(kwargs)
        yield GenerationChunk(text_delta=kwargs.get("lora_path", "base"))
        yield GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(adapter, "generate", native)

    class ChatStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            assert adapter._lora._leases["org/a"] == 1
            yield b'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{"content":"support"},"finish_reason":null}]}\n\n'
            assert adapter._lora._leases["org/a"] == 1
            yield b'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"

    def chat(request):
        body = json.loads(request.content)
        calls.append(body)
        assert "lora_adapter" not in body
        if body.get("lora_path"):
            assert body["lora_path"] == "support"
            assert adapter._lora._leases["org/a"] == 1
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ChatStream())
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "test/model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": body.get("lora_path", "base")},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    monkeypatch.setattr(
        openai_local, "_new_proxy_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(chat))
    )
    registry = MagicMock(spec=ModelRegistry)
    registry.device = "cuda:0"
    registry.engine_config = None
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get_config.return_value = config
    registry.get.return_value = adapter
    registry.ensure_lora_loaded_async = AsyncMock(return_value=(False, True))
    registry.lora_retry_after_s.return_value = 7
    app = FastAPI()
    app.include_router(generate.router)
    app.include_router(openai_local.router)
    app.state.registry = registry
    return TestClient(app), registry, adapter, calls


def request_for(kind, *, stream=False):
    if kind == "native":
        return "/v1/generate/test__model", {"prompt": "hello", "max_new_tokens": 8, "stream": stream}
    return "/v1/chat/completions", {
        "model": "test/model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "stream": stream,
    }


@pytest.mark.parametrize("kind", ["native", "chat"])
@pytest.mark.parametrize("stream", [False, True])
def test_cold_request_retries_then_forwards_selected_alias(local_lora, kind, stream):
    client, registry, adapter, calls = local_lora
    url, body = request_for(kind, stream=stream)
    body["lora_adapter"] = "support"
    cold = client.post(url, json=body)
    assert cold.status_code == 503
    assert cold.headers["retry-after"] == "7"
    assert "LORA_LOADING" in cold.text
    assert not calls

    registry.ensure_lora_loaded_async.return_value = (True, False)
    adapter._lora._loaded.add("org/a")
    ready = client.post(url, json=body)
    assert ready.status_code == 200, ready.text
    assert "support" in ready.text
    assert calls[-1]["lora_path"] == "support"
    assert not adapter._lora._leases.get("org/a", 0)


@pytest.mark.parametrize("kind", ["native", "chat"])
@pytest.mark.parametrize("selector", ["", True, {}, "org/undeclared", "unknown", "--help"])
def test_undeclared_or_malformed_selection_never_starts_loading(local_lora, kind, selector):
    client, registry, _, calls = local_lora
    url, body = request_for(kind)
    body["lora_adapter"] = selector
    response = client.post(url, json=body)
    assert response.status_code == 400
    assert "lora_adapter" in response.text
    registry.ensure_lora_loaded_async.assert_not_called()
    registry.get.assert_not_called()
    assert not calls


@pytest.mark.parametrize("kind", ["native", "chat"])
def test_explicit_null_keeps_base_without_lora_loading(local_lora, kind):
    client, registry, _, calls = local_lora
    url, body = request_for(kind)
    body["lora_adapter"] = None
    response = client.post(url, json=body)
    assert response.status_code == 200, response.text
    assert "base" in response.text
    assert "lora_path" not in calls[-1]
    registry.ensure_lora_loaded_async.assert_not_called()


def test_chat_deferred_parallel_sampling_rejected_before_adapter_fetch(local_lora):
    client, registry, _, calls = local_lora
    url, body = request_for("chat")
    response = client.post(url, json={**body, "lora_adapter": "support", "n": 2})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "n"
    registry.ensure_lora_loaded_async.assert_not_called()
    assert not calls


def test_chat_draining_between_admission_and_lease_returns_retry(local_lora):
    client, registry, adapter, calls = local_lora
    registry.ensure_lora_loaded_async.return_value = (True, False)
    adapter._lora._loaded.add("org/a")
    adapter._lora._draining.add("org/a")
    url, body = request_for("chat")
    response = client.post(url, json={**body, "lora_adapter": "support"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "LORA_LOADING"
    assert response.headers["retry-after"] == "1"
    assert not calls
    assert not adapter._lora._leases


@pytest.mark.parametrize("stream", [False, True])
def test_chat_connection_failure_releases_lease(local_lora, monkeypatch, stream):
    client, registry, adapter, _ = local_lora
    registry.ensure_lora_loaded_async.return_value = (True, False)
    adapter._lora._loaded.add("org/a")

    def fail(request):
        assert adapter._lora._leases["org/a"] == 1
        raise httpx.ConnectError("child unavailable", request=request)

    monkeypatch.setattr(
        openai_local, "_new_proxy_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(fail))
    )
    url, body = request_for("chat", stream=stream)
    response = client.post(url, json={**body, "lora_adapter": "support"})
    assert response.status_code == 502
    assert not adapter._lora._leases.get("org/a", 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_stream_lifetime_closes_on_asgi_error_or_cancellation(cancel):
    released = []
    started = asyncio.Event()

    async def content():
        yield b"chunk"

    async def send(message):
        started.set()
        if cancel:
            await asyncio.Event().wait()
        raise RuntimeError("client send failed")

    async def receive():
        await asyncio.Event().wait()

    lifetime = AsyncExitStack()
    lifetime.callback(released.append, True)
    response = LeasedStreamingResponse(StreamingResponse(content()), lifetime)
    task = asyncio.create_task(response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send))
    await started.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else RuntimeError):
        await task
    assert released == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_chat_closes_acquired_child_before_releasing_lease_on_early_disconnect(local_lora, monkeypatch, cancel):
    client, registry, adapter, _ = local_lora
    registry.ensure_lora_loaded_async.return_value = (True, False)
    adapter._lora._loaded.add("org/a")
    closed = []

    class UnstartedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            pytest.fail("response headers fail before child body iteration")
            yield b"unreachable"

        async def aclose(self):
            assert adapter._lora._leases["org/a"] == 1
            closed.append("upstream")

    child = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=UnstartedStream())
        )
    )
    close_client = child.aclose

    async def tracked_close():
        assert adapter._lora._leases["org/a"] == 1
        closed.append("client")
        await close_client()

    monkeypatch.setattr(child, "aclose", tracked_close)
    monkeypatch.setattr(openai_local, "_new_proxy_client", lambda: child)
    _, body = request_for("chat", stream=True)
    body["lora_adapter"] = "support"

    async def receive_body():
        return {"type": "http.request", "body": json.dumps(body).encode(), "more_body": False}

    scope = {"type": "http", "asgi": {"spec_version": "2.4"}, "app": client.app, "headers": []}
    response = await openai_local.chat_completions(Request(scope, receive_body), None)
    assert adapter._lora._leases["org/a"] == 1
    started = asyncio.Event()

    async def fail_send(message):
        started.set()
        if cancel:
            await asyncio.Event().wait()
        raise OSError("client disconnected before headers")

    task = asyncio.create_task(response(scope, receive_body, fail_send))
    await started.wait()
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else ClientDisconnect):
        await task
    assert closed == ["upstream", "client"]
    assert child.is_closed
    assert not adapter._lora._leases.get("org/a", 0)


@pytest.mark.asyncio
async def test_stream_send_failure_closes_suspended_iterator_before_resources():
    events = []
    read_task = None

    async def content():
        nonlocal read_task
        read_task = asyncio.create_task(asyncio.Event().wait())
        try:
            yield b": keepalive\n\n"
            await read_task
        finally:
            read_task.cancel()
            await asyncio.gather(read_task, return_exceptions=True)
            events.append("iterator")

    async def send(message):
        if message["type"] == "http.response.body":
            raise OSError("disconnected after first keepalive")

    async def receive():
        await asyncio.Event().wait()

    lifetime = AsyncExitStack()
    lifetime.callback(events.append, "resources")
    response = LeasedStreamingResponse(StreamingResponse(content()), lifetime)
    with pytest.raises(ClientDisconnect):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert events == ["iterator", "resources"]
    assert read_task is not None
    assert read_task.cancelled()
