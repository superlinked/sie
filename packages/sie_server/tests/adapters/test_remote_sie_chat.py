import asyncio
import json
from collections.abc import AsyncIterator, Callable
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from sie_sdk import SIEClient
from sie_server.adapters.errors import UpstreamUnavailableError
from sie_server.adapters.remote import sie as remote_sie
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.adapters.remote._limits import upstream_limiter
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.api.openai_local import router
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import RemoteServingDisabledError, Upstream, install_upstreams
from sie_server.core.upstream_client import upstream_client

USAGE = {"prompt_tokens": 37, "completion_tokens": 2, "total_tokens": 39}
BODY = {"model": "caller/model", "messages": [{"role": "user", "content": "question"}], "max_tokens": 9}


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


@pytest.fixture
async def adapter() -> AsyncIterator[SieUpstreamAdapter]:
    upstream = Upstream.model_validate(
        {
            "kind": "sie",
            "base_url": "http://127.0.0.1:8088/prefix",
            "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
        }
    )
    install_upstreams({"chat-test": upstream})
    loaded = SieUpstreamAdapter(upstream="chat-test", upstream_model="operator/model")
    loaded.load("cpu")
    try:
        yield loaded
    finally:
        await loaded.aclose_client()
        loaded.unload()
        install_upstreams({})


def answer_with(adapter: SieUpstreamAdapter, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    assert adapter._upstream is not None
    adapter._async_client = upstream_client(adapter._upstream, transport=httpx.MockTransport(handler))


def completion() -> dict:
    return {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
        "usage": USAGE,
        "model": "private-model",
        "id": "private-id",
    }


def events() -> list[dict | str]:
    return [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "answer"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": USAGE},
        "[DONE]",
    ]


async def test_buffered_chat_pins_route_model_and_usage(adapter: SieUpstreamAdapter) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps(completion()).encode()),
        )

    answer_with(adapter, respond)
    result = await adapter.chat_completion({**BODY, "stream": True}, requested_model="caller/model")
    assert len(requests) == 1
    assert requests[0].url.path == "/prefix/v1/chat/completions"
    assert json.loads(requests[0].content) == {**BODY, "model": "operator/model", "stream": False}
    assert result["model"] == "caller/model"
    assert result["usage"] == USAGE
    assert "private" not in json.dumps(result)


async def test_streaming_chat_requests_exact_usage_and_closes_response(adapter: SieUpstreamAdapter) -> None:
    stream = ChatStream(events())
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    answer_with(adapter, respond)
    result = [
        item
        async for item in adapter.chat_completion_stream(
            {**BODY, "stream_options": {"include_usage": False}}, requested_model="caller/model"
        )
    ]
    assert len(result) == 3
    assert result[-1]["usage"] == USAGE
    assert json.loads(requests[0].content)["stream_options"] == {"include_usage": True}
    assert stream.closed
    assert upstream_limiter("chat-test")._in_flight == 0


@pytest.mark.parametrize("streaming", [False, True])
async def test_pre_output_failure_is_retryable_without_another_call(
    adapter: SieUpstreamAdapter, streaming: bool
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            503, headers={"retry-after": "19"}, stream=httpx.ByteStream(b'{"error":{"message":"private"}}')
        )

    answer_with(adapter, respond)
    with pytest.raises(UpstreamUnavailableError) as raised:
        if streaming:
            await anext(adapter.chat_completion_stream(BODY, requested_model="caller/model"))
        else:
            await adapter.chat_completion(BODY, requested_model="caller/model")
    assert raised.value.retry_after_s == 19
    assert "private" not in str(raised.value)
    assert len(requests) == 1
    assert upstream_limiter("chat-test")._in_flight == 0


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("trickle", [False, True])
async def test_chat_error_body_deadline_closes_and_releases_capacity(
    adapter: SieUpstreamAdapter, monkeypatch: pytest.MonkeyPatch, streaming: bool, trickle: bool
) -> None:
    monkeypatch.setattr(remote_sie, "REQUEST_DEADLINE_S", 0.03)

    class SlowError(ChatStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            while True:
                if trickle:
                    await asyncio.sleep(0.005)
                    yield b"private upstream detail"
                else:
                    await asyncio.Event().wait()

    stream = SlowError([])
    answer_with(adapter, lambda _: httpx.Response(503, headers={"Retry-After": "19"}, stream=stream))
    async with asyncio.timeout(1):
        with pytest.raises(UpstreamUnavailableError) as raised:
            if streaming:
                await anext(adapter.chat_completion_stream(BODY, requested_model=BODY["model"]))
            else:
                await adapter.chat_completion(BODY, requested_model=BODY["model"])
    assert raised.value.retry_after_s == 19
    assert "private" not in str(raised.value)
    assert stream.closed
    assert upstream_limiter("chat-test")._in_flight == 0


async def test_disconnect_after_output_is_final(adapter: SieUpstreamAdapter) -> None:
    stream = ChatStream(events()[:1], disconnect=True)
    answer_with(adapter, lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    iterator = adapter.chat_completion_stream(BODY, requested_model="caller/model")
    await anext(iterator)
    with pytest.raises(RemoteUpstreamError, match="failed during chat"):
        await anext(iterator)
    assert stream.closed
    assert upstream_limiter("chat-test")._in_flight == 0


@pytest.mark.parametrize("keep", [1, 2, 3])
async def test_truncated_chat_stream_fails_closed(adapter: SieUpstreamAdapter, keep: int) -> None:
    stream = ChatStream(events()[:keep])
    answer_with(adapter, lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    with pytest.raises(RemoteUpstreamError):
        _ = [item async for item in adapter.chat_completion_stream(BODY, requested_model="caller/model")]
    assert stream.closed


async def test_cached_chat_client_cannot_bypass_serving_switch(adapter: SieUpstreamAdapter) -> None:
    answer_with(adapter, lambda _: pytest.fail("disabled serving must not dispatch"))
    install_upstreams({"chat-test": adapter._upstream}, remote_serving=False)
    with pytest.raises(RemoteServingDisabledError):
        await adapter.chat_completion(BODY, requested_model="caller/model")


async def test_cancellation_during_a_pending_read_releases_the_slot(adapter: SieUpstreamAdapter) -> None:
    entered = asyncio.Event()

    class PendingStream(ChatStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            entered.set()
            await asyncio.Event().wait()
            yield b"unreachable"

    stream = PendingStream([])
    answer_with(adapter, lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    iterator = adapter.chat_completion_stream(BODY, requested_model="caller/model")
    pending = asyncio.create_task(anext(iterator))
    await entered.wait()
    assert upstream_limiter("chat-test")._in_flight == 1
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert stream.closed
    assert upstream_limiter("chat-test")._in_flight == 0


def test_buffered_and_streamed_chat_work_through_the_sdk(
    adapter: SieUpstreamAdapter, serve_on_loopback: Callable
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if json.loads(request.content).get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ChatStream(events()))
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(json.dumps(completion()).encode()),
        )

    answer_with(adapter, respond)
    config = ModelConfig.model_validate(
        {
            "sie_id": BODY["model"],
            "remote_backed": True,
            "tasks": {"generate": {"context_length": 4096, "max_output_tokens": 64}},
            "profiles": {
                "default": {
                    "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
                    "max_batch_tokens": 8192,
                    "adapter_options": {"loadtime": {"upstream": "chat-test", "upstream_model": "operator/model"}},
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
    with serve_on_loopback(app) as url, SIEClient(url) as client:
        answer = client.chat_completions(BODY["model"], BODY["messages"], max_tokens=9)
        chunks = list(
            client.stream_chat_completions(
                BODY["model"], BODY["messages"], max_tokens=9, stream_options={"include_usage": True}
            )
        )
    assert answer["model"] == BODY["model"]
    assert answer["choices"][0]["message"]["content"] == "answer"
    assert answer["usage"] == USAGE
    assert chunks[-1]["usage"] == USAGE
    assert len(requests) == 2
