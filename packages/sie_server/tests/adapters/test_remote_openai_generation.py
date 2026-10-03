import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sie_sdk import SIEClient
from sie_server.adapters._generation_base import GenerationCapacityError, GenerationUnsupportedFieldError
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.adapters.remote._limits import upstream_limiter
from sie_server.adapters.remote._openai_completions import CompletionStreamParser
from sie_server.adapters.remote.openai import OpenAIUpstreamAdapter
from sie_server.app.app_factory import AppFactory
from sie_server.app.app_state_config import AppStateConfig
from sie_server.config.upstreams import RemoteServingDisabledError, Upstream, install_upstreams
from sie_server.core.upstream_client import upstream_client

USAGE = {"prompt_tokens": 37, "completion_tokens": 2, "total_tokens": 39}


class CompletionStream(httpx.AsyncByteStream):
    def __init__(self, events: list[dict | str], *, disconnect: bool = False) -> None:
        self.events = events
        self.disconnect = disconnect
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self.events:
            data = event if isinstance(event, str) else json.dumps(event)
            wire = f"data: {data}\n\n".encode()
            yield wire[:7]
            yield wire[7:]
        if self.disconnect:
            raise httpx.ReadError("PRIVATE_UPSTREAM_DETAIL")

    async def aclose(self) -> None:
        self.closed = True


def choice(text: str = "answer", *, finish: str | None = None, **extra: Any) -> dict:
    return {"choices": [{"index": 0, "text": text, "finish_reason": finish, **extra}]}


def events() -> list[dict | str]:
    return [choice(), choice("", finish="stop"), {"choices": [], "usage": USAGE}, "[DONE]"]


@pytest.fixture
async def adapter() -> AsyncIterator[OpenAIUpstreamAdapter]:
    upstream = Upstream.model_validate(
        {
            "kind": "openai",
            "base_url": "http://127.0.0.1:8088/prefix/v1",
            "endpoints": ["chat", "completions"],
            "set_params": {"provider": {"data_collection": "deny"}},
            "strip_params": ["user"],
            "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
        }
    )
    install_upstreams({"generation-test": upstream})
    loaded = OpenAIUpstreamAdapter(upstream="generation-test", upstream_model="operator/model")
    loaded.load("cpu")
    try:
        yield loaded
    finally:
        await loaded.aclose_client()
        loaded.unload()
        install_upstreams({})


def answer_with(adapter: OpenAIUpstreamAdapter, handler: Callable[[httpx.Request], httpx.Response]) -> None:
    assert adapter._upstream is not None
    adapter._async_client = upstream_client(adapter._upstream, transport=httpx.MockTransport(handler))


async def test_raw_prompt_pins_endpoint_model_usage_and_operator_fields(adapter: OpenAIUpstreamAdapter) -> None:
    requests: list[httpx.Request] = []
    stream = CompletionStream(events())

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    answer_with(adapter, respond)
    chunks = [chunk async for chunk in adapter.generate("rendered prompt", max_new_tokens=9, seed=4)]
    assert "".join(chunk.text_delta for chunk in chunks) == "answer"
    assert chunks[0].is_first
    assert chunks[-1].done
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].prompt_tokens == 37
    assert chunks[-1].completion_tokens == 2
    assert requests[0].url.path == "/prefix/v1/completions"
    assert json.loads(requests[0].content) == {
        "prompt": "rendered prompt",
        "max_tokens": 9,
        "temperature": 1.0,
        "top_p": 1.0,
        "n": 1,
        "echo": False,
        "seed": 4,
        "model": "operator/model",
        "stream": True,
        "stream_options": {"include_usage": True},
        "provider": {"data_collection": "deny"},
    }
    assert stream.closed
    assert upstream_limiter("generation-test")._in_flight == 0


@pytest.mark.parametrize("streaming", [False, True])
async def test_missing_chat_endpoint_refuses_without_dispatch(adapter: OpenAIUpstreamAdapter, streaming: bool) -> None:
    assert adapter._upstream is not None
    adapter._upstream = adapter._upstream.model_copy(update={"endpoints": frozenset()})
    calls = []
    answer_with(adapter, lambda request: calls.append(request) or httpx.Response(200))
    with pytest.raises(GenerationUnsupportedFieldError):
        if streaming:
            await anext(adapter.chat_completion_stream({"messages": []}, requested_model="caller/model"))
        else:
            await adapter.chat_completion({"messages": []}, requested_model="caller/model")
    assert calls == []


@pytest.mark.parametrize(
    ("field", "value"), [("top_k", 5), ("min_new_tokens", 1), ("repetition_penalty", 1.1), ("images", [{}])]
)
async def test_raw_unsupported_fields_fail_before_dispatch(
    adapter: OpenAIUpstreamAdapter, field: str, value: Any
) -> None:
    calls = []
    answer_with(adapter, lambda request: calls.append(request) or httpx.Response(200))
    with pytest.raises(GenerationUnsupportedFieldError):
        await anext(adapter.generate("prompt", max_new_tokens=9, **{field: value}))
    assert calls == []


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_pre_output_failure_is_retryable_once(adapter: OpenAIUpstreamAdapter, status: int) -> None:
    calls = []
    answer_with(
        adapter,
        lambda request: (
            calls.append(request)
            or httpx.Response(
                status, headers={"retry-after": "19"}, stream=httpx.ByteStream(b"PRIVATE_UPSTREAM_DETAIL")
            )
        ),
    )
    with pytest.raises(GenerationCapacityError) as raised:
        await anext(adapter.generate("prompt", max_new_tokens=9))
    assert raised.value.retry_after_s == 19
    assert "PRIVATE" not in str(raised.value)
    assert len(calls) == 1


async def test_disconnect_after_text_is_final_and_closes_capacity(adapter: OpenAIUpstreamAdapter) -> None:
    stream = CompletionStream([choice()], disconnect=True)
    answer_with(adapter, lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    iterator = adapter.generate("prompt", max_new_tokens=9)
    assert (await anext(iterator)).text_delta == "answer"
    with pytest.raises(RemoteUpstreamError, match="failed during generation"):
        await anext(iterator)
    assert stream.closed
    assert upstream_limiter("generation-test")._in_flight == 0


async def test_cancellation_closes_response_and_releases_capacity(adapter: OpenAIUpstreamAdapter) -> None:
    stream = CompletionStream(events())
    answer_with(adapter, lambda _: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream))
    iterator = adapter.generate("prompt", max_new_tokens=9)
    await anext(iterator)
    await iterator.aclose()
    assert stream.closed
    assert upstream_limiter("generation-test")._in_flight == 0


async def test_existing_client_obeys_global_kill_switch(adapter: OpenAIUpstreamAdapter) -> None:
    answer_with(adapter, lambda _: httpx.Response(200))
    assert adapter._upstream is not None
    install_upstreams({"generation-test": adapter._upstream}, remote_serving=False)
    with pytest.raises(RemoteServingDisabledError):
        await anext(adapter.generate("prompt", max_new_tokens=9))


async def test_sync_unload_keeps_client_for_awaitable_teardown(adapter: OpenAIUpstreamAdapter) -> None:
    answer_with(adapter, lambda _: httpx.Response(200))
    client = adapter._async_client
    await asyncio.to_thread(adapter.unload)
    assert adapter._async_client is client
    await adapter.aclose_client()
    assert client is not None
    assert client.is_closed


@pytest.mark.parametrize(
    "bad_events",
    [
        [choice(), "[DONE]"],
        [choice("", finish="stop"), "[DONE]"],
        [{"choices": [], "usage": USAGE}],
        [choice("", finish="tool_calls")],
        [{"choices": [{"index": True, "text": "answer", "finish_reason": None}]}],
        [choice("", finish="stop"), choice("late")],
        [choice("", finish="stop"), {"choices": [], "usage": {**USAGE, "total_tokens": 40}}],
        [choice(logprobs={"tokens": ["answer"], "token_logprobs": [float("nan")]})],
        [choice(logprobs={"tokens": ["answer"], "token_logprobs": []})],
    ],
)
def test_malformed_completion_events_fail_closed(bad_events: list[dict | str]) -> None:
    parser = CompletionStreamParser()
    with pytest.raises(RemoteUpstreamError):
        for event in bad_events:
            parser.parse((event if isinstance(event, str) else json.dumps(event)).encode())
        parser.finish()


def test_completion_logprobs_are_normalized_without_provider_metadata() -> None:
    parser = CompletionStreamParser(logprobs=True)
    chunk = parser.parse(
        json.dumps(
            choice(
                logprobs={
                    "tokens": ["answer"],
                    "token_logprobs": [-0.2],
                    "top_logprobs": [{"other": -1.0}],
                    "private": "hidden",
                }
            )
        ).encode()
    )
    assert chunk is not None
    assert chunk.logprobs == (
        {"token": "answer", "logprob": -0.2, "bytes": None, "top_logprobs": [{"token": "other", "logprob": -1.0}]},
    )


@pytest.mark.parametrize("operator_cap", [None, 3, 99])
def test_operator_fields_cannot_drop_or_raise_output_ceiling(
    adapter: OpenAIUpstreamAdapter, operator_cap: int | None
) -> None:
    assert adapter._upstream is not None
    changes = (
        {"strip_params": frozenset({"max_tokens"})}
        if operator_cap is None
        else {"set_params": {"max_tokens": operator_cap}}
    )
    adapter._upstream = adapter._upstream.model_copy(update=changes)
    _, request = adapter._generation_request({"messages": [], "max_tokens": 9}, chat=True, stream=False)
    assert json.loads(request.content)["max_tokens"] == (3 if operator_cap == 3 else 9)


@pytest.mark.usefixtures("_offline_apps")
def test_native_generation_crosses_real_wire_through_sdk(
    tmp_path: Path, serve_on_loopback: Callable[[FastAPI], AbstractContextManager[str]]
) -> None:
    upstream_app = FastAPI()
    calls = []

    @upstream_app.post("/v1/completions")
    async def completions(request: Request) -> StreamingResponse:
        calls.append(await request.json())
        return StreamingResponse(CompletionStream(events()).__aiter__(), media_type="text/event-stream")

    model = {
        "sie_id": "acme/remote-generation",
        "remote_backed": True,
        "tasks": {"generate": {"context_length": 4096, "max_output_tokens": 64}},
        "profiles": {
            "default": {
                "adapter_path": "sie_server.adapters.remote.openai:OpenAIUpstreamAdapter",
                "max_batch_tokens": 8192,
                "adapter_options": {"loadtime": {"upstream": "open-host", "upstream_model": "operator/model"}},
            }
        },
    }
    models = tmp_path / "models"
    models.mkdir()
    (models / "generation.yaml").write_text(yaml.safe_dump(model))
    with serve_on_loopback(upstream_app) as upstream_url:
        upstreams = tmp_path / "upstreams.yaml"
        upstreams.write_text(
            yaml.safe_dump(
                {
                    "upstreams": {
                        "open-host": {
                            "kind": "openai",
                            "base_url": upstream_url + "/v1",
                            "endpoints": ["completions"],
                            "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
                        }
                    }
                }
            )
        )
        app = AppFactory.create_app(AppStateConfig(models_dir=str(models), device="cpu", upstreams_file=str(upstreams)))
        with serve_on_loopback(app) as local_url, SIEClient(local_url) as client:
            buffered = client.generate(model["sie_id"], "rendered prompt", max_new_tokens=3)
            streamed = list(client.stream_generate(model["sie_id"], "rendered prompt", max_new_tokens=3))
    assert buffered["text"] == "answer"
    assert buffered["usage"] == USAGE
    assert "".join(chunk.get("text_delta", "") for chunk in streamed) == "answer"
    assert streamed[-1]["usage"] == USAGE
    assert len(calls) == 2
    assert all(call["model"] == "operator/model" and call["prompt"] == "rendered prompt" for call in calls)


@pytest.mark.parametrize("caller_field", ["max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("operator_cap", [3, 99])
def test_operator_alternate_limit_alias_obeys_caller_ceiling(
    adapter: OpenAIUpstreamAdapter, caller_field: str, operator_cap: int
) -> None:
    assert adapter._upstream is not None
    alternate = "max_completion_tokens" if caller_field == "max_tokens" else "max_tokens"
    adapter._upstream = adapter._upstream.model_copy(update={"set_params": {alternate: operator_cap}})
    _, request = adapter._generation_request({"messages": [], caller_field: 9}, chat=True, stream=False)
    sent = json.loads(request.content)
    assert sent[caller_field] == min(9, operator_cap)
    assert sent[alternate] == min(9, operator_cap)


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_private_completion_metadata_cannot_survive_in_logprobs(field: str) -> None:
    parser = CompletionStreamParser(logprobs=True)
    event = choice(logprobs={"tokens": ["PRIVATE"], "token_logprobs": [-0.2]}, **{field: "private"})
    chunk = parser.parse(json.dumps(event).encode())
    assert chunk is not None
    assert chunk.text_delta == "answer"
    assert chunk.logprobs is None


def test_requested_logprobs_cannot_be_silently_omitted() -> None:
    parser = CompletionStreamParser(logprobs=True)
    with pytest.raises(RemoteUpstreamError):
        parser.parse(json.dumps(choice()).encode())
