import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from sie_sdk import SIEClient
from sie_server.adapters._generation_base import GenerationCapacityError, GenerationChunk, GenerationDrainingError
from sie_server.adapters.remote._http import RemoteUpstreamError
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.api.generate import _generation_http_exception
from sie_server.config.upstreams import RemoteServingDisabledError, Upstream, install_upstreams
from sie_server.core.upstream_client import upstream_client


class GenerationStream(httpx.AsyncByteStream):
    def __init__(self, events: list[dict], *, disconnect: bool = False) -> None:
        self.events = events
        self.disconnect = disconnect
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for event in self.events:
            yield b"data: " + json.dumps(event).encode() + b"\n\n"
        if self.disconnect:
            raise httpx.ReadError("untrusted transport detail")

    async def aclose(self) -> None:
        self.closed = True


def terminal(**changes: Any) -> dict:
    return {"done": True, "finish_reason": "stop", "usage": {"prompt_tokens": 37, "completion_tokens": 2}, **changes}


@pytest.fixture
async def adapter() -> AsyncIterator[SieUpstreamAdapter]:
    upstream = Upstream.model_validate(
        {
            "kind": "sie",
            "base_url": "http://127.0.0.1:8088",
            "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
        }
    )
    install_upstreams({"generation-test": upstream})
    loaded = SieUpstreamAdapter(upstream="generation-test", upstream_model="acme/model:remote")
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


async def test_generation_forwards_the_native_request_and_exact_usage(adapter: SieUpstreamAdapter) -> None:
    requests: list[httpx.Request] = []
    stream = GenerationStream([{"done": False, "text_delta": "answer"}, terminal()])

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    answer_with(adapter, respond)
    chunks = [chunk async for chunk in adapter.generate("raw prompt", max_new_tokens=9, top_k=5, seed=17)]
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/generate/acme__model:remote"
    body = json.loads(requests[0].content)
    assert body == {
        "prompt": "raw prompt",
        "max_new_tokens": 9,
        "temperature": 1.0,
        "top_p": 1.0,
        "stream": True,
        "seed": 17,
        "options": {"default_sampling": {"top_k": 5}},
    }
    assert chunks[0].text_delta == "answer"
    assert chunks[0].is_first
    assert (chunks[-1].prompt_tokens, chunks[-1].completion_tokens) == (37, 2)
    assert stream.closed


@pytest.mark.parametrize(
    ("status", "code", "error_class"),
    [
        (429, None, GenerationCapacityError),
        (503, "MODEL_LOADING", GenerationDrainingError),
        (500, None, GenerationCapacityError),
    ],
)
async def test_upstream_failure_before_output_carries_its_retry_after(
    adapter: SieUpstreamAdapter, status: int, code: str | None, error_class: type[GenerationCapacityError]
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            headers={"retry-after": "19"},
            stream=httpx.ByteStream(json.dumps({"error": {"code": code, "message": "private"}}).encode()),
        )

    answer_with(adapter, respond)
    with pytest.raises(error_class) as raised:
        _ = [chunk async for chunk in adapter.generate("prompt", max_new_tokens=9)]
    exception = _generation_http_exception(raised.value)
    assert exception.status_code == 503
    assert exception.headers is not None
    assert exception.headers["Retry-After"] == "19"
    assert "private" not in str(exception.detail)
    assert len(requests) == 1


async def test_a_disconnect_after_output_is_final_and_closes_the_stream(adapter: SieUpstreamAdapter) -> None:
    stream = GenerationStream([{"done": False, "text_delta": "started"}], disconnect=True)
    answer_with(
        adapter, lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    iterator = adapter.generate("prompt", max_new_tokens=9)
    assert (await anext(iterator)).text_delta == "started"
    with pytest.raises(RemoteUpstreamError, match="failed during generation"):
        await anext(iterator)
    assert stream.closed


async def test_cancelling_generation_closes_the_upstream_response(adapter: SieUpstreamAdapter) -> None:
    stream = GenerationStream([{"done": False, "text_delta": "started"}, terminal()])
    answer_with(
        adapter, lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    iterator = adapter.generate("prompt", max_new_tokens=9)
    assert isinstance(await anext(iterator), GenerationChunk)
    await iterator.aclose()
    assert stream.closed


@pytest.mark.parametrize(
    "usage", [None, {}, {"prompt_tokens": True, "completion_tokens": 2}, {"prompt_tokens": 3, "completion_tokens": -1}]
)
async def test_generation_refuses_missing_or_malformed_terminal_usage(adapter: SieUpstreamAdapter, usage: Any) -> None:
    stream = GenerationStream([terminal(usage=usage)])
    answer_with(
        adapter, lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
    )
    with pytest.raises(RemoteUpstreamError, match="usage"):
        _ = [chunk async for chunk in adapter.generate("prompt", max_new_tokens=9)]
    assert stream.closed


def test_native_generation_crosses_a_real_upstream_through_the_sdk(
    tmp_path: Path,
    sie_upstream: Callable[..., Any],
    remote_app: Callable[..., Any],
    serve_on_loopback: Callable[..., Any],
) -> None:
    upstream_config = {
        "sie_id": "acme/upstream-generation",
        "package_backed": True,
        "tasks": {"generate": {"context_length": 4096, "max_output_tokens": 64}},
        "profiles": {
            "default": {
                "adapter_path": "sie_server.adapters.fake.adapter:FakeAdapter",
                "max_batch_tokens": 8192,
                "kv_budget_tokens": 4096,
            }
        },
    }
    models = tmp_path / "upstream-generation-models"
    models.mkdir()
    (models / "generation.yaml").write_text(yaml.safe_dump(upstream_config))
    remote_config = {
        "sie_id": "acme/remote-generation",
        "remote_backed": True,
        "tasks": upstream_config["tasks"],
        "profiles": {
            "default": {
                "adapter_path": "sie_server.adapters.remote.sie:SieUpstreamAdapter",
                "max_batch_tokens": 8192,
                "adapter_options": {"loadtime": {"upstream": "fake-sie", "upstream_model": upstream_config["sie_id"]}},
            }
        },
    }
    with sie_upstream(models_dir=models) as upstream:
        with SIEClient(upstream.url) as direct:
            expected = direct.generate(upstream_config["sie_id"], "the same prompt", max_new_tokens=3)
        app = remote_app(upstream.url, extra_models={"generation.yaml": yaml.safe_dump(remote_config)})
        with serve_on_loopback(app) as local_url, SIEClient(local_url) as client:
            served = client.generate(remote_config["sie_id"], "the same prompt", max_new_tokens=3)
            chunks = list(client.stream_generate(remote_config["sie_id"], "the same prompt", max_new_tokens=3))
    assert served["text"] == expected["text"]
    assert served["usage"] == expected["usage"]
    assert "".join(chunk.get("text_delta", "") for chunk in chunks) == expected["text"]
    assert chunks[-1]["usage"] == expected["usage"]


async def test_a_loaded_generation_client_stops_sending_when_remote_serving_is_disabled(
    adapter: SieUpstreamAdapter,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=GenerationStream([terminal()]))

    answer_with(adapter, respond)
    assert len([chunk async for chunk in adapter.generate("prompt", max_new_tokens=9)]) == 1
    assert adapter._upstream is not None
    install_upstreams({"generation-test": adapter._upstream}, remote_serving=False)
    with pytest.raises(RemoteServingDisabledError):
        _ = [chunk async for chunk in adapter.generate("prompt", max_new_tokens=9)]
    assert len(requests) == 1
