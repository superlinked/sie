"""A remote profile declares the context window its upstream serves.

The model has a local profile at a 16384-token window with a 4096-token output
cap and a remote profile whose upstream serves 131072 tokens with a
32768-token output cap. The configs come from the real loader, the remote
adapter is loaded by the real registry, and only the upstream's HTTP transport
is replaced.
"""

from __future__ import annotations

import functools
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import msgpack
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sie_server.adapters.remote.openai import OpenAIUpstreamAdapter
from sie_server.api.openai_local import router
from sie_server.config.model import ModelConfig
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.loader import load_model_configs
from sie_server.core.registry import ModelRegistry
from sie_server.core.upstream_client import upstream_client
from sie_server.processors.streaming import StreamingProcessor

MODEL = "acme/agent-chat"
REMOTE = f"{MODEL}:remote"
UPSTREAM = "agent-api"
LOCAL_ADAPTER = "sie_server.adapters.fake.adapter:FakeAdapter"
REMOTE_ADAPTER = "sie_server.adapters.remote.openai:OpenAIUpstreamAdapter"
IN_WINDOW = {"prompt_tokens": 100_000, "completion_tokens": 20_000, "total_tokens": 120_000}
OVER_WINDOW = {"prompt_tokens": 131_073, "completion_tokens": 2, "total_tokens": 131_075}
OVER_REMAINDER = {"prompt_tokens": 120_000, "completion_tokens": 12_000, "total_tokens": 132_000}


def model_config(*, remote_loadtime: dict[str, Any] | None = None, remote_output: int = 32768) -> dict[str, Any]:
    return {
        "sie_id": MODEL,
        "hf_id": "acme/not-on-any-hub",
        "hf_revision": "0" * 40,
        "inputs": {"text": True},
        "max_sequence_length": 16384,
        "tasks": {"generate": {"context_length": 16384, "max_output_tokens": 4096, "capabilities": {"tools": True}}},
        "profiles": {
            "default": {
                "adapter_path": LOCAL_ADAPTER,
                "max_batch_tokens": 16384,
                "kv_budget_tokens": 16384,
                "adapter_options": {"loadtime": {"max_seq_length": 16384}},
            },
            "remote": {
                "adapter_path": REMOTE_ADAPTER,
                "max_batch_tokens": 16384,
                "max_output_tokens": remote_output,
                "adapter_options": {
                    "loadtime": remote_loadtime
                    if remote_loadtime is not None
                    else {"upstream": UPSTREAM, "upstream_model": "vendor/agent-chat", "max_seq_length": 131072}
                },
            },
        },
        "routing": {"policy": "fallback", "fallback_profile": "remote"},
    }


def write_models(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    (models / f"{MODEL.replace('/', '__')}.yaml").write_text(yaml.safe_dump(model_config()), encoding="utf-8")
    return models


@pytest.fixture(autouse=True)
def _upstreams(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    install_upstreams(
        {
            UPSTREAM: Upstream.model_validate(
                {
                    "kind": "openai",
                    "base_url": "http://127.0.0.1:9/v1",
                    "endpoints": ["chat"],
                    "rate_cap": {"requests_per_minute": 600, "max_concurrency": 8},
                }
            )
        }
    )
    yield
    install_upstreams({})


def answer(adapter: OpenAIUpstreamAdapter, usages: list[dict[str, int]]) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        usage = usages[len(requests) - 1]
        if json.loads(request.content)["stream"]:
            frames = [
                {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "answer"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": usage},
            ]
            body = b"".join(b"data: " + json.dumps(frame).encode() + b"\n\n" for frame in frames)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(body + b"data: [DONE]\n\n"),
            )
        payload = {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer"}, "finish_reason": "stop"}],
            "usage": usage,
        }
        return httpx.Response(
            200, headers={"content-type": "application/json"}, stream=httpx.ByteStream(json.dumps(payload).encode())
        )

    assert adapter._upstream is not None
    adapter._async_client = upstream_client(adapter._upstream, transport=httpx.MockTransport(respond))
    return requests


def test_the_remote_variant_carries_the_window_its_profile_declares(tmp_path: Path) -> None:
    configs = load_model_configs(write_models(tmp_path))

    bare, remote = configs[MODEL], configs[REMOTE]
    assert bare.tasks.generate is not None
    assert remote.tasks.generate is not None
    assert (bare.max_sequence_length, bare.tasks.generate.context_length) == (16384, 16384)
    assert bare.tasks.generate.max_output_tokens == 4096
    assert (remote.max_sequence_length, remote.tasks.generate.context_length) == (131072, 131072)
    assert remote.tasks.generate.max_output_tokens == 32768
    assert remote.routing is None


def test_the_output_cap_is_checked_against_the_remote_profiles_own_window() -> None:
    with pytest.raises(ValidationError, match=r"max_output_tokens=131073, exceeding its context_length=131072"):
        ModelConfig.model_validate(model_config(remote_output=131073))

    without_window = {"upstream": UPSTREAM, "upstream_model": "vendor/agent-chat"}
    with pytest.raises(ValidationError, match=r"max_output_tokens=32768, exceeding its context_length=16384"):
        ModelConfig.model_validate(model_config(remote_loadtime=without_window))


async def test_the_remote_adapter_receives_the_declared_window(tmp_path: Path) -> None:
    registry = ModelRegistry(models_dir=write_models(tmp_path), device="cpu", enable_hot_reload=False)
    original_init = OpenAIUpstreamAdapter.__init__
    try:
        with patch.object(OpenAIUpstreamAdapter, "__init__", autospec=True, side_effect=original_init) as init:
            adapter = await registry.load_async(REMOTE, device="cpu")

        assert isinstance(adapter, OpenAIUpstreamAdapter)
        assert init.call_args.kwargs["max_seq_length"] == 131072
        assert init.call_args.kwargs["upstream_model"] == "vendor/agent-chat"
    finally:
        await registry.unload_all_async()


@pytest.fixture
def direct(tmp_path: Path) -> Iterator[tuple[TestClient, OpenAIUpstreamAdapter]]:
    registry = ModelRegistry(models_dir=write_models(tmp_path), device="cpu", enable_hot_reload=False)
    app = FastAPI()
    app.include_router(router)
    app.state.registry = registry
    with TestClient(app) as client:
        adapter = client.portal.call(functools.partial(registry.load_async, REMOTE, device="cpu"))
        assert isinstance(adapter, OpenAIUpstreamAdapter)
        try:
            yield client, adapter
        finally:
            client.portal.call(adapter.aclose_client)
            client.portal.call(registry.unload_all_async)


@pytest.mark.parametrize("streaming", [False, True])
def test_the_direct_route_bounds_upstream_usage_by_the_variant_window(direct: tuple, streaming: bool) -> None:
    client, adapter = direct
    body = {"model": REMOTE, "messages": [{"role": "user", "content": "question"}], "max_tokens": 32768}
    if streaming:
        body.update(stream=True, stream_options={"include_usage": True})
    usages = [IN_WINDOW, OVER_WINDOW, OVER_REMAINDER]
    requests = answer(adapter, usages)

    served, *refused = [client.post("/v1/chat/completions", json=body) for _ in usages]

    assert served.status_code == 200, served.text
    assert served.headers["X-SIE-Served-By"] == "remote"
    assert '"prompt_tokens":100000' in served.text.replace(" ", "")
    assert '"error"' not in served.text
    for response in refused:
        if streaming:
            assert response.status_code == 200
            assert '"code":"upstream_error"' in response.text
        else:
            assert response.status_code == 502, response.text
    assert [json.loads(request.content)["max_tokens"] for request in requests] == [32768] * len(usages)


async def test_the_queued_route_bounds_upstream_usage_by_the_variant_window(tmp_path: Path) -> None:
    registry = ModelRegistry(models_dir=write_models(tmp_path), device="cpu", enable_hot_reload=False)
    adapter = await registry.load_async(REMOTE, device="cpu")
    assert isinstance(adapter, OpenAIUpstreamAdapter)
    usages = [IN_WINDOW, OVER_WINDOW, OVER_REMAINDER]
    requests = answer(adapter, usages * 2)

    async def run(*, stream: bool) -> dict[str, Any]:
        nc = AsyncMock()
        proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
        msg = AsyncMock()
        msg.data = msgpack.packb(
            {
                "request_id": "req-1",
                "work_item_id": "req-1.0",
                "item_index": 0,
                "total_items": 1,
                "operation": "generate",
                "model_id": REMOTE,
                "profile_id": "default",
                "pool_name": "default",
                "router_id": "router-1",
                "reply_subject": "_INBOX.router-1.req-1",
                "timestamp": time.time(),
                "generate": {
                    "messages": [{"role": "user", "content": "question"}],
                    "max_new_tokens": 32768,
                    "stream": stream,
                },
            },
            use_bin_type=True,
        )
        await proc.process(msg, REMOTE)
        return msgpack.unpackb(nc.publish.await_args_list[-1].args[1], raw=False)

    try:
        terminals = [await run(stream=stream) for stream in (False, True) for _ in usages]
    finally:
        await adapter.aclose_client()
        await registry.unload_all_async()

    assert [terminal["finish_reason"] for terminal in terminals] == ["stop", "error", "error"] * 2
    assert [terminals[0]["usage"]["prompt_tokens"], terminals[3]["usage"]["prompt_tokens"]] == [100_000, 100_000]
    assert [json.loads(request.content)["stream"] for request in requests] == [False] * 3 + [True] * 3
    assert [json.loads(request.content)["max_tokens"] for request in requests] == [32768] * 6
