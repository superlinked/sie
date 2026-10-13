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
import re
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
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.loader import load_model_configs
from sie_server.core.registry import ModelRegistry
from sie_server.core.upstream_client import upstream_client
from sie_server.processors.streaming import StreamingProcessor
from tokenizers import Tokenizer, models, pre_tokenizers

MODEL = "acme/agent-chat"
REMOTE = f"{MODEL}:remote"
UPSTREAM = "agent-api"
LOCAL_ADAPTER = "sie_server.adapters.fake.adapter:FakeAdapter"
REMOTE_ADAPTER = "sie_server.adapters.remote.openai:OpenAIUpstreamAdapter"
UPSTREAM_LOADTIME = {"upstream": UPSTREAM, "upstream_model": "vendor/agent-chat"}
IN_WINDOW = {"prompt_tokens": 100_000, "completion_tokens": 20_000, "total_tokens": 120_000}
OVER_WINDOW = {"prompt_tokens": 131_073, "completion_tokens": 2, "total_tokens": 131_075}
OVER_REMAINDER = {"prompt_tokens": 120_000, "completion_tokens": 12_000, "total_tokens": 132_000}
IN_BARE_WINDOW = {"prompt_tokens": 12_000, "completion_tokens": 4_000, "total_tokens": 16_000}
OVER_BARE_WINDOW = {"prompt_tokens": 16_385, "completion_tokens": 1, "total_tokens": 16_386}


def model_config(*, remote_loadtime: dict[str, Any] | None = None, remote_output: int | None = 32768) -> dict[str, Any]:
    remote: dict[str, Any] = {
        "adapter_path": REMOTE_ADAPTER,
        "max_batch_tokens": 16384,
        "adapter_options": {
            "loadtime": remote_loadtime
            if remote_loadtime is not None
            else {**UPSTREAM_LOADTIME, "max_seq_length": 131072}
        },
    }
    if remote_output is not None:
        remote["max_output_tokens"] = remote_output
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
            "remote": remote,
        },
        "routing": {"policy": "fallback", "fallback_profile": "remote"},
    }


def remote_backed_config() -> dict[str, Any]:
    return {
        "sie_id": "acme/agent-relay",
        "remote_backed": True,
        "inputs": {"text": True},
        "tasks": {"generate": {"context_length": 16384, "max_output_tokens": 4096}},
        "profiles": {
            "default": {
                "adapter_path": REMOTE_ADAPTER,
                "max_batch_tokens": 16384,
                "max_output_tokens": 32768,
                "adapter_options": {"loadtime": {**UPSTREAM_LOADTIME, "max_seq_length": 131072}},
            }
        },
    }


def write_models(tmp_path: Path, config: dict[str, Any] | None = None) -> Path:
    config = config if config is not None else model_config()
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / f"{config['sie_id'].replace('/', '__')}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return models_dir


def write_tokenizer(path: Path) -> Path:
    path.mkdir()
    vocab = {"[UNK]": 0, "<|user|>": 1, "<|assistant|>": 2, "word": 3, "answer": 4}
    tokenizer = Tokenizer(models.WordLevel(vocab, "[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(["<|user|>", "<|assistant|>"])
    tokenizer.save(str(path / "tokenizer.json"))
    template = (
        "{% for message in messages %}<|{{ message.role }}|>{{ message.content }}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]", "chat_template": template}),
        encoding="utf-8",
    )
    return path


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


async def run_queued(
    registry: ModelRegistry, *, stream: bool, content: str = "question", item: dict[str, Any] | None = None
) -> dict[str, Any]:
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
                "messages": [{"role": "user", "content": content}],
                "max_new_tokens": 32768,
                "stream": stream,
            },
            **(item or {}),
        },
        use_bin_type=True,
    )
    await proc.process(msg, REMOTE)
    return msgpack.unpackb(nc.publish.await_args_list[-1].args[1], raw=False)


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


def test_a_window_without_a_cap_of_its_own_keeps_the_models_cap(tmp_path: Path) -> None:
    config = model_config(remote_loadtime={**UPSTREAM_LOADTIME, "max_seq_length": 8192}, remote_output=None)

    remote = load_model_configs(write_models(tmp_path, config))[REMOTE].tasks.generate

    assert remote is not None
    assert (remote.context_length, remote.max_output_tokens) == (8192, 4096)


@pytest.mark.parametrize(
    ("config", "refusal"),
    [
        pytest.param(
            model_config(remote_output=131073),
            "Profile 'remote' on generation model 'acme/agent-chat' has max_output_tokens=131073, "
            "exceeding its context_length=131072",
            id="cap-above-the-profile-window",
        ),
        pytest.param(
            model_config(remote_loadtime=UPSTREAM_LOADTIME),
            "has max_output_tokens=32768, exceeding its context_length=16384",
            id="cap-above-the-model-window-without-a-profile-window",
        ),
        pytest.param(
            model_config(remote_loadtime={**UPSTREAM_LOADTIME, "max_seq_length": 2048}, remote_output=None),
            "has max_output_tokens=4096, exceeding its context_length=2048",
            id="inherited-cap-above-the-profile-window",
        ),
        pytest.param(
            remote_backed_config(),
            "Remote profile 'default' cannot declare 'max_seq_length': the default profile serves the model's own window",
            id="window-on-a-remote-default-profile",
        ),
    ],
)
def test_the_loader_refuses_an_output_cap_outside_the_window_it_is_served_under(
    tmp_path: Path, config: dict[str, Any], refusal: str
) -> None:
    with pytest.raises(ValidationError, match=re.escape(refusal)):
        load_model_configs(write_models(tmp_path, config))


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


def test_single_node_fallback_holds_a_bare_request_to_the_bare_window(direct: tuple) -> None:
    client, adapter = direct
    registry: ModelRegistry = client.app.state.registry
    body = {"model": MODEL, "messages": [{"role": "user", "content": "question"}]}
    usages = [IN_BARE_WINDOW, OVER_BARE_WINDOW]
    requests = answer(adapter, usages)

    with (
        patch.object(registry, "is_loading", side_effect=lambda name: name == MODEL),
        patch.object(registry, "start_load_async", AsyncMock(return_value=True)),
    ):
        served, refused = [client.post("/v1/chat/completions", json=body) for _ in usages]

    assert served.status_code == 200, served.text
    assert served.headers["X-SIE-Served-By"] == "remote"
    assert served.headers["X-SIE-Fallback-Reason"] == "model_loading"
    assert served.json()["model"] == MODEL
    assert refused.status_code == 503, refused.text
    assert refused.headers["X-SIE-Fallback-Reason"] == "model_loading"
    assert [json.loads(request.content)["max_tokens"] for request in requests] == [4096, 4096]


async def test_the_queued_route_bounds_upstream_usage_by_the_variant_window(tmp_path: Path) -> None:
    registry = ModelRegistry(models_dir=write_models(tmp_path), device="cpu", enable_hot_reload=False)
    adapter = await registry.load_async(REMOTE, device="cpu")
    assert isinstance(adapter, OpenAIUpstreamAdapter)
    usages = [IN_WINDOW, OVER_WINDOW, OVER_REMAINDER]
    requests = answer(adapter, usages * 2)

    try:
        terminals = [await run_queued(registry, stream=stream) for stream in (False, True) for _ in usages]
    finally:
        await adapter.aclose_client()
        await registry.unload_all_async()

    assert [terminal["finish_reason"] for terminal in terminals] == ["stop", "error", "error"] * 2
    assert [terminals[0]["usage"]["prompt_tokens"], terminals[3]["usage"]["prompt_tokens"]] == [100_000, 100_000]
    assert [json.loads(request.content)["stream"] for request in requests] == [False] * 3 + [True] * 3
    assert [json.loads(request.content)["max_tokens"] for request in requests] == [32768] * 6


@pytest.mark.parametrize("stream", [False, True])
async def test_the_queued_route_counts_the_prompt_against_the_variant_window(tmp_path: Path, stream: bool) -> None:
    config = model_config()
    del config["hf_id"], config["hf_revision"]
    config["weights_path"] = str(write_tokenizer(tmp_path / "weights"))
    registry = ModelRegistry(models_dir=write_models(tmp_path, config), device="cpu", enable_hot_reload=False)
    adapter = await registry.load_async(REMOTE, device="cpu")
    assert isinstance(adapter, OpenAIUpstreamAdapter)
    upstream_counts = {"prompt_tokens": 97_000, "completion_tokens": 3, "total_tokens": 97_003}
    requests = answer(adapter, [upstream_counts])

    try:
        served = await run_queued(registry, stream=stream, content=" ".join(["word"] * 98_302))
        refused = await run_queued(registry, stream=stream, content=" ".join(["word"] * 98_303))
    finally:
        await adapter.aclose_client()
        await registry.unload_all_async()

    assert served["finish_reason"] == "stop", served
    assert (served["usage"]["prompt_tokens"], served["usage"]["completion_tokens"]) == (98_304, 1)
    assert served["usage"]["upstream_usage"]["prompt_tokens"] == 97_000
    assert refused["finish_reason"] == "error"
    assert refused["error"]["code"] == "context_exceeded"
    assert "131073 exceeds context_length (131072)" in refused["error"]["message"]
    assert len(requests) == 1
    assert json.loads(requests[0].content)["max_tokens"] == 32768


async def test_a_queued_bare_request_routed_to_the_remote_profile_takes_its_window(tmp_path: Path) -> None:
    registry = ModelRegistry(models_dir=write_models(tmp_path), device="cpu", enable_hot_reload=False)
    adapter = await registry.load_async(REMOTE, device="cpu")
    assert isinstance(adapter, OpenAIUpstreamAdapter)
    answer(adapter, [IN_WINDOW, OVER_WINDOW])
    bridged = {"display_model": MODEL, "fallback_reason": "model_loading"}

    try:
        served = await run_queued(registry, stream=False, item=bridged)
        refused = await run_queued(registry, stream=False, item=bridged)
    finally:
        await adapter.aclose_client()
        await registry.unload_all_async()

    assert served["finish_reason"] == "stop", served
    assert served["usage"]["prompt_tokens"] == 100_000
    assert refused["finish_reason"] == "error"
