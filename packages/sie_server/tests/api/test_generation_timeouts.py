"""Direct-server buffered generation honours the resolved profile timeouts."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_server.adapters._generation_base import GenerationChunk
from sie_server.adapters.fake.adapter import FakeAdapter
from sie_server.api.generate import router as generate_router
from sie_server.api.openai_completions import router as completions_router
from sie_server.api.openai_responses import router as responses_router
from sie_server.config.model import AdapterOptions, GenerateTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.registry import ModelRegistry

_MODEL = "sie-fake/timeouts"


class _ClosingFakeAdapter(FakeAdapter):
    """The fake engine, recording whether the route closed its stream."""

    closed = False

    async def generate(self, prompt: str, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        try:
            async for chunk in super().generate(prompt, **kwargs):
                yield chunk
        finally:
            self.closed = True


def _config(runtime: dict[str, float]) -> ModelConfig:
    return ModelConfig(
        sie_id=_MODEL,
        hf_id=_MODEL,
        tasks=Tasks(generate=GenerateTask(context_length=4096, max_output_tokens=64)),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.fake.adapter:FakeAdapter",
                max_batch_tokens=4096,
                kv_budget_tokens=2048,
                adapter_options=AdapterOptions(runtime=runtime),
            )
        },
    )


def _client(adapter: FakeAdapter, runtime: dict[str, float]) -> TestClient:
    adapter.load("cpu")
    registry = MagicMock(spec=ModelRegistry)
    registry.has_model.return_value = True
    registry.is_loaded.return_value = True
    registry.is_loading.return_value = False
    registry.is_unloading.return_value = False
    registry.is_failed.return_value = False
    registry.get_failure.return_value = None
    registry.get_config.return_value = _config(runtime)
    registry.get.return_value = adapter
    registry.device = "cpu"
    registry.engine_config = None
    app = FastAPI()
    app.include_router(generate_router)
    app.include_router(completions_router)
    app.include_router(responses_router)
    app.state.registry = registry
    return TestClient(app)


def _native_code(body: dict[str, Any]) -> str:
    return body["detail"]["code"]


def _openai_code(body: dict[str, Any]) -> str:
    return body["error"]["code"]


_ROUTES: list[tuple[str, dict[str, Any], Callable[[dict[str, Any]], str]]] = [
    (f"/v1/generate/{_MODEL.replace('/', '__')}", {"prompt": "hello", "max_new_tokens": 8}, _native_code),
    ("/v1/completions", {"model": _MODEL, "prompt": "hello", "max_tokens": 8}, _openai_code),
    ("/v1/responses", {"model": _MODEL, "input": "hello", "max_output_tokens": 8}, _openai_code),
]


@pytest.mark.parametrize(("path", "body", "error_code"), _ROUTES)
def test_first_chunk_timeout_aborts_the_engine_and_returns_504(
    path: str,
    body: dict[str, Any],
    error_code: Callable[[dict[str, Any]], str],
) -> None:
    adapter = _ClosingFakeAdapter(inter_token_latency_s=2.0)
    client = _client(adapter, {"first_chunk_timeout_s": 0.05, "overall_timeout_s": 10})

    response = client.post(path, json=body)

    assert response.status_code == 504, response.text
    assert error_code(response.json()) == "first_chunk_timeout"
    assert adapter.closed


@pytest.mark.parametrize(("path", "body", "error_code"), _ROUTES)
def test_overall_timeout_aborts_a_slow_generation_and_returns_504(
    path: str,
    body: dict[str, Any],
    error_code: Callable[[dict[str, Any]], str],
) -> None:
    adapter = _ClosingFakeAdapter(inter_token_latency_s=0.1)
    client = _client(adapter, {"first_chunk_timeout_s": 5, "overall_timeout_s": 0.3})

    response = client.post(path, json=body)

    assert response.status_code == 504, response.text
    assert error_code(response.json()) == "overall_timeout"
    assert adapter.closed


class _SlowAbortFakeAdapter(FakeAdapter):
    """The fake engine, with an abort that hangs once a request is cancelled."""

    abort_started = False

    async def generate(self, prompt: str, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        _ = (prompt, kwargs)
        try:
            await asyncio.sleep(30)
            yield GenerationChunk(text_delta="late")
        except asyncio.CancelledError:
            self.abort_started = True
            await asyncio.sleep(8)
            raise


def test_native_route_answers_a_timeout_before_a_hung_engine_abort_finishes() -> None:
    adapter = _SlowAbortFakeAdapter()
    client = _client(adapter, {"first_chunk_timeout_s": 0.05, "overall_timeout_s": 10})
    started = time.monotonic()

    response = client.post(_ROUTES[0][0], json=_ROUTES[0][1])

    elapsed = time.monotonic() - started
    assert response.status_code == 504, response.text
    assert response.json()["detail"]["code"] == "first_chunk_timeout"
    assert adapter.abort_started
    assert 1.5 < elapsed < 5.0, elapsed


@pytest.mark.parametrize(("path", "body", "error_code"), _ROUTES)
def test_generation_within_its_timeouts_succeeds(
    path: str,
    body: dict[str, Any],
    error_code: Callable[[dict[str, Any]], str],
) -> None:
    _ = error_code
    adapter = _ClosingFakeAdapter(inter_token_latency_s=0.0)
    client = _client(adapter, {"first_chunk_timeout_s": 5, "overall_timeout_s": 10})

    response = client.post(path, json=body)

    assert response.status_code == 200, response.text
