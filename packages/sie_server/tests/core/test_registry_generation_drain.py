from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import sie_server.core.registry as registry_module
from sie_server.adapters._generation_base import GenerationAdapter, GenerationChunk
from sie_server.adapters._spec import AdapterSpec
from sie_server.config.model import ModelConfig
from sie_server.core.model_loader import LoadedModel


class _DrainAdapter(GenerationAdapter):
    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self, events: list[str], *, block_drain: bool = False) -> None:
        self.events = events
        self.block_drain = block_drain
        self._device = "cpu"

    def load(self, device: str) -> None:
        self._device = device

    async def drain_generation(self) -> None:
        self.events.append("drain")
        if self.block_drain:
            await asyncio.Event().wait()

    def unload(self) -> None:
        self.events.append("unload")

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        _ = (prompt, max_new_tokens, kwargs)
        yield GenerationChunk(text_delta="", done=True, finish_reason="stop")


def _config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "sie_id": "test/generator",
            "hf_id": "test/generator",
            "inputs": {"text": True},
            "tasks": {"generate": {"context_length": 1024, "max_output_tokens": 128}},
            "profiles": {
                "default": {
                    "adapter_path": "tests.fake:Adapter",
                    "max_batch_tokens": 1024,
                    "kv_budget_tokens": 1024,
                }
            },
        }
    )


def _registry(
    adapter: GenerationAdapter,
    *,
    drain_timeout_s: float = 1.0,
) -> registry_module.ModelRegistry:
    registry = registry_module.ModelRegistry(drain_timeout_s=drain_timeout_s)
    config = _config()
    registry.add_config(config)
    registry._loaded[config.name] = LoadedModel(config=config, adapter=adapter, device="cpu")
    registry._loader.unregister = MagicMock()
    return registry


@pytest.mark.asyncio
async def test_generation_drain_precedes_removal_and_sync_unload() -> None:
    events: list[str] = []
    adapter = _DrainAdapter(events)
    registry = _registry(adapter)

    async def drain_with_visibility_check() -> None:
        assert registry.is_loaded("test/generator")
        events.append("drain")

    def unload_with_visibility_check() -> None:
        assert not registry.is_loaded("test/generator")
        events.append("unload")

    adapter.drain_generation = drain_with_visibility_check  # type: ignore[method-assign]
    adapter.unload = unload_with_visibility_check  # type: ignore[method-assign]

    await registry._do_unload("test/generator")

    assert events == ["drain", "unload"]
    assert not registry.is_loaded("test/generator")


@pytest.mark.asyncio
async def test_worker_and_generation_drain_share_one_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    adapter = _DrainAdapter(events, block_drain=True)
    registry = _registry(adapter, drain_timeout_s=0.5)
    worker = MagicMock()
    worker.is_running = True

    async def stop_worker() -> None:
        events.append("worker")
        await asyncio.sleep(0.02)

    worker.stop = stop_worker
    registry._loaded["test/generator"].worker = worker

    real_wait_for = asyncio.wait_for
    timeouts: list[float | None] = []

    async def recording_wait_for(awaitable: Any, **kwargs: Any) -> Any:
        timeout_value = kwargs.get("timeout")
        timeouts.append(timeout_value)
        return await real_wait_for(awaitable, **kwargs)

    monkeypatch.setattr(registry_module.asyncio, "wait_for", recording_wait_for)

    await registry._do_unload("test/generator")

    assert events == ["worker", "drain", "unload"]
    assert len(timeouts) == 2
    worker_timeout, generation_timeout = timeouts
    assert worker_timeout is not None
    assert generation_timeout is not None
    assert generation_timeout < worker_timeout - 0.01


@pytest.mark.asyncio
async def test_generation_drain_is_entered_after_worker_exhausts_deadline() -> None:
    events: list[str] = []
    adapter = _DrainAdapter(events, block_drain=True)
    registry = _registry(adapter, drain_timeout_s=0.01)
    worker = MagicMock()
    worker.is_running = True

    async def stop_worker() -> None:
        events.append("worker")
        await asyncio.Event().wait()

    worker.stop = stop_worker
    registry._loaded["test/generator"].worker = worker

    await registry._do_unload("test/generator")

    assert events == ["worker", "drain", "unload"]
    assert not registry.is_loaded("test/generator")


@pytest.mark.asyncio
async def test_generation_drain_timeout_still_completes_unload() -> None:
    events: list[str] = []
    adapter = _DrainAdapter(events, block_drain=True)
    registry = _registry(adapter, drain_timeout_s=0.01)

    await registry._do_unload("test/generator")

    assert events == ["drain", "unload"]
    assert not registry.is_loaded("test/generator")


@pytest.mark.asyncio
async def test_generation_drain_exception_still_completes_unload() -> None:
    events: list[str] = []
    adapter = _DrainAdapter(events)
    registry = _registry(adapter)

    async def failing_drain() -> None:
        events.append("drain")
        raise RuntimeError("drain failed")

    adapter.drain_generation = failing_drain  # type: ignore[method-assign]

    await registry._do_unload("test/generator")

    assert events == ["drain", "unload"]
    assert not registry.is_loaded("test/generator")


def test_sync_only_generation_unload_requires_async_lifecycle() -> None:
    events: list[str] = []
    registry = _registry(_DrainAdapter(events))

    with pytest.raises(RuntimeError, match="requires an owning event loop"):
        registry.unload("test/generator", reason="preload_pressure")

    assert registry.is_loaded("test/generator")
    assert events == []


@pytest.mark.asyncio
async def test_foreign_thread_generation_unload_runs_on_owner_loop() -> None:
    events: list[str] = []
    registry = _registry(_DrainAdapter(events))
    telemetry = MagicMock()
    owner = asyncio.get_running_loop()
    registry._get_load_lock()

    drain_loops: list[asyncio.AbstractEventLoop] = []

    async def drain_on_owner() -> None:
        drain_loops.append(asyncio.get_running_loop())
        events.append("drain")

    registry.get("test/generator").drain_generation = drain_on_owner  # type: ignore[method-assign]

    with patch("sie_server.core.registry.worker_telemetry", return_value=telemetry):
        await asyncio.to_thread(registry.unload, "test/generator", reason="preload_pressure")

    assert drain_loops == [owner]
    assert events == ["drain", "unload"]
    telemetry.model_evicted.assert_called_once_with(model="test/generator", reason="preload_pressure")


@pytest.mark.asyncio
async def test_same_loop_sync_unload_rejects_while_foreign_bridge_holds_thread_lock() -> None:
    events: list[str] = []
    registry = _registry(_DrainAdapter(events))
    registry._get_load_lock()
    drain_started = asyncio.Event()
    release_drain = asyncio.Event()

    async def blocking_drain() -> None:
        events.append("drain")
        drain_started.set()
        await release_drain.wait()

    registry.get("test/generator").drain_generation = blocking_drain  # type: ignore[method-assign]
    foreign_unload = asyncio.create_task(asyncio.to_thread(registry.unload, "test/generator"))
    await asyncio.wait_for(drain_started.wait(), timeout=1.0)

    with pytest.raises(RuntimeError, match="await unload_async"):
        registry.unload("test/generator")

    release_drain.set()
    await asyncio.wait_for(foreign_unload, timeout=2.0)
    assert events == ["drain", "unload"]


@pytest.mark.asyncio
async def test_async_generation_unload_preserves_reason() -> None:
    events: list[str] = []
    registry = _registry(_DrainAdapter(events))
    telemetry = MagicMock()

    with patch("sie_server.core.registry.worker_telemetry", return_value=telemetry):
        await registry.unload_async("test/generator", reason="config_change")

    assert events == ["drain", "unload"]
    telemetry.model_evicted.assert_called_once_with(model="test/generator", reason="config_change")


def test_sync_only_generation_unload_all_requires_async_lifecycle() -> None:
    events: list[str] = []
    registry = _registry(_DrainAdapter(events))

    with pytest.raises(RuntimeError, match="requires an owning event loop"):
        registry.unload_all()

    assert registry.is_loaded("test/generator")
    assert events == []


@pytest.mark.asyncio
async def test_sync_generation_unload_cannot_bypass_drain_inside_event_loop() -> None:
    events: list[str] = []
    registry = _registry(_DrainAdapter(events))

    with pytest.raises(RuntimeError, match="await unload_async"):
        registry.unload("test/generator")

    assert registry.is_loaded("test/generator")
    assert events == []


@pytest.mark.asyncio
async def test_repeated_generation_unload_is_idempotent() -> None:
    events: list[str] = []
    registry = _registry(_DrainAdapter(events))

    await registry._do_unload("test/generator")
    await registry._do_unload("test/generator")

    assert events == ["drain", "unload"]


@pytest.mark.asyncio
async def test_default_generation_drain_hook_is_noop() -> None:
    events: list[str] = []
    adapter = _DrainAdapter(events)
    adapter.drain_generation = GenerationAdapter.drain_generation.__get__(adapter)  # type: ignore[method-assign]
    registry = _registry(adapter)

    await registry._do_unload("test/generator")

    assert events == ["unload"]
