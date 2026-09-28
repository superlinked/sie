"""A loaded model keeps serving while another model loads or is evicted.

Driven by sie-fake models through the real registry and the real encode
pipeline (``EncodePipeline.run_encode`` -> ``ModelRegistry.start_worker`` ->
``ModelWorker``). Slow loads and slow drains are held open with the fake's
load and dispatch latches, so every assertion is sequenced by state rather
than by timing: under a registry that serializes request admission behind
loads or drains, the encodes below cannot complete until the latch is
released, and ``asyncio.wait_for`` fails the test instead.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from sie_server.config.engine import EngineConfig
from sie_server.core.encode_pipeline import EncodePipeline
from sie_server.core.loader import load_model_configs
from sie_server.core.memory import SIE_FAKE_MEMORY_BUDGET_ENV, MemoryConfig
from sie_server.core.registry import ModelRegistry
from sie_server.core.residency import EvictionResult
from sie_server.core.worker.types import WorkerDrainedError
from sie_server.types.inputs import Item

pytestmark = pytest.mark.fake_stack

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SERVING = "sie-fake:small-a"
OTHER = "sie-fake:small-b"
REQUEST_TIMEOUT_S = 5.0
REQUESTS = 10


def _fake_registry(**kwargs: Any) -> ModelRegistry:
    registry = ModelRegistry(**kwargs)
    registry.add_config(load_model_configs(MODELS_DIR)["sie-fake"])
    return registry


async def _wait_until(predicate, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            msg = "condition not reached within timeout"
            raise TimeoutError(msg)
        await asyncio.sleep(0.01)


async def _encode(registry: ModelRegistry, model: str, text: str) -> list[dict[str, Any]]:
    results, _timing = await EncodePipeline.run_encode(
        registry=registry,
        model=model,
        items=[Item(text=text)],
        output_types=["dense"],
        instruction=None,
        config=registry.get_config(model),
        is_query=False,
        options={},
    )
    return results


async def _serve(registry: ModelRegistry, model: str) -> None:
    for index in range(REQUESTS):
        results = await asyncio.wait_for(_encode(registry, model, f"request {index}"), REQUEST_TIMEOUT_S)
        assert len(results) == 1


def _signal_dispatch(monkeypatch: pytest.MonkeyPatch, registry: ModelRegistry, model: str) -> threading.Event:
    """Set an event when the model's adapter starts an inference call."""
    adapter = registry.get(model)
    dispatched = threading.Event()
    real_encode = adapter.encode

    def encode_and_signal(*args: Any, **kwargs: Any) -> Any:
        dispatched.set()
        return real_encode(*args, **kwargs)

    monkeypatch.setattr(adapter, "encode", encode_and_signal)
    return dispatched


async def test_loaded_model_serves_while_another_model_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "1GiB")
    latch = tmp_path / "release-load"
    monkeypatch.setenv(
        "SIE_FAKE_FAULTS",
        f'{{"{OTHER}": {{"load_latch_file": "{latch}", "latch_timeout_s": 30}}}}',
    )
    registry = _fake_registry()
    await registry.load_async(SERVING, device="cpu")

    load_task = asyncio.create_task(registry.load_async(OTHER, device="cpu"))
    await _wait_until(lambda: registry._get_load_admission_lock().locked())

    # The first request also starts the serving model's worker.
    await _serve(registry, SERVING)
    assert not load_task.done()
    assert registry.is_loading(OTHER)

    latch.touch()
    await asyncio.wait_for(load_task, REQUEST_TIMEOUT_S)
    assert registry.is_loaded(OTHER)


async def test_loaded_model_serves_while_another_model_drains(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "1GiB")
    latch = tmp_path / "release-dispatch"
    monkeypatch.setenv(
        "SIE_FAKE_FAULTS",
        f'{{"{OTHER}": {{"dispatch_latch_file": "{latch}", "latch_timeout_s": 30}}}}',
    )
    registry = _fake_registry()
    await registry.load_async(SERVING, device="cpu")
    await registry.load_async(OTHER, device="cpu")

    # An inference call on OTHER is in flight and held, so its drain waits.
    dispatched = _signal_dispatch(monkeypatch, registry, OTHER)
    in_flight = asyncio.create_task(_encode(registry, OTHER, "held"))
    assert await asyncio.to_thread(dispatched.wait, REQUEST_TIMEOUT_S)
    # Touching SERVING afterwards leaves OTHER as the eviction candidate.
    await _serve(registry, SERVING)

    eviction = asyncio.create_task(registry.evict_lru_excluding(SERVING))
    await _wait_until(lambda: registry.is_unloading(OTHER))

    await _serve(registry, SERVING)
    assert not eviction.done()
    assert registry.is_unloading(OTHER)
    with pytest.raises(RuntimeError, match="currently being unloaded"):
        await registry.start_worker(OTHER)

    latch.touch()
    assert await asyncio.wait_for(eviction, REQUEST_TIMEOUT_S) is EvictionResult.EVICTED
    with pytest.raises(WorkerDrainedError):
        await asyncio.wait_for(in_flight, REQUEST_TIMEOUT_S)
    assert not registry.is_loaded(OTHER)
    assert registry.is_loaded(SERVING)
    with pytest.raises(KeyError, match="not loaded"):
        await registry.start_worker(OTHER)


async def test_load_of_a_draining_model_waits_and_reloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "1GiB")
    latch = tmp_path / "release-dispatch"
    monkeypatch.setenv(
        "SIE_FAKE_FAULTS",
        f'{{"{OTHER}": {{"dispatch_latch_file": "{latch}", "latch_timeout_s": 30}}}}',
    )
    registry = _fake_registry()
    await registry.load_async(SERVING, device="cpu")
    await registry.load_async(OTHER, device="cpu")
    evicted_adapter = registry.get(OTHER)
    dispatched = _signal_dispatch(monkeypatch, registry, OTHER)
    in_flight = asyncio.create_task(_encode(registry, OTHER, "held"))
    assert await asyncio.to_thread(dispatched.wait, REQUEST_TIMEOUT_S)

    eviction = asyncio.create_task(registry.evict_lru_excluding(SERVING))
    await _wait_until(lambda: registry.is_unloading(OTHER))
    reload = asyncio.create_task(registry.load_async(OTHER, device="cpu"))
    for _ in range(5):
        await asyncio.sleep(0)
    # Neither served from the draining adapter nor failed as "being unloaded".
    assert not reload.done()

    latch.touch()
    assert await asyncio.wait_for(eviction, REQUEST_TIMEOUT_S) is EvictionResult.EVICTED
    reloaded_adapter = await asyncio.wait_for(reload, REQUEST_TIMEOUT_S)
    with pytest.raises(WorkerDrainedError):
        await asyncio.wait_for(in_flight, REQUEST_TIMEOUT_S)
    assert reloaded_adapter is not evicted_adapter
    assert registry.get(OTHER) is reloaded_adapter


async def test_request_between_idle_snapshot_and_recheck_keeps_the_model() -> None:
    """The unlocked ``start_worker`` path cannot lose a request to idle eviction.

    The idle evictor snapshots a stale model and then waits for the load lock,
    which the test holds. A request for the model completes while the lock is
    held, and the evictor's re-check under the lock then keeps the model.
    """
    registry = _fake_registry(
        memory_config=MemoryConfig(memory_check_interval_s=0.01),
        engine_config=EngineConfig.model_construct(idle_evict_s=1),
    )
    await registry.load_async(SERVING, device="cpu")
    worker = await registry.start_worker(SERVING)
    manager = registry.memory_manager
    info = manager.get_model_info(SERVING)
    assert info is not None
    info.last_used_at = time.monotonic() - 100.0

    snapshots = 0
    first_snapshot = asyncio.Event()
    second_snapshot = asyncio.Event()
    real_get_idle_models = manager.get_idle_models

    def recording_get_idle_models(**kwargs: Any) -> list[str]:
        nonlocal snapshots
        snapshots += 1
        (first_snapshot if snapshots == 1 else second_snapshot).set()
        return real_get_idle_models(**kwargs)

    lock = registry._get_load_lock()
    with patch.object(manager, "get_idle_models", side_effect=recording_get_idle_models):
        await lock.acquire()
        try:
            await registry.start_idle_evictor()
            await asyncio.wait_for(first_snapshot.wait(), REQUEST_TIMEOUT_S)
            assert await asyncio.wait_for(registry.start_worker(SERVING), REQUEST_TIMEOUT_S) is worker
        finally:
            lock.release()
        await asyncio.wait_for(second_snapshot.wait(), REQUEST_TIMEOUT_S)
        await registry.stop_idle_evictor()

    assert registry.is_loaded(SERVING)
    assert not registry.is_unloading(SERVING)
    assert worker.is_running
    await registry.unload_all_async()
