from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.registry import ModelRegistry


def _config(name: str) -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        hf_id=f"org/{name}",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=768))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
    )


def _adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.capabilities.outputs = ["dense"]
    adapter.load_required_memory_bytes.return_value = None
    adapter.memory_footprint.return_value = 1000
    adapter.requires_main_thread = False
    del adapter.aclose_client
    return adapter


@pytest.fixture(autouse=True)
def _cached_weights() -> Iterator[MagicMock]:
    with patch("sie_sdk.cache.ensure_model_cached", return_value=Path("/fake/cache/model")) as mocked:
        yield mocked


@pytest.mark.asyncio
async def test_foreign_thread_sync_load_eviction_and_unload_all_use_owner_loop() -> None:
    registry = ModelRegistry()
    for name in ("model-a", "model-b"):
        registry.add_config(_config(name))

    adapter_a = _adapter()
    adapter_b = _adapter()
    owner = asyncio.get_running_loop()
    unload_loops: list[asyncio.AbstractEventLoop] = []
    original_do_unload = registry._do_unload

    async def record_unload_loop(name: str, *, reason: str = "other") -> None:
        unload_loops.append(asyncio.get_running_loop())
        await original_do_unload(name, reason=reason)

    registry._do_unload = record_unload_loop  # type: ignore[method-assign]

    with patch("sie_server.core.model_loader.load_adapter", side_effect=[adapter_a, adapter_b]):
        with patch.object(registry._memory_manager, "should_evict_for_load", return_value=False):
            await registry.load_async("model-a", "cpu")

        with patch.object(registry._memory_manager, "should_evict_for_load", side_effect=[True, False]):
            loaded = await asyncio.to_thread(registry.load, "model-b", "cpu")

        assert loaded is adapter_b
        assert not registry.is_loaded("model-a")
        assert registry.is_loaded("model-b")
        assert unload_loops == [owner]

        await asyncio.to_thread(registry.unload_all)

    assert registry.loaded_model_names == []
    assert unload_loops == [owner, owner]
    adapter_a.unload.assert_called_once()
    adapter_b.unload.assert_called_once()


@pytest.mark.asyncio
async def test_async_lifecycle_rejects_a_second_event_loop() -> None:
    registry = ModelRegistry()
    registry._get_load_lock()

    def use_another_loop() -> str:
        try:
            asyncio.run(registry.unload_all_async())
        except RuntimeError as exc:
            return str(exc)
        raise AssertionError("second event loop unexpectedly entered the registry lifecycle")

    message = await asyncio.to_thread(use_another_loop)

    assert message == "model lifecycle is bound to a different event loop"


def test_sync_only_load_and_unload_share_thread_safe_boundary() -> None:
    registry = ModelRegistry()
    registry.add_config(_config("model"))
    adapter = _adapter()
    load_started = threading.Event()
    release_load = threading.Event()
    unload_attempted = threading.Event()
    unload_entered = threading.Event()
    original_unload_sync = registry._unload_sync

    def blocking_load(_device: str) -> None:
        load_started.set()
        assert release_load.wait(timeout=2.0)

    def record_unload(name: str, *, reason: str) -> None:
        unload_entered.set()
        original_unload_sync(name, reason=reason)

    def unload_from_thread() -> None:
        unload_attempted.set()
        registry.unload("model")

    adapter.load.side_effect = blocking_load
    registry._unload_sync = record_unload  # type: ignore[method-assign]

    with (
        patch("sie_server.core.model_loader.load_adapter", return_value=adapter),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        load_future = executor.submit(registry.load, "model", "cpu")
        assert load_started.wait(timeout=2.0)
        unload_future = executor.submit(unload_from_thread)
        assert unload_attempted.wait(timeout=2.0)
        assert not unload_entered.is_set()

        release_load.set()
        assert load_future.result(timeout=2.0) is adapter
        unload_future.result(timeout=2.0)

    assert unload_entered.is_set()
    assert not registry.is_loaded("model")
    adapter.unload.assert_called_once()


def test_nested_sync_lifecycle_cannot_expose_outer_load_to_async_binding() -> None:
    registry = ModelRegistry()
    registry.add_config(_config("model"))
    adapter = _adapter()
    nested_returned = threading.Event()
    release_outer_load = threading.Event()

    def reentrant_load(_device: str) -> None:
        assert registry._sync_lifecycle_depth == 1
        registry.unload_all()
        assert registry._sync_lifecycle_depth == 1
        nested_returned.set()
        assert release_outer_load.wait(timeout=2.0)

    adapter.load.side_effect = reentrant_load

    with (
        patch("sie_server.core.model_loader.load_adapter", return_value=adapter),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        outer_load = executor.submit(registry.load, "model", "cpu")
        assert nested_returned.wait(timeout=2.0)

        with pytest.raises(RuntimeError, match="synchronous model lifecycle operation is already in progress"):
            asyncio.run(registry.unload_all_async())

        assert registry._lifecycle_loop is None
        assert registry._sync_lifecycle_depth == 1
        release_outer_load.set()
        assert outer_load.result(timeout=2.0) is adapter

    assert registry._sync_lifecycle_depth == 0
    assert registry._lifecycle_loop is None
    assert registry.loaded_model_names == ["model"]

    registry.unload("model")
    assert registry.loaded_model_names == []
