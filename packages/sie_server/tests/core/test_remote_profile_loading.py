"""A remote profile loads beside the local model of the same name.

A remote adapter holds no weights and no device memory, and loading it makes no
outbound call. It must not wait for another model's load, download the local
model's weights, or be chosen for eviction. The local model is the fake adapter;
its load and its in-flight work are held open with the fake's latches, so every
assertion is sequenced by registry state rather than by timing.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.config.model import AdapterOptions, EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.encode_pipeline import EncodePipeline
from sie_server.core.registry import ModelRegistry
from sie_server.core.worker.types import WorkerDrainedError
from sie_server.types.inputs import Item

LOCAL_ADAPTER = "sie_server.adapters.fake.adapter:FakeAdapter"
REMOTE_ADAPTER = "sie_server.adapters.remote.sie:SieUpstreamAdapter"
TIMEOUT_S = 10.0


def model_config(
    sie_id: str,
    *,
    faults: dict[str, Any] | None = None,
    upstream_model: str = "sie-fake",
    weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loadtime: dict[str, Any] = {"memory_footprint_bytes": 64 << 20, "fault_key": sie_id}
    if faults is not None:
        loadtime["faults"] = faults
    return {
        "sie_id": sie_id,
        **(weights if weights is not None else {"package_backed": True}),
        "inputs": {"text": True},
        "tasks": {"encode": {"dense": {"dim": 384}}},
        "profiles": {
            "default": {
                "adapter_path": LOCAL_ADAPTER,
                "max_batch_tokens": 8192,
                "adapter_options": {"loadtime": loadtime},
            },
            "remote": {
                "adapter_path": REMOTE_ADAPTER,
                "max_batch_tokens": 8192,
                "adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": upstream_model}},
            },
        },
    }


def registry_for(tmp_path: Path, *configs: dict[str, Any]) -> ModelRegistry:
    models = tmp_path / "models"
    models.mkdir()
    for config in configs:
        (models / f"{config['sie_id'].replace('/', '__')}.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return ModelRegistry(models_dir=models, device="cpu", enable_hot_reload=False)


async def wait_until(predicate: Callable[[], bool], timeout_s: float = TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            msg = "condition not reached within timeout"
            raise TimeoutError(msg)
        await asyncio.sleep(0.01)


@pytest.fixture(autouse=True)
def _upstreams() -> Iterator[None]:
    install_upstreams(
        {
            "team-sie": Upstream.model_validate(
                {
                    "kind": "sie",
                    "base_url": "http://127.0.0.1:9",
                    "rate_cap": {"requests_per_minute": 60, "max_concurrency": 4},
                }
            )
        }
    )
    yield
    install_upstreams({})


async def test_a_remote_profile_loads_while_the_local_model_is_still_loading(tmp_path: Path) -> None:
    latch = tmp_path / "release-local-load"
    registry = registry_for(
        tmp_path, model_config("acme/hybrid", faults={"load_latch_file": str(latch), "latch_timeout_s": 30})
    )
    try:
        assert await registry.start_load_async("acme/hybrid", device="cpu")
        await wait_until(lambda: registry._get_load_admission_lock().locked())

        adapter = await asyncio.wait_for(registry.load_async("acme/hybrid:remote", device="cpu"), TIMEOUT_S)

        assert isinstance(adapter, SieUpstreamAdapter)
        assert registry.is_loaded("acme/hybrid:remote")
        assert registry.is_loading("acme/hybrid")
    finally:
        latch.touch()
        await wait_until(lambda: not registry.is_loading("acme/hybrid"))
        await registry.unload_all_async()


async def test_a_remote_profile_of_a_model_with_weights_downloads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    registry = registry_for(
        tmp_path,
        model_config("acme/weights", weights={"hf_id": "acme/not-on-any-hub", "hf_revision": "0" * 40}),
    )
    try:
        with pytest.raises(Exception, match="acme/not-on-any-hub"):
            registry._loader.ensure_weights_cached(registry.get_config("acme/weights"))

        adapter = await asyncio.wait_for(registry.load_async("acme/weights:remote", device="cpu"), TIMEOUT_S)

        assert isinstance(adapter, SieUpstreamAdapter)
    finally:
        await registry.unload_all_async()


async def test_a_remote_profile_holds_no_memory_and_is_never_chosen_for_eviction(tmp_path: Path) -> None:
    registry = registry_for(tmp_path, model_config("acme/hybrid"))
    try:
        await registry.load_async("acme/hybrid:remote", device="cpu")
        await registry.load_async("acme/hybrid", device="cpu")
        registry.touch_lru("acme/hybrid:remote")
        managers = list(registry.memory_managers.values())

        assert all(manager.get_model_info("acme/hybrid:remote") is None for manager in managers)
        assert [manager.get_lru_model() for manager in managers if manager.loaded_model_count] == ["acme/hybrid"]
        assert [name for manager in managers for name in manager.get_idle_models(idle_threshold_s=0.0)] == [
            "acme/hybrid"
        ]

        await registry.unload_async("acme/hybrid:remote")
        assert not registry.is_loaded("acme/hybrid:remote")
        assert registry.is_loaded("acme/hybrid")
    finally:
        await registry.unload_all_async()


async def test_a_remote_load_overtaken_by_a_config_change_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = registry_for(tmp_path, model_config("acme/hybrid"))
    changed = ModelConfig.model_validate(model_config("acme/hybrid", upstream_model="sie-fake:other"))
    real_load_remote = registry._loader.load_remote_async
    discarded: list[SieUpstreamAdapter] = []

    async def load_then_change(name: str, config: ModelConfig, model_dir: Path, device: str) -> Any:
        loaded = await real_load_remote(name, config, model_dir, device)
        discarded.append(loaded.adapter)
        await registry.add_config_async(changed)
        return loaded

    monkeypatch.setattr(registry._loader, "load_remote_async", load_then_change)
    try:
        with pytest.raises(RuntimeError, match="config changed while it was loading"):
            await registry.load_async("acme/hybrid:remote", device="cpu")

        assert not registry.is_loaded("acme/hybrid:remote")
        assert discarded[0]._client is None
    finally:
        await registry.unload_all_async()


async def test_a_load_started_while_the_model_drains_reloads_it_after_the_unload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    latch = tmp_path / "release-dispatch"
    registry = registry_for(
        tmp_path, model_config("acme/local", faults={"dispatch_latch_file": str(latch), "latch_timeout_s": 30})
    )
    try:
        await registry.load_async("acme/local", device="cpu")
        unloaded_adapter = registry.get("acme/local")
        dispatched = threading.Event()
        real_encode = unloaded_adapter.encode

        def encode_and_signal(*args: Any, **kwargs: Any) -> Any:
            dispatched.set()
            return real_encode(*args, **kwargs)

        monkeypatch.setattr(unloaded_adapter, "encode", encode_and_signal)
        in_flight = asyncio.create_task(
            EncodePipeline.run_encode(
                registry=registry,
                model="acme/local",
                items=[Item(text="held")],
                output_types=["dense"],
                instruction=None,
                config=registry.get_config("acme/local"),
                is_query=False,
                options={},
            )
        )
        assert await asyncio.to_thread(dispatched.wait, TIMEOUT_S)
        unload = asyncio.create_task(registry.unload_async("acme/local"))
        await wait_until(lambda: registry.is_unloading("acme/local"))
        assert registry.is_loaded("acme/local")

        assert await registry.start_load_async("acme/local", device="cpu")

        latch.touch()
        await asyncio.wait_for(unload, TIMEOUT_S)
        with pytest.raises(WorkerDrainedError):
            await asyncio.wait_for(in_flight, TIMEOUT_S)
        await wait_until(lambda: registry.is_loaded("acme/local") and not registry.is_loading("acme/local"))
        assert registry.get("acme/local") is not unloaded_adapter
    finally:
        latch.touch()
        await registry.unload_all_async()


async def test_a_remote_profile_does_not_occupy_a_device_group() -> None:
    wide = ModelConfig(
        sie_id="acme/wide",
        hf_id="acme/wide",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=384))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sglang.embedding:SGLangEmbeddingAdapter",
                max_batch_tokens=8192,
                adapter_options=AdapterOptions(loadtime={"tensor_parallel_size": 2, "request_read_timeout_s": 120.0}),
            )
        },
    )
    remote = ModelConfig.model_validate(
        {
            "sie_id": "acme/remote",
            "remote_backed": True,
            "tasks": {"encode": {"dense": {"dim": 384}}},
            "profiles": {
                "default": {
                    "adapter_path": REMOTE_ADAPTER,
                    "max_batch_tokens": 8192,
                    "adapter_options": {"loadtime": {"upstream": "team-sie", "upstream_model": "sie-fake"}},
                }
            },
        }
    )
    registry = ModelRegistry(device="cuda", devices=["cuda:0", "cuda:1"])
    for manager in registry.memory_managers.values():
        manager.check_pressure = MagicMock(return_value=False)  # type: ignore[method-assign]
    registry.add_config(remote)
    registry.add_config(wide)
    wide_adapter = MagicMock()
    wide_adapter.capabilities.outputs = ["dense"]
    wide_adapter.memory_footprint.return_value = 1000
    try:
        await registry.load_async("acme/remote", device="cuda:0")
        with (
            patch("sie_sdk.cache.ensure_model_cached", return_value=Path("/fake/cache/acme-wide")),
            patch("sie_server.core.model_loader.load_adapter", return_value=wide_adapter),
        ):
            await registry.load_async("acme/wide", device="cuda")

        assert sorted(registry._device_claims) == ["cuda:0", "cuda:1"]
        assert registry.is_loaded("acme/remote")
    finally:
        await registry.unload_all_async()
