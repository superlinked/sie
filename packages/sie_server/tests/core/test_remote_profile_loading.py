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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from sie_server.adapters.remote.sie import SieUpstreamAdapter
from sie_server.config.model import AdapterOptions, EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.config.upstreams import Upstream, install_upstreams
from sie_server.core.encode_pipeline import EncodePipeline
from sie_server.core.load_errors import DevicePlacementError
from sie_server.core.model_loader import LoadedModel
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


async def test_load_now_does_not_wait_for_and_retry_a_failing_background_load(tmp_path: Path) -> None:
    registry = registry_for(tmp_path, model_config("acme/hybrid"))
    entered = threading.Event()
    release = threading.Event()

    def fail_load(*_args: Any, **_kwargs: Any) -> None:
        entered.set()
        assert release.wait(TIMEOUT_S)
        raise RuntimeError("injected background load failure")

    try:
        with patch("sie_server.core.model_loader.load_adapter", side_effect=fail_load) as load:
            assert await registry.start_load_async("acme/hybrid", device="cpu")
            await wait_until(entered.is_set)

            assert not await asyncio.wait_for(registry.load_now("acme/hybrid", device="cpu"), 0.5)

            release.set()
            await wait_until(lambda: not registry.is_loading("acme/hybrid"))
            assert not await registry.load_now("acme/hybrid", device="cpu")
            assert load.call_count == 1
            failure = registry.get_failure("acme/hybrid")
            assert failure is not None
            assert failure.attempts == 1
    finally:
        release.set()
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


@pytest.mark.parametrize("device", ["cuda", "cuda:1"])
@pytest.mark.parametrize("phantom", [False, True])
@pytest.mark.parametrize("claimed", [False, True])
async def test_remote_lifecycle_keeps_configured_device_accounting(device: str, phantom: bool, claimed: bool) -> None:
    registry = ModelRegistry(device="cuda", devices=["cuda:0", "cuda:1"])
    registry.add_config(ModelConfig.model_validate(model_config("acme/hybrid")))
    name = "acme/hybrid:remote"
    if phantom:
        registry._memory_manager_for_device("cuda")
    for manager in registry.memory_managers.values():
        manager.check_pressure = MagicMock(return_value=False)  # type: ignore[method-assign]
    if claimed:
        registry._device_claims = {"cuda:0": "owner", "cuda:1": "owner"}
    before_managers = registry.memory_managers
    before_order = dict(registry._device_order)
    before_claims = dict(registry._device_claims)
    adapter = MagicMock()

    async def loaded_remote(_name: str, config: ModelConfig, _model_dir: Path, metadata_device: str) -> LoadedModel:
        return LoadedModel(config=config, adapter=adapter, device=metadata_device)

    loader = AsyncMock(side_effect=loaded_remote)
    with (
        patch.object(registry._loader, "load_remote_async", loader),
        patch.object(registry._loader, "unregister"),
    ):
        try:
            assert await registry.load_async(name, device=device) is adapter
            assert registry._loaded[name].device == (device if device == "cuda:1" else "cuda:0")
            assert registry.get(name) is adapter
            registry.touch_lru(name)
            await registry.unload_async(name)
            assert await registry.start_load_async(name, device=device)
            await wait_until(lambda: registry.is_loaded(name) and not registry.is_loading(name))
            registry.touch_lru(name)
            await registry.unload_async(name)

            assert registry.memory_managers == before_managers
            assert registry._device_order == before_order
            assert registry._device_claims == before_claims
            assert all(manager.get_model_info(name) is None for manager in registry.memory_managers.values())
            assert loader.await_count == 2
            if claimed:
                with pytest.raises(DevicePlacementError, match="No device is available"):
                    registry._select_device_for_model("cuda")
            else:
                registry._device_claims["cuda:0"] = "owner"
                assert registry._select_device_for_model("cuda") == "cuda:1"
                if not phantom:
                    assert registry._resolve_load_device("cuda") == "cuda:1"
        finally:
            await registry.unload_all_async()


async def test_concurrent_inline_loads_do_not_retry_a_recorded_failure(tmp_path: Path) -> None:
    registry = registry_for(tmp_path, model_config("acme/local"))
    name = "acme/local:remote"
    loader = AsyncMock(side_effect=RuntimeError("remote load failed"))

    async def inline_load(started: asyncio.Event) -> bool:
        started.set()
        return await registry.load_now(name, "cpu")

    with patch.object(registry._loader, "load_remote_async", loader):
        try:
            async with registry._get_config_update_lock():
                first_started, second_started = asyncio.Event(), asyncio.Event()
                first = asyncio.create_task(inline_load(first_started))
                await first_started.wait()
                second = asyncio.create_task(inline_load(second_started))
                await second_started.wait()
            assert await asyncio.wait_for(asyncio.gather(first, second), TIMEOUT_S) == [False, False]
            assert loader.await_count == 1
            assert registry.get_failure(name).attempts == 1
            assert not registry.is_loading(name)
        finally:
            await registry.unload_all_async()


async def test_cancelled_inline_load_clears_its_loading_claim(tmp_path: Path) -> None:
    registry = registry_for(tmp_path, model_config("acme/local"))
    name = "acme/local:remote"
    started = asyncio.Event()

    async def inline_load() -> bool:
        started.set()
        return await registry.load_now(name, "cpu")

    try:
        async with registry._get_config_update_lock():
            task = asyncio.create_task(inline_load())
            await started.wait()
            assert registry.is_loading(name)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not registry.is_loading(name)
        assert registry.get_failure(name) is None
        assert await registry.load_now(name, "cpu")
    finally:
        await registry.unload_all_async()
