import asyncio
import gc
import itertools
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.config.model import EmbeddingDim, EncodeTask, ModelConfig, ProfileConfig, Tasks
from sie_server.core.load_errors import LoadErrorClass
from sie_server.core.memory import MemoryStats
from sie_server.core.registry import ModelRegistry

GIB = 1024**3


class _Allocation:
    """Device bytes held for as long as this object is alive."""

    _ids = itertools.count()

    def __init__(self, held: dict[int, int], nbytes: int) -> None:
        self.nbytes = nbytes
        key = next(self._ids)
        held[key] = nbytes
        weakref.finalize(self, held.pop, key, None)


class _Device:
    """Device memory held by live fake adapters.

    ``load()`` allocates up to the device limit before it fails with an OOM, as
    a real weight load does. That partial allocation belongs to the failing
    call's frame, so the exception's traceback keeps it alive and the adapter's
    ``unload()`` cannot free it. ``unload()`` collects garbage first, as the
    adapter contract does, then frees what the adapter holds.
    """

    def __init__(self, total_bytes: int) -> None:
        self.total_bytes = total_bytes
        self._held: dict[int, int] = {}
        self._partial: dict[int, int] = {}
        self._footprints: dict[str, int] = {}
        self._failing_unloads: set[str] = set()
        self._after_unload: dict[str, Callable[[], None]] = {}
        self.adapters: dict[str, list[MagicMock]] = {}
        self.stranded_bytes: list[int] = []

    def used_bytes(self) -> int:
        return sum(self._held.values()) + sum(self._partial.values())

    def stats(self) -> MemoryStats:
        return MemoryStats(used_bytes=self.used_bytes(), total_bytes=self.total_bytes, device_type="cuda")

    def declare(self, name: str, footprint_bytes: int) -> None:
        self._footprints[name] = footprint_bytes

    def fail_unload(self, name: str) -> None:
        self._failing_unloads.add(name)

    def after_unload(self, name: str, callback: Callable[[], None]) -> None:
        self._after_unload[name] = callback

    def make_adapter(self, name: str) -> MagicMock:
        footprint = self._footprints[name]
        adapter = MagicMock()
        adapter.capabilities.outputs = ["dense"]
        adapter.requires_main_thread = False
        del adapter.aclose_client
        adapter.load_required_memory_bytes.return_value = None
        key = id(adapter)
        held = self._held

        def load(_device: str) -> None:
            self.stranded_bytes.append(sum(self._partial.values()))
            free = self.total_bytes - self.used_bytes()
            if footprint > free:
                partial = _Allocation(self._partial, free)
                msg = f"CUDA out of memory. Tried to allocate {footprint - partial.nbytes} bytes"
                raise RuntimeError(msg)
            held[key] = footprint

        def unload() -> None:
            gc.collect()
            self.stranded_bytes.append(sum(self._partial.values()))
            if name in self._failing_unloads:
                msg = f"unload of {name} failed"
                raise RuntimeError(msg)
            held.pop(key, None)
            if name in self._after_unload:
                self._after_unload[name]()

        adapter.load.side_effect = load
        adapter.unload.side_effect = unload
        weakref.finalize(adapter, held.pop, key, None)
        self.adapters.setdefault(name, []).append(adapter)
        return adapter


def _config(name: str) -> ModelConfig:
    return ModelConfig(
        sie_id=name,
        hf_id=f"org/{name}",
        tasks=Tasks(encode=EncodeTask(dense=EmbeddingDim(dim=8))),
        profiles={
            "default": ProfileConfig(
                adapter_path="sie_server.adapters.sentence_transformer:SentenceTransformerDenseAdapter",
                max_batch_tokens=8192,
            )
        },
    )


@pytest.fixture(autouse=True)
def _no_downloads() -> Iterator[None]:
    with patch("sie_sdk.cache.ensure_model_cached", return_value=Path("/fake/cache/models--org--test")):
        yield


def _sync_registry(device: _Device, models: dict[str, int], *, pinned: list[str]) -> ModelRegistry:
    registry = ModelRegistry(pinned_models=pinned)
    for name, footprint in models.items():
        device.declare(name, footprint)
        registry.add_config(_config(name))
    patch.object(registry._memory_manager, "get_stats", side_effect=device.stats).start()
    patch(
        "sie_server.core.model_loader.load_adapter",
        side_effect=lambda config, *_args, **_kwargs: device.make_adapter(config.sie_id),
    ).start()
    return registry


async def _registry(
    device: _Device, models: dict[str, int], *, pinned: list[str], resident: list[str]
) -> ModelRegistry:
    """A registry on ``device`` with ``resident`` loaded in order, the first least recently used."""
    registry = _sync_registry(device, models, pinned=pinned)
    for name in resident:
        assert await registry.load_now(name, "cpu")
    return registry


@pytest.fixture(autouse=True)
def _stop_patches() -> Iterator[None]:
    yield
    patch.stopall()


async def test_retry_runs_after_the_failed_attempt_is_released() -> None:
    device = _Device(total_bytes=10 * GIB)
    registry = await _registry(
        device,
        {"pinned": 3 * GIB, "cold": 3 * GIB, "new": 5 * GIB},
        pinned=["pinned"],
        resident=["pinned", "cold"],
    )

    assert await registry.load_now("new", "cpu")

    assert registry.loaded_model_names == ["pinned", "new"]
    assert registry.get_failure("new") is None
    failed_attempt = device.adapters["new"][0]
    failed_attempt.unload.assert_called_once()
    assert not any(device.stranded_bytes)
    assert device.used_bytes() == 8 * GIB


async def test_retry_evicts_until_the_load_fits() -> None:
    device = _Device(total_bytes=16 * GIB)
    registry = await _registry(
        device,
        {"pinned": 2 * GIB, "a": 2 * GIB, "b": 2 * GIB, "c": 2 * GIB, "d": 2 * GIB, "new": 9 * GIB},
        pinned=["pinned"],
        resident=["pinned", "a", "b", "c", "d"],
    )

    assert await registry.load_now("new", "cpu")

    assert registry.is_loaded("new")
    assert registry.is_loaded("pinned")
    assert registry.is_loaded("d")
    assert not registry.is_loaded("a")
    assert registry.get_failure("new") is None


async def test_retries_while_unpinned_residents_remain() -> None:
    device = _Device(total_bytes=8 * GIB)
    registry = await _registry(
        device,
        {"pinned": 1 * GIB, "a": 1 * GIB, "b": 1 * GIB, "c": 1 * GIB, "d": 1 * GIB, "e": 1 * GIB, "new": 7 * GIB},
        pinned=["pinned"],
        resident=["pinned", "a", "b", "c", "d", "e"],
    )

    assert await registry.load_now("new", "cpu")

    assert registry.loaded_model_names == ["pinned", "new"]
    assert len(device.adapters["new"]) > 2
    assert registry.get_failure("new") is None


async def test_oom_with_only_pinned_residents_records_the_failure_and_releases_the_attempt() -> None:
    device = _Device(total_bytes=10 * GIB)
    registry = await _registry(
        device,
        {"pinned": 6 * GIB, "new": 5 * GIB},
        pinned=["pinned"],
        resident=["pinned"],
    )

    assert not await registry.load_now("new", "cpu")

    failure = registry.get_failure("new")
    assert failure is not None
    assert failure.error_class is LoadErrorClass.OOM
    assert registry.loaded_model_names == ["pinned"]
    [failed_attempt] = device.adapters["new"]
    failed_attempt.unload.assert_called_once()


def test_sync_load_releases_the_failed_attempt_and_evicts_until_the_load_fits() -> None:
    device = _Device(total_bytes=16 * GIB)
    registry = _sync_registry(
        device,
        {"pinned": 2 * GIB, "a": 2 * GIB, "b": 2 * GIB, "c": 2 * GIB, "d": 2 * GIB, "new": 9 * GIB},
        pinned=["pinned"],
    )
    for name in ["pinned", "a", "b", "c", "d"]:
        registry.load(name, "cpu")

    registry.load("new", "cpu")

    assert registry.is_loaded("new")
    assert registry.is_loaded("pinned")
    assert registry.is_loaded("d")
    assert not registry.is_loaded("a")
    device.adapters["new"][0].unload.assert_called_once()
    assert not any(device.stranded_bytes)


async def test_failed_release_of_the_attempt_evicts_nothing_and_records_the_oom() -> None:
    device = _Device(total_bytes=10 * GIB)
    registry = await _registry(
        device,
        {"pinned": 3 * GIB, "cold": 3 * GIB, "new": 5 * GIB},
        pinned=["pinned"],
        resident=["pinned", "cold"],
    )
    device.fail_unload("new")

    assert not await registry.load_now("new", "cpu")

    failure = registry.get_failure("new")
    assert failure is not None
    assert failure.error_class is LoadErrorClass.OOM
    assert registry.loaded_model_names == ["pinned", "cold"]


def test_sync_failed_release_of_the_attempt_evicts_nothing() -> None:
    device = _Device(total_bytes=10 * GIB)
    registry = _sync_registry(device, {"pinned": 3 * GIB, "cold": 3 * GIB, "new": 5 * GIB}, pinned=["pinned"])
    for name in ["pinned", "cold"]:
        registry.load(name, "cpu")
    device.fail_unload("new")

    with pytest.raises(RuntimeError, match="out of memory"):
        registry.load("new", "cpu")

    assert registry.loaded_model_names == ["pinned", "cold"]


async def _releasing(registry: ModelRegistry, name: str, done: asyncio.Event) -> asyncio.Task[int]:
    """Start an optional-memory release for ``name`` that runs until ``done`` is set."""

    async def release() -> int:
        await done.wait()
        return 0

    task = asyncio.create_task(release())
    registry._optional_release_tasks[name] = task
    await asyncio.sleep(0)
    return task


async def test_retry_eviction_skips_a_model_still_releasing_optional_memory() -> None:
    device = _Device(total_bytes=12 * GIB)
    registry = await _registry(
        device,
        {"pinned": 2 * GIB, "a": 2 * GIB, "b": 4 * GIB, "new": 6 * GIB},
        pinned=["pinned"],
        resident=["pinned", "a", "b"],
    )
    done = asyncio.Event()
    release = await _releasing(registry, "a", done)

    assert await registry.load_now("new", "cpu")

    assert registry.loaded_model_names == ["pinned", "a", "new"]
    done.set()
    await release


async def test_retry_waits_for_the_optional_memory_release_of_the_last_unpinned_model() -> None:
    device = _Device(total_bytes=10 * GIB)
    registry = await _registry(
        device,
        {"pinned": 2 * GIB, "a": 2 * GIB, "new": 7 * GIB},
        pinned=["pinned"],
        resident=["pinned", "a"],
    )
    done = asyncio.Event()
    release = await _releasing(registry, "a", done)
    a_adapter = device.adapters["a"][0]
    unload_a = a_adapter.unload.side_effect
    release_done_at_unload: list[bool] = []

    def record_unload() -> None:
        release_done_at_unload.append(release.done())
        unload_a()

    a_adapter.unload.side_effect = record_unload
    loop = asyncio.get_running_loop()
    device.after_unload("new", lambda: loop.call_soon_threadsafe(loop.call_later, 0.1, done.set))

    assert await registry.load_now("new", "cpu")

    assert registry.loaded_model_names == ["pinned", "new"]
    assert release_done_at_unload == [True]


async def test_retry_fails_when_the_last_unpinned_model_is_still_releasing_optional_memory() -> None:
    device = _Device(total_bytes=10 * GIB)
    registry = await _registry(
        device,
        {"pinned": 2 * GIB, "a": 2 * GIB, "new": 7 * GIB},
        pinned=["pinned"],
        resident=["pinned", "a"],
    )
    done = asyncio.Event()
    release = await _releasing(registry, "a", done)

    with patch("sie_server.core.registry._OPTIONAL_MEMORY_RELEASE_TIMEOUT_S", 0.01):
        assert not await registry.load_now("new", "cpu")

    failure = registry.get_failure("new")
    assert failure is not None
    assert failure.error_class is LoadErrorClass.OOM
    assert registry.loaded_model_names == ["pinned", "a"]
    done.set()
    await release
