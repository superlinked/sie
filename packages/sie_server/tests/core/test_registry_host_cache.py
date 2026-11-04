"""Registry park and wake through the host-RAM tier, with weightless fakes.

The synthetic memory tracker (``SIE_FAKE_MEMORY_BUDGET``) makes device usage
the sum of the fakes' declared footprints, and the fake adapter simulates
parking by moving that footprint to the host tier. Budget 150 MiB on the
device; ``sie-fake:small-a``/``small-b`` declare 64 MiB, ``sie-fake`` 128 MiB.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from sie_server.adapters.base import HostOffload
from sie_server.config.engine import EngineConfig
from sie_server.core import registry as registry_module
from sie_server.core.loader import load_model_configs
from sie_server.core.memory import SIE_FAKE_MEMORY_BUDGET_ENV
from sie_server.core.model_loader import LoadedLora
from sie_server.core.registry import ModelRegistry

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MIB = 1024**2
SMALL_A = "sie-fake:small-a"
SMALL_B = "sie-fake:small-b"
BIG = "sie-fake"


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.lookups: Counter[tuple[str, bool]] = Counter()
        self.wakes: list[tuple[str, str]] = []
        self.host_bytes: dict[str, int] = {}
        self.load_completions: list[str] = []

    def host_cache_lookup(self, *, model: str, hit: bool, **_: Any) -> None:
        self.lookups[(model, hit)] += 1

    def host_cache_wake_completed(self, *, model: str, outcome: str, **_: Any) -> None:
        self.wakes.append((model, outcome))

    def host_cache_memory_changed(self, *, model: str, host_bytes: int, **_: Any) -> None:
        self.host_bytes[model] = host_bytes

    def model_load_completed(self, *, model: str, **_: Any) -> None:
        self.load_completions.append(model)

    def __getattr__(self, name: str) -> Any:
        return lambda **_: None


@pytest.fixture
def telemetry(monkeypatch: pytest.MonkeyPatch) -> _RecordingTelemetry:
    recorder = _RecordingTelemetry()
    monkeypatch.setattr(registry_module, "worker_telemetry", lambda: recorder)
    return recorder


def _registry(monkeypatch: pytest.MonkeyPatch, *, host_budget: str = "256MiB") -> ModelRegistry:
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "150MiB")
    configs = load_model_configs(MODELS_DIR)
    registry = ModelRegistry(engine_config=EngineConfig(host_cache_bytes=host_budget, disk_cache_enabled=False))
    registry.add_config(configs["sie-fake"])
    return registry


async def test_eviction_parks_and_the_next_load_wakes_the_same_adapter(
    monkeypatch: pytest.MonkeyPatch, telemetry: _RecordingTelemetry
) -> None:
    registry = _registry(monkeypatch)
    adapter_a = await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")

    # Loading the big fake needs both small ones out of the way: they park.
    await registry.load_async(BIG, device="cpu")
    assert registry.loaded_model_names == [BIG]
    assert registry.parked_model_names == [SMALL_A, SMALL_B]
    assert registry.is_parked(SMALL_A)
    assert registry.host_cache.used_bytes == 128 * MIB
    assert telemetry.host_bytes[SMALL_A] == 64 * MIB

    # Waking small-a evicts (parks) the big one and brings back the very same
    # adapter object, with no new load.
    load_count = len(telemetry.load_completions)
    woken = await registry.load_async(SMALL_A, device="cpu")
    assert woken is adapter_a
    assert registry.is_loaded(SMALL_A)
    assert not registry.is_parked(SMALL_A)
    assert registry.parked_model_names == [SMALL_B, BIG]
    assert len(telemetry.load_completions) == load_count
    assert telemetry.wakes == [(SMALL_A, "success")]
    assert telemetry.lookups[(SMALL_A, True)] == 1
    assert telemetry.lookups[(SMALL_A, False)] == 1
    assert telemetry.host_bytes[SMALL_A] == 0
    assert registry.host_cache.used_bytes == 192 * MIB
    # The woken model serves.
    worker = await registry.start_worker(SMALL_A)
    assert worker.is_running


async def test_budget_drops_the_least_recently_parked(
    monkeypatch: pytest.MonkeyPatch, telemetry: _RecordingTelemetry
) -> None:
    registry = _registry(monkeypatch, host_budget="128MiB")
    await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")
    assert registry.parked_model_names == [SMALL_A, SMALL_B]

    # Parking the big one (128 MiB) takes the whole budget: both small ones go.
    await registry.load_async("sie-fake:chat", device="cpu")
    assert registry.parked_model_names == [BIG]
    assert registry.host_cache.used_bytes == 128 * MIB
    assert telemetry.host_bytes[SMALL_A] == 0
    assert telemetry.host_bytes[SMALL_B] == 0


async def test_a_waking_copy_stays_charged_until_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(monkeypatch, host_budget="128MiB")
    await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")

    # small-a's host copy exists until its restore finishes, so the big model
    # evicted to make room for it does not fit beside small-b and is unloaded.
    await registry.load_async(SMALL_A, device="cpu")
    assert registry.parked_model_names == [SMALL_B]
    assert not registry.is_loaded(BIG)
    assert registry.host_cache.used_bytes == 64 * MIB


async def test_a_model_larger_than_the_budget_is_unloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(monkeypatch, host_budget="100MiB")
    await registry.load_async(BIG, device="cpu")
    await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    assert BIG not in registry.parked_model_names
    assert not registry.is_loaded(BIG)


async def test_disabled_tier_unloads_as_before(monkeypatch: pytest.MonkeyPatch, telemetry: _RecordingTelemetry) -> None:
    registry = _registry(monkeypatch, host_budget="0")
    await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")
    assert registry.parked_model_names == []
    assert not registry.host_cache.enabled
    assert not telemetry.lookups


async def test_a_failed_wake_falls_back_to_a_cold_load(
    monkeypatch: pytest.MonkeyPatch, telemetry: _RecordingTelemetry
) -> None:
    registry = _registry(monkeypatch)
    adapter_a = await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")

    unloaded: list[bool] = []

    def broken_restore() -> int:
        raise RuntimeError("restore exploded")

    monkeypatch.setattr(adapter_a, "restore_from_host", broken_restore)
    original_unload = adapter_a.unload
    monkeypatch.setattr(adapter_a, "unload", lambda: (unloaded.append(True), original_unload()))

    reloaded = await registry.load_async(SMALL_A, device="cpu")
    assert reloaded is not adapter_a
    assert registry.is_loaded(SMALL_A)
    assert unloaded == [True]
    assert telemetry.wakes == [(SMALL_A, "error")]
    assert SMALL_A not in registry.parked_model_names


async def test_a_failed_park_unloads_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(monkeypatch)
    adapter_a = await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")

    def broken_offload() -> HostOffload:
        raise RuntimeError("offload exploded")

    monkeypatch.setattr(adapter_a, "offload_to_host", broken_offload)
    await registry.load_async(BIG, device="cpu")
    assert registry.parked_model_names == [SMALL_B]
    assert not registry.is_loaded(SMALL_A)


async def test_config_change_drops_the_parked_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(monkeypatch)
    await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")
    assert SMALL_A in registry.parked_model_names

    config = load_model_configs(MODELS_DIR)["sie-fake"]
    changed = config.model_copy(update={"max_sequence_length": (config.max_sequence_length or 512) + 1})
    await registry.add_config_async(changed)
    assert registry.parked_model_names == []
    assert registry.host_cache.used_bytes == 0


async def test_manual_unload_of_a_parked_model_frees_it(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(monkeypatch)
    await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")

    await registry.unload_async(SMALL_A)
    assert registry.parked_model_names == [SMALL_B]
    with pytest.raises(KeyError, match="not loaded"):
        await registry.unload_async(SMALL_A)

    # A manual unload of a resident model never parks it.
    await registry.unload_async(BIG)
    assert registry.parked_model_names == [SMALL_B]


async def test_shutdown_drops_every_parked_model(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(monkeypatch)
    await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")
    await registry.unload_all_async()
    assert registry.loaded_model_names == []
    assert registry.parked_model_names == []
    assert registry.host_cache.used_bytes == 0


async def test_parked_engines_holding_device_memory_are_dropped_when_nothing_resident_remains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(monkeypatch)
    adapter_a = await registry.load_async(SMALL_A, device="cpu")
    original_offload = adapter_a.offload_to_host

    def engine_offload() -> HostOffload:
        offload = original_offload()
        return HostOffload(host_bytes=offload.host_bytes, retains_device_memory=True)

    monkeypatch.setattr(adapter_a, "offload_to_host", engine_offload)
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")
    assert registry.host_cache.holding_device_memory("cpu") == [SMALL_A]

    # Pretend the device is still short after every resident model left: the
    # parked engine is the only thing left that holds memory there.
    manager = registry.memory_managers["cpu"]
    calls = {"n": 0}
    original = manager.should_evict_for_load

    def short_until_engine_dropped(required: int | None = None) -> bool:
        calls["n"] += 1
        return SMALL_A in registry.parked_model_names or original(required)

    monkeypatch.setattr(manager, "should_evict_for_load", short_until_engine_dropped)
    await registry.load_async(SMALL_B, device="cpu")
    assert SMALL_A not in registry.parked_model_names
    assert registry.is_loaded(SMALL_B)


async def test_wake_parked_waits_for_the_wake(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(monkeypatch)
    await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")

    assert not await registry.wake_parked("sie-fake:chat", "cpu")  # never parked
    assert await registry.wake_parked(SMALL_A, "cpu")
    assert registry.is_loaded(SMALL_A)
    assert await registry.wake_parked(SMALL_A, "cpu")  # already loaded


async def test_cancelled_wake_before_restore_keeps_the_parked_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(monkeypatch)
    await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")

    blocked = asyncio.Event()

    async def stuck_make_room(name: str, entry: Any) -> Any:
        blocked.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(registry, "_make_room_to_wake", stuck_make_room)
    task = asyncio.create_task(registry.load_async(SMALL_A, device="cpu"))
    await blocked.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert SMALL_A in registry.parked_model_names
    assert registry.host_cache.retained_bytes(SMALL_A) == 0
    assert registry.host_cache.used_bytes == 128 * MIB


async def test_wake_drops_lora_placeholders_left_by_the_park(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry(monkeypatch)
    await registry.load_async(SMALL_A, device="cpu")
    await registry.load_async(SMALL_B, device="cpu")
    await registry.load_async(BIG, device="cpu")

    parked = registry.host_cache.get(SMALL_A)
    assert parked is not None
    parked.loaded.loras["in-flight"] = LoadedLora(adapter_id="in-flight", loading=True)
    parked.loaded.loras["victim"] = LoadedLora(adapter_id="victim", unloading=True)
    parked.loaded.loras["resident"] = LoadedLora(adapter_id="resident")

    await registry.load_async(SMALL_A, device="cpu")
    loras = registry._loaded[SMALL_A].loras
    assert list(loras) == ["victim", "resident"]
    assert not loras["victim"].unloading
