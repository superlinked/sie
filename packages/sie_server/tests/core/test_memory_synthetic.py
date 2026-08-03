"""Synthetic memory-tracker mode (#1848): deterministic pressure/eviction.

With ``SIE_FAKE_MEMORY_BUDGET`` set, the MemoryManager reports
``total = budget`` and ``used = Σ declared estimated_bytes`` — no device or
psutil queries — so the real ``check_pressure`` / ``should_evict_for_load`` /
LRU logic becomes an exact function of what the registry loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sie_server.core.loader import load_model_configs
from sie_server.core.memory import (
    SIE_FAKE_MEMORY_BUDGET_ENV,
    MemoryManager,
    SyntheticMemoryTracker,
    parse_memory_budget,
)
from sie_server.core.registry import ModelRegistry

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

MIB = 1024**2


# -- Budget parsing ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1048576", 1048576),
        ("64MiB", 64 * MIB),
        ("1.5GiB", int(1.5 * 1024**3)),
        ("2GB", 2 * 1000**3),
        (" 512 KiB ", 512 * 1024),
    ],
)
def test_parse_memory_budget(raw: str, expected: int) -> None:
    assert parse_memory_budget(raw) == expected


@pytest.mark.parametrize("raw", ["", "lots", "12XB", "GiB"])
def test_parse_memory_budget_rejects_garbage(raw: str) -> None:
    with pytest.raises(ValueError, match=SIE_FAKE_MEMORY_BUDGET_ENV):
        parse_memory_budget(raw)


def test_synthetic_tracker_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        SyntheticMemoryTracker(0, lambda: 0)


# -- Manager-level behavior ----------------------------------------------------


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> MemoryManager:
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "100MiB")
    return MemoryManager(device="cpu")


def test_synthetic_stats_track_declared_footprints(manager: MemoryManager) -> None:
    stats = manager.get_stats()
    assert stats.device_type == "synthetic"
    assert stats.total_bytes == 100 * MIB
    assert stats.used_bytes == 0

    manager.register_model("a", estimated_bytes=30 * MIB)
    manager.register_model("b", estimated_bytes=40 * MIB)
    assert manager.get_stats().used_bytes == 70 * MIB

    manager.unregister_model("a")
    assert manager.get_stats().used_bytes == 40 * MIB


def test_synthetic_pressure_is_deterministic(manager: MemoryManager) -> None:
    manager.register_model("a", estimated_bytes=90 * MIB)
    assert not manager.check_pressure()  # 90% < 95% threshold
    manager.register_model("b", estimated_bytes=6 * MIB)
    assert manager.check_pressure()  # 96% > 95% threshold


def test_synthetic_should_evict_for_load(manager: MemoryManager) -> None:
    manager.register_model("a", estimated_bytes=60 * MIB)
    assert not manager.should_evict_for_load(30 * MIB)  # 40 MiB available
    assert manager.should_evict_for_load(50 * MIB)  # needs more than available


def test_real_tracker_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SIE_FAKE_MEMORY_BUDGET_ENV, raising=False)
    stats = MemoryManager(device="cpu").get_stats()
    assert stats.device_type == "cpu"


def test_invalid_budget_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "not-a-size")
    with pytest.raises(ValueError, match=SIE_FAKE_MEMORY_BUDGET_ENV):
        MemoryManager(device="cpu")


# -- Acceptance: eviction under pressure, fakes only, real registry path --------


async def test_eviction_under_pressure_with_fakes_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """#1848 acceptance: load fakes past the declared budget and observe LRU
    eviction fire through the real pre-load decision path — deterministically,
    on a CPU runner, zero downloads.

    Budget 150 MiB; embed+rerank declare 64 MiB each (128 total), generate
    declares 128 MiB. Loading generate needs 128 MiB but only 22 MiB is
    available, so the real ``should_evict_for_load`` loop evicts embed then
    rerank (LRU order) before the load proceeds.
    """
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "150MiB")

    configs = load_model_configs(MODELS_DIR)
    registry = ModelRegistry()
    # Adding the BASE config registers every sie-fake:<profile> variant.
    registry.add_config(configs["sie-fake"])

    await registry.load_async("sie-fake:small-a", device="cpu")
    await registry.load_async("sie-fake:small-b", device="cpu")
    manager = registry.memory_manager
    assert set(manager.loaded_models) == {"sie-fake:small-a", "sie-fake:small-b"}
    assert manager.get_stats().used_bytes == 128 * MIB

    await registry.load_async("sie-fake", device="cpu")

    assert manager.loaded_models == ["sie-fake"]
    assert manager.get_stats().used_bytes == 128 * MIB
    assert registry.get("sie-fake") is not None
    with pytest.raises(KeyError, match="not loaded"):
        registry.get("sie-fake:small-a")
    with pytest.raises(KeyError, match="not loaded"):
        registry.get("sie-fake:small-b")
