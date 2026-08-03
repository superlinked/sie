"""Fake Engine regression tests (#1850): registry race characterization.

Three of the six locked scenarios, driven entirely by sie-fake models with
the synthetic memory budget (#1848) and latch/hang faults (#1849) — no
sleeps as synchronization, no mocks at the decision seams.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from sie_server.core.loader import load_model_configs
from sie_server.core.memory import SIE_FAKE_MEMORY_BUDGET_ENV
from sie_server.core.registry import ModelRegistry
from sie_server.core.residency import EvictionResult

pytestmark = pytest.mark.fake_stack

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MIB = 1024**2


def _fake_registry() -> ModelRegistry:
    """Registry with the whole fake family: adding the BASE ``sie-fake``
    config expands and registers every ``sie-fake:<profile>`` variant
    (adding a variant config alone does not register it).
    """
    configs = load_model_configs(MODELS_DIR)
    registry = ModelRegistry()
    registry.add_config(configs["sie-fake"])
    return registry


async def _wait_until(predicate, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            msg = "condition not reached within timeout"
            raise TimeoutError(msg)
        await asyncio.sleep(0.01)


# -- Scenario: evict during load (latch-sequenced) -------------------------------


async def test_evict_during_load_returns_lock_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Characterizes the drain/load-under-lock stall (registry.py:1668-1678):
    while a load holds the global load-lock, a concurrent eviction attempt
    reports LOCK_TIMEOUT rather than deadlocking, and completes once the
    load finishes.
    """
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "1GiB")
    latch = tmp_path / "release-load"
    monkeypatch.setenv(
        "SIE_FAKE_FAULTS",
        f'{{"sie-fake:small-b": {{"load_latch_file": "{latch}", "latch_timeout_s": 30}}}}',
    )
    registry = _fake_registry()
    await registry.load_async("sie-fake:small-a", device="cpu")

    load_task = asyncio.create_task(registry.load_async("sie-fake:small-b", device="cpu"))
    # Deterministic sequencing: wait until the in-flight load holds the lock.
    # Private-state reach is deliberate — the registry exposes no public
    # "load in flight" observation seam, and polling the lock is the only
    # race-free way to sequence this characterization.
    await _wait_until(lambda: registry._get_load_lock().locked())

    # The race: eviction requested while the load is pinned mid-flight.
    result = await registry.evict_lru_excluding("sie-fake:small-b", timeout_s=0.2)
    assert result is EvictionResult.LOCK_TIMEOUT

    latch.touch()
    await load_task
    assert set(registry.memory_manager.loaded_models) == {"sie-fake:small-a", "sie-fake:small-b"}

    # After the load releases the lock the same eviction succeeds.
    result = await registry.evict_lru_excluding("sie-fake:small-b", timeout_s=5.0)
    assert result is EvictionResult.EVICTED
    assert registry.memory_manager.loaded_models == ["sie-fake:small-b"]


# -- Scenario: concurrent cross-model load under pressure ------------------------


async def test_concurrent_cross_model_load_under_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three models race to load into a 150 MiB budget (64+64+128 MiB
    declared). The global load-lock serializes them FIFO, and the pre-load
    eviction loop must keep the declared usage within budget at every step —
    the last loader evicts both predecessors. Only same-model dedupe had
    coverage before (test_registry_async.py:126); this is the cross-model
    interleaving.
    """
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "150MiB")
    registry = _fake_registry()

    await asyncio.gather(
        registry.load_async("sie-fake:small-a", device="cpu"),
        registry.load_async("sie-fake:small-b", device="cpu"),
        registry.load_async("sie-fake", device="cpu"),
    )

    manager = registry.memory_manager
    # Declared usage never exceeds the budget once the dust settles.
    assert manager.get_stats().used_bytes <= 150 * MIB
    # FIFO lock ordering makes the outcome exact: generate (128 MiB) loads
    # last and evicts both 64 MiB predecessors.
    assert manager.loaded_models == ["sie-fake"]
    assert registry.get("sie-fake") is not None


# -- Scenario: teardown hang ------------------------------------------------------


async def test_teardown_hang_starves_event_loop_but_leaves_no_ghost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterizes the unload/teardown seam (#1600 ghost-model class,
    registry.py:1280-1310). Pinned current contract, measured with a real
    hung ``unload()``: ``adapter.unload()`` runs synchronously ON the event
    loop inside ``_do_unload``, so a hung teardown starves the ENTIRE loop
    for the hang duration — no other request, health probe, or residency op
    can run. Pinned so a future fix (unload in a thread) flips this
    assertion consciously. Post-conditions: the eviction completes, the
    model is fully unregistered (no ghost accounting), and the registry
    accepts new loads immediately.
    """
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "1GiB")
    monkeypatch.setenv(
        "SIE_FAKE_FAULTS",
        '{"sie-fake": {"teardown_hang_s": 1.0}}',
    )
    registry = _fake_registry()
    await registry.load_async("sie-fake", device="cpu")
    await registry.load_async("sie-fake:small-a", device="cpu")

    # Heartbeat task: measures the longest gap between loop iterations while
    # the eviction runs. A blocked loop shows up as one giant gap.
    max_gap = 0.0

    async def _heartbeat() -> None:
        nonlocal max_gap
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.01)
            now = time.monotonic()
            max_gap = max(max_gap, now - last)
            last = now

    beat = asyncio.create_task(_heartbeat())
    await asyncio.sleep(0.05)  # let the heartbeat establish a baseline

    # Evicting from embed's perspective selects generate (the LRU) — whose
    # teardown hangs 1 s, synchronously, on the loop.
    start = time.monotonic()
    result = await registry.evict_lru_excluding("sie-fake:small-a", timeout_s=5.0)
    elapsed = time.monotonic() - start
    # Give the heartbeat one post-hang wakeup so the starvation gap is
    # recorded before we cancel it (the main coroutine resumes first).
    await asyncio.sleep(0.05)
    beat.cancel()

    assert result is EvictionResult.EVICTED
    assert elapsed >= 1.0, "the teardown hang must actually be exercised"
    assert max_gap >= 0.9, "characterization: a hung sync unload starves the event loop today"

    manager = registry.memory_manager
    # No ghost: the hung teardown still unregisters its memory accounting.
    assert manager.loaded_models == ["sie-fake:small-a"]
    assert manager.get_stats().used_bytes == 64 * MIB
    with pytest.raises(KeyError, match="not loaded"):
        registry.get("sie-fake")
    # And the registry accepts new residency work immediately afterwards.
    await registry.load_async("sie-fake", device="cpu")
    assert set(manager.loaded_models) == {"sie-fake:small-a", "sie-fake"}
