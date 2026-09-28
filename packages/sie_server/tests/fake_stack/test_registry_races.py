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


async def test_evict_during_load_does_not_wait_for_the_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An eviction requested while another model's load is pinned mid-flight
    completes at once.

    **A conscious flip.** This test used to characterize the opposite: the
    load held the registry load lock across the adapter load, so the same
    eviction reported LOCK_TIMEOUT until the load finished. The load now
    holds only the admission lock there, and the load lock guards short
    decisions, so an OOM-recovery eviction no longer waits out a cold load.
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
    # Deterministic sequencing: wait until the in-flight load is admitted.
    # Private-state reach is deliberate — the registry exposes no public
    # "load in flight" observation seam, and polling the lock is the only
    # race-free way to sequence this characterization.
    await _wait_until(lambda: registry._get_load_admission_lock().locked())

    # The race: eviction requested while the load is pinned mid-flight.
    result = await registry.evict_lru_excluding("sie-fake:small-b", timeout_s=5.0)
    assert result is EvictionResult.EVICTED
    assert not load_task.done()
    assert registry.memory_manager.loaded_models == []

    latch.touch()
    await load_task
    assert registry.memory_manager.loaded_models == ["sie-fake:small-b"]


# -- Scenario: concurrent cross-model load under pressure ------------------------


async def test_concurrent_cross_model_load_under_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three models race to load into a 150 MiB budget (64+64+128 MiB
    declared). The load admission lock serializes them FIFO, and the pre-load
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


async def test_teardown_hang_does_not_starve_event_loop_and_leaves_no_ghost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterizes the unload/teardown seam (#1600 ghost-model class,
    registry.py:1280-1310), measured with a real hung ``unload()``.

    **This is the conscious flip the previous version of this test asked
    for.** It used to assert the opposite — that a hung teardown starves
    the ENTIRE event loop, because ``adapter.unload()`` ran synchronously
    on it inside ``_do_unload`` — and said so explicitly: "pinned so a
    future fix (unload in a thread) flips this assertion consciously".
    That fix moves teardown onto a worker thread, so the event loop remains
    schedulable while teardown hangs.

    Everything else about the contract is unchanged and still asserted:
    the hang is really exercised (the eviction still takes the full
    teardown duration, because the load lock is deliberately still held
    across it), the eviction completes, the model is fully unregistered
    with no ghost accounting, and the registry accepts new loads
    immediately afterwards.
    """
    monkeypatch.setenv(SIE_FAKE_MEMORY_BUDGET_ENV, "1GiB")
    monkeypatch.setenv(
        "SIE_FAKE_FAULTS",
        '{"sie-fake": {"teardown_hang_s": 1.0}}',
    )
    registry = _fake_registry()
    await registry.load_async("sie-fake", device="cpu")
    await registry.load_async("sie-fake:small-a", device="cpu")

    # Heartbeat task: counts loop iterations completed while the eviction
    # runs. A blocked loop simply stops counting.
    #
    # Counting beats rather than measuring the largest inter-wakeup gap is
    # deliberate. The gap is a max over a noisy sample: a busy CI runner
    # produced 0.61 s on a *healthy* loop here, which is meaningless as a
    # signal but sits well inside any threshold tight enough to catch the
    # 1 s block. The count is cumulative, so one scheduler hiccup barely
    # moves it and the two populations stay decisively apart.
    beats = 0

    async def _heartbeat() -> None:
        nonlocal beats
        while True:
            await asyncio.sleep(0.01)
            beats += 1

    beat = asyncio.create_task(_heartbeat())
    await asyncio.sleep(0.05)  # let the heartbeat establish a baseline
    beats = 0

    # Evicting from embed's perspective selects generate (the LRU) — whose
    # teardown hangs 1 s, now on a worker thread rather than on the loop.
    start = time.monotonic()
    result = await registry.evict_lru_excluding("sie-fake:small-a", timeout_s=5.0)
    elapsed = time.monotonic() - start
    # Cancelled immediately, with no settling sleep: every beat counted
    # below was therefore serviced *during* the hang, which is the whole
    # question. A post-eviction wakeup would credit the blocked case too.
    beat.cancel()

    assert result is EvictionResult.EVICTED
    assert elapsed >= 1.0, "the teardown hang must actually be exercised"
    # The flip. On the loop, the 1 s teardown yields nothing at all and
    # this lands at 0; off the loop the heartbeat keeps its ~10 ms cadence
    # for the whole second and lands near 100. The threshold sits an order
    # of magnitude from both, so a loaded runner cannot flake it.
    assert beats >= 20, f"a hung unload must no longer starve the event loop ({beats} beats during {elapsed:.2f}s)"

    manager = registry.memory_manager
    # No ghost: the hung teardown still unregisters its memory accounting.
    assert manager.loaded_models == ["sie-fake:small-a"]
    assert manager.get_stats().used_bytes == 64 * MIB
    with pytest.raises(KeyError, match="not loaded"):
        registry.get("sie-fake")
    # And the registry accepts new residency work immediately afterwards.
    await registry.load_async("sie-fake", device="cpu")
    assert set(manager.loaded_models) == {"sie-fake:small-a", "sie-fake"}
