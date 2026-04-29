from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sie_server.core.oom import (
    OomRecoveryAction,
    OomRecoveryConfig,
    OomRecoveryStats,
    ResourceExhaustedError,
)
from sie_server.core.worker.oom_recovery import BatchExecutor, ConfigGroup
from sie_server.observability.metrics import (
    OOM_BATCH_SPLITS,
    OOM_CACHE_CLEARS,
    OOM_EVICTIONS_TRIGGERED,
    OOM_RECOVERIES_ATTEMPTED,
    OOM_RECOVERIES_SUCCEEDED,
)

# --------------------------------------------------------------------------
# Test fixtures
# --------------------------------------------------------------------------


class _FakeMetadata:
    """Minimal RequestMetadata stand-in.

    The executor only touches ``future`` and ``_partial_results`` so the
    real dataclass is overkill — keeping this lean clarifies what's tested.
    """

    __slots__ = ("_partial_results", "future", "items")

    def __init__(self, item_count: int = 1) -> None:
        self.future: asyncio.Future[Any] = asyncio.get_event_loop().create_future()
        self._partial_results: dict[int, Any] | None = None
        self.items = list(range(item_count))


class _FakeHandler:
    """Minimal OperationHandler — only ``slice_output`` is exercised."""

    @staticmethod
    def slice_output(output: Any, batch_idx: int) -> Any:
        # Return a sentinel keyed on batch_idx so tests can verify which
        # slice was written to which metadata.
        return ("sliced", output, batch_idx)


def _make_group(size: int) -> tuple[ConfigGroup, list[_FakeMetadata]]:
    """Build a config group of the given size (one metadata per item).

    Returns the group plus the metadata list so tests can assert on
    futures directly.
    """
    metas = [_FakeMetadata() for _ in range(size)]
    items = list(range(size))
    indices = list(range(size))
    prepared = list(range(size))
    return (items, metas, indices, prepared), metas  # type: ignore[return-value]


def _oom() -> RuntimeError:
    return RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


# --------------------------------------------------------------------------
# Strategy-by-strategy behaviour
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_clear_succeeds_on_retry() -> None:
    """First attempt OOMs; cache_clear succeeds → no eviction, batch returns."""
    config = OomRecoveryConfig(strategy=(OomRecoveryAction.CACHE_CLEAR,))
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    group, metas = _make_group(2)
    handler = _FakeHandler()

    call_count = 0

    async def dispatch(_h: Any, _g: ConfigGroup) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _oom()
        return "ok"

    await executor.run(handler, group, dispatch)

    assert call_count == 2
    assert stats.cache_clears == 1
    assert stats.evictions_triggered == 0
    assert stats.batch_splits == 0
    assert stats.terminal_failures == 0
    assert stats.recoveries_succeeded == 1
    # Each metadata got its slice written
    for m in metas:
        assert m._partial_results is not None
        assert ("sliced", "ok", 0) in m._partial_results.values() or len(m._partial_results) == 1


@pytest.mark.asyncio
async def test_eviction_path() -> None:
    """cache_clear fails; evict_lru frees a sibling; retry succeeds."""
    config = OomRecoveryConfig(
        strategy=(OomRecoveryAction.CACHE_CLEAR, OomRecoveryAction.EVICT_LRU),
    )
    stats = OomRecoveryStats()

    registry = AsyncMock()
    registry.evict_lru_excluding = AsyncMock(return_value=True)

    executor = BatchExecutor(model_name="m", registry=registry, config=config, stats=stats)

    group, metas = _make_group(3)
    handler = _FakeHandler()

    call_count = 0

    async def dispatch(_h: Any, _g: ConfigGroup) -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:  # original + after-cache-clear both OOM
            raise _oom()
        return "ok"

    await executor.run(handler, group, dispatch)

    assert call_count == 3
    assert stats.cache_clears == 1
    assert stats.evictions_triggered == 1
    assert stats.recoveries_succeeded == 1
    registry.evict_lru_excluding.assert_awaited_once_with("m", timeout_s=5.0)
    for m in metas:
        assert m._partial_results is not None


@pytest.mark.asyncio
async def test_eviction_no_candidate_falls_through() -> None:
    """Eviction returns False → next strategy is attempted without retry."""
    config = OomRecoveryConfig(
        strategy=(OomRecoveryAction.EVICT_LRU, OomRecoveryAction.CACHE_CLEAR),
    )
    stats = OomRecoveryStats()

    registry = AsyncMock()
    registry.evict_lru_excluding = AsyncMock(return_value=False)

    executor = BatchExecutor(model_name="m", registry=registry, config=config, stats=stats)

    group, _ = _make_group(2)
    handler = _FakeHandler()

    call_count = 0

    async def dispatch(_h: Any, _g: ConfigGroup) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _oom()
        return "ok"

    await executor.run(handler, group, dispatch)

    # Original attempt (1) + evict was a no-op (no extra dispatch) +
    # cache_clear retry (2). The eviction-triggered counter stays at 0.
    assert call_count == 2
    assert stats.evictions_triggered == 0
    assert stats.cache_clears == 1


@pytest.mark.asyncio
async def test_split_batch_halves_until_success() -> None:
    """Batch of 4 OOMs at >1 items, succeeds at size 1.

    Expected dispatch calls (conceptually):
      [1,2,3,4] OOM           -> split
      [1,2]    OOM             -> split
      [1]      OK
      [2]      OK
      [3,4]    OOM             -> split
      [3]      OK
      [4]      OK
    """
    config = OomRecoveryConfig(strategy=(OomRecoveryAction.SPLIT_BATCH,), max_split_depth=4)
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    group, metas = _make_group(4)
    handler = _FakeHandler()

    # Adapter OOMs for any batch with > 1 item.
    async def dispatch(_h: Any, g: ConfigGroup) -> str:
        if len(g[1]) > 1:
            raise _oom()
        return "ok"

    await executor.run(handler, group, dispatch)

    # Every metadata should have a partial result
    for m in metas:
        assert m._partial_results is not None
        assert len(m._partial_results) == 1
    assert stats.batch_splits == 1
    assert stats.terminal_failures == 0


@pytest.mark.asyncio
async def test_terminal_single_item_failure_isolates() -> None:
    """Single-item batch OOMs forever → that future fails, none of the others."""
    config = OomRecoveryConfig(strategy=(OomRecoveryAction.SPLIT_BATCH,), max_split_depth=4)
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    group, metas = _make_group(2)
    handler = _FakeHandler()

    # First metadata always OOMs even at size 1; second succeeds.
    async def dispatch(_h: Any, g: ConfigGroup) -> str:
        if metas[0] in g[1]:
            raise _oom()
        return "ok"

    await executor.run(handler, group, dispatch)

    # First future got ResourceExhaustedError, second got a partial result.
    assert metas[0].future.done()
    err = metas[0].future.exception()
    assert isinstance(err, ResourceExhaustedError)
    assert metas[1]._partial_results is not None
    assert stats.terminal_failures >= 1


@pytest.mark.asyncio
async def test_recovery_disabled_propagates_oom() -> None:
    """``enabled=False`` short-circuits: OOM is set on every future as-is."""
    config = OomRecoveryConfig(enabled=False)
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    group, metas = _make_group(2)
    handler = _FakeHandler()

    async def dispatch(_h: Any, _g: ConfigGroup) -> str:
        raise _oom()

    await executor.run(handler, group, dispatch)

    for m in metas:
        err = m.future.exception()
        assert isinstance(err, RuntimeError)
        # NOT wrapped in ResourceExhaustedError when recovery is off.
        assert not isinstance(err, ResourceExhaustedError)
    assert stats.recoveries_attempted == 0


@pytest.mark.asyncio
async def test_split_depth_limit_caps_recursion() -> None:
    """``max_split_depth=0`` means the first OOM in split is terminal."""
    config = OomRecoveryConfig(strategy=(OomRecoveryAction.SPLIT_BATCH,), max_split_depth=0)
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    group, metas = _make_group(4)
    handler = _FakeHandler()

    async def dispatch(_h: Any, _g: ConfigGroup) -> str:
        raise _oom()

    await executor.run(handler, group, dispatch)

    # All futures fail terminally — depth=0 prevents the halve.
    for m in metas:
        err = m.future.exception()
        assert isinstance(err, ResourceExhaustedError)


@pytest.mark.asyncio
async def test_non_oom_error_propagates_immediately() -> None:
    """A non-OOM exception short-circuits the strategy loop."""
    config = OomRecoveryConfig()
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    group, metas = _make_group(3)
    handler = _FakeHandler()

    sentinel = ValueError("invalid input shape")

    call_count = 0

    async def dispatch(_h: Any, _g: ConfigGroup) -> str:
        nonlocal call_count
        call_count += 1
        raise sentinel

    await executor.run(handler, group, dispatch)

    # Exactly one dispatch attempt; recovery did not engage.
    assert call_count == 1
    assert stats.recoveries_attempted == 0
    for m in metas:
        assert m.future.exception() is sentinel


@pytest.mark.asyncio
async def test_prometheus_counters_fire_on_recovery() -> None:
    """Recovery activity bumps the Prometheus counters in lockstep with stats.

    Without this, dashboards built on ``sie_oom_recoveries_*`` would
    silently undercount (the in-memory ``OomRecoveryStats`` would still
    be correct, but operators rely on the prom metric).
    """
    config = OomRecoveryConfig(strategy=(OomRecoveryAction.CACHE_CLEAR,))
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="prom-test", registry=None, config=config, stats=stats)

    group, _ = _make_group(2)
    handler = _FakeHandler()

    call_count = 0

    async def dispatch(_h: Any, _g: ConfigGroup) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _oom()
        return "ok"

    # Snapshot before/after — Counter doesn't expose .get() directly; use
    # ``_value.get()`` on the labelled child. ``labels(...).inc()`` bumps it.
    attempted_before = OOM_RECOVERIES_ATTEMPTED.labels(model="prom-test")._value.get()
    succeeded_before = OOM_RECOVERIES_SUCCEEDED.labels(model="prom-test")._value.get()
    cache_before = OOM_CACHE_CLEARS.labels(model="prom-test")._value.get()
    splits_before = OOM_BATCH_SPLITS.labels(model="prom-test")._value.get()

    await executor.run(handler, group, dispatch)

    assert OOM_RECOVERIES_ATTEMPTED.labels(model="prom-test")._value.get() == attempted_before + 1
    assert OOM_RECOVERIES_SUCCEEDED.labels(model="prom-test")._value.get() == succeeded_before + 1
    assert OOM_CACHE_CLEARS.labels(model="prom-test")._value.get() == cache_before + 1
    # No split happened — counter must be unchanged.
    assert OOM_BATCH_SPLITS.labels(model="prom-test")._value.get() == splits_before


@pytest.mark.asyncio
async def test_prometheus_counters_fire_for_evict_and_split_actions() -> None:
    """Per-strategy counters bump for evict_lru and split_batch.

    Companion to ``test_prometheus_counters_fire_on_recovery`` (which only
    covers cache_clear). Verifies ``record_oom_recovery_event(action=...)``
    routes to the right counter for every strategy.
    """
    # Force an OOM that walks evict_lru → split_batch.
    config = OomRecoveryConfig(
        strategy=(OomRecoveryAction.EVICT_LRU, OomRecoveryAction.SPLIT_BATCH),
        max_split_depth=2,
    )
    stats = OomRecoveryStats()
    registry = AsyncMock()
    registry.evict_lru_excluding = AsyncMock(return_value=True)
    executor = BatchExecutor(model_name="prom-evict-split", registry=registry, config=config, stats=stats)

    group, _ = _make_group(2)
    handler = _FakeHandler()

    call_count = 0

    async def dispatch(_h: Any, g: ConfigGroup) -> str:
        # OOM persists through evict; then split halves to size 1 and succeeds.
        nonlocal call_count
        call_count += 1
        if len(g[1]) > 1:
            raise _oom()
        return "ok"

    evict_before = OOM_EVICTIONS_TRIGGERED.labels(model="prom-evict-split")._value.get()
    split_before = OOM_BATCH_SPLITS.labels(model="prom-evict-split")._value.get()

    await executor.run(handler, group, dispatch)

    assert OOM_EVICTIONS_TRIGGERED.labels(model="prom-evict-split")._value.get() == evict_before + 1
    assert OOM_BATCH_SPLITS.labels(model="prom-evict-split")._value.get() == split_before + 1


@pytest.mark.asyncio
async def test_partial_split_bumps_both_succeeded_and_terminal() -> None:
    """A partial-success split bumps recoveries_succeeded AND terminal_failures.

    Regression guard for the metric semantics fix: half the items succeed
    via halving, half terminally fail — operator dashboards should see
    both signals so neither "did recovery work" nor "are we losing
    requests" is silently zero.
    """
    config = OomRecoveryConfig(strategy=(OomRecoveryAction.SPLIT_BATCH,), max_split_depth=2)
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    group, metas = _make_group(2)
    handler = _FakeHandler()

    # First metadata always OOMs even at size 1; second succeeds at size 1.
    async def dispatch(_h: Any, g: ConfigGroup) -> str:
        if metas[0] in g[1]:
            raise _oom()
        return "ok"

    await executor.run(handler, group, dispatch)

    # Counter contract: both bump exactly once.
    assert stats.recoveries_succeeded == 1
    assert stats.terminal_failures == 1
    # And the per-future state matches: meta[0] failed, meta[1] succeeded.
    assert metas[0].future.done()
    assert metas[0].future.exception() is not None
    assert metas[1]._partial_results is not None


@pytest.mark.asyncio
async def test_non_oom_exception_mid_split_propagates() -> None:
    """A non-OOM error raised by a halved sub-batch is set on its futures.

    The split-recovery loop must not swallow non-OOM errors as if they
    were terminal OOMs — the upstream cause is otherwise lost.
    """
    config = OomRecoveryConfig(strategy=(OomRecoveryAction.SPLIT_BATCH,), max_split_depth=2)
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    group, metas = _make_group(4)
    handler = _FakeHandler()

    sentinel = ValueError("invalid input shape during split")

    async def dispatch(_h: Any, g: ConfigGroup) -> str:
        if len(g[1]) > 1:
            raise _oom()  # forces a split
        # At size 1, the first-item slice OOMed in the original; this slice
        # is the second half — raise a non-OOM error to test propagation.
        if metas[2] in g[1]:
            raise sentinel
        return "ok"

    await executor.run(handler, group, dispatch)

    # The non-OOM error reaches that slice's metadata exactly.
    assert metas[2].future.done()
    err = metas[2].future.exception()
    assert err is sentinel, f"expected sentinel, got {err!r}"

    # Regression: a non-OOM mid-split error must not bump the executor's
    # terminal-OOM counter. Only the OOM that started the split engaged
    # recovery; the propagated ValueError is not an OOM and should leave
    # ``terminal_failures`` and ``failed_count``-style counters untouched.
    assert stats.terminal_failures == 0


@pytest.mark.asyncio
async def test_eviction_lock_timeout_is_swallowed() -> None:
    """If the registry raises TimeoutError, the executor moves on."""
    config = OomRecoveryConfig(
        strategy=(OomRecoveryAction.EVICT_LRU, OomRecoveryAction.CACHE_CLEAR),
    )
    stats = OomRecoveryStats()

    registry = AsyncMock()
    registry.evict_lru_excluding = AsyncMock(side_effect=TimeoutError())

    executor = BatchExecutor(model_name="m", registry=registry, config=config, stats=stats)

    group, _ = _make_group(2)
    handler = _FakeHandler()

    call_count = 0

    async def dispatch(_h: Any, _g: ConfigGroup) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _oom()
        return "ok"

    await executor.run(handler, group, dispatch)

    # Dispatch ran twice: original (OOM) + after cache_clear (success).
    # Eviction-triggered stays 0 because the timeout meant no actual evict.
    assert call_count == 2
    assert stats.evictions_triggered == 0
    assert stats.cache_clears == 1
