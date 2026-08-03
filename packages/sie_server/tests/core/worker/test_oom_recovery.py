from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import sie_server.core.worker.oom_recovery as oom_recovery_module
from sie_server.core.oom import (
    OomRecoveryAction,
    OomRecoveryConfig,
    OomRecoveryStats,
    ResourceExhaustedError,
)
from sie_server.core.residency import EvictionResult
from sie_server.core.worker.oom_recovery import BatchExecutor, ConfigGroup
from sie_server.types.inputs import InvalidInputError

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
    return cast("ConfigGroup", (items, metas, indices, prepared)), metas


def _oom() -> RuntimeError:
    return RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")


def _capture_recovery_events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    telemetry = SimpleNamespace(oom_recovery_completed=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(oom_recovery_module, "worker_telemetry", lambda: telemetry)
    return calls


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
async def test_managed_oom_metric_records_bounded_strategy_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_recovery_events(monkeypatch)
    executor = BatchExecutor(
        model_name="m",
        registry=None,
        config=OomRecoveryConfig(strategy=(OomRecoveryAction.CACHE_CLEAR,)),
        stats=OomRecoveryStats(),
    )
    group, _ = _make_group(1)
    attempts = 0

    async def dispatch(_handler: Any, _group: ConfigGroup) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _oom()
        return "ok"

    await executor.run(_FakeHandler(), group, dispatch)

    assert calls == [{"model": "m", "strategy": "cache_clear", "outcome": "success"}]


@pytest.mark.asyncio
async def test_managed_oom_final_attempt_records_terminal_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_recovery_events(monkeypatch)
    executor = BatchExecutor(
        model_name="m",
        registry=None,
        config=OomRecoveryConfig(strategy=(OomRecoveryAction.CACHE_CLEAR,)),
        stats=OomRecoveryStats(),
    )
    group, metas = _make_group(1)

    async def dispatch(_handler: Any, _group: ConfigGroup) -> str:
        raise _oom()

    await executor.run(_FakeHandler(), group, dispatch)

    assert calls == [{"model": "m", "strategy": "cache_clear", "outcome": "terminal"}]
    assert isinstance(metas[0].future.exception(), ResourceExhaustedError)


@pytest.mark.asyncio
async def test_managed_oom_records_one_outcome_per_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_recovery_events(monkeypatch)
    registry = AsyncMock()
    registry.evict_lru_excluding = AsyncMock(return_value=EvictionResult.EVICTED)
    executor = BatchExecutor(
        model_name="m",
        registry=registry,
        config=OomRecoveryConfig(
            strategy=(OomRecoveryAction.CACHE_CLEAR, OomRecoveryAction.EVICT_LRU),
        ),
        stats=OomRecoveryStats(),
    )
    group, _ = _make_group(1)
    attempts = 0

    async def dispatch(_handler: Any, _group: ConfigGroup) -> str:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise _oom()
        return "ok"

    await executor.run(_FakeHandler(), group, dispatch)

    assert calls == [
        {"model": "m", "strategy": "cache_clear", "outcome": "failed"},
        {"model": "m", "strategy": "evict_lru", "outcome": "success"},
    ]


@pytest.mark.asyncio
async def test_eviction_path() -> None:
    """cache_clear fails; evict_lru frees a sibling; retry succeeds."""
    config = OomRecoveryConfig(
        strategy=(OomRecoveryAction.CACHE_CLEAR, OomRecoveryAction.EVICT_LRU),
    )
    stats = OomRecoveryStats()

    registry = AsyncMock()
    registry.evict_lru_excluding = AsyncMock(return_value=EvictionResult.EVICTED)

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
    registry.evict_lru_excluding = AsyncMock(return_value=EvictionResult.NO_CANDIDATE)

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
async def test_invalid_input_isolates_fused_requests_when_oom_recovery_disabled() -> None:
    config = OomRecoveryConfig(enabled=False)
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    bad = _FakeMetadata()
    good = _FakeMetadata()
    group = cast("ConfigGroup", (["bad", "good"], [bad, good], [0, 0], [0, 1]))
    handler = _FakeHandler()
    sentinel = InvalidInputError("blank reranker candidate")
    seen_sizes: list[int] = []

    async def dispatch(_h: Any, candidate_group: ConfigGroup) -> str:
        seen_sizes.append(len(candidate_group[0]))
        if "bad" in candidate_group[0]:
            raise sentinel
        return "ok"

    await executor.run(handler, group, dispatch)

    assert seen_sizes == [2, 1, 1]
    assert bad.future.exception() is sentinel
    assert good._partial_results == {0: ("sliced", "ok", 0)}
    assert not good.future.done()
    assert stats.recoveries_attempted == 0
    assert stats.batch_splits == 0


@pytest.mark.asyncio
async def test_invalid_input_keeps_multi_item_request_atomic() -> None:
    config = OomRecoveryConfig(enabled=False)
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    bad = _FakeMetadata(item_count=2)
    good = _FakeMetadata()
    group = cast(
        "ConfigGroup",
        (
            ["bad-0", "bad-1", "good"],
            [bad, bad, good],
            [0, 1, 0],
            [0, 1, 2],
        ),
    )
    handler = _FakeHandler()
    sentinel = InvalidInputError("malformed image")
    seen_groups: list[list[object]] = []

    async def dispatch(_h: Any, candidate_group: ConfigGroup) -> str:
        seen_groups.append(list(candidate_group[0]))
        if bad in candidate_group[1]:
            raise sentinel
        return "ok"

    await executor.run(handler, group, dispatch)

    assert seen_groups == [["bad-0", "bad-1", "good"], ["bad-0", "bad-1"], ["good"]]
    assert bad.future.exception() is sentinel
    assert good._partial_results == {0: ("sliced", "ok", 0)}
    assert not good.future.done()


@pytest.mark.asyncio
async def test_partial_split_bumps_both_succeeded_and_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A partial-success split bumps recoveries_succeeded AND terminal_failures.

    Regression guard for the metric semantics fix: half the items succeed
    via halving, half terminally fail — operator dashboards should see
    both signals so neither "did recovery work" nor "are we losing
    requests" is silently zero.
    """
    config = OomRecoveryConfig(strategy=(OomRecoveryAction.SPLIT_BATCH,), max_split_depth=2)
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)
    managed_calls = _capture_recovery_events(monkeypatch)

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
    # The managed per-attempt outcome is exclusive even though the legacy
    # component counters above intentionally retain both facts.
    assert managed_calls == [{"model": "m", "strategy": "split_batch", "outcome": "terminal"}]


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


# --------------------------------------------------------------------------
# OOM traceback release (#2144)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oom_traceback_released_when_recovery_disabled() -> None:
    """Disabled-recovery branch clears the OOM traceback before _fail_group.

    Future.set_exception preserves the traceback, whose frame locals pin the
    failed forward's batch tensors and activations (#2144).
    """
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
        assert err.__traceback__ is None


@pytest.mark.asyncio
async def test_non_oom_traceback_preserved_when_recovery_disabled() -> None:
    """Non-OOM errors keep their traceback for debugging."""
    config = OomRecoveryConfig(enabled=False)
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    group, metas = _make_group(1)
    handler = _FakeHandler()

    async def dispatch(_h: Any, _g: ConfigGroup) -> str:
        raise ValueError("not an OOM")

    await executor.run(handler, group, dispatch)

    err = metas[0].future.exception()
    assert isinstance(err, ValueError)
    assert err.__traceback__ is not None


@pytest.mark.asyncio
async def test_oom_tracebacks_released_across_enabled_recovery() -> None:
    """Every OOM raised through the enabled strategy loop is released."""
    config = OomRecoveryConfig(strategy=(OomRecoveryAction.CACHE_CLEAR,))
    stats = OomRecoveryStats()
    executor = BatchExecutor(model_name="m", registry=None, config=config, stats=stats)

    group, _metas = _make_group(2)
    handler = _FakeHandler()

    raised: list[BaseException] = []

    async def dispatch(_h: Any, _g: ConfigGroup) -> str:
        e = _oom()
        raised.append(e)
        raise e

    await executor.run(handler, group, dispatch)

    # Original attempt + cache_clear retry, both OOM'd and both released.
    assert len(raised) == 2
    assert all(e.__traceback__ is None for e in raised)
