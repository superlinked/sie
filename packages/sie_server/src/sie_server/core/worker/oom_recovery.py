from __future__ import annotations

import asyncio
import gc
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

import torch

from sie_server.core.oom import (
    OomRecoveryAction,
    OomRecoveryConfig,
    OomRecoveryStats,
    ResourceExhausted,
    ResourceExhaustedError,
    is_oom_error,
)
from sie_server.observability.metrics import record_oom_recovery_event

if TYPE_CHECKING:
    from sie_server.core.batcher import HasCost
    from sie_server.core.worker.handlers import OperationHandler
    from sie_server.core.worker.types import RequestMetadata
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)


# Type alias for the four parallel lists produced by
# ``ModelWorker._group_by_inference_config`` for one config-group.
ConfigGroup = tuple[
    list["Item"],
    list["RequestMetadata"],
    list[int],
    list["HasCost"],
]


class RegistryCallbacks(Protocol):
    """Slim protocol the worker uses to talk to the registry during recovery.

    Decoupled from ``ModelRegistry`` to avoid a cyclic import (registry imports
    worker; worker must not import registry). The registry hands an
    implementation to each worker at construction time.
    """

    async def evict_lru_excluding(self, exclude_name: str, *, timeout_s: float) -> bool:
        """Evict the least-recently-used model other than ``exclude_name``.

        Returns True if a model was actually unloaded, False if there was
        no eligible candidate or the lock could not be acquired in time.
        Implementations are expected to acquire the registry's load-lock
        with the given soft timeout.
        """
        ...


# Type for the dispatch callable injected by the worker for a single config
# group. Returns the typed adapter output for the supplied batch slice. The
# callable is expected to capture per-group context (config_key, active LoRA)
# in its closure so the executor remains operation-agnostic.
DispatchFn = Callable[["OperationHandler[Any]", ConfigGroup], Awaitable[Any]]


class BatchExecutor:
    """Run a single config-group with reactive OOM recovery.

    One ``BatchExecutor`` is created per ``ModelWorker`` and reused across
    every batch the worker processes. State is intentionally minimal: the
    executor is a strategy applier, not a coordinator.

    The dispatch callable is supplied per-group rather than at construction
    time. The worker captures the config_key in a closure, which lets the
    executor remain ignorant of operation parameters while still allowing
    the recursive split path to re-invoke dispatch on sub-batches.
    """

    # Strategy values whose per-action metric branch exists in
    # ``record_oom_recovery_event``. Kept in sync with that function's
    # if/elif ladder; a typo or new ``OomRecoveryAction`` member without a
    # matching metric would otherwise surface only mid-incident as a
    # ``ValueError`` from the metrics layer.
    _RECOGNISED_ACTIONS: frozenset[OomRecoveryAction] = frozenset(
        {
            OomRecoveryAction.CACHE_CLEAR,
            OomRecoveryAction.EVICT_LRU,
            OomRecoveryAction.SPLIT_BATCH,
        }
    )

    def __init__(
        self,
        *,
        model_name: str,
        registry: RegistryCallbacks | None,
        config: OomRecoveryConfig,
        stats: OomRecoveryStats,
    ) -> None:
        """Initialise the executor.

        Args:
            model_name: Name of the worker's model — passed to
                ``evict_lru_excluding`` so the worker never evicts itself.
            registry: Callbacks for cross-model operations. May be None in
                tests / standalone workers; in that case ``EVICT_LRU`` is a
                no-op (treated as "no candidate").
            config: Recovery configuration.
            stats: Mutable counters; updated in-place as recovery proceeds.

        Raises:
            ValueError: If ``config.strategy`` contains an action that has
                no matching per-action metric in
                ``record_oom_recovery_event``. Failing fast at construction
                (worker boot) is preferable to raising during a real OOM.
        """
        unknown = [a for a in config.strategy if a not in self._RECOGNISED_ACTIONS]
        if unknown:
            msg = (
                f"BatchExecutor: unsupported recovery action(s) {unknown!r} for model {model_name!r}; "
                f"expected one of {sorted(a.value for a in self._RECOGNISED_ACTIONS)}"
            )
            raise ValueError(msg)
        self._model_name = model_name
        self._registry = registry
        self._config = config
        self._stats = stats

    async def run(
        self,
        handler: OperationHandler[Any],
        group: ConfigGroup,
        dispatch: DispatchFn,
    ) -> None:
        """Run one config-group, applying recovery on OOM.

        On success: populates ``metadata._partial_results`` for each request
        as the in-line code did before. On non-OOM exception: that exception
        is set on every future in the group (preserves prior behaviour). On
        OOM with recovery exhausted: ``ResourceExhaustedError`` is set on
        every still-unfilled future in the group.
        """
        # Fast path: no recovery configured. Behave exactly like the
        # pre-existing in-line ``try/except``.
        if not self._config.enabled:
            try:
                await self._dispatch_and_fan_out(handler, group, dispatch)
            except Exception as e:  # noqa: BLE001 — preserve prior catch-all
                self._fail_group(group, e)
            return

        # First attempt — original batch.
        try:
            await self._dispatch_and_fan_out(handler, group, dispatch)
            return
        except Exception as e:  # noqa: BLE001
            if not is_oom_error(e):
                self._fail_group(group, e)
                return
            self._stats.recoveries_attempted += 1
            record_oom_recovery_event(self._model_name, attempted=True)
            logger.warning(
                "OOM during dispatch (model=%s, items=%d): %s",
                self._model_name,
                len(group[1]),
                e,
            )
            last_error: BaseException = e

        # Strategy loop. Each non-split strategy is attempted at most once;
        # ``SPLIT_BATCH`` is recursive and terminal.
        for action in self._config.strategy:
            if action is OomRecoveryAction.CACHE_CLEAR:
                self._cache_clear()
                self._stats.cache_clears += 1
                record_oom_recovery_event(self._model_name, action="cache_clear")

            elif action is OomRecoveryAction.EVICT_LRU:
                evicted = await self._try_evict_lru()
                if not evicted:
                    # No-op for this strategy; don't bother retrying the
                    # dispatch — go to the next strategy.
                    continue
                self._stats.evictions_triggered += 1
                record_oom_recovery_event(self._model_name, action="evict_lru")

            elif action is OomRecoveryAction.SPLIT_BATCH:
                # Recursive divide-and-conquer is terminal: each item either
                # succeeds in some sub-batch or fails individually with
                # ``ResourceExhaustedError``. Partial success is a real
                # outcome — half the items can succeed while the other half
                # terminally fails. We record both signals so dashboards
                # don't silently undercount either:
                #   - any successful recovery → ``recoveries_succeeded += 1``
                #   - any terminal failure → ``terminal_failures += 1``
                # A fully-mixed split bumps BOTH counters once.
                self._stats.batch_splits += 1
                record_oom_recovery_event(self._model_name, action="split_batch")
                succeeded_count, oom_failed_count = await self._run_split(handler, group, dispatch, depth=0)
                if succeeded_count > 0:
                    self._stats.recoveries_succeeded += 1
                    record_oom_recovery_event(self._model_name, succeeded=True)
                if oom_failed_count > 0:
                    self._stats.terminal_failures += 1
                    record_oom_recovery_event(self._model_name, terminal=True)
                return

            else:  # pragma: no cover — exhaustive enum
                msg = f"Unknown recovery action: {action!r}"
                raise RuntimeError(msg)

            # Re-dispatch after the (non-split) mitigation.
            try:
                await self._dispatch_and_fan_out(handler, group, dispatch)
                self._stats.recoveries_succeeded += 1
                record_oom_recovery_event(self._model_name, succeeded=True)
                return
            except Exception as e:  # noqa: BLE001
                if not is_oom_error(e):
                    self._fail_group(group, e)
                    return
                last_error = e
                logger.warning(
                    "OOM persists after %s (model=%s, items=%d): %s",
                    action.value,
                    self._model_name,
                    len(group[1]),
                    e,
                )

        # All strategies exhausted without a SPLIT_BATCH terminal step.
        self._stats.terminal_failures += 1
        record_oom_recovery_event(self._model_name, terminal=True)
        self._fail_group(group, self._wrap_oom(last_error, attempts=len(self._config.strategy)))

    # ------------------------------------------------------------------
    # Recursive split
    # ------------------------------------------------------------------

    async def _run_split(
        self,
        handler: OperationHandler[Any],
        group: ConfigGroup,
        dispatch: DispatchFn,
        depth: int,
    ) -> tuple[int, int]:
        """Halve the batch until each slice fits or single items fail.

        Returns ``(succeeded_count, oom_failed_count)`` — the number of
        distinct metadata objects that ended up with a populated partial
        result vs a terminal *OOM* exception. Non-OOM exceptions raised
        by ``dispatch`` are still set on the slice's futures via
        ``_fail_group``, but they do **not** contribute to
        ``oom_failed_count``: the OOM-recovery counters in
        ``OomRecoveryStats`` are reserved for actual OOM-driven terminal
        failures, so a transient non-OOM bug (bad input shape, kernel
        error) doesn't pollute the operator dashboards. Together,
        ``succeeded_count + oom_failed_count + (non-OOM failures)`` cover
        every distinct metadata in ``group``.
        """
        items, metadata_list, indices, prepared = group
        distinct_count = len({id(m) for m in metadata_list})

        # At depth=0 do a full reclaim once: we just came off a failed
        # EVICT_LRU or initial OOM and Python references may still hold
        # tensors. Deeper levels use the cheap CUDA cache empty between
        # halves — there is no Python-level garbage to collect there.
        if depth == 0:
            self._cache_clear()

        # Cooperative cancellation point: bounded recursion can otherwise
        # process up to 2**max_split_depth sub-batches before yielding.
        await asyncio.sleep(0)

        # Try this slice first; if it succeeds, we're done.
        try:
            await self._dispatch_and_fan_out(handler, group, dispatch)
            return distinct_count, 0
        except Exception as e:  # noqa: BLE001
            if not is_oom_error(e):
                # Propagate the non-OOM error onto the slice's futures
                # (preserves the original cause for callers) but do *not*
                # tag it as a terminal-OOM failure — the OOM stats only
                # track OOM-driven terminations.
                self._fail_group(group, e)
                return 0, 0
            last_error: BaseException = e

        # Cannot split further: we're at one item or at the depth cap.
        if len(metadata_list) <= 1 or depth >= self._config.max_split_depth:
            logger.warning(
                "OOM at minimum batch slice (model=%s, items=%d, depth=%d) — marking as resource-exhausted",
                self._model_name,
                len(metadata_list),
                depth,
            )
            self._fail_group(group, self._wrap_oom(last_error, attempts=depth + 1))
            return 0, distinct_count

        # Halve. The split is on the parallel lists — every list shares the
        # same length and ordering by construction.
        mid = len(metadata_list) // 2
        left: ConfigGroup = (items[:mid], metadata_list[:mid], indices[:mid], prepared[:mid])
        right: ConfigGroup = (items[mid:], metadata_list[mid:], indices[mid:], prepared[mid:])

        # Cheap CUDA cache empty between halves (no GC needed at this depth)
        # — second half starts in the same nominal state as the first.
        self._empty_cuda_cache()
        left_ok, left_oom_fail = await self._run_split(handler, left, dispatch, depth + 1)
        self._empty_cuda_cache()
        right_ok, right_oom_fail = await self._run_split(handler, right, dispatch, depth + 1)
        return left_ok + right_ok, left_oom_fail + right_oom_fail

    # ------------------------------------------------------------------
    # Mitigations
    # ------------------------------------------------------------------

    def _cache_clear(self) -> None:
        """Full reclaim: Python GC + CUDA cache drop.

        Used after eviction-style mitigations (or the initial OOM) where
        Python references may still hold tensor weights. ``gc.collect()``
        is relatively expensive — typically tens of ms on a busy heap —
        so it is *not* used between split halves; see
        :meth:`_empty_cuda_cache` for the cheap variant.

        Both reclaim calls are wrapped in defensive ``try/except`` so a
        failure here cannot leak unset futures: the pre-PR behaviour was
        for an exception in the recovery primitives to escape
        ``BatchExecutor.run`` and leave the per-request futures pending
        until the HTTP-layer timeout. Now we log and continue with the
        next strategy.
        """
        try:
            gc.collect()
        except Exception:
            logger.exception("OOM recovery: gc.collect() raised; continuing")
        self._empty_cuda_cache()

    def _empty_cuda_cache(self) -> None:
        """Drop CUDA's caching allocator's free blocks.

        Cheap (microseconds): does not touch Python objects, only releases
        cached blocks back to the driver. Safe to call between recursive
        split halves where no Python-level garbage was created.

        Defensive: any ``torch.cuda`` error during recovery is logged and
        swallowed so the strategy loop can advance. A genuinely-broken
        CUDA context will be revealed on the next dispatch anyway.
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("OOM recovery: torch.cuda.empty_cache() raised; continuing")

    async def _try_evict_lru(self) -> bool:
        """Ask the registry to evict the LRU model, excluding self.

        Returns True if a model was actually unloaded.
        """
        if self._registry is None:
            return False
        try:
            return await self._registry.evict_lru_excluding(
                self._model_name,
                timeout_s=self._config.eviction_lock_timeout_s,
            )
        except TimeoutError:
            logger.warning(
                "Eviction lock timeout (%.1fs) for model=%s; skipping",
                self._config.eviction_lock_timeout_s,
                self._model_name,
            )
            return False
        except Exception:
            logger.exception("Unexpected error during evict_lru_excluding")
            return False

    # ------------------------------------------------------------------
    # Fan-out and failure helpers
    # ------------------------------------------------------------------

    async def _dispatch_and_fan_out(
        self,
        handler: OperationHandler[Any],
        group: ConfigGroup,
        dispatch: DispatchFn,
    ) -> None:
        """Run one inference attempt and write partial results.

        Mirrors the in-line success branch in ``_process_batch``: produces a
        typed output, slices it per metadata, and appends to
        ``metadata._partial_results``.
        """
        _items, metadata_list, indices, _prepared = group
        output = await dispatch(handler, group)
        for batch_idx, (metadata, original_idx) in enumerate(zip(metadata_list, indices, strict=True)):
            if metadata._partial_results is None:
                metadata._partial_results = {}
            metadata._partial_results[original_idx] = handler.slice_output(output, batch_idx)

    def _fail_group(self, group: ConfigGroup, error: BaseException) -> None:
        """Set ``error`` on every distinct, not-yet-done future in ``group``."""
        _items, metadata_list, _indices, _prepared = group
        seen: set[int] = set()
        for metadata in metadata_list:
            mid = id(metadata)
            if mid in seen:
                continue
            seen.add(mid)
            if not metadata.future.done():
                metadata.future.set_exception(error)

    def _wrap_oom(self, error: BaseException, *, attempts: int) -> ResourceExhaustedError:
        """Wrap an OOM exception in our structured terminal-failure error.

        The wrapper preserves the original message for logging while giving
        the API layer a stable type / marker to key on.
        """
        marker = ResourceExhausted(
            operation="inference",
            attempts=attempts,
            original_message=str(error),
        )
        return ResourceExhaustedError(
            f"Resource exhausted after {attempts} recovery attempts: {error}",
            marker=marker,
        )
