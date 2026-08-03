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
from sie_server.core.residency import EvictionResult
from sie_server.observability.worker_telemetry import worker_telemetry
from sie_server.types.inputs import InvalidInputError

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

    async def evict_lru_excluding(self, exclude_name: str, *, timeout_s: float) -> EvictionResult:
        """Evict the least-recently-used model other than ``exclude_name``.

        Returns an :class:`EvictionResult` reporting whether a sibling was
        unloaded (``EVICTED``) or why not (``NO_CANDIDATE`` / ``LOCK_TIMEOUT``
        / ``UNLOAD_FAILED``). Implementations acquire the registry's load-lock
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

    # Recovery strategies implemented by this executor. A new enum member must
    # gain an execution branch before it can enter a production strategy list.
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
            ValueError: If ``config.strategy`` contains an action that this
                executor cannot apply. Failing at worker boot is preferable to
                discovering it during an OOM incident.
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
            except InvalidInputError as e:
                await self._isolate_invalid_input(handler, group, dispatch, e)
            except Exception as e:  # noqa: BLE001 — preserve prior catch-all
                if is_oom_error(e):
                    # Future.set_exception preserves the traceback, which pins
                    # the failed forward's tensors (#2144) — release it for
                    # OOMs here too; non-OOM errors keep theirs for debugging.
                    e.__traceback__ = None
                self._fail_group(group, e)
            return

        # First attempt — original batch.
        try:
            await self._dispatch_and_fan_out(handler, group, dispatch)
            return
        except InvalidInputError as e:
            await self._isolate_invalid_input(handler, group, dispatch, e)
            return
        except Exception as e:  # noqa: BLE001
            if not is_oom_error(e):
                self._fail_group(group, e)
                return
            self._stats.recoveries_attempted += 1
            logger.warning(
                "OOM during dispatch (model=%s, items=%d): %s",
                self._model_name,
                len(group[1]),
                e,
            )
            # Release the traceback: it pins every frame of the failed
            # forward — including the batch tensors and model activations as
            # frame locals — for as long as last_error is held, which is why
            # cache_clear reclaimed nothing and items=1 retries still OOM'd
            # (#2144). _wrap_oom only needs str(e).
            e.__traceback__ = None
            last_error: BaseException = e

        # Strategy loop. Each non-split strategy is attempted at most once;
        # ``SPLIT_BATCH`` is recursive and terminal.
        # Keep an OOM retry pending until we know whether another strategy is
        # actually attempted. If there is a successor, the prior attempt is
        # ``failed``; if recovery exhausts here, that same attempt is
        # ``terminal``. This guarantees one managed outcome per attempt.
        pending_oom_strategy: str | None = None
        for action in self._config.strategy:
            if action is OomRecoveryAction.CACHE_CLEAR:
                if pending_oom_strategy is not None:
                    self._record_recovery(pending_oom_strategy, "failed")
                    pending_oom_strategy = None
                self._cache_clear()
                self._stats.cache_clears += 1

            elif action is OomRecoveryAction.EVICT_LRU:
                evicted = await self._try_evict_lru()
                if not evicted:
                    # No-op for this strategy; don't bother retrying the
                    # dispatch — go to the next strategy.
                    continue
                if pending_oom_strategy is not None:
                    self._record_recovery(pending_oom_strategy, "failed")
                    pending_oom_strategy = None
                self._stats.evictions_triggered += 1

            elif action is OomRecoveryAction.SPLIT_BATCH:
                if pending_oom_strategy is not None:
                    self._record_recovery(pending_oom_strategy, "failed")
                    pending_oom_strategy = None
                # Recursive divide-and-conquer is terminal: each item either
                # succeeds in some sub-batch or fails individually with
                # ``ResourceExhaustedError``. Partial success is a real
                # outcome — half the items can succeed while the other half
                # terminally fails. In-memory stats retain both facts; the
                # contract event is exclusive and gives terminal precedence.
                self._stats.batch_splits += 1
                succeeded_count, oom_failed_count = await self._run_split(handler, group, dispatch, depth=0)
                if succeeded_count > 0:
                    self._stats.recoveries_succeeded += 1
                if oom_failed_count > 0:
                    self._stats.terminal_failures += 1
                managed_outcome = "terminal" if oom_failed_count > 0 else "success" if succeeded_count > 0 else "failed"
                self._record_recovery(action.value, managed_outcome)
                return

            else:  # pragma: no cover — exhaustive enum
                msg = f"Unknown recovery action: {action!r}"
                raise RuntimeError(msg)

            # Re-dispatch after the (non-split) mitigation.
            try:
                await self._dispatch_and_fan_out(handler, group, dispatch)
                self._stats.recoveries_succeeded += 1
                self._record_recovery(action.value, "success")
                return
            except InvalidInputError as e:
                await self._isolate_invalid_input(handler, group, dispatch, e)
                return
            except Exception as e:  # noqa: BLE001
                if not is_oom_error(e):
                    self._record_recovery(action.value, "failed")
                    self._fail_group(group, e)
                    return
                e.__traceback__ = None  # release pinned forward frames (#2144)
                last_error = e
                pending_oom_strategy = action.value
                logger.warning(
                    "OOM persists after %s (model=%s, items=%d): %s",
                    action.value,
                    self._model_name,
                    len(group[1]),
                    e,
                )

        # All strategies exhausted without a SPLIT_BATCH terminal step.
        self._stats.terminal_failures += 1
        self._record_recovery(pending_oom_strategy or "other", "terminal")
        self._fail_group(group, self._wrap_oom(last_error, attempts=len(self._config.strategy)))

    def _record_recovery(self, strategy: str, outcome: str) -> None:
        """Emit one resolved strategy attempt through the semantic facade."""
        worker_telemetry().oom_recovery_completed(
            model=self._model_name,
            strategy=strategy,
            outcome=outcome,
        )

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
        except InvalidInputError as e:
            await self._isolate_invalid_input(handler, group, dispatch, e)
            succeeded = len({id(m) for m in metadata_list if not m.future.done()})
            return succeeded, 0
        except Exception as e:  # noqa: BLE001
            if not is_oom_error(e):
                # Propagate the non-OOM error onto the slice's futures
                # (preserves the original cause for callers) but do *not*
                # tag it as a terminal-OOM failure — the OOM stats only
                # track OOM-driven terminations.
                self._fail_group(group, e)
                return 0, 0
            e.__traceback__ = None  # release pinned forward frames (#2144)
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

    async def _isolate_invalid_input(
        self,
        handler: OperationHandler[Any],
        group: ConfigGroup,
        dispatch: DispatchFn,
        error: InvalidInputError,
    ) -> None:
        """Fail only the request that supplied malformed input.

        Dynamic batches may contain several requests, while a score request
        may contribute several adjacent items. Split by request identity so
        one adapter-level validation error cannot poison valid siblings and a
        multi-item request remains atomic.
        """
        items, metadata_list, indices, prepared = group
        request_ids = list(dict.fromkeys(id(metadata) for metadata in metadata_list))
        if len(request_ids) <= 1:
            self._fail_group(group, error)
            return

        left_ids = set(request_ids[: len(request_ids) // 2])
        left_items: list[Item] = []
        left_metadata: list[RequestMetadata] = []
        left_indices: list[int] = []
        left_prepared: list[HasCost] = []
        right_items: list[Item] = []
        right_metadata: list[RequestMetadata] = []
        right_indices: list[int] = []
        right_prepared: list[HasCost] = []
        for item, metadata, index, prepared_item in zip(
            items,
            metadata_list,
            indices,
            prepared,
            strict=True,
        ):
            if id(metadata) in left_ids:
                left_items.append(item)
                left_metadata.append(metadata)
                left_indices.append(index)
                left_prepared.append(prepared_item)
            else:
                right_items.append(item)
                right_metadata.append(metadata)
                right_indices.append(index)
                right_prepared.append(prepared_item)

        left: ConfigGroup = (left_items, left_metadata, left_indices, left_prepared)
        right: ConfigGroup = (right_items, right_metadata, right_indices, right_prepared)
        await self.run(handler, left, dispatch)
        await self.run(handler, right, dispatch)

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

        Returns True if a sibling model was actually unloaded. The registry
        reports *why* an eviction did not happen via :class:`EvictionResult`;
        we log that reason (so an OOM incident reads as "nothing to evict" vs
        "lock contention") and collapse it to a bool for the recovery
        strategy loop, which today only branches on success.
        """
        if self._registry is None:
            return False
        try:
            result = await self._registry.evict_lru_excluding(
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
        if result is not EvictionResult.EVICTED:
            logger.info(
                "OOM recovery: no eviction for model=%s (%s)",
                self._model_name,
                result.value,
            )
        return result is EvictionResult.EVICTED

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
