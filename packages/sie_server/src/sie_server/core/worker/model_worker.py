"""Model worker for async request handling with dynamic batching.

The ModelWorker manages a single model's inference pipeline:
1. Accepts tokenized requests via submit()
2. Batches requests using BatchFormer
3. Runs inference on batches via operation handlers
4. Fans out results to waiting futures

Architecture:
- ModelWorker: Manages lifecycle, batching, FCFS scheduling, stats
- OperationHandler: Abstract interface for operation-specific logic
- EncodeHandler, ExtractHandler, ScoreHandler: Concrete implementations
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sie_server.core.adaptive_batching import (
    AdaptiveBatchController,
    AdaptiveBatchState,
    BatchEfficiencyTracker,
    LatencyTracker,
)
from sie_server.core.batcher import BatchConfig, BatchFormer, FormattedBatch, HasCost
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.handlers import EncodeHandler, ExtractHandler, OperationHandler, ScoreHandler
from sie_server.core.worker.oom_recovery import BatchExecutor, RegistryCallbacks
from sie_server.core.worker.types import (
    QueueFullError,
    RequestMetadata,
    WorkerConfig,
    WorkerResult,
    WorkerStats,
)
from sie_server.observability.worker_telemetry import worker_telemetry, worker_telemetry_enabled

if TYPE_CHECKING:
    from sie_server.adapters.base import ModelAdapter
    from sie_server.core.postprocessor_registry import PostprocessorRegistry
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PreformedExtractRequest:
    prepared_items: Sequence[HasCost]
    items: list[Item]
    labels: list[str] | None = None
    output_schema: dict[str, Any] | None = None
    instruction: str | None = None
    options: dict[str, Any] | None = None
    request_id: str | None = None
    timing: RequestTiming | None = None


@dataclass(frozen=True, slots=True)
class PreformedScoreRequest:
    prepared_items: Sequence[HasCost]
    query: Item
    items: list[Item]
    instruction: str | None = None
    options: dict[str, Any] | None = None
    request_id: str | None = None
    timing: RequestTiming | None = None


class ModelWorker:
    """Worker that batches and processes inference requests for a single model.

    Thread-safe for async use. Multiple coroutines can submit requests
    concurrently, and they will be batched together for efficient GPU
    utilization.

    Operation-specific logic is delegated to injected handlers:
    - EncodeHandler: Embedding generation
    - ExtractHandler: Entity extraction
    - ScoreHandler: Reranking/scoring

    Usage:
        worker = ModelWorker(adapter, config)
        await worker.start()

        # Submit requests (returns immediately)
        future = await worker.submit(prepared_items, items, output_types)

        # Wait for result
        results = await future

        # Shutdown
        await worker.stop()
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        config: WorkerConfig | None = None,
        *,
        model_name: str | None = None,
        postprocessor_registry: PostprocessorRegistry | None = None,
        handlers: dict[str, OperationHandler[Any]] | None = None,
        registry_callbacks: RegistryCallbacks | None = None,
    ) -> None:
        """Initialize the model worker.

        Args:
            adapter: The model adapter to use for inference.
            config: Worker configuration. Uses defaults if not provided.
            model_name: Name of the model (for postprocessor lookup).
            postprocessor_registry: Registry for postprocessors (optional).
            handlers: Optional dict of operation handlers for dependency injection.
                      Defaults to standard handlers if not provided.
            registry_callbacks: Slim callback protocol used by reactive OOM
                recovery to evict cold models from sibling workers. Optional —
                when None, the EVICT_LRU strategy is a no-op (the rest of
                recovery still works).
        """
        self._adapter = adapter
        self._config = config or WorkerConfig()
        self._model_name = model_name
        self._postprocessor_registry = postprocessor_registry
        self._registry_callbacks = registry_callbacks

        # Protect adapter-global state, especially active LoRA selection,
        # across the direct HTTP batcher and the sidecar IPC pre-formed batch
        # entrypoints. The inference executor serialises GPU work, but LoRA
        # selection happens on the event-loop thread immediately before
        # dispatch, so the set_lora -> forward pair must be atomic.
        self._adapter_dispatch_lock = asyncio.Lock()

        # Initialize operation handlers (dependency injection point)
        if handlers is not None:
            self._handlers: dict[str, OperationHandler[Any]] = handlers
        else:
            self._handlers = {
                "encode": EncodeHandler(model_name, postprocessor_registry),
                "extract": ExtractHandler(),
                "score": ScoreHandler(),
            }

        # Batch config used for all batchers
        self._batch_config = BatchConfig(
            max_batch_tokens=self._config.max_batch_tokens,
            max_batch_requests=self._config.max_batch_requests,
            max_batch_wait_ms=self._config.max_batch_wait_ms,
            coalesce_ms=self._config.coalesce_ms,
            coalesce_ratio=self._config.coalesce_ratio,
        )

        # Per-LoRA batchers: None = base model, "lora-name" = specific LoRA.
        # Each LoRA gets its own batcher for FCFS fairness on the direct
        # Python HTTP path. Sidecar IPC uses submit_preformed* below and
        # bypasses these batchers explicitly.
        self._batchers: dict[str | None, BatchFormer[HasCost, RequestMetadata]] = {
            None: BatchFormer(self._batch_config)
        }

        # Thread pool for running inference (doesn't block event loop)
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,  # Single worker for GPU serialization
            thread_name_prefix="inference",
        )

        # Adaptive batching controller (optional, off by default). It belongs
        # to the direct Python HTTP path; sidecar IPC bypasses the process
        # loop through submit_preformed* and therefore never steps this
        # controller.
        ab = self._config.adaptive_batching
        if ab.enabled:
            self._latency_tracker: LatencyTracker | None = LatencyTracker(
                window_size=ab.window_size,
            )
            self._efficiency_tracker: BatchEfficiencyTracker | None = BatchEfficiencyTracker()
            # min_batch_cost is a *floor* on how small the cost knob can get,
            # not a ceiling. The old value of min(256, max_batch_tokens) was
            # effectively always 256, which lets the PI loop shrink each
            # batch down to a single item whenever p50 stays above target —
            # and under scale-out load (queue latency >> target) p50 *always*
            # stays above target, so cost collapses to 256 and the GPU runs
            # 1-item forwards at ~30k tok/s/pod. Instead, anchor the floor
            # to a quarter of max_batch_tokens so even a fully-collapsed
            # knob still packs ~10 items per forward on the default gte
            # (16384 / 4 = 4096 tokens ≈ 10 items of ~400 tokens each).
            # For tiny max_batch_tokens (<1024), keep the legacy 256 floor
            # so adapters with genuinely small budgets aren't forced above
            # their configured ceiling. See
            # packages/sie_server_sidecar/docs/architecture-guide.md for
            # the related queue-mode regression guard.
            cost_floor = max(256, self._config.max_batch_tokens // 4)
            cost_floor = min(cost_floor, self._config.max_batch_tokens)
            self._adaptive_controller: AdaptiveBatchController | None = AdaptiveBatchController(
                target_p50_ms=ab.target_p50_ms,
                calibration_multiplier=ab.calibration_multiplier,
                min_target_p50_ms=ab.min_target_p50_ms,
                max_target_p50_ms=ab.max_target_p50_ms,
                min_wait_ms=ab.min_wait_ms,
                max_wait_ms=ab.max_wait_ms,
                min_batch_cost=cost_floor,
                max_batch_cost=max(cost_floor, self._config.max_batch_tokens * 4),
                gain=ab.gain,
                integral_gain=ab.integral_gain,
                cost_gain=ab.gain * 0.5,  # cost knob is more conservative
                update_interval=ab.update_interval,
                starvation_recovery_enabled=ab.starvation_recovery_enabled,
                starvation_window=ab.starvation_window,
                starvation_batch_size=ab.starvation_batch_size,
                _current_wait_ms=self._config.max_batch_wait_ms,
                _current_batch_cost=self._config.max_batch_tokens,
            )
        else:
            self._latency_tracker = None
            self._efficiency_tracker = None
            self._adaptive_controller = None

        # Background task and control
        self._running = False
        self._stopping = False  # True when graceful stop has begun
        self._process_task: asyncio.Task[None] | None = None
        self._stats = WorkerStats()

        # Reactive OOM recovery — wraps the per-config-group dispatch. The
        # per-group dispatch closure is built inside ``_process_batch`` (it
        # captures the config_key); the executor itself is constructed once.
        self._batch_executor = BatchExecutor(
            model_name=model_name or "unknown",
            registry=self._registry_callbacks,
            config=self._config.oom_recovery,
            stats=self._stats.oom_recoveries,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def adapter(self) -> ModelAdapter:
        """Return the model adapter."""
        return self._adapter

    @property
    def config(self) -> WorkerConfig:
        """Return the worker configuration."""
        return self._config

    @property
    def stats(self) -> WorkerStats:
        """Return current worker statistics."""
        return self._stats

    @property
    def is_running(self) -> bool:
        """Return True if worker is running."""
        return self._running

    @property
    def pending_count(self) -> int:
        """Return number of pending requests across all batchers."""
        return sum(b.pending_count for b in self._batchers.values())

    def get_adaptive_state(self) -> AdaptiveBatchState | None:
        """Return immutable snapshot of adaptive controller state, or None if disabled."""
        if self._adaptive_controller is None:
            return None
        observed = self._latency_tracker.p50() if self._latency_tracker else None
        fill = self._efficiency_tracker.mean_fill_ratio() if self._efficiency_tracker else None
        return self._adaptive_controller.snapshot(observed_p50_ms=observed, fill_ratio=fill)

    @property
    def pending_tokens(self) -> int:
        """Return total tokens in pending requests across all batchers."""
        return sum(b.pending_tokens for b in self._batchers.values())

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def start(self) -> None:
        """Mark the worker ready to accept requests.

        Should be called before submitting requests.
        Set SIE_INSTRUMENTATION=1 to enable detailed batch statistics.

        The direct HTTP batching loop starts lazily on the first normal
        ``submit*`` call. Sidecar IPC calls use ``submit_preformed*`` and do
        not need a Python batch loop.
        """
        if self._running:
            return

        # Enable instrumentation if configured or env var is set
        env_instrumentation = os.environ.get("SIE_INSTRUMENTATION", "").lower() in ("1", "true", "yes")
        if self._config.instrumentation or env_instrumentation:
            self._stats.enable_instrumentation()
            logger.info("ModelWorker instrumentation enabled")

        self._running = True
        if self._adaptive_controller is not None:
            target_str = (
                f"{self._adaptive_controller.target_p50_ms:.0f}ms"
                if self._adaptive_controller.target_p50_ms is not None
                else "auto-calibrate"
            )
            logger.info(
                "ModelWorker started (adaptive batching: target_p50=%s, gain=%.2f, "
                "integral_gain=%.3f, wait=[%.1f, %.1f]ms, cost=[%d, %d])",
                target_str,
                self._adaptive_controller.gain,
                self._adaptive_controller.integral_gain,
                self._adaptive_controller.min_wait_ms,
                self._adaptive_controller.max_wait_ms,
                self._adaptive_controller.min_batch_cost,
                self._adaptive_controller.max_batch_cost,
            )
        else:
            logger.info("ModelWorker started")

    async def stop(self) -> None:
        """Stop the background processing task.

        Waits for pending batches to complete before returning.
        """
        if not self._running:
            return

        self._running = False

        if self._process_task is not None:
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task
            self._process_task = None

        self._inference_executor.shutdown(wait=True)
        logger.info(
            "ModelWorker stopped (batches=%d, items=%d, tokens=%d)",
            self._stats.batches_processed,
            self._stats.items_processed,
            self._stats.total_tokens_processed,
        )

    # =========================================================================
    # Submit Methods (Public API - unchanged signatures)
    # =========================================================================

    async def submit(
        self,
        prepared_items: Sequence[HasCost],
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        options: dict[str, Any] | None = None,
        request_id: str | None = None,
        timing: RequestTiming | None = None,
    ) -> asyncio.Future[WorkerResult]:
        """Submit prepared items for inference.

        Items are batched with other requests for efficient GPU utilization.
        Returns a future that resolves to a WorkerResult with inference results and timing.

        Args:
            prepared_items: Pre-processed items satisfying HasCost protocol.
            items: Original Item objects (for passing to adapter).
            output_types: Which outputs to return ("dense", "sparse", "multivector").
            instruction: Optional instruction for instruction-tuned models.
            is_query: Whether items are queries (True) or documents (False).
            options: Optional runtime options (e.g., {"muvera": {...}} for postprocessing).
            request_id: Optional request ID for logging/tracing.
            timing: Optional RequestTiming object to track timing for this request.

        Returns:
            Future that resolves to WorkerResult with results and timing.

        Raises:
            RuntimeError: If worker is not running.
            QueueFullError: If queue is full and cannot accept more items.
        """
        self._check_queue_capacity(len(prepared_items))

        future, request_timing = self._create_future_and_timing(timing)

        metadata = RequestMetadata(
            future=future,
            items=items,
            output_types=output_types,
            timing=request_timing,
            instruction=instruction,
            is_query=is_query,
            options=options,
            request_id=request_id,
        )

        lora = options.get("lora") if options else None
        return await self._submit_to_batcher(prepared_items, metadata, lora)

    async def submit_extract(
        self,
        prepared_items: Sequence[HasCost],
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        request_id: str | None = None,
        timing: RequestTiming | None = None,
    ) -> asyncio.Future[WorkerResult]:
        """Submit items for extraction (NER, RE, etc.).

        Items are batched with other extract requests for efficient GPU utilization.
        Items with the same (labels, instruction, options) configuration can batch together.

        Args:
            prepared_items: Pre-processed items with cost for batching (ExtractPreparedItem).
            items: Original Item objects (for passing to adapter).
            labels: Entity types to extract (e.g., ["person", "organization"]).
            output_schema: Optional schema for structured extraction.
            instruction: Optional instruction for instruction-tuned models.
            options: Adapter options to override model config defaults.
            request_id: Optional request ID for logging/tracing.
            timing: Optional RequestTiming object to track timing for this request.

        Returns:
            Future that resolves to WorkerResult with extraction results and timing.

        Raises:
            RuntimeError: If worker is not running.
            QueueFullError: If queue is full and cannot accept more items.
        """
        self._check_queue_capacity(len(prepared_items))

        future, request_timing = self._create_future_and_timing(timing)

        metadata = RequestMetadata(
            future=future,
            items=items,
            timing=request_timing,
            request_id=request_id,
            operation="extract",
            labels=labels,
            output_schema=output_schema,
            instruction=instruction,
            options=options,
        )

        lora = options.get("lora") if options else None
        return await self._submit_to_batcher(prepared_items, metadata, lora)

    async def submit_score(
        self,
        prepared_items: Sequence[HasCost],
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        request_id: str | None = None,
        timing: RequestTiming | None = None,
    ) -> asyncio.Future[WorkerResult]:
        """Submit items for scoring (reranking) against a query.

        Items are batched with other score requests for efficient GPU utilization.
        (query, doc) pairs from different requests can batch together if they
        share the same instruction.

        Args:
            prepared_items: Pre-processed items with cost for batching (ScorePreparedItem).
            query: Query item to score all docs against.
            items: Document items to score.
            instruction: Optional instruction for instruction-tuned rerankers.
            options: Optional runtime options (resolved from profile + overrides).
            request_id: Optional request ID for logging/tracing.
            timing: Optional RequestTiming object to track timing for this request.

        Returns:
            Future that resolves to WorkerResult with score results and timing.
            Each result dict has {"score": float}.

        Raises:
            RuntimeError: If worker is not running.
            QueueFullError: If queue is full and cannot accept more items.
        """
        self._check_queue_capacity(len(prepared_items))

        future, request_timing = self._create_future_and_timing(timing)

        metadata = RequestMetadata(
            future=future,
            items=items,
            timing=request_timing,
            request_id=request_id,
            operation="score",
            query=query,
            instruction=instruction,
            options=options,
        )

        # Score operations use base model batcher (no LoRA support for reranking)
        return await self._submit_to_batcher(prepared_items, metadata, None)

    async def submit_preformed(
        self,
        prepared_items: Sequence[HasCost],
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        options: dict[str, Any] | None = None,
        request_id: str | None = None,
        timing: RequestTiming | None = None,
    ) -> asyncio.Future[WorkerResult]:
        """Run a caller-formed encode batch directly.

        Used by the worker-sidecar IPC path after Rust has already handled
        queue pull, scheduling, batching, and adaptive control. Direct HTTP
        callers should continue to use :meth:`submit` so local single-instance
        serving retains Python batching.
        """
        self._check_running()

        future, request_timing = self._create_future_and_timing(timing)
        metadata = RequestMetadata(
            future=future,
            items=items,
            output_types=output_types,
            timing=request_timing,
            instruction=instruction,
            is_query=is_query,
            options=options,
            request_id=request_id,
        )
        lora = options.get("lora") if options else None
        return await self._submit_preformed_batch(prepared_items, metadata, lora)

    async def submit_extract_preformed(
        self,
        prepared_items: Sequence[HasCost],
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        request_id: str | None = None,
        timing: RequestTiming | None = None,
    ) -> asyncio.Future[WorkerResult]:
        """Run a caller-formed extract batch directly for sidecar IPC."""
        self._check_running()

        future, request_timing = self._create_future_and_timing(timing)
        metadata = RequestMetadata(
            future=future,
            items=items,
            timing=request_timing,
            request_id=request_id,
            operation="extract",
            labels=labels,
            output_schema=output_schema,
            instruction=instruction,
            options=options,
        )
        lora = options.get("lora") if options else None
        return await self._submit_preformed_batch(prepared_items, metadata, lora)

    async def submit_extract_preformed_batch(
        self,
        requests: Sequence[PreformedExtractRequest],
        *,
        lora: str | None,
    ) -> list[asyncio.Future[WorkerResult]]:
        """Run caller-formed extract requests as one sidecar IPC batch.

        The caller has already decided the queue batch. We preserve that
        grouping and let ``_process_batch`` split only on adapter config.
        ``lora`` is explicit because adapter LoRA state is global for the
        forward pass and must be homogeneous for the synthetic batch.
        """
        self._check_running()

        entries: list[tuple[Sequence[HasCost], RequestMetadata]] = []
        for req in requests:
            future, request_timing = self._create_future_and_timing(req.timing)
            entries.append(
                (
                    req.prepared_items,
                    RequestMetadata(
                        future=future,
                        items=req.items,
                        timing=request_timing,
                        request_id=req.request_id,
                        operation="extract",
                        labels=req.labels,
                        output_schema=req.output_schema,
                        instruction=req.instruction,
                        options=req.options,
                    ),
                )
            )

        return await self._submit_preformed_metadata_batch(entries, lora)

    async def submit_score_preformed(
        self,
        prepared_items: Sequence[HasCost],
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        request_id: str | None = None,
        timing: RequestTiming | None = None,
    ) -> asyncio.Future[WorkerResult]:
        """Run a caller-formed score batch directly for sidecar IPC."""
        self._check_running()

        future, request_timing = self._create_future_and_timing(timing)
        metadata = RequestMetadata(
            future=future,
            items=items,
            timing=request_timing,
            request_id=request_id,
            operation="score",
            query=query,
            instruction=instruction,
            options=options,
        )
        return await self._submit_preformed_batch(prepared_items, metadata, None)

    async def submit_score_preformed_batch(
        self,
        requests: Sequence[PreformedScoreRequest],
    ) -> list[asyncio.Future[WorkerResult]]:
        """Run caller-formed score requests as one sidecar IPC batch."""
        self._check_running()

        entries: list[tuple[Sequence[HasCost], RequestMetadata]] = []
        for req in requests:
            future, request_timing = self._create_future_and_timing(req.timing)
            entries.append(
                (
                    req.prepared_items,
                    RequestMetadata(
                        future=future,
                        items=req.items,
                        timing=request_timing,
                        request_id=req.request_id,
                        operation="score",
                        query=req.query,
                        instruction=req.instruction,
                        options=req.options,
                    ),
                )
            )

        return await self._submit_preformed_metadata_batch(entries, None)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _get_batcher(self, lora: str | None) -> BatchFormer[HasCost, RequestMetadata]:
        """Get or create batcher for a LoRA.

        Args:
            lora: LoRA adapter name, or None for base model.

        Returns:
            BatchFormer for the specified LoRA.
        """
        if lora not in self._batchers:
            self._batchers[lora] = BatchFormer(self._batch_config)
            logger.debug("Created batcher for LoRA '%s'", lora)
        return self._batchers[lora]

    def _check_queue_capacity(self, n_items: int) -> None:
        """Check if queue can accept n_items, raise QueueFullError if not.

        Args:
            n_items: Number of items to add to the queue.

        Raises:
            QueueFullError: If queue is full and cannot accept more items.
        """
        if not self._running:
            msg = "ModelWorker is not running"
            raise RuntimeError(msg)
        max_queue = self._config.max_queue_size
        if max_queue > 0:
            current_pending = self.pending_count
            new_count = current_pending + n_items
            if new_count > max_queue:
                msg = f"Queue full: {current_pending} items pending, cannot add {n_items} more (limit: {max_queue})"
                raise QueueFullError(msg)

    def _create_future_and_timing(
        self,
        timing: RequestTiming | None,
    ) -> tuple[asyncio.Future[WorkerResult], RequestTiming]:
        """Create a future and initialize timing for a request.

        Args:
            timing: Optional existing RequestTiming object.

        Returns:
            Tuple of (future, timing) ready for use.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[WorkerResult] = loop.create_future()
        request_timing = timing or RequestTiming()
        request_timing.start_queue()
        return future, request_timing

    async def _submit_to_batcher(
        self,
        prepared_items: Sequence[HasCost],
        metadata: RequestMetadata,
        lora: str | None,
    ) -> asyncio.Future[WorkerResult]:
        """Submit prepared items to the appropriate batcher.

        Args:
            prepared_items: Items to submit.
            metadata: Request metadata (contains the future).
            lora: LoRA adapter name, or None for base model.

        Returns:
            The future from metadata that will resolve to WorkerResult.
        """
        batcher = self._get_batcher(lora)
        await batcher.submit_many([(item, metadata) for item in prepared_items])
        self._ensure_process_loop_started()
        return metadata.future

    def _ensure_process_loop_started(self) -> None:
        if self._process_task is None:
            self._process_task = asyncio.create_task(
                self._process_loop(),
                name="model-worker-process",
            )

    def _check_running(self) -> None:
        if not self._running:
            msg = "ModelWorker is not running"
            raise RuntimeError(msg)

    async def _submit_preformed_batch(
        self,
        prepared_items: Sequence[HasCost],
        metadata: RequestMetadata,
        lora: str | None,
    ) -> asyncio.Future[WorkerResult]:
        await self._submit_preformed_metadata_batch([(prepared_items, metadata)], lora)
        return metadata.future

    async def _submit_preformed_metadata_batch(
        self,
        entries: Sequence[tuple[Sequence[HasCost], RequestMetadata]],
        lora: str | None,
    ) -> list[asyncio.Future[WorkerResult]]:
        """Run a caller-formed batch without the internal BatchFormer.

        Builds a synthetic ``FormattedBatch`` directly from ``prepared_items``
        — bypassing the BatchFormer / FCFS / adaptive controller entirely
        — and dispatches it through the existing ``_process_batch`` path so
        operation handlers, postprocessors and timing instrumentation all
        keep working unchanged.

        Concurrency: held under ``_adapter_dispatch_lock`` so concurrent calls
        to different LoRAs cannot interleave the (``set_active_lora`` →
        ``_process_batch``) pair, which otherwise races on the adapter's
        active-LoRA state. The single-thread inference executor would
        already serialise the GPU work — the lock specifically protects
        the LoRA-set-then-forward atomicity.
        """
        items_list: list[HasCost] = []
        metadata_per_item: list[RequestMetadata] = []
        futures: list[asyncio.Future[WorkerResult]] = []
        total_cost = 0

        # Cost is a sequence-of-protocol read, no I/O — safe outside the lock.
        for prepared_items, metadata in entries:
            request_items = list(prepared_items)
            if not request_items:
                metadata.future.set_exception(ValueError("preformed request contains no prepared items"))
                futures.append(metadata.future)
                continue
            items_list.extend(request_items)
            total_cost += sum(item.cost for item in request_items)
            # ``_process_batch`` keys metadata identity by ``id(metadata)`` so a
            # single shared metadata reference (one per request, N items) is
            # fine; the existing ``_complete_requests`` only fires the future
            # once all items have results regardless of how many slots share it.
            metadata_per_item.extend([metadata] * len(request_items))
            futures.append(metadata.future)

        if not items_list:
            return futures

        synthetic_batch: FormattedBatch[HasCost, RequestMetadata] = FormattedBatch(
            items=items_list,
            metadata=metadata_per_item,
            total_cost=total_cost,
        )

        async with self._adapter_dispatch_lock:
            self._adapter.set_active_lora(lora)
            await self._process_batch(synthetic_batch)

        return futures

    # =========================================================================
    # Batch Processing
    # =========================================================================

    async def _get_next_batch_fcfs(
        self, was_idle: bool = True
    ) -> tuple[str | None, FormattedBatch[HasCost, RequestMetadata], bool]:
        """Get next batch using FCFS (First-Come-First-Serve) selection.

        Selects the batcher whose first pending request has waited the longest.
        This ensures fairness across LoRAs - no LoRA starves even with low traffic.

        When the worker was idle (had to poll for requests), dispatches
        immediately without waiting for batch timeout. This eliminates
        unnecessary latency at low concurrency while preserving batching
        efficiency when requests arrive during inference.

        Args:
            was_idle: Whether the worker was idle before this call. When True,
                dispatches immediately without waiting for batch formation.
                When False, uses the normal timeout/coalesce mechanism to
                accumulate a proper batch.

        Returns:
            Tuple of (lora_name, batch, was_idle) where lora_name is None for
            base model and was_idle indicates whether the worker had to poll
            (was truly idle).
        """
        while True:
            oldest_lora: str | None = None
            oldest_time: float = float("inf")

            # Find batcher with oldest first-request time
            for lora, batcher in self._batchers.items():
                if batcher.pending_count > 0:
                    first_time = batcher._first_request_time
                    if first_time is not None and first_time < oldest_time:
                        oldest_time = first_time
                        oldest_lora = lora

            if oldest_lora is not None or (oldest_lora is None and self._batchers[None].pending_count > 0):
                # Found a batcher with pending items - get batch from it
                selected_lora = oldest_lora if oldest_lora is not None else None
                batch = await self._batchers[selected_lora].get_batch(immediate=was_idle)
                return selected_lora, batch, was_idle

            # No batchers have pending items - worker is idle
            was_idle = True
            await asyncio.sleep(0.001)

            # Check if we should stop
            if not self._running:
                # Return empty batch to exit gracefully
                return None, FormattedBatch(items=[], metadata=[], total_cost=0), was_idle

    async def _drain_active_batcher(self, active_lora: str | None) -> bool:
        """Continuous-batching drain of ``active_lora``'s batcher.

        Bounded to the backlog present at entry: a saturated LoRA whose
        ``submit()`` coroutines keep refilling its own batcher during each GPU
        forward pass would otherwise never let this loop exit, starving sibling
        LoRAs / the base model (they are only re-examined once control returns
        to ``_get_next_batch_fcfs``). Draining at most the entry backlog and
        then returning restores the oldest-first FCFS fairness the docstring
        there advertises. See #1606.

        Returns True if any items were drained.
        """
        batcher = self._batchers.get(active_lora)
        if batcher is None:
            return False

        # Snapshot the backlog at entry; items that arrive mid-drain wait for
        # the next FCFS selection instead of extending the drain indefinitely.
        # Cap every drain to the remaining budget so a batch formed late in the
        # loop can't sweep in post-snapshot arrivals and overshoot the backlog.
        drain_budget = batcher.pending_count
        drained = False
        while drain_budget > 0:
            drain_batch = await batcher.try_drain(max_items=drain_budget)
            if drain_batch is None or drain_batch.size == 0:
                break
            drained = True
            drain_budget -= drain_batch.size
            await self._process_batch(drain_batch, engine_queue_owned=True)
            self._record_runtime_batch(drain_batch)
        return drained

    def _record_runtime_batch(self, batch: FormattedBatch[HasCost, RequestMetadata]) -> None:
        """Emit one engine-owned batch event after a successful dispatch."""
        if not worker_telemetry_enabled():
            return
        worker_telemetry().batch_formed(
            operation="other",
            model=self._model_name or "other",
            profile="default",
            size=batch.size,
            cost=batch.total_cost,
            capacity=self._batch_config.max_batch_cost,
        )

    async def _process_loop(self) -> None:
        """Background loop that processes batches using FCFS across LoRAs."""
        logger.debug("Process loop started")

        # Track idle state across iterations. When idle, the next batch is
        # dispatched immediately (low-concurrency optimization). When busy
        # (just finished inference + drain), we let BatchFormer's
        # timeout/coalesce mechanism accumulate a proper batch.
        was_idle = True

        while self._running:
            try:
                # Track time waiting for batch
                batch_wait_start = time.monotonic()

                # Wait for next batch using FCFS selection across LoRA batchers
                active_lora, batch, was_idle = await self._get_next_batch_fcfs(was_idle)

                # Skip empty batches (can happen during shutdown)
                if batch.size == 0:
                    continue

                telemetry = worker_telemetry() if worker_telemetry_enabled() else None
                if telemetry is not None:
                    pending_after_pull = sum(batcher.pending_count for batcher in self._batchers.values())
                    telemetry.queue_depth_changed(
                        operation="other",
                        model=self._model_name or "other",
                        profile="default",
                        depth=pending_after_pull,
                    )
                    telemetry.queue_pending_observed(
                        operation="other",
                        model=self._model_name or "other",
                        profile="default",
                        pending=pending_after_pull,
                    )

                batch_wait_ms = (time.monotonic() - batch_wait_start) * 1000

                inference_start = time.monotonic()
                drained = False
                async with self._adapter_dispatch_lock:
                    # Set active LoRA before processing the batch and any
                    # immediate drains. The lock also excludes sidecar
                    # pre-formed batches from interleaving a different LoRA
                    # selection between set_active_lora and adapter inference.
                    self._adapter.set_active_lora(active_lora)

                    # Process the batch and track inference time.
                    await self._process_batch(batch, engine_queue_owned=True)
                    self._record_runtime_batch(batch)

                    # Continuous batching: immediately drain items that accumulated
                    # during GPU inference, bypassing the coalesce/timeout wait —
                    # bounded to the backlog present at entry so a saturated LoRA
                    # can't starve sibling LoRAs / the base model (#1606).
                    drained = await self._drain_active_batcher(active_lora)

                # Determine idle state for next iteration.
                # Worker was busy if we drained items or processed a multi-item
                # batch — meaning requests are actively arriving and the next
                # batch should use timeout/coalesce to accumulate properly.
                was_idle = not drained and batch.size <= 1

                inference_ms = (time.monotonic() - inference_start) * 1000

                # Record instrumentation if enabled
                if self._stats.instrumentation_enabled:
                    # Lists are guaranteed to exist when instrumentation is enabled
                    assert self._stats.batch_sizes is not None
                    assert self._stats.batch_tokens is not None
                    assert self._stats.batch_wait_ms is not None
                    assert self._stats.inference_ms is not None
                    assert self._stats.requests_per_batch is not None

                    self._stats.batch_sizes.append(batch.size)
                    self._stats.batch_tokens.append(batch.total_tokens)
                    self._stats.batch_wait_ms.append(batch_wait_ms)
                    self._stats.inference_ms.append(inference_ms)
                    # Count unique requests in this batch
                    unique_requests = len({id(m) for m in batch.metadata})
                    self._stats.requests_per_batch.append(unique_requests)

                # Track batch efficiency for adaptive controller
                if self._efficiency_tracker is not None:
                    self._efficiency_tracker.record(batch.total_cost, self._batch_config.max_batch_cost)

                # Step the adaptive controller after processing. ``apply_step``
                # owns the controller-output → _batch_config write-back (see
                # AdaptiveBatchController); it is synchronous, so the in-place
                # mutation stays safe against BatchFormer's async lock, which
                # cannot interleave with it.
                if self._adaptive_controller is not None and self._latency_tracker is not None:
                    observed_p50 = self._latency_tracker.p50()
                    fill_ratio = self._efficiency_tracker.mean_fill_ratio() if self._efficiency_tracker else None
                    starvation_resets_before = self._adaptive_controller.starvation_resets
                    self._adaptive_controller.apply_step(
                        self._batch_config, observed_p50, fill_ratio, batch_size=batch.size
                    )
                    if telemetry is not None:
                        telemetry.adaptive_snapshot(
                            model=self._model_name or "other",
                            profile="default",
                            wait_ms=self._adaptive_controller.current_wait_ms,
                            cost=self._adaptive_controller.current_batch_cost,
                            observed_p50_ms=observed_p50,
                            target_p50_ms=self._adaptive_controller.target_p50_ms,
                            starvation_resets_delta=(
                                self._adaptive_controller.starvation_resets - starvation_resets_before
                            ),
                        )

                # Log every 10 batches at INFO level for visibility
                if self._stats.batches_processed % 10 == 0:
                    lora_info = f", lora={active_lora}" if active_lora else ""
                    logger.info(
                        "Batch #%d: items=%d, tokens=%d, requests=%d, wait=%.1fms, inference=%.1fms, pending=%d%s",
                        self._stats.batches_processed,
                        batch.size,
                        batch.total_tokens,
                        len({id(m) for m in batch.metadata}),
                        batch_wait_ms,
                        inference_ms,
                        self.pending_count,
                        lora_info,
                    )

            except asyncio.CancelledError:
                logger.debug("Process loop cancelled")
                break
            except Exception:
                logger.exception("Error in process loop")
                # Continue processing despite errors

        # Log summary on shutdown
        if self._stats.instrumentation_enabled:
            logger.info("Worker stats summary:\n%s", self._stats.summary())

        logger.debug("Process loop stopped")

    async def _process_batch(
        self,
        batch: FormattedBatch[HasCost, RequestMetadata],
        *,
        engine_queue_owned: bool = False,
    ) -> None:
        """Process a single batch of requests.

        Items from different requests are batched together for inference if they
        share the same configuration. Delegates to operation handlers for the
        actual inference and result fan-out.

        Args:
            batch: Formatted batch ready for inference.
        """
        if batch.size == 0:
            return

        logger.debug(
            "Processing batch: size=%d, tokens=%d",
            batch.size,
            batch.total_tokens,
        )

        # Group items by inference configuration
        config_groups = self._group_by_inference_config(batch)

        # Mark inference start for all requests in this batch
        telemetry = worker_telemetry() if engine_queue_owned and worker_telemetry_enabled() else None
        seen_metadata: set[int] = set()
        for metadata in batch.metadata:
            meta_id = id(metadata)
            if meta_id not in seen_metadata:
                seen_metadata.add(meta_id)
                if metadata.timing._inference_start is None:
                    metadata.timing.start_inference()
                    if telemetry is not None:
                        telemetry.queue_released(
                            operation=metadata.operation,
                            model=self._model_name or "other",
                            profile="default",
                            duration_s=metadata.timing.queue_ms / 1_000.0,
                        )

        # Run ONE inference call per unique configuration using handlers.
        # The BatchExecutor wraps dispatch with reactive OOM recovery —
        # cache_clear → evict_lru → split_batch → terminal failure. On
        # success it populates ``metadata._partial_results`` exactly as the
        # in-line code did before. On non-OOM exception or terminal OOM it
        # sets the exception on all affected futures.
        #
        # Counter semantics under recovery (preserved across this PR):
        # - ``inference_errors`` bumps once per config group whose dispatch
        #   either OOMed-and-recovered or surfaced any exception. This keeps
        #   pre-existing dashboards meaningful: an OOM that the recovery
        #   layer absorbed still counts as an error event (use
        #   ``oom_recoveries.recoveries_succeeded`` to distinguish recovered
        #   from terminal).
        # - ``items_processed`` counts items whose future completed *without*
        #   an exception, so partial-success splits don't overcount.
        for config_key, group_data in config_groups.items():
            operation = config_key[0]
            items_list, metadata_list, original_indices_list, prepared_items_list = group_data
            handler = self._handlers[operation]

            # Snapshot recovery counters and per-future done-state so we can
            # attribute changes to *this* group only.
            recoveries_before = self._stats.oom_recoveries.recoveries_attempted
            done_before = {id(m): m.future.done() for m in metadata_list}

            # Build a per-group dispatch closure: it captures the config_key
            # (operation params) so the executor can re-invoke dispatch on
            # halved sub-batches without re-deriving config from metadata.
            handler_config_key = config_key[1:]

            async def _dispatch(
                h: OperationHandler[Any],
                group: tuple[list[Item], list[RequestMetadata], list[int], list[HasCost]],
            ) -> Any:
                sub_items, sub_metadata, _sub_indices, sub_prepared = group
                return await self._run_handler_inference(
                    h,
                    sub_items,
                    handler_config_key,
                    sub_prepared,
                    sub_metadata,
                )

            try:
                await self._batch_executor.run(
                    handler,
                    (items_list, metadata_list, original_indices_list, prepared_items_list),
                    _dispatch,
                )
            except Exception as exc:
                # Defensive fan-out: BatchExecutor.run is supposed to fail
                # every per-request future on any exception (see
                # ``_fail_group``), but a bug in recovery primitives could
                # let an exception escape unannotated. Without this fan-out,
                # callers would hang until the HTTP-layer / queue-layer
                # timeout fires. Set the exception on every distinct,
                # not-yet-done future so callers see the failure
                # immediately rather than waiting on a leaked promise.
                logger.exception("Unexpected exception escaping BatchExecutor for %s", config_key)
                _seen: set[int] = set()
                for metadata in metadata_list:
                    mid = id(metadata)
                    if mid in _seen:
                        continue
                    _seen.add(mid)
                    if not metadata.future.done():
                        metadata.future.set_exception(exc)

            # Accounting: ``newly_failed`` is the count of *distinct*
            # request metadata objects whose future transitioned to
            # done-with-exception in this group (terminal failure even
            # under recovery). Multi-item requests appear N times in
            # ``metadata_list`` (one per item); we dedup by ``id(metadata)``
            # so a single 5-item request that fails counts as one error
            # event, not five. ``items_with_partial`` counts items whose
            # partial result was populated by a successful sub-batch
            # dispatch — these are the items that *were* successfully
            # processed, including under split-recovery, regardless of
            # whether their parent metadata future is done yet (the future
            # only completes after assemble in ``_complete_requests``).
            failed_metadata_ids: set[int] = set()
            for metadata in metadata_list:
                meta_id = id(metadata)
                if done_before.get(meta_id) is True:
                    continue  # was done before — not our responsibility
                if meta_id in failed_metadata_ids:
                    continue  # already counted this request
                if metadata.future.done() and metadata.future.exception() is not None:
                    failed_metadata_ids.add(meta_id)
            newly_failed = len(failed_metadata_ids)

            items_with_partial = sum(
                1
                for metadata, original_idx in zip(metadata_list, original_indices_list, strict=True)
                if metadata._partial_results is not None and original_idx in metadata._partial_results
            )

            recovery_engaged = self._stats.oom_recoveries.recoveries_attempted > recoveries_before
            if newly_failed > 0 or recovery_engaged:
                # One error event per config group, regardless of how many
                # futures failed and regardless of whether recovery succeeded.
                # This matches pre-PR dashboard semantics for "an OOM
                # happened on this batch".
                self._stats.inference_errors += 1
                if newly_failed > 0:
                    logger.warning(
                        "Inference error for batch config %s: %d future(s) failed",
                        config_key,
                        newly_failed,
                    )

            # Count successfully-dispatched items (partial result present).
            self._stats.items_processed += items_with_partial

        # Complete requests that have all their results
        self._complete_requests(batch)

        # Update batch stats
        self._stats.batches_processed += 1
        self._stats.total_tokens_processed += batch.total_tokens

    def _group_by_inference_config(
        self,
        batch: FormattedBatch[HasCost, RequestMetadata],
    ) -> dict[
        tuple[Any, ...],
        tuple[list[Item], list[RequestMetadata], list[int], list[HasCost]],
    ]:
        """Group batch items by inference configuration for cross-request batching.

        Items with the same configuration can be batched together in a single
        inference call, even if they come from different requests.

        Delegates config key creation to operation handlers.

        Args:
            batch: The batch to group.

        Returns:
            Dict mapping config tuple to (items_list, metadata_list, original_indices_list, prepared_items_list).
        """
        # Fast path: if all metadata objects are identical (same request),
        # they share the same config — skip per-item hashing
        metadata_list = batch.metadata
        if len(metadata_list) > 1 and all(m is metadata_list[0] for m in metadata_list):
            first_meta = metadata_list[0]
            handler = self._handlers[first_meta.operation]
            handler_key = handler.make_config_key(first_meta)
            config_key = (first_meta.operation, *handler_key)

            items_list: list[Item] = []
            indices_list: list[int] = []
            for prepared_item in batch.items:
                original_idx = prepared_item.original_index
                items_list.append(first_meta.items[original_idx])
                indices_list.append(original_idx)

            return {config_key: (items_list, list(metadata_list), indices_list, list(batch.items))}

        groups: dict[
            tuple[Any, ...],
            tuple[list[Item], list[RequestMetadata], list[int], list[HasCost]],
        ] = {}

        for prepared_item, metadata in zip(batch.items, metadata_list, strict=True):
            # Get handler and create config key
            handler = self._handlers[metadata.operation]
            handler_key = handler.make_config_key(metadata)
            config_key = (metadata.operation, *handler_key)

            if config_key not in groups:
                groups[config_key] = ([], [], [], [])

            group_items, group_metadata, group_indices, group_prepared = groups[config_key]

            # Get the original Item from the request
            original_idx = prepared_item.original_index
            item = metadata.items[original_idx]

            group_items.append(item)
            group_metadata.append(metadata)
            group_indices.append(original_idx)
            group_prepared.append(prepared_item)

        return groups

    async def _run_handler_inference(
        self,
        handler: OperationHandler[Any],
        items: list[Item],
        config_key: tuple[Any, ...],
        prepared_items: list[HasCost] | None,
        metadata_list: list[RequestMetadata],
    ) -> Any:
        """Run inference via handler in thread pool.

        Args:
            handler: The operation handler.
            items: Items to process.
            config_key: Config key (without operation prefix).
            prepared_items: Pre-processed items.
            metadata_list: Request metadata.

        Returns:
            Typed output from handler.
        """
        loop = asyncio.get_running_loop()

        inference_fn = functools.partial(
            handler.run_inference,
            self._adapter,
            items,
            config_key,
            prepared_items,
            metadata_list,
        )

        return await loop.run_in_executor(
            self._inference_executor,
            inference_fn,
        )

    def _complete_requests(self, batch: FormattedBatch[HasCost, RequestMetadata]) -> None:
        """Complete requests that have all their results.

        Uses handlers to assemble partial results into full outputs.

        Args:
            batch: The batch being processed.
        """
        completed_metadata: set[int] = set()
        for metadata in batch.metadata:
            meta_id = id(metadata)
            if meta_id in completed_metadata:
                continue
            completed_metadata.add(meta_id)

            # Check if we have all results for this request
            if metadata._partial_results is not None and len(metadata._partial_results) == len(metadata.items):
                metadata.timing.end_inference()

                # Assemble partial outputs using handler
                handler = self._handlers[metadata.operation]
                output = handler.assemble_output(metadata._partial_results, len(metadata.items))

                # Set result on future with timing
                if not metadata.future.done():
                    worker_result = WorkerResult(output=output, timing=metadata.timing)
                    metadata.future.set_result(worker_result)

                    # Feed latency sample to adaptive controller
                    if self._latency_tracker is not None:
                        self._latency_tracker.record(metadata.timing.total_ms)

                    # Feed inference-only sample for auto-calibration.
                    # Uses inference_ms (GPU forward pass) not total_ms to
                    # avoid a feedback loop where queue/batch wait inflates
                    # the calibration target.
                    if self._adaptive_controller is not None and not self._adaptive_controller.calibrated:
                        self._adaptive_controller.record_inference_sample(metadata.timing.inference_ms)
