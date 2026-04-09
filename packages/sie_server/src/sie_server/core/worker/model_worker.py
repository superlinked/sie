"""Model worker for async request handling with dynamic batching.

The ModelWorker manages a single model's inference pipeline:
1. Accepts tokenized requests via submit()
2. Batches requests using BatchFormer
3. Runs inference on batches via operation handlers
4. Fans out results to waiting futures

See DESIGN.md Section 5.3.

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
from sie_server.core.worker.types import (
    QueueFullError,
    RequestMetadata,
    WorkerConfig,
    WorkerResult,
    WorkerStats,
)

try:
    from prometheus_client import Gauge, Histogram

    GPU_BATCH_ITEMS = Histogram(
        "sie_gpu_batch_items",
        "Number of items per GPU batch",
        ["model"],
        buckets=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    )
    GPU_BATCH_TOKENS = Histogram(
        "sie_gpu_batch_tokens",
        "Number of tokens per GPU batch",
        ["model"],
        buckets=[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    )
    ADAPTIVE_BATCH_WAIT = Gauge(
        "sie_adaptive_batch_wait_ms",
        "Current dynamic max_batch_wait_ms from adaptive controller",
        ["model"],
    )
    ADAPTIVE_P50 = Gauge(
        "sie_adaptive_p50_ms",
        "Observed rolling p50 latency from adaptive controller",
        ["model"],
    )
    ADAPTIVE_HEADROOM = Gauge(
        "sie_adaptive_headroom_ms",
        "Latency headroom (target_p50 - observed_p50)",
        ["model"],
    )
    ADAPTIVE_BATCH_COST = Gauge(
        "sie_adaptive_batch_cost",
        "Current dynamic max_batch_cost (tokens) from adaptive controller",
        ["model"],
    )
    ADAPTIVE_FILL_RATIO = Gauge(
        "sie_adaptive_fill_ratio",
        "Mean batch fill ratio (actual_cost / max_cost)",
        ["model"],
    )
    ADAPTIVE_BATCH_EFFICIENCY = Histogram(
        "sie_adaptive_batch_efficiency",
        "Batch request efficiency: actual_batch_size / max_batch_requests",
        ["model"],
        buckets=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    _HAS_BATCH_METRICS = True
except ImportError:
    _HAS_BATCH_METRICS = False

if TYPE_CHECKING:
    from sie_server.adapters.base import ModelAdapter
    from sie_server.core.postprocessor_registry import PostprocessorRegistry
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)


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
    ) -> None:
        """Initialize the model worker.

        Args:
            adapter: The model adapter to use for inference.
            config: Worker configuration. Uses defaults if not provided.
            model_name: Name of the model (for postprocessor lookup).
            postprocessor_registry: Registry for postprocessors (optional).
            handlers: Optional dict of operation handlers for dependency injection.
                      Defaults to standard handlers if not provided.
        """
        self._adapter = adapter
        self._config = config or WorkerConfig()
        self._model_name = model_name
        self._postprocessor_registry = postprocessor_registry

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
        )

        # Per-LoRA batchers: None = base model, "lora-name" = specific LoRA
        # Each LoRA gets its own batcher for FCFS fairness
        self._batchers: dict[str | None, BatchFormer[HasCost, RequestMetadata]] = {}
        self._batchers[None] = BatchFormer(self._batch_config)  # Base model batcher

        # Thread pool for running inference (doesn't block event loop)
        self._inference_executor = ThreadPoolExecutor(
            max_workers=1,  # Single worker for GPU serialization
            thread_name_prefix="inference",
        )

        # Adaptive batching controller (optional, off by default)
        ab = self._config.adaptive_batching
        if ab.enabled:
            self._latency_tracker: LatencyTracker | None = LatencyTracker(
                window_size=ab.window_size,
            )
            self._efficiency_tracker: BatchEfficiencyTracker | None = BatchEfficiencyTracker()
            self._adaptive_controller: AdaptiveBatchController | None = AdaptiveBatchController(
                target_p50_ms=ab.target_p50_ms,
                calibration_multiplier=ab.calibration_multiplier,
                min_target_p50_ms=ab.min_target_p50_ms,
                max_target_p50_ms=ab.max_target_p50_ms,
                min_wait_ms=ab.min_wait_ms,
                max_wait_ms=ab.max_wait_ms,
                min_batch_cost=min(256, self._config.max_batch_tokens),
                max_batch_cost=max(
                    min(256, self._config.max_batch_tokens), self._config.max_batch_tokens * 4
                ),  # allow up to 4x growth
                gain=ab.gain,
                integral_gain=ab.integral_gain,
                cost_gain=ab.gain * 0.5,  # cost knob is more conservative
                update_interval=ab.update_interval,
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
        """Start the background processing task.

        Should be called before submitting requests.
        Set SIE_INSTRUMENTATION=1 to enable detailed batch statistics.
        """
        if self._running:
            return

        # Enable instrumentation if configured or env var is set
        env_instrumentation = os.environ.get("SIE_INSTRUMENTATION", "").lower() in ("1", "true", "yes")
        if self._config.instrumentation or env_instrumentation:
            self._stats.enable_instrumentation()
            logger.info("ModelWorker instrumentation enabled")

        self._running = True
        self._process_task = asyncio.create_task(
            self._process_loop(),
            name="model-worker-process",
        )
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
        return metadata.future

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

                batch_wait_ms = (time.monotonic() - batch_wait_start) * 1000

                # Set active LoRA before processing batch
                # This allows adapters to switch to the correct LoRA adapter
                self._adapter.set_active_lora(active_lora)

                # Process the batch and track inference time
                inference_start = time.monotonic()
                await self._process_batch(batch)

                # Continuous batching: immediately drain items that accumulated
                # during GPU inference, bypassing the coalesce/timeout wait.
                # This keeps the GPU saturated without re-entering the batch
                # formation loop.
                drained = False
                batcher = self._batchers.get(active_lora)
                while batcher is not None:
                    drain_batch = await batcher.try_drain()
                    if drain_batch is None or drain_batch.size == 0:
                        break
                    drained = True
                    await self._process_batch(drain_batch)
                    if _HAS_BATCH_METRICS:
                        _label_name = self._model_name or "unknown"
                        GPU_BATCH_ITEMS.labels(model=_label_name).observe(drain_batch.size)
                        GPU_BATCH_TOKENS.labels(model=_label_name).observe(drain_batch.total_tokens)

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

                if _HAS_BATCH_METRICS:
                    _label_name = self._model_name or "unknown"
                    GPU_BATCH_ITEMS.labels(model=_label_name).observe(batch.size)
                    GPU_BATCH_TOKENS.labels(model=_label_name).observe(batch.total_tokens)
                    if self._batch_config.max_batch_requests > 0:
                        ADAPTIVE_BATCH_EFFICIENCY.labels(model=_label_name).observe(
                            batch.size / self._batch_config.max_batch_requests
                        )

                # Track batch efficiency for adaptive controller
                if self._efficiency_tracker is not None:
                    self._efficiency_tracker.record(batch.total_cost, self._batch_config.max_batch_cost)

                # Step adaptive controller after processing.
                # Note: mutating _batch_config in-place is safe here because
                # both assignments happen synchronously (no await between them)
                # on the single event-loop thread.  BatchFormer reads these
                # fields only under its own async lock, which cannot interleave
                # with this synchronous block.
                if self._adaptive_controller is not None and self._latency_tracker is not None:
                    observed_p50 = self._latency_tracker.p50()
                    fill_ratio = self._efficiency_tracker.mean_fill_ratio() if self._efficiency_tracker else None
                    new_wait, new_cost = self._adaptive_controller.step(observed_p50, fill_ratio)
                    self._batch_config.max_batch_wait_ms = new_wait
                    self._batch_config.max_batch_cost = new_cost

                    if _HAS_BATCH_METRICS:
                        _label = self._model_name or "unknown"
                        ADAPTIVE_BATCH_WAIT.labels(model=_label).set(new_wait)
                        ADAPTIVE_BATCH_COST.labels(model=_label).set(new_cost)
                        if observed_p50 is not None:
                            ADAPTIVE_P50.labels(model=_label).set(observed_p50)
                            target = self._adaptive_controller.target_p50_ms
                            if target is not None:
                                ADAPTIVE_HEADROOM.labels(model=_label).set(target - observed_p50)
                        if fill_ratio is not None:
                            ADAPTIVE_FILL_RATIO.labels(model=_label).set(fill_ratio)

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

    async def _process_batch(self, batch: FormattedBatch[HasCost, RequestMetadata]) -> None:
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
        seen_metadata: set[int] = set()
        for metadata in batch.metadata:
            meta_id = id(metadata)
            if meta_id not in seen_metadata:
                seen_metadata.add(meta_id)
                if metadata.timing._inference_start is None:
                    metadata.timing.start_inference()

        # Run ONE inference call per unique configuration using handlers
        for config_key, group_data in config_groups.items():
            operation = config_key[0]
            items_list, metadata_list, original_indices_list, prepared_items_list = group_data
            handler = self._handlers[operation]

            try:
                # Run inference via handler (in thread pool)
                output = await self._run_handler_inference(
                    handler,
                    items_list,
                    config_key[1:],  # Strip operation prefix
                    prepared_items_list,
                    metadata_list,
                )

                # Fan out results using handler's slice method
                for batch_idx, (metadata, original_idx) in enumerate(
                    zip(metadata_list, original_indices_list, strict=True)
                ):
                    if metadata._partial_results is None:
                        metadata._partial_results = {}
                    metadata._partial_results[original_idx] = handler.slice_output(output, batch_idx)

                # Update stats
                self._stats.items_processed += len(items_list)

            except Exception as e:
                logger.exception("Inference error for batch config %s", config_key)
                self._stats.inference_errors += 1

                # Set exception on all affected requests
                failed_metadata: set[int] = set()
                for metadata in metadata_list:
                    meta_id = id(metadata)
                    if meta_id not in failed_metadata:
                        failed_metadata.add(meta_id)
                        if not metadata.future.done():
                            metadata.future.set_exception(e)

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
