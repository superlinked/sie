"""Types and dataclasses for model worker.

Contains request metadata, configuration, and statistics types.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from sie_server.core.inference_output import EncodeOutput, ExtractOutput, ScoreOutput
from sie_server.core.oom import OomRecoveryConfig, OomRecoveryStats
from sie_server.core.timing import RequestTiming

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

# Type alias for worker output union
WorkerOutput = EncodeOutput | ScoreOutput | ExtractOutput

# Detailed instrumentation (``SIE_INSTRUMENTATION=1`` / ``WorkerConfig.instrumentation``)
# records five samples per batch. It is a debugging aid, but a flag left on in a
# long-lived deployment would otherwise grow those series without bound for the
# life of the process — and ``WorkerStats.summary`` runs min/max/mean/median over
# the whole history, so it would also get progressively slower. Each series is a
# bounded ring buffer instead: the newest N samples are kept, older ones drop off.
# N is deliberately generous (a busy worker at ~50 batches/s covers the last few
# minutes) because the summary is a shape check, not an exact audit trail.
INSTRUMENTATION_HISTORY_LIMIT = 10_000


@dataclass
class WorkerResult:
    """Result from a worker inference request.

    Contains typed inference output and timing information.
    The output field contains the raw typed output from the adapter.
    API layer is responsible for formatting to JSON-serializable dicts.
    """

    output: WorkerOutput
    timing: RequestTiming


@dataclass(slots=True)
class RequestMetadata:
    """Metadata for a pending inference request.

    Carries the information needed to run inference and return results.
    Supports encode, extract, and score operations via the `operation` field.

    For encode operations:
        - output_types: Which outputs to return ("dense", "sparse", "multivector")
        - instruction: Optional instruction for instruction-tuned models
        - is_query: Whether items are queries (True) or documents (False)

    For extract operations:
        - labels: Entity types to extract (e.g., ["person", "organization"])
        - output_schema: Optional schema for structured extraction
        - instruction: Optional instruction (reused from encode)

    For score operations:
        - query: The query item to score all items against
        - instruction: Optional instruction for instruction-tuned rerankers
    """

    future: asyncio.Future[WorkerResult]
    items: list[Item]  # Original items for adapter (docs for score)
    timing: RequestTiming  # Tracks timing for this request
    request_id: str | None = None
    # Partial results for sub-batching: maps original_index -> typed output (batch_size=1)
    _partial_results: dict[int, EncodeOutput | ScoreOutput | ExtractOutput] | None = None

    # Operation type determines which adapter method to call
    operation: Literal["encode", "extract", "score"] = "encode"

    # Shared params
    instruction: str | None = None
    options: dict[str, Any] | None = None  # Adapter options to override model config defaults

    # Encode-specific params
    output_types: list[str] = field(default_factory=list)
    is_query: bool = False

    # Extract-specific params
    labels: list[str] | None = None
    output_schema: dict[str, Any] | None = None

    # Score-specific params
    query: Item | None = None  # Query item for reranking


class QueueFullError(Exception):
    """Raised when the worker queue is full and cannot accept more requests.

    Carries the structured counts from the capacity check so the HTTP
    layer can distinguish a request that can *never* fit (``requested >
    limit`` — a client error) from genuine transient pressure (queue
    occupied by other work — retryable). All fields are optional so
    legacy raise sites that only pass a message keep working; without
    them the HTTP layer conservatively treats the error as transient.

    Attributes:
        pending: Items already queued when the check ran.
        requested: Items this request tried to add.
        limit: The worker's ``max_queue_size``.
    """

    def __init__(
        self,
        message: str,
        *,
        pending: int | None = None,
        requested: int | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__(message)
        self.pending = pending
        self.requested = requested
        self.limit = limit


class WorkerDrainedError(RuntimeError):
    """Raised on a request that was still queued when its worker stopped.

    ``ModelWorker.stop()`` cancels the batch-processing loop, so anything
    still sitting in a ``BatchFormer`` will never run. Without an explicit
    failure those awaiters hang until some outer timeout fires — the model
    is gone, but the promise is never broken (the eviction-drain contract's
    second stage).

    The condition is transient and the work was never started, so this is a
    *retryable* error: the HTTP surface maps it onto the existing 503
    ``MODEL_LOADING`` shape (with ``Retry-After``) and the queue surface
    maps it onto a ``nak_retry`` redelivery — the same answers both paths
    already give for "this model is not available right now".
    """


@dataclass
class AdaptiveBatchingParams:
    """Parameters for adaptive batch wait control (passed from EngineConfig).

    ``target_p50_ms=None`` means auto-calibrate from observed inference latency.
    An explicit float value means use that as a fixed SLO target.
    """

    enabled: bool = False
    target_p50_ms: float | None = None
    calibration_multiplier: float = 1.5
    min_target_p50_ms: float = 5.0
    max_target_p50_ms: float = 500.0
    # min_wait_ms is the floor the PI controller can shrink the first-request
    # timeout to. It is *not* a mandatory wait — under load the batcher fills
    # (or the coalesce window fires) well before this timeout trips, so
    # raising it has no latency cost in steady state. The floor exists
    # purely to keep the batcher from collapsing to a "flush on every
    # submit" mode under oscillating headroom, which would shred GPU
    # batch sizes from 64 down to single-digits.
    min_wait_ms: float = 15.0
    max_wait_ms: float = 50.0
    gain: float = 0.3
    integral_gain: float = 0.05
    window_size: int = 200
    update_interval: int = 10
    # -- Starvation detection / deadlock recovery ----------------------------
    # The PI loop can deadlock in a "batch-of-1" attractor: once observed p50
    # exceeds the auto-calibrated target, the controller shrinks wait to the
    # floor and cost to the floor, which collapses GPU batches to 1 item,
    # which raises p50 further, which keeps the controller pinned at the
    # floor. Headroom stays negative indefinitely and there is no recovery
    # path. These knobs arm a self-healing escape hatch. Set
    # ``starvation_recovery_enabled=False`` to disable (e.g. for synthetic
    # tests that want bare PI behaviour).
    starvation_recovery_enabled: bool = True
    # How many consecutive batches of ``size <= starvation_batch_size`` at
    # the floor before declaring deadlock. Must be large enough to absorb
    # genuine idle-tail batches where the next burst's first batch is
    # naturally tiny, small enough to recover within a few seconds of load.
    starvation_window: int = 20
    # Treat any batch with ``size <= this`` as contributing to the starvation
    # counter. 1 by default — only "forward pass of a single item" counts.
    starvation_batch_size: int = 1


@dataclass
class WorkerConfig:
    """Configuration for ModelWorker."""

    max_batch_tokens: int = 16384
    max_batch_requests: int = 256
    max_batch_wait_ms: float = 15.0
    # Coalescing window: yield the current batch if no new items have been
    # submitted within this ceiling (capped by ``coalesce_ratio *
    # max_batch_wait_ms``). The default targets typical IPC burst jitter
    # so a worker-sidecar batch of ~64 items lands in a single GPU forward
    # instead of being shredded into several half-full ones.
    coalesce_ms: float = 15.0
    coalesce_ratio: float = 0.5
    # Idle-dispatch accumulation window (#2874). An idle worker used to
    # dispatch IMMEDIATELY with whatever was pending, so a burst arriving at
    # an idle worker degenerated into a train of small serialized forwards
    # (many batch-of-1 dispatches instead of one fused batch). With this
    # window, the first batch after idleness coalesces arrivals until the
    # queue has been quiet for ``idle_coalesce_ms`` (or the batch fills, or
    # ``max_batch_wait_ms`` elapses since the first request). A lone request
    # therefore waits at most this window — single-digit ms — before
    # dispatch, and ``0`` restores the legacy immediate dispatch.
    idle_coalesce_ms: float = 3.0
    max_queue_size: int = 1000  # Maximum pending items in queue (0 = unlimited)
    instrumentation: bool = False
    adaptive_batching: AdaptiveBatchingParams = field(default_factory=AdaptiveBatchingParams)
    # Reactive OOM recovery applied inside `_process_batch`. When disabled,
    # OOM exceptions propagate as before (legacy behaviour).
    oom_recovery: OomRecoveryConfig = field(default_factory=OomRecoveryConfig)


@dataclass
class WorkerStats:
    """Runtime statistics for a ModelWorker."""

    batches_processed: int = 0
    items_processed: int = 0
    total_tokens_processed: int = 0
    inference_errors: int = 0

    # OOM recovery counters (populated by BatchExecutor). Always present so
    # callers can read counters without conditional checks.
    oom_recoveries: OomRecoveryStats = field(default_factory=OomRecoveryStats)

    # Detailed instrumentation (for performance analysis). Bounded ring buffers,
    # each capped at INSTRUMENTATION_HISTORY_LIMIT samples — see that constant.
    batch_sizes: deque[int] | None = None  # Items per batch
    batch_tokens: deque[int] | None = None  # Tokens per batch
    batch_wait_ms: deque[float] | None = None  # Time waiting for batch to form
    inference_ms: deque[float] | None = None  # GPU inference time per batch
    requests_per_batch: deque[int] | None = None  # Unique requests combined per batch

    def enable_instrumentation(self, history_limit: int = INSTRUMENTATION_HISTORY_LIMIT) -> None:
        """Enable detailed instrumentation tracking over a bounded sample window."""
        self.batch_sizes = deque(maxlen=history_limit)
        self.batch_tokens = deque(maxlen=history_limit)
        self.batch_wait_ms = deque(maxlen=history_limit)
        self.inference_ms = deque(maxlen=history_limit)
        self.requests_per_batch = deque(maxlen=history_limit)

    @property
    def instrumentation_enabled(self) -> bool:
        """Check if instrumentation is enabled."""
        return self.batch_sizes is not None

    def summary(self) -> str:
        """Return a summary of collected statistics."""
        lines = [
            f"Batches processed: {self.batches_processed}",
            f"Items processed: {self.items_processed}",
            f"Tokens processed: {self.total_tokens_processed}",
            f"Inference errors: {self.inference_errors}",
        ]
        if self.instrumentation_enabled and self.batch_sizes:
            import statistics

            # All instrumentation lists are initialized together, so assert they exist
            assert self.batch_tokens is not None
            assert self.batch_wait_ms is not None
            assert self.inference_ms is not None
            assert self.requests_per_batch is not None

            lines.extend(
                [
                    "",
                    "=== Batch Size Stats ===",
                    f"  Items/batch: min={min(self.batch_sizes)}, max={max(self.batch_sizes)}, "
                    f"mean={statistics.mean(self.batch_sizes):.1f}, median={statistics.median(self.batch_sizes):.1f}",
                    f"  Tokens/batch: min={min(self.batch_tokens)}, max={max(self.batch_tokens)}, "
                    f"mean={statistics.mean(self.batch_tokens):.1f}",
                    f"  Requests/batch: min={min(self.requests_per_batch)}, max={max(self.requests_per_batch)}, "
                    f"mean={statistics.mean(self.requests_per_batch):.1f}",
                    "",
                    "=== Timing Stats ===",
                    f"  Batch wait (ms): min={min(self.batch_wait_ms):.1f}, max={max(self.batch_wait_ms):.1f}, "
                    f"mean={statistics.mean(self.batch_wait_ms):.1f}, p50={statistics.median(self.batch_wait_ms):.1f}",
                    f"  Inference (ms): min={min(self.inference_ms):.1f}, max={max(self.inference_ms):.1f}, "
                    f"mean={statistics.mean(self.inference_ms):.1f}, p50={statistics.median(self.inference_ms):.1f}",
                ]
            )
        return "\n".join(lines)
