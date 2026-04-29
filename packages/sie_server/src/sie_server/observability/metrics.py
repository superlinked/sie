"""Prometheus metrics for SIE Server.

Exposes metrics for monitoring request throughput, latency, batching efficiency,
and resource utilization.

Metrics follow Prometheus naming conventions:
- sie_ prefix for all metrics
- _total suffix for counters
- _seconds suffix for duration histograms
- _bytes suffix for memory metrics

See DESIGN.md Section 5.6 for observability design.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

if TYPE_CHECKING:
    from sie_server.core.timing import RequestTiming

logger = logging.getLogger(__name__)

# Histogram buckets for request duration (in seconds)
# Covers 1ms to 30s range, optimized for inference workloads
DURATION_BUCKETS = (
    0.001,  # 1ms
    0.005,  # 5ms
    0.01,  # 10ms
    0.025,  # 25ms
    0.05,  # 50ms
    0.1,  # 100ms
    0.25,  # 250ms
    0.5,  # 500ms
    1.0,  # 1s
    2.5,  # 2.5s
    5.0,  # 5s
    10.0,  # 10s
    30.0,  # 30s
)

# Histogram buckets for batch sizes
BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

# Histogram buckets for token counts
TOKEN_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


# -----------------------------------------------------------------------------
# Request Metrics
# -----------------------------------------------------------------------------

REQUESTS_TOTAL = Counter(
    "sie_requests_total",
    "Total number of requests processed",
    ["model", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "sie_request_duration_seconds",
    "Request duration breakdown by phase",
    ["model", "endpoint", "phase"],
    buckets=DURATION_BUCKETS,
)


# -----------------------------------------------------------------------------
# Batching Metrics
# -----------------------------------------------------------------------------

BATCH_SIZE = Histogram(
    "sie_batch_size",
    "Number of items per batch",
    ["model"],
    buckets=BATCH_SIZE_BUCKETS,
)

TOKENS_PROCESSED = Counter(
    "sie_tokens_processed_total",
    "Total number of tokens processed",
    ["model"],
)


# -----------------------------------------------------------------------------
# Queue Metrics
# -----------------------------------------------------------------------------

QUEUE_DEPTH = Gauge(
    "sie_queue_depth",
    "Current number of pending items in queue",
    ["model"],
)


# -----------------------------------------------------------------------------
# Model Metrics
# -----------------------------------------------------------------------------

MODEL_LOADED = Gauge(
    "sie_model_loaded",
    "Whether a model is currently loaded (1=loaded, 0=not loaded)",
    ["model", "device"],
)

MODEL_MEMORY_BYTES = Gauge(
    "sie_model_memory_bytes",
    "Estimated GPU memory usage for a loaded model in bytes",
    ["model", "device"],
)


# -----------------------------------------------------------------------------
# OOM Recovery Metrics
# -----------------------------------------------------------------------------
#
# These mirror ``sie_server.core.oom.OomRecoveryStats`` and are bumped from
# inside ``BatchExecutor`` whenever the corresponding strategy fires. Operators
# use them to:
#   * Detect a sustained recovery rate (= a real memory leak or undersized
#     pool, not just transient pressure).
#   * Tune ``SIE_OOM_RECOVERY__MAX_SPLIT_DEPTH`` based on how often
#     ``batch_splits`` fires relative to ``terminal_failures``.
#   * Dashboard "recovery saved this many requests" via
#     ``recoveries_succeeded`` (analytics value separate from
#     ``terminal_failures``, which is the alert signal).

OOM_RECOVERIES_ATTEMPTED = Counter(
    "sie_oom_recoveries_attempted_total",
    "Number of OOM events caught at the worker dispatch boundary",
    ["model"],
)

OOM_RECOVERIES_SUCCEEDED = Counter(
    "sie_oom_recoveries_succeeded_total",
    "OOM recovery attempts that fully succeeded (every metadata got a result)",
    ["model"],
)

OOM_TERMINAL_FAILURES = Counter(
    "sie_oom_terminal_failures_total",
    "OOM events where every recovery strategy was exhausted (clients see RESOURCE_EXHAUSTED)",
    ["model"],
)

OOM_CACHE_CLEARS = Counter(
    "sie_oom_cache_clears_total",
    "Cache-clear recovery actions executed",
    ["model"],
)

OOM_EVICTIONS_TRIGGERED = Counter(
    "sie_oom_evictions_triggered_total",
    "Sibling-model evictions performed during OOM recovery",
    ["model"],
)

OOM_BATCH_SPLITS = Counter(
    "sie_oom_batch_splits_total",
    "Top-level batch-split recovery invocations (recursive halves not counted separately)",
    ["model"],
)


# -----------------------------------------------------------------------------
# Idle Eviction Metrics
# -----------------------------------------------------------------------------
#
# Bumped from ``ModelRegistry._idle_evict_loop`` each time a cold model is
# unloaded by the proactive idle-TTL evictor. Different from
# ``sie_oom_evictions_triggered_total`` (which is the reactive sibling-eviction
# step inside ``BatchExecutor``) — this metric tracks the proactive cleanup
# loop and is what operators key on when validating ``SIE_IDLE_EVICT_S``.

IDLE_EVICTIONS_TOTAL = Counter(
    "sie_idle_evictions_total",
    "Models unloaded by the proactive idle-TTL evictor",
    ["model"],
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def record_request(
    model: str,
    endpoint: str,
    status: str,
    timing: RequestTiming | None = None,
    *,
    request_id: str | None = None,
    api_key: str | None = None,
    queue_depth: int | None = None,
) -> None:
    """Record metrics and structured log for a completed request.

    Args:
        model: Model name.
        endpoint: Endpoint name (encode, score, extract).
        status: Request status (success, error, queue_full).
        timing: Optional timing information for latency breakdown.
        request_id: Optional request ID for tracing.
        api_key: Optional API key (masked) for audit.
        queue_depth: Optional current queue depth at request time.
    """
    # Increment request counter
    REQUESTS_TOTAL.labels(model=model, endpoint=endpoint, status=status).inc()

    # Record latency breakdown if timing available
    if timing is not None:
        # Total duration
        REQUEST_DURATION.labels(model=model, endpoint=endpoint, phase="total").observe(
            timing.total_ms / 1000
        )  # Convert ms to seconds

        # Queue time (if tracked)
        if timing.queue_ms > 0:
            REQUEST_DURATION.labels(model=model, endpoint=endpoint, phase="queue").observe(timing.queue_ms / 1000)

        # Tokenization time (if tracked)
        if timing.tokenization_ms > 0:
            REQUEST_DURATION.labels(model=model, endpoint=endpoint, phase="tokenize").observe(
                timing.tokenization_ms / 1000
            )

        # Inference time (if tracked)
        if timing.inference_ms > 0:
            REQUEST_DURATION.labels(model=model, endpoint=endpoint, phase="inference").observe(
                timing.inference_ms / 1000
            )

    # Emit structured log for observability (Loki, etc.)
    log_extra: dict[str, object] = {"model": model, "endpoint": endpoint, "status": status}
    if request_id is not None:
        log_extra["request_id"] = request_id
    if api_key is not None:
        log_extra["api_key"] = api_key
    if queue_depth is not None:
        log_extra["queue_depth"] = queue_depth
    if timing is not None:
        log_extra["latency_ms"] = timing.total_ms
        log_extra["tokenization_ms"] = timing.tokenization_ms
        log_extra["queue_ms"] = timing.queue_ms
        log_extra["inference_ms"] = timing.inference_ms
    logger.debug("Request completed", extra=log_extra)


def record_batch(model: str, batch_size: int, tokens: int) -> None:
    """Record metrics for a processed batch.

    Args:
        model: Model name.
        batch_size: Number of items in the batch.
        tokens: Total tokens in the batch.
    """
    BATCH_SIZE.labels(model=model).observe(batch_size)
    TOKENS_PROCESSED.labels(model=model).inc(tokens)


def set_queue_depth(model: str, depth: int) -> None:
    """Update the queue depth gauge for a model.

    Args:
        model: Model name.
        depth: Current queue depth.
    """
    QUEUE_DEPTH.labels(model=model).set(depth)


def set_model_loaded(model: str, device: str, loaded: bool) -> None:
    """Update the model loaded gauge.

    Args:
        model: Model name.
        device: Device the model is loaded on.
        loaded: Whether the model is loaded.
    """
    MODEL_LOADED.labels(model=model, device=device).set(1 if loaded else 0)


def set_model_memory(model: str, device: str, memory_bytes: int) -> None:
    """Update the model memory gauge.

    Args:
        model: Model name.
        device: Device the model is loaded on.
        memory_bytes: Estimated GPU memory usage in bytes.
    """
    MODEL_MEMORY_BYTES.labels(model=model, device=device).set(memory_bytes)


def record_oom_recovery_event(
    model: str,
    *,
    action: str | None = None,
    attempted: bool = False,
    succeeded: bool = False,
    terminal: bool = False,
) -> None:
    """Bump the appropriate OOM recovery counters for ``model``.

    Centralises the prom-metric writes so ``BatchExecutor`` doesn't have
    to know about the prometheus client. Multiple flags can be set in a
    single call (e.g., ``attempted=True`` plus ``succeeded=True``) to
    record both events for the same recovery cycle.

    Args:
        model: Model name (becomes the ``model`` label).
        action: Strategy action name when bumping a per-strategy counter.
            One of ``"cache_clear"``, ``"evict_lru"``, or ``"split_batch"``;
            ``None`` skips the per-strategy bump.
        attempted: Bump ``sie_oom_recoveries_attempted_total``.
        succeeded: Bump ``sie_oom_recoveries_succeeded_total``.
        terminal: Bump ``sie_oom_terminal_failures_total``.
    """
    if attempted:
        OOM_RECOVERIES_ATTEMPTED.labels(model=model).inc()
    if succeeded:
        OOM_RECOVERIES_SUCCEEDED.labels(model=model).inc()
    if terminal:
        OOM_TERMINAL_FAILURES.labels(model=model).inc()
    if action is None:
        return
    if action == "cache_clear":
        OOM_CACHE_CLEARS.labels(model=model).inc()
    elif action == "evict_lru":
        OOM_EVICTIONS_TRIGGERED.labels(model=model).inc()
    elif action == "split_batch":
        OOM_BATCH_SPLITS.labels(model=model).inc()
    else:
        # Fail fast: silently dropping unknown actions causes per-strategy
        # metrics to drift out of sync with the executor's real behaviour
        # (e.g., a typo at the call site in ``BatchExecutor`` would never
        # surface as a missing-counter alert).
        msg = f"record_oom_recovery_event: unknown action {action!r} for model {model!r}"
        raise ValueError(msg)


def record_idle_eviction(model: str) -> None:
    """Bump ``sie_idle_evictions_total`` for ``model``.

    Called from the registry's idle-evict loop after a successful unload.
    """
    IDLE_EVICTIONS_TOTAL.labels(model=model).inc()
