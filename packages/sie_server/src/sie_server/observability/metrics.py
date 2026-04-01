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
