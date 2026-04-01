"""Observability module for SIE Server.

Provides Prometheus metrics and OpenTelemetry tracing.
"""

from sie_server.observability.metrics import (
    BATCH_SIZE,
    MODEL_LOADED,
    MODEL_MEMORY_BYTES,
    QUEUE_DEPTH,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    TOKENS_PROCESSED,
    record_request,
    set_model_loaded,
    set_model_memory,
)

__all__ = [
    "BATCH_SIZE",
    "MODEL_LOADED",
    "MODEL_MEMORY_BYTES",
    "QUEUE_DEPTH",
    "REQUESTS_TOTAL",
    "REQUEST_DURATION",
    "TOKENS_PROCESSED",
    "record_request",
    "set_model_loaded",
    "set_model_memory",
]
