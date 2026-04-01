"""Structured JSON logging for SIE Server.

Provides structured JSON log output for Loki/observability stack compatibility.

Format:
    {"timestamp": "2025-12-18T10:30:00Z", "level": "INFO", "model": "bge-m3",
     "request_id": "abc123", "trace_id": "def456", "message": "Inference completed",
     "latency_ms": 45.2}
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any

OPTIONAL_FIELDS = (
    "model",
    "request_id",
    "trace_id",
    "latency_ms",
    "batch_size",
    "gpu_type",
    "endpoint",
    "api_key",
    "queue_depth",
    "status",
    "tokenization_ms",
    "queue_ms",
    "inference_ms",
)


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs.

    Includes optional fields: model, request_id, trace_id, latency_ms.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add optional structured fields if present
        log_data |= {field: value for field in OPTIONAL_FIELDS if (value := getattr(record, field, None)) is not None}

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Standard text formatter for development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


def configure_logging(*, verbose: bool = False, json_format: bool | None = None) -> None:
    """Configure logging for SIE server.

    Args:
        verbose: Enable DEBUG level logging.
        json_format: Use JSON format. If None, reads from SIE_LOG_JSON env var.
    """
    # Determine log level
    log_level = logging.DEBUG if verbose else logging.INFO

    # Determine format (env var takes precedence if json_format not explicitly set)
    if json_format is None:
        json_format = os.environ.get("SIE_LOG_JSON", "").lower() in ("true", "1", "yes")

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicate logs
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    root_logger.addHandler(handler)

    # Set sie_server modules to appropriate level
    logging.getLogger("sie_server").setLevel(log_level)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
