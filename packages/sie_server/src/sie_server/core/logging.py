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
    # "api_key" was listed here until #2339: no call site ever set it, and a
    # raw credential must never be a pass-through log field. Log a masked
    # token via sie_sdk.redaction.mask_token under a non-secret field instead.
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


# The single source of truth for both validation and resolution. Deliberately
# narrower than ``logging.getLevelNamesMapping()``, which also carries NOTSET
# and the deprecated WARN/FATAL aliases: validating against the wider set while
# advertising this one would accept names the CLI help does not document, and
# NOTSET on the root logger means "inherit", not a severity.
_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# A rejected value is arbitrary caller/environment input, so it is truncated and
# stripped of anything but plain identifier characters before being logged — a
# level name never needs more, and an operator who accidentally pointed
# SIE_LOG_LEVEL at a secret must not have it copied into the log sink.
_REJECTED_MAX_LEN = 20


def valid_log_levels() -> list[str]:
    """Accepted level names, in ascending severity order."""
    return list(_LEVELS)


def is_valid_log_level(name: str) -> bool:
    """Whether ``name`` is one of the advertised levels (case-insensitive)."""
    return name.strip().upper() in _LEVELS


def _safe_for_log(value: str) -> str:
    """Render a rejected level name without echoing arbitrary input."""
    cleaned = "".join(char if char.isalnum() or char in "-_" else "?" for char in value.strip())
    if len(cleaned) > _REJECTED_MAX_LEN:
        return cleaned[:_REJECTED_MAX_LEN] + "..."
    return cleaned or "(empty)"


def _resolve_log_level(*, verbose: bool, level_name: str | None) -> tuple[int, str | None]:
    """Pick root log level: ``--verbose`` wins, then explicit name, then ``SIE_LOG_LEVEL`` env.

    Returns:
        ``(level, rejected_name)``. ``rejected_name`` is the caller's value when
        it is not an advertised level, so :func:`configure_logging` can say so
        once logging is actually running. Silently defaulting to INFO is the
        worst outcome: an operator who set ``DEBUG`` and mistyped it sees a
        normal-looking server with none of the output they asked for, and no
        indication why.
    """
    if verbose:
        return logging.DEBUG, None
    raw = (level_name or os.environ.get("SIE_LOG_LEVEL") or "INFO").strip()
    resolved = _LEVELS.get(raw.upper())
    if resolved is None:
        return logging.INFO, raw
    return resolved, None


def configure_logging(
    *,
    verbose: bool = False,
    json_format: bool | None = None,
    level_name: str | None = None,
) -> None:
    """Configure logging for SIE server.

    Args:
        verbose: Enable DEBUG level logging (overrides ``level_name`` / ``SIE_LOG_LEVEL``).
        json_format: Use JSON format. If None, reads from SIE_LOG_JSON env var.
        level_name: Log level name (e.g. ``DEBUG``, ``INFO``). When None, uses ``SIE_LOG_LEVEL``.
    """
    log_level, rejected = _resolve_log_level(verbose=verbose, level_name=level_name)

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

    # Emitted only now, with handlers installed, so the warning is actually
    # visible. WARNING is above the INFO fallback, so it survives the very
    # downgrade it is reporting.
    if rejected is not None:
        logging.getLogger(__name__).warning(
            "Ignoring unknown log level %r; falling back to INFO. Valid levels: %s.",
            _safe_for_log(rejected),
            ", ".join(valid_log_levels()),
        )

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
