"""Sampled, bounded lifecycle records; never a bridge from Python logging.

The trace's head-sampling decision also controls these logs, including failures.
No exception objects, strings, payloads, or caller-defined fields are accepted.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry._logs import LogRecord, SeverityNumber
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter as GrpcLogExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter as HttpLogExporter
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, LogRecordExporter
from opentelemetry.sdk.resources import Resource

from sie_server.observability.worker_telemetry import worker_resource_attributes

_logger = logging.getLogger(__name__)
_provider: LoggerProvider | None = None
_current: ContextVar[Lifecycle | None] = ContextVar("sie_lifecycle", default=None)
_OUTCOMES = frozenset({"success", "rejected", "error", "cancelled"})
_ERROR_CLASSES = frozenset(
    {"none", "client_error", "server_error", "transport", "timeout", "worker", "cancelled", "protocol", "other"}
)


def configure_lifecycle_logs(exporter: LogRecordExporter, resource: Resource) -> None:
    """Install a dedicated safe logger; deployment owns transport credentials."""
    global _provider
    if _provider is not None:
        return
    provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    _provider = provider


def setup_lifecycle_logs() -> None:
    """OSS transport setup. Signals are independently gated and fail open."""
    if _provider is not None or os.getenv("SIE_OTLP_LOGS_ENABLED", "").strip().lower() not in {"1", "true", "yes"}:
        return
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT", "").strip()
    generic = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    protocol = (
        os.getenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "").strip()
        or os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").strip()
    ).lower()
    if not endpoint and not generic:
        return
    if protocol not in {"grpc", "http/protobuf"}:
        _logger.warning("Unsupported safe lifecycle log transport; export disabled")
        return
    if not endpoint:
        endpoint = generic if protocol == "grpc" else f"{generic.rstrip('/')}/v1/logs"
    try:
        exporter = (
            GrpcLogExporter(endpoint=endpoint, timeout=3)
            if protocol == "grpc"
            else HttpLogExporter(endpoint=endpoint, timeout=3)
        )
        configure_lifecycle_logs(exporter, Resource(worker_resource_attributes()))
    except Exception:  # noqa: BLE001 - telemetry must not prevent serving
        _logger.warning("Safe lifecycle log exporter setup failed; export disabled")


def shutdown_lifecycle_logs() -> None:
    global _provider
    provider, _provider = _provider, None
    if provider is not None:
        try:
            provider.shutdown()
        except Exception:  # noqa: BLE001 - best effort telemetry teardown
            _logger.warning("Safe lifecycle log exporter shutdown failed")


@dataclass
class Lifecycle:
    span: trace.Span
    started: float
    outcome: str = "error"
    error_class: str = "transport"
    terminal: bool = False

    def published_terminal(self, envelope: dict[str, object]) -> None:
        """Terminal means successful transport publication, not client receipt."""
        if self.terminal:
            return
        self.terminal = True
        if envelope.get("finish_reason") == "cancelled":
            self.outcome, self.error_class = "cancelled", "cancelled"
        elif envelope.get("error") is not None or envelope.get("finish_reason") == "error":
            self.outcome, self.error_class = "error", "worker"
        else:
            self.outcome, self.error_class = "success", "none"

    def finish(self) -> None:
        if self.outcome == "error":
            self.span.set_status(trace.StatusCode.ERROR)
        context = self.span.get_span_context()
        if _provider is None or not context.is_valid or not context.trace_flags.sampled:
            return
        _provider.get_logger("sie.lifecycle").emit(
            LogRecord(
                timestamp=time.time_ns(),
                context=trace.set_span_in_context(self.span),
                severity_text="INFO",
                severity_number=SeverityNumber.INFO,
                body="inference.lifecycle.completed",
                attributes={
                    "event.name": "inference.lifecycle.completed",
                    "event.schema.version": "1",
                    "phase": "worker_generation",
                    "operation": "generate",
                    "outcome": self.outcome if self.outcome in _OUTCOMES else "error",
                    "error_class": self.error_class if self.error_class in _ERROR_CLASSES else "other",
                    "duration_ms": max(0.0, (time.perf_counter() - self.started) * 1000),
                },
            )
        )


def current_lifecycle() -> Lifecycle | None:
    return _current.get()


@contextmanager
def observe_generation() -> Iterator[None]:
    span = trace.get_current_span()
    if not span.is_recording() and _provider is None:
        yield
        return
    lifecycle = Lifecycle(span, time.perf_counter())
    token = _current.set(lifecycle)
    try:
        yield
    except asyncio.CancelledError:
        lifecycle.outcome, lifecycle.error_class = "cancelled", "cancelled"
        raise
    except Exception:
        lifecycle.outcome, lifecycle.error_class = "error", "worker"
        raise
    finally:
        lifecycle.finish()
        _current.reset(token)
