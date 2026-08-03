"""OpenTelemetry tracing setup for SIE Server.

Provides distributed tracing via OpenTelemetry with OTLP export.
Disabled by default; export requires SIE_TRACING_ENABLED=true (or --tracing)
and an explicit OTLP endpoint.

Configuration via standard OTel environment variables:
- service.name is fixed by the telemetry contract to sie-worker
- OTEL_TRACES_EXPORTER: Exporter type (otlp, console, none)
- OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: Trace-specific OTLP endpoint URL
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
- OTEL_EXPORTER_OTLP_TRACES_PROTOCOL: Trace-specific grpc or http/protobuf
- OTEL_EXPORTER_OTLP_PROTOCOL: Generic protocol fallback
- OTEL_TRACES_SAMPLER: Sampling strategy
- OTEL_TRACES_SAMPLER_ARG: Sampling rate

See https://opentelemetry.io/docs/concepts/sdk-configuration/ for OTel config.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from sie_server.observability.worker_telemetry import worker_resource_attributes

if TYPE_CHECKING:
    from fastapi import FastAPI
    from opentelemetry.sdk.trace.export import SpanExporter

logger = logging.getLogger(__name__)


def _endpoint_origin_for_log(endpoint: str) -> str:
    """Return a credential- and query-free endpoint origin for diagnostics."""
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        return "<redacted>"
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return "<redacted>"
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{parsed.scheme}://{host}{f':{port}' if port is not None else ''}"


# Module-level tracer for manual spans in endpoint handlers
# This is safe to import even when tracing is disabled (returns no-op tracer)
tracer = trace.get_tracer("sie_server")

# Bounded flush deadline (ms) so process exit can't stall on an unreachable collector.
TRACING_SHUTDOWN_TIMEOUT_MS = 3000

# Retained provider handle so shutdown_tracing() can flush deterministically.
_provider: TracerProvider | None = None


@dataclass(frozen=True)
class _OtlpTraceExportConfig:
    endpoint: str
    protocol: str


def _trace_export_config_from_values(
    enabled: bool,
    traces_endpoint: str | None,
    generic_endpoint: str | None,
    traces_protocol: str | None,
    generic_protocol: str | None,
) -> _OtlpTraceExportConfig | None:
    if not enabled:
        return None
    signal_protocol = (traces_protocol or "").strip()
    fallback_protocol = (generic_protocol or "").strip()
    raw_protocol = (signal_protocol or fallback_protocol or "grpc").lower()
    if raw_protocol == "grpc":
        protocol = "grpc"
    elif raw_protocol == "http/protobuf":
        protocol = "http"
    else:
        raise ValueError(f"unsupported OTLP traces protocol {raw_protocol!r}; expected grpc or http/protobuf")

    explicit = (traces_endpoint or "").strip()
    generic = (generic_endpoint or "").strip()
    if explicit:
        return _OtlpTraceExportConfig(endpoint=explicit, protocol=protocol)
    if not generic:
        return None
    endpoint = generic
    if protocol == "http" and not generic.endswith("/v1/traces"):
        endpoint = f"{generic.rstrip('/')}/v1/traces"
    return _OtlpTraceExportConfig(endpoint=endpoint, protocol=protocol)


def _configured_trace_export_config() -> _OtlpTraceExportConfig | None:
    return _trace_export_config_from_values(
        is_tracing_enabled(),
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"),
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"),
        os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL"),
    )


def _build_span_exporter(config: _OtlpTraceExportConfig) -> SpanExporter:
    if config.protocol == "http":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    else:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    return OTLPSpanExporter(endpoint=config.endpoint, timeout=TRACING_SHUTDOWN_TIMEOUT_MS / 1000)


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled via environment variable."""
    return os.environ.get("SIE_TRACING_ENABLED", "").strip().lower() in ("true", "1", "yes")


def setup_tracing(app: FastAPI) -> None:
    """Initialize OpenTelemetry tracing for the FastAPI app.

    This function:
    1. Creates a TracerProvider with service resource
    2. Configures OTLP exporter based on environment variables
    3. Auto-instruments FastAPI for HTTP span creation
    4. Handles W3C Trace Context propagation automatically

    Args:
        app: FastAPI application instance to instrument.

    Note:
        No-op unless SIE_TRACING_ENABLED is true and an OTLP endpoint is set.
        All configuration is via standard OTel environment variables.
    """
    if not is_tracing_enabled():
        logger.debug("Tracing disabled (SIE_TRACING_ENABLED not set)")
        return

    try:
        config = _configured_trace_export_config()
    except ValueError as error:
        logger.warning(
            "Invalid OTLP trace exporter configuration; tracing disabled (error_type=%s)",
            type(error).__name__,
        )
        return
    if config is None:
        logger.warning("SIE_TRACING_ENABLED is set but no OTLP endpoint configured; tracing disabled")
        return

    # Import here to avoid loading OTel machinery when tracing is disabled
    from opentelemetry.instrumentation.fastapi import (
        FastAPIInstrumentor,
    )

    service_name = worker_resource_attributes()["service.name"]

    logger.info("Initializing OpenTelemetry tracing for service: %s", service_name)

    # Create resource identifying this service
    resource = Resource(worker_resource_attributes())

    # Create tracer provider. We own teardown (see shutdown_tracing), so disable
    # the SDK's unbounded atexit shutdown handler.
    provider = TracerProvider(resource=resource, shutdown_on_exit=False)

    # Configure OTLP exporter. The exporter's per-request timeout is the only
    # effective bound on shutdown flushing in the current SDK, so process exit
    # can't stall on an unreachable collector.
    try:
        exporter = _build_span_exporter(config)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info(
            "OTLP exporter configured for endpoint: %s (%s)",
            _endpoint_origin_for_log(config.endpoint),
            config.protocol,
        )
    except Exception as error:  # noqa: BLE001 - telemetry setup must not block serving
        logger.warning("Failed to configure OTLP exporter (error_type=%s)", type(error).__name__)
        return

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Retain the provider so shutdown_tracing() can flush/shut down deterministically.
    global _provider
    _provider = provider

    # Update module-level tracer to use the new provider
    global tracer
    tracer = trace.get_tracer("sie_server")

    # Auto-instrument FastAPI
    # This creates HTTP spans for all requests and propagates W3C Trace Context
    FastAPIInstrumentor.instrument_app(app)

    logger.info("OpenTelemetry tracing initialized successfully")


def shutdown_tracing() -> None:
    """Bounded shutdown/flush of pending spans so exit can't stall on an unreachable collector.

    The bound comes from the exporter's per-request timeout set in
    ``setup_tracing`` — each export attempt inside ``provider.shutdown()`` is
    capped, so this returns promptly even when the collector is down.
    """
    global _provider
    provider = _provider
    if provider is None:
        return
    try:
        provider.shutdown()
    except Exception:  # best-effort; never block process exit on tracing
        logger.exception("Tracing shutdown failed")
    finally:
        _provider = None


def get_current_trace_id() -> str | None:
    """Get the current trace ID as a hex string.

    Returns:
        Trace ID as 32-character hex string, or None if no active span.
    """
    span = trace.get_current_span()
    if span is None:
        return None

    context = span.get_span_context()
    if context is None or not context.is_valid:
        return None

    return format(context.trace_id, "032x")
