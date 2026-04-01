"""OpenTelemetry tracing setup for SIE Server.

Provides distributed tracing via OpenTelemetry with OTLP export.
Disabled by default; enabled via SIE_TRACING_ENABLED=true or --tracing flag.

Configuration via standard OTel environment variables:
- OTEL_SERVICE_NAME: Service name (default: sie-server)
- OTEL_TRACES_EXPORTER: Exporter type (otlp, console, none)
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
- OTEL_TRACES_SAMPLER: Sampling strategy
- OTEL_TRACES_SAMPLER_ARG: Sampling rate

See DESIGN.md Section 5.6 for observability design.
See https://opentelemetry.io/docs/concepts/sdk-configuration/ for OTel config.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Module-level tracer for manual spans in endpoint handlers
# This is safe to import even when tracing is disabled (returns no-op tracer)
tracer = trace.get_tracer("sie_server")


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled via environment variable."""
    return os.environ.get("SIE_TRACING_ENABLED", "").lower() in ("true", "1", "yes")


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
        No-op if SIE_TRACING_ENABLED is not set to true.
        All configuration is via standard OTel environment variables.
    """
    if not is_tracing_enabled():
        logger.debug("Tracing disabled (SIE_TRACING_ENABLED not set)")
        return

    # Import here to avoid loading OTel machinery when tracing is disabled
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.fastapi import (
        FastAPIInstrumentor,
    )

    # Get service name from env (OTel standard) or use default
    service_name = os.environ.get("OTEL_SERVICE_NAME", "sie-server")

    logger.info("Initializing OpenTelemetry tracing for service: %s", service_name)

    # Create resource identifying this service
    resource = Resource.create({SERVICE_NAME: service_name})

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Configure OTLP exporter
    # OTLPSpanExporter reads OTEL_EXPORTER_OTLP_ENDPOINT from environment
    try:
        exporter = OTLPSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        logger.info("OTLP exporter configured for endpoint: %s", endpoint)
    except Exception:
        logger.exception("Failed to configure OTLP exporter")
        return

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Update module-level tracer to use the new provider
    global tracer
    tracer = trace.get_tracer("sie_server")

    # Auto-instrument FastAPI
    # This creates HTTP spans for all requests and propagates W3C Trace Context
    FastAPIInstrumentor.instrument_app(app)

    logger.info("OpenTelemetry tracing initialized successfully")


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
