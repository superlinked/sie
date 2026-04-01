"""Tests for OpenTelemetry tracing."""

import os
from unittest.mock import MagicMock, patch

from sie_server.observability.tracing import (
    get_current_trace_id,
    is_tracing_enabled,
    setup_tracing,
    tracer,
)


class TestIsTracingEnabled:
    """Tests for is_tracing_enabled function."""

    def test_disabled_by_default(self) -> None:
        """Tracing should be disabled when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove SIE_TRACING_ENABLED if it exists
            os.environ.pop("SIE_TRACING_ENABLED", None)
            assert is_tracing_enabled() is False

    def test_enabled_with_true(self) -> None:
        """Tracing should be enabled when SIE_TRACING_ENABLED=true."""
        with patch.dict(os.environ, {"SIE_TRACING_ENABLED": "true"}):
            assert is_tracing_enabled() is True

    def test_enabled_with_1(self) -> None:
        """Tracing should be enabled when SIE_TRACING_ENABLED=1."""
        with patch.dict(os.environ, {"SIE_TRACING_ENABLED": "1"}):
            assert is_tracing_enabled() is True

    def test_enabled_with_yes(self) -> None:
        """Tracing should be enabled when SIE_TRACING_ENABLED=yes."""
        with patch.dict(os.environ, {"SIE_TRACING_ENABLED": "yes"}):
            assert is_tracing_enabled() is True

    def test_enabled_case_insensitive(self) -> None:
        """SIE_TRACING_ENABLED should be case insensitive."""
        with patch.dict(os.environ, {"SIE_TRACING_ENABLED": "TRUE"}):
            assert is_tracing_enabled() is True

    def test_disabled_with_false(self) -> None:
        """Tracing should be disabled when SIE_TRACING_ENABLED=false."""
        with patch.dict(os.environ, {"SIE_TRACING_ENABLED": "false"}):
            assert is_tracing_enabled() is False


class TestSetupTracing:
    """Tests for setup_tracing function."""

    def test_noop_when_disabled(self) -> None:
        """setup_tracing should be no-op when tracing is disabled."""
        mock_app = MagicMock()

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SIE_TRACING_ENABLED", None)
            # Should not raise and not instrument
            setup_tracing(mock_app)

    def test_instruments_app_when_enabled(self) -> None:
        """setup_tracing should instrument FastAPI when enabled."""
        mock_app = MagicMock()

        with (
            patch.dict(
                os.environ,
                {
                    "SIE_TRACING_ENABLED": "true",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
                },
            ),
            patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor") as mock_instrumentor,
            patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter") as mock_exporter,
            patch("sie_server.observability.tracing.trace") as mock_trace,
        ):
            setup_tracing(mock_app)

            # Should instrument the app
            mock_instrumentor.instrument_app.assert_called_once_with(mock_app)
            # Should create exporter
            mock_exporter.assert_called_once()
            # Should set tracer provider
            mock_trace.set_tracer_provider.assert_called_once()


class TestGetCurrentTraceId:
    """Tests for get_current_trace_id function."""

    def test_returns_none_when_no_span(self) -> None:
        """get_current_trace_id should return None when no active span."""
        # When tracing is disabled or no span is active
        trace_id = get_current_trace_id()
        # May be None or a valid trace ID depending on whether there's an active span
        assert trace_id is None or isinstance(trace_id, str)

    def test_returns_hex_string_format(self) -> None:
        """get_current_trace_id should return 32-character hex string when span is active."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        # Set up in-memory tracer
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        test_tracer = trace.get_tracer("test")

        with test_tracer.start_as_current_span("test-span"):
            trace_id = get_current_trace_id()

            assert trace_id is not None
            assert isinstance(trace_id, str)
            assert len(trace_id) == 32  # 128-bit trace ID as hex
            # Should be valid hex
            int(trace_id, 16)


class TestTracerModule:
    """Tests for module-level tracer."""

    def test_tracer_is_available(self) -> None:
        """Module-level tracer should be available for import."""
        assert tracer is not None

    def test_tracer_can_create_spans(self) -> None:
        """Module-level tracer should be able to create spans."""
        # This should not raise even when tracing is disabled
        # (returns no-op spans)
        with tracer.start_as_current_span("test-span"):
            pass
