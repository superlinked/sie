"""Tests for OpenTelemetry tracing."""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from sie_server.observability.tracing import (
    _build_span_exporter,
    _endpoint_origin_for_log,
    _OtlpTraceExportConfig,
    _trace_export_config_from_values,
    get_current_trace_id,
    is_tracing_enabled,
    setup_tracing,
    shutdown_tracing,
    tracer,
)


def test_endpoint_log_origin_redacts_credentials_path_and_query() -> None:
    raw = "https://user:secret@collector.example:4318/v1/traces?token=private#fragment"
    assert _endpoint_origin_for_log(raw) == "https://collector.example:4318"
    assert _endpoint_origin_for_log("not a URL with secret") == "<redacted>"


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

    def test_enabled_with_whitespace_padding(self) -> None:
        """A whitespace-padded truthy value should still enable tracing."""
        with patch.dict(os.environ, {"SIE_TRACING_ENABLED": " true "}):
            assert is_tracing_enabled() is True

    def test_disabled_with_whitespace_only(self) -> None:
        """A whitespace-only value should be treated as unset (disabled)."""
        with patch.dict(os.environ, {"SIE_TRACING_ENABLED": "   "}):
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
            # Should create exporter with a bounded per-request timeout (seconds)
            mock_exporter.assert_called_once_with(endpoint="http://localhost:4317", timeout=3.0)
            # Should set tracer provider
            mock_trace.set_tracer_provider.assert_called_once()

    def test_noop_when_enabled_without_endpoint(self) -> None:
        """setup_tracing should no-op when the flag is set without an endpoint."""
        mock_app = MagicMock()

        with (
            patch.dict(os.environ, {"SIE_TRACING_ENABLED": "true"}, clear=True),
            patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor") as mock_instrumentor,
        ):
            setup_tracing(mock_app)

            mock_instrumentor.instrument_app.assert_not_called()

    def test_noop_when_endpoint_whitespace_only(self) -> None:
        """A whitespace-only endpoint should be treated as absent (no instrumentation)."""
        mock_app = MagicMock()

        with (
            patch.dict(
                os.environ,
                {"SIE_TRACING_ENABLED": "true", "OTEL_EXPORTER_OTLP_ENDPOINT": "   "},
                clear=True,
            ),
            patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor") as mock_instrumentor,
        ):
            setup_tracing(mock_app)

            mock_instrumentor.instrument_app.assert_not_called()

    def test_invalid_protocol_is_operator_visible_and_fail_open(self, caplog: pytest.LogCaptureFixture) -> None:
        mock_app = MagicMock()
        with (
            patch.dict(
                os.environ,
                {
                    "SIE_TRACING_ENABLED": "true",
                    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4318",
                    "OTEL_EXPORTER_OTLP_PROTOCOL": "http/json",
                },
                clear=True,
            ),
            patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor") as mock_instrumentor,
        ):
            setup_tracing(mock_app)
            mock_instrumentor.instrument_app.assert_not_called()

        assert "Invalid OTLP trace exporter configuration" in caplog.text
        assert "http/json" not in caplog.text

    def test_exporter_setup_failure_does_not_log_credential_bearing_endpoint(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "credential-bearing-path-and-query"  # noqa: S105 - leak-detection sentinel
        endpoint = f"https://collector.example/{secret}/v1/traces?token={secret}"
        mock_app = MagicMock()
        with (
            patch.dict(
                os.environ,
                {
                    "SIE_TRACING_ENABLED": "true",
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": endpoint,
                    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
                },
                clear=True,
            ),
            patch("sie_server.observability.tracing._build_span_exporter", side_effect=RuntimeError(endpoint)),
            patch("opentelemetry.instrumentation.fastapi.FastAPIInstrumentor") as mock_instrumentor,
        ):
            setup_tracing(mock_app)
            mock_instrumentor.instrument_app.assert_not_called()

        assert "error_type=RuntimeError" in caplog.text
        assert secret not in caplog.text


def test_trace_export_config_keeps_specific_endpoint_and_paths_generic_http_base() -> None:
    assert _trace_export_config_from_values(
        True,
        "https://trace.example/custom",
        "https://generic.example",
        "http/protobuf",
        "grpc",
    ) == _OtlpTraceExportConfig(endpoint="https://trace.example/custom", protocol="http")
    assert _trace_export_config_from_values(
        True,
        None,
        "https://collector.example/",
        None,
        "http/protobuf",
    ) == _OtlpTraceExportConfig(endpoint="https://collector.example/v1/traces", protocol="http")


class _TraceCaptureHandler(BaseHTTPRequestHandler):
    request_path = ""
    request_body = b""
    request_headers: ClassVar[dict[str, str]] = {}

    def do_POST(self) -> None:
        type(self).request_path = self.path
        type(self).request_headers = {key.lower(): value for key, value in self.headers.items()}
        type(self).request_body = self.rfile.read(int(self.headers["content-length"]))
        self.send_response(200)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        del format, args


def test_generic_only_http_trace_config_posts_to_standard_trace_path() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _TraceCaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        config = _trace_export_config_from_values(True, None, base, None, "http/protobuf")
        assert config is not None
        provider = TracerProvider(shutdown_on_exit=False)
        provider.add_span_processor(SimpleSpanProcessor(_build_span_exporter(config)))
        with provider.get_tracer("trace-path-capture").start_as_current_span("capture"):
            pass
        assert provider.force_flush(timeout_millis=5_000)
        assert _TraceCaptureHandler.request_path == "/v1/traces"
        provider.shutdown()
    finally:
        server.shutdown()
        thread.join(timeout=5)


class TestShutdownTracing:
    """Tests for shutdown_tracing function."""

    def test_noop_when_not_setup(self) -> None:
        """shutdown_tracing should be a no-op (no error) when no provider is retained."""
        with patch("sie_server.observability.tracing._provider", None):
            shutdown_tracing()  # should not raise

    def test_shuts_down_and_clears_provider(self) -> None:
        """shutdown_tracing should shut the retained provider down, then clear the handle."""
        mock_provider = MagicMock()
        with patch("sie_server.observability.tracing._provider", mock_provider):
            shutdown_tracing()
            # The handle is cleared after the first call, so a second call is a no-op.
            shutdown_tracing()

            mock_provider.shutdown.assert_called_once()


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
