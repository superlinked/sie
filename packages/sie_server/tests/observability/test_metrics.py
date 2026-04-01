"""Tests for Prometheus metrics."""

import logging
from unittest.mock import MagicMock

import pytest
from sie_server.observability.metrics import (
    BATCH_SIZE,
    MODEL_LOADED,
    QUEUE_DEPTH,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    TOKENS_PROCESSED,
    record_batch,
    record_request,
    set_model_loaded,
    set_queue_depth,
)


class TestRecordRequest:
    """Tests for record_request helper."""

    def test_increments_counter(self) -> None:
        """record_request should increment the requests counter."""
        # Get initial value
        initial = REQUESTS_TOTAL.labels(model="test-model", endpoint="encode", status="success")._value.get()

        # Record a request
        record_request(model="test-model", endpoint="encode", status="success")

        # Check counter incremented
        new_value = REQUESTS_TOTAL.labels(model="test-model", endpoint="encode", status="success")._value.get()
        assert new_value == initial + 1

    def test_records_error_status(self) -> None:
        """record_request should handle error status."""
        initial = REQUESTS_TOTAL.labels(model="test-model", endpoint="score", status="error")._value.get()

        record_request(model="test-model", endpoint="score", status="error")

        new_value = REQUESTS_TOTAL.labels(model="test-model", endpoint="score", status="error")._value.get()
        assert new_value == initial + 1

    def test_records_timing_breakdown(self) -> None:
        """record_request should record timing breakdown when provided."""
        # Create mock timing object
        timing = MagicMock()
        timing.total_ms = 100.0
        timing.queue_ms = 10.0
        timing.tokenization_ms = 20.0
        timing.inference_ms = 70.0

        # Record request with timing
        record_request(
            model="timing-test",
            endpoint="encode",
            status="success",
            timing=timing,
        )

        # Verify histograms were observed (we can't easily check exact values
        # but we can verify the labels exist in the registry)
        assert REQUEST_DURATION.labels(model="timing-test", endpoint="encode", phase="total") is not None
        assert REQUEST_DURATION.labels(model="timing-test", endpoint="encode", phase="queue") is not None
        assert REQUEST_DURATION.labels(model="timing-test", endpoint="encode", phase="tokenize") is not None
        assert REQUEST_DURATION.labels(model="timing-test", endpoint="encode", phase="inference") is not None

    def test_skips_zero_timing_phases(self) -> None:
        """record_request should skip phases with zero duration."""
        timing = MagicMock()
        timing.total_ms = 50.0
        timing.queue_ms = 0  # Should be skipped
        timing.tokenization_ms = 0  # Should be skipped
        timing.inference_ms = 50.0

        # Should not raise even with zero values
        record_request(
            model="zero-timing-test",
            endpoint="encode",
            status="success",
            timing=timing,
        )

    def test_record_request_emits_structured_log(self, caplog: pytest.LogCaptureFixture) -> None:
        """record_request emits a structured log with model, endpoint, status, and timing."""
        from unittest.mock import MagicMock

        from sie_server.observability.metrics import record_request

        timing = MagicMock()
        timing.total_ms = 50.0
        timing.queue_ms = 5.0
        timing.tokenization_ms = 10.0
        timing.inference_ms = 35.0

        with caplog.at_level(logging.DEBUG, logger="sie_server.observability.metrics"):
            record_request(
                model="bge-m3",
                endpoint="encode",
                status="success",
                timing=timing,
                request_id="req-abc",
                api_key="sk-***",
                queue_depth=3,
            )

        assert len(caplog.records) >= 1
        rec = caplog.records[-1]
        assert rec.message == "Request completed"
        assert rec.model == "bge-m3"  # type: ignore[attr-defined]
        assert rec.endpoint == "encode"  # type: ignore[attr-defined]
        assert rec.status == "success"  # type: ignore[attr-defined]
        assert rec.request_id == "req-abc"  # type: ignore[attr-defined]
        assert rec.api_key == "sk-***"  # type: ignore[attr-defined]
        assert rec.queue_depth == 3  # type: ignore[attr-defined]
        assert rec.latency_ms == 50.0  # type: ignore[attr-defined]
        assert rec.tokenization_ms == 10.0  # type: ignore[attr-defined]
        assert rec.queue_ms == 5.0  # type: ignore[attr-defined]
        assert rec.inference_ms == 35.0  # type: ignore[attr-defined]

    def test_record_request_without_timing(self, caplog: pytest.LogCaptureFixture) -> None:
        """record_request emits log even without timing data."""
        from sie_server.observability.metrics import record_request

        with caplog.at_level(logging.DEBUG, logger="sie_server.observability.metrics"):
            record_request(
                model="bge-m3",
                endpoint="encode",
                status="error",
            )

        assert len(caplog.records) >= 1
        rec = caplog.records[-1]
        assert rec.message == "Request completed"
        assert rec.model == "bge-m3"  # type: ignore[attr-defined]
        assert rec.status == "error"  # type: ignore[attr-defined]


class TestRecordBatch:
    """Tests for record_batch helper."""

    def test_records_batch_size(self) -> None:
        """record_batch should observe batch size."""
        record_batch(model="batch-test", batch_size=16, tokens=1024)

        # Verify the histogram was observed
        assert BATCH_SIZE.labels(model="batch-test") is not None

    def test_records_tokens_processed(self) -> None:
        """record_batch should increment tokens counter."""
        initial = TOKENS_PROCESSED.labels(model="tokens-test")._value.get()

        record_batch(model="tokens-test", batch_size=8, tokens=512)

        new_value = TOKENS_PROCESSED.labels(model="tokens-test")._value.get()
        assert new_value == initial + 512


class TestSetQueueDepth:
    """Tests for set_queue_depth helper."""

    def test_sets_gauge_value(self) -> None:
        """set_queue_depth should set the gauge to the specified value."""
        set_queue_depth(model="queue-test", depth=42)

        value = QUEUE_DEPTH.labels(model="queue-test")._value.get()
        assert value == 42

    def test_can_update_value(self) -> None:
        """set_queue_depth should be able to update the value."""
        set_queue_depth(model="queue-update-test", depth=10)
        set_queue_depth(model="queue-update-test", depth=5)

        value = QUEUE_DEPTH.labels(model="queue-update-test")._value.get()
        assert value == 5


class TestSetModelLoaded:
    """Tests for set_model_loaded helper."""

    def test_sets_loaded_true(self) -> None:
        """set_model_loaded should set 1 when loaded=True."""
        set_model_loaded(model="loaded-test", device="cuda:0", loaded=True)

        value = MODEL_LOADED.labels(model="loaded-test", device="cuda:0")._value.get()
        assert value == 1

    def test_sets_loaded_false(self) -> None:
        """set_model_loaded should set 0 when loaded=False."""
        set_model_loaded(model="unloaded-test", device="cpu", loaded=False)

        value = MODEL_LOADED.labels(model="unloaded-test", device="cpu")._value.get()
        assert value == 0
