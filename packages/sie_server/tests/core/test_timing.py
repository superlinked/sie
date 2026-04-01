"""Tests for request timing utilities."""

import time

from sie_server.core.timing import RequestTiming


class TestRequestTiming:
    """Tests for RequestTiming dataclass."""

    def test_basic_timing(self) -> None:
        """Can track basic request timing."""
        timing = RequestTiming()

        # Simulate tokenization
        timing.start_tokenization()
        time.sleep(0.01)  # 10ms
        timing.end_tokenization()

        # Simulate queue wait
        timing.start_queue()
        time.sleep(0.005)  # 5ms

        # Simulate inference
        timing.start_inference()
        time.sleep(0.02)  # 20ms
        timing.end_inference()

        timing.finish()

        # Check timing values are reasonable (>= expected, with some tolerance)
        assert timing.tokenization_ms >= 10.0
        assert timing.queue_ms >= 5.0
        assert timing.inference_ms >= 20.0
        assert timing.total_ms >= 35.0

    def test_to_headers(self) -> None:
        """Can convert timing to HTTP headers."""
        timing = RequestTiming()

        timing.start_tokenization()
        timing.end_tokenization()
        timing.start_queue()
        timing.start_inference()
        timing.end_inference()
        timing.finish()

        headers = timing.to_headers()

        assert "X-Queue-Time" in headers
        assert "X-Tokenization-Time" in headers
        assert "X-Inference-Time" in headers
        assert "X-Total-Time" in headers

        # Values should be formatted as strings with 2 decimal places
        for value in headers.values():
            assert isinstance(value, str)
            # Should be a valid float string
            float(value)

    def test_missing_timing_returns_zero(self) -> None:
        """Missing timing phases return 0.0."""
        timing = RequestTiming()

        # No timing recorded
        assert timing.tokenization_ms == 0.0
        assert timing.queue_ms == 0.0
        assert timing.inference_ms == 0.0

        # Total still works (from start to now)
        assert timing.total_ms > 0.0

    def test_partial_timing(self) -> None:
        """Partial timing (start but no end) returns 0.0."""
        timing = RequestTiming()

        timing.start_tokenization()
        # No end_tokenization called
        assert timing.tokenization_ms == 0.0

        timing.start_queue()
        # No start_inference called
        assert timing.queue_ms == 0.0

    def test_total_ms_before_finish(self) -> None:
        """total_ms works before finish() is called."""
        timing = RequestTiming()
        time.sleep(0.005)  # 5ms

        # Should use current time as end
        assert timing.total_ms >= 5.0

    def test_header_values_format(self) -> None:
        """Header values are formatted with 2 decimal places."""
        timing = RequestTiming()
        timing.start_tokenization()
        time.sleep(0.001)
        timing.end_tokenization()
        timing.finish()

        headers = timing.to_headers()

        # All headers should have format X.XX
        for value in headers.values():
            # Check it's a number with decimal point
            parts = value.split(".")
            assert len(parts) == 2
            assert len(parts[1]) == 2  # 2 decimal places
