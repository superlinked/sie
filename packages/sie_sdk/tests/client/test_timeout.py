import time
from unittest.mock import MagicMock, patch

import httpx
import msgpack
import numpy as np
import pytest
from sie_sdk import ModelLoadingError, ProvisioningError, SIEClient, SIEConnectionError
from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE


class TestTimeoutEnforcement:
    """Tests for timeout enforcement safeguards.

    These tests verify that the provision_timeout_s is enforced even when
    individual HTTP requests might block for longer than expected.
    """

    def test_per_request_timeout_capped_to_remaining_provision_time(self) -> None:
        """Per-request timeout is capped to remaining provision time.

        This prevents a single hanging request from exceeding the overall
        provision_timeout_s. The safeguard was added to prevent scenarios where
        httpx timeout > provision_timeout, causing requests to block indefinitely.
        """
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.01", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-type": "application/msgpack"}
        mock_response_200.content = msgpack.packb(
            {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[mock_response_503, mock_response_200])

            # Create client with LONG httpx timeout (300s)
            # but SHORT provision_timeout (0.5s)
            client = SIEClient("http://localhost:8080", timeout_s=300.0)
            client.encode("bge-m3", {"text": "hello"}, provision_timeout_s=0.5)

            # Verify that per-request timeout was passed to httpx
            # (should be capped to remaining provision time, not 300s)
            calls = mock_client.return_value.post.call_args_list
            for call in calls:
                # Every phase of the timeout should be present and <= provision_timeout_s
                timeout_used = call.kwargs.get("timeout")
                assert isinstance(timeout_used, httpx.Timeout), "Per-request timeout should be set"
                for phase in (timeout_used.connect, timeout_used.read, timeout_used.write, timeout_used.pool):
                    assert phase is not None
                    assert phase <= 0.5, f"Request timeout {phase}s should be <= provision_timeout 0.5s"

            client.close()

    def test_provision_timeout_enforced_across_retries(self) -> None:
        """Provision timeout is enforced across multiple retries.

        Even if individual requests complete quickly (returning 503 MODEL_LOADING),
        the cumulative wall-clock time is tracked and the operation times out
        after provision_timeout_s.
        """
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.01", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            # Always return 503 MODEL_LOADING (model never finishes loading)
            mock_client.return_value.post = MagicMock(return_value=mock_response_503)
            client = SIEClient("http://localhost:8080")

            start_time = time.monotonic()
            provision_timeout = 0.01  # 100ms - enough for several retries

            # Either ModelLoadingError (timeout during retry) or ProvisioningError
            # (timeout check before request) are valid timeout behaviors
            with pytest.raises((ModelLoadingError, ProvisioningError)):
                client.encode("bge-m3", {"text": "hello"}, provision_timeout_s=provision_timeout)

            elapsed = time.monotonic() - start_time

            # The operation should complete within a reasonable margin of the timeout
            # Allow some overhead for test execution
            assert elapsed < provision_timeout + 0.15, (
                f"Operation took {elapsed:.2f}s, should timeout around {provision_timeout}s"
            )

            client.close()

    def test_httpx_timeout_exception_respects_provision_timeout(self) -> None:
        """A timeout after the request was sent fails fast without a retry."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=httpx.TimeoutException("Request timed out"))
            client = SIEClient("http://localhost:8080")

            start_time = time.monotonic()
            with pytest.raises(SIEConnectionError, match="timed out"):
                client.encode("bge-m3", {"text": "hello"}, provision_timeout_s=0.15, wait_for_capacity=False)

            elapsed = time.monotonic() - start_time
            assert elapsed < 0.5, f"Should fail quickly, took {elapsed:.2f}s"
            assert mock_client.return_value.post.call_count == 1

            client.close()

    @pytest.mark.parametrize(
        "exc",
        [httpx.ReadTimeout("read timed out"), httpx.TimeoutException("timed out")],
        ids=["read_timeout", "unknown_phase_timeout"],
    )
    def test_sent_request_timeout_not_retried_with_wait_for_capacity(self, exc: httpx.TimeoutException) -> None:
        """A request that may already be running is never re-sent after a timeout."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=exc)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError, match="timed out") as exc_info:
                client.encode("bge-m3", {"text": "hello"}, wait_for_capacity=True, provision_timeout_s=10.0)

            assert "Not retried" in str(exc_info.value)
            assert mock_client.return_value.post.call_count == 1
            assert client.last_retry_count == 0
            client.close()

    @pytest.mark.parametrize(
        "exc",
        [httpx.ConnectTimeout("connect timed out"), httpx.PoolTimeout("pool timed out")],
        ids=["connect_timeout", "pool_timeout"],
    )
    def test_timeout_before_send_retried_with_wait_for_capacity(self, exc: httpx.TimeoutException) -> None:
        """A timeout before the request reached the server is safe to retry."""
        ok = MagicMock()
        ok.status_code = 200
        ok.headers = {"content-type": "application/msgpack"}
        ok.content = msgpack.packb({"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True)
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            mock_client.return_value.post = MagicMock(side_effect=[exc, ok])
            client = SIEClient("http://localhost:8080")

            result = client.encode("bge-m3", {"text": "hello"}, wait_for_capacity=True, provision_timeout_s=10.0)

            assert result["dense"].shape == (4,)
            assert mock_client.return_value.post.call_count == 2
            assert client.last_retry_count == 1
            client.close()
