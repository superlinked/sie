import json
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
from sie_sdk import ServerError, SIEAsyncClient, SIEClient


class TestProvisioningRetry:
    """Tests for 202 provisioning retry functionality."""

    def test_machine_profile_header_sent_when_gpu_specified(self) -> None:
        """X-SIE-MACHINE-PROFILE header is sent when gpu parameter provided."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/msgpack"}
        mock_response.content = msgpack.packb({"items": [{"dense": {"values": np.zeros(1024)}}]}, use_bin_type=True)

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response)
            client = SIEClient("http://localhost:8080")

            client.encode("bge-m3", {"text": "hello"}, gpu="l4")

            # Check that X-SIE-MACHINE-PROFILE header was sent
            call_args = mock_client.return_value.post.call_args
            assert call_args.kwargs["headers"]["X-SIE-MACHINE-PROFILE"] == "l4"
            client.close()

    def test_202_raises_provisioning_error_without_wait(self) -> None:
        """202 response raises ProvisioningError when wait_for_capacity=False."""
        from sie_sdk import ProvisioningError

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {"Retry-After": "30"}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ProvisioningError) as exc_info:
                client.encode("bge-m3", {"text": "hello"}, gpu="l4", wait_for_capacity=False)

            assert exc_info.value.gpu == "l4"
            assert exc_info.value.retry_after == 30.0
            client.close()

    def test_202_retries_with_wait_for_capacity(self) -> None:
        """202 response is retried when wait_for_capacity=True."""
        # First response is 202, second is 200
        mock_response_202 = MagicMock()
        mock_response_202.status_code = 202
        mock_response_202.headers = {"Retry-After": "0.01"}  # Short delay for test

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-type": "application/msgpack"}
        mock_response_200.content = msgpack.packb({"items": [{"dense": {"values": np.zeros(1024)}}]}, use_bin_type=True)

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[mock_response_202, mock_response_200])
            client = SIEClient("http://localhost:8080")

            result = client.encode(
                "bge-m3",
                {"text": "hello"},
                gpu="l4",
                wait_for_capacity=True,
                provision_timeout_s=0.2,
            )

            # Should have retried and succeeded
            assert "dense" in result
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_202_timeout_raises_provisioning_error(self) -> None:
        """Provisioning timeout raises ProvisioningError."""
        from sie_sdk import ProvisioningError

        mock_response_202 = MagicMock()
        mock_response_202.status_code = 202
        mock_response_202.headers = {"Retry-After": "0.01"}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            # Always return 202
            mock_client.return_value.post = MagicMock(return_value=mock_response_202)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ProvisioningError) as exc_info:
                client.encode(
                    "bge-m3",
                    {"text": "hello"},
                    gpu="l4",
                    wait_for_capacity=True,
                    provision_timeout_s=0.05,  # Very short timeout
                )

            assert "timeout" in str(exc_info.value).lower()
            client.close()

    def test_retry_after_header_parsed(self) -> None:
        """Retry-After header is correctly parsed."""
        from sie_sdk import ProvisioningError

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {"Retry-After": "60"}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ProvisioningError) as exc_info:
                client.encode("bge-m3", {"text": "hello"}, gpu="l4", wait_for_capacity=False)

            assert exc_info.value.retry_after == 60.0
            client.close()

    def test_missing_retry_after_uses_default(self) -> None:
        """Missing Retry-After header uses None."""
        from sie_sdk import ProvisioningError

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {}  # No Retry-After

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ProvisioningError) as exc_info:
                client.encode("bge-m3", {"text": "hello"}, gpu="l4", wait_for_capacity=False)

            assert exc_info.value.retry_after is None
            client.close()

    def test_default_wait_for_capacity_is_true(self) -> None:
        """Default wait_for_capacity is True, so 202 is retried automatically."""
        mock_response_202 = MagicMock()
        mock_response_202.status_code = 202
        mock_response_202.headers = {"Retry-After": "0.01"}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-type": "application/msgpack"}
        mock_response_200.content = msgpack.packb({"items": [{"dense": {"values": np.zeros(1024)}}]}, use_bin_type=True)

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[mock_response_202, mock_response_200])
            client = SIEClient("http://localhost:8080")

            # No explicit wait_for_capacity -- default should be True now
            result = client.encode(
                "bge-m3",
                {"text": "hello"},
                gpu="l4",
                provision_timeout_s=0.2,
            )

            assert "dense" in result
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_explicit_wait_for_capacity_false_still_works(self) -> None:
        """Explicitly setting wait_for_capacity=False still raises ProvisioningError on 202."""
        from sie_sdk import ProvisioningError

        mock_response = MagicMock()
        mock_response.status_code = 202
        mock_response.headers = {"Retry-After": "30"}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ProvisioningError):
                client.encode("bge-m3", {"text": "hello"}, gpu="l4", wait_for_capacity=False)

            client.close()


class TestModelLoadingRetry:
    """Tests for 503 MODEL_LOADING retry functionality."""

    def test_503_model_loading_retries_until_success(self) -> None:
        """503 with MODEL_LOADING error code is retried until model is loaded."""
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE

        # First response is 503 MODEL_LOADING, second is 200
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
            client = SIEClient("http://localhost:8080")

            result = client.encode("bge-m3", {"text": "hello"})

            # Should have retried and succeeded
            assert "dense" in result
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_503_model_loading_timeout_raises_error(self) -> None:
        """MODEL_LOADING retry timeout raises timeout error after provision_timeout_s exceeded.

        Either ModelLoadingError (timeout during retry) or ProvisioningError
        (pre-request timeout check) are valid - both indicate the timeout was enforced.
        """
        from sie_sdk import ModelLoadingError, ProvisioningError
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE

        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.01", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            # Always return 503 MODEL_LOADING
            mock_client.return_value.post = MagicMock(return_value=mock_response_503)
            client = SIEClient("http://localhost:8080")

            # Use a very short provision_timeout_s to trigger timeout quickly
            with pytest.raises((ModelLoadingError, ProvisioningError)) as exc_info:
                client.encode("bge-m3", {"text": "hello"}, provision_timeout_s=0.05)

            assert "timeout" in str(exc_info.value).lower()
            client.close()

    def test_503_non_model_loading_not_retried(self) -> None:
        """503 without MODEL_LOADING code raises ServerError immediately."""
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"content-type": "application/json"}
        mock_response_503.json.return_value = {"error": {"code": "OVERLOADED", "message": "Server overloaded"}}

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=mock_response_503)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ServerError) as exc_info:
                client.encode("bge-m3", {"text": "hello"}, wait_for_capacity=False)

            assert exc_info.value.status_code == 503
            assert exc_info.value.code == "OVERLOADED"
            # Should not have retried
            assert mock_client.return_value.post.call_count == 1
            client.close()

    def test_model_loading_retry_respects_retry_after_header(self) -> None:
        """MODEL_LOADING retry uses Retry-After header value."""
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE

        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_503.headers = {"Retry-After": "0.05", "content-type": "application/json"}
        mock_response_503.json.return_value = {"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.headers = {"content-type": "application/msgpack"}
        mock_response_200.content = msgpack.packb(
            {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True
        )

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[mock_response_503, mock_response_200])
            with patch("sie_sdk.client.sync.time.sleep") as mock_sleep:
                client = SIEClient("http://localhost:8080")
                client.encode("bge-m3", {"text": "hello"})

                # Should have slept with the Retry-After value
                mock_sleep.assert_called_with(0.05)
            client.close()


class TestAsyncModelLoadingRetry:
    """Tests for async 503 MODEL_LOADING retry functionality."""

    @pytest.mark.asyncio
    async def test_503_model_loading_retries_until_success(self) -> None:
        """503 with MODEL_LOADING error code is retried until model is loaded."""
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE
        from sie_sdk.client.async_ import _AioResponse

        resp_503 = _AioResponse(
            503,
            json.dumps({"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}).encode(),
            {"Retry-After": "0.01", "content-type": "application/json"},
        )
        resp_200 = _AioResponse(
            200,
            msgpack.packb({"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True),
            {"content-type": "application/msgpack"},
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(side_effect=[resp_503, resp_200])  # type: ignore[method-assign]

        result = await client.encode("bge-m3", {"text": "hello"})

        assert "dense" in result
        assert client._post.call_count == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_503_model_loading_timeout_raises_error(self) -> None:
        """MODEL_LOADING retry timeout raises timeout error after provision_timeout_s exceeded."""
        from sie_sdk import ModelLoadingError, ProvisioningError
        from sie_sdk.client._shared import MODEL_LOADING_ERROR_CODE
        from sie_sdk.client.async_ import _AioResponse

        resp_503 = _AioResponse(
            503,
            json.dumps({"detail": {"code": MODEL_LOADING_ERROR_CODE, "message": "Model loading"}}).encode(),
            {"Retry-After": "0.01", "content-type": "application/json"},
        )

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=resp_503)  # type: ignore[method-assign]

        with pytest.raises((ModelLoadingError, ProvisioningError)) as exc_info:
            await client.encode("bge-m3", {"text": "hello"}, provision_timeout_s=0.05)

        assert "timeout" in str(exc_info.value).lower()
        await client.close()
