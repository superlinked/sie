# Retry coverage for both clients: mid-flight disconnect retries
# (`_RETRYABLE_TRANSPORT_ERRORS`) and connect-time retries (issue #95,
# `httpx.ConnectError` / `aiohttp.ClientConnectorError`). Each surface is
# tested for retry-then-succeed, budget exhaustion, and fail-fast under
# `wait_for_capacity=False`.

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import httpx
import msgpack
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    # No-op the retry sleeps; budget tests still use real time.monotonic.
    monkeypatch.setattr("sie_sdk.client.sync.time.sleep", lambda _: None)

    async def _noop_async_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("sie_sdk.client.async_.asyncio.sleep", _noop_async_sleep)


def _mock_response_200() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/msgpack"}
    resp.content = msgpack.packb(
        {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]},
        use_bin_type=True,
    )
    return resp


def _async_response_200() -> object:
    from sie_sdk.client.async_ import _AioResponse

    return _AioResponse(
        200,
        msgpack.packb(
            {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]},
            use_bin_type=True,
        ),
        {"content-type": "application/msgpack"},
    )


# Sync client — httpx-side transport errors.


class TestSyncTransportErrorRetry:
    @pytest.mark.parametrize(
        "exc",
        [
            httpx.RemoteProtocolError("Server disconnected without sending a response."),
            httpx.ReadError("Connection reset by peer"),
            httpx.WriteError("Broken pipe"),
        ],
        ids=["remote_protocol_error", "read_error", "write_error"],
    )
    def test_transport_error_retried_when_wait_for_capacity_true_then_succeeds(self, exc: Exception) -> None:
        from sie_sdk import SIEClient

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[exc, _mock_response_200()])
            client = SIEClient("http://localhost:8080")

            result = client.encode(
                "bge-m3",
                {"text": "hello"},
                wait_for_capacity=True,
                provision_timeout_s=10.0,
            )

            assert result["dense"].shape == (4,)
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_transport_error_not_retried_when_wait_for_capacity_false(self) -> None:
        from sie_sdk import SIEClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = httpx.RemoteProtocolError("Server disconnected without sending a response.")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[exc])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError):
                client.encode(
                    "bge-m3",
                    {"text": "hello"},
                    wait_for_capacity=False,
                    provision_timeout_s=5.0,
                )

            assert mock_client.return_value.post.call_count == 1
            client.close()

    def test_transport_error_retries_bounded_by_provision_timeout(self) -> None:
        from sie_sdk import ProvisioningError, SIEClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = httpx.RemoteProtocolError("Server disconnected without sending a response.")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=exc)
            client = SIEClient("http://localhost:8080")

            start = time.monotonic()
            # Either error type is valid: SIEConnectionError after budget
            # exhausted, or ProvisioningError if the pre-request budget
            # check caught a freshly-zeroed remaining timeout.
            with pytest.raises((SIEConnectionError, ProvisioningError)):
                client.encode(
                    "bge-m3",
                    {"text": "hello"},
                    wait_for_capacity=True,
                    provision_timeout_s=0.05,
                )
            elapsed = time.monotonic() - start

            assert elapsed < 0.25, f"Retry loop did not honour provision_timeout_s: {elapsed:.2f}s"
            assert mock_client.return_value.post.call_count >= 1
            client.close()

    def test_connect_error_retried_when_wait_for_capacity_true_then_succeeds(self) -> None:
        from sie_sdk import SIEClient

        exc = httpx.ConnectError("Connection refused")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[exc, _mock_response_200()])
            client = SIEClient("http://localhost:8080")

            result = client.encode(
                "bge-m3",
                {"text": "hello"},
                wait_for_capacity=True,
                provision_timeout_s=10.0,
            )

            assert result["dense"].shape == (4,)
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_connect_error_not_retried_when_wait_for_capacity_false(self) -> None:
        from sie_sdk import SIEClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = httpx.ConnectError("Connection refused")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[exc])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError):
                client.encode(
                    "bge-m3",
                    {"text": "hello"},
                    wait_for_capacity=False,
                    provision_timeout_s=10.0,
                )

            assert mock_client.return_value.post.call_count == 1
            client.close()

    def test_connect_error_retries_bounded_by_provision_timeout(self) -> None:
        from sie_sdk import ProvisioningError, SIEClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = httpx.ConnectError("Connection refused")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=exc)
            client = SIEClient("http://localhost:8080")

            start = time.monotonic()
            with pytest.raises((SIEConnectionError, ProvisioningError)):
                client.encode(
                    "bge-m3",
                    {"text": "hello"},
                    wait_for_capacity=True,
                    provision_timeout_s=0.05,
                )
            elapsed = time.monotonic() - start

            assert elapsed < 0.25, f"Retry loop did not honour provision_timeout_s: {elapsed:.2f}s"
            assert mock_client.return_value.post.call_count >= 1
            client.close()


# Async client — aiohttp-side transport errors.


class TestAsyncTransportErrorRetry:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            aiohttp.ServerDisconnectedError("Server disconnected"),
            aiohttp.ClientPayloadError("Response payload is not completed"),
            aiohttp.ServerTimeoutError("Timeout on reading data from socket"),
        ],
        ids=["server_disconnected", "client_payload_error", "server_timeout_error"],
    )
    async def test_transport_error_retried_when_wait_for_capacity_true_then_succeeds(self, exc: Exception) -> None:
        from sie_sdk import SIEAsyncClient

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore
            side_effect=[exc, _async_response_200()]
        )

        result = await client.encode(
            "bge-m3",
            {"text": "hello"},
            wait_for_capacity=True,
            provision_timeout_s=10.0,
        )

        assert result["dense"].shape == (4,)
        assert client._post.call_count == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_transport_error_not_retried_when_wait_for_capacity_false(self) -> None:
        from sie_sdk import SIEAsyncClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = aiohttp.ServerDisconnectedError("Server disconnected")
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(side_effect=[exc])  # type: ignore

        with pytest.raises(SIEConnectionError):
            await client.encode(
                "bge-m3",
                {"text": "hello"},
                wait_for_capacity=False,
                provision_timeout_s=5.0,
            )

        assert client._post.call_count == 1
        await client.close()

    @staticmethod
    def _make_connector_error() -> aiohttp.ClientConnectorError:
        # ClientConnectorError requires a ConnectionKey + OSError.
        key = aiohttp.client_reqrep.ConnectionKey(
            host="localhost",
            port=8080,
            is_ssl=False,
            ssl=None,
            proxy=None,
            proxy_auth=None,
            proxy_headers_hash=None,
        )
        return aiohttp.ClientConnectorError(key, OSError("Connection refused"))

    @pytest.mark.asyncio
    async def test_connector_error_retried_when_wait_for_capacity_true_then_succeeds(
        self,
    ) -> None:
        from sie_sdk import SIEAsyncClient

        exc = self._make_connector_error()
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(side_effect=[exc, _async_response_200()])  # type: ignore

        result = await client.encode(
            "bge-m3",
            {"text": "hello"},
            wait_for_capacity=True,
            provision_timeout_s=10.0,
        )

        assert result["dense"].shape == (4,)
        assert client._post.call_count == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_connector_error_not_retried_when_wait_for_capacity_false(self) -> None:
        from sie_sdk import SIEAsyncClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = self._make_connector_error()
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(side_effect=[exc])  # type: ignore

        with pytest.raises(SIEConnectionError):
            await client.encode(
                "bge-m3",
                {"text": "hello"},
                wait_for_capacity=False,
                provision_timeout_s=10.0,
            )

        assert client._post.call_count == 1
        await client.close()

    @pytest.mark.asyncio
    async def test_connector_error_retries_bounded_by_provision_timeout(self) -> None:
        from sie_sdk import ProvisioningError, SIEAsyncClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = self._make_connector_error()
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(side_effect=exc)  # type: ignore

        start = time.monotonic()
        with pytest.raises((SIEConnectionError, ProvisioningError)):
            await client.encode(
                "bge-m3",
                {"text": "hello"},
                wait_for_capacity=True,
                provision_timeout_s=0.05,
            )
        elapsed = time.monotonic() - start

        assert elapsed < 0.25, f"Retry loop did not honour provision_timeout_s: {elapsed:.2f}s"
        assert client._post.call_count >= 1
        await client.close()


# Cross-method coverage: pin that score/extract share the same retry block.


def _mock_score_response_200() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/msgpack"}
    resp.content = msgpack.packb(
        {
            "model": "bge-reranker-v2-m3",
            "scores": [
                {"item_id": 0, "score": 0.9, "rank": 0},
                {"item_id": 1, "score": 0.1, "rank": 1},
            ],
        },
        use_bin_type=True,
    )
    return resp


def _mock_extract_response_200() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/msgpack"}
    resp.content = msgpack.packb({"items": [{"entities": []}]}, use_bin_type=True)
    return resp


class TestSyncTransportErrorRetryScoreExtract:
    def test_score_retries_on_remote_protocol_error(self) -> None:
        from sie_sdk import SIEClient

        exc = httpx.RemoteProtocolError("Server disconnected")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[exc, _mock_score_response_200()])
            client = SIEClient("http://localhost:8080")

            result = client.score(
                "bge-reranker-v2-m3",
                query={"text": "q"},
                items=[{"text": "a"}, {"text": "b"}],
                wait_for_capacity=True,
                provision_timeout_s=10.0,
            )

            assert len(result["scores"]) == 2
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_extract_retries_on_remote_protocol_error(self) -> None:
        from sie_sdk import SIEClient

        exc = httpx.RemoteProtocolError("Server disconnected")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[exc, _mock_extract_response_200()])
            client = SIEClient("http://localhost:8080")

            result = client.extract(
                "gliner_small-v2.1",
                {"text": "hello"},
                labels=["person"],
                wait_for_capacity=True,
                provision_timeout_s=10.0,
            )

            assert "entities" in result
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_score_retries_on_connect_error(self) -> None:
        from sie_sdk import SIEClient

        exc = httpx.ConnectError("Connection refused")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[exc, _mock_score_response_200()])
            client = SIEClient("http://localhost:8080")

            result = client.score(
                "bge-reranker-v2-m3",
                query={"text": "q"},
                items=[{"text": "a"}, {"text": "b"}],
                wait_for_capacity=True,
                provision_timeout_s=10.0,
            )

            assert len(result["scores"]) == 2
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_extract_retries_on_connect_error(self) -> None:
        from sie_sdk import SIEClient

        exc = httpx.ConnectError("Connection refused")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[exc, _mock_extract_response_200()])
            client = SIEClient("http://localhost:8080")

            result = client.extract(
                "gliner_small-v2.1",
                {"text": "hello"},
                labels=["person"],
                wait_for_capacity=True,
                provision_timeout_s=10.0,
            )

            assert "entities" in result
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_score_fails_fast_on_connect_error_when_wait_for_capacity_false(self) -> None:
        from sie_sdk import SIEClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = httpx.ConnectError("Connection refused")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=exc)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError):
                client.score(
                    "bge-reranker-v2-m3",
                    query={"text": "q"},
                    items=[{"text": "a"}, {"text": "b"}],
                    wait_for_capacity=False,
                    provision_timeout_s=10.0,
                )

            assert mock_client.return_value.post.call_count == 1
            client.close()

    def test_extract_fails_fast_on_connect_error_when_wait_for_capacity_false(self) -> None:
        from sie_sdk import SIEClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = httpx.ConnectError("Connection refused")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=exc)
            client = SIEClient("http://localhost:8080")

            with pytest.raises(SIEConnectionError):
                client.extract(
                    "gliner_small-v2.1",
                    {"text": "hello"},
                    labels=["person"],
                    wait_for_capacity=False,
                    provision_timeout_s=10.0,
                )

            assert mock_client.return_value.post.call_count == 1
            client.close()

    def test_score_fails_fast_on_permanent_connect_error(self) -> None:
        import ssl

        from sie_sdk import SIEClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = httpx.ConnectError("SSL handshake failed")
        exc.__cause__ = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=exc)
            client = SIEClient("https://localhost:8080")

            with pytest.raises(SIEConnectionError):
                client.score(
                    "bge-reranker-v2-m3",
                    query={"text": "q"},
                    items=[{"text": "a"}, {"text": "b"}],
                    wait_for_capacity=True,
                    provision_timeout_s=10.0,
                )

            assert mock_client.return_value.post.call_count == 1
            client.close()

    def test_extract_fails_fast_on_permanent_connect_error(self) -> None:
        import ssl

        from sie_sdk import SIEClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = httpx.ConnectError("SSL handshake failed")
        exc.__cause__ = ssl.SSLError("CERTIFICATE_VERIFY_FAILED")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=exc)
            client = SIEClient("https://localhost:8080")

            with pytest.raises(SIEConnectionError):
                client.extract(
                    "gliner_small-v2.1",
                    {"text": "hello"},
                    labels=["person"],
                    wait_for_capacity=True,
                    provision_timeout_s=10.0,
                )

            assert mock_client.return_value.post.call_count == 1
            client.close()
