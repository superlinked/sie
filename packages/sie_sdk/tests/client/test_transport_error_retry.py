# Retry coverage for both clients: mid-flight disconnect retries
# (`_RETRYABLE_TRANSPORT_ERRORS`) and connect-time retries (issue #95,
# `httpx.ConnectError` / `aiohttp.ClientConnectorError`). Each surface is
# tested for retry-then-succeed, budget exhaustion, and fail-fast under
# `wait_for_capacity=False`.

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import httpx
import msgpack
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client._shared import url_origin_for_logging


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    # No-op ordinary retry sleeps; budget tests install a deterministic clock.
    monkeypatch.setattr("sie_sdk.client.sync.time.sleep", lambda _: None)

    async def _noop_async_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("sie_sdk.client.async_.asyncio.sleep", _noop_async_sleep)


class _RetryClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.now += delay

    async def async_sleep(self, delay: float) -> None:
        self.sleep(delay)


@pytest.fixture
def retry_clock(monkeypatch: pytest.MonkeyPatch) -> _RetryClock:
    clock = _RetryClock()
    monkeypatch.setattr("sie_sdk.client.sync.time", clock)
    monkeypatch.setattr("sie_sdk.client.async_.time", clock)
    monkeypatch.setattr("sie_sdk.client._shared.time", clock)
    monkeypatch.setattr("sie_sdk.client.async_.asyncio.sleep", clock.async_sleep)
    monkeypatch.setattr("sie_sdk.client._shared.apply_jitter", lambda delay, **_kwargs: delay)
    return clock


def _logged_origin(message: str) -> str:
    """Extract the origin the retry WARNING reports.

    The message is ``"... contacting <origin>, retrying in ..."``; returning
    the exact token lets tests assert equality against the expected origin
    instead of a URL substring-membership check (``"<url>" in message``),
    which trips CodeQL's ``py/incomplete-url-substring-sanitization`` rule.
    """
    return message.split(" contacting ", 1)[1].split(",", 1)[0]


def _mock_response_200() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/msgpack"}
    resp.content = msgpack.packb(
        {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]},
        use_bin_type=True,
    )
    return resp


def _mock_response_model_loading() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 503
    resp.headers = {"Retry-After": "0.01", "content-type": "application/json"}
    resp.json.return_value = {"detail": {"code": "MODEL_LOADING", "message": "loading"}}
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
    def test_shared_client_recovers_from_read_error_during_concurrent_model_loading(self) -> None:
        """One transient socket failure must not abort concurrent cold-load requests."""
        from sie_sdk import SIEClient

        worker_count = 4
        first_attempts = threading.Barrier(worker_count)
        attempts: dict[int, int] = {}
        attempts_lock = threading.Lock()
        error_thread: list[int] = []

        def _post(*_args: object, **_kwargs: object) -> MagicMock:
            thread_id = threading.get_ident()
            with attempts_lock:
                attempt = attempts.get(thread_id, 0) + 1
                attempts[thread_id] = attempt
                if not error_thread:
                    error_thread.append(thread_id)

            if attempt == 1:
                first_attempts.wait(timeout=2.0)
                return _mock_response_model_loading()
            if thread_id == error_thread[0] and attempt == 2:
                raise httpx.ReadError("[Errno 9] Bad file descriptor")
            return _mock_response_200()

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post.side_effect = _post
            client = SIEClient("http://localhost:8080")

            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(
                        client.encode,
                        "bge-m3",
                        {"text": f"item-{index}"},
                        wait_for_capacity=True,
                        provision_timeout_s=10.0,
                    )
                    for index in range(worker_count)
                ]
                results = [future.result() for future in futures]

            assert all(result["dense"].shape == (4,) for result in results)
            assert sorted(attempts.values()) == [2, 2, 2, 3]
            assert mock_client.return_value.post.call_count == 9
            client.close()

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

    def test_transport_error_retries_bounded_by_provision_timeout(self, retry_clock: _RetryClock) -> None:
        from sie_sdk import ProvisioningError, SIEClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = httpx.RemoteProtocolError("Server disconnected without sending a response.")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=exc)
            client = SIEClient("http://localhost:8080")

            start = retry_clock.monotonic()
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
            elapsed = retry_clock.monotonic() - start

            assert elapsed == pytest.approx(0.05)
            assert mock_client.return_value.post.call_count == 1
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

    def test_connect_error_retries_bounded_by_provision_timeout(self, retry_clock: _RetryClock) -> None:
        from sie_sdk import ProvisioningError, SIEClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = httpx.ConnectError("Connection refused")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=exc)
            client = SIEClient("http://localhost:8080")

            start = retry_clock.monotonic()
            with pytest.raises((SIEConnectionError, ProvisioningError)):
                client.encode(
                    "bge-m3",
                    {"text": "hello"},
                    wait_for_capacity=True,
                    provision_timeout_s=0.05,
                )
            elapsed = retry_clock.monotonic() - start

            assert elapsed == pytest.approx(0.05)
            assert mock_client.return_value.post.call_count == 1
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
    async def test_connector_error_retries_bounded_by_provision_timeout(self, retry_clock: _RetryClock) -> None:
        from sie_sdk import ProvisioningError, SIEAsyncClient
        from sie_sdk.client.errors import SIEConnectionError

        exc = self._make_connector_error()
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(side_effect=exc)  # type: ignore

        start = retry_clock.monotonic()
        with pytest.raises((SIEConnectionError, ProvisioningError)):
            await client.encode(
                "bge-m3",
                {"text": "hello"},
                wait_for_capacity=True,
                provision_timeout_s=0.05,
            )
        elapsed = retry_clock.monotonic() - start

        assert elapsed == pytest.approx(0.05)
        assert client._post.call_count == 1
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
            assert client.last_retry_count == 1

            mock_client.return_value.post.side_effect = None
            mock_client.return_value.post.return_value = _mock_score_response_200()
            client.score("bge-reranker-v2-m3", query={"text": "q"}, items=[{"text": "a"}])
            assert client.last_retry_count == 0
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
            assert client.last_retry_count == 1

            mock_client.return_value.post.side_effect = None
            mock_client.return_value.post.return_value = _mock_extract_response_200()
            client.extract("gliner_small-v2.1", {"text": "again"}, labels=["person"])
            assert client.last_retry_count == 0
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

    @pytest.mark.usefixtures("_no_sleep")
    def test_connect_retry_logging_first_warning_then_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """The FIRST connect-retry surfaces at WARNING (naming the target URL
        and the total wait budget) so a user at the default log level can see
        the SDK is retrying instead of silently blocking for up to the whole
        provision budget; subsequent retries stay at INFO (OOM convention).

        Declares ``_no_sleep`` via ``usefixtures`` so the retry sleeps are
        patched explicitly (the fixture is also module-autouse).
        """
        exc = httpx.ConnectError("Connection refused")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[exc, exc, _mock_response_200()])
            client = SIEClient("http://localhost:8080")

            with caplog.at_level(logging.INFO, logger="sie_sdk.client._shared"):
                client.encode(
                    "bge-m3",
                    {"text": "hello"},
                    wait_for_capacity=True,
                    provision_timeout_s=10.0,
                )

            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1
            message = warnings[0].getMessage()
            assert _logged_origin(message) == "http://localhost:8080"
            assert "timeout: 10.0s" in message
            infos = [r for r in caplog.records if r.levelno == logging.INFO and "Connect error" in r.getMessage()]
            assert len(infos) == 1
            client.close()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_sleep")
    async def test_async_connect_retry_logging_first_warning_then_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Async mirror of the first-connect-retry WARNING convention.

        Declares ``_no_sleep`` via ``usefixtures`` so the retry sleeps are
        patched explicitly (the fixture is also module-autouse).
        """
        exc = TestAsyncTransportErrorRetry._make_connector_error()
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(side_effect=[exc, exc, _async_response_200()])  # type: ignore

        with caplog.at_level(logging.INFO, logger="sie_sdk.client._shared"):
            await client.encode(
                "bge-m3",
                {"text": "hello"},
                wait_for_capacity=True,
                provision_timeout_s=10.0,
            )

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert _logged_origin(message) == "http://localhost:8080"
        assert "timeout: 10.0s" in message
        infos = [r for r in caplog.records if r.levelno == logging.INFO and "Connect error" in r.getMessage()]
        assert len(infos) == 1
        await client.close()

    # A base_url carrying credentials in userinfo AND a token query param —
    # the log must leak neither. Only the scheme://host:port origin is logged.
    _CREDENTIALED_BASE_URL = "https://user:s3cr3t-token@gateway.example.test:8443/v1?access_token=querysecret"

    @pytest.mark.usefixtures("_no_sleep")
    def test_connect_retry_warning_logs_origin_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """A base_url carrying credentials/tokens must NOT leak them into logs.

        The retry WARNING logs only the
        ``scheme://host:port`` origin — no userinfo, no path, no query.
        """
        # The helper strips userinfo, path, and query — only the origin remains.
        expected_origin = url_origin_for_logging(self._CREDENTIALED_BASE_URL)
        assert expected_origin == "https://gateway.example.test:8443"

        exc = httpx.ConnectError("Connection refused")
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[exc, _mock_response_200()])
            client = SIEClient(self._CREDENTIALED_BASE_URL)

            with caplog.at_level(logging.INFO, logger="sie_sdk.client._shared"):
                client.encode("bge-m3", {"text": "hello"}, wait_for_capacity=True, provision_timeout_s=10.0)

            blob = "\n".join(r.getMessage() for r in caplog.records)
            assert "s3cr3t-token" not in blob
            assert "querysecret" not in blob
            assert "user:" not in blob
            assert "access_token" not in blob
            warning = next(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
            assert _logged_origin(warning) == expected_origin
            client.close()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_sleep")
    async def test_async_connect_retry_warning_logs_origin_only(self, caplog: pytest.LogCaptureFixture) -> None:
        """Async mirror: embedded credentials and query tokens must not reach the log."""
        expected_origin = url_origin_for_logging(self._CREDENTIALED_BASE_URL)
        assert expected_origin == "https://gateway.example.test:8443"

        exc = TestAsyncTransportErrorRetry._make_connector_error()
        client = SIEAsyncClient(self._CREDENTIALED_BASE_URL)
        client._post = AsyncMock(side_effect=[exc, _async_response_200()])  # type: ignore

        with caplog.at_level(logging.INFO, logger="sie_sdk.client._shared"):
            await client.encode("bge-m3", {"text": "hello"}, wait_for_capacity=True, provision_timeout_s=10.0)

        blob = "\n".join(r.getMessage() for r in caplog.records)
        assert "s3cr3t-token" not in blob
        assert "querysecret" not in blob
        assert "user:" not in blob
        assert "access_token" not in blob
        warning = next(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert _logged_origin(warning) == expected_origin
        await client.close()

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
