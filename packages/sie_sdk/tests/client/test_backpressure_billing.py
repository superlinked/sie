# Tests for the pass-2 audit backpressure / billing signals the gateway emits
# that the SDK previously discarded:
#
#   B1 — 429 RATE_LIMIT                     -> retried (Retry-After), typed RateLimitError on give-up
#   B2 — 503 BILLING_CAPACITY_UNAVAILABLE   -> retried (Retry-After), terminal ServerError on give-up
#   B7 — 503 QUEUE_FULL                     -> retried (Retry-After)
#   B3 — 402/403 credit & account failures  -> TERMINAL, typed, NEVER retried (single attempt)
#
# Mirrors the harness in ``test_oom_retry.py``: stub the underlying transport
# with a canned response sequence, patch the sleep to a no-op, and assert the
# retry count / honored Retry-After / final typed error. ``time.monotonic`` is
# stubbed only where a give-up must be forced within the provision-timeout
# budget.

from __future__ import annotations

import asyncio  # noqa: F401 — referenced via the async sleep patch target
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client._shared import handle_error
from sie_sdk.client.async_ import _AioResponse
from sie_sdk.client.errors import (
    AccountInactiveError,
    AccountStateUnavailableError,
    InsufficientCreditsError,
    RateLimitError,
    RequestError,
    ServerError,
    SpendLimitError,
)

# --------------------------------------------------------------------------
# Sync response fixtures
# --------------------------------------------------------------------------


def _resp_429(retry_after: str = "0.01") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 429
    resp.headers = {
        "Retry-After": retry_after,
        "content-type": "application/json",
        "X-SIE-Error-Code": "RATE_LIMIT",
    }
    resp.json.return_value = {"detail": {"code": "RATE_LIMIT", "message": "rate limited"}}
    return resp


def _resp_503(code: str, retry_after: str = "0.01") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 503
    resp.headers = {"Retry-After": retry_after, "content-type": "application/json"}
    resp.json.return_value = {"detail": {"code": code, "message": code.lower()}}
    return resp


def _resp_terminal(status: int, code: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {"detail": {"code": code, "message": code.lower()}}
    resp.text = json.dumps(resp.json.return_value)
    return resp


def _resp_200_encode() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/msgpack"}
    resp.content = msgpack.packb(
        {"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]},
        use_bin_type=True,
    )
    return resp


# ``time.monotonic`` stub: first call (start_time) and the loop-top elapsed
# check stay within budget; the admission-ladder elapsed check jumps past it so
# the give-up branch fires without a second network round-trip.
def _exhausted_ticks(base: float = 1000.0) -> object:
    return iter([base, base, base + 100.0, base + 100.0, base + 100.0, base + 100.0])


# --------------------------------------------------------------------------
# B1 — 429 rate limit (sync)
# --------------------------------------------------------------------------


class TestSyncRateLimit:
    def test_retry_then_success_honors_retry_after(self) -> None:
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep") as mock_sleep,
        ):
            mock_client.return_value.post = MagicMock(
                side_effect=[_resp_429("0.02"), _resp_429("0.02"), _resp_200_encode()]
            )
            client = SIEClient("http://localhost:8080")

            result = client.encode("bge-m3", {"text": "hi"})

            assert result["dense"].shape == (4,)
            assert mock_client.return_value.post.call_count == 3
            # Server Retry-After honored verbatim on every retry.
            assert [c.args[0] for c in mock_sleep.call_args_list] == [0.02, 0.02]
            client.close()

    def test_giveup_raises_rate_limit_error(self) -> None:
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
            patch.object(time, "monotonic", side_effect=_exhausted_ticks()),
        ):
            mock_client.return_value.post = MagicMock(side_effect=[_resp_429("30"), _resp_200_encode()])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(RateLimitError) as excinfo:
                client.encode("bge-m3", {"text": "hi"}, provision_timeout_s=10.0)

            assert excinfo.value.status_code == 429
            assert excinfo.value.retry_after == 30.0
            # Budget spent on the first 429 — no second network round-trip.
            assert mock_client.return_value.post.call_count == 1
            client.close()


# --------------------------------------------------------------------------
# B2 / B7 — retryable 503 backpressure (sync)
# --------------------------------------------------------------------------


class TestSyncBackpressure503:
    @pytest.mark.parametrize("code", ["BILLING_CAPACITY_UNAVAILABLE", "QUEUE_FULL"])
    def test_retry_then_success(self, code: str) -> None:
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep") as mock_sleep,
        ):
            mock_client.return_value.post = MagicMock(side_effect=[_resp_503(code, "1"), _resp_200_encode()])
            client = SIEClient("http://localhost:8080")

            result = client.encode("bge-m3", {"text": "hi"})

            assert result["dense"].shape == (4,)
            assert mock_client.return_value.post.call_count == 2
            assert mock_sleep.call_args_list[0].args[0] == 1.0  # Retry-After honored
            client.close()

    def test_giveup_raises_server_error_preserving_code(self) -> None:
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
            patch.object(time, "monotonic", side_effect=_exhausted_ticks()),
        ):
            mock_client.return_value.post = MagicMock(
                side_effect=[_resp_503("BILLING_CAPACITY_UNAVAILABLE", "1"), _resp_200_encode()]
            )
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ServerError) as excinfo:
                client.encode("bge-m3", {"text": "hi"}, provision_timeout_s=10.0)

            assert excinfo.value.status_code == 503
            assert excinfo.value.code == "BILLING_CAPACITY_UNAVAILABLE"
            # Terminal 503 must NOT be a RateLimitError.
            assert not isinstance(excinfo.value, RateLimitError)
            assert mock_client.return_value.post.call_count == 1
            client.close()


# --------------------------------------------------------------------------
# B3 — terminal credit / account failures (sync) — NEVER retried
# --------------------------------------------------------------------------


class TestSyncTerminalBillingErrors:
    @pytest.mark.parametrize(
        ("status", "code", "exc"),
        [
            (402, "INSUFFICIENT_CREDITS", InsufficientCreditsError),
            (402, "KEY_SPEND_LIMIT_EXCEEDED", SpendLimitError),
            (403, "ACCOUNT_SUSPENDED", AccountInactiveError),
            (403, "ACCOUNT_PENDING_REVIEW", AccountInactiveError),
            (503, "ACCOUNT_STATE_UNAVAILABLE", AccountStateUnavailableError),
        ],
    )
    def test_terminal_typed_and_not_retried(self, status: int, code: str, exc: type[Exception]) -> None:
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep") as mock_sleep,
        ):
            # A 200 is queued AFTER the failure to prove the SDK never reaches it.
            mock_client.return_value.post = MagicMock(side_effect=[_resp_terminal(status, code), _resp_200_encode()])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(exc) as excinfo:
                client.encode("bge-m3", {"text": "hi"})

            assert getattr(excinfo.value, "code", None) == code
            assert getattr(excinfo.value, "status_code", None) == status
            # Single attempt — no retry, no sleep.
            assert mock_client.return_value.post.call_count == 1
            assert mock_sleep.call_count == 0
            client.close()


# --------------------------------------------------------------------------
# Async response fixtures
# --------------------------------------------------------------------------


def _aio_429(retry_after: str = "0.01") -> object:
    return _AioResponse(
        429,
        json.dumps({"detail": {"code": "RATE_LIMIT", "message": "rate limited"}}).encode(),
        {"Retry-After": retry_after, "content-type": "application/json", "X-SIE-Error-Code": "RATE_LIMIT"},
    )


def _aio_503(code: str, retry_after: str = "0.01") -> object:
    return _AioResponse(
        503,
        json.dumps({"detail": {"code": code, "message": code.lower()}}).encode(),
        {"Retry-After": retry_after, "content-type": "application/json"},
    )


def _aio_terminal(status: int, code: str) -> object:
    return _AioResponse(
        status,
        json.dumps({"detail": {"code": code, "message": code.lower()}}).encode(),
        {"content-type": "application/json"},
    )


def _aio_200_encode() -> object:
    return _AioResponse(
        200,
        msgpack.packb({"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True),
        {"content-type": "application/msgpack"},
    )


# --------------------------------------------------------------------------
# Async client
# --------------------------------------------------------------------------


class TestAsyncBackpressureBilling:
    @pytest.mark.asyncio
    async def test_rate_limit_retry_then_success(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession"),
            patch("sie_sdk.client.async_.asyncio.sleep") as mock_sleep,
        ):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(side_effect=[_aio_429("0.02"), _aio_200_encode()])

            result = await client.encode("bge-m3", {"text": "hi"})

            assert result["dense"].shape == (4,)
            assert client._post.await_count == 2
            assert mock_sleep.call_args_list[0].args[0] == 0.02
            await client.close()

    @pytest.mark.asyncio
    async def test_rate_limit_giveup_raises(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession"),
            patch("sie_sdk.client.async_.asyncio.sleep"),
            patch.object(time, "monotonic", side_effect=_exhausted_ticks()),
        ):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(side_effect=[_aio_429("30"), _aio_200_encode()])

            with pytest.raises(RateLimitError) as excinfo:
                await client.encode("bge-m3", {"text": "hi"}, provision_timeout_s=10.0)

            assert excinfo.value.retry_after == 30.0
            assert client._post.await_count == 1
            await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", ["BILLING_CAPACITY_UNAVAILABLE", "QUEUE_FULL"])
    async def test_backpressure_503_retry_then_success(self, code: str) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession"),
            patch("sie_sdk.client.async_.asyncio.sleep") as mock_sleep,
        ):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(side_effect=[_aio_503(code, "1"), _aio_200_encode()])

            result = await client.encode("bge-m3", {"text": "hi"})

            assert result["dense"].shape == (4,)
            assert client._post.await_count == 2
            assert mock_sleep.call_args_list[0].args[0] == 1.0
            await client.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "code", "exc"),
        [
            (402, "INSUFFICIENT_CREDITS", InsufficientCreditsError),
            (402, "KEY_SPEND_LIMIT_EXCEEDED", SpendLimitError),
            (403, "ACCOUNT_SUSPENDED", AccountInactiveError),
            (403, "ACCOUNT_PENDING_REVIEW", AccountInactiveError),
            (503, "ACCOUNT_STATE_UNAVAILABLE", AccountStateUnavailableError),
        ],
    )
    async def test_terminal_typed_and_not_retried(self, status: int, code: str, exc: type[Exception]) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession"),
            patch("sie_sdk.client.async_.asyncio.sleep") as mock_sleep,
        ):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(side_effect=[_aio_terminal(status, code), _aio_200_encode()])

            with pytest.raises(exc) as excinfo:
                await client.encode("bge-m3", {"text": "hi"})

            assert getattr(excinfo.value, "code", None) == code
            assert client._post.await_count == 1
            assert mock_sleep.call_count == 0
            await client.close()


# --------------------------------------------------------------------------
# Direct handle_error mapping (covers terminal / streaming / list paths)
# --------------------------------------------------------------------------


class TestHandleErrorMapping:
    @pytest.mark.parametrize(
        ("status", "code", "exc"),
        [
            (429, "RATE_LIMIT", RateLimitError),
            (402, "INSUFFICIENT_CREDITS", InsufficientCreditsError),
            (402, "KEY_SPEND_LIMIT_EXCEEDED", SpendLimitError),
            (403, "ACCOUNT_SUSPENDED", AccountInactiveError),
            (403, "ACCOUNT_PENDING_REVIEW", AccountInactiveError),
            (503, "ACCOUNT_STATE_UNAVAILABLE", AccountStateUnavailableError),
        ],
    )
    def test_handle_error_maps_code(self, status: int, code: str, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            handle_error(_resp_terminal(status, code))

    def test_unrecognized_403_stays_generic(self) -> None:
        # A 403 with a NON-account code must not become AccountInactiveError.
        with pytest.raises(RequestError) as excinfo:
            handle_error(_resp_terminal(403, "INVALID_KEY"))
        assert not isinstance(excinfo.value, AccountInactiveError)
