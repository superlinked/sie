"""Tests for the ``InputTooLongError`` short-circuit (#849).

A 400 ``INPUT_TOO_LONG`` response on the extract path must:
- raise :class:`InputTooLongError` immediately on the first response
- carry ``code == "INPUT_TOO_LONG"`` and ``status_code == 400``
- expose ``model`` from caller context
- not be confused with generic :class:`RequestError` (so callers can
  branch on token-budget failures specifically)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client._shared import handle_error
from sie_sdk.client.async_ import _AioResponse
from sie_sdk.client.errors import InputTooLongError, RequestError


def _resp_input_too_long(message: str = "Input exceeds capacity (4096 tokens)") -> MagicMock:
    resp = MagicMock()
    resp.status_code = 400
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {"detail": {"code": "INPUT_TOO_LONG", "message": message, "param": "items[0].text"}}
    return resp


def _resp_validation_error() -> MagicMock:
    """Negative case: a different 400 that must NOT be classified as INPUT_TOO_LONG."""
    resp = MagicMock()
    resp.status_code = 400
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {"detail": {"code": "VALIDATION_ERROR", "message": "bad input"}}
    return resp


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class TestSyncInputTooLong:
    def test_extract_raises_immediately_on_first_response(self) -> None:
        """No retries are attempted; the typed error surfaces on the first call."""
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep") as mock_sleep,
        ):
            mock_client.return_value.post = MagicMock(side_effect=[_resp_input_too_long("Too many tokens")])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(InputTooLongError) as excinfo:
                client.extract("gliclass-large", {"text": "hi"}, labels=["a", "b"])

            assert excinfo.value.model == "gliclass-large"
            assert excinfo.value.code == "INPUT_TOO_LONG"
            assert excinfo.value.status_code == 400
            assert excinfo.value.param == "items[0].text"
            assert str(excinfo.value) == "Too many tokens"
            # Critical: no retry happened.
            assert mock_client.return_value.post.call_count == 1
            mock_sleep.assert_not_called()
            client.close()

    def test_is_request_error_subclass(self) -> None:
        """Existing 4xx handlers (`except RequestError`) must still catch it."""
        assert issubclass(InputTooLongError, RequestError)
        assert not issubclass(RequestError, InputTooLongError)

    def test_other_400_falls_through_to_request_error(self) -> None:
        """A 400 with a different code must NOT become InputTooLongError."""
        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[_resp_validation_error()])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(RequestError) as excinfo:
                client.extract("gliclass-large", {"text": "hi"}, labels=["a"])

            assert not isinstance(excinfo.value, InputTooLongError)
            assert excinfo.value.code == "VALIDATION_ERROR"
            assert excinfo.value.status_code == 400
            client.close()


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


def _aio_input_too_long(message: str = "Input exceeds capacity (4096 tokens)") -> object:
    return _AioResponse(
        400,
        json.dumps({"detail": {"code": "INPUT_TOO_LONG", "message": message, "param": "items[0].text"}}).encode(),
        {"content-type": "application/json"},
    )


def _aio_validation_error() -> object:
    return _AioResponse(
        400,
        json.dumps({"detail": {"code": "VALIDATION_ERROR", "message": "bad input"}}).encode(),
        {"content-type": "application/json"},
    )


class TestAsyncInputTooLong:
    @pytest.mark.asyncio
    async def test_extract_raises_immediately(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession"),
            patch("sie_sdk.client.async_.asyncio.sleep") as mock_sleep,
        ):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(side_effect=[_aio_input_too_long()])

            with pytest.raises(InputTooLongError) as excinfo:
                await client.extract("gliclass-large", {"text": "hi"}, labels=["a"])

            assert excinfo.value.model == "gliclass-large"
            assert excinfo.value.code == "INPUT_TOO_LONG"
            assert excinfo.value.status_code == 400
            assert excinfo.value.param == "items[0].text"
            assert client._post.await_count == 1
            mock_sleep.assert_not_called()
            await client.close()

    @pytest.mark.asyncio
    async def test_other_400_falls_through_to_request_error(self) -> None:
        with patch("sie_sdk.client.async_.aiohttp.ClientSession"):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(side_effect=[_aio_validation_error()])

            with pytest.raises(RequestError) as excinfo:
                await client.extract("gliclass-large", {"text": "hi"}, labels=["a"])

            assert not isinstance(excinfo.value, InputTooLongError)
            assert excinfo.value.code == "VALIDATION_ERROR"
            await client.close()


# ---------------------------------------------------------------------------
# Direct ``handle_error`` dispatch (locks in the secondary fallthrough so
# reordering the conditions in ``_shared.handle_error`` cannot silently
# regress the typed dispatch).
# ---------------------------------------------------------------------------


class TestHandleErrorDispatch:
    def test_dispatch_raises_input_too_long(self) -> None:
        with pytest.raises(InputTooLongError) as excinfo:
            handle_error(_resp_input_too_long("Too many tokens"))
        assert excinfo.value.code == "INPUT_TOO_LONG"
        assert excinfo.value.status_code == 400
        assert excinfo.value.param == "items[0].text"
        assert str(excinfo.value) == "Too many tokens"

    def test_dispatch_does_not_classify_other_400(self) -> None:
        with pytest.raises(RequestError) as excinfo:
            handle_error(_resp_validation_error())
        assert not isinstance(excinfo.value, InputTooLongError)
        assert excinfo.value.code == "VALIDATION_ERROR"
