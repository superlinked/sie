"""Tests for the ``ModelLoadFailedError`` short-circuit (sie-test#85).

A 502 ``MODEL_LOAD_FAILED`` response must:
- raise :class:`ModelLoadFailedError` immediately on the first response
- never engage the ``MODEL_LOADING`` retry budget (no sleep, no retry)
- expose the structured ``error_class`` / ``permanent`` / ``attempts``
  fields from the server payload

The 503 ``MODEL_LOADING`` retry behavior must be unchanged.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.async_ import _AioResponse
from sie_sdk.client.errors import ModelLoadFailedError, ModelLoadingError


def _resp_load_failed(
    *,
    error_class: str = "GATED",
    permanent: bool = True,
    attempts: int = 1,
    message: str = "Model 'org/test' failed to load (GATED): missing HF_TOKEN",
) -> MagicMock:
    """Build a 502 MODEL_LOAD_FAILED response with no Retry-After."""
    resp = MagicMock()
    resp.status_code = 502
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {
        "detail": {
            "code": "MODEL_LOAD_FAILED",
            "message": message,
            "error_class": error_class,
            "permanent": permanent,
            "attempts": attempts,
        }
    }
    return resp


def _resp_model_loading() -> MagicMock:
    """503 MODEL_LOADING with a tiny Retry-After (sanity contrast)."""
    resp = MagicMock()
    resp.status_code = 503
    resp.headers = {"Retry-After": "0.01", "content-type": "application/json"}
    resp.json.return_value = {"detail": {"code": "MODEL_LOADING", "message": "loading"}}
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


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class TestSyncModelLoadFailed:
    def test_raises_immediately_on_first_response(self) -> None:
        """No retries are attempted; the error surfaces on the first call."""
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep") as mock_sleep,
        ):
            mock_client.return_value.post = MagicMock(side_effect=[_resp_load_failed()])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ModelLoadFailedError) as excinfo:
                client.encode("org/test", {"text": "hi"})

            assert excinfo.value.model == "org/test"
            assert excinfo.value.error_class == "GATED"
            assert excinfo.value.permanent is True
            assert excinfo.value.attempts == 1
            # Critical: no retry happened.
            assert mock_client.return_value.post.call_count == 1
            mock_sleep.assert_not_called()
            client.close()

    def test_does_not_consume_model_loading_budget(self) -> None:
        """A 502 must never be retried as if it were a 503 MODEL_LOADING."""
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            mock_client.return_value.post = MagicMock(
                side_effect=[_resp_load_failed(error_class="DEPENDENCY", permanent=True)]
            )
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ModelLoadFailedError) as excinfo:
                client.encode("org/test", {"text": "hi"}, provision_timeout_s=300.0)

            assert excinfo.value.error_class == "DEPENDENCY"
            assert mock_client.return_value.post.call_count == 1
            client.close()

    def test_503_model_loading_still_retries(self) -> None:
        """Regression guard: the 502 short-circuit must not break 503 retries."""
        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep"),
        ):
            mock_client.return_value.post = MagicMock(
                side_effect=[_resp_model_loading(), _resp_model_loading(), _resp_200_encode()]
            )
            client = SIEClient("http://localhost:8080")

            result = client.encode("bge-m3", {"text": "hi"})

            assert result["dense"].shape == (4,)
            assert mock_client.return_value.post.call_count == 3
            client.close()

    def test_modelloadfailed_distinct_from_modelloading(self) -> None:
        """The two error classes must not be confused in catch sites."""
        assert not issubclass(ModelLoadFailedError, ModelLoadingError)
        assert not issubclass(ModelLoadingError, ModelLoadFailedError)

    def test_score_endpoint_short_circuits(self) -> None:
        """The 502 short-circuit applies to /score too, not just /encode."""
        score_200 = MagicMock()
        score_200.status_code = 200
        score_200.headers = {"content-type": "application/msgpack"}
        score_200.content = msgpack.packb(
            {"model": "m", "scores": [{"item_id": "a", "score": 0.5, "rank": 0}]},
            use_bin_type=True,
        )

        with (
            patch("sie_sdk.client.sync.httpx.Client") as mock_client,
            patch("sie_sdk.client.sync.time.sleep") as mock_sleep,
        ):
            mock_client.return_value.post = MagicMock(side_effect=[_resp_load_failed()])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ModelLoadFailedError):
                client.score("org/test", query={"text": "q"}, items=[{"text": "d"}])

            assert mock_client.return_value.post.call_count == 1
            mock_sleep.assert_not_called()
            client.close()


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


def _aio_load_failed(
    *,
    error_class: str = "GATED",
    permanent: bool = True,
    attempts: int = 1,
) -> object:
    return _AioResponse(
        502,
        json.dumps(
            {
                "detail": {
                    "code": "MODEL_LOAD_FAILED",
                    "message": "Model 'org/test' failed to load",
                    "error_class": error_class,
                    "permanent": permanent,
                    "attempts": attempts,
                }
            }
        ).encode(),
        {"content-type": "application/json"},
    )


def _aio_model_loading() -> object:
    return _AioResponse(
        503,
        json.dumps({"detail": {"code": "MODEL_LOADING", "message": "loading"}}).encode(),
        {"Retry-After": "0.01", "content-type": "application/json"},
    )


def _aio_200_encode() -> object:
    return _AioResponse(
        200,
        msgpack.packb({"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True),
        {"content-type": "application/msgpack"},
    )


class TestAsyncModelLoadFailed:
    @pytest.mark.asyncio
    async def test_raises_immediately(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession"),
            patch("sie_sdk.client.async_.asyncio.sleep") as mock_sleep,
        ):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(side_effect=[_aio_load_failed()])

            with pytest.raises(ModelLoadFailedError) as excinfo:
                await client.encode("org/test", {"text": "hi"})

            assert excinfo.value.error_class == "GATED"
            assert excinfo.value.permanent is True
            assert client._post.await_count == 1
            mock_sleep.assert_not_called()
            await client.close()

    @pytest.mark.asyncio
    async def test_503_model_loading_still_retries(self) -> None:
        with (
            patch("sie_sdk.client.async_.aiohttp.ClientSession"),
            patch("sie_sdk.client.async_.asyncio.sleep"),
        ):
            client = SIEAsyncClient("http://localhost:8080")
            client._post = AsyncMock(side_effect=[_aio_model_loading(), _aio_model_loading(), _aio_200_encode()])

            result = await client.encode("bge-m3", {"text": "hi"})
            assert result["dense"].shape == (4,)
            assert client._post.await_count == 3
            await client.close()
