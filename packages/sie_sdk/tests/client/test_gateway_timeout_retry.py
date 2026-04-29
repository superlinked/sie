# Tests for the 504 Gateway Timeout retry branch.
#
# The SDK already retries on 503 + X-SIE-Error-Code: MODEL_LOADING. The
# gateway also maps upstream timeouts (e.g. a worker cold-loading a
# model on demand) to that contract in the current release, but the
# SDK adds a defense-in-depth 504 retry so older gateways (or any path
# that still returns plain 504) don't surface as a hard failure to a
# caller who opted into wait_for_capacity=True.
#
# Covers: sync + async clients, encode (representative path; the score
# and extract paths share the same retry block by construction).

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest


def _mock_response_504() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 504
    resp.headers = {"Retry-After": "0.01", "content-type": "application/json"}
    resp.json.return_value = {"error": {"code": "MODEL_LOADING", "message": "Timeout waiting for queue result"}}
    return resp


def _mock_response_200() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/msgpack"}
    resp.content = msgpack.packb({"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True)
    return resp


def _async_response_504() -> object:
    from sie_sdk.client.async_ import _AioResponse

    return _AioResponse(
        504,
        json.dumps({"error": {"code": "MODEL_LOADING", "message": "Timeout waiting for queue result"}}).encode(),
        {"Retry-After": "0.01", "content-type": "application/json"},
    )


def _async_response_200() -> object:
    from sie_sdk.client.async_ import _AioResponse

    return _AioResponse(
        200,
        msgpack.packb({"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True),
        {"content-type": "application/msgpack"},
    )


class TestSync504Retry:
    def test_504_retried_when_wait_for_capacity_true_then_succeeds(self) -> None:
        from sie_sdk import SIEClient

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[_mock_response_504(), _mock_response_200()])
            client = SIEClient("http://localhost:8080")

            result = client.encode(
                "bge-m3",
                {"text": "hello"},
                wait_for_capacity=True,
                provision_timeout_s=2.0,
            )

            assert result["dense"].shape == (4,)
            assert mock_client.return_value.post.call_count == 2
            client.close()

    def test_504_not_retried_when_wait_for_capacity_false(self) -> None:
        from sie_sdk import SIEClient
        from sie_sdk.client.errors import ServerError

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(side_effect=[_mock_response_504()])
            client = SIEClient("http://localhost:8080")

            with pytest.raises(ServerError):
                client.encode(
                    "bge-m3",
                    {"text": "hello"},
                    wait_for_capacity=False,
                    provision_timeout_s=5.0,
                )

            assert mock_client.return_value.post.call_count == 1
            client.close()

    def test_504_retries_bounded_by_provision_timeout(self) -> None:
        from sie_sdk import ProvisioningError, SIEClient
        from sie_sdk.client.errors import ServerError

        with patch("sie_sdk.client.sync.httpx.Client") as mock_client:
            mock_client.return_value.post = MagicMock(return_value=_mock_response_504())
            client = SIEClient("http://localhost:8080")

            start = time.monotonic()
            # Either ServerError (504 surfaced via handle_error once the
            # retry budget is gone) or ProvisioningError (pre-request
            # budget check caught a freshly-zeroed remaining timeout)
            # are valid timeout behaviours — same pattern as
            # test_provision_timeout_enforced_across_retries in
            # test_timeout.py.
            with pytest.raises((ServerError, ProvisioningError)):
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


class TestAsync504Retry:
    @pytest.mark.asyncio
    async def test_504_retried_when_wait_for_capacity_true_then_succeeds(self) -> None:
        from sie_sdk import SIEAsyncClient

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(  # type: ignore
            side_effect=[_async_response_504(), _async_response_200()]
        )

        result = await client.encode(
            "bge-m3",
            {"text": "hello"},
            wait_for_capacity=True,
            provision_timeout_s=2.0,
        )

        assert result["dense"].shape == (4,)
        assert client._post.call_count == 2
        await client.close()

    @pytest.mark.asyncio
    async def test_504_not_retried_when_wait_for_capacity_false(self) -> None:
        from sie_sdk import SIEAsyncClient
        from sie_sdk.client.errors import ServerError

        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(return_value=_async_response_504())  # type: ignore

        with pytest.raises(ServerError):
            await client.encode(
                "bge-m3",
                {"text": "hello"},
                wait_for_capacity=False,
                provision_timeout_s=5.0,
            )

        assert client._post.call_count == 1
        await client.close()
