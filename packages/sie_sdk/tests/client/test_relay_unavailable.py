"""SIE Cloud refusals to serve through SIE Relay surface as ``RelayUnavailableError``.

SIE Cloud answers ``403`` with ``THIRD_PARTY_SERVING_DISABLED``,
``THIRD_PARTY_RUNTIME_UNLISTED`` or ``THIRD_PARTY_REGION_UNAVAILABLE`` when it
will not serve a request through SIE Relay, its partner capacity. The region
refusal also names ``key_region`` and ``deployment_region``. The refusal is
terminal: the SDK raises it on the first response.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import numpy as np
import pytest
from sie_sdk import AccountInactiveError, RelayUnavailableError, RequestError, SIEAsyncClient, SIEClient
from sie_sdk.client._shared import handle_error
from sie_sdk.client.async_ import _AioResponse

RELAY_CODES = (
    "THIRD_PARTY_SERVING_DISABLED",
    "THIRD_PARTY_RUNTIME_UNLISTED",
    "THIRD_PARTY_REGION_UNAVAILABLE",
)


def _refusal(code: str, **fields: Any) -> dict[str, Any]:
    return {"detail": {"code": code, "message": f"refused: {code}", **fields}}


def _response(status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": "application/json", **(headers or {})}
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


def _encode_success() -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/msgpack"}
    resp.content = msgpack.packb({"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True)
    return resp


@pytest.mark.parametrize("code", RELAY_CODES)
def test_each_relay_code_is_a_typed_request_error(code: str) -> None:
    with pytest.raises(RelayUnavailableError) as raised:
        handle_error(_response(403, _refusal(code)))

    error = raised.value
    assert isinstance(error, RequestError)
    assert error.code == code
    assert error.status_code == 403
    assert str(error) == f"refused: {code}"
    assert error.key_region is None
    assert error.deployment_region is None


def test_the_region_refusal_carries_both_regions() -> None:
    body = _refusal("THIRD_PARTY_REGION_UNAVAILABLE", key_region="eu", deployment_region="us")

    with pytest.raises(RelayUnavailableError) as raised:
        handle_error(_response(403, body))

    assert raised.value.key_region == "eu"
    assert raised.value.deployment_region == "us"


def test_a_key_without_a_region_reads_as_none() -> None:
    body = _refusal("THIRD_PARTY_REGION_UNAVAILABLE", key_region=None, deployment_region="us")

    with pytest.raises(RelayUnavailableError) as raised:
        handle_error(_response(403, body))

    assert raised.value.key_region is None
    assert raised.value.deployment_region == "us"


def test_regions_that_are_not_strings_are_dropped() -> None:
    body = _refusal("THIRD_PARTY_REGION_UNAVAILABLE", key_region=7, deployment_region=["us"])

    with pytest.raises(RelayUnavailableError) as raised:
        handle_error(_response(403, body))

    assert raised.value.key_region is None
    assert raised.value.deployment_region is None


def test_the_error_code_header_identifies_a_relay_refusal_in_the_openai_envelope() -> None:
    body = {"error": {"message": "refused", "type": "invalid_request_error", "code": "permission_denied"}}
    headers = {"X-SIE-Error-Code": "THIRD_PARTY_SERVING_DISABLED", "X-SIE-Request-ID": "req-relay-1"}

    with pytest.raises(RelayUnavailableError) as raised:
        handle_error(_response(403, body, headers))

    assert raised.value.code == "THIRD_PARTY_SERVING_DISABLED"
    assert str(raised.value) == "refused"
    assert raised.value.request == {"id": "req-relay-1"}


@pytest.mark.parametrize(
    ("status", "code"),
    [(403, "INVALID_KEY"), (403, "ACCOUNT_SUSPENDED"), (400, "THIRD_PARTY_SERVING_DISABLED")],
)
def test_other_refusals_are_not_relay_refusals(status: int, code: str) -> None:
    with pytest.raises(RequestError) as raised:
        handle_error(_response(status, _refusal(code)))

    assert not isinstance(raised.value, RelayUnavailableError)
    assert isinstance(raised.value, AccountInactiveError) == (code == "ACCOUNT_SUSPENDED")


@pytest.mark.parametrize("code", RELAY_CODES)
def test_the_sync_client_raises_it_on_the_first_response(code: str) -> None:
    with (
        patch("sie_sdk.client.sync.httpx.Client") as mock_client,
        patch("sie_sdk.client.sync.time.sleep") as mock_sleep,
    ):
        mock_client.return_value.post = MagicMock(
            side_effect=[_response(403, _refusal(code, deployment_region="us")), _encode_success()]
        )
        client = SIEClient("http://localhost:8080")

        with pytest.raises(RelayUnavailableError) as raised:
            client.encode("bge-m3", {"text": "hi"})

        assert raised.value.code == code
        assert raised.value.deployment_region == "us"
        assert mock_client.return_value.post.call_count == 1
        assert mock_sleep.call_count == 0
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", RELAY_CODES)
async def test_the_async_client_raises_it_on_the_first_response(code: str) -> None:
    body = _refusal(code, key_region="eu", deployment_region="us")
    with (
        patch("sie_sdk.client.async_.aiohttp.ClientSession"),
        patch("sie_sdk.client.async_.asyncio.sleep") as mock_sleep,
    ):
        client = SIEAsyncClient("http://localhost:8080")
        client._post = AsyncMock(
            side_effect=[
                _AioResponse(403, json.dumps(body).encode(), {"content-type": "application/json"}),
                _AioResponse(
                    200,
                    msgpack.packb({"items": [{"dense": {"dims": 4, "values": np.zeros(4)}}]}, use_bin_type=True),
                    {"content-type": "application/msgpack"},
                ),
            ]
        )

        with pytest.raises(RelayUnavailableError) as raised:
            await client.encode("bge-m3", {"text": "hi"})

        assert raised.value.code == code
        assert (raised.value.key_region, raised.value.deployment_region) == ("eu", "us")
        assert client._post.await_count == 1
        assert mock_sleep.call_count == 0
        await client.close()
