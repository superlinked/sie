"""Terminal encode timing headers remain separate from response-body timing."""

from __future__ import annotations

from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk._msgpack import packb as pack_msgpack
from sie_sdk.client._shared import parse_gateway_timing
from sie_sdk.client.async_ import _AioResponse

_HEADERS = {
    "x-queue-publish-time": "1.2",
    "x-queue-wait-time": "3.4",
    "x-queue-time": "4.6",
    "x-inference-time": "2.1",
    "x-tokenization-time": "0.0",
    "x-postprocessing-time": "0.2",
    "x-payload-fetch-time": "0.3",
}
_EXPECTED = {
    "publish_ms": 1.2,
    "result_wait_ms": 3.4,
    "worker_queue_ms_max": 4.6,
    "worker_inference_ms_max": 2.1,
    "worker_tokenization_ms_max": 0.0,
    "worker_postprocessing_ms_max": 0.2,
    "worker_payload_fetch_ms_max": 0.3,
}
_REQUEST_ID = "69ed6ea7-df59-4b0e-bd1f-73922bedc5f9"


def test_gateway_timing_uses_only_allowlisted_headers_and_keeps_units() -> None:
    headers = httpx.Headers({**_HEADERS, "x-total-time": "99", "x-private": "not-retained"})
    assert parse_gateway_timing(headers) == _EXPECTED


@pytest.mark.parametrize(
    "value",
    [None, True, 1, 1.5, "", "-1", "NaN", "inf", "1e3", "1, 2", " 1", "1 ", "\uff11", "0x10", "9" * 65],
)
def test_gateway_timing_omits_invalid_values_independently(value: object) -> None:
    assert parse_gateway_timing({"x-queue-time": value, "x-queue-wait-time": "0.0"}) == {"result_wait_ms": 0.0}
    assert parse_gateway_timing({"x-queue-time": value}) is None


def test_gateway_timing_missing_headers_do_not_create_values() -> None:
    assert parse_gateway_timing({}) is None
    assert parse_gateway_timing(httpx.Headers({"X-Queue-Time": "0.0", "X-Inference-Time": "0"})) == {
        "worker_queue_ms_max": 0.0,
        "worker_inference_ms_max": 0.0,
    }


def _success(headers: dict[str, str], *, body_timing: bool = False, count: int = 2) -> httpx.Response:
    payload: dict[str, object] = {
        "model": "m",
        "items": [{"dense": {"dims": 1, "values": np.array([0.5], dtype=np.float32)}} for _ in range(count)],
    }
    if body_timing:
        payload["timing"] = {"total_ms": 30.0, "queue_ms": 20.0}
    return httpx.Response(
        200,
        content=pack_msgpack(payload),
        headers={"content-type": "application/msgpack", "x-sie-request-id": _REQUEST_ID, **headers},
    )


def _async_response(response: httpx.Response) -> _AioResponse:
    return _AioResponse(response.status_code, response.content, response.headers)


@pytest.mark.parametrize("body_timing", [False, True])
@pytest.mark.parametrize("request_metadata", [False, True])
def test_sync_encode_preserves_gateway_and_body_timing_separately(body_timing: bool, request_metadata: bool) -> None:
    response = _success(_HEADERS, body_timing=body_timing)
    if not request_metadata:
        del response.headers["x-sie-request-id"]
    with httpx.Client(
        base_url="https://gateway.example.test", transport=httpx.MockTransport(lambda _: response)
    ) as http:
        with SIEClient("https://gateway.example.test", http_client=http) as client:
            results = client.encode("m", [{"text": "one"}, {"text": "two"}])

    assert len(results) == 2
    assert all(result["gateway_timing"] == _EXPECTED for result in results)
    assert all(result.get("request") == ({"id": _REQUEST_ID} if request_metadata else None) for result in results)
    assert all(
        result.get("timing") == ({"total_ms": 30.0, "queue_ms": 20.0} if body_timing else None) for result in results
    )
    results[0]["gateway_timing"]["publish_ms"] = 999
    assert results[1]["gateway_timing"]["publish_ms"] == 1.2


@pytest.mark.parametrize("body_timing", [False, True])
@pytest.mark.parametrize("request_metadata", [False, True])
async def test_async_encode_preserves_gateway_and_body_timing_separately(
    body_timing: bool, request_metadata: bool
) -> None:
    response = _success(_HEADERS, body_timing=body_timing)
    if not request_metadata:
        del response.headers["x-sie-request-id"]
    client = SIEAsyncClient("https://gateway.example.test")
    with patch.object(client, "_post", AsyncMock(return_value=_async_response(response))):
        try:
            results = await client.encode("m", [{"text": "one"}, {"text": "two"}])
        finally:
            await client.close()

    assert len(results) == 2
    assert all(result["gateway_timing"] == _EXPECTED for result in results)
    assert all(result.get("request") == ({"id": _REQUEST_ID} if request_metadata else None) for result in results)
    assert all(
        result.get("timing") == ({"total_ms": 30.0, "queue_ms": 20.0} if body_timing else None) for result in results
    )
    results[0]["gateway_timing"]["publish_ms"] = 999
    assert results[1]["gateway_timing"]["publish_ms"] == 1.2


def _retry() -> httpx.Response:
    return httpx.Response(
        503,
        headers={"Retry-After": "0", "x-queue-time": "999", "x-queue-publish-time": "999"},
        json={"error": {"code": "PROVISIONING", "message": "starting"}},
    )


@pytest.mark.parametrize("terminal_headers", [{}, {"x-queue-time": "NaN"}, {"x-queue-time": "2.5"}])
def test_sync_retry_and_next_call_never_reuse_previous_timing(terminal_headers: dict[str, str]) -> None:
    responses = iter([_success(_HEADERS, count=1), _retry(), _success(terminal_headers, count=1)])
    with httpx.Client(
        base_url="https://gateway.example.test", transport=httpx.MockTransport(lambda _: next(responses))
    ) as http:
        with SIEClient("https://gateway.example.test", http_client=http) as client:
            assert client.encode("m", {"text": "one"})["gateway_timing"] == _EXPECTED
            with patch("sie_sdk.client.sync.time.sleep"):
                result = client.encode("m", {"text": "two"})
            assert client.last_retry_count == 1
    assert result.get("gateway_timing") == (
        {"worker_queue_ms_max": 2.5} if "2.5" in terminal_headers.values() else None
    )
    assert "timing" not in result


@pytest.mark.parametrize("terminal_headers", [{}, {"x-queue-time": "NaN"}, {"x-queue-time": "2.5"}])
async def test_async_retry_and_next_call_never_reuse_previous_timing(terminal_headers: dict[str, str]) -> None:
    responses = [
        _async_response(r) for r in [_success(_HEADERS, count=1), _retry(), _success(terminal_headers, count=1)]
    ]
    client = SIEAsyncClient("https://gateway.example.test")
    with patch.object(client, "_post", AsyncMock(side_effect=responses)):
        try:
            assert (await client.encode("m", {"text": "one"}))["gateway_timing"] == _EXPECTED
            with patch("sie_sdk.client.async_.asyncio.sleep", AsyncMock()):
                result = await client.encode("m", {"text": "two"})
            assert client.last_retry_count == 1
        finally:
            await client.close()
    assert result.get("gateway_timing") == (
        {"worker_queue_ms_max": 2.5} if "2.5" in terminal_headers.values() else None
    )
    assert "timing" not in result


class _AsyncResponseContext:
    def __init__(self, response: httpx.Response) -> None:
        self.status = response.status_code
        self.headers = response.headers
        self._content = response.content

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def read(self) -> bytes:
        return self._content


@pytest.mark.parametrize("terminal_headers", [{}, _HEADERS])
def test_sync_continuation_uses_only_terminal_headers(terminal_headers: dict[str, str]) -> None:
    methods: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "POST":
            return httpx.Response(303, headers={"Location": "/v1/encode/m?__modal_attempt_token=opaque", **_HEADERS})
        return _success(terminal_headers, count=1)

    with httpx.Client(base_url="https://gateway.example.test", transport=httpx.MockTransport(respond)) as http:
        with SIEClient("https://gateway.example.test", http_client=http) as client:
            result = client.encode("m", {"text": "one"})
            assert client.last_retry_count == 0

    assert methods == ["POST", "GET"]
    assert result.get("gateway_timing") == (_EXPECTED if terminal_headers else None)


@pytest.mark.parametrize("terminal_headers", [{}, _HEADERS])
async def test_async_continuation_uses_only_terminal_headers(terminal_headers: dict[str, str]) -> None:
    redirect = _AioResponse(303, b"", {"Location": "/v1/encode/m?__modal_attempt_token=opaque", **_HEADERS})
    session = MagicMock(closed=False)
    session.get.return_value = _AsyncResponseContext(_success(terminal_headers, count=1))
    session.close = AsyncMock()
    client = SIEAsyncClient("https://gateway.example.test")
    client._session = session
    with patch.object(client, "_post", AsyncMock(return_value=redirect)) as post:
        try:
            result = await client.encode("m", {"text": "one"})
            assert client.last_retry_count == 0
        finally:
            await client.close()

    post.assert_awaited_once()
    session.get.assert_called_once()
    assert result.get("gateway_timing") == (_EXPECTED if terminal_headers else None)
