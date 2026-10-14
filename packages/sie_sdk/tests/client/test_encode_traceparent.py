"""Call-local encode trace contexts never mutate pooled-client defaults."""

from __future__ import annotations

import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk._msgpack import packb as pack_msgpack
from sie_sdk.client._shared import validate_encode_traceparent
from sie_sdk.client.errors import RequestError

_BASE = "https://gateway.example.test"
_DEFAULT = "00-11111111111111111111111111111111-1111111111111111-01"
_FIRST = "00-22222222222222222222222222222222-2222222222222222-01"
_SECOND = "00-33333333333333333333333333333333-3333333333333333-00"
_EDGE = {"TraceParent": _DEFAULT, "TraceState": "vendor=default", "X-Edge": "edge-test"}
_INVALID = [
    "",
    _FIRST.replace("2", "A"),
    _FIRST.replace("00-", "01-", 1),
    _FIRST.replace("00-", "ff-", 1),
    "00-" + "0" * 32 + "-2222222222222222-01",
    "00-22222222222222222222222222222222-" + "0" * 16 + "-01",
    _FIRST + "-extra",
    " " + _FIRST,
    _FIRST + "\r\nX-Other: value",
    pytest.param("x" * 10000, id="oversized"),
    True,
    42,
]


def _success() -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "application/msgpack"},
        content=pack_msgpack({"items": [{"dense": {"dims": 1, "values": np.array([0.5], dtype=np.float32)}}]}),
    )


def _retry() -> httpx.Response:
    return httpx.Response(
        503,
        headers={"Retry-After": "0"},
        json={"error": {"code": "PROVISIONING", "message": "starting"}},
    )


def _redirect(location: str = "/v1/encode/m?__modal_attempt_token=opaque") -> httpx.Response:
    return httpx.Response(303, headers={"Location": location})


class _AsyncContext:
    def __init__(self, response: httpx.Response) -> None:
        self.status = response.status_code
        self.headers = response.headers
        self.content = response.content

    async def __aenter__(self) -> Self:
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def read(self) -> bytes:
        return self.content


@pytest.mark.parametrize("value", _INVALID)
def test_sync_invalid_traceparent_fails_before_io(value: Any) -> None:
    with SIEClient(_BASE, api_key="") as client, patch.object(client._client, "post") as post:
        with pytest.raises((TypeError, ValueError), match="traceparent"):
            client.encode("m", {"text": "one"}, traceparent=value)
        post.assert_not_called()


@pytest.mark.parametrize("value", _INVALID)
async def test_async_invalid_traceparent_fails_before_io(value: Any) -> None:
    async with SIEAsyncClient(_BASE, api_key="") as client:
        with patch.object(client, "_post") as post:
            with pytest.raises((TypeError, ValueError), match="traceparent"):
                await client.encode("m", {"text": "one"}, traceparent=value)
            post.assert_not_called()
            assert client._session is None


@pytest.mark.parametrize("flags", ["00", "01", "02", "ff"])
def test_traceparent_preserves_all_w3c_flag_bits(flags: str) -> None:
    validate_encode_traceparent(_FIRST[:-2] + flags)


@pytest.mark.parametrize("edge", [{}, _EDGE])
def test_sync_override_is_call_local_and_keeps_body_and_auth(edge: dict[str, str]) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _success()

    with httpx.Client(base_url=_BASE, transport=httpx.MockTransport(respond)) as http:
        with SIEClient(_BASE, api_key="api-test", base_url_headers=edge, http_client=http) as client:
            for context in [None, _FIRST, None]:
                client.encode("m", {"text": "one"}, traceparent=context)
            assert client._base_url_headers == edge
            assert "traceparent" not in http.headers

    assert len(requests) == 3
    assert requests[0].content == requests[1].content == requests[2].content
    for index in [0, 2]:
        assert requests[index].headers.get("traceparent") == (_DEFAULT if edge else None)
        assert requests[index].headers.get("tracestate") == ("vendor=default" if edge else None)
    assert requests[1].headers["traceparent"] == _FIRST
    assert "tracestate" not in requests[1].headers
    assert all(request.headers["authorization"] == "Bearer api-test" for request in requests)
    assert all(request.headers.get("x-edge") == ("edge-test" if edge else None) for request in requests)


@pytest.mark.parametrize("edge", [{}, _EDGE])
async def test_async_override_is_call_local_and_keeps_body_and_auth(edge: dict[str, str]) -> None:
    session = MagicMock(closed=False)
    session.post.side_effect = lambda *_args, **_kwargs: _AsyncContext(_success())
    session.close = AsyncMock()
    with patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session) as factory:
        async with SIEAsyncClient(_BASE, api_key="api-test", base_url_headers=edge) as client:
            for context in [None, _FIRST, None]:
                await client.encode("m", {"text": "one"}, traceparent=context)
            assert client._base_url_headers == edge
            assert "traceparent" not in client._headers
        factory.assert_called_once()
        assert factory.call_args.kwargs["headers"]["Authorization"] == "Bearer api-test"
    calls = session.post.call_args_list
    headers = [httpx.Headers(call.kwargs.get("headers", {})) for call in calls]
    assert calls[0].kwargs["data"] == calls[1].kwargs["data"] == calls[2].kwargs["data"]
    for index in [0, 2]:
        assert headers[index].get("traceparent") == (_DEFAULT if edge else None)
        assert headers[index].get("tracestate") == ("vendor=default" if edge else None)
    assert headers[1]["traceparent"] == _FIRST
    assert "tracestate" not in headers[1]
    assert all(header.get("x-edge") == ("edge-test" if edge else None) for header in headers)


def test_sync_concurrent_calls_keep_context_through_retry_and_continuation() -> None:
    barrier = Barrier(2)
    calls: list[tuple[str, str]] = []
    posts: Counter[str] = Counter()

    def respond(request: httpx.Request) -> httpx.Response:
        context = request.headers["traceparent"]
        calls.append((context, request.method))
        assert "tracestate" not in request.headers
        if request.method == "POST":
            posts[context] += 1
            if posts[context] == 1:
                barrier.wait(timeout=5)
                return _retry()
            return _redirect()
        return _success()

    with httpx.Client(base_url=_BASE, transport=httpx.MockTransport(respond)) as http:
        with SIEClient(_BASE, api_key="", base_url_headers=_EDGE, http_client=http) as client:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(client.encode, "m", {"text": "one"}, traceparent=c) for c in [_FIRST, _SECOND]
                ]
                assert all(future.result()["dense"].shape == (1,) for future in futures)
            assert client._client is http
            assert client._base_url_headers == _EDGE
    for context in [_FIRST, _SECOND]:
        assert [method for observed, method in calls if observed == context] == ["POST", "POST", "GET"]


async def test_async_concurrent_calls_keep_context_through_retry_and_continuation() -> None:
    calls: list[tuple[str, str]] = []
    posts: Counter[str] = Counter()

    def respond(method: str, _url: str, **kwargs: Any) -> _AsyncContext:
        headers = httpx.Headers(kwargs["headers"])
        context = headers["traceparent"]
        calls.append((context, method))
        assert "tracestate" not in headers
        assert kwargs["allow_redirects"] is False
        if method == "POST":
            posts[context] += 1
            return _AsyncContext(_retry() if posts[context] == 1 else _redirect())
        return _AsyncContext(_success())

    session = MagicMock(closed=False)
    session.post.side_effect = lambda url, **kwargs: respond("POST", url, **kwargs)
    session.get.side_effect = lambda url, **kwargs: respond("GET", url, **kwargs)
    session.close = AsyncMock()
    with patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session) as factory:
        async with SIEAsyncClient(_BASE, api_key="", base_url_headers=_EDGE) as client:
            results = await asyncio.gather(
                client.encode("m", {"text": "one"}, traceparent=_FIRST),
                client.encode("m", {"text": "one"}, traceparent=_SECOND),
            )
            assert all(result["dense"].shape == (1,) for result in results)
            assert client._base_url_headers == _EDGE
        factory.assert_called_once()
    for context in [_FIRST, _SECOND]:
        assert [method for observed, method in calls if observed == context] == ["POST", "POST", "GET"]


def test_sync_call_context_is_not_forwarded_to_cross_origin_or_control_plane() -> None:
    with SIEClient(_BASE, api_key="", base_url_headers=_EDGE) as client:
        hook = client._client.event_hooks["request"][0]
        for url, skip in [("https://other.example.test/path", False), (f"{_BASE}/control-plane", True)]:
            request = httpx.Request(
                "POST", url, extensions={"sie_encode_traceparent": _FIRST, "sie_skip_base_url_headers": skip}
            )
            hook(request)
            assert "traceparent" not in request.headers
            assert "tracestate" not in request.headers


async def test_async_call_context_is_not_forwarded_to_cross_origin_or_control_plane() -> None:
    async with SIEAsyncClient(_BASE, api_key="", base_url_headers=_EDGE) as client:
        for url, include in [("https://other.example.test/path", True), ("/control-plane", False)]:
            headers = client._headers_for_request(url, {}, include_base_url_headers=include, traceparent=_FIRST)
            assert "traceparent" not in {key.lower() for key in headers}
            assert "tracestate" not in {key.lower() for key in headers}


def test_sync_cross_origin_continuation_is_not_followed() -> None:
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _redirect("https://other.example.test/v1/encode/m?__modal_attempt_token=opaque")

    with httpx.Client(base_url=_BASE, transport=httpx.MockTransport(respond)) as http:
        with SIEClient(_BASE, api_key="", http_client=http) as client:
            with pytest.raises(RequestError, match="Unexpected encode HTTP response"):
                client.encode("m", {"text": "one"}, traceparent=_FIRST)
    assert len(seen) == 1
    assert seen[0].url.host == "gateway.example.test"


async def test_async_cross_origin_continuation_is_not_followed() -> None:
    session = MagicMock(closed=False)
    session.post.return_value = _AsyncContext(
        _redirect("https://other.example.test/v1/encode/m?__modal_attempt_token=opaque")
    )
    session.close = AsyncMock()
    with patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session):
        async with SIEAsyncClient(_BASE, api_key="") as client:
            with pytest.raises(RequestError, match="Unexpected encode HTTP response"):
                await client.encode("m", {"text": "one"}, traceparent=_FIRST)
    session.post.assert_called_once()
    session.get.assert_not_called()
