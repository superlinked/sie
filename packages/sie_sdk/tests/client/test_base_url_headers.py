"""Origin confinement for optional gateway-edge headers."""

from __future__ import annotations

import json
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.async_ import _AioResponse

BASE_URL = "https://gateway.example.test"
EDGE_HEADERS = {"Modal-Key": "wk-test", "Modal-Secret": "ws-test"}


def _json_response(body: Any) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.headers = {"content-type": "application/json"}
    response.content = json.dumps(body).encode()
    response.json.return_value = body
    return response


def test_sync_headers_are_copied_and_attached_only_to_exact_base_origin() -> None:
    supplied = dict(EDGE_HEADERS)
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client = SIEClient(BASE_URL, base_url_headers=supplied)
        hook = client_cls.call_args.kwargs["event_hooks"]["request"][0]
        assert client_cls.call_args.kwargs["follow_redirects"] is False

        supplied["Modal-Key"] = "wk-mutated"
        same_origin = httpx.Request("GET", f"{BASE_URL}/v1/models", headers={"Modal-Key": "wk-overridden"})
        hook(same_origin)
        assert same_origin.headers["Modal-Key"] == "wk-test"
        assert same_origin.headers["Modal-Secret"] == "ws-test"

        cross_origin = httpx.Request("GET", "https://attacker.example/v1/models")
        hook(cross_origin)
        assert "Modal-Key" not in cross_origin.headers
        assert "Modal-Secret" not in cross_origin.headers
        assert client._websocket_headers("wss://gateway.example.test/ws/status")["Modal-Key"] == "wk-test"
        assert "Modal-Key" not in client._websocket_headers("wss://attacker.example/ws/status")
        client.close()


def test_sync_headers_never_ride_control_plane_even_on_same_origin() -> None:
    with patch("sie_sdk.client.sync.httpx.Client") as client_cls:
        client_cls.return_value.request.return_value = _json_response({"connections": []})
        client = SIEClient(
            BASE_URL,
            control_plane_url=f"{BASE_URL}/control-plane",
            org="acme",
            base_url_headers=EDGE_HEADERS,
        )
        client.connections.list()

        call = client_cls.return_value.request.call_args
        assert call.kwargs["extensions"] == {"sie_skip_base_url_headers": True}
        control_plane_request = httpx.Request(call.args[0], call.args[1], extensions=call.kwargs["extensions"])
        client_cls.call_args.kwargs["event_hooks"]["request"][0](control_plane_request)
        assert "Modal-Key" not in control_plane_request.headers
        assert "Modal-Secret" not in control_plane_request.headers
        client.close()


def test_sync_headers_never_ride_external_payload_refs() -> None:
    response = MagicMock(status_code=200, content=b"payload")
    with (
        patch("sie_sdk.client.sync.httpx.Client"),
        patch("sie_sdk.client.sync.httpx.get", return_value=response) as external_get,
    ):
        client = SIEClient(BASE_URL, base_url_headers=EDGE_HEADERS)
        assert client.jobs._read_ref("https://payload.example/chunk") == b"payload"
        assert external_get.call_args.kwargs["headers"] == {"Accept": "application/octet-stream"}
        assert "Modal-Key" not in external_get.call_args.kwargs["headers"]
        assert "Modal-Secret" not in external_get.call_args.kwargs["headers"]
        assert external_get.call_args.kwargs["follow_redirects"] is False
        client.close()


def test_sync_same_origin_capability_ref_gets_only_edge_headers() -> None:
    response = MagicMock(status_code=200, content=b"payload")
    with (
        patch("sie_sdk.client.sync.httpx.Client"),
        patch("sie_sdk.client.sync.httpx.get", return_value=response) as capability_get,
    ):
        client = SIEClient(BASE_URL, api_key="sie-secret", base_url_headers=EDGE_HEADERS)
        assert client.jobs._read_ref("https://GATEWAY.EXAMPLE.TEST:443/v1/jobs/job/chunk") == b"payload"

        assert capability_get.call_args.kwargs["headers"] == {
            "Accept": "application/octet-stream",
            **EDGE_HEADERS,
        }
        assert "Authorization" not in capability_get.call_args.kwargs["headers"]
        assert capability_get.call_args.kwargs["follow_redirects"] is False
        client.close()


@pytest.mark.parametrize(
    "ref",
    [
        "https://gateway.example.test.attacker.example/chunk",
        "https://gateway.example.test@attacker.example/chunk",
        "https://gateway.example.test:444/chunk",
        "http://gateway.example.test/chunk",
    ],
)
def test_sync_adversarial_ref_origins_stay_bare(ref: str) -> None:
    response = MagicMock(status_code=302, content=b"redirect", headers={"location": "https://attacker.example/chunk"})
    with (
        patch("sie_sdk.client.sync.httpx.Client"),
        patch("sie_sdk.client.sync.httpx.get", return_value=response) as ref_get,
    ):
        client = SIEClient(BASE_URL, api_key="sie-secret", base_url_headers=EDGE_HEADERS)
        assert client.jobs._read_ref(ref) == b"redirect"

        assert ref_get.call_count == 1
        assert ref_get.call_args.kwargs["headers"] == {"Accept": "application/octet-stream"}
        assert ref_get.call_args.kwargs["follow_redirects"] is False
        client.close()


@pytest.mark.parametrize(
    ("headers", "error"),
    [
        ({"Bad\nName": "value"}, ValueError),
        ({"X-Edge": "value\r\ninjected: yes"}, ValueError),
        ({"Authorization": "other"}, ValueError),
        ({"Keep-Alive": "timeout=5"}, ValueError),
        ({"TE": "trailers"}, ValueError),
        ({"Trailer": "X-Checksum"}, ValueError),
        ({"Upgrade": "websocket"}, ValueError),
        ({"Sec-WebSocket-Protocol": "edge-owned"}, ValueError),
        ({"X-Edge": 3}, TypeError),
    ],
)
def test_sync_rejects_unsafe_base_url_headers(headers: Any, error: type[Exception]) -> None:
    with patch("sie_sdk.client.sync.httpx.Client"), pytest.raises(error):
        SIEClient(BASE_URL, base_url_headers=headers)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://gateway.example.test",
        "http://localhost:8080",
        "gateway.example.test",
        "https://user:password@gateway.example.test",
    ],
)
def test_sync_rejects_origin_credentials_without_secure_absolute_origin(base_url: str) -> None:
    with patch("sie_sdk.client.sync.httpx.Client"), pytest.raises(ValueError, match="absolute https"):
        SIEClient(base_url, base_url_headers=EDGE_HEADERS)


class _AsyncRaw:
    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"{}",
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self.body = body

    async def read(self) -> bytes:
        return self.body


class _AsyncContext:
    def __init__(self, raw: _AsyncRaw | None = None) -> None:
        self.raw = raw or _AsyncRaw()

    async def __aenter__(self) -> _AsyncRaw:
        return self.raw

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeAsyncSession:
    def __init__(self) -> None:
        self.get_calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, **kwargs: Any) -> _AsyncContext:
        self.get_calls.append((url, kwargs))
        return _AsyncContext()

    async def close(self) -> None:
        return None


class _ExternalRefSession:
    def __init__(self, *, raw: _AsyncRaw | None = None, **_kwargs: Any) -> None:
        self.get_calls: list[tuple[str, dict[str, Any]]] = []
        self.raw = raw or _AsyncRaw()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def get(self, url: str, **kwargs: Any) -> _AsyncContext:
        self.get_calls.append((url, kwargs))
        return _AsyncContext(self.raw)


@pytest.mark.asyncio
async def test_async_headers_are_exact_origin_and_redirects_are_disabled() -> None:
    supplied = dict(EDGE_HEADERS)
    client = SIEAsyncClient(BASE_URL, base_url_headers=supplied)
    session = _FakeAsyncSession()

    supplied["Modal-Key"] = "wk-mutated"
    with patch.object(client, "_ensure_session", return_value=session):
        await client._get("/v1/models")
        await client._get("https://attacker.example/v1/models")

    same_kwargs = session.get_calls[0][1]
    assert same_kwargs["headers"]["Modal-Key"] == "wk-test"
    assert same_kwargs["headers"]["Modal-Secret"] == "ws-test"
    assert same_kwargs["allow_redirects"] is False

    cross_kwargs = session.get_calls[1][1]
    assert "Modal-Key" not in cross_kwargs.get("headers", {})
    assert "Modal-Secret" not in cross_kwargs.get("headers", {})
    assert cross_kwargs["allow_redirects"] is False
    assert client._headers_for_request("/v1/score", {"Modal-Key": "wk-overridden"})["Modal-Key"] == "wk-test"
    assert client._websocket_headers("wss://gateway.example.test/ws/status")["Modal-Secret"] == "ws-test"
    assert "Modal-Secret" not in client._websocket_headers("wss://attacker.example/ws/status")
    await client.close()


@pytest.mark.asyncio
async def test_async_headers_never_ride_control_plane_even_on_same_origin() -> None:
    client = SIEAsyncClient(
        BASE_URL,
        control_plane_url=f"{BASE_URL}/control-plane",
        org="acme",
        base_url_headers=EDGE_HEADERS,
    )
    get = AsyncMock(return_value=_AioResponse(200, b'{"connections": []}', {}))

    with patch.object(client, "_get", get):
        await client.connections.list()

    assert get.call_args.kwargs["include_base_url_headers"] is False
    await client.close()


@pytest.mark.asyncio
async def test_async_headers_never_ride_external_payload_refs() -> None:
    session = _ExternalRefSession()
    with patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session):
        client = SIEAsyncClient(BASE_URL, api_key="sie-secret", base_url_headers=EDGE_HEADERS)
        assert await client.jobs._read_ref("https://payload.example/chunk") == b"{}"
        assert session.get_calls == [
            (
                "https://payload.example/chunk",
                {"headers": {"Accept": "application/octet-stream"}, "allow_redirects": False},
            )
        ]
        await client.close()


@pytest.mark.asyncio
async def test_async_same_origin_capability_ref_gets_only_edge_headers_without_redirects() -> None:
    session = _ExternalRefSession()
    with patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session) as session_cls:
        client = SIEAsyncClient(BASE_URL, api_key="sie-secret", base_url_headers=EDGE_HEADERS)
        assert await client.jobs._read_ref("https://GATEWAY.EXAMPLE.TEST:443/v1/jobs/job/chunk") == b"{}"

        assert session.get_calls == [
            (
                "https://GATEWAY.EXAMPLE.TEST:443/v1/jobs/job/chunk",
                {
                    "headers": {"Accept": "application/octet-stream", **EDGE_HEADERS},
                    "allow_redirects": False,
                },
            )
        ]
        assert "Authorization" not in session.get_calls[0][1]["headers"]
        assert "headers" not in session_cls.call_args.kwargs
        await client.close()


@pytest.mark.asyncio
async def test_async_adversarial_redirect_ref_is_not_followed_or_credentialed() -> None:
    session = _ExternalRefSession(
        raw=_AsyncRaw(
            status=302,
            headers={"location": "https://attacker.example/chunk"},
            body=b"redirect",
        )
    )
    with patch("sie_sdk.client.async_.aiohttp.ClientSession", return_value=session):
        client = SIEAsyncClient(BASE_URL, api_key="sie-secret", base_url_headers=EDGE_HEADERS)
        assert await client.jobs._read_ref("https://gateway.example.test.attacker.example/chunk") == b"redirect"

        assert session.get_calls == [
            (
                "https://gateway.example.test.attacker.example/chunk",
                {"headers": {"Accept": "application/octet-stream"}, "allow_redirects": False},
            )
        ]
        await client.close()


@pytest.mark.parametrize(
    "base_url",
    [
        "http://gateway.example.test",
        "http://localhost:8080",
        "gateway.example.test",
        "https://user:password@gateway.example.test",
    ],
)
def test_async_rejects_origin_credentials_without_secure_absolute_origin(base_url: str) -> None:
    with pytest.raises(ValueError, match="absolute https"):
        SIEAsyncClient(base_url, base_url_headers=EDGE_HEADERS)
