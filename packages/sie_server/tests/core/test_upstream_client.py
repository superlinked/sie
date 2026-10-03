"""The upstream egress client: no redirects, no ambient proxy, credential per request."""

from __future__ import annotations

import logging

import httpx
import pytest
from sie_server.config.upstreams import Upstream, UpstreamCredentialError
from sie_server.core.upstream_client import UpstreamRedirectRefusedError, upstream_client

CANARY = "sk-canary-0b8e5d7c3a91f246"


def make_upstream(**overrides: object) -> Upstream:
    spec: dict[str, object] = {
        "kind": "sie",
        "base_url": "https://sie.example.internal",
        "api_key_secret": "TEAM_SIE_KEY",
        "rate_cap": {"requests_per_minute": 60, "max_concurrency": 4},
    }
    spec.update(overrides)
    return Upstream.model_validate(spec)


class Recorder:
    def __init__(self, respond: httpx.Response) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._respond


async def test_the_credential_is_sent_only_as_a_header_and_never_rendered(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("TEAM_SIE_KEY", CANARY)
    recorder = Recorder(httpx.Response(500, json={"error": "boom"}))
    caplog.set_level(logging.DEBUG)

    async with upstream_client(make_upstream(), transport=httpx.MockTransport(recorder)) as client:
        response = await client.post("/v1/encode/BAAI/bge-m3", json={"items": [{"text": "hi"}]})
        with pytest.raises(httpx.HTTPStatusError) as raised:
            response.raise_for_status()
        client_repr = repr(client)

    assert recorder.requests[0].headers["authorization"] == f"Bearer {CANARY}"
    assert str(recorder.requests[0].url) == "https://sie.example.internal/v1/encode/BAAI/bge-m3"
    assert CANARY not in str(raised.value)
    assert CANARY not in repr(raised.value.request.headers)
    assert CANARY not in client_repr
    assert CANARY not in caplog.text


async def test_the_credential_is_read_for_each_request(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(httpx.Response(200, json={}))
    async with upstream_client(make_upstream(), transport=httpx.MockTransport(recorder)) as client:
        monkeypatch.setenv("TEAM_SIE_KEY", "first")
        await client.get("/v1/models")
        monkeypatch.setenv("TEAM_SIE_KEY", "rotated")
        await client.get("/v1/models")

    assert [request.headers["authorization"] for request in recorder.requests] == [
        "Bearer first",
        "Bearer rotated",
    ]


async def test_a_missing_credential_stops_the_request_before_it_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEAM_SIE_KEY", raising=False)
    recorder = Recorder(httpx.Response(200, json={}))

    async with upstream_client(make_upstream(), transport=httpx.MockTransport(recorder)) as client:
        with pytest.raises(UpstreamCredentialError, match="TEAM_SIE_KEY is not set"):
            await client.get("/v1/models")

    assert recorder.requests == []


async def test_a_malformed_credential_is_refused_before_anything_is_sent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("TEAM_SIE_KEY", f"{CANARY} extra")
    recorder = Recorder(httpx.Response(200, json={}))
    caplog.set_level(logging.DEBUG)

    async with upstream_client(make_upstream(), transport=httpx.MockTransport(recorder)) as client:
        with pytest.raises(UpstreamCredentialError) as raised:
            await client.get("/v1/models")

    assert recorder.requests == []
    assert CANARY not in str(raised.value)
    assert CANARY not in caplog.text


async def test_an_upstream_without_a_credential_sends_no_authorization() -> None:
    recorder = Recorder(httpx.Response(200, json={}))
    upstream = make_upstream(api_key_secret=None, base_url="http://127.0.0.1:8080")

    async with upstream_client(upstream, transport=httpx.MockTransport(recorder)) as client:
        await client.get("/v1/models")

    assert "authorization" not in recorder.requests[0].headers


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
async def test_redirects_are_refused_and_never_followed(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    monkeypatch.setenv("TEAM_SIE_KEY", CANARY)
    recorder = Recorder(httpx.Response(status, headers={"location": "https://collector.example/steal"}))

    async with upstream_client(make_upstream(), transport=httpx.MockTransport(recorder)) as client:
        with pytest.raises(UpstreamRedirectRefusedError) as raised:
            await client.get("/v1/models")

    assert [request.url.host for request in recorder.requests] == ["sie.example.internal"]
    assert "collector.example" not in str(raised.value)


def test_ambient_proxy_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://ambient-proxy.example:3128")

    client = upstream_client(make_upstream())

    assert client.trust_env is False
    assert client._mounts == {}


def test_only_the_declared_proxy_is_used() -> None:
    client = upstream_client(make_upstream(proxy_url="http://proxy.example.internal:3128"))

    proxies = [transport._pool._proxy_url for transport in client._mounts.values()]
    assert [(url.host, url.port) for url in proxies] == [(b"proxy.example.internal", 3128)]
