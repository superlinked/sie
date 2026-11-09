"""Credential and direct-connection contracts with real websockets 14 and 15."""

import asyncio
import json
import threading
import urllib.request
from http import HTTPStatus

import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.errors import RequestError
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_watch_direct_connection_auth_and_no_proxy(monkeypatch, synchronous):
    # Invalid ambient proxies make any accidental use fail without contacting
    # a third party. Disable environment and platform bypass rules.
    for name in ("http_proxy", "https_proxy", "all_proxy", "ws_proxy", "wss_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
        monkeypatch.setenv(name.upper(), "http://127.0.0.1:1")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setattr(urllib.request, "proxy_bypass", lambda _host: False)
    seen = []

    async def handler(connection):
        seen.append((connection.request.path, connection.request.headers.get("Authorization")))
        await connection.send(json.dumps({"status": "ready"}))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        if synchronous:

            def consume():
                with SIEClient(url, api_key="test-token") as client:
                    return list(client.watch(mode="worker"))

            result = await asyncio.to_thread(consume)
        else:
            async with SIEAsyncClient(url, api_key="test-token") as client:
                result = [item async for item in client.watch(mode="worker")]
    assert result == [{"status": "ready"}]
    assert seen == [("/ws/status", "Bearer test-token")]


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous", [False, True])
async def test_watch_rejects_redirect_without_forwarding_credentials(synchronous):
    target_called = threading.Event()
    seen = []

    async def target(connection):
        target_called.set()
        await connection.close()

    async with serve(target, "127.0.0.1", 0) as destination:
        destination_port = destination.sockets[0].getsockname()[1]

        def redirect(connection, request):
            seen.append(request.headers.get("Authorization"))
            return Response(
                HTTPStatus.FOUND, "Found", Headers({"Location": f"ws://127.0.0.1:{destination_port}/stolen"})
            )

        async with serve(target, "127.0.0.1", 0, process_request=redirect) as source:
            url = f"http://127.0.0.1:{source.sockets[0].getsockname()[1]}"
            with pytest.raises(RequestError, match="302"):
                if synchronous:

                    def consume():
                        with SIEClient(url, api_key="test-token") as client:
                            return list(client.watch(mode="worker"))

                    await asyncio.to_thread(consume)
                else:
                    async with SIEAsyncClient(url, api_key="test-token") as client:
                        async for _ in client.watch(mode="worker"):
                            pass
    assert seen == ["Bearer test-token"]
    assert not target_called.is_set()
