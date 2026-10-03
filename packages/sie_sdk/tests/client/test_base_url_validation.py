"""Construction-time base_url scheme validation (both clients).

A scheme-less base_url ("localhost:8080") previously surfaced only at request
time — as an opaque ``httpx.UnsupportedProtocol`` on the sync client — or was
silently retried as a connect failure for the whole provision budget. Both
clients must reject it eagerly with a pointed ``ValueError``.
"""

from __future__ import annotations

import pytest
from sie_sdk import SIEAsyncClient, SIEClient

_BAD_BASE_URLS = [
    "localhost:8080",
    "example.com",
    "//host:8080",
    "ftp://host:21",
    "unix:///tmp/sie.sock",
    "",
    # Valid scheme but no host — parses, cannot be connected to.
    "http:///v1",
    "https:///v1",
    "http://",
    "http://:8080",
    # Textual / out-of-range port — must fail at construction, not at request time.
    "http://host:notaport",
    "http://host:99999",
]


@pytest.mark.parametrize("bad", _BAD_BASE_URLS)
def test_sync_client_rejects_base_url_without_http_scheme(bad: str) -> None:
    with pytest.raises(ValueError, match="http:// or https://"):
        SIEClient(bad)


@pytest.mark.parametrize("bad", _BAD_BASE_URLS)
def test_async_client_rejects_base_url_without_http_scheme(bad: str) -> None:
    with pytest.raises(ValueError, match="http:// or https://"):
        SIEAsyncClient(bad)


@pytest.mark.asyncio
@pytest.mark.parametrize("good", ["http://localhost:8080", "https://gateway.example.com", "HTTP://localhost:8080"])
async def test_clients_accept_http_and_https_base_urls(good: str) -> None:
    client = SIEClient(good)
    client.close()
    async_client = SIEAsyncClient(good)
    try:
        assert async_client is not None
    finally:
        # Close the async client so its aiohttp session/destructor cannot emit
        # a ResourceWarning.
        await async_client.close()
