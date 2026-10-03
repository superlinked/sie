"""Async SSE streaming must not be killed by a *total* timeout.

The async client used ``aiohttp.ClientTimeout(total=...)`` on the streaming
POST, which covers the ENTIRE SSE body: any generation streaming longer than
``timeout_s`` (default 30s) died mid-stream with a non-retryable
``SIEConnectionError`` — after the tokens were billed. The read timeout must
be per-read (``sock_read``), so long generations survive while dead
connections still fail within ~``timeout_s``. The sync twin already has
per-read semantics (httpx float timeout is per-phase / per socket read).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from sie_sdk import SIEAsyncClient
from sie_sdk.client.errors import SIEConnectionError


class _FakeRaw:
    """Stand-in for an aiohttp response used as an async context manager."""

    def __init__(self, *, line_bytes: list[bytes]) -> None:
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self._line_bytes = line_bytes
        self.content = self._aiter_bytes()

    async def _aiter_bytes(self):
        for b in self._line_bytes:
            yield b

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _sse_bytes(*chunks: dict[str, Any]) -> list[bytes]:
    out: list[bytes] = []
    for c in chunks:
        out.append(f"data: {json.dumps(c)}\n".encode())
        out.append(b"\n")
    out.append(b"data: [DONE]\n")
    return out


async def test_async_stream_timeout_is_per_read_not_total() -> None:
    """Pin the ClientTimeout shape on the streaming POST: no ``total``."""
    client = SIEAsyncClient("http://localhost:8080", timeout_s=30.0)
    session = MagicMock()
    session.post = MagicMock(return_value=_FakeRaw(line_bytes=_sse_bytes({"text": "hi"})))
    session.close = AsyncMock()
    client._session = session

    out = [c async for c in client.stream_generate("m", "hi", max_new_tokens=8)]

    assert [c["text"] for c in out] == ["hi"]
    timeout = session.post.call_args.kwargs["timeout"]
    assert timeout.total is None, "a total timeout would kill any stream longer than timeout_s"
    assert timeout.sock_read == 30.0
    assert timeout.connect is not None
    assert timeout.connect <= 30.0
    await client.close()


async def _start_sse_server(handler: Any) -> tuple[web.AppRunner, str]:
    app = web.Application()
    app.router.add_post("/v1/generate/m", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    return runner, f"http://{host}:{port}"


async def test_async_stream_survives_longer_than_timeout_s() -> None:
    """End-to-end: chunks each arrive within ``timeout_s`` but the stream as a
    whole runs longer — it must complete, not die mid-stream.
    """

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for i in range(4):
            await asyncio.sleep(0.25)  # inter-chunk gap < timeout_s
            await resp.write(f'data: {{"text": "t{i}"}}\n\n'.encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    runner, base_url = await _start_sse_server(handler)
    try:
        client = SIEAsyncClient(base_url, timeout_s=0.5)  # < total stream duration (~1s)
        out = [c async for c in client.stream_generate("m", "hi", max_new_tokens=8)]
        assert [c["text"] for c in out] == ["t0", "t1", "t2", "t3"]
        await client.close()
    finally:
        await runner.cleanup()


async def test_async_stream_dead_connection_fails_within_read_timeout() -> None:
    """A stalled stream (no chunks arriving) must still fail in ~``timeout_s``,
    not hang forever now that there is no total timeout.
    """

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(b'data: {"text": "t0"}\n\n')
        await asyncio.sleep(30)  # stall: no further reads possible
        return resp

    runner, base_url = await _start_sse_server(handler)
    try:
        client = SIEAsyncClient(base_url, timeout_s=0.4)
        received: list[Any] = []
        start = time.monotonic()
        with pytest.raises(SIEConnectionError):
            async for c in client.stream_generate("m", "hi", max_new_tokens=8):
                received.append(c)
        elapsed = time.monotonic() - start
        assert [c["text"] for c in received] == ["t0"]
        assert elapsed < 5.0, f"dead stream not detected within read timeout: {elapsed:.1f}s"
        await client.close()
    finally:
        await runner.cleanup()
