"""Retry classification, timeouts and partial batches through the real clients.

Both clients talk to a local HTTP server that replays scripted responses, so
every decision below runs through the production transport, classification and
retry code. The response table is shared with the TypeScript SDK through
``packages/wire-fixtures/retry_classification.json``.
"""

from __future__ import annotations

import asyncio
import functools
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import aiohttp
import httpx
import msgpack
import numpy as np
import pytest
from sie_sdk import IncompleteBatchError, SIEAsyncClient, SIEClient, SIEConnectionError, SIEError
from sie_sdk.client import async_ as async_module
from sie_sdk.client import sync as sync_module
from sie_sdk.client._shared import (
    DEFAULT_CONNECT_TIMEOUT_S,
    DEFAULT_READ_TIMEOUT_S,
    get_retry_after,
    resolve_timeouts,
)

_FIXTURE: dict[str, Any] = json.loads(
    (Path(__file__).parents[3] / "wire-fixtures" / "retry_classification.json").read_text()
)
_RESPONSE_CASES: list[dict[str, Any]] = _FIXTURE["responses"]
_RETRY_AFTER_CASES: list[dict[str, Any]] = _FIXTURE["retry_after"]
_OPERATIONS = ("encode", "generate", "chat_completions")
_MSGPACK_HEADERS = {"Content-Type": "application/msgpack"}
_JSON_HEADERS = {"Content-Type": "application/json"}


@dataclass
class _Reply:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    delay_s: float = 0.0


class _StubServer:
    """Threaded HTTP server that replays scripted replies and records request paths."""

    def __init__(self) -> None:
        self.replies: list[_Reply] = []
        self.paths: list[str] = []
        self._lock = threading.Lock()
        self._release = threading.Event()
        stub = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                reply = stub._next(self.path)
                if reply.delay_s:
                    stub._release.wait(reply.delay_s)
                try:
                    self.send_response(reply.status)
                    for name, value in reply.headers.items():
                        self.send_header(name, value)
                    self.send_header("Content-Length", str(len(reply.body)))
                    self.end_headers()
                    self.wfile.write(reply.body)
                except OSError:
                    self.close_connection = True

            def log_message(self, format: str, *args: Any) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, args=(0.01,), daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}"

    def _next(self, path: str) -> _Reply:
        with self._lock:
            self.paths.append(path)
            if len(self.replies) > 1:
                return self.replies.pop(0)
            return self.replies[0]

    def close(self) -> None:
        self._release.set()
        self._server.shutdown()
        self._server.server_close()


class _NoSleepTime:
    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)

    @staticmethod
    def sleep(_seconds: float) -> None:
        return None


class _NoSleepAsyncio:
    def __getattr__(self, name: str) -> Any:
        return getattr(asyncio, name)

    @staticmethod
    async def sleep(_delay: float, result: Any = None) -> Any:
        return result


@pytest.fixture
def stub() -> Iterator[_StubServer]:
    server = _StubServer()
    try:
        yield server
    finally:
        server.close()


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sync_module, "time", _NoSleepTime())
    monkeypatch.setattr(async_module, "asyncio", _NoSleepAsyncio())


def _encode_ok(ids: list[str] | None = None) -> _Reply:
    items = [{"id": item_id, "dense": {"dims": 4, "values": np.zeros(4, dtype=np.float32)}} for item_id in ids or ["a"]]
    return _Reply(200, msgpack.packb({"model": "m", "items": items}, use_bin_type=True), dict(_MSGPACK_HEADERS))


def _json_ok(operation: str) -> _Reply:
    payloads = {
        "generate": {"model": "m", "text": "ok"},
        "chat_completions": {"id": "chat-1", "object": "chat.completion", "model": "m", "choices": []},
        "responses": {"id": "resp-1", "object": "response", "model": "m", "output": []},
    }
    return _Reply(200, json.dumps(payloads[operation]).encode(), dict(_JSON_HEADERS))


def _success(operation: str) -> _Reply:
    return _encode_ok() if operation == "encode" else _json_ok(operation)


def _fixture_reply(case: dict[str, Any]) -> _Reply:
    return _Reply(case["status"], json.dumps(case["body"]).encode(), {**_JSON_HEADERS, **case["headers"]})


def _operation_class(operation: str) -> str:
    return "idempotent" if operation == "encode" else "generation"


def _call_sync(client: SIEClient, operation: str, **kwargs: Any) -> Any:
    if operation == "encode":
        return client.encode("m", {"id": "a", "text": "hi"}, **kwargs)
    if operation == "generate":
        return client.generate("m", "hi", max_new_tokens=4, **kwargs)
    if operation == "responses":
        return client.responses("m", "hi", **kwargs)
    return client.chat_completions("m", [{"role": "user", "content": "hi"}], **kwargs)


async def _call_async(client: SIEAsyncClient, operation: str, **kwargs: Any) -> Any:
    if operation == "encode":
        return await client.encode("m", {"id": "a", "text": "hi"}, **kwargs)
    if operation == "generate":
        return await client.generate("m", "hi", max_new_tokens=4, **kwargs)
    if operation == "responses":
        return await client.responses("m", "hi", **kwargs)
    return await client.chat_completions("m", [{"role": "user", "content": "hi"}], **kwargs)


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("case", _RESPONSE_CASES, ids=[case["name"] for case in _RESPONSE_CASES])
def test_sync_client_follows_shared_retry_table(stub: _StubServer, case: dict[str, Any], operation: str) -> None:
    stub.replies = [_fixture_reply(case), _success(operation)]
    client = SIEClient(stub.url)
    try:
        if case[_operation_class(operation)] == "retry":
            _call_sync(client, operation)
            assert len(stub.paths) == 2
            assert client.last_retry_count == 1
        else:
            with pytest.raises(SIEError):
                _call_sync(client, operation)
            assert len(stub.paths) == 1
            assert client.last_retry_count == 0
    finally:
        client.close()


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("case", _RESPONSE_CASES, ids=[case["name"] for case in _RESPONSE_CASES])
async def test_async_client_follows_shared_retry_table(stub: _StubServer, case: dict[str, Any], operation: str) -> None:
    stub.replies = [_fixture_reply(case), _success(operation)]
    async with SIEAsyncClient(stub.url) as client:
        if case[_operation_class(operation)] == "retry":
            await _call_async(client, operation)
            assert len(stub.paths) == 2
            assert client.last_retry_count == 1
        else:
            with pytest.raises(SIEError):
                await _call_async(client, operation)
            assert len(stub.paths) == 1
            assert client.last_retry_count == 0


@pytest.mark.parametrize("case", _RETRY_AFTER_CASES, ids=[repr(case["header"]) for case in _RETRY_AFTER_CASES])
def test_retry_after_parsing_matches_shared_table(case: dict[str, Any]) -> None:
    parsed = get_retry_after(httpx.Response(503, headers={"Retry-After": case["header"]}))
    if case["seconds"] == "future":
        assert parsed is not None
        assert parsed > 0
    else:
        assert parsed == case["seconds"]


def test_read_timeout_is_not_resent_by_sync_client(stub: _StubServer) -> None:
    stub.replies = [_Reply(**{**_encode_ok().__dict__, "delay_s": 2.0})]
    client = SIEClient(stub.url, read_timeout_s=0.2)
    try:
        with pytest.raises(SIEConnectionError, match="Not retried"):
            client.encode("m", {"text": "hi"}, provision_timeout_s=10.0)
    finally:
        client.close()
    assert stub.paths == ["/v1/encode/m"]


async def test_read_timeout_is_not_resent_by_async_client(stub: _StubServer) -> None:
    stub.replies = [_Reply(**{**_encode_ok().__dict__, "delay_s": 2.0})]
    async with SIEAsyncClient(stub.url, read_timeout_s=0.2) as client:
        with pytest.raises(SIEConnectionError, match="Not retried"):
            await client.encode("m", {"text": "hi"}, provision_timeout_s=10.0)
    assert stub.paths == ["/v1/encode/m"]


@pytest.mark.parametrize("operation", ["generate", "chat_completions", "responses"])
def test_sync_generation_read_timeout_is_per_call(stub: _StubServer, operation: str) -> None:
    slow = _Reply(**{**_json_ok(operation).__dict__, "delay_s": 0.6})
    stub.replies = [slow, slow]
    client = SIEClient(stub.url, read_timeout_s=0.2)
    try:
        assert _call_sync(client, operation, read_timeout_s=5.0)["model"] == "m"
        with pytest.raises(SIEConnectionError):
            _call_sync(client, operation)
    finally:
        client.close()
    assert len(stub.paths) == 2


@pytest.mark.parametrize("operation", ["generate", "chat_completions", "responses"])
async def test_async_generation_read_timeout_is_per_call(stub: _StubServer, operation: str) -> None:
    slow = _Reply(**{**_json_ok(operation).__dict__, "delay_s": 0.6})
    stub.replies = [slow, slow]
    async with SIEAsyncClient(stub.url, read_timeout_s=0.2) as client:
        assert (await _call_async(client, operation, read_timeout_s=5.0))["model"] == "m"
        with pytest.raises(SIEConnectionError):
            await _call_async(client, operation)
    assert len(stub.paths) == 2


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize(
    ("error", "attempts"),
    [(httpx.ConnectTimeout, 2), (httpx.PoolTimeout, 2), (httpx.ReadTimeout, 1)],
    ids=["connect_timeout", "pool_timeout", "read_timeout"],
)
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_sync_transport_timeouts_retry_only_before_send(
    monkeypatch: pytest.MonkeyPatch, operation: str, error: type[httpx.TimeoutException], attempts: int
) -> None:
    seen: list[str] = []
    success = _success(operation)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if len(seen) == 1:
            raise error("timed out", request=request)
        return httpx.Response(success.status, content=success.body, headers=success.headers)

    monkeypatch.setattr(
        sync_module.httpx, "Client", functools.partial(httpx.Client, transport=httpx.MockTransport(handler))
    )
    client = SIEClient("http://sie.test")
    try:
        if attempts == 2:
            _call_sync(client, operation)
            assert client.last_retry_count == 1
        else:
            with pytest.raises(SIEConnectionError):
                _call_sync(client, operation)
    finally:
        client.close()
    assert len(seen) == attempts


class _FirstConnectTimesOut(aiohttp.TCPConnector):
    attempts = 0

    async def connect(self, req: Any, traces: Any, timeout: Any) -> Any:  # noqa: ASYNC109
        type(self).attempts += 1
        if type(self).attempts == 1:
            raise TimeoutError
        return await super().connect(req, traces, timeout)


@pytest.mark.usefixtures("no_sleep")
@pytest.mark.parametrize("operation", _OPERATIONS)
async def test_async_connect_timeout_is_retried(
    monkeypatch: pytest.MonkeyPatch, stub: _StubServer, operation: str
) -> None:
    monkeypatch.setattr(_FirstConnectTimesOut, "attempts", 0)
    monkeypatch.setattr(async_module.aiohttp, "TCPConnector", _FirstConnectTimesOut)
    stub.replies = [_success(operation)]
    async with SIEAsyncClient(stub.url) as client:
        await _call_async(client, operation)
        assert client.last_retry_count == 1
    assert _FirstConnectTimesOut.attempts == 2
    assert len(stub.paths) == 1


def test_timeout_resolution_keeps_timeout_s_meaning() -> None:
    assert resolve_timeouts(None, None, None) == (DEFAULT_CONNECT_TIMEOUT_S, DEFAULT_READ_TIMEOUT_S)
    assert resolve_timeouts(30.0, None, None) == (30.0, 30.0)
    assert resolve_timeouts(30.0, 5.0, None) == (5.0, 30.0)
    assert resolve_timeouts(None, None, 600.0) == (DEFAULT_CONNECT_TIMEOUT_S, 600.0)
    assert DEFAULT_READ_TIMEOUT_S > 120.0


def test_sync_incomplete_batch_keeps_returned_results(stub: _StubServer) -> None:
    stub.replies = [_encode_ok(["b"])]
    client = SIEClient(stub.url)
    try:
        with pytest.raises(IncompleteBatchError) as excinfo:
            client.encode("m", [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}])
    finally:
        client.close()
    error = excinfo.value
    assert error.missing_ids == ["a"]
    assert [result["id"] for result in error.results] == ["b"]
    assert error.results[0]["dense"].shape == (4,)
    assert error.results[0]["model"] == "m"


async def test_async_incomplete_batch_keeps_returned_results(stub: _StubServer) -> None:
    extract_body = {
        "model": "m",
        "items": [{"id": "b", "entities": [], "error": {"code": "INPUT_TOO_LONG", "message": "too long"}}],
    }
    stub.replies = [_Reply(200, msgpack.packb(extract_body, use_bin_type=True), dict(_MSGPACK_HEADERS))]
    async with SIEAsyncClient(stub.url) as client:
        with pytest.raises(IncompleteBatchError) as excinfo:
            await client.extract("m", [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}], labels=["person"])
    error = excinfo.value
    assert error.missing_ids == ["a"]
    assert error.results == [
        {
            "entities": [],
            "relations": [],
            "classifications": [],
            "objects": [],
            "id": "b",
            "error": {"code": "INPUT_TOO_LONG", "message": "too long"},
        }
    ]


@pytest.mark.usefixtures("no_sleep")
async def test_async_retry_count_is_scoped_to_the_calling_task(stub: _StubServer) -> None:
    loading = _fixture_reply(next(case for case in _RESPONSE_CASES if case["name"] == "model_loading"))
    async with SIEAsyncClient(stub.url) as client:
        stub.replies = [loading, _encode_ok()]
        await client.encode("m", {"id": "a", "text": "hi"})
        assert client.last_retry_count == 1

        stub.replies = [_encode_ok()]
        await client.encode("m", {"id": "a", "text": "hi"})
        assert client.last_retry_count == 0

        async def retried_in_task() -> int:
            await client.encode("m", {"id": "a", "text": "hi"})
            return client.last_retry_count

        stub.replies = [loading, _encode_ok()]
        assert await asyncio.create_task(retried_in_task()) == 1
        assert client.last_retry_count == 0


def _sse_reply() -> _Reply:
    chunk = {"request_id": "r", "seq": 0, "text_delta": "ok", "done": True}
    return _Reply(200, f"data: {json.dumps(chunk)}\n\n".encode(), {"Content-Type": "text/event-stream"})


def _backpressure_reply() -> _Reply:
    return _fixture_reply(
        next(case for case in _RESPONSE_CASES if case["name"] == "openai_transport_failure_backpressure")
    )


@pytest.mark.usefixtures("no_sleep")
def test_sync_stream_retries_queue_backpressure_before_opening(stub: _StubServer) -> None:
    stub.replies = [_backpressure_reply(), _sse_reply()]
    client = SIEClient(stub.url)
    try:
        chunks = list(client.stream_generate("m", "hi", max_new_tokens=4))
        assert client.last_retry_count == 1
    finally:
        client.close()
    assert [chunk["text_delta"] for chunk in chunks] == ["ok"]
    assert len(stub.paths) == 2


@pytest.mark.usefixtures("no_sleep")
async def test_async_stream_retries_queue_backpressure_before_opening(stub: _StubServer) -> None:
    stub.replies = [_backpressure_reply(), _sse_reply()]
    async with SIEAsyncClient(stub.url) as client:
        chunks = [chunk async for chunk in client.stream_generate("m", "hi", max_new_tokens=4)]
        assert client.last_retry_count == 1
    assert [chunk["text_delta"] for chunk in chunks] == ["ok"]
    assert len(stub.paths) == 2
