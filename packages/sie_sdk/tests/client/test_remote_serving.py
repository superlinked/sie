"""Forbidding remote serving, and reading which side served, through the real clients.

Both clients talk to a local HTTP server that records each request's headers and
answers with the serving disclosure headers, so the option and the parsing run
through the production transport. The header contract is shared with the
TypeScript SDK through ``packages/wire-fixtures/serving_disclosure.json``.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import msgpack
import numpy as np
import pytest
from sie_sdk import SIEAsyncClient, SIEClient, SIEError
from sie_sdk.client._shared import handle_error, parse_request_metadata

_OPERATIONS = ("encode", "generate", "chat_completions")
_SERVED_REMOTELY = {
    "X-SIE-Served-By": "remote",
    "X-SIE-Upstream": "team-sie",
    "X-SIE-Fallback-Reason": "model_loading",
}


class _RecordingServer:
    """Answers every POST with one success body plus ``reply_headers``, recording request headers."""

    def __init__(self) -> None:
        self.reply_headers: dict[str, str] = {}
        self.request_headers: list[dict[str, str]] = []
        server = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                server.request_headers.append({name.lower(): value for name, value in self.headers.items()})
                content_type, body = _success_body(self.path)
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                for name, value in server.reply_headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

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

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _success_body(path: str) -> tuple[str, bytes]:
    if path.startswith("/v1/encode/"):
        items = [{"id": "a", "dense": {"dims": 4, "values": np.zeros(4, dtype=np.float32)}}]
        return "application/msgpack", msgpack.packb({"model": "m", "items": items}, use_bin_type=True)
    if path.startswith("/v1/generate/"):
        return "application/json", json.dumps({"model": "m", "text": "ok"}).encode()
    return "application/json", json.dumps(
        {"id": "c", "object": "chat.completion", "model": "m", "choices": []}
    ).encode()


@pytest.fixture
def server() -> Iterator[_RecordingServer]:
    recording = _RecordingServer()
    try:
        yield recording
    finally:
        recording.close()


def _request_metadata(result: Any) -> dict[str, Any]:
    first = result[0] if isinstance(result, list) else result
    return first["request"]


def _call_sync(client: SIEClient, operation: str) -> Any:
    if operation == "encode":
        return client.encode("m", {"id": "a", "text": "hi"})
    if operation == "generate":
        return client.generate("m", "hi", max_new_tokens=4)
    return client.chat_completions("m", [{"role": "user", "content": "hi"}])


async def _call_async(client: SIEAsyncClient, operation: str) -> Any:
    if operation == "encode":
        return await client.encode("m", {"id": "a", "text": "hi"})
    if operation == "generate":
        return await client.generate("m", "hi", max_new_tokens=4)
    return await client.chat_completions("m", [{"role": "user", "content": "hi"}])


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_a_forbidding_sync_client_sends_the_header_and_reads_the_serving_side(
    server: _RecordingServer, operation: str
) -> None:
    server.reply_headers = {"X-SIE-Served-By": "local"}
    with SIEClient(server.url, remote="forbid") as client:
        result = _call_sync(client, operation)

    assert server.request_headers[0]["x-sie-remote"] == "forbid"
    assert _request_metadata(result)["served_by"] == "local"


@pytest.mark.parametrize("operation", _OPERATIONS)
async def test_a_forbidding_async_client_sends_the_header_and_reads_the_serving_side(
    server: _RecordingServer, operation: str
) -> None:
    server.reply_headers = {"X-SIE-Served-By": "local"}
    async with SIEAsyncClient(server.url, remote="forbid") as client:
        result = await _call_async(client, operation)

    assert server.request_headers[0]["x-sie-remote"] == "forbid"
    assert _request_metadata(result)["served_by"] == "local"


async def test_a_default_client_leaves_the_choice_to_the_server(server: _RecordingServer) -> None:
    with SIEClient(server.url) as client:
        client.encode("m", {"id": "a", "text": "hi"})
    async with SIEAsyncClient(server.url) as async_client:
        await async_client.encode("m", {"id": "a", "text": "hi"})

    assert [headers.get("x-sie-remote") for headers in server.request_headers] == [None, None]


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_a_remotely_served_result_names_the_upstream_and_the_reason(server: _RecordingServer, operation: str) -> None:
    server.reply_headers = dict(_SERVED_REMOTELY)
    with SIEClient(server.url) as client:
        result = _call_sync(client, operation)

    request = _request_metadata(result)
    assert request["served_by"] == "remote"
    assert request["upstream"] == "team-sie"
    assert request["fallback_reason"] == "model_loading"


@pytest.mark.parametrize("client_class", [SIEClient, SIEAsyncClient])
def test_only_forbid_is_accepted(client_class: type[SIEClient | SIEAsyncClient]) -> None:
    with pytest.raises(ValueError, match="remote must be 'forbid' or None"):
        client_class("http://127.0.0.1:9", remote="allow")  # type: ignore[arg-type]


def test_a_local_refusal_after_a_failed_remote_attempt_carries_both_outcomes() -> None:
    response = httpx.Response(
        503,
        headers={
            "Retry-After": "5",
            "X-SIE-Fallback-Reason": "model_loading",
            "X-SIE-Fallback-Error": "INFERENCE_ERROR",
        },
        json={"detail": {"code": "MODEL_LOADING", "message": "Model 'm' is loading, please retry"}},
    )

    with pytest.raises(SIEError) as raised:
        handle_error(response)

    assert raised.value.request == {"fallback_reason": "model_loading", "fallback_error": "INFERENCE_ERROR"}


@pytest.mark.parametrize(
    ("header", "value"),
    [
        ("X-SIE-Served-By", "elsewhere"),
        ("X-SIE-Served-By", "Remote"),
        ("X-SIE-Upstream", "Team_SIE"),
        ("X-SIE-Upstream", "-team"),
        ("X-SIE-Upstream", "a" * 64),
        ("X-SIE-Fallback-Reason", "sometimes"),
        ("X-SIE-Fallback-Error", "not a code"),
        ("X-SIE-Fallback-Error", "1CODE"),
    ],
)
def test_a_disclosure_value_outside_its_contract_is_dropped(header: str, value: str) -> None:
    assert parse_request_metadata({header: value}) is None
