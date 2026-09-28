"""Python client -> Rust local-ingest generation protocol tests."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing
from pathlib import Path
from typing import Any

import msgpack
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sie_server import local_ingest_client


def _frame(value: dict[str, Any]) -> bytes:
    payload = msgpack.packb(value, use_bin_type=True)
    return len(payload).to_bytes(4, "little") + payload


async def _request(reader: asyncio.StreamReader) -> dict[str, Any]:
    length = int.from_bytes(await reader.readexactly(4), "little")
    return msgpack.unpackb(await reader.readexactly(length), raw=False)


def _meta() -> dict[str, Any]:
    return {
        "lane": "default|a100-80gb|test/model",
        "request_id": "req-1",
        "endpoint": "generate",
        "model": "test/model",
        "engine": "sglang",
        "admission_pool": "default",
        "bundle_config_hash": "hash",
    }


@pytest.fixture
def unix_socket_dir() -> Iterator[Path]:
    # macOS limits AF_UNIX paths to 104 bytes; pytest's tmp_path can exceed it.
    with tempfile.TemporaryDirectory(prefix="sie-li-", dir="/tmp") as directory:
        yield Path(directory)


def test_generation_body_preserves_gateway_transport_binding() -> None:
    meta = {
        **_meta(),
        "dispatch_context": b"authenticated-context",
        "payload_digest": b"d" * local_ingest_client.PAYLOAD_DIGEST_BYTES,
        "timeout_ms": 123,
    }

    body = local_ingest_client.build_generation_request_body(b"items", b"params", meta)

    assert body["dispatch_context"] == meta["dispatch_context"]
    assert body["payload_digest"] == meta["payload_digest"]
    assert body["timeout_ms"] == 123


@pytest.mark.asyncio
async def test_read_frame_normalizes_truncated_and_invalid_protocol_errors() -> None:
    truncated = asyncio.StreamReader()
    truncated.feed_data(b"\x01\x00")
    truncated.feed_eof()
    with pytest.raises(local_ingest_client.LocalIngestStreamError, match="header is truncated"):
        await local_ingest_client._read_frame(truncated)

    invalid = asyncio.StreamReader()
    invalid.feed_data((1).to_bytes(4, "little") + b"\xc1")
    invalid.feed_eof()
    with pytest.raises(local_ingest_client.LocalIngestStreamError, match="invalid MessagePack"):
        await local_ingest_client._read_frame(invalid)


@pytest.mark.asyncio
async def test_stream_generate_bounds_connect_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    async def never_connect(_socket_path: str) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(local_ingest_client, "LOCAL_INGEST_CONNECT_TIMEOUT_S", 0.01)
    monkeypatch.setattr(local_ingest_client.asyncio, "open_unix_connection", never_connect)

    with pytest.raises(local_ingest_client.LocalIngestStreamError, match="connection timed out"):
        _ = [chunk async for chunk in local_ingest_client.stream_generate("unused.sock", b"items", b"params", _meta())]


@pytest.mark.asyncio
async def test_stream_generate_bounds_response_wait(unix_socket_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    socket_path = unix_socket_dir / "timeout.sock"

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _request(reader)
        await asyncio.sleep(0.1)
        writer.close()
        await writer.wait_closed()

    monkeypatch.setattr(local_ingest_client, "DEFAULT_LOCAL_INGEST_RESPONSE_IDLE_TIMEOUT_S", 0.01)
    server = await asyncio.start_unix_server(handle, path=socket_path)
    async with server:
        with pytest.raises(local_ingest_client.LocalIngestStreamError, match="response timed out"):
            _ = [
                chunk
                async for chunk in local_ingest_client.stream_generate(str(socket_path), b"items", b"params", _meta())
            ]


@pytest.mark.asyncio
async def test_stream_generate_preserves_binding_order_and_terminal(unix_socket_dir: Path) -> None:
    socket_path = unix_socket_dir / "ingest.sock"
    observed: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = await _request(reader)
        observed.set_result(request)
        writer.write(
            _frame({"id": 1, "ok": True, "error": None, "body": {"chunk": b"a", "seq": 0}})
            + _frame({"id": 1, "ok": True, "error": None, "body": {"chunk": b"b", "seq": 1}})
            + _frame(
                {
                    "id": 1,
                    "ok": True,
                    "error": None,
                    "body": {"final": True, "outcome": {"status": "complete", "chunks": 2}},
                }
            )
        )
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handle, path=socket_path)
    async with server:
        chunks = [
            chunk async for chunk in local_ingest_client.stream_generate(str(socket_path), b"items", b"params", _meta())
        ]

    assert chunks == [b"a", b"b"]
    request = await observed
    assert request["op"] == "publish_generate_stream"
    body = request["body"]
    assert body["request_id"] == "req-1"
    assert body["payload_digest"] == local_ingest_client.compute_payload_digest(body)


@pytest.mark.asyncio
async def test_stream_generate_disconnect_cancels_connection_owned_stream(unix_socket_dir: Path) -> None:
    socket_path = unix_socket_dir / "cancel.sock"
    disconnected: asyncio.Future[bytes] = asyncio.get_running_loop().create_future()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _request(reader)
        writer.write(_frame({"id": 1, "ok": True, "error": None, "body": {"chunk": b"first", "seq": 0}}))
        await writer.drain()
        disconnected.set_result(await reader.read())
        writer.close()

    server = await asyncio.start_unix_server(handle, path=socket_path)
    async with server:
        stream = local_ingest_client.stream_generate(str(socket_path), b"items", b"params", _meta())
        assert await anext(stream) == b"first"
        await stream.aclose()
        assert await asyncio.wait_for(disconnected, timeout=1.0) == b""


@pytest.mark.asyncio
async def test_stream_generate_rejects_transport_sequence_gap(unix_socket_dir: Path) -> None:
    socket_path = unix_socket_dir / "gap.sock"

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await _request(reader)
        writer.write(_frame({"id": 1, "ok": True, "error": None, "body": {"chunk": b"late", "seq": 1}}))
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handle, path=socket_path)
    async with server:
        with pytest.raises(local_ingest_client.LocalIngestStreamError, match="not expected 0"):
            _ = [
                chunk
                async for chunk in local_ingest_client.stream_generate(str(socket_path), b"items", b"params", _meta())
            ]


def test_trace_carrier_does_not_change_bound_payload() -> None:
    parent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    body = local_ingest_client.build_generation_request_body(b"items", b"params", _meta())
    traced = local_ingest_client.build_generation_request_body(
        b"items", b"params", {**_meta(), "traceparent": parent, "tracestate": "vendor=value"}
    )
    assert traced["traceparent"] == parent
    assert traced["tracestate"] == "vendor=value"
    assert traced["items"] == body["items"]
    assert traced["payload_digest"] == body["payload_digest"]
    assert local_ingest_client.compute_payload_digest(traced) == local_ingest_client.compute_payload_digest(body)
    invalid = local_ingest_client.build_generation_request_body(
        b"items", b"params", {**_meta(), "traceparent": 3, "tracestate": "x" * 513}
    )
    assert "traceparent" not in invalid
    assert "tracestate" not in invalid


@pytest.mark.parametrize("state", [None, "vendor=value", "invalid state"])
async def test_traced_stream_owns_handoff_lifetime_without_leaking_context(
    monkeypatch: pytest.MonkeyPatch, state: str | None
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(local_ingest_client, "_TRACER", provider.get_tracer("test"))
    monkeypatch.setenv("SIE_TRACING_ENABLED", "true")
    parent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    captured = []
    closed = []

    async def transport(_socket: str, items: bytes, params: bytes, meta: dict[str, Any]) -> AsyncIterator[bytes]:
        captured.append((items, params, meta, trace.get_current_span().get_span_context()))
        try:
            yield b"chunk"
            await asyncio.Future()
        finally:
            closed.append(trace.get_current_span().get_span_context())

    monkeypatch.setattr(local_ingest_client, "_stream_generate", transport)
    iterator = local_ingest_client.stream_generate(
        "private/socket", b"secret items", b"secret params", {**_meta(), "traceparent": parent, "tracestate": state}
    )
    async with aclosing(iterator):
        assert await anext(iterator) == b"chunk"
        assert not trace.get_current_span().get_span_context().is_valid
        assert not exporter.get_finished_spans()
        await asyncio.create_task(iterator.aclose())
    (span,) = exporter.get_finished_spans()
    assert closed == [span.context]
    assert captured[0][3] == span.context
    assert span.name == "worker.local_ingest"
    assert span.parent.span_id == int(parent.split("-")[2], 16)
    assert captured[0][0:2] == (b"secret items", b"secret params")
    assert captured[0][2]["traceparent"].split("-")[2] == f"{span.context.span_id:016x}"
    assert captured[0][2].get("tracestate") == (state if state == "vendor=value" else None)
    assert not span.attributes
    assert not span.events
    assert not span.status.description
    provider.shutdown()


async def test_optional_carrier_does_not_overflow_valid_frame(
    unix_socket_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = local_ingest_client.build_generation_request_body(b"items", b"params", _meta())
    limit = len(msgpack.packb({"id": 1, "op": "publish_generate_stream", "body": body}, use_bin_type=True))
    monkeypatch.setattr(local_ingest_client, "MAX_LOCAL_INGEST_FRAME_BYTES", limit)
    socket_path = unix_socket_dir / "bounded.sock"
    observed = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        observed.append(await _request(reader))
        writer.write(_frame({"id": 1, "ok": True, "body": {"final": True, "outcome": {"chunks": 0}}}))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(handle, path=socket_path)
    async with server:
        chunks = [
            chunk
            async for chunk in local_ingest_client.stream_generate(
                str(socket_path),
                b"items",
                b"params",
                {**_meta(), "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"},
            )
        ]
    assert chunks == []
    assert observed[0]["body"] == body
