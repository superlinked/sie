"""Bounded Python client for the Rust sidecar's local-ingest protocol."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import AsyncGenerator
from contextlib import nullcontext, suppress
from typing import Any

import msgpack
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

_TRACER = trace.get_tracer(__name__)
_PROPAGATOR = TraceContextTextMapPropagator()

MAX_LOCAL_INGEST_FRAME_BYTES = 64 * 1024 * 1024
PAYLOAD_DIGEST_BYTES = 32
PAYLOAD_DIGEST_DOMAIN = b"sie-local-ingest-v1\0"
INTERNAL_DISPATCH_CONTEXT = msgpack.packb({"source": "internal"}, use_bin_type=True)
LOCAL_INGEST_CONNECT_TIMEOUT_S = 5.0
DEFAULT_LOCAL_INGEST_RESPONSE_IDLE_TIMEOUT_S = 900.0
LOCAL_INGEST_RESPONSE_CLEANUP_GRACE_S = 11.0


class LocalIngestStreamError(RuntimeError):
    """The colocated Rust sidecar rejected or broke a generation stream."""


def _update_digest_field(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)


def compute_payload_digest(body: dict[str, Any]) -> bytes:
    """Compute the v0.2 trusted-peer consistency digest for one request."""
    hasher = hashlib.sha256()
    hasher.update(PAYLOAD_DIGEST_DOMAIN)
    for value in (
        body["dispatch_context"],
        body["lane"].encode(),
        body["endpoint"].encode(),
        body["model"].encode(),
        body["engine"].encode(),
        body["admission_pool"].encode(),
        body["bundle_config_hash"].encode(),
        body["request_id"].encode(),
        body["params"],
        body["items"],
    ):
        _update_digest_field(hasher, value)
    hasher.update(body["timeout_ms"].to_bytes(8, "big", signed=True))
    return hasher.digest()


def build_generation_request_body(items: bytes, params: bytes, meta: dict[str, Any]) -> dict[str, Any]:
    """Build one generation body while preserving gateway-provided bindings."""
    request_id = meta.get("request_id")
    lane = meta.get("lane")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError("local-ingest generation metadata missing request_id")
    if not isinstance(lane, str) or not lane:
        raise ValueError("local-ingest generation metadata missing lane")

    dispatch_context = meta.get("dispatch_context")
    payload_digest = meta.get("payload_digest")
    timeout_ms = meta.get("timeout_ms", 0)
    bound_transport = dispatch_context is not None or payload_digest is not None
    if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms < 0:
        raise ValueError("local-ingest generation timeout_ms must be a non-negative integer")
    if bound_transport:
        if not isinstance(dispatch_context, bytes) or not dispatch_context:
            raise ValueError("local-ingest generation dispatch_context must be bytes")
        if not isinstance(payload_digest, bytes) or len(payload_digest) != PAYLOAD_DIGEST_BYTES:
            raise ValueError(f"local-ingest generation payload_digest must be {PAYLOAD_DIGEST_BYTES} bytes")
    else:
        dispatch_context = INTERNAL_DISPATCH_CONTEXT
        payload_digest = b""

    body = {
        "lane": lane,
        "endpoint": str(meta.get("endpoint") or "generate"),
        "model": str(meta.get("model") or ""),
        "engine": str(meta.get("engine") or ""),
        "admission_pool": str(meta.get("admission_pool") or ""),
        "bundle_config_hash": str(meta.get("bundle_config_hash") or ""),
        "request_id": request_id,
        "params": params,
        "items": items,
        "dispatch_context": dispatch_context,
        "payload_digest": payload_digest,
        "timeout_ms": timeout_ms,
    }
    for key, limit in (("traceparent", 256), ("tracestate", 512)):
        value = meta.get(key)
        if isinstance(value, str) and len(value) <= limit:
            body[key] = value
    if not bound_transport:
        body["payload_digest"] = compute_payload_digest(body)
    return body


async def _read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError as exc:
        raise LocalIngestStreamError("local-ingest generation response header is truncated") from exc
    length = int.from_bytes(header, "little")
    if length > MAX_LOCAL_INGEST_FRAME_BYTES:
        raise LocalIngestStreamError(
            f"local-ingest generation response frame is {length} bytes; maximum is {MAX_LOCAL_INGEST_FRAME_BYTES}"
        )
    try:
        payload = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise LocalIngestStreamError("local-ingest generation response payload is truncated") from exc
    try:
        response = msgpack.unpackb(payload, raw=False)
    except (ValueError, msgpack.ExtraData) as exc:
        raise LocalIngestStreamError(f"local-ingest generation response is invalid MessagePack: {exc}") from exc
    if not isinstance(response, dict):
        raise LocalIngestStreamError("local-ingest generation response is not a map")
    return response


def stream_generate(socket_path: str, items: bytes, params: bytes, meta: dict[str, Any]) -> AsyncGenerator[bytes, None]:
    if os.environ.get("SIE_TRACING_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return _stream_generate(socket_path, items, params, meta)
    return _traced_generate(socket_path, items, params, meta)


async def _traced_generate(
    socket_path: str, items: bytes, params: bytes, meta: dict[str, Any]
) -> AsyncGenerator[bytes, None]:
    carrier = {
        key: value
        for key, limit in (("traceparent", 256), ("tracestate", 512))
        if isinstance(value := meta.get(key), str) and len(value) <= limit
    }
    span = None
    with suppress(Exception):
        parent = _PROPAGATOR.extract(carrier, context=Context())
        span = _TRACER.start_span(
            "worker.local_ingest",
            context=parent,
            kind=trace.SpanKind.CLIENT,
            record_exception=False,
            set_status_on_exception=False,
        )
        carrier = {}
        _PROPAGATOR.inject(carrier, context=trace.set_span_in_context(span, parent))
        meta = {**meta, **carrier}
    iterator = _stream_generate(socket_path, items, params, meta)
    try:
        while True:
            try:
                with (
                    trace.use_span(span, record_exception=False, set_status_on_exception=False)
                    if span is not None
                    else nullcontext()
                ):
                    chunk = await anext(iterator)
            except StopAsyncIteration:
                break
            yield chunk
    except BaseException as error:
        if span is not None and not isinstance(error, (asyncio.CancelledError, GeneratorExit)):
            with suppress(Exception):
                span.set_status(trace.StatusCode.ERROR)
        raise
    finally:
        try:
            with (
                trace.use_span(span, record_exception=False, set_status_on_exception=False)
                if span is not None
                else nullcontext()
            ):
                await iterator.aclose()
        finally:
            if span is not None:
                with suppress(Exception):
                    span.end()


async def _stream_generate(
    socket_path: str,
    items: bytes,
    params: bytes,
    meta: dict[str, Any],
) -> AsyncGenerator[bytes, None]:
    """Map one caller operation onto sidecar protocol v0.2.

    Pulling one response before requesting the next naturally propagates
    caller backpressure through the sidecar. Closing
    the generator closes this connection; local ingest treats EOF as
    cancellation for the connection-owned request.
    """
    body = build_generation_request_body(items, params, meta)
    operation_id = 1
    payload = msgpack.packb(
        {"id": operation_id, "op": "publish_generate_stream", "body": body},
        use_bin_type=True,
    )
    if len(payload) > MAX_LOCAL_INGEST_FRAME_BYTES and ("traceparent" in body or "tracestate" in body):
        # Optional telemetry must not make a previously valid frame too large.
        body.pop("traceparent", None)
        body.pop("tracestate", None)
        payload = msgpack.packb({"id": operation_id, "op": "publish_generate_stream", "body": body}, use_bin_type=True)
    if len(payload) > MAX_LOCAL_INGEST_FRAME_BYTES:
        raise ValueError(
            f"local-ingest generation frame is {len(payload)} bytes; maximum is {MAX_LOCAL_INGEST_FRAME_BYTES}"
        )

    timeout_ms = body["timeout_ms"]
    response_timeout_s = (
        timeout_ms / 1_000 + LOCAL_INGEST_RESPONSE_CLEANUP_GRACE_S
        if timeout_ms > 0
        else DEFAULT_LOCAL_INGEST_RESPONSE_IDLE_TIMEOUT_S
    )
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(socket_path),
            timeout=LOCAL_INGEST_CONNECT_TIMEOUT_S,
        )
    except TimeoutError as exc:
        raise LocalIngestStreamError("local-ingest generation connection timed out") from exc
    try:
        writer.write(len(payload).to_bytes(4, "little") + payload)
        await writer.drain()
        expected_seq = 0
        while True:
            try:
                response = await asyncio.wait_for(_read_frame(reader), timeout=response_timeout_s)
            except TimeoutError as exc:
                raise LocalIngestStreamError("local-ingest generation response timed out") from exc
            if response.get("id") != operation_id:
                raise LocalIngestStreamError("local-ingest generation response id mismatch")
            if response.get("ok") is not True:
                raise LocalIngestStreamError(str(response.get("error") or "sidecar generation failed"))
            response_body = response.get("body")
            if not isinstance(response_body, dict):
                raise LocalIngestStreamError("local-ingest generation response body is not a map")
            if response_body.get("final") is True:
                outcome = response_body.get("outcome")
                if not isinstance(outcome, dict) or outcome.get("chunks") != expected_seq:
                    raise LocalIngestStreamError("local-ingest generation terminal chunk count mismatch")
                return
            chunk = response_body.get("chunk")
            seq = response_body.get("seq")
            if not isinstance(chunk, bytes):
                raise LocalIngestStreamError("local-ingest generation chunk is not bytes")
            if isinstance(seq, bool) or not isinstance(seq, int) or seq != expected_seq:
                raise LocalIngestStreamError(f"local-ingest generation sequence {seq!r} is not expected {expected_seq}")
            expected_seq += 1
            yield chunk
    finally:
        writer.close()
        await writer.wait_closed()
