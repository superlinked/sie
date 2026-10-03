"""Bounded chat transport shared by SIE and OpenAI upstream adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from sie_server.adapters.errors import UpstreamUnavailableError
from sie_server.adapters.remote._http import RemoteUpstreamError, open_stream, sse_data
from sie_server.adapters.remote._openai_chat import ChatStreamParser

_MAX_EVENT_BYTES = 1 << 20
_MAX_STREAM_BYTES = 64 << 20


async def chat_completion(
    client: httpx.AsyncClient,
    request: httpx.Request,
    *,
    upstream: str,
    requested_model: str,
    choices: int,
    max_response_bytes: int,
    error_body_timeout_s: float,
) -> dict[str, Any]:
    """Return one bounded, normalized chat answer with exact upstream usage."""
    parser = ChatStreamParser(requested_model, choices=choices)
    async with open_stream(client, request, upstream=upstream, error_body_timeout_s=error_body_timeout_s) as response:
        if response.headers.get("content-type", "").partition(";")[0].strip().lower() != "application/json":
            raise RemoteUpstreamError("upstream did not return a chat answer")
        raw = bytearray()
        async for chunk in response.aiter_raw():
            if len(raw) + len(chunk) > min(max_response_bytes, _MAX_STREAM_BYTES):
                raise RemoteUpstreamError("upstream chat answer exceeds the size limit")
            raw.extend(chunk)
        return parser.completion(bytes(raw))


async def chat_completion_stream(
    client: httpx.AsyncClient,
    request: httpx.Request,
    *,
    upstream: str,
    requested_model: str,
    choices: int,
    max_response_bytes: int,
    error_body_timeout_s: float,
) -> AsyncIterator[dict[str, Any]]:
    """Yield normalized events; upstream failures after output are final."""
    parser = ChatStreamParser(requested_model, choices=choices)
    yielded = False
    try:
        async with open_stream(
            client, request, upstream=upstream, error_body_timeout_s=error_body_timeout_s
        ) as response:
            if response.headers.get("content-type", "").partition(";")[0].strip().lower() != "text/event-stream":
                raise RemoteUpstreamError("upstream did not stream its chat answer")
            async for data in sse_data(
                response,
                max_event_bytes=min(_MAX_EVENT_BYTES, max_response_bytes),
                max_total_bytes=min(_MAX_STREAM_BYTES, max_response_bytes),
            ):
                event = parser.parse(data)
                if event is None:
                    return
                yielded = True
                yield event
            parser.finish()
    except UpstreamUnavailableError:
        if yielded:
            raise RemoteUpstreamError("the upstream failed during chat generation") from None
        raise
