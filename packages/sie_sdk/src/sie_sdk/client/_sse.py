"""Server-Sent Events parsing for the SIE SDK streaming surfaces.

The parsers consume an iterator of already-decoded text *lines* (newline
stripped) so they work for both transports the SDK uses: the sync client
feeds httpx's ``Response.iter_lines``; the async client feeds an adapter over
aiohttp's byte ``StreamReader``. The only job here is to pull the ``data:``
payload out of each line and honour the ``[DONE]`` terminator.

Scope mirrors the gateway (``packages/sie_gateway/src/handlers/sse.rs``):
the SIE gateway emits one single-line ``data: <json>`` event per chunk plus
the literal ``data: [DONE]`` terminator. Blank separator lines and ``:``
keep-alive comments are skipped; ``event:`` / ``id:`` / ``retry:`` fields and
multi-line ``data:`` continuations are not produced and are not handled here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import SIEConnectionError

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator

_SSE_DONE = "[DONE]"


class SSECompletion:
    """Distinguish transport EOF from a completed generation.

    A gateway ``[DONE]`` marker ends either surface. Without that marker,
    native generation must have emitted ``done: true``; chat must have
    finished every requested choice. Usage-only chat events do not complete
    a stream and do not erase an earlier finish event.
    """

    def __init__(self, *, chat_choices: int | None = None) -> None:
        self.done = False
        self._native_terminal = False
        self._chat_choices = chat_choices
        if chat_choices is not None and (type(chat_choices) is not int or chat_choices <= 0):
            # The gateway still validates the unchanged request. A malformed
            # option must not make an empty successful body look complete.
            self._chat_choices = 1
        self._finished_choices: set[int] = set()

    def observe(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            return
        if self._chat_choices is None:
            if chunk.get("done") is True:
                self._native_terminal = True
            return
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index")
            finish = choice.get("finish_reason")
            if type(index) is int and 0 <= index < self._chat_choices and isinstance(finish, str) and finish:
                self._finished_choices.add(index)

    def require_complete(self) -> None:
        if self.done or self._native_terminal:
            return
        if self._chat_choices is not None and len(self._finished_choices) == self._chat_choices:
            return
        msg = "Connection lost during stream: response ended before generation completed"
        raise SIEConnectionError(msg)


def _extract_data_payload(line: str) -> str | None:
    """Return the ``data:`` payload of one SSE line, or ``None`` to skip.

    Blank lines (event separators) and ``:`` keep-alive comments yield
    ``None``. Per the SSE spec a single leading space after the colon is
    stripped.
    """
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    value = line[len("data:") :]
    value = value.removeprefix(" ")
    return value


def iter_sse_payloads(lines: Iterable[str], *, completion: SSECompletion | None = None) -> Iterator[str]:
    """Yield ``data:`` payloads from an iterable of SSE lines.

    Stops (without yielding) on the ``[DONE]`` sentinel or a clean EOF.
    When supplied, ``completion`` records the sentinel so the caller can
    validate natural EOF after inspecting the semantic terminal chunks.
    The caller owns the underlying stream; closing it cancels the upstream
    request (the gateway forwards the disconnect so the worker stops
    generating).
    """
    for line in lines:
        payload = _extract_data_payload(line)
        if payload is None:
            continue
        if payload == _SSE_DONE:
            if completion is not None:
                completion.done = True
            return
        yield payload


async def aiter_sse_payloads(
    lines: AsyncIterable[str], *, completion: SSECompletion | None = None
) -> AsyncIterator[str]:
    """Async counterpart to :func:`iter_sse_payloads`."""
    async for line in lines:
        payload = _extract_data_payload(line)
        if payload is None:
            continue
        if payload == _SSE_DONE:
            if completion is not None:
                completion.done = True
            return
        yield payload
