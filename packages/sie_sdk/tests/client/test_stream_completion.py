"""Generation EOF must not silently turn a partial response into success."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import httpx
import pytest
from sie_sdk import SIEAsyncClient, SIEClient
from sie_sdk.client.errors import ServerError, SIEConnectionError

_BASE_URL = "http://synthetic.invalid"


def _frame(payload: dict[str, Any] | str) -> bytes:
    value = payload if isinstance(payload, str) else json.dumps(payload)
    return f"data: {value}\n\n".encode()


def _delta(surface: str, *, index: int = 0) -> dict[str, Any]:
    if surface == "native":
        return {"request_id": "req-test", "seq": 0, "text_delta": "partial", "done": False}
    return {
        "id": "chat-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [{"index": index, "delta": {"content": "partial"}, "finish_reason": None}],
    }


def _terminal(surface: str, *, index: int = 0) -> dict[str, Any]:
    if surface == "native":
        return {
            "request_id": "req-test",
            "seq": 1,
            "text_delta": "",
            "done": True,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    chunk = _delta(surface, index=index)
    chunk["choices"][0].update(delta={}, finish_reason="stop")
    return chunk


class _SyncBody(httpx.SyncByteStream):
    def __init__(self, frames: list[bytes], *, fail: bool = False) -> None:
        self.frames = frames
        self.fail = fail
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        yield from self.frames
        if self.fail:
            raise httpx.ReadError("controlled read failure")

    def close(self) -> None:
        self.closed = True


class _AsyncBody:
    def __init__(self, frames: list[bytes], *, fail: bool = False, block: bool = False) -> None:
        self.frames = frames
        self.fail = fail
        self.block = block
        self.closed = False
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self.waiting = asyncio.Event()
        self.content = self._bytes()

    async def _bytes(self):
        for frame in self.frames:
            yield frame
        if self.fail:
            raise aiohttp.ServerDisconnectedError("controlled read failure")
        if self.block:
            self.waiting.set()
            await asyncio.Event().wait()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc: object) -> bool:
        self.closed = True
        return False


class _Harness:
    def __init__(self, mode: str, frames: list[bytes], *, fail: bool = False, block: bool = False) -> None:
        self.mode = mode
        self.calls = 0
        if mode == "sync":
            self.body = _SyncBody(frames, fail=fail)
            self.client = SIEClient(
                _BASE_URL,
                http_client=httpx.Client(base_url=_BASE_URL, transport=httpx.MockTransport(self._dispatch)),
            )
        else:
            self.body = _AsyncBody(frames, fail=fail, block=block)
            self.client = SIEAsyncClient(_BASE_URL)
            session = MagicMock()
            session.close = AsyncMock()
            session.post.side_effect = self._dispatch
            self.client._session = session

    def _dispatch(self, *args, **kwargs):
        self.calls += 1
        assert self.calls == 1, "an interrupted generation must not be replayed"
        if self.mode == "sync":
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=self.body)
        return self.body

    def stream(self, surface: str, **kwargs):
        if surface == "native":
            return self.client.stream_generate("m", "test", max_new_tokens=4, **kwargs)
        return self.client.stream_chat_completions("m", [{"role": "user", "content": "test"}], **kwargs)

    async def collect(self, surface: str, output: list[Any], **kwargs) -> None:
        if self.mode == "sync":
            output.extend(self.stream(surface, **kwargs))
        else:
            async for chunk in self.stream(surface, **kwargs):
                output.append(chunk)

    async def close(self) -> None:
        if self.mode == "sync":
            self.client.close()
        else:
            await self.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("surface", ["native", "chat"])
@pytest.mark.parametrize("kind", ["empty", "comments", "partial"])
async def test_premature_clean_eof_raises_without_replaying(mode: str, surface: str, kind: str) -> None:
    frames = {"empty": [], "comments": [b": keepalive\n\n"], "partial": [_frame(_delta(surface))]}[kind]
    harness = _Harness(mode, frames)
    output: list[Any] = []
    try:
        with pytest.raises(SIEConnectionError, match="before generation completed"):
            await harness.collect(surface, output)
        assert output == ([_delta(surface)] if kind == "partial" else [])
        assert harness.calls == 1
        assert harness.client.last_retry_count == 0
        assert harness.body.closed
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("surface", ["native", "chat"])
@pytest.mark.parametrize("kind", ["done_only", "delta_done", "terminal_eof", "terminal_done", "terminal_tail"])
async def test_explicit_completion_remains_accepted(mode: str, surface: str, kind: str) -> None:
    delta, terminal = _delta(surface), _terminal(surface)
    frames = {
        "done_only": [_frame("[DONE]")],
        "delta_done": [_frame(delta), _frame("[DONE]")],
        "terminal_eof": [_frame(delta), _frame(terminal)],
        "terminal_done": [_frame(delta), _frame(terminal), _frame("[DONE]")],
        "terminal_tail": [_frame(delta), _frame(terminal).rstrip(b"\n")],
    }[kind]
    harness = _Harness(mode, frames)
    output: list[Any] = []
    try:
        await harness.collect(surface, output)
        expected = [] if kind == "done_only" else [delta]
        if kind.startswith("terminal"):
            expected.append(terminal)
        assert output == expected
        assert harness.calls == 1
        assert harness.body.closed
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("surface", ["native", "chat"])
@pytest.mark.parametrize("before_error", ["empty", "partial", "terminal"])
async def test_error_metadata_survives_eof_guard(mode: str, surface: str, before_error: str) -> None:
    error = {"request_id": "req-failure", "error": {"code": "transport_failure", "message": "controlled failure"}}
    prefix = {"empty": [], "partial": [_frame(_delta(surface))], "terminal": [_frame(_terminal(surface))]}
    frames = prefix[before_error] + [_frame(error)]
    harness = _Harness(mode, frames)
    try:
        with pytest.raises(ServerError, match="controlled failure") as caught:
            await harness.collect(surface, [])
        assert caught.value.code == "transport_failure"
        assert caught.value.request == {"id": "req-failure"}
        assert harness.calls == 1
        assert harness.client.last_retry_count == 0
        assert harness.body.closed
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("surface", ["native", "chat"])
async def test_actual_read_error_is_not_reclassified_or_retried(mode: str, surface: str) -> None:
    harness = _Harness(mode, [_frame(_delta(surface))], fail=True)
    try:
        with pytest.raises(SIEConnectionError, match="controlled read failure"):
            await harness.collect(surface, [])
        assert harness.calls == 1
        assert harness.body.closed
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("surface", ["native", "chat"])
async def test_caller_close_is_not_a_premature_eof(mode: str, surface: str) -> None:
    harness = _Harness(mode, [_frame(_delta(surface)), _frame(_terminal(surface))])
    iterator = harness.stream(surface)
    try:
        if mode == "sync":
            assert next(iterator) == _delta(surface)
            iterator.close()
        else:
            assert await anext(iterator) == _delta(surface)
            await iterator.aclose()
            # The public async wrapper's inner generator finalizes on the loop.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        assert harness.calls == 1
        assert harness.body.closed
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["native", "chat"])
async def test_caller_task_cancellation_is_not_a_premature_eof(surface: str) -> None:
    harness = _Harness("async", [_frame(_delta(surface))], block=True)
    iterator = harness.stream(surface)
    pending = None
    try:
        assert await anext(iterator) == _delta(surface)
        pending = asyncio.create_task(anext(iterator))
        await asyncio.wait_for(harness.body.waiting.wait(), timeout=1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert harness.calls == 1
        assert harness.body.closed
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
        await iterator.aclose()
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("complete", [False, True])
async def test_chat_eof_requires_every_requested_choice(mode: str, override: bool, complete: bool) -> None:
    chunks = [_delta("chat", index=0), _terminal("chat", index=0)]
    if complete:
        chunks += [_terminal("chat", index=1), {"choices": [], "usage": {"total_tokens": 4}}]
    harness = _Harness(mode, [_frame(chunk) for chunk in chunks])
    output: list[Any] = []
    kwargs = {"extra_body": {"n": 2}} if override else {"n": 2}
    try:
        if complete:
            await harness.collect("chat", output, **kwargs)
        else:
            with pytest.raises(SIEConnectionError, match="before generation completed"):
                await harness.collect("chat", output, **kwargs)
        assert output == chunks
        assert harness.calls == 1
        assert harness.body.closed
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("native_marker", [False, True])
async def test_chat_null_n_means_one_chat_choice(mode: str, native_marker: bool) -> None:
    chunk = {"done": True} if native_marker else _terminal("chat")
    harness = _Harness(mode, [_frame(chunk)])
    output: list[Any] = []
    try:
        if native_marker:
            with pytest.raises(SIEConnectionError, match="before generation completed"):
                await harness.collect("chat", output, extra_body={"n": None})
        else:
            await harness.collect("chat", output, extra_body={"n": None})
        assert output == [chunk]
        assert harness.calls == 1
        assert harness.body.closed
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("n", [0, False, -1, "2", [], {}, 1.5, True])
@pytest.mark.parametrize("complete", [False, True])
async def test_malformed_n_cannot_accept_empty_success_or_break_completion(mode: str, n: Any, complete: bool) -> None:
    chunks = [_terminal("chat")] if complete else []
    harness = _Harness(mode, [_frame(chunk) for chunk in chunks])
    try:
        if complete:
            await harness.collect("chat", [], extra_body={"n": n})
        else:
            with pytest.raises(SIEConnectionError, match="before generation completed"):
                await harness.collect("chat", [], extra_body={"n": n})
        assert harness.calls == 1
        assert harness.body.closed
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("indices", [[0, 0], [0, 2], [0, -1], [0, True], [0, 1.0]])
async def test_duplicate_or_invalid_choice_indices_do_not_complete_n_two(mode: str, indices: list[Any]) -> None:
    harness = _Harness(mode, [_frame(_terminal("chat", index=index)) for index in indices])
    try:
        with pytest.raises(SIEConnectionError, match="before generation completed"):
            await harness.collect("chat", [], n=2)
        assert harness.calls == 1
        assert harness.body.closed
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "chunk",
    [
        {"done": "true"},
        {"done": 1},
        {"choices": []},
        {"choices": [{"index": True, "finish_reason": "stop"}]},
        {"choices": [{"index": 1, "finish_reason": "stop"}]},
        {"choices": [{"index": 0, "finish_reason": ""}]},
        {"choices": [{"index": 0, "finish_reason": 1}]},
    ],
)
async def test_malformed_completion_does_not_mask_eof(mode: str, chunk: dict[str, Any]) -> None:
    surface = "native" if "done" in chunk else "chat"
    harness = _Harness(mode, [_frame(chunk)])
    try:
        with pytest.raises(SIEConnectionError, match="before generation completed"):
            await harness.collect(surface, [])
        assert harness.calls == 1
    finally:
        await harness.close()
