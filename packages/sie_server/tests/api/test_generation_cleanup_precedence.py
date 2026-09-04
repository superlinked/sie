"""Cancellation precedence across generation async cleanup boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from sie_server.adapters._generation_base import (
    GenerationChunk,
    aclose_with_error_precedence,
    suppress_thinking_blocks,
)
from sie_server.api.generate import _stream_generate_events
from sie_server.api.openai_completions import _collect_completion, _stream_completion
from sie_server.api.openai_local import _sanitize_sse_stream
from sie_server.processors.tool_call_parser import parse_tool_call_stream


class _BlockingCloseIterator:
    """Finite iterator whose explicit close blocks until cancelled."""

    def __init__(self, values: list[Any]) -> None:
        self._values = list(values)
        self.close_started = asyncio.Event()
        self.close_calls = 0

    def __aiter__(self) -> _BlockingCloseIterator:
        return self

    async def __anext__(self) -> Any:
        if self._values:
            return self._values.pop(0)
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        await asyncio.Event().wait()


class _IteratorAdapter:
    def __init__(self, source: _BlockingCloseIterator) -> None:
        self.source = source

    def generate(self, **_kwargs: Any) -> AsyncIterator[GenerationChunk]:
        return self.source


def _native_stream(
    source: _BlockingCloseIterator,
    *,
    suppress_thinking: bool,
) -> AsyncIterator[str]:
    return _stream_generate_events(
        _IteratorAdapter(source),
        prompt="hi",
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_k=None,
        min_new_tokens=None,
        grammar=None,
        seed=None,
        logit_bias=None,
        logprobs=False,
        top_logprobs=None,
        suppress_thinking=suppress_thinking,
    )


async def _cancel_during_close(
    stream: AsyncIterator[Any],
    source: _BlockingCloseIterator,
) -> None:
    await anext(stream)
    close_task = asyncio.create_task(stream.aclose())
    await source.close_started.wait()
    close_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await close_task


async def test_cleanup_cancellation_supersedes_ordinary_primary_exception() -> None:
    source = _BlockingCloseIterator([])

    async def run() -> None:
        try:
            raise RuntimeError("primary")
        finally:
            await aclose_with_error_precedence(
                source,
                outcome_selected=False,
                context="test helper",
            )

    task = asyncio.create_task(run())
    await source.close_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert source.close_calls == 1


@pytest.mark.parametrize("suppress_thinking", [False, True])
async def test_native_stream_cleanup_cancellation_supersedes_generator_exit(
    suppress_thinking: bool,
) -> None:
    source = _BlockingCloseIterator(
        [
            GenerationChunk(text_delta="visible", done=False),
            GenerationChunk(text_delta="later", done=False),
        ]
    )

    await _cancel_during_close(
        _native_stream(source, suppress_thinking=suppress_thinking),
        source,
    )

    assert source.close_calls == 1


async def test_thinking_wrapper_cleanup_cancellation_supersedes_generator_exit() -> None:
    source = _BlockingCloseIterator(
        [
            GenerationChunk(text_delta="visible", done=False),
            GenerationChunk(text_delta="later", done=False),
        ]
    )

    await _cancel_during_close(suppress_thinking_blocks(source), source)

    assert source.close_calls == 1


async def test_tool_parser_cleanup_cancellation_supersedes_generator_exit() -> None:
    source = _BlockingCloseIterator(
        [
            GenerationChunk(text_delta="visible long enough", done=False, is_first=True),
            GenerationChunk(text_delta="later", done=False),
        ]
    )

    await _cancel_during_close(parse_tool_call_stream(source), source)

    assert source.close_calls == 1


async def test_completion_stream_cleanup_cancellation_supersedes_generator_exit() -> None:
    source = _BlockingCloseIterator(
        [
            GenerationChunk(text_delta="visible", done=False),
            GenerationChunk(text_delta="later", done=False),
        ]
    )
    stream = _stream_completion(
        source,
        completion_id="cmpl-cleanup",
        created=1,
        model="cleanup-model",
        include_usage=False,
        registry=object(),
    )

    await _cancel_during_close(stream, source)

    assert source.close_calls == 1


async def test_completion_collector_cleanup_cancellation_supersedes_selected_outcome() -> None:
    source = _BlockingCloseIterator([GenerationChunk(text_delta="done", done=True, finish_reason="stop")])
    task = asyncio.create_task(_collect_completion(source))
    await source.close_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert source.close_calls == 1


class _BlockingCloseResponse(_BlockingCloseIterator):
    def aiter_bytes(self) -> AsyncIterator[bytes]:
        return self


class _CountingClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


async def test_child_sse_cleanup_cancellation_closes_all_and_releases_slot() -> None:
    response = _BlockingCloseResponse([b'data: {"choices":[]}\n', b"data: [DONE]\n"])
    client = _CountingClient()
    slot = asyncio.Semaphore(0)
    stream = _sanitize_sse_stream(
        response,  # type: ignore[arg-type]
        client,  # type: ignore[arg-type]
        requested_model="cleanup-model",
        hide_thinking_blocks=False,
        initial_inside_thinking=False,
        reasoning_format="qwen3",
        slot=slot,
    )

    await _cancel_during_close(stream, response)

    assert response.close_calls == 1
    assert client.close_calls == 1
    await asyncio.wait_for(slot.acquire(), timeout=0.1)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(slot.acquire(), timeout=0.01)
