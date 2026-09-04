"""Tests for the SIE-native ``/v1/generate`` streaming SSE shaper.

``_stream_generate_events`` emits the ``GenerateChunk`` wire shape that
``sie_sdk.SIEClient.stream_generate`` consumes (mirroring the gateway's
``build_generate_chunk_event``). These run on any platform with a fake adapter —
no MLX/torch — so a regression in the SSE contract is caught in normal CI, not
only the Mac nightly.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from sie_server.adapters._generation_base import (
    GenerationCapacityError,
    GenerationChunk,
    GenerationError,
    GenerationPreflightResult,
)
from sie_server.api.generate import _stream_generate_events


class _FakeAdapter:
    """Duck-typed GenerationAdapter that yields preset chunks (or raises)."""

    def __init__(self, chunks: list[GenerationChunk], raise_after: int | None = None) -> None:
        self._chunks = chunks
        self._raise_after = raise_after
        self.last_kwargs: dict[str, Any] | None = None
        self.closed = 0

    async def generate(self, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
        self.last_kwargs = kwargs
        try:
            for i, c in enumerate(self._chunks):
                if self._raise_after is not None and i == self._raise_after:
                    raise RuntimeError("boom")
                yield c
        finally:
            self.closed += 1


class _BlockingAdapter:
    """Yield once, then block until the consuming task is cancelled."""

    def __init__(self) -> None:
        self.blocked = asyncio.Event()
        self.closed = 0

    async def generate(self, **_kwargs: Any) -> AsyncIterator[GenerationChunk]:
        try:
            yield GenerationChunk(text_delta="visible", done=False)
            self.blocked.set()
            await asyncio.Event().wait()
        finally:
            self.closed += 1


class _FailingCloseIterator:
    """Adversarial iterator whose explicit cleanup always fails."""

    def __init__(
        self,
        chunks: list[GenerationChunk],
        *,
        block_after_chunks: bool = False,
        next_error: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._block_after_chunks = block_after_chunks
        self._next_error = next_error
        self._index = 0
        self.blocked = asyncio.Event()
        self.close_calls = 0

    def __aiter__(self) -> _FailingCloseIterator:
        return self

    async def __anext__(self) -> GenerationChunk:
        if self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk
        if self._next_error is not None:
            error = self._next_error
            self._next_error = None
            raise error
        if self._block_after_chunks:
            self.blocked.set()
            await asyncio.Event().wait()
        raise StopAsyncIteration

    async def aclose(self) -> None:
        self.close_calls += 1
        raise RuntimeError("cleanup failed")


class _IteratorAdapter:
    def __init__(self, iterator: AsyncIterator[GenerationChunk]) -> None:
        self.iterator = iterator

    def generate(self, **_kwargs: Any) -> AsyncIterator[GenerationChunk]:
        return self.iterator


class _PreflightDispatchAdapter:
    """Observe which lower-level generation entrypoint owns a request."""

    def __init__(self) -> None:
        self.generate_calls = 0
        self.generate_with_preflight_calls = 0
        self.received_preflight: GenerationPreflightResult | None = None

    async def generate(self, **_kwargs: Any) -> AsyncIterator[GenerationChunk]:
        self.generate_calls += 1
        yield GenerationChunk(text_delta="", done=True, finish_reason="stop")

    async def generate_with_preflight(
        self,
        _parameters: dict[str, Any],
        preflight: GenerationPreflightResult,
    ) -> AsyncIterator[GenerationChunk]:
        self.generate_with_preflight_calls += 1
        self.received_preflight = preflight
        yield GenerationChunk(text_delta="", done=True, finish_reason="stop")


def _parse_sse(raw: list[str]) -> tuple[list[dict[str, Any]], bool]:
    """Parse emitted SSE strings into events; returns (events, saw_DONE)."""
    events: list[dict[str, Any]] = []
    saw_done = False
    for block in raw:
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                saw_done = True
                continue
            events.append(json.loads(payload))
    return events, saw_done


def _events(
    adapter: Any,
    *,
    suppress_thinking: bool = False,
    preflight_result: GenerationPreflightResult | None = None,
) -> AsyncIterator[str]:
    return _stream_generate_events(
        adapter,
        prompt="hi",
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        stop=None,
        frequency_penalty=0.25,
        presence_penalty=-0.5,
        top_k=12,
        min_new_tokens=2,
        grammar=None,
        seed=None,
        logit_bias=None,
        logprobs=True,
        top_logprobs=3,
        suppress_thinking=suppress_thinking,
        preflight_result=preflight_result,
    )


async def _drain(adapter: Any, *, suppress_thinking: bool = False) -> list[str]:
    return [event async for event in _events(adapter, suppress_thinking=suppress_thinking)]


async def test_stream_without_preflight_uses_duck_typed_generate() -> None:
    adapter = _PreflightDispatchAdapter()

    events, saw_done = _parse_sse([event async for event in _events(adapter)])

    assert saw_done is True
    assert events[-1]["finish_reason"] == "stop"
    assert adapter.generate_calls == 1
    assert adapter.generate_with_preflight_calls == 0


async def test_stream_with_preflight_cannot_bypass_bound_entrypoint() -> None:
    adapter = _PreflightDispatchAdapter()
    preflight = GenerationPreflightResult(adapter, {}, payload="request-evidence")

    events, saw_done = _parse_sse([event async for event in _events(adapter, preflight_result=preflight)])

    assert saw_done is True
    assert events[-1]["finish_reason"] == "stop"
    assert adapter.generate_calls == 0
    assert adapter.generate_with_preflight_calls == 1
    assert adapter.received_preflight is preflight


async def test_stream_shapes_generatechunk() -> None:
    chunks = [
        GenerationChunk(text_delta="Hello", done=False, is_first=True),
        GenerationChunk(text_delta=" world", done=False),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=4, completion_tokens=2),
    ]
    raw = await _drain(_FakeAdapter(chunks))
    events, saw_done = _parse_sse(raw)
    assert saw_done is True

    deltas = [e for e in events if not e.get("done")]
    terminals = [e for e in events if e.get("done")]
    assert [e["text_delta"] for e in deltas] == ["Hello", " world"]
    assert all(e["done"] is False for e in deltas)
    # monotonic seq + stable request_id across the stream
    assert [e["seq"] for e in deltas] == [0, 1]
    assert len({e["request_id"] for e in events}) == 1

    assert len(terminals) == 1
    term = terminals[0]
    assert term["done"] is True
    assert term["finish_reason"] == "stop"
    assert term["usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    assert "ttft_ms" in term  # first non-empty delta was observed


async def test_stream_forwards_sampler_controls_and_emits_logprobs() -> None:
    logprobs = ({"token": "Hello", "logprob": -0.25, "bytes": [72], "top_logprobs": []},)
    adapter = _FakeAdapter(
        [
            GenerationChunk(text_delta="Hello", done=False, logprobs=logprobs),
            GenerationChunk(text_delta="", done=True, finish_reason="stop"),
        ]
    )
    events, _ = _parse_sse(await _drain(adapter))
    assert adapter.last_kwargs is not None
    assert adapter.last_kwargs["frequency_penalty"] == 0.25
    assert adapter.last_kwargs["presence_penalty"] == -0.5
    assert adapter.last_kwargs["top_k"] == 12
    assert adapter.last_kwargs["min_new_tokens"] == 2
    assert adapter.last_kwargs["logprobs"] is True
    assert adapter.last_kwargs["top_logprobs"] == 3
    assert events[0]["logprobs"] == list(logprobs)


async def test_stream_emits_logprobs_only_chunks_without_setting_ttft() -> None:
    logprobs = ({"token": "", "logprob": -0.25, "bytes": [], "top_logprobs": []},)
    adapter = _FakeAdapter(
        [
            GenerationChunk(text_delta="", done=False, logprobs=logprobs),
            GenerationChunk(text_delta="", done=True, finish_reason="stop", logprobs=logprobs),
        ]
    )

    events, saw_done = _parse_sse(await _drain(adapter))

    assert saw_done is True
    assert [event["logprobs"] for event in events[:-1]] == [list(logprobs), list(logprobs)]
    assert all(event["text_delta"] == "" and event["done"] is False for event in events[:-1])
    assert events[-1]["done"] is True
    assert "ttft_ms" not in events[-1]


async def test_stream_error_emits_terminal_error_chunk() -> None:
    chunks = [
        GenerationChunk(text_delta="partial", done=False, is_first=True),
        GenerationChunk(text_delta="never", done=False),
    ]
    adapter = _FakeAdapter(chunks, raise_after=1)
    raw = await _drain(adapter)
    events, saw_done = _parse_sse(raw)
    assert saw_done is True
    terminal = next(e for e in events if e.get("done"))
    assert terminal["finish_reason"] == "error"
    assert terminal["error"]["code"] == "inference_error"
    # The raw exception text must NOT leak to the client (CodeQL info-exposure);
    # it is logged server-side only. The client gets a generic message.
    assert "boom" not in terminal["error"]["message"]
    assert terminal["error"]["message"] == "internal error during generation"
    assert adapter.closed == 1


async def test_stream_capacity_error_carries_configured_retry_hint() -> None:
    iterator = _FailingCloseIterator(
        [GenerationChunk(text_delta="partial", done=False)],
        next_error=GenerationCapacityError("scheduler full"),
    )
    raw = [
        event
        async for event in _stream_generate_events(
            _IteratorAdapter(iterator),
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
            oom_retry_after_s=12,
        )
    ]

    events, saw_done = _parse_sse(raw)
    terminal = next(event for event in events if event.get("done"))
    assert saw_done is True
    assert terminal["error"] == {
        "code": "RESOURCE_EXHAUSTED",
        "message": "scheduler full",
        "retry_after_s": 12,
    }


async def test_stream_terminal_break_closes_adapter_iterator_once() -> None:
    adapter = _FakeAdapter([GenerationChunk(text_delta="done", done=True, finish_reason="stop")])

    await _drain(adapter)

    assert adapter.closed == 1


async def test_stream_consumer_close_closes_adapter_iterator_once() -> None:
    adapter = _FakeAdapter(
        [
            GenerationChunk(text_delta="first", done=False),
            GenerationChunk(text_delta="never", done=False),
        ]
    )
    events = _stream_generate_events(
        adapter,
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
    )

    assert "first" in await anext(events)
    await events.aclose()

    assert adapter.closed == 1


async def test_stream_consumer_close_through_thinking_wrapper_closes_upstream_once() -> None:
    adapter = _FakeAdapter(
        [
            GenerationChunk(text_delta="<think>private</think>visible", done=False),
            GenerationChunk(text_delta="never", done=False),
        ]
    )
    events = _stream_generate_events(
        adapter,
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
        suppress_thinking=True,
    )

    assert "visible" in await anext(events)
    await events.aclose()

    assert adapter.closed == 1


async def test_stream_consumer_cancellation_through_thinking_wrapper_closes_upstream_once() -> None:
    adapter = _BlockingAdapter()
    events = _stream_generate_events(
        adapter,
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
        suppress_thinking=True,
    )

    assert "visible" in await anext(events)
    pending = asyncio.create_task(anext(events))
    await adapter.blocked.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert adapter.closed == 1


@pytest.mark.parametrize("suppress_thinking", [False, True])
async def test_stream_cancellation_preserves_cancelled_error_when_cleanup_fails(
    suppress_thinking: bool,
) -> None:
    iterator = _FailingCloseIterator(
        [GenerationChunk(text_delta="visible", done=False)],
        block_after_chunks=True,
    )
    events = _events(_IteratorAdapter(iterator), suppress_thinking=suppress_thinking)

    assert "visible" in await anext(events)
    pending = asyncio.create_task(anext(events))
    await iterator.blocked.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending

    assert iterator.close_calls == 1


@pytest.mark.parametrize("suppress_thinking", [False, True])
async def test_stream_explicit_close_is_not_masked_by_cleanup_failure(suppress_thinking: bool) -> None:
    iterator = _FailingCloseIterator(
        [GenerationChunk(text_delta="visible", done=False)],
        block_after_chunks=True,
    )
    events = _events(_IteratorAdapter(iterator), suppress_thinking=suppress_thinking)

    assert "visible" in await anext(events)
    await events.aclose()

    assert iterator.close_calls == 1


@pytest.mark.parametrize("suppress_thinking", [False, True])
async def test_stream_terminal_outcome_is_not_masked_by_cleanup_failure(suppress_thinking: bool) -> None:
    iterator = _FailingCloseIterator([GenerationChunk(text_delta="done", done=True, finish_reason="stop")])

    events, saw_done = _parse_sse(await _drain(_IteratorAdapter(iterator), suppress_thinking=suppress_thinking))

    assert saw_done is True
    assert events[-1]["finish_reason"] == "stop"
    assert "error" not in events[-1]
    assert iterator.close_calls == 1


@pytest.mark.parametrize("suppress_thinking", [False, True])
async def test_stream_generation_error_is_not_masked_by_cleanup_failure(suppress_thinking: bool) -> None:
    iterator = _FailingCloseIterator([], next_error=RuntimeError("generation failed"))

    events, saw_done = _parse_sse(await _drain(_IteratorAdapter(iterator), suppress_thinking=suppress_thinking))

    assert saw_done is True
    assert events[-1]["finish_reason"] == "error"
    assert events[-1]["error"] == {
        "code": "inference_error",
        "message": "internal error during generation",
    }
    assert iterator.close_calls == 1


@pytest.mark.parametrize("suppress_thinking", [False, True])
async def test_stream_standalone_cleanup_failure_is_a_terminal_error(suppress_thinking: bool) -> None:
    iterator = _FailingCloseIterator([])

    events, saw_done = _parse_sse(await _drain(_IteratorAdapter(iterator), suppress_thinking=suppress_thinking))

    assert saw_done is True
    assert events[-1]["finish_reason"] == "error"
    assert events[-1]["error"]["code"] == "inference_error"
    assert "cleanup failed" not in events[-1]["error"]["message"]
    assert iterator.close_calls == 1


async def test_stream_propagates_typed_terminal_error_chunk() -> None:
    adapter = _FakeAdapter(
        [
            GenerationChunk(
                text_delta="",
                done=True,
                finish_reason="error",
                error_code="grammar_invalid",
                error_message="invalid grammar",
            )
        ]
    )

    events, saw_done = _parse_sse(await _drain(adapter))

    assert saw_done is True
    terminal = events[-1]
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "error"
    assert terminal["error"] == {"code": "grammar_invalid", "message": "invalid grammar"}


@pytest.mark.parametrize(
    ("code", "message", "expected_code", "expected_message"),
    [
        (
            "inference_error",
            "SENSITIVE_NATIVE_SSE_RUNTIME",
            "inference_error",
            "internal error during generation",
        ),
        (
            "grammar_compile_failed",
            "SENSITIVE_NATIVE_SSE_GRAMMAR",
            "grammar_compile_failed",
            "internal error compiling grammar",
        ),
        (
            "MODEL_OUTPUT_PARSE_ERROR",
            "invalid model JSON",
            "MODEL_OUTPUT_PARSE_ERROR",
            "invalid model JSON",
        ),
        (
            "SENSITIVE_NATIVE_SSE_ERROR_CODE",
            "SENSITIVE_NATIVE_SSE_ERROR_CODE",
            "inference_error",
            "generation terminated with an upstream error",
        ),
    ],
)
async def test_stream_sanitizes_adapter_yielded_terminal_message(
    code: str,
    message: str,
    expected_code: str,
    expected_message: str,
) -> None:
    adapter = _FakeAdapter(
        [
            GenerationChunk(
                text_delta="",
                done=True,
                finish_reason="error",
                error_code=code,
                error_message=message,
            )
        ]
    )

    events, saw_done = _parse_sse(await _drain(adapter))

    assert saw_done is True
    assert events[-1]["finish_reason"] == "error"
    assert events[-1]["error"] == {"code": expected_code, "message": expected_message}
    if message.startswith("SENSITIVE_"):
        assert message not in str(events)


async def test_stream_sanitizes_generic_typed_generation_exception() -> None:
    sentinel = "SENSITIVE_TYPED_NATIVE_SSE"
    failure = GenerationError(sentinel)
    failure.param = sentinel

    class _TypedFailureAdapter:
        async def generate(self, **_kwargs: Any) -> AsyncIterator[GenerationChunk]:
            raise failure
            yield GenerationChunk(text_delta="unreachable")  # pragma: no cover

    events, saw_done = _parse_sse(await _drain(_TypedFailureAdapter()))

    assert saw_done is True
    assert events[-1]["error"] == {
        "code": "inference_error",
        "message": "internal error during generation",
    }
    assert sentinel not in str(events[-1])


async def test_stream_exhaustion_without_terminal_is_an_error() -> None:
    adapter = _FakeAdapter([GenerationChunk(text_delta="partial", done=False)])

    events, saw_done = _parse_sse(await _drain(adapter))

    assert saw_done is True
    terminal = events[-1]
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "error"
    assert terminal["error"] == {
        "code": "inference_error",
        "message": "generation stream ended before a terminal event",
    }


async def test_terminal_text_is_emitted_as_done_false_delta() -> None:
    chunks = [
        GenerationChunk(
            text_delta="final",
            done=True,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )
    ]
    raw = await _drain(_FakeAdapter(chunks))
    events, saw_done = _parse_sse(raw)
    assert saw_done is True
    assert events[0]["text_delta"] == "final"
    assert events[0]["done"] is False
    assert events[1]["done"] is True


async def test_stream_empty_generation_still_terminates() -> None:
    chunks = [GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=0)]
    raw = await _drain(_FakeAdapter(chunks))
    events, saw_done = _parse_sse(raw)
    assert saw_done is True
    terminal = next(e for e in events if e.get("done"))
    assert terminal["finish_reason"] == "stop"
    assert terminal["usage"]["completion_tokens"] == 0
    # No text delta was produced → no ttft_ms on the terminal chunk.
    assert "ttft_ms" not in terminal


async def test_done_is_final_sse_line() -> None:
    chunks = [
        GenerationChunk(text_delta="hi", done=False),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    ]
    raw = await _drain(_FakeAdapter(chunks))
    assert raw[-1].strip() == "data: [DONE]"
