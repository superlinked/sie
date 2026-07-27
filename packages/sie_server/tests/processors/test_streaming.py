"""Tests for the ``StreamingProcessor``.

Exercises the chunk-publishing path with a fake :class:`GenerationAdapter`:

- Coalescing yields one chunk per text-batch (time/count flush).
- Every chunk envelope carries ``request_id`` + ``attempt_id`` + ``seq``.
- The terminal chunk has ``done: true`` and ``finish_reason``.
- Sustained publish failures abort with a transport_failure terminal.
- Queue overflow produces a transport_failure terminal.
- Cancellation event triggers a ``finish_reason: "cancelled"`` terminal.
- Two consecutive ``process()`` calls produce distinct ``attempt_id``s.
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack
import pytest
from sie_server.adapters._generation_base import (
    GenerationAdapter,
    GenerationChunk,
)
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.observability import worker_telemetry as worker_metrics
from sie_server.processors import streaming as streaming_mod
from sie_server.processors.streaming import StreamingProcessor


class _FakeGenAdapter(GenerationAdapter):
    """Yields a scripted sequence of chunks. Optionally blocks before each."""

    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self, script: list[GenerationChunk], *, hold_event: asyncio.Event | None = None) -> None:
        self._device = None
        self._script = script
        self._hold = hold_event

    def load(self, device: str) -> None:  # pragma: no cover — registry-mocked
        self._device = device

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(inputs=["text"], outputs=["tokens"])

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    async def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        _ = (prompt, max_new_tokens, temperature, top_p, stop, kwargs)
        for chunk in self._script:
            if self._hold is not None:
                # Block here until externally released — used for cancel test.
                await self._hold.wait()
            yield chunk


def _make_work_item(
    *,
    generate: dict[str, Any] | None = None,
    messages: list[dict[str, str]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Build a generate-shape WorkItem dict.

    ``generate`` overrides the default ``{prompt, max_new_tokens}`` block
    wholesale. Pass ``messages=[{role, content}, ...]`` to build a chat
    work item without spelling out the full generate dict.
    """
    if generate is None:
        if messages is not None:
            generate = {"messages": messages, "max_new_tokens": 64}
        else:
            generate = {"prompt": "Hello", "max_new_tokens": 64}
    wi: dict[str, Any] = {
        "work_item_id": "req-1.0",
        "request_id": "req-1",
        "item_index": 0,
        "total_items": 1,
        "operation": "generate",
        "model_id": "test/model",
        "profile_id": "default",
        "pool_name": "default",
        "machine_profile": "default",
        "router_id": "router-1",
        "reply_subject": "_INBOX.router-1.req-1",
        "timestamp": time.time(),
        "generate": generate,
    }
    wi.update(overrides)
    return wi


def _make_msg(wi: dict[str, Any]) -> AsyncMock:
    msg = AsyncMock()
    msg.data = msgpack.packb(wi, use_bin_type=True)
    return msg


def _make_registry(adapter: GenerationAdapter) -> MagicMock:
    registry = MagicMock()
    registry.is_loaded.return_value = True
    registry.get.return_value = adapter
    registry.device = "cpu"
    return registry


def _decode_chunks(nc_mock: AsyncMock) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for call in nc_mock.publish.await_args_list:
        _subject, data = call.args
        chunks.append(msgpack.unpackb(data, raw=False))
    return chunks


def _capture_worker_completion(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    telemetry = MagicMock()
    monkeypatch.setattr(worker_metrics, "worker_telemetry_enabled", lambda: True)
    monkeypatch.setattr(worker_metrics, "worker_telemetry", lambda: telemetry)
    return telemetry


def _assert_one_generation_completion(telemetry: MagicMock, *, outcome: str) -> None:
    telemetry.item_completed.assert_called_once()
    kwargs = telemetry.item_completed.call_args.kwargs
    assert kwargs["operation"] == "generate"
    assert kwargs["outcome"] == outcome
    assert kwargs["model"] == "test/model"
    assert kwargs["profile"] == "default"
    assert kwargs["duration_s"] >= 0


@pytest.mark.asyncio
async def test_disabled_terminal_publication_skips_telemetry_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    nc = AsyncMock()
    proc = StreamingProcessor(nc=nc, registry=_make_registry(_FakeGenAdapter([])), worker_id="w1")

    def fail_decode(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("disabled completion path must not decode terminal payloads")

    monkeypatch.setattr(streaming_mod.msgpack, "unpackb", fail_decode)
    await proc._publish_terminal_envelope("_INBOX.test", b"opaque-terminal")

    nc.publish.assert_awaited_once_with("_INBOX.test", b"opaque-terminal")


@pytest.mark.asyncio
async def test_streaming_publishes_per_chunk_with_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three delta chunks + a terminal → at least one delta msg + one terminal."""
    telemetry = _capture_worker_completion(monkeypatch)
    nc = AsyncMock()
    # Script: three short deltas across separate timestamps, then terminal.
    script = [
        GenerationChunk(text_delta="Hello", is_first=True),
        GenerationChunk(text_delta=" world"),
        GenerationChunk(text_delta="!"),
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=3,
        ),
    ]
    adapter = _FakeGenAdapter(script)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")

    msg = _make_msg(_make_work_item())
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    assert len(decoded) >= 2
    # Final chunk is the terminal one.
    terminal = decoded[-1]
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "stop"
    assert terminal["usage"]["completion_tokens"] == 3
    assert terminal["usage"]["total_tokens"] == 8
    # Every chunk carries request_id and attempt_id; attempt_id is consistent.
    attempt_ids = {c["attempt_id"] for c in decoded}
    assert len(attempt_ids) == 1
    attempt_id = next(iter(attempt_ids))
    assert attempt_id  # non-empty
    assert all(c["request_id"] == "req-1" for c in decoded)
    # All chunks carry kind == "chunk".
    assert all(c["kind"] == "chunk" for c in decoded)
    # seq is monotonic non-decreasing.
    seqs = [c["seq"] for c in decoded]
    assert seqs == sorted(seqs)
    # ACK happened.
    msg.ack.assert_awaited()
    _assert_one_generation_completion(telemetry, outcome="success")


@pytest.mark.asyncio
async def test_coalescing_keeps_message_count_bounded() -> None:
    """100 tiny yields should coalesce into ≤ N chunks (≤32 chars/flush)."""
    nc = AsyncMock()
    deltas = [GenerationChunk(text_delta="a", is_first=(i == 0)) for i in range(100)]
    deltas.append(
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=100)
    )
    adapter = _FakeGenAdapter(deltas)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")

    msg = _make_msg(_make_work_item())
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    # 100 single chars × len()==1 each → batched in groups of 32 → ~4 delta
    # messages plus 1 terminal = ≤ 6 total. Time flushes can add a couple
    # more depending on scheduling, so allow up to 8.
    assert len(decoded) <= 8, f"got {len(decoded)} chunks, expected ≤8"
    terminal = decoded[-1]
    assert terminal["done"] is True
    # The concatenated deltas reconstruct the full output.
    body = "".join(c.get("text_delta", "") for c in decoded)
    assert body == "a" * 100


@pytest.mark.asyncio
async def test_two_pickups_yield_distinct_attempt_ids() -> None:
    """Simulated redelivery → fresh attempt_id on every process() call."""
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="hi", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]

    proc = StreamingProcessor(nc=nc, registry=_make_registry(_FakeGenAdapter(script)), worker_id="w1")
    msg1 = _make_msg(_make_work_item())
    await proc.process(msg1, "test/model")
    attempt_a = _decode_chunks(nc)[-1]["attempt_id"]

    nc.publish.reset_mock()
    proc._registry = _make_registry(_FakeGenAdapter(script))
    msg2 = _make_msg(_make_work_item())
    await proc.process(msg2, "test/model")
    attempt_b = _decode_chunks(nc)[-1]["attempt_id"]

    assert attempt_a != attempt_b


@pytest.mark.asyncio
async def test_cancel_event_emits_cancelled_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """signal_cancel() mid-stream → final chunk has finish_reason=cancelled."""
    telemetry = _capture_worker_completion(monkeypatch)
    nc = AsyncMock()
    hold = asyncio.Event()
    script = [
        GenerationChunk(text_delta="partial", is_first=True),
        GenerationChunk(text_delta=" more"),  # will never be yielded — held
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    ]
    # First chunk yields immediately; subsequent yields wait on `hold`.
    # Workaround: tweak the adapter to not hold on the first yield.

    class _AdapterFirstFree(_FakeGenAdapter):
        async def generate(self, prompt, *, max_new_tokens, temperature=1.0, top_p=1.0, stop=None):
            for i, chunk in enumerate(self._script):
                if i > 0 and self._hold is not None:
                    await self._hold.wait()
                yield chunk

    adapter = _AdapterFirstFree(script, hold_event=hold)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    msg = _make_msg(_make_work_item())

    async def _cancel_after_first() -> None:
        # Wait until at least one publish has happened, then signal cancel.
        for _ in range(200):
            if nc.publish.await_count >= 1:
                break
            await asyncio.sleep(0.01)
        proc.signal_cancel("req-1")
        hold.set()  # let the adapter's awaiting wait() unblock so aclose can proceed

    cancel_task = asyncio.create_task(_cancel_after_first())
    await proc.process(msg, "test/model")
    await cancel_task

    decoded = _decode_chunks(nc)
    terminal = decoded[-1]
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "cancelled"
    msg.ack.assert_awaited()
    _assert_one_generation_completion(telemetry, outcome="cancelled")


@pytest.mark.asyncio
async def test_full_queue_at_finally_never_drops_terminal_chunk(monkeypatch) -> None:
    """Fix #1 regression: when the chunk queue is full as the producer
    finishes, the finally block must NOT force-drain a queued payload to make
    room for the ``None`` sentinel — that would discard the terminal chunk
    (lost ACK → full redelivery) or a mid-stream chunk (silent seq gap).

    We shrink the queue to 2 and use a slow publisher so the queue is full
    (text chunk + terminal) exactly when the producer reaches the finally.
    The fix waits for the publisher to drain instead of dropping a payload, so
    every chunk — including the terminal — is published, seqs are contiguous,
    and the message is ACKed.
    """
    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "_CHUNK_QUEUE_MAX", 2)

    nc = AsyncMock()

    # Gate the publisher loop at the very start so it does NOT pull anything
    # off the queue until released. The producer then fills the queue to its
    # cap (text chunk seq 0 + terminal seq 1 == maxsize 2) and reaches the
    # finally with a genuinely full queue, exercising the QueueFull sentinel
    # path. Once released, the publisher drains everything in order.
    release = asyncio.Event()
    real_publisher_loop = StreamingProcessor._publisher_loop

    async def _gated_publisher_loop(self, chunk_queue, reply_subject):
        await release.wait()
        return await real_publisher_loop(self, chunk_queue, reply_subject)

    monkeypatch.setattr(StreamingProcessor, "_publisher_loop", _gated_publisher_loop)

    script = [
        # ≥32 chars so it flushes as its own chunk (seq 0) immediately.
        GenerationChunk(text_delta="x" * 40, is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    adapter = _FakeGenAdapter(script)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    msg = _make_msg(_make_work_item())

    async def _release_when_queue_full() -> None:
        # Give the producer time to fill the queue and park in the finally's
        # blocking sentinel ``put`` (the queue is full so it cannot complete
        # until the publisher drains a slot), then release the publisher.
        await asyncio.sleep(0.2)
        release.set()

    releaser = asyncio.create_task(_release_when_queue_full())
    await proc.process(msg, "test/model")
    await releaser

    decoded = _decode_chunks(nc)
    # No payload was dropped: seqs are contiguous from 0.
    seqs = [c["seq"] for c in decoded]
    assert seqs == list(range(len(seqs)))
    # The terminal survived and was published last.
    terminal = decoded[-1]
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "stop"
    # Full body reconstructs (no mid-stream chunk dropped).
    body = "".join(c.get("text_delta", "") for c in decoded)
    assert body == "x" * 40
    # Terminal confirmed published → message ACKed.
    msg.ack.assert_awaited()


@pytest.mark.asyncio
async def test_publish_failures_emit_transport_failure_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Publish failures leave the JetStream message unacked for redelivery."""
    telemetry = _capture_worker_completion(monkeypatch)
    nc = AsyncMock()
    nc.publish.side_effect = RuntimeError("nats down")

    # Burst many small chunks fast enough to cause coalesced flushes that
    # outpace the publisher loop draining (which immediately fails).
    deltas = [GenerationChunk(text_delta="a" * 40, is_first=(i == 0)) for i in range(200)]
    deltas.append(GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1))
    adapter = _FakeGenAdapter(deltas)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    msg = _make_msg(_make_work_item())

    await proc.process(msg, "test/model")

    # The terminal chunk was not confirmed published, so the worker must not
    # ACK; JetStream redelivery is safer than losing a completed generation.
    msg.ack.assert_not_awaited()
    telemetry.item_completed.assert_not_called()


@pytest.mark.asyncio
async def test_heartbeat_calls_in_progress_during_slow_stream(monkeypatch) -> None:
    """Slow streams trigger ``msg.in_progress()`` heartbeats."""
    from sie_server.processors import streaming as streaming_mod

    # Shrink heartbeat interval so the test runs in well under a second.
    monkeypatch.setattr(streaming_mod, "_INPROGRESS_INTERVAL_S", 0.05)

    nc = AsyncMock()
    hold = asyncio.Event()

    class _SlowAdapter(GenerationAdapter):
        spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

        def load(self, device: str) -> None:  # pragma: no cover
            self._device = device

        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(inputs=["text"], outputs=["tokens"])

        @property
        def dims(self) -> ModelDims:
            return ModelDims()

        async def generate(self, prompt, *, max_new_tokens, temperature=1.0, top_p=1.0, stop=None):
            # Emit one chunk, hold, then terminate.
            yield GenerationChunk(text_delta="hi", is_first=True)
            try:
                await asyncio.wait_for(hold.wait(), timeout=0.3)
            except TimeoutError:
                pass
            yield GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1)

    proc = StreamingProcessor(nc=nc, registry=_make_registry(_SlowAdapter()), worker_id="w1")
    msg = _make_msg(_make_work_item())
    msg.in_progress = AsyncMock()

    await proc.process(msg, "test/model")

    # At least one in_progress heartbeat fired while the adapter was held.
    assert msg.in_progress.await_count >= 1
    msg.ack.assert_awaited()


@pytest.mark.asyncio
async def test_iterator_raises_emits_error_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the adapter iterator raises, we publish an error terminal and ACK."""
    telemetry = _capture_worker_completion(monkeypatch)
    nc = AsyncMock()

    class _RaisingAdapter(GenerationAdapter):
        spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

        def load(self, device: str) -> None:  # pragma: no cover
            self._device = device

        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(inputs=["text"], outputs=["tokens"])

        @property
        def dims(self) -> ModelDims:
            return ModelDims()

        async def generate(self, prompt, *, max_new_tokens, temperature=1.0, top_p=1.0, stop=None):
            yield GenerationChunk(text_delta="oops", is_first=True)
            raise RuntimeError("adapter exploded")

    proc = StreamingProcessor(nc=nc, registry=_make_registry(_RaisingAdapter()), worker_id="w1")
    msg = _make_msg(_make_work_item())
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    terminal = decoded[-1]
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "error"
    assert terminal["error"]["code"] == "inference_error"
    msg.ack.assert_awaited()
    _assert_one_generation_completion(telemetry, outcome="error")


@pytest.mark.asyncio
async def test_retry_nak_is_not_a_completed_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """A routing retry hands ownership off; the eventual attempt owns completion."""
    telemetry = _capture_worker_completion(monkeypatch)
    nc = AsyncMock()
    registry = _make_registry(_FakeGenAdapter([]))
    registry.is_loaded.return_value = False
    registry.load_async = AsyncMock(side_effect=KeyError("not registered"))
    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    msg = _make_msg(_make_work_item())

    await proc.process(msg, "test/model")

    envelope = _decode_chunks(nc)[0]
    assert envelope["kind"] == "nak"
    assert envelope["reason"] == "model_not_loaded"
    msg.ack.assert_awaited_once()
    telemetry.item_completed.assert_not_called()


# ── chat-template rendering, context-exceeded, inert fields ─────────────


def _make_registry_with_chat_config(
    adapter: GenerationAdapter,
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    context_length: int = 32768,
    hf_id: str = "test/model",
    hf_revision: str | None = None,
) -> MagicMock:
    """Variant of :func:`_make_registry` that also fakes ``get_config``.

    The streaming processor's chat-template path reads
    ``config.tasks.generate.chat_template_kwargs`` and the context-length
    path reads ``config.tasks.generate.context_length`` and ``config.hf_id``.
    The tokenizer load also reads ``config.hf_revision`` to pin the load.
    """
    registry = _make_registry(adapter)
    config = MagicMock()
    config.hf_id = hf_id
    config.hf_revision = hf_revision
    config.weights_path = None
    config.tasks.generate.chat_template_kwargs = chat_template_kwargs or {}
    config.tasks.generate.context_length = context_length
    registry.get_config.return_value = config
    return registry


@pytest.mark.asyncio
async def test_streaming_processor_renders_chat_template(monkeypatch) -> None:
    """``Messages`` shape → adapter receives the rendered template string."""
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="hi", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    adapter = _FakeGenAdapter(script)

    captured_prompts: list[str] = []
    original_generate = adapter.generate

    async def _capture_generate(prompt, *, max_new_tokens, temperature=1.0, top_p=1.0, stop=None):
        captured_prompts.append(prompt)
        async for chunk in original_generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, stop=stop
        ):
            yield chunk

    adapter.generate = _capture_generate  # type: ignore[method-assign]

    registry = _make_registry_with_chat_config(adapter, chat_template_kwargs={"enable_thinking": False})

    # Patch ``load_tokenizer`` (called from a thread) with a stub that
    # records the kwargs forwarded to ``apply_chat_template``.
    seen_kwargs: dict[str, Any] = {}

    class _StubTok:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            seen_kwargs.update(kwargs)
            assert tokenize is False
            assert add_generation_prompt is True
            return "<rendered>" + "".join(m["content"] for m in messages) + "</rendered>"

        def encode(self, text, *, add_special_tokens):
            return [0] * len(text.split())

    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "load_tokenizer", lambda *a, **kw: _StubTok())

    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    wi = _make_work_item(messages=[{"role": "user", "content": "ping"}])
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    assert len(captured_prompts) == 1
    assert captured_prompts[0] == "<rendered>ping</rendered>"
    assert seen_kwargs == {"enable_thinking": False}
    decoded = _decode_chunks(nc)
    assert decoded[-1]["done"] is True
    assert decoded[-1]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_streaming_processor_pins_tokenizer_revision(monkeypatch) -> None:
    """The tokenizer load forwards the model's pinned ``hf_revision``.

    Under ``HF_HUB_OFFLINE=1`` a no-revision load resolves ``refs/main``,
    which is absent for repos staged by pinned SHA — pinning the revision
    resolves the ``snapshots/<sha>`` dir directly (#1801).
    """
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="ok", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    adapter = _FakeGenAdapter(script)
    sha = "cdbee75f17c01a7cc42f958dc650907174af0554"
    registry = _make_registry_with_chat_config(adapter, hf_id="test/model", hf_revision=sha)

    seen: dict[str, Any] = {}

    class _StubTok:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            return "<rendered>" + "".join(m["content"] for m in messages) + "</rendered>"

        def encode(self, text, *, add_special_tokens):
            return [0] * len(text.split())

    def _spy_load(source, *, trust_remote_code=False, revision=None):
        seen["source"] = source
        seen["revision"] = revision
        return _StubTok()

    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "load_tokenizer", _spy_load)

    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    wi = _make_work_item(messages=[{"role": "user", "content": "ping"}])
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    assert seen["source"] == "test/model"
    assert seen["revision"] == sha


@pytest.mark.asyncio
async def test_streaming_processor_chat_template_render_error_becomes_terminal_chunk(monkeypatch) -> None:
    """A tokenizer that raises mid-render → terminal ``invalid_request``."""
    nc = AsyncMock()
    adapter = _FakeGenAdapter([])
    registry = _make_registry_with_chat_config(adapter)

    class _BrokenTok:
        def apply_chat_template(self, *a, **kw):
            raise ValueError("template not found")

        def encode(self, text, *, add_special_tokens):  # pragma: no cover
            return []

    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "load_tokenizer", lambda *a, **kw: _BrokenTok())

    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    wi = _make_work_item(messages=[{"role": "user", "content": "hi"}])
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    assert len(decoded) == 1
    terminal = decoded[-1]
    assert terminal["done"] is True
    assert terminal["error"]["code"] == "invalid_request"
    assert "chat template render failed" in terminal["error"]["message"]
    msg.ack.assert_awaited()


@pytest.mark.asyncio
async def test_streaming_processor_rejects_invalid_role() -> None:
    """A messages entry with an unsupported role is rejected pre-template."""
    nc = AsyncMock()
    adapter = _FakeGenAdapter([])
    registry = _make_registry_with_chat_config(adapter)
    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    # ``tool`` is a valid role now (multi-turn tool use); use an unknown
    # role to exercise the rejection path.
    wi = _make_work_item(messages=[{"role": "function", "content": "noop"}])
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    terminal = decoded[-1]
    assert terminal["done"] is True
    assert terminal["error"]["code"] == "invalid_request"
    assert "messages[0].role" in terminal["error"]["message"]


@pytest.mark.asyncio
async def test_streaming_processor_folds_developer_role_to_system(monkeypatch) -> None:
    """A ``developer`` message is accepted and rendered with role ``system``.

    The gateway normally normalizes this; the worker fold is defensive
    (Qwen's chat template has no ``developer`` slot). See roadmap item 1.7.
    """
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="ok", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    adapter = _FakeGenAdapter(script)
    registry = _make_registry_with_chat_config(adapter)

    seen_roles: list[str] = []

    class _StubTok:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            seen_roles.extend(m["role"] for m in messages)
            return "rendered"

        def encode(self, text, *, add_special_tokens):
            return [0]

    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "load_tokenizer", lambda *a, **kw: _StubTok())

    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    wi = _make_work_item(
        messages=[
            {"role": "developer", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]
    )
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    # ``developer`` folded to ``system``; ``user`` unchanged.
    assert seen_roles == ["system", "user"]
    decoded = _decode_chunks(nc)
    assert decoded[-1]["done"] is True
    assert decoded[-1].get("error") is None


@pytest.mark.asyncio
async def test_streaming_processor_context_exceeded_emits_terminal_chunk(monkeypatch) -> None:
    """``prompt_tokens + max_new_tokens > context_length`` → ``context_exceeded``."""
    nc = AsyncMock()
    adapter = _FakeGenAdapter([])
    # Tiny context window so a 5-token prompt + 64-token cap overflows.
    registry = _make_registry_with_chat_config(adapter, context_length=32)

    class _StubTok:
        def apply_chat_template(self, *a, **kw):
            return "rendered"

        def encode(self, text, *, add_special_tokens):
            # 100 fake tokens — guaranteed overflow.
            return list(range(100))

    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "load_tokenizer", lambda *a, **kw: _StubTok())

    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    # Use the prompt shape so the test exercises the non-Messages path.
    wi = _make_work_item(generate={"prompt": "long" * 50, "max_new_tokens": 64})
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    assert len(decoded) == 1
    terminal = decoded[-1]
    assert terminal["done"] is True
    assert terminal["error"]["code"] == "context_exceeded"
    assert "context_length" in terminal["error"]["message"]
    msg.ack.assert_awaited()


@pytest.mark.asyncio
async def test_streaming_processor_ignores_routing_key_this_slice() -> None:
    """Inert routing-affinity fields don't disturb the happy path."""
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="ok", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    adapter = _FakeGenAdapter(script)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    wi = _make_work_item(routing_key="users/42", prompt_cache_key="cache-abc")
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    assert decoded[-1]["done"] is True
    assert decoded[-1]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_streaming_processor_rejects_prompt_and_messages_together() -> None:
    """A work item with both ``prompt`` and ``messages`` → ``invalid_request``."""
    nc = AsyncMock()
    adapter = _FakeGenAdapter([])
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    wi = _make_work_item(
        generate={
            "prompt": "hi",
            "messages": [{"role": "user", "content": "hi"}],
            "max_new_tokens": 8,
        }
    )
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    terminal = decoded[-1]
    assert terminal["done"] is True
    assert terminal["error"]["code"] == "invalid_request"
    assert "mutually exclusive" in terminal["error"]["message"]


# -----------------------------------------------------------------------------
# Grammar compile + cache integration with StreamingProcessor
# -----------------------------------------------------------------------------


def _patch_compile(monkeypatch: pytest.MonkeyPatch, fn: Any) -> list[int]:
    """Install ``fn`` as the worker's ``compile_outlines`` and stub the
    tokenizer loader so the cache path can run without a real HF
    config. Returns a call counter (one int per ``compile_outlines``
    invocation appended).

    Also enables ``SIE_GRAMMAR_PREFLIGHT_DEBUG=1`` so tests that
    exercise the legacy worker-side preflight (now off by default per
    ADR-0002) still run the preflight code path.
    """
    # ADR-0002: the worker-side preflight is off by default. Tests that
    # use this helper are explicitly exercising the preflight path, so
    # enable the debug flag for their scope.
    monkeypatch.setenv("SIE_GRAMMAR_PREFLIGHT_DEBUG", "1")

    calls: list[int] = []

    def _wrapped(tok: Any, grammar: Any) -> Any:
        calls.append(1)
        return fn(tok, grammar)

    monkeypatch.setattr("sie_server.processors.streaming.compile_outlines", _wrapped)

    # Stub the per-model tokenizer fetch so tests don't need an
    # ``hf_id``-shaped fixture. ``_ensure_grammar_ready`` only needs an
    # opaque tokenizer-like object — :func:`compile_outlines` is the
    # one that actually uses it, and we're stubbing that too.
    async def _fake_get_tokenizer(self: Any, model_id: str) -> Any:
        return object()

    monkeypatch.setattr(
        StreamingProcessor,
        "_get_tokenizer",
        _fake_get_tokenizer,
    )
    return calls


def _terminal_chunk(nc_mock: AsyncMock) -> dict[str, Any]:
    decoded = _decode_chunks(nc_mock)
    assert decoded, "expected at least one published chunk"
    return decoded[-1]


@pytest.mark.asyncio
async def test_grammar_compile_runs_once_per_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two requests with the same schema → 1 compile + 1 cache hit."""
    script = [
        GenerationChunk(text_delta="ok", is_first=True),
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        ),
    ]
    # Share the registry across both processors so
    # :meth:`_ensure_grammar_ready` hashes the same tokenizer ID
    # (otherwise each MagicMock ``config.hf_id`` would key into a
    # distinct cache slot and the test would observe two compiles for
    # the same schema).
    registry = _make_registry(_FakeGenAdapter(script))
    nc = AsyncMock()
    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")

    calls = _patch_compile(monkeypatch, lambda *_: True)

    grammar_payload = {
        "kind": "json_schema",
        "value": {"type": "object", "properties": {"x": {"type": "integer"}}},
    }
    wi1 = _make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "grammar": grammar_payload})
    msg1 = _make_msg(wi1)
    await proc.process(msg1, "test/model")

    # Second request — same registry → same tokenizer hash → cache hit.
    # Reset the adapter's scripted iterator (it's drained after the
    # first ``process`` call) by re-pointing the registry at a fresh
    # adapter while keeping the same ``config`` MagicMock identity.
    registry.get.return_value = _FakeGenAdapter(script)
    nc2 = AsyncMock()
    proc2 = StreamingProcessor(
        nc=nc2,
        registry=registry,
        worker_id="w1",
        grammar_cache=proc._grammar_cache,  # share the same LRU
    )
    wi2 = _make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "grammar": grammar_payload})
    msg2 = _make_msg(wi2)
    await proc2.process(msg2, "test/model")

    assert len(calls) == 1, f"expected exactly one compile, got {len(calls)}"


@pytest.mark.asyncio
async def test_grammar_compile_timeout_surfaces_grammar_compile_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A compile that exceeds 5s → terminal-error chunk + ACK."""
    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter([])),
        worker_id="w1",
    )

    # Drop the timeout to 0.05s and make the compile sleep longer so
    # the test is fast. The plan's contractual cap is 5s but the
    # mechanism is identical. Patches the module-level shim
    # ``streaming._wait_for`` so the override is scoped to this
    # module — patching ``asyncio.wait_for`` directly would affect
    # every concurrent coroutine.
    from sie_server.processors.streaming import _wait_for as real_wait_for

    # The ``timeout`` kw mirrors :func:`asyncio.wait_for`'s signature
    # — required because the shim replaces it 1:1. The lint rule
    # that flags ``timeout`` parameters on async funcs (ASYNC109)
    # does not apply to test doubles.
    async def _short_wait(coro: Any, timeout: float) -> Any:  # noqa: ASYNC109
        _ = timeout
        return await real_wait_for(coro, timeout=0.05)

    monkeypatch.setattr("sie_server.processors.streaming._wait_for", _short_wait)

    def _slow_compile(_tok: Any, _g: Any) -> Any:
        import time as _time

        _time.sleep(1.0)
        return True

    _patch_compile(monkeypatch, _slow_compile)

    grammar_payload = {"kind": "regex", "value": r"\d+"}
    wi = _make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "grammar": grammar_payload})
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    terminal = _terminal_chunk(nc)
    assert terminal["done"] is True
    assert terminal["error"]["code"] == "grammar_compile_failed"
    msg.ack.assert_awaited()


@pytest.mark.asyncio
async def test_grammar_singleflight_leader_cancel_does_not_hang_followers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix #2 regression: if the single-flight leader is cancelled between
    creating the shared future and resolving it, the future MUST still be
    resolved and ``_grammar_inflight[key]`` popped — otherwise every follower
    awaits the future forever and the process never ACKs (ack_wait redelivery
    storm).
    """
    from sie_server.types.grammar import GrammarSpec

    registry = _make_registry(_FakeGenAdapter([]))
    nc = AsyncMock()
    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")

    compile_entered = asyncio.Event()
    release_compile = threading.Event()  # never set — keeps the leader parked

    def _hang_compile(_tok: Any, _g: Any) -> Any:
        compile_entered.set()
        release_compile.wait(timeout=10.0)
        return True

    _patch_compile(monkeypatch, _hang_compile)

    grammar = GrammarSpec(kind="regex", value=r"\d+")

    async def _call() -> bool:
        return await proc._ensure_grammar_ready(
            grammar,
            model_id="test/model",
            reply_subject="_INBOX.x",
            request_id="req-1",
            attempt_id="att-1",
            msg=_make_msg(_make_work_item()),
        )

    # Leader starts and parks inside the (hanging) compile.
    leader = asyncio.create_task(_call())
    await asyncio.wait_for(compile_entered.wait(), timeout=2.0)

    # A follower latches onto the same in-flight future.
    follower = asyncio.create_task(_call())
    await asyncio.sleep(0.05)  # let the follower register as a follower

    # Cancel the leader mid-compile (between future-create and resolve).
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader

    # The follower MUST complete promptly (not hang). With the fix the leader's
    # finally resolved the future with CancelledError and popped the inflight
    # key, so the follower surfaces a terminal error and returns False.
    result = await asyncio.wait_for(follower, timeout=5.0)
    assert result is False

    # Inflight bookkeeping is clean: no leaked entry to wedge future requests.
    assert proc._grammar_inflight == {}

    # Free the parked executor thread so it doesn't linger.
    release_compile.set()


@pytest.mark.asyncio
async def test_grammar_compile_validation_error_surfaces_with_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outlines-internal failure → ``grammar_compile_failed`` terminal."""
    from sie_server.types.grammar import GrammarValidationError

    def _fail(_tok: Any, _g: Any) -> Any:
        raise GrammarValidationError(
            "bad schema",
            code="grammar_compile_failed",
            param="grammar",
        )

    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter([])),
        worker_id="w1",
    )
    _patch_compile(monkeypatch, _fail)

    wi = _make_work_item(
        generate={
            "prompt": "Hi",
            "max_new_tokens": 8,
            "grammar": {"kind": "json_schema", "value": {"type": "object"}},
        }
    )
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    terminal = _terminal_chunk(nc)
    assert terminal["error"]["code"] == "grammar_compile_failed"
    assert "bad schema" in terminal["error"]["message"]
    msg.ack.assert_awaited()


@pytest.mark.asyncio
async def test_grammar_absent_skips_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default request path: no grammar → no compile, no metric activity."""
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="x", is_first=True),
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        ),
    ]
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter(script)),
        worker_id="w1",
    )
    calls = _patch_compile(monkeypatch, lambda *_: True)

    msg = _make_msg(_make_work_item())  # no grammar field
    await proc.process(msg, "test/model")

    assert calls == []
    terminal = _terminal_chunk(nc)
    assert terminal["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_xgrammar_adapter_skips_outlines_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """XGrammar requests must not enter the Outlines preflight path."""
    nc = AsyncMock()
    adapter = _RecordingGenAdapter()
    adapter._grammar_backend = "xgrammar"
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(adapter),
        worker_id="w1",
    )
    calls = _patch_compile(monkeypatch, lambda *_: True)

    grammar_payload = {"kind": "regex", "value": "blue"}
    msg = _make_msg(_make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "grammar": grammar_payload}))
    await proc.process(msg, "test/model")

    assert calls == []
    assert adapter.captured is not None
    assert adapter.captured["grammar"].kind == "regex"
    terminal = _terminal_chunk(nc)
    assert terminal["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_grammar_concurrent_first_requests_collapse_to_one_compile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two concurrent requests with the same schema → 1 compile + 1 hit.

    Exercises the single-flight path (``_grammar_inflight`` table) that
    collapses thundering-herd cold-start traffic into a single
    Outlines compile.
    """
    script = [
        GenerationChunk(text_delta="ok", is_first=True),
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        ),
    ]
    registry = _make_registry(_FakeGenAdapter(script))

    # Slow the compile so the second request observes the in-flight
    # future instead of a populated cache. Use ``threading.Event``
    # because the compile runs in a worker thread (via the dedicated
    # ``_GRAMMAR_EXECUTOR``), not the asyncio loop.
    compile_event = threading.Event()

    def _slow_compile(_tok: Any, _g: Any) -> Any:
        # Block in the executor thread until the event is set by the
        # test (which it sets immediately after the second request
        # arrives at the single-flight wait).
        compile_event.wait()
        return True

    calls = _patch_compile(monkeypatch, _slow_compile)

    nc1 = AsyncMock()
    proc1 = StreamingProcessor(nc=nc1, registry=registry, worker_id="w1")
    cache = proc1._grammar_cache
    inflight = proc1._grammar_inflight
    inflight_lock = proc1._grammar_inflight_lock
    nc2 = AsyncMock()
    # Share cache + inflight table across both processors (production
    # would be one processor handling many concurrent messages, but
    # the shared state is the relevant invariant).
    proc2 = StreamingProcessor(nc=nc2, registry=registry, worker_id="w1", grammar_cache=cache)
    proc2._grammar_inflight = inflight
    proc2._grammar_inflight_lock = inflight_lock

    grammar_payload = {
        "kind": "json_schema",
        "value": {"type": "object", "properties": {"x": {"type": "integer"}}},
    }

    # Fresh adapter per request because the _FakeGenAdapter iterator
    # is drained after one ``process()`` call.
    async def _run(proc: StreamingProcessor, nc: AsyncMock) -> None:
        registry.get.return_value = _FakeGenAdapter(script)
        wi = _make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "grammar": grammar_payload})
        await proc.process(_make_msg(wi), "test/model")

    task1 = asyncio.create_task(_run(proc1, nc1))
    # Yield so task1 is scheduled and gets into the executor.
    await asyncio.sleep(0.05)
    task2 = asyncio.create_task(_run(proc2, nc2))
    await asyncio.sleep(0.05)
    # Both tasks are now waiting: task1 inside the executor, task2 on
    # the in-flight future. Release the compile.
    compile_event.set()
    await asyncio.gather(task1, task2)

    assert len(calls) == 1, f"expected exactly one compile under single-flight, got {len(calls)}"


class _RecordingGenAdapter(GenerationAdapter):
    """Captures generate() kwargs so tests can assert the worker
    forwarded fields it received.
    """

    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(self) -> None:
        self._device = None
        self.captured: dict[str, Any] | None = None

    def load(self, device: str) -> None:  # pragma: no cover — registry-mocked
        self._device = device

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(inputs=["text"], outputs=["tokens"])

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    async def generate(self, prompt, *, max_new_tokens, **kwargs) -> AsyncIterator[GenerationChunk]:
        self.captured = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }
        yield GenerationChunk(text_delta="ok", done=False)
        yield GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        )


@pytest.mark.asyncio
async def test_penalties_flow_through_to_adapter() -> None:
    """Worker receives ``frequency_penalty`` / ``presence_penalty`` from
    the work envelope and forwards them as adapter ``generate`` kwargs.
    Guards against a regression where the streaming processor adds the
    fields to its dataclass but forgets to thread them into the
    adapter call.
    """
    nc = AsyncMock()
    adapter = _RecordingGenAdapter()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(adapter),
        worker_id="w1",
    )
    wi = _make_work_item(
        generate={
            "prompt": "Hi",
            "max_new_tokens": 8,
            "frequency_penalty": 0.5,
            "presence_penalty": -1.5,
        }
    )
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    assert adapter.captured is not None, "adapter.generate was not invoked"
    assert adapter.captured.get("frequency_penalty") == 0.5
    assert adapter.captured.get("presence_penalty") == -1.5


@pytest.mark.asyncio
async def test_penalties_absent_when_not_provided() -> None:
    """When the request omits penalties the worker does NOT pass
    ``frequency_penalty`` / ``presence_penalty`` to the adapter so the
    adapter's own default (typically 0.0) stays in effect.
    """
    nc = AsyncMock()
    adapter = _RecordingGenAdapter()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(adapter),
        worker_id="w1",
    )
    wi = _make_work_item()
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    assert adapter.captured is not None
    assert "frequency_penalty" not in adapter.captured
    assert "presence_penalty" not in adapter.captured


@pytest.mark.asyncio
async def test_negative_seed_flows_through_to_adapter_unchanged() -> None:
    """The signed gateway wire value must not be normalized to u64 in the worker."""
    nc = AsyncMock()
    adapter = _RecordingGenAdapter()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(adapter),
        worker_id="w1",
    )
    wi = _make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "seed": -1})
    await proc.process(_make_msg(wi), "test/model")

    assert adapter.captured is not None, "adapter.generate was not invoked"
    assert adapter.captured.get("seed") == -1


@pytest.mark.parametrize("seed", [True, "1", 1.5])
@pytest.mark.asyncio
async def test_invalid_seed_is_not_forwarded_to_adapter(seed: object) -> None:
    """Malformed internal work items cannot bypass the gateway's seed bounds."""
    nc = AsyncMock()
    adapter = _RecordingGenAdapter()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(adapter),
        worker_id="w1",
    )
    wi = _make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "seed": seed})
    await proc.process(_make_msg(wi), "test/model")

    assert adapter.captured is not None, "adapter.generate was not invoked"
    assert "seed" not in adapter.captured


@pytest.mark.parametrize("invalid_seed", [1 << 63, (1 << 64) - 1])
@pytest.mark.asyncio
async def test_unsigned_seed_above_i64_is_not_forwarded(invalid_seed: int) -> None:
    nc = AsyncMock()
    adapter = _RecordingGenAdapter()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(adapter),
        worker_id="w1",
    )
    wi = _make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "seed": invalid_seed})
    await proc.process(_make_msg(wi), "test/model")

    assert adapter.captured is not None, "adapter.generate was not invoked"
    assert "seed" not in adapter.captured


@pytest.mark.asyncio
async def test_top_k_and_repetition_penalty_flow_through_to_adapter() -> None:
    """Worker receives ``top_k`` / ``repetition_penalty`` from the work
    envelope and forwards them as adapter ``generate`` kwargs. Guards
    against the regression where the fields are added to the dataclass
    but not threaded into the adapter call.
    """
    nc = AsyncMock()
    adapter = _RecordingGenAdapter()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(adapter),
        worker_id="w1",
    )
    wi = _make_work_item(
        generate={
            "prompt": "Hi",
            "max_new_tokens": 8,
            "top_k": 10,
            "repetition_penalty": 1.1,
        }
    )
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    assert adapter.captured is not None, "adapter.generate was not invoked"
    assert adapter.captured.get("top_k") == 10
    assert adapter.captured.get("repetition_penalty") == 1.1


@pytest.mark.asyncio
async def test_top_k_and_repetition_penalty_absent_when_not_provided() -> None:
    """When the request omits ``top_k`` / ``repetition_penalty`` the
    worker does NOT pass them to the adapter so the model/sampler default
    stays in effect.
    """
    nc = AsyncMock()
    adapter = _RecordingGenAdapter()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(adapter),
        worker_id="w1",
    )
    wi = _make_work_item()
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    assert adapter.captured is not None
    assert "top_k" not in adapter.captured
    assert "repetition_penalty" not in adapter.captured


def test_encode_chunk_includes_logprobs() -> None:
    """Per-token logprobs ride on the wire chunk envelope when provided,
    and are omitted otherwise so non-logprobs requests pay nothing extra.
    """
    from sie_server.processors.streaming import _encode_chunk

    lps = [{"token": "Hi", "logprob": -0.5, "bytes": [72, 105], "top_logprobs": []}]
    payload = _encode_chunk(
        kind="chunk",
        request_id="r",
        attempt_id="a",
        seq=0,
        text_delta="Hi",
        done=False,
        logprobs=lps,
    )
    decoded = msgpack.unpackb(payload, raw=False)
    assert decoded["logprobs"][0]["token"] == "Hi"  # noqa: S105 — OpenAI logprob field name, not a secret
    assert decoded["logprobs"][0]["logprob"] == -0.5

    omitted = msgpack.unpackb(
        _encode_chunk(kind="chunk", request_id="r", attempt_id="a", seq=0, text_delta="Hi", done=False),
        raw=False,
    )
    assert "logprobs" not in omitted


@pytest.mark.asyncio
async def test_grammar_malformed_payload_surfaces_invalid_request() -> None:
    """Worker-side guard: ``grammar.kind`` outside the allowed set
    surfaces an ``invalid_request`` terminal even though the gateway
    should have caught it. The worker is defence-in-depth, not the
    primary filter.

    Allowed kinds are ``json_schema``, ``regex``, and ``ebnf`` — any
    other discriminator (here a fabricated ``cfg-future``) trips the
    defence-in-depth check.
    """
    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter([])),
        worker_id="w1",
    )
    wi = _make_work_item(
        generate={
            "prompt": "Hi",
            "max_new_tokens": 8,
            "grammar": {"kind": "cfg-future", "value": "..."},
        }
    )
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")
    terminal = _terminal_chunk(nc)
    assert terminal["error"]["code"] == "invalid_request"
    msg.ack.assert_awaited()


@pytest.mark.asyncio
async def test_ebnf_on_outlines_backend_not_rejected_at_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """EBNF on an Outlines-backed generation model must NOT be rejected at
    the worker-side preflight.

    The adapter forwards ``ebnf`` to SGLang on both backends; the bundled
    Outlines surface has no EBNF factory. Before the fix the preflight
    raised ``grammar_invalid`` and the request died with an error
    terminal. After the fix the request streams normally and the raw EBNF
    is forwarded to the adapter.
    """
    nc = AsyncMock()
    adapter = _RecordingGenAdapter()
    # Default backend is "outlines" — the preflight path runs for this
    # adapter (``_adapter_uses_outlines_grammar`` is True).
    assert getattr(adapter, "_grammar_backend", "outlines") == "outlines"
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(adapter),
        worker_id="w1",
    )

    # The real ``compile_outlines`` is exercised — but with the fix it
    # short-circuits for EBNF before resolving any Outlines factory, so we
    # don't need to stub it. Stub only the tokenizer fetch.
    async def _fake_get_tokenizer(self: Any, model_id: str) -> Any:
        return object()

    monkeypatch.setattr(StreamingProcessor, "_get_tokenizer", _fake_get_tokenizer)

    grammar_payload = {"kind": "ebnf", "value": 'root ::= "yes" | "no"'}
    msg = _make_msg(_make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "grammar": grammar_payload}))
    await proc.process(msg, "test/model")

    # Not rejected: the adapter was invoked and the stream ended cleanly.
    assert adapter.captured is not None, "EBNF request was rejected before reaching the adapter"
    assert adapter.captured["grammar"].kind == "ebnf"
    terminal = _terminal_chunk(nc)
    assert terminal["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_cancel_during_grammar_compile_window_is_honored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancel arriving during the grammar-compile/admission window (before
    the stream starts) is honored — the request does not run to completion.

    Regression for the cancel-handle being registered too late (only after
    compile + admission). With the handle registered right after
    ``request_id`` is known, ``signal_cancel`` flips the event during the
    slow compile so the stream tears down with ``finish_reason=cancelled``.
    """
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="should-not-fully-stream", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    ]
    registry = _make_registry(_FakeGenAdapter(script))
    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")

    async def _fake_get_tokenizer(self: Any, model_id: str) -> Any:
        return object()

    monkeypatch.setattr(StreamingProcessor, "_get_tokenizer", _fake_get_tokenizer)

    compile_started = asyncio.Event()
    compile_release = threading.Event()

    def _slow_compile(_tok: Any, _g: Any) -> Any:
        # Runs in the grammar executor thread. Signal the loop that we're
        # in the compile window, then block until released.
        compile_started.set()
        compile_release.wait()
        return True

    _patch_compile(monkeypatch, _slow_compile)

    grammar_payload = {
        "kind": "json_schema",
        "value": {"type": "object", "properties": {"x": {"type": "integer"}}},
    }
    msg = _make_msg(_make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "grammar": grammar_payload}))

    async def _cancel_during_compile() -> None:
        # Wait until the compile is in flight, then cancel and release.
        for _ in range(200):
            if compile_started.is_set():
                break
            await asyncio.sleep(0.01)
        # The handle must already be registered even though the stream
        # hasn't started — this is the property under test.
        assert proc.signal_cancel("req-1") is True, "cancel handle not registered during compile window"
        compile_release.set()

    cancel_task = asyncio.create_task(_cancel_during_compile())
    await proc.process(msg, "test/model")
    await cancel_task

    terminal = _terminal_chunk(nc)
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "cancelled"
    msg.ack.assert_awaited()


@pytest.mark.asyncio
async def test_terminal_error_publish_failure_naks_not_acks(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the terminal-error chunk fails to publish, the work item is
    NAKed (for redelivery), NOT ACKed (which would orphan the request).
    """
    telemetry = _capture_worker_completion(monkeypatch)
    nc = AsyncMock()
    nc.publish.side_effect = RuntimeError("nats down")
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter([])),
        worker_id="w1",
    )
    # A malformed grammar kind takes the pre-stream terminal-error path,
    # which now settles via ACK-on-success / NAK-on-failure.
    wi = _make_work_item(
        generate={
            "prompt": "Hi",
            "max_new_tokens": 8,
            "grammar": {"kind": "cfg-future", "value": "..."},
        }
    )
    msg = _make_msg(wi)
    await proc.process(msg, "test/model")

    # Publish failed → NAK for redelivery, no ACK.
    msg.nak.assert_awaited()
    msg.ack.assert_not_awaited()
    telemetry.item_completed.assert_not_called()


@pytest.mark.asyncio
async def test_pre_stream_error_terminal_records_one_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = _capture_worker_completion(monkeypatch)
    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter([])),
        worker_id="w1",
    )
    wi = _make_work_item(
        generate={
            "prompt": "Hi",
            "max_new_tokens": 8,
            "grammar": {"kind": "cfg-future", "value": "..."},
        }
    )
    msg = _make_msg(wi)

    await proc.process(msg, "test/model")

    terminal = _terminal_chunk(nc)
    assert terminal["finish_reason"] == "error"
    msg.ack.assert_awaited_once()
    _assert_one_generation_completion(telemetry, outcome="error")


@pytest.mark.asyncio
async def test_external_completion_owner_suppresses_processor_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry = _capture_worker_completion(monkeypatch)
    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter([])),
        worker_id="w1",
        record_worker_completion=False,
    )
    wi = _make_work_item(
        generate={
            "prompt": "Hi",
            "max_new_tokens": 8,
            "grammar": {"kind": "cfg-future", "value": "..."},
        }
    )

    await proc.process(_make_msg(wi), "test/model")

    assert _terminal_chunk(nc)["finish_reason"] == "error"
    telemetry.item_completed.assert_not_called()


@pytest.mark.asyncio
async def test_grammar_ready_ebnf_does_not_satisfy_same_value_regex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BUG 4 regression (end-to-end at the cache layer): readying an ebnf
    grammar must NOT mark a same-``value`` regex grammar as ready. Each kind
    runs its own Outlines preflight (one compile per kind), so a malformed
    regex can't slip through on an ebnf's cache hit.
    """
    from sie_server.types.grammar import GrammarSpec

    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter([])),
        worker_id="w1",
    )
    calls = _patch_compile(monkeypatch, lambda *_: True)

    ebnf = GrammarSpec(kind="ebnf", value="[a-z]+")
    regex = GrammarSpec(kind="regex", value="[a-z]+")

    common = {
        "model_id": "test/model",
        "reply_subject": "_INBOX.router-1.req-1",
        "request_id": "req-1",
        "attempt_id": "att-1",
        "msg": AsyncMock(),
    }

    ready_ebnf = await proc._ensure_grammar_ready(ebnf, **common)  # type: ignore[arg-type]
    assert ready_ebnf is True
    assert len(calls) == 1, "ebnf should have compiled once"

    # Same value, different kind: must NOT hit the ebnf cache entry — its own
    # compile must run, so the counter advances to 2.
    ready_regex = await proc._ensure_grammar_ready(regex, **common)  # type: ignore[arg-type]
    assert ready_regex is True
    assert len(calls) == 2, "regex must run its own preflight, not reuse ebnf's cache entry"


@pytest.mark.asyncio
async def test_grammar_hash_typeerror_surfaces_grammar_invalid() -> None:
    """``hash_grammar`` raising ``TypeError`` inside ``_ensure_grammar_ready``
    (e.g. a GrammarSpec whose value type mismatches its kind, built outside
    the validated request path — forcing-grammar / prewarm / future code)
    surfaces a clean ``grammar_invalid`` terminal + ACK instead of an
    unhandled exception that left the request hung with no terminal and no
    ACK → JetStream redeliver loop.

    Exercised directly against ``_ensure_grammar_ready`` because the
    request-shape validation guards value-vs-kind before the stream starts,
    so the TypeError is only reachable from un-validated GrammarSpec
    sources. The fix is defence-in-depth at the cache-key computation.
    """
    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter([])),
        worker_id="w1",
    )

    from sie_server.types.grammar import GrammarSpec

    # kind=json_schema but value is a string — ``hash_grammar`` raises
    # TypeError when it tries to ``json.dumps`` a non-dict for json_schema.
    bad = GrammarSpec(kind="json_schema", value="not-a-dict")  # type: ignore[arg-type]
    msg = AsyncMock()

    ready = await proc._ensure_grammar_ready(
        bad,
        model_id="test/model",
        reply_subject="_INBOX.router-1.req-1",
        request_id="req-1",
        attempt_id="att-1",
        msg=msg,
    )

    assert ready is False
    terminal = _terminal_chunk(nc)
    assert terminal["done"] is True
    assert terminal["error"]["code"] == "grammar_invalid"
    msg.ack.assert_awaited()


# ── Tool calling (streaming wiring) ────────────────────────────────


def _tool_call_deltas(decoded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the OpenAI ``delta.tool_calls`` entries across chunks."""
    out: list[dict[str, Any]] = []
    for c in decoded:
        for tc in c.get("tool_calls", []) or []:
            out.append(tc)
    return out


_WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]


@pytest.mark.asyncio
async def test_streaming_tool_call_emits_incremental_deltas() -> None:
    """A <tool_call> block in the model output surfaces as well-formed
    incremental OpenAI ``delta.tool_calls`` (id+name first, args after).
    """
    nc = AsyncMock()
    script = [
        GenerationChunk(
            text_delta='<tool_call>{"name":"get_weather","arguments":{"city":"Tokyo"}}</tool_call>',
            is_first=True,
        ),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=5, completion_tokens=9),
    ]
    proc = StreamingProcessor(nc=nc, registry=_make_registry(_FakeGenAdapter(script)), worker_id="w1")
    wi = _make_work_item(generate={"prompt": "weather?", "max_new_tokens": 64, "tools": _WEATHER_TOOLS})
    await proc.process(_make_msg(wi), "test/model")

    decoded = _decode_chunks(nc)
    tcs = _tool_call_deltas(decoded)
    assert len(tcs) == 2
    # Announcement delta: id + function name, empty arguments.
    assert tcs[0]["index"] == 0
    assert tcs[0]["id"].startswith("call_")
    assert tcs[0]["function"]["name"] == "get_weather"
    assert tcs[0]["function"]["arguments"] == ""
    # Body delta: arguments fragment, no name.
    assert tcs[1]["index"] == 0
    assert tcs[1]["function"]["arguments"] == '{"city":"Tokyo"}'
    # Terminal reason flips to tool_calls.
    assert decoded[-1]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_streaming_tool_choice_required_forwards_forcing_grammar() -> None:
    """tool_choice='required' builds a regex forcing grammar and forwards
    it to the adapter's ``grammar`` kwarg.
    """
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="<tool_call><function=get_weather></function></tool_call>", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    ]
    adapter = _FakeGenAdapter(script)
    # Pretend the adapter uses the xgrammar backend so the worker skips
    # the Outlines preflight (which would need a real tokenizer).
    adapter._grammar_backend = "xgrammar"  # type: ignore[attr-defined]

    captured: dict[str, Any] = {}
    original = adapter.generate

    async def _capture(prompt, *, max_new_tokens, temperature=1.0, top_p=1.0, stop=None, **kwargs):
        captured.update(kwargs)
        async for chunk in original(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, stop=stop
        ):
            yield chunk

    adapter.generate = _capture  # type: ignore[method-assign]

    registry = _make_registry(adapter)
    # The forcing grammar is only built when the model's on-wire tool-call
    # format resolves confidently (fix #7). Configure the resolved profile so
    # ``_resolve_tool_call_format`` returns ``qwen_xml``.
    resolved = MagicMock()
    resolved.loadtime = {"tool_call_parser": "qwen3_coder"}
    registry.get_config.return_value.resolve_profile.return_value = resolved

    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    wi = _make_work_item(
        generate={
            "prompt": "weather?",
            "max_new_tokens": 64,
            "tools": _WEATHER_TOOLS,
            "tool_choice": "required",
        }
    )
    await proc.process(_make_msg(wi), "test/model")

    grammar = captured.get("grammar")
    assert grammar is not None
    assert grammar.kind == "regex"
    assert "get_weather" in grammar.value


@pytest.mark.asyncio
async def test_streaming_tool_choice_none_hides_tools_from_chat_template(monkeypatch) -> None:
    """``tool_choice='none'`` must prevent the model from ever seeing the
    tool catalogue.

    Per the OpenAI contract, "none" means the model is forbidden from
    calling tools — the worker enforces this at the *prompt* layer by
    passing ``tools=None`` to ``apply_chat_template`` regardless of what
    the caller put in ``params.tools``. With the tool definitions absent
    from the rendered prompt, the model has nothing to call, so no
    ``<tool_call>`` syntax can leak into assistant content.

    Replaces the prior parser-disabled behavior, which kept the tool
    definitions visible to the model and let raw ``<tool_call>`` blocks
    surface as plain text.
    """
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="ok", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    adapter = _FakeGenAdapter(script)
    registry = _make_registry_with_chat_config(adapter)

    seen_kwargs: dict[str, Any] = {}

    class _StubTok:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
            seen_kwargs.update(kwargs)
            return "<rendered>" + "".join(m["content"] for m in messages) + "</rendered>"

        def encode(self, text, *, add_special_tokens):
            return [0] * len(text.split())

    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "load_tokenizer", lambda *a, **kw: _StubTok())

    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    wi = _make_work_item(
        generate={
            "messages": [{"role": "user", "content": "weather?"}],
            "max_new_tokens": 64,
            "tools": _WEATHER_TOOLS,
            "tool_choice": "none",
        }
    )
    await proc.process(_make_msg(wi), "test/model")

    # The tool catalogue must not have been forwarded to the chat template.
    # Either the key was omitted entirely, or it was passed as None / empty.
    assert not seen_kwargs.get("tools"), (
        f"tools must be hidden from the chat template when tool_choice='none'; got {seen_kwargs.get('tools')!r}"
    )

    # No tool-call deltas surface, no raw ``<tool_call>`` syntax leaks
    # into assistant content, and the stream finishes as ``stop``.
    decoded = _decode_chunks(nc)
    assert _tool_call_deltas(decoded) == []
    body = "".join(c.get("text_delta", "") for c in decoded)
    assert "<tool_call>" not in body
    assert decoded[-1]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_streaming_tool_choice_named_unknown_function_rejected() -> None:
    """A named tool_choice for a function absent from tools → invalid_request."""
    nc = AsyncMock()
    proc = StreamingProcessor(nc=nc, registry=_make_registry(_FakeGenAdapter([])), worker_id="w1")
    wi = _make_work_item(
        generate={
            "prompt": "weather?",
            "max_new_tokens": 64,
            "tools": _WEATHER_TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "not_a_tool"}},
        }
    )
    await proc.process(_make_msg(wi), "test/model")

    terminal = _terminal_chunk(nc)
    assert terminal["error"]["code"] == "invalid_request"
    assert "not_a_tool" in terminal["error"]["message"]


# ---------------------------------------------------------------------------
# ADR-0002 — SGLang owns request-time grammar compilation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grammar_preflight_default_does_not_call_compile_outlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0002: with ``SIE_GRAMMAR_PREFLIGHT_DEBUG`` unset the default
    structured-output path MUST NOT invoke ``compile_outlines``. The raw
    schema is forwarded straight to SGLang.
    """
    # Make sure the env flag is OFF for this test (a parent process /
    # other test fixture could have set it).
    monkeypatch.delenv("SIE_GRAMMAR_PREFLIGHT_DEBUG", raising=False)

    calls: list[int] = []

    def _spy(_tok: Any, _g: Any) -> Any:
        calls.append(1)
        return True

    monkeypatch.setattr("sie_server.processors.streaming.compile_outlines", _spy)

    # Stub the tokenizer fetch in case the env flag is accidentally on
    # somewhere — we still don't want the test crashing on a missing HF id.
    async def _fake_get_tokenizer(self: Any, model_id: str) -> Any:
        return object()

    monkeypatch.setattr(StreamingProcessor, "_get_tokenizer", _fake_get_tokenizer)

    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="ok", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter(script)),
        worker_id="w1",
    )

    grammar_payload = {
        "kind": "json_schema",
        "value": {"type": "object", "properties": {"x": {"type": "integer"}}},
    }
    wi = _make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "grammar": grammar_payload})
    await proc.process(_make_msg(wi), "test/model")

    assert calls == [], "Default path (SIE_GRAMMAR_PREFLIGHT_DEBUG unset) must NOT invoke compile_outlines per ADR-0002"
    terminal = _terminal_chunk(nc)
    assert terminal["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_grammar_preflight_enabled_via_env_calls_compile_outlines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0002: ``SIE_GRAMMAR_PREFLIGHT_DEBUG=1`` re-enables the legacy
    worker-side Outlines preflight. The preflight code path runs and
    ``compile_outlines`` is invoked exactly once for a cache-miss
    request.
    """
    monkeypatch.setenv("SIE_GRAMMAR_PREFLIGHT_DEBUG", "1")

    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="ok", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter(script)),
        worker_id="w1",
    )

    # ``_patch_compile`` also sets the env var; calling it after our
    # explicit setenv is harmless — both end up with "1".
    calls = _patch_compile(monkeypatch, lambda *_: True)

    grammar_payload = {
        "kind": "json_schema",
        "value": {"type": "object", "properties": {"x": {"type": "integer"}}},
    }
    wi = _make_work_item(generate={"prompt": "Hi", "max_new_tokens": 8, "grammar": grammar_payload})
    await proc.process(_make_msg(wi), "test/model")

    assert len(calls) == 1, f"expected one preflight compile, got {len(calls)}"


@pytest.mark.asyncio
async def test_grammar_malformed_payload_still_rejected_without_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0002 regression: removing the worker preflight from the hot
    path MUST NOT weaken validation. Cheap shape checks (``kind`` must be
    one of the allowed strings) still reject a malformed grammar at the
    worker before any inference cost is spent.
    """
    monkeypatch.delenv("SIE_GRAMMAR_PREFLIGHT_DEBUG", raising=False)

    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter([])),
        worker_id="w1",
    )
    # ``cfg-future`` is outside the allowed ``{json_schema, regex, ebnf}``
    # set — must surface ``invalid_request`` regardless of the preflight
    # debug flag.
    wi = _make_work_item(
        generate={
            "prompt": "Hi",
            "max_new_tokens": 8,
            "grammar": {"kind": "cfg-future", "value": "..."},
        }
    )
    await proc.process(_make_msg(wi), "test/model")
    terminal = _terminal_chunk(nc)
    assert terminal["error"]["code"] == "invalid_request"


def test_generation_diagnostics_are_noop_until_meter_activation() -> None:
    """Typed generation events remain harmless while telemetry is disabled."""
    from sie_server.observability import worker_telemetry as obs_metrics

    telemetry = obs_metrics.worker_telemetry()
    telemetry.grammar_requested(model="test/model", backend="outlines", grammar="json_schema")
    telemetry.grammar_cache_lookup(
        model="test/model",
        backend="outlines",
        grammar="json_schema",
        phase="request",
        result="miss",
    )
    telemetry.grammar_compile_completed(
        model="test/model",
        backend="outlines",
        grammar="json_schema",
        phase="request",
        outcome="success",
        duration_s=0.01,
    )


# ---------------------------------------------------------------------------
# H6 regression: no-silent-drop on chunk-queue backpressure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_n2_queue_full_emits_transport_failure_and_withholds_ack(
    monkeypatch,
) -> None:
    """H6: per-choice (n>1) delta enqueue failure → transport_failure + no ACK.

    Stall the publisher briefly so the bounded-await ``chunk_queue.put``
    times out on a per-choice delta. The worker must publish a
    ``transport_failure`` terminal at the un-advanced ``seq`` and MUST
    NOT ACK the JetStream message (so redelivery can recover).
    """
    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "_CHUNK_QUEUE_MAX", 1)
    # Short put timeout so the producer trips quickly during the stall.
    monkeypatch.setattr(streaming_mod, "_CHUNK_PUT_TIMEOUT_S", 0.02)

    real_publisher_loop = StreamingProcessor._publisher_loop

    async def _slow_publisher_loop(self, chunk_queue, reply_subject):
        # Pull one chunk, publish it, then stall ~250ms before falling
        # through to the real loop. During the stall the producer fills
        # the queue (size 1) and then times out trying to enqueue the
        # next per-choice delta — the H6 path. After the stall the
        # publisher drains the transport_failure terminal the producer
        # enqueued in its place.
        item = await chunk_queue.get()
        if item is not None:
            payload, _ = item
            await self._nc.publish(reply_subject, payload)
        await asyncio.sleep(0.25)
        return await real_publisher_loop(self, chunk_queue, reply_subject)

    monkeypatch.setattr(StreamingProcessor, "_publisher_loop", _slow_publisher_loop)

    nc = AsyncMock()

    # Streaming n=2: each delta is its own wire chunk (no coalescing).
    deltas = [
        GenerationChunk(text_delta="a", choice_index=0, is_first=True),
        GenerationChunk(text_delta="b", choice_index=1),
        GenerationChunk(text_delta="c", choice_index=0),
        GenerationChunk(text_delta="d", choice_index=1),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    adapter = _FakeGenAdapter(deltas)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    msg = _make_msg(_make_work_item(generate={"prompt": "hi", "max_new_tokens": 8, "n": 2, "stream": True}))

    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    terminals = [c for c in decoded if c.get("done") is True]
    assert terminals, "expected at least one terminal chunk on the wire"
    assert any(t.get("error", {}).get("code") == "transport_failure" for t in terminals), (
        f"expected transport_failure terminal, got: {[t.get('error') for t in terminals]}"
    )

    # H6: ACK withheld even though transport_failure terminal published.
    msg.ack.assert_not_awaited()

    # H6 invariant: seqs are strictly monotonic with no gaps.
    seqs = [c["seq"] for c in decoded]
    assert seqs == sorted(seqs), f"non-monotonic seqs: {seqs}"
    assert len(seqs) == len(set(seqs)), f"duplicate seqs: {seqs}"
    # transport_failure terminal occupies the contiguous next seq —
    # seq did not advance past the failed per-choice enqueue.
    delta_seqs = [c["seq"] for c in decoded if not c.get("done")]
    failure_terminal = next(t for t in terminals if t.get("error", {}).get("code") == "transport_failure")
    if delta_seqs:
        assert failure_terminal["seq"] == max(delta_seqs) + 1, (
            f"transport_failure seq {failure_terminal['seq']} not contiguous with last delta seq {max(delta_seqs)}"
        )


@pytest.mark.asyncio
async def test_streaming_terminal_pending_text_flush_failure_withholds_ack(
    monkeypatch,
) -> None:
    """H6: terminal-path pending_text flush failure → transport_failure + no ACK.

    If ``_flush_pending`` returns False at the ``chunk.done`` branch
    (publisher stalled, queue full), the worker must NOT build a
    ``stop`` terminal on top of the dropped text. It must publish a
    ``transport_failure`` terminal at the un-advanced seq and withhold
    the ACK.
    """
    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "_CHUNK_QUEUE_MAX", 1)
    monkeypatch.setattr(streaming_mod, "_CHUNK_PUT_TIMEOUT_S", 0.02)

    real_publisher_loop = StreamingProcessor._publisher_loop

    async def _slow_publisher_loop(self, chunk_queue, reply_subject):
        # Pull and publish one chunk, then stall ~250ms before falling
        # through to the real loop. During the stall the queue is full
        # and the terminal-time pending-text flush returns False.
        item = await chunk_queue.get()
        if item is not None:
            payload, _ = item
            await self._nc.publish(reply_subject, payload)
        await asyncio.sleep(0.25)
        return await real_publisher_loop(self, chunk_queue, reply_subject)

    monkeypatch.setattr(StreamingProcessor, "_publisher_loop", _slow_publisher_loop)

    nc = AsyncMock()
    # Two non-empty deltas then a terminal carrying trailing text. The
    # first flush succeeds, then the queue fills with the second flush.
    # The terminal-time flush of pending_text returns False.
    deltas = [
        GenerationChunk(text_delta="x" * 40, is_first=True),  # seq 0 → published
        GenerationChunk(text_delta="y" * 40),  # seq 1 → fills queue
        GenerationChunk(
            text_delta="z" * 40,  # trailing text the terminal will try to flush
            done=True,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=3,
        ),
    ]
    adapter = _FakeGenAdapter(deltas)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    msg = _make_msg(_make_work_item())

    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    terminals = [c for c in decoded if c.get("done") is True]
    assert terminals
    # The terminal must be transport_failure, not a normal "stop".
    last = terminals[-1]
    assert last.get("finish_reason") == "error"
    assert last.get("error", {}).get("code") == "transport_failure"

    # ACK withheld per H6.
    msg.ack.assert_not_awaited()


@pytest.mark.asyncio
async def test_seq_does_not_advance_on_failed_enqueue(monkeypatch) -> None:
    """H6: seq must not advance when chunk_queue.put fails (bounded-await timeout).

    Direct invariant: after a failed flush, the next successfully
    enqueued chunk MUST reuse the same seq the failed chunk would have
    consumed. Equivalently: the transport_failure terminal sits at the
    contiguous next-seq, never at next+1.
    """
    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "_CHUNK_QUEUE_MAX", 1)
    monkeypatch.setattr(streaming_mod, "_CHUNK_PUT_TIMEOUT_S", 0.02)

    real_publisher_loop = StreamingProcessor._publisher_loop

    async def _slow_publisher_loop(self, chunk_queue, reply_subject):
        # Drain one chunk (seq 0), stall, then hand off to the real loop
        # so the transport_failure terminal can drain after the producer
        # exits with a contiguous seq.
        item = await chunk_queue.get()
        if item is not None:
            payload, _ = item
            await self._nc.publish(reply_subject, payload)
        await asyncio.sleep(0.25)
        return await real_publisher_loop(self, chunk_queue, reply_subject)

    monkeypatch.setattr(StreamingProcessor, "_publisher_loop", _slow_publisher_loop)

    nc = AsyncMock()
    deltas = [
        GenerationChunk(text_delta="a" * 40, is_first=True),  # seq 0 → wire
        GenerationChunk(text_delta="b" * 40),  # seq 1 → fills queue
        GenerationChunk(text_delta="c" * 40),  # would be seq 2 but enqueue may time out
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=3,
        ),
    ]
    adapter = _FakeGenAdapter(deltas)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    msg = _make_msg(_make_work_item())

    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    seqs = [c["seq"] for c in decoded]
    # Strict invariant: no duplicates and no gaps in published seqs.
    # The transport_failure terminal must occupy the un-advanced next
    # seq, never skip ahead past the chunk that failed to enqueue.
    assert len(seqs) == len(set(seqs)), f"duplicate seqs: {seqs}"
    assert seqs == sorted(seqs), f"non-monotonic seqs: {seqs}"
    # Contiguous from 0 (no gap left by a failed enqueue that advanced seq).
    assert seqs == list(range(min(seqs), min(seqs) + len(seqs))), (
        f"non-contiguous seqs (gap suggests seq advanced on a failed enqueue): {seqs}"
    )


# -----------------------------------------------------------------------------
# H9 — Cancel tombstone (first-chunk-fallback double-execution defence)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_before_register_refuses_decode() -> None:
    """H9: cancel arrives before any decode attempt registers; the next
    decode attempt for the same request_id is refused via the tombstone.

    Simulates the gateway's first-chunk-fallback race: the direct-dispatched
    worker hasn't yet picked the message off JetStream when the gateway
    fires a cancel signal. Today ``signal_cancel`` would no-op (no live
    attempt); the tombstone makes it leave a marker so the eventual decode
    refuses to run and emits a ``transport_failure`` terminal instead.
    """
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="should-not-decode", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop"),
    ]
    adapter = _FakeGenAdapter(script)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")

    # Cancel BEFORE any process() call → no in-flight attempt → tombstone written.
    matched = proc.signal_cancel("req-1")
    assert matched is False  # no live attempt to signal
    assert "req-1" in proc._cancel_tombstones

    msg = _make_msg(_make_work_item())
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    # One terminal, transport_failure, done=true. No adapter output.
    assert len(decoded) == 1, f"expected only the tombstone terminal; got {decoded}"
    terminal = decoded[0]
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "error"
    assert terminal["error"]["code"] == "transport_failure"
    # ACK fires: JetStream must not redeliver indefinitely.
    msg.ack.assert_awaited()
    # Tombstone consumed on hit.
    assert "req-1" not in proc._cancel_tombstones


@pytest.mark.asyncio
async def test_tombstone_expires_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    """H9: an expired tombstone does NOT block a legitimate later request.

    Drives the tombstone-TTL clock forwards by monkey-patching
    ``time.monotonic`` inside the streaming module. After expiry the next
    decode for the same request_id runs normally.
    """
    from sie_server.processors import streaming as streaming_mod

    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="ok", is_first=True),
        GenerationChunk(
            text_delta="",
            done=True,
            finish_reason="stop",
            prompt_tokens=1,
            completion_tokens=1,
        ),
    ]
    adapter = _FakeGenAdapter(script)
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")

    # Drop a tombstone via signal_cancel.
    proc.signal_cancel("req-1")
    assert "req-1" in proc._cancel_tombstones

    # Advance monotonic clock past the TTL.
    real_monotonic = streaming_mod.time.monotonic
    base = real_monotonic()
    offset = streaming_mod._CANCEL_TOMBSTONE_TTL_S + 1.0

    def _bumped_monotonic() -> float:
        return real_monotonic() + offset

    monkeypatch.setattr(streaming_mod.time, "monotonic", _bumped_monotonic)
    _ = base  # silence unused

    msg = _make_msg(_make_work_item())
    await proc.process(msg, "test/model")

    decoded = _decode_chunks(nc)
    terminal = decoded[-1]
    # Normal completion — not blocked by the expired tombstone.
    assert terminal["done"] is True
    assert terminal["finish_reason"] == "stop"
    msg.ack.assert_awaited()


@pytest.mark.asyncio
async def test_tombstone_lazy_cleanup_evicts_when_over_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """H9: the tombstone map sweeps + evicts when it exceeds the soft cap.

    Shrinks the cap to a tiny value, fills with expired + live entries, then
    adds one more — the expired entries must be swept and the surviving
    eldest evicted so the map stays bounded.
    """
    from sie_server.processors import streaming as streaming_mod

    monkeypatch.setattr(streaming_mod, "_CANCEL_TOMBSTONE_MAX", 4)

    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_FakeGenAdapter([])),
        worker_id="w1",
    )

    # Fill with 4 live tombstones.
    for i in range(4):
        proc.signal_cancel(f"req-live-{i}")
    assert len(proc._cancel_tombstones) == 4

    # Force two of them to be expired by rewriting their deadlines.
    proc._cancel_tombstones["req-live-0"] = 0.0
    proc._cancel_tombstones["req-live-1"] = 0.0

    # Now insert a 5th: cleanup should reclaim the expired pair, leaving 3
    # live + the new one = 4.
    proc.signal_cancel("req-new")
    assert "req-new" in proc._cancel_tombstones
    assert "req-live-0" not in proc._cancel_tombstones
    assert "req-live-1" not in proc._cancel_tombstones
    assert len(proc._cancel_tombstones) <= 4


# -----------------------------------------------------------------------------
# Vision input (#1233): image content parsing + plumbing to the adapter
# -----------------------------------------------------------------------------


def _png_data_uri(payload: bytes = b"PNGDATA") -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def test_decode_data_uri_image_parses_bytes_and_format() -> None:
    from sie_server.processors.streaming import _decode_data_uri_image

    data, fmt = _decode_data_uri_image(_png_data_uri(b"hello-png"))
    assert data == b"hello-png"
    assert fmt == "png"


def test_decode_data_uri_image_rejects_remote_and_malformed() -> None:
    from sie_server.processors.streaming import _decode_data_uri_image

    with pytest.raises(ValueError, match="inline base64 'data:' URI"):
        _decode_data_uri_image("https://example.com/cat.png")
    with pytest.raises(ValueError, match="base64-encoded"):
        # data URI without ;base64
        _decode_data_uri_image("data:image/png,rawbytes")
    with pytest.raises(ValueError, match="invalid base64"):
        _decode_data_uri_image("data:image/png;base64,!!!notbase64!!!")
    with pytest.raises(ValueError, match="empty bytes"):
        _decode_data_uri_image("data:image/png;base64,")
    # Non-image media types reject (mirrors the gateway's image/* check).
    with pytest.raises(ValueError, match="media type"):
        _decode_data_uri_image("data:text/plain;base64," + base64.b64encode(b"hi").decode())
    with pytest.raises(ValueError, match="media type"):
        _decode_data_uri_image("data:;base64," + base64.b64encode(b"hi").decode())


def test_parse_chat_content_accepts_input_text() -> None:
    # ``input_text`` is accepted alongside ``text`` (gateway/OpenAPI accept both).
    from sie_server.processors.streaming import _parse_chat_content

    text, images, parts = _parse_chat_content([{"type": "input_text", "text": "hi"}], 0)
    assert text == "hi"
    assert images == ()
    assert parts is None  # no image → no ordered layout (renders from ``content``)


def test_parse_chat_content_string_passthrough() -> None:
    from sie_server.processors.streaming import _parse_chat_content

    text, images, parts = _parse_chat_content("plain text", 0)
    assert text == "plain text"
    assert images == ()
    assert parts is None


def test_parse_chat_content_extracts_text_and_images_in_order() -> None:
    from sie_server.processors.streaming import _parse_chat_content

    content = [
        {"type": "text", "text": "describe "},
        {"type": "image_url", "image_url": {"url": _png_data_uri(b"img1")}},
        {"type": "text", "text": "and this"},
        # ``input_image`` alias + bare-string url form.
        {"type": "input_image", "image_url": _png_data_uri(b"img2")},
    ]
    text, images, parts = _parse_chat_content(content, 0)
    assert text == "describe and this"
    assert [img["data"] for img in images] == [b"img1", b"img2"]
    # #1294: the ordered layout preserves the text↔image interleaving.
    assert parts is not None
    assert [p["type"] for p in parts] == ["text", "image", "text", "image"]
    assert parts[0] == {"type": "text", "text": "describe "}
    assert parts[2] == {"type": "text", "text": "and this"}


def test_parse_chat_content_rejects_unknown_part_and_bad_text() -> None:
    from sie_server.processors.streaming import _parse_chat_content, _ValidationError

    bad_type = _parse_chat_content([{"type": "audio_url", "audio_url": {"url": "x"}}], 2)
    assert isinstance(bad_type, _ValidationError)
    assert bad_type.code == "invalid_request"
    assert "messages[2].content[0].type" in bad_type.message

    bad_text = _parse_chat_content([{"type": "text", "text": 123}], 0)
    assert isinstance(bad_text, _ValidationError)
    assert "must be a string" in bad_text.message

    bad_url = _parse_chat_content([{"type": "image_url", "image_url": {"url": ""}}], 0)
    assert isinstance(bad_url, _ValidationError)
    assert "image_url.url" in bad_url.message


def test_render_chat_template_builds_image_placeholders(monkeypatch: pytest.MonkeyPatch) -> None:
    from sie_server.processors.streaming import _ChatMessage

    captured: dict[str, Any] = {}

    class _FakeTok:
        def apply_chat_template(self, message_dicts: Any, **_kw: Any) -> str:
            captured["dicts"] = message_dicts
            return "RENDERED"

    async def _fake_get_tokenizer(self: Any, model_id: str) -> Any:
        return _FakeTok()

    monkeypatch.setattr(StreamingProcessor, "_get_tokenizer", _fake_get_tokenizer)
    proc = StreamingProcessor(nc=AsyncMock(), registry=_make_registry(_FakeGenAdapter([])), worker_id="w1")

    messages = (
        _ChatMessage(role="system", content="be helpful"),
        _ChatMessage(
            role="user",
            content="what is this?",
            images=({"data": b"img1", "format": "png"}, {"data": b"img2", "format": "png"}),
        ),
    )

    async def _run() -> Any:
        return await proc._render_chat_template("test/model", messages)

    rendered = asyncio.run(_run())
    assert rendered == "RENDERED"
    dicts = captured["dicts"]
    # Text-only system message stays a plain string.
    assert dicts[0]["content"] == "be helpful"
    # Vision user message becomes a parts list: 2 image placeholders + text.
    user_content = dicts[1]["content"]
    assert [p["type"] for p in user_content] == ["image", "image", "text"]
    assert user_content[2] == {"type": "text", "text": "what is this?"}


def test_render_chat_template_interleaves_content_parts(monkeypatch: pytest.MonkeyPatch) -> None:
    # #1294: with an ordered layout, placeholders render in their original
    # positions (text↔image interleaved), NOT all-images-first.
    from sie_server.processors.streaming import _ChatMessage

    captured: dict[str, Any] = {}

    class _FakeTok:
        def apply_chat_template(self, message_dicts: Any, **_kw: Any) -> str:
            captured["dicts"] = message_dicts
            return "RENDERED"

    async def _fake_get_tokenizer(self: Any, model_id: str) -> Any:
        return _FakeTok()

    monkeypatch.setattr(StreamingProcessor, "_get_tokenizer", _fake_get_tokenizer)
    proc = StreamingProcessor(nc=AsyncMock(), registry=_make_registry(_FakeGenAdapter([])), worker_id="w1")

    messages = (
        _ChatMessage(
            role="user",
            content="Page 1:Page 2:which has a cat?",
            images=({"data": b"imgA", "format": "png"}, {"data": b"imgB", "format": "png"}),
            content_parts=(
                {"type": "text", "text": "Page 1:"},
                {"type": "image"},
                {"type": "text", "text": "Page 2:"},
                {"type": "image"},
                {"type": "text", "text": "which has a cat?"},
            ),
        ),
    )

    async def _run() -> Any:
        return await proc._render_chat_template("test/model", messages)

    rendered = asyncio.run(_run())
    assert rendered == "RENDERED"
    user_content = captured["dicts"][0]["content"]
    assert [p["type"] for p in user_content] == ["text", "image", "text", "image", "text"]
    assert user_content[0] == {"type": "text", "text": "Page 1:"}
    assert user_content[2] == {"type": "text", "text": "Page 2:"}
    assert user_content[4] == {"type": "text", "text": "which has a cat?"}


def test_parse_content_parts_field() -> None:
    from sie_server.processors.streaming import _parse_content_parts_field, _ValidationError

    assert _parse_content_parts_field(None, 0) is None
    ok = _parse_content_parts_field([{"type": "text", "text": "a"}, {"type": "image"}], 0)
    assert ok == ({"type": "text", "text": "a"}, {"type": "image"})
    bad_type = _parse_content_parts_field([{"type": "audio"}], 1)
    assert isinstance(bad_type, _ValidationError)
    assert "content_parts[0].type" in bad_type.message
    bad_text = _parse_content_parts_field([{"type": "text", "text": 5}], 0)
    assert isinstance(bad_text, _ValidationError)
    assert "content_parts[0].text" in bad_text.message


def test_validate_rejects_content_parts_image_count_mismatch() -> None:
    # Gateway path: 2 image placeholders but only 1 image byte would desync
    # placeholders from ``image_data`` at the engine — the validator rejects it.
    from sie_server.processors.streaming import _ValidationError

    wi = _make_work_item(
        generate={
            "messages": [
                {
                    "role": "user",
                    "content": "a b",
                    "images": [{"data": base64.b64encode(b"img").decode(), "format": "png"}],
                    "content_parts": [
                        {"type": "text", "text": "a"},
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": "b"},
                    ],
                }
            ],
            "max_new_tokens": 8,
        }
    )
    result = StreamingProcessor._validate_generate_params(wi)
    assert isinstance(result, _ValidationError)
    assert "image placeholder" in result.message


def test_validate_threads_matching_gateway_content_parts() -> None:
    # Gateway path with consistent counts: the ordered layout reaches the
    # decoded message in order so the worker can interleave placeholders.
    from sie_server.processors.streaming import _MessagesInput, _ValidationError

    wi = _make_work_item(
        generate={
            "messages": [
                {
                    "role": "user",
                    "content": "Page 1:Page 2:",
                    "images": [
                        {"data": base64.b64encode(b"A").decode(), "format": "png"},
                        {"data": base64.b64encode(b"B").decode(), "format": "png"},
                    ],
                    "content_parts": [
                        {"type": "text", "text": "Page 1:"},
                        {"type": "image"},
                        {"type": "text", "text": "Page 2:"},
                        {"type": "image"},
                    ],
                }
            ],
            "max_new_tokens": 8,
        }
    )
    result = StreamingProcessor._validate_generate_params(wi)
    assert not isinstance(result, _ValidationError)
    assert isinstance(result.input, _MessagesInput)
    parts = result.input.messages[0].content_parts
    assert parts is not None
    assert [p["type"] for p in parts] == ["text", "image", "text", "image"]


def test_validate_rejects_both_image_bearing_layout_sources() -> None:
    # A malformed/internal producer can send an image-bearing ``content`` array
    # (the real parsed layout) AND an image-bearing ``content_parts`` field whose
    # layout differs. The field would override the layout ``content`` was parsed
    # from, so the validated text ("REAL") and the rendered layout ("FAKE") would
    # diverge. Exactly one image-bearing layout source is allowed — reject it.
    from sie_server.processors.streaming import _ValidationError

    wi = _make_work_item(
        generate={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "REAL"},
                        {"type": "image_url", "image_url": {"url": _png_data_uri(b"x")}},
                    ],
                    "content_parts": [
                        {"type": "text", "text": "FAKE"},
                        {"type": "image"},
                    ],
                }
            ],
            "max_new_tokens": 8,
        }
    )
    result = StreamingProcessor._validate_generate_params(wi)
    assert isinstance(result, _ValidationError)
    assert "exactly one layout source" in result.message


def test_validate_image_free_content_parts_field_does_not_shadow_layout() -> None:
    # Defensive: a request carrying BOTH an array content (direct-path layout
    # with an image) AND an image-free content_parts field must keep the real
    # layout — the image-free field must not shadow it.
    from sie_server.processors.streaming import _ValidationError

    wi = _make_work_item(
        generate={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image_url", "image_url": {"url": _png_data_uri(b"x")}},
                    ],
                    "content_parts": [{"type": "text", "text": "ignore me"}],  # image-free
                }
            ],
            "max_new_tokens": 8,
        }
    )
    result = StreamingProcessor._validate_generate_params(wi)
    assert not isinstance(result, _ValidationError)
    parts = result.input.messages[0].content_parts
    assert parts is not None
    assert [p["type"] for p in parts] == ["text", "image"]


def test_parse_message_images_field() -> None:
    from sie_server.processors.streaming import _parse_message_images_field, _ValidationError

    # Missing field → no images.
    assert _parse_message_images_field(None, 0) == ()
    # Valid gateway base64 string → decoded to bytes.
    out = _parse_message_images_field([{"data": base64.b64encode(b"catbytes").decode(), "format": "png"}], 0)
    assert not isinstance(out, _ValidationError)
    assert out[0]["data"] == b"catbytes"
    assert out[0]["format"] == "png"
    # Non-list rejects.
    assert isinstance(_parse_message_images_field({"data": "x"}, 1), _ValidationError)
    # Non-string data rejects (gateway sends a base64 string).
    bad = _parse_message_images_field([{"data": 123}], 2)
    assert isinstance(bad, _ValidationError)
    assert "messages[2].images[0]" in bad.message
    # Invalid base64 rejects.
    bad_b64 = _parse_message_images_field([{"data": "!!!notbase64!!!"}], 0)
    assert isinstance(bad_b64, _ValidationError)
    assert "invalid base64" in bad_b64.message
    # Empty data rejects.
    empty = _parse_message_images_field([{"data": ""}], 0)
    assert isinstance(empty, _ValidationError)
    assert "non-empty base64" in empty.message


@pytest.mark.asyncio
async def test_images_rejected_on_text_only_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker-side defense-in-depth gate: a model whose config says
    ``inputs.image`` is False rejects an image request (queue-path bypass of
    the gateway gate) with a clean ``invalid_request`` instead of a broken
    prompt / opaque SGLang error.
    """

    async def _fake_get_tokenizer(self: Any, model_id: str) -> Any:
        class _Tok:
            def apply_chat_template(self, message_dicts: Any, **_kw: Any) -> str:
                return "rendered"

        return _Tok()

    monkeypatch.setattr(StreamingProcessor, "_get_tokenizer", _fake_get_tokenizer)
    nc = AsyncMock()
    adapter = _FakeGenAdapter([GenerationChunk(text_delta="", done=True, finish_reason="stop")])
    registry = _make_registry(adapter)
    # Model config declares text-only input.
    registry.get_config.return_value.inputs.image = False
    proc = StreamingProcessor(nc=nc, registry=registry, worker_id="w1")
    wi = _make_work_item(
        generate={
            "messages": [
                {
                    "role": "user",
                    "content": "what is this?",
                    "images": [{"data": base64.b64encode(b"img").decode(), "format": "png"}],
                }
            ],
            "max_new_tokens": 8,
        }
    )
    await proc.process(_make_msg(wi), "test/model")
    terminal = _decode_chunks(nc)[-1]
    assert terminal["done"] is True
    assert terminal["error"]["code"] == "invalid_request"
    assert "does not support image input" in terminal["error"]["message"]


@pytest.mark.asyncio
async def test_messages_images_field_reaches_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gateway path: a message-level ``images`` field of decoded bytes is
    threaded to ``adapter.generate(images=...)``. ``content`` is plain text.
    """

    class _RecordingAdapter(_FakeGenAdapter):
        def __init__(self, script: list[GenerationChunk]) -> None:
            super().__init__(script)
            self.received_images: Any = "UNSET"

        async def generate(self, prompt: str, *, max_new_tokens: int, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
            self.received_images = kwargs.get("images")
            for chunk in self._script:
                yield chunk

    async def _fake_get_tokenizer(self: Any, model_id: str) -> Any:
        class _Tok:
            def apply_chat_template(self, message_dicts: Any, **_kw: Any) -> str:
                return "rendered prompt"

        return _Tok()

    monkeypatch.setattr(StreamingProcessor, "_get_tokenizer", _fake_get_tokenizer)
    script = [
        GenerationChunk(text_delta="a cat", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=5, completion_tokens=2),
    ]
    adapter = _RecordingAdapter(script)
    nc = AsyncMock()
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")
    wi = _make_work_item(
        generate={
            "messages": [
                {
                    "role": "user",
                    "content": "what is this?",
                    "images": [{"data": base64.b64encode(b"catbytes").decode(), "format": "png"}],
                }
            ],
            "max_new_tokens": 8,
        }
    )
    await proc.process(_make_msg(wi), "test/model")
    assert isinstance(adapter.received_images, list)
    assert len(adapter.received_images) == 1
    assert adapter.received_images[0]["data"] == b"catbytes"


@pytest.mark.asyncio
async def test_messages_image_reaches_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: a generate work item carrying an OpenAI image part is
    decoded and forwarded to ``adapter.generate(images=...)`` as bytes.
    """

    class _RecordingAdapter(_FakeGenAdapter):
        def __init__(self, script: list[GenerationChunk]) -> None:
            super().__init__(script)
            self.received_images: Any = "UNSET"

        async def generate(self, prompt: str, *, max_new_tokens: int, **kwargs: Any) -> AsyncIterator[GenerationChunk]:
            self.received_images = kwargs.get("images")
            for chunk in self._script:
                yield chunk

    async def _fake_get_tokenizer(self: Any, model_id: str) -> Any:
        class _Tok:
            def apply_chat_template(self, message_dicts: Any, **_kw: Any) -> str:
                return "rendered prompt"

        return _Tok()

    monkeypatch.setattr(StreamingProcessor, "_get_tokenizer", _fake_get_tokenizer)

    script = [
        GenerationChunk(text_delta="a cat", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=5, completion_tokens=2),
    ]
    adapter = _RecordingAdapter(script)
    nc = AsyncMock()
    proc = StreamingProcessor(nc=nc, registry=_make_registry(adapter), worker_id="w1")

    wi = _make_work_item(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": _png_data_uri(b"catbytes")}},
                ],
            }
        ]
    )
    await proc.process(_make_msg(wi), "test/model")

    assert isinstance(adapter.received_images, list)
    assert len(adapter.received_images) == 1
    assert adapter.received_images[0]["data"] == b"catbytes"
    assert adapter.received_images[0]["format"] == "png"
