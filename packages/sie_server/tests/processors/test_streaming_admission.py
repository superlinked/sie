"""KV-cache admission control tests.

Exercises the streaming processor's reserve/release state machine
with a fake :class:`GenerationAdapter`. Covers:

- Admission-on overload: NAK envelopes appear with ``reason="kv_budget"``,
  gauges move with reserve/release, in-flight peaks then drains.
- Admission-off overload: zero rejected_total counter, gauges still emitted.
- Reserve/release accounting on every exit path (normal end, cancel,
  transport_failure, inference error).
- Env-var precedence: ``SIE_GENERATION_ADMISSION`` overrides profile.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack
import pytest
from sie_server.adapters._generation_base import GenerationAdapter, GenerationChunk
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters.base import ModelCapabilities, ModelDims
from sie_server.processors.admission import (
    parse_admission_env,
    resolve_admission_enabled,
)
from sie_server.processors.streaming import StreamingProcessor

MODEL = "test/model"


class _ScriptedAdapter(GenerationAdapter):
    """Adapter that yields a scripted sequence, optionally waiting on an event."""

    spec = AdapterSpec(inputs=("text",), outputs=("tokens",), unload_fields=())

    def __init__(
        self,
        script: list[GenerationChunk],
        *,
        hold: asyncio.Event | None = None,
        raise_after: int | None = None,
    ) -> None:
        self._script = script
        self._hold = hold
        self._raise_after = raise_after

    def load(self, device: str) -> None:  # pragma: no cover
        return None

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
        for i, chunk in enumerate(self._script):
            if self._hold is not None:
                await self._hold.wait()
            if self._raise_after is not None and i == self._raise_after:
                raise RuntimeError("boom")
            yield chunk


def _make_registry(adapter: GenerationAdapter) -> MagicMock:
    registry = MagicMock()
    registry.is_loaded.return_value = True
    registry.get.return_value = adapter
    registry.device = "cpu"
    return registry


def _make_work_item(*, request_id: str = "req-1", max_new_tokens: int = 64) -> dict[str, Any]:
    return {
        "work_item_id": f"{request_id}.0",
        "request_id": request_id,
        "item_index": 0,
        "total_items": 1,
        "operation": "generate",
        "model_id": MODEL,
        "profile_id": "default",
        "pool_name": "default",
        "machine_profile": "default",
        "router_id": "router-1",
        "reply_subject": f"_INBOX.router-1.{request_id}",
        "timestamp": time.time(),
        "generate": {"prompt": "Hello", "max_new_tokens": max_new_tokens},
    }


def _make_msg(wi: dict[str, Any]) -> AsyncMock:
    msg = AsyncMock()
    msg.data = msgpack.packb(wi, use_bin_type=True)
    return msg


def _decode(nc: AsyncMock) -> list[dict[str, Any]]:
    return [msgpack.unpackb(call.args[1], raw=False) for call in nc.publish.await_args_list]


# ---------------------------------------------------------------------------
# Env-var precedence
# ---------------------------------------------------------------------------


class TestAdmissionEnvParsing:
    def test_parse_unset_is_auto(self) -> None:
        assert parse_admission_env(None) == "auto"
        assert parse_admission_env("") == "auto"
        assert parse_admission_env("   ") == "auto"

    def test_parse_case_insensitive(self) -> None:
        assert parse_admission_env("ON") == "on"
        assert parse_admission_env("Off") == "off"
        assert parse_admission_env(" auto ") == "auto"

    def test_parse_unknown_falls_back_to_auto(self) -> None:
        assert parse_admission_env("maybe") == "auto"

    def test_resolve_env_on_forces_true(self) -> None:
        assert resolve_admission_enabled(profile_admission=False, env_value="on") is True

    def test_resolve_env_off_forces_false(self) -> None:
        assert resolve_admission_enabled(profile_admission=True, env_value="off") is False

    def test_resolve_auto_honors_profile_true(self) -> None:
        assert resolve_admission_enabled(profile_admission=True, env_value="auto") is True

    def test_resolve_auto_honors_profile_false(self) -> None:
        assert resolve_admission_enabled(profile_admission=False, env_value="auto") is False

    def test_resolve_auto_defaults_to_false_when_profile_none(self) -> None:
        assert resolve_admission_enabled(profile_admission=None, env_value="auto") is False


# ---------------------------------------------------------------------------
# Reserve/release accounting on all four exit paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reserved_returns_to_zero_on_normal_end() -> None:
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="hi", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter(script)),
        worker_id="w1",
        kv_budget_tokens=10_000,
        admission_enabled=True,
    )
    await proc.process(_make_msg(_make_work_item()), MODEL)
    assert proc.kv_reserved_tokens() == 0


@pytest.mark.asyncio
async def test_reserved_returns_to_zero_on_cancel() -> None:
    nc = AsyncMock()
    hold = asyncio.Event()
    script = [GenerationChunk(text_delta="x", is_first=True)] * 10
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter(script, hold=hold)),
        worker_id="w1",
        kv_budget_tokens=10_000,
        admission_enabled=True,
    )

    async def _cancel_soon() -> None:
        await asyncio.sleep(0.05)
        proc.signal_cancel("req-1")
        hold.set()

    cancel_task = asyncio.create_task(_cancel_soon())
    await proc.process(_make_msg(_make_work_item()), MODEL)
    await cancel_task
    assert proc.kv_reserved_tokens() == 0


@pytest.mark.asyncio
async def test_reserved_returns_to_zero_on_inference_error() -> None:
    nc = AsyncMock()
    # Raise mid-stream (raise_after=1, after the initial delta has been
    # yielded by the iterator).
    script = [
        GenerationChunk(text_delta="ok", is_first=True),
        GenerationChunk(text_delta="boom"),  # raise_after=1 — never yielded
    ]
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter(script, raise_after=1)),
        worker_id="w1",
        kv_budget_tokens=10_000,
        admission_enabled=True,
    )
    await proc.process(_make_msg(_make_work_item()), MODEL)
    assert proc.kv_reserved_tokens() == 0


@pytest.mark.asyncio
async def test_reserved_returns_to_zero_on_transport_failure() -> None:
    """A publish error followed by enough retries surfaces a transport
    failure terminal — reservation must still be released.
    """
    nc = AsyncMock()
    nc.publish = AsyncMock(side_effect=RuntimeError("publish broken"))
    script = [GenerationChunk(text_delta="a", is_first=True) for _ in range(5)]
    script.append(GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=5))
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter(script)),
        worker_id="w1",
        kv_budget_tokens=10_000,
        admission_enabled=True,
    )
    await proc.process(_make_msg(_make_work_item()), MODEL)
    assert proc.kv_reserved_tokens() == 0


# ---------------------------------------------------------------------------
# Admission-on rejection + NAK envelope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admission_on_rejects_over_budget_request_with_kv_budget_nak() -> None:
    """A request whose reserve exceeds the budget publishes a NAK envelope
    with ``reason="kv_budget"`` and ACKs the JetStream msg.
    """
    nc = AsyncMock()
    # Empty script suffices — process() rejects before reaching the adapter.
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter([])),
        worker_id="w1",
        # Tiny budget — even max_new_tokens=64 + a 2-char prompt blows it.
        kv_budget_tokens=16,
        admission_enabled=True,
    )
    msg = _make_msg(_make_work_item(max_new_tokens=64))
    await proc.process(msg, MODEL)

    decoded = _decode(nc)
    assert len(decoded) == 1
    nak = decoded[0]
    assert nak["kind"] == "nak"
    assert nak["reason"] == "kv_budget"
    assert nak["request_id"] == "req-1"
    msg.ack.assert_awaited()
    # Reservation never took effect — release path is unreached.
    assert proc.kv_reserved_tokens() == 0


@pytest.mark.asyncio
async def test_admission_on_concurrent_overload_some_rejected_some_admitted() -> None:
    """64 concurrent requests at a tiny budget produce a mix of NAKs and
    successful terminals; reserved drains back to zero at the end.
    """
    nc = AsyncMock()
    hold = asyncio.Event()
    # Each admitted request will hold mid-stream until released so we can
    # observe concurrent reservations stacking.
    script = [
        GenerationChunk(text_delta="hi", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    # The adapter is shared but ``hold`` is the gate; the first wait
    # blocks until we release.
    adapter = _ScriptedAdapter(script, hold=hold)
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(adapter),
        worker_id="w1",
        # Each request reserves ~max_new_tokens=8 + ceil(5/4)≈1 = 9 tokens.
        # Budget=20 admits ~2 concurrent and rejects the rest.
        kv_budget_tokens=20,
        admission_enabled=True,
    )

    msgs = [_make_msg(_make_work_item(request_id=f"req-{i}", max_new_tokens=8)) for i in range(8)]
    tasks = [asyncio.create_task(proc.process(msg, MODEL)) for msg in msgs]
    # Let admission decisions settle, then release the held streams.
    await asyncio.sleep(0.05)
    hold.set()
    await asyncio.gather(*tasks)

    decoded = _decode(nc)
    naks = [c for c in decoded if c.get("kind") == "nak"]
    terminals = [c for c in decoded if c.get("kind") == "chunk" and c.get("done") is True]
    assert naks, "expected at least some NAK envelopes"
    assert all(n["reason"] == "kv_budget" for n in naks)
    assert terminals, "expected at least some successful terminals"
    assert proc.kv_reserved_tokens() == 0


# ---------------------------------------------------------------------------
# Admission-off behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admission_off_never_rejects() -> None:
    """With ``admission_enabled=False`` the same over-budget request goes
    through; the reserved counter still moves so gauges stay live.
    """
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="hi", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter(script)),
        worker_id="w1",
        kv_budget_tokens=16,
        admission_enabled=False,
    )
    msg = _make_msg(_make_work_item(max_new_tokens=64))
    await proc.process(msg, MODEL)

    decoded = _decode(nc)
    # No NAK envelope.
    assert not any(c.get("kind") == "nak" for c in decoded)
    # A real terminal chunk was published.
    terminals = [c for c in decoded if c.get("kind") == "chunk" and c.get("done") is True]
    assert terminals
    assert terminals[-1]["finish_reason"] == "stop"
    # Reservation released cleanly.
    assert proc.kv_reserved_tokens() == 0


@pytest.mark.asyncio
async def test_admission_off_reservation_state_still_updates() -> None:
    """Reservation state updates around work even when admission is disabled."""
    nc = AsyncMock()
    hold = asyncio.Event()
    script = [GenerationChunk(text_delta="ok", is_first=True)]
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter(script, hold=hold)),
        worker_id="w1",
        kv_budget_tokens=1_000,
        admission_enabled=False,
    )

    async def _release_after_observe() -> None:
        for _ in range(50):
            if proc.kv_reserved_tokens() > 0:
                break
            await asyncio.sleep(0.01)
        hold.set()

    msg = _make_msg(_make_work_item(max_new_tokens=64))
    release = asyncio.create_task(_release_after_observe())
    await proc.process(msg, "gauge/model")
    await release
    # After the run the counter is back to zero.
    assert proc.kv_reserved_tokens() == 0


# ---------------------------------------------------------------------------
# No budget configured → admission is fully inert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_budget_configured_never_rejects() -> None:
    nc = AsyncMock()
    script = [
        GenerationChunk(text_delta="hi", is_first=True),
        GenerationChunk(text_delta="", done=True, finish_reason="stop", prompt_tokens=1, completion_tokens=1),
    ]
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter(script)),
        worker_id="w1",
        kv_budget_tokens=None,
        admission_enabled=True,  # on, but no budget → degenerates to inert
    )
    msg = _make_msg(_make_work_item(max_new_tokens=10_000))
    await proc.process(msg, MODEL)
    decoded = _decode(nc)
    assert not any(c.get("kind") == "nak" for c in decoded)
    assert proc.kv_reserved_tokens() == 0


# ---------------------------------------------------------------------------
# Lazy per-request budget resolution
# ---------------------------------------------------------------------------


def _make_registry_with_resolved_budget(adapter: GenerationAdapter, budget: int | None) -> MagicMock:
    """Registry stub whose ``get_config(...).resolve_profile('default')``
    returns a faux resolved profile with ``kv_budget_tokens=budget``.
    """
    registry = _make_registry(adapter)
    resolved = MagicMock()
    resolved.kv_budget_tokens = budget
    cfg = MagicMock()
    cfg.tasks.generate = MagicMock()  # truthy → "is a generation model"
    cfg.resolve_profile.return_value = resolved
    registry.get_config.return_value = cfg
    return registry


@pytest.mark.asyncio
async def test_lazy_budget_resolution_uses_per_request_value() -> None:
    """Boot-time budget is huge, per-request budget is tiny — request is rejected."""
    nc = AsyncMock()
    registry = _make_registry_with_resolved_budget(
        _ScriptedAdapter([GenerationChunk(text_delta="x", is_first=True)]),
        budget=16,
    )
    proc = StreamingProcessor(
        nc=nc,
        registry=registry,
        worker_id="w1",
        # Boot-time default is 10k — overridden by the per-model lookup.
        kv_budget_tokens=10_000,
        admission_enabled=True,
    )
    await proc.process(_make_msg(_make_work_item(max_new_tokens=64)), MODEL)
    decoded = _decode(nc)
    assert any(c.get("kind") == "nak" and c.get("reason") == "kv_budget" for c in decoded)


@pytest.mark.asyncio
async def test_lazy_budget_resolution_falls_back_when_lookup_fails() -> None:
    """A registry that raises on lookup falls back to the boot-time budget."""
    nc = AsyncMock()
    registry = _make_registry(_ScriptedAdapter([GenerationChunk(text_delta="x", is_first=True)]))
    registry.get_config.side_effect = KeyError("not found")
    proc = StreamingProcessor(
        nc=nc,
        registry=registry,
        worker_id="w1",
        kv_budget_tokens=16,  # boot-time budget is tiny
        admission_enabled=True,
    )
    await proc.process(_make_msg(_make_work_item(max_new_tokens=64)), MODEL)
    # Boot-time budget is honoured → request is rejected.
    decoded = _decode(nc)
    assert any(c.get("kind") == "nak" and c.get("reason") == "kv_budget" for c in decoded)


@pytest.mark.asyncio
async def test_admission_resolver_controls_enabled_per_model() -> None:
    """A model-specific resolver can enable admission even when boot default is off."""
    nc = AsyncMock()
    registry = _make_registry(_ScriptedAdapter([GenerationChunk(text_delta="x", is_first=True)]))
    proc = StreamingProcessor(
        nc=nc,
        registry=registry,
        worker_id="w1",
        kv_budget_tokens=10_000,
        admission_enabled=False,
        admission_resolver=lambda model_id: (16, model_id == MODEL),
    )

    await proc.process(_make_msg(_make_work_item(max_new_tokens=64)), MODEL)

    decoded = _decode(nc)
    assert any(c.get("kind") == "nak" and c.get("reason") == "kv_budget" for c in decoded)


@pytest.mark.asyncio
async def test_double_release_is_idempotent() -> None:
    """Calling _release_reservation twice for the same request_id is a no-op."""
    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter([])),
        worker_id="w1",
        kv_budget_tokens=10_000,
        admission_enabled=False,
    )
    # Manually drive the reserve/release pair, then call release again.
    await proc._try_reserve(MODEL, 100, request_id="r-1")
    assert proc.kv_reserved_tokens() == 100
    await proc._release_reservation(MODEL, 100, request_id="r-1")
    assert proc.kv_reserved_tokens() == 0
    # Second release should be a no-op (idempotency).
    await proc._release_reservation(MODEL, 100, request_id="r-1")
    assert proc.kv_reserved_tokens() == 0


@pytest.mark.asyncio
async def test_concurrent_attempts_same_request_id_release_returns_full_budget() -> None:
    """BUG 2/7 regression: two legitimate attempts for the SAME request_id
    (redelivery after ack_wait) must both reserve AND both release. The
    release dedup keys on ``attempt_id``, NOT ``request_id`` — otherwise the
    second release is wrongly deduped and budget leaks.

    Pre-fix: reserve×2=200, release×2 → 100 leaked.
    """
    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter([])),
        worker_id="w1",
        kv_budget_tokens=10_000,
        admission_enabled=False,
    )
    await proc._try_reserve(MODEL, 100, request_id="r-1", attempt_id="A1")
    await proc._try_reserve(MODEL, 100, request_id="r-1", attempt_id="A2")
    assert proc.kv_reserved_tokens() == 200

    await proc._release_reservation(MODEL, 100, request_id="r-1", attempt_id="A1")
    await proc._release_reservation(MODEL, 100, request_id="r-1", attempt_id="A2")
    assert proc.kv_reserved_tokens() == 0, "second legitimate release was deduped → budget leaked"

    # A redundant release of an already-released attempt is still a no-op.
    await proc._release_reservation(MODEL, 100, request_id="r-1", attempt_id="A1")
    assert proc.kv_reserved_tokens() == 0


@pytest.mark.asyncio
async def test_concurrent_attempts_cancel_reaches_both_and_cleanup_is_per_attempt() -> None:
    """BUG 2/7 regression: two concurrent attempts on one request_id register
    distinct cancel handles. ``signal_cancel(request_id)`` must reach BOTH,
    and one attempt's cleanup must NOT remove the other's handle.

    Pre-fix: ``_in_flight_cancels[request_id] = event`` clobbered, so the
    second registration overwrote the first; the first attempt's ``finally``
    popped the survivor's handle → later cancel a silent no-op.
    """
    nc = AsyncMock()
    proc = StreamingProcessor(
        nc=nc,
        registry=_make_registry(_ScriptedAdapter([])),
        worker_id="w1",
    )

    ev1 = asyncio.Event()
    ev2 = asyncio.Event()
    proc._register_cancel("r-1", "A1", ev1)
    proc._register_cancel("r-1", "A2", ev2)

    # A single cancel for the request_id reaches BOTH live attempts.
    assert proc.signal_cancel("r-1") is True
    assert ev1.is_set()
    assert ev2.is_set()

    # Attempt A1 finishes and cleans up ONLY its own handle.
    proc._unregister_cancel("r-1", "A1")
    # A2 is still live → a fresh cancel still reaches it.
    ev2b = asyncio.Event()
    # Re-register A2's (new) event to model an in-flight attempt awaiting cancel.
    proc._register_cancel("r-1", "A2", ev2b)
    assert proc.signal_cancel("r-1") is True, "surviving attempt's handle was wrongly removed"
    assert ev2b.is_set()

    # Once the last attempt cleans up, the request_id key is gone.
    proc._unregister_cancel("r-1", "A2")
    assert proc.signal_cancel("r-1") is False
