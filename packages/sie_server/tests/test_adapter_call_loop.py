"""Unit tests for :mod:`sie_server.adapter_call_loop`.

Scope:
    * Dispatches to the correct per-op ``QueueExecutor`` method.
    * Preserves item order when forwarding.
    * Rejects mixed-op batches with per-item error outcomes.
    * Produces typed errors (``run_batch_*``) for malformed inputs
      without crashing.

Out of scope: the IPC round-trip (that's ``test_ipc_server.py``) and
the actual adapter forward pass (that's ``test_queue_executor*.py``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sie_server import adapter_call_loop
from sie_server.adapter_call_loop import (
    RUN_BATCH_INVALID_ITEM,
    RUN_BATCH_MIXED_OP,
    RUN_BATCH_UNKNOWN_OP,
    handle_run_batch,
)
from sie_server.ipc_types import (
    BatchedF16MultivectorItem,
    BatchedF16MultivectorOutput,
    BatchOutcome,
    EncodeBatchItem,
    ExtractBatchItem,
    ItemOutcome,
    RunBatchItem,
    RunBatchRequest,
    ScoreBatchItem,
    UnitCounts,
)

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------


def _mk_encode(idx: int, text: str = "hello") -> EncodeBatchItem:
    return EncodeBatchItem(
        work_item_id=f"w{idx}",
        request_id=f"r{idx}",
        item_index=idx,
        total_items=1,
        timestamp=0.0,
        item={"text": text},
    )


def _mk_score(idx: int) -> ScoreBatchItem:
    return ScoreBatchItem(
        work_item_id=f"w{idx}",
        request_id=f"r{idx}",
        item_index=idx,
        total_items=1,
        timestamp=0.0,
        query_item={"text": "q"},
        score_items=[{"text": "d1"}, {"text": "d2"}],
    )


def _mk_extract(idx: int) -> ExtractBatchItem:
    return ExtractBatchItem(
        work_item_id=f"w{idx}",
        request_id=f"r{idx}",
        item_index=idx,
        total_items=1,
        timestamp=0.0,
        item={"text": "extract me"},
    )


def _ok_outcome(idx: int, result: bytes = b"\x80") -> ItemOutcome:
    return ItemOutcome(
        work_item_id=f"w{idx}",
        request_id=f"r{idx}",
        item_index=idx,
        disposition="publish_and_ack",
        result_msgpack=result,
    )


def _mk_request(
    *items: RunBatchItem,
    model_id: str = "test/model",
    batch_id: int = 1,
    lora_key: str = "",
    total_cost: int = 0,
    accepts_batched_f16_multivectors: bool = False,
) -> RunBatchRequest:
    return RunBatchRequest(
        model_id=model_id,
        batch_id=batch_id,
        lora_key=lora_key,
        total_cost=total_cost,
        items=list(items),
        accepts_batched_f16_multivectors=accepts_batched_f16_multivectors,
    )


def _mock_executor(**responses: Any) -> AsyncMock:
    """Build an AsyncMock executor with the three per-op methods.

    Pass ``encode=BatchOutcome(...)`` etc to control what each method
    returns.
    """
    exe = AsyncMock()
    exe.process_encode_batch = AsyncMock(
        return_value=responses.get("encode", BatchOutcome(outcomes=[])),
    )
    exe.process_score_batch = AsyncMock(
        return_value=responses.get("score", BatchOutcome(outcomes=[])),
    )
    exe.process_extract_batch = AsyncMock(
        return_value=responses.get("extract", BatchOutcome(outcomes=[])),
    )
    return exe


# --------------------------------------------------------------------
# Happy paths
# --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encode_batch_dispatches_to_process_encode() -> None:
    expected = BatchOutcome(outcomes=[_ok_outcome(0), _ok_outcome(1)])
    exe = _mock_executor(encode=expected)
    req = _mk_request(
        RunBatchItem(op="encode", encode=_mk_encode(0)),
        RunBatchItem(op="encode", encode=_mk_encode(1)),
    )

    result = await handle_run_batch(exe, req)

    assert result.outcomes == expected.outcomes
    exe.process_encode_batch.assert_awaited_once()
    inner = exe.process_encode_batch.call_args.args[0]
    assert inner.model_id == "test/model"
    assert len(inner.items) == 2
    # Order must be preserved so per-item index alignment holds.
    assert [it.item_index for it in inner.items] == [0, 1]
    # Score / extract handlers never called.
    exe.process_score_batch.assert_not_called()
    exe.process_extract_batch.assert_not_called()


@pytest.mark.asyncio
async def test_encode_batch_preserves_batched_f16_transport_capability() -> None:
    f16_batch = BatchedF16MultivectorOutput(
        values_f16=b"\x00<\x00@",
        items=[
            BatchedF16MultivectorItem(
                work_item_id="w0",
                byte_offset=0,
                byte_len=4,
                num_tokens=1,
                token_dims=2,
            )
        ],
    )
    expected = BatchOutcome(
        outcomes=[_ok_outcome(0)],
        batched_f16_multivectors=[f16_batch],
    )
    exe = _mock_executor(encode=expected)
    req = _mk_request(
        RunBatchItem(op="encode", encode=_mk_encode(0)),
        accepts_batched_f16_multivectors=True,
    )

    result = await handle_run_batch(exe, req)

    inner = exe.process_encode_batch.call_args.args[0]
    assert inner.accepts_batched_f16_multivectors is True
    assert result.batched_f16_multivectors == [f16_batch]


@pytest.mark.asyncio
async def test_score_batch_dispatches_to_process_score() -> None:
    expected = BatchOutcome(outcomes=[_ok_outcome(0)])
    exe = _mock_executor(score=expected)
    req = _mk_request(RunBatchItem(op="score", score=_mk_score(0)))

    result = await handle_run_batch(exe, req)

    assert result.outcomes == expected.outcomes
    exe.process_score_batch.assert_awaited_once()
    exe.process_encode_batch.assert_not_called()
    exe.process_extract_batch.assert_not_called()


@pytest.mark.asyncio
async def test_extract_batch_dispatches_to_process_extract() -> None:
    expected = BatchOutcome(outcomes=[_ok_outcome(0)])
    exe = _mock_executor(extract=expected)
    req = _mk_request(RunBatchItem(op="extract", extract=_mk_extract(0)))

    result = await handle_run_batch(exe, req)

    assert result.outcomes == expected.outcomes
    exe.process_extract_batch.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_batch_records_engine_metrics_without_sidecar_owned_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = BatchOutcome(
        outcomes=[
            ItemOutcome(
                work_item_id="w0",
                request_id="r0",
                item_index=0,
                disposition="publish_and_ack",
                result_msgpack=b"\x80",
                inference_ms=20.0,
                tokenization_ms=3.0,
                postprocessing_ms=2.0,
                units=UnitCounts(input_tokens=11, images=1),
            )
        ]
    )
    calls: list[dict[str, Any]] = []
    telemetry = SimpleNamespace(item_completed=lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(adapter_call_loop, "worker_telemetry", lambda: telemetry)
    monkeypatch.setattr(adapter_call_loop, "worker_telemetry_enabled", lambda: True)
    item = _mk_encode(0)
    item.profile_id = "default"
    result = await handle_run_batch(
        _mock_executor(encode=expected),
        _mk_request(RunBatchItem(op="encode", encode=item)),
    )

    assert result == expected
    assert len(calls) == 1
    assert calls[0]["operation"] == "encode"
    assert calls[0]["outcome"] == "success"
    assert calls[0]["model"] == "test/model"
    assert calls[0]["profile"] == "default"
    assert calls[0]["inference_s"] == pytest.approx(0.02)
    assert calls[0]["tokenization_s"] == pytest.approx(0.003)
    assert calls[0]["postprocessing_s"] == pytest.approx(0.002)
    assert calls[0]["units"] == {"input_tokens": 11, "images": 1}
    # The Rust sidecar owns queue/batch metrics and identity/error fields never
    # cross this engine metric seam.
    forbidden = {
        "queue_s",
        "batch_size",
        "batch_cost",
        "request_id",
        "work_item_id",
        "error",
        "error_code",
    }
    assert forbidden.isdisjoint(calls[0])


@pytest.mark.asyncio
async def test_run_batch_skips_telemetry_timer_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = BatchOutcome(outcomes=[_ok_outcome(0)])
    monkeypatch.setattr(adapter_call_loop, "worker_telemetry_enabled", lambda: False)

    def unexpected_timer() -> float:
        raise AssertionError("disabled telemetry must not sample its completion timer")

    monkeypatch.setattr(adapter_call_loop.time, "perf_counter", unexpected_timer)

    result = await handle_run_batch(
        _mock_executor(encode=expected),
        _mk_request(RunBatchItem(op="encode", encode=_mk_encode(0))),
    )

    assert result == expected


# --------------------------------------------------------------------
# Error paths
# --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_batch_returns_empty_outcome() -> None:
    exe = _mock_executor()
    req = _mk_request()  # no items

    result = await handle_run_batch(exe, req)

    assert result.outcomes == []
    exe.process_encode_batch.assert_not_called()


@pytest.mark.asyncio
async def test_mixed_op_batch_is_rejected_wholesale() -> None:
    exe = _mock_executor()
    req = _mk_request(
        RunBatchItem(op="encode", item_index=7, encode=_mk_encode(7)),
        RunBatchItem(op="score", item_index=9, score=_mk_score(9)),
    )

    result = await handle_run_batch(exe, req)

    assert len(result.outcomes) == 2
    for o in result.outcomes:
        assert o.disposition == "publish_error_and_ack"
        assert o.error_code == RUN_BATCH_MIXED_OP
    assert [o.item_index for o in result.outcomes] == [7, 9]
    # Neither underlying handler was called — protocol violation
    # short-circuits before any executor work.
    exe.process_encode_batch.assert_not_called()
    exe.process_score_batch.assert_not_called()


@pytest.mark.asyncio
async def test_unknown_op_is_rejected_with_typed_error_code() -> None:
    exe = _mock_executor()
    # msgspec can't enforce the Literal at construction time without
    # being given a RunBatchItem instance; we bypass here to simulate
    # a future op the current Python doesn't know.
    req = RunBatchRequest(
        model_id="m",
        batch_id=1,
        lora_key="",
        total_cost=0,
        items=[RunBatchItem(op="future_op")],  # type: ignore[arg-type]
    )

    result = await handle_run_batch(exe, req)

    assert len(result.outcomes) == 1
    assert result.outcomes[0].error_code == RUN_BATCH_UNKNOWN_OP
    assert result.outcomes[0].disposition == "publish_error_and_ack"


@pytest.mark.asyncio
async def test_invalid_item_tagged_but_batch_continues() -> None:
    # Second item has op=encode but encode payload missing — that
    # one becomes an invalid_item outcome, and the rest of the batch
    # still goes to process_encode_batch for item 0.
    expected = BatchOutcome(outcomes=[_ok_outcome(0)])
    exe = _mock_executor(encode=expected)
    req = _mk_request(
        RunBatchItem(op="encode", encode=_mk_encode(0)),
        RunBatchItem(op="encode", work_item_id="w1", request_id="r1", item_index=1, encode=None),
    )

    result = await handle_run_batch(exe, req)

    assert len(result.outcomes) == 2
    # Invalid outcomes come first, followed by the executor's outcomes.
    assert result.outcomes[0].error_code == RUN_BATCH_INVALID_ITEM
    assert result.outcomes[0].disposition == "publish_error_and_ack"
    assert result.outcomes[0].work_item_id == "w1"
    assert result.outcomes[0].request_id == "r1"
    assert result.outcomes[0].item_index == 1
    assert result.outcomes[1].work_item_id == "w0"
    # The executor was called with only the valid item.
    exe.process_encode_batch.assert_awaited_once()
    inner = exe.process_encode_batch.call_args.args[0]
    assert len(inner.items) == 1


@pytest.mark.asyncio
async def test_all_items_invalid_short_circuits_executor() -> None:
    exe = _mock_executor()
    req = _mk_request(
        RunBatchItem(op="encode", work_item_id="w0", request_id="r0", item_index=0, encode=None),
        RunBatchItem(op="encode", work_item_id="w1", request_id="r1", item_index=1, encode=None),
    )

    result = await handle_run_batch(exe, req)

    assert len(result.outcomes) == 2
    assert all(o.error_code == RUN_BATCH_INVALID_ITEM for o in result.outcomes)
    assert [o.work_item_id for o in result.outcomes] == ["w0", "w1"]
    assert [o.request_id for o in result.outcomes] == ["r0", "r1"]
    exe.process_encode_batch.assert_not_called()


@pytest.mark.asyncio
async def test_lora_key_is_passed_through_to_model_id_untouched() -> None:
    # The lora_key isn't plumbed into the per-op request types for
    # v1 (the Python adapter picks the active LoRA from its own
    # state) — we just assert that model_id passes through verbatim.
    # When the adapter becomes LoRA-aware in a later stage this test
    # can be tightened to also check set_active_lora(lora_key).
    expected = BatchOutcome(outcomes=[_ok_outcome(0)])
    exe = _mock_executor(encode=expected)
    req = _mk_request(
        RunBatchItem(op="encode", encode=_mk_encode(0)),
        model_id="special/model-v7",
        lora_key="some-lora",
    )

    await handle_run_batch(exe, req)

    inner = exe.process_encode_batch.call_args.args[0]
    assert inner.model_id == "special/model-v7"
