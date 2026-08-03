"""Adapter call loop for the Rust-driven scheduler.

When a request lands on the worker-sidecar path, the worker-sidecar forms
the batch itself (``sie_server_sidecar::scheduler::Scheduler``) and
dispatches it over a single :data:`~sie_server.ipc_types.METHOD_RUN_BATCH`
RPC. This module is the Python receiver for that RPC.

The previous ``SIE_RUST_SCHEDULER_MODELS`` allowlist has been retired:
routing decisions now live in the gateway's model-config API and pool
selection. Any model that lands here is running through the worker-sidecar
scheduler path.

Design rationale — why this isn't just a method on ``QueueExecutor``:

* The existing ``process_encode_batch`` / ``process_score_batch`` /
  ``process_extract_batch`` methods on ``QueueExecutor`` are the
  IPC execution path. They prepare adapter-coupled inputs and call
  ``ModelWorker.submit_preformed*`` so Rust-owned batches do not re-enter
  Python's local ``BatchFormer``.
* ``adapter_call_loop`` is a **strict subset** of what those three
  methods do: it accepts an already-formed batch (no batching, no
  coalesce, no adaptive control — those live in Rust now) and fans
  each ``RunBatchItem`` into the same per-op handlers the executor
  already calls. All the adapter-side logic (``handlers.make_config_key``,
  ``EncodePipeline.run_encode``, ``ModelWorker.submit_preformed``,
  ``_group_by_inference_config``, ``_complete_requests``) is reused.
* Keeping the dispatch thin here lets the Rust side own the only
  behaviourally-novel piece (batch formation + FCFS across LoRAs + PI
  loop) without duplicating the Python handler vocabulary.

LoRA plumbing (single source of truth for the wire-boundary contract):

* The Rust scheduler hands us **one** ``RunBatchRequest.lora_key`` per
  batch — items inside the batch are guaranteed homogeneous by the
  upstream FCFS-across-LoRAs scheduler. We forward that key into each
  ``EncodeBatchItem.options["lora"]`` / ``ExtractBatchItem.options["lora"]``
  before fanning out to the per-op handler. From there the pre-formed
  ``ModelWorker`` entrypoint picks it up via ``options.get("lora")`` and
  sets the adapter's active LoRA immediately before the forward pass.
* Empty / missing ``lora_key`` → no injection → ``options.get("lora")``
  resolves to ``None`` → base-model batcher. The empty-string-to-``None``
  coercion is the wire-boundary contract; a future native adapter must
  apply the same normalisation.
* Score is base-only — Python's ``ModelWorker.submit_score_preformed`` documents
  this as "Score operations use base model batcher (no LoRA support
  for reranking)". When ``RunBatch`` arrives with ``op="score"`` and a
  non-empty ``lora_key`` we **log** and serve the base path rather
  than fail the batch.
* Pre-existing ``options["lora"]`` on individual items is **never**
  silently overwritten — if a caller has already set it (e.g. legacy
  client code that constructs ``EncodeBatchItem`` directly) and it
  conflicts with the batch-level ``lora_key``, we log and trust the
  per-item value. The Rust scheduler never produces this combination
  today, so the warning surfaces a real protocol bug rather than
  papering over one.

Fallback contract:

* Unknown ``op`` → typed error per item (``error_code =
  "run_batch_unknown_op"``). Forward-compat safety: a newer Rust
  build can ship an op this Python doesn't know, and the operator
  sees a clean message instead of a crash.
* Mixed-op batch → wholesale rejection with every outcome tagged
  ``run_batch_mixed_op``. The Rust scheduler MUST only form
  homogeneous-op batches; this is defence-in-depth, not an expected
  code path.
* Op tag / payload mismatch (e.g. ``op = "encode"`` but
  ``encode = None``) → per-item ``run_batch_invalid_item``. Keeps
  one bad item from sinking the whole batch.

Every error path produces a ``publish_error_and_ack`` disposition —
not a ``nak_retry`` — because the payload is ill-formed at the wire
level and retrying would deliver the same bad frame. Matches the
existing behaviour of the per-op handlers when ``msgspec.convert``
fails.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from opentelemetry import propagate, trace
from opentelemetry.context import Context

from sie_server.ipc_types import (
    BatchOutcome,
    EncodeBatchItem,
    ExtractBatchItem,
    ItemOutcome,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    RunBatchItem,
    RunBatchRequest,
)
from sie_server.observability.worker_telemetry import worker_telemetry, worker_telemetry_enabled

if TYPE_CHECKING:
    from sie_server.queue_executor import QueueExecutor

logger = logging.getLogger(__name__)


# Error codes emitted by this module. Kept as module-level constants
# so callers (tests, log-greppers, dashboards) can import them without
# stringly-typed drift. Mirrors the pattern in
# ``queue_executor._RAW_OUTPUT_ERROR_CODE``.
RUN_BATCH_EMPTY = "run_batch_empty"
RUN_BATCH_MIXED_OP = "run_batch_mixed_op"
RUN_BATCH_UNKNOWN_OP = "run_batch_unknown_op"
RUN_BATCH_INVALID_ITEM = "run_batch_invalid_item"


def _batch_profile(req: RunBatchRequest) -> str:
    """Return the release profile for a homogeneous batch, else ``other``."""
    profiles: set[str] = set()
    for item in req.items:
        payload = item.encode or item.score or item.extract
        if payload is not None:
            profiles.add(str(payload.profile_id or "default"))
    return profiles.pop() if len(profiles) == 1 else "other"


def _record_run_batch_metrics(req: RunBatchRequest, result: BatchOutcome, duration_s: float) -> None:
    """Emit one engine completion event per outcome, never sidecar queue/batch."""
    if not worker_telemetry_enabled():
        return
    operations = {item.op for item in req.items}
    operation = operations.pop() if len(operations) == 1 else "other"
    profile = _batch_profile(req)
    for outcome in result.outcomes:
        managed_outcome = (
            "success"
            if outcome.disposition == "publish_and_ack"
            else "retry"
            if outcome.disposition == "nak_retry"
            else "error"
        )
        units = None
        if managed_outcome == "success" and outcome.units is not None:
            units = {
                key: value
                for key, value in (
                    ("input_tokens", outcome.units.input_tokens),
                    ("pages", outcome.units.pages),
                    ("images", outcome.units.images),
                )
                if value is not None
            }
        worker_telemetry().item_completed(
            operation=operation,
            outcome=managed_outcome,
            model=req.model_id,
            profile=profile,
            duration_s=duration_s,
            tokenization_s=(outcome.tokenization_ms / 1000.0 if outcome.tokenization_ms is not None else None),
            inference_s=(outcome.inference_ms / 1000.0 if outcome.inference_ms is not None else None),
            postprocessing_s=(outcome.postprocessing_ms / 1000.0 if outcome.postprocessing_ms is not None else None),
            units=units,
        )


def _merge_lora_into_options(
    existing: dict[str, Any] | None,
    lora_key: str,
    *,
    item_id: str,
    batch_id: int,
    op: str,
) -> dict[str, Any] | None:
    """Return an options dict with ``options["lora"] = lora_key`` set.

    Matches the ``ModelWorker.submit_preformed`` contract where the
    batcher selection is sourced from ``options.get("lora")``. Returns
    ``existing`` unchanged when ``lora_key`` is empty (the base-model
    case — leaving ``options["lora"]`` unset rather than setting it to
    ``None`` keeps the two paths byte-identical).

    Pre-existing per-item ``options["lora"]`` is never silently
    overwritten: if it conflicts with the batch-level key we log and
    trust the per-item value. The Rust scheduler never produces this
    combination today (it partitions by lora_key before forming the
    batch), so any conflict is a real upstream bug worth a WARN.
    """
    if not lora_key:
        return existing

    if existing is None:
        return {"lora": lora_key}

    prior = existing.get("lora")
    if prior is not None and prior != lora_key:
        logger.warning(
            "run_batch (%s) item=%r batch_id=%d has options['lora']=%r but "
            "RunBatchRequest.lora_key=%r — keeping per-item value; the Rust "
            "scheduler should not produce mixed-LoRA batches",
            op,
            item_id,
            batch_id,
            prior,
            lora_key,
        )
        return existing

    if prior == lora_key:
        # Already set to the same value — no copy needed.
        return existing

    # Avoid mutating the caller's dict. The msgspec deserialisation
    # produces a fresh dict per call so this copy is cheap (a handful
    # of items per batch in practice) and keeps the function pure.
    merged = dict(existing)
    merged["lora"] = lora_key
    return merged


async def handle_run_batch(
    executor: QueueExecutor,
    req: RunBatchRequest,
) -> BatchOutcome:
    """Dispatch a Rust-formed batch into the existing per-op handlers.

    Contract:
        * ``req.items`` must be non-empty and homogeneous in ``op``.
        * ``req.model_id`` must be a model the executor can serve —
          the gateway is responsible for only routing models served
          by this worker image to the worker-sidecar; we don't
          re-validate here because a model reaching this RPC means
          the gateway has already decided we can serve it.

    Returns a :class:`BatchOutcome` matching the per-op RPCs' shape
    so the Rust publisher path is identical regardless of which RPC
    produced the result (``process_*_batch`` or ``RunBatch``).
    """
    telemetry_enabled = worker_telemetry_enabled()
    started = time.perf_counter() if telemetry_enabled else None
    if not req.items:
        logger.warning(
            "run_batch received empty items list (model=%s, batch_id=%d, lora=%r)",
            req.model_id,
            req.batch_id,
            req.lora_key,
        )
        # Empty batch is a protocol violation on the Rust side —
        # scheduler should never flush zero items. Return an empty
        # BatchOutcome; the Rust publisher treats this as a no-op.
        return BatchOutcome(outcomes=[])

    op = req.items[0].op
    if any(item.op != op for item in req.items):
        logger.error(
            "run_batch got mixed-op batch (model=%s, batch_id=%d): "
            "ops=%s — Rust scheduler MUST emit homogeneous-op batches",
            req.model_id,
            req.batch_id,
            sorted({i.op for i in req.items}),
        )
        result = _reject_all(req, RUN_BATCH_MIXED_OP, "mixed-op run_batch is not supported")
        if started is not None:
            _record_run_batch_metrics(req, result, time.perf_counter() - started)
        return result

    # W3C Trace Context propagation (mirrors the streaming path in
    # ``processors/streaming.py``). The gateway serialises its span into
    # each work item's ``traceparent`` / ``tracestate``; the Rust
    # scheduler copies those onto the wire items. We re-extract them here
    # so ``worker.run_batch`` attaches to the gateway span instead of
    # rooting a fresh trace. A batch can fan in items from several
    # requests (each with its own trace) — we parent the span to the
    # first distinct context and add OTel links to the rest.
    parent_ctx, links = _extract_batch_trace_context(req.items)
    tracer = trace.get_tracer("sie_server.adapter_call_loop")
    with tracer.start_as_current_span(
        "worker.run_batch",
        context=parent_ctx,
        links=links,
        attributes={
            "sie.model": req.model_id,
            "sie.batch_id": req.batch_id,
            "sie.op": op,
            "sie.adapter": "run_batch",
            "sie.batch_size": len(req.items),
        },
    ):
        if op == "encode":
            result = await _dispatch_encode(executor, req)
        elif op == "score":
            result = await _dispatch_score(executor, req)
        elif op == "extract":
            result = await _dispatch_extract(executor, req)
        else:
            logger.error(
                "run_batch got unknown op=%r (model=%s, batch_id=%d)",
                op,
                req.model_id,
                req.batch_id,
            )
            result = _reject_all(req, RUN_BATCH_UNKNOWN_OP, f"unknown op {op!r}")

        if started is not None:
            _record_run_batch_metrics(req, result, time.perf_counter() - started)
        return result


def _extract_batch_trace_context(
    items: list[RunBatchItem],
) -> tuple[Context, list[trace.Link]]:
    """Resolve the parent context + span links for a ``worker.run_batch`` span.

    A run_batch holds N items, each potentially from a different request
    with its own ``traceparent`` / ``tracestate``. We:

    * collect the DISTINCT ``(traceparent, tracestate)`` pairs, preserving
      first-seen order;
    * :func:`opentelemetry.propagate.extract` each into a parent
      :class:`~opentelemetry.context.Context` and keep only the ones that
      carry a *valid* span context;
    * return the FIRST valid context as the parent and OTel
      :class:`~opentelemetry.trace.Link` s to the remaining valid ones so
      the fan-in is visible in the trace graph.

    When no item carries a valid trace context, the empty context from
    ``propagate.extract({})`` is returned (a safe no-op) so the caller
    opens a root span.
    """
    distinct: list[tuple[str | None, str | None]] = []
    for item in items:
        key = (item.traceparent, item.tracestate)
        if key not in distinct:
            distinct.append(key)

    valid_ctxs: list[Context] = []
    for traceparent, tracestate in distinct:
        carrier: dict[str, str] = {}
        if traceparent is not None:
            carrier["traceparent"] = traceparent
        if tracestate is not None:
            carrier["tracestate"] = tracestate
        ctx = propagate.extract(carrier)
        if trace.get_current_span(ctx).get_span_context().is_valid:
            valid_ctxs.append(ctx)

    if not valid_ctxs:
        # No valid parent — open a root span. ``propagate.extract({})``
        # yields an empty context that ``start_as_current_span`` treats
        # as "no parent" (root).
        return propagate.extract({}), []

    parent_ctx = valid_ctxs[0]
    links = [trace.Link(trace.get_current_span(ctx).get_span_context()) for ctx in valid_ctxs[1:]]
    return parent_ctx, links


# --------------------------------------------------------------------
# Per-op dispatchers
# --------------------------------------------------------------------
#
# Each dispatcher validates that every RunBatchItem has its matching
# payload populated, then unwraps into the existing Process*Request
# type and calls the same method the per-op RPC already uses.
# Any validation miss on a single item becomes a typed error on that
# item only — the rest of the batch proceeds.


async def _dispatch_encode(
    executor: QueueExecutor,
    req: RunBatchRequest,
) -> BatchOutcome:
    items: list[EncodeBatchItem] = []
    invalid: list[ItemOutcome] = []
    for ri in req.items:
        if ri.encode is None:
            invalid.append(
                _invalid_item_outcome(
                    request_id=ri.request_id,
                    work_item_id=ri.work_item_id,
                    item_index=ri.item_index,
                    reason="op=encode but encode payload missing",
                )
            )
            continue
        # Plumb RunBatchRequest.lora_key into the per-item options so
        # ``ModelWorker.submit_preformed`` sets the adapter's active LoRA
        # immediately before the forward pass.
        merged_options = _merge_lora_into_options(
            ri.encode.options,
            req.lora_key,
            item_id=ri.encode.work_item_id,
            batch_id=req.batch_id,
            op="encode",
        )
        if merged_options is ri.encode.options:
            items.append(ri.encode)
        else:
            # ``msgspec.Struct.replace``-equivalent: build a new struct
            # rather than mutating the deserialised one. We only enter
            # this branch when an injection actually happens, so the
            # base-LoRA hot path stays allocation-free.
            items.append(_with_options(ri.encode, merged_options))

    if not items:
        return BatchOutcome(outcomes=invalid)

    inner = ProcessEncodeBatchRequest(
        model_id=req.model_id,
        items=items,
        accepts_batched_f16_multivectors=req.accepts_batched_f16_multivectors,
    )
    outcome = await executor.process_encode_batch(inner)
    return BatchOutcome(
        outcomes=[*invalid, *outcome.outcomes],
        batched_f16_multivectors=outcome.batched_f16_multivectors,
    )


async def _dispatch_score(
    executor: QueueExecutor,
    req: RunBatchRequest,
) -> BatchOutcome:
    # Score is base-only: ``ModelWorker.submit_score_preformed`` ignores
    # LoRA. We "downgrade to WARN, serve base" here
    # so a misrouted batch produces a visible log line rather than a
    # silent drop or a NAK that confuses retry logic.
    if req.lora_key:
        logger.warning(
            "run_batch op=score for model=%s arrived with lora_key=%r — "
            "score has no LoRA support (ModelWorker.submit_score_preformed), "
            "serving base weights for batch_id=%d",
            req.model_id,
            req.lora_key,
            req.batch_id,
        )

    items = []
    invalid: list[ItemOutcome] = []
    for ri in req.items:
        if ri.score is None:
            invalid.append(
                _invalid_item_outcome(
                    request_id=ri.request_id,
                    work_item_id=ri.work_item_id,
                    item_index=ri.item_index,
                    reason="op=score but score payload missing",
                )
            )
            continue
        items.append(ri.score)

    if not items:
        return BatchOutcome(outcomes=invalid)

    inner = ProcessScoreBatchRequest(model_id=req.model_id, items=items)
    outcome = await executor.process_score_batch(inner)
    return BatchOutcome(outcomes=[*invalid, *outcome.outcomes])


async def _dispatch_extract(
    executor: QueueExecutor,
    req: RunBatchRequest,
) -> BatchOutcome:
    items: list[ExtractBatchItem] = []
    invalid: list[ItemOutcome] = []
    for ri in req.items:
        if ri.extract is None:
            invalid.append(
                _invalid_item_outcome(
                    request_id=ri.request_id,
                    work_item_id=ri.work_item_id,
                    item_index=ri.item_index,
                    reason="op=extract but extract payload missing",
                )
            )
            continue
        # Same reasoning as ``_dispatch_encode``: plumb the batch-level
        # lora_key into ``options["lora"]`` so the pre-formed ModelWorker
        # path selects the right LoRA for this IPC batch.
        merged_options = _merge_lora_into_options(
            ri.extract.options,
            req.lora_key,
            item_id=ri.extract.work_item_id,
            batch_id=req.batch_id,
            op="extract",
        )
        if merged_options is ri.extract.options:
            items.append(ri.extract)
        else:
            items.append(_with_options(ri.extract, merged_options))

    if not items:
        return BatchOutcome(outcomes=invalid)

    inner = ProcessExtractBatchRequest(model_id=req.model_id, items=items)
    outcome = await executor.process_extract_batch(inner)
    return BatchOutcome(outcomes=[*invalid, *outcome.outcomes])


def _with_options[OptionsItemT: (EncodeBatchItem, ExtractBatchItem)](
    item: OptionsItemT,
    options: dict[str, Any] | None,
) -> OptionsItemT:
    """Return a copy of ``item`` with ``options`` replaced.

    Uses ``msgspec.structs.replace`` semantics — preserves every other
    field byte-for-byte. We avoid mutating the input struct because the
    deserialised request may be referenced by RPC framing/logging
    helpers downstream and silent mutation is a debugging hazard.

    Generic over the two item structs that actually carry an
    ``options`` field so callers preserve their narrowed type
    (``items.append(_with_options(ri.encode, ...))`` keeps
    ``list[EncodeBatchItem]``).
    """
    import msgspec.structs  # noqa: PLC0415  (local import keeps the cold path cold)

    return msgspec.structs.replace(item, options=options)


# --------------------------------------------------------------------
# Error helpers
# --------------------------------------------------------------------


def _reject_all(req: RunBatchRequest, error_code: str, error: str) -> BatchOutcome:
    """Produce a BatchOutcome that fails every item uniformly.

    Used for batch-level protocol errors (empty, mixed-op, unknown
    op). Every item's disposition is ``publish_error_and_ack``; the
    Rust publisher surfaces the error_code on the wire so the SDK
    sees a structured error rather than a timeout.
    """
    outcomes: list[ItemOutcome] = []
    for idx, ri in enumerate(req.items):
        work_item_id = ri.work_item_id
        request_id = ri.request_id
        item_index = getattr(ri, "item_index", idx)
        # Pull the id out of whichever payload is populated — we do not know
        # which, because the whole point of this path is the protocol error.
        # Prefer the wrapped payload when present so malformed frames still get
        # useful IDs.
        payload = ri.encode or ri.score or ri.extract
        if payload is not None:
            work_item_id = payload.work_item_id
            request_id = payload.request_id
            item_index = payload.item_index
        outcomes.append(
            ItemOutcome(
                work_item_id=work_item_id,
                request_id=request_id,
                item_index=item_index,
                disposition="publish_error_and_ack",
                error=error,
                error_code=error_code,
            )
        )
    return BatchOutcome(outcomes=outcomes)


def _invalid_item_outcome(
    *,
    request_id: str,
    work_item_id: str,
    item_index: int,
    reason: str,
) -> ItemOutcome:
    """Single-item error outcome for op-tag / payload mismatches."""
    return ItemOutcome(
        work_item_id=work_item_id,
        request_id=request_id,
        item_index=item_index,
        disposition="publish_error_and_ack",
        error=reason,
        error_code=RUN_BATCH_INVALID_ITEM,
    )
