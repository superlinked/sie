"""Standalone IPC test harness — boots ``IpcServer`` with a stub executor.

This module exists solely for the worker-sidecar's integration smoke test
(``packages/sie_server_sidecar/tests/integration_smoke.rs``). It is **not**
used by production code paths and must not import any GPU / torch / model
dependencies — the whole point is to prove the UDS wire contract without
needing a real ``sie-server`` with its heavy dependency graph.

Run as::

    python -m sie_server._ipc_test_harness --socket /tmp/ipc.sock --worker-id w1

The process exits on SIGINT / SIGTERM. All stubbed RPCs:

* ``Ping`` — handled by ``IpcServer`` itself.
* ``EnsureModelReady`` — always returns ``"ready"``.
* ``ProcessEncodeBatch`` / ``ProcessScoreBatch`` / ``ProcessExtractBatch`` —
  returns ``publish_and_ack`` for every item with a canned ``result_msgpack``
  payload that the test can decode and assert on.
* ``ProcessGenerate`` — when ``--fake-generate-model`` is set, streams
  ``in_progress`` → ``publish`` → ``ack`` events through the IPC generation
  protocol without loading a real generation adapter.
* ``SetPinnedModels`` — records the requested pinned set and returns the applied
  count so the sidecar pinned reconciler can exercise the IPC method.
* ``Drain`` — handled by ``IpcServer`` itself.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from typing import Any, cast

import msgpack

from sie_server.ipc_server import IpcServer
from sie_server.ipc_types import (
    BatchOutcome,
    ItemOutcome,
    ModelDescriptor,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    ReadinessState,
    SetPinnedModelsRequest,
    SetPinnedModelsResponse,
)

logger = logging.getLogger(__name__)


# Fixed canned result the Rust smoke test can pattern-match on.
_CANNED_RESULT = {"smoke": "ok", "source": "ipc_test_harness"}
_IPC_TEST_RESULT_BYTES_OPTION = "ipc_test_result_bytes"
_MAX_IPC_TEST_RESULT_BYTES = 40 * 1024 * 1024


def _canned_result_bytes() -> bytes:
    return msgpack.packb(_CANNED_RESULT, use_bin_type=True)


def _requested_test_result_bytes(item: Any) -> int:
    """Return the bounded test-only result size requested on an IPC item."""
    options = getattr(item, "options", None)
    if not isinstance(options, dict) or _IPC_TEST_RESULT_BYTES_OPTION not in options:
        return 0
    value = options[_IPC_TEST_RESULT_BYTES_OPTION]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{_IPC_TEST_RESULT_BYTES_OPTION} must be an integer")
    if not 0 <= value <= _MAX_IPC_TEST_RESULT_BYTES:
        raise ValueError(f"{_IPC_TEST_RESULT_BYTES_OPTION} must be between 0 and {_MAX_IPC_TEST_RESULT_BYTES}")
    return value


class _StubGenerationTasks:
    def __init__(self, enabled: bool) -> None:
        self.generate = object() if enabled else None


class _StubGenerationConfig:
    def __init__(self, enabled: bool) -> None:
        self.tasks = _StubGenerationTasks(enabled)


class _StubRegistry:
    def __init__(
        self,
        generation_model: str | None = None,
        generation_hidden_polls: int = 0,
    ) -> None:
        self._generation_model = generation_model
        self._generation_hidden_polls = generation_hidden_polls
        self._generation_capability_polls = 0

    def get_configs_snapshot(self) -> dict[str, _StubGenerationConfig]:
        if self._generation_model is None:
            return {}
        self._generation_capability_polls += 1
        if self._generation_capability_polls <= self._generation_hidden_polls:
            return {}
        return {self._generation_model: _StubGenerationConfig(enabled=True)}


class _StubExecutor:
    """Duck-typed replacement for :class:`sie_server.queue_executor.QueueExecutor`.

    We only need the methods ``IpcServer`` actually calls. The test asserts
    on the outcome shape, not on any real inference output.

    The optional ``per_request_delay_ms`` knob injects an ``asyncio.sleep`` at
    the top of every ``process_*_batch`` call. It exists purely so the Rust
    pool integration test can *observe* concurrency: with fast RPCs the pool
    returns slots faster than Prometheus can scrape ``ipc_pool_inflight``,
    making the metric invisible to real-time assertions. A 100ms delay gives
    the test a comfortable window to scrape ``> 1`` in-flight.
    """

    def __init__(
        self,
        readiness: ReadinessState = "ready",
        batch_budget: int | None = None,
        descriptor: ModelDescriptor | None = None,
        per_request_delay_ms: float = 0.0,
        generation_model: str | None = None,
        generation_hidden_polls: int = 0,
    ) -> None:
        self._readiness = readiness
        self._batch_budget = batch_budget
        self._descriptor = descriptor
        self._per_request_delay_ms = per_request_delay_ms
        self._pinned_models: frozenset[str] = frozenset()
        self.registry = _StubRegistry(generation_model, generation_hidden_polls)

    async def ensure_model_ready(self, model_id: str) -> ReadinessState:  # noqa: ARG002
        return self._readiness

    def get_batch_budget(self, model_id: str) -> int | None:  # noqa: ARG002
        return self._batch_budget

    def get_model_descriptor(self, model_id: str) -> ModelDescriptor | None:  # noqa: ARG002
        return self._descriptor

    def loaded_model_names(self) -> list[str]:
        return []

    async def set_pinned_models(self, req: SetPinnedModelsRequest) -> SetPinnedModelsResponse:
        self._pinned_models = frozenset(model.strip() for model in req.models if model.strip())
        return SetPinnedModelsResponse(applied=True, pinned_count=len(self._pinned_models))

    async def _maybe_sleep(self) -> None:
        if self._per_request_delay_ms > 0:
            await asyncio.sleep(self._per_request_delay_ms / 1000.0)

    async def process_encode_batch(self, req: ProcessEncodeBatchRequest) -> BatchOutcome:
        await self._maybe_sleep()
        return _canned_batch_outcome_echoing_prepared_tokens(req.items)

    async def process_score_batch(self, req: ProcessScoreBatchRequest) -> BatchOutcome:
        await self._maybe_sleep()
        return _canned_batch_outcome_echoing_prepared_tokens(req.items)

    async def process_extract_batch(self, req: ProcessExtractBatchRequest) -> BatchOutcome:
        await self._maybe_sleep()
        return _canned_extract_batch_outcome(req.items)


class _FakeGenerateProcessor:
    def signal_cancel(self, request_id: str) -> bool:  # noqa: ARG002
        return False

    async def process(self, msg: Any, model_id: str) -> None:
        work_item = msgpack.unpackb(msg.data, raw=False)
        await msg.in_progress()
        payload = msgpack.packb(
            {
                "smoke": "generate",
                "source": "ipc_test_harness",
                "model_id": model_id,
                "request_id": work_item.get("request_id"),
                "work_item_id": work_item.get("work_item_id"),
            },
            use_bin_type=True,
        )
        await msg._sink.publish(work_item["reply_subject"], payload)
        await msg.ack()


def _canned_batch_outcome(items: list[Any]) -> BatchOutcome:
    payload = _canned_result_bytes()
    return BatchOutcome(
        outcomes=[
            ItemOutcome(
                work_item_id=item.work_item_id,
                request_id=item.request_id,
                item_index=item.item_index,
                disposition="publish_and_ack",
                result_msgpack=payload,
                inference_ms=0.1,
                tokenization_ms=0.05,
                postprocessing_ms=0.01,
            )
            for item in items
        ]
    )


def _extract_document_echo(item: Any) -> dict[str, Any]:
    document = item.item.get("document") if isinstance(item.item, dict) else None
    if not isinstance(document, dict):
        return {
            "present": False,
            "data_is_bytes": False,
            "data": b"",
            "data_len": 0,
            "format": None,
        }

    data = document.get("data")
    data_is_bytes = isinstance(data, bytes | bytearray)
    data_bytes = bytes(data) if data_is_bytes else b""
    return {
        "present": True,
        "data_is_bytes": data_is_bytes,
        "data": data_bytes,
        "data_len": len(data_bytes),
        "format": document.get("format"),
    }


def _canned_extract_batch_outcome(items: list[Any]) -> BatchOutcome:
    outcomes: list[ItemOutcome] = []
    for item in items:
        payload = msgpack.packb(
            {**_CANNED_RESULT, "extract_document": _extract_document_echo(item)},
            use_bin_type=True,
        )
        outcomes.append(
            ItemOutcome(
                work_item_id=item.work_item_id,
                request_id=item.request_id,
                item_index=item.item_index,
                disposition="publish_and_ack",
                result_msgpack=payload,
                inference_ms=0.1,
                tokenization_ms=0.05,
                postprocessing_ms=0.01,
            )
        )
    return BatchOutcome(outcomes=outcomes)


def _canned_batch_outcome_echoing_prepared_tokens(items: list[Any]) -> BatchOutcome:
    """Like :func:`_canned_batch_outcome` but folds each item's
    ``prepared_tokens`` presence / content into the per-item
    ``result_msgpack``. The Rust integration smoke test uses this to
    assert that the Rust tokenise-and-attach hook actually fired
    for the request without needing to run a real adapter.

    Shape of the echoed result::

        {
            "smoke": "ok",
            "source": "ipc_test_harness",
            "prepared_tokens": {
                "present": bool,
                "tokenizer_id": str | None,
                "input_ids_first_seq": list[int] | None,
                "max_seq_len": int | None,
            },
        }
    """
    outcomes: list[ItemOutcome] = []
    for item in items:
        pt = getattr(item, "prepared_tokens", None)
        if pt is not None:
            pt_echo: dict[str, Any] = {
                "present": True,
                "tokenizer_id": pt.tokenizer_id,
                "input_ids_first_seq": list(pt.input_ids[0]) if pt.input_ids else None,
                "max_seq_len": int(pt.max_seq_len),
            }
        else:
            pt_echo = {
                "present": False,
                "tokenizer_id": None,
                "input_ids_first_seq": None,
                "max_seq_len": None,
            }
        payload_shape: dict[str, Any] = {**_CANNED_RESULT, "prepared_tokens": pt_echo}
        requested_bytes = _requested_test_result_bytes(item)
        if requested_bytes:
            payload_shape["test_blob"] = b"x" * requested_bytes
        payload = msgpack.packb(payload_shape, use_bin_type=True)
        outcomes.append(
            ItemOutcome(
                work_item_id=item.work_item_id,
                request_id=item.request_id,
                item_index=item.item_index,
                disposition="publish_and_ack",
                result_msgpack=payload,
                inference_ms=0.1,
                tokenization_ms=0.05,
                postprocessing_ms=0.01,
            )
        )
    return BatchOutcome(outcomes=outcomes)


async def _main(
    socket_path: str,
    worker_id: str,
    readiness: ReadinessState,
    per_request_delay_ms: float,
    descriptor: ModelDescriptor | None,
    fake_generate_model: str | None,
    fake_generate_hidden_polls: int,
) -> None:
    executor = _StubExecutor(
        readiness=readiness,
        descriptor=descriptor,
        per_request_delay_ms=per_request_delay_ms,
        generation_model=fake_generate_model,
        generation_hidden_polls=fake_generate_hidden_polls,
    )
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows / some embedded runtimes
            pass

    server = IpcServer(socket_path, cast("Any", executor), worker_id=worker_id)
    if fake_generate_model is not None:
        server._streaming_processor = _FakeGenerateProcessor()

    async with server:
        # Ready marker the test can grep for on stdout.
        print(f"HARNESS_READY socket={socket_path}", flush=True)
        await stop.wait()


def main() -> None:
    parser = argparse.ArgumentParser(prog="sie_server._ipc_test_harness")
    parser.add_argument("--socket", required=True, help="UDS path to bind")
    parser.add_argument("--worker-id", default="smoke-worker", help="worker_id to report on Ping")
    parser.add_argument(
        "--readiness",
        default="ready",
        choices=["ready", "loading_started", "loading_in_progress", "retry_later", "failed"],
        help="Canned readiness state returned for every EnsureModelReady",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--per-request-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Inject this many milliseconds of asyncio.sleep at the top of every "
            "process_*_batch RPC. Used by Rust pool integration tests to make "
            "concurrent in-flight RPCs observable on `ipc_pool_inflight` / "
            "`sie_worker_inflight_batches`; set to 0 in production-shaped tests."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Optional tokenizer.json path to expose through ModelDescriptor",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=None,
        help="Optional tokenizer content hash to expose through ModelDescriptor",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Optional max sequence length to expose through ModelDescriptor",
    )
    parser.add_argument(
        "--fake-generate-model",
        default=None,
        help=(
            "Expose this model as generation-capable and handle ProcessGenerate "
            "with a fake streaming processor for sidecar smoke tests."
        ),
    )
    parser.add_argument(
        "--fake-generate-hidden-polls",
        type=int,
        default=0,
        help=(
            "Return no generation models for this many WorkerCapabilities polls "
            "before exposing --fake-generate-model. Used to exercise sidecar "
            "activation after live config reconciliation."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    descriptor = None
    if args.tokenizer_path:
        descriptor = ModelDescriptor(
            tokenizer_path=args.tokenizer_path,
            tokenizer_id=args.tokenizer_id,
            max_seq_len=args.max_seq_len,
        )

    asyncio.run(
        _main(
            args.socket,
            args.worker_id,
            args.readiness,
            args.per_request_delay_ms,
            descriptor,
            args.fake_generate_model,
            args.fake_generate_hidden_polls,
        )
    )


if __name__ == "__main__":
    main()
