"""Rust output-framing tests for ``QueueExecutor``.

The gate is per-request: ``_maybe_*_raw_output`` returns a typed
``RawOutput`` whenever the adapter's response is the exact single-key,
supported-dtype, well-shaped form the Rust framing path knows how to encode
byte-identically; otherwise the legacy ``msgpack.packb`` path runs. See
the sidecar runtime architecture guide at
``packages/sie_server_sidecar/docs/architecture-guide.md`` (component/crate
name ``sie_server_sidecar``) for the full rationale.

These tests pin:

* The per-helper safety rules (single output key, supported dtype,
  multi-output items fall back, dim-mismatch falls back, ...).
* End-to-end wiring of ``process_encode_batch`` /
  ``process_score_batch``: dense / sparse / multivector / score requests
  that meet the safety rules emit a typed ``RawOutput``. Multi-output /
  unsupported dtype paths continue to fall back to ``result_msgpack``.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack_numpy
import numpy as np
import pytest
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.timing import RequestTiming
from sie_server.core.worker.types import WorkerResult
from sie_server.ipc_types import (
    EncodeBatchItem,
    ProcessEncodeBatchRequest,
    ProcessScoreBatchRequest,
    ScoreBatchItem,
)
from sie_server.queue_executor import (
    QueueExecutor,
    _maybe_dense_raw_output,
    _maybe_multivector_raw_output,
    _maybe_sparse_raw_output,
)

# The Python-framed reference path (``msgpack.packb(output, use_bin_type=True)``
# inside ``QueueExecutor.process_encode_batch``) relies on
# ``msgpack_numpy.patch()`` being active to serialise the ``np.ndarray``
# ``dense``/``sparse``/``multivector`` payloads. Production wires this
# through ``sie_server.api.serialization`` at import time. This test
# module exercises that path directly, so we activate the patch here
# as well — a no-op when another module already did it.
msgpack_numpy.patch()

MODEL_ID = "test/model"


def _make_registry() -> MagicMock:
    reg = MagicMock()
    reg.model_names = [MODEL_ID]
    reg.device = "cpu"
    reg.is_loaded.return_value = True
    reg.is_loading.return_value = False
    config = MagicMock()
    config.sie_id = MODEL_ID
    config.outputs = ["dense", "sparse"]
    config.resolve_profile.return_value.runtime = {}
    config.tasks.encode.dense.dim = 4
    reg.get_config.return_value = config
    return reg


def _encode_item(
    wiid: str = "req-1.0",
    *,
    output_types: list[str] | None = None,
    item_index: int = 0,
) -> EncodeBatchItem:
    return EncodeBatchItem(
        work_item_id=wiid,
        request_id=wiid.split(".", maxsplit=1)[0],
        item_index=item_index,
        total_items=1,
        timestamp=time.time(),
        item={"text": "hello"},
        output_types=output_types,
    )


# -----------------------------------------------------------------------------
# Dense fast-path gate (per-request shape rules)
# -----------------------------------------------------------------------------


class TestDenseFastPathGate:
    def _config(self, dim: int = 4) -> MagicMock:
        cfg = MagicMock()
        cfg.tasks.encode.dense.dim = dim
        return cfg

    def test_float32_dense_is_eligible(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        raw = _maybe_dense_raw_output({"dense": arr}, self._config(), ["dense"])
        assert raw is not None
        assert raw.dense is not None
        assert raw.dense.dim == 4
        assert raw.dense.values == [1.0, 2.0, 3.0, 4.0]
        assert raw.dense.normalize is False

    def test_non_float32_dense_is_skipped(self) -> None:
        for dtype in (np.float16, np.int8, np.uint8):
            arr = np.ones(4, dtype=dtype)
            assert _maybe_dense_raw_output({"dense": arr}, self._config(), ["dense"]) is None

    def test_sparse_or_multivector_disables_fastpath(self) -> None:
        dense = np.ones(4, dtype=np.float32)
        formatted = {
            "dense": dense,
            "sparse": {"indices": np.array([0, 1]), "values": np.array([0.1, 0.2], dtype=np.float32)},
        }
        assert _maybe_dense_raw_output(formatted, self._config(), ["dense", "sparse"]) is None

    def test_multiple_output_types_disables_fastpath(self) -> None:
        arr = np.ones(4, dtype=np.float32)
        assert _maybe_dense_raw_output({"dense": arr}, self._config(), ["dense", "multivector"]) is None

    def test_dim_mismatch_falls_back(self) -> None:
        # Adapter returned 5 dims but config says 4 — do NOT take the
        # fast path; let the legacy path mis-label consistently with
        # the existing Python code.
        arr = np.ones(5, dtype=np.float32)
        assert _maybe_dense_raw_output({"dense": arr}, self._config(dim=4), ["dense"]) is None

    def test_non_array_input_is_skipped(self) -> None:
        # The mocked EncodeHandler in tests sometimes emits plain lists;
        # the fast-path must not crash, it must fall back silently.
        assert _maybe_dense_raw_output({"dense": [0.1, 0.2, 0.3, 0.4]}, self._config(), ["dense"]) is None


# -----------------------------------------------------------------------------
# process_encode_batch — Rust output framing is unconditional when safe
# -----------------------------------------------------------------------------


class TestProcessEncodeBatchRustOutputFraming:
    @pytest.mark.asyncio
    async def test_dense_emits_raw_output_by_default(self) -> None:
        """Dense output framing is on by default. Any single-output
        float32 dense item produces a typed ``RawOutput`` and leaves
        ``result_msgpack`` at ``None`` for the Rust publisher to fill in.
        """
        reg = _make_registry()
        ex = QueueExecutor(reg)

        arr = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=([{"dense": arr}], RequestTiming()),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id=MODEL_ID,
                    items=[_encode_item(output_types=["dense"])],
                ),
            )

        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        assert o.raw_output is not None
        assert o.result_msgpack is None
        assert o.raw_output.dense is not None
        dense = o.raw_output.dense
        assert dense.dim == 4
        assert np.array(dense.values, dtype=np.float32).tolist() == arr.tolist()
        assert dense.normalize is False

    @pytest.mark.asyncio
    async def test_multi_output_falls_back_to_legacy_path(self) -> None:
        """Per-request safety rule: an item requesting dense + sparse
        does NOT produce a typed ``RawOutput`` — the v1 wire contract
        is one variant per ``RawOutput``. The Rust publisher then sees
        the legacy ``result_msgpack`` and ships it through unchanged.
        """
        reg = _make_registry()
        ex = QueueExecutor(reg)

        dense = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        sparse = {
            "indices": np.array([1, 2], dtype=np.int64),
            "values": np.array([0.5, 0.6], dtype=np.float32),
        }
        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=([{"dense": dense, "sparse": sparse}], RequestTiming()),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id=MODEL_ID,
                    items=[_encode_item(output_types=["dense", "sparse"])],
                ),
            )

        o = outcome.outcomes[0]
        assert o.raw_output is None
        assert o.result_msgpack is not None


# -----------------------------------------------------------------------------
# process_score_batch — score is always Rust-frameable
# -----------------------------------------------------------------------------


class TestProcessScoreBatchRustOutputFraming:
    def _make_score_item(self, wiid: str = "req-1.0") -> ScoreBatchItem:
        return ScoreBatchItem(
            work_item_id=wiid,
            request_id=wiid.split(".", maxsplit=1)[0],
            item_index=0,
            total_items=1,
            timestamp=time.time(),
            query_item={"text": "q"},
            score_items=[
                {"text": "a", "id": "doc-a"},
                {"text": "b", "id": "doc-b"},
                {"text": "c", "id": "doc-c"},
            ],
        )

    async def _run_score(self, reg: MagicMock, scores: list[float]):
        worker = AsyncMock()
        score_output = ScoreOutput(scores=np.array(scores, dtype=np.float32))
        wr = WorkerResult(output=score_output, timing=RequestTiming())
        fut: asyncio.Future[WorkerResult] = asyncio.Future()
        fut.set_result(wr)
        worker.submit_score_preformed_batch = AsyncMock(return_value=[fut])
        reg.start_worker = AsyncMock(return_value=worker)

        ex = QueueExecutor(reg)
        return await ex.process_score_batch(
            ProcessScoreBatchRequest(model_id=MODEL_ID, items=[self._make_score_item()]),
        )

    @pytest.mark.asyncio
    async def test_score_emits_raw_parallel_arrays_in_input_order_unconditionally(
        self,
    ) -> None:
        """Score is always Rust-frameable. The Python adapter ships the
        unsorted parallel arrays; Rust does the descending stable sort
        + rank assignment in
        ``sie_server_sidecar::output::build_score_payload``.
        """
        outcome = await self._run_score(_make_registry(), [0.1, 0.9, 0.5])

        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        assert o.result_msgpack is None
        assert o.raw_output is not None
        assert o.raw_output.score is not None
        score = o.raw_output.score
        # Critical: input order is preserved; sort happens in Rust.
        assert score.item_ids == ["doc-a", "doc-b", "doc-c"]
        # ``float(np.float32(x))`` widening — Rust narrows back to f32
        # losslessly, and the production Python path uses the same
        # widening too (see ``_score_success_outcome``).
        assert score.scores == pytest.approx([0.1, 0.9, 0.5])


# -----------------------------------------------------------------------------
# Sparse fast-path gate (_maybe_sparse_raw_output) — per-request rules
# -----------------------------------------------------------------------------


class TestSparseFastPathGate:
    def _config(self, dim: int | None = 30522) -> MagicMock:
        cfg = MagicMock()
        cfg.tasks.encode.sparse.dim = dim
        return cfg

    def test_int_indices_float32_values_eligible(self) -> None:
        formatted = {
            "sparse": {
                "indices": np.array([3, 7, 42], dtype=np.int32),
                "values": np.array([0.5, 1.5, 2.5], dtype=np.float32),
            },
        }
        raw = _maybe_sparse_raw_output(formatted, self._config(), ["sparse"])
        assert raw is not None
        assert raw.sparse is not None
        assert raw.sparse.indices == [3, 7, 42]
        assert raw.sparse.values == [0.5, 1.5, 2.5]
        assert raw.sparse.dims == 30522

    def test_float16_values_falls_back(self) -> None:
        formatted = {
            "sparse": {
                "indices": np.array([1, 2], dtype=np.int32),
                "values": np.array([0.1, 0.2], dtype=np.float16),
            },
        }
        assert _maybe_sparse_raw_output(formatted, self._config(), ["sparse"]) is None

    def test_multi_output_falls_back(self) -> None:
        formatted = {
            "sparse": {
                "indices": np.array([1, 2], dtype=np.int32),
                "values": np.array([0.1, 0.2], dtype=np.float32),
            },
            "dense": np.ones(4, dtype=np.float32),
        }
        assert _maybe_sparse_raw_output(formatted, self._config(), ["sparse", "dense"]) is None


# -----------------------------------------------------------------------------
# Multivector fast-path gate (_maybe_multivector_raw_output) — per-request rules
# -----------------------------------------------------------------------------


class TestMultivectorFastPathGate:
    def _config(self, mv_dim: int = 4) -> MagicMock:
        cfg = MagicMock()
        cfg.tasks.encode.multivector.dim = mv_dim
        return cfg

    def test_float32_2d_array_eligible(self) -> None:
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        raw = _maybe_multivector_raw_output({"multivector": arr}, self._config(), ["multivector"])
        assert raw is not None
        assert raw.multivector is not None
        assert raw.multivector.num_tokens == 3
        assert raw.multivector.token_dims == 4

    def test_float16_2d_array_eligible(self) -> None:
        arr = np.zeros((2, 4), dtype=np.float16)
        raw = _maybe_multivector_raw_output({"multivector": arr}, self._config(), ["multivector"])
        assert raw is not None
        assert raw.multivector is not None
        assert raw.multivector.num_tokens == 2
        assert raw.multivector.token_dims == 4
        assert raw.multivector.dtype == "float16"

    def test_bit_packed_binary_falls_back(self) -> None:
        # ``shape[1] < mv_dim`` signals binary multivector packed into
        # bytes; framing in Rust isn't supported yet, must fall back.
        arr = np.zeros((3, 1), dtype=np.uint8)
        assert _maybe_multivector_raw_output({"multivector": arr}, self._config(mv_dim=8), ["multivector"]) is None


# -----------------------------------------------------------------------------
# process_encode_batch — sparse + multivector end-to-end
# -----------------------------------------------------------------------------


class TestProcessEncodeBatchSparseMV:
    def _sparse_registry(self, sparse_dim: int = 30522) -> MagicMock:
        reg = MagicMock()
        reg.model_names = [MODEL_ID]
        reg.device = "cpu"
        reg.is_loaded.return_value = True
        reg.is_loading.return_value = False
        config = MagicMock()
        config.sie_id = MODEL_ID
        config.outputs = ["sparse"]
        config.resolve_profile.return_value.runtime = {}
        config.tasks.encode.sparse.dim = sparse_dim
        reg.get_config.return_value = config
        return reg

    def _mv_registry(self, mv_dim: int = 4) -> MagicMock:
        reg = MagicMock()
        reg.model_names = [MODEL_ID]
        reg.device = "cpu"
        reg.is_loaded.return_value = True
        reg.is_loading.return_value = False
        config = MagicMock()
        config.sie_id = MODEL_ID
        config.outputs = ["multivector"]
        config.resolve_profile.return_value.runtime = {}
        config.tasks.encode.multivector.dim = mv_dim
        reg.get_config.return_value = config
        return reg

    @pytest.mark.asyncio
    async def test_sparse_emits_raw_output_unconditionally(self) -> None:
        reg = self._sparse_registry()
        ex = QueueExecutor(reg)

        formatted = {
            "sparse": {
                "indices": np.array([3, 7, 42], dtype=np.int32),
                "values": np.array([0.5, 1.5, 2.5], dtype=np.float32),
            },
        }
        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=([formatted], RequestTiming()),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id=MODEL_ID,
                    items=[_encode_item(output_types=["sparse"])],
                ),
            )

        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        assert o.result_msgpack is None
        assert o.raw_output is not None
        assert o.raw_output.sparse is not None
        assert o.raw_output.sparse.indices == [3, 7, 42]
        assert o.raw_output.sparse.dims == 30522

    @pytest.mark.asyncio
    async def test_multivector_emits_raw_output_unconditionally(self) -> None:
        reg = self._mv_registry()
        ex = QueueExecutor(reg)

        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=([{"multivector": arr}], RequestTiming()),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id=MODEL_ID,
                    items=[_encode_item(output_types=["multivector"])],
                ),
            )

        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        assert o.result_msgpack is None
        assert o.raw_output is not None
        assert o.raw_output.multivector is not None
        mv = o.raw_output.multivector
        assert mv.num_tokens == 3
        assert mv.token_dims == 4
        assert len(mv.values) == 12

    @pytest.mark.asyncio
    async def test_float16_multivector_emits_raw_output(self) -> None:
        """Float16 multivectors stay on RawOutput so the sidecar can
        write the compact f16 msgpack-numpy sentinel.
        """
        reg = self._mv_registry()
        ex = QueueExecutor(reg)

        arr = np.zeros((2, 4), dtype=np.float16)
        with patch(
            "sie_server.core.encode_pipeline.EncodePipeline.run_encode",
            new_callable=AsyncMock,
            return_value=([{"multivector": arr}], RequestTiming()),
        ):
            outcome = await ex.process_encode_batch(
                ProcessEncodeBatchRequest(
                    model_id=MODEL_ID,
                    items=[_encode_item(output_types=["multivector"])],
                ),
            )

        o = outcome.outcomes[0]
        assert o.disposition == "publish_and_ack"
        assert o.result_msgpack is None
        assert o.raw_output is not None
        assert o.raw_output.multivector is not None
        assert o.raw_output.multivector.dtype == "float16"


# -----------------------------------------------------------------------------
# Score legacy-path verification — the unsorted/sorted equivalence
# -----------------------------------------------------------------------------


class TestScoreLegacySortParity:
    """Score is always emitted as an unsorted ``RawOutput``.
    Pin the Rust-side sort+rank semantics by re-implementing the
    Python-framed fallback path in test code and asserting equivalence to the
    raw arrays — this is the contract Rust mirrors in
    ``sie_server_sidecar::output::build_score_payload``.
    """

    def test_legacy_sort_matches_input_arrays(self) -> None:
        item_ids = ["doc-a", "doc-b", "doc-c"]
        raw_scores = [0.1, 0.9, 0.5]

        # Legacy Python path (kept here as the spec the Rust framing mirrors).
        scored = list(zip(item_ids, raw_scores, strict=True))
        scored.sort(key=lambda x: x[1], reverse=True)
        legacy = [{"item_id": item_id, "score": sc, "rank": rank} for rank, (item_id, sc) in enumerate(scored)]
        assert [e["item_id"] for e in legacy] == ["doc-b", "doc-c", "doc-a"]
        assert [e["rank"] for e in legacy] == [0, 1, 2]


# -----------------------------------------------------------------------------
# ModelDescriptor handshake
# -----------------------------------------------------------------------------


class TestGetModelDescriptor:
    """Cover ``QueueExecutor.get_model_descriptor`` and tokenizer materialisation.

    The dispatcher re-handshakes on every batch so we pin two contracts:

    1. The descriptor carries the right tokenizer_id / max_seq_len.
    2. Subsequent calls hit the per-model cache (no extra file I/O,
       no extra calls into the preprocessor registry).
    """

    def _descriptor_registry(self, *, tokenizer_id: str, max_len: int, canonical: bytes) -> MagicMock:
        """Build a registry whose ``preprocessor_registry`` returns a
        preprocessor that mimics the production ``TextPreprocessor``
        contract: a ``tokenizer_id`` property returning a real ``str``
        and a ``_tokenizer.backend_tokenizer.to_str`` returning real
        canonical JSON bytes.
        """
        reg = MagicMock()
        reg.model_names = [MODEL_ID]
        reg.is_loaded.return_value = True
        reg.is_loading.return_value = False
        reg.get_worker.return_value = MagicMock()  # any non-None object satisfies the gate

        backend = MagicMock()
        backend.to_str.return_value = canonical.decode("utf-8")
        inner = MagicMock()
        inner.backend_tokenizer = backend
        inner.model_max_length = max_len

        preprocessor = MagicMock()
        type(preprocessor).tokenizer_id = property(lambda _self: tokenizer_id)
        preprocessor._tokenizer = inner

        reg.preprocessor_registry.get_preprocessor.return_value = preprocessor
        return reg

    def test_descriptor_carries_tokenizer_id_and_max_seq_len(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Point the staging dir at the test's tmp dir so we don't
        # litter ``$TMPDIR/sie-tokenizers`` and so different test
        # runs don't see each other's files.
        monkeypatch.setattr(
            "sie_server.queue_executor._TOKENIZER_STAGING_DIR",
            tmp_path,
        )
        canonical = b'{"version":"1.0","model":{"type":"WordLevel"}}'
        reg = self._descriptor_registry(tokenizer_id="abc123", max_len=512, canonical=canonical)
        ex = QueueExecutor(reg)

        descriptor = ex.get_model_descriptor(MODEL_ID)
        assert descriptor is not None
        assert descriptor.tokenizer_id == "abc123"
        assert descriptor.max_seq_len == 512
        assert descriptor.supports_run_batch is True
        assert descriptor.tokenizer_path is not None
        # The materialised file lives under the per-model dir and
        # holds the canonical bytes byte-for-byte. The sidecar will
        # hash this on load and reconcile against ``tokenizer_id``.
        assert Path(descriptor.tokenizer_path).read_bytes() == canonical

    def test_descriptor_is_cached_per_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            "sie_server.queue_executor._TOKENIZER_STAGING_DIR",
            tmp_path,
        )
        canonical = b'{"version":"1.0"}'
        reg = self._descriptor_registry(tokenizer_id="abc", max_len=128, canonical=canonical)
        ex = QueueExecutor(reg)

        first = ex.get_model_descriptor(MODEL_ID)
        second = ex.get_model_descriptor(MODEL_ID)
        # Same struct object → cache hit, no re-materialisation, no
        # re-call into the preprocessor registry. Three calls total
        # are allowed (other code paths in this test file may have
        # called it earlier on this fixture); the contract we pin is
        # "exactly one call by ``get_model_descriptor``".
        assert first is second
        # ``get_preprocessor`` should have been invoked exactly once
        # by the two ``get_model_descriptor`` calls above (the second
        # short-circuited at the cache).
        assert reg.preprocessor_registry.get_preprocessor.call_count == 1

    def test_invalidate_clears_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            "sie_server.queue_executor._TOKENIZER_STAGING_DIR",
            tmp_path,
        )
        canonical = b'{"version":"1.0"}'
        reg = self._descriptor_registry(tokenizer_id="v1", max_len=128, canonical=canonical)
        ex = QueueExecutor(reg)

        assert ex.get_model_descriptor(MODEL_ID) is not None
        ex.invalidate_model_descriptor(MODEL_ID)
        # After invalidation we expect a fresh registry call.
        ex.get_model_descriptor(MODEL_ID)
        assert reg.preprocessor_registry.get_preprocessor.call_count == 2

    def test_unsane_max_length_reads_as_unset(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """HF defaults ``model_max_length`` to ``int(1e30)`` when no
        cap is declared; the descriptor must report ``None`` rather
        than ship a bogus billion-token cap to the sidecar.
        """
        monkeypatch.setattr(
            "sie_server.queue_executor._TOKENIZER_STAGING_DIR",
            tmp_path,
        )
        reg = self._descriptor_registry(
            tokenizer_id="v1",
            max_len=int(1e30),
            canonical=b'{"version":"1.0"}',
        )
        ex = QueueExecutor(reg)
        descriptor = ex.get_model_descriptor(MODEL_ID)
        assert descriptor is not None
        assert descriptor.max_seq_len is None

    def test_returns_none_when_model_not_loaded(self) -> None:
        reg = MagicMock()
        reg.get_worker.return_value = None
        ex = QueueExecutor(reg)
        assert ex.get_model_descriptor("any/model") is None
