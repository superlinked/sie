from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from sie_server.adapters.bge_m3_flash import BGEM3FlashAdapter
from sie_server.config.model import ModelConfig
from sie_server.core.inference_output import EncodeOutput, SparseVector
from sie_server.types.inputs import Item

# bge_m3_flash requires CUDA at load() time. The score() / score_pairs() unit
# tests in this module bypass load() by:
#   * setting ``adapter._model = MagicMock()`` so ``_check_loaded()`` passes,
#   * monkey-patching ``adapter.encode`` to return synthetic ``EncodeOutput``
#     fixtures that exercise the dense / sparse / multivector code paths.
# This keeps the tests CPU-only and focused on the scoring math.

_RNG = np.random.default_rng(0)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    return vec / norm


def _make_dense_output(vectors: np.ndarray) -> EncodeOutput:
    return EncodeOutput(
        dense=vectors.astype(np.float32),
        batch_size=vectors.shape[0],
        dense_dim=vectors.shape[1],
    )


def _make_sparse_output(weights: list[dict[int, float]]) -> EncodeOutput:
    sparse = []
    for w in weights:
        if w:
            indices = np.array(sorted(w.keys()), dtype=np.int32)
            values = np.array([w[k] for k in indices], dtype=np.float32)
        else:
            indices = np.array([], dtype=np.int32)
            values = np.array([], dtype=np.float32)
        sparse.append(SparseVector(indices=indices, values=values))
    return EncodeOutput(sparse=sparse, batch_size=len(weights))


def _make_multivector_output(mvs: list[np.ndarray]) -> EncodeOutput:
    return EncodeOutput(
        multivector=[m.astype(np.float32) for m in mvs],
        batch_size=len(mvs),
        multivector_token_dim=mvs[0].shape[-1] if mvs else None,
    )


class TestBGEM3FlashAdapterSpec:
    """Static spec / capability assertions (no load required)."""

    def test_spec_advertises_score(self) -> None:
        spec = BGEM3FlashAdapter.spec
        assert "score" in spec.outputs
        assert {"dense", "sparse", "multivector"}.issubset(set(spec.outputs))

    def test_capabilities_include_score(self) -> None:
        adapter = BGEM3FlashAdapter()
        caps = adapter.capabilities
        assert "score" in caps.outputs


class TestBGEM3YamlConfig:
    """Validate the shipped BAAI__bge-m3.yaml advertises scoring."""

    @pytest.fixture
    def config(self) -> ModelConfig:
        models_dir = Path(__file__).resolve().parents[2] / "models"
        path = models_dir / "BAAI__bge-m3.yaml"
        raw = yaml.safe_load(path.read_text())
        return ModelConfig(**raw)

    def test_score_task_enabled(self, config: ModelConfig) -> None:
        # Regression guard for sie-internal#728: bge-m3 was advertised as a
        # scoring model but the YAML had `score: null`, causing /v1/score
        # requests to be rejected before reaching the adapter.
        assert config.tasks.score is not None

    def test_score_in_outputs(self, config: ModelConfig) -> None:
        assert "score" in config.outputs


class TestResolveScoreMode:
    """Validation of options['score_mode'] / options['score_weights']."""

    @pytest.fixture
    def adapter(self) -> BGEM3FlashAdapter:
        return BGEM3FlashAdapter()

    def test_default_is_dense(self, adapter: BGEM3FlashAdapter) -> None:
        mode, weights = adapter._resolve_score_mode(None)
        assert mode == "dense"
        # Weights default to paper values regardless of mode (only used in hybrid)
        assert weights == {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}

    @pytest.mark.parametrize("mode", ["dense", "sparse", "colbert", "hybrid"])
    def test_valid_modes(self, adapter: BGEM3FlashAdapter, mode: str) -> None:
        resolved, _ = adapter._resolve_score_mode({"score_mode": mode})
        assert resolved == mode

    def test_invalid_mode_raises(self, adapter: BGEM3FlashAdapter) -> None:
        with pytest.raises(ValueError, match="Invalid score_mode"):
            adapter._resolve_score_mode({"score_mode": "wat"})

    @pytest.mark.parametrize("bad", [[], {}, ["dense"], None.__class__, 5])
    def test_invalid_mode_type_raises_value_error(self, adapter: BGEM3FlashAdapter, bad: object) -> None:
        # Unhashable inputs (list/dict) would raise TypeError from
        # frozenset.__contains__ before our validation; ensure we catch the
        # type up-front and surface a uniform ValueError (→ 400, not 500).
        with pytest.raises(ValueError, match="Invalid score_mode"):
            adapter._resolve_score_mode({"score_mode": bad})

    def test_score_weights_override(self, adapter: BGEM3FlashAdapter) -> None:
        _, weights = adapter._resolve_score_mode(
            {"score_mode": "hybrid", "score_weights": {"dense": 1.0, "sparse": 0.0}}
        )
        assert weights["dense"] == 1.0
        assert weights["sparse"] == 0.0
        # colbert preserves default
        assert weights["colbert"] == 0.4

    def test_score_weights_must_be_mapping(self, adapter: BGEM3FlashAdapter) -> None:
        with pytest.raises(ValueError, match="score_weights must be a mapping"):
            adapter._resolve_score_mode({"score_weights": [1.0, 0.0, 0.0]})

    def test_score_weights_unknown_key(self, adapter: BGEM3FlashAdapter) -> None:
        with pytest.raises(ValueError, match="Unknown score_weights keys"):
            adapter._resolve_score_mode({"score_weights": {"foo": 1.0}})

    def test_score_weights_negative(self, adapter: BGEM3FlashAdapter) -> None:
        with pytest.raises(ValueError, match="must be a non-negative"):
            adapter._resolve_score_mode({"score_weights": {"dense": -1.0}})

    def test_score_weights_bool_rejected(self, adapter: BGEM3FlashAdapter) -> None:
        # bool is a subclass of int; we reject it to avoid silent True/False -> 1/0.
        with pytest.raises(ValueError, match="must be a non-negative"):
            adapter._resolve_score_mode({"score_weights": {"dense": True}})

    def test_score_weights_string_rejected(self, adapter: BGEM3FlashAdapter) -> None:
        with pytest.raises(ValueError, match="must be a non-negative"):
            adapter._resolve_score_mode({"score_weights": {"dense": "0.5"}})

    def test_hybrid_zero_weights_rejected(self, adapter: BGEM3FlashAdapter) -> None:
        with pytest.raises(ValueError, match="at least one positive weight"):
            adapter._resolve_score_mode(
                {
                    "score_mode": "hybrid",
                    "score_weights": {"dense": 0.0, "sparse": 0.0, "colbert": 0.0},
                }
            )


class TestOutputTypesForMode:
    @pytest.fixture
    def adapter(self) -> BGEM3FlashAdapter:
        return BGEM3FlashAdapter()

    def test_dense_only_for_dense_mode(self, adapter: BGEM3FlashAdapter) -> None:
        assert adapter._output_types_for_mode("dense", {}) == ["dense"]

    def test_sparse_only_for_sparse_mode(self, adapter: BGEM3FlashAdapter) -> None:
        assert adapter._output_types_for_mode("sparse", {}) == ["sparse"]

    def test_multivector_for_colbert_mode(self, adapter: BGEM3FlashAdapter) -> None:
        assert adapter._output_types_for_mode("colbert", {}) == ["multivector"]

    def test_hybrid_uses_all_with_positive_weights(self, adapter: BGEM3FlashAdapter) -> None:
        out = adapter._output_types_for_mode("hybrid", {"dense": 0.4, "sparse": 0.2, "colbert": 0.4})
        assert set(out) == {"dense", "sparse", "multivector"}

    def test_hybrid_skips_zero_weight_outputs(self, adapter: BGEM3FlashAdapter) -> None:
        out = adapter._output_types_for_mode("hybrid", {"dense": 1.0, "sparse": 0.0, "colbert": 0.0})
        assert out == ["dense"]


class TestSimilarityHelpers:
    """Direct tests for the four similarity primitives."""

    def test_dense_sim_identical_vectors_is_one(self) -> None:
        vec = _RNG.standard_normal((1, 8)).astype(np.float32)
        vec = _normalize(vec)
        out = _make_dense_output(vec)
        sim = BGEM3FlashAdapter._dense_sim(out, 0, out, 0)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_dense_sim_orthogonal_vectors_is_zero(self) -> None:
        q = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        d = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        sim = BGEM3FlashAdapter._dense_sim(_make_dense_output(q), 0, _make_dense_output(d), 0)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_dense_sim_handles_unnormalized_input(self) -> None:
        # Cosine still yields 1.0 for parallel vectors of different magnitudes.
        q = np.array([[2.0, 0.0]], dtype=np.float32)
        d = np.array([[5.0, 0.0]], dtype=np.float32)
        sim = BGEM3FlashAdapter._dense_sim(_make_dense_output(q), 0, _make_dense_output(d), 0)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_sparse_sim_overlapping_tokens(self) -> None:
        q_out = _make_sparse_output([{1: 0.5, 2: 0.3}])
        d_out = _make_sparse_output([{1: 0.4, 3: 0.9}])
        sim = BGEM3FlashAdapter._sparse_sim(q_out, 0, d_out, 0)
        # Only token id 1 overlaps: 0.5 * 0.4 = 0.2
        assert sim == pytest.approx(0.2, abs=1e-6)

    def test_sparse_sim_disjoint_tokens_is_zero(self) -> None:
        q_out = _make_sparse_output([{1: 0.5}])
        d_out = _make_sparse_output([{2: 0.9}])
        assert BGEM3FlashAdapter._sparse_sim(q_out, 0, d_out, 0) == 0.0

    def test_sparse_sim_empty_vector_is_zero(self) -> None:
        q_out = _make_sparse_output([{}])
        d_out = _make_sparse_output([{1: 0.5}])
        assert BGEM3FlashAdapter._sparse_sim(q_out, 0, d_out, 0) == 0.0

    def test_colbert_sim_identical_sequences_is_one(self) -> None:
        vecs = _normalize(_RNG.standard_normal((4, 8)).astype(np.float32))
        out = _make_multivector_output([vecs])
        sim = BGEM3FlashAdapter._colbert_sim(out, 0, out, 0)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_colbert_sim_disjoint_directions_is_low(self) -> None:
        q = np.array([[1.0, 0.0]], dtype=np.float32)
        d = np.array([[0.0, 1.0]], dtype=np.float32)
        sim = BGEM3FlashAdapter._colbert_sim(_make_multivector_output([q]), 0, _make_multivector_output([d]), 0)
        assert sim == pytest.approx(0.0, abs=1e-6)


class TestScoreEndToEnd:
    """Test the public score() / score_pairs() entry points with mocked encode()."""

    @pytest.fixture
    def adapter(self) -> BGEM3FlashAdapter:
        ad = BGEM3FlashAdapter()
        # Bypass _check_loaded()
        ad._model = MagicMock()
        return ad

    @staticmethod
    def _patch_encode_dense(
        adapter: BGEM3FlashAdapter,
        *,
        query_vec: np.ndarray,
        doc_vecs: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Patch ``encode`` with role-aware fixture and capture the calls."""
        calls: list[dict[str, Any]] = []

        def fake_encode(
            items: list[Item],
            output_types: list[str],
            *,
            instruction: str | None = None,
            is_query: bool = False,
            options: dict[str, Any] | None = None,
            **_: Any,
        ) -> EncodeOutput:
            calls.append(
                {
                    "n": len(items),
                    "is_query": is_query,
                    "output_types": list(output_types),
                    "instruction": instruction,
                }
            )
            if is_query:
                return _make_dense_output(np.tile(query_vec, (len(items), 1)))
            return _make_dense_output(doc_vecs[: len(items)])

        adapter.encode = fake_encode  # type: ignore[method-assign]
        return calls

    def test_score_returns_list_of_floats(self, adapter: BGEM3FlashAdapter) -> None:
        q = _normalize(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        # Doc 0 == query (sim 1.0); doc 1 orthogonal (sim 0.0); doc 2 anti-aligned (sim -1.0)
        docs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
        self._patch_encode_dense(adapter, query_vec=q[0], doc_vecs=docs)

        scores = adapter.score(Item(text="q"), [Item(text="a"), Item(text="b"), Item(text="c")])

        assert isinstance(scores, list)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)
        assert scores[0] == pytest.approx(1.0, abs=1e-5)
        assert scores[1] == pytest.approx(0.0, abs=1e-6)
        assert scores[2] == pytest.approx(-1.0, abs=1e-5)

    def test_score_pairs_returns_score_output(self, adapter: BGEM3FlashAdapter) -> None:
        q = _normalize(np.array([[1.0, 0.0]], dtype=np.float32))
        docs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self._patch_encode_dense(adapter, query_vec=q[0], doc_vecs=docs)

        out = adapter.score_pairs(
            [Item(text="q1"), Item(text="q2")],
            [Item(text="d1"), Item(text="d2")],
        )

        assert out.scores.dtype == np.float32
        assert out.scores.shape == (2,)
        assert out.scores[0] == pytest.approx(1.0, abs=1e-5)
        assert out.scores[1] == pytest.approx(0.0, abs=1e-6)

    def test_score_pairs_empty_inputs(self, adapter: BGEM3FlashAdapter) -> None:
        out = adapter.score_pairs([], [])
        assert out.scores.shape == (0,)
        assert out.scores.dtype == np.float32

    def test_score_pairs_length_mismatch_raises(self, adapter: BGEM3FlashAdapter) -> None:
        with pytest.raises(ValueError, match="equal-length"):
            adapter.score_pairs([Item(text="q")], [Item(text="d1"), Item(text="d2")])

    def test_score_empty_items(self, adapter: BGEM3FlashAdapter) -> None:
        # encode() should never be called when items is empty
        adapter.encode = MagicMock(side_effect=AssertionError("encode should not be called"))  # type: ignore[method-assign]
        assert adapter.score(Item(text="q"), []) == []

    def test_score_uses_is_query_flag(self, adapter: BGEM3FlashAdapter) -> None:
        q = _normalize(np.array([[1.0, 0.0]], dtype=np.float32))
        docs = np.array([[1.0, 0.0]], dtype=np.float32)
        calls = self._patch_encode_dense(adapter, query_vec=q[0], doc_vecs=docs)

        adapter.score(Item(text="q"), [Item(text="d")])

        # First call encodes the query, second the docs
        assert len(calls) == 2
        assert calls[0]["is_query"] is True
        assert calls[0]["n"] == 1
        assert calls[1]["is_query"] is False
        assert calls[1]["n"] == 1

    def test_invalid_score_mode_raises(self, adapter: BGEM3FlashAdapter) -> None:
        with pytest.raises(ValueError, match="Invalid score_mode"):
            adapter.score(Item(text="q"), [Item(text="d")], options={"score_mode": "bogus"})

    def test_score_sparse_mode(self, adapter: BGEM3FlashAdapter) -> None:
        def fake_encode(
            items: list[Item],
            output_types: list[str],
            *,
            instruction: str | None = None,
            is_query: bool = False,
            options: dict[str, Any] | None = None,
            **_: Any,
        ) -> EncodeOutput:
            assert output_types == ["sparse"]
            if is_query:
                return _make_sparse_output([{10: 0.5, 20: 0.3}])
            # Two docs: first overlaps tokens 10 & 20, second only token 999
            return _make_sparse_output(
                [
                    {10: 0.4, 20: 0.6, 30: 0.1},  # 0.5*0.4 + 0.3*0.6 = 0.38
                    {999: 0.9},  # disjoint => 0
                ]
            )

        adapter.encode = fake_encode  # type: ignore[method-assign]

        scores = adapter.score(
            Item(text="q"),
            [Item(text="d1"), Item(text="d2")],
            options={"score_mode": "sparse"},
        )
        assert scores[0] == pytest.approx(0.38, abs=1e-5)
        assert scores[1] == pytest.approx(0.0, abs=1e-6)

    def test_score_colbert_mode(self, adapter: BGEM3FlashAdapter) -> None:
        q_mv = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        # Doc identical to query (perfect MaxSim)
        d_mv = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        def fake_encode(
            items: list[Item],
            output_types: list[str],
            *,
            instruction: str | None = None,
            is_query: bool = False,
            options: dict[str, Any] | None = None,
            **_: Any,
        ) -> EncodeOutput:
            assert output_types == ["multivector"]
            return _make_multivector_output([q_mv if is_query else d_mv])

        adapter.encode = fake_encode  # type: ignore[method-assign]

        scores = adapter.score(Item(text="q"), [Item(text="d")], options={"score_mode": "colbert"})
        assert scores == [pytest.approx(1.0, abs=1e-5)]

    def test_hybrid_dense_only_weight_equals_dense_mode(self, adapter: BGEM3FlashAdapter) -> None:
        """Hybrid with weights {dense: 1.0, others: 0.0} matches pure dense path."""
        q = _normalize(np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
        docs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        def fake_encode(
            items: list[Item],
            output_types: list[str],
            *,
            instruction: str | None = None,
            is_query: bool = False,
            options: dict[str, Any] | None = None,
            **_: Any,
        ) -> EncodeOutput:
            # Weight-pruning should request only "dense"
            assert output_types == ["dense"]
            if is_query:
                return _make_dense_output(np.tile(q[0], (len(items), 1)))
            return _make_dense_output(docs[: len(items)])

        adapter.encode = fake_encode  # type: ignore[method-assign]

        scores = adapter.score(
            Item(text="q"),
            [Item(text="d1"), Item(text="d2")],
            options={
                "score_mode": "hybrid",
                "score_weights": {"dense": 1.0, "sparse": 0.0, "colbert": 0.0},
            },
        )
        assert scores[0] == pytest.approx(1.0, abs=1e-5)
        assert scores[1] == pytest.approx(0.0, abs=1e-6)

    def test_hybrid_combines_modes(self, adapter: BGEM3FlashAdapter) -> None:
        """Hybrid score is a linear combination of the per-mode scores."""
        q_dense = _normalize(np.array([[1.0, 0.0]], dtype=np.float32))
        d_dense = _normalize(np.array([[1.0, 0.0]], dtype=np.float32))  # sim 1.0
        # Sparse: only token 5 overlaps with weights 0.5 & 0.4 -> sim 0.2

        def fake_encode(
            items: list[Item],
            output_types: list[str],
            *,
            instruction: str | None = None,
            is_query: bool = False,
            options: dict[str, Any] | None = None,
            **_: Any,
        ) -> EncodeOutput:
            # colbert weight is 0 -> multivector is pruned out.
            assert set(output_types) == {"dense", "sparse"}
            dense = q_dense if is_query else d_dense
            sparse = _make_sparse_output([{5: 0.5}]).sparse if is_query else _make_sparse_output([{5: 0.4}]).sparse
            return EncodeOutput(
                dense=dense,
                sparse=sparse,
                batch_size=1,
            )

        adapter.encode = fake_encode  # type: ignore[method-assign]

        scores = adapter.score(
            Item(text="q"),
            [Item(text="d")],
            options={
                "score_mode": "hybrid",
                # Use convenient weights that make arithmetic obvious.
                "score_weights": {"dense": 0.5, "sparse": 0.5, "colbert": 0.0},
            },
        )
        # 0.5 * 1.0 (dense) + 0.5 * 0.2 (sparse) + 0 = 0.6
        assert scores == [pytest.approx(0.6, abs=1e-5)]

    def test_hybrid_combines_all_three_modes(self, adapter: BGEM3FlashAdapter) -> None:
        """Hybrid with all three weights > 0 fans encode out to all output types."""
        q_dense = _normalize(np.array([[1.0, 0.0]], dtype=np.float32))
        d_dense = _normalize(np.array([[1.0, 0.0]], dtype=np.float32))
        mv = np.array([[1.0, 0.0]], dtype=np.float32)

        def fake_encode(
            items: list[Item],
            output_types: list[str],
            *,
            instruction: str | None = None,
            is_query: bool = False,
            options: dict[str, Any] | None = None,
            **_: Any,
        ) -> EncodeOutput:
            assert set(output_types) == {"dense", "sparse", "multivector"}
            dense = q_dense if is_query else d_dense
            sparse = _make_sparse_output([{5: 0.5}]).sparse if is_query else _make_sparse_output([{5: 0.4}]).sparse
            return EncodeOutput(
                dense=dense,
                sparse=sparse,
                multivector=[mv],
                batch_size=1,
            )

        adapter.encode = fake_encode  # type: ignore[method-assign]

        scores = adapter.score(
            Item(text="q"),
            [Item(text="d")],
            options={
                "score_mode": "hybrid",
                "score_weights": {"dense": 0.4, "sparse": 0.2, "colbert": 0.4},
            },
        )
        # 0.4*1.0 + 0.2*0.2 + 0.4*1.0 = 0.84
        assert scores == [pytest.approx(0.84, abs=1e-5)]
