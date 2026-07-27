from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters.gte_sparse_flash import GTESparseFlashAdapter
from sie_server.adapters.splade_flash.adapter import SPLADEFlashAdapter
from sie_server.core.inference_output import SparseVector


def _reference_aggregate(
    weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lengths: list[int],
) -> list[SparseVector]:
    """Per-sequence reference implementation (the old loop-based approach)."""
    results: list[SparseVector] = []
    for i in range(len(seq_lengths)):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        max_weights, _ = weights[start:end].max(dim=0)
        row = max_weights.cpu().float().numpy()
        mask = row > 0
        results.append(
            SparseVector(
                indices=np.where(mask)[0].astype(np.int32),
                values=row[mask],
            )
        )
    return results


def _build_cu_seqlens(seq_lengths: list[int]) -> torch.Tensor:
    cu = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(seq_lengths, dtype=torch.int32).cumsum(0)
    return cu


class TestSPLADEAggregation:
    """Tests for SPLADEFlashAdapter._aggregate_sparse using segment_reduce."""

    @pytest.fixture
    def adapter(self) -> SPLADEFlashAdapter:
        return SPLADEFlashAdapter("test-splade-model", max_seq_length=128)

    @pytest.mark.parametrize(
        "seq_lengths",
        [
            [5],
            [3, 7],
            [1, 4, 2, 6],
            [10, 1, 10],
        ],
        ids=["single", "two", "four-varied", "mixed"],
    )
    def test_segment_reduce_matches_reference(
        self,
        adapter: SPLADEFlashAdapter,
        seq_lengths: list[int],
    ) -> None:
        vocab_size = 64
        total_tokens = sum(seq_lengths)
        torch.manual_seed(42)
        weights = torch.rand(total_tokens, vocab_size)
        cu_seqlens = _build_cu_seqlens(seq_lengths)

        actual = adapter._aggregate_sparse(weights, cu_seqlens, seq_lengths)
        expected = _reference_aggregate(weights, cu_seqlens, seq_lengths)

        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a.indices, e.indices)
            np.testing.assert_allclose(a.values, e.values, rtol=1e-5)

    def test_all_zero_weights_produce_empty_vectors(
        self,
        adapter: SPLADEFlashAdapter,
    ) -> None:
        seq_lengths = [3, 5]
        total_tokens = sum(seq_lengths)
        vocab_size = 32
        weights = torch.zeros(total_tokens, vocab_size)
        cu_seqlens = _build_cu_seqlens(seq_lengths)

        result = adapter._aggregate_sparse(weights, cu_seqlens, seq_lengths)

        assert len(result) == 2
        for sv in result:
            assert len(sv.indices) == 0
            assert len(sv.values) == 0

    def test_output_dtypes(self, adapter: SPLADEFlashAdapter) -> None:
        seq_lengths = [4, 3]
        total_tokens = sum(seq_lengths)
        vocab_size = 16
        torch.manual_seed(7)
        weights = torch.rand(total_tokens, vocab_size)
        cu_seqlens = _build_cu_seqlens(seq_lengths)

        result = adapter._aggregate_sparse(weights, cu_seqlens, seq_lengths)

        for sv in result:
            assert sv.indices.dtype == np.int32, f"Expected int32, got {sv.indices.dtype}"
            assert sv.values.dtype == np.float32, f"Expected float32, got {sv.values.dtype}"


class TestGTESparseAggregation:
    """Tests for GTESparseFlashAdapter._aggregate_sparse with special token masking."""

    @pytest.fixture
    def adapter(self) -> GTESparseFlashAdapter:
        a = GTESparseFlashAdapter("test-gte-sparse-model", trust_remote_code=True)
        a._special_token_ids = [0, 1, 2]
        return a

    @pytest.mark.parametrize(
        "seq_lengths",
        [
            [5],
            [3, 7],
            [1, 4, 2, 6],
        ],
        ids=["single", "two", "four-varied"],
    )
    def test_segment_reduce_matches_reference_with_special_tokens(
        self,
        adapter: GTESparseFlashAdapter,
        seq_lengths: list[int],
    ) -> None:
        vocab_size = 64
        total_tokens = sum(seq_lengths)
        torch.manual_seed(42)
        weights = torch.rand(total_tokens, vocab_size)
        cu_seqlens = _build_cu_seqlens(seq_lengths)

        actual = adapter._aggregate_sparse(weights, cu_seqlens, seq_lengths)

        # Build reference: aggregate then zero special tokens
        ref = _reference_aggregate(weights, cu_seqlens, seq_lengths)
        expected: list[SparseVector] = []
        for sv in ref:
            mask = np.ones(len(sv.indices), dtype=bool)
            for sid in adapter._special_token_ids:
                mask &= sv.indices != sid
            expected.append(
                SparseVector(
                    indices=sv.indices[mask],
                    values=sv.values[mask],
                )
            )

        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a.indices, e.indices)
            np.testing.assert_allclose(a.values, e.values, rtol=1e-5)

    def test_no_special_tokens(self) -> None:
        adapter = GTESparseFlashAdapter("test-gte-sparse-model", trust_remote_code=True)
        adapter._special_token_ids = []
        seq_lengths = [3, 4]
        total_tokens = sum(seq_lengths)
        vocab_size = 32
        torch.manual_seed(99)
        weights = torch.rand(total_tokens, vocab_size)
        cu_seqlens = _build_cu_seqlens(seq_lengths)

        actual = adapter._aggregate_sparse(weights, cu_seqlens, seq_lengths)
        expected = _reference_aggregate(weights, cu_seqlens, seq_lengths)

        assert len(actual) == len(expected)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(a.indices, e.indices)
            np.testing.assert_allclose(a.values, e.values, rtol=1e-5)

    def test_all_zero_weights(self) -> None:
        adapter = GTESparseFlashAdapter("test-gte-sparse-model", trust_remote_code=True)
        adapter._special_token_ids = [0, 1]
        seq_lengths = [2, 3]
        total_tokens = sum(seq_lengths)
        vocab_size = 16
        weights = torch.zeros(total_tokens, vocab_size)
        cu_seqlens = _build_cu_seqlens(seq_lengths)

        result = adapter._aggregate_sparse(weights, cu_seqlens, seq_lengths)
        for sv in result:
            assert len(sv.indices) == 0
            assert len(sv.values) == 0

    def test_output_dtypes(self) -> None:
        adapter = GTESparseFlashAdapter("test-gte-sparse-model", trust_remote_code=True)
        adapter._special_token_ids = [0]
        seq_lengths = [4]
        total_tokens = sum(seq_lengths)
        vocab_size = 16
        torch.manual_seed(7)
        weights = torch.rand(total_tokens, vocab_size)
        cu_seqlens = _build_cu_seqlens(seq_lengths)

        result = adapter._aggregate_sparse(weights, cu_seqlens, seq_lengths)
        for sv in result:
            assert sv.indices.dtype == np.int32
            assert sv.values.dtype == np.float32


class TestDenseToSparseList:
    """Tests for SPLADEFlashAdapter._dense_to_sparse_list."""

    def test_matches_reference(self) -> None:
        torch.manual_seed(42)
        max_weights = torch.rand(4, 32)
        result = SPLADEFlashAdapter._dense_to_sparse_list(max_weights)

        assert len(result) == 4
        dense = max_weights.cpu().float().numpy()
        for i, sv in enumerate(result):
            row = dense[i]
            mask = row > 0
            np.testing.assert_array_equal(sv.indices, np.where(mask)[0].astype(np.int32))
            np.testing.assert_allclose(sv.values, row[mask], rtol=1e-5)

    def test_all_zero(self) -> None:
        max_weights = torch.zeros(2, 16)
        result = SPLADEFlashAdapter._dense_to_sparse_list(max_weights)
        assert len(result) == 2
        for sv in result:
            assert len(sv.indices) == 0
            assert len(sv.values) == 0

    def test_output_dtypes(self) -> None:
        torch.manual_seed(7)
        max_weights = torch.rand(1, 8)
        result = SPLADEFlashAdapter._dense_to_sparse_list(max_weights)
        for sv in result:
            assert sv.indices.dtype == np.int32
            assert sv.values.dtype == np.float32


class TestInPlaceRelu:
    """Verify that in-place relu_ produces same results as out-of-place relu."""

    def test_relu_inplace_matches_outofplace(self) -> None:
        torch.manual_seed(42)
        logits = torch.randn(10, 64)

        expected = torch.log1p(torch.relu(logits.clone()))
        actual = torch.log1p(torch.relu_(logits))

        torch.testing.assert_close(actual, expected)

    def test_relu_inplace_on_float_copy(self) -> None:
        """Simulates GTE pattern: relu_ on logits.float() (a new tensor)."""
        torch.manual_seed(42)
        logits_half = torch.randn(10, 64, dtype=torch.float16)

        values_copy = logits_half.float()
        original_half = logits_half.clone()

        _ = torch.log1p(torch.relu_(values_copy))

        # The original half-precision tensor must be untouched
        torch.testing.assert_close(logits_half, original_half)

    def test_double_log1p_relu_inplace(self) -> None:
        """Verify the v3 activation: log1p(log1p(relu_(.)))."""
        torch.manual_seed(42)
        logits = torch.randn(5, 32)

        expected = torch.log1p(torch.log1p(torch.relu(logits.clone())))
        actual = torch.log1p(torch.log1p(torch.relu_(logits)))

        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("compute_precision", "expected"),
    [("float32", False), ("float16", True), ("bfloat16", True)],
)
def test_flash_path_requires_half_precision(
    monkeypatch,
    compute_precision: ComputePrecision,
    expected: bool,
) -> None:
    tokenizer = SimpleNamespace(model_max_length=512)
    model = SimpleNamespace(
        bert=object(),
        config=SimpleNamespace(
            hidden_size=32,
            max_position_embeddings=512,
            position_embedding_type="absolute",
            vocab_size=128,
        ),
        eval=MagicMock(),
        to=MagicMock(),
    )
    adapter = SPLADEFlashAdapter("test-splade-model", compute_precision=compute_precision)
    monkeypatch.setattr(adapter, "_try_load_idf_vector", lambda _tokenizer: None)
    monkeypatch.setattr(
        "sie_server.adapters.splade_flash.adapter._has_flash_attn",
        lambda: True,
    )
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForMaskedLM.from_pretrained", return_value=model),
    ):
        adapter.load("cuda:0")

    assert adapter._use_flash is expected


@pytest.mark.parametrize("revision", [None, "main", "v1.0"])
def test_remote_code_load_requires_immutable_revision(revision: str | None) -> None:
    adapter = SPLADEFlashAdapter(
        "remote-org/remote-model",
        trust_remote_code=True,
        revision=revision,
    )

    with pytest.raises(ValueError, match="immutable 40-character revision"):
        adapter.load("cpu")


@pytest.mark.parametrize(
    ("model_path", "revision"),
    [
        ("remote-org/remote-model", "a" * 40),
        (None, None),
    ],
    ids=["pinned-remote", "local-path"],
)
def test_remote_code_load_allows_pinned_remote_or_local_path(
    tmp_path: Path,
    model_path: str | None,
    revision: str | None,
) -> None:
    resolved_path = model_path
    if resolved_path is None:
        local_path = tmp_path / "model"
        local_path.mkdir()
        resolved_path = str(local_path)

    adapter = SPLADEFlashAdapter(
        resolved_path,
        trust_remote_code=True,
        revision=revision,
    )
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained",
            side_effect=RuntimeError("validated load reached"),
        ),
        pytest.raises(RuntimeError, match="validated load reached"),
    ):
        adapter.load("cpu")


class _TokenizerWithAddedTokens:
    vocab_size = 2

    def __len__(self) -> int:
        return 4

    def get_vocab(self) -> dict[str, int]:
        return {"base": 0, "added": 3, "out-of-range": 7}

    def __call__(self, _texts: list[str], **_kwargs: Any) -> dict[str, list[list[int]]]:
        return {"input_ids": [[0, 3, 7]]}


def test_parse_idf_json_rejects_non_mapping_and_invalid_weights(tmp_path: Path) -> None:
    artifact = tmp_path / "idf.json"
    artifact.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
    assert SPLADEFlashAdapter._parse_idf_json(str(artifact)) is None

    artifact.write_text(json.dumps({"valid": 1.0, "invalid": []}), encoding="utf-8")
    assert SPLADEFlashAdapter._parse_idf_json(str(artifact)) is None


@pytest.mark.parametrize("invalid_weight", [float("nan"), float("inf"), float("-inf")])
def test_parse_idf_json_rejects_non_finite_weights(
    tmp_path: Path,
    invalid_weight: float,
) -> None:
    artifact = tmp_path / "idf.json"
    artifact.write_text(json.dumps({"invalid": invalid_weight}), encoding="utf-8")

    assert SPLADEFlashAdapter._parse_idf_json(str(artifact)) is None


def test_malformed_query_weights_fall_back_to_idf_json(monkeypatch, tmp_path: Path) -> None:
    query_weights = tmp_path / "query_token_weights.txt"
    query_weights.write_text("base\t1.0\nmalformed-row\n", encoding="utf-8")
    idf = tmp_path / "idf.json"
    idf.write_text(json.dumps({"added": 2.0}), encoding="utf-8")
    adapter = SPLADEFlashAdapter("test-splade-model")
    artifacts = {
        "query_token_weights.txt": str(query_weights),
        "idf.json": str(idf),
    }
    monkeypatch.setattr(
        adapter,
        "_resolve_repo_file",
        lambda _path, filename, _revision: artifacts[filename],
    )

    vector = adapter._try_load_idf_vector(_TokenizerWithAddedTokens())

    assert vector is not None
    assert vector.tolist() == [0.0, 0.0, 0.0, 2.0]


def test_unmapped_query_weights_fall_back_to_idf_json(monkeypatch, tmp_path: Path) -> None:
    query_weights = tmp_path / "query_token_weights.txt"
    query_weights.write_text("unknown\t1.0\n", encoding="utf-8")
    idf = tmp_path / "idf.json"
    idf.write_text(json.dumps({"added": 2.0}), encoding="utf-8")
    adapter = SPLADEFlashAdapter("test-splade-model")
    artifacts = {
        "query_token_weights.txt": str(query_weights),
        "idf.json": str(idf),
    }
    monkeypatch.setattr(
        adapter,
        "_resolve_repo_file",
        lambda _path, filename, _revision: artifacts[filename],
    )

    vector = adapter._try_load_idf_vector(_TokenizerWithAddedTokens())

    assert vector is not None
    assert vector.tolist() == [0.0, 0.0, 0.0, 2.0]


def test_idf_vector_covers_added_tokens_and_ignores_out_of_range_ids(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "idf.json"
    artifact.write_text(
        json.dumps({"base": 1.0, "added": 2.0, "out-of-range": 3.0}),
        encoding="utf-8",
    )
    adapter = SPLADEFlashAdapter("test-splade-model")
    monkeypatch.setattr(
        adapter,
        "_resolve_repo_file",
        lambda _path, filename, _revision: str(artifact) if filename == "idf.json" else None,
    )

    vector = adapter._try_load_idf_vector(_TokenizerWithAddedTokens())

    assert vector is not None
    assert vector.tolist() == [1.0, 0.0, 0.0, 2.0]


def test_query_idf_ignores_token_ids_outside_vector() -> None:
    adapter = SPLADEFlashAdapter("test-splade-model")
    adapter._model = object()
    adapter._tokenizer = _TokenizerWithAddedTokens()
    adapter._idf = torch.tensor([1.0, 0.0, 0.0, 2.0])

    output = adapter._encode_query_idf(["query"], is_query=True)

    assert output.sparse is not None
    np.testing.assert_array_equal(output.sparse[0].indices, np.array([0, 3], dtype=np.int32))
    np.testing.assert_array_equal(output.sparse[0].values, np.array([1.0, 2.0], dtype=np.float32))
    assert output.extra == {"input_token_counts": [3]}
