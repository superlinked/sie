from __future__ import annotations

import numpy as np
import pytest
import torch
from sie_server.adapters.gte_sparse_flash import GTESparseFlashAdapter
from sie_server.adapters.splade_flash import SPLADEFlashAdapter
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
