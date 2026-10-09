"""``sparse_rows`` and the pooled sparse paths of the doc-only sparse encoders."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sie_server.adapters._sparse_rows import sparse_rows
from sie_server.adapters.gte_sparse_flash import GTESparseFlashAdapter
from sie_server.core.inference_output import EncodeOutput, SparseVector
from sie_server.types.inputs import Item


def _reference(weights: torch.Tensor) -> list[SparseVector]:
    """The previous conversion: copy the dense rows to the host, then numpy.where per row."""
    dense = weights.cpu().float().numpy()
    out = []
    for row in dense:
        mask = row > 0
        out.append(SparseVector(indices=np.where(mask)[0].astype(np.int32), values=row[mask]))
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sparse_rows_matches_dense_conversion(dtype: torch.dtype) -> None:
    torch.manual_seed(3)
    weights = torch.relu(torch.randn(6, 97)).to(dtype)
    weights[2] = 0  # a row with no nonzero entry
    actual = sparse_rows(weights)
    expected = _reference(weights)
    assert len(actual) == len(expected) == 6
    for got, want in zip(actual, expected, strict=True):
        assert got.indices.dtype == np.int32
        assert got.values.dtype == np.float32
        np.testing.assert_array_equal(got.indices, want.indices)
        np.testing.assert_array_equal(got.values, want.values)
    assert len(actual[2].indices) == 0


def test_sparse_rows_of_an_empty_batch() -> None:
    assert sparse_rows(torch.zeros(0, 10)) == []


def test_gte_pool_then_activate_equals_activate_then_pool() -> None:
    adapter = GTESparseFlashAdapter("test-gte-sparse-model", trust_remote_code=True)
    adapter._special_token_ids = [0, 1]
    torch.manual_seed(11)
    seq_lengths = [3, 1, 6]
    logits = torch.randn(sum(seq_lengths), 64).half()
    cu_seqlens = torch.tensor([0, 3, 4, 10], dtype=torch.int32)

    before = adapter._aggregate_sparse(adapter._sparse_activation(logits.float()), cu_seqlens, seq_lengths)
    pooled = torch.segment_reduce(logits, "max", offsets=cu_seqlens)
    after = adapter._pooled_to_sparse(adapter._sparse_activation(pooled.float()))

    for got, want in zip(after, before, strict=True):
        np.testing.assert_array_equal(got.indices, want.indices)
        np.testing.assert_array_equal(got.values, want.values)


def _stub_flash(adapter: GTESparseFlashAdapter, counts: list[int]) -> None:
    def encode_flash(texts: list[str], is_query: bool) -> EncodeOutput:
        output = EncodeOutput(
            sparse=[SparseVector(indices=np.zeros(0, np.int32), values=np.zeros(0, np.float32)) for _ in texts],
            batch_size=len(texts),
            is_query=is_query,
        )
        output.extra["input_token_counts"] = counts
        return output

    adapter._encode_flash = encode_flash  # type: ignore[method-assign]
    adapter._use_flash = True
    adapter._tokenizer = object()  # type: ignore[assignment]
    adapter._model = object()
    adapter._idf = None


def test_gte_reports_its_own_token_counts_for_untemplated_documents() -> None:
    adapter = GTESparseFlashAdapter("test-gte-sparse-model", trust_remote_code=True)
    _stub_flash(adapter, [5, 7])
    output = adapter.encode([Item(text="red shoe"), Item(text="blue hat")], ["sparse"])
    assert output.extra["input_token_counts"] == [5, 7]


def test_gte_leaves_templated_documents_to_the_metering_hook() -> None:
    adapter = GTESparseFlashAdapter("test-gte-sparse-model", trust_remote_code=True)
    _stub_flash(adapter, [5, 7])
    output = adapter.encode(
        [Item(text="red shoe"), Item(text="blue hat")], ["sparse"], options={"doc_template": "passage: {text}"}
    )
    assert "input_token_counts" not in output.extra
