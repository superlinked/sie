"""Position-id parity for the SPLADE packed flash path across architectures.

The SPLADE flash adapter packs a batch of sequences into one flat token stream
and drives ``flash_attn_varlen_func`` with ``cu_seqlens``. Each packed sequence
needs position ids restarting at the model's base offset: 0-based for
BERT/DistilBERT, ``padding_idx + 1`` for RoBERTa. These CPU tests pin that the
packed builder is bit-identical to HuggingFace's own position-id construction
(``create_position_ids_from_input_ids``) — the correctness gate that let the
RoBERTa packed path be enabled (granite-embedding-30m-sparse) without a GPU.

The end-to-end packed-vs-padded output-equivalence test (real model, flash-attn)
lives in ``test_splade_packed_parity.py`` and is GPU-gated.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from sie_server.adapters._flash_pack import build_position_ids
from sie_server.adapters.splade_flash.adapter import SPLADEFlashAdapter
from sie_server.types.inputs import Item


def _cu_seqlens(seq_lengths: list[int]) -> torch.Tensor:
    cu = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32)
    cu[1:] = torch.tensor(seq_lengths, dtype=torch.int32).cumsum(0)
    return cu


class _StubEmbeddings:
    def __init__(self, padding_idx: int) -> None:
        self.padding_idx = padding_idx


class _StubBaseModel:
    def __init__(self, padding_idx: int) -> None:
        self.embeddings = _StubEmbeddings(padding_idx)


def test_packed_roberta_position_ids_match_hf_reference() -> None:
    """Packed per-sequence RoBERTa position ids == HF create_position_ids_from_input_ids.

    HF builds RoBERTa position ids from the input_ids + attention mask
    (padding_idx + cumsum(non_pad)). For unpadded input — which is exactly what
    varlen packing feeds — that reduces to ``[pad+1, pad+2, ...]`` per sequence.
    Varied lengths incl. the len-1 boundary and the 512 max-seq boundary.
    """
    from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

    padding_idx = 1
    seq_lengths = [1, 5, 12, 3, 512]
    cu = _cu_seqlens(seq_lengths)

    got = build_position_ids(cu, offset=padding_idx + 1)

    expected_parts = []
    for length in seq_lengths:
        # Any non-pad token id works; the mask is all-ones for unpadded input.
        ids = torch.full((1, length), fill_value=padding_idx + 100, dtype=torch.long)
        expected_parts.append(create_position_ids_from_input_ids(ids, padding_idx).squeeze(0))
    expected = torch.cat(expected_parts)

    assert got.dtype == torch.int64
    assert torch.equal(got, expected)


def test_adapter_build_position_ids_roberta_uses_padding_offset() -> None:
    adapter = SPLADEFlashAdapter("stub-roberta", max_seq_length=128)
    adapter._arch = "roberta"
    adapter._device = "cpu"
    adapter._get_base_model = lambda: _StubBaseModel(padding_idx=1)  # type: ignore[method-assign]

    cu = _cu_seqlens([3, 1, 7])
    got = adapter._build_position_ids(cu)

    # per-seq (offset pad+1=2): [2,3,4] · [2] · [2,3,4,5,6,7,8]
    expected = torch.tensor([2, 3, 4, 2, 2, 3, 4, 5, 6, 7, 8])
    assert torch.equal(got, expected)


def test_adapter_build_position_ids_bert_is_zero_based() -> None:
    adapter = SPLADEFlashAdapter("stub-bert", max_seq_length=128)
    adapter._arch = "bert"
    adapter._device = "cpu"

    cu = _cu_seqlens([3, 1, 4])
    got = adapter._build_position_ids(cu)

    # per-seq (offset 0): [0,1,2] · [0] · [0,1,2,3]
    expected = torch.tensor([0, 1, 2, 0, 0, 1, 2, 3])
    assert torch.equal(got, expected)


def test_adapter_build_position_ids_distilbert_is_zero_based() -> None:
    adapter = SPLADEFlashAdapter("stub-distilbert", max_seq_length=128)
    adapter._arch = "distilbert"
    adapter._device = "cpu"

    cu = _cu_seqlens([2, 5])
    got = adapter._build_position_ids(cu)

    expected = torch.tensor([0, 1, 0, 1, 2, 3, 4])
    assert torch.equal(got, expected)


def test_runtime_max_sequence_length_is_honored_and_clamped() -> None:
    adapter = SPLADEFlashAdapter("stub-bert", max_seq_length=512)
    adapter._model = MagicMock()
    adapter._tokenizer = MagicMock()
    adapter._device = "cpu"

    for use_flash, method_name in ((False, "_encode_native"), (True, "_encode_flash")):
        adapter._use_flash = use_flash
        for requested, expected in ((128, 128), (1024, 512), (None, 512), (True, 512)):
            options = {} if requested is None else {"max_seq_length": requested}
            with patch.object(adapter, method_name, return_value=([MagicMock()], [1])) as encode_method:
                adapter.encode([Item(text="hello")], ["sparse"], options=options)
            encode_method.assert_called_once_with(["hello"], max_length=expected)

    adapter._idf = MagicMock()
    with patch.object(adapter, "_encode_query_idf", return_value=MagicMock()) as encode_query_idf:
        adapter.encode(
            [Item(text="hello")],
            ["sparse"],
            is_query=True,
            options={"max_seq_length": 1024},
        )
    encode_query_idf.assert_called_once_with(["hello"], True, max_length=512)
