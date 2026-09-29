"""Unit tests for TopK-Embed's packed (variable-length) path.

The kernels and the packed forward over real Qwen3.5 modules need transformers >= 5.2
and, for the fast kernels, CUDA; ``test_topk_embed_parity.py`` covers those on real
weights. These tests pin what runs anywhere: kernel selection, the PyTorch reference
kernels against unpacked computation, and the adapter's packing plumbing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sie_server.adapters.topk_embed import adapter as adapter_module
from sie_server.adapters.topk_embed import packed
from sie_server.adapters.topk_embed.adapter import TopkEmbedAdapter
from sie_server.types.inputs import Item

from .test_topk_embed import _fake_vision, _png, make_adapter


def _stub(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(0)


class TestResolveKernels:
    def test_auto_does_not_pack_off_cuda(self) -> None:
        assert packed.resolve_kernels("cpu", mode=None) is None

    def test_off_never_packs(self) -> None:
        fast = packed.PackedKernels(delta_rule=_stub, causal_conv=_stub, attention=_stub)
        with patch.object(packed, "_fast_kernels", return_value=fast):
            assert packed.resolve_kernels("cuda:0", mode=False) is None

    def test_forced_packing_off_cuda_uses_reference_kernels(self) -> None:
        kernels = packed.resolve_kernels("cpu", mode=True)
        assert kernels is not None
        assert kernels.delta_rule is packed.reference_delta_rule
        assert kernels.causal_conv is packed.reference_causal_conv
        assert kernels.attention is packed.reference_attention

    def test_auto_on_cuda_needs_the_fast_kernels(self) -> None:
        with patch.object(packed, "_fast_kernels", return_value=None):
            assert packed.resolve_kernels("cuda:0", mode=None) is None
            forced = packed.resolve_kernels("cuda:0", mode=True)
            assert forced is not None
            assert forced.names["delta_rule"] == "reference"
        fast = packed.PackedKernels(delta_rule=_stub, causal_conv=_stub, attention=_stub)
        with patch.object(packed, "_fast_kernels", return_value=fast):
            assert packed.resolve_kernels("cuda:0", mode=None) is fast


def _cu(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0, *np.cumsum(lengths)], dtype=torch.long)


class TestReferenceKernels:
    def test_segments(self) -> None:
        assert packed.segments(_cu([3, 1, 4])) == [(0, 3), (3, 4), (4, 8)]

    def test_causal_conv_matches_each_sequence_alone(self) -> None:
        torch.manual_seed(0)
        lengths, dim, width = [5, 2, 7], 6, 4
        x = torch.randn(1, sum(lengths), dim)
        weight, bias = torch.randn(dim, width), torch.randn(dim)
        out = packed.reference_causal_conv(x, weight, bias, "silu", _cu(lengths))
        start = 0
        for length in lengths:
            seq = x[:, start : start + length].transpose(1, 2)
            # transformers' fallback: symmetric padding, then keep the first ``length`` outputs.
            expected = torch.nn.functional.conv1d(seq, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
            expected = torch.nn.functional.silu(expected[..., :length]).transpose(1, 2)
            torch.testing.assert_close(out[:, start : start + length], expected)
            start += length

    def test_causal_conv_keeps_sequences_apart(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(1, 9, 4)
        weight = torch.randn(4, 4)
        changed = x.clone()
        changed[:, 5:] += 10.0
        a = packed.reference_causal_conv(x, weight, None, None, _cu([5, 4]))
        b = packed.reference_causal_conv(changed, weight, None, None, _cu([5, 4]))
        torch.testing.assert_close(a[:, :5], b[:, :5])

    def test_attention_is_block_diagonal_and_bidirectional(self) -> None:
        torch.manual_seed(2)
        lengths, heads, kv_heads, dim = [4, 3, 5], 4, 2, 8
        total = sum(lengths)
        q = torch.randn(total, heads, dim)
        k = torch.randn(total, kv_heads, dim)
        v = torch.randn(total, kv_heads, dim)
        out = packed.reference_attention(q, k, v, _cu(lengths), max(lengths), dim**-0.5)
        segment = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
        mask = segment[:, None] == segment[None, :]
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).unsqueeze(0),
            v.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
        )[0].transpose(0, 1)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


def _identity_text_forward(calls: list[dict[str, Any]]):
    def forward(language_model, embeds, position_ids, cu_seqlens, max_len, kernels):
        calls.append({"embeds": embeds.shape, "positions": position_ids, "cu_seqlens": cu_seqlens, "max_len": max_len})
        return embeds

    return forward


class TestAdapterPacking:
    """With an identity backbone, the packed and padded paths must give the same vectors."""

    @pytest.fixture
    def adapters(self) -> tuple[TopkEmbedAdapter, TopkEmbedAdapter]:
        packed_adapter = make_adapter(packed=True)
        packed_adapter._kernels = packed.resolve_kernels("cpu", mode=True)
        return make_adapter(), packed_adapter

    def test_text_matches_the_padded_path(self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter]) -> None:
        padded_adapter, packed_adapter = adapters
        items = [Item(text="a short one"), Item(text="a somewhat longer document, with commas."), Item(text="x")]
        calls: list[dict[str, Any]] = []
        with patch.object(adapter_module, "text_forward", _identity_text_forward(calls)):
            ours = packed_adapter.encode(items, ["multivector"])
        reference = padded_adapter.encode(items, ["multivector"])
        assert ours.multivector is not None
        assert reference.multivector is not None
        for a, b in zip(ours.multivector, reference.multivector, strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        assert ours.extra["input_token_counts"] == reference.extra["input_token_counts"]
        # One packed call: boundaries at the (length-sorted) row ends, positions restart per row.
        assert len(calls) == 1
        lengths = sorted(reference.extra["input_token_counts"])
        assert calls[0]["cu_seqlens"].tolist() == [0, *np.cumsum(lengths).tolist()]
        assert calls[0]["max_len"] == max(lengths)
        positions = calls[0]["positions"]
        assert positions.shape == (3, 1, sum(lengths))
        assert positions[0, 0, : lengths[0]].tolist() == list(range(lengths[0]))
        assert positions[0, 0, lengths[0]] == 0

    def test_images_match_the_padded_path(self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter]) -> None:
        padded_adapter, packed_adapter = adapters
        items = [Item(images=[_png()]), Item(images=[_png("black", (96, 64))])]
        calls: list[dict[str, Any]] = []
        with _fake_vision(packed_adapter), patch.object(adapter_module, "text_forward", _identity_text_forward(calls)):
            ours = packed_adapter.encode(items, ["multivector"])
        with _fake_vision(padded_adapter):
            reference = padded_adapter.encode(items, ["multivector"])
        assert ours.multivector is not None
        assert reference.multivector is not None
        for a, b in zip(ours.multivector, reference.multivector, strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        assert calls[0]["positions"].shape[0] == 3

    def test_packed_batches_budget_the_sum_of_lengths(
        self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter]
    ) -> None:
        padded_adapter, packed_adapter = adapters
        padded_adapter._text_batch_tokens = packed_adapter._text_batch_tokens = 10
        lengths = [2, 2, 3, 3, 9]
        # Padded: count x longest; packed: plain sum, so more rows fit per batch.
        assert padded_adapter._plan_text_batches(lengths) == [[0, 1, 2], [3], [4]]
        assert packed_adapter._plan_text_batches(lengths) == [[0, 1, 2, 3], [4]]

    def test_unload_forgets_the_kernels(self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter]) -> None:
        _, packed_adapter = adapters
        packed_adapter.unload()
        assert packed_adapter._kernels is None
