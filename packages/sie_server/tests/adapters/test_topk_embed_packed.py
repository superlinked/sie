"""Unit tests for TopK-Embed's packed (variable-length) path.

The fast kernels need CUDA and flash-linear-attention; ``test_topk_embed_parity.py``
covers them on real weights. These tests pin what runs anywhere: kernel selection,
the PyTorch reference kernels against the stock modules' math, the packed forward
against the stock text model on a small random Qwen3.5 (transformers >= 5.2), and
the adapter's packing plumbing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sie_server.adapters.topk_embed import packed
from sie_server.adapters.topk_embed.adapter import TopkEmbedAdapter
from sie_server.types.inputs import Item
from torch.nn import functional as F

from .test_topk_embed import _fake_vision, _png, make_adapter


def _stub(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(0)


def _stub_kernels() -> packed.PackedKernels:
    return packed.PackedKernels(
        delta_rule=_stub, causal_conv=_stub, attention=_stub, rms_norm=_stub, gated_rms_norm=_stub, swiglu=_stub
    )


class TestResolveKernels:
    def test_auto_does_not_pack_off_cuda(self) -> None:
        assert packed.resolve_kernels("cpu", mode=None) is None

    def test_off_never_packs(self) -> None:
        with patch.object(packed, "_fast_kernels", return_value=_stub_kernels()):
            assert packed.resolve_kernels("cuda:0", mode=False) is None

    def test_forced_packing_off_cuda_uses_reference_kernels(self) -> None:
        kernels = packed.resolve_kernels("cpu", mode=True)
        assert kernels is not None
        assert kernels.delta_rule is packed.reference_delta_rule
        assert kernels.causal_conv is packed.reference_causal_conv
        assert kernels.attention is packed.reference_attention
        assert kernels.rms_norm is packed.reference_rms_norm
        assert kernels.gated_rms_norm is packed.reference_gated_rms_norm
        assert kernels.swiglu is packed.reference_swiglu

    def test_auto_on_cuda_needs_the_fast_kernels(self) -> None:
        with patch.object(packed, "_fast_kernels", return_value=None):
            assert packed.resolve_kernels("cuda:0", mode=None) is None
            forced = packed.resolve_kernels("cuda:0", mode=True)
            assert forced is not None
            assert forced.names["delta_rule"] == "reference"
        fast = _stub_kernels()
        with patch.object(packed, "_fast_kernels", return_value=fast):
            assert packed.resolve_kernels("cuda:0", mode=None) is fast


def _packing(lengths: list[int]) -> packed.Packing:
    return packed.Packing.from_lengths(lengths, "cpu")


class TestPacking:
    def test_offsets_and_segments(self) -> None:
        packing = _packing([3, 1, 4])
        assert packing.cu_seqlens.tolist() == [0, 3, 4, 8]
        assert packing.cu_seqlens.dtype == torch.long
        assert packing.cu_seqlens_int32.dtype == torch.int32
        assert packing.cu_seqlens_int32.tolist() == [0, 3, 4, 8]
        assert packing.cu_seqlens_cpu.device.type == "cpu"
        assert packing.max_len == 4
        assert packing.segments() == [(0, 3), (3, 4), (4, 8)]


class TestReferenceKernels:
    def test_causal_conv_matches_each_sequence_alone(self) -> None:
        torch.manual_seed(0)
        lengths, dim, width = [5, 2, 7], 6, 4
        x = torch.randn(1, sum(lengths), dim)
        weight, bias = torch.randn(dim, width), torch.randn(dim)
        out = packed.reference_causal_conv(x, weight, bias, "silu", _packing(lengths))
        start = 0
        for length in lengths:
            seq = x[:, start : start + length].transpose(1, 2)
            # transformers' fallback: symmetric padding, then keep the first ``length`` outputs.
            expected = F.conv1d(seq, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
            expected = F.silu(expected[..., :length]).transpose(1, 2)
            torch.testing.assert_close(out[:, start : start + length], expected)
            start += length

    def test_causal_conv_keeps_sequences_apart(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(1, 9, 4)
        weight = torch.randn(4, 4)
        changed = x.clone()
        changed[:, 5:] += 10.0
        a = packed.reference_causal_conv(x, weight, None, None, _packing([5, 4]))
        b = packed.reference_causal_conv(changed, weight, None, None, _packing([5, 4]))
        torch.testing.assert_close(a[:, :5], b[:, :5])

    def test_attention_is_block_diagonal_and_bidirectional(self) -> None:
        torch.manual_seed(2)
        lengths, heads, kv_heads, dim = [4, 3, 5], 4, 2, 8
        total = sum(lengths)
        q = torch.randn(total, heads, dim)
        k = torch.randn(total, kv_heads, dim)
        v = torch.randn(total, kv_heads, dim)
        out = packed.reference_attention(q, k, v, _packing(lengths), dim**-0.5)
        segment = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
        mask = segment[:, None] == segment[None, :]
        expected = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).unsqueeze(0),
            v.repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1).unsqueeze(0),
            attn_mask=mask,
        )[0].transpose(0, 1)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_rms_norm_without_residual_is_qwen3_5_rms_norm(self) -> None:
        torch.manual_seed(3)
        x = torch.randn(2, 5, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, dtype=torch.bfloat16) * 0.1
        # Qwen3_5RMSNorm.forward (zero-centred weight, float32 math, cast back).
        xf = x.float()
        expected = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + weight.float())).type_as(x)
        normed, residual = packed.reference_rms_norm(x, 1.0 + weight.float(), 1e-6)
        assert torch.equal(normed, expected)
        assert residual is x

    def test_rms_norm_adds_the_residual_first(self) -> None:
        torch.manual_seed(4)
        x = torch.randn(1, 6, 16, dtype=torch.bfloat16)
        residual = torch.randn(1, 6, 16, dtype=torch.bfloat16)
        scale = torch.rand(16) + 0.5
        normed, summed = packed.reference_rms_norm(x, scale, 1e-6, residual)
        # The residual stream is the stock bf16 add, bit for bit ...
        assert summed.dtype == torch.bfloat16
        assert torch.equal(summed, residual + x)
        # ... and the norm reads the unrounded float32 sum.
        total = x.float() + residual.float()
        expected = total * torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + 1e-6) * scale
        assert torch.equal(normed, expected.to(torch.bfloat16))
        stock = packed.reference_rms_norm(residual + x, scale, 1e-6)[0]
        torch.testing.assert_close(normed.float(), stock.float(), rtol=0.02, atol=0.02)

    def test_gated_rms_norm_is_the_float32_formula(self) -> None:
        torch.manual_seed(5)
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        gate = torch.randn(8, 16, dtype=torch.bfloat16)
        weight = (torch.rand(16) + 0.5).to(torch.bfloat16)
        out = packed.reference_gated_rms_norm(x, gate, weight, 1e-6)
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert torch.equal(out, (normed * weight.float() * F.silu(gate.float())).to(torch.bfloat16))
        # Close to transformers' Qwen3_5RMSNormGated, which rounds to bf16 between steps.
        stock = (weight * normed.to(torch.bfloat16)) * F.silu(gate.float())
        torch.testing.assert_close(out.float(), stock.to(torch.bfloat16).float(), rtol=0.02, atol=0.02)

    def test_swiglu(self) -> None:
        torch.manual_seed(6)
        gate, up = torch.randn(3, 10, dtype=torch.bfloat16), torch.randn(3, 10, dtype=torch.bfloat16)
        out = packed.reference_swiglu(gate, up)
        assert out.dtype == torch.bfloat16
        assert torch.equal(out, (F.silu(gate.float()) * up.float()).to(torch.bfloat16))


def tiny_qwen3_5_text_model() -> Any:
    """A random 4-layer Qwen3.5 text model: three Gated DeltaNet layers, then full attention.

    Norm weights are randomized (zero-centred RMSNorms around 0, the Gated DeltaNet
    norm around 1), and there are more value heads than key heads, so the
    grouped-value repeat runs too. Skips without transformers >= 5.2.
    """
    pytest.importorskip("transformers.models.qwen3_5", reason="needs transformers >= 5.2 (the transformers5 bundle)")
    from transformers.models.qwen3_5 import (  # ty: ignore[unresolved-import]
        Qwen3_5TextConfig,
        Qwen3_5TextModel,
    )

    torch.manual_seed(7)
    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=256,
    )
    config.is_causal = False
    config.use_cache = False
    model = Qwen3_5TextModel(config).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("linear_attn.norm.weight"):
                parameter.normal_(1.0, 0.2)
            elif "norm" in name:
                parameter.normal_(0.0, 0.2)
    for layer in model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.is_causal = False
    return model


class TestPackedTextModel:
    """The packed forward (reference kernels) against the stock text model run one input at a time."""

    @pytest.fixture(scope="class")
    def model(self) -> Any:
        return tiny_qwen3_5_text_model()

    def test_matches_the_stock_model_per_input(self, model: Any) -> None:
        assert [
            type(layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn).__name__
            for layer in model.layers
        ] == [
            "Qwen3_5GatedDeltaNet",
            "Qwen3_5GatedDeltaNet",
            "Qwen3_5GatedDeltaNet",
            "Qwen3_5Attention",
        ]
        torch.manual_seed(8)
        lengths = [7, 1, 70, 12]  # 70 spans two delta-rule chunks
        embeds = [torch.randn(1, n, model.config.hidden_size) for n in lengths]
        positions = [torch.arange(n).view(1, 1, -1).expand(3, 1, -1) for n in lengths]
        text = packed.PackedTextModel(model, packed.reference_kernels())
        with torch.inference_mode():
            ours = text(torch.cat(embeds, dim=1), torch.cat(positions, dim=2), _packing(lengths))
            start = 0
            for n, embed, position in zip(lengths, embeds, positions, strict=True):
                stock = model(
                    inputs_embeds=embed, attention_mask=torch.ones(1, n, dtype=torch.long), position_ids=position
                ).last_hidden_state
                torch.testing.assert_close(ours[:, start : start + n], stock, rtol=1e-4, atol=1e-4)
                start += n

    def test_rejects_a_non_silu_mlp(self, model: Any) -> None:
        with patch.object(model.config, "hidden_act", "gelu"), pytest.raises(ValueError, match="SiLU"):
            packed.PackedTextModel(model, packed.reference_kernels())


class _IdentityText:
    """Stands in for ``PackedTextModel``: records its inputs and returns the embeddings."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, embeds: torch.Tensor, position_ids: torch.Tensor, packing: packed.Packing) -> torch.Tensor:
        self.calls.append({"embeds": embeds.shape, "positions": position_ids, "packing": packing})
        return embeds


class TestAdapterPacking:
    """With an identity backbone, the packed and padded paths must give the same vectors."""

    @pytest.fixture
    def adapters(self) -> tuple[TopkEmbedAdapter, TopkEmbedAdapter, _IdentityText]:
        text = _IdentityText()
        packed_adapter = make_adapter(packed=True)
        packed_adapter._packed_text = text  # ty: ignore[invalid-assignment]
        return make_adapter(), packed_adapter, text

    def test_text_matches_the_padded_path(
        self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter, _IdentityText]
    ) -> None:
        padded_adapter, packed_adapter, text = adapters
        items = [Item(text="a short one"), Item(text="a somewhat longer document, with commas."), Item(text="x")]
        ours = packed_adapter.encode(items, ["multivector"])
        reference = padded_adapter.encode(items, ["multivector"])
        assert ours.multivector is not None
        assert reference.multivector is not None
        for a, b in zip(ours.multivector, reference.multivector, strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        assert ours.extra["input_token_counts"] == reference.extra["input_token_counts"]
        # One packed call: boundaries at the (length-sorted) row ends, positions restart per row.
        assert len(text.calls) == 1
        lengths = sorted(reference.extra["input_token_counts"])
        packing = text.calls[0]["packing"]
        assert packing.cu_seqlens.tolist() == [0, *np.cumsum(lengths).tolist()]
        assert packing.max_len == max(lengths)
        positions = text.calls[0]["positions"]
        assert positions.shape == (3, 1, sum(lengths))
        assert positions[0, 0, : lengths[0]].tolist() == list(range(lengths[0]))
        assert positions[0, 0, lengths[0]] == 0

    def test_images_match_the_padded_path(
        self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter, _IdentityText]
    ) -> None:
        padded_adapter, packed_adapter, text = adapters
        items = [Item(images=[_png()]), Item(images=[_png("black", (96, 64))])]
        with _fake_vision(packed_adapter):
            ours = packed_adapter.encode(items, ["multivector"])
        with _fake_vision(padded_adapter):
            reference = padded_adapter.encode(items, ["multivector"])
        assert ours.multivector is not None
        assert reference.multivector is not None
        for a, b in zip(ours.multivector, reference.multivector, strict=True):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
        assert text.calls[0]["positions"].shape[0] == 3

    def test_packed_batches_budget_the_sum_of_lengths(
        self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter, _IdentityText]
    ) -> None:
        padded_adapter, packed_adapter, _ = adapters
        padded_adapter._text_batch_tokens = packed_adapter._text_batch_tokens = 10
        lengths = [2, 2, 3, 3, 9]
        # Padded: count x longest; packed: plain sum, so more rows fit per batch.
        assert padded_adapter._plan_text_batches(lengths) == [[0, 1, 2], [3], [4]]
        assert packed_adapter._plan_text_batches(lengths) == [[0, 1, 2, 3], [4]]

    def test_unload_forgets_the_packed_model(
        self, adapters: tuple[TopkEmbedAdapter, TopkEmbedAdapter, _IdentityText]
    ) -> None:
        _, packed_adapter, _ = adapters
        packed_adapter.unload()
        assert packed_adapter._packed_text is None
