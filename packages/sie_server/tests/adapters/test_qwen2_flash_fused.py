"""Qwen2FlashAdapter's fused-kernel layers: opt-in per profile, bit-identical to the eager layers on CUDA."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from sie_server.adapters import qwen2_flash
from sie_server.adapters.qwen2_flash import Qwen2FlashAdapter
from sie_server.core.loader import load_model_config

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

try:
    import triton  # ty: ignore[unresolved-import]

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

needs_cuda_triton = pytest.mark.skipif(not (torch.cuda.is_available() and _HAS_TRITON), reason="needs CUDA and Triton")


def test_qwen3_embedding_8b_turns_fused_kernels_on() -> None:
    profile = load_model_config(_MODELS_DIR / "Qwen__Qwen3-Embedding-8B.yaml").resolve_profile("default")
    assert profile.loadtime["fused_kernels"] is True


def test_fused_kernels_are_off_unless_a_profile_asks() -> None:
    assert Qwen2FlashAdapter("unused")._fused_kernels is False


def test_fused_kernels_need_a_silu_mlp() -> None:
    assert not qwen2_flash._fused_kernels_supported(SimpleNamespace(hidden_act="gelu"))
    assert qwen2_flash._fused_kernels_supported(SimpleNamespace(hidden_act="silu")) is _HAS_TRITON


def test_the_option_routes_the_layers_to_the_fused_path(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = Qwen2FlashAdapter("unused", fused_kernels=True)
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(adapter, "_run_transformer_fused", lambda *args: calls.append(args) or args[0])
    hidden = torch.zeros(3, 4)
    assert adapter._run_transformer_flash(hidden, torch.tensor([0, 3]), 3, 3, torch.arange(3)) is hidden
    assert len(calls) == 1


@needs_cuda_triton
def test_each_fused_op_is_bit_identical_to_the_eager_ops() -> None:
    from sie_server.adapters._utils import apply_rotary_pos_emb
    from sie_server.adapters.qwen2_flash._fused_ops import rms_norm, rotary_, silu_mul
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    tokens, hidden_size, head_dim, heads, kv_heads, inter = 777, 512, 128, 8, 2, 1536
    norm = Qwen3RMSNorm(hidden_size, eps=1e-6).to(dev, dt)
    head_norm = Qwen3RMSNorm(head_dim, eps=1e-6).to(dev, dt)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden_size, device=dev, dtype=dt) * 0.1 + 1)
        head_norm.weight.copy_(torch.randn(head_dim, device=dev, dtype=dt) * 0.1 + 1)
    x = torch.randn(tokens, hidden_size, device=dev, dtype=dt) * 3
    residual = torch.randn(tokens, hidden_size, device=dev, dtype=dt) * 10
    query = torch.randn(tokens, heads, head_dim, device=dev, dtype=dt)
    key = torch.randn(tokens, kv_heads, head_dim, device=dev, dtype=dt)
    positions = torch.cat([torch.arange(n, device=dev) for n in (300, 400, 77)]).float()
    inv_freq = 1.0 / (1_000_000 ** (torch.arange(0, head_dim, 2, device=dev).float() / head_dim))
    angles = torch.cat([positions[:, None] * inv_freq] * 2, dim=-1)
    cos, sin = angles.cos().to(dt), angles.sin().to(dt)
    gate = torch.randn(tokens, inter, device=dev, dtype=dt) * 4
    up = torch.randn(tokens, inter, device=dev, dtype=dt)

    with torch.no_grad():
        out, summed = rms_norm(x, norm.weight, norm.variance_epsilon, residual)
        assert torch.equal(summed, residual + x)
        assert torch.equal(out, norm(residual + x))
        out, same = rms_norm(x, norm.weight, norm.variance_epsilon)
        assert same is x
        assert torch.equal(out, norm(x))
        assert torch.equal(rms_norm(query, head_norm.weight, head_norm.variance_epsilon)[0], head_norm(query))
        eager_q, eager_k = apply_rotary_pos_emb(query, key, cos, sin)
        assert torch.equal(rotary_(query.clone(), cos, sin), eager_q)
        assert torch.equal(rotary_(key.clone(), cos, sin), eager_k)
        assert torch.equal(silu_mul(gate, up), torch.nn.functional.silu(gate) * up)


@needs_cuda_triton
def test_fused_layers_match_the_eager_layers_on_a_small_qwen3() -> None:
    pytest.importorskip("flash_attn")
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=1024,
    )
    model = Qwen3Model(config).to("cuda", torch.bfloat16).eval()
    lengths = [5, 300, 1, 64]
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    total = sum(lengths)
    position_ids = torch.cat([torch.arange(n) for n in lengths]).to("cuda")
    hidden = model.embed_tokens(torch.randint(0, 128, (total,), device="cuda"))

    def run(fused: bool) -> torch.Tensor:
        adapter = Qwen2FlashAdapter("unused", causal=True, fused_kernels=fused)
        adapter._model = model
        adapter._device = "cuda"
        with torch.inference_mode():
            return adapter._run_transformer_flash(hidden, cu_seqlens, max(lengths), total, position_ids)

    assert torch.equal(run(fused=True), run(fused=False))
