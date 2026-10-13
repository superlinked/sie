"""Qwen2FlashCrossEncoderAdapter's fused-kernel layers: opt-in per profile, bit-identical to the eager layers on CUDA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from sie_server.adapters.qwen2_flash import _fused_layers
from sie_server.adapters.qwen2_flash_cross_encoder.adapter import Qwen2FlashCrossEncoderAdapter
from sie_server.core.loader import load_model_config

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

try:
    import triton  # ty: ignore[unresolved-import]

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False

needs_cuda_triton = pytest.mark.skipif(not (torch.cuda.is_available() and _HAS_TRITON), reason="needs CUDA and Triton")


def _small_reranker(layers: int = 3) -> Any:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=768,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=1024,
    )
    return Qwen3ForCausalLM(config)


def _adapter(model: Any, device: str, *, fused: bool) -> Qwen2FlashCrossEncoderAdapter:
    adapter = Qwen2FlashCrossEncoderAdapter(
        "unused", input_format="qwen3", score_mode="log_softmax", fused_kernels=fused
    )
    config = model.config
    adapter._model = model
    adapter._device = device
    adapter._dtype = model.dtype
    adapter._num_heads = config.num_attention_heads
    adapter._num_kv_heads = config.num_key_value_heads
    adapter._hidden_size = config.hidden_size
    adapter._head_dim = config.head_dim
    rows = torch.tensor([1, 2], device=device)
    adapter._score_weight = model.lm_head.weight.index_select(0, rows).detach().float()
    return adapter


@pytest.mark.parametrize("config", ["Qwen__Qwen3-Reranker-4B.yaml", "Qwen__Qwen3-Reranker-0.6B.yaml"])
def test_the_qwen3_rerankers_turn_fused_kernels_on(config: str) -> None:
    profile = load_model_config(_MODELS_DIR / config).resolve_profile("default")
    assert profile.loadtime["fused_kernels"] is True


def test_fused_kernels_are_off_unless_a_profile_asks() -> None:
    assert Qwen2FlashCrossEncoderAdapter("unused")._fused_kernels is False


def test_a_model_the_kernels_cannot_run_keeps_the_eager_layers(caplog: pytest.LogCaptureFixture) -> None:
    model = _small_reranker(layers=1)
    model.config.hidden_act = "gelu"
    adapter = _adapter(model, "cpu", fused=True)

    with caplog.at_level("WARNING"):
        adapter._resolve_serving_options("cpu")

    assert adapter._fused_kernels is False
    assert "runs the eager layers instead" in caplog.text


def test_a_supported_model_keeps_the_option(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(_small_reranker(layers=1), "cpu", fused=True)
    seen: list[Any] = []
    monkeypatch.setattr(_fused_layers, "supported", lambda model: seen.append(model) or True)

    adapter._resolve_serving_options("cpu")

    assert adapter._fused_kernels is True
    # The layer stack is the causal LM's base model.
    assert seen == [adapter._model.model]


def test_the_option_routes_the_layers_to_the_fused_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _adapter(_small_reranker(layers=1), "cpu", fused=True)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(_fused_layers, "run_layers", lambda *args, **kwargs: calls.append((args, kwargs)) or args[1])
    hidden, table = torch.zeros(3, 4), torch.zeros(3, 4)
    cu_seqlens = torch.tensor([0, 3])

    assert adapter._run_layers(hidden, cu_seqlens, 3, 3, table, table) is hidden

    ((args, kwargs),) = calls
    assert args[0] is adapter._model.model
    assert kwargs == {"causal": True, "attend": None}


@needs_cuda_triton
def test_fused_layers_score_as_the_eager_layers_bit_for_bit() -> None:
    pytest.importorskip("flash_attn")
    model = _small_reranker().to("cuda", torch.bfloat16).eval()
    lengths = [5, 300, 1, 64]
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    total = sum(lengths)
    position_ids = torch.cat([torch.arange(n) for n in lengths]).to("cuda")
    input_ids = torch.randint(0, 128, (total,), device="cuda")

    def logits(fused: bool) -> torch.Tensor:
        adapter = _adapter(model, "cuda", fused=fused)
        adapter._resolve_serving_options("cuda")
        assert adapter._fused_kernels is fused
        with torch.inference_mode():
            return adapter._forward_flash(input_ids, cu_seqlens, max(lengths), total, len(lengths), position_ids)

    fused, eager = logits(fused=True), logits(fused=False)
    assert fused.dtype == torch.float32
    assert torch.equal(fused, eager)


@needs_cuda_triton
def test_the_fused_loop_takes_the_callers_attention_step() -> None:
    pytest.importorskip("flash_attn")
    from flash_attn import flash_attn_varlen_func

    model = _small_reranker().to("cuda", torch.bfloat16).eval()
    base = model.model
    lengths = [7, 120, 33]
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    total = sum(lengths)
    adapter = _adapter(model, "cuda", fused=True)
    seen: list[int] = []

    def attend(index: int, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        seen.append(index)
        return flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(lengths),
            causal=True,
            softmax_scale=1.0 / (adapter._head_dim**0.5),
        )

    with torch.inference_mode():
        hidden = base.embed_tokens(torch.randint(0, 128, (total,), device="cuda"))
        positions = torch.cat([torch.arange(n) for n in lengths]).to("cuda")
        cos, sin = adapter._compute_rope(base.rotary_emb, positions)
        default = _fused_layers.run_layers(base, hidden, cu_seqlens, max(lengths), total, cos, sin, causal=True)
        hooked = _fused_layers.run_layers(
            base, hidden, cu_seqlens, max(lengths), total, cos, sin, causal=True, attend=attend
        )

    assert seen == [0, 1, 2]
    assert torch.equal(hooked, default)
