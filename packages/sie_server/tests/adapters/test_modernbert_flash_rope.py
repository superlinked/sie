"""The shared ModernBERT layer stack rotates queries and keys with one fused kernel where a model opts in.

On CPU the tests stand a PyTorch reference in for the kernel and for
flash-attn's varlen attention, so they check what the stack hands the
kernel: the per-token cos/sin rows, the first half of each table, and the
queries and keys it then attends with. The ``gpu_hw`` test checks the fused
stack against the same loop rotated by flash-attn's own rotary kernel, bit for
bit.
"""

from __future__ import annotations

import sys
import types
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
import torch
from sie_server.adapters import _modernbert_flash as stack
from sie_server.adapters import _modernbert_flash_graphs as graphs_module
from sie_server.adapters._modernbert_flash import parse_fused_rope
from sie_server.adapters.colbert_modernbert_flash.adapter import ColBERTModernBERTFlashAdapter
from sie_server.adapters.modernbert_flash import ModernBERTFlashAdapter
from sie_server.core.loader import load_model_configs, reject_unknown_loadtime_options
from torch.nn import functional
from transformers import ModernBertConfig, ModernBertModel


def _reference_varlen(query, key, value, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs):
    out = torch.empty_like(query)
    window = kwargs.get("window_size", (-1, -1))
    for start, end in pairwise(cu_seqlens_q.tolist()):
        rows = [t[start:end].transpose(0, 1) for t in (query, key, value)]
        mask = None
        if window != (-1, -1):
            positions = torch.arange(end - start)
            mask = (positions[:, None] - positions[None, :]).abs() <= window[0]
        out[start:end] = functional.scaled_dot_product_attention(
            *rows, attn_mask=mask, scale=kwargs.get("softmax_scale")
        ).transpose(0, 1)
    return out


def _model() -> ModernBertModel:
    torch.manual_seed(0)
    config = ModernBertConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=2,
        global_attn_every_n_layers=3,
        local_attention=8,
        max_position_embeddings=128,
        pad_token_id=0,
        attn_implementation="eager",
    )
    return ModernBertModel(config).eval()


def _inputs(lengths: list[int]) -> tuple[torch.Tensor, ...]:
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    positions = torch.cat([torch.arange(n) for n in lengths])
    tables = [
        t
        for theta in (160000.0, 10000.0)
        for t in stack.modernbert_rope_cos_sin(positions, head_dim=32, theta=theta, dtype=torch.float32)
    ]
    return (torch.randn(total, 64), cu, max(lengths), total, *tables)


@pytest.mark.parametrize(("available", "fused_rope"), [(False, True), (True, False)], ids=["off-cuda", "not-opted-in"])
def test_the_stack_rotates_with_pytorch_operations_unless_fused_where_it_runs(
    monkeypatch: pytest.MonkeyPatch, available: bool, fused_rope: bool
) -> None:
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace(flash_attn_varlen_func=_reference_varlen))
    monkeypatch.setattr(stack, "packed_rope_available", lambda device: available)

    def unexpected(*args: object) -> None:
        raise AssertionError("the fused kernel runs only when opted in, on CUDA")

    monkeypatch.setattr(stack, "rotate_packed_qkv_", unexpected)
    model, args = _model(), _inputs([3, 9, 1])
    with torch.inference_mode():
        stack.run_modernbert_flash_layers(model, *args, fused_rope=fused_rope)


def test_the_kernel_gets_each_tokens_row_and_the_tables_first_half(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a PyTorch stand-in for the kernel, the fused stack matches the unfused one (float32)."""
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace(flash_attn_varlen_func=_reference_varlen))
    calls: list[tuple[int, int]] = []

    def rotate(qkv: torch.Tensor, rows: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
        calls.append((rows.shape[0], cos.shape[-1]))
        assert torch.equal(rows, torch.arange(qkv.shape[0], dtype=torch.int32))
        half = cos.shape[-1]
        x0, x1 = qkv[:, :2, :, :half], qkv[:, :2, :, half:]
        c, s = cos[rows.long()][:, None, None, :], sin[rows.long()][:, None, None, :]
        qkv[:, :2] = torch.cat([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1)

    model, args = _model(), _inputs([3, 9, 1, 20])
    with torch.inference_mode():
        unfused = stack.run_modernbert_flash_layers(model, *args)
        monkeypatch.setattr(stack, "packed_rope_available", lambda device: True)
        monkeypatch.setattr(stack, "rotate_packed_qkv_", rotate)
        fused = stack.run_modernbert_flash_layers(model, *args, fused_rope=True)
    assert calls == [(33, 16)] * 3
    torch.testing.assert_close(fused, unfused, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu_hw
def test_the_shared_layer_stack_rotates_with_flash_attn_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_modernbert_flash_layers`` on CUDA: flash-attn's rotation of every token, bit for bit."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rotary = pytest.importorskip("flash_attn.ops.triton.rotary")

    torch.manual_seed(0)
    config = ModernBertConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=3,
        num_attention_heads=2,
        global_attn_every_n_layers=3,
        local_attention=16,
        max_position_embeddings=512,
        pad_token_id=0,
        attn_implementation="eager",
    )
    model = ModernBertModel(config).eval().to("cuda", torch.bfloat16)
    lengths = [5, 200, 1, 64]
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])
    tables = [
        stack.modernbert_rope_cos_sin(positions, head_dim=64, theta=theta, dtype=torch.bfloat16)
        for theta in (160000.0, 10000.0)
    ]
    hidden = torch.randn(total, 128, device="cuda", dtype=torch.bfloat16)

    def run(*, fused_rope: bool) -> torch.Tensor:
        with torch.inference_mode():
            return stack.run_modernbert_flash_layers(
                model, hidden, cu, max(lengths), total, *tables[0], *tables[1], fused_rope=fused_rope
            )

    fused = run(fused_rope=True)
    unfused = run(fused_rope=False)

    def flash_rotation(query, key, cos, sin):
        qk = torch.stack([query, key], dim=1).contiguous().view(1, total, -1, 64)
        half = cos.shape[-1] // 2
        rotary.apply_rotary(qk, cos[:, :half].contiguous(), sin[:, :half].contiguous(), inplace=True)
        qk = qk.view(total, 2, -1, 64)
        return qk[:, 0], qk[:, 1]

    monkeypatch.setattr(stack, "apply_rotary_pos_emb", flash_rotation)
    assert torch.equal(fused, run(fused_rope=False))
    # the unfused rotation rounds three times, so it differs from the fused one, but only by rounding
    assert not torch.equal(fused, unfused)
    cosine = torch.nn.functional.cosine_similarity(fused.float(), unfused.float(), dim=-1)
    assert cosine.min() > 0.99


# -- the per-model option ----------------------------------------------------------

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
_ADAPTERS = (ModernBERTFlashAdapter, ColBERTModernBERTFlashAdapter)
# The shipped models whose profiles enable the fused rotation: those at least as
# accurate as the unfused one against float32 (see the server README).
_FUSED_BY_DEFAULT = {
    "nomic-ai/modernbert-embed-base",
    "lightonai/Reason-ModernColBERT",
    "lightonai/mLateOn",
}


@pytest.mark.parametrize("adapter", _ADAPTERS)
def test_fused_rope_is_a_boolean_load_time_option(adapter: Any) -> None:
    reject_unknown_loadtime_options(adapter, {"fused_rope": True}, model_name="m")
    assert adapter("m")._fused_rope is False
    assert adapter("m", fused_rope=True)._fused_rope is True
    for value in ("true", 1, None):
        with pytest.raises(ValueError, match="fused_rope must be true or false"):
            adapter("m", fused_rope=value)


@pytest.mark.parametrize("adapter", _ADAPTERS)
@pytest.mark.parametrize("fused_rope", [False, True])
def test_the_adapters_pass_their_option_to_the_stack(
    monkeypatch: pytest.MonkeyPatch, adapter: Any, fused_rope: bool
) -> None:
    seen: list[bool] = []

    def layers(model: Any, hidden: torch.Tensor, *args: Any, fused_rope: bool = False, **kwargs: Any) -> torch.Tensor:
        seen.append(fused_rope)
        return hidden

    monkeypatch.setattr(sys.modules[adapter.__module__], "run_modernbert_flash_layers", layers)
    monkeypatch.setattr(graphs_module, "run_modernbert_flash_layers", layers)
    instance = adapter("m", fused_rope=fused_rope)
    instance._model = _model()
    hidden = torch.zeros(4, 64)
    instance._run_transformer_flash(hidden, torch.tensor([0, 4], dtype=torch.int32), 4, 4, *_inputs([4])[4:])
    encode, _ = graphs_module.modernbert_encoder(
        instance._model, window=8, dtype=torch.float32, fused_rope=instance._fused_rope
    )
    ids = torch.zeros(4, dtype=torch.int32)
    encode(ids, ids, torch.tensor([0, 4], dtype=torch.int32), 4, 4)
    assert seen == [fused_rope, fused_rope]


def test_only_models_at_least_as_accurate_against_float32_ship_fused() -> None:
    fused = set()
    for config in load_model_configs(_MODELS_DIR).values():
        for name in config.profiles:
            profile = config.resolve_profile(name)
            if "fused_rope" in profile.loadtime:
                assert parse_fused_rope(profile.loadtime["fused_rope"], adapter=profile.adapter_path)
                fused.add(config.sie_id.split(":")[0])
    assert fused == _FUSED_BY_DEFAULT
