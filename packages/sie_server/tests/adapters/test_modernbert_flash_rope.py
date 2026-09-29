"""The shared ModernBERT layer stack rotates queries and keys with one fused kernel on CUDA.

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

import pytest
import torch
from sie_server.adapters import _modernbert_flash as stack
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


def test_off_cuda_the_stack_rotates_with_pytorch_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "flash_attn", types.SimpleNamespace(flash_attn_varlen_func=_reference_varlen))

    def unexpected(*args: object) -> None:
        raise AssertionError("the fused kernel runs only on CUDA")

    monkeypatch.setattr(stack, "rotate_packed_qkv_", unexpected)
    model, args = _model(), _inputs([3, 9, 1])
    with torch.inference_mode():
        stack.run_modernbert_flash_layers(model, *args)


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
        fused = stack.run_modernbert_flash_layers(model, *args)
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

    def run() -> torch.Tensor:
        with torch.inference_mode():
            return stack.run_modernbert_flash_layers(model, hidden, cu, max(lengths), total, *tables[0], *tables[1])

    fused = run()

    def flash_rotation(query, key, cos, sin):
        qk = torch.stack([query, key], dim=1).contiguous().view(1, total, -1, 64)
        half = cos.shape[-1] // 2
        rotary.apply_rotary(qk, cos[:, :half].contiguous(), sin[:, :half].contiguous(), inplace=True)
        qk = qk.view(total, 2, -1, 64)
        return qk[:, 0], qk[:, 1]

    monkeypatch.setattr(stack, "packed_rope_available", lambda device: False)
    monkeypatch.setattr(stack, "apply_rotary_pos_emb", flash_rotation)
    assert torch.equal(fused, run())
    monkeypatch.undo()
    # the unfused rotation rounds three times, so it differs from both, but only by rounding
    monkeypatch.setattr(stack, "packed_rope_available", lambda device: False)
    unfused = run()
    assert not torch.equal(fused, unfused)
    cosine = torch.nn.functional.cosine_similarity(fused.float(), unfused.float(), dim=-1)
    assert cosine.min() > 0.99
