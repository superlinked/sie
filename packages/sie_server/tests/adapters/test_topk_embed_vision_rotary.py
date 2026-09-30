"""The vision tower's fused rotary kernel: used only on CUDA, bit-identical to the PyTorch chain it replaces."""

from __future__ import annotations

import pytest
import torch
from sie_server.adapters.topk_embed import adapter as adapter_module
from sie_server.adapters.topk_embed import vision_rotary


def _eager(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The PyTorch chain: float32 rotation, rounded back to the input dtype."""
    xf = x.float()
    c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    return ((xf * c) + (adapter_module._rotate_half(xf) * s)).to(x.dtype)


def _inputs(device: str, tokens: int = 37, heads: int = 12, dim: int = 64) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    # A strided view, as the vision tower's unbind of its fused qkv projection gives.
    qkv = torch.randn(tokens, 3, heads, dim, device=device, dtype=torch.bfloat16)
    query = qkv.permute(1, 0, 2, 3).unbind(0)[0]
    angles = torch.randn(tokens, dim, device=device).to(torch.bfloat16)
    return query, angles.cos(), angles.sin()


def test_cpu_tensors_keep_the_pytorch_chain() -> None:
    query, cos, sin = _inputs("cpu")
    assert not vision_rotary.available(query)
    out_q, out_k = adapter_module._apply_rotary_pos_emb_vision(query, query, cos, sin)
    assert torch.equal(out_q, _eager(query, cos, sin))
    assert torch.equal(out_k, out_q)


@pytest.mark.skipif(not torch.cuda.is_available() or vision_rotary._kernel is None, reason="needs CUDA and Triton")
@pytest.mark.parametrize(("tokens", "heads", "dim"), [(37, 12, 64), (4920, 16, 64), (5, 3, 80)])
def test_the_kernel_is_bit_identical_on_cuda(tokens: int, heads: int, dim: int) -> None:
    query, cos, sin = _inputs("cuda", tokens, heads, dim)
    assert vision_rotary.available(query)
    assert torch.equal(vision_rotary.rotate(query, cos, sin), _eager(query, cos, sin))
