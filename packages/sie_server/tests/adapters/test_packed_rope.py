"""The packed rotary kernel: when it applies, what it accepts, and its arithmetic against flash-attn's."""

from __future__ import annotations

import pytest
import torch
from sie_server.adapters import _packed_rope
from sie_server.adapters._packed_rope import packed_rope_available, rotate_packed_qkv_


def _inputs(
    tokens: int = 5, heads: int = 2, head_dim: int = 8, dtype: torch.dtype = torch.float16
) -> tuple[torch.Tensor, ...]:
    qkv = torch.zeros(tokens, 3, heads, head_dim, dtype=dtype)
    positions = torch.arange(tokens, dtype=torch.int32)
    cos = torch.ones(16, head_dim // 2, dtype=dtype)
    return qkv, positions, cos, torch.zeros_like(cos)


def test_it_applies_on_cuda_with_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_packed_rope, "_TRITON", True)
    assert packed_rope_available("cuda:0")
    assert packed_rope_available(torch.device("cuda", 1))
    assert not packed_rope_available("cpu")
    assert not packed_rope_available("mps")
    monkeypatch.setattr(_packed_rope, "_TRITON", False)
    assert not packed_rope_available("cuda:0")
    with pytest.raises(RuntimeError, match="needs Triton"):
        rotate_packed_qkv_(*_inputs())


@pytest.mark.skipif(not _packed_rope._TRITON, reason="requires Triton")
@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda q, p, c, s: (q[:, :2], p, c, s), "contiguous"),
        (lambda q, p, c, s: (torch.zeros(5, 3, 2, 7, dtype=q.dtype), p, c, s), "contiguous"),  # odd head_dim
        (lambda q, p, c, s: (q, p, c[:, :3], s[:, :3]), "cos and sin tables"),
        (lambda q, p, c, s: (q, p, c, s[:8]), "cos and sin tables"),
        (lambda q, p, c, s: (q, p, c.float(), s.float()), "must be torch.float16"),
        (lambda q, p, c, s: (q, p[:4], c, s), "expected 5 positions"),
    ],
)
def test_layouts_it_does_not_handle_are_refused(change, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rotate_packed_qkv_(*change(*_inputs()))


@pytest.mark.skipif(not _packed_rope._TRITON, reason="requires Triton")
def test_no_tokens_launch_nothing() -> None:
    rotate_packed_qkv_(*_inputs(tokens=0))  # CPU tensors: a launch would fail


@pytest.mark.gpu_hw
def test_rotation_matches_flash_attn_bit_for_bit() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rotary = pytest.importorskip("flash_attn.ops.triton.rotary")
    torch.manual_seed(0)
    lengths = [7, 300, 1, 64, 129]
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    positions = torch.cat([torch.arange(n, device="cuda") for n in lengths]).int()
    for dtype in (torch.float16, torch.bfloat16):
        for heads, base in ((12, 160000.0), (6, 10000.0)):
            qkv = torch.randn(total, 3, heads, 64, device="cuda", dtype=dtype) * 3
            inv_freq = 1.0 / (base ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64))
            freqs = torch.outer(torch.arange(512, device="cuda", dtype=torch.float32), inv_freq)
            cos, sin = freqs.cos().to(dtype), freqs.sin().to(dtype)
            want = qkv.clone()
            rotary.apply_rotary(
                want[:, :2].view(total, -1, 64), cos, sin, cu_seqlens=cu, max_seqlen=max(lengths), inplace=True
            )
            got = qkv.clone()
            rotate_packed_qkv_(got, positions, cos, sin)
            assert torch.equal(got, want)
            # a [positions, head_dim] table's first half, as the shared layer stack holds them
            wide = qkv.clone()
            rotate_packed_qkv_(
                wide, positions.long(), torch.cat([cos, cos], -1)[:, :32], torch.cat([sin, sin], -1)[:, :32]
            )
            assert torch.equal(wide, want)


@pytest.mark.gpu_hw
def test_strided_positions_and_rows_outside_the_tables() -> None:
    """Positions are read at their stride; a position outside the tables leaves its token unrotated."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(0)
    qkv = torch.randn(4, 3, 2, 8, device="cuda", dtype=torch.float16)
    cos = torch.rand(16, 4, device="cuda", dtype=torch.float16)
    sin = torch.rand(16, 4, device="cuda", dtype=torch.float16)
    strided = torch.tensor([3, 99, 0, 99, 7, 99, 15, 99], device="cuda", dtype=torch.int32)[::2]
    got, want = qkv.clone(), qkv.clone()
    rotate_packed_qkv_(got, strided, cos, sin)
    rotate_packed_qkv_(want, strided.contiguous(), cos, sin)
    assert torch.equal(got, want)
    assert not torch.equal(got, qkv)

    outside = torch.tensor([2, -1, 16, 5], device="cuda", dtype=torch.int32)
    qkv[2, 0, 0, 5] = float("inf")  # left as is, not multiplied by a zero sine
    rotated = qkv.clone()
    rotate_packed_qkv_(rotated, outside, cos, sin)
    assert torch.equal(rotated[1:3], qkv[1:3])
    reference = qkv.clone()
    rotate_packed_qkv_(reference, torch.tensor([2, 0, 0, 5], device="cuda", dtype=torch.int32), cos, sin)
    assert torch.equal(rotated[[0, 3]], reference[[0, 3]])
