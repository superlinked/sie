"""Triton kernel of ``vision_rotary.py`` (imported only where Triton is installed)."""

# Triton reads a kernel's parameter annotations itself, so the kernel's pointer and stride
# arguments stay unannotated.
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import triton  # ty: ignore[unresolved-import]
import triton.language as tl  # ty: ignore[unresolved-import]


@triton.jit
def rotate_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    n_heads,
    stride_xt,
    stride_xh,
    stride_ct,
    stride_ot,
    stride_oh,
    half: tl.constexpr,
    block_h: tl.constexpr,
    block_d: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, block_h)
    # Triton ranges are powers of two: block_d covers half the head size, masked past it.
    dims = tl.arange(0, block_d)
    in_dims = dims < half
    in_heads = (heads < n_heads)[:, None] & in_dims[None, :]
    cos1 = tl.load(cos_ptr + t * stride_ct + dims, mask=in_dims).to(tl.float32)[None, :]
    cos2 = tl.load(cos_ptr + t * stride_ct + half + dims, mask=in_dims).to(tl.float32)[None, :]
    sin1 = tl.load(sin_ptr + t * stride_ct + dims, mask=in_dims).to(tl.float32)[None, :]
    sin2 = tl.load(sin_ptr + t * stride_ct + half + dims, mask=in_dims).to(tl.float32)[None, :]
    rows = x_ptr + t * stride_xt + heads[:, None] * stride_xh
    x1 = tl.load(rows + dims[None, :], mask=in_heads).to(tl.float32)
    x2 = tl.load(rows + half + dims[None, :], mask=in_heads).to(tl.float32)
    # rotate_half(x) is cat(-x2, x1): the same products and sums as the PyTorch chain.
    out1 = x1 * cos1 + (-x2) * sin1
    out2 = x2 * cos2 + x1 * sin2
    out = out_ptr + t * stride_ot + heads[:, None] * stride_oh
    tl.store(out + dims[None, :], out1.to(out_ptr.dtype.element_ty), mask=in_heads)
    tl.store(out + half + dims[None, :], out2.to(out_ptr.dtype.element_ty), mask=in_heads)
