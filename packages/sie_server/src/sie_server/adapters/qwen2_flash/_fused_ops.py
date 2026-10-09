# Triton reads a kernel's parameter annotations itself, so the kernels' pointer and stride
# arguments stay unannotated.
# ruff: noqa: ANN001, ANN202
"""Fused Triton kernels for the elementwise work of a Qwen2/Qwen3 decoder layer.

Each kernel does in one pass what the eager layer does in a chain of PyTorch
ops, and rounds to the activation dtype at the same points the eager ops do, so
it reproduces their results bit for bit:

* ``rms_norm``: optional residual add, then HF's RMSNorm
  ``weight * (x * rsqrt(mean(x**2) + eps)).to(dtype)``. The add is rounded
  to the activation dtype first, as ``residual + hidden`` is, and the mean is
  PyTorch's own reduction of the same float32 squares.
* ``rotary_``: HF's ``q * cos + rotate_half(q) * sin`` in place, each product
  and the sum rounded as the eager bfloat16 ops round them.
* ``silu_mul``: ``silu(gate) * up``, the SiLU rounded before the product.

The eager layer launches about 48 elementwise kernels per layer and these about
14, which matters because each of those kernels reads and writes the whole
activation.
"""

from __future__ import annotations

import torch
import triton  # ty: ignore[unresolved-import]
import triton.language as tl  # ty: ignore[unresolved-import]
from triton.language.extra.cuda import libdevice  # ty: ignore[unresolved-import]

# Elements per program of the elementwise SiLU-product kernel.
_SILU_BLOCK = 1024


@triton.jit
def _round(x, dtype: tl.constexpr):
    """``x`` rounded to ``dtype`` to nearest-even, as PyTorch rounds each op's float result.

    The products, sums and casts are explicit intrinsics so the compiler can
    neither contract a product into the next add nor drop a round trip.
    """
    return tl.cast(x, dtype, fp_downcast_rounding="rtne")


@triton.jit
def _add_square_kernel(
    x_ptr,
    residual_ptr,
    summed_ptr,
    squares_ptr,
    stride_x,
    stride_residual,
    stride_summed,
    n_cols,
    has_residual: tl.constexpr,
    block: tl.constexpr,
):
    """``summed = x + residual`` rounded (or ``x``), and ``float(summed) ** 2`` for the variance."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if has_residual:
        residual = tl.load(residual_ptr + row * stride_residual + cols, mask=mask, other=0.0).to(tl.float32)
        summed = _round(libdevice.add_rn(x, residual), summed_ptr.dtype.element_ty)
        tl.store(summed_ptr + row * stride_summed + cols, summed, mask=mask)
        x = summed.to(tl.float32)
    tl.store(squares_ptr + row * n_cols + cols, libdevice.mul_rn(x, x), mask=mask)


@triton.jit
def _normalize_kernel(
    x_ptr,
    variance_ptr,
    weight_ptr,
    out_ptr,
    stride_x,
    stride_out,
    n_cols,
    eps,
    block: tl.constexpr,
):
    """``weight * (x * rsqrt(variance + eps)).to(dtype)``, rounded where HF's RMSNorm rounds."""
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, block)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = libdevice.rsqrt(libdevice.add_rn(tl.load(variance_ptr + row), eps))
    normed = _round(libdevice.mul_rn(x, rstd), out_ptr.dtype.element_ty).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out_ptr + row * stride_out + cols, _round(libdevice.mul_rn(normed, weight), out_ptr.dtype.element_ty), mask=mask
    )


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rms_norm(x + residual), x + residual)`` over the last dimension (``x`` alone without a residual).

    The variance is ``squares.mean(-1)`` in PyTorch over the same float32
    squares HF's ``x.float().pow(2)`` makes, so it is reduced in PyTorch's own
    order and matches HF's RMSNorm bit for bit; the add, the squares and the
    scaling run as two Triton kernels.
    """
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    n_rows, n_cols = x2.shape
    if residual is None:
        summed2 = x2
        residual2 = x2
    else:
        summed2 = torch.empty_like(x2)
        residual2 = residual.reshape(-1, shape[-1])
    squares = torch.empty(shape, dtype=torch.float32, device=x.device)
    out = torch.empty_like(x2)
    if n_rows:
        block = triton.next_power_of_2(n_cols)
        _add_square_kernel[(n_rows,)](
            x2,
            residual2,
            summed2,
            squares,
            x2.stride(0),
            residual2.stride(0),
            summed2.stride(0),
            n_cols,
            has_residual=residual is not None,
            block=block,
        )
        variance = squares.mean(-1)
        _normalize_kernel[(n_rows,)](
            summed2, variance, weight, out, summed2.stride(0), out.stride(0), n_cols, eps, block=block
        )
    return out.view(shape), (x if residual is None else summed2.view(shape))


@triton.jit
def _rotary_kernel(x_ptr, cos_ptr, sin_ptr, stride_token, stride_head, stride_table, half: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, half)
    base = x_ptr + token * stride_token + head * stride_head
    table = token * stride_table
    x1 = tl.load(base + offs)
    x2 = tl.load(base + half + offs)
    dtype = x1.dtype
    x1 = x1.to(tl.float32)
    x2 = x2.to(tl.float32)
    cos1 = tl.load(cos_ptr + table + offs).to(tl.float32)
    cos2 = tl.load(cos_ptr + table + half + offs).to(tl.float32)
    sin1 = tl.load(sin_ptr + table + offs).to(tl.float32)
    sin2 = tl.load(sin_ptr + table + half + offs).to(tl.float32)
    # q * cos + rotate_half(q) * sin, rotate_half(q) = cat(-x2, x1): round each product, then the sum.
    out1 = _round(
        libdevice.add_rn(
            _round(libdevice.mul_rn(x1, cos1), dtype).to(tl.float32),
            _round(libdevice.mul_rn(-x2, sin1), dtype).to(tl.float32),
        ),
        dtype,
    )
    out2 = _round(
        libdevice.add_rn(
            _round(libdevice.mul_rn(x2, cos2), dtype).to(tl.float32),
            _round(libdevice.mul_rn(x1, sin2), dtype).to(tl.float32),
        ),
        dtype,
    )
    tl.store(base + offs, out1)
    tl.store(base + half + offs, out2)


def rotary_(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` (``[tokens, heads, head_dim]``) in place by per-token ``cos``/``sin`` (``[tokens, head_dim]``)."""
    tokens, heads, head_dim = x.shape
    half = head_dim // 2
    if cos.shape != (tokens, head_dim) or sin.shape != cos.shape or head_dim % 2 or half & (half - 1):
        msg = "rotary_ needs cos/sin of shape [tokens, head_dim] and a head_dim that is twice a power of two"
        raise ValueError(msg)
    if x.stride(-1) != 1 or cos.stride(-1) != 1 or sin.stride(-1) != 1 or cos.stride(0) != sin.stride(0):
        msg = "rotary_ needs a contiguous last dimension and matching cos/sin layouts"
        raise ValueError(msg)
    if tokens:
        _rotary_kernel[(tokens, heads)](x, cos, sin, x.stride(0), x.stride(1), cos.stride(0), half=half)
    return x


@triton.jit
def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n_elements, block: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    mask = offs < n_elements
    gate = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # PyTorch's SiLU: x / (1 + exp(-x)) in float, rounded; then the product, rounded.
    dtype = out_ptr.dtype.element_ty
    silu = _round(libdevice.div_rn(gate, libdevice.add_rn(1.0, libdevice.exp(-gate))), dtype).to(tl.float32)
    tl.store(out_ptr + offs, _round(libdevice.mul_rn(silu, up), dtype), mask=mask)


def silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` for contiguous tensors of one shape."""
    if not (gate.is_contiguous() and up.is_contiguous()) or gate.shape != up.shape:
        msg = "silu_mul needs contiguous gate and up tensors of one shape"
        raise ValueError(msg)
    out = torch.empty_like(gate)
    n = gate.numel()
    if n:
        _silu_mul_kernel[(triton.cdiv(n, _SILU_BLOCK),)](gate, up, out, n, block=_SILU_BLOCK)
    return out
