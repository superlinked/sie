# Triton reads a kernel's parameter annotations, and accepts only its own types
# there: the kernel's tensor and integer arguments stay unannotated.
# ruff: noqa: ANN001
"""Rotary position embedding for a packed token stream, one position per token.

Flash-attention encoders pack a batch into one unpadded token stream. The
rotary kernel flash-attn ships (``flash_attn.ops.triton.rotary``) finds each
token's position from ``cu_seqlens``: it launches one program per block of
``max_seqlen`` positions per sequence per head. That is the right launch for
a batch of real rows, but not for a CUDA graph, whose ``cu_seqlens`` has a
fixed number of sequence slots and whose ``max_seqlen`` is a bucket: most of
its programs would find no token to rotate.

:func:`rotate_packed_qkv_` takes each token's position from a tensor instead,
so its launch grows with the number of tokens only. Its arithmetic is
flash-attn's for the non-interleaved (GPT-NeoX) layout: the halves ``x0`` and
``x1`` of a head are read in float32 and written back rounded once, as
``x0 * cos - x1 * sin`` and ``x0 * sin + x1 * cos``, with the float32 cosine
and sine of the model's dtype tables.
"""

from __future__ import annotations

import torch

try:  # Triton ships with CUDA builds of PyTorch; CPU-only installs lack it.
    import triton  # ty: ignore[unresolved-import]
    import triton.language as tl  # ty: ignore[unresolved-import]
except ImportError:  # pragma: no cover - exercised only on installs without Triton
    _TRITON = False
else:
    _TRITON = True

# Tokens per program.
_BLOCK_TOKENS = 32


def packed_rope_available(device: str | torch.device) -> bool:
    """Whether :func:`rotate_packed_qkv_` can run on ``device``."""
    return _TRITON and str(device).startswith("cuda")


if _TRITON:

    @triton.jit(do_not_specialize=["n_tokens", "n_rows"])
    def _rotate_packed_kernel(
        qk_ptr,
        positions_ptr,
        cos_ptr,
        sin_ptr,
        n_tokens,
        n_rows,
        stride_position,
        stride_token,
        stride_head,
        stride_table,
        half: tl.constexpr,
        block_half: tl.constexpr,
        block_tokens: tl.constexpr,
    ) -> None:
        pid_tokens = tl.program_id(axis=0)
        pid_head = tl.program_id(axis=1)
        tokens = pid_tokens * block_tokens + tl.arange(0, block_tokens)
        cols = tl.arange(0, block_half)
        mask = (tokens[:, None] < n_tokens) & (cols[None, :] < half)
        positions = tl.load(positions_ptr + tokens * stride_position, mask=tokens < n_tokens, other=0)
        # A position outside the tables reads and writes nothing: its token is
        # left exactly as it was.
        mask = mask & ((positions >= 0) & (positions < n_rows))[:, None]
        table = positions[:, None] * stride_table + cols[None, :]
        cos = tl.load(cos_ptr + table, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + table, mask=mask, other=0.0).to(tl.float32)
        x = qk_ptr + tokens[:, None] * stride_token + pid_head * stride_head + cols[None, :]
        x0 = tl.load(x, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x + half, mask=mask, other=0.0).to(tl.float32)
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(x, o0, mask=mask)
        tl.store(x + half, o1, mask=mask)


def rotate_packed_qkv_(qkv: torch.Tensor, positions: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """Rotate the queries and keys of a packed QKV tensor in place.

    Args:
        qkv: ``[tokens, 3, heads, head_dim]``, contiguous, on CUDA. The
            queries (``qkv[:, 0]``) and keys (``qkv[:, 1]``) are rotated; the
            values are untouched.
        positions: ``[tokens]`` integer position of each token, any stride.
            A position outside ``[0, max_positions)`` leaves its token
            exactly as it was (flash-attn's rotary kernel, too, does not
            rotate past its tables): checking positions on the host would
            wait for the device, which a CUDA graph being recorded cannot do.
        cos, sin: ``[max_positions, head_dim / 2]`` tables in ``qkv``'s dtype,
            with rows laid out ``stride(0)`` apart (a ``[max_positions,
            head_dim]`` table's first half is accepted as a view).

    Raises:
        RuntimeError: If Triton is unavailable.
        ValueError: On a layout the kernel does not handle.
    """
    if not _TRITON:
        raise RuntimeError("rotate_packed_qkv_ needs Triton")
    tokens, three, heads, head_dim = qkv.shape
    half = head_dim // 2
    if three != 3 or head_dim % 2 or not qkv.is_contiguous():
        raise ValueError(f"expected a contiguous [tokens, 3, heads, head_dim] tensor, got {tuple(qkv.shape)}")
    if cos.shape[-1] != half or sin.shape != cos.shape or cos.stride(-1) != 1 or sin.stride() != cos.stride():
        raise ValueError(f"expected [positions, {half}] cos and sin tables with unit column stride")
    if cos.dtype != qkv.dtype or sin.dtype != qkv.dtype:
        raise ValueError(f"cos and sin must be {qkv.dtype}, like qkv")
    if positions.shape != (tokens,):
        raise ValueError(f"expected {tokens} positions, got {tuple(positions.shape)}")
    if not tokens:
        return
    # Queries and keys are the first two thirds of every token: heads 0..2H-1.
    grid = (triton.cdiv(tokens, _BLOCK_TOKENS), 2 * heads)
    with torch.cuda.device(qkv.device):
        _rotate_packed_kernel[grid](
            qkv,
            positions,
            cos,
            sin,
            tokens,
            cos.shape[0],
            positions.stride(0),
            qkv.stride(0),
            qkv.stride(2),
            cos.stride(0),
            half=half,  # ty: ignore[invalid-argument-type] -- Triton makes constexpr arguments of plain values
            block_half=triton.next_power_of_2(half),
            block_tokens=_BLOCK_TOKENS,  # ty: ignore[invalid-argument-type]
        )
