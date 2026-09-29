"""The vision tower's rotary embedding as one GPU kernel, bit-identical to the PyTorch ops it replaces.

In PyTorch the rotation of a query or key tensor is a chain of element-wise kernels:
two upcasts to float32, the half rotation (a concatenation), two multiplies, an add and
a downcast. On a page image (about 5,000 patches) that chain is about a fifth of the
vision tower's GPU time. The kernel below reads each tensor once and does the same
three rounded float32 operations per element (``x * cos``, ``rotate_half(x) * sin``,
their sum), then rounds to the input dtype. Fused multiply-add is turned off, so the
compiler cannot merge a multiply and the add into one rounding: the result is
bit-identical to the PyTorch chain, and the page vectors cannot move.

Only used on CUDA with Triton importable; anything else keeps the PyTorch chain.
"""

from __future__ import annotations

from typing import Any

import torch


def _load_kernel() -> Any:
    """The jitted kernel, or ``None`` on CPU-only installs (no Triton).

    Typed ``Any``: a Triton kernel takes plain ints for its constexpr parameters, and compiler
    options such as ``enable_fp_fusion``, at launch, which its Python signature does not describe.
    """
    try:
        from sie_server.adapters.topk_embed._vision_rotary_kernel import rotate_kernel
    except ImportError:  # CPU-only installs: no Triton
        return None
    return rotate_kernel


_kernel: Any = _load_kernel()


def available(tensor: torch.Tensor) -> bool:
    """Whether the kernel can rotate ``tensor`` (CUDA, Triton importable, an even head size)."""
    return _kernel is not None and tensor.is_cuda and tensor.shape[-1] % 2 == 0


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` (``[tokens, heads, dim]``) by ``cos``/``sin`` (``[tokens, dim]``); same dtype out."""
    kernel = _kernel
    if kernel is None:
        msg = "the fused vision rotary kernel needs Triton"
        raise RuntimeError(msg)
    tokens, heads, dim = x.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    cos, sin = cos.contiguous(), sin.contiguous()
    out = torch.empty((tokens, heads, dim), dtype=x.dtype, device=x.device)
    kernel[(tokens,)](
        x,
        cos,
        sin,
        out,
        heads,
        x.stride(0),
        x.stride(1),
        cos.stride(0),
        out.stride(0),
        out.stride(1),
        half=dim // 2,
        block_h=_next_power_of_2(heads),
        block_d=_next_power_of_2(dim // 2),
        enable_fp_fusion=False,
    )
    return out
