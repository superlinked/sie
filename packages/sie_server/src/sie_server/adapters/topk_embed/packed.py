"""Packed (variable-length) execution for TopK-Embed-V1.

A batch runs as one sequence: the inputs are concatenated with no padding, and
cumulative sequence lengths (``cu_seqlens``) keep them apart in every layer. This
is how TopK's own pipeline runs the backbone (``hf_backbone.py`` in the model
repo), and it avoids the padding the row-per-input path computes on.

On CUDA the kernels are flash-linear-attention's (the Gated DeltaNet chunked
delta rule and the causal conv, both variable-length) and FlashAttention's
variable-length attention. The reference kernels run the same math one sequence
at a time in PyTorch; they back the CPU tests and are never chosen automatically.

The layer functions reuse the loaded ``transformers.models.qwen3_5`` modules'
weights and sub-modules and mirror their forward passes, minus the KV cache and
padding masks that a packed, cache-free encoder does not need.
"""

from __future__ import annotations

import inspect
import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PackedKernels:
    """The three variable-length kernels the packed forward needs.

    ``delta_rule(q, k, v, *, g, beta, cu_seqlens)`` returns ``[1, T, HV, V]``;
    ``causal_conv(x, weight, bias, activation, cu_seqlens)`` maps ``[1, T, D]`` to
    ``[1, T, D]``; ``attention(q, k, v, cu_seqlens, max_len, scale)`` takes
    ``[T, H, d]`` queries and ``[T, H_kv, d]`` keys and values (bidirectional).
    """

    delta_rule: Callable[..., torch.Tensor]
    causal_conv: Callable[..., torch.Tensor]
    attention: Callable[..., torch.Tensor]
    names: dict[str, str] = field(default_factory=dict)


def resolve_kernels(device: str, *, mode: bool | None) -> PackedKernels | None:
    """Pick the packed kernels for ``device``.

    ``mode=None`` (auto) packs only on CUDA with flash-linear-attention importable;
    otherwise it returns ``None`` and the caller keeps its row-per-input path.
    ``mode=True`` always packs, with the reference kernels where the fast ones are
    missing. ``mode=False`` never packs.
    """
    if mode is False:
        return None
    fast = _fast_kernels() if str(device).startswith("cuda") else None
    if fast is not None:
        return fast
    if mode is None:
        return None
    return PackedKernels(
        delta_rule=reference_delta_rule,
        causal_conv=reference_causal_conv,
        attention=reference_attention,
        names={"delta_rule": "reference", "causal_conv": "reference", "attention": "reference"},
    )


def _fast_kernels() -> PackedKernels | None:
    try:
        from fla.modules.convolution import causal_conv1d  # ty: ignore[unresolved-import]
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # ty: ignore[unresolved-import]
    except Exception:  # noqa: BLE001 - fla missing or unusable (no Triton): not packing
        logger.info("flash-linear-attention unavailable; TopK-Embed keeps the row-per-input path")
        return None

    def delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        return out

    def causal_conv(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        out, _ = causal_conv1d(x=x, weight=weight, bias=bias, activation=activation, cu_seqlens=cu_seqlens)
        return out

    attention, attention_name = reference_attention, "sdpa per sequence"
    try:
        from flash_attn import flash_attn_varlen_func  # ty: ignore[unresolved-import]

        def attention(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, max_len: int, scale: float
        ) -> torch.Tensor:
            cu = cu_seqlens.to(torch.int32)
            return flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len, softmax_scale=scale, causal=False)

        attention_name = "flash_attn varlen"
    except ImportError:
        logger.info("flash_attn unavailable; TopK-Embed packs attention with per-sequence sdpa")
    return PackedKernels(
        delta_rule=delta_rule,
        causal_conv=causal_conv,
        attention=attention,
        names={"delta_rule": "fla chunk", "causal_conv": "fla", "attention": attention_name},
    )


def segments(cu_seqlens: torch.Tensor) -> list[tuple[int, int]]:
    return list(itertools.pairwise(cu_seqlens.tolist()))


# ---------------------------------------------------------------------------
# Reference kernels (PyTorch, one sequence at a time)
# ---------------------------------------------------------------------------


def reference_delta_rule(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, g: torch.Tensor, beta: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """The PyTorch chunked delta rule from transformers, run per packed sequence."""
    from transformers.models.qwen3_5 import modeling_qwen3_5  # ty: ignore[unresolved-import]

    rule = inspect.unwrap(modeling_qwen3_5.torch_chunk_gated_delta_rule)
    outs = [
        rule(
            q[:, s:e],
            k[:, s:e],
            v[:, s:e],
            g=g[:, s:e],
            beta=beta[:, s:e],
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )[0]
        for s, e in segments(cu_seqlens)
    ]
    return torch.cat(outs, dim=1)


def reference_causal_conv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """Depthwise causal conv per packed sequence: left-pad by ``width - 1``, then activate."""
    width = weight.shape[-1]
    outs = []
    for s, e in segments(cu_seqlens):
        seq = F.pad(x[:, s:e].transpose(1, 2), (width - 1, 0))
        out = F.conv1d(seq, weight.unsqueeze(1), bias, groups=x.shape[-1])
        outs.append(F.silu(out) if activation in ("silu", "swish") else out)
    return torch.cat(outs, dim=2).transpose(1, 2)


def reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor, max_len: int, scale: float
) -> torch.Tensor:
    """Bidirectional attention per packed sequence (``[T, H, d]`` in and out)."""
    del max_len
    outs = []
    for s, e in segments(cu_seqlens):
        out = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1).unsqueeze(0),
            k[s:e].transpose(0, 1).unsqueeze(0),
            v[s:e].transpose(0, 1).unsqueeze(0),
            scale=scale,
            enable_gqa=True,
        )
        outs.append(out[0].transpose(0, 1))
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# Packed forward over the stock Qwen3.5 modules
# ---------------------------------------------------------------------------


def text_forward(
    language_model: Any,
    embeds: torch.Tensor,
    position_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_len: int,
    kernels: PackedKernels,
) -> torch.Tensor:
    """``Qwen3_5TextModel.forward`` for one packed sequence ``[1, T, H]``.

    ``position_ids`` are ``[3, 1, T]`` (temporal, height, width), restarting per
    packed input.
    """
    position_embeddings = language_model.rotary_emb(embeds, position_ids)
    hidden = embeds
    for layer in language_model.layers[: language_model.config.num_hidden_layers]:
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        if hasattr(layer, "linear_attn"):
            hidden = gated_delta_net(layer.linear_attn, hidden, cu_seqlens, kernels)
        else:
            hidden = full_attention(layer.self_attn, hidden, position_embeddings, cu_seqlens, max_len, kernels)
        hidden = residual + hidden
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return language_model.norm(hidden)


def gated_delta_net(attn: Any, hidden: torch.Tensor, cu_seqlens: torch.Tensor, kernels: PackedKernels) -> torch.Tensor:
    """``Qwen3_5GatedDeltaNet.forward`` without the cache, over packed sequences."""
    batch, length, _ = hidden.shape
    mixed = attn.in_proj_qkv(hidden)
    z = attn.in_proj_z(hidden).reshape(batch, length, -1, attn.head_v_dim)
    b = attn.in_proj_b(hidden)
    a = attn.in_proj_a(hidden)
    mixed = kernels.causal_conv(mixed, attn.conv1d.weight.squeeze(1), attn.conv1d.bias, attn.activation, cu_seqlens)
    query, key, value = torch.split(mixed, [attn.key_dim, attn.key_dim, attn.value_dim], dim=-1)
    query = query.reshape(batch, length, -1, attn.head_k_dim)
    key = key.reshape(batch, length, -1, attn.head_k_dim)
    value = value.reshape(batch, length, -1, attn.head_v_dim)
    beta = b.sigmoid()
    # float32 as transformers does: with fp16 weights A would otherwise overflow to -inf.
    g = -attn.A_log.float().exp() * F.softplus(a.float() + attn.dt_bias)
    repeats = attn.num_v_heads // attn.num_k_heads
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)
    out = kernels.delta_rule(query, key, value, g=g, beta=beta, cu_seqlens=cu_seqlens)
    out = attn.norm(out.reshape(-1, attn.head_v_dim), z.reshape(-1, attn.head_v_dim))
    return attn.out_proj(out.reshape(batch, length, -1))


def full_attention(
    attn: Any,
    hidden: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cu_seqlens: torch.Tensor,
    max_len: int,
    kernels: PackedKernels,
) -> torch.Tensor:
    """``Qwen3_5Attention.forward`` (gated output, partial rotary), bidirectional per packed sequence."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb  # ty: ignore[unresolved-import]

    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    query, gate = torch.chunk(attn.q_proj(hidden).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    query = attn.q_norm(query.view(hidden_shape)).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
    value = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    out = kernels.attention(
        query[0].transpose(0, 1).contiguous(),
        key[0].transpose(0, 1).contiguous(),
        value[0].transpose(0, 1).contiguous(),
        cu_seqlens,
        max_len,
        attn.scaling,
    )
    out = out.reshape(*input_shape, -1).contiguous() * torch.sigmoid(gate)
    return attn.o_proj(out)
