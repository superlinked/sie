"""Qwen2/Qwen3 decoder layers with their elementwise work in fused Triton kernels.

``run_layers`` is the layer loop of ``Qwen2FlashAdapter._run_transformer_flash``
with each residual add folded into the next RMSNorm, q/k-norm as one kernel per
tensor, the rotary embedding as one in-place kernel per tensor and the MLP's
``silu(gate) * up`` as one kernel (see ``_fused_ops``). The kernels round where
the eager ops round, so the result matches the eager layers bit for bit; the
projections and attention are the same calls.
"""

from __future__ import annotations

from typing import Any

import torch


def supported(model: Any) -> bool:
    """Whether ``run_layers`` can run ``model``.

    Needs Triton, a SiLU-gated MLP, and decoder layers with the modules the
    loop reads (checked on the first layer): separate gate/up/down projections
    and RMSNorms exposing ``weight`` and ``variance_epsilon``. A remote-code
    variant with a fused ``gate_up_proj`` or a norm with ``eps`` keeps the
    eager layers.
    """
    if getattr(model.config, "hidden_act", None) != "silu":
        return False
    try:
        import triton  # ty: ignore[unresolved-import]
    except ImportError:
        return False
    return _layer_layout(model.layers)


def _layer_layout(layers: Any) -> bool:
    if not len(layers):
        return False
    layer = layers[0]
    norms = [layer.input_layernorm, layer.post_attention_layernorm]
    attn = layer.self_attn
    norms += [getattr(attn, name) for name in ("q_norm", "k_norm") if hasattr(attn, name)]
    return all(hasattr(layer.mlp, name) for name in ("gate_proj", "up_proj", "down_proj")) and all(
        hasattr(norm, "weight") and hasattr(norm, "variance_epsilon") for norm in norms
    )


def run_layers(
    model: Any,
    hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """Run ``model``'s decoder layers over the packed ``hidden``; returns the residual stream before the final norm."""
    from flash_attn import flash_attn_varlen_func

    from sie_server.adapters.qwen2_flash._fused_ops import rms_norm, rotary_, silu_mul

    config = model.config
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
    softmax_scale = 1.0 / (head_dim**0.5)
    cos, sin = cos.contiguous(), sin.contiguous()

    residual: torch.Tensor | None = None
    for layer in model.layers:
        attn = layer.self_attn
        norm = layer.input_layernorm
        normed, residual = rms_norm(hidden, norm.weight, norm.variance_epsilon, residual)
        query = attn.q_proj(normed).view(total_tokens, num_heads, head_dim)
        key = attn.k_proj(normed).view(total_tokens, num_kv_heads, head_dim)
        value = attn.v_proj(normed).view(total_tokens, num_kv_heads, head_dim)
        if hasattr(attn, "q_norm"):
            query, _ = rms_norm(query, attn.q_norm.weight, attn.q_norm.variance_epsilon)
        if hasattr(attn, "k_norm"):
            key, _ = rms_norm(key, attn.k_norm.weight, attn.k_norm.variance_epsilon)
        query = rotary_(query.contiguous(), cos, sin)
        key = rotary_(key.contiguous(), cos, sin)
        attn_out = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=causal,
            softmax_scale=softmax_scale,
        ).reshape(total_tokens, num_heads * head_dim)
        norm = layer.post_attention_layernorm
        normed, residual = rms_norm(attn.o_proj(attn_out), norm.weight, norm.variance_epsilon, residual)
        mlp = layer.mlp
        hidden = mlp.down_proj(silu_mul(mlp.gate_proj(normed), mlp.up_proj(normed)))
    return hidden if residual is None else residual + hidden
