"""Shared ModernBERT backbone pieces for the flash-attention adapters.

ModernBERT-family encoders (ModernBERT, mmBERT, and checkpoints fine-tuned from
them) run the same pre-norm layer stack: alternating global and sliding-window
attention, each with its own RoPE base. Several adapters drive that stack with
``flash_attn_varlen_func`` over a packed, unpadded token stream (dense
embeddings, late interaction, Laya decisions). This module is the single copy
of the RoPE-base resolution and the varlen layer loop they share.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from sie_server.adapters._utils import apply_rotary_pos_emb

# ModernBertConfig class defaults in transformers 4.x.
_DEFAULT_GLOBAL_ROPE_THETA = 160000.0
_DEFAULT_LOCAL_ROPE_THETA = 10000.0


def nested_rope_theta(config: Any, layer_kind: str) -> float | None:
    """Return ``rope_theta`` for ``layer_kind`` from a transformers>=5 nested
    ``rope_parameters`` mapping, or None when it is absent or malformed.

    transformers 5.x re-serializes ModernBERT rope as
    ``rope_parameters[{"full_attention", "sliding_attention"}]["rope_theta"]``
    and drops the flat ``global_rope_theta``/``local_rope_theta`` keys. The 4.x
    config class does not ingest the nested mapping — ``PretrainedConfig``
    retains it as an opaque attribute while the flat attrs fill in from class
    defaults — so without this read a 5.x-serialized checkpoint is silently
    served with sliding-window layers at the class-default theta 10000 instead
    of its trained value (mmBERT-based checkpoints train both layer kinds at
    160000; the #2807 parity probe measured per-token cosine vs pylate down to
    0.588 from this alone).

    Precedence: nested-when-present wins — a 5.x-serialized config is
    authoritative about itself, and 4.x-written configs carry no
    ``rope_parameters`` key, so their flat-attr path stays byte-identical. A
    malformed nested declaration (non-mapping, missing layer kind, non-numeric
    theta) falls back to the flat attrs rather than failing the load.
    """
    rope_parameters = getattr(config, "rope_parameters", None)
    if not isinstance(rope_parameters, Mapping):
        return None
    entry = rope_parameters.get(layer_kind)
    if not isinstance(entry, Mapping):
        return None
    theta = entry.get("rope_theta")
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        return None
    return float(theta)


def modernbert_rope_theta(config: Any, *, use_global: bool) -> float:
    """Resolve the RoPE base for global (full) or local (sliding-window) layers.

    A nested transformers>=5 ``rope_parameters`` declaration wins (see
    :func:`nested_rope_theta`); otherwise the flat 4.x attributes
    (``global_rope_theta`` / ``local_rope_theta``), then ``rope_theta``, then
    the 4.x class defaults.
    """
    if use_global:
        theta = nested_rope_theta(config, "full_attention")
        if theta is None:
            theta = getattr(config, "global_rope_theta", getattr(config, "rope_theta", _DEFAULT_GLOBAL_ROPE_THETA))
    else:
        theta = nested_rope_theta(config, "sliding_attention")
        if theta is None:
            theta = getattr(config, "local_rope_theta", getattr(config, "rope_theta", _DEFAULT_LOCAL_ROPE_THETA))
    return theta


def modernbert_rope_cos_sin(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE ``cos``/``sin`` tables ``[len(position_ids), head_dim]`` for one base.

    Computed in float32 (like the Hugging Face rotary embedding) and cast to
    ``dtype`` last, which is the rounding HF applies before rotating the
    attention inputs.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=position_ids.device).float() / head_dim))
    freqs = torch.outer(position_ids.float(), inv_freq)  # [total_tokens, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [total_tokens, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def run_modernbert_flash_layers(
    model: Any,
    hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    total_tokens: int,
    global_cos: torch.Tensor,
    global_sin: torch.Tensor,
    local_cos: torch.Tensor,
    local_sin: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run a ModernBERT layer stack over a packed batch with flash attention.

    ModernBERT is pre-norm with local/global attention alternation: every
    ``global_attn_every_n_layers``-th layer (0-indexed) uses full attention
    with the global RoPE base; the rest use a sliding window of
    ``local_attention`` tokens with the local RoPE base.

    ``compute_dtype`` lets a caller keep the residual stream (``hidden``) and
    the layer norms in a wider type than the projections, the way mixed
    precision autocast runs the reference forward: norm outputs are cast to
    ``compute_dtype`` before each projection and projection outputs are cast
    back before the residual add. When it is ``None`` (or equal to
    ``hidden.dtype``) every cast is a no-op and the loop is the plain
    single-dtype forward.

    Args:
        model: A Hugging Face ``ModernBertModel`` (``.config`` and ``.layers``).
        hidden: Packed embeddings ``[total_tokens, hidden_size]``.
        cu_seqlens: Cumulative sequence lengths ``[num_seqs + 1]`` (int32).
        max_seqlen: Longest packed sequence.
        total_tokens: ``cu_seqlens[-1]``.
        global_cos, global_sin, local_cos, local_sin: Per-token RoPE tables.
        compute_dtype: Dtype of the attention/MLP projections, or ``None``.

    Returns:
        Hidden states ``[total_tokens, hidden_size]`` before ``final_norm``.
    """
    from flash_attn import flash_attn_varlen_func  # ty: ignore[unresolved-import]

    cfg = model.config
    num_heads = cfg.num_attention_heads
    hidden_size = cfg.hidden_size
    head_dim = hidden_size // num_heads
    softmax_scale = 1.0 / (head_dim**0.5)
    compute_dtype = compute_dtype or hidden.dtype
    residual_dtype = hidden.dtype

    global_every_n = getattr(cfg, "global_attn_every_n_layers", 1)
    local_window = getattr(cfg, "local_attention", -1)
    # flash_attn_varlen_func expects window_size as (left, right) tuple
    window = (local_window // 2, local_window // 2) if local_window > 0 else (-1, -1)

    for layer_idx, layer in enumerate(model.layers):
        is_global = (layer_idx % global_every_n == 0) if global_every_n > 1 else True
        cos = global_cos if is_global else local_cos
        sin = global_sin if is_global else local_sin

        # Pre-attention norm (ModernBERT is pre-norm)
        normed_hidden = layer.attn_norm(hidden).to(compute_dtype)

        # Fused QKV projection
        qkv = layer.attn.Wqkv(normed_hidden)
        qkv = qkv.view(total_tokens, 3, num_heads, head_dim)
        query = qkv[:, 0]  # [total_tokens, num_heads, head_dim]
        key = qkv[:, 1]
        value = qkv[:, 2]

        # Apply RoPE to Q and K (using layer-appropriate theta)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # Flash attention — global layers use full attention,
        # local layers use sliding window
        attn_kwargs: dict[str, Any] = {}
        if not is_global and local_window > 0:
            attn_kwargs["window_size"] = window

        attn_out = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
            softmax_scale=softmax_scale,
            **attn_kwargs,
        )
        attn_out = attn_out.reshape(total_tokens, hidden_size)

        # Output projection + residual connection
        hidden = hidden + layer.attn.Wo(attn_out).to(residual_dtype)

        # MLP block with pre-norm
        normed_hidden = layer.mlp_norm(hidden).to(compute_dtype)
        hidden = hidden + layer.mlp(normed_hidden).to(residual_dtype)

    return hidden
