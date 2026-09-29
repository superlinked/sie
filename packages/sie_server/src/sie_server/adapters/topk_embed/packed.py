"""Packed (variable-length) execution for TopK-Embed-V1.

A batch runs as one sequence: the inputs are concatenated with no padding, and
cumulative sequence lengths (``cu_seqlens``) keep them apart in every layer. This
is how TopK's own pipeline runs the backbone (``hf_backbone.py`` in the model
repo), and it avoids the padding the row-per-input path computes on.

On CUDA the kernels are flash-linear-attention's and FlashAttention's:

* the Gated DeltaNet chunked delta rule (decay gate computed in the kernel) and
  causal conv, both variable-length;
* FlashAttention's variable-length attention for the full-attention layers;
* fused RMSNorm (with the residual add), gated RMSNorm and SwiGLU, each one pass
  over the activations where the stock modules run a chain of float32 element-wise
  ops. The Gated DeltaNet norm is the one TopK's stack (transformers 5.9 with
  flash-linear-attention) already fuses.

The reference kernels run the same math in PyTorch, one sequence at a time where a
kernel mixes positions. They back the CPU tests and are never chosen automatically.

``PackedTextModel`` reuses the loaded ``transformers.models.qwen3_5`` modules'
weights and mirrors their forward passes, minus the KV cache and padding masks
that a packed, cache-free encoder does not need. It also runs a fixed-size
layout, right-padded rows (``Padded``), which is what a CUDA graph records
(``graphs.py``): the Gated DeltaNet layers read each row left to right, so the
padding after an input never reaches it, and attention masks the padded keys.
"""

from __future__ import annotations

import inspect
import itertools
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Packing:
    """Where each input of a packed batch starts and ends.

    ``cu_seqlens`` holds the ``N + 1`` int64 offsets on the model's device;
    ``cu_seqlens_cpu`` holds the same offsets on the host, so kernels can plan
    without a device sync; ``cu_seqlens_int32`` is the device copy FlashAttention
    takes. ``max_len`` is the longest input.
    """

    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    cu_seqlens_int32: torch.Tensor
    max_len: int

    @classmethod
    def from_lengths(cls, lengths: Sequence[int], device: str | torch.device) -> Packing:
        offsets = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.long)
        return cls(
            cu_seqlens=offsets.to(device),
            cu_seqlens_cpu=offsets,
            cu_seqlens_int32=offsets.to(device, torch.int32),
            max_len=max(lengths),
        )

    def segments(self) -> list[tuple[int, int]]:
        """``(start, end)`` of each packed input."""
        return list(itertools.pairwise(self.cu_seqlens_cpu.tolist()))


@dataclass(frozen=True)
class Padded:
    """Inputs as rows of one length, each right-padded: ``mask`` is ``[B, L]``, True on real tokens."""

    mask: torch.Tensor


Layout = Packing | Padded


@dataclass(frozen=True)
class PackedKernels:
    """The kernels the packed forward needs.

    Sequence mixers, which see the packing (``packing=None``: every row of the
    batch is one input, as in the ``Padded`` layout):

    * ``delta_rule(q, k, v, *, a, beta, a_log, dt_bias, packing)`` returns
      ``[B, T, HV, V]``. The decay is ``-exp(a_log) * softplus(a + dt_bias)`` in
      float32, as the stock layer computes it.
    * ``causal_conv(x, weight, bias, activation, packing)`` maps ``[B, T, D]`` to
      ``[B, T, D]``.
    * ``attention(q, k, v, packing, scale)`` takes ``[T, H, d]`` queries and
      ``[T, H_kv, d]`` keys and values of a packed batch (bidirectional).

    Per-token ops, computed in float32 and returned in the input dtype:

    * ``rms_norm(x, scale, eps, residual)`` returns ``(rms(s) * scale, s)`` with
      ``s = x + residual`` (``s = x`` without a residual).
    * ``gated_rms_norm(x, gate, weight, eps)`` returns ``rms(x) * weight * silu(gate)``.
    * ``swiglu(gate, up)`` returns ``silu(gate) * up``.
    """

    delta_rule: Callable[..., torch.Tensor]
    causal_conv: Callable[..., torch.Tensor]
    attention: Callable[..., torch.Tensor]
    rms_norm: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    gated_rms_norm: Callable[..., torch.Tensor]
    swiglu: Callable[..., torch.Tensor]
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
    return reference_kernels()


def reference_kernels() -> PackedKernels:
    return PackedKernels(
        delta_rule=reference_delta_rule,
        causal_conv=reference_causal_conv,
        attention=reference_attention,
        rms_norm=reference_rms_norm,
        gated_rms_norm=reference_gated_rms_norm,
        swiglu=reference_swiglu,
        names={"delta_rule": "reference", "causal_conv": "reference", "attention": "reference", "norms": "reference"},
    )


def _fast_kernels() -> PackedKernels | None:
    try:
        from fla.modules.activations import swiglu  # ty: ignore[unresolved-import]
        from fla.modules.convolution import causal_conv1d  # ty: ignore[unresolved-import]
        from fla.modules.fused_norm_gate import rms_norm_gated  # ty: ignore[unresolved-import]
        from fla.modules.layernorm import rms_norm  # ty: ignore[unresolved-import]
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # ty: ignore[unresolved-import]
    except Exception:  # noqa: BLE001 - fla missing or unusable (no Triton): not packing
        logger.info("flash-linear-attention unavailable; TopK-Embed keeps the row-per-input path")
        return None

    def delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        a: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        packing: Packing | None,
    ) -> torch.Tensor:
        out, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g=a,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            **_offsets(packing),
        )
        return out

    def causal_conv(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        packing: Packing | None,
    ) -> torch.Tensor:
        out, _ = causal_conv1d(x=x, weight=weight, bias=bias, activation=activation, **_offsets(packing))
        return out

    def fused_rms_norm(
        x: torch.Tensor, scale: torch.Tensor, eps: float, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return rms_norm(x, scale, None, residual=residual, eps=eps, prenorm=True)

    def fused_gated_rms_norm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return rms_norm_gated(x, gate, weight, None, activation="swish", eps=eps)

    attention, attention_name = reference_attention, "sdpa per sequence"
    try:
        from flash_attn import flash_attn_varlen_func  # ty: ignore[unresolved-import]

        def attention(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, packing: Packing, scale: float
        ) -> torch.Tensor:
            cu = packing.cu_seqlens_int32
            return flash_attn_varlen_func(
                q, k, v, cu, cu, packing.max_len, packing.max_len, softmax_scale=scale, causal=False
            )

        attention_name = "flash_attn varlen"
    except ImportError:
        logger.info("flash_attn unavailable; TopK-Embed packs attention with per-sequence sdpa")
    return PackedKernels(
        delta_rule=delta_rule,
        causal_conv=causal_conv,
        attention=attention,
        rms_norm=fused_rms_norm,
        gated_rms_norm=fused_gated_rms_norm,
        swiglu=swiglu,
        names={"delta_rule": "fla chunk", "causal_conv": "fla", "attention": attention_name, "norms": "fla fused"},
    )


def _offsets(packing: Packing | None) -> dict[str, torch.Tensor]:
    """Fla's variable-length arguments for ``packing``; none for fixed rows."""
    if packing is None:
        return {}
    return {"cu_seqlens": packing.cu_seqlens, "cu_seqlens_cpu": packing.cu_seqlens_cpu}


# ---------------------------------------------------------------------------
# Reference kernels (PyTorch)
# ---------------------------------------------------------------------------


def reference_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    a: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    packing: Packing | None,
) -> torch.Tensor:
    """The PyTorch chunked delta rule from transformers, run per packed sequence (or per row)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5  # ty: ignore[unresolved-import]

    rule = inspect.unwrap(modeling_qwen3_5.torch_chunk_gated_delta_rule)
    g = -a_log.float().exp() * F.softplus(a.float() + dt_bias)
    spans = packing.segments() if packing is not None else [(0, q.shape[1])]
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
        for s, e in spans
    ]
    return torch.cat(outs, dim=1)


def reference_causal_conv(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str | None, packing: Packing | None
) -> torch.Tensor:
    """Depthwise causal conv per packed sequence (or per row): left-pad by ``width - 1``, then activate."""
    width = weight.shape[-1]
    outs = []
    spans = packing.segments() if packing is not None else [(0, x.shape[1])]
    for s, e in spans:
        seq = F.pad(x[:, s:e].transpose(1, 2), (width - 1, 0))
        out = F.conv1d(seq, weight.unsqueeze(1), bias, groups=x.shape[-1])
        outs.append(F.silu(out) if activation in ("silu", "swish") else out)
    return torch.cat(outs, dim=2).transpose(1, 2)


def reference_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, packing: Packing, scale: float
) -> torch.Tensor:
    """Bidirectional attention per packed sequence (``[T, H, d]`` in and out)."""
    outs = []
    for s, e in packing.segments():
        out = F.scaled_dot_product_attention(
            q[s:e].transpose(0, 1).unsqueeze(0),
            k[s:e].transpose(0, 1).unsqueeze(0),
            v[s:e].transpose(0, 1).unsqueeze(0),
            scale=scale,
            enable_gqa=True,
        )
        outs.append(out[0].transpose(0, 1))
    return torch.cat(outs, dim=0)


def padded_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float
) -> torch.Tensor:
    """Bidirectional attention over right-padded rows (``[B, H, L, d]``); padded keys are masked.

    Keys and values repeat to the query heads rather than using sdpa's grouped-query
    mode, which not every sdpa backend takes together with a mask.
    """
    repeats = q.shape[1] // k.shape[1]
    if repeats > 1:
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :], scale=scale)


def reference_rms_norm(
    x: torch.Tensor, scale: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm of ``x + residual``, summed and normalised in float32 as the fused kernel does."""
    total = x.float() if residual is None else x.float() + residual.float()
    normed = total * torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + eps) * scale
    return normed.to(x.dtype), x if residual is None else total.to(residual.dtype)


def reference_gated_rms_norm(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``rms(x) * weight * silu(gate)`` in float32 (flash-linear-attention's ``FusedRMSNormGated``)."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return (normed * F.silu(gate.float())).to(x.dtype)


def reference_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``silu(gate) * up`` in float32."""
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)


# ---------------------------------------------------------------------------
# Packed forward over the stock Qwen3.5 modules
# ---------------------------------------------------------------------------


class PackedTextModel:
    """``Qwen3_5TextModel.forward`` over one packed sequence, on the loaded model's weights.

    Each RMSNorm also performs the residual add that precedes it. The residual stream
    is summed in float32 and stored in the model dtype, which rounds it exactly as
    the stock bf16 add does; only the normalised values skip that rounding.
    """

    def __init__(self, language_model: Any, kernels: PackedKernels) -> None:
        if language_model.config.hidden_act not in ("silu", "swish"):
            msg = f"packed TopK-Embed needs a SiLU MLP, got {language_model.config.hidden_act!r}"
            raise ValueError(msg)
        self.model = language_model
        self.kernels = kernels
        self.layers = list(language_model.layers[: language_model.config.num_hidden_layers])
        # Qwen3.5's RMSNorm multiplies by ``1 + weight`` in float32; the kernels take that factor.
        norms = [language_model.norm]
        for layer in self.layers:
            norms += [layer.input_layernorm, layer.post_attention_layernorm]
            if hasattr(layer, "self_attn"):
                norms += [layer.self_attn.q_norm, layer.self_attn.k_norm]
        self._scales = {norm: 1.0 + norm.weight.detach().float() for norm in norms}

    def __call__(self, embeds: torch.Tensor, position_ids: torch.Tensor, layout: Layout) -> torch.Tensor:
        """Final hidden states ``[B, T, H]`` for ``embeds`` ``[B, T, H]``.

        With ``Packing``, ``B`` is 1 and the inputs are packed along ``T``; with
        ``Padded``, each of the ``B`` rows is one right-padded input. ``position_ids``
        are ``[3, B, T]`` (temporal, height, width), restarting per input.
        """
        position_embeddings = self.model.rotary_emb(embeds, position_ids)
        packing = layout if isinstance(layout, Packing) else None
        hidden, residual = embeds, None
        for layer in self.layers:
            normed, residual = self._norm(layer.input_layernorm, hidden, residual)
            if hasattr(layer, "linear_attn"):
                hidden = self._gated_delta_net(layer.linear_attn, normed, packing)
            else:
                hidden = self._attention(layer.self_attn, normed, position_embeddings, layout)
            normed, residual = self._norm(layer.post_attention_layernorm, hidden, residual)
            hidden = self._mlp(layer.mlp, normed)
        return self._norm(self.model.norm, hidden, residual)[0]

    def _norm(
        self, norm: Any, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kernels.rms_norm(x, self._scales[norm], norm.eps, residual)

    def _gated_delta_net(self, attn: Any, hidden: torch.Tensor, packing: Packing | None) -> torch.Tensor:
        """``Qwen3_5GatedDeltaNet.forward`` without the cache."""
        batch, length, _ = hidden.shape
        mixed = attn.in_proj_qkv(hidden)
        z = attn.in_proj_z(hidden)
        b = attn.in_proj_b(hidden)
        a = attn.in_proj_a(hidden)
        mixed = self.kernels.causal_conv(
            mixed, attn.conv1d.weight.squeeze(1), attn.conv1d.bias, attn.activation, packing
        )
        query, key, value = torch.split(mixed, [attn.key_dim, attn.key_dim, attn.value_dim], dim=-1)
        query = query.reshape(batch, length, -1, attn.head_k_dim)
        key = key.reshape(batch, length, -1, attn.head_k_dim)
        value = value.reshape(batch, length, -1, attn.head_v_dim)
        repeats = attn.num_v_heads // attn.num_k_heads
        if repeats > 1:
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        out = self.kernels.delta_rule(
            query, key, value, a=a, beta=b.sigmoid(), a_log=attn.A_log, dt_bias=attn.dt_bias, packing=packing
        )
        out = self.kernels.gated_rms_norm(
            out.reshape(-1, attn.head_v_dim),
            z.reshape(-1, attn.head_v_dim),
            attn.norm.weight,
            attn.norm.variance_epsilon,
        )
        return attn.out_proj(out.reshape(batch, length, -1))

    def _attention(
        self,
        attn: Any,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        layout: Layout,
    ) -> torch.Tensor:
        """``Qwen3_5Attention.forward`` (gated output, partial rotary), bidirectional per input."""
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb  # ty: ignore[unresolved-import]

        input_shape = hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query, gate = torch.chunk(attn.q_proj(hidden).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query = self._norm(attn.q_norm, query.reshape(hidden_shape))[0].transpose(1, 2)
        key = self._norm(attn.k_norm, attn.k_proj(hidden).view(hidden_shape))[0].transpose(1, 2)
        value = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        if isinstance(layout, Packing):
            out = self.kernels.attention(
                query[0].transpose(0, 1).contiguous(),
                key[0].transpose(0, 1).contiguous(),
                value[0].transpose(0, 1).contiguous(),
                layout,
                attn.scaling,
            )
        else:
            out = padded_attention(query, key, value, layout.mask, attn.scaling).transpose(1, 2)
        out = out.reshape(*input_shape, -1) * torch.sigmoid(gate)
        return attn.o_proj(out)

    def _mlp(self, mlp: Any, hidden: torch.Tensor) -> torch.Tensor:
        """``Qwen3_5MLP.forward``: ``down(silu(gate(x)) * up(x))``."""
        return mlp.down_proj(self.kernels.swiglu(mlp.gate_proj(hidden), mlp.up_proj(hidden)))
