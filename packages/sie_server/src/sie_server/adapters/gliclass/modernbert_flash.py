"""Flash-attention varlen encoder for GLiClass models with a ModernBERT backbone.

GLiClass models built on ModernBERT or mmBERT (the edge models,
``gliclass-multilang-edge``, the Opir edge models, ``gliclass-modern-base-v3.0``
and ``gliclass-modern-large-v3.0``) run their encoder through the Hugging Face
``ModernBertModel`` forward. With flash-attn on CUDA, that forward unpads and
repads the batch around every call and runs its MLPs through ``torch.compile``
guards. At the batch sizes classification traffic has, the host launching
kernels bounds a forward, and the GPU mostly waits.

This runner feeds the same weights through SIE's shared ModernBERT layer loop
(``sie_server.adapters._modernbert_flash``, also used by the dense, late
interaction, cross-encoder and Laya adapters): one packed, unpadded token
stream through ``flash_attn_varlen_func``, with RoPE tables built once at load.
The encoder output is repadded, with zeros at padded positions as the Hugging
Face flash-attention forward leaves them, and the gliclass scoring head runs
unchanged on it: label-token features, pooling, projections and scorer are the
library's own code with the forward's own inputs and label slots.

What the encoder computes is the Hugging Face forward's math: token embeddings
(plus segment embeddings on instruct models) and the embedding norm, the
pre-norm layers with global attention every ``global_attn_every_n_layers``
layers and a sliding window of ``local_attention`` tokens elsewhere, each with
its RoPE base, then the final norm. Scores differ from the Hugging Face path
only by floating-point rounding.

Past a number of packed tokens that depends on the encoder's width, the GPU,
not kernel launches, bounds a forward (``token_bound``). There the Hugging
Face path is faster: it rotates queries and keys with one fused kernel per
layer and compiles its MLPs, where the shared layer loop runs each step as its
own kernel. Such forwards run the gliclass forward. On an L4, the flash path
runs a ``gliclass-modern-large-v3.0`` forward (1,024 wide) 1.13x as fast at
1,024 packed tokens and 0.90x at 1,289, a ``gliclass-modern-base-v3.0``
forward (768 wide) 1.27x at 2,048 and 0.96x at 2,560, and the 384-wide edge
models 1.3x at about 3,650 tokens and 0.81x at 5,700 on 22 layers.

Operators enable it per model, at load, with
``adapter_options.loadtime.modernbert_flash: true``. It applies to uni-encoder
models whose encoder is a ModernBERT (mmBERT included) with float16 or bfloat16
weights, on a CUDA device with flash-attn; anything else runs the gliclass
forward as before, and the load logs why. So does a model that scores
intermediate encoder layers (``squeeze_layers`` or ``encoder_layer_id``),
which no published checkpoint does. A forward whose inputs the runner does not
recognise (other input tensors, rows that are not right-padded) runs the
gliclass forward too.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from sie_server.adapters._flash_pack import build_position_ids
from sie_server.adapters._modernbert_flash import (
    modernbert_rope_cos_sin,
    modernbert_rope_theta,
    run_modernbert_flash_layers,
)
from sie_server.adapters.gliclass.cuda_graphs import segment_ids

logger = logging.getLogger(__name__)

# ``encoder_config.model_type`` of the encoders this runner drives. mmBERT
# checkpoints are ModernBERT models with another vocabulary and RoPE bases.
_ENCODERS = frozenset({"modernbert"})
_DTYPES = frozenset({torch.float16, torch.bfloat16})
_INPUTS = frozenset({"input_ids", "attention_mask"})

# Why a forward ran the gliclass forward instead.
EagerReason = Literal[
    "unsupported_inputs",  # inputs other than token ids and mask, or not a 2-D batch
    "not_right_padded",  # a row's mask is not a run of ones followed by zeros
    "too_long",  # a row longer than the RoPE tables, which cover the model window
    "too_large",  # more packed tokens than the token bound
]


def token_bound(hidden_size: int) -> int:
    """Packed tokens a flash forward of an encoder this wide may hold (see the module docstring)."""
    if hidden_size <= 384:
        return 4096
    if hidden_size <= 768:
        return 2048
    return 1024


def unsupported_reason(model: Any, device: str | torch.device) -> str | None:
    """Why ``model`` cannot run its encoder through this runner on ``device``; None when it can."""
    if not str(device).startswith("cuda"):
        return "flash attention needs a CUDA device"
    # Imported here: sie_server.core.inference imports the model loader, which imports adapters.
    from sie_server.core.inference import is_flash_attention_available

    if not is_flash_attention_available(str(device)):
        return "flash-attn is not installed, or the GPU predates Ampere"
    config = getattr(model, "config", None)
    if getattr(config, "architecture_type", None) != "uni-encoder":
        return "only uni-encoder GLiClass models are supported"
    encoder_type = getattr(getattr(config, "encoder_config", None), "model_type", None)
    if encoder_type not in _ENCODERS:
        return f"the {encoder_type} encoder is not a ModernBERT"
    if getattr(config, "squeeze_layers", False) or getattr(config, "encoder_layer_id", -1) not in (-1, None):
        return "the model scores intermediate encoder layers"
    inner = getattr(model, "model", None)
    encoder = getattr(inner, "encoder_model", None)
    embeddings = getattr(encoder, "embeddings", None)
    if not all(hasattr(encoder, name) for name in ("layers", "final_norm")) or not hasattr(embeddings, "norm"):
        return "the encoder is not a Hugging Face ModernBertModel"
    if not callable(getattr(inner, "process_encoder_output", None)):
        return "the model has no scoring head to run apart from its encoder"
    dtype = embeddings.tok_embeddings.weight.dtype
    if dtype not in _DTYPES:
        return f"flash attention needs float16 or bfloat16 weights, not {dtype}"
    return None


@dataclass
class FlashStats:
    """Forwards the runner was offered since the model loaded: run here, or eagerly by reason."""

    flash: int = 0
    eager: dict[str, int] = field(default_factory=dict)


class ModernBertFlashEncoder:
    """Runs a GLiClass uni-encoder forward with its ModernBERT encoder on packed, unpadded rows.

    ``max_length`` is the model window: the longest row the RoPE tables cover.
    The runner holds no per-forward state, so threads may share it.
    """

    def __init__(self, model: Any, *, max_length: int) -> None:
        self._inner = model.model
        self._config = model.config
        self._encoder = self._inner.encoder_model
        self._segments = bool(getattr(self._config, "use_segment_embeddings", False))
        self._max_length = max_length
        encoder_config = self._encoder.config
        weight = self._encoder.embeddings.tok_embeddings.weight
        positions = torch.arange(max_length, device=weight.device)
        head_dim = encoder_config.hidden_size // encoder_config.num_attention_heads
        self._max_tokens = token_bound(encoder_config.hidden_size)
        # Per-position cos/sin tables for the global and the sliding-window
        # layers, each with its own RoPE base, in the weights' dtype.
        self._rope = {
            kind: modernbert_rope_cos_sin(
                positions,
                head_dim=head_dim,
                theta=modernbert_rope_theta(encoder_config, use_global=kind == "global"),
                dtype=weight.dtype,
            )
            for kind in ("global", "local")
        }
        self.stats = FlashStats()

    def run(self, inputs: Mapping[str, torch.Tensor], max_num_classes: int | None) -> torch.Tensor | None:
        """Logits for a padded batch of token ids, or None to run the gliclass forward.

        ``max_num_classes`` is the forward's label-slot count, passed to the
        scoring head as the gliclass forward passes it.
        """
        logits, reason = self._run(inputs, max_num_classes)
        if reason is not None:
            self.stats.eager[reason] = self.stats.eager.get(reason, 0) + 1
        else:
            self.stats.flash += 1
        return logits

    def _run(
        self, inputs: Mapping[str, torch.Tensor], max_num_classes: int | None
    ) -> tuple[torch.Tensor | None, EagerReason | None]:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        if input_ids is None or attention_mask is None or set(inputs) - _INPUTS or input_ids.dim() != 2:
            return None, "unsupported_inputs"
        batch, length = input_ids.shape
        # The row lengths, read once from the device: the packed layout's
        # shape. Nothing is queued ahead of this forward but its input copies.
        mask = attention_mask.cpu()
        lengths = mask.sum(dim=1)
        if not torch.equal(mask.bool(), torch.arange(length)[None, :] < lengths[:, None]):
            return None, "not_right_padded"
        max_seqlen = int(lengths.max()) if batch else 0
        if max_seqlen > self._max_length:
            return None, "too_long"
        total_tokens = int(lengths.sum())
        if not total_tokens:
            return None, "unsupported_inputs"
        if total_tokens > self._max_tokens:
            return None, "too_large"
        device = input_ids.device
        cu_seqlens = torch.zeros(batch + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
        cu_seqlens = cu_seqlens.to(device)
        positions = build_position_ids(cu_seqlens, total_tokens=total_tokens)
        rows = torch.repeat_interleave(
            torch.arange(batch, device=device), cu_seqlens[1:] - cu_seqlens[:-1], output_size=total_tokens
        )

        encoder = self._encoder
        hidden = encoder.embeddings.tok_embeddings(input_ids[rows, positions])
        if self._segments:
            # As the uni-encoder forward adds them before the encoder's own
            # embedding norm. Segment ids depend on a whole padded row.
            hidden = hidden + self._inner.segment_embeddings(segment_ids(input_ids, self._config)[rows, positions])
        hidden = encoder.embeddings.norm(hidden)
        global_cos, global_sin = self._rope["global"]
        local_cos, local_sin = self._rope["local"]
        hidden = run_modernbert_flash_layers(
            encoder,
            hidden,
            cu_seqlens,
            max_seqlen,
            total_tokens,
            global_cos[positions],
            global_sin[positions],
            local_cos[positions],
            local_sin[positions],
        )
        hidden = encoder.final_norm(hidden)
        padded = hidden.new_zeros(batch, length, hidden.shape[-1])
        padded[rows, positions] = hidden
        logits = self._inner.process_encoder_output(input_ids, attention_mask, padded, None, max_num_classes)[0]
        return logits, None
