"""Laya network: a ModernBERT-family encoder followed by a typed decision head.

The checkpoint's ``model.safetensors`` holds ``encoder.*`` (a stock Hugging Face
``ModernBertModel``), ``head.layers.{0,1}`` (two pre-norm
``nn.TransformerEncoderLayer``), ``type_emb`` (one embedding per question type),
and ``scorer`` (LayerNorm -> Linear -> GELU -> Linear -> 1 logit). It also holds
an ``act_head`` (an escalation classifier) and a ``temperature`` buffer; neither
is used here — calibration comes from ``rl_agent_config.json``.

Two backbones run the same math:

* ``forward_flash``: CUDA + flash-attn. Rows are packed without padding and the
  encoder runs ``flash_attn_varlen_func``; the projections run in the compute
  dtype while the residual stream, layer norms, and embeddings stay in float32,
  matching the reference forward under bf16 autocast. The encoder output is
  re-padded for the (small) decision head, which runs its stock modules under
  autocast exactly like the reference.
* ``forward_padded``: everything else (CPU, MPS, CUDA without flash-attn). The
  Hugging Face ``ModernBertModel`` with SDPA attention plus the same head — the
  reference math, in float32 off-CUDA.

The flash path is split in two so that the encoder can be replayed as a CUDA
graph (``sie_server.adapters._modernbert_flash_graphs``): ``encode_packed`` is
what a graph records, and ``head_logits`` runs eagerly on its output, as it
does after an eager forward.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
from torch import nn

from sie_server.adapters._flash_pack import build_position_ids
from sie_server.adapters._modernbert_flash import (
    modernbert_rope_cos_sin,
    modernbert_rope_theta,
    run_modernbert_flash_layers,
)
from sie_server.adapters._modernbert_flash_graphs import EncodeFn

ENCODER_PREFIX = "encoder."
HEAD_PREFIXES = ("head.", "type_emb.", "scorer.")
# Present in every checkpoint but not used for typed answers.
UNUSED_PREFIXES = ("act_head.", "temperature")


class LayaDecisionHead(nn.Module):
    """Typed decision head: question-type embedding, full-attention layers, marker scorer."""

    def __init__(self, hidden_size: int, num_layers: int = 2) -> None:
        super().__init__()
        nhead = max(1, hidden_size // 64)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                hidden_size, nhead, 4 * hidden_size, dropout=0.1, batch_first=True, norm_first=True
            )
            for _ in range(num_layers)
        )
        self.type_emb = nn.Embedding(3, hidden_size)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor,
        qtype: torch.Tensor,
        marker_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Score every option marker.

        Args:
            hidden: Encoder output ``[rows, seq, hidden]``.
            padding_mask: ``True`` at padding positions ``[rows, seq]``.
            qtype: Question type index per row ``[rows]``.
            marker_pos: Marker positions ``[rows, max_options]`` (padded with 0).

        Returns:
            Float32 logits ``[rows, max_options]``; entries past a row's option
            count are meaningless.
        """
        h = hidden + self.type_emb(qtype)[:, None, :]
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=padding_mask)
        idx = marker_pos[:, :, None].expand(-1, -1, h.size(-1))
        return self.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()


def split_checkpoint(weights: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Split a Laya ``model.safetensors`` state dict into encoder and head state dicts.

    Raises:
        ValueError: On a key that belongs to no known component (not a Laya checkpoint).
    """
    encoder: dict[str, torch.Tensor] = {}
    head: dict[str, torch.Tensor] = {}
    for key, value in weights.items():
        if key.startswith(ENCODER_PREFIX):
            encoder[key[len(ENCODER_PREFIX) :]] = value
        elif key.startswith(HEAD_PREFIXES):
            head[key.removeprefix("head.")] = value
        elif not key.startswith(UNUSED_PREFIXES):
            raise ValueError(f"Unexpected tensor {key!r} in Laya checkpoint")
    if not encoder or not head:
        raise ValueError("Laya checkpoint is missing encoder or decision-head weights")
    return encoder, head


def apply_rope_parameters(encoder_config: Any) -> None:
    """Carry a transformers>=5 ``rope_parameters`` declaration onto the 4.x config attrs.

    transformers 4.x keeps ``rope_parameters`` as an opaque attribute and fills
    ``global_rope_theta``/``local_rope_theta`` from class defaults (160000 and
    10000). mmBERT trains both layer kinds at 160000, so without this its
    sliding-window layers would silently run the wrong RoPE base.
    """
    encoder_config.global_rope_theta = float(modernbert_rope_theta(encoder_config, use_global=True))
    encoder_config.local_rope_theta = float(modernbert_rope_theta(encoder_config, use_global=False))


class LayaModel(nn.Module):
    """Encoder + decision head with the flash and padded forward paths."""

    def __init__(self, encoder: nn.Module, head: LayaDecisionHead) -> None:
        super().__init__()
        # A Hugging Face ModernBertModel (typed loosely: its submodules are read by name).
        self.encoder: Any = encoder
        self.head = head
        self._use_flash = False
        self._compute_dtype: torch.dtype = torch.float32
        self._fused_rope = False
        self._rope: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def use_flash(self) -> bool:
        return self._use_flash

    def prepare(
        self,
        device: str,
        *,
        use_flash: bool,
        compute_dtype: torch.dtype,
        max_positions: int,
        embedding_dtype: torch.dtype | None = None,
        fused_rope: bool = False,
    ) -> None:
        """Place the model for the chosen path. Call once after loading weights (float32).

        Args:
            device: Target device.
            use_flash: Run the packed flash-attention backbone (CUDA only).
            compute_dtype: Projection dtype on the flash path.
            max_positions: Longest row the RoPE tables must cover.
            embedding_dtype: dtype of the token-embedding table in the checkpoint.
                On the flash path a float16 or bfloat16 table is kept at that
                precision, which is lossless because the values came from it;
                any other table stays float32.
            fused_rope: On the flash path, rotate queries and keys with one
                fused kernel per layer where it runs (see
                ``run_modernbert_flash_layers``). Outputs differ by rounding.
        """
        self._use_flash = use_flash
        self._compute_dtype = compute_dtype
        self._fused_rope = fused_rope and use_flash
        if use_flash:
            # Projections in the compute dtype; norms stay float32 for the float32
            # residual stream. The embedding lookup is upcast to float32.
            for module in self.encoder.modules():
                if isinstance(module, nn.Linear):
                    module.to(compute_dtype)
            if embedding_dtype in (torch.float16, torch.bfloat16):
                self.encoder.embeddings.tok_embeddings.to(embedding_dtype)
        self.to(device)
        self.eval()
        if use_flash:
            cfg = self.encoder.config
            head_dim = cfg.hidden_size // cfg.num_attention_heads
            positions = torch.arange(max_positions, device=device)
            self._rope = {
                kind: modernbert_rope_cos_sin(
                    positions,
                    head_dim=head_dim,
                    theta=modernbert_rope_theta(cfg, use_global=kind == "global"),
                    dtype=compute_dtype,
                )
                for kind in ("global", "local")
            }

    def autocast(self, device: str) -> AbstractContextManager[Any]:
        """Autocast context for the forward pass (CUDA reduced precision only)."""
        if device.startswith("cuda") and self._compute_dtype != torch.float32:
            return torch.autocast(device_type="cuda", dtype=self._compute_dtype)
        return nullcontext()

    def encode_packed(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
    ) -> torch.Tensor:
        """The packed varlen encoder: token embeddings and their norm, the layers, the final norm.

        This is what a CUDA graph records (an ``EncodeFn``), and what the eager
        flash forward runs. ``input_ids`` and ``positions`` are integer device
        tensors of ``total_tokens`` entries; ``cu_seqlens`` is int32.

        Returns:
            Float32 hidden states ``[total_tokens, hidden]``.
        """
        encoder = self.encoder
        embeddings = encoder.embeddings
        hidden = embeddings.norm(embeddings.tok_embeddings(input_ids).float())
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
            compute_dtype=self._compute_dtype,
            fused_rope=self._fused_rope,
        )
        return encoder.final_norm(hidden)

    def graph_encoder(self) -> tuple[EncodeFn, list[torch.Tensor]]:
        """The encoder a CUDA graph records, and the RoPE tables it reads (for the runner's memory budget)."""
        return self.encode_packed, [table for pair in self._rope.values() for table in pair]

    def head_logits(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        lengths: list[int],
        marker_pos: torch.Tensor,
        qtype: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Re-pad packed encoder output and run the decision head on it.

        The head's layers attend over every token of a row, so the packed
        ``[total_tokens, hidden]`` states are laid out as ``[rows, max_seqlen,
        hidden]`` with the padding masked.

        Args:
            hidden: Packed encoder output of the real rows, row after row.
            cu_seqlens: Their cumulative lengths ``[rows + 1]`` (int32).
            lengths: Tokens per row (host copy of the same lengths).
            marker_pos: Marker positions ``[rows, max_options]``.
            qtype: Question type index per row.
            device: The model's device.

        Returns:
            Float32 logits ``[rows, max_options]``.
        """
        num_rows, max_seqlen, total_tokens = len(lengths), max(lengths), hidden.shape[0]
        lengths_t = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        positions = build_position_ids(cu_seqlens, total_tokens=total_tokens)
        row_index = torch.repeat_interleave(torch.arange(num_rows, device=device), lengths_t, output_size=total_tokens)
        padded = hidden.new_zeros(num_rows, max_seqlen, hidden.shape[-1])
        padded[row_index, positions] = hidden
        padding_mask = torch.arange(max_seqlen, device=device)[None, :] >= lengths_t[:, None]
        with self.autocast(device):
            return self.head(padded, padding_mask, qtype, marker_pos)

    def forward_flash(
        self,
        rows: list[list[int]],
        marker_pos: torch.Tensor,
        qtype: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Packed varlen encoder + re-padded decision head. Returns float32 logits ``[rows, max_options]``."""
        lengths = [len(r) for r in rows]
        total_tokens = sum(lengths)
        num_rows = len(rows)

        flat = [token for row in rows for token in row]
        input_ids = torch.tensor(flat, dtype=torch.long, device=device)
        lengths_t = torch.tensor(lengths, dtype=torch.int32, device=device)
        cu_seqlens = torch.zeros(num_rows + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = torch.cumsum(lengths_t, dim=0)
        positions = build_position_ids(cu_seqlens, total_tokens=total_tokens)
        hidden = self.encode_packed(input_ids, positions, cu_seqlens, max(lengths), total_tokens)
        return self.head_logits(hidden, cu_seqlens, lengths, marker_pos, qtype, device)

    def forward_padded(
        self,
        rows: list[list[int]],
        marker_pos: torch.Tensor,
        qtype: torch.Tensor,
        device: str,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Hugging Face SDPA encoder over a padded batch + decision head (reference math)."""
        max_seqlen = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), max_seqlen), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_seqlen), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[i, : len(row)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        with self.autocast(device):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            return self.head(hidden, ~attention_mask.bool(), qtype, marker_pos)
