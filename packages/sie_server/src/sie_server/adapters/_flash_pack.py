"""Shared variable-length packing helpers for flash-attention adapters.

Flash-attention encoders pack a batch of sequences into one flat token stream
and drive attention with cumulative sequence lengths (``cu_seqlens``). Each
packed sequence needs position ids restarting at the model's base offset.

Historically every ``*_flash`` adapter reimplemented this with a per-sequence
``torch.arange`` loop (with a ``.item()`` CUDA sync per sequence). This module
is the single deep implementation those adapters delegate to. See issue #1538.
"""

from __future__ import annotations

import torch


def build_position_ids(
    cu_seqlens: torch.Tensor,
    *,
    offset: int = 0,
    total_tokens: int | None = None,
) -> torch.Tensor:
    """Position ids for a varlen-packed batch, restarting per sequence.

    Given ``cu_seqlens`` (cumulative token counts, shape ``[num_seqs + 1]``),
    return a flat ``int64`` tensor where packed sequence ``i`` carries
    ``[offset, offset + 1, ..., offset + len_i - 1]``.

    Fully vectorized — no per-sequence ``.item()`` CUDA sync — and bit-identical
    to the per-sequence ``torch.arange`` loops it replaces. The result is placed
    on ``cu_seqlens.device``.

    Args:
        cu_seqlens: Cumulative sequence lengths, shape ``[num_seqs + 1]``.
        offset: Base position of each sequence. ``0`` for most encoders;
            ``padding_idx + 1`` for XLM-RoBERTa-style position ids.
        total_tokens: Optional host-known final cumulative token count. This
            must equal ``cu_seqlens[-1]``; supplying it avoids a device-to-host
            synchronization when ``cu_seqlens`` is on CUDA.

    Returns:
        A 1-D ``int64`` tensor of length ``cu_seqlens[-1]`` on ``cu_seqlens.device``.
    """
    if total_tokens is None:
        total_tokens = int(cu_seqlens[-1].item())
    positions = torch.arange(total_tokens, device=cu_seqlens.device)
    seq_starts = torch.repeat_interleave(
        cu_seqlens[:-1],
        cu_seqlens[1:] - cu_seqlens[:-1],
        output_size=total_tokens,
    )
    positions = positions - seq_starts
    return positions + offset if offset else positions


def mean_pool_packed(hidden: torch.Tensor, cu_seqlens: torch.Tensor, num_seqs: int) -> torch.Tensor:
    """Mean-pool each sequence of a varlen-packed hidden-state tensor.

    ``hidden`` is the flat ``[total_tokens, dim]`` output of a flash encoder;
    ``cu_seqlens`` (shape ``[num_seqs + 1]``) marks the packed sequence
    boundaries. Returns ``[num_seqs, dim]`` — each row the mean of one
    sequence's token vectors.

    This is the exact per-sequence loop the ``*_flash`` embedding adapters
    (nomic / rope / qwen2 / xlm_roberta) each carried inline, relocated here
    unchanged so a pooling fix lands once. Behaviour is bit-identical to the
    inline loops (same slice + ``mean(dim=0)`` + ``stack``); it is intentionally
    NOT vectorized so no adapter's numerics shift.
    """
    mean_embeddings = []
    for i in range(num_seqs):
        start = cu_seqlens[i].item()
        end = cu_seqlens[i + 1].item()
        mean_embeddings.append(hidden[start:end].mean(dim=0))
    return torch.stack(mean_embeddings)
