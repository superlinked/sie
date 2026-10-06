"""Batched greedy decoding for token-and-duration transducers (TDT).

The loop follows NeMo's greedy TDT decoder. At each step the joint network
scores the current encoder frame against the prediction network's state and
predicts a token and a duration (how many encoder frames to advance):

* The token is the argmax over the vocabulary slice of the joint output (blank
  included); the duration is ``durations[argmax]`` over the remaining logits.
* A blank never stays on its frame: a blank with duration 0 advances one frame.
* Only rows that emit a token feed it to the prediction network; the other
  rows keep their state.
* A row that emits ``max_symbols_per_frame`` tokens in a row without advancing
  is forced one frame forward. This bounds the decode at
  ``max_symbols_per_frame`` steps per frame and stops a row from repeating a
  token forever.
* Each row stops at its own encoder length and never reads padded frames.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

MAX_SYMBOLS_PER_FRAME = 10


@dataclass(slots=True)
class TdtHypothesis:
    """One row's greedy decode.

    ``token_ids`` are the emitted (non-blank) tokens. ``step_ids`` and
    ``step_durations`` record every decode step, blanks included, with the
    number of encoder frames it advanced; their running sum is each step's
    frame index, which the processor's timestamp decoding consumes.
    """

    token_ids: list[int] = field(default_factory=list)
    step_ids: list[int] = field(default_factory=list)
    step_durations: list[int] = field(default_factory=list)


@torch.inference_mode()
def greedy_tdt_decode(
    decoder: Any,
    joint: Any,
    encoder_output: torch.Tensor,
    lengths: torch.Tensor,
    *,
    blank_id: int,
    vocab_size: int,
    durations: Sequence[int],
    max_symbols_per_frame: int = MAX_SYMBOLS_PER_FRAME,
) -> list[TdtHypothesis]:
    """Decode a padded batch of encoder outputs.

    Args:
        decoder: Prediction network with ``embedding``, ``lstm`` (batch-first)
            and ``decoder_projector``.
        joint: Joint network with ``activation`` and ``head``; ``head`` returns
            ``vocab_size`` token logits followed by one logit per duration.
        encoder_output: Projected encoder states, shape ``(batch, frames, hidden)``.
        lengths: Valid encoder frames per row, shape ``(batch,)``.
        blank_id: Blank token id (inside the vocabulary slice).
        vocab_size: Width of the token slice of the joint output.
        durations: Frame advance for each duration logit.
        max_symbols_per_frame: Consecutive emissions allowed on one frame
            before the row is forced to advance.
    """
    if max_symbols_per_frame < 1:
        msg = "max_symbols_per_frame must be positive"
        raise ValueError(msg)
    batch, frames, _ = encoder_output.shape
    hypotheses = [TdtHypothesis() for _ in range(batch)]
    if batch == 0 or frames == 0:
        return hypotheses

    device = encoder_output.device
    lstm = decoder.lstm
    hidden = torch.zeros(
        (lstm.num_layers, batch, lstm.hidden_size),
        device=device,
        dtype=encoder_output.dtype,
    )
    cell = torch.zeros_like(hidden)
    # The prediction network starts from a zero state fed the blank token.
    start = torch.full((batch, 1), blank_id, device=device, dtype=torch.long)
    output, (hidden, cell) = lstm(decoder.embedding(start), (hidden, cell))
    prediction = decoder.decoder_projector(output)[:, 0]

    duration_values = torch.as_tensor(list(durations), device=device, dtype=torch.long)
    rows = torch.arange(batch, device=device)
    lengths = lengths.to(device=device, dtype=torch.long)
    frame = torch.zeros(batch, device=device, dtype=torch.long)
    symbols = torch.zeros_like(frame)
    one = torch.ones_like(frame)
    zero = torch.zeros_like(frame)
    active = frame < lengths

    step_tokens: list[torch.Tensor] = []
    step_durations: list[torch.Tensor] = []
    step_active: list[torch.Tensor] = []
    step_emitted: list[torch.Tensor] = []
    while bool(active.any()):
        encoded = encoder_output[rows, frame.clamp(max=frames - 1)]
        logits = joint.head(joint.activation(encoded + prediction)).float()
        token = logits[:, :vocab_size].argmax(dim=-1)
        duration = duration_values[logits[:, vocab_size:].argmax(dim=-1)]
        blank = token == blank_id
        duration = torch.where(blank & (duration == 0), one, duration)
        emitted = active & ~blank
        symbols = torch.where(emitted & (duration == 0), symbols + 1, zero)
        forced = symbols >= max_symbols_per_frame
        duration = torch.where(forced, one, duration)
        symbols = torch.where(forced, zero, symbols)

        step_tokens.append(token)
        step_durations.append(duration)
        step_active.append(active)
        step_emitted.append(emitted)

        if bool(emitted.any()):
            output, (next_hidden, next_cell) = lstm(decoder.embedding(token[:, None]), (hidden, cell))
            next_prediction = decoder.decoder_projector(output)[:, 0]
            state_mask = emitted.view(1, batch, 1)
            hidden = torch.where(state_mask, next_hidden, hidden)
            cell = torch.where(state_mask, next_cell, cell)
            prediction = torch.where(emitted.view(batch, 1), next_prediction, prediction)

        frame = frame + torch.where(active, duration, zero)
        active = frame < lengths

    if not step_tokens:
        return hypotheses
    tokens = torch.stack(step_tokens, dim=1).tolist()
    advances = torch.stack(step_durations, dim=1).tolist()
    was_active = torch.stack(step_active, dim=1).tolist()
    was_emitted = torch.stack(step_emitted, dim=1).tolist()
    for row, hypothesis in enumerate(hypotheses):
        for token_id, advance, row_active, row_emitted in zip(
            tokens[row], advances[row], was_active[row], was_emitted[row], strict=True
        ):
            if not row_active:
                continue
            hypothesis.step_ids.append(token_id)
            hypothesis.step_durations.append(advance)
            if row_emitted:
                hypothesis.token_ids.append(token_id)
    return hypotheses
