"""Laya network code on a tiny random encoder, against the reference forward (no downloads).

The reference ``DecisionModel`` below is laya 0.3.11's (``laya/common.py``,
Apache-2.0) with its training-only branches removed. Its state dict has the same
keys as a released ``model.safetensors``; loading it through
``split_checkpoint`` and ``LayaModel`` and comparing logits checks the decision
head's wiring (head count, activations, pre-norm, marker gather, question-type
embedding) and the checkpoint split on real modules, on CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from sie_server.adapters.laya.model import LayaDecisionHead, LayaModel, split_checkpoint
from torch import nn
from transformers import AutoModel, ModernBertConfig

HIDDEN = 128  # two head attention heads (hidden // 64), so the head split is exercised
PAD_ID = 0


class ReferenceDecisionModel(nn.Module):
    """laya 0.3.11 ``DecisionModel``, inference path only."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, max(1, d // 64), 4 * d, dropout, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = self.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


def _encoder_config() -> ModernBertConfig:
    config = ModernBertConfig(
        vocab_size=128,
        hidden_size=HIDDEN,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        global_attn_every_n_layers=2,  # layer 0 global, layer 1 sliding-window
        local_attention=8,
        pad_token_id=PAD_ID,
        bos_token_id=1,
        cls_token_id=1,
        eos_token_id=2,
        sep_token_id=2,
    )
    config.reference_compile = False
    return config


# Three rows of different lengths (padding), one per question type, with 3/2/4 option markers.
ROWS = [
    [1, *range(10, 40), 2],
    [1, *range(40, 52), 2],
    [1, *range(60, 80), 2, 3, 4, 2],
]
MARKERS = [[3, 9, 20], [2, 7], [1, 5, 12, 22]]
QTYPES = [0, 1, 2]


def _reference_logits(reference: ReferenceDecisionModel) -> torch.Tensor:
    n, width, k = len(ROWS), max(len(r) for r in ROWS), max(len(m) for m in MARKERS)
    input_ids = torch.full((n, width), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((n, width), dtype=torch.long)
    marker_pos = torch.zeros((n, k), dtype=torch.long)
    marker_mask = torch.zeros((n, k), dtype=torch.bool)
    for i, (row, markers) in enumerate(zip(ROWS, MARKERS, strict=True)):
        input_ids[i, : len(row)] = torch.tensor(row)
        attention_mask[i, : len(row)] = 1
        marker_pos[i, : len(markers)] = torch.tensor(markers)
        marker_mask[i, : len(markers)] = True
    with torch.inference_mode():
        return reference(input_ids, attention_mask, marker_pos, marker_mask, torch.tensor(QTYPES))


def _marker_tensor() -> torch.Tensor:
    marker_pos = torch.zeros((len(MARKERS), max(len(m) for m in MARKERS)), dtype=torch.long)
    for i, markers in enumerate(MARKERS):
        marker_pos[i, : len(markers)] = torch.tensor(markers)
    return marker_pos


def _load_like_adapter(checkpoint: Path, config: ModernBertConfig) -> tuple[LayaModel, torch.dtype]:
    """Build the served model from a ``model.safetensors`` the way ``LayaAdapter.load`` does."""
    encoder_weights, head_weights = split_checkpoint(load_file(str(checkpoint)))
    encoder = AutoModel.from_config(config, attn_implementation="sdpa")
    head = LayaDecisionHead(config.hidden_size, 2)
    encoder.load_state_dict(encoder_weights, strict=True)
    head.load_state_dict(head_weights, strict=True)
    return LayaModel(encoder, head), encoder_weights["embeddings.tok_embeddings.weight"].dtype


@pytest.fixture
def reference_checkpoint(tmp_path: Path) -> tuple[ReferenceDecisionModel, Path]:
    torch.manual_seed(0)
    reference = ReferenceDecisionModel(AutoModel.from_config(_encoder_config(), attn_implementation="sdpa")).eval()
    checkpoint = tmp_path / "model.safetensors"
    save_file(reference.state_dict(), str(checkpoint))
    return reference, checkpoint


def test_padded_forward_matches_reference(reference_checkpoint: tuple[ReferenceDecisionModel, Path]) -> None:
    reference, checkpoint = reference_checkpoint
    expected = _reference_logits(reference)
    model, embedding_dtype = _load_like_adapter(checkpoint, _encoder_config())
    model.prepare(
        "cpu", use_flash=False, compute_dtype=torch.float32, max_positions=64, embedding_dtype=embedding_dtype
    )

    with torch.inference_mode():
        logits = model.forward_padded(ROWS, _marker_tensor(), torch.tensor(QTYPES), "cpu", pad_token_id=PAD_ID)

    assert logits.dtype == torch.float32
    for i, markers in enumerate(MARKERS):
        mine, ref = logits[i, : len(markers)], expected[i, : len(markers)]
        torch.testing.assert_close(mine, ref, atol=1e-5, rtol=1e-5)
        # Guard against a vacuous pass: the options get distinct scores.
        assert float(ref.max() - ref.min()) > 1e-3


@pytest.mark.parametrize(
    ("checkpoint_dtype", "expected"),
    [
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (None, torch.float32),
    ],
)
def test_flash_prepare_keeps_embedding_at_checkpoint_precision(
    reference_checkpoint: tuple[ReferenceDecisionModel, Path],
    checkpoint_dtype: torch.dtype | None,
    expected: torch.dtype,
) -> None:
    """The flash path stores the embedding table at the checkpoint's own precision (never narrower)."""
    _, checkpoint = reference_checkpoint
    model, _ = _load_like_adapter(checkpoint, _encoder_config())
    # prepare() only places weights and builds RoPE tables; no flash kernel runs here.
    model.prepare(
        "cpu", use_flash=True, compute_dtype=torch.bfloat16, max_positions=64, embedding_dtype=checkpoint_dtype
    )
    encoder = model.encoder
    assert encoder.embeddings.tok_embeddings.weight.dtype == expected
    assert encoder.layers[0].attn.Wqkv.weight.dtype == torch.bfloat16
    assert encoder.embeddings.norm.weight.dtype == torch.float32
    assert encoder.final_norm.weight.dtype == torch.float32
