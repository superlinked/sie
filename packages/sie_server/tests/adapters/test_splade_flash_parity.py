"""Flash-vs-native parity for SPLADEFlashAdapter (#1685).

The packed flash path (hand-rolled attention + ``flash_attn_varlen_func``)
and the native path (the model's own eager forward) must produce the same
sparse vectors within fp16 tolerance. The 13.5% StackOverflowQA regression
shipped precisely because nothing pinned the two paths (and the floors)
to each other.

Needs CUDA + a working flash_attn, so it runs on the GPU lanes and is
skipped elsewhere (also gated behind the ``model`` marker because it
downloads the checkpoint).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sie_server.adapters.splade_flash.adapter import SPLADEFlashAdapter, _has_flash_attn
from sie_server.types.inputs import Item

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and _has_flash_attn()),
        reason="flash-vs-native parity needs CUDA and a working flash_attn",
    ),
]

MODEL_ID = "naver/splade-v3"

# Short single-segment queries — the shape that regressed on StackOverflowQA —
# plus a longer passage so the packed path exercises mixed sequence lengths.
TEXTS = [
    "how to sort a list in python",
    "segfault when freeing a pointer twice in C",
    "what does the yield keyword do",
    "SPLADE builds sparse lexical representations by max-pooling the "
    "log-saturated MLM logits over the sequence, producing one weight per "
    "vocabulary term; padding and structural tokens must not contribute.",
]


def _encode_sparse(use_flash: bool) -> list[dict[int, float]]:
    adapter = SPLADEFlashAdapter(MODEL_ID)
    adapter.load("cuda:0")
    assert adapter._use_flash, "flash path unexpectedly unavailable on this runner"
    adapter._use_flash = use_flash
    items: list[Item] = [{"text": t} for t in TEXTS]
    out = adapter.encode(items, ["sparse"])
    return [dict(zip(sv.indices.tolist(), sv.values.tolist(), strict=True)) for sv in out.sparse]


def test_flash_matches_native_within_fp16_tolerance() -> None:
    flash = _encode_sparse(use_flash=True)
    native = _encode_sparse(use_flash=False)

    for f, n in zip(flash, native, strict=True):
        union = set(f) | set(n)
        assert union, "empty sparse vector from both paths"
        diffs = np.array([abs(f.get(i, 0.0) - n.get(i, 0.0)) for i in union])
        # SPLADE weights sit in ~[0, 3.5]; fp16 accumulation across the two
        # attention implementations stays well inside this envelope. Terms
        # present on only one side must be threshold-boundary noise.
        assert diffs.max() < 0.05, f"max sparse-weight divergence {diffs.max():.4f}"

        f_top = {i for i, _ in sorted(f.items(), key=lambda kv: -kv[1])[:20]}
        n_top = {i for i, _ in sorted(n.items(), key=lambda kv: -kv[1])[:20]}
        overlap = len(f_top & n_top) / 20
        assert overlap >= 0.9, f"top-20 term overlap only {overlap:.2f}"
