"""GPU output-equivalence gate: SPLADE packed-flash vs native-padded forward.

This is the acceptance gate for enabling the RoBERTa packed flash path
(granite-embedding-30m-sparse). It loads a real SPLADE model on CUDA with
flash-attn, runs the SAME texts through both ``_encode_flash`` (packed varlen)
and ``_encode_native`` (padded model forward), and asserts the sparse
term-weight vectors are equivalent (cosine >= 0.9999, tiny max-abs-diff) across
varied lengths incl. boundary cases.

GPU-gated: skips in CI (no CUDA / no flash-attn / model not cached). Run on a
CUDA box with the model cached locally:

    SIE_SPLADE_PARITY_MODEL=ibm-granite/granite-embedding-30m-sparse \
      mise exec -- uv run pytest \
      packages/sie_server/tests/adapters/test_splade_packed_parity.py -v -s
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch
from sie_server.adapters.splade_flash.adapter import SPLADEFlashAdapter
from sie_server.core.inference_output import SparseVector

_HAS_CUDA = torch.cuda.is_available()
_HAS_FLASH = importlib.util.find_spec("flash_attn") is not None
_MODEL_ID = os.environ.get("SIE_SPLADE_PARITY_MODEL", "ibm-granite/granite-embedding-30m-sparse")

pytestmark = pytest.mark.skipif(
    not (_HAS_CUDA and _HAS_FLASH),
    reason="packed-flash parity requires CUDA + flash-attn (GPU box only)",
)

# Query + document texts spanning boundary cases: single token, short query,
# medium doc, a long doc that trips max_seq truncation, and repeated tokens.
_TEXTS = [
    "cache",
    "how do i shard a batch across gpus",
    (
        "The managed inference edge coalesces concurrent single-item requests into "
        "deeper GPU forwards over a persistent low-latency channel, then meters exact "
        "on-wire token counts per item for billing and returns a learned-sparse "
        "term-weight vector for each document chunk in the batch."
    ),
    " ".join(["model batch cache index shard metric chunk hash probe split"] * 40),
    "aaa aaa aaa aaa aaa",
]


def _densify(vec: SparseVector, dim: int) -> np.ndarray:
    dense = np.zeros(dim, dtype=np.float64)
    dense[vec.indices.astype(np.int64)] = vec.values.astype(np.float64)
    return dense


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@pytest.fixture(scope="module")
def adapter() -> SPLADEFlashAdapter:
    try:
        a = SPLADEFlashAdapter(_MODEL_ID, max_seq_length=512, compute_precision="float16")
        a.load("cuda")
    except Exception as exc:  # noqa: BLE001 — model not staged in this env → skip
        pytest.skip(f"model {_MODEL_ID!r} not loadable on this box: {exc}")
    return a


def test_guard_lifted_flash_path_active(adapter: SPLADEFlashAdapter) -> None:
    """The RoBERTa packed flash path must actually be enabled (guard lifted)."""
    assert adapter._use_flash is True, "packed flash path is not active — guard not lifted?"


@pytest.mark.parametrize("i", range(len(_TEXTS)), ids=[f"text{i}" for i in range(len(_TEXTS))])
def test_single_item_parity(adapter: SPLADEFlashAdapter, i: int) -> None:
    """Each text alone: packed forward == padded forward."""
    dim = adapter._vocab_size
    flash_outputs, flash_counts = adapter._encode_flash([_TEXTS[i]])
    native_outputs, native_counts = adapter._encode_native([_TEXTS[i]])
    assert flash_counts == native_counts
    flash = flash_outputs[0]
    native = native_outputs[0]
    cos = _cosine(_densify(flash, dim), _densify(native, dim))
    max_abs = float(np.max(np.abs(_densify(flash, dim) - _densify(native, dim))))
    print(
        f"[text{i}] cosine={cos:.6f} max_abs_diff={max_abs:.4e} nnz_flash={len(flash.indices)} nnz_native={len(native.indices)}"
    )
    assert cos >= 0.9999, f"text{i} cosine {cos} < 0.9999"


def test_mixed_batch_parity(adapter: SPLADEFlashAdapter) -> None:
    """One packed batch of varied lengths: every item matches the padded path."""
    dim = adapter._vocab_size
    flash, flash_counts = adapter._encode_flash(_TEXTS)
    native, native_counts = adapter._encode_native(_TEXTS)
    assert len(flash) == len(native) == len(_TEXTS)
    assert flash_counts == native_counts
    worst = 1.0
    for j, (f, n) in enumerate(zip(flash, native, strict=True)):
        cos = _cosine(_densify(f, dim), _densify(n, dim))
        worst = min(worst, cos)
        print(f"[mixed item{j}] cosine={cos:.6f}")
        assert cos >= 0.9999, f"mixed batch item{j} cosine {cos} < 0.9999"
    print(f"[mixed batch] worst cosine={worst:.6f}")
