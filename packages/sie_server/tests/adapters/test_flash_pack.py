"""Equivalence tests for the shared flash varlen packing helper (issue #1538).

``build_position_ids`` replaces per-adapter ``torch.arange`` loops. These tests
pin it to be bit-identical to those loops (zero-based and padding-offset) across
``int32``/``int64`` ``cu_seqlens`` and zero-length sequences.
"""

from __future__ import annotations

import pytest
import torch
from sie_server.adapters._flash_pack import build_position_ids, mean_pool_packed

# Includes a zero-length middle sequence ([0, 2, 2, 5]) and single-token seqs.
_CASES = [[0, 3, 7, 8], [0, 5], [0, 1, 4, 4, 9], [0, 2, 2, 5], [0, 1, 2, 3]]


def _loop(cu: list[int], *, offset: int) -> torch.Tensor:
    parts = []
    for i in range(len(cu) - 1):
        seq_len = cu[i + 1] - cu[i]
        parts.append(torch.arange(offset, offset + seq_len, dtype=torch.long))
    return torch.cat(parts)


def test_matches_zero_based_loop() -> None:
    for dtype in (torch.int32, torch.int64):
        for case in _CASES:
            cu = torch.tensor(case, dtype=dtype)
            assert torch.equal(build_position_ids(cu), _loop(case, offset=0))


def test_matches_padding_offset_loop() -> None:
    # XLM-RoBERTa style: positions start at padding_idx + 1 (here padding_idx=1).
    for dtype in (torch.int32, torch.int64):
        for case in _CASES:
            cu = torch.tensor(case, dtype=dtype)
            assert torch.equal(build_position_ids(cu, offset=2), _loop(case, offset=2))


def test_result_is_int64() -> None:
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    assert build_position_ids(cu).dtype == torch.int64


def test_single_token_sequences() -> None:
    cu = torch.tensor([0, 1, 2, 3])
    assert torch.equal(build_position_ids(cu), torch.tensor([0, 0, 0]))


def test_known_total_matches_default_path() -> None:
    cu = torch.tensor([0, 2, 2, 5], dtype=torch.int32)
    assert torch.equal(
        build_position_ids(cu, offset=2, total_tokens=5),
        build_position_ids(cu, offset=2),
    )


def test_known_total_avoids_item() -> None:
    class ItemForbiddenTensor(torch.Tensor):
        def item(self) -> int:
            raise AssertionError("known-total path must not read a device scalar")

    cu = torch.tensor([0, 2, 2, 5], dtype=torch.int32).as_subclass(ItemForbiddenTensor)
    got = build_position_ids(cu, offset=2, total_tokens=5)

    assert torch.equal(got, torch.tensor([2, 3, 2, 3, 4]))


def test_known_total_must_match_cumulative_lengths() -> None:
    cu = torch.tensor([0, 2, 5], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="allocated size does not match required size"):
        build_position_ids(cu, total_tokens=4)


def _mean_loop(hidden: torch.Tensor, cu: torch.Tensor, num_seqs: int) -> torch.Tensor:
    """The exact per-sequence mean-pool loop the flash adapters carried inline."""
    means = []
    for i in range(num_seqs):
        start = cu[i].item()
        end = cu[i + 1].item()
        means.append(hidden[start:end].mean(dim=0))
    return torch.stack(means)


# Non-empty sequence boundaries: real flash batches never mean-pool an empty
# slice (that is nan, and nan != nan under torch.equal).
_MEAN_CASES = [[0, 3, 7, 8], [0, 5], [0, 1, 4, 9], [0, 1, 2, 3]]


def test_mean_pool_packed_matches_inline_loop() -> None:
    torch.manual_seed(0)
    for case in _MEAN_CASES:
        cu = torch.tensor(case)
        num_seqs = len(case) - 1
        hidden = torch.randn(case[-1], 4)
        assert torch.equal(mean_pool_packed(hidden, cu, num_seqs), _mean_loop(hidden, cu, num_seqs))
