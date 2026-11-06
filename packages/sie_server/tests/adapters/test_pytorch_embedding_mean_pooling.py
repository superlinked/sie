# Copyright 2026 Massimiliano Brighindi
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from sie_server.adapters.pytorch_embedding import PyTorchEmbeddingAdapter


def original(hidden_state, attention_mask, pooling=None):
    if pooling == "cls":
        return hidden_state[:, 0]
    if pooling == "last_token":
        # Fixtures below are unpadded, so the last token is the reference.
        return hidden_state[:, -1]
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def patched(hidden_state, attention_mask, pooling=None):
    adapter = PyTorchEmbeddingAdapter("unused-no-download", pooling="mean")
    return adapter._apply_pooling(hidden_state, attention_mask, pooling=pooling)


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 17, 64), (8, 257, 128), (2, 0, 16)])
@pytest.mark.parametrize("mask_kind", ["right", "left", "random", "all_zero"])
def test_mean_reference_and_no_mutation(shape, mask_kind):
    gen = torch.Generator().manual_seed(20261019)
    h = torch.randn(shape, generator=gen)
    b, s, _ = shape
    if mask_kind == "right":
        mask = (torch.arange(s).expand(b, s) < s // 2).long()
    elif mask_kind == "left":
        mask = (torch.arange(s).expand(b, s) >= s // 2).long()
    elif mask_kind == "random":
        mask = (torch.rand(b, s, generator=gen) > 0.3).long()
    else:
        mask = torch.zeros(b, s, dtype=torch.int64)
    before_h, before_mask = h.clone(), mask.clone()
    actual = patched(h, mask)
    torch.testing.assert_close(actual, original(h, mask), rtol=2e-5, atol=1e-6)
    assert actual.dtype == torch.float32
    assert torch.equal(h, before_h)
    assert torch.equal(mask, before_mask)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_other_precisions_use_unchanged_reference(dtype):
    h = torch.randn(3, 31, 17).to(dtype)
    mask = torch.ones(3, 31, dtype=torch.int64)
    torch.testing.assert_close(patched(h, mask), original(h, mask), rtol=0, atol=0)


@pytest.mark.parametrize("pooling", ["cls", "last_token"])
def test_other_pooling_strategies_unchanged(pooling):
    h = torch.randn(3, 31, 17)
    mask = torch.ones(3, 31, dtype=torch.int64)
    torch.testing.assert_close(patched(h, mask, pooling), original(h, mask, pooling), rtol=0, atol=0)


def test_noncontiguous_and_expanded_inputs():
    for h in (torch.randn(3, 17, 31).transpose(1, 2), torch.randn(1, 31, 17).expand(3, -1, -1)):
        mask = torch.ones(3, 31, dtype=torch.int64)
        torch.testing.assert_close(patched(h, mask), original(h, mask), rtol=2e-5, atol=1e-6)


def test_outer_autocast_does_not_reduce_precision():
    h = torch.randn(3, 31, 17)
    mask = torch.ones(3, 31, dtype=torch.int64)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual, expected = patched(h, mask), original(h, mask)
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)


def test_nonfinite_values_keep_reference_propagation():
    h = torch.randn(2, 5, 3)
    h[0, 1, 0] = float("nan")
    h[1, 4, 1] = float("inf")
    mask = torch.tensor([[1, 0, 1, 0, 1], [1, 1, 1, 1, 0]])
    torch.testing.assert_close(patched(h, mask), original(h, mask), equal_nan=True, rtol=2e-5, atol=1e-6)


def test_normalized_embeddings_and_top1_on_synthetic_fixture():
    gen = torch.Generator().manual_seed(20261019)
    h = torch.randn(128, 97, 64, generator=gen)
    mask = (torch.rand(128, 97, generator=gen) > 0.2).long()
    a = torch.nn.functional.normalize(original(h, mask), dim=-1)
    b = torch.nn.functional.normalize(patched(h, mask), dim=-1)
    torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-6)
    candidates = torch.randn(512, 64, generator=gen)
    candidates = torch.nn.functional.normalize(candidates, dim=-1)
    assert torch.equal((a @ candidates.T).argmax(dim=-1), (b @ candidates.T).argmax(dim=-1))
