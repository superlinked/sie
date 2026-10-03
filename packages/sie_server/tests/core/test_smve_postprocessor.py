"""SMVE (Sparse Multi-Vector Encoding) postprocessor: multivector to sparse."""

from __future__ import annotations

import numpy as np
import pytest
from sie_server.core import postprocessor as postprocessor_module
from sie_server.core.inference_output import EncodeOutput, SparseVector
from sie_server.core.postprocessor import SmveConfig, SmvePostprocessor

TOKEN_DIM = 16


def _tokens(rng: np.random.Generator, n: int, dim: int = TOKEN_DIM) -> np.ndarray:
    tokens = rng.standard_normal((n, dim)).astype(np.float32)
    return tokens / np.linalg.norm(tokens, axis=1, keepdims=True)


def _anchors(config: SmveConfig, dim: int = TOKEN_DIM) -> list[np.ndarray]:
    """The anchors as the method defines them: seeded Gaussian columns scaled to unit length."""
    anchors = []
    for rep in range(config.num_repetitions):
        matrix = np.random.default_rng(config.seed + rep).standard_normal((dim, config.width), dtype=np.float32)
        anchors.append(matrix / np.linalg.norm(matrix, axis=0, keepdims=True))
    return anchors


def _reference(tokens: np.ndarray, config: SmveConfig, *, is_query: bool) -> np.ndarray:
    """SMVE written out per token: project, keep the top k, then sum (query) or average (document)."""
    pooled = []
    for anchors in _anchors(config, tokens.shape[1]):
        projections = tokens.astype(np.float64) @ anchors.astype(np.float64)
        sparse = np.zeros_like(projections)
        top = np.argsort(-projections, axis=1)[:, : config.k]
        np.put_along_axis(sparse, top, np.take_along_axis(projections, top, axis=1), axis=1)
        total = sparse.sum(axis=0)
        if not is_query:
            total /= np.maximum((sparse != 0).sum(axis=0), 1)
        pooled.append(total)
    return np.concatenate(pooled)


def _dense(vector: SparseVector, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float64)
    out[vector.indices] = vector.values
    return out


@pytest.mark.parametrize("is_query", [True, False])
@pytest.mark.parametrize("num_repetitions", [1, 2])
def test_matches_the_reference_definition(is_query: bool, num_repetitions: int) -> None:
    rng = np.random.default_rng(0)
    config = SmveConfig(width=256, k=8, num_repetitions=num_repetitions, seed=7)
    items = [_tokens(rng, n) for n in (1, 5, 13)]

    encoded = SmvePostprocessor(TOKEN_DIM, config).encode(items, is_query=is_query)

    for tokens, vector in zip(items, encoded, strict=True):
        np.testing.assert_allclose(
            _dense(vector, config.output_dim), _reference(tokens, config, is_query=is_query), rtol=1e-5, atol=1e-6
        )


def test_a_batch_encodes_each_item_as_if_alone() -> None:
    rng = np.random.default_rng(1)
    items = [_tokens(rng, n) for n in (3, 1, 9, 4)]
    smve = SmvePostprocessor(TOKEN_DIM, SmveConfig(width=128, k=4))

    together = smve.encode(items, is_query=False)
    alone = [smve.encode([tokens], is_query=False)[0] for tokens in items]

    for a, b in zip(together, alone, strict=True):
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_allclose(a.values, b.values, rtol=1e-6)


def test_chunks_and_groups_do_not_change_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(2)
    items = [_tokens(rng, n) for n in (7, 2, 11)]
    config = SmveConfig(width=64, k=4)
    expected = SmvePostprocessor(TOKEN_DIM, config).encode(items, is_query=True)

    # One token per projection chunk and one item per accumulator group.
    monkeypatch.setattr(postprocessor_module, "_SMVE_PROJECTION_CHUNK_BYTES", 1)
    monkeypatch.setattr(postprocessor_module, "_SMVE_ACCUMULATOR_BYTES", 1)
    got = SmvePostprocessor(TOKEN_DIM, config).encode(items, is_query=True)

    for a, b in zip(expected, got, strict=True):
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_allclose(a.values, b.values, rtol=1e-6)


def test_anchors_come_from_the_seed() -> None:
    config = SmveConfig(width=32, k=2, num_repetitions=2, seed=11)
    built = [a.numpy() for a in SmvePostprocessor(TOKEN_DIM, config)._get_anchors()]

    for got, expected in zip(built, _anchors(config), strict=True):
        np.testing.assert_array_equal(got, expected)
        np.testing.assert_allclose(np.linalg.norm(got, axis=0), 1.0, rtol=1e-6)
    other = SmvePostprocessor(TOKEN_DIM, SmveConfig(width=32, k=2, seed=12))._get_anchors()[0].numpy()
    assert not np.array_equal(other, built[0])


def test_queries_sum_and_documents_average() -> None:
    token = _tokens(np.random.default_rng(3), 1)
    smve = SmvePostprocessor(TOKEN_DIM, SmveConfig(width=64, k=4))
    single_query, double_query = smve.encode([token, np.vstack([token, token])], is_query=True)
    single_doc, double_doc = smve.encode([token, np.vstack([token, token])], is_query=False)

    np.testing.assert_array_equal(double_query.indices, single_query.indices)
    np.testing.assert_allclose(double_query.values, 2 * single_query.values, rtol=1e-6)
    np.testing.assert_array_equal(double_doc.indices, single_doc.indices)
    np.testing.assert_allclose(double_doc.values, single_doc.values, rtol=1e-6)


def test_output_layout() -> None:
    rng = np.random.default_rng(4)
    config = SmveConfig(width=100, k=5, num_repetitions=3)
    smve = SmvePostprocessor(TOKEN_DIM, config)
    (single,), (many,) = smve.encode([_tokens(rng, 1)], is_query=True), smve.encode([_tokens(rng, 40)], is_query=False)

    assert smve.target_dim == config.output_dim == 300
    # One token keeps exactly k projections per repetition.
    assert single.indices.size == config.k * config.num_repetitions
    for vector in (single, many):
        assert vector.indices.dtype == np.int32
        assert vector.values.dtype == np.float32
        assert np.all(np.diff(vector.indices) > 0)
        assert vector.indices.min() >= 0
        assert vector.indices.max() < config.output_dim
    assert many.indices.size <= 40 * config.k * config.num_repetitions


def test_items_without_tokens_encode_as_empty_vectors() -> None:
    rng = np.random.default_rng(5)
    empty = np.zeros((0, TOKEN_DIM), dtype=np.float32)
    smve = SmvePostprocessor(TOKEN_DIM, SmveConfig(width=32, k=2))

    assert all(v.indices.size == 0 for v in smve.encode([empty, empty], is_query=True))
    first, middle, last = smve.encode([empty, _tokens(rng, 3), empty], is_query=False)
    assert first.indices.size == 0
    assert middle.indices.size > 0
    assert last.indices.size == 0


def test_max_nonzeros_keeps_the_largest_values() -> None:
    tokens = _tokens(np.random.default_rng(6), 30)
    full = SmvePostprocessor(TOKEN_DIM, SmveConfig(width=128, k=8)).encode([tokens], is_query=False)[0]
    pruned = SmvePostprocessor(TOKEN_DIM, SmveConfig(width=128, k=8, max_nonzeros=10)).encode([tokens], is_query=False)[
        0
    ]

    largest = np.sort(np.argsort(-np.abs(full.values))[:10])
    np.testing.assert_array_equal(pruned.indices, full.indices[largest])
    np.testing.assert_allclose(pruned.values, full.values[largest], rtol=1e-6)


def test_dot_product_ranks_a_matching_document_first() -> None:
    rng = np.random.default_rng(7)
    query = _tokens(rng, 6)
    matching = query + 0.05 * rng.standard_normal(query.shape).astype(np.float32)
    documents = [_tokens(rng, 6) for _ in range(9)]
    documents.insert(4, matching / np.linalg.norm(matching, axis=1, keepdims=True))
    smve = SmvePostprocessor(TOKEN_DIM, SmveConfig(width=2048, k=16))

    (encoded_query,) = smve.encode([query], is_query=True)
    scores = [_dense(encoded_query, 2048) @ _dense(d, 2048) for d in smve.encode(documents, is_query=False)]

    assert int(np.argmax(scores)) == 4


def test_transform_adds_sparse_output() -> None:
    rng = np.random.default_rng(8)
    output = EncodeOutput(multivector=[_tokens(rng, 3), _tokens(rng, 2)])
    SmvePostprocessor(TOKEN_DIM, SmveConfig(width=32, k=2)).transform(output, is_query=True)

    assert output.sparse is not None
    assert len(output.sparse) == 2
    assert output.multivector is not None


def test_transform_requires_multivectors() -> None:
    with pytest.raises(ValueError, match="requires multivector"):
        SmvePostprocessor(TOKEN_DIM).transform(EncodeOutput(dense=np.zeros((1, 4), dtype=np.float32)))


def test_rejects_tokens_of_another_width() -> None:
    with pytest.raises(ValueError, match="token embeddings"):
        SmvePostprocessor(TOKEN_DIM, SmveConfig(width=32, k=2)).encode(
            [np.zeros((2, TOKEN_DIM + 1), dtype=np.float32)], is_query=True
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"width": 8, "k": 0},
        {"width": 8, "k": 9},
        {"num_repetitions": 0},
        {"max_nonzeros": 0},
    ],
)
def test_config_rejects_invalid_settings(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="SMVE"):
        SmveConfig(**kwargs)
