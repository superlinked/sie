"""Fake adapter family (#1847): determinism properties + catalog-path loading.

Per the Fake Engine determinism contract, these tests assert self-equality,
distinctness, shape, and ordering — never pinned output values.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from sie_server.adapters._generation_base import collect_generation
from sie_server.adapters.fake.adapter import FakeAdapter
from sie_server.types.inputs import Item


@pytest.fixture
def embed() -> FakeAdapter:
    adapter = FakeAdapter(dense_dim=384)
    adapter.load("cpu")
    return adapter


@pytest.fixture
def reranker() -> FakeAdapter:
    adapter = FakeAdapter()
    adapter.load("cpu")
    return adapter


@pytest.fixture
def generator() -> FakeAdapter:
    adapter = FakeAdapter(default_completion_tokens=8)
    adapter.load("cpu")
    return adapter


# -- Embedding properties -----------------------------------------------------


def test_embed_same_input_identical(embed: FakeAdapter) -> None:
    a = embed.encode([Item(text="hello world")], output_types=["dense"])
    b = embed.encode([Item(text="hello world")], output_types=["dense"])
    assert a.dense is not None
    assert b.dense is not None
    np.testing.assert_array_equal(a.dense, b.dense)


def test_embed_distinct_inputs_differ(embed: FakeAdapter) -> None:
    out = embed.encode([Item(text="alpha"), Item(text="beta")], output_types=["dense"])
    assert out.dense is not None
    assert not np.array_equal(out.dense[0], out.dense[1])


def test_embed_shape_matches_config(embed: FakeAdapter) -> None:
    out = embed.encode([Item(text="x"), Item(text="y"), Item(text="z")], output_types=["dense"])
    assert out.dense is not None
    assert out.dense.shape == (3, 384)
    assert out.dense.dtype == np.float32
    assert not np.isnan(out.dense).any()
    # Unit-normalized vectors.
    np.testing.assert_allclose(np.linalg.norm(out.dense, axis=1), 1.0, rtol=1e-5)


def test_embed_dim_is_configurable() -> None:
    adapter = FakeAdapter(dense_dim=17)
    adapter.load("cpu")
    out = adapter.encode([Item(text="x")], output_types=["dense"])
    assert out.dense is not None
    assert out.dense.shape == (1, 17)
    assert adapter.dims.dense == 17


def test_embed_requires_load() -> None:
    adapter = FakeAdapter()
    with pytest.raises(RuntimeError):
        adapter.encode([Item(text="x")], output_types=["dense"])


# -- Reranker properties ------------------------------------------------------


def test_rerank_deterministic_and_distinct(reranker: FakeAdapter) -> None:
    query = Item(text="query")
    docs = [Item(text="doc one"), Item(text="doc two"), Item(text="doc three")]
    first = reranker.score(query, docs)
    second = reranker.score(query, docs)
    assert first == second
    assert len(first) == 3
    assert len(set(first)) == 3
    assert all(0.0 <= s < 1.0 for s in first)


def test_rerank_score_pairs_batched(reranker: FakeAdapter) -> None:
    queries = [Item(text="q1"), Item(text="q1"), Item(text="q2")]
    docs = [Item(text="d1"), Item(text="d2"), Item(text="d1")]
    out = reranker.score_pairs(queries, docs)
    assert out.scores.shape == (3,)
    assert out.input_token_counts == [4, 4, 4]
    # Same (query, doc) pair scores identically regardless of batch shape.
    solo = reranker.score(Item(text="q2"), [Item(text="d1")])
    assert out.scores[2] == pytest.approx(solo[0])


# -- Generation properties ----------------------------------------------------


async def test_generate_stream_contract(generator: FakeAdapter) -> None:
    chunks = [chunk async for chunk in generator.generate("a prompt", max_new_tokens=64)]
    assert len(chunks) == 9  # 8 deltas + terminal
    assert chunks[0].is_first
    assert all(not c.done for c in chunks[:-1])
    terminal = chunks[-1]
    assert terminal.done
    assert terminal.finish_reason == "stop"
    assert terminal.completion_tokens == 8
    assert terminal.prompt_tokens is not None
    assert terminal.prompt_tokens >= 1


async def test_generate_deterministic_and_distinct(generator: FakeAdapter) -> None:
    first = await collect_generation(generator.generate("same prompt", max_new_tokens=64))
    second = await collect_generation(generator.generate("same prompt", max_new_tokens=64))
    other = await collect_generation(generator.generate("different prompt", max_new_tokens=64))
    assert first.text
    assert first.text == second.text
    assert first.text != other.text


async def test_generate_truncates_to_max_new_tokens(generator: FakeAdapter) -> None:
    result = await collect_generation(generator.generate("p", max_new_tokens=3))
    assert result.completion_tokens == 3
    assert result.finish_reason == "length"


def test_generate_requires_load() -> None:
    adapter = FakeAdapter()

    async def _drain() -> None:
        async for _ in adapter.generate("p", max_new_tokens=1):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(_drain())
