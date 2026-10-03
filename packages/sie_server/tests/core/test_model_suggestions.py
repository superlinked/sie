"""Near-match suggestions for unknown model ids."""

from __future__ import annotations

import pytest
from sie_server.core.model_suggestions import suggest_models, suggestion_suffix

# A representative slice of the shipped catalog, including the near-collisions
# that make naive matching dangerous (three bge rerankers, two gte-Qwen2 sizes).
CATALOG = [
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "BAAI/bge-m3",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-v2-m3",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "jinaai/jina-colbert-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]


def test_missing_org_prefix_is_the_headline_case() -> None:
    """`bge-m3` must resolve to `BAAI/bge-m3`.

    The catalog has no short-name aliasing, so dropping the org prefix is the
    single most likely way to miss — including from the SDK's own docstrings.
    """
    assert suggest_models("bge-m3", CATALOG)[0] == "BAAI/bge-m3"


def test_case_slip_is_matched() -> None:
    assert suggest_models("baai/BGE-M3", CATALOG)[0] == "BAAI/bge-m3"


def test_wrong_org_prefix_still_finds_the_model() -> None:
    assert suggest_models("huggingface/bge-m3", CATALOG)[0] == "BAAI/bge-m3"


def test_typo_in_the_basename_is_matched() -> None:
    assert "Qwen/Qwen3-0.6B" in suggest_models("Qwen/Qwen3-0.6b", CATALOG)


def test_partial_reranker_name_suggests_rerankers() -> None:
    """`bge-reranker-v2` (a real docs typo) should surface the v2 variant."""
    matches = suggest_models("bge-reranker-v2", CATALOG)
    assert "BAAI/bge-reranker-v2-m3" in matches


def test_nothing_close_returns_nothing() -> None:
    """A confidently wrong guess is worse than no guess."""
    assert suggest_models("completely-unrelated-xyzzy", CATALOG) == []


def test_limit_is_respected() -> None:
    assert len(suggest_models("bge-reranker", CATALOG, limit=2)) == 2


def test_results_are_deterministic() -> None:
    """Equal scores must not reorder between runs."""
    assert suggest_models("bge-reranker", CATALOG) == suggest_models("bge-reranker", CATALOG)


@pytest.mark.parametrize("empty", ["", "   "])
def test_empty_query_returns_nothing(empty: str) -> None:
    assert suggest_models(empty, CATALOG) == []


def test_empty_catalog_returns_nothing() -> None:
    assert suggest_models("bge-m3", []) == []


def test_zero_limit_returns_nothing() -> None:
    assert suggest_models("bge-m3", CATALOG, limit=0) == []


def test_suffix_is_appendable_prose() -> None:
    suffix = suggestion_suffix("bge-m3", CATALOG)
    assert suffix.startswith(" Did you mean ")
    assert "'BAAI/bge-m3'" in suffix
    assert suffix.endswith("?")


def test_suffix_is_empty_when_there_is_no_match() -> None:
    """Callers append unconditionally, so "no suggestion" must add nothing."""
    assert suggestion_suffix("completely-unrelated-xyzzy", CATALOG) == ""
