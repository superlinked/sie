"""Retrieve source chunks, then rerank them without losing their identity."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from sie_sdk import SIEClient
from sie_sdk.types import Item


@dataclass(frozen=True)
class SourceChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedChunk:
    chunk: SourceChunk
    score: float


def normalize_chunks(results: object) -> list[SourceChunk]:
    """Accept released Cognee CHUNKS entries, rejecting derived completions."""
    from cognee.modules.recall.types.SearchResultItem import SearchResultItem

    if not isinstance(results, list):
        raise ValueError("Expected a list of Cognee CHUNKS results")
    chunks = []
    seen = set()
    for entry in results:
        if isinstance(entry, SearchResultItem):
            result = entry.model_dump()
        elif isinstance(entry, dict):
            result = cast("dict[str, Any]", entry)
        else:
            raise ValueError("Expected source chunks")
        if result.get("kind") != "chunk":
            raise ValueError("Expected source chunks; graph completions and session answers are not source evidence")
        text = result.get("text")
        metadata = result.get("metadata")
        raw = result.get("raw", {})
        if not isinstance(text, str) or not text.strip() or not isinstance(metadata, dict) or not isinstance(raw, dict):
            raise ValueError("Malformed source chunk text or metadata")
        chunk_id = metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id or chunk_id in seen:
            raise ValueError("Source chunks need unique nonempty chunk_id values")
        seen.add(chunk_id)
        chunks.append(SourceChunk(chunk_id, text, dict(metadata), dict(raw)))
    return chunks


async def retrieve_chunks(query: str, *, dataset: str, top_k: int = 5) -> list[SourceChunk]:
    import cognee
    from cognee.api.v1.search import SearchType

    if not query.strip() or not dataset.strip() or top_k < 1:
        raise ValueError("query, dataset and positive top_k are required")
    results = await cognee.recall(
        query,
        query_type=SearchType.CHUNKS,
        datasets=[dataset],
        top_k=top_k,
        auto_route=False,
    )
    return normalize_chunks(results)


def rerank_chunks(
    client: SIEClient,
    query: str,
    chunks: Sequence[SourceChunk],
    *,
    model: str,
    top_k: int = 3,
    provision_timeout_s: float = 180,
) -> list[RankedChunk]:
    if top_k < 1 or not query.strip():
        raise ValueError("query and positive top_k are required")
    ids = [chunk.chunk_id for chunk in chunks]
    if len(set(ids)) != len(ids) or any(not chunk.chunk_id or not chunk.text.strip() for chunk in chunks):
        raise ValueError("Candidates need unique nonempty identifiers and text")
    if not chunks:
        return []
    reply = client.score(
        model,
        Item(id="query", text=query),
        [Item(id=chunk.chunk_id, text=chunk.text) for chunk in chunks],
        wait_for_capacity=True,
        provision_timeout_s=provision_timeout_s,
    )
    entries = reply.get("scores") if isinstance(reply, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Reranker returned no score list")
    scores: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Reranker returned malformed score entries")
        item_id, score = entry.get("item_id"), entry.get("score")
        if not isinstance(item_id, str) or item_id not in ids or item_id in scores:
            raise ValueError("Reranker returned unknown, missing or duplicate item identifiers")
        if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(score):
            raise ValueError("Reranker scores must be finite numbers")
        scores[item_id] = float(score)
    if len(scores) != len(chunks):
        raise ValueError("Reranker did not score every candidate")
    ranked = [RankedChunk(chunk, scores[chunk.chunk_id]) for chunk in chunks]
    return sorted(ranked, key=lambda result: -result.score)[:top_k]
