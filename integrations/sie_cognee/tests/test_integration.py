"""Optional real-model lane: existing embedding/rerank fixtures only."""

import os

import pytest
from sie_cognee.config import Endpoint
from sie_cognee.retrieval import SourceChunk, rerank_chunks
from sie_sdk import SIEClient

pytestmark = pytest.mark.integration


def test_rerank_sources():
    with SIEClient(os.environ.get("SIE_SERVER_URL", "http://localhost:8080")) as client:
        chunks = [
            SourceChunk("python", "Python is a programming language."),
            SourceChunk("fruit", "Apples grow on trees."),
        ]
        result = rerank_chunks(
            client, "programming", chunks, model="jinaai/jina-reranker-v2-base-multilingual", top_k=2
        )
    assert {item.chunk.chunk_id for item in result} == {"python", "fruit"}
    assert result[0].score >= result[1].score


@pytest.mark.asyncio
async def test_compatible_embedding_route():
    from cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine import (
        OpenAICompatibleEmbeddingEngine,
    )

    endpoint = Endpoint(os.environ.get("SIE_SERVER_URL", "http://localhost:8080"), os.environ.get("SIE_API_KEY"))
    engine = OpenAICompatibleEmbeddingEngine(
        model="BAAI/bge-m3", dimensions=1024, endpoint=endpoint.compatible_url(), api_key=endpoint.compatible_key()
    )
    result = await engine.embed_text(["Python is a programming language."])
    assert len(result) == 1
    assert len(result[0]) == 1024
