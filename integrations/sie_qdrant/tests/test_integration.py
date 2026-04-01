"""Integration tests for sie-qdrant.

These tests require a running SIE server and a Qdrant instance.

Run with: pytest -m integration integrations/sie_qdrant/tests/

Prerequisites:
    # SIE server
    mise run serve -d cpu -p 8080

    # Qdrant (Docker)
    docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.13.2
"""

from __future__ import annotations

import os
import uuid

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def sie_url() -> str:
    return os.environ.get("SIE_SERVER_URL", "http://localhost:8080")


@pytest.fixture
def qdrant_client():
    """Create a Qdrant client connected to the test instance."""
    from qdrant_client import QdrantClient

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url)
    yield client
    client.close()


def _unique_name() -> str:
    return f"test_{uuid.uuid4().hex[:8]}"


class TestDenseSearch:
    """Dense vector search with SIEVectorizer."""

    def test_basic_search(self, sie_url: str, qdrant_client) -> None:
        """Add documents with SIE embeddings and search."""
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from sie_qdrant import SIEVectorizer

        vectorizer = SIEVectorizer(base_url=sie_url, model="BAAI/bge-m3")
        name = _unique_name()

        try:
            # Get embedding dimension from a test encode
            test_vec = vectorizer.embed_query("test")
            dim = len(test_vec)

            qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

            texts = [
                "Machine learning algorithms learn patterns from data.",
                "The weather forecast predicts rain tomorrow.",
                "Neural networks are inspired by biological neurons.",
                "Stock prices fluctuated today.",
                "Deep learning is a subset of machine learning.",
            ]

            vectors = vectorizer.embed_documents(texts)
            qdrant_client.upsert(
                collection_name=name,
                points=[
                    PointStruct(id=i, vector=v, payload={"text": t}) for i, (t, v) in enumerate(zip(texts, vectors))
                ],
            )

            query_vec = vectorizer.embed_query("How do neural networks work?")
            results = qdrant_client.query_points(
                collection_name=name,
                query=query_vec,
                limit=2,
            )

            assert len(results.points) == 2
        finally:
            qdrant_client.delete_collection(name)


class TestNamedVectorSearch:
    """Named vector search with SIENamedVectorizer."""

    def test_dense_and_sparse_named_vectors(self, sie_url: str, qdrant_client) -> None:
        """Store dense + sparse named vectors from one SIE call."""
        from qdrant_client.models import (
            Distance,
            PointStruct,
            SparseVector,
            SparseVectorParams,
            VectorParams,
        )
        from sie_qdrant import SIENamedVectorizer

        vectorizer = SIENamedVectorizer(
            base_url=sie_url,
            model="BAAI/bge-m3",
            output_types=["dense", "sparse"],
        )
        name = _unique_name()

        try:
            # Get embedding dimension from a test encode
            test_named = vectorizer.embed_query("test")
            dim = len(test_named["dense"])

            qdrant_client.create_collection(
                collection_name=name,
                vectors_config={"dense": VectorParams(size=dim, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams()},
            )

            texts = [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with multiple layers.",
                "Natural language processing analyzes human language.",
            ]

            named_vectors = vectorizer.embed_documents(texts)
            qdrant_client.upsert(
                collection_name=name,
                points=[
                    PointStruct(
                        id=i,
                        vector={
                            "dense": v["dense"],
                            "sparse": SparseVector(**v["sparse"]),
                        },
                        payload={"text": t},
                    )
                    for i, (t, v) in enumerate(zip(texts, named_vectors))
                ],
            )

            # Search with dense vector
            query_named = vectorizer.embed_query("neural networks")
            results = qdrant_client.query_points(
                collection_name=name,
                query=query_named["dense"],
                using="dense",
                limit=2,
            )

            assert len(results.points) == 2
        finally:
            qdrant_client.delete_collection(name)
