"""Integration tests for sie-weaviate.

These tests require a running SIE server and a Weaviate instance.

Run with: pytest -m integration integrations/sie_weaviate/tests/

Prerequisites:
    # SIE server
    mise run serve -d cpu -p 8080

    # Weaviate (Docker)
    docker run -d -p 8090:8080 -p 50051:50051 \
        -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
        -e DEFAULT_VECTORIZER_MODULE=none \
        cr.weaviate.io/semitechnologies/weaviate:1.28.0
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
def weaviate_client():
    """Create a Weaviate client connected to the test instance."""
    import weaviate

    port = int(os.environ.get("WEAVIATE_PORT", "8090"))
    grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
    client = weaviate.connect_to_local(port=port, grpc_port=grpc_port)
    yield client
    client.close()


def _unique_name() -> str:
    return f"Test_{uuid.uuid4().hex[:8]}"


class TestDenseSearch:
    """Dense vector search with SIEVectorizer."""

    def test_basic_search(self, sie_url: str, weaviate_client) -> None:
        """Add documents with SIE embeddings and search."""
        import weaviate.classes as wvc
        from sie_weaviate import SIEVectorizer

        vectorizer = SIEVectorizer(base_url=sie_url, model="BAAI/bge-m3")
        name = _unique_name()

        try:
            collection = weaviate_client.collections.create(
                name,
                properties=[wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT)],
                vector_config=wvc.config.Configure.Vectors.self_provided(),
            )

            texts = [
                "Machine learning algorithms learn patterns from data.",
                "The weather forecast predicts rain tomorrow.",
                "Neural networks are inspired by biological neurons.",
                "Stock prices fluctuated today.",
                "Deep learning is a subset of machine learning.",
            ]

            vectors = vectorizer.embed_documents(texts)
            collection.data.insert_many(
                [wvc.data.DataObject(properties={"text": t}, vector=v) for t, v in zip(texts, vectors)]
            )

            query_vec = vectorizer.embed_query("How do neural networks work?")
            results = collection.query.near_vector(near_vector=query_vec, limit=2)

            assert len(results.objects) == 2
        finally:
            weaviate_client.collections.delete(name)


class TestNamedVectorSearch:
    """Named vector search with SIENamedVectorizer."""

    def test_dense_and_sparse_named_vectors(self, sie_url: str, weaviate_client) -> None:
        """Store dense + sparse named vectors from one SIE call."""
        import weaviate.classes as wvc
        from sie_weaviate import SIENamedVectorizer

        vectorizer = SIENamedVectorizer(
            base_url=sie_url,
            model="BAAI/bge-m3",
            output_types=["dense", "sparse"],
        )
        name = _unique_name()

        try:
            collection = weaviate_client.collections.create(
                name,
                properties=[wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT)],
                vector_config=[
                    wvc.config.Configure.Vectors.self_provided(name="dense"),
                    wvc.config.Configure.Vectors.self_provided(name="sparse"),
                ],
            )

            texts = [
                "Machine learning is a subset of artificial intelligence.",
                "Deep learning uses neural networks with multiple layers.",
                "Natural language processing analyzes human language.",
            ]

            named_vectors = vectorizer.embed_documents(texts)
            collection.data.insert_many(
                [
                    wvc.data.DataObject(
                        properties={"text": t},
                        vector=v,
                    )
                    for t, v in zip(texts, named_vectors)
                ]
            )

            query_named = vectorizer.embed_query("neural networks")
            results = collection.query.near_vector(
                near_vector=query_named["dense"],
                target_vector="dense",
                limit=2,
            )

            assert len(results.objects) == 2
        finally:
            weaviate_client.collections.delete(name)
