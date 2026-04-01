"""SIE integration for Qdrant.

This package provides vectorizer helpers for Qdrant:

- SIEVectorizer: Compute dense embeddings via SIE for Qdrant collections
- SIENamedVectorizer: Compute multiple vector types (dense + sparse) for Qdrant named vectors

Qdrant supports both dense and sparse vectors natively. Dense vectors are
stored as ``list[float]``, and sparse vectors use ``SparseVector(indices, values)``
— no expansion to full vocabulary length is needed (unlike Weaviate).

Example usage (dense):

    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from sie_qdrant import SIEVectorizer

    vectorizer = SIEVectorizer(
        base_url="http://localhost:8080",
        model="BAAI/bge-m3",
    )

    qdrant = QdrantClient("http://localhost:6333")
    qdrant.create_collection(
        collection_name="documents",
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

    # Embed and insert
    texts = ["first doc", "second doc"]
    vectors = vectorizer.embed_documents(texts)
    qdrant.upsert(
        collection_name="documents",
        points=[
            PointStruct(id=i, vector=v, payload={"text": t})
            for i, (t, v) in enumerate(zip(texts, vectors))
        ],
    )

    # Embed query and search
    query_vector = vectorizer.embed_query("search text")
    results = qdrant.query_points(
        collection_name="documents",
        query=query_vector,
        limit=5,
    )

Example usage (named vectors for hybrid search):

    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        SparseVectorParams, SparseVector,
    )
    from sie_qdrant import SIENamedVectorizer

    vectorizer = SIENamedVectorizer(
        base_url="http://localhost:8080",
        model="BAAI/bge-m3",
    )

    qdrant = QdrantClient("http://localhost:6333")
    qdrant.create_collection(
        collection_name="documents",
        vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )

    # Embed with both dense and sparse in one SIE call
    texts = ["first doc", "second doc"]
    named = vectorizer.embed_documents(texts)
    qdrant.upsert(
        collection_name="documents",
        points=[
            PointStruct(
                id=i,
                vector={
                    "dense": v["dense"],
                    "sparse": SparseVector(**v["sparse"]),
                },
                payload={"text": t},
            )
            for i, (t, v) in enumerate(zip(texts, named))
        ],
    )
"""

from sie_qdrant.vectorizer import SIENamedVectorizer, SIEVectorizer

__all__ = [
    "SIENamedVectorizer",
    "SIEVectorizer",
]
