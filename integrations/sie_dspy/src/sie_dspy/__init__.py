"""SIE integration for DSPy.

This package provides DSPy-compatible components for SIE:

- SIEEmbedder: Dense embedding function for dspy.retrievers.Embeddings
- SIESparseEmbedder: Sparse embedding function for hybrid search workflows
- SIEReranker: Module to rerank passages by relevance
- SIEExtractor: Module to extract entities from text

Example usage with DSPy's built-in FAISS retriever (dense only):

    import dspy
    from sie_dspy import SIEEmbedder

    embedder = SIEEmbedder(
        base_url="http://localhost:8080",
        model="BAAI/bge-m3",
    )

    retriever = dspy.retrievers.Embeddings(
        corpus=["doc1", "doc2", "doc3"],
        embedder=embedder,
        k=2,
    )

    results = retriever("search query")

Example usage with hybrid search (dense + sparse):

    from sie_dspy import SIEEmbedder, SIESparseEmbedder

    dense_embedder = SIEEmbedder(model="BAAI/bge-m3")
    sparse_embedder = SIESparseEmbedder(model="BAAI/bge-m3")

    # Index corpus - store in your vector DB (Qdrant, Weaviate, etc.)
    dense_vecs = dense_embedder(corpus)
    sparse_vecs = sparse_embedder.embed_documents(corpus)

    # Query
    dense_query = dense_embedder(query)
    sparse_query = sparse_embedder.embed_query(query)
"""

from sie_dspy.embedder import SIEEmbedder, SIESparseEmbedder
from sie_dspy.modules import (
    Classification,
    DetectedObject,
    Entity,
    Relation,
    SIEExtractor,
    SIEReranker,
)

__all__ = [
    "Classification",
    "DetectedObject",
    "Entity",
    "Relation",
    "SIEEmbedder",
    "SIEExtractor",
    "SIEReranker",
    "SIESparseEmbedder",
]
