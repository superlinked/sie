"""SIE integration for Haystack.

This package provides Haystack components that use SIE for inference:

Dense Embedders:
- SIETextEmbedder: Embeds single text strings (queries)
- SIEDocumentEmbedder: Embeds documents and stores embeddings on them

Sparse Embedders (for hybrid search):
- SIESparseTextEmbedder: Sparse embeddings for queries
- SIESparseDocumentEmbedder: Sparse embeddings for documents

Rankers and Extractors:
- SIERanker: Reranks documents by relevance to a query
- SIEExtractor: Extracts entities from text

Example usage:
    from haystack import Document
    from sie_haystack import SIETextEmbedder, SIEDocumentEmbedder, SIERanker

    # Embed a query
    text_embedder = SIETextEmbedder(base_url="http://localhost:8080", model="BAAI/bge-m3")
    result = text_embedder.run(text="What is machine learning?")
    query_embedding = result["embedding"]

    # Embed documents
    doc_embedder = SIEDocumentEmbedder(base_url="http://localhost:8080", model="BAAI/bge-m3")
    docs = [Document(content="Python is a programming language.")]
    result = doc_embedder.run(documents=docs)
    embedded_docs = result["documents"]

    # Rerank documents
    ranker = SIERanker(base_url="http://localhost:8080", model="jinaai/jina-reranker-v2-base-multilingual")
    result = ranker.run(query="What is Python?", documents=embedded_docs, top_k=3)
    ranked_docs = result["documents"]

Hybrid search example:
    from sie_haystack import SIESparseTextEmbedder, SIESparseDocumentEmbedder

    # Sparse embeddings for hybrid search with Qdrant
    sparse_text_embedder = SIESparseTextEmbedder(model="BAAI/bge-m3")
    result = sparse_text_embedder.run(text="What is machine learning?")
    sparse_embedding = result["sparse_embedding"]  # {"indices": [...], "values": [...]}
"""

from sie_haystack.embedders import (
    SIEDocumentEmbedder,
    SIESparseDocumentEmbedder,
    SIESparseTextEmbedder,
    SIETextEmbedder,
)
from sie_haystack.extractors import SIEExtractor
from sie_haystack.rankers import SIERanker

__all__ = [
    "SIEDocumentEmbedder",
    "SIEExtractor",
    "SIERanker",
    "SIESparseDocumentEmbedder",
    "SIESparseTextEmbedder",
    "SIETextEmbedder",
]
