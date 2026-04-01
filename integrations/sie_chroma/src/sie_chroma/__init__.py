"""SIE integration for ChromaDB.

This package provides embedding functions for ChromaDB:

- SIEEmbeddingFunction: Dense embeddings for standard ChromaDB collections
- SIESparseEmbeddingFunction: Sparse embeddings for Chroma Cloud hybrid search

Example usage (dense):

    import chromadb
    from sie_chroma import SIEEmbeddingFunction

    embedding_function = SIEEmbeddingFunction(
        base_url="http://localhost:8080",
        model="BAAI/bge-m3",
    )

    client = chromadb.Client()
    collection = client.create_collection(
        name="my_collection",
        embedding_function=embedding_function,
    )

    collection.add(
        documents=["doc1", "doc2"],
        ids=["id1", "id2"],
    )

Example usage (sparse for hybrid search with Chroma Cloud):

    from sie_chroma import SIESparseEmbeddingFunction

    sparse_ef = SIESparseEmbeddingFunction(
        base_url="http://localhost:8080",
        model="BAAI/bge-m3",
    )
    # Use with SparseVectorIndexConfig for hybrid search
"""

from sie_chroma.embedding_function import SIEEmbeddingFunction, SIESparseEmbeddingFunction

__all__ = [
    "SIEEmbeddingFunction",
    "SIESparseEmbeddingFunction",
]
