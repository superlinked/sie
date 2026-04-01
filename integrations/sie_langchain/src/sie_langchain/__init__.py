"""SIE integration for LangChain.

Provides LangChain-compatible wrappers for SIE's encoding, reranking,
and entity extraction capabilities.

Example:
    >>> from sie_langchain import SIEEmbeddings, SIEReranker, SIEExtractor
    >>>
    >>> # Create embeddings
    >>> embeddings = SIEEmbeddings(base_url="http://localhost:8080", model="BAAI/bge-m3")
    >>> vectors = embeddings.embed_documents(["Hello world"])
    >>>
    >>> # Create reranker
    >>> reranker = SIEReranker(base_url="http://localhost:8080")
    >>> reranked = reranker.compress_documents(documents, query)
    >>>
    >>> # Create extractor tool
    >>> extractor = SIEExtractor(base_url="http://localhost:8080")
    >>> entities = extractor.invoke("John Smith works at Acme Corp")

Hybrid search example:
    >>> from langchain_pinecone import PineconeHybridSearchRetriever
    >>> from sie_langchain import SIEEmbeddings, SIESparseEncoder
    >>>
    >>> retriever = PineconeHybridSearchRetriever(
    ...     embeddings=SIEEmbeddings(model="BAAI/bge-m3"),
    ...     sparse_encoder=SIESparseEncoder(model="BAAI/bge-m3"),
    ...     index=pinecone_index,
    ... )
"""

from sie_langchain.embeddings import SIEEmbeddings, SIESparseEncoder
from sie_langchain.extractors import SIEExtractor
from sie_langchain.rerankers import SIEReranker

__all__ = ["SIEEmbeddings", "SIEExtractor", "SIEReranker", "SIESparseEncoder"]
