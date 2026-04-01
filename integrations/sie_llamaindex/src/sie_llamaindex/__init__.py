"""SIE integration for LlamaIndex.

Provides embeddings, reranking, and extraction using SIE's inference server.

Example:
    >>> from llama_index.core import Settings, VectorStoreIndex
    >>> from sie_llamaindex import SIEEmbedding, SIENodePostprocessor
    >>>
    >>> # Set SIE as the embedding model
    >>> Settings.embed_model = SIEEmbedding(base_url="http://localhost:8080", model_name="BAAI/bge-m3")
    >>>
    >>> # Create index and add reranking
    >>> index = VectorStoreIndex.from_documents(documents)
    >>> reranker = SIENodePostprocessor(
    ...     base_url="http://localhost:8080", model="jinaai/jina-reranker-v2-base-multilingual"
    ... )
    >>> query_engine = index.as_query_engine(node_postprocessors=[reranker])

Hybrid search example:
    >>> from llama_index.vector_stores.qdrant import QdrantVectorStore
    >>> from sie_llamaindex import SIEEmbedding, SIESparseEmbeddingFunction
    >>>
    >>> vector_store = QdrantVectorStore(
    ...     client=qdrant_client,
    ...     collection_name="hybrid_docs",
    ...     enable_hybrid=True,
    ...     sparse_embedding_function=SIESparseEmbeddingFunction(model_name="BAAI/bge-m3"),
    ... )
"""

from sie_llamaindex.embeddings import SIEEmbedding, SIESparseEmbeddingFunction
from sie_llamaindex.extractors import SIEExtractorTool, create_sie_extractor_tool
from sie_llamaindex.rerankers import SIENodePostprocessor

__all__ = [
    "SIEEmbedding",
    "SIEExtractorTool",
    "SIENodePostprocessor",
    "SIESparseEmbeddingFunction",
    "create_sie_extractor_tool",
]
