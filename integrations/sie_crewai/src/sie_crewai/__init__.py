"""SIE integration for CrewAI.

This package provides CrewAI tools and embedders that use SIE for inference:

- SIERerankerTool: Rerank documents by relevance to a query
- SIEExtractorTool: Extract entities from text
- SIESparseEmbedder: Sparse embeddings for hybrid search workflows

For dense embeddings, use SIE's OpenAI-compatible API directly with CrewAI:

    crew = Crew(
        agents=[...],
        tasks=[...],
        memory=True,
        embedder={
            "provider": "openai",
            "config": {
                "model": "BAAI/bge-m3",
                "api_base": "http://localhost:8080/v1",
            }
        }
    )

For hybrid search (dense + sparse), use SIESparseEmbedder alongside the OpenAI API:

    from sie_crewai import SIESparseEmbedder

    sparse_embedder = SIESparseEmbedder(
        base_url="http://localhost:8080",
        model="BAAI/bge-m3",
    )

    # Get sparse embeddings for your corpus
    sparse_vecs = sparse_embedder.embed_documents(my_documents)
    # Store in your vector DB alongside dense embeddings

    # Query
    sparse_query = sparse_embedder.embed_query(query)
"""

from sie_crewai.embedders import SIESparseEmbedder
from sie_crewai.tools import SIEExtractorTool, SIERerankerTool

__all__ = [
    "SIEExtractorTool",
    "SIERerankerTool",
    "SIESparseEmbedder",
]
