"""SIE integration for LanceDB.

This package provides LanceDB embedding functions, rerankers, and extractors
that use SIE for inference:

Embedding Functions (registered in LanceDB's embedding function registry):
- SIEEmbeddingFunction: Dense embeddings via the "sie" registry name
- SIEMultiVectorEmbeddingFunction: ColBERT/ColPali multi-vector via "sie-multivector"

Rerankers:
- SIEReranker: Cross-encoder reranking for hybrid search pipelines

Extractors:
- SIEExtractor: Entity extraction with table enrichment support

Example usage with embedding function registry:

    import lancedb
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector
    import sie_lancedb  # registers "sie" and "sie-multivector"

    sie = get_registry().get("sie").create(model="BAAI/bge-m3")

    class Documents(LanceModel):
        text: str = sie.SourceField()
        vector: Vector(sie.ndims()) = sie.VectorField()

    db = lancedb.connect("~/.lancedb")
    table = db.create_table("docs", schema=Documents)
    table.add([{"text": "hello world"}])  # auto-embeds
    results = table.search("hello").limit(5).to_list()  # auto-embeds query

Example usage with reranker:

    from sie_lancedb import SIEReranker

    table.create_fts_index("text")
    results = (
        table.search("hello", query_type="hybrid")
        .rerank(SIEReranker(model="jinaai/jina-reranker-v2-base-multilingual"))
        .limit(10)
        .to_list()
    )

Example usage with extractor:

    from sie_lancedb import SIEExtractor

    extractor = SIEExtractor(model="urchade/gliner_multi-v2.1")
    extractor.enrich_table(
        table,
        source_column="text",
        target_column="entities",
        labels=["person", "organization", "location"],
        id_column="id",
    )
"""

from sie_lancedb.embeddings import SIEEmbeddingFunction, SIEMultiVectorEmbeddingFunction
from sie_lancedb.extractors import ENTITY_STRUCT, SIEExtractor
from sie_lancedb.rerankers import SIEReranker

__all__ = [
    "ENTITY_STRUCT",
    "SIEEmbeddingFunction",
    "SIEExtractor",
    "SIEMultiVectorEmbeddingFunction",
    "SIEReranker",
]
