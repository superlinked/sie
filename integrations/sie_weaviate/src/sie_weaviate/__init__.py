"""SIE integration for Weaviate.

This package provides vectorizer and enrichment helpers for Weaviate v4:

- SIEVectorizer: Compute dense embeddings via SIE for Weaviate collections
- SIENamedVectorizer: Compute multiple vector types (dense + multivector)
  for Weaviate named vectors
- SIEDocumentEnricher: Enrich documents with embeddings and extracted
  properties (entities, classifications) for Weaviate's Query Agent

Weaviate does not have a client-side embedding function protocol like ChromaDB.
Instead, you configure collections with ``Configure.Vectors.self_provided()`` and
pass pre-computed vectors on insert and query. This package handles the SIE encoding
so you don't have to manage Item creation, output type selection, or format conversion.

Example usage (dense):

    import weaviate
    import weaviate.classes as wvc
    from sie_weaviate import SIEVectorizer

    vectorizer = SIEVectorizer(
        base_url="http://localhost:8080",
        model="BAAI/bge-m3",
    )

    client = weaviate.connect_to_local()
    collection = client.collections.create(
        "Documents",
        vector_config=wvc.config.Configure.Vectors.self_provided(),
    )

    # Embed and insert
    texts = ["first doc", "second doc"]
    vectors = vectorizer.embed_documents(texts)
    objects = [
        wvc.data.DataObject(properties={"text": t}, vector=v)
        for t, v in zip(texts, vectors)
    ]
    collection.data.insert_many(objects)

    # Embed query and search
    query_vector = vectorizer.embed_query("search text")
    results = collection.query.near_vector(near_vector=query_vector, limit=5)

Example usage (enrichment for Query Agent):

    from sie_weaviate import SIEDocumentEnricher

    enricher = SIEDocumentEnricher(
        base_url="http://localhost:8080",
        labels=["person", "organization", "location"],
    )

    # Embed + extract entities in one call
    docs = enricher.enrich(["John Smith works at Google in NYC."])
    objects = [
        wvc.data.DataObject(properties=doc.properties, vector=doc.vector)
        for doc in docs
    ]
    collection.data.insert_many(objects)
    # Each object now has filterable person/organization/location properties
"""

from sie_weaviate.enricher import EnrichedDocument, SIEDocumentEnricher
from sie_weaviate.vectorizer import SIENamedVectorizer, SIEVectorizer

__all__ = [
    "EnrichedDocument",
    "SIEDocumentEnricher",
    "SIENamedVectorizer",
    "SIEVectorizer",
]
