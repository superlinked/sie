"""SIE integration for Weaviate.

This package provides vectorizer helpers for Weaviate v4:

- SIEVectorizer: Compute dense embeddings via SIE for Weaviate collections
- SIENamedVectorizer: Compute multiple vector types (dense + sparse) for Weaviate named vectors

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

Example usage (named vectors for hybrid search):

    from sie_weaviate import SIENamedVectorizer

    vectorizer = SIENamedVectorizer(
        base_url="http://localhost:8080",
        model="BAAI/bge-m3",
    )

    collection = client.collections.create(
        "Documents",
        vector_config=[
            wvc.config.Configure.Vectors.self_provided(name="dense"),
            wvc.config.Configure.Vectors.self_provided(name="sparse"),
        ],
    )

    # Embed with both dense and sparse in one SIE call
    named = vectorizer.embed_documents(texts)
    objects = [
        wvc.data.DataObject(
            properties={"text": t},
            vector={"dense": v["dense"], "sparse": v["sparse"]},
        )
        for t, v in zip(texts, named)
    ]
    collection.data.insert_many(objects)
"""

from sie_weaviate.vectorizer import SIENamedVectorizer, SIEVectorizer

__all__ = [
    "SIENamedVectorizer",
    "SIEVectorizer",
]
