# sie-weaviate

SIE integration for Weaviate v4.

## Two integration paths

### 1. Client-side (this package, works now)

`sie-weaviate` provides vectorizer and enrichment helpers that call SIE's
encode() and extract() and return data in the format Weaviate expects. You
configure collections with `Configure.Vectors.self_provided()` and pass
vectors on insert/query.

```bash
pip install sie-weaviate
```

```python
import weaviate
import weaviate.classes as wvc
from sie_weaviate import SIEVectorizer

vectorizer = SIEVectorizer(base_url="http://localhost:8080", model="BAAI/bge-m3")

client = weaviate.connect_to_local()
try:
    collection = client.collections.create(
        "Documents",
        properties=[wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT)],
        vector_config=wvc.config.Configure.Vectors.self_provided(),
    )

    texts = ["first doc", "second doc"]
    vectors = vectorizer.embed_documents(texts)
    collection.data.insert_many([
        wvc.data.DataObject(properties={"text": t}, vector=v)
        for t, v in zip(texts, vectors)
    ])

    query_vec = vectorizer.embed_query("search text")
    results = collection.query.near_vector(near_vector=query_vec, limit=5)
finally:
    client.close()
```

### 2. Server-side module (partnership, planned)

A `text2vec-sie` Go module for the Weaviate server that enables native
vectorizer config (`Configure.Vectorizer.text2vec_sie(...)`). See
`weaviate-module-spec/` for the spec and reference implementation.

## Named vectors (dense + multivector)

`SIENamedVectorizer` produces multiple vector types in one SIE call.
Use it with ColBERT models that output both dense and multivector
(per-token) embeddings:

```python
from sie_weaviate import SIENamedVectorizer

vectorizer = SIENamedVectorizer(
    base_url="http://localhost:8080",
    model="jinaai/jina-colbert-v2",
    output_types=["dense", "multivector"],
)

collection = client.collections.create(
    "Documents",
    properties=[wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT)],
    vector_config=[
        wvc.config.Configure.Vectors.self_provided(name="dense"),
        wvc.config.Configure.Vectors.self_provided(name="multivector"),
    ],
)

named = vectorizer.embed_documents(["hello world"])
collection.data.insert_many([
    wvc.data.DataObject(properties={"text": "hello world"}, vector=named[0])
])
```

For hybrid search, Weaviate has built-in BM25 — no extra vectors needed:

```python
results = collection.query.hybrid(query="search text", alpha=0.75)
```

## Document enrichment for Query Agent

`SIEDocumentEnricher` combines SIE's embedding and entity extraction
pipelines to produce documents with dense vectors **and** structured
metadata. The extracted properties (persons, organizations, locations,
categories) are exactly what Weaviate's Query Agent uses to construct
filters from natural language queries.

```python
import weaviate
import weaviate.classes as wvc
from sie_weaviate import SIEDocumentEnricher

enricher = SIEDocumentEnricher(
    base_url="http://localhost:8080",
    labels=["person", "organization", "location"],
    classify_model="knowledgator/gliclass-large-v3.0",
    classify_labels=["technical", "business", "legal"],
)

client = weaviate.connect_to_local()
try:
    collection = client.collections.create(
        "Documents",
        description="Documents with extracted entity and classification metadata.",
        properties=[
            wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
            wvc.config.Property(
                name="person", data_type=wvc.config.DataType.TEXT_ARRAY,
                description="People mentioned in the document",
            ),
            wvc.config.Property(
                name="organization", data_type=wvc.config.DataType.TEXT_ARRAY,
                description="Organizations mentioned in the document",
            ),
            wvc.config.Property(
                name="location", data_type=wvc.config.DataType.TEXT_ARRAY,
                description="Locations mentioned in the document",
            ),
            wvc.config.Property(
                name="classification", data_type=wvc.config.DataType.TEXT,
                description="Document category: technical, business, or legal",
            ),
            wvc.config.Property(
                name="classification_score", data_type=wvc.config.DataType.NUMBER,
                description="Confidence score for the classification",
            ),
        ],
        vector_config=wvc.config.Configure.Vectors.self_provided(),
    )

    # Embed + extract in one call
    texts = [
        "John Smith presented Google's new AI strategy in New York.",
        "The court ruling on patent law affects tech companies.",
    ]
    docs = enricher.enrich(texts)
    collection.data.insert_many([
        wvc.data.DataObject(properties=doc.properties, vector=doc.vector)
        for doc in docs
    ])

    # The Query Agent can now filter on extracted properties:
    # "find documents about Google" → organization filter + vector search
    # "show me legal documents mentioning John Smith" → classification + person filter
    query_vec = enricher.enrich_query("AI strategy announcements")
    results = collection.query.near_vector(near_vector=query_vec, limit=5)
finally:
    client.close()
```

## Testing

```bash
# Unit tests (no server needed)
pytest

# Integration tests (requires SIE + Weaviate)
pytest -m integration
```
