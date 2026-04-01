# sie-weaviate

SIE integration for Weaviate v4.

## Two integration paths

### 1. Client-side (this package, works now)

`sie-weaviate` provides vectorizer helpers that call SIE's encode() and return
vectors in the format Weaviate expects. You configure collections with
`Configure.Vectors.self_provided()` and pass vectors on insert/query.

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

## Named vectors (dense + sparse)

SIE's multi-output encode produces dense and sparse vectors in one call.
Weaviate's named vectors feature stores them separately:

```python
from sie_weaviate import SIENamedVectorizer

vectorizer = SIENamedVectorizer(
    base_url="http://localhost:8080",
    model="BAAI/bge-m3",
    output_types=["dense", "sparse"],
)

collection = client.collections.create(
    "Documents",
    properties=[wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT)],
    vector_config=[
        wvc.config.Configure.Vectors.self_provided(name="dense"),
        wvc.config.Configure.Vectors.self_provided(name="sparse"),
    ],
)

named = vectorizer.embed_documents(["hello world"])
collection.data.insert_many([
    wvc.data.DataObject(properties={"text": "hello world"}, vector=named[0])
])
```

**Storage note:** SIE sparse vectors (SPLADE/BGE-M3) are expanded to full
vocabulary length (~30K floats per document for BERT-based models) so that
positional information is preserved for similarity search. At large scale this
is significant storage. If you only need keyword-style hybrid search, use
Weaviate's built-in BM25 instead — it requires no extra vectors:

```python
results = collection.query.hybrid(query="search text", alpha=0.75)
```

## Testing

```bash
# Unit tests (no server needed)
pytest

# Integration tests (requires SIE + Weaviate)
pytest -m integration
```
