# sie-qdrant

SIE integration for Qdrant.

## Installation

```bash
pip install sie-qdrant
```

## Dense embeddings

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sie_qdrant import SIEVectorizer

vectorizer = SIEVectorizer(base_url="http://localhost:8080", model="BAAI/bge-m3")

qdrant = QdrantClient("http://localhost:6333")
qdrant.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

texts = ["first doc", "second doc"]
vectors = vectorizer.embed_documents(texts)
qdrant.upsert(
    collection_name="documents",
    points=[
        PointStruct(id=i, vector=v, payload={"text": t})
        for i, (t, v) in enumerate(zip(texts, vectors))
    ],
)

query_vec = vectorizer.embed_query("search text")
results = qdrant.query_points(
    collection_name="documents", query=query_vec, limit=5
)
```

## Named vectors (dense + sparse)

SIE's multi-output encode produces dense and sparse vectors in one call.
Qdrant supports sparse vectors natively via `SparseVector(indices, values)`,
so no expansion to full vocabulary length is needed:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    SparseVectorParams, SparseVector,
)
from sie_qdrant import SIENamedVectorizer

vectorizer = SIENamedVectorizer(
    base_url="http://localhost:8080",
    model="BAAI/bge-m3",
    output_types=["dense", "sparse"],
)

qdrant = QdrantClient("http://localhost:6333")
qdrant.create_collection(
    collection_name="documents",
    vectors_config={"dense": VectorParams(size=1024, distance=Distance.COSINE)},
    sparse_vectors_config={"sparse": SparseVectorParams()},
)

named = vectorizer.embed_documents(["hello world"])
qdrant.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=0,
            vector={
                "dense": named[0]["dense"],
                "sparse": SparseVector(**named[0]["sparse"]),
            },
            payload={"text": "hello world"},
        )
    ],
)
```

**Storage advantage:** Unlike integrations that expand sparse vectors to full
vocabulary length (~30K floats), Qdrant stores sparse vectors in their native
indices+values form, making hybrid search storage-efficient.

## Testing

```bash
# Unit tests (no server needed)
pytest

# Integration tests (requires SIE + Qdrant)
pytest -m integration
```
