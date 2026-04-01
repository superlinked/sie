# sie-haystack

SIE integration for Haystack.

## Installation

```bash
pip install sie-haystack
```

## Usage

```python
from haystack import Document
from sie_haystack import SIETextEmbedder, SIEDocumentEmbedder, SIERanker

# Embed a query
text_embedder = SIETextEmbedder(base_url="http://localhost:8080", model="BAAI/bge-m3")
result = text_embedder.run(text="What is machine learning?")
query_embedding = result["embedding"]

# Embed documents
doc_embedder = SIEDocumentEmbedder(base_url="http://localhost:8080", model="BAAI/bge-m3")
docs = [Document(content="Python is a programming language.")]
result = doc_embedder.run(documents=docs)
embedded_docs = result["documents"]

# Rerank documents
ranker = SIERanker(
    base_url="http://localhost:8080",
    model="jinaai/jina-reranker-v2-base-multilingual"
)
result = ranker.run(query="What is Python?", documents=embedded_docs, top_k=3)
ranked_docs = result["documents"]
```
