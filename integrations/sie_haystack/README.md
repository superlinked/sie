# sie-haystack

SIE integration for Haystack.

## Installation

```bash
pip install sie-haystack
```

## Imports

Preferred import paths follow Haystack's namespace convention:

```python
from haystack_integrations.components.embedders.sie import (
    SIEDocumentEmbedder,
    SIETextEmbedder,
)
from haystack_integrations.components.rankers.sie import SIERanker
from haystack_integrations.components.extractors.sie import SIEExtractor
```

The legacy flat imports remain supported for compatibility:

```python
from sie_haystack import SIEDocumentEmbedder, SIEExtractor, SIERanker, SIETextEmbedder
```

## Usage

```python
from haystack import Document
from haystack_integrations.components.embedders.sie import SIEDocumentEmbedder, SIETextEmbedder
from haystack_integrations.components.rankers.sie import SIERanker

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

Relation models such as GLiREL classify relations between supplied entity
spans. Extract those spans first, then pass them through the relation
extractor's `entities` input:

```python
from haystack_integrations.components.extractors.sie import SIEExtractor

text = "Tim Cook is the CEO of Apple Inc."
entity_extractor = SIEExtractor(
    model="urchade/gliner_multi-v2.1",
    labels=["person", "organization"],
)
relation_extractor = SIEExtractor(
    model="jackboyla/glirel-large-v0",
    labels=["ceo_of"],
)

entities = entity_extractor.run(text=text)["entities"]
result = relation_extractor.run(text=text, entities=entities)
```
