# sie-dspy

SIE integration for [DSPy](https://dspy.ai/).

## Installation

```bash
pip install sie-dspy
```

## Features

- **SIEEmbedder**: Embedding function for use with `dspy.Embedder` or `dspy.retrievers.Embeddings`
- **SIEReranker**: Module to rerank passages by relevance to a query
- **SIEExtractor**: Module to extract entities from text

## Quick Start

### Embeddings with FAISS Retriever

```python
import dspy
from sie_dspy import SIEEmbedder

# Create SIE embedder
embedder = SIEEmbedder(
    base_url="http://localhost:8080",
    model="BAAI/bge-m3",
)

# Use with DSPy's built-in FAISS retriever
corpus = [
    "Machine learning enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing analyzes human language.",
]

retriever = dspy.retrievers.Embeddings(
    corpus=corpus,
    embedder=embedder,
    k=2,
)

# Retrieve relevant passages
results = retriever("What is deep learning?")
print(results.passages)
```

### Reranking Retrieved Passages

```python
from sie_dspy import SIEReranker

# Create reranker module
reranker = SIEReranker(
    base_url="http://localhost:8080",
    model="jinaai/jina-reranker-v2-base-multilingual",
)

# Rerank passages
query = "How do neural networks learn?"
passages = [
    "The weather is sunny today.",
    "Neural networks learn through backpropagation.",
    "Deep learning models require large datasets.",
]

result = reranker(query=query, passages=passages, k=2)
print(result.passages)  # Top 2 most relevant passages
```

### Entity Extraction

```python
from sie_dspy import SIEExtractor

# Create extractor module
extractor = SIEExtractor(
    base_url="http://localhost:8080",
    model="urchade/gliner_multi-v2.1",
    labels=["person", "organization", "location"],
)

# Extract entities
text = "John Smith is the CEO of TechCorp in San Francisco."
result = extractor(text=text)
print(result.entities)  # [{"text": "John Smith", "label": "person", ...}, ...]
```

### RAG Pipeline with Reranking

```python
import dspy
from sie_dspy import SIEEmbedder, SIEReranker

class RAGWithReranking(dspy.Module):
    def __init__(self, corpus, embedder, reranker, k=5, rerank_k=3):
        super().__init__()
        self.retriever = dspy.retrievers.Embeddings(
            corpus=corpus,
            embedder=embedder,
            k=k,
        )
        self.reranker = reranker
        self.rerank_k = rerank_k
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # Retrieve initial candidates
        retrieved = self.retriever(question)

        # Rerank to get most relevant
        reranked = self.reranker(
            query=question,
            passages=retrieved.passages,
            k=self.rerank_k,
        )

        # Generate answer
        context = "\n".join(reranked.passages)
        return self.generate(context=context, question=question)

# Usage
embedder = SIEEmbedder(base_url="http://localhost:8080", model="BAAI/bge-m3")
reranker = SIEReranker(base_url="http://localhost:8080")

rag = RAGWithReranking(
    corpus=["...your documents..."],
    embedder=embedder,
    reranker=reranker,
)
result = rag("What is machine learning?")
```

## SIE Server

Start the SIE server before using this integration:

```bash
mise run serve -d cpu -p 8080
```

## Testing

```bash
# Unit tests (no server required)
pytest

# Integration tests (requires running server)
pytest -m integration
```
