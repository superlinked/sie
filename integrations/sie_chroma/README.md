# sie-chroma

SIE integration for [ChromaDB](https://www.trychroma.com/).

## Installation

```bash
pip install sie-chroma
```

## Features

- **SIEEmbeddingFunction**: Custom embedding function for ChromaDB collections

## Quick Start

### Basic Usage

```python
import chromadb
from sie_chroma import SIEEmbeddingFunction

# Create SIE embedding function
embedding_function = SIEEmbeddingFunction(
    base_url="http://localhost:8080",
    model="BAAI/bge-m3",
)

# Create ChromaDB client and collection
client = chromadb.Client()
collection = client.create_collection(
    name="my_collection",
    embedding_function=embedding_function,
)

# Add documents (embeddings are generated automatically)
collection.add(
    documents=[
        "Machine learning enables pattern recognition.",
        "Deep learning uses neural networks.",
        "Natural language processing analyzes text.",
    ],
    ids=["doc1", "doc2", "doc3"],
)

# Query the collection
results = collection.query(
    query_texts=["What is deep learning?"],
    n_results=2,
)
print(results["documents"])
```

### With Persistent Storage

```python
import chromadb
from sie_chroma import SIEEmbeddingFunction

# Persistent client
client = chromadb.PersistentClient(path="./chroma_data")

embedding_function = SIEEmbeddingFunction(
    base_url="http://localhost:8080",
    model="BAAI/bge-m3",
)

# Get or create collection
collection = client.get_or_create_collection(
    name="research_papers",
    embedding_function=embedding_function,
)

# Add documents with metadata
collection.add(
    documents=["Paper about transformers...", "Study on attention mechanisms..."],
    metadatas=[{"year": 2023}, {"year": 2024}],
    ids=["paper1", "paper2"],
)

# Query with metadata filtering
results = collection.query(
    query_texts=["attention in neural networks"],
    n_results=5,
    where={"year": {"$gte": 2023}},
)
```

### With LangChain or LlamaIndex

The SIEEmbeddingFunction works with ChromaDB's LangChain and LlamaIndex integrations:

```python
# LangChain
from langchain_chroma import Chroma
from sie_chroma import SIEEmbeddingFunction

embedding_function = SIEEmbeddingFunction(model="BAAI/bge-m3")
vectorstore = Chroma(
    collection_name="docs",
    embedding_function=embedding_function,  # Works directly!
)

# LlamaIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

# SIE can also be used via LlamaIndex's SIEEmbedding
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
