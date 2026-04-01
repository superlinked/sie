# SIE Framework Integrations

This directory contains framework integrations for the Search Inference Engine (SIE).
Each integration is a separate PyPI package that wraps `SIEClient` to implement
standard framework interfaces.

## Available Integrations

| Package | Framework | Install | Status |
|---------|-----------|---------|--------|
| `sie-chroma` | Chroma | `pip install sie-chroma` | Ready |
| `sie-crewai` | CrewAI | `pip install sie-crewai` | Ready |
| `sie-dspy` | DSPy | `pip install sie-dspy` | Ready |
| `sie-haystack` | Haystack | `pip install sie-haystack` | Ready |
| `sie-langchain` | LangChain | `pip install sie-langchain` | Ready |
| `sie-llamaindex` | LlamaIndex | `pip install sie-llamaindex` | Ready |
| `sie-qdrant` | Qdrant | `pip install sie-qdrant` | Ready |
| `sie-weaviate` | Weaviate | `pip install sie-weaviate` | Ready |

## Quick Start

```python
# LangChain example
from sie_langchain import SIEEmbeddings

embeddings = SIEEmbeddings(
    base_url="http://localhost:8080",
    model="BAAI/bge-m3"
)

# Use with any LangChain vector store
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents, embeddings)
```

## Directory Structure

```
integrations/
├── README.md                    # This file
├── conftest.py                  # Shared pytest fixtures
├── sie_langchain/               # LangChain integration
│   ├── pyproject.toml
│   ├── src/sie_langchain/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # SIEEmbeddings
│   │   ├── rerankers.py         # SIEReranker
│   │   └── extractors.py        # SIEExtractor
│   └── tests/
├── sie_llamaindex/              # LlamaIndex integration
│   └── ...
└── ...
```

## Development

### Prerequisites

```bash
# Install all packages including integrations
uv sync --all-packages

# Or install a specific integration for development
cd integrations/sie_langchain
uv sync
```

### Testing Strategy

We use a layered testing approach:

| Layer | Responsibility | What We Test |
|-------|----------------|--------------|
| **Unit tests** | Mock `SIEClient` | Framework interface compliance |
| **Integration tests** | Real SIE server | End-to-end functionality |

**Unit tests** (run on every PR):

```bash
# Run all unit tests (excludes integration by default)
mise run test

# Run tests for a specific integration
mise run test integrations/sie_langchain/tests/
```

**Integration tests** (require running server):

```bash
# Start SIE server
mise run serve -d cpu -p 8080

# Run all integration tests
mise run test -i

# Run integration tests for specific package
mise run test -i integrations/sie_langchain/
```

### Creating a New Integration

1. Create the package directory:

   ```bash
   mkdir -p integrations/sie_myframework/src/sie_myframework
   mkdir -p integrations/sie_myframework/tests
   ```

2. Create `pyproject.toml`:

   ```toml
   [project]
   name = "sie-myframework"
   version = "0.1.0"
   description = "SIE integration for MyFramework"
   dependencies = [
       "sie-sdk>=0.1.0",
       "myframework-core>=1.0",
   ]

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"

   [tool.hatch.build.targets.wheel]
   packages = ["src/sie_myframework"]
   ```

3. Implement the framework interface in `src/sie_myframework/`

4. Add tests using the shared fixtures from `conftest.py`

### Shared Fixtures

The `conftest.py` file provides common fixtures for all integrations:

- `mock_sie_client` - Mocked `SIEClient` that returns test embeddings
- `mock_sie_async_client` - Mocked async client
- `sie_server_url` - URL of running SIE server (for integration tests)

Example usage:

```python
def test_embeddings(mock_sie_client):
    embeddings = SIEEmbeddings(client=mock_sie_client, model="test-model")
    result = embeddings.embed_query("Hello")
    assert len(result) == 384  # Mock returns 384-dim embeddings
```

## Framework Primitives Mapping

Each SIE primitive maps to framework-specific interfaces:

| SIE Primitive | LangChain | LlamaIndex | Haystack | CrewAI | DSPy | Chroma |
|---------------|-----------|------------|----------|--------|------|--------|
| `encode()` | `Embeddings` | `BaseEmbedding` | `SIETextEmbedder`, `SIEDocumentEmbedder` | OpenAI-compatible API | `SIEEmbedder` | `EmbeddingFunction` |
| `score()` | `BaseDocumentCompressor` | `BaseNodePostprocessor` | `SIERanker` | `SIERerankerTool` | `SIEReranker` | N/A |
| `extract()` | `BaseTool` | `FunctionTool` | `SIEExtractor` | `SIEExtractorTool` | `SIEExtractor` | N/A |

## When to Use Integrations vs SDK Directly

Framework integrations implement **callback protocols** - the framework calls SIE automatically during its pipeline. This is valuable when frameworks have standardized interfaces.

| Use Case | Recommendation | Why |
|----------|---------------|-----|
| **Dense embeddings** | Framework integrations | Frameworks have standard interfaces (`Embeddings`, `BaseEmbedding`, etc.) |
| **Sparse/hybrid search** | Framework integrations | Most frameworks support sparse via `SIESparseEncoder`/`SIESparseEmbedder` |
| **Reranking** | Framework integrations | All frameworks have reranker interfaces (works with ColBERT models too!) |
| **Entity extraction** | Framework integrations | All frameworks have tool/component interfaces |
| **Multivector/ColBERT retrieval** | SDK directly | No framework has native multi-vector interfaces. Use SDK + vector DB (Qdrant, Weaviate, Vespa) |
| **Multimodal (CLIP, ColPali)** | SDK directly | No framework has standardized image embedding interfaces |

### Sparse Embeddings

All integrations support sparse embeddings for hybrid search:

```python
# LangChain - works with PineconeHybridSearchRetriever
from sie_langchain import SIESparseEncoder
sparse_encoder = SIESparseEncoder(model="BAAI/bge-m3")

# LlamaIndex
from sie_llamaindex import SIESparseEmbeddingFunction
sparse_fn = SIESparseEmbeddingFunction(model="BAAI/bge-m3")

# Haystack
from sie_haystack import SIESparseTextEmbedder
sparse_embedder = SIESparseTextEmbedder(model="BAAI/bge-m3")

# DSPy, CrewAI
from sie_dspy import SIESparseEmbedder  # or sie_crewai
sparse_embedder = SIESparseEmbedder(model="BAAI/bge-m3")

# Chroma (Cloud only)
from sie_chroma import SIESparseEmbeddingFunction
sparse_fn = SIESparseEmbeddingFunction(model="BAAI/bge-m3")
```

### Multivector/ColBERT (SDK Directly)

For ColBERT-style late interaction retrieval, use the SDK with a vector DB that supports multi-vector:

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, MultiVectorConfig, MultiVectorComparator

sie = SIEClient("http://localhost:8080")
qdrant = QdrantClient("http://localhost:6333")

# Create collection with multi-vector support
qdrant.create_collection(
    collection_name="docs",
    vectors_config={
        "colbert": VectorParams(
            size=128,
            distance=Distance.COSINE,
            multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM)
        )
    }
)

# Index documents
doc_results = sie.encode(
    "jinaai/jina-colbert-v2",
    [Item(text=doc) for doc in documents],
    output_types=["multivector"],
)
qdrant.upsert("docs", points=[
    {"id": i, "vector": {"colbert": r["multivector"].tolist()}}
    for i, r in enumerate(doc_results)
])

# Query with MaxSim
query_result = sie.encode(
    "jinaai/jina-colbert-v2",
    Item(text="What is machine learning?"),
    output_types=["multivector"],
    options={"is_query": True},
)
results = qdrant.query_points("docs", query_vector=("colbert", query_result["multivector"].tolist()), limit=10)
```

**Note:** For ColBERT **reranking** (not retrieval), use the framework rerankers with a ColBERT model name - the `score()` API handles encoding + MaxSim internally:

```python
# LangChain - ColBERT reranking just works
reranker = SIEReranker(model="jinaai/jina-colbert-v2")
reranked = reranker.compress_documents(documents, query)
```

### Multimodal/Images (SDK Directly)

For CLIP, SigLIP, ColPali image embeddings:

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item
from PIL import Image

client = SIEClient("http://localhost:8080")

# Text-to-image search
text_emb = client.encode("openai/clip-vit-large-patch14", "a cat on a windowsill")

# Image embedding
image = Image.open("photo.jpg")
image_emb = client.encode("openai/clip-vit-large-patch14", Item(image=image))

# Visual document retrieval with ColPali
page_image = Image.open("document_page.png")
page_emb = client.encode("vidore/colpali-v1.2", Item(image=page_image), output_types=["multivector"])
```

**Supported vector databases for multi-vector:**

- [Qdrant](https://qdrant.tech/documentation/concepts/vectors/) (v1.10+) - Native MaxSim via `MultiVectorConfig`
- [Weaviate](https://weaviate.io/developers/weaviate/config-refs/schema/multi-vector) (v1.29+) - Multi-vector embeddings GA
- [Vespa](https://docs.vespa.ai/) - Native late interaction support

## Release Process

Each integration is versioned and released independently:

1. Update version in `pyproject.toml`
2. Create release commit
3. Tag with `sie-myframework-vX.Y.Z`
4. GitHub Actions publishes to PyPI

## Contributing

See the main repository's CONTRIBUTING.md for general guidelines.

For integration-specific contributions:

1. Follow the framework's conventions and best practices
2. Ensure compatibility with the framework's latest stable version
3. Add both unit and integration tests
4. Update this README if adding a new integration
