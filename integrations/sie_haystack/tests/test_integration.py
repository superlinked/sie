"""Integration tests for sie-haystack.

These tests require a running SIE server and serve as runnable examples.
Run with: pytest -m integration integrations/sie_haystack/tests/

Prerequisites:
    mise run serve -d cpu -p 8080
"""

from __future__ import annotations

import os

import pytest

# Skip all tests in this module if not running integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def sie_url() -> str:
    """Get SIE server URL from environment or default."""
    return os.environ.get("SIE_SERVER_URL", "http://localhost:8080")


class TestTextEmbedderIntegration:
    """Integration tests for SIETextEmbedder with real server."""

    def test_embed_text(self, sie_url: str) -> None:
        """Test embedding text with real SIE server."""
        from sie_haystack import SIETextEmbedder

        embedder = SIETextEmbedder(base_url=sie_url, model="BAAI/bge-m3")

        result = embedder.run(text="What is vector search?")

        assert "embedding" in result
        assert len(result["embedding"]) == 1024  # BGE-M3 dimension
        assert all(isinstance(x, float) for x in result["embedding"])

    def test_embed_multiple_texts(self, sie_url: str) -> None:
        """Test embedding multiple texts for similarity comparison."""
        from sie_haystack import SIETextEmbedder

        embedder = SIETextEmbedder(base_url=sie_url, model="BAAI/bge-m3")

        result1 = embedder.run(text="Machine learning models")
        result2 = embedder.run(text="Deep learning neural networks")
        result3 = embedder.run(text="The weather is sunny today")

        # All should be valid embeddings
        assert len(result1["embedding"]) == 1024
        assert len(result2["embedding"]) == 1024
        assert len(result3["embedding"]) == 1024


class TestDocumentEmbedderIntegration:
    """Integration tests for SIEDocumentEmbedder with real server."""

    def test_embed_documents(self, sie_url: str) -> None:
        """Test embedding documents with real SIE server."""
        from haystack import Document
        from sie_haystack import SIEDocumentEmbedder

        embedder = SIEDocumentEmbedder(base_url=sie_url, model="BAAI/bge-m3")

        docs = [
            Document(content="Python is a programming language."),
            Document(content="JavaScript runs in web browsers."),
        ]
        result = embedder.run(documents=docs)

        assert len(result["documents"]) == 2
        for doc in result["documents"]:
            assert doc.embedding is not None
            assert len(doc.embedding) == 1024

    def test_embed_with_metadata(self, sie_url: str) -> None:
        """Test embedding documents that include metadata fields."""
        from haystack import Document
        from sie_haystack import SIEDocumentEmbedder

        embedder = SIEDocumentEmbedder(
            base_url=sie_url,
            model="BAAI/bge-m3",
            meta_fields_to_embed=["title"],
        )

        docs = [
            Document(
                content="This article explains vector databases.",
                meta={"title": "Introduction to Vector Databases"},
            ),
        ]
        result = embedder.run(documents=docs)

        assert len(result["documents"]) == 1
        assert result["documents"][0].embedding is not None


class TestInMemoryDocumentStoreIntegration:
    """Integration test using Haystack's InMemoryDocumentStore.

    This serves as an example of using SIE embeddings with a document store.
    """

    def test_similarity_search_in_memory(self, sie_url: str) -> None:
        """Example: Semantic search with InMemoryDocumentStore."""
        from haystack import Document, Pipeline
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from sie_haystack import SIEDocumentEmbedder, SIETextEmbedder

        # 1. Create document store
        document_store = InMemoryDocumentStore()

        # 2. Create and embed documents
        doc_embedder = SIEDocumentEmbedder(base_url=sie_url, model="BAAI/bge-m3")
        documents = [
            Document(content="Python is a popular programming language."),
            Document(content="JavaScript runs in web browsers."),
            Document(content="Machine learning uses statistical models."),
            Document(content="Vector databases store embeddings efficiently."),
        ]
        result = doc_embedder.run(documents=documents)
        embedded_docs = result["documents"]

        # 3. Write to document store
        document_store.write_documents(embedded_docs)

        # 4. Create retrieval pipeline
        text_embedder = SIETextEmbedder(base_url=sie_url, model="BAAI/bge-m3")
        retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=2)

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("retriever", retriever)
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

        # 5. Run search
        result = pipeline.run({"text_embedder": {"text": "What programming languages are commonly used?"}})

        assert len(result["retriever"]["documents"]) == 2
        contents = [d.content for d in result["retriever"]["documents"]]
        assert any("Python" in c or "JavaScript" in c for c in contents)


class TestChromaIntegration:
    """Integration test using Chroma vector store.

    This verifies SIE embeddings work with real vector stores.
    """

    def test_chroma_similarity_search(self, sie_url: str) -> None:
        """Example: Using SIE embeddings with Chroma."""
        import chromadb
        from haystack import Document, Pipeline
        from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
        from haystack_integrations.document_stores.chroma import ChromaDocumentStore
        from sie_haystack import SIEDocumentEmbedder, SIETextEmbedder

        # 1. Set up Chroma (ephemeral for test)
        _ = chromadb.Client()  # Ensure ephemeral client
        document_store = ChromaDocumentStore(
            collection_name="test_sie_haystack",
            embedding_function=None,  # We provide embeddings via SIE
        )

        # 2. Create and embed documents
        doc_embedder = SIEDocumentEmbedder(base_url=sie_url, model="BAAI/bge-m3")
        documents = [
            Document(
                content="SIE provides fast GPU inference for embeddings.",
                meta={"source": "docs"},
            ),
            Document(
                content="Chroma is an open-source vector database.",
                meta={"source": "docs"},
            ),
            Document(
                content="RAG combines retrieval with generation.",
                meta={"source": "tutorial"},
            ),
        ]
        result = doc_embedder.run(documents=documents)

        # 3. Write to Chroma
        document_store.write_documents(result["documents"])

        # 4. Create retrieval pipeline
        text_embedder = SIETextEmbedder(base_url=sie_url, model="BAAI/bge-m3")
        retriever = ChromaEmbeddingRetriever(document_store=document_store, top_k=2)

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("retriever", retriever)
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

        # 5. Search
        result = pipeline.run({"text_embedder": {"text": "How do I generate embeddings quickly?"}})

        assert len(result["retriever"]["documents"]) == 2
        # Results should have content
        assert all(d.content for d in result["retriever"]["documents"])


class TestRankerIntegration:
    """Integration tests for SIERanker with real server."""

    def test_rerank_documents(self, sie_url: str) -> None:
        """Example: Reranking retrieval results."""
        from haystack import Document
        from sie_haystack import SIERanker

        # 1. Create ranker
        ranker = SIERanker(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
            top_k=3,
        )

        # 2. Simulate initial retrieval results (not in relevance order)
        documents = [
            Document(content="The weather is sunny today."),
            Document(content="Vector search finds similar embeddings."),
            Document(content="Reranking improves search result quality."),
            Document(content="Python is used for data science."),
            Document(content="Semantic search understands meaning."),
        ]

        # 3. Rerank based on query
        result = ranker.run(
            query="How does semantic search work?",
            documents=documents,
        )

        assert len(result["documents"]) == 3  # top_k=3
        # All should have scores
        assert all("score" in doc.meta for doc in result["documents"])
        # Scores should be in descending order
        scores = [doc.meta["score"] for doc in result["documents"]]
        assert scores == sorted(scores, reverse=True)


class TestRAGPipelineIntegration:
    """Full RAG pipeline example using SIE components.

    This demonstrates a complete retrieval-augmented generation setup
    with embeddings and reranking (without the LLM generation step).
    """

    def test_two_stage_retrieval_pipeline(self, sie_url: str) -> None:
        """Example: Two-stage retrieval with embedding + reranking."""
        from haystack import Document, Pipeline
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from sie_haystack import SIEDocumentEmbedder, SIERanker, SIETextEmbedder

        # === Stage 1: Index documents with SIE embeddings ===
        document_store = InMemoryDocumentStore()
        doc_embedder = SIEDocumentEmbedder(base_url=sie_url, model="BAAI/bge-m3")

        documents = [
            Document(
                content="SIE is a GPU inference server for search workloads.",
                meta={"title": "SIE Overview"},
            ),
            Document(
                content="The encode endpoint generates embeddings from text.",
                meta={"title": "Encoding API"},
            ),
            Document(
                content="The score endpoint reranks documents by relevance.",
                meta={"title": "Scoring API"},
            ),
            Document(
                content="BGE-M3 is a multilingual embedding model.",
                meta={"title": "Models"},
            ),
            Document(
                content="Python SDK provides SIEClient for API access.",
                meta={"title": "SDK Guide"},
            ),
        ]

        result = doc_embedder.run(documents=documents)
        document_store.write_documents(result["documents"])

        # === Stage 2: Create two-stage retrieval pipeline ===
        text_embedder = SIETextEmbedder(base_url=sie_url, model="BAAI/bge-m3")
        retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=4)
        ranker = SIERanker(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
            top_k=2,
        )

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("ranker", ranker)
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "ranker.documents")

        # === Stage 3: Run pipeline ===
        query = "How do I generate embeddings with SIE?"
        result = pipeline.run(
            {
                "text_embedder": {"text": query},
                "ranker": {"query": query},
            }
        )

        final_docs = result["ranker"]["documents"]
        assert len(final_docs) == 2
        # The encoding API doc should be highly ranked
        top_content = final_docs[0].content.lower()
        assert "encode" in top_content or "embedding" in top_content

    def test_rag_pipeline_with_metadata_filtering(self, sie_url: str) -> None:
        """Example: RAG pipeline with metadata filtering."""
        from haystack import Document, Pipeline
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
        from haystack.document_stores.in_memory import InMemoryDocumentStore
        from sie_haystack import SIEDocumentEmbedder, SIERanker, SIETextEmbedder

        # Set up document store with categorized documents
        document_store = InMemoryDocumentStore()
        doc_embedder = SIEDocumentEmbedder(base_url=sie_url, model="BAAI/bge-m3")

        documents = [
            Document(content="Python is great for machine learning.", meta={"category": "programming"}),
            Document(content="TensorFlow is a deep learning framework.", meta={"category": "ml"}),
            Document(content="Pandas handles data manipulation.", meta={"category": "data"}),
            Document(content="PyTorch is popular for research.", meta={"category": "ml"}),
        ]

        result = doc_embedder.run(documents=documents)
        document_store.write_documents(result["documents"])

        # Create pipeline
        text_embedder = SIETextEmbedder(base_url=sie_url, model="BAAI/bge-m3")
        retriever = InMemoryEmbeddingRetriever(
            document_store=document_store,
            top_k=3,
            filters={"field": "meta.category", "operator": "==", "value": "ml"},
        )
        ranker = SIERanker(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
            top_k=2,
        )

        pipeline = Pipeline()
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("ranker", ranker)
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever.documents", "ranker.documents")

        # Run with ML filter
        query = "What frameworks are used for deep learning?"
        result = pipeline.run(
            {
                "text_embedder": {"text": query},
                "ranker": {"query": query},
            }
        )

        # Should only get ML category documents
        final_docs = result["ranker"]["documents"]
        assert len(final_docs) == 2
        for doc in final_docs:
            assert doc.meta.get("category") == "ml"


class TestExtractorIntegration:
    """Integration tests for SIEExtractor with real server."""

    def test_extract_entities(self, sie_url: str) -> None:
        """Example: Extract entities from text."""
        from sie_haystack import SIEExtractor

        extractor = SIEExtractor(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
            labels=["person", "organization", "location"],
        )

        result = extractor.run(text="John Smith works at Google in New York.")

        assert "entities" in result
        assert isinstance(result["entities"], list)
        # Should find at least some entities
        assert len(result["entities"]) > 0

        # Check entity structure
        for entity in result["entities"]:
            assert entity.text
            assert entity.label in ["person", "organization", "location"]
            assert isinstance(entity.score, float)

    def test_extract_custom_entities(self, sie_url: str) -> None:
        """Example: Extract custom entity types."""
        from sie_haystack import SIEExtractor

        extractor = SIEExtractor(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
            labels=["programming language", "framework", "database"],
        )

        result = extractor.run(text="We built our application using Python, FastAPI, and PostgreSQL.")

        assert "entities" in result
        # Should extract programming language, framework, database mentions
