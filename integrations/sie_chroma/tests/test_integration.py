"""Integration tests for sie-chroma.

These tests require a running SIE server and serve as runnable examples
of ChromaDB workflows using SIE embeddings.

Run with: pytest -m integration integrations/sie_chroma/tests/

Prerequisites:
    mise run serve -d cpu -p 8080

ChromaDB Use Cases Demonstrated:
- Semantic search over document collections
- RAG pipeline document retrieval
- Similarity-based recommendation
- Research paper search
"""

from __future__ import annotations

import os

import chromadb
import pytest

# Skip all tests in this module if not running integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def sie_url() -> str:
    """Get SIE server URL from environment or default."""
    return os.environ.get("SIE_SERVER_URL", "http://localhost:8080")


@pytest.fixture
def chroma_client() -> chromadb.Client:
    """Create an in-memory ChromaDB client for testing."""
    return chromadb.Client()


class TestSemanticSearch:
    """Integration tests demonstrating semantic search use case.

    ChromaDB's primary use case: Store documents as embeddings and
    retrieve the most semantically similar ones to a query.
    """

    def test_basic_semantic_search(self, sie_url: str, chroma_client: chromadb.Client) -> None:
        """Example: Basic semantic search over documents."""
        from sie_chroma import SIEEmbeddingFunction

        embedding_function = SIEEmbeddingFunction(
            base_url=sie_url,
            model="BAAI/bge-m3",
        )

        collection = chroma_client.create_collection(
            name="test_semantic",
            embedding_function=embedding_function,
        )

        # Add documents about different topics
        collection.add(
            documents=[
                "Machine learning algorithms learn patterns from data.",
                "The weather forecast predicts rain tomorrow.",
                "Neural networks are inspired by biological neurons.",
                "Stock prices fluctuated today.",
                "Deep learning is a subset of machine learning.",
            ],
            ids=["ml1", "weather1", "nn1", "stocks1", "dl1"],
        )

        # Query for ML-related documents
        results = collection.query(
            query_texts=["How do neural networks work?"],
            n_results=2,
        )

        assert len(results["documents"][0]) == 2
        # Results should be ML/neural network related

    def test_search_with_metadata_filter(self, sie_url: str, chroma_client: chromadb.Client) -> None:
        """Example: Semantic search with metadata filtering."""
        from sie_chroma import SIEEmbeddingFunction

        embedding_function = SIEEmbeddingFunction(
            base_url=sie_url,
            model="BAAI/bge-m3",
        )

        collection = chroma_client.create_collection(
            name="test_metadata",
            embedding_function=embedding_function,
        )

        # Add documents with metadata
        collection.add(
            documents=[
                "Introduction to machine learning concepts.",
                "Advanced deep learning architectures.",
                "Beginner's guide to Python programming.",
                "Expert-level neural network optimization.",
            ],
            metadatas=[
                {"level": "beginner", "topic": "ml"},
                {"level": "advanced", "topic": "ml"},
                {"level": "beginner", "topic": "programming"},
                {"level": "expert", "topic": "ml"},
            ],
            ids=["doc1", "doc2", "doc3", "doc4"],
        )

        # Query with metadata filter
        results = collection.query(
            query_texts=["learning about neural networks"],
            n_results=3,
            where={"level": {"$ne": "beginner"}},  # Exclude beginner content
        )

        assert len(results["documents"][0]) <= 3
        # Should only get advanced/expert level documents


class TestRAGPipeline:
    """Integration tests demonstrating RAG pipeline use case.

    ChromaDB is commonly used as the retrieval component in
    Retrieval-Augmented Generation (RAG) systems.
    """

    def test_rag_document_retrieval(self, sie_url: str, chroma_client: chromadb.Client) -> None:
        """Example: Document retrieval for RAG pipeline."""
        from sie_chroma import SIEEmbeddingFunction

        embedding_function = SIEEmbeddingFunction(
            base_url=sie_url,
            model="BAAI/bge-m3",
        )

        collection = chroma_client.create_collection(
            name="test_rag",
            embedding_function=embedding_function,
        )

        # Knowledge base documents
        knowledge_base = [
            "Python was created by Guido van Rossum in 1991.",
            "JavaScript was developed by Brendan Eich at Netscape.",
            "The Rust programming language was started by Mozilla.",
            "Go was designed at Google by Robert Griesemer and others.",
            "TypeScript is a superset of JavaScript developed by Microsoft.",
        ]

        collection.add(
            documents=knowledge_base,
            ids=[f"fact_{i}" for i in range(len(knowledge_base))],
        )

        # Retrieve context for a question
        question = "Who created Python?"
        results = collection.query(
            query_texts=[question],
            n_results=3,
        )

        # The retrieved documents would be passed to an LLM
        context = results["documents"][0]
        assert len(context) == 3


class TestResearchPaperSearch:
    """Integration tests demonstrating research paper search use case.

    Academic use case: Search through research papers or technical
    documentation using semantic similarity.
    """

    def test_paper_search_by_topic(self, sie_url: str, chroma_client: chromadb.Client) -> None:
        """Example: Search research papers by topic."""
        from sie_chroma import SIEEmbeddingFunction

        embedding_function = SIEEmbeddingFunction(
            base_url=sie_url,
            model="BAAI/bge-m3",
        )

        collection = chroma_client.create_collection(
            name="test_papers",
            embedding_function=embedding_function,
        )

        # Paper abstracts
        papers = [
            {
                "abstract": "We introduce BERT, a bidirectional transformer model for NLP.",
                "year": 2018,
                "title": "BERT",
            },
            {
                "abstract": "Attention mechanisms enable models to focus on relevant inputs.",
                "year": 2017,
                "title": "Attention is All You Need",
            },
            {
                "abstract": "GPT-3 demonstrates few-shot learning capabilities.",
                "year": 2020,
                "title": "GPT-3",
            },
            {
                "abstract": "We study the scaling laws of language models.",
                "year": 2020,
                "title": "Scaling Laws",
            },
        ]

        collection.add(
            documents=[p["abstract"] for p in papers],
            metadatas=[{"year": p["year"], "title": p["title"]} for p in papers],
            ids=[f"paper_{i}" for i in range(len(papers))],
        )

        # Find papers about attention mechanisms
        results = collection.query(
            query_texts=["attention mechanisms in neural networks"],
            n_results=2,
        )

        assert len(results["documents"][0]) == 2


class TestCollectionManagement:
    """Integration tests demonstrating collection management.

    Shows how to manage collections with different embedding functions.
    """

    def test_multiple_collections(self, sie_url: str, chroma_client: chromadb.Client) -> None:
        """Example: Managing multiple collections with different models."""
        from sie_chroma import SIEEmbeddingFunction

        # Create different embedding functions for different purposes
        general_embedder = SIEEmbeddingFunction(
            base_url=sie_url,
            model="BAAI/bge-m3",
        )

        # Create collections for different document types
        docs_collection = chroma_client.create_collection(
            name="documentation",
            embedding_function=general_embedder,
        )

        code_collection = chroma_client.create_collection(
            name="code_snippets",
            embedding_function=general_embedder,
        )

        # Add different types of content
        docs_collection.add(
            documents=["How to install the package", "API reference guide"],
            ids=["install", "api"],
        )

        code_collection.add(
            documents=["def hello(): print('Hello')", "class MyClass: pass"],
            ids=["func", "class"],
        )

        # Query each collection
        doc_results = docs_collection.query(
            query_texts=["installation instructions"],
            n_results=1,
        )

        code_results = code_collection.query(
            query_texts=["function definition"],
            n_results=1,
        )

        assert len(doc_results["documents"][0]) == 1
        assert len(code_results["documents"][0]) == 1
