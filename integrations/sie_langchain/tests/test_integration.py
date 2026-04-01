"""Integration tests for sie-langchain.

These tests require a running SIE server and serve as runnable examples.
Run with: pytest -m integration integrations/sie_langchain/tests/

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


class TestEmbeddingsIntegration:
    """Integration tests for SIEEmbeddings with real server."""

    def test_embed_documents(self, sie_url: str) -> None:
        """Test embedding documents with real SIE server."""
        from sie_langchain import SIEEmbeddings

        embeddings = SIEEmbeddings(base_url=sie_url, model="BAAI/bge-m3")

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can understand natural language.",
        ]
        result = embeddings.embed_documents(texts)

        assert len(result) == 2
        assert len(result[0]) == 1024  # BGE-M3 dimension
        assert all(isinstance(x, float) for x in result[0])

    def test_embed_query(self, sie_url: str) -> None:
        """Test embedding a query with real SIE server."""
        from sie_langchain import SIEEmbeddings

        embeddings = SIEEmbeddings(base_url=sie_url, model="BAAI/bge-m3")

        result = embeddings.embed_query("What is vector search?")

        assert len(result) == 1024
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.asyncio
    async def test_aembed_documents(self, sie_url: str) -> None:
        """Test async embedding with real SIE server."""
        from sie_langchain import SIEEmbeddings

        embeddings = SIEEmbeddings(base_url=sie_url, model="BAAI/bge-m3")

        texts = ["Hello world", "Async embedding test"]
        result = await embeddings.aembed_documents(texts)

        assert len(result) == 2
        assert len(result[0]) == 1024


class TestInMemoryVectorStoreIntegration:
    """Integration test using LangChain's InMemoryVectorStore.

    This serves as an example of using SIEEmbeddings with a vector store.
    """

    def test_similarity_search(self, sie_url: str) -> None:
        """Example: Semantic search with InMemoryVectorStore."""
        from langchain_core.documents import Document
        from langchain_core.vectorstores import InMemoryVectorStore
        from sie_langchain import SIEEmbeddings

        # 1. Create embeddings
        embeddings = SIEEmbeddings(base_url=sie_url, model="BAAI/bge-m3")

        # 2. Create documents
        documents = [
            Document(page_content="Python is a popular programming language."),
            Document(page_content="JavaScript runs in web browsers."),
            Document(page_content="Machine learning uses statistical models."),
            Document(page_content="Vector databases store embeddings efficiently."),
        ]

        # 3. Create vector store and add documents
        vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)

        # 4. Search for similar documents
        query = "What programming languages are commonly used?"
        results = vectorstore.similarity_search(query, k=2)

        assert len(results) == 2
        # Python and JavaScript should be most relevant
        contents = [r.page_content for r in results]
        assert any("Python" in c or "JavaScript" in c for c in contents)


class TestChromaIntegration:
    """Integration test using Chroma vector store (in-process).

    This verifies SIEEmbeddings works with real vector stores.
    """

    def test_chroma_similarity_search(self, sie_url: str) -> None:
        """Example: Using SIEEmbeddings with Chroma."""
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from sie_langchain import SIEEmbeddings

        # 1. Create embeddings
        embeddings = SIEEmbeddings(base_url=sie_url, model="BAAI/bge-m3")

        # 2. Create documents
        documents = [
            Document(
                page_content="SIE provides fast GPU inference for embeddings.",
                metadata={"source": "docs"},
            ),
            Document(
                page_content="Chroma is an open-source vector database.",
                metadata={"source": "docs"},
            ),
            Document(
                page_content="RAG combines retrieval with generation.",
                metadata={"source": "tutorial"},
            ),
        ]

        # 3. Create Chroma with in-memory storage
        vectorstore = Chroma.from_documents(
            documents,
            embeddings,
            collection_name="test_collection",
        )

        try:
            # 4. Search
            results = vectorstore.similarity_search_with_score("How do I generate embeddings quickly?", k=2)

            assert len(results) == 2
            # Results are (Document, score) tuples
            assert all(isinstance(r[1], float) for r in results)

            # 5. Search with metadata filter
            filtered = vectorstore.similarity_search("documentation", k=2, filter={"source": "docs"})
            assert all(d.metadata.get("source") == "docs" for d in filtered)

        finally:
            # Cleanup
            vectorstore.delete_collection()


class TestRerankerIntegration:
    """Integration tests for SIEReranker with real server."""

    def test_rerank_documents(self, sie_url: str) -> None:
        """Example: Reranking search results."""
        from langchain_core.documents import Document
        from sie_langchain import SIEReranker

        # 1. Create reranker
        reranker = SIEReranker(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
            top_k=3,
        )

        # 2. Simulate initial retrieval results (not in relevance order)
        documents = [
            Document(page_content="The weather is sunny today."),
            Document(page_content="Vector search finds similar embeddings."),
            Document(page_content="Reranking improves search result quality."),
            Document(page_content="Python is used for data science."),
            Document(page_content="Semantic search understands meaning."),
        ]

        # 3. Rerank based on query
        query = "How does semantic search work?"
        reranked = reranker.compress_documents(documents, query)

        assert len(reranked) == 3  # top_k=3
        # All should have relevance scores
        assert all("relevance_score" in d.metadata for d in reranked)
        # Scores should be in descending order
        scores = [d.metadata["relevance_score"] for d in reranked]
        assert scores == sorted(scores, reverse=True)


class TestRAGPipelineIntegration:
    """Full RAG pipeline example using SIE components.

    This demonstrates a complete retrieval-augmented generation setup
    with embeddings and reranking (without the LLM generation step).
    """

    def test_rag_retrieval_pipeline(self, sie_url: str) -> None:
        """Example: Two-stage retrieval with embedding + reranking."""
        from langchain_core.documents import Document
        from langchain_core.vectorstores import InMemoryVectorStore
        from sie_langchain import SIEEmbeddings, SIEReranker

        # === Stage 1: Index documents ===
        embeddings = SIEEmbeddings(base_url=sie_url, model="BAAI/bge-m3")

        documents = [
            Document(
                page_content="SIE is a GPU inference server for search workloads.",
                metadata={"title": "SIE Overview"},
            ),
            Document(
                page_content="The encode endpoint generates embeddings from text.",
                metadata={"title": "Encoding API"},
            ),
            Document(
                page_content="The score endpoint reranks documents by relevance.",
                metadata={"title": "Scoring API"},
            ),
            Document(
                page_content="BGE-M3 is a multilingual embedding model.",
                metadata={"title": "Models"},
            ),
            Document(
                page_content="Python SDK provides SIEClient for API access.",
                metadata={"title": "SDK Guide"},
            ),
        ]

        vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)

        # === Stage 2: Retrieve candidates ===
        query = "How do I generate embeddings with SIE?"
        candidates = vectorstore.similarity_search(query, k=4)
        assert len(candidates) == 4

        # === Stage 3: Rerank for precision ===
        reranker = SIEReranker(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
            top_k=2,
        )
        final_results = reranker.compress_documents(candidates, query)

        assert len(final_results) == 2
        # The encoding API doc should be highly ranked
        top_content = final_results[0].page_content
        assert "encode" in top_content.lower() or "embedding" in top_content.lower()
