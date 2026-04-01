"""Integration tests for sie-llamaindex.

These tests require a running SIE server and serve as runnable examples.
Run with: pytest -m integration integrations/sie_llamaindex/tests/

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


class TestEmbeddingIntegration:
    """Integration tests for SIEEmbedding with real server."""

    def test_get_text_embedding(self, sie_url: str) -> None:
        """Test embedding text with real SIE server."""
        from sie_llamaindex import SIEEmbedding

        embed_model = SIEEmbedding(base_url=sie_url, model_name="BAAI/bge-m3")

        result = embed_model._get_text_embedding("Hello world")

        assert len(result) == 1024  # BGE-M3 dimension
        assert all(isinstance(x, float) for x in result)

    def test_get_text_embeddings_batch(self, sie_url: str) -> None:
        """Test batch embedding with real SIE server."""
        from sie_llamaindex import SIEEmbedding

        embed_model = SIEEmbedding(base_url=sie_url, model_name="BAAI/bge-m3")

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning models can understand natural language.",
        ]
        results = embed_model._get_text_embeddings(texts)

        assert len(results) == 2
        assert len(results[0]) == 1024

    def test_get_query_embedding(self, sie_url: str) -> None:
        """Test query embedding with real SIE server."""
        from sie_llamaindex import SIEEmbedding

        embed_model = SIEEmbedding(base_url=sie_url, model_name="BAAI/bge-m3")

        result = embed_model._get_query_embedding("What is vector search?")

        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_aget_text_embedding(self, sie_url: str) -> None:
        """Test async embedding with real SIE server."""
        from sie_llamaindex import SIEEmbedding

        embed_model = SIEEmbedding(base_url=sie_url, model_name="BAAI/bge-m3")

        result = await embed_model._aget_text_embedding("Async embedding test")

        assert len(result) == 1024


class TestVectorStoreIntegration:
    """Integration test using LlamaIndex's VectorStoreIndex.

    This serves as an example of using SIEEmbedding with a vector store.
    """

    def test_similarity_search_in_memory(self, sie_url: str) -> None:
        """Example: Semantic search with VectorStoreIndex (in-memory)."""
        from llama_index.core import Document, Settings, VectorStoreIndex
        from sie_llamaindex import SIEEmbedding

        # 1. Configure SIE as embedding model
        Settings.embed_model = SIEEmbedding(base_url=sie_url, model_name="BAAI/bge-m3")

        # 2. Create documents
        documents = [
            Document(text="Python is a popular programming language."),
            Document(text="JavaScript runs in web browsers."),
            Document(text="Machine learning uses statistical models."),
            Document(text="Vector databases store embeddings efficiently."),
        ]

        # 3. Create index (uses SIE for embeddings)
        index = VectorStoreIndex.from_documents(documents)

        # 4. Create retriever and search
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve("What programming languages are commonly used?")

        assert len(nodes) == 2
        # Python and JavaScript should be most relevant
        contents = [n.node.get_content() for n in nodes]
        assert any("Python" in c or "JavaScript" in c for c in contents)


class TestChromaIntegration:
    """Integration test using Chroma vector store.

    This verifies SIEEmbedding works with real vector stores.
    """

    def test_chroma_similarity_search(self, sie_url: str) -> None:
        """Example: Using SIEEmbedding with Chroma."""
        import chromadb
        from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from sie_llamaindex import SIEEmbedding

        # 1. Configure SIE as embedding model
        Settings.embed_model = SIEEmbedding(base_url=sie_url, model_name="BAAI/bge-m3")

        # 2. Set up Chroma (ephemeral for test)
        chroma_client = chromadb.Client()
        chroma_collection = chroma_client.create_collection("test_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 3. Create documents
        documents = [
            Document(
                text="SIE provides fast GPU inference for embeddings.",
                metadata={"source": "docs"},
            ),
            Document(
                text="Chroma is an open-source vector database.",
                metadata={"source": "docs"},
            ),
            Document(
                text="RAG combines retrieval with generation.",
                metadata={"source": "tutorial"},
            ),
        ]

        # 4. Create index with Chroma storage
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )

        # 5. Search
        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve("How do I generate embeddings quickly?")

        assert len(nodes) == 2
        # Results should have scores
        assert all(n.score is not None for n in nodes)


class TestRerankerIntegration:
    """Integration tests for SIENodePostprocessor with real server."""

    def test_rerank_nodes(self, sie_url: str) -> None:
        """Example: Reranking retrieval results."""
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
        from sie_llamaindex import SIENodePostprocessor

        # 1. Create reranker
        reranker = SIENodePostprocessor(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
            top_n=3,
        )

        # 2. Simulate initial retrieval results (not in relevance order)
        nodes = [
            NodeWithScore(node=TextNode(text="The weather is sunny today."), score=0.8),
            NodeWithScore(node=TextNode(text="Vector search finds similar embeddings."), score=0.7),
            NodeWithScore(node=TextNode(text="Reranking improves search result quality."), score=0.6),
            NodeWithScore(node=TextNode(text="Python is used for data science."), score=0.5),
            NodeWithScore(node=TextNode(text="Semantic search understands meaning."), score=0.4),
        ]

        # 3. Rerank based on query
        query = QueryBundle(query_str="How does semantic search work?")
        reranked = reranker._postprocess_nodes(nodes, query)

        assert len(reranked) == 3  # top_n=3
        # All should have scores
        assert all(n.score is not None for n in reranked)
        # Scores should be in descending order (reranker sorts by relevance)
        scores = [n.score for n in reranked]
        assert scores == sorted(scores, reverse=True)


class TestRAGPipelineIntegration:
    """Full RAG pipeline example using SIE components.

    This demonstrates a complete retrieval-augmented generation setup
    with embeddings and reranking (without the LLM generation step).
    """

    def test_rag_retrieval_pipeline(self, sie_url: str) -> None:
        """Example: Two-stage retrieval with embedding + reranking."""
        from llama_index.core import Document, Settings, VectorStoreIndex
        from sie_llamaindex import SIEEmbedding, SIENodePostprocessor

        # === Stage 1: Index documents with SIE embeddings ===
        Settings.embed_model = SIEEmbedding(base_url=sie_url, model_name="BAAI/bge-m3")

        documents = [
            Document(
                text="SIE is a GPU inference server for search workloads.",
                metadata={"title": "SIE Overview"},
            ),
            Document(
                text="The encode endpoint generates embeddings from text.",
                metadata={"title": "Encoding API"},
            ),
            Document(
                text="The score endpoint reranks documents by relevance.",
                metadata={"title": "Scoring API"},
            ),
            Document(
                text="BGE-M3 is a multilingual embedding model.",
                metadata={"title": "Models"},
            ),
            Document(
                text="Python SDK provides SIEClient for API access.",
                metadata={"title": "SDK Guide"},
            ),
        ]

        index = VectorStoreIndex.from_documents(documents)

        # === Stage 2: Retrieve candidates ===
        retriever = index.as_retriever(similarity_top_k=4)
        candidates = retriever.retrieve("How do I generate embeddings with SIE?")
        assert len(candidates) == 4

        # === Stage 3: Rerank for precision ===
        reranker = SIENodePostprocessor(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
            top_n=2,
        )
        from llama_index.core.schema import QueryBundle

        final_results = reranker._postprocess_nodes(
            candidates,
            QueryBundle(query_str="How do I generate embeddings with SIE?"),
        )

        assert len(final_results) == 2
        # The encoding API doc should be highly ranked
        top_content = final_results[0].node.get_content()
        assert "encode" in top_content.lower() or "embedding" in top_content.lower()

    def test_query_engine_with_reranking(self, sie_url: str) -> None:
        """Example: Using reranker in query engine pipeline."""
        from llama_index.core import Document, Settings, VectorStoreIndex
        from sie_llamaindex import SIEEmbedding, SIENodePostprocessor

        # Configure embeddings
        Settings.embed_model = SIEEmbedding(base_url=sie_url, model_name="BAAI/bge-m3")

        # Create simple index
        documents = [
            Document(text="Vector search is efficient for semantic retrieval."),
            Document(text="Reranking improves retrieval precision."),
            Document(text="The weather today is sunny."),
        ]
        index = VectorStoreIndex.from_documents(documents)

        # Create reranker
        reranker = SIENodePostprocessor(
            base_url=sie_url,
            model="jinaai/jina-reranker-v2-base-multilingual",
            top_n=2,
        )

        # Get retriever with reranking
        retriever = index.as_retriever(
            similarity_top_k=3,
            node_postprocessors=[reranker],
        )

        # This would be used in a full query engine with LLM
        # For now, just test retrieval works
        nodes = retriever.retrieve("How does search work?")
        assert len(nodes) == 2  # Limited by top_n


class TestExtractorIntegration:
    """Integration tests for SIE extractor tool with real server."""

    def test_extract_entities(self, sie_url: str) -> None:
        """Example: Extract entities from text."""
        from sie_llamaindex.extractors import _SIEExtractor

        # Create extractor directly for testing
        extractor = _SIEExtractor(
            base_url=sie_url,
            model="urchade/gliner_multi-v2.1",
            labels=["person", "organization", "location"],
            options=None,
            gpu=None,
            timeout_s=180.0,
        )

        result = extractor.extract("John Smith works at Google in New York.")

        assert isinstance(result, list)
        # Should find at least some entities
        assert len(result) > 0

        # Check entity structure
        for entity in result:
            assert "text" in entity
            assert "label" in entity
            assert "score" in entity
            assert entity["label"] in ["person", "organization", "location"]
