"""Unit tests for SIE CrewAI tools and embedders."""

from __future__ import annotations

from sie_crewai import SIEExtractorTool, SIERerankerTool, SIESparseEmbedder


class TestSIERerankerTool:
    """Tests for SIERerankerTool.

    CrewAI use case: Research agents that need to find the most relevant
    sources from a collection of documents.
    """

    def test_rerank_research_documents(self, mock_sie_client: object, research_documents: list[str]) -> None:
        """Test reranking documents for research agent."""
        reranker = SIERerankerTool(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker._run(
            query="How does machine learning work?",
            documents=research_documents,
            top_k=3,
        )

        assert "Ranked documents" in result
        assert "Score:" in result
        # Should only have 3 results due to top_k
        assert result.count("[Score:") == 3

    def test_rerank_empty_documents(self, mock_sie_client: object) -> None:
        """Test reranking with no documents."""
        reranker = SIERerankerTool(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker._run(query="test query", documents=[])

        assert "No documents provided" in result

    def test_rerank_no_top_k(self, mock_sie_client: object, research_documents: list[str]) -> None:
        """Test reranking without top_k limit."""
        reranker = SIERerankerTool(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker._run(
            query="What is deep learning?",
            documents=research_documents,
        )

        # Should return all documents
        assert result.count("[Score:") == len(research_documents)

    def test_custom_model(self, mock_sie_client: object, research_documents: list[str]) -> None:
        """Test using a custom reranker model."""
        reranker = SIERerankerTool(model="custom/reranker-model")
        reranker._client = mock_sie_client

        reranker._run(query="test", documents=research_documents)

        call_args = mock_sie_client.score.call_args
        assert call_args[0][0] == "custom/reranker-model"

    def test_tool_metadata(self) -> None:
        """Test tool name and description for agent discovery."""
        reranker = SIERerankerTool()

        assert reranker.name == "sie_reranker"
        assert "rerank" in reranker.description.lower()
        assert "relevance" in reranker.description.lower()


class TestSIEExtractorTool:
    """Tests for SIEExtractorTool.

    CrewAI use case: Lead qualification agents that extract company,
    person, and other entity information from text.
    """

    def test_extract_lead_info(self, mock_sie_client: object, lead_info_text: str) -> None:
        """Test extracting entities for lead qualification."""
        extractor = SIEExtractorTool(
            model="test-extractor",
            labels=["person", "organization", "location"],
        )
        extractor._client = mock_sie_client

        result = extractor._run(text=lead_info_text)

        assert "Extracted entities" in result or "No entities found" in result

    def test_extract_with_custom_labels(self, mock_sie_client: object, lead_info_text: str) -> None:
        """Test extraction with custom labels for business use case."""
        extractor = SIEExtractorTool(model="test-extractor")
        extractor._client = mock_sie_client

        # Business-specific labels for lead scoring
        extractor._run(
            text=lead_info_text,
            labels=["company", "job_title", "funding_amount"],
        )

        # Check that custom labels were passed
        call_kwargs = mock_sie_client.extract.call_args.kwargs
        assert call_kwargs.get("labels") == ["company", "job_title", "funding_amount"]

    def test_extract_empty_result(self) -> None:
        """Test extraction with no entities found."""
        from unittest.mock import MagicMock

        extractor = SIEExtractorTool(
            model="test-extractor",
            labels=["very_specific_label"],
        )
        # Create a fresh mock that returns empty
        empty_mock = MagicMock()
        empty_mock.extract.return_value = []
        extractor._client = empty_mock

        result = extractor._run(text="Simple text with no entities.")

        assert "No entities found" in result

    def test_custom_model(self, mock_sie_client: object, lead_info_text: str) -> None:
        """Test using a custom extraction model."""
        extractor = SIEExtractorTool(model="custom/extraction-model")
        extractor._client = mock_sie_client

        extractor._run(text=lead_info_text)

        call_args = mock_sie_client.extract.call_args
        assert call_args[0][0] == "custom/extraction-model"

    def test_tool_metadata(self) -> None:
        """Test tool name and description for agent discovery."""
        extractor = SIEExtractorTool()

        assert extractor.name == "sie_extractor"
        assert "extract" in extractor.description.lower()
        assert "entities" in extractor.description.lower()

    def test_default_labels(self) -> None:
        """Test that default labels are set."""
        extractor = SIEExtractorTool()

        assert "person" in extractor.labels
        assert "organization" in extractor.labels
        assert "location" in extractor.labels


class TestSIESparseEmbedder:
    """Tests for SIESparseEmbedder.

    Use alongside SIE's OpenAI-compatible API (for dense) in hybrid search workflows.
    """

    def test_embed_documents_single(self, mock_sie_client: object) -> None:
        """Test embedding a single document."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.embed_documents(["Hello world"])

        assert len(result) == 1
        assert "indices" in result[0]
        assert "values" in result[0]
        assert len(result[0]["indices"]) == len(result[0]["values"])
        assert len(result[0]["indices"]) > 0

    def test_embed_documents_batch(self, mock_sie_client: object, research_documents: list[str]) -> None:
        """Test embedding multiple documents."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.embed_documents(research_documents)

        assert len(result) == len(research_documents)
        for sparse in result:
            assert "indices" in sparse
            assert "values" in sparse

    def test_embed_documents_empty(self, mock_sie_client: object) -> None:
        """Test embedding empty list returns empty."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.embed_documents([])

        assert result == []

    def test_embed_query(self, mock_sie_client: object) -> None:
        """Test embedding a query."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.embed_query("What is machine learning?")

        assert "indices" in result
        assert "values" in result
        assert len(result["indices"]) == len(result["values"])

    def test_embed_query_uses_is_query(self, mock_sie_client: object) -> None:
        """Test that embed_query sets is_query=True."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        embedder.embed_query("test query")

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options", {}).get("is_query") is True
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_embed_documents_no_is_query(self, mock_sie_client: object) -> None:
        """Test that embed_documents doesn't set is_query."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        embedder.embed_documents(["test doc"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options") is None
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        embedder = SIESparseEmbedder(model="custom/sparse-model")
        embedder._client = mock_sie_client

        embedder.embed_query("test")

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/sparse-model"

    def test_lazy_client_initialization(self) -> None:
        """Test that client is not created until first use."""
        embedder = SIESparseEmbedder(model="test-model")

        assert embedder._client is None
