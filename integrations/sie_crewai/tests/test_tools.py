"""Unit tests for SIE CrewAI tools and embedders."""

from __future__ import annotations

from typing import Any
from unittest.mock import NonCallableMagicMock, create_autospec

import pytest
from sie_crewai import SIEExtractorTool, SIERerankerTool, SIESparseEmbedder
from sie_sdk import RequestError, SIEClient


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

    def test_rerank_maps_scores_by_item_id(self, mock_sie_client: object) -> None:
        """Top-ranked document is the relevant one, scores mapped by item_id."""
        documents = ["alpha", "bravo", "charlie", "delta", "echo"]
        # Ranked entries reference input positions via item_id (delta = index 3
        # is most relevant), out of input order.
        mock_sie_client.score.side_effect = None
        mock_sie_client.score.return_value = {
            "model": "test-reranker",
            "scores": [
                {"item_id": "3", "score": 0.9, "rank": 0},
                {"item_id": "1", "score": 0.7, "rank": 1},
                {"item_id": "4", "score": 0.5, "rank": 2},
                {"item_id": "0", "score": 0.3, "rank": 3},
                {"item_id": "2", "score": 0.1, "rank": 4},
            ],
        }
        reranker = SIERerankerTool(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker._run(query="query", documents=documents)

        # Ranked most-relevant first: delta, bravo, echo, alpha, charlie.
        assert (
            result.index("delta")
            < result.index("bravo")
            < result.index("echo")
            < result.index("alpha")
            < result.index("charlie")
        )
        assert "[Score: 0.9000] delta" in result
        # All five documents are represented (no duplication or loss).
        assert all(doc in result for doc in documents)

    def test_rerank_skips_malformed_item_id(self, mock_sie_client: object) -> None:
        """Malformed item_ids are skipped (no crash, no misassignment)."""
        documents = ["alpha", "bravo", "charlie"]
        # Only item_id "1" (bravo) is usable; the rest are malformed. The float
        # 1.5 and bool True come after the valid "1": if int() accepted them
        # (int(1.5) == 1, int(True) == 1) they would overwrite bravo's score.
        mock_sie_client.score.side_effect = None
        mock_sie_client.score.return_value = {
            "model": "test-reranker",
            "scores": [
                {"item_id": "1", "score": 0.8, "rank": 0},
                {"item_id": "not-an-int", "score": 0.95, "rank": 1},
                {"item_id": "-1", "score": 0.9, "rank": 2},
                {"item_id": "99", "score": 0.7, "rank": 3},
                {"score": 0.5, "rank": 4},
                {"item_id": 1.5, "score": 0.99, "rank": 5},
                {"item_id": True, "score": 0.98, "rank": 6},
            ],
        }
        reranker = SIERerankerTool(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker._run(query="query", documents=documents)

        # All three documents represented, none dropped or duplicated.
        assert result.count("[Score:") == 3
        assert "[Score: 0.8000] bravo" in result
        # bravo (the only scored doc) ranks first.
        assert result.index("bravo") < result.index("alpha")
        assert result.index("bravo") < result.index("charlie")

    def test_rerank_ranks_through_sie_client(self, score_stub_server: Any) -> None:
        """Scores returned by a real ``SIEClient.score()`` call rank the matching document first."""
        documents = [
            "The weather today is sunny with clear skies.",
            "Python is a popular programming language.",
            "Nearest neighbor search uses distance metrics.",
            "Vector similarity search finds similar embeddings.",
        ]
        reranker = SIERerankerTool(base_url=score_stub_server.url, model="test-reranker")

        result = reranker._run(query="vector similarity search", documents=documents)

        assert result.splitlines() == [
            "Ranked documents (most relevant first):",
            f"1. [Score: 3.0000] {documents[3]}",
            f"2. [Score: 1.0000] {documents[2]}",
            f"3. [Score: 0.0000] {documents[0]}",
            f"4. [Score: 0.0000] {documents[1]}",
        ]

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

        assert "Extracted entities" in result or "No extraction results found" in result

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
        extractor = SIEExtractorTool(
            model="test-extractor",
            labels=["very_specific_label"],
        )
        # Create a fresh mock that returns empty
        empty_mock = create_autospec(SIEClient, instance=True)
        empty_mock.extract.return_value = {"entities": [], "relations": [], "classifications": [], "objects": []}
        extractor._client = empty_mock

        result = extractor._run(text="Simple text with no entities.")

        assert "No extraction results found" in result

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


def test_extractor_tool_raises_on_item_error(
    mock_sie_client: NonCallableMagicMock, extract_item_error: dict[str, str]
) -> None:
    mock_sie_client.extract.side_effect = None
    mock_sie_client.extract.return_value = {
        "entities": [],
        "relations": [],
        "classifications": [],
        "objects": [],
        "error": dict(extract_item_error),
    }
    extractor = SIEExtractorTool(model="test-extractor", labels=["person"])
    extractor._client = mock_sie_client

    with pytest.raises(RequestError, match="Extraction failed") as excinfo:
        extractor._run(text="text")

    assert excinfo.value.code == extract_item_error["code"]
