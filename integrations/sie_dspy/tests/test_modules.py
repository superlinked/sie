"""Unit tests for SIE DSPy modules."""

from __future__ import annotations

import dspy
from sie_dspy import SIEExtractor, SIEReranker
from sie_dspy.modules import Entity


class TestSIEReranker:
    """Tests for SIEReranker module.

    DSPy use case: Two-stage retrieval where initial retrieval candidates
    are reranked for higher precision before generation.
    """

    def test_rerank_passages(self, mock_sie_client: object, ml_corpus: list[str]) -> None:
        """Test reranking passages for improved retrieval."""
        reranker = SIEReranker(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker(
            query="How do neural networks learn?",
            passages=ml_corpus,
            k=3,
        )

        assert isinstance(result, dspy.Prediction)
        assert hasattr(result, "passages")
        assert hasattr(result, "scores")
        assert len(result.passages) == 3
        assert len(result.scores) == 3
        # Scores should be sorted descending
        assert result.scores == sorted(result.scores, reverse=True)

    def test_rerank_empty_passages(self, mock_sie_client: object) -> None:
        """Test reranking with no passages."""
        reranker = SIEReranker(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker(query="test query", passages=[])

        assert result.passages == []
        assert result.scores == []

    def test_rerank_no_k(self, mock_sie_client: object, ml_corpus: list[str]) -> None:
        """Test reranking without k limit returns all passages."""
        reranker = SIEReranker(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker(
            query="What is deep learning?",
            passages=ml_corpus,
        )

        assert len(result.passages) == len(ml_corpus)
        assert len(result.scores) == len(ml_corpus)

    def test_rerank_k_larger_than_passages(self, mock_sie_client: object) -> None:
        """Test reranking when k is larger than passage count."""
        reranker = SIEReranker(model="test-reranker")
        reranker._client = mock_sie_client

        passages = ["doc1", "doc2"]
        result = reranker(query="test", passages=passages, k=10)

        assert len(result.passages) == 2

    def test_custom_model(self, mock_sie_client: object, ml_corpus: list[str]) -> None:
        """Test using a custom reranker model."""
        reranker = SIEReranker(model="custom/reranker-model")
        reranker._client = mock_sie_client

        reranker(query="test", passages=ml_corpus[:2])

        call_args = mock_sie_client.score.call_args
        assert call_args[0][0] == "custom/reranker-model"

    def test_is_dspy_module(self) -> None:
        """Test that SIEReranker is a DSPy module."""
        reranker = SIEReranker()

        assert isinstance(reranker, dspy.Module)

    def test_forward_method(self, mock_sie_client: object, ml_corpus: list[str]) -> None:
        """Test that forward method works (DSPy module interface)."""
        reranker = SIEReranker(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker.forward(
            query="test query",
            passages=ml_corpus[:2],
        )

        assert isinstance(result, dspy.Prediction)


class TestSIEExtractor:
    """Tests for SIEExtractor module.

    DSPy use case: Extracting structured information from text for
    knowledge graph construction, information retrieval enhancement,
    or structured output generation.
    """

    def test_extract_entities(self, mock_sie_client: object, research_text: str) -> None:
        """Test extracting entities from research text."""
        extractor = SIEExtractor(
            model="test-extractor",
            labels=["person", "organization", "location"],
        )
        extractor._client = mock_sie_client

        result = extractor(text=research_text)

        assert isinstance(result, dspy.Prediction)
        assert hasattr(result, "entities")
        assert hasattr(result, "entities_dict")
        # Entities should be Entity objects
        for entity in result.entities:
            assert isinstance(entity, Entity)
            assert hasattr(entity, "text")
            assert hasattr(entity, "label")
            assert hasattr(entity, "score")

    def test_extract_with_custom_labels(self, mock_sie_client: object, research_text: str) -> None:
        """Test extraction with custom labels."""
        extractor = SIEExtractor(model="test-extractor")
        extractor._client = mock_sie_client

        extractor(
            text=research_text,
            labels=["researcher", "funding_amount", "institution"],
        )

        # Check that custom labels were passed
        call_kwargs = mock_sie_client.extract.call_args.kwargs
        assert call_kwargs.get("labels") == ["researcher", "funding_amount", "institution"]

    def test_extract_empty_result(self) -> None:
        """Test extraction with no entities found."""
        from unittest.mock import MagicMock

        extractor = SIEExtractor(
            model="test-extractor",
            labels=["very_specific_label"],
        )
        # Create a fresh mock that returns empty
        empty_mock = MagicMock()
        empty_mock.extract.return_value = []
        extractor._client = empty_mock

        result = extractor(text="Simple text with no entities.")

        assert result.entities == []
        assert result.entities_dict == []

    def test_entities_dict_format(self, mock_sie_client: object, research_text: str) -> None:
        """Test that entities_dict is JSON-serializable."""
        extractor = SIEExtractor(model="test-extractor")
        extractor._client = mock_sie_client

        result = extractor(text=research_text)

        # Should be a list of dicts
        assert isinstance(result.entities_dict, list)
        for entity_dict in result.entities_dict:
            assert isinstance(entity_dict, dict)
            assert "text" in entity_dict
            assert "label" in entity_dict
            assert "score" in entity_dict
            assert "start" in entity_dict
            assert "end" in entity_dict

    def test_custom_model(self, mock_sie_client: object, research_text: str) -> None:
        """Test using a custom extraction model."""
        extractor = SIEExtractor(model="custom/extraction-model")
        extractor._client = mock_sie_client

        extractor(text=research_text)

        call_args = mock_sie_client.extract.call_args
        assert call_args[0][0] == "custom/extraction-model"

    def test_is_dspy_module(self) -> None:
        """Test that SIEExtractor is a DSPy module."""
        extractor = SIEExtractor()

        assert isinstance(extractor, dspy.Module)

    def test_default_labels(self) -> None:
        """Test that default labels are set."""
        extractor = SIEExtractor()

        assert "person" in extractor._labels
        assert "organization" in extractor._labels
        assert "location" in extractor._labels

    def test_forward_method(self, mock_sie_client: object, research_text: str) -> None:
        """Test that forward method works (DSPy module interface)."""
        extractor = SIEExtractor(model="test-extractor")
        extractor._client = mock_sie_client

        result = extractor.forward(text=research_text)

        assert isinstance(result, dspy.Prediction)
