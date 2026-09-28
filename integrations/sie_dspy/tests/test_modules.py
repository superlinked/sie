"""Unit tests for SIE DSPy modules."""

from __future__ import annotations

from collections import Counter
from unittest.mock import create_autospec

import dspy
from sie_dspy import SIEExtractor, SIEReranker
from sie_dspy.modules import Entity
from sie_sdk import SIEClient


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
        # Every passage comes back exactly as many times as it went in (Counter
        # checks multiplicity, not just membership). The envelope bug tried to
        # zip passages against the ScoreResult dict's keys.
        assert Counter(result.passages) == Counter(ml_corpus)
        # Distinct and descending.
        assert result.scores == sorted(result.scores, reverse=True)
        assert len(set(result.scores)) == len(result.scores)

    def test_rerank_maps_scores_by_item_id(self, mock_sie_client: object) -> None:
        """Top-ranked passage is the relevant one, scores mapped by item_id."""
        passages = [f"doc-{i}" for i in range(5)]
        # Ranked entries reference input positions via item_id (doc-3 = index 3
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
        reranker = SIEReranker(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker(query="query", passages=passages)

        assert result.passages == ["doc-3", "doc-1", "doc-4", "doc-0", "doc-2"]
        assert result.scores == [0.9, 0.7, 0.5, 0.3, 0.1]

    def test_rerank_skips_malformed_item_id(self, mock_sie_client: object) -> None:
        """Malformed item_ids are skipped (no crash, no misassignment)."""
        passages = [f"doc-{i}" for i in range(3)]
        # Only item_id "1" is usable; the rest are malformed. The float 1.5 and
        # bool True come after the valid "1": if int() accepted them
        # (int(1.5) == 1, int(True) == 1) they would overwrite doc-1's score.
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
        reranker = SIEReranker(model="test-reranker")
        reranker._client = mock_sie_client

        result = reranker(query="query", passages=passages)

        assert len(result.passages) == 3
        assert Counter(result.passages) == Counter(["doc-0", "doc-1", "doc-2"])
        by_passage = dict(zip(result.passages, result.scores, strict=True))
        assert by_passage == {"doc-1": 0.8, "doc-0": 0.0, "doc-2": 0.0}
        assert result.passages[0] == "doc-1"

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
        extractor = SIEExtractor(
            model="test-extractor",
            labels=["very_specific_label"],
        )
        # Create a fresh mock that returns empty
        empty_mock = create_autospec(SIEClient, instance=True)
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
