"""Unit tests for SIEExtractor component."""

from __future__ import annotations

from sie_haystack import SIEExtractor
from sie_haystack.extractors import Entity


class TestSIEExtractor:
    """Tests for SIEExtractor component."""

    def test_run_extracts_entities(self, mock_sie_client: object, test_ner_text: str) -> None:
        """Test that run extracts entities."""
        extractor = SIEExtractor(
            model="test-extractor",
            labels=["person", "organization", "location"],
        )
        extractor._client = mock_sie_client

        result = extractor.run(text=test_ner_text)

        assert "entities" in result
        assert isinstance(result["entities"], list)
        # Mock should return at least some entities
        for entity in result["entities"]:
            assert isinstance(entity, Entity)
            assert entity.text
            assert entity.label
            assert isinstance(entity.score, float)
            assert isinstance(entity.start, int)
            assert isinstance(entity.end, int)

    def test_run_with_custom_labels(self, mock_sie_client: object) -> None:
        """Test extraction with custom labels."""
        custom_labels = ["product", "price", "date"]
        extractor = SIEExtractor(model="test-extractor", labels=custom_labels)
        extractor._client = mock_sie_client

        extractor.run(text="Test text")

        call_kwargs = mock_sie_client.extract.call_args.kwargs
        assert call_kwargs.get("labels") == custom_labels

    def test_run_with_label_override(self, mock_sie_client: object) -> None:
        """Test that per-call labels override configured labels."""
        extractor = SIEExtractor(
            model="test-extractor",
            labels=["person", "organization"],
        )
        extractor._client = mock_sie_client

        override_labels = ["product", "price"]
        extractor.run(text="Test", labels=override_labels)

        call_kwargs = mock_sie_client.extract.call_args.kwargs
        assert call_kwargs.get("labels") == override_labels

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        extractor = SIEExtractor(model="custom/extraction-model")
        extractor._client = mock_sie_client

        extractor.run(text="Test")

        call_args = mock_sie_client.extract.call_args
        assert call_args[0][0] == "custom/extraction-model"

    def test_default_labels(self, mock_sie_client: object) -> None:
        """Test that default labels are used."""
        extractor = SIEExtractor(model="test-extractor")
        extractor._client = mock_sie_client

        extractor.run(text="Test")

        call_kwargs = mock_sie_client.extract.call_args.kwargs
        # Default labels should be person, organization, location
        assert call_kwargs.get("labels") == ["person", "organization", "location"]

    def test_entity_structure(self, mock_sie_client: object) -> None:
        """Test Entity dataclass structure."""
        entity = Entity(
            text="John Smith",
            label="person",
            score=0.95,
            start=0,
            end=10,
        )

        assert entity.text == "John Smith"
        assert entity.label == "person"
        assert entity.score == 0.95
        assert entity.start == 0
        assert entity.end == 10
