"""Unit tests for SIEExtractor."""

from __future__ import annotations

import pytest
from sie_langchain import SIEExtractor


class TestSIEExtractor:
    """Tests for SIEExtractor class."""

    def test_extract_returns_multi_type_dict(self, mock_sie_client: object, test_ner_text: str) -> None:
        """Test extracting returns dict with all extraction types."""
        extractor = SIEExtractor(
            client=mock_sie_client, model="test-ner", labels=["PERSON", "ORGANIZATION", "LOCATION"]
        )

        result = extractor._run(test_ner_text)

        assert isinstance(result, dict)
        assert "entities" in result
        assert "relations" in result
        assert "classifications" in result
        assert "objects" in result
        assert isinstance(result["entities"], list)
        assert isinstance(result["relations"], list)
        assert isinstance(result["classifications"], list)
        assert isinstance(result["objects"], list)
        # Mock may or may not return entities based on random seed
        for entity in result["entities"]:
            assert "text" in entity
            assert "label" in entity
            assert "score" in entity
            assert "start" in entity
            assert "end" in entity

    def test_extract_invoke(self, mock_sie_client: object, test_ner_text: str) -> None:
        """Test extracting via invoke interface."""
        extractor = SIEExtractor(client=mock_sie_client, model="test-ner", labels=["PERSON", "ORGANIZATION"])

        result = extractor.invoke(test_ner_text)

        assert isinstance(result, dict)
        assert "entities" in result

    def test_custom_labels(self, mock_sie_client: object) -> None:
        """Test using custom entity labels."""
        custom_labels = ["PRODUCT", "PRICE", "DATE"]
        extractor = SIEExtractor(client=mock_sie_client, model="test-ner", labels=custom_labels)

        assert extractor.labels == custom_labels

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        extractor = SIEExtractor(client=mock_sie_client, model="custom/ner-model")

        extractor._run("test text")

        mock_sie_client.extract.assert_called()
        call_args = mock_sie_client.extract.call_args
        assert call_args[0][0] == "custom/ner-model"

    def test_tool_name_and_description(self) -> None:
        """Test tool has proper name and description."""
        extractor = SIEExtractor()

        assert extractor.name == "sie_extract"
        assert "extract" in extractor.description.lower()


class TestSIEExtractorAsync:
    """Tests for async SIEExtractor methods."""

    @pytest.mark.asyncio
    async def test_aextract(self, mock_sie_async_client: object, test_ner_text: str) -> None:
        """Test async extracting."""
        extractor = SIEExtractor(
            async_client=mock_sie_async_client, model="test-ner", labels=["PERSON", "ORGANIZATION"]
        )

        result = await extractor._arun(test_ner_text)

        assert isinstance(result, dict)
        assert "entities" in result
        assert "relations" in result

    @pytest.mark.asyncio
    async def test_aextract_invoke(self, mock_sie_async_client: object, test_ner_text: str) -> None:
        """Test async extracting via ainvoke."""
        extractor = SIEExtractor(async_client=mock_sie_async_client, model="test-ner")

        result = await extractor.ainvoke(test_ner_text)

        assert isinstance(result, dict)
