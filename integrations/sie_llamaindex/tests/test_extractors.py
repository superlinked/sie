"""Unit tests for SIEExtractorTool."""

from __future__ import annotations

from sie_llamaindex import create_sie_extractor_tool


class TestSIEExtractorTool:
    """Tests for SIE extractor tool."""

    def test_create_extractor_tool(self, mock_sie_client: object) -> None:
        """Test creating extractor tool."""
        tool = create_sie_extractor_tool(
            model="test-extractor",
            labels=["person", "organization"],
        )

        assert tool.metadata.name == "sie_extract"
        assert "person" in tool.metadata.description
        assert "organization" in tool.metadata.description

    def test_extract_entities(self, mock_sie_client: object) -> None:
        """Test extracting entities through the tool."""
        from sie_llamaindex.extractors import _SIEExtractor

        extractor = _SIEExtractor(
            base_url="http://localhost:8080",
            model="test-extractor",
            labels=["person", "organization", "location"],
            options=None,
            gpu=None,
            timeout_s=180.0,
        )
        extractor._client = mock_sie_client

        result = extractor.extract("John Smith works at Acme Corp in New York")

        assert isinstance(result, dict)
        assert "entities" in result
        assert "relations" in result
        assert "classifications" in result
        assert "objects" in result
        # Mock uses probabilistic entity extraction, so result may be empty
        if len(result["entities"]) > 0:
            entity = result["entities"][0]
            assert "text" in entity
            assert "label" in entity
            assert "score" in entity
            assert "start" in entity
            assert "end" in entity

    def test_custom_labels(self, mock_sie_client: object) -> None:
        """Test using custom labels."""
        from sie_llamaindex.extractors import _SIEExtractor

        custom_labels = ["product", "price", "date"]
        extractor = _SIEExtractor(
            base_url="http://localhost:8080",
            model="test-extractor",
            labels=custom_labels,
            options=None,
            gpu=None,
            timeout_s=180.0,
        )
        extractor._client = mock_sie_client

        extractor.extract("test text")

        call_kwargs = mock_sie_client.extract.call_args.kwargs
        assert call_kwargs.get("labels") == custom_labels

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        from sie_llamaindex.extractors import _SIEExtractor

        extractor = _SIEExtractor(
            base_url="http://localhost:8080",
            model="custom/extraction-model",
            labels=["person"],
            options=None,
            gpu=None,
            timeout_s=180.0,
        )
        extractor._client = mock_sie_client

        extractor.extract("test")

        call_args = mock_sie_client.extract.call_args
        assert call_args[0][0] == "custom/extraction-model"

    def test_custom_tool_name_and_description(self) -> None:
        """Test custom tool name and description."""
        tool = create_sie_extractor_tool(
            name="my_custom_extractor",
            description="My custom extraction tool",
        )

        assert tool.metadata.name == "my_custom_extractor"
        assert tool.metadata.description == "My custom extraction tool"

    def test_tool_callable(self, mock_sie_client: object) -> None:
        """Test that tool is callable."""
        # Create tool with mocked extractor
        tool = create_sie_extractor_tool(model="test-model")

        # The tool wraps a function that uses _SIEExtractor
        assert callable(tool.fn)
