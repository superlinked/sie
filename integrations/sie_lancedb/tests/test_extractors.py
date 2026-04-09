"""Unit tests for SIE LanceDB extractor."""

from __future__ import annotations

import lancedb
import pyarrow as pa
import pytest
from sie_lancedb import SIEExtractor
from sie_lancedb.extractors import _build_entities_array, _format_entities


@pytest.fixture
def db(tmp_path):
    """Ephemeral in-process LanceDB instance."""
    return lancedb.connect(tmp_path / "test.lance")


class TestSIEExtractor:
    """Tests for SIEExtractor."""

    def test_extract_returns_entities(self, mock_sie_client: object) -> None:
        """Extracting from texts returns entity lists."""
        extractor = SIEExtractor(model="test-model", client=mock_sie_client)

        results = extractor.extract(
            ["John Smith works at Apple Inc. in California."],
            labels=["PERSON", "ORGANIZATION", "LOCATION"],
        )

        assert isinstance(results, list)
        assert len(results) == 1
        for entity in results[0]:
            assert "text" in entity
            assert "label" in entity
            assert "score" in entity
            assert "start" in entity
            assert "end" in entity
            assert "bbox" in entity

    def test_extract_batch(self, mock_sie_client: object) -> None:
        """Extracting from multiple texts returns one list per text."""
        extractor = SIEExtractor(model="test-model", client=mock_sie_client)

        texts = [
            "John Smith works at Apple.",
            "Tim Cook leads the company.",
            "They are in California.",
        ]
        results = extractor.extract(texts, labels=["PERSON", "ORGANIZATION"])

        assert len(results) == 3

    def test_extract_empty(self, mock_sie_client: object) -> None:
        """Empty input returns empty list."""
        extractor = SIEExtractor(model="test-model", client=mock_sie_client)

        results = extractor.extract([], labels=["PERSON"])

        assert results == []
        mock_sie_client.extract.assert_not_called()

    def test_extract_batching(self, mock_sie_client: object) -> None:
        """Large inputs are split into batches."""
        extractor = SIEExtractor(model="test-model", client=mock_sie_client)

        texts = [f"Text number {i}" for i in range(250)]
        results = extractor.extract(texts, labels=["PERSON"], batch_size=100)

        assert len(results) == 250
        # 3 batches: 100 + 100 + 50
        assert mock_sie_client.extract.call_count == 3

    def test_extract_entity_fields(self, mock_sie_client: object) -> None:
        """Extracted entities have correct field types."""
        extractor = SIEExtractor(model="test-model", client=mock_sie_client)

        results = extractor.extract(
            ["John Smith works at Apple Inc. in California."],
            labels=["PERSON", "ORGANIZATION"],
        )

        for entity in results[0]:
            assert isinstance(entity["text"], str)
            assert isinstance(entity["label"], str)
            assert isinstance(entity["score"], float)
            if entity["start"] is not None:
                assert isinstance(entity["start"], int)
            if entity["end"] is not None:
                assert isinstance(entity["end"], int)

    def test_extract_labels_forwarded(self, mock_sie_client: object) -> None:
        """Labels are forwarded to the SIE client."""
        extractor = SIEExtractor(model="test-model", client=mock_sie_client)

        extractor.extract(["test"], labels=["CUSTOM_LABEL"])

        call_kwargs = mock_sie_client.extract.call_args.kwargs
        assert call_kwargs.get("labels") == ["CUSTOM_LABEL"]

    def test_custom_model_forwarded(self, mock_sie_client: object) -> None:
        """Model name is forwarded to SIE client."""
        extractor = SIEExtractor(model="custom/extract-model", client=mock_sie_client)

        extractor.extract(["test"], labels=["PERSON"])

        call_args = mock_sie_client.extract.call_args
        assert call_args[0][0] == "custom/extract-model"

    def test_lazy_client_initialization(self) -> None:
        """Client is not created until first use."""
        extractor = SIEExtractor(model="test-model")
        assert extractor._client is None

    def test_enrich_table(self, mock_sie_client: object, db) -> None:
        """enrich_table reads, extracts, and merges entities into a real LanceDB table."""
        extractor = SIEExtractor(model="test-model", client=mock_sie_client)

        # Create a real LanceDB table
        table = db.create_table(
            "enrich_test",
            data=[
                {"id": 0, "text": "John Smith works at Apple."},
                {"id": 1, "text": "Tim Cook leads the company."},
                {"id": 2, "text": "They are based in California."},
            ],
            mode="overwrite",
        )

        assert "entities" not in table.to_pandas().columns

        # Enrich the table
        extractor.enrich_table(
            table,
            source_column="text",
            target_column="entities",
            labels=["PERSON", "ORGANIZATION"],
            id_column="id",
        )

        # Verify the entities column was added
        df = table.to_pandas()
        assert "entities" in df.columns
        assert len(df) == 3

        # Each row should have entities (possibly empty)
        for entities in df["entities"]:
            for entity in entities:
                assert "text" in entity
                assert "label" in entity
                assert "score" in entity

    def test_enrich_table_preserves_existing_data(self, mock_sie_client: object, db) -> None:
        """enrich_table adds the column without modifying existing columns."""
        extractor = SIEExtractor(model="test-model", client=mock_sie_client)

        table = db.create_table(
            "preserve_test",
            data=[
                {"id": 0, "text": "Hello world.", "category": "greeting"},
                {"id": 1, "text": "Goodbye world.", "category": "farewell"},
            ],
            mode="overwrite",
        )

        extractor.enrich_table(
            table,
            source_column="text",
            target_column="entities",
            labels=["PERSON"],
            id_column="id",
        )

        df = table.to_pandas()
        # Original columns preserved
        assert list(df["category"]) == ["greeting", "farewell"]
        assert list(df["text"]) == ["Hello world.", "Goodbye world."]
        # New column added
        assert "entities" in df.columns


class TestFormatEntities:
    """Tests for _format_entities helper."""

    def test_format_dict_entities(self) -> None:
        """Dict-style entities are formatted correctly."""
        entities = [
            {"text": "John", "label": "PERSON", "score": 0.95, "start": 0, "end": 4, "bbox": None},
        ]

        result = _format_entities(entities)

        assert len(result) == 1
        assert result[0]["text"] == "John"
        assert result[0]["label"] == "PERSON"
        assert result[0]["score"] == 0.95
        assert result[0]["start"] == 0
        assert result[0]["end"] == 4
        assert result[0]["bbox"] is None

    def test_format_empty_entities(self) -> None:
        """Empty entity list returns empty list."""
        assert _format_entities([]) == []

    def test_format_entities_with_bbox(self) -> None:
        """Entities with bbox (vision models) are preserved."""
        entities = [
            {"text": "person", "label": "object", "score": 0.9, "start": None, "end": None, "bbox": [10, 20, 50, 60]},
        ]

        result = _format_entities(entities)

        assert result[0]["bbox"] == [10, 20, 50, 60]
        assert result[0]["start"] is None

    def test_format_entities_missing_fields(self) -> None:
        """Entities with missing optional fields get defaults."""
        entities = [{"text": "test", "label": "X", "score": 0.5}]

        result = _format_entities(entities)

        assert result[0]["start"] is None
        assert result[0]["end"] is None
        assert result[0]["bbox"] is None


class TestBuildEntitiesArray:
    """Tests for _build_entities_array helper."""

    def test_builds_arrow_array(self) -> None:
        """Builds a proper Arrow array of ENTITY_STRUCT lists."""
        entities_per_row = [
            [{"text": "John", "label": "PERSON", "score": 0.95, "start": 0, "end": 4, "bbox": None}],
            [],  # empty entities for second row
        ]

        result = _build_entities_array(entities_per_row)

        assert isinstance(result, pa.Array)
        assert pa.types.is_list(result.type)
        assert len(result) == 2
        assert len(result[0].as_py()) == 1
        assert len(result[1].as_py()) == 0

    def test_empty_input(self) -> None:
        """Empty input produces empty array."""
        result = _build_entities_array([])

        assert isinstance(result, pa.Array)
        assert len(result) == 0
