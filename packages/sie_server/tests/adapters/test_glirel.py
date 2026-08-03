from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import Item


class TestGLiRELAdapterExtractEntities:
    """Tests for GLiREL entity input handling."""

    @pytest.fixture
    def adapter(self) -> GLiRELAdapter:
        from sie_server.adapters.glirel import GLiRELAdapter

        return GLiRELAdapter(
            "jackboyla/glirel-large-v0",
            compute_precision="float32",
        )

    def test_extract_entities_with_metadata(self, adapter: GLiRELAdapter) -> None:
        """Entities are extracted from item metadata dict."""
        item = Item(text="test", metadata={"entities": [{"text": "Alice", "label": "PER"}]})
        result = adapter._extract_entities(item)
        assert result == [{"text": "Alice", "label": "PER"}]

    def test_extract_entities_no_metadata(self, adapter: GLiRELAdapter) -> None:
        """Returns empty list when metadata is absent."""
        item = Item(text="test")
        result = adapter._extract_entities(item)
        assert result == []

    def test_extract_entities_metadata_none(self, adapter: GLiRELAdapter) -> None:
        """Returns empty list when metadata is explicitly None."""
        item = Item(text="test", metadata=None)
        result = adapter._extract_entities(item)
        assert result == []

    def test_extract_entities_metadata_no_entities_key(self, adapter: GLiRELAdapter) -> None:
        """Returns empty list when metadata has no 'entities' key."""
        item = Item(text="test", metadata={"other": "data"})
        result = adapter._extract_entities(item)
        assert result == []

    def test_extract_converts_character_offsets_to_glirel_tokens(self, adapter: GLiRELAdapter) -> None:
        """GLiREL receives its expected tokenized text and inclusive token spans."""
        adapter._model = MagicMock()
        adapter._model.predict_relations.return_value = [
            {
                "head_pos": [0, 2],
                "tail_pos": [6, 9],
                "head_text": ["Tim", "Cook"],
                "tail_text": ["Apple", "Inc", "."],
                "label": "ceo_of",
                "score": 0.98,
            }
        ]
        item = Item(
            text="Tim Cook is the CEO of Apple Inc.",
            metadata={
                "entities": [
                    {"text": "Tim Cook", "label": "PERSON", "start": 0, "end": 8},
                    {"text": "Apple Inc.", "label": "ORG", "start": 23, "end": 33},
                ]
            },
        )

        output = adapter.extract([item], labels=["ceo_of"])

        assert isinstance(output, ExtractOutput)
        assert output.relations == [[{"head": "Tim Cook", "tail": "Apple Inc.", "relation": "ceo_of", "score": 0.98}]]
        adapter._model.predict_relations.assert_called_once_with(
            text=["Tim", "Cook", "is", "the", "CEO", "of", "Apple", "Inc", "."],
            labels=["ceo_of"],
            threshold=0.3,
            ner=[[0, 1, "PERSON", "Tim Cook"], [6, 8, "ORG", "Apple Inc."]],
            top_k=10,
        )

    def test_extract_requires_entity_spans(self, adapter: GLiRELAdapter) -> None:
        """Missing entity candidates fail clearly instead of returning an empty success."""
        adapter._model = MagicMock()

        with pytest.raises(ValueError, match="requires entities in item metadata"):
            adapter.extract([Item(text="Tim Cook is the CEO of Apple Inc.")], labels=["ceo_of"])
