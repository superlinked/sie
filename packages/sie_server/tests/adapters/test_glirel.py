from __future__ import annotations

import pytest
from sie_server.types.inputs import Item


class TestGLiRELAdapterExtractEntities:
    """Tests for GLiRELAdapter._extract_entities with TypedDict items.

    Item is a TypedDict -- at runtime it's a plain dict. The adapter must use
    dict access (.get()) rather than attribute access (.metadata).
    """

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
