"""Unit tests for SIE Weaviate document enricher."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sie_weaviate import EnrichedDocument, SIEDocumentEnricher


class TestSIEDocumentEnricher:
    """Tests for SIEDocumentEnricher."""

    def test_enrich_returns_vectors_and_properties(self, mock_sie_client: MagicMock) -> None:
        """Enriched documents contain vectors and text properties."""
        enricher = SIEDocumentEnricher(
            extract_model="ner-model",
            labels=["person", "organization"],
        )
        enricher._client = mock_sie_client

        docs = enricher.enrich(["John works at Google."])

        assert len(docs) == 1
        assert isinstance(docs[0], EnrichedDocument)
        assert isinstance(docs[0].vector, list)
        assert len(docs[0].vector) == 384
        assert all(isinstance(v, float) for v in docs[0].vector)
        assert docs[0].properties["text"] == "John works at Google."

    def test_enrich_groups_entities_by_label(self, mock_sie_client: MagicMock) -> None:
        """Entities are grouped by label into list properties."""
        enricher = SIEDocumentEnricher(
            extract_model="ner-model",
            labels=["person", "organization", "location"],
        )
        enricher._client = mock_sie_client

        docs = enricher.enrich(["test text"])

        props = docs[0].properties
        assert "person" in props
        assert "organization" in props
        assert "location" in props
        assert isinstance(props["person"], list)
        assert isinstance(props["organization"], list)

    def test_enrich_with_classification(self, mock_classify_client: MagicMock) -> None:
        """Classification adds top label as property."""
        enricher = SIEDocumentEnricher(
            extract_model="ner-model",
            labels=["person"],
            classify_model="classify-model",
            classify_labels=["technical", "business", "legal"],
        )
        enricher._client = mock_classify_client

        docs = enricher.enrich(["test text"])

        props = docs[0].properties
        assert "classification" in props
        assert props["classification"] == "technical"  # highest score in mock
        assert "classification_score" in props
        assert isinstance(props["classification_score"], float)

    def test_enrich_empty_input(self, mock_sie_client: MagicMock) -> None:
        """Empty input returns empty list without calling SIE."""
        enricher = SIEDocumentEnricher()
        enricher._client = mock_sie_client

        result = enricher.enrich([])

        assert result == []
        mock_sie_client.encode.assert_not_called()
        mock_sie_client.extract.assert_not_called()

    def test_enrich_query(self, mock_sie_client: MagicMock) -> None:
        """Query enrichment returns only a vector, no extraction."""
        enricher = SIEDocumentEnricher()
        enricher._client = mock_sie_client

        vec = enricher.enrich_query("search text")

        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)
        mock_sie_client.extract.assert_not_called()

    def test_enrich_without_extract_model(self, mock_sie_client: MagicMock) -> None:
        """With extract_model=None, only embeddings are produced."""
        enricher = SIEDocumentEnricher(extract_model=None)
        enricher._client = mock_sie_client

        docs = enricher.enrich(["test text"])

        assert len(docs) == 1
        assert docs[0].properties == {"text": "test text"}
        assert docs[0].entities is None
        assert docs[0].classifications is None
        assert len(docs[0].vector) == 384
        mock_sie_client.extract.assert_not_called()

    def test_enrich_raw_entities_preserved(self, mock_sie_client: MagicMock) -> None:
        """Raw entity dicts are available on EnrichedDocument."""
        enricher = SIEDocumentEnricher(
            extract_model="ner-model",
            labels=["person"],
        )
        enricher._client = mock_sie_client

        docs = enricher.enrich(["test text"])

        assert docs[0].entities is not None
        assert len(docs[0].entities) > 0
        entity = docs[0].entities[0]
        assert "text" in entity
        assert "label" in entity
        assert "score" in entity

    def test_enrich_multiple_documents(self, mock_sie_client: MagicMock, sample_documents: list[str]) -> None:
        """Multiple documents each get their own vectors and properties."""
        enricher = SIEDocumentEnricher(
            extract_model="ner-model",
            labels=["person", "organization"],
        )
        enricher._client = mock_sie_client

        docs = enricher.enrich(sample_documents)

        assert len(docs) == len(sample_documents)
        for i, doc in enumerate(docs):
            assert doc.properties["text"] == sample_documents[i]
            assert len(doc.vector) == 384

    def test_enrich_deduplicates_entities(self, mock_sie_client: MagicMock) -> None:
        """Duplicate entity texts within a label are deduplicated."""
        enricher = SIEDocumentEnricher(extract_model="ner-model", labels=["person"])
        enricher._client = mock_sie_client

        # Manually set up a result with duplicate entities
        mock_sie_client.extract = MagicMock(
            return_value=[
                {
                    "entities": [
                        {"text": "John", "label": "person", "score": 0.9, "start": 0, "end": 4},
                        {"text": "John", "label": "person", "score": 0.8, "start": 20, "end": 24},
                    ],
                }
            ]
        )

        docs = enricher.enrich(["John met John."])

        assert docs[0].properties["person"] == ["John"]

    def test_lazy_client_initialization(self) -> None:
        """Client is not created until first use."""
        enricher = SIEDocumentEnricher()
        assert enricher._client is None

    def test_enriched_document_dataclass(self) -> None:
        """EnrichedDocument has expected fields and defaults."""
        doc = EnrichedDocument(
            vector=[0.1, 0.2],
            properties={"text": "hello"},
        )
        assert doc.vector == [0.1, 0.2]
        assert doc.properties == {"text": "hello"}
        assert doc.entities is None
        assert doc.classifications is None

    # -- Chunking tests ------------------------------------------------------

    def test_enrich_iter_yields_chunks(self, mock_sie_client: MagicMock) -> None:
        """enrich_iter yields one batch of documents per chunk."""
        enricher = SIEDocumentEnricher(extract_model=None)
        enricher._client = mock_sie_client

        texts = ["a", "b", "c", "d", "e"]
        chunks = list(enricher.enrich_iter(texts, batch_size=2))

        assert len(chunks) == 3  # 2, 2, 1
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1

    def test_enrich_with_batch_size(self, mock_sie_client: MagicMock) -> None:
        """Enrich with batch_size calls encode once per chunk."""
        enricher = SIEDocumentEnricher(extract_model=None)
        enricher._client = mock_sie_client

        docs = enricher.enrich(["a", "b", "c", "d", "e"], batch_size=2)

        assert len(docs) == 5
        assert mock_sie_client.encode.call_count == 3  # chunks of 2, 2, 1

    def test_enrich_iter_empty(self, mock_sie_client: MagicMock) -> None:
        """enrich_iter on empty input yields nothing."""
        enricher = SIEDocumentEnricher()
        enricher._client = mock_sie_client

        chunks = list(enricher.enrich_iter([]))

        assert chunks == []

    # -- Lifecycle tests -----------------------------------------------------

    def test_close_closes_client(self, mock_sie_client: MagicMock) -> None:
        """close() calls client.close() and resets to None."""
        enricher = SIEDocumentEnricher()
        enricher._client = mock_sie_client

        enricher.close()

        mock_sie_client.close.assert_called_once()
        assert enricher._client is None

    def test_context_manager_sync(self, mock_sie_client: MagicMock) -> None:
        """Sync context manager calls close on exit."""
        enricher = SIEDocumentEnricher()
        enricher._client = mock_sie_client

        with enricher:
            pass

        mock_sie_client.close.assert_called_once()

    def test_async_client_lazy_init(self) -> None:
        """Async client is None until first use."""
        enricher = SIEDocumentEnricher()
        assert enricher._async_client is None


class TestSIEDocumentEnricherAsync:
    """Async tests for SIEDocumentEnricher."""

    @pytest.mark.asyncio
    async def test_aenrich_returns_vectors_and_properties(self, mock_sie_async_client: AsyncMock) -> None:
        """Aenrich produces same output shape as enrich."""
        enricher = SIEDocumentEnricher(
            extract_model="ner-model",
            labels=["person", "organization"],
        )
        enricher._async_client = mock_sie_async_client

        docs = await enricher.aenrich(["John works at Google."])

        assert len(docs) == 1
        assert isinstance(docs[0], EnrichedDocument)
        assert len(docs[0].vector) == 384
        assert docs[0].properties["text"] == "John works at Google."
        assert "person" in docs[0].properties

    @pytest.mark.asyncio
    async def test_aenrich_query(self, mock_sie_async_client: AsyncMock) -> None:
        """aenrich_query returns a vector without extraction."""
        enricher = SIEDocumentEnricher()
        enricher._async_client = mock_sie_async_client

        vec = await enricher.aenrich_query("search text")

        assert isinstance(vec, list)
        assert len(vec) == 384
        mock_sie_async_client.extract.assert_not_called()

    @pytest.mark.asyncio
    async def test_aenrich_iter_yields_chunks(self, mock_sie_async_client: AsyncMock) -> None:
        """aenrich_iter yields per-chunk results."""
        enricher = SIEDocumentEnricher(extract_model=None)
        enricher._async_client = mock_sie_async_client

        chunks = []
        async for chunk in enricher.aenrich_iter(["a", "b", "c", "d", "e"], batch_size=2):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1

    @pytest.mark.asyncio
    async def test_aenrich_empty_input(self, mock_sie_async_client: AsyncMock) -> None:
        """Aenrich on empty input returns empty list."""
        enricher = SIEDocumentEnricher()
        enricher._async_client = mock_sie_async_client

        result = await enricher.aenrich([])

        assert result == []
        mock_sie_async_client.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_aenrich_with_classification(self, mock_async_classify_client: AsyncMock) -> None:
        """Aenrich runs encode+extract+classify concurrently."""
        enricher = SIEDocumentEnricher(
            extract_model="ner-model",
            labels=["person"],
            classify_model="classify-model",
            classify_labels=["technical", "business"],
        )
        enricher._async_client = mock_async_classify_client

        docs = await enricher.aenrich(["test text"])

        assert len(docs) == 1
        props = docs[0].properties
        assert "person" in props
        assert "classification" in props

    @pytest.mark.asyncio
    async def test_aclose_closes_async_client(self, mock_sie_async_client: AsyncMock) -> None:
        """aclose() calls async_client.close() and resets to None."""
        enricher = SIEDocumentEnricher()
        enricher._async_client = mock_sie_async_client

        await enricher.aclose()

        mock_sie_async_client.close.assert_called_once()
        assert enricher._async_client is None

    @pytest.mark.asyncio
    async def test_context_manager_async(self, mock_sie_async_client: AsyncMock) -> None:
        """Async context manager calls aclose on exit."""
        enricher = SIEDocumentEnricher()
        enricher._async_client = mock_sie_async_client

        async with enricher:
            pass

        mock_sie_async_client.close.assert_called_once()
