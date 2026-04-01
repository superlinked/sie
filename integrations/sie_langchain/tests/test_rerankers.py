"""Unit tests for SIEReranker."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from sie_langchain import SIEReranker


class TestSIEReranker:
    """Tests for SIEReranker class."""

    def test_compress_documents(self, mock_sie_client: object, test_documents: list[str]) -> None:
        """Test reranking documents."""
        reranker = SIEReranker(client=mock_sie_client, model="test-reranker")
        documents = [Document(page_content=text) for text in test_documents]

        result = reranker.compress_documents(documents, "test query")

        assert len(result) > 0
        # All results should have relevance_score in metadata
        for doc in result:
            assert "relevance_score" in doc.metadata
            assert isinstance(doc.metadata["relevance_score"], float)

    def test_compress_documents_empty(self, mock_sie_client: object) -> None:
        """Test reranking empty list returns empty."""
        reranker = SIEReranker(client=mock_sie_client, model="test-reranker")

        result = reranker.compress_documents([], "test query")

        assert result == []

    def test_compress_documents_top_k(self, mock_sie_client: object, test_documents: list[str]) -> None:
        """Test reranking with top_k limit."""
        reranker = SIEReranker(client=mock_sie_client, model="test-reranker", top_k=2)
        documents = [Document(page_content=text) for text in test_documents]

        result = reranker.compress_documents(documents, "test query")

        assert len(result) <= 2

    def test_compress_documents_preserves_metadata(self, mock_sie_client: object) -> None:
        """Test that reranking preserves original document metadata."""
        reranker = SIEReranker(client=mock_sie_client, model="test-reranker")
        documents = [Document(page_content="test doc", metadata={"source": "test.txt", "page": 1})]

        result = reranker.compress_documents(documents, "test query")

        assert len(result) == 1
        assert result[0].metadata["source"] == "test.txt"
        assert result[0].metadata["page"] == 1
        assert "relevance_score" in result[0].metadata

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        reranker = SIEReranker(client=mock_sie_client, model="custom/reranker-model")
        documents = [Document(page_content="test")]

        reranker.compress_documents(documents, "query")

        mock_sie_client.score.assert_called()
        call_args = mock_sie_client.score.call_args
        assert call_args[0][0] == "custom/reranker-model"


class TestSIERerankerAsync:
    """Tests for async SIEReranker methods."""

    @pytest.mark.asyncio
    async def test_acompress_documents(self, mock_sie_async_client: object, test_documents: list[str]) -> None:
        """Test async reranking documents."""
        reranker = SIEReranker(async_client=mock_sie_async_client, model="test-reranker")
        documents = [Document(page_content=text) for text in test_documents]

        result = await reranker.acompress_documents(documents, "test query")

        assert len(result) > 0
        for doc in result:
            assert "relevance_score" in doc.metadata

    @pytest.mark.asyncio
    async def test_acompress_documents_empty(self, mock_sie_async_client: object) -> None:
        """Test async reranking empty list returns empty."""
        reranker = SIEReranker(async_client=mock_sie_async_client, model="test-reranker")

        result = await reranker.acompress_documents([], "test query")

        assert result == []
