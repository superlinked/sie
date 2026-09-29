"""Unit tests for SIEReranker."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
from langchain_core.documents import Document
from sie_langchain import SIEReranker

# Ranked entries reference input positions via item_id, out of input order and
# sorted by relevance (doc-3 most relevant). Shared across the mapping tests.
_RANKED_ENVELOPE = {
    "model": "test-reranker",
    "scores": [
        {"item_id": "3", "score": 0.9, "rank": 0},
        {"item_id": "1", "score": 0.7, "rank": 1},
        {"item_id": "4", "score": 0.5, "rank": 2},
        {"item_id": "0", "score": 0.3, "rank": 3},
        {"item_id": "2", "score": 0.1, "rank": 4},
    ],
}

# Envelope for three inputs where only item_id "1" is usable; the rest are
# malformed and must be skipped without crashing so their inputs keep the 0.0
# default. The float 1.5 and bool True come AFTER the valid "1": if int()
# accepted them (int(1.5) == 1, int(True) == 1) they would overwrite position 1.
_MALFORMED_ENVELOPE = {
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

_STUB_QUERY = "vector similarity search"
_STUB_DOCUMENTS = [
    "The weather today is sunny with clear skies.",
    "Python is a popular programming language.",
    "Nearest neighbor search uses distance metrics.",
    "Vector similarity search finds similar embeddings.",
]
_STUB_RANKING = [_STUB_DOCUMENTS[i] for i in (3, 2, 0, 1)]


class TestSIEReranker:
    """Tests for SIEReranker class."""

    def test_compress_documents(self, mock_sie_client: object, test_documents: list[str]) -> None:
        """Test reranking documents."""
        reranker = SIEReranker(client=mock_sie_client, model="test-reranker")
        documents = [Document(page_content=text) for text in test_documents]

        result = reranker.compress_documents(documents, "test query")

        # Every input document comes back exactly as many times as it went in
        # (Counter checks multiplicity, not just membership). The envelope bug
        # iterated the ScoreResult dict's keys, returning documents[0] duplicated.
        assert len(result) == len(documents)
        assert Counter(d.page_content for d in result) == Counter(d.page_content for d in documents)
        scores = [doc.metadata["relevance_score"] for doc in result]
        assert all(isinstance(s, float) for s in scores)
        # Distinct and descending — the bug produced identical 0.0 scores.
        assert scores == sorted(scores, reverse=True)
        assert len(set(scores)) == len(scores)

    def test_compress_documents_maps_scores_by_item_id(self, mock_sie_client: object) -> None:
        """Top-ranked document is the relevant one, scores mapped by item_id."""
        documents = [Document(page_content=f"doc-{i}") for i in range(5)]
        mock_sie_client.score.side_effect = None
        mock_sie_client.score.return_value = _RANKED_ENVELOPE
        reranker = SIEReranker(client=mock_sie_client, model="test-reranker")

        result = reranker.compress_documents(documents, "query")

        assert [d.page_content for d in result] == ["doc-3", "doc-1", "doc-4", "doc-0", "doc-2"]
        assert result[0].metadata["relevance_score"] == 0.9
        assert [d.metadata["relevance_score"] for d in result] == [0.9, 0.7, 0.5, 0.3, 0.1]

    def test_compress_documents_skips_malformed_item_id(self, mock_sie_client: object) -> None:
        """Malformed item_ids are skipped (no crash, no misassignment)."""
        documents = [Document(page_content=f"doc-{i}") for i in range(3)]
        mock_sie_client.score.side_effect = None
        mock_sie_client.score.return_value = _MALFORMED_ENVELOPE
        reranker = SIEReranker(client=mock_sie_client, model="test-reranker")

        result = reranker.compress_documents(documents, "query")

        assert len(result) == 3
        assert Counter(d.page_content for d in result) == Counter(["doc-0", "doc-1", "doc-2"])
        by_content = {d.page_content: d.metadata["relevance_score"] for d in result}
        assert by_content == {"doc-1": 0.8, "doc-0": 0.0, "doc-2": 0.0}
        assert result[0].page_content == "doc-1"

    def test_compress_documents_ranks_through_sie_client(self, score_stub_server: Any) -> None:
        """Scores returned by a real ``SIEClient.score()`` call rank the matching document first."""
        reranker = SIEReranker(base_url=score_stub_server.url, model="test-reranker")
        documents = [Document(page_content=text) for text in _STUB_DOCUMENTS]

        result = reranker.compress_documents(documents, _STUB_QUERY)

        assert [d.page_content for d in result] == _STUB_RANKING
        assert [d.metadata["relevance_score"] for d in result] == [3.0, 1.0, 0.0, 0.0]

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
    async def test_acompress_documents_ranks_through_sie_client(self, score_stub_server: Any) -> None:
        """Scores returned by a real ``SIEAsyncClient.score()`` call rank the matching document first."""
        reranker = SIEReranker(base_url=score_stub_server.url, model="test-reranker")
        documents = [Document(page_content=text) for text in _STUB_DOCUMENTS]

        try:
            result = await reranker.acompress_documents(documents, _STUB_QUERY)
        finally:
            await reranker.async_client.close()

        assert [d.page_content for d in result] == _STUB_RANKING
        assert [d.metadata["relevance_score"] for d in result] == [3.0, 1.0, 0.0, 0.0]

    @pytest.mark.asyncio
    async def test_acompress_documents_empty(self, mock_sie_async_client: object) -> None:
        """Test async reranking empty list returns empty."""
        reranker = SIEReranker(async_client=mock_sie_async_client, model="test-reranker")

        result = await reranker.acompress_documents([], "test query")

        assert result == []
