"""Unit tests for SIERanker component."""

from __future__ import annotations

from collections import Counter
from typing import Any

from haystack import Document
from sie_haystack import SIERanker

# Ranked entries reference input positions via item_id, out of input order and
# sorted by relevance (doc-3 most relevant).
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


class TestSIERanker:
    """Tests for SIERanker component."""

    def test_run_reranks_documents(
        self,
        mock_sie_client: object,
        haystack_documents: list[Document],
        test_query: str,
    ) -> None:
        """Test that run reranks documents."""
        ranker = SIERanker(model="test-reranker")
        ranker._client = mock_sie_client

        result = ranker.run(query=test_query, documents=haystack_documents)

        assert "documents" in result
        docs = result["documents"]
        # Every input document comes back exactly as many times as it went in
        # (Counter checks multiplicity, not just membership). The envelope bug
        # tried to zip documents against the ScoreResult dict's keys.
        assert len(docs) == len(haystack_documents)
        assert Counter(d.content for d in docs) == Counter(d.content for d in haystack_documents)
        scores = [d.meta["score"] for d in docs]
        assert all(isinstance(s, float) for s in scores)
        # Distinct and descending.
        assert scores == sorted(scores, reverse=True)
        assert len(set(scores)) == len(scores)

    def test_run_maps_scores_by_item_id(self, mock_sie_client: object) -> None:
        """Top-ranked document is the relevant one, scores mapped by item_id."""
        documents = [Document(content=f"doc-{i}") for i in range(5)]
        mock_sie_client.score.side_effect = None
        mock_sie_client.score.return_value = _RANKED_ENVELOPE
        ranker = SIERanker(model="test-reranker")
        ranker._client = mock_sie_client

        ranked = ranker.run(query="query", documents=documents)["documents"]

        assert [d.content for d in ranked] == ["doc-3", "doc-1", "doc-4", "doc-0", "doc-2"]
        assert ranked[0].meta["score"] == 0.9
        assert [d.meta["score"] for d in ranked] == [0.9, 0.7, 0.5, 0.3, 0.1]

    def test_run_skips_malformed_item_id(self, mock_sie_client: object) -> None:
        """Malformed item_ids are skipped (no crash, no misassignment)."""
        documents = [Document(content=f"doc-{i}") for i in range(3)]
        mock_sie_client.score.side_effect = None
        mock_sie_client.score.return_value = _MALFORMED_ENVELOPE
        ranker = SIERanker(model="test-reranker")
        ranker._client = mock_sie_client

        ranked = ranker.run(query="query", documents=documents)["documents"]

        assert len(ranked) == 3
        by_content = {d.content: d.meta["score"] for d in ranked}
        assert by_content == {"doc-1": 0.8, "doc-0": 0.0, "doc-2": 0.0}
        assert ranked[0].content == "doc-1"

    def test_run_ranks_through_sie_client(self, score_stub_server: Any) -> None:
        """Scores returned by a real ``SIEClient.score()`` call rank the matching document first."""
        documents = [
            Document(content="The weather today is sunny with clear skies."),
            Document(content="Python is a popular programming language."),
            Document(content="Nearest neighbor search uses distance metrics."),
            Document(content="Vector similarity search finds similar embeddings."),
        ]
        ranker = SIERanker(base_url=score_stub_server.url, model="test-reranker")

        ranked = ranker.run(query="vector similarity search", documents=documents)["documents"]

        assert [d.content for d in ranked] == [documents[i].content for i in (3, 2, 0, 1)]
        assert [d.meta["score"] for d in ranked] == [3.0, 1.0, 0.0, 0.0]

    def test_run_empty_list(self, mock_sie_client: object) -> None:
        """Test that run handles empty document list."""
        ranker = SIERanker(model="test-reranker")
        ranker._client = mock_sie_client

        result = ranker.run(query="test query", documents=[])

        assert result == {"documents": []}

    def test_run_with_top_k(
        self,
        mock_sie_client: object,
        haystack_documents: list[Document],
        test_query: str,
    ) -> None:
        """Test that run respects top_k parameter."""
        ranker = SIERanker(model="test-reranker", top_k=2)
        ranker._client = mock_sie_client

        result = ranker.run(query=test_query, documents=haystack_documents)

        assert len(result["documents"]) == 2

    def test_run_with_top_k_override(
        self,
        mock_sie_client: object,
        haystack_documents: list[Document],
        test_query: str,
    ) -> None:
        """Test that per-call top_k overrides configured value."""
        ranker = SIERanker(model="test-reranker", top_k=2)
        ranker._client = mock_sie_client

        result = ranker.run(query=test_query, documents=haystack_documents, top_k=3)

        assert len(result["documents"]) == 3

    def test_results_sorted_by_score(
        self,
        mock_sie_client: object,
        haystack_documents: list[Document],
        test_query: str,
    ) -> None:
        """Test that results are sorted by score descending."""
        ranker = SIERanker(model="test-reranker")
        ranker._client = mock_sie_client

        result = ranker.run(query=test_query, documents=haystack_documents)

        scores = [doc.meta["score"] for doc in result["documents"]]
        assert scores == sorted(scores, reverse=True)

    def test_custom_model(self, mock_sie_client: object, haystack_documents: list[Document]) -> None:
        """Test using a custom model name."""
        ranker = SIERanker(model="custom/reranker-model")
        ranker._client = mock_sie_client

        ranker.run(query="test", documents=haystack_documents)

        call_args = mock_sie_client.score.call_args
        assert call_args[0][0] == "custom/reranker-model"

    def test_preserves_document_metadata(self, mock_sie_client: object) -> None:
        """Test that original document metadata is preserved."""
        ranker = SIERanker(model="test-reranker")
        ranker._client = mock_sie_client

        docs = [
            Document(content="Test", meta={"source": "docs", "important": True}),
        ]
        result = ranker.run(query="test", documents=docs)

        assert result["documents"][0].meta["source"] == "docs"
        assert result["documents"][0].meta["important"] is True
        assert "score" in result["documents"][0].meta

    def test_preserves_document_embedding(self, mock_sie_client: object) -> None:
        """Test that document embeddings are preserved."""
        ranker = SIERanker(model="test-reranker")
        ranker._client = mock_sie_client

        original_embedding = [0.1, 0.2, 0.3]
        docs = [Document(content="Test", embedding=original_embedding)]

        result = ranker.run(query="test", documents=docs)

        assert result["documents"][0].embedding == original_embedding
