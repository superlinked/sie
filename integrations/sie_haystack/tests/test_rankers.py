"""Unit tests for SIERanker component."""

from __future__ import annotations

from haystack import Document
from sie_haystack import SIERanker


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
        assert len(result["documents"]) == len(haystack_documents)
        # All results should have scores in metadata
        for doc in result["documents"]:
            assert "score" in doc.meta
            assert isinstance(doc.meta["score"], float)

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
