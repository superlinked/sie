"""Unit tests for SIE LanceDB reranker."""

from __future__ import annotations

from typing import Any
from unittest.mock import NonCallableMagicMock

import lancedb
import pyarrow as pa
import pytest
from sie_lancedb import SIEReranker


@pytest.fixture
def db(tmp_path):
    """Ephemeral in-process LanceDB instance."""
    return lancedb.connect(tmp_path / "test.lance")


class TestSIEReranker:
    """Tests for SIEReranker."""

    def test_rerank_vector(self, mock_sie_client: object, sample_table: pa.Table) -> None:
        """Reranking vector results adds _relevance_score and sorts."""
        reranker = SIEReranker(model="test-model")
        reranker._client = mock_sie_client

        result = reranker.rerank_vector("What is vector search?", sample_table)

        assert isinstance(result, pa.Table)
        assert "_relevance_score" in result.column_names
        # Should be sorted descending by relevance
        scores = result.column("_relevance_score").to_pylist()
        assert scores == sorted(scores, reverse=True)

    def test_rerank_fts(self, mock_sie_client: object, sample_table: pa.Table) -> None:
        """Reranking FTS results adds _relevance_score."""
        reranker = SIEReranker(model="test-model")
        reranker._client = mock_sie_client

        result = reranker.rerank_fts("What is vector search?", sample_table)

        assert "_relevance_score" in result.column_names
        scores = result.column("_relevance_score").to_pylist()
        assert scores == sorted(scores, reverse=True)

    def test_rerank_hybrid(self, mock_sie_client: object) -> None:
        """Reranking hybrid results merges and scores."""
        reranker = SIEReranker(model="test-model")
        reranker._client = mock_sie_client

        vector_results = pa.table(
            {
                "text": ["Vector search is fast.", "Embeddings represent meaning."],
                "_rowid": [0, 1],
                "_distance": [0.1, 0.3],
            }
        )
        fts_results = pa.table(
            {
                "text": ["Embeddings represent meaning.", "Full text search uses BM25."],
                "_rowid": [1, 2],
                "_score": [2.5, 1.8],
            }
        )

        result = reranker.rerank_hybrid("embeddings", vector_results, fts_results)

        assert isinstance(result, pa.Table)
        assert "_relevance_score" in result.column_names
        # Merged results should have unique rows
        assert result.num_rows >= 2

    def test_rerank_empty_table(self, mock_sie_client: object) -> None:
        """Empty table returns empty table with _relevance_score column."""
        reranker = SIEReranker(model="test-model")
        reranker._client = mock_sie_client

        empty = pa.table({"text": pa.array([], type=pa.utf8()), "_rowid": pa.array([], type=pa.int64())})
        result = reranker.rerank_vector("query", empty)

        assert result.num_rows == 0
        assert "_relevance_score" in result.column_names

    def test_return_score_relevance(self, mock_sie_client: object, sample_table: pa.Table) -> None:
        """return_score='relevance' drops _distance and _score columns."""
        reranker = SIEReranker(model="test-model", return_score="relevance")
        reranker._client = mock_sie_client

        table_with_distance = sample_table.append_column(
            "_distance", pa.array([0.1, 0.2, 0.3, 0.4, 0.5], type=pa.float32())
        )

        result = reranker.rerank_vector("query", table_with_distance)

        assert "_relevance_score" in result.column_names
        assert "_distance" not in result.column_names

    def test_return_score_all(self, mock_sie_client: object, sample_table: pa.Table) -> None:
        """return_score='all' keeps _distance and _score columns."""
        reranker = SIEReranker(model="test-model", return_score="all")
        reranker._client = mock_sie_client

        table_with_distance = sample_table.append_column(
            "_distance", pa.array([0.1, 0.2, 0.3, 0.4, 0.5], type=pa.float32())
        )

        result = reranker.rerank_vector("query", table_with_distance)

        assert "_relevance_score" in result.column_names
        assert "_distance" in result.column_names

    def test_custom_column(self, mock_sie_client: object) -> None:
        """Custom column name is used for text extraction."""
        reranker = SIEReranker(model="test-model", column="content")
        reranker._client = mock_sie_client

        table = pa.table(
            {
                "content": ["Hello world", "Goodbye world"],
                "_rowid": [0, 1],
            }
        )

        result = reranker.rerank_vector("hello", table)

        assert result.num_rows == 2
        assert "_relevance_score" in result.column_names

    def test_custom_model_forwarded(self, mock_sie_client: object, sample_table: pa.Table) -> None:
        """Model name is forwarded to SIE client."""
        reranker = SIEReranker(model="custom/reranker-model")
        reranker._client = mock_sie_client

        reranker.rerank_vector("query", sample_table)

        call_args = mock_sie_client.score.call_args
        assert call_args[0][0] == "custom/reranker-model"

    def test_lazy_client_initialization(self) -> None:
        """Client is not created until first use."""
        reranker = SIEReranker(model="test-model")
        assert reranker._client is None

    def test_all_rows_scored(self, mock_sie_client: object, sample_table: pa.Table) -> None:
        """Every row gets a relevance score."""
        reranker = SIEReranker(model="test-model")
        reranker._client = mock_sie_client

        result = reranker.rerank_vector("query", sample_table)

        scores = result.column("_relevance_score").to_pylist()
        assert len(scores) == sample_table.num_rows
        assert all(isinstance(s, float) for s in scores)
        assert all(s > 0 for s in scores)

    def test_rerank_ranks_through_sie_client(self, score_stub_server: Any) -> None:
        """Scores returned by a real ``SIEClient.score()`` call rank the matching row first."""
        texts = [
            "The weather today is sunny with clear skies.",
            "Python is a popular programming language.",
            "Nearest neighbor search uses distance metrics.",
            "Vector similarity search finds similar embeddings.",
        ]
        table = pa.table({"text": texts, "_rowid": [0, 1, 2, 3]})
        reranker = SIEReranker(base_url=score_stub_server.url, model="test-reranker")

        result = reranker.rerank_vector("vector similarity search", table)

        assert result.column("text").to_pylist()[:2] == [texts[3], texts[2]]
        assert result.column("_relevance_score").to_pylist() == [3.0, 1.0, 0.0, 0.0]

    def test_rerank_skips_malformed_item_id(self, mock_sie_client: NonCallableMagicMock) -> None:
        """Malformed item_ids are skipped (no crash, no misassignment)."""
        mock_sie_client.score.side_effect = None
        mock_sie_client.score.return_value = {
            "model": "test-model",
            "scores": [
                {"item_id": "1", "score": 0.5, "rank": 0},
                {"item_id": "item-0", "score": 0.95, "rank": 1},
                {"item_id": "-1", "score": 0.9, "rank": 2},
                {"item_id": "99", "score": 0.7, "rank": 3},
                {"score": 0.6, "rank": 4},
                {"item_id": 1.5, "score": 0.99, "rank": 5},
                {"item_id": True, "score": 0.98, "rank": 6},
            ],
        }
        reranker = SIEReranker(model="test-model")
        reranker._client = mock_sie_client
        table = pa.table({"text": ["alpha", "bravo", "charlie"], "_rowid": [0, 1, 2]})

        result = reranker.rerank_vector("query", table)

        by_text = dict(
            zip(result.column("text").to_pylist(), result.column("_relevance_score").to_pylist(), strict=True)
        )
        assert by_text == {"bravo": 0.5, "alpha": 0.0, "charlie": 0.0}
        assert result.column("text").to_pylist()[0] == "bravo"

    def test_rerank_on_real_lance_table(self, mock_sie_client: object, db) -> None:
        """Reranker works with a real LanceDB table's search results."""
        import numpy as np

        # Create a table with pre-computed vectors
        rng = np.random.default_rng(42)
        data = [
            {"id": i, "text": text, "vector": rng.standard_normal(8).astype(np.float32).tolist()}
            for i, text in enumerate(
                [
                    "Vector search finds similar items.",
                    "The weather is sunny.",
                    "Embedding models convert text to vectors.",
                    "Python is popular for ML.",
                ]
            )
        ]
        table = db.create_table("rerank_test", data=data, mode="overwrite")

        # Run a vector search to get pa.Table results
        query_vec = rng.standard_normal(8).astype(np.float32).tolist()
        search_results = table.search(query_vec).limit(4).to_arrow()

        # Rerank those results with SIE
        reranker = SIEReranker(model="test-model")
        reranker._client = mock_sie_client

        reranked = reranker.rerank_vector("What are embeddings?", search_results)

        assert "_relevance_score" in reranked.column_names
        assert reranked.num_rows == 4
        scores = reranked.column("_relevance_score").to_pylist()
        assert scores == sorted(scores, reverse=True)
