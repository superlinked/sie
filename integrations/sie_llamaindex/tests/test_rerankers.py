"""Unit tests for SIENodePostprocessor."""

from __future__ import annotations

from collections import Counter

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from sie_llamaindex import SIENodePostprocessor

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


class TestSIENodePostprocessor:
    """Tests for SIENodePostprocessor class."""

    def test_postprocess_nodes(self, mock_sie_client: object, test_documents: list[str]) -> None:
        """Test reranking nodes."""
        postprocessor = SIENodePostprocessor(model="test-reranker")
        postprocessor._client = mock_sie_client

        nodes = [NodeWithScore(node=TextNode(text=text), score=0.5) for text in test_documents]
        query_bundle = QueryBundle(query_str="test query")

        result = postprocessor._postprocess_nodes(nodes, query_bundle)

        # Every input node comes back exactly as many times as it went in
        # (Counter checks multiplicity, not just membership). The envelope bug
        # iterated the ScoreResult dict's keys, returning nodes[0] duplicated.
        assert len(result) == len(nodes)
        assert Counter(n.node.get_content() for n in result) == Counter(n.node.get_content() for n in nodes)
        scores = [n.score for n in result]
        assert all(isinstance(s, float) for s in scores)
        # Distinct and descending — the bug produced identical 0.0 scores.
        assert scores == sorted(scores, reverse=True)
        assert len(set(scores)) == len(scores)

    def test_postprocess_nodes_maps_scores_by_item_id(self, mock_sie_client: object) -> None:
        """Top-ranked node is the relevant one, scores mapped by item_id."""
        nodes = [NodeWithScore(node=TextNode(text=f"doc-{i}"), score=0.5) for i in range(5)]
        mock_sie_client.score.side_effect = None
        mock_sie_client.score.return_value = _RANKED_ENVELOPE
        postprocessor = SIENodePostprocessor(model="test-reranker")
        postprocessor._client = mock_sie_client

        result = postprocessor._postprocess_nodes(nodes, QueryBundle(query_str="query"))

        assert [n.node.get_content() for n in result] == ["doc-3", "doc-1", "doc-4", "doc-0", "doc-2"]
        assert result[0].score == 0.9
        assert [n.score for n in result] == [0.9, 0.7, 0.5, 0.3, 0.1]

    def test_postprocess_nodes_skips_malformed_item_id(self, mock_sie_client: object) -> None:
        """Malformed item_ids are skipped (no crash, no misassignment)."""
        nodes = [NodeWithScore(node=TextNode(text=f"doc-{i}"), score=0.5) for i in range(3)]
        mock_sie_client.score.side_effect = None
        mock_sie_client.score.return_value = _MALFORMED_ENVELOPE
        postprocessor = SIENodePostprocessor(model="test-reranker")
        postprocessor._client = mock_sie_client

        result = postprocessor._postprocess_nodes(nodes, QueryBundle(query_str="query"))

        assert len(result) == 3
        by_content = {n.node.get_content(): n.score for n in result}
        assert by_content == {"doc-1": 0.8, "doc-0": 0.0, "doc-2": 0.0}
        assert result[0].node.get_content() == "doc-1"

    def test_postprocess_nodes_empty(self, mock_sie_client: object) -> None:
        """Test reranking empty list returns empty."""
        postprocessor = SIENodePostprocessor(model="test-reranker")
        postprocessor._client = mock_sie_client

        result = postprocessor._postprocess_nodes([], QueryBundle(query_str="test"))

        assert result == []

    def test_postprocess_nodes_no_query(self, mock_sie_client: object) -> None:
        """Test reranking without query returns original nodes."""
        postprocessor = SIENodePostprocessor(model="test-reranker")
        postprocessor._client = mock_sie_client

        nodes = [NodeWithScore(node=TextNode(text="test"), score=0.5)]

        result = postprocessor._postprocess_nodes(nodes, query_bundle=None)

        assert len(result) == 1
        assert result[0].node.text == "test"

    def test_postprocess_nodes_top_n(self, mock_sie_client: object, test_documents: list[str]) -> None:
        """Test reranking with top_n limit."""
        postprocessor = SIENodePostprocessor(model="test-reranker", top_n=2)
        postprocessor._client = mock_sie_client

        nodes = [NodeWithScore(node=TextNode(text=text), score=0.5) for text in test_documents]
        query_bundle = QueryBundle(query_str="test query")

        result = postprocessor._postprocess_nodes(nodes, query_bundle)

        assert len(result) <= 2

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        postprocessor = SIENodePostprocessor(model="custom/reranker-model")
        postprocessor._client = mock_sie_client

        nodes = [NodeWithScore(node=TextNode(text="test"), score=0.5)]
        query_bundle = QueryBundle(query_str="test query")

        postprocessor._postprocess_nodes(nodes, query_bundle)

        call_args = mock_sie_client.score.call_args
        assert call_args[0][0] == "custom/reranker-model"

    def test_class_name(self) -> None:
        """Test class_name returns correct identifier."""
        assert SIENodePostprocessor.class_name() == "SIENodePostprocessor"
