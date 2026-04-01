"""Unit tests for SIENodePostprocessor."""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from sie_llamaindex import SIENodePostprocessor


class TestSIENodePostprocessor:
    """Tests for SIENodePostprocessor class."""

    def test_postprocess_nodes(self, mock_sie_client: object, test_documents: list[str]) -> None:
        """Test reranking nodes."""
        postprocessor = SIENodePostprocessor(model="test-reranker")
        postprocessor._client = mock_sie_client

        nodes = [NodeWithScore(node=TextNode(text=text), score=0.5) for text in test_documents]
        query_bundle = QueryBundle(query_str="test query")

        result = postprocessor._postprocess_nodes(nodes, query_bundle)

        assert len(result) > 0
        # All results should have scores
        for node in result:
            assert node.score is not None
            assert isinstance(node.score, float)

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
