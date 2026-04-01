"""Unit tests for SIEEmbedding and SIESparseEmbeddingFunction."""

from __future__ import annotations

import pytest
from sie_llamaindex import SIEEmbedding, SIESparseEmbeddingFunction


class TestSIEEmbedding:
    """Tests for SIEEmbedding class."""

    def test_get_text_embedding(self, mock_sie_client: object) -> None:
        """Test getting embedding for a single text."""
        embedding = SIEEmbedding(model_name="test-model")
        embedding._client = mock_sie_client

        result = embedding._get_text_embedding("Hello world")

        assert len(result) == 384  # Mock returns 384-dim
        assert all(isinstance(x, float) for x in result)

    def test_get_text_embeddings_batch(self, mock_sie_client: object) -> None:
        """Test getting embeddings for multiple texts."""
        embedding = SIEEmbedding(model_name="test-model")
        embedding._client = mock_sie_client

        texts = ["Hello", "World", "Test"]
        results = embedding._get_text_embeddings(texts)

        assert len(results) == 3
        for result in results:
            assert len(result) == 384

    def test_get_text_embeddings_empty(self, mock_sie_client: object) -> None:
        """Test getting embeddings for empty list."""
        embedding = SIEEmbedding(model_name="test-model")
        embedding._client = mock_sie_client

        results = embedding._get_text_embeddings([])

        assert results == []

    def test_get_query_embedding(self, mock_sie_client: object) -> None:
        """Test getting query embedding."""
        embedding = SIEEmbedding(model_name="test-model")
        embedding._client = mock_sie_client

        result = embedding._get_query_embedding("What is the meaning of life?")

        assert len(result) == 384
        mock_sie_client.encode.assert_called()
        # Verify is_query=True was passed in options
        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options", {}).get("is_query") is True

    def test_custom_model_name(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        embedding = SIEEmbedding(model_name="custom/embedding-model")
        embedding._client = mock_sie_client

        embedding._get_text_embedding("test")

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/embedding-model"

    def test_custom_instruction(self, mock_sie_client: object) -> None:
        """Test passing custom instruction."""
        embedding = SIEEmbedding(
            model_name="test-model",
            instruction="Represent this for search:",
        )
        embedding._client = mock_sie_client

        embedding._get_text_embedding("test")

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("instruction") == "Represent this for search:"

    def test_class_name(self) -> None:
        """Test class_name returns correct identifier."""
        assert SIEEmbedding.class_name() == "SIEEmbedding"


class TestSIEEmbeddingAsync:
    """Tests for async SIEEmbedding methods."""

    @pytest.mark.asyncio
    async def test_aget_text_embedding(self, mock_sie_async_client: object) -> None:
        """Test async getting embedding for text."""
        embedding = SIEEmbedding(model_name="test-model")
        embedding._async_client = mock_sie_async_client

        result = await embedding._aget_text_embedding("Hello world")

        assert len(result) == 384
        mock_sie_async_client.encode.assert_called()

    @pytest.mark.asyncio
    async def test_aget_query_embedding(self, mock_sie_async_client: object) -> None:
        """Test async getting query embedding."""
        embedding = SIEEmbedding(model_name="test-model")
        embedding._async_client = mock_sie_async_client

        result = await embedding._aget_query_embedding("test query")

        assert len(result) == 384
        call_kwargs = mock_sie_async_client.encode.call_args.kwargs
        assert call_kwargs.get("options", {}).get("is_query") is True


class TestSIESparseEmbeddingFunction:
    """Tests for SIESparseEmbeddingFunction class.

    Used with LlamaIndex vector stores that support hybrid search.
    """

    def test_encode_queries_single(self, mock_sie_client: object) -> None:
        """Test encoding a single query."""
        sparse_fn = SIESparseEmbeddingFunction(model_name="test-model")
        sparse_fn._client = mock_sie_client

        indices, values = sparse_fn.encode_queries(["Hello world"])

        assert len(indices) == 1
        assert len(values) == 1
        assert len(indices[0]) == len(values[0])
        assert len(indices[0]) > 0

    def test_encode_queries_batch(self, mock_sie_client: object) -> None:
        """Test encoding multiple queries."""
        sparse_fn = SIESparseEmbeddingFunction(model_name="test-model")
        sparse_fn._client = mock_sie_client

        texts = ["Hello", "World", "Test"]
        indices, values = sparse_fn.encode_queries(texts)

        assert len(indices) == 3
        assert len(values) == 3

    def test_encode_queries_empty(self, mock_sie_client: object) -> None:
        """Test encoding empty list returns empty tuples."""
        sparse_fn = SIESparseEmbeddingFunction(model_name="test-model")
        sparse_fn._client = mock_sie_client

        indices, values = sparse_fn.encode_queries([])

        assert indices == []
        assert values == []
        mock_sie_client.encode.assert_not_called()

    def test_encode_documents_single(self, mock_sie_client: object) -> None:
        """Test encoding a single document."""
        sparse_fn = SIESparseEmbeddingFunction(model_name="test-model")
        sparse_fn._client = mock_sie_client

        indices, values = sparse_fn.encode_documents(["Hello world"])

        assert len(indices) == 1
        assert len(values) == 1

    def test_encode_documents_batch(self, mock_sie_client: object) -> None:
        """Test encoding multiple documents."""
        sparse_fn = SIESparseEmbeddingFunction(model_name="test-model")
        sparse_fn._client = mock_sie_client

        texts = ["Hello", "World", "Test"]
        indices, values = sparse_fn.encode_documents(texts)

        assert len(indices) == 3
        assert len(values) == 3

    def test_encode_documents_empty(self, mock_sie_client: object) -> None:
        """Test encoding empty list returns empty tuples."""
        sparse_fn = SIESparseEmbeddingFunction(model_name="test-model")
        sparse_fn._client = mock_sie_client

        indices, values = sparse_fn.encode_documents([])

        assert indices == []
        assert values == []
        mock_sie_client.encode.assert_not_called()

    def test_encode_queries_uses_is_query(self, mock_sie_client: object) -> None:
        """Test that encode_queries sets is_query=True."""
        sparse_fn = SIESparseEmbeddingFunction(model_name="test-model")
        sparse_fn._client = mock_sie_client

        sparse_fn.encode_queries(["test query"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options", {}).get("is_query") is True
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_encode_documents_no_is_query(self, mock_sie_client: object) -> None:
        """Test that encode_documents doesn't set is_query."""
        sparse_fn = SIESparseEmbeddingFunction(model_name="test-model")
        sparse_fn._client = mock_sie_client

        sparse_fn.encode_documents(["test doc"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        # Documents don't set is_query
        assert call_kwargs.get("options") is None
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        sparse_fn = SIESparseEmbeddingFunction(model_name="custom/model-name")
        sparse_fn._client = mock_sie_client

        sparse_fn.encode_queries(["test"])

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/model-name"

    def test_lazy_client_initialization(self) -> None:
        """Test that client is not created until first use."""
        sparse_fn = SIESparseEmbeddingFunction(model_name="test-model")

        assert sparse_fn._client is None
