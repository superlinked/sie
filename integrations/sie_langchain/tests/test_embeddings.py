"""Unit tests for SIEEmbeddings and SIESparseEncoder."""

from __future__ import annotations

import pytest
from sie_langchain import SIEEmbeddings, SIESparseEncoder


class TestSIEEmbeddings:
    """Tests for SIEEmbeddings class."""

    def test_embed_documents_single(self, mock_sie_client: object) -> None:
        """Test embedding a single document."""
        embeddings = SIEEmbeddings(client=mock_sie_client, model="test-model")

        result = embeddings.embed_documents(["Hello world"])

        assert len(result) == 1
        assert len(result[0]) == 384  # Default mock embedding dim
        assert all(isinstance(x, float) for x in result[0])

    def test_embed_documents_batch(self, mock_sie_client: object, test_texts: list[str]) -> None:
        """Test embedding multiple documents."""
        embeddings = SIEEmbeddings(client=mock_sie_client, model="test-model")

        result = embeddings.embed_documents(test_texts)

        assert len(result) == len(test_texts)
        for vec in result:
            assert len(vec) == 384

    def test_embed_documents_empty(self, mock_sie_client: object) -> None:
        """Test embedding empty list returns empty."""
        embeddings = SIEEmbeddings(client=mock_sie_client, model="test-model")

        result = embeddings.embed_documents([])

        assert result == []

    def test_embed_query(self, mock_sie_client: object) -> None:
        """Test embedding a query."""
        embeddings = SIEEmbeddings(client=mock_sie_client, model="test-model")

        result = embeddings.embed_query("test query")

        assert len(result) == 384
        assert all(isinstance(x, float) for x in result)

    def test_embed_query_different_from_document(self, mock_sie_client: object) -> None:
        """Test that query and document embeddings are different (is_query flag)."""
        embeddings = SIEEmbeddings(client=mock_sie_client, model="test-model")

        query_result = embeddings.embed_query("test")
        doc_result = embeddings.embed_documents(["test"])

        # They should be embeddings of the same text but potentially different
        # due to is_query flag (model-dependent)
        assert len(query_result) == len(doc_result[0])

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        embeddings = SIEEmbeddings(client=mock_sie_client, model="custom/model-name")

        embeddings.embed_query("test")

        # Check that model was passed to client
        mock_sie_client.encode.assert_called()
        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/model-name"

    def test_custom_instruction(self, mock_sie_client: object) -> None:
        """Test using custom instruction prefix."""
        embeddings = SIEEmbeddings(client=mock_sie_client, model="test-model", instruction="Represent this for search:")

        embeddings.embed_query("test")

        call_kwargs = mock_sie_client.encode.call_args[1]
        assert call_kwargs["instruction"] == "Represent this for search:"


class TestSIEEmbeddingsAsync:
    """Tests for async SIEEmbeddings methods."""

    @pytest.mark.asyncio
    async def test_aembed_documents(self, mock_sie_async_client: object) -> None:
        """Test async embedding documents."""
        embeddings = SIEEmbeddings(async_client=mock_sie_async_client, model="test-model")

        result = await embeddings.aembed_documents(["Hello world"])

        assert len(result) == 1
        assert len(result[0]) == 384

    @pytest.mark.asyncio
    async def test_aembed_query(self, mock_sie_async_client: object) -> None:
        """Test async embedding query."""
        embeddings = SIEEmbeddings(async_client=mock_sie_async_client, model="test-model")

        result = await embeddings.aembed_query("test query")

        assert len(result) == 384

    @pytest.mark.asyncio
    async def test_aembed_documents_empty(self, mock_sie_async_client: object) -> None:
        """Test async embedding empty list returns empty."""
        embeddings = SIEEmbeddings(async_client=mock_sie_async_client, model="test-model")

        result = await embeddings.aembed_documents([])

        assert result == []


class TestSIESparseEncoder:
    """Tests for SIESparseEncoder class.

    Used with PineconeHybridSearchRetriever for hybrid search.
    """

    def test_encode_queries_single(self, mock_sie_client: object) -> None:
        """Test encoding a single query."""
        encoder = SIESparseEncoder(model="test-model")
        encoder._client = mock_sie_client

        result = encoder.encode_queries(["Hello world"])

        assert len(result) == 1
        assert "indices" in result[0]
        assert "values" in result[0]
        assert len(result[0]["indices"]) == len(result[0]["values"])
        assert len(result[0]["indices"]) > 0

    def test_encode_queries_batch(self, mock_sie_client: object, test_texts: list[str]) -> None:
        """Test encoding multiple queries."""
        encoder = SIESparseEncoder(model="test-model")
        encoder._client = mock_sie_client

        result = encoder.encode_queries(test_texts)

        assert len(result) == len(test_texts)
        for sparse in result:
            assert "indices" in sparse
            assert "values" in sparse

    def test_encode_queries_empty(self, mock_sie_client: object) -> None:
        """Test encoding empty list returns empty."""
        encoder = SIESparseEncoder(model="test-model")
        encoder._client = mock_sie_client

        result = encoder.encode_queries([])

        assert result == []
        mock_sie_client.encode.assert_not_called()

    def test_encode_documents_single(self, mock_sie_client: object) -> None:
        """Test encoding a single document."""
        encoder = SIESparseEncoder(model="test-model")
        encoder._client = mock_sie_client

        result = encoder.encode_documents(["Hello world"])

        assert len(result) == 1
        assert "indices" in result[0]
        assert "values" in result[0]

    def test_encode_documents_batch(self, mock_sie_client: object, test_texts: list[str]) -> None:
        """Test encoding multiple documents."""
        encoder = SIESparseEncoder(model="test-model")
        encoder._client = mock_sie_client

        result = encoder.encode_documents(test_texts)

        assert len(result) == len(test_texts)

    def test_encode_documents_empty(self, mock_sie_client: object) -> None:
        """Test encoding empty list returns empty."""
        encoder = SIESparseEncoder(model="test-model")
        encoder._client = mock_sie_client

        result = encoder.encode_documents([])

        assert result == []
        mock_sie_client.encode.assert_not_called()

    def test_encode_queries_uses_is_query(self, mock_sie_client: object) -> None:
        """Test that encode_queries sets is_query=True."""
        encoder = SIESparseEncoder(model="test-model")
        encoder._client = mock_sie_client

        encoder.encode_queries(["test query"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options", {}).get("is_query") is True
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_encode_documents_no_is_query(self, mock_sie_client: object) -> None:
        """Test that encode_documents doesn't set is_query."""
        encoder = SIESparseEncoder(model="test-model")
        encoder._client = mock_sie_client

        encoder.encode_documents(["test doc"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        # Documents don't set is_query
        assert call_kwargs.get("options") is None
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        encoder = SIESparseEncoder(model="custom/model-name")
        encoder._client = mock_sie_client

        encoder.encode_queries(["test"])

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/model-name"

    def test_lazy_client_initialization(self) -> None:
        """Test that client is not created until first use."""
        encoder = SIESparseEncoder(model="test-model")

        assert encoder._client is None
