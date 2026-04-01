"""Unit tests for SIE ChromaDB embedding functions."""

from __future__ import annotations

import contextlib

import numpy as np
from sie_chroma import SIEEmbeddingFunction, SIESparseEmbeddingFunction


class TestSIEEmbeddingFunction:
    """Tests for SIEEmbeddingFunction.

    ChromaDB use case: Vector database for semantic search, RAG pipelines,
    and similarity-based retrieval.

    Note: ChromaDB's Embeddings type is List[numpy.ndarray], so embeddings
    are expected to be numpy arrays, not Python lists.
    """

    def test_embed_documents(self, mock_sie_client: object, sample_documents: list[str]) -> None:
        """Test embedding documents for collection storage."""
        embedding_function = SIEEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        embeddings = embedding_function(sample_documents)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_documents)
        # ChromaDB expects numpy arrays for each embedding
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 384  # Default dim

    def test_embed_empty_list_no_client_call(self, mock_sie_client: object) -> None:
        """Test that empty input doesn't call the client.

        Note: ChromaDB validates against empty embeddings, so we just verify
        that our implementation short-circuits before calling the client.
        """
        embedding_function = SIEEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        # Call with empty - our implementation returns [] before calling client
        # But ChromaDB's validation might throw, so we catch it
        with contextlib.suppress(ValueError):
            embedding_function([])

        # The key test: client.encode should NOT have been called
        mock_sie_client.encode.assert_not_called()

    def test_embed_single_document(self, mock_sie_client: object) -> None:
        """Test embedding a single document."""
        embedding_function = SIEEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        embeddings = embedding_function(["Single document"])

        assert len(embeddings) == 1
        assert isinstance(embeddings[0], np.ndarray)
        assert len(embeddings[0]) == 384

    def test_deterministic_embeddings(self, mock_sie_client: object) -> None:
        """Test that same text produces same embedding."""
        embedding_function = SIEEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        docs = ["Machine learning is fascinating."]
        embeddings1 = embedding_function(docs)
        embeddings2 = embedding_function(docs)

        np.testing.assert_array_equal(embeddings1[0], embeddings2[0])

    def test_different_documents_different_embeddings(self, mock_sie_client: object) -> None:
        """Test that different documents produce different embeddings."""
        embedding_function = SIEEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        embeddings = embedding_function(["Machine learning", "Weather forecast"])

        # Embeddings should be different
        assert not np.array_equal(embeddings[0], embeddings[1])

    def test_custom_model(self, mock_sie_client: object, sample_documents: list[str]) -> None:
        """Test using a custom model name."""
        embedding_function = SIEEmbeddingFunction(model="custom/embedding-model")
        embedding_function._client = mock_sie_client

        embedding_function(sample_documents[:1])

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/embedding-model"

    def test_lazy_client_initialization(self) -> None:
        """Test that client is not created until first use."""
        embedding_function = SIEEmbeddingFunction(model="test-model")

        # Client should not be initialized yet
        assert embedding_function._client is None

    def test_embedding_function_interface(self, mock_sie_client: object, sample_documents: list[str]) -> None:
        """Test that embedding function follows ChromaDB interface."""
        embedding_function = SIEEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        # Should be callable
        assert callable(embedding_function)

        # Should accept Documents (list[str])
        result = embedding_function(sample_documents)

        # Should return Embeddings (list[numpy.ndarray])
        assert isinstance(result, list)
        assert all(isinstance(e, np.ndarray) for e in result)
        assert all(e.dtype == np.float32 for e in result)

    def test_normalized_embeddings(self, mock_sie_client: object) -> None:
        """Test that embeddings are unit vectors (normalized)."""
        embedding_function = SIEEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        embeddings = embedding_function(["Test document"])

        # Calculate norm using numpy
        embedding = embeddings[0]
        norm = np.linalg.norm(embedding)

        # Should be approximately 1
        assert abs(norm - 1.0) < 0.01

    def test_output_types_set_to_dense(self, mock_sie_client: object, sample_documents: list[str]) -> None:
        """Test that output_types is set to dense for embeddings."""
        embedding_function = SIEEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        embedding_function(sample_documents[:1])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_types") == ["dense"]


class TestSIESparseEmbeddingFunction:
    """Tests for SIESparseEmbeddingFunction.

    Chroma Cloud use case: Hybrid search combining dense and sparse vectors
    using Reciprocal Rank Fusion (RRF).
    """

    def test_embed_documents_returns_sparse(self, mock_sie_client: object, sample_documents: list[str]) -> None:
        """Test embedding documents returns sparse embeddings."""
        embedding_function = SIESparseEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        embeddings = embedding_function(sample_documents)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_documents)
        # Each embedding should be a dict mapping int -> float
        for embedding in embeddings:
            assert isinstance(embedding, dict)
            assert len(embedding) > 0
            # Keys should be ints (token indices), values should be floats (weights)
            for key, value in embedding.items():
                assert isinstance(key, int)
                assert isinstance(value, float)

    def test_embed_empty_list_no_client_call(self, mock_sie_client: object) -> None:
        """Test that empty input doesn't call the client."""
        embedding_function = SIESparseEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        result = embedding_function([])

        assert result == []
        mock_sie_client.encode.assert_not_called()

    def test_embed_single_document(self, mock_sie_client: object) -> None:
        """Test embedding a single document."""
        embedding_function = SIESparseEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        embeddings = embedding_function(["Single document"])

        assert len(embeddings) == 1
        assert isinstance(embeddings[0], dict)
        assert len(embeddings[0]) > 0

    def test_deterministic_sparse_embeddings(self, mock_sie_client: object) -> None:
        """Test that same text produces same sparse embedding."""
        embedding_function = SIESparseEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        docs = ["Machine learning is fascinating."]
        embeddings1 = embedding_function(docs)
        embeddings2 = embedding_function(docs)

        assert embeddings1[0] == embeddings2[0]

    def test_different_documents_different_sparse_embeddings(self, mock_sie_client: object) -> None:
        """Test that different documents produce different sparse embeddings."""
        embedding_function = SIESparseEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        embeddings = embedding_function(["Machine learning", "Weather forecast"])

        # Embeddings should be different
        assert embeddings[0] != embeddings[1]

    def test_output_types_set_to_sparse(self, mock_sie_client: object, sample_documents: list[str]) -> None:
        """Test that output_types is set to sparse."""
        embedding_function = SIESparseEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        embedding_function(sample_documents[:1])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_custom_model(self, mock_sie_client: object, sample_documents: list[str]) -> None:
        """Test using a custom model name."""
        embedding_function = SIESparseEmbeddingFunction(model="custom/embedding-model")
        embedding_function._client = mock_sie_client

        embedding_function(sample_documents[:1])

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/embedding-model"

    def test_lazy_client_initialization(self) -> None:
        """Test that client is not created until first use."""
        embedding_function = SIESparseEmbeddingFunction(model="test-model")

        # Client should not be initialized yet
        assert embedding_function._client is None

    def test_sparse_embedding_function_interface(self, mock_sie_client: object, sample_documents: list[str]) -> None:
        """Test that sparse embedding function follows expected interface."""
        embedding_function = SIESparseEmbeddingFunction(model="test-model")
        embedding_function._client = mock_sie_client

        # Should be callable
        assert callable(embedding_function)

        # Should accept Documents (list[str])
        result = embedding_function(sample_documents)

        # Should return SparseEmbeddings (list[dict[int, float]])
        assert isinstance(result, list)
        assert all(isinstance(e, dict) for e in result)
        for embedding in result:
            for k, v in embedding.items():
                assert isinstance(k, int)
                assert isinstance(v, float)
