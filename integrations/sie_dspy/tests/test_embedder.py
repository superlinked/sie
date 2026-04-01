"""Unit tests for SIE DSPy embedders."""

from __future__ import annotations

import numpy as np
from sie_dspy import SIEEmbedder, SIESparseEmbedder


class TestSIEEmbedder:
    """Tests for SIEEmbedder.

    DSPy use case: Embeddings for FAISS-based retrieval in RAG pipelines,
    program optimization, and few-shot example selection.
    """

    def test_embed_single_query(self, mock_sie_client: object) -> None:
        """Test embedding a single query string."""
        embedder = SIEEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder("What is machine learning?")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result.shape) == 1  # 1D array for single query
        assert result.shape[0] == 384  # Default embedding dim

    def test_embed_corpus(self, mock_sie_client: object, ml_corpus: list[str]) -> None:
        """Test embedding a corpus of documents."""
        embedder = SIEEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder(ml_corpus)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result.shape) == 2  # 2D array for corpus
        assert result.shape[0] == len(ml_corpus)
        assert result.shape[1] == 384

    def test_embeddings_are_normalized(self, mock_sie_client: object) -> None:
        """Test that embeddings are unit vectors (normalized)."""
        embedder = SIEEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder(["Test document"])

        # Check norm is approximately 1
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 0.01

    def test_deterministic_embeddings(self, mock_sie_client: object) -> None:
        """Test that same text produces same embedding."""
        embedder = SIEEmbedder(model="test-model")
        embedder._client = mock_sie_client

        text = "Machine learning is fascinating."
        result1 = embedder(text)
        result2 = embedder(text)

        np.testing.assert_array_almost_equal(result1, result2)

    def test_different_texts_different_embeddings(self, mock_sie_client: object) -> None:
        """Test that different texts produce different embeddings."""
        embedder = SIEEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder(["Machine learning", "Weather forecast"])

        # Embeddings should be different
        assert not np.allclose(result[0], result[1])

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        embedder = SIEEmbedder(model="custom/embedding-model")
        embedder._client = mock_sie_client

        embedder("test query")

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/embedding-model"

    def test_query_vs_document_embedding(self, mock_sie_client: object) -> None:
        """Test that queries use options.is_query=True and corpus uses default (no is_query)."""
        embedder = SIEEmbedder(model="test-model")
        embedder._client = mock_sie_client

        # Single string = query
        embedder("search query")
        query_call = mock_sie_client.encode.call_args
        assert query_call.kwargs.get("options", {}).get("is_query") is True

        # A list of texts should be treated as corpus (no is_query option = default False)
        embedder(["doc1", "doc2"])
        corpus_call = mock_sie_client.encode.call_args
        # Corpus calls don't pass is_query (defaults to False on server)
        assert corpus_call.kwargs.get("options") is None

    def test_lazy_client_initialization(self) -> None:
        """Test that client is not created until first use."""
        embedder = SIEEmbedder(model="test-model")

        # Client should not be initialized yet
        assert embedder._client is None

    def test_callable_interface(self, mock_sie_client: object) -> None:
        """Test that embedder is callable (required by DSPy)."""
        embedder = SIEEmbedder(model="test-model")
        embedder._client = mock_sie_client

        # Should be callable
        assert callable(embedder)

        # Should work with __call__
        result = embedder.__call__("test")
        assert isinstance(result, np.ndarray)


class TestSIESparseEmbedder:
    """Tests for SIESparseEmbedder.

    Use alongside SIEEmbedder for hybrid search workflows.
    """

    def test_embed_documents_single(self, mock_sie_client: object) -> None:
        """Test embedding a single document."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.embed_documents(["Hello world"])

        assert len(result) == 1
        assert "indices" in result[0]
        assert "values" in result[0]
        assert len(result[0]["indices"]) == len(result[0]["values"])
        assert len(result[0]["indices"]) > 0

    def test_embed_documents_batch(self, mock_sie_client: object, ml_corpus: list[str]) -> None:
        """Test embedding multiple documents."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.embed_documents(ml_corpus)

        assert len(result) == len(ml_corpus)
        for sparse in result:
            assert "indices" in sparse
            assert "values" in sparse

    def test_embed_documents_empty(self, mock_sie_client: object) -> None:
        """Test embedding empty list returns empty."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.embed_documents([])

        assert result == []

    def test_embed_query(self, mock_sie_client: object) -> None:
        """Test embedding a query."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        result = embedder.embed_query("What is machine learning?")

        assert "indices" in result
        assert "values" in result
        assert len(result["indices"]) == len(result["values"])

    def test_embed_query_uses_is_query(self, mock_sie_client: object) -> None:
        """Test that embed_query sets is_query=True."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        embedder.embed_query("test query")

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options", {}).get("is_query") is True
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_embed_documents_no_is_query(self, mock_sie_client: object) -> None:
        """Test that embed_documents doesn't set is_query."""
        embedder = SIESparseEmbedder(model="test-model")
        embedder._client = mock_sie_client

        embedder.embed_documents(["test doc"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options") is None
        assert call_kwargs.get("output_types") == ["sparse"]

    def test_custom_model(self, mock_sie_client: object) -> None:
        """Test using a custom model name."""
        embedder = SIESparseEmbedder(model="custom/sparse-model")
        embedder._client = mock_sie_client

        embedder.embed_query("test")

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/sparse-model"

    def test_lazy_client_initialization(self) -> None:
        """Test that client is not created until first use."""
        embedder = SIESparseEmbedder(model="test-model")

        assert embedder._client is None
