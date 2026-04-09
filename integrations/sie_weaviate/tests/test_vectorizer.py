"""Unit tests for SIE Weaviate vectorizer."""

from __future__ import annotations

import numpy as np
import pytest
from sie_weaviate import SIENamedVectorizer, SIEVectorizer


class TestSIEVectorizer:
    """Tests for SIEVectorizer (dense embeddings)."""

    def test_embed_documents(self, mock_sie_client: object, sample_documents: list[str]) -> None:
        """Embedding multiple documents returns correct count and type."""
        vectorizer = SIEVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        vectors = vectorizer.embed_documents(sample_documents)

        assert isinstance(vectors, list)
        assert len(vectors) == len(sample_documents)
        for vec in vectors:
            assert isinstance(vec, list)
            assert all(isinstance(v, float) for v in vec)
            assert len(vec) == 384

    def test_embed_empty_list_no_client_call(self, mock_sie_client: object) -> None:
        """Empty input returns empty list without calling SIE."""
        vectorizer = SIEVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        result = vectorizer.embed_documents([])

        assert result == []
        mock_sie_client.encode.assert_not_called()

    def test_embed_query(self, mock_sie_client: object) -> None:
        """Embedding a single query returns a flat list of floats."""
        vectorizer = SIEVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        vec = vectorizer.embed_query("search text")

        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)
        assert len(vec) == 384

    def test_deterministic_embeddings(self, mock_sie_client: object) -> None:
        """Same text produces same embedding."""
        vectorizer = SIEVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        vectors1 = vectorizer.embed_documents(["Machine learning"])
        vectors2 = vectorizer.embed_documents(["Machine learning"])

        assert vectors1[0] == vectors2[0]

    def test_different_documents_different_embeddings(self, mock_sie_client: object) -> None:
        """Different texts produce different embeddings."""
        vectorizer = SIEVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        vectors = vectorizer.embed_documents(["Machine learning", "Weather forecast"])

        assert vectors[0] != vectors[1]

    def test_custom_model_passed_to_encode(self, mock_sie_client: object) -> None:
        """Model name is forwarded to SIE client."""
        vectorizer = SIEVectorizer(model="custom/model")
        vectorizer._client = mock_sie_client

        vectorizer.embed_documents(["test"])

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/model"

    def test_output_types_set_to_dense(self, mock_sie_client: object) -> None:
        """encode() is called with output_types=["dense"]."""
        vectorizer = SIEVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        vectorizer.embed_documents(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_types") == ["dense"]

    def test_lazy_client_initialization(self) -> None:
        """Client is not created until first use."""
        vectorizer = SIEVectorizer(model="test-model")
        assert vectorizer._client is None

    def test_normalized_embeddings(self, mock_sie_client: object) -> None:
        """Embeddings are unit vectors."""
        vectorizer = SIEVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        vectors = vectorizer.embed_documents(["Test document"])

        norm = float(np.linalg.norm(vectors[0]))
        assert abs(norm - 1.0) < 0.01

    def test_embed_query_passes_is_query(self, mock_sie_client: object) -> None:
        """embed_query() passes is_query=True for instruction-tuned models."""
        vectorizer = SIEVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        vectorizer.embed_query("search text")

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options") == {"is_query": True}

    def test_embed_documents_does_not_pass_is_query(self, mock_sie_client: object) -> None:
        """embed_documents() does not pass is_query flag."""
        vectorizer = SIEVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        vectorizer.embed_documents(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert "options" not in call_kwargs or call_kwargs.get("options") is None

    def test_instruction_forwarded(self, mock_sie_client: object) -> None:
        """Instruction parameter is passed to encode()."""
        vectorizer = SIEVectorizer(model="test-model", instruction="Represent this document:")
        vectorizer._client = mock_sie_client

        vectorizer.embed_documents(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("instruction") == "Represent this document:"

    def test_output_dtype_forwarded(self, mock_sie_client: object) -> None:
        """output_dtype parameter is passed to encode()."""
        vectorizer = SIEVectorizer(model="test-model", output_dtype="float16")
        vectorizer._client = mock_sie_client

        vectorizer.embed_documents(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_dtype") == "float16"

    def test_instruction_forwarded_in_query(self, mock_sie_client: object) -> None:
        """Instruction is also passed in embed_query()."""
        vectorizer = SIEVectorizer(model="test-model", instruction="Represent this query:")
        vectorizer._client = mock_sie_client

        vectorizer.embed_query("test")

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("instruction") == "Represent this query:"
        assert call_kwargs.get("options") == {"is_query": True}

    def test_extract_dense_raises_on_none(self) -> None:
        """extract_dense raises ValueError when dense is missing."""
        from sie_sdk.encoding import dense_embedding

        with pytest.raises(ValueError, match="missing dense embedding"):
            dense_embedding({})


class TestSIENamedVectorizer:
    """Tests for SIENamedVectorizer (multi-output for named vectors)."""

    def test_embed_documents_returns_named(self, mock_sie_client: object, sample_documents: list[str]) -> None:
        """Returns dicts with requested vector types."""
        vectorizer = SIENamedVectorizer(model="test-model", output_types=["dense", "multivector"])
        vectorizer._client = mock_sie_client

        named = vectorizer.embed_documents(sample_documents)

        assert isinstance(named, list)
        assert len(named) == len(sample_documents)
        for item in named:
            assert isinstance(item, dict)
            assert "dense" in item
            assert "multivector" in item
            assert isinstance(item["dense"], list)
            assert isinstance(item["multivector"], list)
            # multivector is list of token vectors
            assert isinstance(item["multivector"][0], list)

    def test_embed_query_returns_named(self, mock_sie_client: object) -> None:
        """Query embedding returns dict with all vector types."""
        vectorizer = SIENamedVectorizer(model="test-model", output_types=["dense", "multivector"])
        vectorizer._client = mock_sie_client

        named = vectorizer.embed_query("search text")

        assert isinstance(named, dict)
        assert "dense" in named
        assert "multivector" in named

    def test_empty_documents(self, mock_sie_client: object) -> None:
        """Empty input returns empty list without calling SIE."""
        vectorizer = SIENamedVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        result = vectorizer.embed_documents([])

        assert result == []
        mock_sie_client.encode.assert_not_called()

    def test_output_types_forwarded(self, mock_sie_client: object) -> None:
        """Requested output_types are passed to encode()."""
        vectorizer = SIENamedVectorizer(model="test-model", output_types=["dense", "multivector"])
        vectorizer._client = mock_sie_client

        vectorizer.embed_documents(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_types") == ["dense", "multivector"]

    def test_default_output_types(self, mock_sie_client: object) -> None:
        """Default output types are dense only."""
        vectorizer = SIENamedVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        vectorizer.embed_documents(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_types") == ["dense"]

    def test_lazy_client_initialization(self) -> None:
        """Client is not created until first use."""
        vectorizer = SIENamedVectorizer(model="test-model")
        assert vectorizer._client is None

    def test_embed_query_passes_is_query(self, mock_sie_client: object) -> None:
        """embed_query() passes is_query=True."""
        vectorizer = SIENamedVectorizer(model="test-model")
        vectorizer._client = mock_sie_client

        vectorizer.embed_query("search text")

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options") == {"is_query": True}

    def test_instruction_forwarded(self, mock_sie_client: object) -> None:
        """Instruction parameter is passed to encode()."""
        vectorizer = SIENamedVectorizer(model="test-model", instruction="Represent:")
        vectorizer._client = mock_sie_client

        vectorizer.embed_documents(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("instruction") == "Represent:"

    def test_output_dtype_forwarded(self, mock_sie_client: object) -> None:
        """output_dtype parameter is passed to encode()."""
        vectorizer = SIENamedVectorizer(model="test-model", output_dtype="float16")
        vectorizer._client = mock_sie_client

        vectorizer.embed_documents(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_dtype") == "float16"

    def test_multivector_shape(self, mock_sie_client: object) -> None:
        """Multivector output has correct nested list structure."""
        vectorizer = SIENamedVectorizer(model="test-model", output_types=["multivector"])
        vectorizer._client = mock_sie_client

        named = vectorizer.embed_documents(["test"])

        mv = named[0]["multivector"]
        assert isinstance(mv, list)
        assert len(mv) == 8  # MULTIVECTOR_TOKENS from conftest
        for token_vec in mv:
            assert isinstance(token_vec, list)
            assert len(token_vec) == 128  # MULTIVECTOR_DIM from conftest
            assert all(isinstance(v, float) for v in token_vec)

    def test_extract_multivector_from_numpy(self) -> None:
        """extract_multivector converts 2D numpy array to list of lists."""
        from sie_sdk.encoding import multivector_embedding

        mv = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = multivector_embedding(mv)

        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_extract_multivector_from_list_of_arrays(self) -> None:
        """extract_multivector handles list of 1D numpy arrays."""
        from sie_sdk.encoding import multivector_embedding

        mv = [np.array([1.0, 2.0], dtype=np.float32), np.array([3.0, 4.0], dtype=np.float32)]
        result = multivector_embedding(mv)

        assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_dense_only_named(self, mock_sie_client: object) -> None:
        """Dense-only output works without multivector."""
        vectorizer = SIENamedVectorizer(model="test-model", output_types=["dense"])
        vectorizer._client = mock_sie_client

        named = vectorizer.embed_documents(["test"])

        assert "dense" in named[0]
        assert "multivector" not in named[0]
