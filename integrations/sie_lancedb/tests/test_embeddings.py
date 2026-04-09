"""Unit tests for SIE LanceDB embedding functions."""

from __future__ import annotations

import lancedb
import numpy as np
import pytest
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from sie_lancedb import SIEEmbeddingFunction, SIEMultiVectorEmbeddingFunction


@pytest.fixture
def db(tmp_path):
    """Ephemeral in-process LanceDB instance."""
    return lancedb.connect(tmp_path / "test.lance")


class TestSIEEmbeddingFunction:
    """Tests for SIEEmbeddingFunction (dense embeddings)."""

    def test_generate_embeddings(self, mock_sie_client: object, sample_texts: list[str]) -> None:
        """Generating embeddings returns correct count and shape."""
        func = SIEEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        results = func.generate_embeddings(sample_texts)

        assert isinstance(results, list)
        assert len(results) == len(sample_texts)
        for vec in results:
            assert isinstance(vec, list)
            assert all(isinstance(v, float) for v in vec)
            assert len(vec) == 384

    def test_generate_embeddings_empty(self, mock_sie_client: object) -> None:
        """Empty input returns empty list without calling SIE."""
        func = SIEEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        result = func.generate_embeddings([])

        assert result == []
        mock_sie_client.encode.assert_not_called()

    def test_ndims_from_metadata(self, mock_sie_client: object) -> None:
        """ndims() queries GET /v1/models/{model}, not inference."""
        func = SIEEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        assert func.ndims() == 384
        mock_sie_client.get_model.assert_called_once_with("test-model")
        mock_sie_client.encode.assert_not_called()

    def test_ndims_cached(self, mock_sie_client: object) -> None:
        """ndims() caches after first call."""
        func = SIEEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        func.ndims()
        func.ndims()

        assert mock_sie_client.get_model.call_count == 1

    def test_ndims_unknown_model_raises(self, mock_sie_client: object) -> None:
        """ndims() raises for unknown model."""
        from sie_sdk import RequestError

        func = SIEEmbeddingFunction(model="nonexistent/model")
        func._sie_client = mock_sie_client

        with pytest.raises(RequestError):
            func.ndims()

    def test_deterministic_embeddings(self, mock_sie_client: object) -> None:
        """Same text produces same embedding."""
        func = SIEEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        r1 = func.generate_embeddings(["Machine learning"])
        r2 = func.generate_embeddings(["Machine learning"])

        assert r1[0] == r2[0]

    def test_different_texts_different_embeddings(self, mock_sie_client: object) -> None:
        """Different texts produce different embeddings."""
        func = SIEEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        results = func.generate_embeddings(["Machine learning", "Weather forecast"])

        assert results[0] != results[1]

    def test_custom_model_forwarded(self, mock_sie_client: object) -> None:
        """Model name is forwarded to SIE client."""
        func = SIEEmbeddingFunction(model="custom/model")
        func._sie_client = mock_sie_client

        func.generate_embeddings(["test"])

        call_args = mock_sie_client.encode.call_args
        assert call_args[0][0] == "custom/model"

    def test_output_types_set_to_dense(self, mock_sie_client: object) -> None:
        """encode() is called with output_types=["dense"]."""
        func = SIEEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        func.generate_embeddings(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_types") == ["dense"]

    def test_instruction_forwarded(self, mock_sie_client: object) -> None:
        """Instruction parameter is passed to encode()."""
        func = SIEEmbeddingFunction(model="test-model", instruction="Represent this document:")
        func._sie_client = mock_sie_client

        func.generate_embeddings(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("instruction") == "Represent this document:"

    def test_output_dtype_forwarded(self, mock_sie_client: object) -> None:
        """output_dtype parameter is passed to encode()."""
        func = SIEEmbeddingFunction(model="test-model", output_dtype="float16")
        func._sie_client = mock_sie_client

        func.generate_embeddings(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_dtype") == "float16"

    def test_lazy_client_initialization(self) -> None:
        """Client is not created until first use."""
        func = SIEEmbeddingFunction(model="test-model")
        assert func._sie_client is None

    def test_extract_dense_raises_on_none(self) -> None:
        """_extract_dense raises ValueError when dense is missing."""
        func = SIEEmbeddingFunction(model="test-model")

        with pytest.raises(ValueError, match="missing dense embedding"):
            func._extract_dense({})

    def test_normalized_embeddings(self, mock_sie_client: object) -> None:
        """Embeddings are unit vectors (mock returns normalized)."""
        func = SIEEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        results = func.generate_embeddings(["Test document"])

        norm = float(np.linalg.norm(results[0]))
        assert abs(norm - 1.0) < 0.01

    def test_registered_in_registry(self) -> None:
        """SIEEmbeddingFunction is registered as 'sie' in the LanceDB registry."""
        cls = get_registry().get("sie")
        assert cls is SIEEmbeddingFunction

    def test_source_field_vector_field(self, mock_sie_client: object) -> None:
        """SourceField/VectorField annotations work for schema definition."""
        func = SIEEmbeddingFunction.create(model="test-model")
        func._sie_client = mock_sie_client
        func._ndims = 384

        class TestSchema(LanceModel):
            text: str = func.SourceField()
            vector: Vector(384) = func.VectorField()

        # Schema should have the correct fields
        schema = TestSchema.to_arrow_schema()
        assert "text" in schema.names
        assert "vector" in schema.names

    def test_auto_embed_table(self, mock_sie_client: object, db) -> None:
        """Full round-trip: create table, add with auto-embed, search.

        Patches the client property so LanceDB's internal retry wrapper
        also uses the mock (it re-instantiates from serialized config).
        """
        from unittest.mock import patch

        with patch.object(SIEEmbeddingFunction, "client", new_callable=lambda: property(lambda self: mock_sie_client)):
            func = SIEEmbeddingFunction.create(model="test-model")

            class TestSchema(LanceModel):
                text: str = func.SourceField()
                vector: Vector(func.ndims()) = func.VectorField()

            table = db.create_table("test_auto_embed", schema=TestSchema, mode="overwrite")
            table.add(
                [
                    {"text": "Machine learning is great."},
                    {"text": "Weather is sunny today."},
                    {"text": "Deep learning uses neural networks."},
                ]
            )

            assert table.count_rows() == 3

            # Vectors were actually computed and stored
            rows = table.to_pandas()
            assert "vector" in rows.columns
            for vec in rows["vector"]:
                assert len(vec) == 384

            # Search works (auto-embeds query)
            results = table.search("machine learning").limit(2).to_list()
            assert len(results) == 2
            assert "text" in results[0]


class TestSIEMultiVectorEmbeddingFunction:
    """Tests for SIEMultiVectorEmbeddingFunction (ColBERT/ColPali)."""

    def test_compute_source_embeddings(self, mock_sie_client: object, sample_texts: list[str]) -> None:
        """Source embeddings return 2D arrays (tokens x dims)."""
        func = SIEMultiVectorEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        results = func.compute_source_embeddings(sample_texts)

        assert isinstance(results, list)
        assert len(results) == len(sample_texts)
        for mv in results:
            assert isinstance(mv, list)
            assert len(mv) > 0  # at least one token
            assert len(mv[0]) == 128  # token dim

    def test_compute_query_embeddings(self, mock_sie_client: object) -> None:
        """Query embeddings return 2D arrays and pass is_query=True."""
        func = SIEMultiVectorEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        results = func.compute_query_embeddings("search text")

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], list)
        assert len(results[0][0]) == 128

        # Check is_query was passed
        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("options") == {"is_query": True}

    def test_compute_query_embeddings_string_input(self, mock_sie_client: object) -> None:
        """String query is handled (wrapped in list internally)."""
        func = SIEMultiVectorEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        results = func.compute_query_embeddings("single query")

        assert len(results) == 1

    def test_ndims_from_metadata(self, mock_sie_client: object) -> None:
        """ndims() queries GET /v1/models/{model} for multivector dim."""
        func = SIEMultiVectorEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        assert func.ndims() == 128
        mock_sie_client.get_model.assert_called_once_with("test-model")
        mock_sie_client.encode.assert_not_called()

    def test_ndims_cached(self, mock_sie_client: object) -> None:
        """ndims() caches after first call."""
        func = SIEMultiVectorEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        func.ndims()
        func.ndims()

        assert mock_sie_client.get_model.call_count == 1

    def test_ndims_no_multivector_raises(self, mock_sie_client: object) -> None:
        """ndims() raises for model without multivector output."""
        func = SIEMultiVectorEmbeddingFunction(model="custom/model")
        func._sie_client = mock_sie_client

        with pytest.raises(ValueError, match="does not support multivector"):
            func.ndims()

    def test_output_types_set_to_multivector(self, mock_sie_client: object) -> None:
        """encode() is called with output_types=["multivector"]."""
        func = SIEMultiVectorEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        func.compute_source_embeddings(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("output_types") == ["multivector"]

    def test_source_does_not_pass_is_query(self, mock_sie_client: object) -> None:
        """compute_source_embeddings() does not pass is_query."""
        func = SIEMultiVectorEmbeddingFunction(model="test-model")
        func._sie_client = mock_sie_client

        func.compute_source_embeddings(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert "options" not in call_kwargs or call_kwargs.get("options") is None

    def test_instruction_forwarded(self, mock_sie_client: object) -> None:
        """Instruction parameter is passed to encode()."""
        func = SIEMultiVectorEmbeddingFunction(model="test-model", instruction="Represent:")
        func._sie_client = mock_sie_client

        func.compute_source_embeddings(["test"])

        call_kwargs = mock_sie_client.encode.call_args.kwargs
        assert call_kwargs.get("instruction") == "Represent:"

    def test_lazy_client_initialization(self) -> None:
        """Client is not created until first use."""
        func = SIEMultiVectorEmbeddingFunction(model="test-model")
        assert func._sie_client is None

    def test_extract_multivector_raises_on_none(self) -> None:
        """_extract_multivector raises ValueError when multivector is missing."""
        func = SIEMultiVectorEmbeddingFunction(model="test-model")

        with pytest.raises(ValueError, match="missing multivector embedding"):
            func._extract_multivector({})

    def test_registered_in_registry(self) -> None:
        """SIEMultiVectorEmbeddingFunction is registered as 'sie-multivector'."""
        cls = get_registry().get("sie-multivector")
        assert cls is SIEMultiVectorEmbeddingFunction
