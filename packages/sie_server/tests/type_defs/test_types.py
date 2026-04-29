"""Tests for types used in the SIE Server API."""

import numpy as np
import pytest
from sie_server.types.inputs import Item
from sie_server.types.outputs import DenseVector, EncodeResult, MultiVector, SparseVector
from sie_server.types.requests import (
    EncodeParams,
    EncodeRequest,
    ExtractParams,
    ExtractRequest,
    ScoreRequest,
)
from sie_server.types.requests import (
    Item as RequestItem,
)
from sie_server.types.responses import (
    EncodeResponse,
    EntityResult,
    ErrorCode,
    ErrorDetail,
    ErrorResponse,
    ExtractResponse,
    ExtractResult,
    ScoreEntry,
    ScoreResponse,
)


class TestInputTypes:
    """Tests for input types: Item (msgspec Struct)."""

    def test_item_text_only(self) -> None:
        """Item with just text."""
        item = Item(text="Hello world")
        assert item.text == "Hello world"
        assert item.id is None

    def test_item_with_id(self) -> None:
        """Item with ID."""
        item = Item(id="doc-1", text="Hello")
        assert item.id == "doc-1"

    def test_item_empty_allowed(self) -> None:
        """Empty Item is allowed (all fields optional for multimodal flexibility)."""
        item = Item()
        assert item.text is None
        assert item.images is None
        assert item.audio is None
        assert item.video is None
        assert item.document is None

    def test_item_with_document(self) -> None:
        """Item carries a document payload for composite-document extractors."""
        item = Item(document={"data": b"%PDF-1.4", "format": "pdf"})
        assert item.document is not None
        assert item.document["data"] == b"%PDF-1.4"
        assert item.document["format"] == "pdf"


class TestOutputTypes:
    """Tests for output types (TypedDicts): DenseVector, SparseVector, MultiVector."""

    def test_dense_vector(self) -> None:
        """DenseVector holds numpy array."""
        values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vec = DenseVector(dims=3, dtype="float32", values=values)
        assert vec["dims"] == 3
        assert vec["dtype"] == "float32"
        np.testing.assert_array_equal(vec["values"], values)

    def test_dense_vector_serialization(self) -> None:
        """DenseVector values can be converted to list for JSON."""
        values = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        vec = DenseVector(dims=3, dtype="float32", values=values)
        # TypedDict is just a dict - convert values manually
        serialized = {**vec, "values": vec["values"].tolist()}
        assert serialized["values"] == [0.10000000149011612, 0.20000000298023224, 0.30000001192092896]

    def test_sparse_vector(self) -> None:
        """SparseVector holds indices and values as numpy arrays."""
        indices = np.array([1, 5, 10], dtype=np.int64)
        values = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        vec = SparseVector(dims=30522, dtype="float32", indices=indices, values=values)
        assert vec["dims"] == 30522
        np.testing.assert_array_equal(vec["indices"], indices)
        np.testing.assert_array_equal(vec["values"], values)

    def test_sparse_vector_unknown_dims(self) -> None:
        """SparseVector can have unknown vocabulary size."""
        vec = SparseVector(
            dims=None,
            dtype="float32",
            indices=np.array([1, 2]),
            values=np.array([0.5, 0.5]),
        )
        assert vec["dims"] is None

    def test_multi_vector(self) -> None:
        """MultiVector holds per-token embeddings as numpy array."""
        rng = np.random.default_rng(42)
        values = rng.standard_normal((10, 128)).astype(np.float32)
        vec = MultiVector(token_dims=128, num_tokens=10, dtype="float32", values=values)
        assert vec["token_dims"] == 128
        assert vec["num_tokens"] == 10
        assert vec["values"].shape == (10, 128)

    def test_encode_result(self) -> None:
        """EncodeResult can have multiple output types."""
        dense = DenseVector(dims=768, dtype="float32", values=np.zeros(768, dtype=np.float32))
        sparse = SparseVector(
            dims=30522,
            dtype="float32",
            indices=np.array([1]),
            values=np.array([0.5]),
        )
        result = EncodeResult(id="doc-1", dense=dense, sparse=sparse)
        assert result["id"] == "doc-1"
        assert result.get("dense") is not None
        assert result.get("sparse") is not None
        assert result.get("multivector") is None


class TestRequestTypes:
    """Tests for request Struct types (msgspec)."""

    def test_encode_request_minimal(self) -> None:
        """EncodeRequest with just items."""
        req = EncodeRequest(items=[RequestItem(text="Hello")])
        assert len(req.items) == 1
        assert req.params is None

    def test_encode_request_with_params(self) -> None:
        """EncodeRequest with parameters."""
        req = EncodeRequest(
            items=[RequestItem(text="Hello")],
            params=EncodeParams(output_types=["dense", "sparse"], options={"is_query": True}),
        )
        assert req.params is not None
        assert req.params.output_types == ["dense", "sparse"]
        assert req.params.options is not None
        assert req.params.options.get("is_query") is True

    def test_encode_request_empty_items_rejected(self) -> None:
        """EncodeRequest with empty items raises ValidationError in __post_init__."""
        import msgspec as _msgspec

        with pytest.raises(_msgspec.ValidationError, match="items"):
            EncodeRequest(items=[])

    def test_encode_params_defaults(self) -> None:
        """EncodeParams fields default to None."""
        params = EncodeParams()
        assert params.output_types is None
        assert params.options is None
        assert params.instruction is None

    def test_score_request(self) -> None:
        """ScoreRequest for reranking."""
        req = ScoreRequest(
            query=RequestItem(id="q1", text="What is Python?"),
            items=[RequestItem(id="d1", text="Python is a programming language")],
        )
        assert req.query.id == "q1"
        assert len(req.items) == 1

    def test_extract_request(self) -> None:
        """ExtractRequest for NER."""
        req = ExtractRequest(
            items=[RequestItem(text="John works at Anthropic")],
            params=ExtractParams(labels=["person", "organization"]),
        )
        assert req.params is not None
        assert "person" in req.params.labels


class TestResponseTypes:
    """Tests for response types (TypedDicts)."""

    def test_encode_response(self) -> None:
        """EncodeResponse structure."""
        dense = DenseVector(dims=3, dtype="float32", values=np.array([0.1, 0.2, 0.3]))
        result = EncodeResult(dense=dense)
        response = EncodeResponse(model="bge-m3", items=[result])
        assert response["model"] == "bge-m3"
        assert len(response["items"]) == 1

    def test_score_response(self) -> None:
        """ScoreResponse structure."""
        response = ScoreResponse(
            model="jina-reranker",
            scores=[
                ScoreEntry(item_id="d1", score=0.95, rank=0),
                ScoreEntry(item_id="d2", score=0.82, rank=1),
            ],
        )
        assert response["scores"][0]["score"] > response["scores"][1]["score"]

    def test_extract_response(self) -> None:
        """ExtractResponse structure."""
        response = ExtractResponse(
            model="gliner",
            items=[
                ExtractResult(
                    id="doc-1",
                    entities=[EntityResult(text="John", label="person", score=0.98, start=0, end=4)],
                )
            ],
        )
        assert response["items"][0]["entities"][0]["label"] == "person"

    def test_error_response(self) -> None:
        """ErrorResponse structure."""
        response = ErrorResponse(error=ErrorDetail(code=ErrorCode.MODEL_NOT_FOUND.value, message="Model not found"))
        assert response["error"]["code"] == ErrorCode.MODEL_NOT_FOUND.value
