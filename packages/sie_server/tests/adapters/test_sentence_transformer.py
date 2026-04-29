from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sie_server.adapters.sentence_transformer import (
    SentenceTransformerDenseAdapter,
    SentenceTransformerSparseAdapter,
)
from sie_server.types.inputs import Item

# Create a random generator for tests
_RNG = np.random.default_rng(42)


class TestSentenceTransformerDenseAdapter:
    """Tests for SentenceTransformerDenseAdapter with mocked model."""

    @pytest.fixture
    def mock_st_model(self) -> MagicMock:
        """Create a mock SentenceTransformer model."""
        mock = MagicMock()
        mock.get_embedding_dimension.return_value = 384

        # Return correct batch size based on input
        def mock_encode(texts, **kwargs):
            return _RNG.standard_normal((len(texts), 384)).astype(np.float32)

        mock.encode.side_effect = mock_encode
        return mock

    @pytest.fixture
    def adapter(self) -> SentenceTransformerDenseAdapter:
        """Create an adapter instance."""
        return SentenceTransformerDenseAdapter(
            "test-model",
            normalize=True,
            max_seq_length=512,
        )

    def test_capabilities(self, adapter: SentenceTransformerDenseAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["dense"]

    def test_dims_before_load_returns_none(self, adapter: SentenceTransformerDenseAdapter) -> None:
        """Dims returns None values before load (BaseAdapter derives from spec)."""
        dims = adapter.dims
        assert dims.dense is None

    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    def test_load(
        self,
        mock_st_class: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Load initializes the model."""
        mock_st_class.return_value = mock_st_model

        adapter.load("cpu")

        mock_st_class.assert_called_once_with(
            "test-model",
            device="cpu",
            trust_remote_code=True,
            config_kwargs=None,
        )
        assert adapter.dims.dense == 384

    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    def test_encode(
        self,
        mock_st_class: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Encode returns dense embeddings."""
        mock_st_class.return_value = mock_st_model
        adapter.load("cpu")

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["dense"])

        assert output.batch_size == 2
        assert output.dense is not None
        assert output.dense[0].shape == (384,)

        # First call is warmup during load, second is actual encode
        assert mock_st_model.encode.call_count == 2
        call_args = mock_st_model.encode.call_args
        assert call_args[0][0] == ["hello", "world"]

    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    def test_encode_with_instruction(
        self,
        mock_st_class: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Encode prepends instruction to text."""
        mock_st_class.return_value = mock_st_model
        adapter.load("cpu")

        items = [Item(text="query")]
        adapter.encode(items, output_types=["dense"], instruction="search:")

        call_args = mock_st_model.encode.call_args
        assert call_args[0][0] == ["search: query"]

    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    def test_encode_unsupported_output_type(
        self,
        mock_st_class: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Encode raises for unsupported output types."""
        mock_st_class.return_value = mock_st_model
        adapter.load("cpu")

        items = [Item(text="hello")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    @patch("sie_server.adapters.sentence_transformer.SentenceTransformer")
    def test_encode_without_text_raises(
        self,
        mock_st_class: MagicMock,
        adapter: SentenceTransformerDenseAdapter,
        mock_st_model: MagicMock,
    ) -> None:
        """Encode raises if item has no text."""
        mock_st_class.return_value = mock_st_model
        adapter.load("cpu")

        items = [Item()]  # No text
        with pytest.raises(ValueError, match="requires text input"):
            adapter.encode(items, output_types=["dense"])


class TestSentenceTransformerSparseAdapter:
    """Tests for SentenceTransformerSparseAdapter with mocked model."""

    @pytest.fixture
    def mock_sparse_model(self) -> MagicMock:
        """Create a mock SparseEncoder model."""
        import torch

        mock = MagicMock()
        mock.get_embedding_dimension.return_value = 30522

        # Create sparse COO tensor output
        # 2 rows, vocab_size columns, few non-zero values
        # indices: [row_indices, col_indices], values: weights
        row_indices = torch.tensor([0, 0, 0, 1, 1])
        col_indices = torch.tensor([100, 500, 1000, 200, 800])
        indices = torch.stack([row_indices, col_indices])
        values = torch.tensor([0.5, 0.3, 0.8, 0.4, 0.6], dtype=torch.float32)
        sparse_result = torch.sparse_coo_tensor(indices, values, size=(2, 30522))

        mock.encode_query.return_value = sparse_result
        mock.encode_document.return_value = sparse_result
        return mock

    @pytest.fixture
    def adapter(self) -> SentenceTransformerSparseAdapter:
        """Create an adapter instance."""
        return SentenceTransformerSparseAdapter("test-sparse-model")

    def test_capabilities(self, adapter: SentenceTransformerSparseAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["sparse"]

    def test_dims_before_load_returns_none(self, adapter: SentenceTransformerSparseAdapter) -> None:
        """Dims returns None values before load (BaseAdapter derives from spec)."""
        dims = adapter.dims
        assert dims.sparse is None

    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    def test_load(
        self,
        mock_sparse_class: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Load initializes sparse model."""
        mock_sparse_class.return_value = mock_sparse_model

        adapter.load("cpu")

        mock_sparse_class.assert_called_once()
        assert adapter.dims.sparse == 30522
        assert adapter.dims.dense is None

    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    def test_encode_document(
        self,
        mock_sparse_class: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Encode returns sparse embeddings for documents."""
        mock_sparse_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["sparse"], is_query=False)

        assert output.batch_size == 2
        assert output.sparse is not None
        assert len(output.sparse) == 2
        # SparseVector has indices and values attributes
        assert hasattr(output.sparse[0], "indices")
        assert hasattr(output.sparse[0], "values")

        mock_sparse_model.encode_document.assert_called_once()

    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    def test_encode_query(
        self,
        mock_sparse_class: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Encode uses encode_query for queries."""
        mock_sparse_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [Item(text="query")]
        adapter.encode(items, output_types=["sparse"], is_query=True)

        mock_sparse_model.encode_query.assert_called_once()

    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    def test_rejects_dense_output(
        self,
        mock_sparse_class: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Sparse model rejects dense output type."""
        mock_sparse_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [Item(text="hello")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    def test_encode_without_text_raises(
        self,
        mock_sparse_class: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Encode raises if item has no text."""
        mock_sparse_class.return_value = mock_sparse_model
        adapter.load("cpu")

        items = [Item()]  # No text
        with pytest.raises(ValueError, match="requires text input"):
            adapter.encode(items, output_types=["sparse"])

    def test_encode_before_load_raises(self, adapter: SentenceTransformerSparseAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["sparse"])

    @patch("gc.collect")
    @patch("sie_server.adapters.sentence_transformer.SparseEncoder")
    @patch("sie_server.adapters.sentence_transformer.torch")
    def test_unload(
        self,
        mock_torch: MagicMock,
        mock_sparse_class: MagicMock,
        mock_gc: MagicMock,
        adapter: SentenceTransformerSparseAdapter,
        mock_sparse_model: MagicMock,
    ) -> None:
        """Unload clears the model."""
        mock_sparse_class.return_value = mock_sparse_model

        adapter.load("cpu")
        adapter.unload()

        dims = adapter.dims
        assert dims.sparse is None
