from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sie_server.adapters.bge_m3 import BGEM3Adapter
from sie_server.adapters.bge_m3_flag import BGEM3FlagAdapter
from sie_server.types.inputs import Item

# Create a random generator for tests
_RNG = np.random.default_rng(42)


class TestBGEM3FlagAdapter:
    """Tests for BGEM3FlagAdapter with mocked model."""

    @pytest.fixture
    def mock_bgem3_model(self) -> MagicMock:
        """Create a mock BGEM3FlagModel."""
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": _RNG.standard_normal((2, 1024)).astype(np.float32),
            "lexical_weights": [
                {1: 0.5, 100: 0.3, 500: 0.8},
                {2: 0.4, 200: 0.6},
            ],
            "colbert_vecs": [
                _RNG.standard_normal((10, 1024)).astype(np.float32),
                _RNG.standard_normal((8, 1024)).astype(np.float32),
            ],
        }
        return mock

    @pytest.fixture
    def adapter(self) -> BGEM3FlagAdapter:
        """Create an adapter instance."""
        return BGEM3FlagAdapter(
            "BAAI/bge-m3",
            normalize=True,
            max_seq_length=8192,
        )

    def test_capabilities(self, adapter: BGEM3FlagAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["dense", "sparse", "multivector", "score"]

    def test_dims(self, adapter: BGEM3FlagAdapter) -> None:
        """Adapter reports correct dimensions."""
        dims = adapter.dims
        assert dims.dense == 1024
        assert dims.sparse == 250002
        assert dims.multivector == 1024

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_load(
        self,
        mock_bgem3_class: MagicMock,
        adapter: BGEM3FlagAdapter,
    ) -> None:
        """Load initializes the model with resolved path."""
        adapter.load("cuda:0")

        # Model should be loaded (path may be cached or downloaded)
        mock_bgem3_class.assert_called_once()
        call_kwargs = mock_bgem3_class.call_args
        # Should use fp16 on CUDA
        assert call_kwargs.kwargs["use_fp16"] is True
        assert call_kwargs.kwargs["device"] == "cuda:0"
        # First positional arg is the model path (local or HF ID)
        model_path = call_kwargs.args[0]
        assert isinstance(model_path, str)
        assert len(model_path) > 0

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_dense_only(
        self,
        mock_bgem3_class: MagicMock,
        adapter: BGEM3FlagAdapter,
        mock_bgem3_model: MagicMock,
    ) -> None:
        """Encode can return dense embeddings only."""
        mock_bgem3_class.return_value = mock_bgem3_model
        adapter.load("cuda:0")

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["dense"])

        assert output.batch_size == 2
        assert output.dense is not None
        assert output.dense[0].shape == (1024,)
        # Should not have sparse or multivector
        assert output.sparse is None
        assert output.multivector is None

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_all_outputs(
        self,
        mock_bgem3_class: MagicMock,
        adapter: BGEM3FlagAdapter,
        mock_bgem3_model: MagicMock,
    ) -> None:
        """Encode can return all output types."""
        mock_bgem3_class.return_value = mock_bgem3_model
        adapter.load("cuda:0")

        items = [Item(text="hello"), Item(text="world")]
        output = adapter.encode(items, output_types=["dense", "sparse", "multivector"])

        assert output.batch_size == 2

        # Check dense
        assert output.dense is not None
        assert output.dense[0].shape == (1024,)

        # Check sparse
        assert output.sparse is not None
        assert len(output.sparse) == 2
        assert len(output.sparse[0].indices) == 3  # 3 tokens in mock

        # Check multivector
        assert output.multivector is not None
        assert output.multivector[0].shape == (10, 1024)

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_sparse_only(
        self,
        mock_bgem3_class: MagicMock,
        adapter: BGEM3FlagAdapter,
    ) -> None:
        """Encode can return sparse embeddings only."""
        # Create mock with single item outputs
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": _RNG.standard_normal((1, 1024)).astype(np.float32),
            "lexical_weights": [{1: 0.5, 100: 0.3}],
            "colbert_vecs": [_RNG.standard_normal((5, 1024)).astype(np.float32)],
        }
        mock_bgem3_class.return_value = mock
        adapter.load("cuda:0")

        items = [Item(text="hello")]
        output = adapter.encode(items, output_types=["sparse"])

        assert output.batch_size == 1
        assert output.sparse is not None
        assert output.dense is None

    def test_encode_before_load_raises(self, adapter: BGEM3FlagAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    @patch("FlagEmbedding.BGEM3FlagModel")
    def test_encode_without_text_raises(
        self,
        mock_bgem3_class: MagicMock,
        adapter: BGEM3FlagAdapter,
        mock_bgem3_model: MagicMock,
    ) -> None:
        """Encode raises if item has no text."""
        mock_bgem3_class.return_value = mock_bgem3_model
        adapter.load("cuda:0")

        items = [Item()]  # No text
        with pytest.raises(ValueError, match="requires text input"):
            adapter.encode(items, output_types=["dense"])

    @patch("gc.collect")
    @patch("FlagEmbedding.BGEM3FlagModel")
    @patch("sie_server.adapters.bge_m3_flag.torch")
    def test_unload(
        self,
        mock_torch: MagicMock,
        mock_bgem3_class: MagicMock,
        mock_gc: MagicMock,
        adapter: BGEM3FlagAdapter,
        mock_bgem3_model: MagicMock,
    ) -> None:
        """Unload clears the model."""
        mock_bgem3_class.return_value = mock_bgem3_model

        adapter.load("cuda:0")
        adapter.unload()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode([Item(text="test")], output_types=["dense"])


class TestBGEM3Adapter:
    """Tests for native BGEM3Adapter with mocked transformers."""

    @pytest.fixture
    def adapter(self) -> BGEM3Adapter:
        """Create an adapter instance."""
        return BGEM3Adapter(
            "BAAI/bge-m3",
            normalize=True,
            max_seq_length=8192,
        )

    def test_capabilities(self, adapter: BGEM3Adapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["dense", "sparse", "multivector", "score"]

    def test_dims(self, adapter: BGEM3Adapter) -> None:
        """Adapter reports correct dimensions."""
        dims = adapter.dims
        assert dims.dense == 1024
        assert dims.sparse == 250002
        assert dims.multivector == 1024

    def test_encode_before_load_raises(self, adapter: BGEM3Adapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])
