from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters.siglip import SiglipAdapter
from sie_server.types.inputs import Item


class TestSiglipAdapter:
    """Tests for SiglipAdapter with mocked model."""

    @pytest.fixture
    def mock_siglip_model(self) -> MagicMock:
        """Create a mock SiglipModel."""
        mock = MagicMock()
        # Mock config with vision_config.hidden_size (SigLIP uses hidden_size, not projection_dim)
        mock.config.vision_config.hidden_size = 1152
        return mock

    @pytest.fixture
    def mock_siglip_processor(self) -> MagicMock:
        """Create a mock SiglipProcessor."""
        mock = MagicMock()
        mock.return_value = {"pixel_values": MagicMock(), "input_ids": MagicMock()}
        return mock

    @pytest.fixture
    def adapter(self) -> SiglipAdapter:
        """Create an adapter instance."""
        return SiglipAdapter(
            "google/siglip-so400m-patch14-384",
            normalize=True,
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: SiglipAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["dense"]

    def test_dims_before_load_raises(self, adapter: SiglipAdapter) -> None:
        """Accessing dims before load raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims

    def test_encode_before_load_raises(self, adapter: SiglipAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["dense"])

    def test_encode_without_input_raises(self, adapter: SiglipAdapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["dense"])

    def test_validate_output_types(self, adapter: SiglipAdapter) -> None:
        """Only dense output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])

    @patch("transformers.SiglipModel")
    @patch("transformers.SiglipProcessor")
    def test_load(
        self,
        mock_processor_class: MagicMock,
        mock_model_class: MagicMock,
        adapter: SiglipAdapter,
        mock_siglip_model: MagicMock,
        mock_siglip_processor: MagicMock,
    ) -> None:
        """Load initializes the model."""
        mock_model_class.from_pretrained.return_value = mock_siglip_model
        mock_processor_class.from_pretrained.return_value = mock_siglip_processor

        adapter.load("cpu")

        mock_model_class.from_pretrained.assert_called_once()
        mock_processor_class.from_pretrained.assert_called_once()
        assert adapter.dims.dense == 1152

    @patch("sie_server.adapters.siglip.torch")
    def test_unload(self, mock_torch: MagicMock, adapter: SiglipAdapter) -> None:
        """Unload clears the model."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._dense_dim = 1152

        adapter.unload()

        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = adapter.dims
