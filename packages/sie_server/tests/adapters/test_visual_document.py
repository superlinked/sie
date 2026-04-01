"""Tests for visual document retrieval adapters (ColPali, ColQwen2, NemoColEmbed).

These adapters encode document images into multi-vector representations
for late interaction retrieval using MaxSim scoring.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.adapters.colpali import ColPaliAdapter
from sie_server.adapters.colqwen2 import ColQwen2Adapter
from sie_server.adapters.nemo_colembed import NemoColEmbedAdapter
from sie_server.types.inputs import Item

# Create a random generator for tests
_RNG = np.random.default_rng(42)


class TestColPaliAdapter:
    """Tests for ColPaliAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> ColPaliAdapter:
        """Create an adapter instance."""
        return ColPaliAdapter(
            "vidore/colpali-v1.3-hf",
            normalize=True,
            compute_precision="float32",
        )

    def test_capabilities(self, adapter: ColPaliAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: ColPaliAdapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 128  # ColPali default

    def test_encode_before_load_raises(self, adapter: ColPaliAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_encode_without_input_raises(self, adapter: ColPaliAdapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: ColPaliAdapter) -> None:
        """Only multivector output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])


class TestColQwen2Adapter:
    """Tests for ColQwen2Adapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> ColQwen2Adapter:
        """Create an adapter instance."""
        return ColQwen2Adapter(
            "vidore/colqwen2.5-v0.2",
            normalize=True,
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: ColQwen2Adapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: ColQwen2Adapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 128  # ColQwen2 default

    def test_encode_before_load_raises(self, adapter: ColQwen2Adapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_encode_without_input_raises(self, adapter: ColQwen2Adapter) -> None:
        """Encode raises if item has no text or images."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item()]  # No text or images
        with pytest.raises(ValueError, match="requires either text or images"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: ColQwen2Adapter) -> None:
        """Only multivector output type is supported."""
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])


class TestNemoColEmbedAdapter:
    """Tests for NemoColEmbedAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> NemoColEmbedAdapter:
        """Create an adapter instance."""
        return NemoColEmbedAdapter(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            normalize=True,
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: NemoColEmbedAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text", "image"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_has_default(self, adapter: NemoColEmbedAdapter) -> None:
        """Dims returns default value before load."""
        dims = adapter.dims
        assert dims.multivector == 128  # NemoColEmbed default

    def test_encode_before_load_raises(self, adapter: NemoColEmbedAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_validate_output_types(self, adapter: NemoColEmbedAdapter) -> None:
        """Only multivector output type is supported."""
        # Mock model as loaded
        adapter._model = MagicMock()
        adapter._device = "cpu"

        items = [Item(text="test")]
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter.encode(items, output_types=["sparse"])


class TestNemoColEmbedPreprocessor:
    """Tests for NemoColEmbedPreprocessor infrastructure."""

    def test_preprocessor_class_exists(self) -> None:
        """NemoColEmbedPreprocessor class is defined."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        assert NemoColEmbedPreprocessor is not None

    def test_preprocessor_payload_class_exists(self) -> None:
        """NemoColEmbedPayload dataclass is defined."""
        from sie_server.core.prepared import NemoColEmbedPayload

        assert NemoColEmbedPayload is not None

    def test_adapter_has_preprocessor_method(self) -> None:
        """NemoColEmbedAdapter has get_preprocessor method."""
        adapter = NemoColEmbedAdapter(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            normalize=True,
        )
        # get_preprocessor returns CharCountPreprocessor for cost estimation
        preprocessor = adapter.get_preprocessor()
        assert preprocessor is not None

    def test_adapter_processor_created_on_load(self) -> None:
        """NemoColEmbedAdapter creates _processor after load().

        Note: Full _create_processor() test requires loaded model with
        tokenizer/config attributes. This test just verifies the interface.
        """
        adapter = NemoColEmbedAdapter(
            "nvidia/llama-nemoretriever-colembed-3b-v1",
            normalize=True,
        )
        # Before load, _processor should be None
        assert adapter._processor is None


class TestVLMCudaCacheClearing:
    """Tests that VLM adapters contain torch.cuda.empty_cache() after inference.

    This prevents GPU memory accumulation (OOM) on L4 22GB GPUs when
    encoding many document images in sequence.
    """

    def test_colqwen2_encode_images_has_empty_cache(self) -> None:
        """ColQwen2 _encode_images source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColQwen2Adapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "ColQwen2._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_colpali_encode_prepared_batch_has_empty_cache(self) -> None:
        """ColPali _encode_prepared_batch source contains empty_cache call."""
        import inspect

        source = inspect.getsource(ColPaliAdapter._encode_prepared_batch)
        assert "torch.cuda.empty_cache()" in source, (
            "ColPali._encode_prepared_batch must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_nemo_colembed_encode_images_has_empty_cache(self) -> None:
        """NemoColEmbed _encode_images source contains empty_cache call."""
        import inspect

        source = inspect.getsource(NemoColEmbedAdapter._encode_images)
        assert "torch.cuda.empty_cache()" in source, (
            "NemoColEmbed._encode_images must call torch.cuda.empty_cache() to prevent OOM"
        )

    def test_nemo_colembed_preprocessed_has_empty_cache(self) -> None:
        """NemoColEmbed _encode_images_preprocessed source contains empty_cache call."""
        import inspect

        source = inspect.getsource(NemoColEmbedAdapter._encode_images_preprocessed)
        assert "torch.cuda.empty_cache()" in source, (
            "NemoColEmbed._encode_images_preprocessed must call torch.cuda.empty_cache() to prevent OOM"
        )
