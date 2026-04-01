"""Tests for PreprocessorRegistry."""

import io
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from sie_server.core.preprocessor import Preprocessor
from sie_server.core.preprocessor_registry import PreprocessorRegistry
from sie_server.types.inputs import ImageInput, Item


class TestPreprocessorRegistry:
    """Tests for PreprocessorRegistry."""

    @pytest.fixture
    def registry(self):
        """Create a PreprocessorRegistry instance."""
        return PreprocessorRegistry(max_workers=2)

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [[101, 2023, 102]],
            "attention_mask": [[1, 1, 1]],
        }
        return tokenizer

    @pytest.fixture
    def mock_processor(self):
        """Create a mock image processor."""
        processor = MagicMock()
        processor.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }
        return processor

    @pytest.fixture
    def mock_config(self):
        """Create a mock model config."""
        config = MagicMock()
        config.max_sequence_length = 512
        return config

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image as bytes."""
        img = Image.new("RGB", (100, 100), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_init(self, registry):
        """Registry initializes with correct worker count."""
        assert registry.max_workers == 2
        assert registry.registered_models == []

    def test_register_text(self, registry, mock_tokenizer):
        """Register text preprocessor."""
        registry.register_text("test-model", mock_tokenizer)

        assert "test-model" in registry.registered_models
        assert registry.has_preprocessor("test-model", "text")
        assert not registry.has_preprocessor("test-model", "image")

    def test_register_image(self, registry, mock_processor):
        """Register image preprocessor."""
        registry.register_image("test-model", mock_processor)

        assert "test-model" in registry.registered_models
        assert registry.has_preprocessor("test-model", "image")
        assert not registry.has_preprocessor("test-model", "text")

    def test_register_both(self, registry, mock_tokenizer, mock_processor):
        """Register both text and image preprocessors."""
        registry.register_text("test-model", mock_tokenizer)
        registry.register_image("test-model", mock_processor)

        assert registry.has_preprocessor("test-model", "text")
        assert registry.has_preprocessor("test-model", "image")
        assert registry.get_modalities("test-model") == ["text", "image"]

    def test_unregister(self, registry, mock_tokenizer, mock_processor):
        """Unregister removes all preprocessors for a model."""
        registry.register_text("test-model", mock_tokenizer)
        registry.register_image("test-model", mock_processor)

        registry.unregister("test-model")

        assert "test-model" not in registry.registered_models
        assert not registry.has_preprocessor("test-model", "text")
        assert not registry.has_preprocessor("test-model", "image")

    def test_unregister_nonexistent(self, registry):
        """Unregister nonexistent model doesn't raise."""
        registry.unregister("nonexistent")  # Should not raise

    def test_get_preprocessor(self, registry, mock_tokenizer):
        """Get preprocessor returns correct instance."""
        registry.register_text("test-model", mock_tokenizer)

        preprocessor = registry.get_preprocessor("test-model", "text")
        assert preprocessor is not None
        assert isinstance(preprocessor, Preprocessor)
        assert preprocessor.modality == "text"

    def test_get_preprocessor_not_found(self, registry):
        """Get preprocessor returns None for unregistered."""
        assert registry.get_preprocessor("test-model", "text") is None
        assert registry.get_preprocessor("test-model", "image") is None

    def test_get_modalities_empty(self, registry):
        """Get modalities for unregistered model returns empty list."""
        assert registry.get_modalities("nonexistent") == []

    @pytest.mark.asyncio
    async def test_prepare_text(self, registry, mock_tokenizer, mock_config):
        """Prepare text items asynchronously."""
        registry.register_text("test-model", mock_tokenizer)
        items = [Item(text="Hello world")]

        batch = await registry.prepare("test-model", items, mock_config)

        assert batch.modality == "text"
        assert batch.size == 1

    @pytest.mark.asyncio
    async def test_prepare_image(self, registry, mock_processor, mock_config, sample_image_bytes):
        """Prepare image items asynchronously."""
        registry.register_image("test-model", mock_processor)
        items = [Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")])]

        batch = await registry.prepare("test-model", items, mock_config)

        assert batch.modality == "image"
        assert batch.size == 1

    @pytest.mark.asyncio
    async def test_prepare_image_priority(
        self, registry, mock_tokenizer, mock_processor, mock_config, sample_image_bytes
    ):
        """Image modality takes priority over text when both present."""
        registry.register_text("test-model", mock_tokenizer)
        registry.register_image("test-model", mock_processor)

        # Item with both text and image
        items = [
            Item(
                text="Some text",
                images=[ImageInput(data=sample_image_bytes, format="jpeg")],
            )
        ]

        batch = await registry.prepare("test-model", items, mock_config)

        # Should use image preprocessor
        assert batch.modality == "image"

    @pytest.mark.asyncio
    async def test_prepare_no_preprocessor(self, registry, mock_config):
        """Prepare raises KeyError when no preprocessor registered."""
        items = [Item(text="Hello")]

        with pytest.raises(KeyError, match="No text preprocessor"):
            await registry.prepare("test-model", items, mock_config)

    @pytest.mark.asyncio
    async def test_prepare_empty_items(self, registry, mock_tokenizer, mock_config):
        """Prepare raises ValueError for items without content."""
        registry.register_text("test-model", mock_tokenizer)
        items = [Item()]  # No text, no images

        with pytest.raises(ValueError, match="must have text or images"):
            await registry.prepare("test-model", items, mock_config)

    def test_prepare_sync_text(self, registry, mock_tokenizer, mock_config):
        """Prepare text items synchronously."""
        registry.register_text("test-model", mock_tokenizer)
        items = [Item(text="Hello world")]

        batch = registry.prepare_sync("test-model", items, mock_config)

        assert batch.modality == "text"
        assert batch.size == 1

    def test_prepare_sync_image(self, registry, mock_processor, mock_config, sample_image_bytes):
        """Prepare image items synchronously."""
        registry.register_image("test-model", mock_processor)
        items = [Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")])]

        batch = registry.prepare_sync("test-model", items, mock_config)

        assert batch.modality == "image"
        assert batch.size == 1

    def test_shutdown(self, registry, mock_tokenizer):
        """Shutdown clears preprocessors and stops executor."""
        registry.register_text("test-model", mock_tokenizer)

        registry.shutdown()

        assert registry.registered_models == []


class TestPreprocessorRegistryIntegration:
    """Integration tests for PreprocessorRegistry with real tokenizers."""

    @pytest.fixture
    def registry(self):
        """Create registry."""
        return PreprocessorRegistry(max_workers=2)

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.max_sequence_length = 512
        return config

    @pytest.mark.asyncio
    async def test_text_preprocessing_matches_tokenizer(self, registry, mock_config):
        """Text preprocessing produces same output as direct tokenization."""
        # Create a simple mock tokenizer that returns predictable output
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [[101, 7592, 2088, 102], [101, 3231, 102]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1]],
        }

        registry.register_text("test-model", tokenizer)

        items = [
            Item(text="Hello world"),
            Item(text="Test"),
        ]

        batch = await registry.prepare("test-model", items, mock_config)

        # Verify output matches tokenizer
        assert batch.size == 2
        assert batch.items[0].payload.input_ids == [101, 7592, 2088, 102]
        assert batch.items[0].payload.attention_mask == [1, 1, 1, 1]
        assert batch.items[0].cost == 4  # token count

        assert batch.items[1].payload.input_ids == [101, 3231, 102]
        assert batch.items[1].payload.attention_mask == [1, 1, 1]
        assert batch.items[1].cost == 3

        # Total cost is sum
        assert batch.total_cost == 7

        # Original indices preserved
        assert batch.items[0].original_index == 0
        assert batch.items[1].original_index == 1
