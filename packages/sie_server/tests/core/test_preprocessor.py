"""Tests for preprocessor implementations."""

import io
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from sie_server.core.prepared import ImagePayload, PreparedItem, TextPayload
from sie_server.core.preprocessor import ImagePreprocessor, Preprocessor, TextPreprocessor
from sie_server.core.preprocessor.image import OpenCLIPImagePreprocessor
from sie_server.types.inputs import ImageInput, Item


class TestPreprocessorProtocol:
    """Tests for Preprocessor protocol."""

    def test_text_preprocessor_is_preprocessor(self):
        """TextPreprocessor satisfies Preprocessor protocol."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

        preprocessor = TextPreprocessor(tokenizer, "test-model")
        assert isinstance(preprocessor, Preprocessor)

    def test_image_preprocessor_is_preprocessor(self):
        """ImagePreprocessor satisfies Preprocessor protocol."""
        processor = MagicMock()
        preprocessor = ImagePreprocessor(processor, "test-model")
        assert isinstance(preprocessor, Preprocessor)


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create mock model config."""
        config = MagicMock()
        config.max_sequence_length = 512
        return config

    def test_modality(self, mock_tokenizer):
        """Modality is 'text'."""
        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")
        assert preprocessor.modality == "text"

    def test_prepare_single_item(self, mock_tokenizer, mock_config):
        """Prepare single text item."""
        mock_tokenizer.return_value = {
            "input_ids": [[101, 2023, 2003, 1037, 3231, 102]],
            "attention_mask": [[1, 1, 1, 1, 1, 1]],
        }

        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")
        items = [Item(text="This is a test")]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 1
        assert batch.modality == "text"
        assert batch.total_cost == 6  # 6 tokens

        item = batch.items[0]
        assert item.cost == 6
        assert item.original_index == 0
        assert isinstance(item.payload, TextPayload)
        assert item.payload.token_count == 6

    def test_prepare_multiple_items(self, mock_tokenizer, mock_config):
        """Prepare multiple text items."""
        mock_tokenizer.return_value = {
            "input_ids": [
                [101, 2023, 102],  # 3 tokens
                [101, 2023, 2003, 1037, 102],  # 5 tokens
                [101, 102],  # 2 tokens
            ],
            "attention_mask": [
                [1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1],
            ],
        }

        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")
        items: list[Item] = [
            Item(text="Short"),
            Item(text="A bit longer"),
            Item(text="X"),
        ]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 3
        assert batch.total_cost == 10  # 3 + 5 + 2

        # Check original indices preserved
        indices = [item.original_index for item in batch.items]
        assert indices == [0, 1, 2]

        # Check costs
        costs = [item.cost for item in batch.items]
        assert costs == [3, 5, 2]

    def test_prepare_empty_text(self, mock_tokenizer, mock_config):
        """Handle items with None text."""
        mock_tokenizer.return_value = {
            "input_ids": [[101, 102]],
            "attention_mask": [[1, 1]],
        }

        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")
        items = [Item(text=None)]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 1
        # Empty string tokenized
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args
        assert call_args[0][0] == [""]  # Empty string passed

    def test_collate_single_item(self, mock_tokenizer):
        """Collate single prepared item."""
        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")

        prepared = [
            PreparedItem(
                payload=TextPayload(
                    input_ids=[101, 2023, 102],
                    attention_mask=[1, 1, 1],
                ),
                cost=3,
                original_index=0,
            )
        ]

        result = preprocessor.collate(prepared, device="cpu")

        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape == (1, 3)
        assert result["attention_mask"].shape == (1, 3)
        assert result["input_ids"].tolist() == [[101, 2023, 102]]

    def test_collate_with_padding(self, mock_tokenizer):
        """Collate pads shorter sequences."""
        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")

        prepared = [
            PreparedItem(
                payload=TextPayload(
                    input_ids=[101, 2023, 102],
                    attention_mask=[1, 1, 1],
                ),
                cost=3,
                original_index=0,
            ),
            PreparedItem(
                payload=TextPayload(
                    input_ids=[101, 102],
                    attention_mask=[1, 1],
                ),
                cost=2,
                original_index=1,
            ),
        ]

        result = preprocessor.collate(prepared, device="cpu", pad_token_id=0)

        assert result["input_ids"].shape == (2, 3)  # Padded to max length
        assert result["attention_mask"].shape == (2, 3)

        # First item unchanged
        assert result["input_ids"][0].tolist() == [101, 2023, 102]
        assert result["attention_mask"][0].tolist() == [1, 1, 1]

        # Second item padded
        assert result["input_ids"][1].tolist() == [101, 102, 0]
        assert result["attention_mask"][1].tolist() == [1, 1, 0]

    def test_collate_empty(self, mock_tokenizer):
        """Collate empty list returns empty tensors."""
        preprocessor = TextPreprocessor(mock_tokenizer, "test-model")

        result = preprocessor.collate([], device="cpu")

        assert result["input_ids"].numel() == 0
        assert result["attention_mask"].numel() == 0


class TestImagePreprocessor:
    """Tests for ImagePreprocessor."""

    @pytest.fixture
    def mock_processor(self):
        """Create mock image processor."""
        processor = MagicMock()
        # Return processed pixel values
        processor.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }
        return processor

    @pytest.fixture
    def mock_config(self):
        """Create mock model config."""
        config = MagicMock()
        return config

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image as bytes."""
        img = Image.new("RGB", (640, 480), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_modality(self, mock_processor):
        """Modality is 'image'."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")
        assert preprocessor.modality == "image"

    def test_prepare_single_image(self, mock_processor, mock_config, sample_image_bytes):
        """Prepare single image item."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        items = [Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")])]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 1
        assert batch.modality == "image"
        assert batch.total_cost == 1  # 1 image

        item = batch.items[0]
        assert item.cost == 1
        assert item.original_index == 0
        assert isinstance(item.payload, ImagePayload)
        assert item.payload.original_size == (640, 480)

    def test_prepare_multiple_images(self, mock_processor, mock_config, sample_image_bytes):
        """Prepare multiple image items."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        items: list[Item] = [
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
        ]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 3
        assert batch.total_cost == 3

        # Check original indices
        indices = [item.original_index for item in batch.items]
        assert indices == [0, 1, 2]

    def test_prepare_skips_items_without_images(self, mock_processor, mock_config, sample_image_bytes):
        """Items without images are skipped."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        items: list[Item] = [
            Item(text="text only"),
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
            Item(text="also text only"),
        ]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 1
        assert batch.total_cost == 1
        # Only item at index 1 has image
        assert batch.items[0].original_index == 1

    def test_prepare_rgba_conversion(self, mock_processor, mock_config):
        """RGBA images are converted to RGB."""
        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        rgba_bytes = buffer.getvalue()

        preprocessor = ImagePreprocessor(mock_processor, "test-model")
        items = [Item(images=[ImageInput(data=rgba_bytes, format="png")])]

        # Should not raise - RGBA converted to RGB
        batch = preprocessor.prepare(items, config=mock_config)
        assert batch.size == 1

    def test_collate_single_image(self, mock_processor):
        """Collate single image item."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        pixel_values = torch.randn(3, 224, 224)
        prepared = [
            PreparedItem(
                payload=ImagePayload(
                    pixel_values=pixel_values,
                    original_size=(640, 480),
                ),
                cost=1,
                original_index=0,
            )
        ]

        result = preprocessor.collate(prepared, device="cpu")

        assert "pixel_values" in result
        assert result["pixel_values"].shape == (1, 3, 224, 224)

    def test_collate_multiple_images(self, mock_processor):
        """Collate batches images correctly."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        prepared = [
            PreparedItem(
                payload=ImagePayload(
                    pixel_values=torch.randn(3, 224, 224),
                    original_size=(640, 480),
                ),
                cost=1,
                original_index=i,
            )
            for i in range(4)
        ]

        result = preprocessor.collate(prepared, device="cpu")

        assert result["pixel_values"].shape == (4, 3, 224, 224)

    def test_collate_empty(self, mock_processor):
        """Collate empty list returns empty tensor."""
        preprocessor = ImagePreprocessor(mock_processor, "test-model")

        result = preprocessor.collate([], device="cpu")

        assert result["pixel_values"].numel() == 0


class TestOpenCLIPImagePreprocessor:
    """Tests for OpenCLIPImagePreprocessor.

    Uses a lambda ``val_preproc`` so the tests are pure unit tests with no
    model download.
    """

    @pytest.fixture
    def fake_val_preproc(self):
        """A torchvision-compose-shaped callable: PIL.Image -> Tensor[3, 224, 224]."""

        def _fn(img: Image.Image) -> torch.Tensor:
            # Deterministic encoding of (width, height) into the first two
            # channels of the corner pixel so tests can verify which image
            # produced the tensor.
            tensor = torch.zeros(3, 224, 224)
            tensor[0, 0, 0] = float(img.width)
            tensor[1, 0, 0] = float(img.height)
            return tensor

        return _fn

    @pytest.fixture
    def mock_config(self):
        """Create mock model config."""
        return MagicMock()

    @pytest.fixture
    def sample_image_bytes(self):
        """Create a 640x480 JPEG."""
        img = Image.new("RGB", (640, 480), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_is_preprocessor(self, fake_val_preproc):
        """OpenCLIPImagePreprocessor satisfies the Preprocessor protocol."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        assert isinstance(preprocessor, Preprocessor)

    def test_modality(self, fake_val_preproc):
        """Modality is 'image'."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        assert preprocessor.modality == "image"

    def test_prepare_runs_val_preproc_and_records_payload(self, fake_val_preproc, mock_config, sample_image_bytes):
        """prepare() invokes val_preproc and stores the resulting tensor on the payload."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        items = [Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")])]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 1
        assert batch.modality == "image"
        assert batch.total_cost == 1

        item = batch.items[0]
        assert item.cost == 1
        assert item.original_index == 0
        assert isinstance(item.payload, ImagePayload)
        assert item.payload.original_size == (640, 480)
        # Tensor shape is [C, H, W] (no batch dim) — same contract as ImagePreprocessor
        assert item.payload.pixel_values.shape == (3, 224, 224)
        # Verify our fake val_preproc actually ran (encoded width/height)
        assert item.payload.pixel_values[0, 0, 0].item() == 640.0
        assert item.payload.pixel_values[1, 0, 0].item() == 480.0

    def test_prepare_skips_text_only_items(self, fake_val_preproc, mock_config, sample_image_bytes):
        """Items without images are skipped; original_index of image items is preserved."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        items: list[Item] = [
            Item(text="text only"),
            Item(images=[ImageInput(data=sample_image_bytes, format="jpeg")]),
            Item(text="also text only"),
        ]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.size == 1
        assert batch.total_cost == 1
        assert batch.items[0].original_index == 1

    def test_prepare_rgba_conversion(self, fake_val_preproc, mock_config):
        """RGBA images are converted to RGB before val_preproc."""
        img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        rgba_bytes = buffer.getvalue()

        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        items = [Item(images=[ImageInput(data=rgba_bytes, format="png")])]

        # Should not raise — RGBA → RGB happens before the lambda is invoked.
        batch = preprocessor.prepare(items, config=mock_config)
        assert batch.size == 1
        assert batch.items[0].payload.pixel_values.shape == (3, 224, 224)

    def test_collate_stacks_into_batch(self, fake_val_preproc):
        """collate() stacks [C, H, W] payloads into [B, C, H, W] on the requested device."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        prepared = [
            PreparedItem(
                payload=ImagePayload(
                    pixel_values=torch.randn(3, 224, 224),
                    original_size=(640, 480),
                ),
                cost=1,
                original_index=i,
            )
            for i in range(4)
        ]

        result = preprocessor.collate(prepared, device="cpu")

        assert "pixel_values" in result
        assert result["pixel_values"].shape == (4, 3, 224, 224)

    def test_collate_empty_returns_empty_tensor(self, fake_val_preproc):
        """collate() on an empty list returns a 0-element tensor (matches ImagePreprocessor)."""
        preprocessor = OpenCLIPImagePreprocessor(fake_val_preproc, "test-model")
        result = preprocessor.collate([], device="cpu")

        assert result["pixel_values"].numel() == 0


class TestNemoColEmbedPreprocessor:
    """Tests for NemoColEmbedPreprocessor."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        return tokenizer

    @pytest.fixture
    def mock_model_config(self):
        """Create mock model config."""
        config = MagicMock()
        config.template = "bidirectional-llama-retriever"
        return config

    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image as bytes."""
        img = Image.new("RGB", (640, 480), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()

    @pytest.fixture
    def mock_config(self):
        """Create mock model config for prepare()."""
        config = MagicMock()
        config.max_sequence_length = 10240
        return config

    def test_modality(self, mock_tokenizer, mock_model_config):
        """NemoColEmbedPreprocessor reports correct modality."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
        )

        assert preprocessor.modality == "image"

    def test_build_prompt_single_tile(self, mock_tokenizer, mock_model_config):
        """Build prompt creates correct format for single tile."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
            num_image_token=256,
        )

        prompt = preprocessor._build_prompt(num_tiles=1)

        # Should have 256 IMG_CONTEXT tokens for 1 tile
        assert prompt.startswith("passage: <img>")
        assert prompt.endswith("</img> ")
        assert prompt.count("<IMG_CONTEXT>") == 256

    def test_build_prompt_multiple_tiles(self, mock_tokenizer, mock_model_config):
        """Build prompt creates correct format for multiple tiles."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
            num_image_token=256,
        )

        prompt = preprocessor._build_prompt(num_tiles=3)

        # Should have 256 * 3 = 768 IMG_CONTEXT tokens for 3 tiles
        assert prompt.count("<IMG_CONTEXT>") == 768

    def test_dynamic_preprocess_single_tile(self):
        """Dynamic preprocess creates single tile for square image."""
        from sie_server.core.preprocessor import _dynamic_preprocess

        img = Image.new("RGB", (448, 448), color="blue")
        tiles = _dynamic_preprocess(img, image_size=448, max_num=6)

        # Square 448x448 image should produce 1 tile
        assert len(tiles) == 1
        assert tiles[0].size == (448, 448)

    def test_dynamic_preprocess_multiple_tiles(self):
        """Dynamic preprocess creates multiple tiles for wide image."""
        from sie_server.core.preprocessor import _dynamic_preprocess

        # Wide image should create multiple tiles
        img = Image.new("RGB", (896, 448), color="green")
        tiles = _dynamic_preprocess(img, image_size=448, max_num=6)

        # 2:1 aspect ratio should produce 2 tiles
        assert len(tiles) == 2

    def test_prepare_creates_payload(self, mock_tokenizer, mock_model_config, mock_config, sample_image_bytes):
        """Prepare creates NemoColEmbedPayload with correct fields."""
        from sie_server.core.prepared import NemoColEmbedPayload
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        # Mock tokenizer return value
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
        )

        items: list[Item] = [
            Item(images=[ImageInput(data=sample_image_bytes)]),
        ]

        batch = preprocessor.prepare(items, config=mock_config)

        assert batch.modality == "image"
        assert len(batch.items) == 1
        assert batch.total_cost > 0

        payload = batch.items[0].payload
        assert isinstance(payload, NemoColEmbedPayload)
        assert payload.pixel_values is not None
        assert payload.input_ids is not None
        assert payload.attention_mask is not None
        assert payload.num_tiles > 0
        assert payload.original_size == (640, 480)

    def test_collate_concatenates_variable_tiles(self, mock_tokenizer, mock_model_config):
        """Collate concatenates pixel values from items with different tile counts."""
        from sie_server.core.prepared import NemoColEmbedPayload, PreparedItem
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        mock_tokenizer.pad_token_id = 0

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
        )

        # Create items with different tile counts
        prepared = [
            PreparedItem(
                payload=NemoColEmbedPayload(
                    pixel_values=torch.randn(2, 3, 448, 448),  # 2 tiles
                    input_ids=torch.tensor([1, 2, 3]),
                    attention_mask=torch.tensor([1, 1, 1]),
                    num_tiles=2,
                    original_size=(896, 448),
                ),
                cost=2,
                original_index=0,
            ),
            PreparedItem(
                payload=NemoColEmbedPayload(
                    pixel_values=torch.randn(1, 3, 448, 448),  # 1 tile
                    input_ids=torch.tensor([1, 2]),
                    attention_mask=torch.tensor([1, 1]),
                    num_tiles=1,
                    original_size=(448, 448),
                ),
                cost=1,
                original_index=1,
            ),
        ]

        result = preprocessor.collate(prepared, device="cpu")

        # pixel_values should be concatenated: 2 + 1 = 3 tiles total
        assert result["pixel_values"].shape == (3, 3, 448, 448)

        # input_ids should be padded to max length (3)
        assert result["input_ids"].shape == (2, 3)
        assert result["attention_mask"].shape == (2, 3)

    def test_collate_empty(self, mock_tokenizer, mock_model_config):
        """Collate empty list returns empty tensors."""
        from sie_server.core.preprocessor import NemoColEmbedPreprocessor

        preprocessor = NemoColEmbedPreprocessor(
            tokenizer=mock_tokenizer,
            model_config=mock_model_config,
            model_name="test-model",
        )

        result = preprocessor.collate([], device="cpu")

        assert result["pixel_values"].numel() == 0
        assert result["input_ids"].numel() == 0
        assert result["attention_mask"].numel() == 0
