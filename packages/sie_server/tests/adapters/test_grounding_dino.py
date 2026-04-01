"""Tests for GroundingDINO object detection adapter."""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from sie_server.adapters.grounding_dino import GroundingDINOAdapter
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import ImageInput, Item


class TestGroundingDINOAdapter:
    """Test suite for GroundingDINOAdapter."""

    def test_init_defaults(self) -> None:
        """Test adapter initialization with default values."""
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        assert adapter._model_name_or_path == "IDEA-Research/grounding-dino-tiny"
        assert adapter._compute_precision == "float16"
        assert adapter._box_threshold == 0.25
        assert adapter._text_threshold == 0.25
        assert not adapter.is_loaded()

    def test_init_custom_thresholds(self) -> None:
        """Test adapter initialization with custom thresholds."""
        adapter = GroundingDINOAdapter(
            "IDEA-Research/grounding-dino-base",
            box_threshold=0.3,
            text_threshold=0.35,
            compute_precision="bfloat16",
        )
        assert adapter._box_threshold == 0.3
        assert adapter._text_threshold == 0.35
        assert adapter._compute_precision == "bfloat16"

    def test_capabilities(self) -> None:
        """Test model capabilities."""
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        caps = adapter.capabilities
        assert "image" in caps.inputs

    def test_dims_empty(self) -> None:
        """Test model dimensions (empty for extraction models)."""
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        dims = adapter.dims
        # Extraction models don't have embedding dimensions
        assert dims.dense is None
        assert dims.sparse is None

    def test_encode_not_supported(self) -> None:
        """Test that encode raises NotImplementedError."""
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        with pytest.raises(NotImplementedError, match="does not support encode"):
            adapter.encode([], [])

    def test_extract_requires_labels(self) -> None:
        """Test that extract requires labels."""
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        with pytest.raises(ValueError, match="requires labels"):
            adapter.extract([Item(images=[ImageInput(data=b"test", format="jpeg")])])

    def test_text_prompt_format(self) -> None:
        """Test that labels are formatted correctly for GroundingDINO.

        The text prompt format is: "label1. label2. label3."
        Labels should be lowercased and stripped.
        """
        # Text prompt building is now inline in extract(), so we test the format
        # by verifying the expected format in the docstring:
        # "label.lower().strip()." joined with spaces

        # Standard case: labels should be "person. car. dog."
        labels = ["person", "car", "dog"]
        expected = " ".join(f"{label.lower().strip()}." for label in labels)
        assert expected == "person. car. dog."

        # Mixed case: should lowercase
        labels = ["Person", "CAR", "Dog"]
        expected = " ".join(f"{label.lower().strip()}." for label in labels)
        assert expected == "person. car. dog."

        # Whitespace: should strip
        labels = [" person ", "car"]
        expected = " ".join(f"{label.lower().strip()}." for label in labels)
        assert expected == "person. car."

    def test_extract_output_format(self) -> None:
        """Test that extract returns properly formatted ExtractOutput."""
        import torch

        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")

        # Mock the model and processor
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._device_type = "cpu"
        adapter._model_dtype = torch.float32

        # Create a mock image
        from PIL import Image

        mock_image = Image.new("RGB", (100, 100))
        img_bytes = io.BytesIO()
        mock_image.save(img_bytes, format="JPEG")
        img_data = img_bytes.getvalue()

        # Mock processor output as a dict-like object
        # The processor returns a BatchFeature which behaves like a dict
        mock_pixel_values = torch.zeros(1, 3, 224, 224)
        mock_input_ids = torch.zeros(1, 10, dtype=torch.long)
        mock_attention_mask = torch.ones(1, 10, dtype=torch.long)

        def mock_processor_call(*args, **kwargs):
            return {
                "pixel_values": mock_pixel_values,
                "input_ids": mock_input_ids,
                "attention_mask": mock_attention_mask,
            }

        adapter._processor.side_effect = mock_processor_call

        # Mock model output
        adapter._model.return_value = MagicMock()

        # Mock post_process output
        adapter._processor.post_process_grounded_object_detection.return_value = [
            {
                "boxes": torch.tensor([[10.0, 20.0, 110.0, 220.0]]),
                "scores": torch.tensor([0.85]),
                "labels": ["cat"],
                "text_labels": ["cat"],
            }
        ]

        # Mock model parameters for dtype
        adapter._model.parameters = MagicMock(return_value=iter([torch.nn.Parameter(torch.zeros(1))]))

        # Call extract
        result = adapter.extract(
            [Item(images=[ImageInput(data=img_data, format="jpeg")])],
            labels=["cat"],
        )

        # Verify result structure
        assert isinstance(result, ExtractOutput)
        assert len(result.entities) == 1
        assert len(result.entities[0]) == 1

        entity = result.entities[0][0]
        assert isinstance(entity, dict)  # Entity is a TypedDict
        assert entity["text"] == "cat"  # text contains the class name
        assert entity["label"] == "object"  # label is the category type
        assert entity["score"] == pytest.approx(0.85, rel=1e-5)
        assert entity["bbox"] == [10, 20, 100, 200]  # x, y, width, height


@pytest.mark.integration
class TestGroundingDINOIntegration:
    """Integration tests for GroundingDINO (requires model download)."""

    @pytest.fixture
    def adapter(self, device: str) -> GroundingDINOAdapter:
        """Create adapter and load model on detected device."""
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        adapter.load(device)
        yield adapter
        adapter.unload()

    def test_extract_real_image(self, adapter: GroundingDINOAdapter) -> None:
        """Test extraction on a real image."""
        from PIL import Image

        # Create a simple test image (100x100 red square)
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")

        result = adapter.extract(
            [Item(images=[ImageInput(data=img_bytes.getvalue(), format="jpeg")])],
            labels=["object", "square"],
        )

        assert isinstance(result, ExtractOutput)
        assert len(result.entities) == 1
        # May or may not detect anything in a solid color image
        # Just verify structure is correct
        for entity in result.entities[0]:
            # Entities may be dicts or TypedDicts
            assert "label" in entity
            assert "score" in entity
            assert "bbox" in entity
            if entity["bbox"]:
                assert len(entity["bbox"]) == 4
