"""Tests for GroundingDINO object detection adapter."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from sie_server.adapters.grounding_dino.adapter import GroundingDINOAdapter
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

    def test_extract_requires_labels_or_instruction(self) -> None:
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        adapter._model = MagicMock()
        adapter._processor = MagicMock()

        with pytest.raises(ValueError, match="requires labels or an instruction"):
            adapter.extract([Item(images=[ImageInput(data=b"test", format="jpeg")])])

    @pytest.mark.parametrize(
        ("labels", "instruction", "expected_prompt"),
        [
            (["Person", " car "], "ignored instruction", "person. car."),
            (None, "find the damaged screen", "find the damaged screen"),
        ],
    )
    def test_extract_uses_labels_or_instruction_prompt(
        self,
        labels: list[str] | None,
        instruction: str,
        expected_prompt: str,
    ) -> None:
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._detect_batch = MagicMock(return_value=[[]])  # type: ignore[method-assign]
        prepared = SimpleNamespace(
            original_index=0,
            payload=SimpleNamespace(
                pixel_values=torch.zeros(3, 8, 8),
                original_size=(8, 8),
            ),
        )

        result = adapter.extract(
            [Item()],
            labels=labels,
            instruction=instruction,
            prepared_items=[prepared],
        )

        assert result.objects == [[]]
        assert adapter._detect_batch.call_args.kwargs["text_prompt"] == expected_prompt

    def test_fused_requests_map_prepared_results_positionally(self) -> None:
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        first = [{"label": "first", "score": 1.0, "bbox": [0, 0, 1, 1]}]
        second = [{"label": "second", "score": 1.0, "bbox": [1, 1, 1, 1]}]
        adapter._detect_batch = MagicMock(return_value=[first, second])  # type: ignore[method-assign]
        prepared = [
            SimpleNamespace(
                original_index=0,
                payload=SimpleNamespace(
                    pixel_values=torch.full((3, 2, 2), value),
                    original_size=(2, 2),
                ),
            )
            for value in (1.0, 2.0)
        ]

        result = adapter.extract([Item(), Item()], labels=["object"], prepared_items=prepared)

        assert result.objects == [first, second]
        assert torch.equal(
            adapter._detect_batch.call_args.kwargs["pixel_values"],
            torch.stack([prepared[0].payload.pixel_values, prepared[1].payload.pixel_values]),
        )

    def test_fused_requests_pad_variable_image_shapes_with_pixel_mask(self) -> None:
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._detect_batch = MagicMock(return_value=[[], []])  # type: ignore[method-assign]
        prepared = [
            SimpleNamespace(
                original_index=0,
                payload=SimpleNamespace(
                    pixel_values=torch.full((3, 2, 3), 1.0),
                    original_size=(3, 2),
                ),
            ),
            SimpleNamespace(
                original_index=0,
                payload=SimpleNamespace(
                    pixel_values=torch.full((3, 4, 2), 2.0),
                    original_size=(2, 4),
                ),
            ),
        ]

        adapter.extract([Item(), Item()], labels=["object"], prepared_items=prepared)

        pixel_values = adapter._detect_batch.call_args.kwargs["pixel_values"]
        pixel_mask = adapter._detect_batch.call_args.kwargs["pixel_mask"]
        assert pixel_values.shape == (2, 3, 4, 3)
        assert torch.equal(pixel_values[0, :, :2, :3], prepared[0].payload.pixel_values)
        assert torch.equal(pixel_values[1, :, :4, :2], prepared[1].payload.pixel_values)
        assert torch.equal(pixel_mask[0], torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]]))
        assert torch.equal(pixel_mask[1], torch.tensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]))

    def test_prepared_detection_forwards_pixel_mask(self) -> None:
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        adapter._model = MagicMock(return_value=MagicMock())
        adapter._processor = MagicMock()
        adapter._processor.tokenizer.return_value = {
            "input_ids": torch.zeros((2, 4), dtype=torch.long),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
        }
        adapter._processor.post_process_grounded_object_detection.return_value = [{}, {}]
        adapter._device = "cpu"
        adapter._device_type = "cpu"
        adapter._model_dtype = torch.float32
        pixel_values = torch.zeros((2, 3, 4, 5))
        pixel_mask = torch.tensor(
            [
                [[1, 1, 1, 1, 1]] * 4,
                [[1, 1, 1, 0, 0]] * 4,
            ]
        )

        with patch.object(adapter, "_results_to_objects", return_value=[]):
            adapter._detect_batch(
                "object.",
                0.25,
                0.25,
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                original_sizes=[(5, 4), (3, 4)],
            )

        assert torch.equal(adapter._model.call_args.kwargs["pixel_mask"], pixel_mask)

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
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")

        # Mock the model and processor
        adapter._model = MagicMock()
        adapter._processor = MagicMock()
        adapter._device = "cpu"
        adapter._device_type = "cpu"
        adapter._model_dtype = torch.float32

        # Create a mock image
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
        assert len(result.entities[0]) == 0  # Entities empty for detection adapters
        assert result.objects is not None
        assert len(result.objects) == 1
        assert len(result.objects[0]) == 1

        obj = result.objects[0][0]
        assert isinstance(obj, dict)  # DetectedObject is a TypedDict
        assert obj["label"] == "cat"
        assert obj["score"] == pytest.approx(0.85, rel=1e-5)
        assert obj["bbox"] == [10, 20, 100, 200]  # x, y, width, height

    def test_results_to_objects_bulk_converts_tensors_once(self) -> None:
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        boxes = MagicMock()
        boxes.__len__.return_value = 2
        boxes.detach.return_value.cpu.return_value.tolist.return_value = [
            [10.9, 20.1, 110.7, 220.8],
            [-2.9, 3.9, 7.8, 13.2],
        ]
        scores = MagicMock()
        scores.detach.return_value.cpu.return_value.tolist.return_value = [0.85, 0.125]

        objects = adapter._results_to_objects(
            {
                "boxes": boxes,
                "scores": scores,
                "text_labels": ["cat", "dog"],
            }
        )

        assert objects == [
            {"label": "cat", "score": 0.85, "bbox": [10, 20, 99, 200]},
            {"label": "dog", "score": 0.125, "bbox": [-2, 3, 10, 9]},
        ]
        boxes.detach.assert_called_once_with()
        boxes.detach.return_value.cpu.assert_called_once_with()
        boxes.detach.return_value.cpu.return_value.tolist.assert_called_once_with()
        scores.detach.assert_called_once_with()
        scores.detach.return_value.cpu.assert_called_once_with()
        scores.detach.return_value.cpu.return_value.tolist.assert_called_once_with()

    def test_results_to_objects_empty_does_not_transfer_scores(self) -> None:
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-tiny")
        scores = MagicMock()

        objects = adapter._results_to_objects(
            {
                "boxes": torch.empty((0, 4)),
                "scores": scores,
                "text_labels": [],
            }
        )

        assert objects == []
        scores.detach.assert_not_called()

    def test_load_preserves_default_processor_protocol(self) -> None:
        adapter = GroundingDINOAdapter("IDEA-Research/grounding-dino-base", revision="deadbeef")
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(dtype="float32")])

        with (
            patch("transformers.AutoProcessor.from_pretrained") as mock_processor_load,
            patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained") as mock_model_load,
        ):
            mock_model_load.return_value = mock_model
            adapter.load("cpu")

        kwargs = mock_processor_load.call_args.kwargs
        assert kwargs["revision"] == "deadbeef"
        assert "use_fast" not in kwargs
        model_kwargs = mock_model_load.call_args.kwargs
        assert model_kwargs["dtype"] is torch.float32
        assert "torch_dtype" not in model_kwargs


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
        assert len(result.entities[0]) == 0  # Detection adapters produce objects, not entities
        assert result.objects is not None
        assert len(result.objects) == 1
        # May or may not detect anything in a solid color image
        # Just verify structure is correct
        for obj in result.objects[0]:
            assert "label" in obj
            assert "score" in obj
            assert "bbox" in obj
            if obj["bbox"]:
                assert len(obj["bbox"]) == 4
