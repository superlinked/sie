"""Tests for the OWLv2 object detection adapter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image
from sie_server.adapters.owlv2.adapter import Owlv2Adapter
from sie_server.types.inputs import Item


def test_load_uses_slow_processor_protocol() -> None:
    adapter = Owlv2Adapter("google/owlv2-base-patch16-ensemble", revision="deadbeef")
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(dtype="float32")])

    with (
        patch("transformers.Owlv2Processor.from_pretrained", return_value=mock_processor) as load_processor,
        patch("transformers.Owlv2ForObjectDetection.from_pretrained", return_value=mock_model) as load_model,
    ):
        adapter.load("cpu")

    kwargs = load_processor.call_args.kwargs
    assert kwargs["revision"] == "deadbeef"
    assert kwargs["use_fast"] is False
    model_kwargs = load_model.call_args.kwargs
    assert model_kwargs["dtype"] is torch.float32
    assert "torch_dtype" not in model_kwargs


def test_results_to_objects_bulk_converts_tensors_once() -> None:
    adapter = Owlv2Adapter("google/owlv2-base-patch16-ensemble")
    boxes = MagicMock()
    boxes.__len__.return_value = 2
    boxes.detach.return_value.cpu.return_value.tolist.return_value = [
        [10.9, 20.1, 110.7, 220.8],
        [-2.9, 3.9, 7.8, 13.2],
    ]
    scores = MagicMock()
    scores.detach.return_value.cpu.return_value.tolist.return_value = [0.85, 0.125]
    label_indices = MagicMock()
    label_indices.detach.return_value.cpu.return_value.tolist.return_value = [0, 7]

    objects = adapter._results_to_objects(
        {
            "boxes": boxes,
            "scores": scores,
            "labels": label_indices,
        },
        ["cat", "dog"],
    )

    assert objects == [
        {"label": "cat", "score": 0.85, "bbox": [10, 20, 99, 200]},
        {"label": "class_7", "score": 0.125, "bbox": [-2, 3, 10, 9]},
    ]
    for tensor in (boxes, scores, label_indices):
        tensor.detach.assert_called_once_with()
        tensor.detach.return_value.cpu.assert_called_once_with()
        tensor.detach.return_value.cpu.return_value.tolist.assert_called_once_with()


def test_detection_postprocesses_with_image_processor() -> None:
    """Postprocessing uses the component that owns the stable API."""
    adapter = Owlv2Adapter("google/owlv2-base-patch16-ensemble")
    adapter._model = MagicMock(return_value=MagicMock())
    adapter._processor = MagicMock()
    adapter._device = "cpu"
    adapter._device_type = "cpu"
    adapter._model_dtype = torch.float32

    adapter._processor.return_value = {
        "pixel_values": torch.zeros(1, 3, 224, 224),
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    adapter._processor.image_processor.post_process_object_detection.return_value = [
        {
            "boxes": torch.tensor([[10.0, 20.0, 110.0, 220.0]]),
            "scores": torch.tensor([0.85]),
            "labels": torch.tensor([0]),
        }
    ]

    result = adapter._detect_batch(
        text_queries=["a photo of cat"],
        labels=["cat"],
        score_threshold=0.1,
        images=[Image.new("RGB", (320, 240))],
    )

    adapter._processor.image_processor.post_process_object_detection.assert_called_once()
    adapter._processor.post_process_object_detection.assert_not_called()
    postprocess = adapter._processor.image_processor.post_process_object_detection
    assert torch.equal(postprocess.call_args.kwargs["target_sizes"], torch.tensor([[320, 320]]))
    assert result == [[{"label": "cat", "score": pytest.approx(0.85), "bbox": [10, 20, 100, 200]}]]


def test_detection_with_prepared_pixels_tokenizes_flat_queries_and_restores_target_sizes() -> None:
    """The production prepared-tensor path preserves OWLv2's batch protocol."""
    adapter = Owlv2Adapter("google/owlv2-base-patch16-ensemble")
    outputs = MagicMock()
    adapter._model = MagicMock(return_value=outputs)
    adapter._processor = MagicMock()
    adapter._device = "cpu"
    adapter._device_type = "cpu"
    adapter._model_dtype = torch.float32
    input_ids = torch.arange(32).reshape(4, 8)
    attention_mask = torch.ones(4, 8, dtype=torch.long)
    adapter._processor.tokenizer.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    adapter._processor.image_processor.post_process_object_detection.return_value = [
        {
            "boxes": torch.tensor([[1.0, 2.0, 11.0, 22.0]]),
            "scores": torch.tensor([0.75]),
            "labels": torch.tensor([1]),
        },
        {
            "boxes": torch.empty((0, 4)),
            "scores": torch.empty(0),
            "labels": torch.empty(0, dtype=torch.long),
        },
    ]
    pixel_values = torch.ones(2, 3, 4, 4)

    result = adapter._detect_batch(
        text_queries=["a photo of cat", "a photo of dog"],
        labels=["cat", "dog"],
        score_threshold=0.2,
        pixel_values=pixel_values,
        original_sizes=[(320, 240), (640, 480)],
    )

    adapter._processor.tokenizer.assert_called_once_with(
        text=["a photo of cat", "a photo of dog", "a photo of cat", "a photo of dog"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    model_inputs = adapter._model.call_args.kwargs
    assert torch.equal(model_inputs["pixel_values"], pixel_values)
    assert torch.equal(model_inputs["input_ids"], input_ids)
    assert torch.equal(model_inputs["attention_mask"], attention_mask)
    postprocess = adapter._processor.image_processor.post_process_object_detection
    assert postprocess.call_args.kwargs["outputs"] is outputs
    assert torch.equal(postprocess.call_args.kwargs["target_sizes"], torch.tensor([[320, 320], [640, 640]]))
    assert postprocess.call_args.kwargs["threshold"] == 0.2
    assert result == [
        [{"label": "dog", "score": pytest.approx(0.75), "bbox": [1, 2, 10, 20]}],
        [],
    ]


def test_extract_normalizes_label_whitespace() -> None:
    adapter = Owlv2Adapter("google/owlv2-base-patch16-ensemble")
    adapter._model = MagicMock()
    adapter._processor = MagicMock()
    prepared = SimpleNamespace(
        original_index=0,
        payload=SimpleNamespace(
            pixel_values=torch.zeros(3, 224, 224),
            original_size=(224, 224),
        ),
    )

    with patch.object(adapter, "_detect_batch", return_value=[[]]) as detect_batch:
        adapter.extract(
            [Item(images=[{"data": b"not-decoded"}])],
            labels=[" Car "],
            prepared_items=[prepared],
        )

    assert detect_batch.call_args.kwargs["text_queries"] == ["a photo of car"]


def test_fused_requests_map_prepared_results_positionally() -> None:
    adapter = Owlv2Adapter("google/owlv2-base-patch16-ensemble")
    adapter._model = MagicMock()
    adapter._processor = MagicMock()
    first = [{"label": "first", "score": 1.0, "bbox": [0, 0, 1, 1]}]
    second = [{"label": "second", "score": 1.0, "bbox": [1, 1, 1, 1]}]
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

    with patch.object(adapter, "_detect_batch", return_value=[first, second]) as detect_batch:
        result = adapter.extract([Item(), Item()], labels=["object"], prepared_items=prepared)

    assert result.objects == [first, second]
    assert torch.equal(
        detect_batch.call_args.kwargs["pixel_values"],
        torch.stack([prepared[0].payload.pixel_values, prepared[1].payload.pixel_values]),
    )
