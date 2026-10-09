"""CPU contract tests for the SAM 3 detection adapter.

Pretrained loading and the model forward pass are mocked. The adapter scores
detections as Transformers 5.19's
``Sam3ImageProcessor.post_process_object_detection`` does: score =
sigmoid(logit) x sigmoid(presence), kept when strictly above the threshold, and
relative ``xyxy`` boxes scaled to the image's ``(width, height)``.
"""

from __future__ import annotations

import io
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image
from sie_server.adapters.sam3.adapter import (
    _MAX_PAIRS_PER_FORWARD,
    DEFAULT_SCORE_THRESHOLD,
    Sam3Adapter,
    _NoMasks,
    _to_objects,
)
from sie_server.core.inference_output import ExtractOutput
from sie_server.core.prepared import DetectionPayload, PreparedItem
from sie_server.core.preprocessor import DetectionPreprocessor
from sie_server.types.inputs import InvalidMediaError, Item
from sie_server.types.responses import DetectedObject

MODEL = "facebook/sam3"
REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
QUERIES = 4


def _score(logit: float) -> float:
    """The score the post-processor gives a query with this logit and a certain presence."""
    return float(torch.sigmoid(torch.tensor(logit)))


class _FakeProcessor:
    """Sam3Processor's text tokenization, without assets."""

    START, END = 49406, 49407

    def __init__(self) -> None:
        self.image_processor = MagicMock()
        self.image_processor.size = {"height": 1008, "width": 1008}
        self.texts: list[str] = []
        self._vocab: dict[str, int] = {}
        self.prompts: dict[tuple[int, ...], str] = {}

    def __call__(self, *, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        """One id per lowercased word between start and end tokens, padded to 32, like the CLIP tokenizer."""
        assert return_tensors == "pt"
        self.texts.append(text)
        words = text.lower().split()
        ids = [self.START, *(self._vocab.setdefault(word, len(self._vocab) + 1) for word in words), self.END]
        self.prompts[tuple(ids)] = " ".join(words)
        width = max(32, len(ids))
        input_ids = torch.full((1, width), self.END, dtype=torch.long)
        input_ids[0, : len(ids)] = torch.tensor(ids)
        attention_mask = torch.zeros(1, width, dtype=torch.long)
        attention_mask[0, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _detections(rows: list[tuple[float, list[float]]]) -> tuple[list[float], list[list[float]]]:
    """Pad ``(logit, relative xyxy box)`` rows to ``QUERIES`` with queries that score near zero.

    Logits and box coordinates are exact in bfloat16, so the expected boxes
    and scores do not depend on rounding.
    """
    logits = [logit for logit, _ in rows] + [-10.0] * (QUERIES - len(rows))
    boxes = [box for _, box in rows] + [[0.0, 0.0, 0.125, 0.125]] * (QUERIES - len(rows))
    return logits, boxes


def _loaded(per_label: dict[str, list[list[tuple[float, list[float]]]]] | None = None) -> Sam3Adapter:
    """A loaded adapter whose model returns ``per_label[prompt][image]`` detections.

    The fake text encoder returns a label's token ids as its features and the
    fake vision encoder returns each image's index as its features, so every
    row of a detector forward pass can be traced back to its (image, label)
    pair. ``per_label`` is keyed by the lowercased prompt the model sees.
    """
    per_label = per_label or {}
    adapter = Sam3Adapter(MODEL)
    processor = _FakeProcessor()
    model = MagicMock()
    adapter.forwards = []  # type: ignore[attr-defined]

    def vision(*, pixel_values):
        # _prepared fills image i's pixels with i, so the features carry each image's index.
        index = pixel_values[:, 0, 0, 0].float().view(-1, 1)
        # Four FPN levels as Sam3VisionModel returns them; the detector reads the third.
        levels = (torch.full_like(index, -1.0), torch.full_like(index, -2.0), index, torch.full_like(index, -4.0))
        return SimpleNamespace(fpn_hidden_states=levels, fpn_position_encoding=levels)

    def text(*, input_ids, attention_mask, return_dict):
        assert return_dict is True
        assert input_ids.shape[0] == attention_mask.shape[0] == 1
        return SimpleNamespace(pooler_output=input_ids.float().unsqueeze(-1))

    def forward(*, vision_embeds, text_embeds, attention_mask):
        features, last = vision_embeds.fpn_hidden_states
        assert last is None
        images = [int(value) for value in features[:, 0].tolist()]
        rows = []
        for row, image in enumerate(images):
            ids = text_embeds.pooler_output[row, :, 0].long()
            tokens = ids[attention_mask[row].bool()].tolist()
            prompt = processor.prompts[tuple(tokens)]
            batch = per_label.get(prompt, [])
            rows.append(_detections(batch[image] if image < len(batch) else []))
        adapter.forwards.append(len(images))  # type: ignore[attr-defined]
        return SimpleNamespace(
            pred_logits=torch.tensor([logits for logits, _ in rows], dtype=torch.bfloat16),
            pred_boxes=torch.tensor([boxes for _, boxes in rows], dtype=torch.bfloat16),
            # A certain presence (sigmoid(30) is 1.0 in float32): the score is sigmoid(logit).
            presence_logits=torch.full((len(rows), 1), 30.0, dtype=torch.bfloat16),
            pred_masks=None,
        )

    model.get_vision_features.side_effect = vision
    model.get_text_features.side_effect = text
    model.side_effect = forward
    adapter._model = model
    adapter._processor = processor
    adapter._device = "cpu"
    adapter._device_type = "cpu"
    adapter._model_dtype = torch.float32
    return adapter


def _prepared(sizes: list[tuple[int, int]]) -> list[PreparedItem[DetectionPayload]]:
    return [
        PreparedItem(
            payload=DetectionPayload(pixel_values=torch.full((3, 4, 4), float(index)), original_size=size),
            cost=1,
            original_index=index,
        )
        for index, size in enumerate(sizes)
    ]


def _objects(output: ExtractOutput) -> list[list[DetectedObject]]:
    assert output.objects is not None
    return output.objects


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 40, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


# -- Contract and lifecycle -------------------------------------------------


def test_image_json_spec_and_unloaded_errors() -> None:
    adapter = Sam3Adapter(MODEL)
    assert adapter.capabilities.inputs == ["image"]
    assert adapter.capabilities.outputs == ["json"]
    assert adapter.dims.dense is None
    assert adapter._score_threshold == DEFAULT_SCORE_THRESHOLD == 0.3
    assert adapter._compute_precision == "bfloat16"
    assert not adapter.is_loaded()
    with pytest.raises(NotImplementedError, match=r"Use extract\(\)"):
        adapter.encode([Item()], ["dense"])
    with pytest.raises(RuntimeError, match="Model not loaded"):
        adapter.extract([Item()], labels=["cat"])


@pytest.mark.parametrize(
    ("device", "precision", "dtype"),
    [
        ("cpu", "bfloat16", torch.float32),
        ("cuda:0", "bfloat16", torch.bfloat16),
        ("cuda:0", "float16", torch.float16),
        ("cuda:0", "float32", torch.float32),
    ],
)
def test_load_uses_sam3_classes_revision_and_dtype(monkeypatch, device, precision, dtype) -> None:
    processor = SimpleNamespace(image_processor=SimpleNamespace(size={"height": 1008, "width": 1008}))
    model = MagicMock()
    model.config.text_config.max_position_embeddings = 32
    processor_loader = MagicMock(return_value=processor)
    model_loader = MagicMock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            Sam3Processor=SimpleNamespace(from_pretrained=processor_loader),
            Sam3Model=SimpleNamespace(from_pretrained=model_loader),
        ),
    )
    adapter = Sam3Adapter(MODEL, revision=REVISION, compute_precision=precision)
    adapter.load(device)

    processor_loader.assert_called_once_with(MODEL, revision=REVISION)
    model_loader.assert_called_once_with(MODEL, dtype=dtype, revision=REVISION)
    model.to.assert_called_once_with(device)
    model.eval.assert_called_once_with()
    assert adapter.is_loaded()
    assert adapter._model_dtype is dtype
    assert adapter._text_positions == 32
    # Masks are not returned, so the mask decoder is not run.
    assert isinstance(model.mask_decoder, _NoMasks)
    assert model.mask_decoder(decoder_queries=None).pred_masks is None
    preprocessor = adapter.get_preprocessor()
    assert isinstance(preprocessor, DetectionPreprocessor)
    # Photos far larger than the 1008 px model input are shrunk on the CPU first.
    assert preprocessor._max_side == 2016


def test_load_without_transformers5_sam3_fails_before_loading_assets(monkeypatch) -> None:
    processor_loader = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(Sam3Processor=SimpleNamespace(from_pretrained=processor_loader)),
    )
    with pytest.raises(RuntimeError, match="transformers5 bundle"):
        Sam3Adapter(MODEL, revision=REVISION).load("cpu")
    processor_loader.assert_not_called()


def test_unload_clears_loaded_state() -> None:
    adapter = _loaded()
    adapter.unload()
    assert adapter._model is None
    assert adapter._processor is None
    assert adapter._preprocessor is None
    assert adapter._device is None
    assert not adapter.is_loaded()


# -- Detection ----------------------------------------------------------------


def test_one_encoding_per_image_and_label_and_one_pass_per_pair() -> None:
    adapter = _loaded(
        {
            "cat": [
                [(2.0, [0.125, 0.25, 0.5, 0.625]), (-0.5, [0.5, 0.5, 0.75, 1.0])],
                [],
            ],
            "remote control": [
                [(0.5, [0.0, 0.0, 0.25, 0.25])],
                [(3.0, [0.5, 0.25, 1.0, 0.5])],
            ],
        }
    )
    output = adapter.extract(
        [Item(), Item()],
        labels=["cat", "remote control"],
        prepared_items=_prepared([(640, 480), (200, 400)]),
    )

    model = adapter._model
    # Each image is encoded once, on its own.
    assert model.get_vision_features.call_count == 2
    for call in model.get_vision_features.call_args_list:
        assert call.kwargs["pixel_values"].shape == (1, 3, 4, 4)
        assert call.kwargs["pixel_values"].dtype == torch.float32
    # Each label is encoded once, and each (image, label) pair runs the detector once.
    assert model.get_text_features.call_count == 2
    assert adapter.forwards == [1, 1, 1, 1]
    call = model.call_args
    assert call.kwargs["attention_mask"].shape == (1, 32)
    assert call.kwargs["text_embeds"].pooler_output.shape == (1, 32, 1)
    assert adapter._processor.texts == ["cat", "remote control"]

    assert output.entities == [[], []]
    first, second = _objects(output)
    # All labels' detections for an image, highest score first.
    assert [(found["label"], found["bbox"]) for found in first] == [
        ("cat", [80, 120, 240, 180]),
        ("remote control", [0, 0, 160, 120]),
        ("cat", [320, 240, 160, 240]),
    ]
    assert [found["score"] for found in first] == pytest.approx([_score(2.0), _score(0.5), _score(-0.5)])
    assert second == [{"label": "remote control", "score": pytest.approx(_score(3.0)), "bbox": [100, 100, 100, 100]}]


@pytest.mark.parametrize("pairs_per_forward", [_MAX_PAIRS_PER_FORWARD, 32])
def test_pairs_are_split_into_forward_passes_and_keep_their_detections(pairs_per_forward: int) -> None:
    labels = [f"thing{index}" for index in range(9)]
    images = 4
    per_label = {label: [[(1.0 + image, [0.0, 0.0, 0.5, 0.5])] for image in range(images)] for label in labels}
    adapter = _loaded(per_label)
    adapter._max_pairs_per_forward = pairs_per_forward
    output = adapter.extract(
        [Item() for _ in range(images)], labels=labels, prepared_items=_prepared([(100, 100)] * images)
    )

    pairs = len(labels) * images
    full, rest = divmod(pairs, pairs_per_forward)
    assert adapter.forwards == [pairs_per_forward] * full + ([rest] if rest else [])
    for image, found in enumerate(_objects(output)):
        assert [item["label"] for item in found] == labels
        assert [item["score"] for item in found] == pytest.approx([_score(1.0 + image)] * len(labels))


def test_label_encodings_are_reused_across_requests() -> None:
    adapter = _loaded({"cat": [[(2.0, [0.0, 0.0, 0.5, 0.5])]]})
    first = adapter.extract([Item()], labels=["cat", "dog"], prepared_items=_prepared([(10, 10)]))
    second = adapter.extract([Item()], labels=["dog", "cat"], prepared_items=_prepared([(10, 10)]))

    assert adapter._model.get_text_features.call_count == 2
    assert first.objects == second.objects


def test_default_threshold_and_request_options_that_override_it() -> None:
    # sigmoid(-0.84375) is 0.3007 and sigmoid(-1) is 0.2689: either side of the 0.3 default.
    rows = {"dog": [[(-0.84375, [0.0, 0.0, 0.5, 0.5]), (-1.0, [0.5, 0.5, 1.0, 1.0])]]}

    default = _loaded(rows).extract([Item()], labels=["dog"], prepared_items=_prepared([(100, 100)]))
    assert [found["score"] for found in _objects(default)[0]] == pytest.approx([_score(-0.84375)])

    lower = _loaded(rows).extract(
        [Item()], labels=["dog"], options={"score_threshold": 0.2}, prepared_items=_prepared([(100, 100)])
    )
    assert [found["score"] for found in _objects(lower)[0]] == pytest.approx([_score(-0.84375), _score(-1.0)])

    alias = _loaded(rows).extract(
        [Item()], labels=["dog"], options={"threshold": 0.5}, prepared_items=_prepared([(100, 100)])
    )
    assert alias.objects == [[]]

    loadtime = Sam3Adapter(MODEL, score_threshold=0.0)
    assert loadtime._score_threshold == 0.0


def test_labels_are_returned_verbatim_prompted_stripped_and_deduplicated() -> None:
    adapter = _loaded({"traffic light": [[(1.5, [0.0, 0.0, 1.0, 1.0])]]})
    output = adapter.extract(
        [Item()],
        labels=[" traffic light ", " traffic light "],
        prepared_items=_prepared([(10, 20)]),
    )
    assert adapter._processor.texts == ["traffic light"]
    assert adapter._model.get_text_features.call_count == 1
    assert adapter.forwards == [1]
    assert output.objects == [
        [{"label": " traffic light ", "score": pytest.approx(_score(1.5)), "bbox": [0, 0, 10, 20]}]
    ]


def test_labels_that_tokenize_identically_are_prompted_once_under_the_first_label() -> None:
    adapter = _loaded({"cat": [[(2.0, [0.0, 0.0, 0.5, 0.5])]], "remote control": [[(1.0, [0.5, 0.5, 1.0, 1.0])]]})
    output = adapter.extract(
        [Item()],
        labels=["Cat", "cat", "remote control", "CAT ", "Remote Control"],
        prepared_items=_prepared([(100, 100)]),
    )
    # Every label is tokenized (and checked for length); each distinct model input runs once.
    assert adapter._processor.texts == ["Cat", "cat", "remote control", "CAT", "Remote Control"]
    assert adapter._model.get_text_features.call_count == 2
    assert adapter.forwards == [1, 1]
    assert _objects(output) == [
        [
            {"label": "Cat", "score": pytest.approx(_score(2.0)), "bbox": [0, 0, 50, 50]},
            {"label": "remote control", "score": pytest.approx(_score(1.0)), "bbox": [50, 50, 50, 50]},
        ]
    ]


def test_no_detections_and_missing_images_return_empty_objects() -> None:
    adapter = _loaded()
    output = adapter.extract([Item(), Item()], labels=["zebra"], prepared_items=_prepared([(64, 64), (32, 32)]))
    assert output.objects == [[], []]
    assert output.entities == [[], []]

    no_payloads = [SimpleNamespace(payload=None), SimpleNamespace(payload=None)]
    adapter = _loaded()
    output = adapter.extract([Item(), Item()], labels=["zebra"], prepared_items=no_payloads)
    assert output.objects == [[], []]
    adapter._model.get_vision_features.assert_not_called()

    adapter = _loaded()
    output = adapter.extract([Item(), Item(images=[])], labels=["zebra"])
    assert output.objects == [[], []]
    adapter._model.get_vision_features.assert_not_called()


def test_prepared_items_without_payload_keep_their_positions() -> None:
    adapter = _loaded({"cup": [[(1.0, [0.0, 0.0, 0.5, 0.5])]]})
    prepared: list[Any] = [SimpleNamespace(payload=None), *_prepared([(50, 50)])]
    output = adapter.extract([Item(), Item()], labels=["cup"], prepared_items=prepared)
    assert output.objects == [[], [{"label": "cup", "score": pytest.approx(_score(1.0)), "bbox": [0, 0, 25, 25]}]]


def test_inline_images_use_the_image_processor_and_their_own_sizes() -> None:
    adapter = _loaded({"square": [[(2.0, [0.25, 0.5, 0.75, 1.0])]]})
    adapter._processor.image_processor.return_value = {"pixel_values": torch.zeros(1, 3, 4, 4)}
    output = adapter.extract([Item(images=[{"data": _png(300, 120), "format": "png"}])], labels=["square"])

    image_call = adapter._processor.image_processor.call_args
    assert [image.size for image in image_call.kwargs["images"]] == [(300, 120)]
    assert image_call.kwargs["return_tensors"] == "pt"
    assert output.objects == [[{"label": "square", "score": pytest.approx(_score(2.0)), "bbox": [75, 60, 150, 60]}]]


# -- Box conversion -----------------------------------------------------------


def test_boxes_become_clipped_integer_xywh() -> None:
    result = {
        "boxes": torch.tensor(
            [
                [10.9, 20.1, 110.7, 220.8],
                [-3.5, -0.2, 40.2, 30.9],
                [600.0, 470.0, 655.0, 490.0],
                [700.0, 500.0, 720.0, 510.0],
            ]
        ),
        "scores": torch.tensor([0.9, 0.8, 0.7, 0.6]),
    }
    objects = _to_objects(result, "box", (640, 480))
    assert [found["bbox"] for found in objects] == [
        [10, 20, 99, 200],
        [0, 0, 40, 30],
        [600, 470, 40, 10],
        [640, 480, 0, 0],
    ]
    for found in objects:
        x, y, w, h = found["bbox"]
        assert all(isinstance(value, int) for value in found["bbox"])
        assert x >= 0
        assert x + w <= 640
        assert y >= 0
        assert y + h <= 480
    assert [found["score"] for found in objects] == pytest.approx([0.9, 0.8, 0.7, 0.6])
    assert all(found["label"] == "box" for found in objects)


def test_empty_post_processed_result_has_no_objects() -> None:
    assert _to_objects({"boxes": torch.empty(0, 4), "scores": torch.empty(0)}, "box", (10, 10)) == []


# -- Error paths ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (None, "requires labels"),
        ([], "requires labels"),
        (["cat", "   "], "non-empty strings"),
        (["cat", 7], "non-empty strings"),
    ],
)
def test_invalid_labels_never_reach_the_model(labels, message) -> None:
    adapter = _loaded()
    with pytest.raises(ValueError, match=message):
        adapter.extract([Item()], labels=labels, prepared_items=_prepared([(10, 10)]))
    adapter._model.get_vision_features.assert_not_called()
    adapter._model.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"score_threshold": True},
        {"score_threshold": "0.3"},
        {"score_threshold": float("nan")},
        {"score_threshold": -0.1},
        {"score_threshold": 1.5},
        {"threshold": None},
    ],
)
def test_invalid_threshold_never_reaches_the_model(options) -> None:
    adapter = _loaded()
    with pytest.raises(ValueError, match="score_threshold must be"):
        adapter.extract([Item()], labels=["cat"], options=options, prepared_items=_prepared([(10, 10)]))
    adapter._model.get_vision_features.assert_not_called()


@pytest.mark.parametrize("value", [0, -1, True, 2.0])
def test_invalid_pairs_per_forward_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="max_pairs_per_forward must be a positive integer"):
        Sam3Adapter(MODEL, max_pairs_per_forward=value)


@pytest.mark.parametrize("value", [True, -1, 2.0, float("inf")])
def test_invalid_loadtime_threshold_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="score_threshold must be"):
        Sam3Adapter(MODEL, score_threshold=value)


def test_label_longer_than_the_text_encoder_is_rejected_before_any_image_work() -> None:
    adapter = _loaded()
    long_label = " ".join(["very"] * 40) + " long object"
    with pytest.raises(ValueError, match=r"is 44 tokens; the text encoder takes at most 32"):
        adapter.extract([Item()], labels=["cat", long_label], prepared_items=_prepared([(10, 10)]))
    adapter._model.get_vision_features.assert_not_called()
    adapter._model.assert_not_called()


def test_undecodable_inline_image_names_the_item() -> None:
    adapter = _loaded()
    items = [Item(), Item(images=[{"data": b"not an image", "format": "png"}])]
    with pytest.raises(InvalidMediaError, match=r"at `\$\.items\[1\]\.images\[0\]\.data`"):
        adapter.extract(items, labels=["cat"])
