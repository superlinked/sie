"""SAM 3 adapter for open-vocabulary object detection.

SAM 3 (Segment Anything Model 3, Meta) finds every instance of a concept given
as a short noun phrase ("yellow school bus"). Each caller label is one concept
prompt. The adapter encodes each image once and runs the detector once per
label against those shared vision features, the multi-prompt pattern the
Transformers documentation describes.

Scores come from ``Sam3Processor.post_process_object_detection``: the
per-instance probability multiplied by the per-prompt presence probability.
A detection is kept when its score is strictly above ``score_threshold``.

SAM 3 also predicts an instance mask for each detection. This adapter does not
return masks. It returns the standard ``objects`` list (label, score and an
``[x, y, w, h]`` box in integer pixels of the original image).

Target model: facebook/sam3 (SAM License, gated on the Hugging Face Hub).
It needs Transformers 5 (``Sam3Model`` and ``Sam3Processor``).

Usage:
    client.extract(
        "facebook/sam3",
        [Item(images=["photo.jpg"])],
        labels=["person", "yellow school bus"],
    )

See: https://huggingface.co/docs/transformers/model_doc/sam3
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.core.preprocessor import DetectionPreprocessor
from sie_server.core.preprocessor.vision import collect_detection_prepared_items
from sie_server.types.inputs import decode_image
from sie_server.types.responses import DetectedObject

if TYPE_CHECKING:
    from PIL.Image import Image

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

# Default of Sam3Processor.post_process_object_detection (Transformers 5.14 to 5.19).
DEFAULT_SCORE_THRESHOLD = 0.3
# Sam3ImageProcessor resizes every image to a 1008 x 1008 square.
_DEFAULT_IMAGE_SIDE = 1008
# Sam3Config's CLIP text encoder has 32 positions. Sam3Processor pads every prompt to 32 tokens.
_DEFAULT_TEXT_POSITIONS = 32
# Fields of a Sam3 output that post-processing reads. They are scored in float32 so that
# a bfloat16 forward pass does not quantise scores near the threshold.
_SCORED_OUTPUTS = ("pred_logits", "pred_boxes", "presence_logits")

_ERR_NO_LABELS = "Sam3Adapter requires labels: one short noun phrase per kind of object to detect"
_ERR_BAD_LABEL = "Sam3Adapter labels must be non-empty strings"
_ERR_THRESHOLD = "score_threshold must be a number between 0 and 1"
_ERR_ENCODE_NOT_SUPPORTED = "Sam3Adapter does not support encode(). Use extract() instead."
_ERR_TRANSFORMERS = (
    "SAM 3 needs Sam3Model and Sam3Processor from Transformers 5; serve it from the transformers5 bundle"
)


def _score_threshold(value: Any) -> float:
    """``value`` as a threshold in [0, 1], or ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(_ERR_THRESHOLD)
    if not 0 <= value <= 1:
        raise ValueError(_ERR_THRESHOLD)
    return float(value)


def _label_prompts(labels: list[str] | None) -> list[tuple[str, str]]:
    """``(label, prompt)`` for each distinct caller label, in request order.

    The prompt is the label with surrounding whitespace removed; the label is
    returned to the caller unchanged. A repeated label is prompted once.
    """
    if not labels:
        raise ValueError(_ERR_NO_LABELS)
    prompts: dict[str, str] = {}
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError(_ERR_BAD_LABEL)
        prompts.setdefault(label, label.strip())
    return list(prompts.items())


def _image_side(image_processor: Any) -> int:
    """The side of the square Sam3ImageProcessor resizes every image to."""
    size = getattr(image_processor, "size", None) or {}
    side = size.get("height") if isinstance(size, dict) else getattr(size, "height", None)
    return int(side) if side else _DEFAULT_IMAGE_SIDE


def _scored_in_float32(outputs: Any) -> Any:
    """``outputs`` with the fields post-processing reads cast to float32."""
    for name in _SCORED_OUTPUTS:
        value = getattr(outputs, name, None)
        if isinstance(value, torch.Tensor):
            setattr(outputs, name, value.float())
    return outputs


def _to_objects(result: dict[str, Any], label: str, size: tuple[int, int]) -> list[DetectedObject]:
    """Post-processed ``xyxy`` boxes as ``[x, y, w, h]`` integer boxes clipped to the image.

    ``size`` is the original ``(width, height)``. Coordinates are clipped to
    the image, then truncated to integers, the way the other detectors convert
    boxes, so ``x + w`` never exceeds the image width.
    """
    boxes = result["boxes"].detach().float().cpu().tolist()
    scores = result["scores"].detach().float().cpu().tolist()
    width, height = size
    objects: list[DetectedObject] = []
    for (x1, y1, x2, y2), score in zip(boxes, scores, strict=True):
        x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
        y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
        objects.append(
            DetectedObject(
                label=label,
                score=float(score),
                bbox=[int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            )
        )
    return objects


class Sam3Adapter(BaseAdapter):
    """Adapter for SAM 3 text-prompted, open-vocabulary object detection.

    Given images and text labels, returns every instance of each label with a
    bounding box and a confidence score. Masks are not returned.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("image",),
        outputs=("json",),
        unload_fields=("_model", "_processor", "_preprocessor"),
        default_preprocessor="image",
    )

    __slots__ = (
        "_compute_precision",
        "_device",
        "_device_type",
        "_model",
        "_model_dtype",
        "_model_name_or_path",
        "_preprocessor",
        "_processor",
        "_revision",
        "_score_threshold",
        "_text_positions",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._revision = revision
        self._score_threshold = _score_threshold(score_threshold)

        self._model: Any = None
        self._processor: Any = None
        self._preprocessor: Any = None
        self._device: str | None = None
        self._device_type: str = "cpu"
        self._model_dtype: torch.dtype = torch.float32
        self._text_positions: int = _DEFAULT_TEXT_POSITIONS

    def load(self, device: str) -> None:
        import transformers

        model_class = getattr(transformers, "Sam3Model", None)
        processor_class = getattr(transformers, "Sam3Processor", None)
        if model_class is None or processor_class is None:
            raise RuntimeError(_ERR_TRANSFORMERS)

        self._device = device
        self._device_type = "cuda" if device.startswith("cuda") else "cpu"
        dtype = self._resolve_dtype()

        shared_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        logger.info("Loading SAM 3 model %s on device=%s with dtype=%s", self._model_name_or_path, device, dtype)

        self._processor = processor_class.from_pretrained(self._model_name_or_path, **shared_kwargs)
        self._model = model_class.from_pretrained(self._model_name_or_path, dtype=dtype, **shared_kwargs)
        self._model.to(device)
        self._model.eval()
        self._model_dtype = dtype

        text_config = getattr(self._model.config, "text_config", None)
        positions = getattr(text_config, "max_position_embeddings", None)
        self._text_positions = positions if isinstance(positions, int) and positions > 0 else _DEFAULT_TEXT_POSITIONS

        image_processor = self._processor.image_processor
        # Images larger than twice the model input are shrunk on the CPU first,
        # as for OWLv2. Boxes still map to the size the caller sent.
        self._preprocessor = DetectionPreprocessor(
            image_processor=image_processor,
            model_name=self._model_name_or_path,
            max_side=2 * _image_side(image_processor),
        )

        logger.info("SAM 3 model loaded successfully")

    def _resolve_dtype(self) -> torch.dtype:
        if not self._device or not str(self._device).startswith("cuda"):
            return torch.float32
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
            self._compute_precision, torch.bfloat16
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: list[Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        raise NotImplementedError(_ERR_ENCODE_NOT_SUPPORTED)

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        """Detect every instance of each label in each item's first image.

        Every label and option is validated, and every label tokenized,
        before any image reaches the model.
        """
        del output_schema, instruction  # Unused

        self._check_loaded()
        if self._processor is None:
            raise RuntimeError(ERR_NOT_LOADED)

        prompts = _label_prompts(labels)
        opts = options or {}
        score_threshold = _score_threshold(opts.get("score_threshold", opts.get("threshold", self._score_threshold)))
        text_inputs = self._tokenize(prompts)

        n_items = len(items)
        no_objects = ExtractOutput(entities=[[] for _ in range(n_items)], objects=[[] for _ in range(n_items)])

        if prepared_items:
            prepared_batch = collect_detection_prepared_items(prepared_items)
            if prepared_batch is None:
                return no_objects
            pixel_values, _pixel_mask, original_sizes, image_indices = prepared_batch
        else:
            images: list[Image] = []
            image_indices = []
            for idx, item in enumerate(items):
                img = self._extract_image(item, item_index=idx)
                if img is not None:
                    images.append(img)
                    image_indices.append(idx)
            if not images:
                return no_objects
            pixel_values = self._processor.image_processor(images=images, return_tensors="pt")["pixel_values"]
            original_sizes = [(img.width, img.height) for img in images]

        detections = self._detect(pixel_values, original_sizes, text_inputs, score_threshold)

        all_objects: list[list[DetectedObject]] = [[] for _ in range(n_items)]
        for result_idx, item_idx in enumerate(image_indices):
            all_objects[item_idx] = detections[result_idx]

        return ExtractOutput(entities=[[] for _ in range(n_items)], objects=all_objects)

    def _tokenize(self, prompts: list[tuple[str, str]]) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """``(label, input_ids, attention_mask)`` per label, each ``[1, seq_len]``.

        Uses the processor's own text path, which pads every prompt to the
        text encoder's 32 positions. A longer prompt is rejected, not
        truncated: SAM 3 is prompted with short noun phrases.
        """
        tokenized: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        with self._tokenizer_guard():
            for label, prompt in prompts:
                encoded = self._processor(text=prompt, return_tensors="pt")
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                if input_ids.shape[-1] > self._text_positions:
                    n_tokens = int(attention_mask.sum())
                    msg = (
                        f"SAM 3 label {label!r} is {n_tokens} tokens; the text encoder takes at most "
                        f"{self._text_positions}. Use a short noun phrase."
                    )
                    raise ValueError(msg)
                tokenized.append((label, input_ids, attention_mask))
        return tokenized

    def _detect(
        self,
        pixel_values: torch.Tensor,
        original_sizes: list[tuple[int, int]],
        text_inputs: list[tuple[str, torch.Tensor, torch.Tensor]],
        score_threshold: float,
    ) -> list[list[DetectedObject]]:
        """Detect each label in a batch of images, encoding the images once.

        Args:
            pixel_values: Processed images ``[B, 3, H, W]``.
            original_sizes: ``(width, height)`` of each image as the caller sent it.
            text_inputs: Tokenized labels from ``_tokenize``.
            score_threshold: Keep detections scoring strictly above this.

        Returns:
            One list of detections per image, highest score first.
        """
        model = self._model
        processor = self._processor
        device = self._device
        batch_size = pixel_values.shape[0]
        # post_process_object_detection takes (height, width).
        target_sizes = [(height, width) for width, height in original_sizes]
        detections: list[list[DetectedObject]] = [[] for _ in range(batch_size)]

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self._device_type,
                dtype=self._model_dtype,
                enabled=(self._device_type == "cuda"),
            ),
        ):
            vision_embeds = model.get_vision_features(pixel_values=pixel_values.to(device, dtype=self._model_dtype))
            for label, input_ids, attention_mask in text_inputs:
                outputs = model(
                    vision_embeds=vision_embeds,
                    input_ids=input_ids.to(device).repeat(batch_size, 1),
                    attention_mask=attention_mask.to(device).repeat(batch_size, 1),
                )
                results = processor.post_process_object_detection(
                    _scored_in_float32(outputs),
                    threshold=score_threshold,
                    target_sizes=target_sizes,
                )
                for index, result in enumerate(results):
                    detections[index].extend(_to_objects(result, label, original_sizes[index]))

        for found in detections:
            found.sort(key=lambda detected: detected["score"], reverse=True)
        return detections

    def _extract_image(self, item: Item, *, item_index: int | None = None) -> Image | None:
        """The item's first image as RGB, or ``None`` when it has none.

        Expects the SDK wire format (an ImageInput dict with ``data`` bytes).
        """
        images = item.images
        if not images:
            return None

        img = images[0]
        if not isinstance(img, dict) or "data" not in img:
            return None

        # decode_image raises InvalidMediaError (-> 400 INVALID_INPUT) on
        # non-bytes or undecodable payloads, and converts to RGB.
        return decode_image(img, item_index=item_index, image_index=0)

    def get_preprocessor(self) -> Any | None:
        """The DetectionPreprocessor that decodes and resizes images on the CPU."""
        return self._preprocessor
