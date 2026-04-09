"""GroundingDINO adapter for open-vocabulary object detection.

GroundingDINO enables zero-shot object detection by combining DINO with
grounded pre-training. Given an image and text labels, it detects objects
matching those labels with bounding boxes and confidence scores.

Key features:
- Zero-shot detection: No fine-tuning needed for new object classes
- Open-vocabulary: Detect any object describable in text
- COCO zero-shot: 52.5 AP (grounding-dino-base)

Architecture: Swin Transformer (vision) + BERT (text) + cross-modal fusion

Target models:
- IDEA-Research/grounding-dino-tiny (172M params)
- IDEA-Research/grounding-dino-base (~340M params)

Usage:
    client.extract(
        "IDEA-Research/grounding-dino-base",
        [Item(images=["photo.jpg"])],
        labels=["person", "car", "dog"],
    )

See: https://huggingface.co/docs/transformers/model_doc/grounding-dino
"""

from __future__ import annotations

import gc
import logging
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from PIL import Image as PILImage

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.types.inputs import ImageInput
from sie_server.types.responses import DetectedObject

if TYPE_CHECKING:
    from PIL.Image import Image

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_NO_LABELS = "GroundingDINOAdapter requires labels for object detection"
_ERR_ENCODE_NOT_SUPPORTED = "GroundingDINOAdapter does not support encode(). Use extract() instead."


class GroundingDINOAdapter(ModelAdapter):
    """Adapter for GroundingDINO open-vocabulary object detection.

    GroundingDINO combines DINO with grounded pre-training for zero-shot
    object detection. Given an image and text labels, it detects objects
    matching those labels with bounding boxes and confidence scores.

    This adapter implements extract() for object detection tasks with
    efficient batched inference.
    """

    __slots__ = (
        "_box_threshold",
        "_compute_precision",
        "_device",
        "_device_type",
        "_model",
        "_model_dtype",
        "_model_name_or_path",
        "_preprocessor",
        "_processor",
        "_text_threshold",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "float16",
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold

        self._model: Any = None
        self._processor: Any = None
        self._preprocessor: Any = None
        self._device: str | None = None
        self._device_type: str = "cpu"
        self._model_dtype: torch.dtype = torch.float32

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            inputs=["image"],
            outputs=["json"],
        )

    @property
    def dims(self) -> ModelDims:
        return ModelDims()

    def load(self, device: str) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        from sie_server.core.preprocessor import DetectionPreprocessor

        self._device = device
        self._device_type = "cuda" if device.startswith("cuda") else "cpu"
        dtype = self._resolve_dtype(device)

        logger.info(
            "Loading GroundingDINO model %s on device=%s with dtype=%s",
            self._model_name_or_path,
            device,
            dtype,
        )

        self._processor = AutoProcessor.from_pretrained(self._model_name_or_path)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
        )
        self._model.to(device)
        self._model.eval()
        self._model_dtype = next(self._model.parameters()).dtype

        # Create preprocessor with image_processor for CPU/GPU overlap
        self._preprocessor = DetectionPreprocessor(
            image_processor=self._processor.image_processor,
            model_name=self._model_name_or_path,
        )

        logger.info("GroundingDINO model loaded successfully")

    def _resolve_dtype(self, device: str) -> torch.dtype:
        if not device.startswith("cuda"):
            return torch.float32
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
            self._compute_precision, torch.float16
        )

    def unload(self) -> None:
        device = self._device
        self._model = None
        self._processor = None
        self._preprocessor = None
        self._device = None
        self._device_type = "cpu"

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

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
        """Detect objects in images matching the given labels.

        Uses batched inference for efficient GPU utilization.
        If prepared_items is provided (from DetectionPreprocessor), uses
        preprocessed pixel_values tensors for efficiency.
        """
        del output_schema, instruction  # Unused

        model = self._model
        processor = self._processor
        if model is None or processor is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        if not labels:
            raise ValueError(_ERR_NO_LABELS)

        # Extract options once
        opts = options or {}
        box_threshold = opts.get("box_threshold", self._box_threshold)
        text_threshold = opts.get("text_threshold", self._text_threshold)

        # Build text prompt once (shared across batch)
        text_prompt = " ".join(f"{label.lower().strip()}." for label in labels)

        n_items = len(items)

        if prepared_items:
            # Use preprocessed tensors from DetectionPreprocessor
            valid_items = [p for p in prepared_items if hasattr(p, "payload") and p.payload is not None]

            if not valid_items:
                return ExtractOutput(entities=[[] for _ in range(n_items)], objects=[[] for _ in range(n_items)])

            # Stack pixel values and collect metadata
            pixel_values = torch.stack([p.payload.pixel_values for p in valid_items])
            original_sizes = [p.payload.original_size for p in valid_items]
            image_indices = [p.original_index for p in valid_items]

            # Batched inference with preprocessed tensors
            batch_results = self._detect_batch(
                text_prompt=text_prompt,
                labels=labels,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                pixel_values=pixel_values,
                original_sizes=original_sizes,
            )
        else:
            # Fallback: decode images inline
            images: list[Image] = []
            image_indices: list[int] = []

            for idx, item in enumerate(items):
                img = self._extract_image(item)
                if img is not None:
                    images.append(img)
                    image_indices.append(idx)

            if not images:
                return ExtractOutput(entities=[[] for _ in range(n_items)], objects=[[] for _ in range(n_items)])

            # Batched inference with PIL images
            batch_results = self._detect_batch(
                text_prompt=text_prompt,
                labels=labels,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                images=images,
            )

        # Map results back to original item positions
        all_objects: list[list[DetectedObject]] = [[] for _ in range(n_items)]
        for result_idx, item_idx in enumerate(image_indices):
            all_objects[item_idx] = batch_results[result_idx]

        return ExtractOutput(entities=[[] for _ in range(n_items)], objects=all_objects)

    def _detect_batch(
        self,
        text_prompt: str,
        labels: list[str],
        box_threshold: float,
        text_threshold: float,
        *,
        pixel_values: torch.Tensor | None = None,
        original_sizes: list[tuple[int, int]] | None = None,
        images: list[Image] | None = None,
    ) -> list[list[DetectedObject]]:
        """Run batched detection on multiple images.

        Supports two modes:
        1. Preprocessed: pixel_values tensor + original_sizes (from DetectionPreprocessor)
        2. Fallback: PIL images directly (legacy path, for testing)

        Args:
            text_prompt: Formatted text prompt for all images.
            labels: Original labels for output mapping.
            box_threshold: Minimum box confidence.
            text_threshold: Minimum text confidence.
            pixel_values: Preprocessed image tensor [B, C, H, W] (optional).
            original_sizes: List of (width, height) tuples for bbox denormalization.
            images: List of PIL Images (fallback if pixel_values not provided).

        Returns:
            List of detected object lists, one per input image.
        """
        processor = self._processor
        model = self._model
        device = self._device

        if pixel_values is not None and original_sizes is not None:
            # Preprocessed path: use tensors + tokenizer only
            batch_size = pixel_values.shape[0]

            # Tokenize text (fast: ~0.1ms)
            text_inputs = processor.tokenizer(
                [text_prompt] * batch_size,
                return_tensors="pt",
                padding=True,
            )

            # Build inputs dict with preprocessed pixel_values
            inputs = {
                "pixel_values": pixel_values.to(device),
                "input_ids": text_inputs["input_ids"].to(device),
                "attention_mask": text_inputs["attention_mask"].to(device),
            }

            # Target sizes for post-processing (height, width)
            target_sizes = [(h, w) for w, h in original_sizes]

        elif images is not None:
            # Fallback: full processor with PIL images
            target_sizes = [(img.height, img.width) for img in images]

            inputs = processor(
                images=images,
                text=[text_prompt] * len(images),
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

        else:
            raise ValueError("Either pixel_values+original_sizes or images must be provided")

        # Single forward pass with autocast
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self._device_type,
                dtype=self._model_dtype,
                enabled=(self._device_type == "cuda"),
            ),
        ):
            outputs = model(**inputs)

        # Batched post-processing - returns list of dicts
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )

        # Convert results to DetectedObject format
        return [self._results_to_objects(result) for result in results]

    def _results_to_objects(self, result: dict[str, Any]) -> list[DetectedObject]:
        """Convert post-processed detection result to DetectedObject list."""
        boxes = result["boxes"]
        scores = result["scores"]
        result_labels = result.get("text_labels", result.get("labels", []))

        n_detections = len(boxes)
        if n_detections == 0:
            return []

        objects = []
        for i in range(n_detections):
            box = boxes[i]
            x1, y1, x2, y2 = box.tolist()
            label_text = result_labels[i]

            objects.append(
                DetectedObject(
                    label=label_text,
                    score=float(scores[i]),
                    bbox=[int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                )
            )

        return objects

    def _extract_image(self, item: Item) -> Image | None:
        """Extract PIL Image from item.

        Expects ImageInput format (SDK wire format with .data bytes).
        Returns None if no valid image.
        """
        images = item.images
        if not images:
            return None

        img = images[0]
        # ImageInput is a TypedDict (dict) - check if it has the required "data" key
        if not isinstance(img, dict) or "data" not in img:
            return None

        pil_img = PILImage.open(BytesIO(img["data"]))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return pil_img

    def get_preprocessor(self) -> Any | None:
        """Return preprocessor for CPU/GPU overlap.

        Returns DetectionPreprocessor which preprocesses images on CPU
        (PIL decode + image_processor → tensors) while GPU processes
        previous batch.
        """
        return self._preprocessor
