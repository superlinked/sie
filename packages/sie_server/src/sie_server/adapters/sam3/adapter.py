"""SAM 3 adapter for open-vocabulary object detection.

SAM 3 (Segment Anything Model 3, Meta) finds every instance of a concept given
as a short noun phrase ("yellow school bus"). Each caller label is one concept
prompt. The adapter encodes each image once, encodes each label once (and
keeps recent label encodings), then runs the detector for every
(image, label) pair against those shared features, ``max_pairs_per_forward``
pairs per forward pass.

Scores come from ``Sam3Processor.post_process_object_detection``: the
per-instance probability multiplied by the per-prompt presence probability.
A detection is kept when its score is strictly above ``score_threshold``.

SAM 3 also predicts an instance mask for each detection. This adapter does not
return masks, so it does not compute them: the mask decoder is replaced by a
stub at load. Boxes and scores come from the detector before the mask decoder,
so they are unchanged. It returns the standard ``objects`` list (label, score
and an ``[x, y, w, h]`` box in integer pixels of the original image).

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
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
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
# (image, label) pairs in one detector forward pass. One pair per pass is the
# per-label forward of a single image exactly, so a request's boxes and scores are
# bit-identical to running each label alone. More pairs per pass (the loadtime
# option max_pairs_per_forward) run faster under concurrency, but bfloat16 GEMMs
# over a different batch shape move scores slightly, as batching images does.
_MAX_PAIRS_PER_FORWARD = 1
# Label encodings kept between requests (each 32 x 256 values; labels repeat across images).
_TEXT_CACHE_SIZE = 1024

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
    returned to the caller unchanged. A repeated label is prompted once (see
    ``Sam3Adapter._tokenize`` for labels that tokenize identically).
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


class _NoMasks(torch.nn.Module):
    """Stands in for ``Sam3Model.mask_decoder``: masks are not returned, so they are not computed.

    ``Sam3Model.forward`` calls the mask decoder after the boxes, logits and
    presence scores are final, so skipping it leaves those outputs unchanged.
    """

    def forward(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(pred_masks=None, semantic_seg=None, attentions=None)


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
        "_max_pairs_per_forward",
        "_model",
        "_model_dtype",
        "_model_name_or_path",
        "_preprocessor",
        "_processor",
        "_revision",
        "_score_threshold",
        "_text_cache",
        "_text_positions",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        max_pairs_per_forward: int = _MAX_PAIRS_PER_FORWARD,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._revision = revision
        self._score_threshold = _score_threshold(score_threshold)
        if (
            isinstance(max_pairs_per_forward, bool)
            or not isinstance(max_pairs_per_forward, int)
            or max_pairs_per_forward < 1
        ):
            msg = "max_pairs_per_forward must be a positive integer"
            raise ValueError(msg)
        self._max_pairs_per_forward = max_pairs_per_forward

        self._model: Any = None
        self._processor: Any = None
        self._preprocessor: Any = None
        self._device: str | None = None
        self._device_type: str = "cpu"
        self._model_dtype: torch.dtype = torch.float32
        self._text_positions: int = _DEFAULT_TEXT_POSITIONS
        # Prompt token ids -> the text encoder's output for that prompt ([1, 32, 256]).
        self._text_cache: OrderedDict[tuple[int, ...], torch.Tensor] = OrderedDict()

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
        self._model.mask_decoder = _NoMasks()
        self._model_dtype = dtype
        self._text_cache.clear()

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
        """``(label, input_ids, attention_mask)`` per distinct prompt, each ``[1, seq_len]``.

        Uses the processor's own text path, which pads every prompt to the
        text encoder's 32 positions. A longer prompt is rejected, not
        truncated: SAM 3 is prompted with short noun phrases. The CLIP
        tokenizer lowercases, so labels that differ only in case ("Cat" and
        "cat") are the same model input. Such a prompt runs once, and its
        detections carry the first of those labels in request order.
        """
        tokenized: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        seen: set[tuple[int, ...]] = set()
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
                key = tuple(input_ids[0][attention_mask[0].bool()].tolist())
                if key in seen:
                    continue
                seen.add(key)
                tokenized.append((label, input_ids, attention_mask))
        return tokenized

    def _detect(
        self,
        pixel_values: torch.Tensor,
        original_sizes: list[tuple[int, int]],
        text_inputs: list[tuple[str, torch.Tensor, torch.Tensor]],
        score_threshold: float,
    ) -> list[list[DetectedObject]]:
        """Detect each label in a batch of images, encoding each image once.

        Every (image, label) pair goes through the detector, up to
        ``max_pairs_per_forward`` pairs per pass, with the image's vision
        features and the label's text features. Scores and boxes are computed
        as ``Sam3Processor.post_process_object_detection`` computes them, in
        float32, and copied to the host once.

        Args:
            pixel_values: Processed images ``[B, 3, H, W]``.
            original_sizes: ``(width, height)`` of each image as the caller sent it.
            text_inputs: Tokenized labels from ``_tokenize``.
            score_threshold: Keep detections scoring strictly above this.

        Returns:
            One list of detections per image, highest score first.
        """
        model = self._model
        device = self._device
        batch_size = pixel_values.shape[0]
        detections: list[list[DetectedObject]] = [[] for _ in range(batch_size)]
        pairs = [
            (label_index, image_index) for label_index in range(len(text_inputs)) for image_index in range(batch_size)
        ]

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self._device_type,
                dtype=self._model_dtype,
                enabled=(self._device_type == "cuda"),
            ),
        ):
            # Each image is encoded on its own. Images share nothing in the vision
            # encoder, and a batch of one keeps every convolution at one input shape,
            # so cuDNN's autotuning (enabled server-wide) runs once rather than again
            # for every new batch size. The detector reads one FPN level, the last
            # one Sam3Model.forward keeps (the others feed only the mask decoder).
            features_parts: list[torch.Tensor] = []
            positions_parts: list[torch.Tensor] = []
            vision_outputs: list[type] = []
            for index in range(batch_size):
                vision = model.get_vision_features(
                    pixel_values=pixel_values[index : index + 1].to(device, dtype=self._model_dtype)
                )
                features_parts.append(vision.fpn_hidden_states[:-1][-1])
                positions_parts.append(vision.fpn_position_encoding[:-1][-1])
                vision_outputs.append(type(vision))
            # A pair's vision input: its image's features at that level. Sam3Model.forward drops the
            # last entry of each tuple and reads the one before it, so (features, None) is that level.
            vision_output = vision_outputs[0]
            features = torch.cat(features_parts)
            positions = torch.cat(positions_parts)
            text_embeds = torch.cat([self._text_embedding(ids, mask) for _, ids, mask in text_inputs])
            text_masks = torch.cat([mask for _, _, mask in text_inputs]).to(device)
            # post_process_object_detection scales relative xyxy boxes by (width, height, width, height).
            scale = torch.tensor(
                [[width, height, width, height] for width, height in original_sizes], dtype=torch.float32, device=device
            )
            scores_parts: list[torch.Tensor] = []
            boxes_parts: list[torch.Tensor] = []
            for start in range(0, len(pairs), self._max_pairs_per_forward):
                chunk = pairs[start : start + self._max_pairs_per_forward]
                label_rows = torch.tensor([label_index for label_index, _ in chunk], device=device)
                image_rows = torch.tensor([image_index for _, image_index in chunk], device=device)
                outputs = model(
                    vision_embeds=vision_output(
                        fpn_hidden_states=(features[image_rows], None),
                        fpn_position_encoding=(positions[image_rows], None),
                    ),
                    text_embeds=SimpleNamespace(pooler_output=text_embeds[label_rows]),
                    attention_mask=text_masks[label_rows],
                )
                scores = outputs.pred_logits.float().sigmoid()
                if outputs.presence_logits is not None:
                    scores = scores * outputs.presence_logits.float().sigmoid()
                scores_parts.append(scores)
                boxes_parts.append(outputs.pred_boxes.float() * scale[image_rows].unsqueeze(1))
            all_scores = torch.cat(scores_parts).cpu()
            all_boxes = torch.cat(boxes_parts).cpu()

        for row, (label_index, image_index) in enumerate(pairs):
            keep = all_scores[row] > score_threshold
            result = {"scores": all_scores[row][keep], "boxes": all_boxes[row][keep]}
            detections[image_index].extend(
                _to_objects(result, text_inputs[label_index][0], original_sizes[image_index])
            )

        for found in detections:
            found.sort(key=lambda detected: detected["score"], reverse=True)
        return detections

    def _text_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """The text encoder's output for one tokenized label (``[1, seq_len, hidden]``), cached by token ids."""
        key = tuple(input_ids[0].tolist()) + tuple(attention_mask[0].tolist())
        cached = self._text_cache.get(key)
        if cached is not None:
            self._text_cache.move_to_end(key)
            return cached
        encoded = self._model.get_text_features(
            input_ids=input_ids.to(self._device), attention_mask=attention_mask.to(self._device), return_dict=True
        ).pooler_output
        self._text_cache[key] = encoded
        if len(self._text_cache) > _TEXT_CACHE_SIZE:
            self._text_cache.popitem(last=False)
        return encoded

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
