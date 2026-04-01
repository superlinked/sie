"""Florence-2 adapter for document understanding and visual extraction.

Florence-2 is Microsoft's vision foundation model supporting multiple tasks:
- OCR: Text extraction from images
- OCR_WITH_REGION: Text extraction with bounding boxes
- OD: Object detection
- CAPTION: Image captioning
- DENSE_REGION_CAPTION: Dense captioning with regions

Architecture: DaViT vision encoder + BERT text encoder + Transformer decoder

For KIE (Key Information Extraction), use OCR_WITH_REGION task which returns
text with quad_boxes (4-corner bounding boxes).

See: https://huggingface.co/microsoft/Florence-2-base
"""

from __future__ import annotations

import gc
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from transformers import PreTrainedModel

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.types.responses import Entity

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_NO_IMAGES = "Florence2Adapter requires image input for extraction"
_ERR_ENCODE_NOT_SUPPORTED = "Florence2Adapter does not support encode(). Use extract() instead."

# Florence-2 task prompts
TASK_OCR = "<OCR>"
TASK_OCR_WITH_REGION = "<OCR_WITH_REGION>"
TASK_OD = "<OD>"
TASK_CAPTION = "<CAPTION>"
TASK_DETAILED_CAPTION = "<DETAILED_CAPTION>"
TASK_DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
TASK_REGION_PROPOSAL = "<REGION_PROPOSAL>"
TASK_CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
TASK_DOCVQA = "<DocVQA>"


class Florence2Adapter(ModelAdapter):
    """Adapter for Florence-2 vision-language models.

    Florence-2 is a multi-task vision foundation model that can perform:
    - OCR (text extraction)
    - Object detection
    - Image captioning
    - Region-based tasks

    This adapter implements extract() for document understanding tasks.
    Uses HuggingFace AutoModelForCausalLM and AutoProcessor.

    Performance note (Dec 2025):
        CPU preprocessing uses ImagePreprocessor pattern via prepared_items.
        Batched inference supported. Flash attention enabled when available.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        default_task: str = TASK_OCR_WITH_REGION,
        compute_precision: ComputePrecision = "float16",
        trust_remote_code: bool = True,
        max_new_tokens: int = 1024,
        num_beams: int = 1,
        attn_implementation: str = "eager",
        **kwargs: Any,  # Accept and ignore extra kwargs (e.g., normalize, max_seq_length)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            default_task: Default task prompt if not specified in options.
            compute_precision: Compute precision for inference.
            trust_remote_code: Whether to trust remote code (required for Florence-2).
            max_new_tokens: Maximum tokens to generate.
            num_beams: Number of beams for beam search.
            attn_implementation: Attention implementation - "eager", "sdpa", or "flash_attention_2".
                Use "flash_attention_2" for optimized inference (requires flash-attn package).
            **kwargs: Ignored extra arguments from the loader.
        """
        del kwargs  # Unused - accepts normalize, max_seq_length, etc.
        self._model_name_or_path = str(model_name_or_path)
        self._default_task = default_task
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams
        self._attn_implementation = attn_implementation

        self._model: Any = None
        self._processor: Any = None  # HuggingFace AutoProcessor
        self._preprocessor: Any = None  # SIE Florence2Preprocessor for CPU preprocessing
        self._device: str | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            inputs=["image"],
            outputs=["json"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return model dimensions (empty for extraction models)."""
        return ModelDims()

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = device

        # Determine dtype
        dtype = self._resolve_dtype(device)

        logger.info(
            "Loading Florence-2 model %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._attn_implementation,
        )

        # Load processor with trust_remote_code (required for Florence-2 custom processor)
        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # Load model with trust_remote_code to use model's custom code
        # attn_implementation options: "eager", "sdpa", "flash_attention_2"
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=self._trust_remote_code,
            attn_implementation=self._attn_implementation,
        )

        self._model.to(device)  # ty: ignore[invalid-argument-type]
        self._model.eval()

        # Create SIE preprocessor for CPU preprocessing
        self._create_preprocessor()

        logger.info("Florence-2 model loaded successfully")

    def _resolve_dtype(self, device: str) -> torch.dtype:
        """Resolve dtype based on device and config."""
        if not device.startswith("cuda"):
            return torch.float32

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.float16)

    def _create_preprocessor(self) -> None:
        """Create SIE preprocessor for CPU preprocessing.

        Uses the HuggingFace processor for actual preprocessing but wraps it
        in SIE's Florence2Preprocessor for integration with the worker infrastructure.
        """
        from sie_server.core.preprocessor import Florence2Preprocessor

        self._preprocessor = Florence2Preprocessor(
            processor=self._processor,
            model_name=self._model_name_or_path,
            default_task=self._default_task,
        )

        logger.info("Created Florence2Preprocessor for CPU preprocessing")

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device

        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        if self._preprocessor is not None:
            del self._preprocessor
            self._preprocessor = None

        self._device = None

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    def get_preprocessor(self) -> Any | None:
        """Return the Florence2Preprocessor for CPU/GPU overlap.

        Returns:
            Florence2Preprocessor instance or None if not loaded.
        """
        return self._preprocessor

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
        """Not supported - Florence-2 is an extraction model."""
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
        """Extract information from document images.

        Args:
            items: List of items with images.
            labels: Entity labels (used for phrase grounding if provided).
            output_schema: Optional schema (unused, for API compatibility).
            instruction: Optional instruction (used as custom prompt if provided).
            options: Adapter options:
                - task: Task prompt (default: OCR_WITH_REGION)
                - max_new_tokens: Override max tokens to generate
                - num_beams: Override beam search width
            prepared_items: Optional preprocessed items from Florence2Preprocessor.
                If provided, uses precomputed pixel_values and input_ids.

        Returns:
            List of extraction results with keys:
            - entities: List of {text, label, score, bbox} for OCR_WITH_REGION
            - objects: List of {label, score, bbox} for OD
            - data: Raw parsed output from model
        """
        if self._model is None or self._processor is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        options = options or {}
        task = options.get("task", self._default_task)
        max_new_tokens = options.get("max_new_tokens", self._max_new_tokens)
        num_beams = options.get("num_beams", self._num_beams)

        # Build task prompt
        prompt = self._build_prompt(task, labels, instruction)

        # Use preprocessed items if available
        if prepared_items is not None and len(prepared_items) > 0:
            return self._extract_preprocessed(
                items=items,
                prepared_items=prepared_items,
                task=task,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

        # Fallback to inline preprocessing
        all_entities = []
        for item in items:
            entities = self._extract_single(
                item,
                prompt=prompt,
                task=task,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )
            all_entities.append(entities)

        return ExtractOutput(entities=all_entities)

    def _extract_preprocessed(
        self,
        items: list[Item],
        prepared_items: list[Any],
        *,
        task: str,
        max_new_tokens: int,
        num_beams: int,
    ) -> ExtractOutput:
        """Extract from preprocessed items.

        Uses precomputed pixel_values and input_ids from Florence2Preprocessor.
        Processes items one at a time (autoregressive decoding).

        Args:
            items: Original items (for reference).
            prepared_items: Preprocessed items with Florence2Payload.
            task: Task token for post-processing.
            max_new_tokens: Max tokens to generate.
            num_beams: Beam search width.

        Returns:
            ExtractOutput with entities.
        """
        from sie_server.core.prepared import Florence2Payload, PreparedItem

        all_entities = []

        for i, prepared in enumerate(prepared_items):
            # Extract payload
            if isinstance(prepared, PreparedItem):
                payload = prepared.payload
            else:
                payload = getattr(prepared, "payload", prepared)

            if not isinstance(payload, Florence2Payload):
                # Fallback to inline preprocessing if payload is wrong type
                entities = self._extract_single(
                    items[i],
                    prompt=task,
                    task=task,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
                all_entities.append(entities)
                continue

            # Move tensors to device with correct dtype
            pixel_values = payload.pixel_values.unsqueeze(0).to(self._device, dtype=self._model.dtype)
            input_ids = payload.input_ids.unsqueeze(0).to(self._device)

            # Generate
            # NOTE: use_cache=False works around a bug in Florence-2's cached model code
            # where prepare_inputs_for_generation doesn't handle past_key_values=None.
            # Still broken in transformers 4.57.6 despite prepare_inputs_for_generation
            # looking fixed — decoder forward pass crashes with 'NoneType' has no 'shape'.
            with torch.inference_mode():
                generated_ids = self._model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                    use_cache=False,
                )

            # Decode
            generated_text = self._processor.batch_decode(
                generated_ids,
                skip_special_tokens=False,
            )[0]

            # Post-process
            parsed = self._processor.post_process_generation(
                generated_text,
                task=task,
                image_size=payload.original_size,
            )

            # Convert to SIE format
            entities = self._convert_output(parsed, task, payload.original_size)
            all_entities.append(entities)

        return ExtractOutput(entities=all_entities)

    def _build_prompt(
        self,
        task: str,
        labels: list[str] | None,
        instruction: str | None,
    ) -> str:
        """Build the task prompt.

        Args:
            task: Task token (e.g., <OCR_WITH_REGION>).
            labels: Optional labels for phrase grounding.
            instruction: Optional custom instruction.

        Returns:
            Complete prompt string.
        """
        # Use instruction as custom prompt if provided
        if instruction:
            return f"{task}{instruction}"

        # For phrase grounding, append labels
        if task == TASK_CAPTION_TO_PHRASE_GROUNDING and labels:
            label_text = ", ".join(labels)
            return f"{task}{label_text}"

        return task

    def _extract_single(
        self,
        item: Item,
        *,
        prompt: str,
        task: str,
        max_new_tokens: int,
        num_beams: int,
    ) -> list[Entity]:
        """Extract from a single item.

        Args:
            item: Item with images.
            prompt: Task prompt.
            task: Task token for post-processing.
            max_new_tokens: Max tokens to generate.
            num_beams: Beam search width.

        Returns:
            List of entities extracted from the item.
        """
        from PIL import Image as PILImage

        # Validate input
        images = item.images
        if not images or len(images) == 0:
            raise ValueError(_ERR_NO_IMAGES)

        # Load image
        img_bytes = images[0]["data"]
        pil_img = PILImage.open(io.BytesIO(img_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        image_size = (pil_img.width, pil_img.height)

        # Process inputs
        inputs = self._processor(
            text=prompt,
            images=pil_img,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(self._device, dtype=self._model.dtype) if v.dtype == torch.float32 else v.to(self._device)
            for k, v in inputs.items()
        }

        # Generate
        # NOTE: use_cache=False works around a bug in Florence-2's cached model code.
        # Still broken in transformers 4.57.6 — see comment in _extract_preprocessed.
        with torch.inference_mode():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                use_cache=False,
            )

        # Decode
        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]

        # Post-process
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=image_size,
        )

        # Convert to SIE format
        return self._convert_output(parsed, task, image_size)

    def _convert_output(
        self,
        parsed: dict[str, Any],
        task: str,
        image_size: tuple[int, int],
    ) -> list[Entity]:
        """Convert Florence-2 output to SIE extraction format.

        Args:
            parsed: Parsed output from processor.post_process_generation().
            task: Task token.
            image_size: Original image size (width, height).

        Returns:
            List of entities extracted from the output.
        """
        entities: list[Entity] = []

        task_output = parsed.get(task, {})

        if task == TASK_OCR_WITH_REGION:
            # OCR with region returns quad_boxes (4-corner points) and labels (text)
            quad_boxes = task_output.get("quad_boxes", [])
            labels = task_output.get("labels", [])

            for quad_box, text in zip(quad_boxes, labels, strict=False):
                # Convert quad box to normalized bbox [x1, y1, x2, y2]
                bbox = self._quad_to_bbox(quad_box, image_size)
                entities.append(
                    Entity(
                        text=text,
                        label="text",  # Generic label for OCR
                        score=1.0,  # Florence-2 doesn't return confidence
                        bbox=[int(b) for b in bbox],  # Convert to int list
                    )
                )

        elif task in (TASK_OD, TASK_DENSE_REGION_CAPTION, TASK_REGION_PROPOSAL):
            # Object detection returns bboxes and labels
            bboxes = task_output.get("bboxes", [])
            labels = task_output.get("labels", [])

            for bbox, label in zip(bboxes, labels, strict=False):
                # Normalize bbox
                norm_bbox = self._normalize_bbox(bbox, image_size)
                entities.append(
                    Entity(
                        text=label or "object",
                        label="object",
                        score=1.0,
                        bbox=[int(b) for b in norm_bbox],  # Convert to int list
                    )
                )

        elif task == TASK_OCR:
            # Plain OCR returns just text
            text = task_output if isinstance(task_output, str) else str(task_output)
            entities.append(
                Entity(
                    text=text,
                    label="text",
                    score=1.0,
                )
            )

        elif task in (TASK_CAPTION, TASK_DETAILED_CAPTION):
            # Captioning returns text
            text = task_output if isinstance(task_output, str) else str(task_output)
            entities.append(
                Entity(
                    text=text,
                    label="caption",
                    score=1.0,
                )
            )

        elif task == TASK_DOCVQA:
            # DocVQA returns answer text
            text = task_output if isinstance(task_output, str) else str(task_output)
            entities.append(
                Entity(
                    text=text,
                    label="answer",
                    score=1.0,
                )
            )

        return entities

    def _quad_to_bbox(
        self,
        quad_box: list[float],
        image_size: tuple[int, int],
    ) -> list[float]:
        """Convert quad box (4 corners) to normalized bbox.

        Args:
            quad_box: [x1, y1, x2, y2, x3, y3, x4, y4] - 4 corner points.
            image_size: (width, height) of original image.

        Returns:
            Normalized bbox [x1, y1, x2, y2] in range [0, 1].
        """
        width, height = image_size

        # Extract x and y coordinates
        xs = [quad_box[i] for i in range(0, 8, 2)]
        ys = [quad_box[i] for i in range(1, 8, 2)]

        # Get bounding rectangle
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # Normalize to [0, 1]
        return [
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height,
        ]

    def _normalize_bbox(
        self,
        bbox: list[float],
        image_size: tuple[int, int],
    ) -> list[float]:
        """Normalize bbox to [0, 1] range.

        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates.
            image_size: (width, height) of original image.

        Returns:
            Normalized bbox [x1, y1, x2, y2] in range [0, 1].
        """
        width, height = image_size
        return [
            bbox[0] / width,
            bbox[1] / height,
            bbox[2] / width,
            bbox[3] / height,
        ]
