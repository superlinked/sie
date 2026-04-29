from __future__ import annotations

import gc
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.types.responses import Entity

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_NO_IMAGES = "GlmOcrAdapter requires image input for extraction"
_ERR_ENCODE_NOT_SUPPORTED = "GlmOcrAdapter does not support encode(). Use extract() instead."

DEFAULT_USER_TEXT = "Text Recognition:"


class GlmOcrAdapter(ModelAdapter):
    """Adapter for zai-org/GLM-OCR document OCR model.

    GLM-OCR uses a CogViT visual encoder + GLM-0.5B autoregressive decoder
    to produce Markdown text from document images.

    Requires transformers >= 5.0 for GlmOcrForConditionalGeneration.
    Must use bfloat16 precision (float16 produces garbage output).

    This adapter implements extract() for document OCR tasks.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "bfloat16",
        max_new_tokens: int = 8192,
        num_beams: int = 1,
        attn_implementation: str = "sdpa",
        user_text: str = DEFAULT_USER_TEXT,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            compute_precision: Compute precision for inference.
            max_new_tokens: Maximum tokens to generate.
            num_beams: Number of beams for beam search.
            attn_implementation: Attention implementation - "eager", "sdpa", or "flash_attention_2".
            user_text: Default user text appended to the image in the chat prompt.
            **kwargs: Ignored extra arguments from the loader.
        """
        del kwargs  # Unused - accepts normalize, max_seq_length, etc.
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams
        self._attn_implementation = attn_implementation
        self._user_text = user_text

        self._model: Any = None
        self._processor: Any = None
        self._preprocessor: Any = None
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
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._device = device

        dtype = self._resolve_dtype(device)

        logger.info(
            "Loading GLM-OCR model %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._attn_implementation,
        )

        self._processor = AutoProcessor.from_pretrained(self._model_name_or_path)

        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=self._attn_implementation,
        )

        self._model.to(device)  # ty: ignore[invalid-argument-type]
        self._model.eval()

        self._create_preprocessor()

        logger.info("GLM-OCR model loaded successfully")

    def _resolve_dtype(self, device: str) -> torch.dtype:
        """Resolve dtype based on device and config."""
        if not device.startswith("cuda"):
            return torch.float32

        if self._compute_precision == "float16":
            msg = "GLM-OCR does not support float16 on CUDA (produces garbage output). Use bfloat16."
            raise ValueError(msg)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._compute_precision)
        if dtype is None:
            msg = f"Unsupported compute_precision: {self._compute_precision!r}. Use 'bfloat16' or 'float32'."
            raise ValueError(msg)
        return dtype

    def _create_preprocessor(self) -> None:
        """Create SIE preprocessor for CPU preprocessing."""
        from sie_server.core.preprocessor import GlmOcrPreprocessor

        self._preprocessor = GlmOcrPreprocessor(
            processor=self._processor,
            model_name=self._model_name_or_path,
            user_text=self._user_text,
        )

        logger.info("Created GlmOcrPreprocessor for CPU preprocessing")

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
        """Return the GlmOcrPreprocessor for CPU/GPU overlap.

        Returns:
            GlmOcrPreprocessor instance or None if not loaded.
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
        """Not supported - GLM-OCR is an extraction model."""
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
        """Extract Markdown text from document images.

        Args:
            items: List of items with images.
            labels: Entity labels (unused, for API compatibility).
            output_schema: Optional schema (unused, for API compatibility).
            instruction: Optional instruction replacing the default user text.
            options: Adapter options:
                - max_new_tokens: Override max tokens to generate
                - num_beams: Override beam search width
            prepared_items: Optional preprocessed items from GlmOcrPreprocessor.

        Returns:
            ExtractOutput with entities containing Markdown text.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        options = options or {}
        max_new_tokens = options.get("max_new_tokens", self._max_new_tokens)
        num_beams = options.get("num_beams", self._num_beams)

        if prepared_items is not None and len(prepared_items) > 0:
            if len(prepared_items) != len(items):
                msg = f"prepared_items length ({len(prepared_items)}) must match items length ({len(items)})"
                raise ValueError(msg)
            return self._extract_preprocessed(
                items=items,
                prepared_items=prepared_items,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

        all_entities = []
        for item in items:
            entities = self._extract_single(
                item,
                instruction=instruction,
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
        max_new_tokens: int,
        num_beams: int,
    ) -> ExtractOutput:
        """Extract from preprocessed items.

        Args:
            items: Original items (for reference/fallback).
            prepared_items: Preprocessed items with GlmOcrPayload.
            max_new_tokens: Max tokens to generate.
            num_beams: Beam search width.

        Returns:
            ExtractOutput with entities.
        """
        from sie_server.core.prepared import GlmOcrPayload, PreparedItem

        all_entities = []

        for i, prepared in enumerate(prepared_items):
            if isinstance(prepared, PreparedItem):
                payload = prepared.payload
            else:
                payload = getattr(prepared, "payload", prepared)

            if not isinstance(payload, GlmOcrPayload):
                entities = self._extract_single(
                    items[i],
                    instruction=None,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
                all_entities.append(entities)
                continue

            inputs = {
                k: v.to(device=self._device, dtype=self._model.dtype) if v.is_floating_point() else v.to(self._device)
                for k, v in payload.inputs.items()
            }
            prompt_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                )

            generated_ids = output_ids[0, prompt_len:]
            generated_text = self._processor.decode(generated_ids, skip_special_tokens=True)

            entities = self._convert_output(generated_text)
            all_entities.append(entities)

        return ExtractOutput(entities=all_entities)

    def _extract_single(
        self,
        item: Item,
        *,
        instruction: str | None,
        max_new_tokens: int,
        num_beams: int,
    ) -> list[Entity]:
        """Extract from a single item.

        Args:
            item: Item with images.
            instruction: Optional instruction replacing the default user text.
            max_new_tokens: Max tokens to generate.
            num_beams: Beam search width.

        Returns:
            List of entities extracted from the item.
        """
        from PIL import Image as PILImage

        images = item.images
        if not images or len(images) == 0:
            raise ValueError(_ERR_NO_IMAGES)

        img_bytes = images[0]["data"]
        pil_img = PILImage.open(io.BytesIO(img_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        text = instruction or self._user_text
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": text},
                ],
            }
        ]

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: v.to(device=self._device, dtype=self._model.dtype) if v.is_floating_point() else v.to(self._device)
            for k, v in inputs.items()
        }
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
            )

        generated_ids = output_ids[0, prompt_len:]
        generated_text = self._processor.decode(generated_ids, skip_special_tokens=True)

        return self._convert_output(generated_text)

    @staticmethod
    def _convert_output(text: str) -> list[Entity]:
        """Convert generated text to SIE entity format.

        Args:
            text: Generated Markdown text from the model.

        Returns:
            List with a single Entity containing the Markdown text.
        """
        return [Entity(text=text.strip(), label="markdown", score=1.0)]
