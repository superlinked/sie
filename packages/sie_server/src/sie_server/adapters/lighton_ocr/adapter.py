from __future__ import annotations

import gc
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.core.prepared import LightOnOCRPayload, PreparedItem
from sie_server.core.preprocessor import LightOnOCRPreprocessor
from sie_server.types.inputs import InvalidMediaError, media_bytes
from sie_server.types.responses import Entity

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_NO_IMAGES = "LightOnOCRAdapter requires image input for extraction"
_ERR_ENCODE_NOT_SUPPORTED = "LightOnOCRAdapter does not support encode(). Use extract() instead."

DEFAULT_SYSTEM_PROMPT = "You are an OCR engine. Return the markdown representation of the document."


class LightOnOCRAdapter(ModelAdapter):
    """Adapter for lightonai/LightOnOCR-2-1B document OCR model.

    LightOnOCR-2-1B uses a Pixtral vision encoder + Qwen3 text decoder
    to produce Markdown text from document images.

    Requires transformers >= 5.0 for LightOnOcrForConditionalGeneration.
    Must use bfloat16 precision (float16 produces garbage output).

    This adapter implements extract() for document OCR tasks.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "bfloat16",
        max_new_tokens: int = 4096,
        num_beams: int = 1,
        attn_implementation: str = "sdpa",
        revision: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_batch_images: int = 4,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            compute_precision: Compute precision for inference.
            max_new_tokens: Maximum tokens to generate.
            num_beams: Number of beams for beam search.
            attn_implementation: Attention implementation - "eager", "sdpa", or "flash_attention_2".
            revision: Optional HuggingFace revision/branch/commit SHA to pin when loading model artifacts.
                Forwarded to ``from_pretrained(..., revision=...)``.
            system_prompt: System prompt for the chat template.
            max_batch_images: Max images per batched ``generate()`` call. Bounds KV-cache /
                vision-activation memory so a large incoming batch cannot OOM (issue #33,
                decoder-batching KV-OOM). The
                default (4) is safe on a 24GB L4 with ``max_new_tokens=4096``; raise it on
                higher-VRAM GPUs to batch more pages per call.
            **kwargs: Ignored extra arguments from the loader.
        """
        del kwargs  # Unused - accepts normalize, max_seq_length, etc.
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams
        self._attn_implementation = attn_implementation
        self._revision = revision
        self._system_prompt = system_prompt
        self._max_batch_images = max(1, max_batch_images)

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
        from transformers import (
            LightOnOcrForConditionalGeneration,  # ty: ignore[unresolved-import]
            LightOnOcrProcessor,  # ty: ignore[unresolved-import]
        )

        self._device = device

        dtype = self._resolve_dtype(device)

        shared_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        logger.info(
            "Loading LightOnOCR model %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._attn_implementation,
        )

        self._processor = LightOnOcrProcessor.from_pretrained(
            self._model_name_or_path,
            **shared_kwargs,
        )

        # Batched generation left-pads input_ids (prompt length varies with image
        # resolution), so the decoder must pad on the left and have a pad token.
        tokenizer = getattr(self._processor, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

        self._model = LightOnOcrForConditionalGeneration.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=self._attn_implementation,
            **shared_kwargs,
        )

        self._model.to(device)  # ty: ignore[invalid-argument-type]
        self._model.eval()

        self._create_preprocessor()

        logger.info("LightOnOCR model loaded successfully")

    def _resolve_dtype(self, device: str) -> torch.dtype:
        """Resolve dtype based on device and config."""
        if not device.startswith("cuda"):
            return torch.float32

        if self._compute_precision == "float16":
            msg = "LightOnOCR does not support float16 on CUDA (produces garbage output). Use bfloat16."
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
        self._preprocessor = LightOnOCRPreprocessor(
            processor=self._processor,
            model_name=self._model_name_or_path,
            system_prompt=self._system_prompt,
        )

        logger.info("Created LightOnOCRPreprocessor for CPU preprocessing")

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
        """Return the LightOnOCRPreprocessor for CPU/GPU overlap.

        Returns:
            LightOnOCRPreprocessor instance or None if not loaded.
        """
        return self._preprocessor

    def count_input_images(self, items: list[Item]) -> None:
        """OCR images are pages; avoid also emitting the generic image unit."""
        del items

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
        """Not supported - LightOnOCR is an extraction model."""
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
            instruction: Optional instruction appended to user message.
            options: Adapter options:
                - max_new_tokens: Override max tokens to generate
                - num_beams: Override beam search width
            prepared_items: Optional preprocessed items from LightOnOCRPreprocessor.

        Returns:
            ExtractOutput with entities containing Markdown text.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        for item in items:
            if item.images is None or len(item.images) != 1:
                msg = "LightOn OCR requires exactly one image per item"
                raise InvalidMediaError(msg)

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

        return ExtractOutput(entities=all_entities, pages=[1 for _ in items])

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
            items: Original items (for reference).
            prepared_items: Preprocessed items with LightOnOCRPayload.
            max_new_tokens: Max tokens to generate.
            num_beams: Beam search width.

        Returns:
            ExtractOutput with entities in original item order.

        Preprocessed payloads are run through a single batched ``generate()`` per
        sub-batch instead of one call per image, so GPU concurrency actually scales
        OCR throughput (issue #601). Images may differ in resolution (Pixtral is
        variable-resolution), so pixel_values are zero-padded to the per-batch max and
        input_ids are left-padded — mirroring transformers' own batched Pixtral path.
        """
        results: list[list[Entity]] = [[] for _ in items]
        payloads: list[Any] = []
        payload_indices: list[int] = []

        for i, prepared in enumerate(prepared_items):
            if isinstance(prepared, PreparedItem):
                payload = prepared.payload
            else:
                payload = getattr(prepared, "payload", prepared)

            if isinstance(payload, LightOnOCRPayload):
                payloads.append(payload)
                payload_indices.append(i)
            else:
                results[i] = self._extract_single(
                    items[i],
                    instruction=None,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )

        # Sub-batch to bound KV-cache / vision-activation memory (issue #33, decoder-batching KV-OOM): a large
        # incoming batch under high concurrency could otherwise OOM a 24GB L4.
        for start in range(0, len(payloads), self._max_batch_images):
            chunk = payloads[start : start + self._max_batch_images]
            chunk_indices = payload_indices[start : start + self._max_batch_images]
            texts = self._generate_batch(chunk, max_new_tokens=max_new_tokens, num_beams=num_beams)
            for text, i in zip(texts, chunk_indices):
                results[i] = self._convert_output(text)

        return ExtractOutput(entities=results, pages=[1 for _ in items])

    def _generate_batch(
        self,
        payloads: list[Any],
        *,
        max_new_tokens: int,
        num_beams: int,
    ) -> list[str]:
        """Run one batched ``generate()`` over preprocessed LightOnOCR payloads.

        Args:
            payloads: LightOnOCRPayload items (pixel_values [C,H,W], input_ids [L],
                attention_mask [L], image_sizes [2]) that may have different resolutions.
            max_new_tokens: Max tokens to generate.
            num_beams: Beam search width.

        Returns:
            Decoded Markdown text per payload, in input order.
        """
        device = self._device
        dtype = self._model.dtype

        # pixel_values: zero-pad each [C, H_i, W_i] to the per-batch max (bottom/right),
        # then stack to [N, C, H_max, W_max]. image_sizes carries the TRUE (h, w) per image
        # so the Pixtral encoder masks the padding out (mirrors HF _pad_for_batching).
        max_h = max(int(p.pixel_values.shape[1]) for p in payloads)
        max_w = max(int(p.pixel_values.shape[2]) for p in payloads)
        padded_pixels = [
            F.pad(p.pixel_values, (0, max_w - int(p.pixel_values.shape[2]), 0, max_h - int(p.pixel_values.shape[1])))
            for p in payloads
        ]
        pixel_values = torch.stack(padded_pixels).to(device=device, dtype=dtype)
        image_sizes = torch.stack([p.image_sizes for p in payloads]).to(device)

        # input_ids / attention_mask: LEFT-pad to the longest prompt so every row's real
        # tokens align at the right edge (required for correct batched greedy decode;
        # prompt length varies with image resolution).
        pad_id = self._pad_token_id()
        max_len = max(int(p.input_ids.shape[0]) for p in payloads)
        input_ids_rows: list[torch.Tensor] = []
        attention_rows: list[torch.Tensor] = []
        for p in payloads:
            n_pad = max_len - int(p.input_ids.shape[0])
            ids = p.input_ids
            mask = p.attention_mask
            if n_pad:
                ids = torch.cat([torch.full((n_pad,), pad_id, dtype=ids.dtype), ids])
                mask = torch.cat([torch.zeros(n_pad, dtype=mask.dtype), mask])
            input_ids_rows.append(ids)
            attention_rows.append(mask)
        input_ids = torch.stack(input_ids_rows).to(device)
        attention_mask = torch.stack(attention_rows).to(device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
            )

        # Left-padding right-aligns every prompt, so generation starts at column max_len.
        generated = output_ids[:, max_len:]
        return self._processor.batch_decode(generated, skip_special_tokens=True)

    def _pad_token_id(self) -> int:
        """Resolve the pad token id for left-padding (Qwen3 has one; fall back to eos)."""
        tokenizer = getattr(self._processor, "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if pad_id is None:
            pad_id = self._model.config.text_config.eos_token_id
        if isinstance(pad_id, (list, tuple)):
            pad_id = pad_id[0]
        return int(pad_id)

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
            instruction: Optional instruction to append to user message.
            max_new_tokens: Max tokens to generate.
            num_beams: Beam search width.

        Returns:
            List of entities extracted from the item.
        """
        from PIL import Image as PILImage

        images = item.images
        if not images or len(images) == 0:
            raise ValueError(_ERR_NO_IMAGES)

        img_bytes = media_bytes(images[0], kind="image")
        pil_img = PILImage.open(io.BytesIO(img_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        messages = self._build_messages(instruction)

        text = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self._processor(
            text=text,
            images=[pil_img],
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

    def _build_messages(self, instruction: str | None = None) -> list[dict[str, Any]]:
        """Build chat messages for the model.

        Args:
            instruction: Optional instruction to append to user content.

        Returns:
            List of message dicts with system and user roles.
        """
        user_content: list[dict[str, str]] = [{"type": "image"}]
        if instruction:
            user_content.append({"type": "text", "text": instruction})

        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _convert_output(text: str) -> list[Entity]:
        """Convert generated text to SIE entity format.

        Args:
            text: Generated Markdown text from the model.

        Returns:
            List with a single Entity containing the Markdown text.
        """
        return [Entity(text=text.strip(), label="markdown", score=1.0)]
