"""Donut adapter for document understanding and visual extraction.

Donut (Document Understanding Transformer) is an OCR-free visual document
understanding model that directly analyzes document images without requiring
an external OCR engine.

Architecture: Swin vision encoder + BART text decoder

Supported tasks:
- Document parsing (CORD): Extracts structured key-value pairs from receipts
- Document VQA (DocVQA): Answers questions about documents
- Document classification (RVL-CDIP): Classifies document types

The model generates JSON-like structured output that is parsed using
processor.token2json().

See: https://huggingface.co/naver-clova-ix/donut-base
Paper: https://arxiv.org/abs/2111.15664
"""

from __future__ import annotations

import gc
import io
import json
import logging
import re
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
_ERR_NO_IMAGES = "DonutAdapter requires image input for extraction"
_ERR_ENCODE_NOT_SUPPORTED = "DonutAdapter does not support encode(). Use extract() instead."

# Task prompt patterns for different fine-tuned models
TASK_DOCVQA = "<s_docvqa>"
TASK_CORD = "<s_cord-v2>"
TASK_RVLCDIP = "<s_rvlcdip>"
TASK_SYNTHDOG = "<s_synthdog>"


class DonutAdapter(ModelAdapter):
    """Adapter for Donut vision-language models.

    Donut is an OCR-free document understanding transformer that:
    - Uses a Swin vision encoder to process document images
    - Uses a BART decoder to generate structured text output
    - Outputs JSON-like structured data parsed via token2json

    This adapter implements extract() for document understanding tasks.
    Uses HuggingFace VisionEncoderDecoderModel and DonutProcessor.

    Performance note (Dec 2025):
        CPU preprocessing uses ImagePreprocessor pattern via prepared_items.
        Batched inference supported. Flash attention enabled when available.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        default_task: str = TASK_CORD,
        compute_precision: ComputePrecision = "float16",
        max_new_tokens: int = 512,
        num_beams: int = 1,
        attn_implementation: str = "eager",
        **kwargs: Any,  # Accept and ignore extra kwargs (e.g., normalize, max_seq_length)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            default_task: Default task prompt token (e.g., <s_cord-v2>).
            compute_precision: Compute precision for inference.
            max_new_tokens: Maximum tokens to generate.
            num_beams: Number of beams for beam search (1 = greedy).
            attn_implementation: Attention implementation - "eager" or "sdpa".
                Use "sdpa" for optimized inference with PyTorch's scaled dot-product attention.
            **kwargs: Ignored extra arguments from the loader.
        """
        del kwargs  # Unused - accepts normalize, max_seq_length, etc.
        self._model_name_or_path = str(model_name_or_path)
        self._default_task = default_task
        self._compute_precision = compute_precision
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams
        self._attn_implementation = attn_implementation

        self._model: Any = None
        self._processor: Any = None  # HuggingFace DonutProcessor
        self._preprocessor: Any = None  # SIE DonutPreprocessor for CPU preprocessing
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
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        self._device = device

        # Determine dtype
        dtype = self._resolve_dtype(device)

        logger.info(
            "Loading Donut model %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._attn_implementation,
        )

        # Load processor
        self._processor = DonutProcessor.from_pretrained(self._model_name_or_path)

        # Load model with optional SDPA
        # Note: VisionEncoderDecoderModel supports SDPA but not flash_attention_2
        self._model = VisionEncoderDecoderModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=self._attn_implementation,
        )
        self._model.to(device)
        self._model.eval()

        # Create SIE preprocessor for CPU preprocessing
        self._create_preprocessor()

        logger.info("Donut model loaded successfully")

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
        in SIE's DonutPreprocessor for integration with the worker infrastructure.
        """
        from sie_server.core.preprocessor import DonutPreprocessor

        self._preprocessor = DonutPreprocessor(
            processor=self._processor,
            model_name=self._model_name_or_path,
            default_task=self._default_task,
        )

        logger.info("Created DonutPreprocessor for CPU preprocessing")

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
        """Return the DonutPreprocessor for CPU/GPU overlap.

        Returns:
            DonutPreprocessor instance or None if not loaded.
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
        """Not supported - Donut is an extraction model."""
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
            labels: Entity labels (unused for Donut, kept for API compatibility).
            output_schema: Optional schema (unused, for API compatibility).
            instruction: For DocVQA, this is the question to answer.
            options: Adapter options:
                - task: Task prompt token (default from config)
                - max_new_tokens: Override max tokens to generate
                - num_beams: Override beam search width
            prepared_items: Optional preprocessed items from DonutPreprocessor.
                If provided, uses precomputed pixel_values and decoder_input_ids.

        Returns:
            List of extraction results with keys:
            - entities: List of extracted entities (for CORD/parsing tasks)
            - data: Parsed JSON output from model
            - raw_text: Raw generated text before JSON parsing
        """
        if self._model is None or self._processor is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        options = options or {}
        task = options.get("task", self._default_task)
        max_new_tokens = options.get("max_new_tokens", self._max_new_tokens)
        num_beams = options.get("num_beams", self._num_beams)

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
                task=task,
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
        task: str,
        max_new_tokens: int,
        num_beams: int,
    ) -> ExtractOutput:
        """Extract from preprocessed items.

        Uses precomputed pixel_values and decoder_input_ids from DonutPreprocessor.
        Processes items one at a time (autoregressive decoding).

        Args:
            items: Original items (for reference).
            prepared_items: Preprocessed items with DonutPayload.
            task: Task token for post-processing.
            max_new_tokens: Max tokens to generate.
            num_beams: Beam search width.

        Returns:
            ExtractOutput with entities.
        """
        import re

        from sie_server.core.prepared import DonutPayload, PreparedItem

        all_entities = []

        for i, prepared in enumerate(prepared_items):
            # Extract payload
            if isinstance(prepared, PreparedItem):
                payload = prepared.payload
            else:
                payload = getattr(prepared, "payload", prepared)

            if not isinstance(payload, DonutPayload):
                # Fallback to inline preprocessing if payload is wrong type
                entities = self._extract_single(
                    items[i],
                    task=task,
                    instruction=None,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                )
                all_entities.append(entities)
                continue

            # Move tensors to device with correct dtype
            pixel_values = payload.pixel_values.unsqueeze(0).to(self._device, dtype=self._model.dtype)
            decoder_input_ids = payload.decoder_input_ids.unsqueeze(0).to(self._device)

            # Generate
            with torch.inference_mode():
                outputs = self._model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=self._model.decoder.config.max_position_embeddings,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self._processor.tokenizer.pad_token_id,
                    eos_token_id=self._processor.tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                    num_beams=num_beams,
                )

            # Decode
            sequence = self._processor.batch_decode(outputs.sequences)[0]

            # Clean up the sequence
            sequence = sequence.replace(self._processor.tokenizer.eos_token, "")
            sequence = sequence.replace(self._processor.tokenizer.pad_token, "")
            # Remove first task start token
            sequence = re.sub(r"<s_\w+>", "", sequence, count=1).strip()

            # Parse to JSON
            try:
                parsed = self._processor.token2json(sequence)
            except (ValueError, KeyError, AttributeError):
                # Fallback: try direct JSON parsing
                parsed = self._try_parse_json(sequence)

            # Convert to SIE format
            entities = self._convert_output(parsed, task, sequence)
            all_entities.append(entities)

        return ExtractOutput(entities=all_entities)

    def _extract_single(
        self,
        item: Item,
        *,
        task: str,
        instruction: str | None,
        max_new_tokens: int,
        num_beams: int,
    ) -> list[Entity]:
        """Extract from a single item.

        Args:
            item: Item with images.
            task: Task prompt token.
            instruction: Question for DocVQA tasks.
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

        # Build decoder input prompt
        prompt = self._build_prompt(task, instruction)

        # Process image
        pixel_values = self._processor(pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self._device, dtype=self._model.dtype)

        # Create decoder input ids
        decoder_input_ids = self._processor.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self._device)

        # Generate
        with torch.inference_mode():
            outputs = self._model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self._model.decoder.config.max_position_embeddings,
                max_new_tokens=max_new_tokens,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                num_beams=num_beams,
            )

        # Decode
        sequence = self._processor.batch_decode(outputs.sequences)[0]

        # Clean up the sequence
        sequence = sequence.replace(self._processor.tokenizer.eos_token, "")
        sequence = sequence.replace(self._processor.tokenizer.pad_token, "")
        # Remove first task start token
        sequence = re.sub(r"<s_\w+>", "", sequence, count=1).strip()

        # Parse to JSON
        try:
            parsed = self._processor.token2json(sequence)
        except (ValueError, KeyError, AttributeError):
            # Fallback: try direct JSON parsing
            parsed = self._try_parse_json(sequence)

        # Convert to SIE format
        return self._convert_output(parsed, task, sequence)

    def _build_prompt(self, task: str, instruction: str | None) -> str:
        """Build the decoder input prompt.

        Args:
            task: Task prompt token (e.g., <s_cord-v2>).
            instruction: Optional question for DocVQA.

        Returns:
            Complete prompt string.
        """
        if task == TASK_DOCVQA and instruction:
            # DocVQA format: <s_docvqa><s_question>{question}</s_question><s_answer>
            return f"{task}<s_question>{instruction}</s_question><s_answer>"

        return task

    def _try_parse_json(self, text: str) -> dict[str, Any]:
        """Try to parse JSON from text.

        Args:
            text: Text that might contain JSON.

        Returns:
            Parsed dict or empty dict with raw text.
        """
        # Try to find JSON-like content
        text = text.strip()

        # Handle special tokens that might remain
        text = re.sub(r"<[^>]+>", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

    def _convert_output(
        self,
        parsed: dict[str, Any],
        task: str,
        raw_text: str,
    ) -> list[Entity]:
        """Convert Donut output to SIE extraction format.

        Args:
            parsed: Parsed JSON output from token2json.
            task: Task token.
            raw_text: Raw generated text.

        Returns:
            List of entities extracted from the output.
        """
        entities: list[Entity] = []

        if task == TASK_CORD:
            # CORD receipt parsing - extract entities from menu items
            entities = self._extract_cord_entities(parsed)

        elif task == TASK_DOCVQA:
            # DocVQA - extract answer
            answer = parsed.get("answer", parsed.get("raw", ""))
            if answer:
                entities.append(
                    Entity(
                        text=answer,
                        label="answer",
                        score=1.0,
                    )
                )

        elif task == TASK_RVLCDIP:
            # Document classification - extract class
            doc_class = parsed.get("class", parsed.get("raw", ""))
            if doc_class:
                entities.append(
                    Entity(
                        text=doc_class,
                        label="document_class",
                        score=1.0,
                    )
                )

        return entities

    def _extract_cord_entities(self, parsed: dict[str, Any]) -> list[Entity]:
        """Extract entities from CORD receipt parsing output.

        CORD output typically has structure like:
        {
            "menu": [{"nm": "item name", "price": "10.00", "cnt": "1"}],
            "total": {"total_price": "10.00"},
            ...
        }

        Args:
            parsed: Parsed CORD output.

        Returns:
            List of Entity objects.
        """
        entities: list[Entity] = []

        def extract_recursive(obj: Any, prefix: str = "") -> None:
            """Recursively extract key-value pairs."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, str):
                        entities.append(
                            Entity(
                                text=value,
                                label=new_prefix,
                                score=1.0,
                            )
                        )
                    else:
                        extract_recursive(value, new_prefix)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_recursive(item, f"{prefix}[{i}]")

        extract_recursive(parsed)
        return entities
