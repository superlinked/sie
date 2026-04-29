from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.types.responses import Entity

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_ERR_NO_IMAGES = "PaddleOCRVLAdapter requires image input for extraction"
_ERR_ENCODE_NOT_SUPPORTED = "PaddleOCRVLAdapter does not support encode(). Use extract() instead."
_ERR_FP16_UNSUPPORTED = "PaddleOCR-VL does not support float16 on CUDA (config pins bfloat16); use bfloat16 or float32."

# Canonical task -> prompt mapping from the PaddleOCR-VL-1.5 model card.
# Keep in sync with preprocessor/vision.py::_PADDLEOCR_VL_TASK_PROMPTS.
_TASK_PROMPTS: dict[str, str] = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
    "spotting": "Spotting:",
    "seal": "Seal Recognition:",
}
_VALID_TASKS: tuple[str, ...] = tuple(_TASK_PROMPTS)
_DEFAULT_TASK = "ocr"

_COMPAT_PATCHED_ATTR = "_sie_paddleocr_vl_compat_patched"


def _apply_transformers_compat_shim() -> None:
    """Bridge a parameter rename so PaddleOCR-VL's modeling code loads.

    PaddleOCR-VL-1.5's custom modeling code (pinned at revision
    6819afc8509ac9afa50e91b34627a7cf8f7900bb) calls
    ``transformers.masking_utils.create_causal_mask(inputs_embeds=...)``,
    but the signature in transformers 4.57.x uses ``input_embeds`` (singular).
    Without this shim, ``model.generate()`` raises:

        TypeError: create_causal_mask() got an unexpected keyword argument 'inputs_embeds'

    This patch aliases the two spellings. Remove once either (a) upstream
    updates the modeling code, or (b) SIE bumps past a transformers version
    where the parameter is spelled ``inputs_embeds``.

    Idempotent: safe to call on every ``load()``.
    """
    from transformers import masking_utils  # ty: ignore[unresolved-import]

    if getattr(masking_utils.create_causal_mask, _COMPAT_PATCHED_ATTR, False):
        return

    original = masking_utils.create_causal_mask

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        return original(*args, **kwargs)

    wrapped.__wrapped__ = original  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    setattr(wrapped, _COMPAT_PATCHED_ATTR, True)
    masking_utils.create_causal_mask = wrapped  # ty: ignore[invalid-assignment]


class PaddleOCRVLAdapter(BaseAdapter):
    """Adapter for PaddlePaddle/PaddleOCR-VL-1.5 OCR VLM.

    PaddleOCR-VL-1.5 is a 0.9B-param autoregressive VLM combining a NaViT-style
    SigLIP vision encoder with an ERNIE-4.5-0.3B decoder. Supports 109
    languages and six task modes: ocr, table, formula, chart, spotting, seal.

    Ships custom modeling code via ``auto_map`` — requires
    ``trust_remote_code=True``. Runs natively under PyTorch + HuggingFace
    transformers; no PaddlePaddle dependency.

    Uses ``bfloat16`` on CUDA (per ``config.json``; fp16 is unsupported).
    Falls back to fp32 on CPU/MPS for smoke testing.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("image",),
        outputs=("json",),
        unload_fields=("_model", "_processor", "_preprocessor"),
        default_preprocessor="image",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "bfloat16",
        trust_remote_code: bool = True,
        revision: str | None = None,
        default_task: str = _DEFAULT_TASK,
        max_new_tokens: int = 4096,
        num_beams: int = 1,
        attn_implementation: str = "sdpa",
        **kwargs: Any,
    ) -> None:
        del kwargs  # Accept + discard loader extras (normalize, max_seq_length, etc.)
        if default_task not in _VALID_TASKS:
            msg = f"default_task {default_task!r} must be one of {_VALID_TASKS}"
            raise ValueError(msg)
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._revision = revision
        self._default_task = default_task
        self._max_new_tokens = max_new_tokens
        self._num_beams = num_beams
        self._attn_implementation = attn_implementation

        self._model: Any = None
        self._processor: Any = None
        self._preprocessor: Any = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        # Apply the create_causal_mask compat shim before from_pretrained
        # triggers the custom modeling code to bind its imports.
        _apply_transformers_compat_shim()

        from transformers import AutoModelForCausalLM, AutoProcessor

        self._device = device
        dtype = self._resolve_dtype_for(device)

        shared_kwargs: dict[str, Any] = {"trust_remote_code": self._trust_remote_code}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        logger.info(
            "Loading PaddleOCR-VL model %s on device=%s dtype=%s revision=%s attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._revision,
            self._attn_implementation,
        )

        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path,
            **shared_kwargs,
        )
        # Model's ``auto_map`` registers ``AutoModelForCausalLM`` (not
        # ``AutoModelForImageTextToText``), even though the README suggests
        # the latter. Using the registered class avoids "Unrecognized
        # configuration class" on transformers 4.57.
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name_or_path,
            dtype=dtype,
            attn_implementation=self._attn_implementation,
            **shared_kwargs,
        )
        self._model.to(device)  # ty: ignore[invalid-argument-type]
        self._model.eval()

        self._create_preprocessor()
        logger.info("PaddleOCR-VL model loaded successfully")

    def _resolve_dtype_for(self, device: str) -> torch.dtype:
        if not device.startswith("cuda"):
            return torch.float32
        if self._compute_precision == "float16":
            raise ValueError(_ERR_FP16_UNSUPPORTED)
        dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(self._compute_precision)
        if dtype is None:
            msg = f"Unsupported compute_precision: {self._compute_precision!r}. Use 'bfloat16' or 'float32'."
            raise ValueError(msg)
        return dtype

    def _create_preprocessor(self) -> None:
        from sie_server.core.preprocessor.vision import PaddleOCRVLPreprocessor

        self._preprocessor = PaddleOCRVLPreprocessor(
            processor=self._processor,
            model_name=self._model_name_or_path,
            default_task=self._default_task,
        )

    def get_preprocessor(self) -> Any | None:
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
        self._check_loaded()
        if self._processor is None:
            raise RuntimeError(ERR_NOT_LOADED)

        options = options or {}
        task = options.get("task", self._default_task)
        if task not in _VALID_TASKS:
            msg = f"task {task!r} must be one of {_VALID_TASKS}"
            raise ValueError(msg)
        max_new_tokens = options.get("max_new_tokens", self._max_new_tokens)
        num_beams = options.get("num_beams", self._num_beams)

        if prepared_items is not None and len(prepared_items) > 0:
            if len(prepared_items) != len(items):
                msg = f"prepared_items length ({len(prepared_items)}) must match items length ({len(items)})"
                raise ValueError(msg)
            return self._extract_preprocessed(
                items=items,
                prepared_items=prepared_items,
                task=task,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
            )

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
        instruction: str | None,
        max_new_tokens: int,
        num_beams: int,
    ) -> ExtractOutput:
        from sie_server.core.prepared import PaddleOCRVLPayload, PreparedItem

        all_entities = []
        for i, prepared in enumerate(prepared_items):
            payload = prepared.payload if isinstance(prepared, PreparedItem) else getattr(prepared, "payload", prepared)
            if not isinstance(payload, PaddleOCRVLPayload):
                all_entities.append(
                    self._extract_single(
                        items[i],
                        task=task,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                    )
                )
                continue

            pixel_values = payload.pixel_values.to(device=self._device, dtype=self._model.dtype)
            input_ids = payload.input_ids.unsqueeze(0).to(self._device)
            attention_mask = payload.attention_mask.unsqueeze(0).to(self._device)
            image_grid_thw = payload.image_grid_thw.to(self._device)
            prompt_len = input_ids.shape[1]

            with torch.inference_mode():
                output_ids = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=num_beams,
                )

            generated_ids = output_ids[0, prompt_len:]
            generated_text = self._processor.decode(generated_ids, skip_special_tokens=True)
            all_entities.append(self._convert_output(generated_text, task))
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
        from PIL import Image as PILImage

        images = item.images
        if not images:
            raise ValueError(_ERR_NO_IMAGES)

        img_bytes = images[0]["data"]
        pil_img = PILImage.open(io.BytesIO(img_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        messages = self._build_messages(task=task, instruction=instruction)
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
        return self._convert_output(generated_text, task)

    def _build_messages(self, *, task: str, instruction: str | None) -> list[dict[str, Any]]:
        prompt_text = instruction or _TASK_PROMPTS[task]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

    @staticmethod
    def _convert_output(text: str, task: str) -> list[Entity]:
        """Wrap generated text in a single-entity list.

        Downstream detection/structured parsing for ``spotting``/``seal`` is
        deferred — those tasks emit bbox-annotated JSON that we return as raw
        text for now (labelled by task name). Consumers can post-process.
        """
        label = "markdown" if task in ("ocr", "table", "formula", "chart") else task
        return [Entity(text=text.strip(), label=label, score=1.0)]
