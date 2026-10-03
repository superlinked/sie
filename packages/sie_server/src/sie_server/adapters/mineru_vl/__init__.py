from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision
from sie_server.core.inference_output import EncodeOutput, ExtractOutput
from sie_server.types.inputs import decode_image
from sie_server.types.responses import Entity

if TYPE_CHECKING:
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_ERR_NO_IMAGES = "MinerUVLAdapter requires image input for extraction"
_ERR_ENCODE_NOT_SUPPORTED = "MinerUVLAdapter does not support encode(). Use extract() instead."
_ERR_FP16_UNSUPPORTED = (
    "MinerU2.5-Pro does not support float16 on CUDA (config pins bfloat16); use bfloat16 or float32."
)

# Canonical task -> prompt mapping from upstream mineru_vl_utils.mineru_client.
# Bracketed task names (``[default]``, ``[layout]``) mirror the upstream
# convention — preserve them verbatim. Leading newline in the prompt strings
# is intentional (MinerU's prompts start with "\n"); do NOT strip it. Keep in
# sync with preprocessor/vision.py::_MINERU_VL_TASK_PROMPTS.
_TASK_PROMPTS: dict[str, str] = {
    "[default]": "\nText Recognition:",
    "table": "\nTable Recognition:",
    "equation": "\nFormula Recognition:",
    "image": "\nImage Analysis:",
    "chart": "\nImage Analysis:",
    "[layout]": "\nLayout Detection:",
}
_VALID_TASKS: tuple[str, ...] = tuple(_TASK_PROMPTS)
_DEFAULT_TASK = "[default]"

# HF-applicable subset of mineru_vl_utils' DEFAULT_SAMPLING_PARAMS.
# ``presence_penalty`` and ``frequency_penalty`` from MinerU's defaults are
# vLLM-only knobs — HF's ``generate`` silently drops them. Greedy decoding
# (``do_sample=False``) is MinerU's production setting; ``repetition_penalty``
# reproduces MinerU's transformers-backend recipe. ``no_repeat_ngram_size`` is
# applied via a fast logits processor (see ``_ngram_logits_processor``).
_NO_REPEAT_NGRAM_SIZE = 100
_MINERU_GENERATE_KWARGS: dict[str, Any] = {
    "do_sample": False,
    "repetition_penalty": 1.0,
}


class _IncrementalNoRepeatNGramLogitsProcessor:
    """Bit-identical, O(ngram_size)-per-step drop-in for HF's ``NoRepeatNGramLogitsProcessor``.

    HF's stock processor rebuilds the entire n-gram table and does a full
    GPU->CPU ``.tolist()`` of the growing sequence on *every* decode step —
    O(L^2) pure-Python work plus one host sync per token. On MinerU's dense
    pages that reach the 4096-token cap this single-threaded loop pegs one CPU
    core and starves the GPU (measured ~20% util, ``load``~=1.0). This version
    keeps the table incrementally: each step it adds only the single n-gram
    completed by the previous token and syncs only that token, so per-step cost
    is O(ngram_size) instead of O(sequence_length * ngram_size). It produces
    identical bans (verified in tests) — at ngram_size=100 the guard still
    fires only on a verbatim 100-token repeat (the runaway-loop case).

    Stateful, hence single-sequence / greedy only: beam search reorders
    hypotheses between steps and would invalidate the cached table, so the
    adapter only wires this in for ``num_beams == 1`` and keeps HF's stateless
    processor otherwise. Construct a fresh instance per ``generate`` call.
    """

    def __init__(self, ngram_size: int) -> None:
        self._n = ngram_size
        self._tokens: list[int] = []
        self._ngrams: dict[tuple[int, ...], list[int]] = {}
        self._seen_len = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        n = self._n
        cur_len = input_ids.shape[-1]
        if self._seen_len == 0:
            # First call: ingest the whole prompt prefix once (O(prefix)).
            self._tokens = input_ids[0].tolist()
            for end in range(n - 1, len(self._tokens)):
                self._ngrams.setdefault(tuple(self._tokens[end - n + 1 : end]), []).append(self._tokens[end])
        else:
            # Later calls: append only tokens added since the last step
            # (normally exactly one) and extend the table incrementally.
            for tok in input_ids[0, self._seen_len :].tolist():
                self._tokens.append(tok)
                if len(self._tokens) >= n:
                    self._ngrams.setdefault(tuple(self._tokens[-n:-1]), []).append(self._tokens[-1])
        self._seen_len = cur_len

        if cur_len + 1 < n:
            return scores
        banned = self._ngrams.get(tuple(self._tokens[cur_len - n + 1 : cur_len]))
        if not banned:
            return scores
        scores_processed = scores.clone()
        scores_processed[0, banned] = -float("inf")
        return scores_processed


def _ngram_logits_processor(num_beams: int) -> dict[str, Any]:
    """No-repeat-ngram generate kwargs, fast path for greedy decoding.

    ``num_beams == 1`` (MinerU's production setting) uses the incremental
    processor; beam search keeps HF's stateless ``no_repeat_ngram_size`` kwarg
    because the incremental table is invalidated by beam reordering.
    """
    if num_beams == 1:
        from transformers import LogitsProcessorList

        return {
            "logits_processor": LogitsProcessorList([_IncrementalNoRepeatNGramLogitsProcessor(_NO_REPEAT_NGRAM_SIZE)])
        }
    return {"no_repeat_ngram_size": _NO_REPEAT_NGRAM_SIZE}


class MinerUVLAdapter(BaseAdapter):
    """Adapter for opendatalab/MinerU2.5-Pro-2604-1.2B document OCR VLM.

    MinerU2.5-Pro-2604-1.2B is a 1.2B-param Qwen2-VL-based document parser
    (Apache-2.0). Architecture: 32-layer ViT (embed 1280, patch 14, spatial
    merge 2) over a 24-layer Qwen2 decoder (hidden 896, GQA 14/2, vocab
    151936, mRoPE). OmniDocBench v1.6 Overall 95.75 — best of the
    fast-tier open-weight document VLMs.

    Uses upstream ``Qwen2VLForConditionalGeneration`` — no
    ``trust_remote_code``, no custom modeling code. ``bfloat16`` on CUDA
    (per ``config.json``; fp16 is unsupported), ``float32`` on CPU/MPS for
    smoke testing only.

    Supports six task modes via the ``task`` runtime option:
    ``[default]`` (text recognition), ``table``, ``equation``, ``image``,
    ``chart``, ``[layout]``. Bracketed task names mirror upstream's
    convention. Each call returns a single ``Entity`` whose ``label`` is
    ``mineru_<task-without-brackets>`` (e.g. ``mineru_text``,
    ``mineru_table``, ``mineru_layout``).
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
        # Use upstream Qwen2VLForConditionalGeneration directly. MinerU ships
        # no custom modeling code, so no trust_remote_code and no
        # create_causal_mask shim are needed (compare PaddleOCR-VL).
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self._device = device
        dtype = self._resolve_dtype_for(device)

        shared_kwargs: dict[str, Any] = {}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        logger.info(
            "Loading MinerU2.5-Pro model %s on device=%s dtype=%s revision=%s attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._revision,
            self._attn_implementation,
        )

        self._processor = AutoProcessor.from_pretrained(
            self._model_name_or_path,
            use_fast=True,
            **shared_kwargs,
        )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self._model_name_or_path,
            dtype=dtype,
            attn_implementation=self._attn_implementation,
            **shared_kwargs,
        )
        self._model.to(device)  # ty: ignore[invalid-argument-type]
        self._model.eval()

        self._create_preprocessor()
        logger.info("MinerU2.5-Pro model loaded successfully")

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
        from sie_server.core.preprocessor.vision import MinerUVLPreprocessor

        self._preprocessor = MinerUVLPreprocessor(
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
        for i, item in enumerate(items):
            entities = self._extract_single(
                item,
                task=task,
                instruction=instruction,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                item_index=i,
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
        from sie_server.core.prepared import MinerUVLPayload, PreparedItem

        all_entities = []
        for i, prepared in enumerate(prepared_items):
            payload = prepared.payload if isinstance(prepared, PreparedItem) else getattr(prepared, "payload", prepared)
            if not isinstance(payload, MinerUVLPayload):
                all_entities.append(
                    self._extract_single(
                        items[i],
                        task=task,
                        instruction=instruction,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        item_index=i,
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
                    num_beams=num_beams,
                    use_cache=True,
                    **_MINERU_GENERATE_KWARGS,
                    **_ngram_logits_processor(num_beams),
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
        item_index: int | None = None,
    ) -> list[Entity]:
        images = item.images
        if not images:
            raise ValueError(_ERR_NO_IMAGES)

        # decode_image raises InvalidMediaError (-> 400 INVALID_INPUT) on
        # non-bytes or undecodable payloads.
        pil_img = decode_image(images[0], item_index=item_index, image_index=0)

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
                num_beams=num_beams,
                use_cache=True,
                **_MINERU_GENERATE_KWARGS,
                **_ngram_logits_processor(num_beams),
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
        """Wrap generated text in a single-entity list labelled ``mineru_<task>``.

        Bracketed task names ``[default]``/``[layout]`` map to bracket-free
        labels ``mineru_text``/``mineru_layout``; all other task names pass
        through unchanged (``mineru_table``, ``mineru_equation``,
        ``mineru_image``, ``mineru_chart``). Downstream JSON parsing for
        ``[layout]`` (bbox-annotated output) is deferred — consumers can
        post-process.
        """
        label_suffix_map = {"[default]": "text", "[layout]": "layout"}
        label_suffix = label_suffix_map.get(task, task)
        return [Entity(text=text.strip(), label=f"mineru_{label_suffix}", score=1.0)]
