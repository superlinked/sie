from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ComputePrecision
from sie_server.adapters._vision_patch_embed import rebind_vision_patch_embed
from sie_server.core.inference_output import ScoreOutput
from sie_server.types.inputs import InvalidInputError, InvalidMediaError, media_bytes

if TYPE_CHECKING:
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_ERR_NO_INPUT = "Qwen3VLRerankerAdapter requires nonblank text or one image in every query and document item"
_ERR_MULTIPLE_IMAGES = "Qwen3VLRerankerAdapter accepts at most one image per query or document item"
_ERR_UNSUPPORTED_MEDIA = "Qwen3VLRerankerAdapter accepts only text and image inputs"

# Chat template markers used by the reranker to structure (query, document) pairs.
_DEFAULT_INSTRUCTION = "Retrieve relevant documents for the query."
_SYSTEM_PROMPT = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)


def _build_reranker_conversation(
    *,
    query_text: str | None = None,
    query_image: Image.Image | None = None,
    doc_text: str | None = None,
    doc_image: Image.Image | None = None,
    instruction: str = _DEFAULT_INSTRUCTION,
    include_document_placeholder: bool = True,
) -> list[dict[str, Any]]:
    """Build chat conversation for reranking a (query, document) pair.

    The Qwen3-VL-Reranker reference format puts instruction, query, and
    document content in one user message. The model chat template emits vision
    placeholder tokens for images in user content, but not assistant content.
    """
    content: list[dict[str, Any]] = [
        {"type": "text", "text": "<Instruct>: " + instruction},
        {"type": "text", "text": "<Query>:"},
    ]
    if query_image is not None:
        content.append({"type": "image", "image": query_image})
    if query_text:
        content.append({"type": "text", "text": query_text})
    if not query_text and query_image is None:
        content.append({"type": "text", "text": "NULL"})

    content.append({"type": "text", "text": "\n<Document>:"})
    if doc_image is not None:
        content.append({"type": "image", "image": doc_image})
    if doc_text:
        content.append({"type": "text", "text": doc_text})
    if include_document_placeholder and not doc_text and doc_image is None:
        content.append({"type": "text", "text": "NULL"})

    return [
        {"role": "system", "content": [{"type": "text", "text": _SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


class Qwen3VLRerankerAdapter(BaseAdapter):
    """Adapter for Qwen3-VL-Reranker multimodal cross-encoder reranking models.

    Qwen3-VL-Reranker-2B accepts (query, document) pairs where both query and
    document may contain text, images, or a mix. It outputs a relevance score
    based on the difference between "yes" and "no" logits at the final position.

    Key features:
    - Multimodal cross-attention reranking (text+image query × text+image doc)
    - Chat-template-based instruction-aware scoring
    - Apache 2.0 license, 2B parameters
    - Designed for two-stage retrieval: embed then rerank

    Target models:
    - Qwen/Qwen3-VL-Reranker-2B
    - Qwen/Qwen3-VL-Reranker-8B (future)
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("text", "image"),
        outputs=("score",),
        unload_fields=("_model", "_processor", "_yes_token_id", "_no_token_id"),
        default_preprocessor="image",
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        compute_precision: ComputePrecision = "bfloat16",
        trust_remote_code: bool = False,
        revision: str | None = None,
        max_seq_length: int | None = None,
        default_instruction: str = _DEFAULT_INSTRUCTION,
        base_processor_model: str = "Qwen/Qwen3-VL-2B-Instruct",
        base_processor_revision: str | None = None,
    ) -> None:
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._revision = revision
        self._max_seq_length = max_seq_length
        self._default_instruction = default_instruction
        self._base_processor_model = base_processor_model
        self._base_processor_revision = base_processor_revision

        self._model: Qwen3VLForConditionalGeneration | None = None
        self._processor: AutoProcessor | None = None
        self._device: str | None = None
        self._yes_token_id: int | None = None
        self._attn_implementation: str | None = None
        self._no_token_id: int | None = None

    def load(self, device: str) -> None:
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        self._device = device
        dtype = self._resolve_dtype()
        attn_impl = self._resolve_attn_implementation(device)
        self._attn_implementation = attn_impl

        logger.info(
            "Loading Qwen3-VL-Reranker %s on device=%s dtype=%s attn=%s max_seq_length=%s",
            self._model_name_or_path,
            device,
            dtype,
            attn_impl,
            self._max_seq_length,
        )

        # The reranker model may not ship a valid processor config (template
        # file reference is None in some transformers versions). Try
        # AutoProcessor first; if that fails, load processor from the base
        # Qwen3-VL-2B-Instruct model and override its tokenizer.
        try:
            self._processor = AutoProcessor.from_pretrained(
                self._model_name_or_path,
                trust_remote_code=self._trust_remote_code,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
                revision=self._revision,
            )
        except (TypeError, OSError) as exc:
            logger.info(
                "AutoProcessor failed for %s (%s), loading processor from base model",
                self._model_name_or_path,
                exc,
            )
            if self._base_processor_revision is None:
                raise RuntimeError("Qwen3-VL processor fallback requires an immutable base_processor_revision") from exc
            self._processor = AutoProcessor.from_pretrained(
                self._base_processor_model,
                trust_remote_code=self._trust_remote_code,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
                revision=self._base_processor_revision,
            )
            # Replace tokenizer with the reranker's own tokenizer (has
            # reranker-specific chat template and special tokens)
            self._processor.tokenizer = AutoTokenizer.from_pretrained(  # ty: ignore[unresolved-attribute]
                self._model_name_or_path,
                trust_remote_code=self._trust_remote_code,
                revision=self._revision,
            )

        if self._max_seq_length is not None and hasattr(self._processor, "tokenizer"):
            self._processor.tokenizer.model_max_length = self._max_seq_length  # ty: ignore[unresolved-attribute]

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device,
            "trust_remote_code": self._trust_remote_code,
        }
        if attn_impl is not None:
            load_kwargs["attn_implementation"] = attn_impl
        if self._revision is not None:
            load_kwargs["revision"] = self._revision

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_name_or_path,
            **load_kwargs,
        )
        self._model.eval()

        # FIX[#1151]: rebind the Qwen3-VL vision Conv3d patch-embed to its matmul
        # equivalent (same weights). The non-overlapping per-patch Conv3d hits a
        # pathologically slow cuDNN path that dominates the vision-tower forward.
        rebind_vision_patch_embed(self._model, "qwen3_vl_reranker")

        # Pre-resolve the yes/no token IDs for scoring and validate they
        # are real tokens (convert_tokens_to_ids silently returns the UNK id
        # if the token is missing from the vocabulary).
        tokenizer = self._processor.tokenizer  # ty: ignore[unresolved-attribute]
        unk_id = getattr(tokenizer, "unk_token_id", None)

        self._yes_token_id = tokenizer.convert_tokens_to_ids("yes")
        if self._yes_token_id == unk_id or tokenizer.convert_ids_to_tokens(self._yes_token_id) != "yes":
            msg = (
                f"Tokenizer for {self._model_name_or_path} does not contain a 'yes' token "
                f"(resolved to id={self._yes_token_id}, unk_id={unk_id}). "
                "Scoring requires dedicated 'yes'/'no' tokens; consider using a "
                "tokenizer that includes them or adding them via add_tokens()."
            )
            raise ValueError(msg)

        self._no_token_id = tokenizer.convert_tokens_to_ids("no")
        if self._no_token_id == unk_id or tokenizer.convert_ids_to_tokens(self._no_token_id) != "no":
            msg = (
                f"Tokenizer for {self._model_name_or_path} does not contain a 'no' token "
                f"(resolved to id={self._no_token_id}, unk_id={unk_id}). "
                "Scoring requires dedicated 'yes'/'no' tokens; consider using a "
                "tokenizer that includes them or adding them via add_tokens()."
            )
            raise ValueError(msg)

    def _resolve_dtype(self) -> torch.dtype:
        if not self._device or not str(self._device).startswith("cuda"):
            return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)

    def _resolve_attn_implementation(self, device: str) -> str | None:
        if not device.startswith("cuda"):
            return None
        try:
            import flash_attn  # ty: ignore[unresolved-import]

            return "flash_attention_2"
        except (ImportError, RuntimeError) as exc:
            logger.info("flash_attn not available (%s), using sdpa attention", exc)
            return "sdpa"

    # ------------------------------------------------------------------
    # Score (single query, multiple documents)
    # ------------------------------------------------------------------

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        self._check_loaded()

        if not items:
            return []
        output = self.score_pairs(
            [query] * len(items),
            items,
            instruction=instruction,
            options=options,
        )
        return output.scores.tolist()

    # ------------------------------------------------------------------
    # Score pairs (parallel query-doc pairs)
    # ------------------------------------------------------------------

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        self._check_loaded()

        if len(queries) != len(docs):
            msg = f"score_pairs requires equal-length queries and docs, got {len(queries)} queries and {len(docs)} docs"
            raise ValueError(msg)

        if not queries:
            return ScoreOutput(
                scores=np.empty(0, dtype=np.float32),
                batch_size=0,
                input_token_counts=[],
                input_image_counts=[],
            )

        inst = instruction or self._default_instruction
        max_length = self._runtime_max_length((options or {}).get("max_seq_length"))
        scores, input_token_counts, input_image_counts = self._score_pair_batch(
            queries,
            docs,
            instruction=inst,
            max_length=max_length,
        )

        return ScoreOutput(
            scores=scores,
            batch_size=len(docs),
            input_token_counts=input_token_counts,
            input_image_counts=input_image_counts,
        )

    def count_pair_input_images(
        self,
        query: Item,
        docs: list[Item],
        *,
        instruction: str | None = None,
    ) -> list[int]:
        """Count the validated query/doc images consumed for each scored pair."""
        _ = instruction
        query_image_count = int(bool(query.images))
        return [query_image_count + int(bool(doc.images)) for doc in docs]

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_pair_batch(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str,
        max_length: int | None,
    ) -> tuple[np.ndarray, list[int], list[int]]:
        """Score a model-worker batch in one padded multimodal forward.

        The Qwen3-VL reranker scores the final attended token's "yes" and "no"
        logits. Explicit final-token indices make this independent of tokenizer
        padding direction.
        """
        assert self._model is not None
        assert self._processor is not None
        assert self._yes_token_id is not None
        assert self._no_token_id is not None

        prompts: list[str] = []
        sentinel_prompts: list[str] = []
        sentinels: list[str] = []
        document_texts: list[str] = []
        images: list[Any] = []
        input_image_counts: list[int] = []
        for pair_index, (query, doc) in enumerate(zip(queries, docs, strict=True)):
            self._validate_item(query)
            self._validate_item(doc)
            self._validate_control_tokens(instruction, query.text, doc.text)

            query_image = self._load_first_image(query) if query.images else None
            doc_image = self._load_first_image(doc) if doc.images else None
            conversation = _build_reranker_conversation(
                query_text=query.text,
                query_image=query_image,
                doc_text=doc.text,
                doc_image=doc_image,
                instruction=instruction,
            )
            sentinel = self._document_sentinel(pair_index, instruction, query.text, doc.text)
            sentinel_conversation = _build_reranker_conversation(
                query_text=query.text,
                query_image=query_image,
                doc_text=sentinel,
                doc_image=doc_image,
                instruction=instruction,
                include_document_placeholder=False,
            )
            with self._tokenizer_guard():
                prompts.append(
                    self._processor.apply_chat_template(  # ty: ignore[unresolved-attribute]
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                sentinel_prompts.append(
                    self._processor.apply_chat_template(  # ty: ignore[unresolved-attribute]
                        sentinel_conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            sentinels.append(sentinel)
            document_texts.append(doc.text or "")
            if query_image is not None:
                images.append(query_image)
            if doc_image is not None:
                images.append(doc_image)
            input_image_counts.append(int(query_image is not None) + int(doc_image is not None))

        proc_kwargs: dict[str, Any] = {
            "text": prompts,
            "return_tensors": None,
            "padding": False,
            "truncation": False,
        }
        if max_length is not None:
            proc_kwargs["return_offsets_mapping"] = True
        if images:
            proc_kwargs["images"] = images
        try:
            with self._tokenizer_guard():
                processor_inputs = self._processor(**proc_kwargs)  # ty: ignore[call-non-callable]
        except (IndexError, ValueError) as exc:
            raise InvalidInputError("Qwen3-VL processor rejected the supplied text/image layout") from exc

        inputs, input_token_counts = self._prepare_model_inputs(
            processor_inputs,
            prompts=prompts,
            sentinel_prompts=sentinel_prompts,
            sentinels=sentinels,
            document_texts=document_texts,
            max_length=max_length,
        )
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        if self._attn_implementation == "flash_attention_2":
            # Transformers' Qwen3-VL FlashAttention path rejects a text-only
            # singleton with a zero-dimension query. Keep FlashAttention for
            # image work and real batches; only this shape needs SDPA.
            attention = "sdpa" if len(prompts) == 1 and not images else "flash_attention_2"
            self._model.set_attn_implementation(attention)
        with torch.inference_mode():
            outputs = self._model(**inputs, return_dict=True)

        device_attention_mask = inputs.get("attention_mask")
        if device_attention_mask is None:
            device_attention_mask = torch.ones_like(inputs["input_ids"])
        positions = torch.arange(
            device_attention_mask.shape[1],
            device=device_attention_mask.device,
        ).expand_as(device_attention_mask)
        last_indices = positions.masked_fill(~device_attention_mask.bool(), -1).max(dim=1).values
        if bool((last_indices < 0).any()):
            raise InvalidInputError(_ERR_NO_INPUT)
        batch_indices = torch.arange(len(prompts), device=outputs.logits.device)
        last_logits = outputs.logits[batch_indices, last_indices, :]

        yes_logits = last_logits[:, self._yes_token_id].float()
        no_logits = last_logits[:, self._no_token_id].float()
        scores = torch.sigmoid(yes_logits - no_logits).cpu().numpy().astype(np.float32)

        del outputs, inputs, last_logits
        return scores, input_token_counts, input_image_counts

    def _prepare_model_inputs(
        self,
        processor_inputs: Any,
        *,
        prompts: list[str],
        sentinel_prompts: list[str],
        sentinels: list[str],
        document_texts: list[str],
        max_length: int | None,
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        """Truncate only document text, then pad without reprocessing images."""
        assert self._processor is not None
        input_ids = [[int(token_id) for token_id in row] for row in processor_inputs["input_ids"]]
        trimmed_ids = input_ids
        if max_length is not None:
            raw_offsets = processor_inputs.get("offset_mapping")
            if raw_offsets is None:
                raise RuntimeError("Qwen3-VL tokenizer did not return offset mappings")
            expanded_prompts = self._expand_image_placeholders(
                prompts,
                processor_inputs.get("image_grid_thw"),
            )
            expanded_sentinel_prompts = self._expand_image_placeholders(
                sentinel_prompts,
                processor_inputs.get("image_grid_thw"),
            )
            trimmed_ids = []
            for pair_index, (ids, offsets, prompt, sentinel_prompt, sentinel, document_text) in enumerate(
                zip(
                    input_ids,
                    raw_offsets,
                    expanded_prompts,
                    expanded_sentinel_prompts,
                    sentinels,
                    document_texts,
                    strict=True,
                )
            ):
                document_span = self._document_text_span(
                    prompt,
                    sentinel_prompt,
                    sentinel,
                    document_text,
                )
                trimmed_ids.append(
                    self._trim_document_tokens(
                        ids,
                        [(int(start), int(end)) for start, end in offsets],
                        document_span,
                        max_length,
                        has_image=self._processor.image_token in prompt,  # ty: ignore[unresolved-attribute]
                        pair_index=pair_index,
                    )
                )

        tokenizer = self._processor.tokenizer  # ty: ignore[unresolved-attribute]
        with self._tokenizer_guard():
            padded = tokenizer.pad(
                {"input_ids": trimmed_ids},
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
        inputs = dict(padded)
        for key, value in processor_inputs.items():
            if key in {"input_ids", "attention_mask", "offset_mapping"}:
                continue
            inputs[key] = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
        return inputs, [len(ids) for ids in trimmed_ids]

    def _expand_image_placeholders(
        self,
        prompts: list[str],
        image_grid_thw: Any,
    ) -> list[str]:
        """Mirror Qwen3VLProcessor's grid-driven image-token expansion."""
        assert self._processor is not None
        image_token = self._processor.image_token  # ty: ignore[unresolved-attribute]
        if image_grid_thw is None:
            if any(image_token in prompt for prompt in prompts):
                raise InvalidInputError("Qwen3-VL prompt contains image tokens without image data")
            return list(prompts)

        grids = image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else list(image_grid_thw)
        merge_length = int(self._processor.image_processor.merge_size) ** 2  # ty: ignore[unresolved-attribute]
        expanded: list[str] = []
        grid_index = 0
        for original_prompt in prompts:
            prompt = original_prompt
            while image_token in prompt:
                if grid_index >= len(grids):
                    raise InvalidInputError("Qwen3-VL prompt contains more image tokens than supplied images")
                image_tokens = int(np.prod(grids[grid_index])) // merge_length
                prompt = prompt.replace(image_token, "<|placeholder|>" * image_tokens, 1)
                grid_index += 1
            expanded.append(prompt.replace("<|placeholder|>", image_token))
        if grid_index != len(grids):
            raise InvalidInputError("Qwen3-VL request supplied images without matching prompt tokens")
        return expanded

    @staticmethod
    def _document_text_span(
        prompt: str,
        sentinel_prompt: str,
        sentinel: str,
        document_text: str,
    ) -> tuple[int, int]:
        if sentinel_prompt.count(sentinel) != 1:
            raise RuntimeError("Qwen3-VL document sentinel was not preserved by the chat template")
        prefix, suffix = sentinel_prompt.split(sentinel, 1)
        if prompt != f"{prefix}{document_text}{suffix}":
            raise RuntimeError("Qwen3-VL chat template changed outside the document text span")
        return len(prefix), len(prefix) + len(document_text)

    @staticmethod
    def _trim_document_tokens(
        input_ids: list[int],
        offsets: list[tuple[int, int]],
        document_span: tuple[int, int],
        max_length: int,
        *,
        has_image: bool,
        pair_index: int,
    ) -> list[int]:
        if len(input_ids) != len(offsets):
            raise RuntimeError("Qwen3-VL tokenizer returned misaligned token offsets")
        if len(input_ids) <= max_length:
            return input_ids

        document_start, document_end = document_span
        document_indices = [
            index for index, (start, end) in enumerate(offsets) if document_start <= start < end <= document_end
        ]
        protected_count = len(input_ids) - len(document_indices)
        if protected_count > max_length:
            image_note = " and complete image tokens" if has_image else ""
            raise InvalidInputError(
                f"options.max_seq_length={max_length} is too small for reranker pair {pair_index} "
                f"to preserve the instruction, query, template{image_note}"
            )

        keep_document_count = max_length - protected_count
        kept_document_indices = set(document_indices[:keep_document_count])
        document_index_set = set(document_indices)
        return [
            token_id
            for index, token_id in enumerate(input_ids)
            if index not in document_index_set or index in kept_document_indices
        ]

    def _validate_control_tokens(self, *values: str | None) -> None:
        """Reject user text that could forge processor/chat control tokens."""
        assert self._processor is not None
        tokenizer = self._processor.tokenizer  # ty: ignore[unresolved-attribute]
        reserved = {str(token) for token in getattr(tokenizer, "all_special_tokens", []) if token}
        for attribute in ("image_token", "video_token", "vision_start_token", "vision_end_token"):
            token = getattr(self._processor, attribute, None)
            if token:
                reserved.add(str(token))
        reserved.add("<|placeholder|>")
        if any(token in value for value in values if value for token in reserved):
            raise InvalidInputError("reranker text must not contain reserved model control tokens")

    @staticmethod
    def _document_sentinel(
        pair_index: int,
        *values: str | None,
    ) -> str:
        sentinel = f"SIE_DOCUMENT_TEXT_BOUNDARY_{pair_index}"
        source = "\n".join(value for value in values if value)
        while sentinel in source:
            sentinel += "_X"
        return sentinel

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _runtime_max_length(self, value: Any) -> int | None:
        """Validate a request override and clamp it to the load-time ceiling."""
        if value is None:
            return self._max_seq_length
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise InvalidInputError("options.max_seq_length must be a positive integer")
        if self._max_seq_length is None:
            return value
        return min(value, self._max_seq_length)

    @staticmethod
    def _validate_item(item: Item) -> None:
        if item.audio is not None or item.video is not None or item.document is not None:
            raise InvalidMediaError(_ERR_UNSUPPORTED_MEDIA)
        if item.images is not None and len(item.images) > 1:
            raise InvalidMediaError(_ERR_MULTIPLE_IMAGES)
        if not (item.text and item.text.strip()) and not item.images:
            raise InvalidInputError(_ERR_NO_INPUT)

    def _load_first_image(self, item: Any) -> Image.Image:
        from PIL import Image

        img_input = item.images[0]
        try:
            with Image.open(io.BytesIO(media_bytes(img_input, kind="image"))) as source:
                source.load()
                return source.convert("RGB") if source.mode != "RGB" else source.copy()
        except InvalidMediaError:
            raise
        except (OSError, ValueError) as exc:
            raise InvalidMediaError("image input must contain a valid decodable image") from exc

    def get_preprocessor(self) -> Any | None:
        # Qwen3-VL processor requires text alongside images (for chat template
        # token insertion). Return None to use the direct adapter call path.
        return None
