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
from sie_server.core.inference_output import ScoreOutput

if TYPE_CHECKING:
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)

_ERR_NO_INPUT = "Qwen3VLRerankerAdapter requires text or images input"

# Chat template markers used by the reranker to structure (query, document) pairs.
_DEFAULT_INSTRUCTION = "Retrieve relevant documents for the query."


def _build_reranker_conversation(
    *,
    query_text: str | None = None,
    query_image: Image.Image | None = None,
    doc_text: str | None = None,
    doc_image: Image.Image | None = None,
    instruction: str = _DEFAULT_INSTRUCTION,
) -> list[dict[str, Any]]:
    """Build chat conversation for reranking a (query, document) pair.

    The reranker model expects:
      system: <instruction>
      user: <query content>
      assistant: <document content>

    Both query and document can contain text, image, or both.
    """
    # Build query content
    query_content: list[dict[str, Any]] = []
    if query_image is not None:
        query_content.append({"type": "image", "image": query_image})
    if query_text:
        query_content.append({"type": "text", "text": query_text})
    if not query_content:
        query_content.append({"type": "text", "text": ""})

    # Build document content
    doc_content: list[dict[str, Any]] = []
    if doc_image is not None:
        doc_content.append({"type": "image", "image": doc_image})
    if doc_text:
        doc_content.append({"type": "text", "text": doc_text})
    if not doc_content:
        doc_content.append({"type": "text", "text": ""})

    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": query_content},
        {"role": "assistant", "content": doc_content},
    ]


class Qwen3VLRerankerAdapter(BaseAdapter):
    """Adapter for Qwen3-VL-Reranker multimodal cross-encoder reranking models.

    Qwen3-VL-Reranker-2B accepts (query, document) pairs where both query and
    document may contain text, images, or a mix. It outputs a relevance score
    based on the difference between "yes" and "no" logits at the final position.

    Key features:
    - Multimodal cross-attention reranking (text+image query × text+image doc)
    - Chat-template-based instruction-aware scoring
    - Apache 2.0 license, 2B parameters, fits L4 with headroom
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
        max_seq_length: int | None = None,
        default_instruction: str = _DEFAULT_INSTRUCTION,
    ) -> None:
        self._model_name_or_path = str(model_name_or_path)
        self._compute_precision = compute_precision
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._default_instruction = default_instruction

        self._model: Qwen3VLForConditionalGeneration | None = None
        self._processor: AutoProcessor | None = None
        self._device: str | None = None
        self._yes_token_id: int | None = None
        self._no_token_id: int | None = None

    def load(self, device: str) -> None:
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        self._device = device
        dtype = self._resolve_dtype()
        attn_impl = self._resolve_attn_implementation(device)

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
            )
        except (TypeError, OSError) as exc:
            logger.info(
                "AutoProcessor failed for %s (%s), loading processor from base model",
                self._model_name_or_path,
                exc,
            )
            # Load processor from base model (same vision architecture)
            base_model = "Qwen/Qwen3-VL-2B-Instruct"
            self._processor = AutoProcessor.from_pretrained(
                base_model,
                trust_remote_code=self._trust_remote_code,
                min_pixels=256 * 28 * 28,
                max_pixels=1280 * 28 * 28,
            )
            # Replace tokenizer with the reranker's own tokenizer (has
            # reranker-specific chat template and special tokens)
            self._processor.tokenizer = AutoTokenizer.from_pretrained(  # ty: ignore[unresolved-attribute]
                self._model_name_or_path,
                trust_remote_code=self._trust_remote_code,
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

        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_name_or_path,
            **load_kwargs,
        )
        self._model.eval()

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

        inst = instruction or self._default_instruction
        scores = []
        for doc in items:
            s = self._score_pair(query, doc, instruction=inst)
            scores.append(s)

        # Free GPU memory once after the full scoring pass
        if self._device and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

        return scores

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

        inst = instruction or self._default_instruction
        scores = []
        for query, doc in zip(queries, docs):
            s = self._score_pair(query, doc, instruction=inst)
            scores.append(s)

        # Free GPU memory once after the full scoring pass
        if self._device and self._device.startswith("cuda"):
            torch.cuda.empty_cache()

        return ScoreOutput(
            scores=np.array(scores, dtype=np.float32),
            batch_size=len(docs),
        )

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_pair(self, query: Item, doc: Item, *, instruction: str) -> float:
        """Score a single (query, document) pair.

        Uses the Qwen3-VL reranker pattern: feed (query, doc) through the model,
        extract logits at the last position for "yes" vs "no" tokens, and return
        sigmoid(logit_yes - logit_no) as the relevance score.
        """
        assert self._model is not None
        assert self._processor is not None

        # Build conversation
        query_image = self._load_first_image(query) if (query.images and len(query.images) > 0) else None
        doc_image = self._load_first_image(doc) if (doc.images and len(doc.images) > 0) else None

        # Reject if both query and doc are completely empty
        query_empty = not query.text and query_image is None
        doc_empty = not doc.text and doc_image is None
        if query_empty and doc_empty:
            raise ValueError(_ERR_NO_INPUT)

        conversation = _build_reranker_conversation(
            query_text=query.text,
            query_image=query_image,
            doc_text=doc.text,
            doc_image=doc_image,
            instruction=instruction,
        )

        # Apply chat template
        prompt = self._processor.apply_chat_template(  # ty: ignore[unresolved-attribute]
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Collect images
        images = []
        for msg in conversation:
            content = msg.get("content", [])
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image":
                        img = part.get("image")
                        if img is not None:
                            images.append(img)

        # Tokenize
        proc_kwargs: dict[str, Any] = {
            "text": [prompt],
            "return_tensors": "pt",
            "padding": True,
        }
        if self._max_seq_length is not None:
            proc_kwargs["truncation"] = True
            proc_kwargs["max_length"] = self._max_seq_length
        if images:
            proc_kwargs["images"] = images

        inputs = self._processor(**proc_kwargs)  # ty: ignore[call-non-callable]
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self._model(**inputs, return_dict=True)

        # Extract logits at last position
        logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Score = sigmoid(logit_yes - logit_no)
        yes_logit = logits[self._yes_token_id].float()
        no_logit = logits[self._no_token_id].float()
        score = torch.sigmoid(yes_logit - no_logit).item()

        # Free intermediate tensors (batch-level empty_cache is in score()/score_pairs())
        del outputs, inputs, logits

        return score

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_first_image(self, item: Any) -> Image.Image:
        from PIL import Image

        img_input = item.images[0]
        pil_img = Image.open(io.BytesIO(img_input["data"]))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return pil_img

    def get_preprocessor(self) -> Any | None:
        # Qwen3-VL processor requires text alongside images (for chat template
        # token insertion). Return None to use the direct adapter call path.
        return None
