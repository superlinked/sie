from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._utils import extract_text
from sie_server.core.inference_output import ScoreOutput
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_ERR_CUDA_REQUIRED = "JinaFlashCrossEncoder requires CUDA for Flash Attention."


class JinaFlashCrossEncoderAdapter(FlashBaseAdapter):
    """Cross-encoder adapter for Jina Rerankers with built-in flash attention.

    Uses the model's native flash attention implementation which already
    handles variable-length sequences with unpadding internally.
    """

    fallback_adapter_path: ClassVar[str | None] = "cross_encoder:CrossEncoderAdapter"
    fallback_kwargs_overrides: ClassVar[dict[str, Any]] = {"attn_implementation": "sdpa"}

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("score",),
        unload_fields=("_model", "_tokenizer", "_dtype"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = True,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code (required for Jina).
            max_seq_length: Maximum sequence length for query+document.
            compute_precision: Compute precision (bfloat16 recommended for Jina).
            **kwargs: Additional arguments (ignored).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision

        # Loaded state
        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None
        # Resolved at load() to min(configured, tokenizer.model_max_length,
        # model.config.max_position_embeddings). Used as a hard ceiling at
        # tokenization time so runtime overrides cannot escape the clamp.
        # Initialised to 0 so a pre-load read would zero-out tokenization
        # rather than silently use the unclamped configured value; in
        # practice every entry point goes through ``_check_loaded()`` first.
        self._tokenizer_max_length: int = 0

    def load(self, device: str) -> None:
        """Load model weights onto the specified device."""
        if not device.startswith("cuda"):
            raise RuntimeError(_ERR_CUDA_REQUIRED)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = device
        self._dtype = self._resolve_dtype()

        logger.info(
            "Loading %s with built-in Flash Attention (dtype=%s)",
            self._model_name_or_path,
            self._dtype,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # Load model - let it use its built-in flash attention (use_flash_attn in config)
        # Don't pass attn_implementation - the model's custom code handles attention
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name_or_path,
            torch_dtype=self._dtype,
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        config = self._model.config
        logger.info(
            "Loaded: hidden=%d, heads=%d, layers=%d, use_flash_attn=%s",
            config.hidden_size,
            config.num_attention_heads,
            config.num_hidden_layers,
            getattr(config, "use_flash_attn", "N/A"),
        )

        # Clamp max_seq_length to whatever the tokenizer/model actually support.
        # jina-reranker-v2-base-multilingual has model_max_length=1024 and
        # max_position_embeddings=1024; a stale 8192 here would silently let
        # over-long inputs through and crash the CUDA worker.
        self._max_seq_length = self._resolve_tokenizer_ceiling(
            self._tokenizer,
            self._model,
            self._max_seq_length,
        )
        self._tokenizer_max_length = self._max_seq_length

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score items against a query using flash attention.

        Args:
            query: Query item (must have text).
            items: Items to score against the query.
            instruction: Optional instruction to prepend to query.

        Returns:
            List of relevance scores (higher = more relevant).
        """
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        query_text = extract_text(query, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="JinaFlashCrossEncoder"))
        if instruction:
            query_text = f"{instruction} {query_text}"

        # Tokenize all pairs
        pairs = [
            (query_text, extract_text(item, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="JinaFlashCrossEncoder")))
            for item in items
        ]

        # Batch tokenize with padding (model handles unpadding internally).
        # No runtime override on this path — use the ceiling resolved at load().
        encodings = self._tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            max_length=self._tokenizer_max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        encodings = {k: v.to(self._device) for k, v in encodings.items()}

        with torch.inference_mode():
            # Forward pass - model's flash attention handles unpadding internally
            outputs = self._model(**encodings)
            logits = outputs.logits

            # Apply sigmoid for single-label classification
            if self._model.config.num_labels == 1:
                scores_tensor = torch.sigmoid(logits.squeeze(-1)).float()
            else:
                scores_tensor = logits.squeeze(-1).float()

            scores = scores_tensor.cpu().tolist()

        return scores

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        """Score (query, doc) pairs in a batch.

        Batched version of score() for cross-request batching.

        Args:
            queries: Query items (parallel to docs).
            docs: Document items to score.
            instruction: Optional instruction to prepend to queries.
            options: Runtime options (config defaults -> profile -> request overrides).

        Returns:
            ScoreOutput containing scores for each (query, doc) pair.
        """
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        opts = options or {}
        # Hard-clamp to the tokenizer/model ceiling resolved at load time.
        # Runtime overrides must not push past the model's positional capacity.
        # Malformed overrides (None, strings, negatives) fall back to the ceiling.
        max_length = self._coerce_runtime_max_length(opts.get("max_seq_length"), self._tokenizer_max_length)

        # Build (query, doc) pairs
        pairs = []
        for query, doc in zip(queries, docs, strict=True):
            query_text = extract_text(query, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="JinaFlashCrossEncoder"))
            if instruction:
                query_text = f"{instruction} {query_text}"
            doc_text = extract_text(doc, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="JinaFlashCrossEncoder"))
            pairs.append((query_text, doc_text))

        # Batch tokenize with padding
        encodings = self._tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        encodings = {k: v.to(self._device) for k, v in encodings.items()}

        with torch.inference_mode():
            outputs = self._model(**encodings)
            logits = outputs.logits

            if self._model.config.num_labels == 1:
                scores_tensor = torch.sigmoid(logits.squeeze(-1)).float()
            else:
                scores_tensor = logits.squeeze(-1).float()

            # Convert to float32 numpy array and wrap in ScoreOutput
            import numpy as np

            scores_array = scores_tensor.cpu().numpy().astype(np.float32)

        return ScoreOutput(scores=scores_array)
