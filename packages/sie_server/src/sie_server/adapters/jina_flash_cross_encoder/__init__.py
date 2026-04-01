"""Jina Flash Attention Cross-Encoder adapter for reranking.

This adapter uses Jina's built-in flash attention with varlen batching.
The Jina reranker models have their own optimized flash attention implementation
that already handles unpadding and cu_seqlens internally.

Supports:
- jinaai/jina-reranker-v2-base-multilingual

Key optimizations (built into model):
- Flash Attention 2 with variable-length sequences (no padding waste)
- Automatic unpadding/padding via model's custom code
- BF16 inference
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any, Literal

import torch

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "JinaFlashCrossEncoder requires text input"
_ERR_CUDA_REQUIRED = "JinaFlashCrossEncoder requires CUDA for Flash Attention."


class JinaFlashCrossEncoderAdapter(ModelAdapter):
    """Cross-encoder adapter for Jina Rerankers with built-in flash attention.

    Uses the model's native flash attention implementation which already
    handles variable-length sequences with unpadding internally.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = True,
        max_length: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code (required for Jina).
            max_length: Maximum sequence length for query+document.
            compute_precision: Compute precision (bfloat16 recommended for Jina).
            **kwargs: Additional arguments (ignored).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_length = max_length
        self._compute_precision = compute_precision

        # Loaded state
        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> ModelAdapter:
        """Factory method that returns the appropriate adapter for the device.

        For non-CUDA devices or when flash-attn is unavailable, returns CrossEncoderAdapter.

        Args:
            device: Device string (e.g., "cuda:0", "mps", "cpu").
            **kwargs: Adapter initialization parameters.

        Returns:
            JinaFlashCrossEncoderAdapter for CUDA with flash-attn, CrossEncoderAdapter otherwise.
        """
        from sie_server.adapters.cross_encoder import CrossEncoderAdapter

        return cls._create_flash_or_fallback(
            device,
            fallback_class=CrossEncoderAdapter,
            fallback_kwargs={**kwargs, "attn_implementation": "sdpa"},
            **kwargs,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            inputs=["text"],
            outputs=["score"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return model dimensions (none for cross-encoders)."""
        return ModelDims()

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

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)

    def unload(self) -> None:
        """Unload model and free GPU memory."""
        device = self._device

        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._device = None

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()

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
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        query_text = self._extract_text(query)
        if instruction:
            query_text = f"{instruction} {query_text}"

        # Tokenize all pairs
        pairs = [(query_text, self._extract_text(item)) for item in items]

        # Batch tokenize with padding (model handles unpadding internally)
        encodings = self._tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            max_length=self._max_length,
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
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        opts = options or {}
        max_length = opts.get("max_length", self._max_length)

        # Build (query, doc) pairs
        pairs = []
        for query, doc in zip(queries, docs, strict=True):
            query_text = self._extract_text(query)
            if instruction:
                query_text = f"{instruction} {query_text}"
            doc_text = self._extract_text(doc)
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

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(_ERR_REQUIRES_TEXT)
        return item.text

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
