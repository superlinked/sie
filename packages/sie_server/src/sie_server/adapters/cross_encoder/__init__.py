"""Cross-encoder adapter for reranker models.

This module provides adapters for cross-encoder models used for reranking:
- CrossEncoderAdapter: For sentence-transformers CrossEncoder models (e.g., Jina Rerankers, BGE Rerankers)

Cross-encoders score (query, document) pairs directly, producing relevance scores.
Unlike bi-encoders which encode queries and documents separately, cross-encoders
process them together and are more accurate but less efficient at scale.

Typical usage in a retrieval pipeline:
1. Initial retrieval with bi-encoder (fast, approximate)
2. Reranking top-k results with cross-encoder (accurate, slow)

Performance optimizations:
- Uses SDPA (Scaled Dot-Product Attention) by default for efficient attention
- Supports FP16/BF16 inference for reduced memory and faster computation
"""

from pathlib import Path
from typing import Any, Literal

import torch
from sentence_transformers import CrossEncoder

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import ScoreOutput
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item

# Compute precision type (for interface compatibility)
ComputePrecision = Literal["float16", "bfloat16", "float32"]

# Attention implementation options
AttnImplementation = Literal["sdpa", "eager"]

# Error messages
_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "CrossEncoder adapter requires text input"


class CrossEncoderAdapter(ModelAdapter):
    """Adapter for cross-encoder reranker models.

    Uses the sentence-transformers CrossEncoder class for models like:
    - Jina Rerankers (jinaai/jina-reranker-v2-base-multilingual)
    - BGE Rerankers (BAAI/bge-reranker-v2-m3)
    - MS MARCO cross-encoders

    These models take (query, document) pairs and output relevance scores.
    Higher scores indicate more relevant documents.

    Performance optimizations:
    - SDPA attention (default): Uses PyTorch's scaled_dot_product_attention
      which can dispatch to flash attention when available
    - FP16 inference: Reduces memory and improves throughput on GPU
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = False,
        max_length: int | None = None,
        compute_precision: ComputePrecision = "float16",
        attn_implementation: AttnImplementation = "sdpa",
        **kwargs: Any,  # Accept extra args from loader (e.g., pooling)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code in model files.
            max_length: Override default max sequence length for query+document.
            compute_precision: Compute precision for inference (float16, bfloat16, float32).
            attn_implementation: Attention implementation ("sdpa" for optimized, "eager" for baseline).
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_length = max_length
        self._compute_precision = compute_precision
        self._attn_implementation = attn_implementation

        self._model: CrossEncoder | None = None
        self._device: str | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities.

        Cross-encoders support scoring but not encoding.
        """
        return ModelCapabilities(
            inputs=["text"],
            outputs=["score"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return model dimensions.

        Cross-encoders don't produce embeddings, so all dims are None.
        """
        return ModelDims()

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu", "mps").
        """
        self._device = device

        # Build model_kwargs for performance optimizations
        model_kwargs: dict[str, Any] = {}

        # Use SDPA for efficient attention on GPU
        if device.startswith("cuda") or device == "mps":
            model_kwargs["attn_implementation"] = self._attn_implementation

            # Use FP16/BF16 for faster inference
            if self._compute_precision == "float16":
                model_kwargs["dtype"] = torch.float16
            elif self._compute_precision == "bfloat16":
                model_kwargs["dtype"] = torch.bfloat16
            # float32 is the default, no need to specify

        self._model = CrossEncoder(
            self._model_name_or_path,
            device=device,
            trust_remote_code=self._trust_remote_code,
            model_kwargs=model_kwargs or None,
        )

        if self._max_length is not None:
            self._model.max_length = self._max_length

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device  # Save before clearing

        if self._model is not None:
            del self._model
            self._model = None

        self._device = None

        # Release GPU memory per the memory management contract in base.py
        import gc

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score items against a query using the cross-encoder.

        Args:
            query: Query item (must have text).
            items: Items to score against the query.
            instruction: Optional instruction to prepend to query (for instruction-tuned models).

        Returns:
            List of scores, one per item. Higher scores indicate more relevant documents.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If query or items lack text.
        """
        _ = options  # Delegated to self._model.predict() which handles tokenization internally
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        query_text = self._extract_text(query)
        if instruction is not None:
            query_text = f"{instruction} {query_text}"

        # Build (query, document) pairs
        pairs = [(query_text, self._extract_text(item)) for item in items]

        # Score pairs
        with torch.inference_mode():
            scores = self._model.predict(pairs)

        # Convert to Python floats
        return [float(s) for s in scores]

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

        Returns:
            ScoreOutput containing scores for each (query, doc) pair.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If any item lacks text.
        """
        _ = options  # Delegated to self._model.predict() which handles tokenization internally
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        # Build (query, doc) pairs
        pairs = []
        for query, doc in zip(queries, docs, strict=True):
            query_text = self._extract_text(query)
            if instruction is not None:
                query_text = f"{instruction} {query_text}"
            doc_text = self._extract_text(doc)
            pairs.append((query_text, doc_text))

        # Score pairs
        with torch.inference_mode():
            scores = self._model.predict(pairs)

        # Convert to float32 numpy array and wrap in ScoreOutput
        import numpy as np

        return ScoreOutput(scores=np.asarray(scores, dtype=np.float32))

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(_ERR_REQUIRES_TEXT)
        return item.text

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
