"""SentenceTransformer-based model adapters.

This module provides adapters for models compatible with the sentence-transformers library:
- SentenceTransformerDenseAdapter: For dense embedding models (SentenceTransformer class)
- SentenceTransformerSparseAdapter: For sparse embedding models (SparseEncoder class)

For models that output both dense and sparse (like BGE-M3), use BGEM3Adapter.
"""

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, SparseEncoder

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import EncodeOutput, SparseVector
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item

# Compute precision type (for interface compatibility)
ComputePrecision = Literal["float16", "bfloat16", "float32"]

# Error messages
_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "SentenceTransformer adapters require text input"


class SentenceTransformerDenseAdapter(ModelAdapter):
    """Adapter for dense sentence-transformers models.

    Uses the SentenceTransformer class for models like BGE, E5, GTE, all-MiniLM, etc.
    These models output dense (fixed-dimension) vector embeddings.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = True,
        normalize: bool = True,
        max_seq_length: int | None = None,
        compute_precision: ComputePrecision = "float16",
        config_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,  # Accept extra args from loader (e.g., pooling)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code in model files.
            normalize: Whether to L2-normalize embeddings.
            max_seq_length: Override default max sequence length.
            compute_precision: Compute precision (ignored, sentence-transformers handles internally).
            config_kwargs: Additional kwargs passed to model config (e.g., for models
                with custom code that need specific settings like
                {"use_memory_efficient_attention": False, "unpad_inputs": False}).
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._config_kwargs = config_kwargs

        self._model: SentenceTransformer | None = None
        self._device: str | None = None
        self._dense_dim: int | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            inputs=["text"],
            outputs=["dense"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return model dimensions."""
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        return ModelDims(dense=self._dense_dim)

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        self._device = device
        self._model = SentenceTransformer(
            self._model_name_or_path,
            device=device,
            trust_remote_code=self._trust_remote_code,
            config_kwargs=self._config_kwargs,
        )
        _ = self._model.encode(["warmup"], convert_to_numpy=True, show_progress_bar=False)

        if self._max_seq_length is not None:
            self._model.max_seq_length = self._max_seq_length

        self._dense_dim = self._model.get_sentence_embedding_dimension()

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device  # Save before clearing

        if self._model is not None:
            del self._model
            self._model = None

        self._device = None
        self._dense_dim = None

        # Release GPU memory per the memory management contract in base.py
        import gc

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: Any = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        """Run inference returning standardized batched output.

        Args:
            items: List of items to encode.
            output_types: Should contain "dense".
            instruction: Optional instruction prefix for instruction-tuned models.
            is_query: Whether items are queries. If True and model has a "query" prompt,
                uses it. If False and model has a "document" prompt, uses it.
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with dense embeddings.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If output_types contains unsupported types.
        """
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        unsupported = set(output_types) - {"dense"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. This model only supports 'dense'."
            raise ValueError(msg)

        # Resolve runtime options (query_template / doc_template)
        opts = options or {}
        query_template = opts.get("query_template")
        doc_template = opts.get("doc_template")

        texts = [self._extract_text(item) for item in items]

        # Apply query/doc template from runtime options if provided
        template = query_template if is_query else doc_template
        if template:
            texts = [template.format(text=text, instruction=instruction or "") for text in texts]
            prompt_name = None
        else:
            # Determine prompt_name based on is_query and model's available prompts
            prompt_name = None
            if self._model.prompts:
                if is_query and "query" in self._model.prompts:
                    prompt_name = "query"
                elif not is_query and "document" in self._model.prompts:
                    prompt_name = "document"

            # If instruction is provided explicitly, prepend it (fallback for models without prompts)
            if instruction is not None and prompt_name is None:
                texts = [f"{instruction} {text}" for text in texts]

        with torch.inference_mode():
            embeddings: np.ndarray = self._model.encode(
                texts,
                prompt_name=prompt_name,
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

        return EncodeOutput(
            dense=embeddings,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=self._dense_dim,
        )

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(_ERR_REQUIRES_TEXT)
        return item.text

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)


class SentenceTransformerSparseAdapter(ModelAdapter):
    """Adapter for sparse sentence-transformers models.

    Uses the SparseEncoder class for models like SPLADE that output sparse
    (variable-dimension, mostly-zero) vector embeddings.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = True,
        max_seq_length: int | None = None,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,  # Accept extra args from loader (e.g., pooling)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code in model files.
            max_seq_length: Override default max sequence length.
            compute_precision: Compute precision (ignored, sentence-transformers handles internally).
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length

        self._model: SparseEncoder | None = None
        self._device: str | None = None
        self._sparse_dim: int | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            inputs=["text"],
            outputs=["sparse"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return model dimensions."""
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        return ModelDims(sparse=self._sparse_dim)

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        self._device = device
        self._model = SparseEncoder(
            self._model_name_or_path,
            device=device,
            trust_remote_code=self._trust_remote_code,
        )

        if self._max_seq_length is not None:
            self._model.max_seq_length = self._max_seq_length

        # Sparse dim is vocabulary size
        self._sparse_dim = self._model.get_sentence_embedding_dimension()

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device  # Save before clearing

        if self._model is not None:
            del self._model
            self._model = None

        self._device = None
        self._sparse_dim = None

        # Release GPU memory per the memory management contract in base.py
        import gc

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    def encode(
        self,
        items: list[Item],
        output_types: list[str],
        *,
        instruction: str | None = None,
        is_query: bool = False,
        prepared_items: Any = None,
        options: dict[str, Any] | None = None,
    ) -> EncodeOutput:
        """Run inference returning standardized batched output.

        Args:
            items: List of items to encode.
            output_types: Should contain "sparse".
            instruction: Optional instruction/prompt for the encoder.
            is_query: Whether items are queries (uses encode_query vs encode_document).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with sparse embeddings.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If output_types contains unsupported types.
        """
        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        unsupported = set(output_types) - {"sparse"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. This model only supports 'sparse'."
            raise ValueError(msg)

        texts = [self._extract_text(item) for item in items]

        with torch.inference_mode():
            # SparseEncoder has separate methods for query vs document
            # Use sparse COO tensor output for efficiency
            if is_query:
                embeddings = self._model.encode_query(
                    texts,
                    prompt=instruction,
                    convert_to_tensor=True,
                    convert_to_sparse_tensor=True,
                )
            else:
                embeddings = self._model.encode_document(
                    texts,
                    prompt=instruction,
                    convert_to_tensor=True,
                    convert_to_sparse_tensor=True,
                )

        # Convert sparse COO tensor to our format (indices + values per item)
        # embeddings is a sparse COO tensor with shape [batch, vocab_size]
        # indices[0] = row indices, indices[1] = column indices (token IDs)
        embeddings = cast("torch.Tensor", embeddings)
        sparse_indices = embeddings._indices()
        sparse_values = embeddings._values()

        sparse_list = []
        for i in range(len(items)):
            # Get entries for this item (where row index == i)
            item_mask = sparse_indices[0] == i
            token_ids = sparse_indices[1][item_mask].cpu().numpy().astype(np.int32)
            weights = sparse_values[item_mask].cpu().numpy().astype(np.float32)
            sparse_list.append(SparseVector(indices=token_ids, values=weights))

        return EncodeOutput(
            sparse=sparse_list,
            batch_size=len(items),
            is_query=is_query,
        )

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(_ERR_REQUIRES_TEXT)
        return item.text

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
