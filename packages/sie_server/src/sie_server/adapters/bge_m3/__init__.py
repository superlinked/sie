"""BGE-M3 model adapter.

BGE-M3 is a multi-functional embedding model that supports:
- Dense embeddings (1024 dims)
- Sparse embeddings (lexical, SPLADE-like)
- Multi-vector embeddings (ColBERT-like)

Note: BGE-M3 uses XLMRoberta architecture which does NOT support Flash Attention 2.
This adapter always uses SDPA (Scaled Dot-Product Attention).

See: https://huggingface.co/BAAI/bge-m3
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from torch.nn import functional

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.core.inference_output import EncodeOutput, SparseVector
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast, XLMRobertaModel

logger = logging.getLogger(__name__)

# Compute precision type
ComputePrecision = Literal["float16", "bfloat16", "float32"]

# Error messages
_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "BGEM3Adapter requires text input"


class BGEM3Adapter(ModelAdapter):
    """Adapter for BAAI/bge-m3 model.

    This adapter uses direct PyTorch inference with Flash Attention 2
    for optimal performance (dense, sparse, and multi-vector outputs).
    """

    # BGE-M3 specific dimensions
    DENSE_DIM = 1024
    SPARSE_DIM = 250002  # Vocabulary size
    MULTIVECTOR_DIM = 1024  # Per-token dimension

    def __init__(
        self,
        model_name_or_path: str | Path = "BAAI/bge-m3",
        *,
        normalize: bool = True,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        **kwargs: Any,  # Accept extra args from loader (e.g., pooling)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize dense embeddings.
            max_seq_length: Maximum sequence length (default 8192).
            compute_precision: Compute precision (bfloat16 recommended).
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision

        self._model: XLMRobertaModel | None = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._colbert_linear: nn.Linear | None = None
        self._sparse_linear: nn.Linear | None = None
        self._device: str | None = None

    @property
    def capabilities(self) -> ModelCapabilities:
        """Return model capabilities."""
        return ModelCapabilities(
            inputs=["text"],
            outputs=["dense", "sparse", "multivector"],
        )

    @property
    def dims(self) -> ModelDims:
        """Return model dimensions."""
        return ModelDims(
            dense=self.DENSE_DIM,
            sparse=self.SPARSE_DIM,
            multivector=self.MULTIVECTOR_DIM,
        )

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        from transformers import AutoModel, AutoTokenizer

        self._device = device

        # Determine dtype and attention implementation
        dtype, attn_impl = self._resolve_dtype_and_attn(device)

        logger.info(
            "Loading %s on device=%s with dtype=%s, attn=%s",
            self._model_name_or_path,
            device,
            dtype,
            attn_impl,
        )

        # Load tokenizer (fast Rust tokenizer)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        # Load base model with Flash Attention 2 if available
        self._model = AutoModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
        self._model.to(device)
        self._model.eval()

        # Load colbert and sparse linear layers
        self._load_linear_layers(self._model_name_or_path, dtype, device)

    def _resolve_dtype_and_attn(self, device: str) -> tuple[torch.dtype, str]:
        """Resolve dtype and attention implementation based on device and config.

        Note: BGE-M3 uses XLMRoberta architecture which does NOT support Flash Attention 2.
        This method always returns "sdpa" for attention implementation.

        Returns:
            Tuple of (torch.dtype, attention_implementation string).
        """
        # CPU should use FP32
        if not device.startswith("cuda"):
            return torch.float32, "sdpa"

        # Map precision to dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self._compute_precision, torch.bfloat16)

        # XLMRoberta (BGE-M3) does not support Flash Attention 2, always use SDPA
        return dtype, "sdpa"

    def _load_linear_layers(self, model_path: str, dtype: torch.dtype, device: str) -> None:
        """Load the colbert and sparse linear layers from checkpoint.

        BGE-M3 has additional linear layers for multi-vector and sparse outputs.
        """
        hidden_size = self._model.config.hidden_size  # type: ignore[union-attr]

        # Resolve the actual directory: could be a local path or HF model ID
        base_path = Path(model_path)
        if not base_path.is_dir():
            base_path = Path(snapshot_download(model_path))

        # ColBERT linear: hidden_size -> hidden_size (1024 -> 1024)
        colbert_path = base_path / "colbert_linear.pt"
        if colbert_path.exists():
            self._colbert_linear = nn.Linear(hidden_size, hidden_size)
            state_dict = torch.load(colbert_path, map_location=device, weights_only=True)
            self._colbert_linear.load_state_dict(state_dict)
            self._colbert_linear.to(device=device, dtype=dtype)
            self._colbert_linear.eval()
        else:
            logger.warning("colbert_linear.pt not found at %s", base_path)

        # Sparse linear: hidden_size -> 1 (for token weights)
        sparse_path = base_path / "sparse_linear.pt"
        if sparse_path.exists():
            self._sparse_linear = nn.Linear(hidden_size, 1)
            state_dict = torch.load(sparse_path, map_location=device, weights_only=True)
            self._sparse_linear.load_state_dict(state_dict)
            self._sparse_linear.to(device=device, dtype=dtype)
            self._sparse_linear.eval()
        else:
            logger.warning("sparse_linear.pt not found at %s", base_path)

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device  # Save before clearing

        if self._model is not None:
            del self._model
            self._model = None

        if self._colbert_linear is not None:
            del self._colbert_linear
            self._colbert_linear = None

        if self._sparse_linear is not None:
            del self._sparse_linear
            self._sparse_linear = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._device = None

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
            output_types: Which outputs to return ("dense", "sparse", "multivector").
            instruction: Optional instruction (not commonly used with BGE-M3).
            is_query: Whether items are queries (affects instruction handling).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with requested output types.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        self._validate_output_types(output_types)

        # Resolve runtime options (config defaults -> profile -> request overrides)
        opts = options or {}
        normalize = opts.get("normalize", self._normalize)

        texts = self._extract_texts(items, instruction)

        # Tokenize
        inputs = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self._device)

        # Forward pass
        with torch.inference_mode():
            outputs = self._model(**inputs, return_dict=True)
            last_hidden_state = outputs.last_hidden_state

            results = self._compute_embeddings(
                last_hidden_state,
                inputs["input_ids"],
                inputs["attention_mask"],
                output_types,
                normalize=normalize,
            )

        return self._to_encode_output(results, output_types, len(items), is_query)

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"dense", "sparse", "multivector"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}"
            raise ValueError(msg)

    def _extract_texts(self, items: list[Item], instruction: str | None) -> list[str]:
        """Extract texts from items, optionally prepending instruction."""
        texts = []
        for item in items:
            if item.text is None:
                raise ValueError(_ERR_REQUIRES_TEXT)
            text = item.text
            if instruction is not None:
                text = f"{instruction} {text}"
            texts.append(text)
        return texts

    def _compute_embeddings(
        self,
        last_hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_types: list[str],
        *,
        normalize: bool | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | list[dict[int, float]]]:
        """Compute requested embeddings from the model output."""
        normalize = normalize if normalize is not None else self._normalize
        results: dict[str, torch.Tensor | list[torch.Tensor] | list[dict[int, float]]] = {}

        if "dense" in output_types:
            # CLS pooling
            dense_vecs = last_hidden_state[:, 0]
            if normalize:
                dense_vecs = functional.normalize(dense_vecs, p=2, dim=-1)
            results["dense"] = dense_vecs

        if "sparse" in output_types and self._sparse_linear is not None:
            # Token weights via sparse linear
            token_weights = torch.relu(self._sparse_linear(last_hidden_state)).squeeze(-1)
            results["sparse"] = self._compute_sparse_weights(token_weights, input_ids)

        if "multivector" in output_types and self._colbert_linear is not None:
            # ColBERT vectors (skip CLS token)
            colbert_vecs = self._colbert_linear(last_hidden_state[:, 1:])
            # Mask padding tokens
            mask = attention_mask[:, 1:].unsqueeze(-1).float()
            colbert_vecs = colbert_vecs * mask
            if normalize:
                colbert_vecs = functional.normalize(colbert_vecs, p=2, dim=-1)
            results["multivector"] = colbert_vecs

        return results

    def _compute_sparse_weights(
        self,
        token_weights: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> list[dict[int, float]]:
        """Compute sparse lexical weights per item.

        Returns list of {token_id: weight} dicts.
        """
        # Get special token IDs to exclude
        if self._tokenizer is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        special_tokens = {
            self._tokenizer.cls_token_id,
            self._tokenizer.eos_token_id,
            self._tokenizer.pad_token_id,
            self._tokenizer.unk_token_id,
        }
        special_tokens.discard(None)

        batch_size = token_weights.shape[0]
        results = []

        for i in range(batch_size):
            weights = token_weights[i].float().cpu().numpy()
            ids = input_ids[i].cpu().numpy()

            # Build sparse dict: max weight per unique token
            sparse_dict: dict[int, float] = {}
            for tid, weight in zip(ids, weights, strict=True):
                if tid in special_tokens or weight <= 0:
                    continue
                tid_int = int(tid)
                if tid_int not in sparse_dict or weight > sparse_dict[tid_int]:
                    sparse_dict[tid_int] = float(weight)

            results.append(sparse_dict)

        return results

    def _to_encode_output(
        self,
        embeddings: dict[str, Any],
        output_types: list[str],
        batch_size: int,
        is_query: bool,
    ) -> EncodeOutput:
        """Convert embeddings dict to EncodeOutput."""
        dense_np = None
        sparse_list = None
        multivector_list = None

        if "dense" in output_types and "dense" in embeddings:
            dense_np = embeddings["dense"].float().cpu().numpy()

        if "sparse" in output_types and "sparse" in embeddings:
            sparse_list = []
            for weights in embeddings["sparse"]:
                if weights:
                    indices = np.array(list(weights.keys()), dtype=np.int32)
                    values = np.array(list(weights.values()), dtype=np.float32)
                else:
                    indices = np.array([], dtype=np.int32)
                    values = np.array([], dtype=np.float32)
                sparse_list.append(SparseVector(indices=indices, values=values))

        if "multivector" in output_types and "multivector" in embeddings:
            multivec_tensor = embeddings["multivector"]
            multivector_list = []
            for i in range(batch_size):
                vecs = multivec_tensor[i].float().cpu().numpy()
                # Remove zero-padded vectors
                non_zero = np.any(vecs != 0, axis=-1)
                multivector_list.append(vecs[non_zero])

        return EncodeOutput(
            dense=dense_np,
            sparse=sparse_list,
            multivector=multivector_list,
            batch_size=batch_size,
            is_query=is_query,
            dense_dim=self.DENSE_DIM if dense_np is not None else None,
            multivector_token_dim=self.MULTIVECTOR_DIM if multivector_list is not None else None,
        )

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
