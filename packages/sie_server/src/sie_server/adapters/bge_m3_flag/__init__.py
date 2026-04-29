"""BGE-M3 model adapter using FlagEmbedding library.

BGE-M3 is a multi-functional embedding model that supports:
- Dense embeddings (1024 dims)
- Sparse embeddings (lexical, SPLADE-like)
- Multi-vector embeddings (ColBERT-like)

This adapter uses the FlagEmbedding library. For better performance,
use the default BGEM3Adapter (bge_m3.py) which uses Flash Attention 2.

See: https://huggingface.co/BAAI/bge-m3
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters.bge_m3_score_mixin import BGEM3ScoreMixin
from sie_server.core.inference_output import EncodeOutput, SparseVector

if TYPE_CHECKING:
    from pathlib import Path

    from FlagEmbedding import BGEM3FlagModel

    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)


class BGEM3FlagAdapter(BGEM3ScoreMixin, BaseAdapter):
    """Adapter for BAAI/bge-m3 using FlagEmbedding library.

    This adapter uses the FlagEmbedding library's BGEM3FlagModel.
    For better performance, use BGEM3Adapter which uses Flash Attention 2.

    Scoring (`/v1/score`) is supported via :class:`BGEM3ScoreMixin`, which
    composes scores from the encoder outputs (dense / sparse / multivector).
    """

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("dense", "sparse", "multivector", "score"),
        dense_dim=1024,
        sparse_dim=250002,
        multivector_dim=1024,
        unload_fields=("_model",),
    )

    # BGE-M3 specific dimensions
    DENSE_DIM = 1024
    SPARSE_DIM = 250002  # Vocabulary size
    MULTIVECTOR_DIM = 1024  # Per-token dimension

    # Default batch size for encoding - balances memory usage and throughput
    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        model_name_or_path: str | Path = "BAAI/bge-m3",
        *,
        normalize: bool = True,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "float16",
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize dense embeddings.
            max_seq_length: Maximum sequence length (default 8192).
            compute_precision: Compute precision (float16, bfloat16, float32).
        """
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision

        self._model: BGEM3FlagModel | None = None
        self._device: str | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu").
        """
        from FlagEmbedding import BGEM3FlagModel

        self._device = device

        # Resolve use_fp16 based on compute_precision and device
        # BGEM3FlagModel only supports use_fp16 boolean, not bfloat16
        use_fp16 = self._resolve_use_fp16(device)

        logger.info(
            "Loading %s on device=%s with use_fp16=%s (precision=%s)",
            self._model_name_or_path,
            device,
            use_fp16,
            self._compute_precision,
        )

        self._model = BGEM3FlagModel(
            self._model_name_or_path,
            use_fp16=use_fp16,
            device=device,
        )

    def _resolve_use_fp16(self, device: str) -> bool:
        """Resolve use_fp16 flag based on compute_precision and device.

        BGEM3FlagModel only supports use_fp16 boolean. This method maps
        compute_precision to the appropriate value.

        Args:
            device: Target device string.

        Returns:
            True if FP16 should be used, False for FP32.
        """
        # CPU should always use FP32 (FP16 is very slow on CPU)
        if not device.startswith("cuda"):
            if self._compute_precision != "float32":
                logger.debug(
                    "Precision %s requested on %s - using FP32 for CPU",
                    self._compute_precision,
                    device,
                )
            return False

        # On GPU, map precision to use_fp16
        if self._compute_precision == "float32":
            return False
        if self._compute_precision == "bfloat16":
            # BGEM3FlagModel doesn't support bfloat16, use fp16 as fallback
            logger.warning("BFloat16 requested but BGEM3FlagModel only supports FP16; using FP16")
            return True
        return True

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

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If output_types contains unsupported types.
        """
        self._check_loaded()
        if self._model is None:
            raise RuntimeError(ERR_NOT_LOADED)

        self._validate_output_types(output_types)
        texts = self._extract_texts(items, instruction)

        # Determine what to compute
        return_dense = "dense" in output_types
        return_sparse = "sparse" in output_types
        return_colbert = "multivector" in output_types

        # Encode with BGEM3FlagModel
        with torch.inference_mode():
            outputs = self._model.encode(
                texts,
                batch_size=self.DEFAULT_BATCH_SIZE,
                max_length=self._max_seq_length,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert,
            )

        return self._to_encode_output(
            cast("dict[str, Any]", outputs),
            len(items),
            is_query,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert=return_colbert,
        )

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
                raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="BGEM3FlagAdapter"))
            text = item.text
            if instruction is not None:
                text = f"{instruction} {text}"
            texts.append(text)
        return texts

    def _to_encode_output(
        self,
        outputs: Mapping[str, Any],
        batch_size: int,
        is_query: bool,
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> EncodeOutput:
        """Convert model outputs to EncodeOutput."""
        dense_np = None
        sparse_list = None
        multivector_list = None

        if return_dense:
            embeddings = outputs["dense_vecs"]
            if self._normalize:
                norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-12)
            dense_np = embeddings

        if return_sparse:
            sparse_list = []
            for weights in outputs["lexical_weights"]:
                if weights:
                    indices = np.array(list(weights.keys()), dtype=np.int32)
                    values = np.array(list(weights.values()), dtype=np.float32)
                else:
                    indices = np.array([], dtype=np.int32)
                    values = np.array([], dtype=np.float32)
                sparse_list.append(SparseVector(indices=indices, values=values))

        if return_colbert:
            multivector_list = outputs["colbert_vecs"]

        return EncodeOutput(
            dense=dense_np,
            sparse=sparse_list,
            multivector=multivector_list,
            batch_size=batch_size,
            is_query=is_query,
            dense_dim=self.DENSE_DIM if dense_np is not None else None,
            multivector_token_dim=self.MULTIVECTOR_DIM if multivector_list is not None else None,
        )
