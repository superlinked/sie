"""ModernBERT Flash Attention Cross-Encoder adapter for reranking.

This adapter implements cross-encoder scoring with Flash Attention 2 varlen,
eliminating padding waste for variable-length query-document pairs.

Supports:
- Alibaba-NLP/gte-reranker-modernbert-base

Key optimizations:
- Flash Attention 2 with variable-length sequences (no padding waste)
- FP16/BF16 inference
- Packed batching via cu_seqlens
- Uses model's native flash attention with sliding window support

Architecture (ModernBERT):
- model.embeddings: tok_embeddings + norm + drop
- model.layers[i].attn_norm / mlp_norm: layer norms (pre-norm)
- model.layers[i].attn.Wqkv: fused QKV projection
- model.layers[i].attn.rotary_emb: rotary embeddings (flash version)
- model.layers[i].attn.Wo: output projection
- model.layers[i].attn.local_attention: window size for sliding window attention
- head: dense + act + norm
- classifier: final linear layer
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
_ERR_REQUIRES_TEXT = "ModernBertFlashCrossEncoder requires text input"
_ERR_CUDA_REQUIRED = "ModernBertFlashCrossEncoder requires CUDA for Flash Attention."


class ModernBertFlashCrossEncoderAdapter(ModelAdapter):
    """Cross-encoder adapter for ModernBERT with Flash Attention 2 varlen.

    Uses flash_attn_varlen_qkvpacked_func to avoid wasting compute on padding tokens.
    Supports ModernBERT architecture with RoPE, pre-norm, and sliding window attention.
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
            trust_remote_code: Whether to trust remote code.
            max_length: Maximum sequence length for query+document.
            compute_precision: Compute precision (bfloat16 recommended for ModernBERT).
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

        # Model config (set during load)
        self._num_heads: int = 0
        self._head_dim: int = 0
        self._hidden_size: int = 0
        self._use_sigmoid: bool = True
        self._use_mean_pooling: bool = False

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> ModelAdapter:
        """Factory method that returns the appropriate adapter for the device.

        For non-CUDA devices or when flash-attn is unavailable, returns CrossEncoderAdapter.

        Args:
            device: Device string (e.g., "cuda:0", "mps", "cpu").
            **kwargs: Adapter initialization parameters.

        Returns:
            ModernBertFlashCrossEncoderAdapter for CUDA with flash-attn, CrossEncoderAdapter otherwise.
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
            "Loading %s with Flash Attention 2 (dtype=%s)",
            self._model_name_or_path,
            self._dtype,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # Load model with flash_attention_2 to use model's optimized rotary embeddings
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name_or_path,
            torch_dtype=self._dtype,
            attn_implementation="flash_attention_2",
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        # Cache config values
        config = self._model.config
        self._num_heads = config.num_attention_heads
        self._hidden_size = config.hidden_size
        self._head_dim = self._hidden_size // self._num_heads

        # Use sigmoid for single-label classification
        self._use_sigmoid = config.num_labels == 1

        # Check pooling type (mean vs cls)
        self._use_mean_pooling = getattr(config, "classifier_pooling", "cls") == "mean"

        logger.info(
            "Loaded: hidden=%d, heads=%d, head_dim=%d, layers=%d, sigmoid=%s, pooling=%s",
            self._hidden_size,
            self._num_heads,
            self._head_dim,
            config.num_hidden_layers,
            self._use_sigmoid,
            "mean" if self._use_mean_pooling else "cls",
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

        # Tokenize all pairs individually (no padding)
        pairs = [(query_text, self._extract_text(item)) for item in items]
        encodings = [
            self._tokenizer(
                q,
                d,
                max_length=self._max_length,
                truncation=True,
                return_tensors="pt",
            )
            for q, d in pairs
        ]

        # Build packed representation
        seq_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack tensors
        input_ids = torch.cat([enc["input_ids"].squeeze(0) for enc in encodings]).to(self._device)

        # Build cu_seqlens
        cu_seqlens = torch.zeros(len(pairs) + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            # Run forward pass with flash attention
            logits = self._forward_flash(
                input_ids,
                cu_seqlens,
                max_seqlen,
                total_tokens,
                len(pairs),
                seq_lengths,
            )

            # Apply activation function
            scores_tensor = logits.squeeze(-1).float()
            if self._use_sigmoid:
                scores_tensor = torch.sigmoid(scores_tensor)
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

        # Tokenize all pairs individually (no padding)
        encodings = [
            self._tokenizer(
                q,
                d,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            for q, d in pairs
        ]

        # Build packed representation
        seq_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack tensors
        input_ids = torch.cat([enc["input_ids"].squeeze(0) for enc in encodings]).to(self._device)

        # Build cu_seqlens
        cu_seqlens = torch.zeros(len(pairs) + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            logits = self._forward_flash(
                input_ids,
                cu_seqlens,
                max_seqlen,
                total_tokens,
                len(pairs),
                seq_lengths,
            )

            scores_tensor = logits.squeeze(-1).float()
            if self._use_sigmoid:
                scores_tensor = torch.sigmoid(scores_tensor)

            # Convert to float32 numpy array and wrap in ScoreOutput
            import numpy as np

            scores_array = scores_tensor.cpu().numpy().astype(np.float32)

        return ScoreOutput(scores=scores_array)

    def _forward_flash(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        batch_size: int,
        seq_lengths: list[int],
    ) -> torch.Tensor:
        """Run forward pass with flash attention varlen.

        Args:
            input_ids: Packed input IDs [total_tokens].
            cu_seqlens: Cumulative sequence lengths [batch_size + 1].
            max_seqlen: Maximum sequence length in batch.
            total_tokens: Total number of tokens.
            batch_size: Number of sequences.
            seq_lengths: List of individual sequence lengths.

        Returns:
            Logits tensor [batch_size, 1].
        """
        from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

        # Get model components
        backbone = self._model.model
        embeddings = backbone.embeddings
        layers = backbone.layers
        final_norm = backbone.final_norm
        head = self._model.head
        drop = self._model.drop
        classifier = self._model.classifier

        # Compute embeddings (ModernBERT: tok_embeddings -> norm -> drop)
        hidden = embeddings.tok_embeddings(input_ids)
        if hasattr(embeddings, "norm"):
            hidden = embeddings.norm(hidden)
        if hasattr(embeddings, "drop"):
            hidden = embeddings.drop(hidden)

        softmax_scale = 1.0 / (self._head_dim**0.5)

        # Run transformer layers with flash attention
        for layer in layers:
            # Pre-attention norm (ModernBERT is pre-norm)
            normed_hidden = layer.attn_norm(hidden)

            # Fused QKV projection
            qkv = layer.attn.Wqkv(normed_hidden)
            qkv = qkv.view(total_tokens, 3, self._num_heads, self._head_dim)

            # Apply rotary embeddings using model's optimized implementation
            qkv = layer.attn.rotary_emb(qkv, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

            # Get window_size from layer (for sliding window attention)
            window_size = layer.attn.local_attention  # e.g., (-1,-1) for global or (64,64) for local

            # Flash attention with variable-length sequences
            attn_out = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen=max_seqlen,
                softmax_scale=softmax_scale,
                causal=False,
                window_size=window_size,
            )
            attn_out = attn_out.reshape(total_tokens, self._hidden_size)

            # Output projection
            attn_out = layer.attn.Wo(attn_out)

            # Residual connection
            hidden = hidden + attn_out

            # MLP block with pre-norm
            normed_hidden = layer.mlp_norm(hidden)
            mlp_out = layer.mlp(normed_hidden)
            hidden = hidden + mlp_out

        # Apply final layer norm
        hidden = final_norm(hidden)

        # Pool hidden states based on config
        if self._use_mean_pooling:
            # Mean pooling over each sequence
            pooled_list = []
            for i in range(batch_size):
                start = int(cu_seqlens[i].item())
                end = int(cu_seqlens[i + 1].item())
                seq_hidden = hidden[start:end]  # [seq_len, hidden]
                pooled_list.append(seq_hidden.mean(dim=0))  # [hidden]
            pooled_hidden = torch.stack(pooled_list)  # [batch, hidden]
        else:
            # CLS pooling: extract first token of each sequence
            cls_indices = cu_seqlens[:-1].long()
            pooled_hidden = hidden[cls_indices]  # [batch, hidden]

        # Apply classification head: head -> drop -> classifier
        pooled = head(pooled_hidden)
        pooled = drop(pooled)
        logits = classifier(pooled)

        return logits

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(_ERR_REQUIRES_TEXT)
        return item.text

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
