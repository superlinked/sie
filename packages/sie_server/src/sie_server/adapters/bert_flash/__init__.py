"""BERT Flash Attention adapter using flash_attn_varlen_func.

This adapter uses Flash Attention 2's variable-length attention to process
sequences without padding, eliminating padding waste and improving throughput.

Supports BERT-based models like:
- intfloat/e5-small-v2, e5-base-v2, e5-large-v2
- sentence-transformers/all-MiniLM-L6-v2

Key features:
- Uses flash_attn_varlen_func with cu_seqlens for packed sequences
- No padding tokens = no wasted compute
- Position IDs start at 0 (standard BERT convention, unlike XLMRoberta which starts at 2)

See: https://github.com/Dao-AILab/flash-attention
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from torch.nn import functional

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import BertModel, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]
PoolingStrategy = Literal["cls", "mean"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "BertFlashAdapter requires text input"
_ERR_CPU_NOT_SUPPORTED = "BertFlashAdapter requires CUDA. Use pytorch_embedding adapter for CPU."


class BertFlashAdapter(PEFTLoRAMixin, ModelAdapter):
    """BERT adapter using Flash Attention 2 with variable-length sequences.

    This adapter eliminates padding waste by packing sequences and using
    flash_attn_varlen_func. Achieves higher throughput than SDPA-based adapters.

    Works with any BERT-based model for dense embeddings.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        max_seq_length: int = 512,
        compute_precision: ComputePrecision = "float16",
        pooling: PoolingStrategy = "mean",
        query_template: str | None = None,
        doc_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize dense embeddings.
            max_seq_length: Maximum sequence length.
            compute_precision: Compute precision (float16 recommended for flash).
            pooling: Pooling strategy - "cls" or "mean".
            query_template: Optional template for queries, e.g. "query: {text}".
            doc_template: Optional template for documents, e.g. "passage: {text}".
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._pooling = pooling
        self._query_template = query_template
        self._doc_template = doc_template

        self._model: BertModel | None = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dense_dim: int | None = None
        self._fused_qkv_weights: list[torch.Tensor] = []
        self._fused_qkv_biases: list[torch.Tensor] = []
        self._token_type_emb_0: torch.Tensor | None = None

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> ModelAdapter:
        """Factory method that returns the appropriate adapter for the device.

        For non-CUDA devices or when flash-attn is unavailable, returns SentenceTransformerDenseAdapter.

        Args:
            device: Device string (e.g., "cuda:0", "mps", "cpu").
            **kwargs: Adapter initialization parameters.

        Returns:
            BertFlashAdapter for CUDA with flash-attn, SentenceTransformerDenseAdapter otherwise.
        """
        from sie_server.adapters.sentence_transformer import SentenceTransformerDenseAdapter

        # Use base class helper (fallback gets same kwargs)
        return cls._create_flash_or_fallback(device, fallback_class=SentenceTransformerDenseAdapter, **kwargs)

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
        if self._dense_dim is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        return ModelDims(dense=self._dense_dim)

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (must be "cuda" or "cuda:X").

        Raises:
            RuntimeError: If device is not CUDA (flash attention requires GPU).
        """
        if not device.startswith("cuda"):
            raise RuntimeError(_ERR_CPU_NOT_SUPPORTED)

        from transformers import AutoModel, AutoTokenizer

        self._device = device
        dtype = self._resolve_dtype()

        logger.info(
            "Loading %s on device=%s with dtype=%s, attn=flash_varlen, pooling=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._pooling,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        # Load model with eager attention - we'll run our own flash attention
        self._model = AutoModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",  # We handle attention manually
        )
        self._model.to(device)
        self._model.eval()

        self._dense_dim = self._model.config.hidden_size

        # Fuse QKV weights at load time to avoid per-call concatenation
        self._fused_qkv_weights = []
        self._fused_qkv_biases = []
        for layer in self._model.encoder.layer:
            attention = layer.attention.self
            qkv_w = torch.cat([attention.query.weight, attention.key.weight, attention.value.weight], dim=0)
            qkv_b = torch.cat([attention.query.bias, attention.key.bias, attention.value.bias], dim=0)
            self._fused_qkv_weights.append(qkv_w)
            self._fused_qkv_biases.append(qkv_b)

        # Cache token_type_embeddings[0] to avoid creating zeros_like tensor each call
        self._token_type_emb_0 = self._model.embeddings.token_type_embeddings.weight[0]

        logger.debug("BERT hidden_size: %d", self._dense_dim)

        # Warmup flash attention kernels
        logger.info("Warming up CUDA kernels...")
        warmup_items = [Item(text="warmup")]
        self.encode(warmup_items, ["dense"])
        logger.info("Warmup complete")

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.float16)

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device

        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._device = None
        self._dense_dim = None
        self._fused_qkv_weights = []
        self._fused_qkv_biases = []
        self._token_type_emb_0 = None

        gc.collect()
        if device and device.startswith("cuda"):
            torch.cuda.empty_cache()

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
            output_types: Which outputs to compute (only "dense" supported).
            instruction: Optional instruction prefix.
            is_query: Whether items are queries (affects template selection).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with dense embeddings.
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        self._validate_output_types(output_types)

        # Resolve runtime options (config defaults -> profile -> request overrides)
        opts = options or {}
        query_template = opts.get("query_template", self._query_template)
        doc_template = opts.get("doc_template", self._doc_template)
        normalize = opts.get("normalize", self._normalize)
        pooling = opts.get("pooling", self._pooling)

        texts = self._extract_texts(
            items,
            instruction,
            is_query=is_query,
            query_template=query_template,
            doc_template=doc_template,
        )

        # Batch tokenization — single call instead of per-text loop
        batch_encoding = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
        )

        # Extract input_ids as lists and compute seq_lengths
        input_ids_lists = batch_encoding["input_ids"]
        seq_lengths = [len(ids) for ids in input_ids_lists]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack input_ids — build flat list first, create one tensor
        all_input_ids: list[int] = []
        for ids in input_ids_lists:
            all_input_ids.extend(ids)
        input_ids_packed = torch.tensor(all_input_ids, dtype=torch.long, device=self._device)

        # Build cu_seqlens with vectorized cumsum
        seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=self._device)
        cu_seqlens = torch.zeros(len(texts) + 1, dtype=torch.int32, device=self._device)
        cu_seqlens[1:] = torch.cumsum(seq_lengths_tensor, dim=0)

        with torch.inference_mode():
            # Build BERT-style position IDs (start at 0) — vectorized
            position_ids_packed = self._build_position_ids(seq_lengths_tensor, total_tokens)

            # Run embeddings
            hidden = self._run_embeddings(input_ids_packed, position_ids_packed)

            # Run transformer layers with flash attention
            hidden = self._run_transformer_flash(hidden, cu_seqlens, max_seqlen, total_tokens)

            # Pool to get dense embeddings
            dense_vecs = self._pool_embeddings(
                hidden,
                cu_seqlens,
                seq_lengths,
                seq_lengths_tensor=seq_lengths_tensor,
                normalize=normalize,
                pooling=pooling,
            )

        # Transfer 16-bit to CPU first, then upcast to float32 — halves GPU→CPU bandwidth
        dense_np = dense_vecs.cpu().float().numpy()
        return EncodeOutput(
            dense=dense_np,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=self._dense_dim,
        )

    def _build_position_ids(self, seq_lengths_tensor: torch.Tensor, total_tokens: int) -> torch.Tensor:
        """Build BERT-style position IDs for packed sequences (vectorized).

        BERT uses position IDs starting at 0, unlike XLMRoberta which starts at
        padding_idx + 1 = 2. This is critical for matching the padded model's output.

        Args:
            seq_lengths_tensor: Tensor of sequence lengths on device.
            total_tokens: Total number of tokens (pre-computed on CPU to avoid GPU sync).
        """
        offsets = torch.zeros_like(seq_lengths_tensor)
        offsets[1:] = torch.cumsum(seq_lengths_tensor[:-1], dim=0)
        flat_offsets = torch.repeat_interleave(offsets, seq_lengths_tensor)
        return torch.arange(total_tokens, device=self._device) - flat_offsets

    def _run_embeddings(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for packed input."""
        embeddings = self._model.embeddings  # type: ignore[union-attr]

        word_emb = embeddings.word_embeddings(input_ids)
        pos_emb = embeddings.position_embeddings(position_ids)
        # Use pre-cached token_type_embeddings[0] — broadcasts, avoids zeros_like allocation
        token_type_emb = self._token_type_emb_0

        hidden = word_emb + pos_emb + token_type_emb
        hidden = embeddings.LayerNorm(hidden)
        return embeddings.dropout(hidden)

    def _run_transformer_flash(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
    ) -> torch.Tensor:
        """Run transformer layers using flash_attn_varlen_func."""
        from flash_attn import flash_attn_varlen_func

        num_heads = self._model.config.num_attention_heads  # type: ignore[union-attr]
        hidden_size = self._model.config.hidden_size  # type: ignore[union-attr]
        head_dim = hidden_size // num_heads
        softmax_scale = 1.0 / (head_dim**0.5)

        for layer_idx, layer in enumerate(self._model.encoder.layer):  # type: ignore[union-attr]
            # Cache layer sub-modules to avoid repeated attribute chain lookups
            attn_output = layer.attention.output
            intermediate = layer.intermediate
            output = layer.output

            # Fused QKV projection — single matmul instead of 3 separate
            qkv_out = functional.linear(hidden, self._fused_qkv_weights[layer_idx], self._fused_qkv_biases[layer_idx])
            query, key, value = qkv_out.split(hidden_size, dim=-1)
            query = query.view(total_tokens, num_heads, head_dim)
            key = key.view(total_tokens, num_heads, head_dim)
            value = value.view(total_tokens, num_heads, head_dim)

            # Flash attention with variable-length sequences
            attn_out = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=False,
                softmax_scale=softmax_scale,
            )
            attn_out = attn_out.reshape(total_tokens, hidden_size)

            # Output projection and residual
            attn_out = attn_output.dense(attn_out)
            attn_out = attn_output.dropout(attn_out)
            hidden = attn_output.LayerNorm(attn_out + hidden)

            # FFN
            inter = intermediate.dense(hidden)
            inter = intermediate.intermediate_act_fn(inter)
            out = output.dense(inter)
            out = output.dropout(out)
            hidden = output.LayerNorm(out + hidden)

        return hidden

    def _pool_embeddings(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
        *,
        seq_lengths_tensor: torch.Tensor | None = None,
        normalize: bool | None = None,
        pooling: str | None = None,
    ) -> torch.Tensor:
        """Pool hidden states to get sequence embeddings."""
        normalize = normalize if normalize is not None else self._normalize
        pooling = pooling if pooling is not None else self._pooling
        num_seqs = len(seq_lengths)

        if pooling == "cls":
            # Vectorized CLS extraction — index directly with cu_seqlens
            pooled = hidden[cu_seqlens[:-1].long()]
        else:  # mean pooling — vectorized with scatter_add in float32
            if seq_lengths_tensor is None:
                seq_lengths_tensor = torch.tensor(seq_lengths, dtype=torch.int32, device=hidden.device)
            # Create segment IDs for each token
            seg_ids = torch.repeat_interleave(
                torch.arange(num_seqs, device=hidden.device),
                seq_lengths_tensor,
            )
            # Accumulate in float32 to match torch.mean() precision behavior
            segment_sums = torch.zeros(num_seqs, hidden.shape[-1], device=hidden.device, dtype=torch.float32)
            segment_sums.scatter_add_(0, seg_ids.unsqueeze(-1).expand_as(hidden), hidden.float())
            pooled = (segment_sums / seq_lengths_tensor.unsqueeze(-1).float()).to(hidden.dtype)

        if normalize:
            pooled = functional.normalize(pooled, p=2, dim=-1)

        return pooled

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"dense"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. BertFlashAdapter only supports 'dense'."
            raise ValueError(msg)

    def _extract_texts(
        self,
        items: list[Item],
        instruction: str | None,
        *,
        is_query: bool,
        query_template: str | None = None,
        doc_template: str | None = None,
    ) -> list[str]:
        """Extract texts from items, applying templates if configured."""
        query_template = query_template if query_template is not None else self._query_template
        doc_template = doc_template if doc_template is not None else self._doc_template
        texts = []
        for item in items:
            if item.text is None:
                raise ValueError(_ERR_REQUIRES_TEXT)

            text = item.text

            # Apply template based on query/document mode
            template = query_template if is_query else doc_template
            if template:
                text = template.format(text=text, instruction=instruction or "")
            elif instruction:
                text = f"{instruction} {text}"

            texts.append(text)
        return texts

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
