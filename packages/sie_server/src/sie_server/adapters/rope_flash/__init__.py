from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.nn import functional

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision, PoolingStrategy
from sie_server.adapters._utils import apply_rotary_pos_emb, extract_texts, validate_output_types
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_ERR_CPU_NOT_SUPPORTED = "RoPEFlashAdapter requires CUDA. Use pytorch_embedding adapter for CPU."


class RoPEFlashAdapter(PEFTLoRAMixin, FlashBaseAdapter):
    """RoPE-based encoder adapter using Flash Attention 2 with variable-length sequences.

    This adapter eliminates padding waste by packing sequences and using
    flash_attn_varlen_func. Supports models with Rotary Position Embeddings.

    Works with NewModel architecture (gte-multilingual-base).
    """

    fallback_adapter_path: ClassVar[str | None] = "sentence_transformer:SentenceTransformerDenseAdapter"

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("dense",),
        unload_fields=("_model", "_tokenizer", "_dense_dim", "_rope_dummy"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "float16",
        pooling: PoolingStrategy = "cls",
        query_template: str | None = None,
        doc_template: str | None = None,
        uses_legacy_transformers_cache: bool = False,
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
            uses_legacy_transformers_cache: If True, disable the KV cache after
                loading by setting model.config.use_cache = False. Required for
                models that use the legacy transformers cache API (pre-4.54).
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
        self._uses_legacy_transformers_cache = uses_legacy_transformers_cache

        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dense_dim: int | None = None
        self._rope_dummy: torch.Tensor | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (must be "cuda" or "cuda:X").

        Raises:
            RuntimeError: If device is not CUDA (flash attention requires GPU).
        """
        if not device.startswith("cuda"):
            raise RuntimeError(_ERR_CPU_NOT_SUPPORTED)

        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self._device = device
        dtype = self._resolve_dtype()

        logger.info(
            "Loading %s on device=%s with dtype=%s, attn=rope_flash_varlen, pooling=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._pooling,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        # Load config first to disable optional xformers features
        # Some models (e.g., stella_en_400M_v5) have these enabled in their saved config
        # but we replace attention with flash_attn_varlen_func anyway
        config = AutoConfig.from_pretrained(self._model_name_or_path, trust_remote_code=True)
        if hasattr(config, "use_memory_efficient_attention"):
            config.use_memory_efficient_attention = False
        if hasattr(config, "unpad_inputs"):
            config.unpad_inputs = False

        # Load model with eager attention - we handle attention manually
        self._model = AutoModel.from_pretrained(
            self._model_name_or_path,
            config=config,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        )

        # Disable KV cache for models using the legacy transformers cache API
        if self._uses_legacy_transformers_cache:
            self._model.config.use_cache = False

        self._model.to(device)
        self._model.eval()

        self._dense_dim = self._model.config.hidden_size
        logger.debug("RoPE model hidden_size: %d", self._dense_dim)

        # Clamp configured max_seq_length to whatever the tokenizer/model
        # actually support to avoid OOB position embeddings on long inputs.
        self._max_seq_length = self._resolve_tokenizer_ceiling(
            self._tokenizer,
            self._model,
            self._max_seq_length,
        )

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
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        validate_output_types(output_types, {"dense"}, "RoPEFlashAdapter")

        # Resolve runtime options (config defaults -> profile -> request overrides)
        opts = options or {}
        query_template = opts.get("query_template", self._query_template)
        doc_template = opts.get("doc_template", self._doc_template)
        normalize = opts.get("normalize", self._normalize)
        pooling = opts.get("pooling", self._pooling)

        texts = extract_texts(
            items,
            instruction,
            is_query=is_query,
            query_template=query_template,
            doc_template=doc_template,
            err_msg="RoPEFlashAdapter requires text input",
        )

        # Tokenize all sequences in a single batched call (no padding)
        batch_encoding = self._tokenizer(
            texts,
            max_length=self._max_seq_length,
            truncation=True,
            padding=False,
            return_length=True,
            return_tensors=None,  # Return lists, not tensors -- we pack ourselves
        )

        # Build packed representation
        seq_lengths = batch_encoding.get("length") or [len(ids) for ids in batch_encoding["input_ids"]]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack input_ids into a single 1-D tensor
        input_ids_packed = torch.cat(
            [torch.as_tensor(ids, dtype=torch.long) for ids in batch_encoding["input_ids"]],
        ).to(self._device)

        # Build cu_seqlens using cumsum (no Python loop)
        cu_seqlens = torch.zeros(len(texts) + 1, dtype=torch.int32, device=self._device)
        cu_seqlens[1:] = torch.cumsum(torch.tensor(seq_lengths, dtype=torch.int32, device=self._device), dim=0)

        with torch.inference_mode():
            # Build position IDs for RoPE
            position_ids_packed = self._build_position_ids(cu_seqlens, len(texts))

            # Get RoPE cos/sin values
            cos, sin = self._compute_rope(position_ids_packed, max_seqlen)

            # Run embeddings (no position embeddings - RoPE applied in attention)
            hidden = self._run_embeddings(input_ids_packed)

            # Run transformer layers with flash attention and RoPE
            hidden = self._run_transformer_flash(hidden, cu_seqlens, max_seqlen, total_tokens, cos, sin)

            # Pool to get dense embeddings
            dense_vecs = self._pool_embeddings(
                hidden,
                cu_seqlens,
                seq_lengths,
                normalize=normalize,
                pooling=pooling,
            )

        # Convert to numpy and return EncodeOutput
        dense_np = dense_vecs.float().cpu().numpy()
        return EncodeOutput(
            dense=dense_np,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=self._dense_dim,
        )

    def _build_position_ids(self, cu_seqlens: torch.Tensor, num_seqs: int) -> torch.Tensor:
        """Build position IDs for packed sequences.

        Each sequence has positions starting from 0.
        """
        total_tokens = cu_seqlens[-1].item()
        positions = torch.arange(total_tokens, device=self._device)
        seq_starts = torch.repeat_interleave(
            cu_seqlens[:-1],
            cu_seqlens[1:] - cu_seqlens[:-1],
        )
        return positions - seq_starts

    def _compute_rope(
        self,
        position_ids: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin values for packed positions.

        Returns:
            cos, sin tensors of shape [total_tokens, head_dim].
        """
        rotary_emb = self._model.embeddings.rotary_emb
        dtype = self._resolve_dtype()

        # Reuse a cached dummy tensor instead of allocating one every call
        if self._rope_dummy is None:
            self._rope_dummy = torch.zeros(1, 1, 1, 1, device=self._device, dtype=dtype)
        cos_cached, sin_cached = rotary_emb(self._rope_dummy, seq_len=max_seqlen)

        # Index into cached values using position IDs
        cos = cos_cached[position_ids]  # [total_tokens, head_dim]
        sin = sin_cached[position_ids]  # [total_tokens, head_dim]

        return cos, sin

    def _run_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for packed input (no position embeddings - RoPE in attention)."""
        embeddings = self._model.embeddings

        word_emb = embeddings.word_embeddings(input_ids)

        # token_type_embeddings (all zeros for this model)
        if hasattr(embeddings, "token_type_embeddings"):
            token_type_ids = torch.zeros_like(input_ids)
            token_type_emb = embeddings.token_type_embeddings(token_type_ids)
            hidden = word_emb + token_type_emb
        else:
            hidden = word_emb

        hidden = embeddings.LayerNorm(hidden)
        hidden = embeddings.dropout(hidden)

        return hidden

    def _run_transformer_flash(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run transformer layers using flash_attn_varlen_func with RoPE."""
        from flash_attn import flash_attn_varlen_func

        num_heads = self._model.config.num_attention_heads
        hidden_size = self._model.config.hidden_size
        head_dim = hidden_size // num_heads
        softmax_scale = 1.0 / (head_dim**0.5)

        for layer in self._model.encoder.layer:
            # QKV projection (combined)
            qkv = layer.attention.qkv_proj(hidden)
            # Split into Q, K, V (each is hidden_size)
            qkv = qkv.view(total_tokens, 3, num_heads, head_dim)
            query = qkv[:, 0]  # [total_tokens, num_heads, head_dim]
            key = qkv[:, 1]
            value = qkv[:, 2]

            # Apply RoPE to Q and K
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

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

            # Output projection
            attn_out = layer.attention.o_proj(attn_out)

            # Residual + dropout + LayerNorm (post-norm style)
            if layer.hidden_dropout is not None:
                attn_out = layer.hidden_dropout(attn_out)
            hidden = hidden + attn_out
            hidden = layer.attn_ln(hidden)

            # MLP (gated)  # section header
            residual = hidden
            mlp_out = layer.mlp(hidden)
            if layer.hidden_dropout is not None:
                mlp_out = layer.hidden_dropout(mlp_out)
            hidden = residual + mlp_out
            hidden = layer.mlp_ln(hidden)

        return hidden

    def _pool_embeddings(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
        *,
        normalize: bool | None = None,
        pooling: str | None = None,
    ) -> torch.Tensor:
        """Pool hidden states to get sequence embeddings."""
        normalize = normalize if normalize is not None else self._normalize
        pooling = pooling if pooling is not None else self._pooling
        num_seqs = len(seq_lengths)

        if pooling == "cls":
            # Extract CLS token from each sequence
            cls_embeddings = []
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                cls_embeddings.append(hidden[start])
            pooled = torch.stack(cls_embeddings)
        else:  # mean pooling
            # Average all tokens
            mean_embeddings = []
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                mean_embeddings.append(hidden[start:end].mean(dim=0))
            pooled = torch.stack(mean_embeddings)

        if normalize:
            pooled = functional.normalize(pooled, p=2, dim=-1)

        return pooled
