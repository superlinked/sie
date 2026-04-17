from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch.nn import functional

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ComputePrecision, PoolingStrategy
from sie_server.adapters._utils import (
    apply_rotary_pos_emb,
    extract_texts,
    resolve_embedding_options,
    validate_output_types,
)
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_ERR_CPU_NOT_SUPPORTED = "ModernBERTFlashAdapter requires CUDA for Flash Attention."


class ModernBERTFlashAdapter(PEFTLoRAMixin, FlashBaseAdapter):
    """Dense embedding adapter for ModernBERT with RoPE and Flash Attention 2 varlen.

    Combines the ModernBERT forward pass (pre-norm, fused QKV, RoPE,
    tok_embeddings) from ColBERTModernBERTFlashAdapter with dense output
    pooling (CLS/mean + normalize + runtime options) from BertFlashAdapter.

    Works with ModernBERT-based dense embedding models (gte-modernbert-base,
    granite-embedding-english-r2, granite-embedding-small-english-r2).
    """

    fallback_adapter_path: ClassVar[str | None] = "sentence_transformer:SentenceTransformerDenseAdapter"

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("dense",),
        unload_fields=("_model", "_tokenizer", "_dense_dim"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        pooling: PoolingStrategy = "cls",
        query_template: str | None = None,
        doc_template: str | None = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize dense embeddings.
            max_seq_length: Maximum sequence length.
            compute_precision: Compute precision (bfloat16 recommended).
            pooling: Pooling strategy - "cls" or "mean".
            query_template: Optional template for queries, e.g. "query: {text}".
            doc_template: Optional template for documents, e.g. "passage: {text}".
            trust_remote_code: Whether to trust remote code for model/tokenizer.
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
        self._trust_remote_code = trust_remote_code

        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dense_dim: int | None = None

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (must be "cuda" or "cuda:X").

        Raises:
            RuntimeError: If device is not CUDA.
        """
        if not device.startswith("cuda"):
            raise RuntimeError(_ERR_CPU_NOT_SUPPORTED)

        from transformers import AutoModel, AutoTokenizer

        self._device = device
        dtype = self._resolve_dtype()

        logger.info(
            "Loading ModernBERT %s on device=%s with dtype=%s, pooling=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._pooling,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # Load model with eager attention — we handle flash attention manually
        self._model = AutoModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        self._dense_dim = self._model.config.hidden_size

        logger.debug("ModernBERT hidden_size: %d", self._dense_dim)

        # Warmup flash attention kernels
        logger.info("Warming up CUDA kernels...")
        warmup_items = [Item(text="warmup")]
        self.encode(warmup_items, ["dense"])
        logger.info("Warmup complete")

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
        """Run inference returning dense embeddings.

        Args:
            items: List of items to encode.
            output_types: Which outputs to compute (only "dense" supported).
            instruction: Optional instruction prefix.
            is_query: Whether items are queries (affects template selection).
            prepared_items: Not used by this adapter.
            options: Runtime options for normalize, pooling, templates.

        Returns:
            EncodeOutput with dense embeddings.
        """
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        validate_output_types(output_types, {"dense"}, "ModernBERTFlashAdapter")

        # Resolve runtime options (config defaults -> profile -> request overrides)
        normalize, pooling, query_template, doc_template = resolve_embedding_options(
            options,
            default_normalize=self._normalize,
            default_pooling=self._pooling,
            default_query_template=self._query_template,
            default_doc_template=self._doc_template,
        )

        texts = extract_texts(
            items,
            instruction,
            is_query=is_query,
            query_template=query_template,
            doc_template=doc_template,
            err_msg="ModernBERTFlashAdapter requires text input",
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
            # Build position IDs for RoPE
            position_ids_packed = self._build_position_ids(seq_lengths_tensor, total_tokens)

            # Run embeddings (tok_embeddings -> norm -> drop)
            hidden = self._run_embeddings(input_ids_packed)

            # Pre-compute RoPE cos/sin for global and local layers
            global_cos, global_sin = self._compute_rope(position_ids_packed, use_global=True)
            local_cos, local_sin = self._compute_rope(position_ids_packed, use_global=False)

            # Run transformer layers with flash attention and RoPE
            hidden = self._run_transformer_flash(
                hidden,
                cu_seqlens,
                max_seqlen,
                total_tokens,
                global_cos,
                global_sin,
                local_cos,
                local_sin,
            )

            # Apply final layer norm (ModernBERT has a final_norm after all layers)
            if hasattr(self._model, "final_norm"):
                hidden = self._model.final_norm(hidden)

            # Pool to get dense embeddings
            dense_vecs = self._pool_embeddings(
                hidden,
                cu_seqlens,
                seq_lengths,
                seq_lengths_tensor=seq_lengths_tensor,
                normalize=normalize,
                pooling=pooling,
            )

        # Transfer 16-bit to CPU first, then upcast to float32 — halves GPU->CPU bandwidth
        dense_np = dense_vecs.cpu().float().numpy()
        return EncodeOutput(
            dense=dense_np,
            batch_size=len(items),
            is_query=is_query,
            dense_dim=self._dense_dim,
        )

    def _build_position_ids(self, seq_lengths_tensor: torch.Tensor, total_tokens: int) -> torch.Tensor:
        """Build position IDs for packed sequences (vectorized).

        Args:
            seq_lengths_tensor: Tensor of sequence lengths on device.
            total_tokens: Total number of tokens.
        """
        offsets = torch.zeros_like(seq_lengths_tensor)
        offsets[1:] = torch.cumsum(seq_lengths_tensor[:-1], dim=0)
        flat_offsets = torch.repeat_interleave(offsets, seq_lengths_tensor)
        return (torch.arange(total_tokens, device=self._device) - flat_offsets).long()

    def _compute_rope(
        self,
        position_ids: torch.Tensor,
        *,
        use_global: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin values for packed positions.

        ModernBERT uses different rope_theta values for global vs local attention
        layers: ``global_rope_theta`` (default 160000) for global layers and
        ``local_rope_theta`` (default 10000) for local (sliding-window) layers.

        Args:
            position_ids: Packed position IDs [total_tokens].
            use_global: If True, use global_rope_theta; otherwise local_rope_theta.

        Returns:
            cos, sin tensors of shape [total_tokens, head_dim].
        """
        head_dim = self._model.config.hidden_size // self._model.config.num_attention_heads
        cfg = self._model.config

        if use_global:
            base = getattr(cfg, "global_rope_theta", getattr(cfg, "rope_theta", 160000.0))
        else:
            base = getattr(cfg, "local_rope_theta", getattr(cfg, "rope_theta", 10000.0))

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=self._device).float() / head_dim))

        pos = position_ids.float()
        freqs = torch.outer(pos, inv_freq)  # [total_tokens, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [total_tokens, head_dim]

        return emb.cos().to(self._resolve_dtype()), emb.sin().to(self._resolve_dtype())

    def _run_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for packed input (no position embeddings — RoPE in attention)."""
        embeddings = self._model.embeddings

        # ModernBERT: tok_embeddings -> norm -> drop
        hidden = embeddings.tok_embeddings(input_ids)
        if hasattr(embeddings, "norm"):
            hidden = embeddings.norm(hidden)
        if hasattr(embeddings, "drop"):
            hidden = embeddings.drop(hidden)

        return hidden

    def _run_transformer_flash(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        global_cos: torch.Tensor,
        global_sin: torch.Tensor,
        local_cos: torch.Tensor,
        local_sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run transformer layers using flash_attn_varlen_func with RoPE.

        ModernBERT uses pre-norm architecture with local/global attention
        patterns.  Every ``global_attn_every_n_layers``-th layer (0-indexed)
        uses full (global) attention with ``global_rope_theta``; the remaining
        layers use sliding-window (local) attention of size
        ``local_attention`` with ``local_rope_theta``.
        """
        from flash_attn import flash_attn_varlen_func  # ty: ignore[unresolved-import]

        cfg = self._model.config
        num_heads = cfg.num_attention_heads
        hidden_size = cfg.hidden_size
        head_dim = hidden_size // num_heads
        softmax_scale = 1.0 / (head_dim**0.5)

        global_every_n = getattr(cfg, "global_attn_every_n_layers", 1)
        local_window = getattr(cfg, "local_attention", -1)
        # flash_attn_varlen_func expects window_size as (left, right) tuple
        window = (local_window // 2, local_window // 2) if local_window > 0 else (-1, -1)

        for layer_idx, layer in enumerate(self._model.layers):
            is_global = (layer_idx % global_every_n == 0) if global_every_n > 1 else True
            cos = global_cos if is_global else local_cos
            sin = global_sin if is_global else local_sin

            # Pre-attention norm (ModernBERT is pre-norm)
            normed_hidden = layer.attn_norm(hidden)

            # Fused QKV projection
            qkv = layer.attn.Wqkv(normed_hidden)
            qkv = qkv.view(total_tokens, 3, num_heads, head_dim)
            query = qkv[:, 0]  # [total_tokens, num_heads, head_dim]
            key = qkv[:, 1]
            value = qkv[:, 2]

            # Apply RoPE to Q and K (using layer-appropriate theta)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            # Flash attention — global layers use full attention,
            # local layers use sliding window
            attn_kwargs: dict[str, Any] = {}
            if not is_global and local_window > 0:
                attn_kwargs["window_size"] = window

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
                **attn_kwargs,
            )
            attn_out = attn_out.reshape(total_tokens, hidden_size)

            # Output projection
            attn_out = layer.attn.Wo(attn_out)

            # Residual connection
            hidden = hidden + attn_out

            # MLP block with pre-norm
            normed_hidden = layer.mlp_norm(hidden)
            mlp_out = layer.mlp(normed_hidden)
            hidden = hidden + mlp_out

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
