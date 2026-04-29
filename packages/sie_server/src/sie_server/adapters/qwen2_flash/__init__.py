from __future__ import annotations

import json
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

_ERR_CPU_NOT_SUPPORTED = "Qwen2FlashAdapter requires CUDA. Use sentence_transformer adapter for CPU."


class Qwen2FlashAdapter(PEFTLoRAMixin, FlashBaseAdapter):
    """Qwen2-based encoder adapter using Flash Attention 2 with variable-length sequences.

    This adapter eliminates padding waste by packing sequences and using
    flash_attn_varlen_func. Specifically designed for Qwen2 architecture models.

    Works with models like dunzhang/stella_en_1.5B_v5.
    """

    fallback_adapter_path: ClassVar[str | None] = "sentence_transformer:SentenceTransformerDenseAdapter"

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("dense",),
        unload_fields=("_model", "_tokenizer", "_dense_dim", "_dense_projection"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        pooling: PoolingStrategy = "mean",
        query_template: str | None = None,
        doc_template: str | None = None,
        uses_legacy_transformers_cache: bool = False,
        dense_projection_path: str | None = None,
        causal: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            normalize: Whether to L2-normalize dense embeddings.
            max_seq_length: Maximum sequence length.
            compute_precision: Compute precision (bfloat16 recommended for Qwen2).
            pooling: Pooling strategy - "cls", "mean", or "last".
            query_template: Optional template for queries, e.g. "query: {text}".
            doc_template: Optional template for documents, e.g. "passage: {text}".
            **kwargs: Additional arguments (ignored, for compatibility).
            uses_legacy_transformers_cache: If True, disable the KV cache after
                loading by setting model.config.use_cache = False. Required for
                models that use the legacy transformers cache API (pre-4.54).
            dense_projection_path: Optional subfolder in the HuggingFace repo
                containing a dense projection layer (config.json + weights).
            causal: If True, use causal (autoregressive) attention instead of
                bidirectional. Required for decoder-based embedding models like
                Qwen3-Embedding that use last-token pooling with causal masking.
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._pooling = pooling
        self._query_template = query_template
        self._doc_template = doc_template
        self._causal = causal
        self._uses_legacy_transformers_cache = uses_legacy_transformers_cache
        self._dense_projection_path = dense_projection_path

        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dense_dim: int | None = None
        self._dense_projection: torch.nn.Linear | None = None

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype (defaults to bfloat16 for Qwen2)."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)

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
            "Loading %s on device=%s with dtype=%s, attn=qwen2_flash_varlen, pooling=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._pooling,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        # Load config first to disable optional xformers features
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

        if self._dense_projection_path:
            self._load_dense_projection(device)

        logger.debug("Qwen2 model dense_dim: %d", self._dense_dim)

        # Clamp configured max_seq_length to whatever the tokenizer/model
        # actually support to avoid OOB position embeddings on long inputs.
        self._max_seq_length = self._resolve_tokenizer_ceiling(
            self._tokenizer,
            self._model,
            self._max_seq_length,
        )

    def _load_dense_projection(self, device: str) -> None:
        """Load the sentence-transformers dense projection layer from HuggingFace."""
        import safetensors.torch
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        config_file = hf_hub_download(
            self._model_name_or_path,
            f"{self._dense_projection_path}/config.json",
        )

        with open(config_file) as f:
            proj_config = json.load(f)

        in_features = proj_config["in_features"]
        out_features = proj_config["out_features"]
        bias = proj_config.get("bias", True)

        self._dense_projection = torch.nn.Linear(in_features, out_features, bias=bias)

        try:
            weights_path = hf_hub_download(
                self._model_name_or_path,
                f"{self._dense_projection_path}/model.safetensors",
            )
            state_dict = safetensors.torch.load_file(weights_path)
        except (EntryNotFoundError, OSError):
            weights_path = hf_hub_download(
                self._model_name_or_path,
                f"{self._dense_projection_path}/pytorch_model.bin",
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        if any(k.startswith("linear.") for k in state_dict):
            state_dict = {k.removeprefix("linear."): v for k, v in state_dict.items()}
        self._dense_projection.load_state_dict(state_dict)
        self._dense_projection.to(device=device, dtype=self._resolve_dtype())
        self._dense_projection.eval()
        self._dense_dim = out_features

        logger.info(
            "Loaded dense projection %s: %d -> %d",
            self._dense_projection_path,
            in_features,
            out_features,
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

        validate_output_types(output_types, {"dense"}, "Qwen2FlashAdapter")

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
            err_msg="Qwen2FlashAdapter requires text input",
        )

        # Tokenize each sequence individually (no padding)
        encodings = [
            self._tokenizer(
                text,
                max_length=self._max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            for text in texts
        ]

        # Build packed representation
        seq_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack input_ids
        input_ids_packed = torch.cat([enc["input_ids"].squeeze(0) for enc in encodings]).to(self._device)

        # Build cu_seqlens using cumsum (no Python loop, no GPU sync points)
        cu_seqlens = torch.zeros(len(texts) + 1, dtype=torch.int32, device=self._device)
        cu_seqlens[1:] = torch.cumsum(torch.tensor(seq_lengths, dtype=torch.int32, device=self._device), dim=0)

        with torch.inference_mode():
            # Build position IDs for RoPE
            position_ids_packed = self._build_position_ids(cu_seqlens, len(texts))

            # Get word embeddings (Qwen2 has no position embeddings - uses RoPE)
            hidden = self._model.embed_tokens(input_ids_packed)

            # Run transformer layers with flash attention
            hidden = self._run_transformer_flash(hidden, cu_seqlens, max_seqlen, total_tokens, position_ids_packed)

            # Final layer norm
            hidden = self._model.norm(hidden)

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

        Each sequence has positions starting from 0.  Fully vectorized
        to avoid per-sequence .item() CUDA sync points.
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
        rotary_emb: Any,
        position_ids: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin values for packed positions using layer's rotary_emb.

        Args:
            rotary_emb: The Qwen2RotaryEmbedding module from the attention layer.
            position_ids: Position IDs for all tokens [total_tokens].
            max_seqlen: Maximum sequence length in the batch.

        Returns:
            cos, sin tensors of shape [total_tokens, head_dim].
        """
        dtype = self._resolve_dtype()

        # Try new-style API first (transformers >= 4.45, Qwen3):
        # rotary_emb(x, position_ids) -> (cos, sin)
        if not hasattr(rotary_emb, "cos_cached"):
            dummy_x = torch.zeros(
                1,
                1,
                1,
                getattr(
                    self._model.config,
                    "head_dim",
                    self._model.config.hidden_size // self._model.config.num_attention_heads,
                ),
                device=self._device,
                dtype=dtype,
            )
            pos_ids = position_ids.unsqueeze(0)  # [1, total_tokens]
            cos, sin = rotary_emb(dummy_x, pos_ids)
            cos = cos.squeeze(0).to(dtype)  # [total_tokens, head_dim]
            sin = sin.squeeze(0).to(dtype)  # [total_tokens, head_dim]
            return cos, sin

        # Old-style API (transformers < 4.45, Qwen2):
        # rotary_emb(x, seq_len) populates cos_cached/sin_cached
        dummy_x = torch.zeros(1, 1, max_seqlen, 1, device=self._device, dtype=dtype)
        _ = rotary_emb(dummy_x, seq_len=max_seqlen)
        cos = rotary_emb.cos_cached[position_ids].to(dtype)  # [total_tokens, head_dim]
        sin = rotary_emb.sin_cached[position_ids].to(dtype)  # [total_tokens, head_dim]
        return cos, sin

    def _run_transformer_flash(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run transformer layers using flash_attn_varlen_func with RoPE.

        Qwen2 uses:
        - Pre-norm (RMSNorm before attention and MLP)
        - GQA (grouped query attention with fewer KV heads)
        - Separate Q/K/V projections
        """
        from flash_attn import flash_attn_varlen_func

        num_heads = self._model.config.num_attention_heads
        num_kv_heads = self._model.config.num_key_value_heads
        hidden_size = self._model.config.hidden_size
        head_dim = getattr(self._model.config, "head_dim", hidden_size // num_heads)
        softmax_scale = 1.0 / (head_dim**0.5)

        # Precompute RoPE once (Qwen3 stores rotary_emb at model level, Qwen2 per-layer)
        if hasattr(self._model, "rotary_emb"):
            rotary_emb = self._model.rotary_emb
        else:
            rotary_emb = self._model.layers[0].self_attn.rotary_emb
        cos, sin = self._compute_rope(rotary_emb, position_ids, max_seqlen)

        for layer in self._model.layers:
            attn = layer.self_attn

            # Pre-norm for attention
            normed_hidden = layer.input_layernorm(hidden)

            # Separate Q, K, V projections
            query = attn.q_proj(normed_hidden)  # [total_tokens, hidden_size]
            key = attn.k_proj(normed_hidden)  # [total_tokens, kv_hidden_size]
            value = attn.v_proj(normed_hidden)  # [total_tokens, kv_hidden_size]

            # Reshape for attention
            query = query.view(total_tokens, num_heads, head_dim)
            key = key.view(total_tokens, num_kv_heads, head_dim)
            value = value.view(total_tokens, num_kv_heads, head_dim)

            # Qwen3 QK-normalization (RMSNorm per-head, before RoPE)
            if hasattr(attn, "q_norm"):
                query = attn.q_norm(query)
            if hasattr(attn, "k_norm"):
                key = attn.k_norm(key)

            # Apply RoPE to Q and K
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            # Flash attention with variable-length sequences and GQA
            attn_out = flash_attn_varlen_func(
                query,
                key,
                value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=self._causal,
                softmax_scale=softmax_scale,
            )
            attn_out = attn_out.reshape(total_tokens, num_heads * head_dim)

            # Output projection
            attn_out = attn.o_proj(attn_out)

            # Residual connection
            hidden = hidden + attn_out

            # Pre-norm for MLP
            normed_hidden = layer.post_attention_layernorm(hidden)

            # MLP
            mlp_out = layer.mlp(normed_hidden)

            # Residual connection
            hidden = hidden + mlp_out

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
        """Pool hidden states to get sequence embeddings.

        Uses vectorized indexing for cls/last pooling to avoid per-sequence
        .item() CUDA sync points.  Mean pooling still uses a Python loop
        due to variable-length segment reduction.
        """
        normalize = normalize if normalize is not None else self._normalize
        pooling = pooling if pooling is not None else self._pooling

        if pooling == "cls":
            # Vectorized: extract first token of each sequence
            pooled = hidden[cu_seqlens[:-1].long()]
        elif pooling == "last":
            # Vectorized: extract last token of each sequence
            pooled = hidden[(cu_seqlens[1:] - 1).long()]
        else:  # mean pooling
            num_seqs = len(seq_lengths)
            mean_embeddings = []
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                mean_embeddings.append(hidden[start:end].mean(dim=0))
            pooled = torch.stack(mean_embeddings)

        if self._dense_projection is not None:
            pooled = self._dense_projection(pooled)

        if normalize:
            pooled = functional.normalize(pooled, p=2, dim=-1)

        return pooled
