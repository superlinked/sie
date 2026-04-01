"""Nomic BERT MoE Flash Attention adapter using flash_attn_varlen_func.

This adapter uses Flash Attention 2's variable-length attention to process
sequences without padding, eliminating padding waste and improving throughput.

Supports nomic-ai/nomic-embed-text-v2-moe model:
- 12 layers, 768 hidden, 12 heads
- MoE on every 2nd layer (layers 1,3,5,7,9,11) with 8 experts, top-2 routing
- Regular MLP on even layers (0,2,4,6,8,10)
- RoPE positional embeddings
- Task prefixes: search_query: / search_document:

Key features:
- Uses flash_attn_varlen_func with cu_seqlens for packed sequences
- Applies Rotary Position Embeddings (RoPE) to Q and K
- Implements MoE routing in pure PyTorch (no megablocks dependency)
- No padding tokens = no wasted compute

See: https://github.com/Dao-AILab/flash-attention
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from sie_server.adapters.base import ModelAdapter, ModelCapabilities, ModelDims
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.preprocessor import CharCountPreprocessor
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]
PoolingStrategy = Literal["cls", "mean"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "NomicFlashAdapter requires text input"
_ERR_CPU_NOT_SUPPORTED = "NomicFlashAdapter requires CUDA. Use pytorch_embedding adapter for CPU."


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors.

    Args:
        q: Query tensor [total_tokens, num_heads, head_dim].
        k: Key tensor [total_tokens, num_heads, head_dim].
        cos: Cosine part [total_tokens, head_dim].
        sin: Sine part [total_tokens, head_dim].

    Returns:
        Rotated query and key tensors.
    """
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class NomicFlashAdapter(PEFTLoRAMixin, ModelAdapter):
    """Nomic BERT MoE adapter using Flash Attention 2 with variable-length sequences.

    This adapter eliminates padding waste by packing sequences and using
    flash_attn_varlen_func. Implements MoE routing in pure PyTorch.
    """

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        normalize: bool = True,
        max_seq_length: int = 2048,
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
            query_template: Template for queries, e.g. "search_query: {text}".
            doc_template: Template for documents, e.g. "search_document: {text}".
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._pooling = pooling
        self._query_template = query_template or "search_query: {text}"
        self._doc_template = doc_template or "search_document: {text}"

        # Model components (loaded in load())
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None
        self._dense_dim: int | None = None

        # Model weights
        self._word_embeddings: torch.Tensor | None = None
        self._token_type_embeddings: torch.Tensor | None = None
        self._emb_ln_weight: torch.Tensor | None = None
        self._emb_ln_bias: torch.Tensor | None = None
        self._layers: list[dict[str, torch.Tensor]] | None = None

        # Config
        self._num_heads: int = 12
        self._head_dim: int = 64  # 768 / 12
        self._hidden_size: int = 768
        self._intermediate_size: int = 3072
        self._num_experts: int = 8
        self._moe_top_k: int = 2
        self._rotary_base: float = 10000.0

    @classmethod
    def create_for_device(cls, device: str, **kwargs: Any) -> ModelAdapter:
        """Factory method that returns the appropriate adapter for the device.

        For non-CUDA devices or when flash-attn is unavailable, returns SentenceTransformerDenseAdapter.

        Args:
            device: Device string (e.g., "cuda:0", "mps", "cpu").
            **kwargs: Adapter initialization parameters.

        Returns:
            NomicFlashAdapter for CUDA with flash-attn, SentenceTransformerDenseAdapter otherwise.
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

        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        from transformers import AutoTokenizer

        self._device = device
        self._dtype = self._resolve_dtype()

        logger.info(
            "Loading %s on device=%s with dtype=%s, attn=nomic_flash_varlen, pooling=%s",
            self._model_name_or_path,
            device,
            self._dtype,
            self._pooling,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)

        # Download and load weights
        model_path = hf_hub_download(self._model_name_or_path, "model.safetensors")
        self._load_weights(model_path)

        self._dense_dim = self._hidden_size
        logger.info("Nomic model loaded: %d layers, %d hidden", 12, self._hidden_size)

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.float16)

    def _load_weights(self, model_path: str) -> None:
        """Load model weights from safetensors file."""
        from safetensors import safe_open

        with safe_open(model_path, framework="pt", device=self._device) as f:
            # Embedding weights
            self._word_embeddings = f.get_tensor("embeddings.word_embeddings.weight").to(self._dtype)
            self._token_type_embeddings = f.get_tensor("embeddings.token_type_embeddings.weight").to(self._dtype)
            self._emb_ln_weight = f.get_tensor("emb_ln.weight").to(self._dtype)
            self._emb_ln_bias = f.get_tensor("emb_ln.bias").to(self._dtype)

            # Load all 12 layers
            self._layers = []
            for i in range(12):
                layer = self._load_layer_weights(f, i)
                self._layers.append(layer)

    def _load_layer_weights(self, f: Any, layer_idx: int) -> dict[str, torch.Tensor]:
        """Load weights for a single transformer layer."""
        prefix = f"encoder.layers.{layer_idx}"
        is_moe = layer_idx % 2 == 1  # MoE on odd layers

        layer = {
            # Attention weights
            "Wqkv_weight": f.get_tensor(f"{prefix}.attn.Wqkv.weight").to(self._dtype),
            "Wqkv_bias": f.get_tensor(f"{prefix}.attn.Wqkv.bias").to(self._dtype),
            "out_proj_weight": f.get_tensor(f"{prefix}.attn.out_proj.weight").to(self._dtype),
            "out_proj_bias": f.get_tensor(f"{prefix}.attn.out_proj.bias").to(self._dtype),
            # Layer norms
            "norm1_weight": f.get_tensor(f"{prefix}.norm1.weight").to(self._dtype),
            "norm1_bias": f.get_tensor(f"{prefix}.norm1.bias").to(self._dtype),
            "norm2_weight": f.get_tensor(f"{prefix}.norm2.weight").to(self._dtype),
            "norm2_bias": f.get_tensor(f"{prefix}.norm2.bias").to(self._dtype),
            "is_moe": is_moe,
        }

        if is_moe:
            # MoE weights
            layer["router_weight"] = f.get_tensor(f"{prefix}.mlp.router.layer.weight").to(self._dtype)
            layer["experts_w1"] = f.get_tensor(f"{prefix}.mlp.experts.mlp.w1").to(self._dtype)
            layer["experts_w2"] = f.get_tensor(f"{prefix}.mlp.experts.mlp.w2").to(self._dtype)
            layer["experts_bias"] = f.get_tensor(f"{prefix}.mlp.experts.bias").to(self._dtype)
        else:
            # Regular MLP weights
            layer["fc1_weight"] = f.get_tensor(f"{prefix}.mlp.fc1.weight").to(self._dtype)
            layer["fc1_bias"] = f.get_tensor(f"{prefix}.mlp.fc1.bias").to(self._dtype)
            layer["fc2_weight"] = f.get_tensor(f"{prefix}.mlp.fc2.weight").to(self._dtype)
            layer["fc2_bias"] = f.get_tensor(f"{prefix}.mlp.fc2.bias").to(self._dtype)

        return layer

    def unload(self) -> None:
        """Unload the model and free resources."""
        device = self._device

        # Clear all weight tensors
        self._word_embeddings = None
        self._token_type_embeddings = None
        self._emb_ln_weight = None
        self._emb_ln_bias = None
        self._layers = None
        self._tokenizer = None
        self._device = None
        self._dense_dim = None

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
            instruction: Optional instruction prefix (unused, template-based).
            is_query: Whether items are queries (affects template selection).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with dense embeddings.
        """
        if self._tokenizer is None or self._layers is None:
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

        # Build cu_seqlens (cumulative sequence lengths)
        cu_seqlens = torch.zeros(len(texts) + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            # Build position IDs for RoPE
            position_ids = self._build_position_ids(cu_seqlens, len(texts))

            # Compute RoPE cos/sin
            cos, sin = self._compute_rope(position_ids)

            # Run embeddings
            hidden = self._run_embeddings(input_ids_packed)

            # Run transformer layers
            hidden = self._run_transformer(hidden, cu_seqlens, max_seqlen, total_tokens, cos, sin)

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
        """Build position IDs for packed sequences (starting from 0 for each)."""
        pos_list = []
        for i in range(num_seqs):
            seq_len = cu_seqlens[i + 1].item() - cu_seqlens[i].item()
            pos_list.append(torch.arange(0, seq_len, device=self._device))
        return torch.cat(pos_list)

    def _compute_rope(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin values for packed positions.

        The nomic model uses non-interleaved RoPE where cos/sin are computed on
        (seqlen, rotary_dim/2) then concatenated to (seqlen, rotary_dim).

        Returns:
            cos, sin tensors of shape [total_tokens, head_dim].
        """
        # Compute inverse frequencies (half of head_dim)
        rotary_dim = self._head_dim
        inv_freq = 1.0 / (
            self._rotary_base ** (torch.arange(0, rotary_dim, 2, device=self._device, dtype=torch.float32) / rotary_dim)
        )

        # Compute cos/sin for each position - shape (seqlen, rotary_dim/2)
        freqs = torch.outer(position_ids.float(), inv_freq)
        cos_half = freqs.cos().to(self._dtype)  # (seqlen, 32)
        sin_half = freqs.sin().to(self._dtype)  # (seqlen, 32)

        # Concatenate to full head_dim: (seqlen, 32) -> (seqlen, 64)
        # Pattern: [c0, c1, ..., c31, c0, c1, ..., c31]  # layout explanation
        cos = torch.cat([cos_half, cos_half], dim=-1)
        sin = torch.cat([sin_half, sin_half], dim=-1)

        return cos, sin

    def _run_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for packed input."""
        # Word embeddings
        hidden = F.embedding(input_ids, self._word_embeddings)

        # Token type embeddings (all zeros for this model)
        token_type_ids = torch.zeros_like(input_ids)
        hidden = hidden + F.embedding(token_type_ids, self._token_type_embeddings)

        # Embedding layer norm
        hidden = F.layer_norm(
            hidden,
            [self._hidden_size],
            weight=self._emb_ln_weight,
            bias=self._emb_ln_bias,
        )

        return hidden

    def _run_transformer(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run all transformer layers."""
        for layer_idx, layer in enumerate(self._layers):
            hidden = self._run_layer(hidden, layer, cu_seqlens, max_seqlen, total_tokens, cos, sin)
        return hidden

    def _run_layer(
        self,
        hidden: torch.Tensor,
        layer: dict[str, torch.Tensor],
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run a single transformer layer (post-norm style)."""
        from flash_attn import flash_attn_varlen_func

        # Self-attention
        # QKV projection (fused)
        qkv = F.linear(hidden, layer["Wqkv_weight"], layer["Wqkv_bias"])
        qkv = qkv.view(total_tokens, 3, self._num_heads, self._head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Flash attention
        softmax_scale = 1.0 / (self._head_dim**0.5)
        attn_out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=False,
            softmax_scale=softmax_scale,
        )
        attn_out = attn_out.reshape(total_tokens, self._hidden_size)

        # Output projection
        attn_out = F.linear(attn_out, layer["out_proj_weight"], layer["out_proj_bias"])

        # Residual + post-norm
        hidden = hidden + attn_out
        hidden = F.layer_norm(
            hidden,
            [self._hidden_size],
            weight=layer["norm1_weight"],
            bias=layer["norm1_bias"],
        )

        # MLP or MoE
        if layer["is_moe"]:
            mlp_out = self._run_moe(hidden, layer)
        else:
            mlp_out = self._run_mlp(hidden, layer)

        # Residual + post-norm
        hidden = hidden + mlp_out
        hidden = F.layer_norm(
            hidden,
            [self._hidden_size],
            weight=layer["norm2_weight"],
            bias=layer["norm2_bias"],
        )

        return hidden

    def _run_mlp(self, hidden: torch.Tensor, layer: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run regular MLP layer."""
        # Up projection
        out = F.linear(hidden, layer["fc1_weight"], layer["fc1_bias"])
        # GELU activation
        out = F.gelu(out)
        # Down projection
        out = F.linear(out, layer["fc2_weight"], layer["fc2_bias"])
        return out

    def _run_moe(self, hidden: torch.Tensor, layer: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run MoE layer with top-k routing.

        Uses sorted-expert dispatch: flatten all top-k assignments, sort by
        expert ID so each expert gets a contiguous slice, then scatter-add
        weighted results back. This avoids expensive boolean masking and
        fancy indexing per expert.
        """
        total_tokens = hidden.shape[0]

        # Router: compute expert scores [total_tokens, num_experts]
        router_logits = F.linear(hidden, layer["router_weight"])
        router_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32).to(hidden.dtype)

        # Select top-k experts per token
        top_weights, top_indices = torch.topk(router_weights, self._moe_top_k, dim=-1)

        # Get expert weights
        w1 = layer["experts_w1"].view(self._num_experts, self._intermediate_size, self._hidden_size)
        w2 = layer["experts_w2"].view(self._num_experts, self._intermediate_size, self._hidden_size)
        bias = layer["experts_bias"]

        # Flatten top-k: each token appears top_k times
        flat_expert_idx = top_indices.view(-1)  # [N * top_k]
        flat_weights = top_weights.view(-1, 1)  # [N * top_k, 1]
        flat_token_idx = torch.arange(total_tokens, device=hidden.device).repeat_interleave(self._moe_top_k)
        flat_hidden = hidden[flat_token_idx]  # [N * top_k, hidden]

        # Sort by expert for contiguous GEMM slices
        sorted_order = flat_expert_idx.argsort()
        sorted_expert = flat_expert_idx[sorted_order]
        sorted_hidden = flat_hidden[sorted_order]
        sorted_weights = flat_weights[sorted_order]
        sorted_token_idx = flat_token_idx[sorted_order]

        # Find expert boundaries
        expert_counts = torch.zeros(self._num_experts, dtype=torch.long, device=hidden.device)
        expert_counts.scatter_add_(0, sorted_expert.long(), torch.ones_like(sorted_expert, dtype=torch.long))
        expert_offsets = torch.zeros(self._num_experts + 1, dtype=torch.long, device=hidden.device)
        expert_offsets[1:] = expert_counts.cumsum(0)

        # Process each expert on its contiguous slice
        all_outputs = torch.empty_like(sorted_hidden)
        for e in range(self._num_experts):
            start_e = expert_offsets[e].item()
            end_e = expert_offsets[e + 1].item()
            if start_e == end_e:
                continue
            eh = sorted_hidden[start_e:end_e]
            up = F.linear(eh, w1[e])
            up = F.gelu(up)
            all_outputs[start_e:end_e] = up.matmul(w2[e])

        # Apply routing weights and scatter-add back to token positions
        weighted = all_outputs * sorted_weights
        output = torch.zeros(total_tokens, self._hidden_size, device=hidden.device, dtype=hidden.dtype)
        output.scatter_add_(0, sorted_token_idx.unsqueeze(-1).expand_as(weighted), weighted)

        return output + bias

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
            cls_embeddings = []
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                cls_embeddings.append(hidden[start])
            pooled = torch.stack(cls_embeddings)
        else:  # mean pooling
            mean_embeddings = []
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                mean_embeddings.append(hidden[start:end].mean(dim=0))
            pooled = torch.stack(mean_embeddings)

        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        return pooled

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"dense"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. NomicFlashAdapter only supports 'dense'."
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
        """Extract texts from items, applying task prefixes."""
        query_template = query_template if query_template is not None else self._query_template
        doc_template = doc_template if doc_template is not None else self._doc_template
        texts = []
        for item in items:
            if item.text is None:
                raise ValueError(_ERR_REQUIRES_TEXT)

            text = item.text

            # Apply task prefix template
            template = query_template if is_query else doc_template
            text = template.format(text=text, instruction=instruction or "")

            texts.append(text)
        return texts

    def get_preprocessor(self) -> CharCountPreprocessor:
        """Return CharCountPreprocessor for cost estimation without tokenization overhead."""
        return CharCountPreprocessor(model_name=self._model_name_or_path)
