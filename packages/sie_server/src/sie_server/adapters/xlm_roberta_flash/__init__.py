"""XLMRoberta Flash Attention adapter using flash_attn_varlen_func.

This adapter uses Flash Attention 2's variable-length attention to process
sequences without padding, eliminating padding waste and improving throughput.

Supports XLMRoberta-based models like:
- Alibaba-NLP/gte-multilingual-base
- intfloat/multilingual-e5-large(-instruct)

Key features:
- Uses flash_attn_varlen_func with cu_seqlens for packed sequences
- No padding tokens = no wasted compute
- Position IDs correctly offset for XLMRoberta (start at padding_idx + 1 = 2)

See: https://github.com/Dao-AILab/flash-attention
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
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
    from transformers import PreTrainedTokenizerFast, XLMRobertaModel

logger = logging.getLogger(__name__)

ComputePrecision = Literal["float16", "bfloat16", "float32"]
PoolingStrategy = Literal["cls", "mean"]

_ERR_NOT_LOADED = "Model not loaded. Call load() first."
_ERR_REQUIRES_TEXT = "XLMRobertaFlashAdapter requires text input"


class XLMRobertaFlashAdapter(PEFTLoRAMixin, ModelAdapter):
    """XLMRoberta adapter using Flash Attention 2 with variable-length sequences.

    This adapter eliminates padding waste by packing sequences and using
    flash_attn_varlen_func. Achieves higher throughput than SDPA-based adapters.

    Works with any XLMRoberta-based model for dense embeddings.
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

        self._model: XLMRobertaModel | None = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._padding_idx: int = 1  # XLMRoberta default, set properly in load()
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
        if self._dense_dim is None:
            raise RuntimeError(_ERR_NOT_LOADED)
        return ModelDims(dense=self._dense_dim)

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (must be "cuda" or "cuda:X").

        """
        from transformers import AutoModel, AutoTokenizer

        self._device = device
        dtype = self._resolve_dtype()

        self._use_flash = device.startswith("cuda")

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
            trust_remote_code=True,
        )
        self._model.to(device)
        self._model.eval()

        # Get padding_idx for XLMRoberta position ID calculation
        self._padding_idx = self._model.embeddings.padding_idx
        self._dense_dim = self._model.config.hidden_size
        logger.debug("XLMRoberta padding_idx: %d, hidden_size: %d", self._padding_idx, self._dense_dim)

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
            # Build XLMRoberta-style position IDs (start at padding_idx + 1)
            position_ids_packed = self._build_position_ids(cu_seqlens, len(texts))

            # Run embeddings
            hidden = self._run_embeddings(input_ids_packed, position_ids_packed)

            # Run transformer layers with flash attention
            if self._use_flash:
                hidden = self._run_transformer_flash(hidden, cu_seqlens, max_seqlen, total_tokens)
            else:
                hidden = self._run_transformer_standard(hidden, cu_seqlens, max_seqlen, total_tokens)

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
        """Build XLMRoberta-style position IDs for packed sequences.

        XLMRoberta uses position IDs starting at padding_idx + 1, not 0.
        This is critical for matching the padded model's output.
        """
        pos_list = []
        for i in range(num_seqs):
            seq_len = cu_seqlens[i + 1].item() - cu_seqlens[i].item()
            # XLMRoberta positions: [padding_idx+1, padding_idx+2, ..., padding_idx+seq_len]
            pos_list.append(
                torch.arange(
                    self._padding_idx + 1,
                    self._padding_idx + 1 + seq_len,
                    device=self._device,
                )
            )
        return torch.cat(pos_list)

    def _run_embeddings(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for packed input."""
        embeddings = self._model.embeddings  # type: ignore[union-attr]

        word_emb = embeddings.word_embeddings(input_ids)
        pos_emb = embeddings.position_embeddings(position_ids)
        token_type_emb = embeddings.token_type_embeddings(torch.zeros_like(input_ids))

        hidden = word_emb + pos_emb + token_type_emb
        hidden = embeddings.LayerNorm(hidden)
        hidden = embeddings.dropout(hidden)

        return hidden

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

        for layer in self._model.encoder.layer:  # type: ignore[union-attr]
            attention = layer.attention.self

            # QKV projections
            query = attention.query(hidden).view(total_tokens, num_heads, head_dim)
            key = attention.key(hidden).view(total_tokens, num_heads, head_dim)
            value = attention.value(hidden).view(total_tokens, num_heads, head_dim)

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
            attn_out = layer.attention.output.dense(attn_out)
            attn_out = layer.attention.output.dropout(attn_out)
            hidden = layer.attention.output.LayerNorm(attn_out + hidden)

            # FFN
            inter = layer.intermediate.dense(hidden)
            inter = layer.intermediate.intermediate_act_fn(inter)
            out = layer.output.dense(inter)
            out = layer.output.dropout(out)
            hidden = layer.output.LayerNorm(out + hidden)

        return hidden

    def _run_transformer_standard(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
    ) -> torch.Tensor:
        """Run transformer layers using standard PyTorch SDPA (CPU/MPS compatible).

        Expects `hidden` to be packed as shape [total_tokens, hidden_size] and uses
        `cu_seqlens` to iterate per-sequence slices (no padding required).

        Notes:
        - Uses torch.nn.functional.scaled_dot_product_attention
        - Produces attention output equivalent to non-causal self-attention
        - Handles dropout via the module dropout layers (p=0 in SDPA when eval)
        """
        import torch.nn.functional as F

        if self._model is None:
            raise RuntimeError(_ERR_NOT_LOADED)

        model = self._model
        num_heads = model.config.num_attention_heads  # type: ignore[union-attr]
        hidden_size = model.config.hidden_size  # type: ignore[union-attr]
        head_dim = hidden_size // num_heads

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads}).")

        # Ensure cu_seqlens is on CPU for cheap .item() access; it's tiny.
        # (If you prefer keeping on device, remove this and keep .item() calls.)
        if cu_seqlens.is_cuda:
            cu_seqlens_cpu = cu_seqlens.detach().to("cpu")
        else:
            cu_seqlens_cpu = cu_seqlens

        for layer in model.encoder.layer:  # type: ignore[union-attr]
            attn_self = layer.attention.self

            # QKV projections on packed hidden
            q = attn_self.query(hidden).view(total_tokens, num_heads, head_dim)
            k = attn_self.key(hidden).view(total_tokens, num_heads, head_dim)
            v = attn_self.value(hidden).view(total_tokens, num_heads, head_dim)

            # SDPA expects [B, H, L, D]; we run per sequence with B=1 to avoid padding.
            attn_out_chunks: list[torch.Tensor] = []
            for i in range(cu_seqlens_cpu.numel() - 1):
                start = int(cu_seqlens_cpu[i].item())
                end = int(cu_seqlens_cpu[i + 1].item())
                if end <= start:
                    continue  # defensive: skip empty

                qs = q[start:end].transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
                ks = k[start:end].transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
                vs = v[start:end].transpose(0, 1).unsqueeze(0)  # [1, H, L, D]

                # No mask; non-causal self-attention
                # dropout_p must be 0.0 in eval; in train you could pass attn_self.dropout.p
                dropout_p = 0.0 if not model.training else float(getattr(attn_self, "dropout", 0.0))
                out = F.scaled_dot_product_attention(
                    qs,
                    ks,
                    vs,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=False,
                )  # [1, H, L, D]

                out = out.squeeze(0).transpose(0, 1).contiguous()  # [L, H, D]
                attn_out_chunks.append(out)

            attn_out = torch.cat(attn_out_chunks, dim=0).view(total_tokens, hidden_size)

            # Output projection + residual + LN (mirrors HF)
            attn_out = layer.attention.output.dense(attn_out)
            attn_out = layer.attention.output.dropout(attn_out)
            hidden = layer.attention.output.LayerNorm(attn_out + hidden)

            # FFN block (mirrors HF)
            inter = layer.intermediate.dense(hidden)
            inter = layer.intermediate.intermediate_act_fn(inter)
            out = layer.output.dense(inter)
            out = layer.output.dropout(out)
            hidden = layer.output.LayerNorm(out + hidden)

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
            # Average all tokens (excluding padding - but we have none!)
            mean_embeddings = []
            for i in range(num_seqs):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                mean_embeddings.append(hidden[start:end].mean(dim=0))
            pooled = torch.stack(mean_embeddings)

        if normalize:
            pooled = functional.normalize(pooled, p=2, dim=-1)

        return pooled

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"dense"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. XLMRobertaFlashAdapter only supports 'dense'."
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
