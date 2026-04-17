from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.nn import functional

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._utils import apply_rotary_pos_emb, validate_output_types
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from transformers import PreTrainedTokenizerFast

# Runtime imports
import numpy as np

logger = logging.getLogger(__name__)

_ERR_CPU_NOT_SUPPORTED = "ColBERTModernBERTFlashAdapter requires CUDA for Flash Attention."


class ColBERTModernBERTFlashAdapter(PEFTLoRAMixin, FlashBaseAdapter):
    """ColBERT adapter for ModernBERT with RoPE and Flash Attention 2 varlen.

    This adapter eliminates padding waste by packing sequences and using
    flash_attn_varlen_func. Supports ModernBERT architecture with Rotary
    Position Embeddings and Matryoshka dimension truncation.

    Works with ModernBERT-based ColBERT models (GTE-ModernColBERT, Reason-ModernColBERT,
    mxbai-edge-colbert).
    """

    fallback_adapter_path: ClassVar[str | None] = "colbert:ColBERTAdapter"

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("multivector", "score"),
        unload_fields=("_model", "_tokenizer"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        token_dim: int = 128,
        normalize: bool = True,
        max_seq_length: int = 8192,
        query_max_length: int = 32,
        compute_precision: ComputePrecision = "bfloat16",
        skip_special_tokens: bool = True,
        query_prefix: str = "",
        doc_prefix: str = "",
        muvera_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            token_dim: Output dimension per token (Matryoshka truncation).
            normalize: Whether to L2-normalize token embeddings.
            max_seq_length: Maximum sequence length for documents.
            query_max_length: Maximum sequence length for queries.
            compute_precision: Compute precision (bfloat16 recommended).
            skip_special_tokens: Whether to exclude special tokens from output.
            query_prefix: Prefix to prepend to queries.
            doc_prefix: Prefix to prepend to documents.
            muvera_config: MUVERA configuration dict with keys like num_repetitions,
                num_simhash_projections, normalize. Used for FDE postprocessing.
            **kwargs: Additional arguments (ignored).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._token_dim = token_dim
        self._multivector_dim = token_dim
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._query_max_length = query_max_length
        self._compute_precision = compute_precision
        self._skip_special_tokens = skip_special_tokens
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._muvera_config = muvera_config

        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None

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
            "Loading ColBERT ModernBERT model %s on device=%s with dtype=%s",
            self._model_name_or_path,
            device,
            dtype,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=True,
        )

        # Load model with eager attention - we handle flash attention manually
        self._model = AutoModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=True,
        )
        self._model.to(device)
        self._model.eval()

        logger.info(
            "ColBERT ModernBERT: hidden=%d, token_dim=%d (Matryoshka truncation)",
            self._model.config.hidden_size,
            self._token_dim,
        )

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype (default: bfloat16)."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)

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
        """Run inference returning per-token embeddings.

        Args:
            items: List of items to encode.
            output_types: Which outputs to return (only "multivector" supported).
            instruction: Optional instruction prefix.
            is_query: Whether items are queries.
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with multivector embeddings.
        """
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        validate_output_types(output_types, {"multivector"}, "ColBERTModernBERTFlashAdapter")
        texts = self._extract_texts(items, instruction, is_query=is_query)

        max_length = self._query_max_length if is_query else self._max_seq_length

        # Tokenize each sequence individually (no padding)
        encodings = [
            self._tokenizer(
                text,
                max_length=max_length,
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
            position_ids_packed = self._build_position_ids(cu_seqlens, len(texts))

            # Run embeddings
            hidden = self._run_embeddings(input_ids_packed)

            # Get RoPE cos/sin values
            cos, sin = self._compute_rope(position_ids_packed)

            # Run transformer layers with flash attention and RoPE
            hidden = self._run_transformer_flash(hidden, cu_seqlens, max_seqlen, total_tokens, cos, sin)

            # Matryoshka truncation: take first token_dim dimensions
            hidden = hidden[:, : self._token_dim]

            # L2 normalize
            if self._normalize:
                hidden = functional.normalize(hidden, p=2, dim=-1)

            # Split back into per-item results
            multivectors = self._split_embeddings(hidden, cu_seqlens, seq_lengths, input_ids_packed)

        return EncodeOutput(
            multivector=multivectors,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._token_dim,
        )

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
    ) -> list[float]:
        """Score items against a query using MaxSim.

        Args:
            query: Query item.
            items: List of items to score.
            instruction: Optional instruction for the query.

        Returns:
            List of MaxSim scores, one per item.
        """
        self._check_loaded()

        # Encode query
        query_output = self.encode(
            [query],
            output_types=["multivector"],
            instruction=instruction,
            is_query=True,
        )
        query_vecs = query_output.multivector[0]

        # Encode documents
        doc_output = self.encode(
            items,
            output_types=["multivector"],
            is_query=False,
        )

        # Compute MaxSim for all documents in a single batched operation
        query_tensor = torch.from_numpy(query_vecs).to(self._device)

        doc_list = doc_output.multivector
        doc_lengths = [d.shape[0] for d in doc_list]
        max_doc_tokens = max(doc_lengths)
        dim = query_vecs.shape[1]

        docs_padded = torch.zeros(
            (len(doc_list), max_doc_tokens, dim),
            dtype=query_tensor.dtype,
            device=self._device,
        )
        for i, doc_vecs in enumerate(doc_list):
            t = torch.from_numpy(doc_vecs).to(self._device)
            docs_padded[i, : t.shape[0]] = t

        sim = torch.matmul(query_tensor, docs_padded.transpose(1, 2))

        lengths_t = torch.tensor(doc_lengths, device=self._device)
        mask = torch.arange(max_doc_tokens, device=self._device).unsqueeze(0) < lengths_t.unsqueeze(1)
        sim.masked_fill_(~mask.unsqueeze(1), float("-inf"))

        return sim.max(dim=-1).values.sum(dim=-1).tolist()

    def _build_position_ids(self, cu_seqlens: torch.Tensor, num_seqs: int) -> torch.Tensor:
        """Build position IDs for packed sequences."""
        pos_list = []
        for i in range(num_seqs):
            seq_len = int(cu_seqlens[i + 1].item() - cu_seqlens[i].item())
            pos_list.append(torch.arange(0, seq_len, device=self._device, dtype=torch.long))
        return torch.cat(pos_list)

    def _compute_rope(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin values for packed positions.

        Returns:
            cos, sin tensors of shape [total_tokens, head_dim].
        """
        head_dim = self._model.config.hidden_size // self._model.config.num_attention_heads

        # ModernBERT uses rope_theta for base frequency
        base = getattr(self._model.config, "rope_theta", 160000.0)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=self._device).float() / head_dim))

        # Compute sin/cos for all positions
        pos = position_ids.float()
        freqs = torch.outer(pos, inv_freq)  # [total_tokens, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [total_tokens, head_dim]

        cos = emb.cos()
        sin = emb.sin()

        return cos.to(self._resolve_dtype()), sin.to(self._resolve_dtype())

    def _run_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for packed input (no position embeddings - RoPE in attention)."""
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
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Run transformer layers using flash_attn_varlen_func with RoPE.

        ModernBERT uses pre-norm architecture with local/global attention patterns.
        We use global attention for all layers (no window restriction).
        """
        from flash_attn import flash_attn_varlen_func

        num_heads = self._model.config.num_attention_heads
        hidden_size = self._model.config.hidden_size
        head_dim = hidden_size // num_heads
        softmax_scale = 1.0 / (head_dim**0.5)

        for layer in self._model.layers:
            # Pre-attention norm (ModernBERT is pre-norm)
            normed_hidden = layer.attn_norm(hidden)

            # Fused QKV projection
            qkv = layer.attn.Wqkv(normed_hidden)
            qkv = qkv.view(total_tokens, 3, num_heads, head_dim)
            query = qkv[:, 0]  # [total_tokens, num_heads, head_dim]
            key = qkv[:, 1]
            value = qkv[:, 2]

            # Apply RoPE to Q and K
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

            # Flash attention with variable-length sequences
            # Note: We use global attention (no window_size) for simplicity
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
            attn_out = layer.attn.Wo(attn_out)

            # Residual connection
            hidden = hidden + attn_out

            # MLP block with pre-norm
            normed_hidden = layer.mlp_norm(hidden)
            mlp_out = layer.mlp(normed_hidden)
            hidden = hidden + mlp_out

        return hidden

    def _split_embeddings(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
        input_ids: torch.Tensor,
    ) -> list[np.ndarray]:
        """Split packed embeddings back into per-item arrays."""
        results = []
        num_seqs = len(seq_lengths)

        # Get special token IDs
        special_ids = set()
        if self._skip_special_tokens:
            if self._tokenizer.cls_token_id is not None:
                special_ids.add(self._tokenizer.cls_token_id)
            if self._tokenizer.sep_token_id is not None:
                special_ids.add(self._tokenizer.sep_token_id)
            if self._tokenizer.pad_token_id is not None:
                special_ids.add(self._tokenizer.pad_token_id)
            if self._tokenizer.bos_token_id is not None:
                special_ids.add(self._tokenizer.bos_token_id)
            if self._tokenizer.eos_token_id is not None:
                special_ids.add(self._tokenizer.eos_token_id)

        # Build a batched filter mask on GPU
        if self._skip_special_tokens and special_ids:
            filter_tensor = torch.tensor(sorted(special_ids), device=self._device)
            keep_mask_all = ~torch.isin(input_ids, filter_tensor)
        else:
            keep_mask_all = None

        # Transfer to CPU in bulk
        offsets = cu_seqlens.tolist()
        hidden_cpu = hidden.float().cpu().numpy()
        if keep_mask_all is not None:
            mask_cpu = keep_mask_all.cpu().numpy()

        for i in range(num_seqs):
            start, end = offsets[i], offsets[i + 1]
            seq_hidden = hidden_cpu[start:end]
            if keep_mask_all is not None:
                seq_hidden = seq_hidden[mask_cpu[start:end]]
            results.append(seq_hidden)

        return results

    def _extract_texts(self, items: list[Item], instruction: str | None, *, is_query: bool) -> list[str]:
        """Extract texts from items, applying prefixes."""
        texts = []
        prefix = self._query_prefix if is_query else self._doc_prefix

        for item in items:
            if item.text is None:
                raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="ColBERTModernBERTFlashAdapter"))

            text = item.text

            if instruction:
                text = f"{instruction} {text}"

            if prefix:
                text = f"{prefix}{text}"

            texts.append(text)

        return texts

    def get_postprocessors(self) -> dict[str, Any] | None:
        """Return MUVERA postprocessor for converting multivector to dense.

        Returns:
            Dict with "muvera" key mapping to MuveraPostprocessor instance.
        """
        from sie_server.core.postprocessor import MuveraConfig, MuveraPostprocessor

        # Build MuveraConfig from loadtime options or use defaults
        if self._muvera_config:
            config = MuveraConfig(**self._muvera_config)
        else:
            config = MuveraConfig()
        return {"muvera": MuveraPostprocessor(token_dim=self._token_dim, config=config)}
