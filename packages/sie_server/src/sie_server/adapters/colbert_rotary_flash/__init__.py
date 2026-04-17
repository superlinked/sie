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

_ERR_CPU_NOT_SUPPORTED = "ColBERTRotaryFlashAdapter requires CUDA for Flash Attention."


class ColBERTRotaryFlashAdapter(PEFTLoRAMixin, FlashBaseAdapter):
    """ColBERT adapter with RoPE and Flash Attention 2 variable-length sequences.

    This adapter eliminates padding waste by packing sequences and using
    flash_attn_varlen_func. Supports models with Rotary Position Embeddings
    and Matryoshka dimension truncation.

    Works with jina-colbert-v2 architecture (XLM-RoBERTa with flash + RoPE).
    """

    fallback_adapter_path: ClassVar[str | None] = "colbert:ColBERTAdapter"

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("multivector", "score"),
        unload_fields=("_model", "_tokenizer", "_expansion_token_id"),
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
        query_expansion: bool = True,
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
            query_expansion: Whether to pad queries with MASK tokens to query_max_length.
                This is a core ColBERT feature where MASK tokens become additional
                "virtual" query tokens. Default: True.
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
        self._query_expansion = query_expansion
        self._muvera_config = muvera_config

        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._expansion_token_id: int | None = None  # Set during load() if query_expansion

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
            "Loading ColBERT model %s on device=%s with dtype=%s (rotary flash varlen)",
            self._model_name_or_path,
            device,
            dtype,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=True,
        )

        # Configure tokenizer for query expansion (pad queries with MASK tokens)
        if self._query_expansion and self._tokenizer.mask_token_id is not None:
            self._expansion_token_id = self._tokenizer.mask_token_id
            logger.info("Query expansion enabled, using MASK token for padding")
        else:
            self._expansion_token_id = None
            if self._query_expansion:
                logger.warning("Query expansion requested but model has no MASK token, disabling")

        # Resolve prefix token IDs (for special tokens like [QueryMarker], [DocumentMarker])
        # These need to be inserted as token IDs, not as text
        self._query_prefix_id = self._resolve_prefix_token_id(self._query_prefix)
        self._doc_prefix_id = self._resolve_prefix_token_id(self._doc_prefix)
        if self._query_prefix_id:
            logger.info("Query prefix '%s' resolved to token ID %d", self._query_prefix.strip(), self._query_prefix_id)
        if self._doc_prefix_id:
            logger.info("Doc prefix '%s' resolved to token ID %d", self._doc_prefix.strip(), self._doc_prefix_id)

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
            "ColBERT rotary flash: hidden=%d, token_dim=%d (Matryoshka truncation)",
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

    def _resolve_prefix_token_id(self, prefix: str) -> int | None:
        """Resolve a prefix string to a single token ID if possible.

        Special tokens like [QueryMarker], [DocumentMarker] need to be inserted as token IDs,
        not as text (which would be split into multiple tokens).

        Args:
            prefix: Prefix string (e.g., "[QueryMarker] " or "query: ").

        Returns:
            Token ID if prefix is a single special token, None otherwise.
        """
        if not prefix or self._tokenizer is None:
            return None

        # Strip whitespace to get the token
        token = prefix.strip()
        if not token:
            return None

        # Check if this token exists as a single token in vocab
        # Use convert_tokens_to_ids which correctly handles vocab tokens
        # (encode() would split special tokens like [QueryMarker] into multiple pieces)
        if token in self._tokenizer.vocab:
            token_ids = self._tokenizer.convert_tokens_to_ids([token])
            if len(token_ids) == 1 and token_ids[0] != self._tokenizer.unk_token_id:
                return token_ids[0]

        return None

    def _insert_prefix_token(self, batch: dict[str, torch.Tensor], prefix_id: int) -> dict[str, torch.Tensor]:
        """Insert a prefix token ID at position 1 (after [CLS]) in the batch.

        This is used for special tokens like [QueryMarker] that need to be inserted
        as token IDs rather than as text.

        Args:
            batch: Tokenized batch with input_ids, attention_mask, etc.
            prefix_id: Token ID to insert.

        Returns:
            Modified batch with prefix token inserted.
        """
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]

        # Create prefix tensor
        prefix_tensor = torch.full((batch_size, 1), prefix_id, dtype=input_ids.dtype)

        # Insert at position 1 (after [CLS])
        new_input_ids = torch.cat(
            [input_ids[:, :1], prefix_tensor, input_ids[:, 1:]],
            dim=1,
        )
        batch["input_ids"] = new_input_ids

        # Update attention_mask
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"]
            prefix_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype)
            new_attention_mask = torch.cat(
                [attention_mask[:, :1], prefix_mask, attention_mask[:, 1:]],
                dim=1,
            )
            batch["attention_mask"] = new_attention_mask

        # Update token_type_ids if present
        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"]
            prefix_type = torch.zeros((batch_size, 1), dtype=token_type_ids.dtype)
            new_token_type_ids = torch.cat(
                [token_type_ids[:, :1], prefix_type, token_type_ids[:, 1:]],
                dim=1,
            )
            batch["token_type_ids"] = new_token_type_ids

        return batch

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

        validate_output_types(output_types, {"multivector"}, "ColBERTRotaryFlashAdapter")
        texts = self._extract_texts(items, instruction, is_query=is_query)

        max_length = self._query_max_length if is_query else self._max_seq_length

        # Determine if query expansion should be applied
        use_expansion = is_query and self._expansion_token_id is not None

        # Get prefix token ID (for special tokens like [QueryMarker])
        prefix_id = self._query_prefix_id if is_query else self._doc_prefix_id

        # Adjust max_length if we need to insert a prefix token
        effective_max_length = max_length - 1 if prefix_id is not None else max_length

        # For query expansion, pad to exact max_length using MASK tokens
        if use_expansion:
            original_pad_id = self._tokenizer.pad_token_id
            self._tokenizer.pad_token_id = self._expansion_token_id
            encodings = [
                self._tokenizer(
                    text,
                    max_length=effective_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                for text in texts
            ]
            self._tokenizer.pad_token_id = original_pad_id
        else:
            # Tokenize each sequence individually (no padding)
            encodings = [
                self._tokenizer(
                    text,
                    max_length=effective_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                for text in texts
            ]

        # Insert prefix token ID at position 1 (after [CLS]) for each encoding
        if prefix_id is not None:
            encodings = [self._insert_prefix_token(enc, prefix_id) for enc in encodings]

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

            # Run embeddings (no position embeddings - RoPE in attention)
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
            multivectors = self._split_embeddings(
                hidden, cu_seqlens, seq_lengths, input_ids_packed, keep_mask=use_expansion
            )

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
        # Compute cos/sin for positions
        # The rotary_emb expects [batch, seqlen] but we have packed positions
        # We need to use the internal computation
        head_dim = self._model.config.hidden_size // self._model.config.num_attention_heads

        # Compute frequencies
        base = getattr(self._model.config, "rotary_emb_base", 10000.0)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=self._device).float() / head_dim))

        # Compute sin/cos for all positions
        pos = position_ids.float()
        freqs = torch.outer(pos, inv_freq)  # [total_tokens, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [total_tokens, head_dim]

        cos = emb.cos()
        sin = emb.sin()

        return cos.to(self._resolve_dtype()), sin.to(self._resolve_dtype())

    def _run_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for packed input (no position embeddings - RoPE in attention).

        Uses word_embeddings + token_type_embeddings directly because:
        1. input_ids is 1D (packed sequences), but some custom embedding layers (e.g., Jina)
           expect 2D input (batch_size, seqlen) in their forward method
        2. Position embeddings are not needed - we use RoPE in attention instead
        """
        # Use word_embeddings directly to avoid custom embedding forward() that may expect 2D input
        hidden = self._model.embeddings.word_embeddings(input_ids)

        # Add token_type_embeddings (all zeros for single segment, but NOT zero values!)
        if hasattr(self._model.embeddings, "token_type_embeddings"):
            token_type_ids = torch.zeros_like(input_ids)
            hidden = hidden + self._model.embeddings.token_type_embeddings(token_type_ids)

        # Apply embedding dropout and layer norm if present
        if hasattr(self._model, "emb_drop"):
            hidden = self._model.emb_drop(hidden)
        if hasattr(self._model, "emb_ln"):
            hidden = self._model.emb_ln(hidden)

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
        """Run transformer layers using flash_attn_varlen_func with pure PyTorch RoPE.

        Uses our own apply_rotary_pos_emb instead of the native mixer.rotary_emb
        to avoid Triton JIT compilation which requires gcc/libcuda.so at runtime.
        """
        from flash_attn import flash_attn_varlen_func

        num_heads = self._model.config.num_attention_heads
        hidden_size = self._model.config.hidden_size
        head_dim = hidden_size // num_heads
        softmax_scale = 1.0 / (head_dim**0.5)

        for layer in self._model.encoder.layers:
            mixer = layer.mixer

            # Fused QKV projection
            # Some models (e.g., Jina) use LinearResidual which returns (output, input) tuple
            qkv = mixer.Wqkv(hidden)
            if isinstance(qkv, tuple):
                qkv = qkv[0]

            # Reshape and extract Q, K, V
            qkv = qkv.view(total_tokens, 3, num_heads, head_dim)
            query = qkv[:, 0]  # [total_tokens, num_heads, head_dim]
            key = qkv[:, 1]
            value = qkv[:, 2]

            # Apply RoPE using pure PyTorch (avoids Triton JIT compilation)
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
            attn_out = mixer.out_proj(attn_out)
            if isinstance(attn_out, tuple):
                attn_out = attn_out[0]

            # Residual + layer norm (post-norm pattern for jina-colbert-v2)
            hidden = hidden + attn_out
            if hasattr(layer, "norm1"):
                hidden = layer.norm1(hidden)

            # MLP
            # Some models (e.g., Jina) use Mlp that returns (output, input) tuple
            residual = hidden
            mlp_out = layer.mlp(hidden)
            if isinstance(mlp_out, tuple):
                mlp_out = mlp_out[0]
            hidden = residual + mlp_out
            if hasattr(layer, "norm2"):
                hidden = layer.norm2(hidden)

        return hidden

    def _split_embeddings(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
        input_ids: torch.Tensor,
        *,
        keep_mask: bool = False,
    ) -> list[np.ndarray]:
        """Split packed embeddings back into per-item arrays.

        Args:
            hidden: Hidden states tensor.
            cu_seqlens: Cumulative sequence lengths.
            seq_lengths: List of sequence lengths.
            input_ids: Input token IDs.
            keep_mask: If True, keep MASK tokens (for query expansion).
        """
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

            # For query expansion, keep MASK tokens (they're semantic expansion tokens)
            if keep_mask and self._tokenizer.mask_token_id in special_ids:
                special_ids.discard(self._tokenizer.mask_token_id)

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
        """Extract texts from items, applying prefixes.

        Note: If the prefix is a special token (resolved to a token ID), it won't be
        applied here - it will be inserted as a token ID during tokenization.
        """
        texts = []
        # Only use text prefix if we don't have a resolved token ID
        prefix_id = self._query_prefix_id if is_query else self._doc_prefix_id
        prefix = self._query_prefix if is_query else self._doc_prefix
        use_text_prefix = prefix and prefix_id is None

        for item in items:
            if item.text is None:
                raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="ColBERTRotaryFlashAdapter"))

            text = item.text

            if instruction:
                text = f"{instruction} {text}"

            # Apply text prefix only if we don't have a special token ID
            if use_text_prefix:
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
