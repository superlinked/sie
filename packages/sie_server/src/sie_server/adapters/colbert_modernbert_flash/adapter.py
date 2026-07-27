from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch
from torch.nn import functional

from sie_server.adapters._colbert_utils import punctuation_token_ids
from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._flash_pack import build_position_ids
from sie_server.adapters._multivector import maxsim_scores_batched
from sie_server.adapters._pylate_dense import apply_dense_chain, load_pylate_dense_chain
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._utils import (
    apply_rotary_pos_emb,
    grouped_score_pairs,
    validate_output_types,
)
from sie_server.adapters.peft_lora_mixin import PEFTLoRAMixin
from sie_server.core.inference_output import EncodeOutput, ScoreOutput
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_ERR_CPU_NOT_SUPPORTED = "ColBERTModernBERTFlashAdapter requires CUDA for Flash Attention."


class ColBERTModernBERTFlashAdapter(PEFTLoRAMixin, FlashBaseAdapter):
    """ColBERT adapter for ModernBERT with RoPE and Flash Attention 2 varlen.

    This adapter eliminates padding waste by packing sequences and using
    flash_attn_varlen_func. Mirrors the ModernBERT forward faithfully:
    local/global attention alternation (per-layer sliding window + rope theta),
    final_norm, and the trained pylate Dense chain when the checkpoint ships
    one (falling back to Matryoshka truncation otherwise). See #1680.

    Works with ModernBERT-based ColBERT models (GTE-ModernColBERT, Reason-ModernColBERT,
    mxbai-edge-colbert).
    """

    fallback_adapter_path: ClassVar[str | None] = "colbert:ColBERTAdapter"
    # Align the fallback with this adapter's faithful serving of the ModernBERT
    # ColBERT trio: no query expansion (their checkpoints set
    # do_query_expansion: false) and the flash default max_seq_length
    # (ColBERTAdapter defaults to 512). Applies ONLY to fallback instances created for models
    # configured with this adapter, never to models whose YAML names
    # colbert:ColBERTAdapter directly.
    fallback_kwargs_overrides: ClassVar[dict[str, Any]] = {
        "query_expansion": False,
        "max_seq_length": 8192,
    }

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("multivector", "score"),
        unload_fields=("_model", "_tokenizer", "_dense_chain"),
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
        doc_punctuation_skiplist: bool = False,
        query_prefix: str = "",
        doc_prefix: str = "",
        muvera_config: dict[str, Any] | None = None,
        revision: str | None = None,
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
            doc_punctuation_skiplist: Whether to remove punctuation token
                embeddings from documents before MaxSim scoring.
            query_prefix: Prefix to prepend to queries.
            doc_prefix: Prefix to prepend to documents.
            muvera_config: MUVERA configuration dict with keys like num_repetitions,
                num_simhash_projections, normalize. Used for FDE postprocessing.
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading the tokenizer, model, and Dense-chain artifacts.
                Forwarded to ``from_pretrained(..., revision=...)``.
            **kwargs: Additional arguments (ignored).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._revision = revision
        self._token_dim = token_dim
        self._multivector_dim = token_dim
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._query_max_length = query_max_length
        self._compute_precision = compute_precision
        self._skip_special_tokens = skip_special_tokens
        self._doc_punctuation_skiplist = doc_punctuation_skiplist
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._muvera_config = muvera_config

        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dense_chain: list[torch.Tensor] | None = None
        self._doc_skiplist_ids: set[int] = set()

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (must be "cuda" or "cuda:X").

        Raises:
            RuntimeError: If device is not CUDA.
        """
        if not device.startswith("cuda"):
            raise RuntimeError(_ERR_CPU_NOT_SUPPORTED)

        from transformers import AutoModel

        self._device = device
        dtype = self._resolve_dtype()

        logger.info(
            "Loading ColBERT ModernBERT model %s on device=%s with dtype=%s",
            self._model_name_or_path,
            device,
            dtype,
        )

        self._tokenizer = self._load_tokenizer()

        if self._doc_punctuation_skiplist:
            self._doc_skiplist_ids = punctuation_token_ids(self._tokenizer)

        # Load model with eager attention - we handle flash attention manually
        self._model = AutoModel.from_pretrained(
            self._model_name_or_path,
            torch_dtype=dtype,
            attn_implementation="eager",
            trust_remote_code=False,
            revision=self._revision,
        )
        self._model.to(device)
        self._model.eval()

        self._dense_chain = load_pylate_dense_chain(
            self._model_name_or_path,
            hidden_size=self._model.config.hidden_size,
            token_dim=self._token_dim,
            device=self._device,
            dtype=dtype,
            revision=self._revision,
        )
        if self._dense_chain is None:
            logger.warning(
                "No usable pylate Dense chain for %s; using backbone truncation. "
                "If this checkpoint ships Dense modules (e.g. partial weight cache), embeddings are degraded.",
                self._model_name_or_path,
            )

        logger.info(
            "ColBERT ModernBERT: hidden=%d, token_dim=%d, dense_chain=%s",
            self._model.config.hidden_size,
            self._token_dim,
            None if self._dense_chain is None else [tuple(w.shape) for w in self._dense_chain],
        )

    def _load_tokenizer(self) -> PreTrainedTokenizerFast:
        """Load the tokenizer, tolerating checkpoints saved by newer transformers.

        Some ModernBERT ColBERT checkpoints (e.g. topk-io/Iso-ModernColBERT) are
        saved by transformers>=5 and declare a tokenizer_class this transformers
        version cannot resolve (e.g. "TokenizersBackend"). These ship a standard
        fast tokenizer, so fall back to PreTrainedTokenizerFast, which is
        byte-equivalent to AutoTokenizer for the fast-tokenizer checkpoints this
        adapter serves.
        """
        from transformers import AutoTokenizer, PreTrainedTokenizerFast

        try:
            return AutoTokenizer.from_pretrained(
                self._model_name_or_path,
                trust_remote_code=False,
                revision=self._revision,
            )
        except (ValueError, KeyError) as exc:
            logger.warning(
                "AutoTokenizer could not resolve the tokenizer class for %s (%s); "
                "falling back to PreTrainedTokenizerFast",
                self._model_name_or_path,
                exc,
            )
            return PreTrainedTokenizerFast.from_pretrained(self._model_name_or_path, revision=self._revision)

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype (default: bfloat16)."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the trained pylate Dense chain if available, then Matryoshka-truncate
        to token_dim. Dimension-preserving: output is always token_dim wide.

        The trailing truncation is the whole projection when the chain is absent
        and a no-op when it is present (the chain loader guarantees the chain
        ends at token_dim).
        """
        if self._dense_chain is not None:
            hidden = apply_dense_chain(hidden, self._dense_chain)
        return hidden[:, : self._token_dim]

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
        if not texts:
            return EncodeOutput(
                multivector=[],
                batch_size=0,
                is_query=is_query,
                multivector_token_dim=self._token_dim,
                extra={"input_token_counts": []},
            )

        opts = options or {}
        if is_query:
            max_length = self._coerce_runtime_max_length(
                opts.get("query_max_length"),
                self._query_max_length,
            )
        else:
            max_length = self._coerce_runtime_max_length(
                opts.get("max_seq_length"),
                self._max_seq_length,
            )

        encodings = self._tokenize_inputs(texts, max_length)

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

            # Apply final layer norm (ModernBERT has a final_norm after all layers);
            # the trained Dense projection was fit on final-normed activations.
            if hasattr(self._model, "final_norm"):
                hidden = self._model.final_norm(hidden)

            # Apply the trained pylate Dense chain (if the checkpoint ships one)
            # then Matryoshka-truncate to token_dim; see #1680.
            hidden = self._project(hidden)

            # L2 normalize
            if self._normalize:
                hidden = functional.normalize(hidden, p=2, dim=-1)

            # Split back into per-item results
            multivectors = self._split_embeddings(
                hidden,
                cu_seqlens,
                seq_lengths,
                input_ids_packed,
                is_query=is_query,
            )

        return EncodeOutput(
            multivector=multivectors,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._token_dim,
            extra={"input_token_counts": [int(length) for length in seq_lengths]},
        )

    def _tokenize_inputs(self, texts: list[str], max_length: int) -> list[Any]:
        """Tokenize packed inputs without racing metering tokenization."""
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)
        with self._tokenizer_guard():
            return [
                self._tokenizer(
                    text,
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                for text in texts
            ]

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
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
            options=options,
        )
        query_vecs = query_output.multivector[0]

        # Encode documents
        doc_output = self.encode(
            items,
            output_types=["multivector"],
            is_query=False,
            options=options,
        )

        # MaxSim over all documents in one padded, masked batched matmul.
        query_tensor = torch.from_numpy(query_vecs).to(self._device)
        doc_tensors = [torch.from_numpy(d).to(self._device) for d in doc_output.multivector]
        return maxsim_scores_batched(query_tensor, doc_tensors)

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        """Score parallel (query, doc) pairs via per-query MaxSim grouping.

        Sequence-length runtime options are forwarded to query/document encode.
        """

        def score_with_options(
            query: Item,
            items: list[Item],
            *,
            instruction: str | None = None,
        ) -> list[float]:
            return self.score(query, items, instruction=instruction, options=options)

        output = grouped_score_pairs(score_with_options, queries, docs, instruction=instruction)
        output.input_token_counts = self._pair_input_token_counts(queries, docs, instruction, options)
        return output

    def _pair_input_token_counts(
        self,
        queries: list[Item],
        docs: list[Item],
        instruction: str | None,
        options: dict[str, Any] | None = None,
    ) -> list[int] | None:
        """Return faithful logical token counts for bi-encoder pair scoring.

        ColBERT encodes each query and document separately, with different
        truncation caps. The generic score fallback jointly tokenizes pairs as
        if this were a cross-encoder, so it cannot represent the work performed
        here. Re-tokenize the transformed inputs with the same tokenizer and
        caps, then report query + document tokens for each logical pair.
        """
        if self._tokenizer is None:
            return None
        if not queries:
            return []

        opts = options or {}
        query_max_length = self._coerce_runtime_max_length(
            opts.get("query_max_length"),
            self._query_max_length,
        )
        doc_max_length = self._coerce_runtime_max_length(
            opts.get("max_seq_length"),
            self._max_seq_length,
        )

        try:
            query_texts = self._extract_texts(queries, instruction, is_query=True)
            doc_texts = self._extract_texts(docs, None, is_query=False)
            with self._tokenizer_guard():
                query_encodings = self._tokenizer(
                    query_texts,
                    max_length=query_max_length,
                    truncation=True,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                doc_encodings = self._tokenizer(
                    doc_texts,
                    max_length=doc_max_length,
                    truncation=True,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
            query_counts = [len(ids) for ids in query_encodings["input_ids"]]
            doc_counts = [len(ids) for ids in doc_encodings["input_ids"]]
        except Exception:  # noqa: BLE001 -- metering must never fail inference
            return None

        if len(query_counts) != len(queries) or len(doc_counts) != len(docs):
            return None
        return [query + doc for query, doc in zip(query_counts, doc_counts, strict=True)]

    def _build_position_ids(self, cu_seqlens: torch.Tensor, num_seqs: int) -> torch.Tensor:
        """Build position IDs for packed sequences."""
        return build_position_ids(cu_seqlens)

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
        from flash_attn import flash_attn_varlen_func

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

    def _split_embeddings(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
        input_ids: torch.Tensor,
        *,
        is_query: bool = True,
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

        if not is_query:
            special_ids |= self._doc_skiplist_ids

        # Build a batched filter mask on GPU
        if special_ids:
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
