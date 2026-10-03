from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import torch

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._modernbert_flash_graphs import (
    EncodeFn,
    PackedForward,
    VarlenGraphRunner,
    graph_runner,
    parse_graph_mode,
)
from sie_server.adapters._packed_rope import packed_rope_available, rotate_packed_qkv_
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._utils import extract_text
from sie_server.core.inference_output import ScoreOutput
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_ERR_CUDA_REQUIRED = "ModernBertFlashCrossEncoder requires CUDA for Flash Attention."


class ModernBertFlashCrossEncoderAdapter(FlashBaseAdapter):
    """Cross-encoder adapter for ModernBERT with Flash Attention 2 varlen.

    Uses flash_attn_varlen_qkvpacked_func to avoid wasting compute on padding tokens.
    Supports ModernBERT architecture with RoPE, pre-norm, and sliding window attention.
    """

    fallback_adapter_path: ClassVar[str | None] = "cross_encoder:CrossEncoderAdapter"
    fallback_kwargs_overrides: ClassVar[dict[str, Any]] = {"attn_implementation": "sdpa"}

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("score",),
        unload_fields=("_model", "_tokenizer", "_dtype", "_num_heads", "_head_dim", "_hidden_size", "_graphs"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = True,
        max_seq_length: int = 8192,
        compute_precision: ComputePrecision = "bfloat16",
        revision: str | None = None,
        cuda_graphs: str | bool = "off",
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code.
            max_seq_length: Maximum sequence length for query+document.
            compute_precision: Compute precision (bfloat16 recommended for ModernBERT).
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading model artifacts. Forwarded to ``from_pretrained(..., revision=...)``.
            cuda_graphs: "off" (the default) or "bucketed": whether forwards
                replay the encoder as CUDA graphs (see
                ``sie_server.adapters._modernbert_flash_graphs``). An operator
                setting, fixed at load.
            **kwargs: Additional arguments (ignored).

        Raises:
            ValueError: If ``cuda_graphs`` is not "off" or "bucketed".
        """
        _ = kwargs
        self._cuda_graphs = parse_graph_mode(cuda_graphs, adapter="ModernBertFlashCrossEncoder")
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision
        self._revision = revision

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
        # Replays the encoder as CUDA graphs, when the operator enabled them.
        self._graphs: VarlenGraphRunner | None = None

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

        shared_kwargs: dict[str, Any] = {"trust_remote_code": self._trust_remote_code}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            **shared_kwargs,
        )

        # Load model with flash_attention_2 to use model's optimized rotary embeddings
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name_or_path,
            torch_dtype=self._dtype,
            attn_implementation="flash_attention_2",
            **shared_kwargs,
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

        # Clamp configured max_seq_length to whatever the tokenizer/model
        # actually support to avoid OOB position embeddings on long inputs.
        self._max_seq_length = self._resolve_tokenizer_ceiling(
            self._tokenizer,
            self._model,
            self._max_seq_length,
        )
        self._graphs = self._graph_runner()

    def _graph_runner(self) -> VarlenGraphRunner | None:
        """The CUDA graph runner for the loaded model; None when graphs are off or do not apply."""
        if self._cuda_graphs == "off" or self._device is None or self._tokenizer is None:
            return None
        reason = self._graph_unsupported_reason()
        if reason is not None:
            logger.warning(
                "ModernBERT cuda_graphs=%s does not apply to %s, which runs eagerly: %s",
                self._cuda_graphs,
                self._model_name_or_path,
                reason,
            )
            return None
        window = self._max_seq_length
        return graph_runner(
            lambda: self._graph_encoder(window),
            mode=self._cuda_graphs,
            device=self._device,
            hidden_size=self._hidden_size,
            dtype=self._dtype or self._resolve_dtype(),
            window=window,
            pad_token_id=self._tokenizer.pad_token_id,
            name=self._model_name_or_path,
        )

    def _graph_unsupported_reason(self) -> str | None:
        """Why this model's forward cannot be recorded; None when it can."""
        if not packed_rope_available(self._device or "cpu"):
            return "the graph's rotary kernel needs Triton on CUDA"
        for layer in self._model.model.layers:
            rotary = getattr(layer.attn, "rotary_emb", None)
            if not callable(getattr(rotary, "_update_cos_sin_cache", None)):
                return "its attention layers have no flash-attn rotary embedding"
        return None

    def _graph_encoder(self, window: int) -> tuple[EncodeFn, list[torch.Tensor]]:
        """The encoder a graph records: ``_forward_flash`` up to the final norm.

        The eager forward rotates queries and keys with each layer's
        flash-attn rotary embedding, whose kernel finds positions from
        ``cu_seqlens``. A graph's ``cu_seqlens`` has fixed slots, so the graph
        rotates by token position instead (``rotate_packed_qkv_``), with the
        same arithmetic and the same cosine and sine tables: each rotary
        module's own cache, grown to the model window once here.

        Returns:
            The encoder and the tables it reads.
        """
        backbone = self._model.model
        embeddings = backbone.embeddings
        layers = list(backbone.layers)
        final_norm = backbone.final_norm
        tables: dict[Any, tuple[torch.Tensor, torch.Tensor]] = {}
        layer_tables: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in layers:
            rotary = layer.attn.rotary_emb
            base = getattr(rotary, "base", None)
            key = float(base) if isinstance(base, (int, float)) else id(rotary)
            if key not in tables:
                rotary._update_cos_sin_cache(window, device=self._device, dtype=self._dtype)
                # Held here: the module replaces its cache when it grows, and a graph keeps reading these.
                tables[key] = (rotary._cos_cached, rotary._sin_cached)
            layer_tables.append(tables[key])
        num_heads, head_dim, hidden_size = self._num_heads, self._head_dim, self._hidden_size
        softmax_scale = 1.0 / (head_dim**0.5)

        def encode(
            input_ids: torch.Tensor, positions: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int, total: int
        ) -> torch.Tensor:
            from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

            hidden = embeddings.tok_embeddings(input_ids)
            if hasattr(embeddings, "norm"):
                hidden = embeddings.norm(hidden)
            if hasattr(embeddings, "drop"):
                hidden = embeddings.drop(hidden)
            for layer, (cos, sin) in zip(layers, layer_tables, strict=True):
                qkv = layer.attn.Wqkv(layer.attn_norm(hidden)).view(total, 3, num_heads, head_dim)
                rotate_packed_qkv_(qkv, positions, cos, sin)
                attn_out = flash_attn_varlen_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_seqlen=max_seqlen,
                    softmax_scale=softmax_scale,
                    causal=False,
                    window_size=layer.attn.local_attention,
                )
                hidden = hidden + layer.attn.Wo(attn_out.reshape(total, hidden_size))
                hidden = hidden + layer.mlp(layer.mlp_norm(hidden))
            return final_norm(hidden)

        return encode, [table for pair in tables.values() for table in pair]

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.bfloat16)

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
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        query_text = extract_text(query, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="ModernBertFlashCrossEncoder"))
        if instruction:
            query_text = f"{instruction} {query_text}"

        # Tokenize all pairs individually (no padding)
        pairs = [
            (
                query_text,
                extract_text(item, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="ModernBertFlashCrossEncoder")),
            )
            for item in items
        ]
        encodings = [
            self._tokenizer(
                q,
                d,
                max_length=self._max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            for q, d in pairs
        ]
        return self._scores(encodings).tolist()

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
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        opts = options or {}
        # Hard-clamp to the load-time ceiling so runtime overrides cannot
        # push past the model's positional capacity. Malformed overrides
        # (None, strings, negatives) fall back to the ceiling.
        max_length = self._coerce_runtime_max_length(opts.get("max_seq_length"), self._max_seq_length)

        # Build (query, doc) pairs
        pairs = []
        for query, doc in zip(queries, docs, strict=True):
            query_text = extract_text(
                query, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="ModernBertFlashCrossEncoder")
            )
            if instruction:
                query_text = f"{instruction} {query_text}"
            doc_text = extract_text(doc, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="ModernBertFlashCrossEncoder"))
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
        return ScoreOutput(scores=self._scores(encodings))

    def _scores(self, encodings: list[Any]) -> np.ndarray:
        """float32 relevance scores for tokenized (query, doc) pairs: a replayed CUDA graph, or eagerly."""
        seq_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        if self._graphs is not None:
            flat_ids = torch.cat([enc["input_ids"].squeeze(0) for enc in encodings]).numpy()

            def head(packed: PackedForward) -> np.ndarray:
                logits = self._head_logits(packed.hidden, packed.cu_seqlens, len(seq_lengths))
                return self._activate(logits)

            with torch.inference_mode():
                scores = self._graphs.run(flat_ids, seq_lengths, head)
            if scores is not None:
                return scores

        # Build packed representation
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack tensors
        input_ids = torch.cat([enc["input_ids"].squeeze(0) for enc in encodings]).to(self._device)

        # Build cu_seqlens
        cu_seqlens = torch.zeros(len(encodings) + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            logits = self._forward_flash(
                input_ids,
                cu_seqlens,
                max_seqlen,
                total_tokens,
                len(encodings),
                seq_lengths,
            )
            return self._activate(logits)

    def _activate(self, logits: torch.Tensor) -> np.ndarray:
        """Scores from logits: sigmoid for single-label models, as float32 on the host."""
        scores_tensor = logits.squeeze(-1).float()
        if self._use_sigmoid:
            scores_tensor = torch.sigmoid(scores_tensor)
        return scores_tensor.cpu().numpy().astype(np.float32)

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
        return self._head_logits(hidden, cu_seqlens, batch_size)

    def _head_logits(self, hidden: torch.Tensor, cu_seqlens: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Classification logits ``[batch_size, 1]`` from final-normed packed hidden states."""
        head = self._model.head
        drop = self._model.drop
        classifier = self._model.classifier

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
