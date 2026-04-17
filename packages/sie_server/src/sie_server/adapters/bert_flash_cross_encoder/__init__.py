from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from torch.nn import functional as F

from sie_server.adapters._flash_base import FlashBaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._utils import extract_text
from sie_server.core.inference_output import ScoreOutput
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_ERR_CUDA_REQUIRED = "BertFlashCrossEncoder requires CUDA for Flash Attention."


class BertFlashCrossEncoderAdapter(FlashBaseAdapter):
    """Cross-encoder adapter with Flash Attention 2 varlen.

    Uses flash_attn_varlen_func to avoid wasting compute on padding tokens.
    Loads BERT/RoBERTa weights and runs a custom forward pass with flash attention.
    """

    fallback_adapter_path: ClassVar[str | None] = "cross_encoder:CrossEncoderAdapter"
    fallback_kwargs_overrides: ClassVar[dict[str, Any]] = {"attn_implementation": "sdpa"}

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("score",),
        unload_fields=("_model", "_tokenizer", "_dtype", "_num_heads", "_head_dim", "_hidden_size", "_use_sigmoid"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = False,
        max_seq_length: int = 512,
        compute_precision: ComputePrecision = "float16",
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code.
            max_seq_length: Maximum sequence length for query+document.
            compute_precision: Compute precision (float16, bfloat16, float32).
            **kwargs: Additional arguments (ignored).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_seq_length = max_seq_length
        self._compute_precision = compute_precision

        # Loaded state
        self._model: Any = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None

        # Model config (set during load)
        self._num_heads: int = 0
        self._head_dim: int = 0
        self._hidden_size: int = 0

    def load(self, device: str) -> None:
        """Load model weights onto the specified device.

        Note: This adapter requires CUDA. Use the create_for_device() factory method
        for automatic fallback selection on non-CUDA devices.

        Args:
            device: Device string (must start with "cuda").

        Raises:
            RuntimeError: If device is not CUDA (should use factory method instead).
        """
        if not device.startswith("cuda"):
            msg = (
                f"BertFlashCrossEncoderAdapter requires CUDA, got device='{device}'. "
                "Use create_for_device() factory method for automatic fallback selection."
            )
            raise RuntimeError(msg)

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._device = device
        self._dtype = self._resolve_dtype()

        logger.info(
            "Loading %s with Flash Attention varlen (dtype=%s)",
            self._model_name_or_path,
            self._dtype,
        )

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            trust_remote_code=self._trust_remote_code,
        )

        # Load model with eager attention (we handle flash attention manually)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name_or_path,
            torch_dtype=self._dtype,
            attn_implementation="eager",
            trust_remote_code=self._trust_remote_code,
        )
        self._model.to(device)
        self._model.eval()

        # Cache config values
        config = self._model.config
        self._num_heads = config.num_attention_heads
        self._hidden_size = config.hidden_size
        self._head_dim = self._hidden_size // self._num_heads

        # Detect activation function (matching sentence-transformers CrossEncoder logic)
        # Some models (e.g., MiniLM) override the default with Identity
        self._use_sigmoid = self._should_use_sigmoid(config)

        logger.info(
            "Loaded: hidden=%d, heads=%d, head_dim=%d, layers=%d, sigmoid=%s",
            self._hidden_size,
            self._num_heads,
            self._head_dim,
            config.num_hidden_layers,
            self._use_sigmoid,
        )

    def _should_use_sigmoid(self, config: Any) -> bool:
        """Determine if sigmoid should be applied (matching sentence-transformers logic).

        CrossEncoder models may specify a custom activation function in their config.
        The logic is:
        1. Check config.sentence_transformers["activation_fn"]
        2. Check config.sbert_ce_default_activation_function
        3. Default: Sigmoid for num_labels=1, else Identity
        """
        # Check sentence_transformers config first (new format)
        if hasattr(config, "sentence_transformers"):
            st_config = config.sentence_transformers
            if isinstance(st_config, dict) and "activation_fn" in st_config:
                activation_fn = st_config["activation_fn"]
                return "Sigmoid" in activation_fn

        # Check old format (sbert_ce_default_activation_function)
        if hasattr(config, "sbert_ce_default_activation_function"):
            activation_fn = config.sbert_ce_default_activation_function
            if activation_fn is not None:
                return "Sigmoid" in activation_fn

        # Default: Sigmoid for num_labels=1
        return config.num_labels == 1

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

        query_text = extract_text(query, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="BertFlashCrossEncoder"))
        if instruction:
            query_text = f"{instruction} {query_text}"

        # Tokenize all pairs individually (no padding)
        pairs = [
            (query_text, extract_text(item, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="BertFlashCrossEncoder")))
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

        # Build packed representation
        seq_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack tensors
        input_ids = torch.cat([enc["input_ids"].squeeze(0) for enc in encodings]).to(self._device)

        # Token type IDs - BERT returns them from tokenizer, XLMRoberta doesn't but still needs them (all zeros)
        if "token_type_ids" in encodings[0]:
            token_type_ids = torch.cat([enc["token_type_ids"].squeeze(0) for enc in encodings]).to(self._device)
        else:
            # XLMRoberta: create zeros (required for token_type_embeddings)
            token_type_ids = torch.zeros(total_tokens, dtype=torch.long, device=self._device)

        # Build cu_seqlens
        cu_seqlens = torch.zeros(len(pairs) + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            # Run forward pass with flash attention
            logits = self._forward_flash(
                input_ids,
                token_type_ids,
                cu_seqlens,
                max_seqlen,
                total_tokens,
                len(pairs),
            )

            # Apply activation function (matching sentence-transformers CrossEncoder.predict())
            scores_tensor = logits.squeeze(-1).float()
            if self._use_sigmoid:
                scores_tensor = torch.sigmoid(scores_tensor)
            scores = scores_tensor.cpu().tolist()

        return scores

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
        max_length = opts.get("max_seq_length", self._max_seq_length)

        # Build (query, doc) pairs
        pairs = []
        for query, doc in zip(queries, docs, strict=True):
            query_text = extract_text(query, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="BertFlashCrossEncoder"))
            if instruction:
                query_text = f"{instruction} {query_text}"
            doc_text = extract_text(doc, err_msg=ERR_REQUIRES_TEXT.format(adapter_name="BertFlashCrossEncoder"))
            pairs.append((query_text, doc_text))

        # Tokenize all pairs individually (no padding - flash attention handles it)
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

        # Build packed representation
        seq_lengths = [enc["input_ids"].shape[1] for enc in encodings]
        total_tokens = sum(seq_lengths)
        max_seqlen = max(seq_lengths)

        # Pack tensors
        input_ids = torch.cat([enc["input_ids"].squeeze(0) for enc in encodings]).to(self._device)

        # Token type IDs
        if "token_type_ids" in encodings[0]:
            token_type_ids = torch.cat([enc["token_type_ids"].squeeze(0) for enc in encodings]).to(self._device)
        else:
            token_type_ids = torch.zeros(total_tokens, dtype=torch.long, device=self._device)

        # Build cu_seqlens
        cu_seqlens = torch.zeros(len(pairs) + 1, dtype=torch.int32, device=self._device)
        for i, length in enumerate(seq_lengths):
            cu_seqlens[i + 1] = cu_seqlens[i] + length

        with torch.inference_mode():
            logits = self._forward_flash(
                input_ids,
                token_type_ids,
                cu_seqlens,
                max_seqlen,
                total_tokens,
                len(pairs),
            )

            scores_tensor = logits.squeeze(-1).float()
            if self._use_sigmoid:
                scores_tensor = torch.sigmoid(scores_tensor)

            # Convert to float32 numpy array and wrap in ScoreOutput
            import numpy as np

            scores_array = scores_tensor.cpu().numpy().astype(np.float32)

        return ScoreOutput(scores=scores_array)

    def _forward_flash(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Run forward pass with flash attention varlen.

        Args:
            input_ids: Packed input IDs [total_tokens].
            token_type_ids: Packed token type IDs [total_tokens] or None for RoBERTa.
            cu_seqlens: Cumulative sequence lengths [batch_size + 1].
            max_seqlen: Maximum sequence length in batch.
            total_tokens: Total number of tokens.
            batch_size: Number of sequences.

        Returns:
            Logits tensor [batch_size, 1].
        """
        from flash_attn import flash_attn_varlen_func

        # Get model components - handle both BERT and XLMRoberta naming
        if hasattr(self._model, "bert"):
            backbone = self._model.bert
        elif hasattr(self._model, "roberta"):
            backbone = self._model.roberta
        else:
            raise RuntimeError("Unknown model architecture")

        embeddings = backbone.embeddings
        encoder = backbone.encoder
        pooler = backbone.pooler if hasattr(backbone, "pooler") else None
        classifier = self._model.classifier

        # Determine position ID offset (BERT: 0, RoBERTa: padding_idx + 1)
        pos_emb = embeddings.position_embeddings
        if hasattr(pos_emb, "padding_idx") and pos_emb.padding_idx is not None:
            pos_offset = pos_emb.padding_idx + 1  # RoBERTa-style
        else:
            pos_offset = 0  # BERT-style

        # Build position IDs for each sequence
        position_ids = self._build_position_ids(cu_seqlens, batch_size, offset=pos_offset)

        # Compute embeddings: word + position (+ token_type for BERT)
        hidden = embeddings.word_embeddings(input_ids)
        hidden = hidden + embeddings.position_embeddings(position_ids)
        if token_type_ids is not None and hasattr(embeddings, "token_type_embeddings"):
            hidden = hidden + embeddings.token_type_embeddings(token_type_ids)
        hidden = embeddings.LayerNorm(hidden)
        hidden = embeddings.dropout(hidden)

        # Run transformer layers with flash attention
        softmax_scale = 1.0 / (self._head_dim**0.5)

        for layer in encoder.layer:
            # Self-attention with flash
            attn = layer.attention

            # Compute Q, K, V
            query = attn.self.query(hidden)
            key = attn.self.key(hidden)
            value = attn.self.value(hidden)

            # Reshape for flash attention: [total_tokens, num_heads, head_dim]
            query = query.view(total_tokens, self._num_heads, self._head_dim)
            key = key.view(total_tokens, self._num_heads, self._head_dim)
            value = value.view(total_tokens, self._num_heads, self._head_dim)

            # Flash attention with variable-length sequences
            attn_output = flash_attn_varlen_func(
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
            attn_output = attn_output.view(total_tokens, self._hidden_size)

            # Attention output projection + residual + LayerNorm
            attn_output = attn.output.dense(attn_output)
            attn_output = attn.output.dropout(attn_output)
            hidden = attn.output.LayerNorm(hidden + attn_output)

            # MLP (intermediate + output)  # section header
            intermediate = layer.intermediate.dense(hidden)
            intermediate = layer.intermediate.intermediate_act_fn(intermediate)
            mlp_output = layer.output.dense(intermediate)
            mlp_output = layer.output.dropout(mlp_output)
            hidden = layer.output.LayerNorm(hidden + mlp_output)

        # Extract [CLS] token for each sequence (position 0 of each sequence)
        cls_indices = cu_seqlens[:-1].long()  # Start of each sequence
        cls_hidden = hidden[cls_indices]  # [batch_size, hidden_size]

        # Pooler and classifier - handle different architectures
        if pooler is not None:
            # BERT-style: pooler + classifier
            pooled = pooler.dense(cls_hidden)
            pooled = pooler.activation(pooled)
            if hasattr(self._model, "dropout"):
                pooled = self._model.dropout(pooled)
            logits = classifier(pooled)
        elif hasattr(classifier, "dense") and hasattr(classifier, "out_proj"):
            # XLMRoberta-style: classification head with dense + out_proj
            # Manually apply the head since it expects 3D input but we have 2D
            x = classifier.dropout(cls_hidden)
            x = classifier.dense(x)
            x = torch.tanh(x)
            x = classifier.dropout(x)
            logits = classifier.out_proj(x)
        else:
            # Fallback: just apply classifier directly
            logits = classifier(cls_hidden)

        return logits

    def _build_position_ids(
        self,
        cu_seqlens: torch.Tensor,
        batch_size: int,
        offset: int = 0,
    ) -> torch.Tensor:
        """Build position IDs for packed sequences.

        Args:
            cu_seqlens: Cumulative sequence lengths.
            batch_size: Number of sequences.
            offset: Starting position (0 for BERT, padding_idx+1 for RoBERTa).
        """
        pos_list = []
        for i in range(batch_size):
            seq_len = int(cu_seqlens[i + 1].item() - cu_seqlens[i].item())
            pos_list.append(torch.arange(offset, offset + seq_len, device=self._device, dtype=torch.long))
        return torch.cat(pos_list)
