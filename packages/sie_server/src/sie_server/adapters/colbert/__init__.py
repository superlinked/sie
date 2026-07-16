"""ColBERT adapter for late interaction models.

This adapter supports ColBERT-style models that return per-token embeddings
for late interaction retrieval and reranking.

Supports models like:
- answerdotai/answerai-colbert-small-v1
- colbert-ir/colbertv2.0
- jinaai/jina-colbert-v2

Key features:
- Uses Flash Attention 2's variable-length attention (no padding waste)
- Returns ALL token embeddings (no pooling) for late interaction
- Linear projection layer: hidden_size → token_dim (e.g., 768 → 128)
- L2 normalization of token embeddings
- Supports both encoding and MaxSim scoring

Performance note (Dec 2025):
    This adapter already uses flash_attn_varlen_func for BERT-based models via
    _encode_manual_flash(). Benchmarks show 22-37x speedup vs Infinity baseline
    on NFCorpus. No separate "ColBERTBertFlashAdapter" is needed - the optimization
    is already built-in. See _should_use_native_mode() for mode selection logic.

See: https://github.com/stanford-futuredata/ColBERT
"""

from __future__ import annotations

import logging
import string
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.nn import functional

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._flash_pack import build_position_ids
from sie_server.adapters._multivector import maxsim_scores_batched
from sie_server.adapters._pylate_dense import apply_dense_chain, load_pylate_dense_chain
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, ComputePrecision
from sie_server.adapters._utils import grouped_score_pairs
from sie_server.core.inference_output import EncodeOutput, ScoreOutput

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedModel, PreTrainedTokenizerFast

# Runtime imports (needed for actual execution, not just type hints)
import numpy as np

from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)


class ColBERTAdapter(BaseAdapter):
    """Adapter for ColBERT-style late interaction models.

    ColBERT models produce per-token embeddings that enable late interaction
    scoring via MaxSim. This adapter supports two execution modes:

    1. **Native mode** (for models with built-in flash attention like jina-colbert-v2):
       - Uses the model's native forward pass with flash_attention_2
       - Supports rotary embeddings, custom architectures, etc.

    2. **Manual flash mode** (for standard BERT-based ColBERT models):
       - Uses Flash Attention 2 with varlen for efficient variable-length batching
       - Requires standard BERT-style position embeddings

    Both modes return per-token embeddings (shape: [num_tokens, token_dim]),
    apply linear projection if present, and L2 normalize.

    Supports both encode() for getting embeddings and score() for MaxSim reranking.
    """

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("multivector", "score"),
        unload_fields=("_model", "_tokenizer", "_linear", "_actual_token_dim", "_dense_chain"),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        token_dim: int = 128,
        normalize: bool = True,
        max_seq_length: int = 512,
        query_max_length: int = 32,
        doc_max_length: int | None = None,
        compute_precision: ComputePrecision = "float16",
        skip_special_tokens: bool = True,
        query_prefix: str = "",
        doc_prefix: str = "",
        use_native_attention: bool | None = None,
        query_expansion: bool = True,
        doc_punctuation_skiplist: bool = True,
        muvera_config: dict[str, Any] | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            token_dim: Output dimension per token (after projection).
            normalize: Whether to L2-normalize token embeddings.
            max_seq_length: Maximum sequence length (model capacity).
            query_max_length: Maximum sequence length for queries.
            doc_max_length: Maximum sequence length for documents.
                Defaults to max_seq_length if not set.  PyLate uses 180
                for mxbai-colbert; truncating documents to a shorter
                length can improve quality by reducing noise tokens.
            compute_precision: Compute precision (float16 recommended for flash).
            skip_special_tokens: Whether to exclude [CLS], [SEP], [PAD] from output.
            query_prefix: Prefix to prepend to queries (e.g., "[Q] ").
            doc_prefix: Prefix to prepend to documents (e.g., "[D] ").
            use_native_attention: If True, use model's native forward pass.
                If None (default), auto-detect based on model architecture.
            query_expansion: Whether to pad queries with MASK tokens to query_max_length.
                This is a core ColBERT feature where MASK tokens become additional
                "virtual" query tokens. Default: True.
            doc_punctuation_skiplist: Whether to remove punctuation tokens from
                document multivectors (pylate applies a punctuation skiplist to
                document embeddings; disable to match a serving path that does
                not). Default: True.
            muvera_config: MUVERA configuration dict with keys like num_repetitions,
                num_simhash_projections, normalize. Used for FDE postprocessing.
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading the tokenizer, config, model, and Dense-chain/projection
                artifacts. Forwarded to ``from_pretrained(..., revision=...)``.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs
        self._model_name_or_path = str(model_name_or_path)
        self._revision = revision
        self._token_dim = token_dim
        self._normalize = normalize
        self._max_seq_length = max_seq_length
        self._query_max_length = query_max_length
        self._doc_max_length = doc_max_length if doc_max_length is not None else max_seq_length
        self._compute_precision = compute_precision
        self._skip_special_tokens = skip_special_tokens
        self._query_prefix = query_prefix
        self._doc_prefix = doc_prefix
        self._use_native_attention = use_native_attention
        self._query_expansion = query_expansion
        self._doc_punctuation_skiplist = doc_punctuation_skiplist
        self._muvera_config = muvera_config

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerFast | None = None
        self._linear: torch.nn.Linear | None = None
        self._dense_chain: list[torch.Tensor] | None = None
        self._device: str | None = None
        self._actual_token_dim: int | None = None
        self._is_cuda: bool = False  # Set during load()
        self._native_mode: bool = False  # Set during load()
        self._expansion_token_id: int | None = None  # Set during load() if query_expansion
        self._query_prefix_id: int | None = None  # Set during load() for special token prefixes
        self._doc_prefix_id: int | None = None  # Set during load() for special token prefixes
        self._doc_skiplist_ids: set[int] = set()  # Set during load(): punctuation token IDs to skip in docs

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g. "cuda", "cuda:0", "mps", "cpu").
        """
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self._device = device
        self._is_cuda = device.startswith("cuda")
        dtype = self._resolve_dtype()

        logger.info(
            "Loading ColBERT model %s on device=%s with dtype=%s revision=%s",
            self._model_name_or_path,
            device,
            dtype,
            self._revision,
        )

        shared_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self._revision is not None:
            shared_kwargs["revision"] = self._revision

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name_or_path,
            **shared_kwargs,
        )

        # Register prefix tokens (e.g. [Q], [D]) as special tokens if they are
        # not already in the vocabulary.  PyLate's ColBERT does the same via
        # ``tokenizer.add_tokens(["[Q]", "[D]"])``.  Without this, tokens like
        # "[Q]" are decomposed into multiple sub-word pieces (``[``, ``q``,
        # ``]``) whose embeddings are meaningless.  After the model is loaded we
        # call ``model.resize_token_embeddings`` so the new IDs get a randomly-
        # initialised embedding (overwritten by the checkpoint's trained weights
        # when the safetensors file already contains them).
        new_tokens: list[str] = []
        for prefix in (self._query_prefix, self._doc_prefix):
            token = prefix.strip()
            if token and token not in self._tokenizer.vocab:
                new_tokens.append(token)
        if new_tokens:
            num_added = self._tokenizer.add_tokens(new_tokens)
            if num_added:
                logger.info("Registered %d new prefix token(s) in tokenizer: %s", num_added, new_tokens)

        # Configure tokenizer for query expansion (pad queries with MASK tokens)
        # This matches PyLate's behavior: use MASK token for padding queries
        if self._query_expansion and self._tokenizer.mask_token_id is not None:
            self._expansion_token_id = self._tokenizer.mask_token_id
            logger.info("Query expansion enabled, using MASK token for padding")
        else:
            self._expansion_token_id = None
            if self._query_expansion:
                logger.warning("Query expansion requested but model has no MASK token, disabling")

        # Resolve prefix token IDs (for special tokens like [unused0], [unused1])
        # These need to be inserted as token IDs, not as text
        self._query_prefix_id = self._resolve_prefix_token_id(self._query_prefix)
        self._doc_prefix_id = self._resolve_prefix_token_id(self._doc_prefix)
        if self._query_prefix_id:
            logger.info("Query prefix '%s' resolved to token ID %d", self._query_prefix.strip(), self._query_prefix_id)
        if self._doc_prefix_id:
            logger.info("Doc prefix '%s' resolved to token ID %d", self._doc_prefix.strip(), self._doc_prefix_id)

        # Build punctuation skiplist for documents (matches PyLate's default
        # ``skiplist_words = string.punctuation``).  Document token embeddings
        # whose token ID is in this set are removed before returning
        # multivectors, preventing punctuation from participating in MaxSim.
        if self._doc_punctuation_skiplist:
            skiplist_ids: set[int] = set()
            for ch in string.punctuation:
                ids = self._tokenizer.encode(ch, add_special_tokens=False)
                skiplist_ids.update(ids)
            self._doc_skiplist_ids = skiplist_ids
            logger.info(
                "Built document skiplist with %d token IDs from %d punctuation chars",
                len(skiplist_ids),
                len(string.punctuation),
            )
        else:
            logger.info("Document punctuation skiplist disabled (doc_punctuation_skiplist=False)")

        # Determine whether to use native attention or manual flash attention
        config = AutoConfig.from_pretrained(
            self._model_name_or_path,
            **shared_kwargs,
        )
        self._native_mode = self._is_cuda and self._should_use_native_mode(config)

        if self._native_mode:
            # Use model's native forward with flash_attention_2, or sdpa when
            # flash-attn is unavailable (exactly the situation that selects the
            # ColBERTAdapter fallback for e.g. ModernBERT on pre-Ampere CUDA).
            # Import lazily to avoid circular deps (core.inference -> core.loader -> base)
            from sie_server.core.inference import is_flash_attention_available

            attn_impl = "flash_attention_2" if is_flash_attention_available(device) else "sdpa"
            logger.info("Using native attention mode with attn_implementation=%s", attn_impl)
            self._model = AutoModel.from_pretrained(
                self._model_name_or_path,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                **shared_kwargs,
            )
        else:
            # Load model with eager attention - we'll run our own flash attention
            logger.info("Using manual flash attention mode")
            self._model = AutoModel.from_pretrained(
                self._model_name_or_path,
                torch_dtype=dtype,
                attn_implementation="eager",
                **shared_kwargs,
            )

        # Resize token embeddings if we added new prefix tokens above.
        # The checkpoint's safetensors already contain trained embeddings for
        # these token IDs (PyLate saves them), so the weights are loaded
        # correctly — we just need to make sure the embedding matrix is big
        # enough to hold them.
        if new_tokens:
            self._model.resize_token_embeddings(len(self._tokenizer))

        self._model.to(device)
        self._model.eval()

        # Probe the trained pylate Dense chain first (#1680). When a checkpoint
        # ships one it IS the projection head, so the legacy probes must not
        # run; the loader returns None for a missing modules.json AND for a
        # Dense-less modules.json (e.g. mxbai-colbert-large-v1), so every
        # checkpoint without a chain falls through to the exact same probes.
        self._dense_chain = load_pylate_dense_chain(
            self._model_name_or_path,
            hidden_size=self._model.config.hidden_size,
            token_dim=self._token_dim,
            device=self._device,
            dtype=dtype,
            revision=self._revision,
        )
        # Check for linear projection layer (ColBERT-specific)
        # Different ColBERT implementations use different names
        self._linear = None if self._dense_chain is not None else self._find_projection_layer()

        # Determine actual output dimension
        hidden_size = self._model.config.hidden_size
        if self._dense_chain is not None:
            self._actual_token_dim = self._dense_chain[-1].shape[0]
            logger.info(
                "ColBERT pylate Dense chain: %s",
                [tuple(w.shape) for w in self._dense_chain],
            )
        elif self._linear is not None:
            self._actual_token_dim = self._linear.out_features
            logger.info(
                "ColBERT projection: %d -> %d",
                hidden_size,
                self._actual_token_dim,
            )
        elif self._token_dim < hidden_size:
            # No projection layer but token_dim < hidden_size: Matryoshka truncation
            self._actual_token_dim = self._token_dim
            logger.info(
                "No projection layer found, using Matryoshka truncation: %d -> %d",
                hidden_size,
                self._token_dim,
            )
        else:
            # No projection layer, use hidden_size directly
            self._actual_token_dim = hidden_size
            logger.info(
                "No projection layer found, using hidden_size=%d as token_dim",
                hidden_size,
            )

    def _should_use_native_mode(self, config: Any) -> bool:
        """Determine whether to use native attention or manual flash attention.

        Uses native mode for:
        - Models with rotary position embeddings (not standard BERT)
        - Models with explicit flash attention support in config
        - Models with custom architectures (not BERT/RoBERTa/etc.)
        """
        # If explicitly specified in adapter options, use that
        if self._use_native_attention is not None:
            return self._use_native_attention

        # ModernBERT requires native mode (the manual flash loop is BERT-only:
        # it iterates model.encoder.layer, which ModernBERT does not have)
        if getattr(config, "model_type", None) == "modernbert":
            logger.info("Detected ModernBERT architecture, using native mode (manual flash loop is BERT-only)")
            return True

        # Auto-detect based on model config
        # Rotary embeddings require native mode (our manual loop doesn't support them)
        if getattr(config, "position_embedding_type", None) == "rotary":
            logger.info("Detected rotary position embeddings, using native mode")
            return True

        # Models with explicit flash attention config should use native mode
        if getattr(config, "use_flash_attn", False):
            logger.info("Model has use_flash_attn=True, using native mode")
            return True

        # Default to manual flash attention for standard BERT-style models
        return False

    def _find_projection_layer(self) -> torch.nn.Linear | None:
        """Find the ColBERT projection layer if it exists.

        Different ColBERT implementations use different names:
        - colbert_linear (HF_ColBERT)
        - linear (some implementations)
        - projection (others)

        Some models (like answerai-colbert-small-v1) store the projection layer
        in the safetensors file but don't register it as a model attribute.
        """
        # Called from load() after model is set
        assert self._model is not None

        # Check common projection layer names as model attributes
        for name in ["colbert_linear", "linear", "projection", "dense"]:
            if hasattr(self._model, name):
                layer = getattr(self._model, name)
                if isinstance(layer, torch.nn.Linear):
                    return layer

        # Check if there's a projection in the model's named modules
        for name, module in self._model.named_modules():
            if "colbert" in name.lower() and isinstance(module, torch.nn.Linear):
                return module

        # Try loading projection layer weights directly from safetensors
        # Some models store linear.weight in safetensors but don't register as module
        return self._load_projection_from_weights()

    def _load_projection_from_weights(self) -> torch.nn.Linear | None:
        """Try to load projection layer weights from model files.

        Some ColBERT models store the projection layer as 'linear.weight' in the
        safetensors file without registering it as a model attribute.
        """
        try:
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open

            # Try to get safetensors file
            try:
                file_path = hf_hub_download(self._model_name_or_path, "model.safetensors", revision=self._revision)
            except (OSError, ValueError):
                return None

            with safe_open(file_path, framework="pt") as f:
                keys = list(f.keys())

                # Look for standalone linear projection layer
                if "linear.weight" in keys:
                    weight = f.get_tensor("linear.weight")
                    out_features, in_features = weight.shape

                    # Create Linear layer and load weights
                    # Check for bias
                    has_bias = "linear.bias" in keys
                    linear = torch.nn.Linear(in_features, out_features, bias=has_bias)
                    linear.weight.data = weight

                    if has_bias:
                        linear.bias.data = f.get_tensor("linear.bias")

                    # Move to device and set dtype
                    dtype = self._resolve_dtype()
                    linear = linear.to(self._device, dtype=dtype)

                    logger.info(
                        "Loaded projection layer from safetensors: %d -> %d",
                        in_features,
                        out_features,
                    )
                    return linear

        except (ImportError, OSError, RuntimeError, KeyError) as e:
            logger.debug("Failed to load projection from weights: %s", e)

        return None

    def _resolve_dtype(self) -> torch.dtype:
        """Resolve compute dtype.

        Off-CUDA (CPU/MPS) ALWAYS uses fp32 for numerical safety — ColBERT is not
        validated for fp16 on MPS, so unlike ``bge_m3``/``pytorch_embedding`` (which
        honor an explicit precision off-CUDA for benchmarked models) it ignores
        ``compute_precision`` off-CUDA. fp16/bf16 are only used on CUDA.
        """
        if not self._device or not str(self._device).startswith("cuda"):
            return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self._compute_precision, torch.float16)

    def _resolve_prefix_token_id(self, prefix: str) -> int | None:
        """Resolve a prefix string to a single token ID if possible.

        Special tokens like [unused0], [unused1] need to be inserted as token IDs,
        not as text (which would be split into multiple tokens).

        Args:
            prefix: Prefix string (e.g., "[unused0] " or "query: ").

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
        # (encode() would split special tokens like [unused0] into multiple pieces)
        if token in self._tokenizer.vocab:
            token_ids = self._tokenizer.convert_tokens_to_ids([token])
            # convert_tokens_to_ids returns int for single token, list for list input
            # isinstance guard for type checker since it returns int | list[int]
            if isinstance(token_ids, list) and len(token_ids) == 1 and token_ids[0] != self._tokenizer.unk_token_id:
                return token_ids[0]

        return None

    def _insert_prefix_token(self, batch: dict[str, torch.Tensor], prefix_id: int) -> dict[str, torch.Tensor]:
        """Insert a prefix token ID at position 1 (after [CLS]) in the batch.

        This is used for special tokens like [unused0] that need to be inserted
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
            instruction: Optional instruction (prepended to text).
            is_query: Whether items are queries (affects max length and prefix).
            prepared_items: Not used by this adapter.

        Returns:
            EncodeOutput with multivector embeddings.
        """
        self._check_loaded()
        if self._tokenizer is None:
            raise RuntimeError(ERR_NOT_LOADED)

        self._validate_output_types(output_types)
        texts = self._extract_texts(items, instruction, is_query=is_query)

        max_length = self._query_max_length if is_query else self._doc_max_length

        # Determine if query expansion should be applied
        # Query expansion pads queries with MASK tokens to exactly max_length
        use_expansion = is_query and self._expansion_token_id is not None

        # Get prefix token ID (for special tokens like [unused0])
        prefix_id = self._query_prefix_id if is_query else self._doc_prefix_id

        if self._is_cuda and not self._native_mode:
            # Manual flash attention mode: use packed varlen sequences (CUDA only)
            multivectors = self._encode_manual_flash(texts, max_length, use_expansion, prefix_id, is_query=is_query)
        else:
            # Native mode: use model's forward pass with padding (works on any device)
            multivectors = self._encode_native(texts, max_length, use_expansion, prefix_id, is_query=is_query)

        return EncodeOutput(
            multivector=multivectors,
            batch_size=len(items),
            is_query=is_query,
            multivector_token_dim=self._actual_token_dim,
        )

    def _encode_native(
        self,
        texts: list[str],
        max_length: int,
        use_expansion: bool,
        prefix_id: int | None = None,
        *,
        is_query: bool = True,
    ) -> list[np.ndarray]:
        """Encode using model's native forward pass (for rotary/custom models).

        Args:
            texts: Texts to encode.
            max_length: Maximum sequence length.
            use_expansion: Whether to use query expansion (MASK padding to max_length).
            prefix_id: Optional prefix token ID to insert after [CLS].
            is_query: Whether these are queries (affects skiplist filtering).
        """
        # Called from encode() which checks these
        assert self._model is not None
        assert self._tokenizer is not None

        # Adjust max_length if we need to insert a prefix token
        effective_max_length = max_length - 1 if prefix_id is not None else max_length

        # Serialise all tokenizer access (including the transient
        # ``pad_token_id`` swap) against the §7.3 metering tokenization that
        # runs on a separate thread — a bare fast tokenizer would raise
        # ``Already borrowed`` under concurrency (#1800).
        with self._tokenizer_guard():
            # For query expansion, pad to exact length using MASK tokens
            if use_expansion:
                # Temporarily swap pad_token_id to mask_token_id for MASK token
                # padding. Restore in ``finally`` so a tokenizer failure (bad
                # input / OOM) cannot leave the instance's ``pad_token_id`` stuck
                # at the MASK id — under ``_tokenizer_guard()`` that corruption
                # would leak to every later guarded caller (including metering),
                # silently mis-padding until reload (CodeRabbit #1800).
                original_pad_id = self._tokenizer.pad_token_id
                self._tokenizer.pad_token_id = self._expansion_token_id
                try:
                    batch = self._tokenizer(
                        texts,
                        max_length=effective_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                finally:
                    self._tokenizer.pad_token_id = original_pad_id
            else:
                # Standard padding for documents
                batch = self._tokenizer(
                    texts,
                    max_length=effective_max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )

        # Insert prefix token ID at position 1 (after [CLS])
        if prefix_id is not None:
            batch = self._insert_prefix_token(batch, prefix_id)

        batch = batch.to(self._device)  # ty: ignore[unresolved-attribute, possibly-missing-attribute]

        # NOTE: We do NOT set attention_mask to all-ones for query expansion.
        # PyLate's default (attend_to_expansion_tokens=False) keeps MASK
        # expansion tokens with attention_mask=0 during the forward pass,
        # meaning other tokens do not attend to them.  The MASK token
        # embeddings are still produced and kept for MaxSim scoring (see
        # the post-forward filtering below).

        with torch.inference_mode():
            # Model forward pass - returns BaseModelOutput with last_hidden_state
            outputs = self._model(**batch)
            hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

            # Apply the trained pylate Dense chain if present, else the single
            # projection layer, else Matryoshka truncation
            if self._dense_chain is not None:
                hidden = apply_dense_chain(hidden, self._dense_chain)
            elif self._linear is not None:
                hidden = self._linear(hidden)
            elif self._actual_token_dim is not None and self._actual_token_dim < hidden.shape[-1]:
                hidden = hidden[:, :, : self._actual_token_dim]

            # L2 normalize
            if self._normalize:
                hidden = functional.normalize(hidden, p=2, dim=-1)

            # Split into per-item results.
            # PyLate keeps ALL query tokens (including MASK expansion tokens)
            # regardless of attention_mask.  For documents, attention_mask
            # filters out actual PAD tokens, and the skiplist removes
            # punctuation.
            multivectors = []
            attention_mask = batch["attention_mask"]
            input_ids = batch["input_ids"]

            for i in range(len(texts)):
                if is_query and use_expansion:
                    # Expanded queries: keep ALL tokens (matches PyLate: "we do
                    # not want to prune expansion tokens in queries even if we
                    # do not attend to them in attention layers")
                    seq_hidden = hidden[i]
                    seq_ids = input_ids[i]
                else:
                    # Documents and non-expanded queries: use attention_mask to
                    # drop real PAD tokens (without expansion, PAD rows carry
                    # no semantics and must not leak into the multivectors)
                    mask = attention_mask[i].bool()
                    seq_hidden = hidden[i][mask]
                    seq_ids = input_ids[i][mask]

                # Filter special tokens if configured, and punctuation for documents
                # Note: For query expansion, MASK tokens are NOT filtered (they're semantic)
                if self._skip_special_tokens or (not is_query and self._doc_skiplist_ids):
                    seq_hidden = self._filter_special_tokens(
                        seq_hidden,
                        seq_ids,
                        keep_mask=use_expansion,
                        is_document=not is_query,
                    )

                multivectors.append(seq_hidden.float().cpu().numpy())

        return multivectors

    def _encode_manual_flash(
        self,
        texts: list[str],
        max_length: int,
        use_expansion: bool,
        prefix_id: int | None = None,
        *,
        is_query: bool = True,
    ) -> list[np.ndarray]:
        """Encode using manual flash attention varlen (for standard BERT models).

        Args:
            texts: Texts to encode.
            max_length: Maximum sequence length.
            use_expansion: Whether to use query expansion (MASK padding to max_length).
            prefix_id: Optional prefix token ID to insert after [CLS].
            is_query: Whether these are queries (affects skiplist filtering).
        """
        # Called from encode() which checks these
        assert self._model is not None
        assert self._tokenizer is not None

        # Adjust max_length if we need to insert a prefix token
        effective_max_length = max_length - 1 if prefix_id is not None else max_length

        # Serialise all tokenizer access (including the transient
        # ``pad_token_id`` swap) against the §7.3 metering tokenization that
        # runs on a separate thread — a bare fast tokenizer would raise
        # ``Already borrowed`` under concurrency (#1800).
        with self._tokenizer_guard():
            # For query expansion, pad to exact max_length using MASK tokens.
            # Restore ``pad_token_id`` in ``finally`` so a mid-comprehension
            # tokenizer failure cannot leave it stuck at the MASK id and leak the
            # corruption to every later guarded caller under ``_tokenizer_guard()``
            # (CodeRabbit #1800).
            if use_expansion:
                original_pad_id = self._tokenizer.pad_token_id
                self._tokenizer.pad_token_id = self._expansion_token_id
                try:
                    encodings = [
                        self._tokenizer(
                            text,
                            max_length=effective_max_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                            return_attention_mask=True,
                        )
                        for text in texts
                    ]
                finally:
                    self._tokenizer.pad_token_id = original_pad_id
            else:
                # Tokenize each sequence individually (no padding)
                encodings = [
                    self._tokenizer(
                        text,
                        max_length=effective_max_length,
                        truncation=True,
                        return_tensors="pt",
                        return_attention_mask=True,
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
            # Build position IDs
            position_ids_packed = self._build_position_ids(cu_seqlens, len(texts))

            # Run embeddings
            hidden = self._run_embeddings(input_ids_packed, position_ids_packed)

            # Run transformer layers with flash attention
            hidden = self._run_transformer_flash(hidden, cu_seqlens, max_seqlen, total_tokens)

            # Apply the trained pylate Dense chain if present, else the single
            # projection layer
            if self._dense_chain is not None:
                hidden = apply_dense_chain(hidden, self._dense_chain)
            elif self._linear is not None:
                hidden = self._linear(hidden)

            # L2 normalize
            if self._normalize:
                hidden = functional.normalize(hidden, p=2, dim=-1)

            # Split back into per-item results
            multivectors = self._split_embeddings(
                hidden, cu_seqlens, seq_lengths, input_ids_packed, keep_mask=use_expansion, is_query=is_query
            )

        return multivectors

    def _filter_special_tokens(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        keep_mask: bool = False,
        is_document: bool = False,
    ) -> torch.Tensor:
        """Filter out special tokens (and optionally punctuation) from hidden states.

        Args:
            hidden: Hidden states tensor.
            input_ids: Input token IDs.
            keep_mask: If True, keep MASK tokens (for query expansion).
            is_document: If True, also filter out punctuation tokens (skiplist).
        """
        assert self._tokenizer is not None

        filter_ids: set[int] = set()

        # Only filter CLS/SEP/PAD if skip_special_tokens is enabled
        if self._skip_special_tokens:
            if self._tokenizer.cls_token_id is not None:
                filter_ids.add(self._tokenizer.cls_token_id)
            if self._tokenizer.sep_token_id is not None:
                filter_ids.add(self._tokenizer.sep_token_id)
            if self._tokenizer.pad_token_id is not None:
                filter_ids.add(self._tokenizer.pad_token_id)

            # For query expansion, keep MASK tokens (they're semantic expansion tokens)
            if keep_mask and self._tokenizer.mask_token_id in filter_ids:
                filter_ids.discard(self._tokenizer.mask_token_id)

        # For documents, also filter punctuation tokens (matches PyLate skiplist)
        if is_document and self._doc_skiplist_ids:
            filter_ids |= self._doc_skiplist_ids

        if not filter_ids:
            return hidden

        mask = torch.tensor(
            [tok_id.item() not in filter_ids for tok_id in input_ids],
            device=hidden.device,
        )
        return hidden[mask]

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score items against a query using MaxSim.

        MaxSim computes the sum of maximum cosine similarities between
        each query token and all document tokens.

        Args:
            query: Query item.
            items: List of items to score against the query.
            instruction: Optional instruction for the query.
            options: Optional options for the query.

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
        if query_output.multivector is None:
            raise RuntimeError("Failed to encode query: no multivector output")
        query_vecs = query_output.multivector[0]  # [num_query_tokens, dim]

        # Encode documents
        doc_output = self.encode(
            items,
            output_types=["multivector"],
            is_query=False,
        )

        # MaxSim over all documents in one padded, masked batched matmul.
        query_tensor = torch.from_numpy(query_vecs).to(self._device)
        assert doc_output.multivector is not None
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

        ColBERT scoring always uses ``output_types=["multivector"]`` and computes
        MaxSim directly. Encode-time runtime ``options`` resolved from the profile
        (e.g. ``muvera``/``output_types``/``output_similarity``) are irrelevant to
        MaxSim, so they are accepted and ignored here rather than routed into
        ``score()``.
        """
        _ = options  # Encode-time options are irrelevant to MaxSim scoring.
        return grouped_score_pairs(self.score, queries, docs, instruction=instruction)

    def _build_position_ids(self, cu_seqlens: torch.Tensor, num_seqs: int) -> torch.Tensor:
        """Build position IDs for packed sequences."""
        return build_position_ids(cu_seqlens)

    def _run_embeddings(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for packed input.

        Supports both BERT (word_embeddings, position_embeddings, LayerNorm) and
        ModernBERT (tok_embeddings, no position embeddings, norm) architectures.
        """
        assert self._model is not None
        embeddings = self._model.embeddings

        # Support both BERT (word_embeddings) and ModernBERT (tok_embeddings)
        if hasattr(embeddings, "word_embeddings"):
            word_emb = embeddings.word_embeddings(input_ids)  # type: ignore
        elif hasattr(embeddings, "tok_embeddings"):
            word_emb = embeddings.tok_embeddings(input_ids)  # type: ignore
        else:
            msg = f"Embeddings layer missing 'word_embeddings' and 'tok_embeddings'. Got: {type(embeddings).__name__}"
            raise AttributeError(msg)

        # Handle position embeddings (BERT has them, ModernBERT uses RoPE in attention)
        if hasattr(embeddings, "position_embeddings"):
            pos_emb = embeddings.position_embeddings(position_ids)  # type: ignore
            # Token type embeddings (BERT only)
            if hasattr(embeddings, "token_type_embeddings"):
                token_type_emb = embeddings.token_type_embeddings(torch.zeros_like(input_ids))  # type: ignore
                hidden = word_emb + pos_emb + token_type_emb
            else:
                hidden = word_emb + pos_emb
        else:
            # ModernBERT: no position embeddings (RoPE in attention layers)
            hidden = word_emb

        # Handle LayerNorm variants (BERT: LayerNorm, ModernBERT: norm)
        if hasattr(embeddings, "LayerNorm"):
            hidden = embeddings.LayerNorm(hidden)  # type: ignore
        elif hasattr(embeddings, "norm"):
            hidden = embeddings.norm(hidden)

        # Handle dropout variants (BERT: dropout, ModernBERT: drop)
        if hasattr(embeddings, "dropout"):
            hidden = embeddings.dropout(hidden)  # type: ignore
        elif hasattr(embeddings, "drop"):
            hidden = embeddings.drop(hidden)  # type: ignore

        return hidden

    def _run_transformer_flash(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        total_tokens: int,
    ) -> torch.Tensor:
        """Run transformer layers using flash_attn_varlen_func.

        Currently supports BERT-style architectures. Add support for other
        architectures (ModernBERT, etc.) as needed.
        """
        assert self._model is not None

        from flash_attn import flash_attn_varlen_func  # ty: ignore[unresolved-import]

        config = self._model.config
        num_heads = config.num_attention_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads
        softmax_scale = 1.0 / (head_dim**0.5)

        for layer_untyped in self._model.encoder.layer:  # type: ignore
            layer = cast("Any", layer_untyped)
            attention = layer.attention.self

            # QKV projections (standard BERT-style)
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

    def _split_embeddings(
        self,
        hidden: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seq_lengths: list[int],
        input_ids: torch.Tensor,
        *,
        keep_mask: bool = False,
        is_query: bool = True,
    ) -> list[np.ndarray]:
        """Split packed embeddings back into per-item arrays.

        Optionally filters out special tokens ([CLS], [SEP], [PAD]) and,
        for documents, punctuation tokens from the skiplist.

        Args:
            hidden: Hidden states tensor.
            cu_seqlens: Cumulative sequence lengths.
            seq_lengths: List of sequence lengths.
            input_ids: Input token IDs.
            keep_mask: If True, keep MASK tokens (for query expansion).
            is_query: Whether these are queries (affects skiplist filtering).
        """
        assert self._tokenizer is not None

        is_document = not is_query
        should_filter = self._skip_special_tokens or (is_document and self._doc_skiplist_ids)

        # Build the set of token IDs to remove
        filter_ids: set[int] = set()
        if should_filter:
            if self._skip_special_tokens:
                if self._tokenizer.cls_token_id is not None:
                    filter_ids.add(self._tokenizer.cls_token_id)
                if self._tokenizer.sep_token_id is not None:
                    filter_ids.add(self._tokenizer.sep_token_id)
                if self._tokenizer.pad_token_id is not None:
                    filter_ids.add(self._tokenizer.pad_token_id)

                # For query expansion, keep MASK tokens (they're semantic expansion tokens)
                if keep_mask and self._tokenizer.mask_token_id in filter_ids:
                    filter_ids.discard(self._tokenizer.mask_token_id)

            # For documents, also filter punctuation tokens (matches PyLate skiplist)
            if is_document and self._doc_skiplist_ids:
                filter_ids |= self._doc_skiplist_ids

        # Build a batched filter mask on GPU (avoids per-token .item() calls)
        if should_filter and filter_ids:
            filter_tensor = torch.tensor(sorted(filter_ids), device=self._device)
            keep_mask_all = ~torch.isin(input_ids, filter_tensor)
        else:
            keep_mask_all = None

        # Transfer cu_seqlens to CPU once
        offsets = cu_seqlens.tolist()

        # Transfer all hidden states to CPU in one go
        hidden_cpu = hidden.float().cpu().numpy()
        if keep_mask_all is not None:
            mask_cpu = keep_mask_all.cpu().numpy()

        results = []
        for i in range(len(seq_lengths)):
            start, end = offsets[i], offsets[i + 1]
            seq_hidden = hidden_cpu[start:end]

            if keep_mask_all is not None:
                seq_hidden = seq_hidden[mask_cpu[start:end]]

            results.append(seq_hidden)

        return results

    def _validate_output_types(self, output_types: list[str]) -> None:
        """Validate that output types are supported."""
        unsupported = set(output_types) - {"multivector"}
        if unsupported:
            msg = f"Unsupported output types: {unsupported}. ColBERTAdapter only supports 'multivector'."
            raise ValueError(msg)

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
                raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="ColBERTAdapter"))

            text = item.text

            # Apply instruction if provided
            if instruction:
                text = f"{instruction} {text}"

            # Apply text prefix only if we don't have a special token ID
            if use_text_prefix:
                text = f"{prefix}{text}"

            texts.append(text)

        return texts

    def get_postprocessors(self) -> dict[str, Any] | None:
        """Return MUVERA postprocessor for converting multivector to dense.

        MUVERA enables using ColBERT embeddings with standard HNSW search
        by converting variable-length per-token embeddings to fixed-dimension
        dense vectors.

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
