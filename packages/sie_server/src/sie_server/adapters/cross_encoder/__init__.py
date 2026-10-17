"""Cross-encoder adapter for reranker models.

This module provides adapters for cross-encoder models used for reranking:
- CrossEncoderAdapter: For sentence-transformers CrossEncoder models (e.g., Jina Rerankers, BGE Rerankers)

Cross-encoders score (query, document) pairs directly, producing relevance scores.
Unlike bi-encoders which encode queries and documents separately, cross-encoders
process them together and are more accurate but less efficient at scale.

Typical usage in a retrieval pipeline:
1. Initial retrieval with bi-encoder (fast, approximate)
2. Reranking top-k results with cross-encoder (accurate, slow)

Performance optimizations:
- Uses SDPA (Scaled Dot-Product Attention) by default for efficient attention
- Supports FP16/BF16 inference for reduced memory and faster computation
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from sentence_transformers import CrossEncoder

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.adapters._types import ERR_NOT_LOADED, ERR_REQUIRES_TEXT, AttnImplementation, ComputePrecision
from sie_server.core.inference_output import ScoreOutput
from sie_server.types.inputs import Item

if TYPE_CHECKING:
    from sie_server.adapters.cross_encoder._lora import CrossEncoderLora
    from sie_server.adapters.lora import LoraRuntimeConfig


class CrossEncoderAdapter(BaseAdapter):
    """Adapter for cross-encoder reranker models.

    Uses the sentence-transformers CrossEncoder class for models like:
    - Jina Rerankers (jinaai/jina-reranker-v2-base-multilingual)
    - BGE Rerankers (BAAI/bge-reranker-v2-m3)
    - MS MARCO cross-encoders

    These models take (query, document) pairs and output relevance scores.
    Higher scores indicate more relevant documents.

    Runtime LoRA is opt-in through ``lora_paths={served_name: source}``.
    Only ordinary A/B LoRA matrices targeting encoder Linear modules in a
    standard, unquantized ``BertForSequenceClassification`` or
    ``XLMRobertaForSequenceClassification`` are supported.
    The scoring head and activation stay those of the base checkpoint;
    trained replacement heads, causal rerankers and custom model classes
    are rejected when LoRA is enabled. Adapters load on demand under the
    registry's resident limit. ``options.lora=None`` selects the base model.
    Configurations without declarations use the original model and scoring path.
    Use an explicit generic CrossEncoder profile; native BertFlash defaults are
    separate implementations and do not gain LoRA support from this adapter.

    Performance optimizations:
    - SDPA attention (default): Uses PyTorch's scaled_dot_product_attention
      which can dispatch to flash attention when available
    - FP16 inference: Reduces memory and improves throughput on GPU
    """

    spec = AdapterSpec(
        inputs=("text",),
        outputs=("score",),
        unload_fields=("_model",),
    )

    def __init__(
        self,
        model_name_or_path: str | Path,
        *,
        trust_remote_code: bool = False,
        max_length: int | None = None,
        compute_precision: ComputePrecision | None = None,
        attn_implementation: AttnImplementation = "sdpa",
        revision: str | None = None,
        lora_paths: dict[str, str] | None = None,
        lora_revisions: dict[str, str | None] | None = None,
        max_loaded_loras: int = 10,
        **kwargs: Any,  # Accept extra args from loader (e.g., pooling)
    ) -> None:
        """Initialize the adapter.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            trust_remote_code: Whether to trust remote code in model files.
            max_length: Override default max sequence length for query+document.
            compute_precision: Compute precision for inference. None (default) selects
                fp32 off-CUDA (safe on MPS) and fp16 on CUDA; set explicitly to override
                (e.g. float16 to opt a curated model into fp16 on MPS).
            attn_implementation: Attention implementation ("sdpa" for optimized, "eager" for baseline).
            revision: Optional HuggingFace revision/branch/commit SHA to pin when
                loading the model. If None, the default branch is used. Forwarded
                to ``CrossEncoder(..., revision=...)``.
            lora_paths: Declared adapter aliases/sources. Enables deferred runtime
                loading for standard BERT/XLM-R encoder projections only. Served names
                use letters, digits, underscores, dots and hyphens; sources may
                be local directories or Hugging Face repositories.
            lora_revisions: Source revisions supplied by the model loader.
            max_loaded_loras: Resident adapter limit, enforced with registry LRU eviction.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        _ = kwargs  # Unused, but accepted for loader compatibility
        self._model_name_or_path = str(model_name_or_path)
        self._trust_remote_code = trust_remote_code
        self._max_length = max_length
        self._compute_precision = compute_precision
        self._attn_implementation = attn_implementation
        self._revision = revision

        self._model: CrossEncoder | None = None
        self._device: str | None = None
        self._lora: CrossEncoderLora | None = None
        if lora_paths:
            # Keep optional dependency imports and all LoRA state off the ordinary path.
            from sie_server.adapters.cross_encoder._lora import CrossEncoderLora as LoraManager

            self._lora = LoraManager(lora_paths, lora_revisions, max_loaded_loras)

    def load(self, device: str) -> None:
        """Load the model onto the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cpu", "mps").
        """
        self._device = device

        # Build model_kwargs for performance optimizations
        model_kwargs: dict[str, Any] = {}

        # Use SDPA for efficient attention on accelerators
        if device.startswith("cuda") or device == "mps":
            # FA2 is CUDA-only; off-CUDA (MPS) coerce anything but "eager" to "sdpa"
            # (mirrors PyTorchEmbeddingAdapter._resolve_dtype_and_attn) so a reranker
            # profile requesting flash_attention_2 still loads on Metal.
            if device.startswith("cuda") or self._attn_implementation == "eager":
                model_kwargs["attn_implementation"] = self._attn_implementation
            else:
                model_kwargs["attn_implementation"] = "sdpa"

            # Resolve precision. Default to fp32 off-CUDA (numerically safe on MPS);
            # honor an explicit compute_precision everywhere so curated Mac models can
            # opt into fp16 on MPS. CUDA keeps the historical fp16 default.
            precision = self._compute_precision
            if precision is None:
                precision = "float16" if device.startswith("cuda") else "float32"
            # NOTE: bf16 is coerced to fp16 on MPS at the loader (MPS bf16 is incomplete
            # in torch and can hang load). See core/loader.load_adapter.
            if precision == "float16":
                model_kwargs["dtype"] = torch.float16
            elif precision == "bfloat16":
                model_kwargs["dtype"] = torch.bfloat16
            # float32 -> CrossEncoder default, no dtype kwarg needed

        self._model = CrossEncoder(
            self._model_name_or_path,
            device=device,
            trust_remote_code=self._trust_remote_code,
            revision=self._revision,
            model_kwargs=model_kwargs or None,
        )

        if self._max_length is not None:
            self._model.max_length = self._max_length
        if self._lora is not None:
            self._lora.bind(self._model)

    def unload(self) -> None:
        try:
            if self._lora is not None:
                self._lora.close()
        finally:
            super().unload()

    def supports_lora(self) -> bool:
        return self._lora is not None

    def supports_hot_lora_reload(self) -> bool:
        return self._lora is not None

    def lora_runtime_config(self) -> "LoraRuntimeConfig | None":
        return self._lora.runtime_config() if self._lora is not None else None

    def fetch_lora(self, lora_path: str, revision: str | None = None) -> str:
        if self._lora is None:
            return super().fetch_lora(lora_path, revision)
        return self._lora.fetch(lora_path, revision)

    def discard_lora_source(self, lora_path: str) -> None:
        if self._lora is not None:
            self._lora.discard(lora_path)

    def load_lora(self, lora_path: str, revision: str | None = None, *, source: str | None = None) -> int:
        if self._lora is None:
            return super().load_lora(lora_path, revision, source=source)
        return self._lora.load(lora_path, revision, source=source)

    def unload_lora(self, lora_name: str) -> None:
        if self._lora is None:
            super().unload_lora(lora_name)
        else:
            self._lora.unload(lora_name)

    def set_active_lora(self, lora_name: str | None) -> None:
        if self._lora is not None:
            self._lora.select(lora_name)
        elif lora_name is not None:
            raise ValueError("No LoRA sources are declared for this cross-encoder")

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score items against a query using the cross-encoder.

        Args:
            query: Query item (must have text).
            items: Items to score against the query.
            instruction: Optional instruction to prepend to query (for instruction-tuned models).

        Returns:
            List of scores, one per item. Higher scores indicate more relevant documents.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If query or items lack text.
        """
        _ = options  # Delegated to self._model.predict() which handles tokenization internally
        self._check_loaded()
        if self._model is None:
            raise RuntimeError(ERR_NOT_LOADED)

        query_text = self._extract_text(query)
        if instruction is not None:
            query_text = f"{instruction} {query_text}"

        # Build (query, document) pairs
        pairs = [(query_text, self._extract_text(item)) for item in items]

        # Score pairs. ``CrossEncoder.predict`` tokenizes ``self._model.tokenizer``
        # internally (fused with the forward, unlike colbert where the two are
        # separable); the §7.3 metering fallback re-tokenizes that same tokenizer
        # on a separate thread-pool thread, so a bare HF fast tokenizer would race
        # into ``RuntimeError("Already borrowed")``. Serialise with the shared
        # ``_tokenizer_guard()`` (class-fix for #1800). The inference executor is
        # single-threaded, so predicts are already serial — the guard adds no GPU
        # throughput cost, it only holds off the concurrent metering re-tokenize.
        with torch.inference_mode(), self._tokenizer_guard():
            scores = self._model.predict(pairs)

        # Convert to Python floats
        return [float(s) for s in scores]

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

        Returns:
            ScoreOutput containing scores for each (query, doc) pair.

        Raises:
            RuntimeError: If model not loaded.
            ValueError: If any item lacks text.
        """
        _ = options  # Delegated to self._model.predict() which handles tokenization internally
        self._check_loaded()
        if self._model is None:
            raise RuntimeError(ERR_NOT_LOADED)

        # Build (query, doc) pairs
        pairs = []
        for query, doc in zip(queries, docs, strict=True):
            query_text = self._extract_text(query)
            if instruction is not None:
                query_text = f"{instruction} {query_text}"
            doc_text = self._extract_text(doc)
            pairs.append((query_text, doc_text))

        # Score pairs. See ``score()``: ``predict`` tokenizes ``self._model``'s
        # tokenizer internally (fused with the forward), and ``_pair_input_token_counts``
        # below re-tokenizes it for §7.3 metering, while the shared metering
        # fallback re-tokenizes it on another thread — all serialised through
        # ``_tokenizer_guard()`` to avoid the ``Already borrowed`` race (#1800).
        # The guard is released between predict and the metering re-tokenize
        # (separate ``with`` blocks); the lock is non-reentrant, so they must not
        # nest.
        with torch.inference_mode(), self._tokenizer_guard():
            scores = self._model.predict(pairs)

        # Convert to float32 numpy array and wrap in ScoreOutput
        import numpy as np

        return ScoreOutput(
            scores=np.asarray(scores, dtype=np.float32),
            input_token_counts=self._pair_input_token_counts(pairs),
        )

    def _metering_tokenizer(self) -> Any | None:
        """Return the CrossEncoder's own tokenizer for §7.3 pair counting.

        Unlike in-tree text adapters, this adapter keeps no ``self._tokenizer``:
        the tokenizer lives inside the sentence-transformers ``CrossEncoder``
        (``self._model``). Sourcing it here lets the shared BaseAdapter counting
        helpers (:meth:`count_pair_input_tokens`, :meth:`_token_counts_or_none`)
        operate over the exact tokenizer the forward pass uses.
        """
        return getattr(self._model, "tokenizer", None)

    def _metering_max_length(self) -> int | None:
        """Resolve the truncation cap ``predict()`` applies, for §7.3 counting.

        ``CrossEncoder.predict`` always tokenizes with ``truncation=True`` at the
        model's ``max_length`` (falling back to the tokenizer's
        ``model_max_length`` when unset). Returning that same cap keeps the
        shared counter's truncation ON and its length equal to the forward
        pass's real joint-pair length — reusing BaseAdapter's tokenizer/count
        core without under- or over-billing long pairs. ``None`` only when no
        real cap exists (the base counter then leaves truncation off, matching
        ``predict`` tokenizing to an effectively-unbounded ``model_max_length``).
        """
        max_length = getattr(self._model, "max_length", None) or self._max_length
        if isinstance(max_length, int) and not isinstance(max_length, bool) and max_length > 0:
            return max_length
        model_max = getattr(self._metering_tokenizer(), "model_max_length", None)
        if isinstance(model_max, int) and not isinstance(model_max, bool) and 0 < model_max <= 100_000:
            return model_max
        return None

    def _pair_input_token_counts(self, pairs: list[tuple[str, str]]) -> list[int] | None:
        """Real per-pair input-token counts for the unit meter (§7.3).

        A cross-encoder tokenizes each (query, doc) pair JOINTLY (query and doc
        packed into one sequence with the model's separators), so the pair's
        billable input tokens is the length of that joint encoding. ``predict``
        owns the forward-pass tokenization opaquely; we recover the count by
        reusing BaseAdapter's shared joint-encoding counter
        (:meth:`_token_counts_or_none`) over the model's own tokenizer
        (:meth:`_metering_tokenizer`) and the same truncation cap ``predict``
        applies (:meth:`_metering_max_length`). Best-effort: any tokenizer quirk
        returns ``None``, so the counts are absent rather than failing the
        score or presenting an approximation as a count.

        Runs inline on the inference-executor thread inside :meth:`score_pairs`,
        re-tokenizing the very tokenizer ``CrossEncoder.predict`` just used (and
        that the §7.3 metering fallback re-tokenizes on a separate thread-pool
        thread for another concurrent request). A bare HF fast tokenizer wraps a
        Rust ``RefCell`` and raises ``RuntimeError("Already borrowed")`` under
        concurrent use, so this re-tokenize is serialised with
        ``_tokenizer_guard()`` — the encode/score-side sibling of the guard the
        shared ``count_pair_input_tokens`` entry point already holds. The lock is
        held only for the pure tokenization (no GPU forward), matching colbert's
        fix for the same class (#1800). NOTE: reached only via ``score_pairs`` —
        never through ``count_pair_input_tokens`` (which already holds the guard
        and routes to ``_token_counts_or_none`` directly), so no non-reentrant
        deadlock.
        """
        tokenizer = self._metering_tokenizer()
        if tokenizer is None:
            return None
        with self._tokenizer_guard():
            return self._token_counts_or_none(
                tokenizer,
                [q for q, _ in pairs],
                [d for _, d in pairs],
                expected_len=len(pairs),
            )

    def _extract_text(self, item: Item) -> str:
        """Extract text from an item."""
        if item.text is None:
            raise ValueError(ERR_REQUIRES_TEXT.format(adapter_name="CrossEncoder adapter"))
        return item.text
