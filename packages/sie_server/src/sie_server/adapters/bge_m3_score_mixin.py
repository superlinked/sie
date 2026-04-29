from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from sie_server.core.inference_output import ScoreOutput

if TYPE_CHECKING:
    from sie_server.core.inference_output import EncodeOutput
    from sie_server.types.inputs import Item


class BGEM3ScoreMixin:
    """Adds BGE-M3 score()/score_pairs() to any adapter implementing encode().

    BGE-M3 supports four scoring modes composable from any of its encode
    outputs (dense / sparse / multivector). This mixin implements the modes
    once and is consumed by every BGE-M3 adapter (``bge_m3_flash``,
    ``bge_m3``, ``bge_m3_flag``) so they all expose the same ``/v1/score``
    semantics.

    Modes:
      - ``dense``  : cosine similarity between CLS-pooled, L2-normalized vectors.
      - ``sparse`` : ``Σ q_w * d_w`` over shared token ids (BGE-M3 paper /
        FlagEmbedding's ``compute_lexical_matching_score``).
      - ``colbert``: ColBERT-style MaxSim over the multi-vector projection,
        normalized by query length (matches FlagEmbedding's ``colbert_score``).
      - ``hybrid`` : weighted sum (default ``{dense: 0.4, sparse: 0.2, colbert: 0.4}``,
        override via ``options["score_weights"]``).

    Runtime-side companion to ``"score"`` being declared in
    :class:`AdapterSpec.outputs`; the class-level validator at
    :mod:`sie_server.adapters._base_adapter` only checks that ``score`` /
    ``score_pairs`` are overridden on the class — both come from this mixin.

    Subclasses must provide ``encode()`` and ``_check_loaded()`` (the latter
    is supplied by the standard adapter base classes). The TYPE_CHECKING
    stubs below let the type checker resolve ``self.encode`` / ``self._check_loaded``
    inside this module without affecting Python's runtime MRO — they only
    exist for the type checker, never as live attributes.
    """

    if TYPE_CHECKING:

        def encode(
            self,
            items: list[Item],
            output_types: list[str],
            *,
            instruction: str | None = ...,
            is_query: bool = ...,
            prepared_items: Any = ...,
            options: dict[str, Any] | None = ...,
        ) -> EncodeOutput: ...

        def _check_loaded(self) -> None: ...

    # Default hybrid weights from the BGE-M3 paper (Chen et al., 2024).
    _DEFAULT_HYBRID_WEIGHTS: ClassVar[dict[str, float]] = {"dense": 0.4, "sparse": 0.2, "colbert": 0.4}
    _VALID_SCORE_MODES: ClassVar[frozenset[str]] = frozenset({"dense", "sparse", "colbert", "hybrid"})
    _MODE_TO_OUTPUT: ClassVar[dict[str, str]] = {
        "dense": "dense",
        "sparse": "sparse",
        "colbert": "multivector",
    }

    # ------------------------------------------------------------------ public

    def score(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> list[float]:
        """Score items against a query using bi-encoder similarity.

        Args:
            query: Query item.
            items: Document items to score.
            instruction: Optional instruction prepended to query and items.
            options: Runtime options. Recognized keys:
                ``score_mode``: one of
                    ``"dense" | "sparse" | "colbert" | "hybrid"`` (default ``"dense"``).
                ``score_weights``: mapping
                    ``{dense, sparse, colbert} -> float`` for hybrid mode.

        Returns:
            List of scores parallel to ``items``.
        """
        self._check_loaded()
        if not items:
            return []

        score_mode, weights = self._resolve_score_mode(options)
        output_types = self._output_types_for_mode(score_mode, weights)

        query_out = self.encode(
            [query],
            output_types=output_types,
            instruction=instruction,
            is_query=True,
            options=options,
        )
        items_out = self.encode(
            items,
            output_types=output_types,
            instruction=instruction,
            is_query=False,
            options=options,
        )

        return [self._compute_pair_score(query_out, 0, items_out, i, score_mode, weights) for i in range(len(items))]

    def score_pairs(
        self,
        queries: list[Item],
        docs: list[Item],
        *,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ScoreOutput:
        """Score parallel (query, doc) pairs in a single batched encode."""
        self._check_loaded()
        if len(queries) != len(docs):
            msg = f"score_pairs requires equal-length queries and docs, got {len(queries)} queries and {len(docs)} docs"
            raise ValueError(msg)
        if not queries:
            return ScoreOutput(scores=np.empty(0, dtype=np.float32))

        score_mode, weights = self._resolve_score_mode(options)
        output_types = self._output_types_for_mode(score_mode, weights)

        queries_out = self.encode(
            queries,
            output_types=output_types,
            instruction=instruction,
            is_query=True,
            options=options,
        )
        docs_out = self.encode(
            docs,
            output_types=output_types,
            instruction=instruction,
            is_query=False,
            options=options,
        )

        scores = np.asarray(
            [self._compute_pair_score(queries_out, i, docs_out, i, score_mode, weights) for i in range(len(queries))],
            dtype=np.float32,
        )
        return ScoreOutput(scores=scores)

    # ------------------------------------------------------------ option resolve

    def _resolve_score_mode(self, options: dict[str, Any] | None) -> tuple[str, dict[str, float]]:
        """Validate and resolve ``score_mode`` and ``score_weights`` from options."""
        opts = options or {}
        score_mode = opts.get("score_mode", "dense")
        # Validate type before membership (frozenset.__contains__ would raise
        # TypeError on unhashable inputs like list/dict, leaking a 500).
        if not isinstance(score_mode, str) or score_mode not in self._VALID_SCORE_MODES:
            msg = f"Invalid score_mode '{score_mode}'. Expected one of {sorted(self._VALID_SCORE_MODES)}."
            raise ValueError(msg)

        weights = dict(self._DEFAULT_HYBRID_WEIGHTS)
        override = opts.get("score_weights")
        if override is not None:
            if not isinstance(override, dict):
                msg = "score_weights must be a mapping of {dense, sparse, colbert} -> float"
                raise ValueError(msg)
            unknown = set(override) - set(self._DEFAULT_HYBRID_WEIGHTS)
            if unknown:
                msg = f"Unknown score_weights keys: {sorted(unknown)}. Allowed: dense, sparse, colbert"
                raise ValueError(msg)
            for key, value in override.items():
                # bool is a subclass of int — reject it explicitly to avoid silently
                # treating True/False as 1.0/0.0 weights.
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                    msg = f"score_weights['{key}'] must be a non-negative number, got {value!r}"
                    raise ValueError(msg)
                weights[key] = float(value)

        if score_mode == "hybrid":
            total = sum(weights.values())
            if total <= 0:
                msg = "score_weights for hybrid mode must contain at least one positive weight"
                raise ValueError(msg)

        return score_mode, weights

    def _output_types_for_mode(self, score_mode: str, weights: dict[str, float]) -> list[str]:
        """Resolve which encode outputs are needed for the requested score mode."""
        if score_mode == "hybrid":
            # Skip outputs whose weight is zero to save compute.
            return [self._MODE_TO_OUTPUT[mode] for mode in ("dense", "sparse", "colbert") if weights.get(mode, 0.0) > 0]
        return [self._MODE_TO_OUTPUT[score_mode]]

    # ------------------------------------------------------------------ similarity

    def _compute_pair_score(
        self,
        q_out: EncodeOutput,
        q_idx: int,
        d_out: EncodeOutput,
        d_idx: int,
        score_mode: str,
        weights: dict[str, float],
    ) -> float:
        """Compute a single (query, doc) score under the resolved mode."""
        if score_mode == "dense":
            return self._dense_sim(q_out, q_idx, d_out, d_idx)
        if score_mode == "sparse":
            return self._sparse_sim(q_out, q_idx, d_out, d_idx)
        if score_mode == "colbert":
            return self._colbert_sim(q_out, q_idx, d_out, d_idx)
        # hybrid
        score = 0.0
        if weights.get("dense", 0.0) > 0:
            score += weights["dense"] * self._dense_sim(q_out, q_idx, d_out, d_idx)
        if weights.get("sparse", 0.0) > 0:
            score += weights["sparse"] * self._sparse_sim(q_out, q_idx, d_out, d_idx)
        if weights.get("colbert", 0.0) > 0:
            score += weights["colbert"] * self._colbert_sim(q_out, q_idx, d_out, d_idx)
        return float(score)

    @staticmethod
    def _dense_sim(q_out: EncodeOutput, q_idx: int, d_out: EncodeOutput, d_idx: int) -> float:
        """Cosine similarity between dense vectors (normalized inside encode)."""
        if q_out.dense is None or d_out.dense is None:
            msg = "Dense vectors required for dense scoring but missing from encode output"
            raise RuntimeError(msg)
        q = q_out.dense[q_idx]
        d = d_out.dense[d_idx]
        # Defensive normalization in case caller disabled normalize at runtime.
        q_norm = float(np.linalg.norm(q))
        d_norm = float(np.linalg.norm(d))
        if q_norm == 0.0 or d_norm == 0.0:
            return 0.0
        return float(np.dot(q, d) / (q_norm * d_norm))

    @staticmethod
    def _sparse_sim(q_out: EncodeOutput, q_idx: int, d_out: EncodeOutput, d_idx: int) -> float:
        """BGE-M3 lexical-match score: sum of q_w * d_w over shared token ids."""
        if q_out.sparse is None or d_out.sparse is None:
            msg = "Sparse vectors required for sparse scoring but missing from encode output"
            raise RuntimeError(msg)
        q_vec = q_out.sparse[q_idx]
        d_vec = d_out.sparse[d_idx]
        if len(q_vec.indices) == 0 or len(d_vec.indices) == 0:
            return 0.0
        d_lookup = dict(zip(d_vec.indices.tolist(), d_vec.values.tolist(), strict=True))
        total = 0.0
        for tid, q_w in zip(q_vec.indices.tolist(), q_vec.values.tolist(), strict=True):
            d_w = d_lookup.get(tid)
            if d_w is not None:
                total += float(q_w) * float(d_w)
        return float(total)

    @staticmethod
    def _colbert_sim(q_out: EncodeOutput, q_idx: int, d_out: EncodeOutput, d_idx: int) -> float:
        """ColBERT MaxSim: sum over query tokens of max-dot against doc tokens, normalized by query length.

        Matches FlagEmbedding's ``BGEM3FlagModel.colbert_score`` exactly.
        """
        if q_out.multivector is None or d_out.multivector is None:
            msg = "Multivector outputs required for colbert scoring but missing from encode output"
            raise RuntimeError(msg)
        q_mv = q_out.multivector[q_idx]
        d_mv = d_out.multivector[d_idx]
        if q_mv.size == 0 or d_mv.size == 0:
            return 0.0
        # Defensive normalization (multivector is normalized inside encode by default).
        q_norms = np.linalg.norm(q_mv, axis=-1, keepdims=True)
        d_norms = np.linalg.norm(d_mv, axis=-1, keepdims=True)
        q_normed = np.divide(q_mv, q_norms, out=np.zeros_like(q_mv), where=q_norms > 0)
        d_normed = np.divide(d_mv, d_norms, out=np.zeros_like(d_mv), where=d_norms > 0)
        sim = q_normed @ d_normed.T  # [q_len, d_len]
        max_per_query_token = sim.max(axis=-1)
        return float(max_per_query_token.sum() / q_mv.shape[0])
