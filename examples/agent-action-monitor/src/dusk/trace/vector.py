"""SIE-backed similarity with deterministic offline fallbacks."""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

from dusk.config import Config, get_config
from dusk.trace.models import TraceDecision

logger = logging.getLogger(__name__)

#: Default zero-shot labels for pulling privileged terms out of an action.
DEFAULT_EXTRACT_LABELS = ["role", "privilege", "resource", "segment", "port"]

#: Per-call provisioning bound; one gate request can make several sequential SIE calls.
_PROVISION_TIMEOUT_S = 1.5


def _sigmoid(x: float) -> float:
    """Bound a raw cross-encoder logit into [0, 1]; monotonic, not a calibrated probability."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


@dataclass
class ExtractedTerm:
    """One entity GLiNER pulled out of an action's text, with its confidence."""

    text: str
    label: str
    score: float


@dataclass
class SimilarDecision:
    id: str
    agent_id: str
    action: str
    similarity: float
    verdict: str
    score: int


#: Conservative label for records created before verdict persistence existed.
_UNKNOWN_VERDICT_FALLBACK = "WOULD-BLOCK"


def _sie_client(config: Config) -> Any | None:  # noqa: ANN401
    """Return a constructed SIEClient if the sie-sdk package is installed, else None."""
    try:
        from sie_sdk import SIEClient  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        api_key = os.getenv("SIE_API_KEY") or None
        return SIEClient(
            config.sie_endpoint.rstrip("/"),
            api_key=api_key,
            timeout_s=config.sie_timeout_ms / 1000,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("SIE client construction failed: %s", exc)
        return None


def sie_encode(text: str, config: Config | None = None) -> list[float] | None:
    """encode: text -> dense vector via SIE. Returns None to trigger the n-gram fallback."""
    cfg = config or get_config()
    client = _sie_client(cfg)
    if client is None:
        return None
    try:
        from sie_sdk.types import Item  # type: ignore[import-not-found]

        # Cold or saturated models must not stall the inline gate.
        result = client.encode(
            cfg.sie_encode_model,
            Item(text=text),
            wait_for_capacity=False,
            provision_timeout_s=_PROVISION_TIMEOUT_S,
        )
        dense = result["dense"] if isinstance(result, dict) else getattr(result, "dense", None)
        return [float(v) for v in dense] if dense is not None else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("SIE encode failed: %s", exc)
        return None


def sie_score(
    query: str, candidates: list[str], config: Config | None = None
) -> list[float] | None:
    """score: rerank candidates against query via SIE's cross-encoder.

    Returns one [0, 1]-bounded score per candidate (see :func:`_sigmoid`) in
    the same order as ``candidates`` (never reordered), or None when SIE is
    unavailable or there are no candidates to score.
    """
    if not candidates:
        return None
    cfg = config or get_config()
    client = _sie_client(cfg)
    if client is None:
        return None
    try:
        from sie_sdk.types import Item  # type: ignore[import-not-found]

        query_item = Item(text=query)
        candidate_items = [Item(text=text, id=str(i)) for i, text in enumerate(candidates)]
        result = client.score(
            cfg.sie_score_model,
            query_item,
            candidate_items,
            wait_for_capacity=False,
            provision_timeout_s=_PROVISION_TIMEOUT_S,
        )
        entries = result["scores"] if isinstance(result, dict) else getattr(result, "scores", None)
        if not entries:
            return None
        score_by_id = {str(e["item_id"]): _sigmoid(float(e["score"])) for e in entries}
        return [score_by_id.get(str(i), 0.0) for i in range(len(candidates))]
    except Exception as exc:  # noqa: BLE001
        logger.warning("SIE score failed: %s", exc)
        return None


def sie_extract(
    text: str, labels: list[str] | None = None, config: Config | None = None
) -> list[ExtractedTerm]:
    """extract: pull entities / privileged terms from text via SIE's GLiNER model.

    Returns an empty list when SIE is unavailable, never raises. Each result
    keeps its label and confidence score so callers can weight by how
    confident this zero-shot extraction actually was, rather than treating
    every hit as equally privileged.
    """
    cfg = config or get_config()
    client = _sie_client(cfg)
    if client is None:
        return []
    try:
        from sie_sdk.types import Item  # type: ignore[import-not-found]

        item_labels = labels or DEFAULT_EXTRACT_LABELS
        result = client.extract(
            cfg.sie_extract_model,
            Item(text=text),
            labels=item_labels,
            wait_for_capacity=False,
            provision_timeout_s=_PROVISION_TIMEOUT_S,
        )
        entities = (
            result["entities"] if isinstance(result, dict) else getattr(result, "entities", None)
        )
        if not entities:
            return []
        extracted: list[ExtractedTerm] = []
        for e in entities:
            if e is None:
                continue
            if isinstance(e, dict):
                extracted.append(
                    ExtractedTerm(
                        text=str(e["text"]),
                        label=str(e.get("label", "")),
                        score=float(e.get("score", 0.0)),
                    )
                )
            else:
                extracted.append(
                    ExtractedTerm(
                        text=str(e.text),
                        label=str(getattr(e, "label", "")),
                        score=float(getattr(e, "score", 0.0)),
                    )
                )
        return extracted
    except Exception as exc:  # noqa: BLE001
        logger.warning("SIE extract failed: %s", exc)
        return []


def _stable_hash(token: str) -> int:
    """Return a process-stable token hash for persisted fallback embeddings."""
    return int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)


def _ngram_fallback(text: str, dims: int = 64) -> list[float]:
    vec = [0.0] * dims
    for token in text.lower().split():
        vec[_stable_hash(token) % dims] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / mag if mag else 0.0


def embed_text(text: str, config: Config | None = None) -> list[float]:
    """Return a dense vector for text via SIE encode, or the n-gram fallback."""
    return sie_encode(text, config) or _ngram_fallback(text)


def _rank_candidates(
    query: str,
    scored: list[tuple[float, TraceDecision]],
    top_k: int,
) -> list[SimilarDecision]:
    """Shared top_k selection + optional SIE rerank pass, given pre-scored candidates."""
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_k]

    # Prefer the cross-encoder ordering when SIE can rerank the shortlist.
    rerank = sie_score(query, [f"{d.agent_id} {d.action}" for _, d in top])
    if rerank and len(rerank) == len(top):
        reranked = sorted(zip(rerank, top, strict=True), key=lambda x: x[0], reverse=True)
        top = [t for _, t in reranked]

    return [
        SimilarDecision(
            id=d.id,
            agent_id=d.agent_id,
            action=d.action,
            similarity=round(sim, 3),
            verdict=d.verdict or _UNKNOWN_VERDICT_FALLBACK,
            score=d.score,
        )
        for sim, d in top
    ]


def find_similar(
    target_action: str,
    target_agent: str,
    past_decisions: list[TraceDecision],
    top_k: int = 3,
) -> list[SimilarDecision]:
    """Return the top_k most similar past decisions to a new one.

    Re-embeds every past decision on every call -- fine for an occasional,
    small lookup, but O(len(past_decisions)) SIE encode calls. A live
    request path with growing history should use :func:`find_similar_cached`
    instead, which embeds each decision once at record time.

    Returns an empty list when fewer than 2 past decisions exist.
    Never raises -- worst case is an empty list.
    """
    if len(past_decisions) < 2:
        return []

    query = f"{target_agent} {target_action}"
    query_vec = embed_text(query)

    scored: list[tuple[float, TraceDecision]] = []
    for d in past_decisions:
        candidate = f"{d.agent_id} {d.action} {d.reasoning}"
        candidate_vec = embed_text(candidate)
        sim = _cosine(query_vec, candidate_vec)
        if sim > 0.3:
            scored.append((sim, d))

    return _rank_candidates(query, scored, top_k)


def find_similar_cached(
    target_action: str,
    target_agent: str,
    past_decisions: list[tuple[TraceDecision, list[float]]],
    top_k: int = 3,
) -> list[SimilarDecision]:
    """Like find_similar, but each past decision already carries its embedding.

    Costs exactly one SIE encode call (the query) plus an optional rerank on
    the shortlist, independent of how much history exists -- unlike
    find_similar, which re-embeds the entire history on every call. Intended
    for a live request path where each decision's vector is computed once,
    when it's recorded, and reused for every future lookup.

    Returns an empty list when fewer than 2 past decisions exist.
    """
    if len(past_decisions) < 2:
        return []

    query = f"{target_agent} {target_action}"
    query_vec = embed_text(query)

    scored: list[tuple[float, TraceDecision]] = []
    for decision, vec in past_decisions:
        sim = _cosine(query_vec, vec)
        if sim > 0.3:
            scored.append((sim, decision))

    return _rank_candidates(query, scored, top_k)
