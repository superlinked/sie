"""Local search fallback for demo mode and offline browsing."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.db.models import Model

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return {match.group(0) for match in _WORD_RE.finditer(text.lower())}


def _join_parts(parts: Iterable[str | None]) -> str:
    return " ".join(part for part in parts if part)


def _short_text(model: Model) -> str:
    tags = " ".join(model.tags or [])
    return _join_parts(
        [
            model.hf_id.replace("/", " "),
            model.author,
            model.pipeline_tag,
            tags,
            model.short_description,
        ]
    )


def _long_text(model: Model) -> str:
    return _join_parts([_short_text(model), model.long_description])


def _score(query: str, text: str) -> float:
    if not text.strip():
        return 0.0

    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0

    text_tokens = _tokenize(text)
    overlap = len(query_tokens & text_tokens)
    coverage = overlap / len(query_tokens)
    phrase_bonus = 0.35 if query.lower() in text.lower() else 0.0
    return coverage + phrase_bonus


def _distance(score: float) -> float:
    return round(1.0 / (1.0 + score), 6)


def search_local(models: list[Model], query: str, n_results: int) -> dict:
    ranked = sorted(
        (
            {
                "hf_id": model.hf_id,
                "distance": _distance(_score(query, _short_text(model))),
                "short_description": model.short_description,
                "downloads_30d": model.downloads_30d or 0,
            }
            for model in models
        ),
        key=lambda row: (row["distance"], -row["downloads_30d"], row["hf_id"]),
    )
    return {"results": ranked[:n_results]}


def search_local_with_rerank(models: list[Model], query: str, n_results: int) -> dict:
    first_pass = []
    for model in models:
        short_score = _score(query, _short_text(model))
        long_score = _score(query, _long_text(model)) if model.long_description else None
        first_pass.append(
            {
                "hf_id": model.hf_id,
                "short_distance": _distance(short_score),
                "rerank_distance": _distance(long_score) if long_score is not None else None,
                "short_description": model.short_description,
                "downloads_30d": model.downloads_30d or 0,
            }
        )

    reranked = sorted(
        first_pass,
        key=lambda row: (
            row["rerank_distance"] is None,
            row["rerank_distance"] if row["rerank_distance"] is not None else row["short_distance"],
            row["short_distance"],
            -row["downloads_30d"],
            row["hf_id"],
        ),
    )
    return {"results": reranked[:n_results]}
