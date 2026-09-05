from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from sie_server.core.inference_output import ScoreOutput

if TYPE_CHECKING:
    import torch

    from sie_server.types.inputs import Item


class _ScoreFn(Protocol):
    def __call__(
        self,
        query: Item,
        items: list[Item],
        *,
        instruction: str | None = ...,
    ) -> list[float]: ...


# ---------------------------------------------------------------------------
# RoPE utilities (eliminates 7 identical copies)
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    import torch as _torch

    return _torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors.

    Args:
        q: Query tensor ``[total_tokens, num_heads, head_dim]``.
        k: Key tensor ``[total_tokens, num_kv_heads, head_dim]``.
        cos: Cosine part ``[total_tokens, head_dim]``.
        sin: Sine part ``[total_tokens, head_dim]``.

    Returns:
        Rotated query and key tensors.
    """
    cos = cos.unsqueeze(1).to(q.dtype)
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Output type validation (eliminates 9+ copies)
# ---------------------------------------------------------------------------


def validate_output_types(
    output_types: list[str],
    supported: set[str],
    adapter_name: str,
) -> None:
    """Raise ``ValueError`` if any requested output type is unsupported."""
    unsupported = set(output_types) - supported
    if unsupported:
        msg = f"Unsupported output types: {unsupported}. {adapter_name} only supports {supported!r}."
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Text extraction (eliminates 9+ copies)
# ---------------------------------------------------------------------------


def extract_texts(
    items: list[Item],
    instruction: str | None,
    *,
    is_query: bool,
    query_template: str | None = None,
    doc_template: str | None = None,
    err_msg: str = "Item must have text",
) -> list[str]:
    """Extract text from items, applying query/doc templates.

    Args:
        items: List of input items.
        instruction: Optional instruction string.
        is_query: Whether items are queries (selects template).
        query_template: Template for queries, e.g. ``"query: {text}"``.
        doc_template: Template for documents, e.g. ``"passage: {text}"``.
        err_msg: Error message when ``item.text`` is ``None``.

    Returns:
        List of formatted text strings.
    """
    texts: list[str] = []
    template = query_template if is_query else doc_template

    for item in items:
        if item.text is None:
            raise ValueError(err_msg)

        text = item.text

        if template:
            kwargs: dict[str, str] = {"text": text}
            if "{instruction}" in template:
                kwargs["instruction"] = instruction or ""

            try:
                text = template.format(**kwargs)
            except KeyError as exc:
                raise ValueError(f"Unsupported template placeholder: {exc.args[0]!r}") from exc
        elif instruction:
            text = f"{instruction} {text}"

        texts.append(text)
    return texts


def extract_text(item: Item, *, err_msg: str = "Item must have text") -> str:
    """Extract text from a single item (cross-encoder use)."""
    if item.text is None:
        raise ValueError(err_msg)
    return item.text


# ---------------------------------------------------------------------------
# Runtime options resolution (eliminates 5+ copies)
# ---------------------------------------------------------------------------


def resolve_embedding_options(
    options: dict[str, Any] | None,
    *,
    default_normalize: bool,
    default_pooling: str,
    default_query_template: str | None,
    default_doc_template: str | None,
) -> tuple[bool, str, str | None, str | None]:
    """Resolve runtime options with adapter defaults as fallback.

    Returns:
        ``(normalize, pooling, query_template, doc_template)``
    """
    opts = options or {}
    return (
        opts.get("normalize", default_normalize),
        opts.get("pooling", default_pooling),
        opts.get("query_template", default_query_template),
        opts.get("doc_template", default_doc_template),
    )


# ---------------------------------------------------------------------------
# Score-pair grouping (shared by ColBERT-family adapters)
# ---------------------------------------------------------------------------


def grouped_score_pairs(
    score_fn: _ScoreFn,
    queries: list[Item],
    docs: list[Item],
    *,
    instruction: str | None = None,
) -> ScoreOutput:
    """Run a per-query ``score()`` callable over parallel (query, doc) pairs.

    Groups pairs by ``(query.text, query.id, instruction)`` so each unique
    query is encoded once and its docs are scored as one batch. Used by
    ColBERT-family adapters to satisfy the worker's ``score_pairs()``
    contract while reusing the optimized batched ``score()``.

    Queries with ``text is None`` are not supported and raise ``ValueError``
    (ColBERT scoring requires text). The grouping key is
    ``(query.text, query.id or "", instruction or "")`` — two distinct
    ``Item`` objects with identical text/id/instruction collapse to one
    encoding pass.

    Args:
        score_fn: Bound ``adapter.score(query, items, *, instruction=None)``.
        queries: Query items (parallel to docs).
        docs: Document items to score.
        instruction: Optional instruction passed through to ``score_fn``.

    Returns:
        ``ScoreOutput`` with one float per pair, in the original input order.

    Raises:
        ValueError: If ``queries`` and ``docs`` lengths differ, or any query
            lacks text.
    """
    if len(queries) != len(docs):
        msg = f"queries and docs must be parallel; got {len(queries)} vs {len(docs)}"
        raise ValueError(msg)

    if not docs:
        return ScoreOutput(scores=np.zeros(0, dtype=np.float32), batch_size=0)

    groups: dict[tuple[str, str, str], list[int]] = {}
    for i, q in enumerate(queries):
        if q.text is None:
            msg = f"grouped_score_pairs requires queries[{i}].text; got None"
            raise ValueError(msg)
        key = (q.text, q.id or "", instruction or "")
        groups.setdefault(key, []).append(i)

    scores = np.zeros(len(docs), dtype=np.float32)
    for indices in groups.values():
        q = queries[indices[0]]
        group_docs = [docs[i] for i in indices]
        group_scores = score_fn(q, group_docs, instruction=instruction)
        for idx, s in zip(indices, group_scores, strict=True):
            scores[idx] = float(s)

    return ScoreOutput(scores=scores, batch_size=len(docs))
