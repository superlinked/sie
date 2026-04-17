from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from sie_server.types.inputs import Item


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
            text = template.format(text=text, instruction=instruction or "")
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
