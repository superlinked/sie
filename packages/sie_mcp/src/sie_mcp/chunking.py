"""Transient text chunking for ``answer_questions``.

SIE has no chunking pipeline today, so this is a deliberately small,
self-contained primitive: a fixed-size sliding character window with overlap. It
runs entirely in the MCP edge per request — chunks are never persisted and no
index is built, preserving a transient retrieval boundary.

Character windows (not model-tokenizer windows) keep the job dependency-free and
predictable; sizes are expressed in characters and approximate a token budget at
the ~4-chars-per-token heuristic the edge already uses elsewhere. Precise token
accounting is the cluster's job, not the edge's.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A single passage carved from one source document.

    ``id`` is stable within a single call (``d{doc}-c{chunk}``) so the reranker
    can echo it back; ``doc_index`` and ``start`` point back to the source so a
    caller can locate the passage in the original text.
    """

    id: str
    text: str
    doc_index: int
    start: int


def chunk_text(text: str, *, window: int, overlap: int) -> list[tuple[int, str]]:
    """Slide a fixed character window over ``text``, yielding ``(start, piece)``.

    ``overlap`` characters are shared between neighbouring windows so a passage
    straddling a boundary still lands whole in at least one window. Returns an
    empty list for empty text.
    """
    if window <= 0:
        msg = f"window must be positive, got {window}"
        raise ValueError(msg)
    if not 0 <= overlap < window:
        msg = f"overlap must satisfy 0 <= overlap < window, got overlap={overlap}, window={window}"
        raise ValueError(msg)
    stride = window - overlap
    windows: list[tuple[int, str]] = []
    start = 0
    length = len(text)
    while start < length:
        windows.append((start, text[start : start + window]))
        if start + window >= length:
            break
        start += stride
    return windows


def chunk_documents(documents: Sequence[str], *, window: int, overlap: int) -> list[Chunk]:
    """Chunk a set of documents into overlapping windows.

    Empty / whitespace-only documents contribute no chunks. Chunk ids are unique
    within the returned list, so a reranker can echo them back unambiguously.
    """
    chunks: list[Chunk] = []
    for doc_index, document in enumerate(documents):
        if not document or not document.strip():
            continue
        for chunk_index, (start, piece) in enumerate(chunk_text(document, window=window, overlap=overlap)):
            chunks.append(Chunk(id=f"d{doc_index}-c{chunk_index}", text=piece, doc_index=doc_index, start=start))
    return chunks
