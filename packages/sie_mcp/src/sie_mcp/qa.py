"""Transient question answering for ``answer_questions``.

Grounded question answering over a document set too large to drop whole into a
single context window. Composed from shipped SIE primitives and run client-side
in the MCP edge: TRANSIENT retrieval, so nothing is persisted between calls — no
standing index, no cross-call cache, no user-facing search (the M5 boundary).

Per request the chunks are encoded once; then, for each question, the job:

1. ``encode`` the question and rank chunks by dense cosine similarity to
   shortlist the most promising candidates — the cheap first pass that keeps a
   document larger than one context window tractable without reranking every
   chunk;
2. ``score`` (rerank) the shortlist with a cross-encoder to select the final
   passages;
3. generate an answer grounded ONLY in those passages, via a grounded
   chat-completion (system + passages + question).

Each answer is returned with the passages it used, so the grounding is visible.
"""

import math
from collections.abc import Sequence
from typing import Any, Protocol, TypedDict

from sie_mcp.chunking import Chunk, chunk_documents

_SYSTEM = (
    "You answer the question using ONLY the supplied passages. Ground every "
    "statement in them and do not rely on outside knowledge. If the passages do "
    "not contain the answer, say that they do not rather than guessing."
)


class AnswerQuestionsError(Exception):
    """Raised when the QA pipeline cannot produce a grounded answer."""


def _validate_params(
    *,
    top_k: int,
    rerank_candidates: int,
    chunk_chars: int,
    chunk_overlap_chars: int,
    max_tokens: int,
    max_document_chars: int,
    max_questions: int,
    max_chunks: int,
) -> None:
    """Reject invalid retrieval knobs up front, in the job's own error type.

    ``MCPConfig.from_env`` already clamps these for the server path; this guards
    direct callers of :func:`answer_questions` so a bad knob surfaces as
    ``AnswerQuestionsError`` rather than a raw ``ValueError`` from
    ``chunk_documents`` deeper in the pipeline. All checks run before any chunking
    or model call.
    """
    must_be_positive = {
        "top_k": top_k,
        "rerank_candidates": rerank_candidates,
        "chunk_chars": chunk_chars,
        "max_tokens": max_tokens,
        "max_document_chars": max_document_chars,
        "max_questions": max_questions,
        "max_chunks": max_chunks,
    }
    for name, value in must_be_positive.items():
        if value < 1:
            msg = f"{name} must be >= 1, got {value}"
            raise AnswerQuestionsError(msg)
    if not 0 <= chunk_overlap_chars < chunk_chars:
        msg = f"chunk_overlap_chars must satisfy 0 <= overlap < chunk_chars ({chunk_chars}), got {chunk_overlap_chars}"
        raise AnswerQuestionsError(msg)


class QAClient(Protocol):
    """The slice of ``SIEAsyncClient`` this job composes (keeps it unit-testable)."""

    async def encode(self, model: str, items: Any, **kwargs: Any) -> Any:
        """Encode items against an embedding model on the SIE cluster."""

    async def score(self, model: str, query: Any, items: list[Any], **kwargs: Any) -> Any:
        """Rerank items against a query with a reranker model on the SIE cluster."""

    async def chat_completions(self, model: str, messages: list[Any], **kwargs: Any) -> Any:
        """Run an OpenAI-compatible chat completion on the SIE cluster."""


class Passage(TypedDict):
    """A retrieved passage, where it came from, and the score that selected it.

    ``doc_index`` and ``start`` point back to the caller's input — ``doc_index``
    indexes the ``documents`` list, ``start`` is the character offset of the
    passage within that document — so the grounding can be located in the source.
    """

    id: str
    doc_index: int
    start: int
    text: str
    score: float


class Answer(TypedDict):
    """One question, its grounded answer, and the passages it was grounded in."""

    question: str
    answer: str
    passages: list[Passage]


class AnswerQuestionsResult(TypedDict):
    answers: list[Answer]


def _dense_floats(result: Any) -> list[float]:
    dense = result.get("dense") if isinstance(result, dict) else None
    if dense is None:
        msg = "encode result has no dense embedding"
        raise AnswerQuestionsError(msg)
    return [float(x) for x in dense]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _shortlist(question_vec: Sequence[float], chunk_vecs: Sequence[Sequence[float]], *, limit: int) -> list[int]:
    """Rank chunk indices by cosine similarity to the question, highest first."""
    ranked = sorted(
        range(len(chunk_vecs)),
        key=lambda i: _cosine(question_vec, chunk_vecs[i]),
        reverse=True,
    )
    return ranked[:limit]


def _select_passages(score_result: Any, shortlist: Sequence[Chunk], *, top_k: int) -> list[Passage]:
    """Pick the top-``top_k`` reranked passages; ``[]`` if no score maps to a chunk.

    Raises only when the reranker returned *no* scores at all (a genuine
    failure); an id mismatch returns ``[]`` so the caller can fall back to the
    dense shortlist rather than failing the whole question.
    """
    scores = score_result.get("scores") if isinstance(score_result, dict) else None
    if not scores:
        msg = "reranker returned no scores"
        raise AnswerQuestionsError(msg)
    # The reranker contract returns scores sorted by relevance descending; sort
    # defensively so selection is correct even if a backend returns them unsorted.
    ranked = sorted(
        (entry for entry in scores if isinstance(entry, dict)),
        key=lambda entry: float(entry.get("score", 0.0)),
        reverse=True,
    )
    by_id = {chunk.id: chunk for chunk in shortlist}
    passages: list[Passage] = []
    for entry in ranked:
        if len(passages) >= top_k:
            break
        chunk = by_id.get(entry.get("item_id"))
        if chunk is None:
            continue
        passages.append(
            Passage(
                id=chunk.id,
                doc_index=chunk.doc_index,
                start=chunk.start,
                text=chunk.text,
                score=float(entry.get("score", 0.0)),
            )
        )
    return passages


def _format_passages(passages: Sequence[Passage]) -> str:
    blocks = [f"[{i}] (document {passage['doc_index']})\n{passage['text']}" for i, passage in enumerate(passages, 1)]
    return "\n\n".join(blocks)


def _message_content(resp: Any) -> str:
    choices = resp.get("choices") if isinstance(resp, dict) else None
    if not choices:
        msg = "cluster returned no choices for the answer"
        raise AnswerQuestionsError(msg)
    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        msg = "cluster returned no answer content"
        raise AnswerQuestionsError(msg)
    return content


def _dense_passages(
    chunks: Sequence[Chunk],
    chunk_vecs: Sequence[Sequence[float]],
    indices: Sequence[int],
    question_vec: Sequence[float],
) -> list[Passage]:
    return [
        Passage(
            id=chunks[i].id,
            doc_index=chunks[i].doc_index,
            start=chunks[i].start,
            text=chunks[i].text,
            score=_cosine(question_vec, chunk_vecs[i]),
        )
        for i in indices
    ]


async def _answer_one(
    client: QAClient,
    *,
    question: str,
    chunks: Sequence[Chunk],
    chunk_vecs: Sequence[Sequence[float]],
    encode_model: str,
    rerank_model: str,
    generate_model: str,
    top_k: int,
    rerank_candidates: int,
    max_tokens: int,
    gpu: str | None,
) -> Answer:
    # 1. Dense shortlist: cheap bi-encoder pass over every chunk.
    q_encoded = await client.encode(encode_model, {"text": question}, output_types=["dense"], is_query=True, gpu=gpu)
    question_vec = _dense_floats(q_encoded)
    shortlist_idx = _shortlist(question_vec, chunk_vecs, limit=rerank_candidates)
    shortlist = [chunks[i] for i in shortlist_idx]

    # 2. Rerank the shortlist with the cross-encoder to select final passages.
    score_result = await client.score(
        rerank_model,
        {"text": question},
        [{"id": chunk.id, "text": chunk.text} for chunk in shortlist],
        gpu=gpu,
    )
    passages = _select_passages(score_result, shortlist, top_k=top_k)
    if not passages:
        # The reranker returned scores but none mapped to our chunks; fall back to
        # the dense shortlist (already ordered by similarity) so the question still
        # gets a grounded answer instead of failing outright.
        passages = _dense_passages(chunks, chunk_vecs, shortlist_idx[:top_k], question_vec)

    # 3. Generate an answer grounded only in the selected passages.
    user = (
        f"<passages>\n{_format_passages(passages)}\n</passages>\n\n"
        f"Question: {question}\n\n"
        "Answer using only the passages above."
    )
    response = await client.chat_completions(
        generate_model,
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        # Deterministic, grounded answer — no creative sampling.
        temperature=0.0,
        max_completion_tokens=max_tokens,
        gpu=gpu,
    )
    return Answer(question=question, answer=_message_content(response), passages=passages)


async def answer_questions(
    client: QAClient,
    *,
    documents: Sequence[str],
    questions: Sequence[str],
    encode_model: str,
    rerank_model: str,
    generate_model: str,
    top_k: int,
    rerank_candidates: int,
    chunk_chars: int,
    chunk_overlap_chars: int,
    max_tokens: int,
    max_document_chars: int,
    max_questions: int,
    max_chunks: int,
    gpu: str | None = None,
) -> AnswerQuestionsResult:
    """Answer each question grounded in passages retrieved from ``documents``.

    Chunks the documents, encodes the chunks once, then for every question runs
    dense-shortlist → rerank → grounded generation. Nothing is persisted between
    calls and no standing index is built. Returns each answer with the passages
    it was grounded in.

    The inputs are bounded before any chunking or encoding: an oversize corpus,
    too many questions, or too many chunks is rejected up front so a single
    request can't exhaust edge memory or fan out into an unbounded model call.
    """
    _validate_params(
        top_k=top_k,
        rerank_candidates=rerank_candidates,
        chunk_chars=chunk_chars,
        chunk_overlap_chars=chunk_overlap_chars,
        max_tokens=max_tokens,
        max_document_chars=max_document_chars,
        max_questions=max_questions,
        max_chunks=max_chunks,
    )
    if not any(d and d.strip() for d in documents):
        msg = "no non-empty documents supplied"
        raise AnswerQuestionsError(msg)
    # Bound the total text held in memory before chunking it.
    total_chars = sum(len(d) for d in documents)
    if total_chars > max_document_chars:
        msg = f"documents exceed the {max_document_chars}-character limit ({total_chars} supplied)"
        raise AnswerQuestionsError(msg)
    qs = [q for q in questions if q and q.strip()]
    if not qs:
        msg = "no non-empty questions supplied"
        raise AnswerQuestionsError(msg)
    # Each question fans out into its own encode → rerank → generate pass.
    if len(qs) > max_questions:
        msg = f"number of questions ({len(qs)}) exceeds the limit of {max_questions}"
        raise AnswerQuestionsError(msg)

    # Chunk over the original ``documents`` (empties skipped, indices preserved)
    # so every passage's ``doc_index`` points back to the caller's input.
    chunks = chunk_documents(documents, window=chunk_chars, overlap=chunk_overlap_chars)
    if not chunks:
        msg = "documents produced no chunks"
        raise AnswerQuestionsError(msg)
    # Every chunk goes into one encode request; bound that fan-out.
    if len(chunks) > max_chunks:
        msg = f"documents produced {len(chunks)} chunks, which exceeds the limit of {max_chunks}"
        raise AnswerQuestionsError(msg)

    # Encode every chunk once for this call; the vectors live only for its
    # duration — no index is built or stored (transient retrieval).
    encoded = await client.encode(
        encode_model,
        [{"id": chunk.id, "text": chunk.text} for chunk in chunks],
        output_types=["dense"],
        is_query=False,
        gpu=gpu,
    )
    chunk_vecs = [_dense_floats(result) for result in encoded]

    answers = [
        await _answer_one(
            client,
            question=question,
            chunks=chunks,
            chunk_vecs=chunk_vecs,
            encode_model=encode_model,
            rerank_model=rerank_model,
            generate_model=generate_model,
            top_k=top_k,
            rerank_candidates=rerank_candidates,
            max_tokens=max_tokens,
            gpu=gpu,
        )
        for question in qs
    ]
    return AnswerQuestionsResult(answers=answers)
