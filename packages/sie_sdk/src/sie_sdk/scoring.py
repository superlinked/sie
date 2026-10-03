"""Scoring utilities for late interaction models (ColBERT-style).

Provides MaxSim computation for client-side scoring when query and document
multivectors are already available (e.g., retrieved from a vector database).

This enables the "encode once, score many" pattern:
1. Encode documents once and store multivectors in a vector DB
2. At query time, encode query and compute MaxSim locally
3. Avoid re-encoding documents for each query

Example:
    >>> from sie_sdk import SIEClient
    >>> from sie_sdk.scoring import maxsim
    >>>
    >>> client = SIEClient("http://localhost:8080")
    >>>
    >>> # Encode query
    >>> query_result = client.encode(
    ...     "jinaai/jina-colbert-v2",
    ...     {"text": "What is ML?"},
    ...     output_types=["multivector"],
    ...     is_query=True,
    ... )
    >>>
    >>> # Assume doc_vectors retrieved from your vector DB
    >>> # doc_vectors: list of np.ndarray, each shape [num_tokens, dim]
    >>>
    >>> # Compute MaxSim scores
    >>> scores = maxsim(query_result["multivector"], doc_vectors)

Naming differs from the TypeScript SDK and the two are not interchangeable:

===========================  =================================================
Python                       TypeScript
===========================  =================================================
``maxsim(q, docs) -> list``  ``maxsimDocuments(q, docs) -> number[]``
(no equivalent)              ``maxsim(q, doc) -> number`` (one document)
``maxsim_batch -> [Q, D]``   ``maxsimBatch -> flat Float32Array`` (length Q*D)
===========================  =================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    FloatMultivector = NDArray[np.float16] | NDArray[np.float32]

_MULTIVECTOR_NDIM = 2


def _as_multivector(array: object, *, label: str) -> np.ndarray:
    """Coerce one multivector to a 2-D float array, or explain what is wrong.

    Callers of this module hold vectors they fetched back out of a vector
    database, so shape faults are the normal failure — a stored row read back
    flat, a corpus encoded with a different model, an empty result set. NumPy
    reports those in terms of its own gufunc signature
    (``(n?,k),(k,m?)->(n?,m?)``), which says nothing about queries, documents,
    or which side was wrong.
    """
    coerced = np.asarray(array, dtype=np.float32)
    if coerced.ndim != _MULTIVECTOR_NDIM:
        msg = (
            f"{label} must be a 2-D array of shape [num_tokens, dim], got shape {coerced.shape}. "
            "A single token vector needs an outer dimension (use `vector[None, :]`); "
            "a list of documents belongs in the `documents` argument, not here."
        )
        raise ValueError(msg)
    if coerced.shape[0] == 0:
        msg = f"{label} has 0 tokens (shape {coerced.shape}); MaxSim is undefined with no tokens to match."
        raise ValueError(msg)
    return coerced


def _check_dim(query: np.ndarray, document: np.ndarray, *, index: int) -> None:
    """Reject a query/document embedding-width mismatch by name."""
    if query.shape[1] != document.shape[1]:
        msg = (
            f"dim mismatch: query has dim {query.shape[1]} but documents[{index}] has dim {document.shape[1]}. "
            "Both sides must come from the same model — check that the stored document vectors "
            "were produced by the model now encoding the query."
        )
        raise ValueError(msg)


def maxsim(
    query: FloatMultivector,
    documents: list[FloatMultivector] | FloatMultivector,
) -> list[float]:
    """Compute MaxSim scores between a query and documents.

    MaxSim is the late interaction scoring function used by ColBERT-style models.
    For each query token, it finds the maximum similarity with any document token,
    then sums these maximums across all query tokens.

    Args:
        query: Float16 or float32 query multivector of shape
            [num_query_tokens, dim].
            Should be L2-normalized (as returned by ColBERT encode).
        documents: Either:
            - A list of float16 or float32 document multivectors, each of shape
              [num_doc_tokens, dim]
            - A single float16 or float32 document multivector of shape
              [num_doc_tokens, dim]

    Returns:
        List of MaxSim scores, one per document.
        Higher scores indicate greater relevance. Similarities and the final
        token sum are accumulated in float32 for both float16 and float32 inputs.

    Raises:
        ValueError: If ``query`` or any document is not a 2-D
            ``[num_tokens, dim]`` array, has zero tokens, or has an embedding
            width that differs from the query's.

    Note:
        The TypeScript SDK's ``maxsim`` scores a *single* document and returns
        one number; its ``maxsimDocuments`` is the analogue of this function.
        The two SDKs' ``maxsimBatch``/``maxsim_batch`` also differ in return
        shape (flat array vs 2-D matrix). Do not port a call between them by
        name alone.

    Example:
        >>> query = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 query tokens
        >>> doc1 = np.array([[1.0, 0.0], [0.5, 0.5]])  # 2 doc tokens
        >>> doc2 = np.array([[0.0, 1.0]])  # 1 doc token
        >>> scores = maxsim(query, [doc1, doc2])
        >>> # scores[0] > scores[1] because doc1 matches both query tokens
    """
    # Handle single document case (2D array = single document)
    doc_list: list[FloatMultivector]
    if isinstance(documents, np.ndarray) and documents.ndim == _MULTIVECTOR_NDIM:
        doc_list = cast("list[FloatMultivector]", [documents])
    elif isinstance(documents, np.ndarray):
        # A 3-D stack is a list of multivectors; anything else (notably a 1-D
        # vector passed unwrapped) is caught per-element by _as_multivector.
        doc_list = list(documents)
    else:
        doc_list = documents

    query_f32 = _as_multivector(query, label="query")
    scores: list[float] = []

    for index, doc in enumerate(doc_list):
        # Cast f16 transport values before matmul so NumPy does not accumulate
        # an entire late-interaction score at f16 precision.
        doc_f32 = _as_multivector(doc, label=f"documents[{index}]")
        _check_dim(query_f32, doc_f32, index=index)

        # Compute all pairwise similarities: [num_query_tokens, num_doc_tokens]
        # This is just matrix multiplication since vectors are L2-normalized.
        sim = np.matmul(query_f32, doc_f32.T)

        # For each query token, find max similarity with any doc token
        max_sims = np.max(sim, axis=-1)  # [num_query_tokens]

        # Sum over query tokens to get final MaxSim score
        score = float(np.sum(max_sims))
        scores.append(score)

    return scores


def maxsim_batch(
    queries: list[FloatMultivector],
    documents: list[FloatMultivector],
) -> NDArray[np.float32]:
    """Compute MaxSim scores for multiple queries against multiple documents.

    This is a batch version of maxsim() for efficiency when scoring
    multiple queries against the same document set.

    Args:
        queries: List of float16 or float32 query multivectors, each of shape
            [num_tokens, dim].
        documents: List of float16 or float32 document multivectors, each of
            shape [num_tokens, dim].

    Returns:
        Score matrix of shape [num_queries, num_documents].
        scores[i, j] is the MaxSim score between query i and document j.
        Similarities and token sums are accumulated in float32.

    Raises:
        ValueError: If any query or document is not a 2-D
            ``[num_tokens, dim]`` array, has zero tokens, or has an embedding
            width that differs from the queries'.

    Note:
        The TypeScript SDK's ``maxsimBatch`` returns a *flat* array of length
        ``num_queries * num_documents``; this returns a 2-D matrix.

    Example:
        >>> queries = [query1, query2]  # 2 queries
        >>> docs = [doc1, doc2, doc3]  # 3 documents
        >>> scores = maxsim_batch(queries, docs)
        >>> scores.shape  # (2, 3)
    """
    num_queries = len(queries)
    num_docs = len(documents)
    scores = np.zeros((num_queries, num_docs), dtype=np.float32)
    queries_f32 = [_as_multivector(query, label=f"queries[{i}]") for i, query in enumerate(queries)]

    # Cast one document at a time so f16-backed corpora are not duplicated in
    # full at f32 precision. Queries are typically the much smaller side and
    # stay cached across the document loop.
    for j, doc in enumerate(documents):
        doc_f32 = _as_multivector(doc, label=f"documents[{j}]")
        for i, query in enumerate(queries_f32):
            _check_dim(query, doc_f32, index=j)
            # Compute pairwise similarities
            sim = np.matmul(query, doc_f32.T)
            # MaxSim: max over doc tokens, sum over query tokens
            scores[i, j] = np.sum(np.max(sim, axis=-1))

    return scores
