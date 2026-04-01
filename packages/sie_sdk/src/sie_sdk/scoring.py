"""Scoring utilities for late interaction models (ColBERT-style).

Provides MaxSim computation for client-side scoring when query and document
multivectors are already available (e.g., retrieved from a vector database).

Per DESIGN.md Section 8.4, this enables the "encode once, score many" pattern:
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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def maxsim(
    query: NDArray[np.float32],
    documents: list[NDArray[np.float32]] | NDArray[np.float32],
) -> list[float]:
    """Compute MaxSim scores between a query and documents.

    MaxSim is the late interaction scoring function used by ColBERT-style models.
    For each query token, it finds the maximum similarity with any document token,
    then sums these maximums across all query tokens.

    Args:
        query: Query multivector of shape [num_query_tokens, dim].
            Should be L2-normalized (as returned by ColBERT encode).
        documents: Either:
            - A list of document multivectors, each of shape [num_doc_tokens, dim]
            - A single document multivector of shape [num_doc_tokens, dim]

    Returns:
        List of MaxSim scores, one per document.
        Higher scores indicate greater relevance.

    Example:
        >>> query = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 query tokens
        >>> doc1 = np.array([[1.0, 0.0], [0.5, 0.5]])  # 2 doc tokens
        >>> doc2 = np.array([[0.0, 1.0]])  # 1 doc token
        >>> scores = maxsim(query, [doc1, doc2])
        >>> # scores[0] > scores[1] because doc1 matches both query tokens
    """
    # Handle single document case (2D array = single document)
    multivector_ndim = 2
    doc_list: list[NDArray[np.float32]]
    if isinstance(documents, np.ndarray) and documents.ndim == multivector_ndim:
        doc_list = cast("list[NDArray[np.float32]]", [documents])
    elif isinstance(documents, np.ndarray):
        doc_list = list(documents)
    else:
        doc_list = documents

    scores: list[float] = []

    for doc in doc_list:
        # Compute all pairwise similarities: [num_query_tokens, num_doc_tokens]
        # This is just matrix multiplication since vectors are L2-normalized
        sim = np.matmul(query, doc.T)

        # For each query token, find max similarity with any doc token
        max_sims = np.max(sim, axis=-1)  # [num_query_tokens]

        # Sum over query tokens to get final MaxSim score
        score = float(np.sum(max_sims))
        scores.append(score)

    return scores


def maxsim_batch(
    queries: list[NDArray[np.float32]],
    documents: list[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Compute MaxSim scores for multiple queries against multiple documents.

    This is a batch version of maxsim() for efficiency when scoring
    multiple queries against the same document set.

    Args:
        queries: List of query multivectors, each of shape [num_tokens, dim].
        documents: List of document multivectors, each of shape [num_tokens, dim].

    Returns:
        Score matrix of shape [num_queries, num_documents].
        scores[i, j] is the MaxSim score between query i and document j.

    Example:
        >>> queries = [query1, query2]  # 2 queries
        >>> docs = [doc1, doc2, doc3]  # 3 documents
        >>> scores = maxsim_batch(queries, docs)
        >>> scores.shape  # (2, 3)
    """
    num_queries = len(queries)
    num_docs = len(documents)
    scores = np.zeros((num_queries, num_docs), dtype=np.float32)

    for i, query in enumerate(queries):
        for j, doc in enumerate(documents):
            # Compute pairwise similarities
            sim = np.matmul(query, doc.T)
            # MaxSim: max over doc tokens, sum over query tokens
            scores[i, j] = np.sum(np.max(sim, axis=-1))

    return scores
