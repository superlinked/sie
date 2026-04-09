"""Helpers for converting SIE encode results to plain Python types.

These functions bridge SDK result types (numpy arrays, TypedDicts) to the
plain ``list[float]`` / ``dict`` formats that framework integrations and
vector databases expect.

Example:
    >>> from sie_sdk import SIEClient
    >>> from sie_sdk.encoding import dense_embedding, sparse_embedding
    >>>
    >>> client = SIEClient("http://localhost:8080")
    >>> result = client.encode("BAAI/bge-m3", {"text": "hello"}, output_types=["dense", "sparse"])
    >>> vec = dense_embedding(result)  # list[float]
    >>> sp = sparse_embedding(result)  # {"indices": list[int], "values": list[float]}
"""

from __future__ import annotations

from typing import Any

SparseVector = dict[str, list]
"""``{"indices": list[int], "values": list[float]}``"""


def dense_embedding(result: Any, *, strict: bool = True) -> list[float]:
    """Extract the dense embedding from an encode result as ``list[float]``.

    Args:
        result: An :class:`~sie_sdk.EncodeResult` (or any dict/object with
            a ``dense`` attribute).
        strict: If *True* (default), raise :class:`ValueError` when the
            result has no dense embedding.  If *False*, return ``[]``.

    Returns:
        The dense vector as a plain Python list of floats.
    """
    dense = result.get("dense") if isinstance(result, dict) else getattr(result, "dense", None)
    if dense is None:
        if strict:
            msg = "Encode result missing dense embedding"
            raise ValueError(msg)
        return []
    return dense.tolist() if hasattr(dense, "tolist") else list(dense)


def sparse_embedding(result: Any) -> SparseVector:
    """Extract the sparse embedding from an encode result.

    Args:
        result: An :class:`~sie_sdk.EncodeResult` with a ``sparse`` key.

    Returns:
        ``{"indices": list[int], "values": list[float]}``.
        Empty lists if sparse data is absent.
    """
    sparse = result.get("sparse") if isinstance(result, dict) else getattr(result, "sparse", None)
    if sparse is None:
        return {"indices": [], "values": []}

    indices = sparse.get("indices") if isinstance(sparse, dict) else getattr(sparse, "indices", None)
    values = sparse.get("values") if isinstance(sparse, dict) else getattr(sparse, "values", None)

    return {
        "indices": indices.tolist() if hasattr(indices, "tolist") else list(indices or []),
        "values": values.tolist() if hasattr(values, "tolist") else list(values or []),
    }


def sparse_embedding_dict(result: Any) -> dict[int, float]:
    """Extract the sparse embedding as a ``{token_id: weight}`` mapping.

    ChromaDB expects sparse embeddings in this format.

    Args:
        result: An :class:`~sie_sdk.EncodeResult` with a ``sparse`` key.

    Returns:
        Dict mapping integer token indices to float weights.
        Empty dict if sparse data is absent.
    """
    sparse = result.get("sparse") if isinstance(result, dict) else getattr(result, "sparse", None)
    if sparse is None:
        return {}

    indices = sparse.get("indices") if isinstance(sparse, dict) else getattr(sparse, "indices", None)
    values = sparse.get("values") if isinstance(sparse, dict) else getattr(sparse, "values", None)

    if indices is None or values is None:
        return {}

    indices_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    values_list = values.tolist() if hasattr(values, "tolist") else list(values)

    return dict(zip(indices_list, values_list))


def normalize_sparse_vector(sparse: Any) -> SparseVector:
    """Convert a raw sparse sub-object to plain Python lists.

    Unlike :func:`sparse_embedding`, this takes the *sparse value itself*
    (already extracted from the result) -- useful inside named-vector
    loops where you've already pulled ``result["sparse"]``.

    Args:
        sparse: A :class:`~sie_sdk.SparseResult` or dict with
            ``indices`` and ``values``.

    Returns:
        ``{"indices": list[int], "values": list[float]}``.
    """
    indices = sparse.get("indices") if isinstance(sparse, dict) else getattr(sparse, "indices", None)
    values = sparse.get("values") if isinstance(sparse, dict) else getattr(sparse, "values", None)

    if indices is None or values is None:
        return {"indices": [], "values": []}

    return {
        "indices": indices.tolist() if hasattr(indices, "tolist") else list(indices),
        "values": values.tolist() if hasattr(values, "tolist") else list(values),
    }


def multivector_embedding(raw: Any) -> list[list[float]]:
    """Convert a multivector (ColBERT) result to ``list[list[float]]``.

    SIE returns multivectors as a 2-D numpy array of shape
    ``(num_tokens, token_dim)``.  Vector databases expect nested Python lists.

    Args:
        raw: A 2-D numpy array, or a list of 1-D arrays / lists.

    Returns:
        Nested list of float vectors.
    """
    if hasattr(raw, "tolist"):
        return raw.tolist()
    if isinstance(raw, list) and raw and hasattr(raw[0], "tolist"):
        return [v.tolist() for v in raw]
    return [list(v) for v in raw]
