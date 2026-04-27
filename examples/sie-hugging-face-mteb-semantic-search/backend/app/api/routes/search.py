import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import models as db_models
from app.db.schemas import (
    RerankSearchResponse,
    RerankSearchResult,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from app.db.session import get_db
from app.services.chroma import search as chroma_search
from app.services.chroma import search_with_rerank as chroma_search_with_rerank
from app.services.fallback_search import search_local, search_local_with_rerank

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


def _require_storage(db: Session, storage_id: str) -> None:
    storage = (
        db.query(db_models.Storage)
        .filter(db_models.Storage.storage_id == storage_id)
        .one_or_none()
    )
    if storage is None:
        raise HTTPException(
            status_code=404,
            detail=f"Storage id '{storage_id}' not found",
        )


def _load_models(db: Session, storage_id: str) -> list[db_models.Model]:
    storage = (
        db.query(db_models.Storage)
        .filter(db_models.Storage.storage_id == storage_id)
        .one_or_none()
    )
    if storage is None:
        return []
    return (
        db.query(db_models.Model)
        .filter(db_models.Model.storage_id == storage.id)
        .all()
    )


@router.post("/semantic", response_model=SemanticSearchResponse)
def semantic_search(
    payload: SemanticSearchRequest,
    db: Session = Depends(get_db),
):
    """Semantic search over indexed model short descriptions using ChromaDB."""
    _require_storage(db, payload.storage_id)

    try:
        raw = chroma_search(payload.storage_id, payload.query, payload.n_results)
    except Exception as exc:
        logger.warning("Chroma search failed, falling back to local ranking: %s", exc)
        raw = search_local(_load_models(db, payload.storage_id), payload.query, payload.n_results)
        results = [
            SemanticSearchResult(
                hf_id=item["hf_id"],
                distance=item["distance"],
                short_description=item.get("short_description"),
            )
            for item in raw["results"]
        ]
        return SemanticSearchResponse(
            storage_id=payload.storage_id,
            query=payload.query,
            results=results,
        )

    results: list[SemanticSearchResult] = []
    ids = raw.get("ids", [[]])[0]
    distances = raw.get("distances", [[]])[0]
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]

    for entry_id, distance, doc, meta in zip(ids, distances, documents, metadatas):
        hf_id = (meta or {}).get("hf_id") or entry_id.split("::", 1)[0]
        results.append(
            SemanticSearchResult(
                hf_id=hf_id,
                distance=round(distance, 6),
                short_description=doc or None,
            )
        )

    return SemanticSearchResponse(
        storage_id=payload.storage_id,
        query=payload.query,
        results=results,
    )


@router.post("/semantic-rerank", response_model=RerankSearchResponse)
def semantic_search_rerank(
    payload: SemanticSearchRequest,
    db: Session = Depends(get_db),
):
    """Two-stage search: short-description kNN then long-description rerank."""
    _require_storage(db, payload.storage_id)

    try:
        raw = chroma_search_with_rerank(
            payload.storage_id, payload.query, payload.n_results
        )
    except Exception as exc:
        logger.warning("Chroma rerank search failed, falling back to local ranking: %s", exc)
        raw = search_local_with_rerank(
            _load_models(db, payload.storage_id), payload.query, payload.n_results
        )
        results = [
            RerankSearchResult(
                hf_id=item["hf_id"],
                rerank_distance=item["rerank_distance"],
                short_distance=item["short_distance"],
                short_description=item.get("short_description"),
            )
            for item in raw["results"]
        ]
        return RerankSearchResponse(
            storage_id=payload.storage_id,
            query=payload.query,
            results=results,
        )

    results: list[RerankSearchResult] = []
    for item in raw.get("results", []):
        rerank = item.get("rerank_distance")
        results.append(
            RerankSearchResult(
                hf_id=item["hf_id"],
                rerank_distance=round(rerank, 6) if rerank is not None else None,
                short_distance=round(item["short_distance"], 6),
                short_description=item.get("short_description") or None,
            )
        )

    return RerankSearchResponse(
        storage_id=payload.storage_id,
        query=payload.query,
        results=results,
    )
