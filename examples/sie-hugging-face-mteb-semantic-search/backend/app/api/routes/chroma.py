import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import models as db_models
from app.db.session import get_db
from app.services.chroma import reindex_all

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chroma", tags=["chroma"])


class ReindexRequest(BaseModel):
    storage_id: str


class ReindexResponse(BaseModel):
    storage_id: str
    total: int
    indexed: int
    skipped: int


@router.post("/reindex", response_model=ReindexResponse)
def reindex(payload: ReindexRequest, db: Session = Depends(get_db)):
    """Rebuild the Chroma collection for a storage id from current SQLite descriptions."""
    storage = (
        db.query(db_models.Storage)
        .filter(db_models.Storage.storage_id == payload.storage_id)
        .one_or_none()
    )
    if storage is None:
        raise HTTPException(404, f"Storage id '{payload.storage_id}' not found")

    rows = (
        db.query(db_models.Model)
        .filter(db_models.Model.storage_id == storage.id)
        .all()
    )
    if not rows:
        raise HTTPException(404, f"No models found for storage id '{payload.storage_id}'")

    result = reindex_all(payload.storage_id, rows)

    return ReindexResponse(storage_id=payload.storage_id, **result)
