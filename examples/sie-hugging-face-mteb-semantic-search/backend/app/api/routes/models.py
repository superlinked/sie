import logging
import re
from collections import Counter, OrderedDict
from threading import Lock
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from huggingface_hub import HfApi, ModelCard
from sqlalchemy.orm import Session, load_only

from app.db.session import get_db
from app.db import models as db_models
from app.db.schemas import (
    DownloadRequest,
    DownloadResponse,
    EnrichRequest,
    EnrichResponse,
    ModelOut,
    ModelSummary,
)

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/models", tags=["models"])

_MTEB_NAME_PAREN_RE = re.compile(r"_\(.*\)$")


def _mteb_name_to_hf_id(mteb_name: str) -> str:
    """Convert an MTEB cache folder name like ``org__model-name`` to a HF id ``org/model-name``.

    Strips parenthetical suffixes such as ``_(output_dtype=binary)`` that
    represent MTEB-specific run variants, not separate HF repos.
    """
    clean = _MTEB_NAME_PAREN_RE.sub("", mteb_name)
    return clean.replace("__", "/", 1)


def _get_top_mteb_model_ids(cache, limit: int) -> list[str]:
    """Return the top *limit* HF model ids ranked by number of MTEB benchmark tasks.

    Counting tasks from cache paths is near-instant (no result parsing).
    """
    paths = cache.get_cache_paths()
    model_task_count: Counter[str] = Counter()
    for p in paths:
        parts = p.parts
        try:
            idx = parts.index("results")
            model_task_count[parts[idx + 1]] += 1
        except (ValueError, IndexError):
            pass

    # Deduplicate parenthetical variants (keep the base model id)
    hf_counts: Counter[str] = Counter()
    for mteb_name, count in model_task_count.items():
        hf_id = _mteb_name_to_hf_id(mteb_name)
        hf_counts[hf_id] += count

    return [hf_id for hf_id, _ in hf_counts.most_common(limit)]


def _fetch_hf_model_row(
    api: HfApi,
    hf_id: str,
    storage_pk: int,
) -> db_models.Model | None:
    """Fetch HF metadata for a single model and return an unsaved Model row.

    README is intentionally NOT persisted — it's served live via
    GET /api/models/readme/{hf_id}.
    """
    try:
        m = api.model_info(hf_id, securityStatus=False)
    except Exception as exc:
        logger.warning("HF model_info failed for %s: %s", hf_id, exc)
        return None

    card_data_raw = getattr(m, "card_data", None)
    card_data_data = (
        dict(card_data_raw)
        if card_data_raw and hasattr(card_data_raw, "__iter__")
        else None
    )

    return db_models.Model(
        storage_id=storage_pk,
        hf_id=m.id,
        author=getattr(m, "author", None),
        sha=getattr(m, "sha", None),
        created_at_hf=getattr(m, "created_at", None),
        last_modified=getattr(m, "last_modified", None),
        private=getattr(m, "private", None),
        disabled=getattr(m, "disabled", None),
        downloads=getattr(m, "downloads", None),
        downloads_all_time=getattr(m, "downloads_all_time", None),
        downloads_30d=getattr(m, "downloads_30d", None),
        likes=getattr(m, "likes", None),
        trending_score=getattr(m, "trending_score", None),
        tags=getattr(m, "tags", None),
        pipeline_tag=getattr(m, "pipeline_tag", None),
        library_name=getattr(m, "library_name", None),
        mask_token=getattr(m, "mask_token", None),
        config=getattr(m, "config", None),
        card_data=card_data_data,
    )


@router.post("/download", response_model=DownloadResponse)
def download_models(
    payload: DownloadRequest,
    db: Session = Depends(get_db),
):
    from mteb import ResultCache

    cache = ResultCache()
    logger.info("Downloading / updating MTEB results cache …")
    cache.download_from_remote()
    logger.info("MTEB cache ready.")

    top_hf_ids = _get_top_mteb_model_ids(cache, payload.limit)
    logger.info(
        "Selected %d models from MTEB (by benchmark-task count).", len(top_hf_ids)
    )

    # Get or create storage row
    storage = (
        db.query(db_models.Storage)
        .filter(db_models.Storage.storage_id == payload.storage_id)
        .one_or_none()
    )
    if storage is None:
        storage = db_models.Storage(storage_id=payload.storage_id)
        db.add(storage)
        db.flush()

    # Clear existing models for this storage id for idempotency
    db.query(db_models.Model).filter(
        db_models.Model.storage_id == storage.id
    ).delete()

    api = HfApi()
    stored_models: list[db_models.Model] = []
    for hf_id in top_hf_ids:
        logger.info("Fetching HF metadata for %s …", hf_id)
        model_row = _fetch_hf_model_row(api, hf_id, storage.id)
        if model_row is None:
            continue
        db.add(model_row)
        stored_models.append(model_row)

    db.commit()
    for m in stored_models:
        db.refresh(m)

    enriched, skipped = _enrich_rows_with_mteb(stored_models, cache)
    db.commit()
    for m in stored_models:
        db.refresh(m)

    return DownloadResponse(
        storage_id=payload.storage_id,
        stored_count=len(stored_models),
        models=[ModelOut.model_validate(m) for m in stored_models],
    )


@router.get("/search")
def search_models(
    storage_id: str = Query(..., description="Logical storage id, e.g. test01"),
    hf_id: Optional[str] = Query(
        None, description="Optional Hugging Face model id filter"
    ),
    max_count: int = Query(
        30, ge=1, le=10000, description="Maximum number of items to return"
    ),
    db: Session = Depends(get_db),
):
    """List stored models for a given storage id, optionally filtered by hf_id.

    Returns lightweight summaries; use GET /models/detail for full data.
    """
    storage = (
        db.query(db_models.Storage)
        .filter(db_models.Storage.storage_id == storage_id)
        .one_or_none()
    )
    if storage is None:
        return {"models": []}

    query = db.query(db_models.Model).options(
        load_only(
            db_models.Model.hf_id,
            db_models.Model.author,
            db_models.Model.created_at,
            db_models.Model.downloads_30d,
            db_models.Model.likes,
            db_models.Model.pipeline_tag,
            db_models.Model.short_description,
        )
    ).filter(
        db_models.Model.storage_id == storage.id
    )

    if hf_id:
        like_pattern = f"%{hf_id}%"
        query = query.filter(db_models.Model.hf_id.ilike(like_pattern))

    rows = query.order_by(
        db_models.Model.created_at.desc(),
    ).limit(max_count).all()

    return {"models": [ModelSummary.model_validate(m) for m in rows]}


# --- Live README (fetched from HuggingFace on demand; not stored in SQLite) ---

_README_CACHE_MAX = 64  # roughly one open detail page worth of unique views
_readme_cache: "OrderedDict[str, str]" = OrderedDict()
_readme_cache_lock = Lock()


def _readme_cache_get(hf_id: str) -> Optional[str]:
    with _readme_cache_lock:
        if hf_id in _readme_cache:
            _readme_cache.move_to_end(hf_id)
            return _readme_cache[hf_id]
    return None


def _readme_cache_put(hf_id: str, text_value: str) -> None:
    with _readme_cache_lock:
        _readme_cache[hf_id] = text_value
        _readme_cache.move_to_end(hf_id)
        while len(_readme_cache) > _README_CACHE_MAX:
            _readme_cache.popitem(last=False)


@router.get("/readme/{hf_id:path}")
def get_readme(hf_id: str):
    """Live-fetch a model's README from HuggingFace.

    Not stored in SQLite (too large at 14K-model scale). A small LRU in
    memory avoids hammering HF when a user toggles between views.
    """
    cached = _readme_cache_get(hf_id)
    if cached is not None:
        return {"hf_id": hf_id, "readme": cached, "cached": True}

    from app.db.session import SessionLocal

    with SessionLocal() as db:
        demo_model = (
            db.query(db_models.Model)
            .filter(db_models.Model.hf_id == hf_id)
            .first()
        )
        demo_readme = (demo_model.card_data or {}).get("demo_readme") if demo_model else None
        if demo_readme:
            _readme_cache_put(hf_id, demo_readme)
            return {"hf_id": hf_id, "readme": demo_readme, "cached": True}
    try:
        card = ModelCard.load(hf_id)
        text_value = card.text or ""
    except Exception as exc:
        logger.info("README fetch failed for %s: %s", hf_id, exc)
        raise HTTPException(status_code=404, detail="README not available")
    _readme_cache_put(hf_id, text_value)
    return {"hf_id": hf_id, "readme": text_value, "cached": False}


@router.get("/detail", response_model=ModelOut)
def get_model_detail(
    storage_id: str = Query(..., description="Logical storage id"),
    hf_id: str = Query(..., description="Hugging Face model id"),
    db: Session = Depends(get_db),
):
    """Return full model data for a single model."""
    storage = (
        db.query(db_models.Storage)
        .filter(db_models.Storage.storage_id == storage_id)
        .one_or_none()
    )
    if storage is None:
        raise HTTPException(status_code=404, detail="Storage id not found")

    model = (
        db.query(db_models.Model)
        .filter(
            db_models.Model.storage_id == storage.id,
            db_models.Model.hf_id == hf_id,
        )
        .first()
    )
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelOut.model_validate(model)


def _load_mteb_scores_for_model(
    cache, model_id: str
) -> list[dict] | None:
    """Return compact MTEB scores as ``[{"task_name": str, "main_score": float}, ...]``.

    Collapses every split/subset by averaging ``main_score`` per task. This
    is exactly what the prompt builder and the UI consume — storing
    anything richer is waste.
    """
    try:
        benchmark_results = cache.load_results(
            models=[model_id],
            require_model_meta=False,
        )
    except Exception as exc:
        logger.warning("MTEB load_results failed for %s: %s", model_id, exc)
        return None

    import math

    compact: list[dict] = []
    for model_result in benchmark_results.model_results:
        for task_result in model_result.task_results:
            all_scores: list[float] = []
            for subsets in (task_result.scores or {}).values():
                for s in subsets:
                    main = s.get("main_score")
                    if main is None:
                        continue
                    try:
                        main_f = float(main)
                    except (TypeError, ValueError):
                        continue
                    if math.isnan(main_f) or math.isinf(main_f):
                        continue
                    all_scores.append(main_f)
            if not all_scores:
                continue
            compact.append(
                {
                    "task_name": task_result.task_name,
                    "main_score": sum(all_scores) / len(all_scores),
                }
            )

    return compact or None


def _enrich_rows_with_mteb(
    rows: list[db_models.Model],
    cache=None,
) -> tuple[int, int]:
    """Enrich rows with compact MTEB scores. Returns (enriched, skipped)."""
    if cache is None:
        from mteb import ResultCache

        cache = ResultCache()
        logger.info("Downloading / updating MTEB results cache...")
        cache.download_from_remote()
        logger.info("MTEB cache ready.")

    enriched = 0
    skipped = 0
    for model_row in rows:
        mteb_data = _load_mteb_scores_for_model(cache, model_row.hf_id)
        if mteb_data is not None:
            model_row.mteb_scores = mteb_data
            enriched += 1
        else:
            skipped += 1

    return enriched, skipped


@router.post("/enrich", response_model=EnrichResponse)
def enrich_models(
    payload: EnrichRequest,
    db: Session = Depends(get_db),
):
    """Enrich stored models with MTEB benchmark results.

    Downloads / updates the MTEB results cache, then loads per-task scores
    for every model under the given storage_id.
    """

    storage = (
        db.query(db_models.Storage)
        .filter(db_models.Storage.storage_id == payload.storage_id)
        .one_or_none()
    )
    if storage is None:
        raise HTTPException(
            status_code=404,
            detail=f"Storage id '{payload.storage_id}' not found",
        )

    rows: list[db_models.Model] = (
        db.query(db_models.Model)
        .filter(db_models.Model.storage_id == storage.id)
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No models found for storage id '{payload.storage_id}'",
        )

    enriched_count, skipped_count = _enrich_rows_with_mteb(rows)
    db.commit()

    return EnrichResponse(
        storage_id=payload.storage_id,
        total_models=len(rows),
        enriched_count=enriched_count,
        skipped_count=skipped_count,
    )
