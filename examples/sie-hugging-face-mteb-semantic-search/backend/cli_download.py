"""CLI utility to download model metadata from MTEB / HuggingFace into a storage.

By default, **appends** new models without touching existing ones. Use
``--overwrite`` to wipe all existing models first (same behaviour as the
web UI Download button).

HuggingFace metadata fetches run in parallel via a thread pool since the
HfApi / ModelCard calls are synchronous blocking I/O.

Usage:
    python cli_download.py test01                       # top 30 MTEB models, append
    python cli_download.py test01 --limit 100           # top 100
    python cli_download.py test01 --limit 70 --parallel 20
    python cli_download.py test01 --overwrite           # wipe existing models first
    python cli_download.py test01 --dry-run             # show which models would be fetched
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi

from app.config import settings
from app.db.migrate import ensure_schema
from app.db.models import Model, Storage
from app.db.session import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_top_mteb_model_ids(cache, limit: int) -> list[str]:
    """Reuses the same ranking logic as the web API endpoint."""
    from collections import Counter
    import re

    paren_re = re.compile(r"_\(.*\)$")

    paths = cache.get_cache_paths()
    model_task_count: Counter[str] = Counter()
    for p in paths:
        parts = p.parts
        try:
            idx = parts.index("results")
            model_task_count[parts[idx + 1]] += 1
        except (ValueError, IndexError):
            pass

    hf_counts: Counter[str] = Counter()
    for mteb_name, count in model_task_count.items():
        clean = paren_re.sub("", mteb_name)
        hf_id = clean.replace("__", "/", 1)
        hf_counts[hf_id] += count

    return [hf_id for hf_id, _ in hf_counts.most_common(limit)]


def _fetch_hf_model_row(api: HfApi, hf_id: str, storage_pk: int) -> Model | None:
    """Fetch HF metadata for a single model and return an unsaved Model row.

    README is intentionally NOT fetched or stored — it's served live via
    GET /api/models/readme/{hf_id} when the user opens a detail view. This
    keeps SQLite small enough to hold the full 14K-model catalog.
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

    return Model(
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


def _load_mteb_scores_for_model(cache, model_id: str) -> list[dict] | None:
    """Return compact MTEB scores as ``[{"task_name": str, "main_score": float}, ...]``.

    We collapse every split/subset by averaging their ``main_score`` values
    per task. This is exactly what the backend prompt builder and the
    frontend MTEB table consume, so storing anything richer is waste.
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


def main():
    parser = argparse.ArgumentParser(
        description="Download model metadata from MTEB / HuggingFace into a storage."
    )
    parser.add_argument("storage_id", help="Logical storage id (e.g. test01)")
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="How many top MTEB-benchmarked models to consider (default: 30)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=10,
        help="Max parallel HuggingFace API fetches (default: 10)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete all existing models in the storage before downloading (default: append).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which models would be fetched without actually downloading.",
    )
    args = parser.parse_args()

    ensure_schema()

    # --- Step 1: MTEB cache ---
    from mteb import ResultCache

    cache = ResultCache()
    logger.info("Downloading / updating MTEB results cache …")
    cache.download_from_remote()
    logger.info("MTEB cache ready.")

    top_hf_ids = _get_top_mteb_model_ids(cache, args.limit)
    logger.info("Selected %d models from MTEB (by benchmark-task count).", len(top_hf_ids))

    # --- Step 2: DB setup ---
    db = SessionLocal()
    try:
        storage = (
            db.query(Storage)
            .filter(Storage.storage_id == args.storage_id)
            .one_or_none()
        )
        if storage is None:
            storage = Storage(storage_id=args.storage_id)
            db.add(storage)
            db.flush()
            logger.info("Created new storage '%s'.", args.storage_id)

        if args.overwrite:
            deleted = (
                db.query(Model)
                .filter(Model.storage_id == storage.id)
                .delete()
            )
            db.flush()
            logger.info("Overwrite mode: deleted %d existing model(s).", deleted)

        # Filter out models already in this storage
        existing_hf_ids: set[str] = {
            row[0]
            for row in db.query(Model.hf_id)
            .filter(Model.storage_id == storage.id)
            .all()
        }
        new_hf_ids = [hf_id for hf_id in top_hf_ids if hf_id not in existing_hf_ids]
        skipped = len(top_hf_ids) - len(new_hf_ids)
        if skipped:
            logger.info("Skipping %d model(s) already in storage.", skipped)

        logger.info(
            "%d new model(s) to fetch (parallel=%d).", len(new_hf_ids), args.parallel
        )

        if args.dry_run:
            for i, hf_id in enumerate(new_hf_ids, 1):
                logger.info("[DRY RUN] %3d. %s", i, hf_id)
            logger.info("Dry run complete. %d model(s) would be fetched.", len(new_hf_ids))
            return

        if not new_hf_ids:
            logger.info("Nothing to download.")
            return

        # --- Step 3: Parallel HF fetch ---
        api = HfApi(token=settings.hf_token or None)
        stored_models: list[Model] = []
        failed = 0
        wall_t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            future_to_hf_id = {
                pool.submit(_fetch_hf_model_row, api, hf_id, storage.id): hf_id
                for hf_id in new_hf_ids
            }

            for i, future in enumerate(as_completed(future_to_hf_id), 1):
                hf_id = future_to_hf_id[future]
                try:
                    model_row = future.result()
                except Exception:
                    logger.exception("Failed to fetch %s", hf_id)
                    failed += 1
                    continue

                if model_row is None:
                    failed += 1
                    continue

                db.add(model_row)
                stored_models.append(model_row)
                logger.info(
                    "[%d/%d] Fetched %s", i, len(new_hf_ids), hf_id
                )

        db.commit()
        for m in stored_models:
            db.refresh(m)

        fetch_elapsed = time.time() - wall_t0
        logger.info(
            "HF fetch complete: %d stored, %d failed (%.1fs).",
            len(stored_models),
            failed,
            fetch_elapsed,
        )

        # --- Step 4: MTEB enrichment ---
        logger.info("Enriching %d model(s) with MTEB results …", len(stored_models))
        enriched = 0
        mteb_skipped = 0
        for model_row in stored_models:
            mteb_data = _load_mteb_scores_for_model(cache, model_row.hf_id)
            if mteb_data is not None:
                model_row.mteb_scores = mteb_data
                enriched += 1
            else:
                mteb_skipped += 1

        db.commit()
        logger.info(
            "MTEB enrichment: %d enriched, %d skipped (no results).",
            enriched,
            mteb_skipped,
        )

        total_in_storage = (
            db.query(Model)
            .filter(Model.storage_id == storage.id)
            .count()
        )
        logger.info(
            "Done. Storage '%s' now has %d model(s) total.",
            args.storage_id,
            total_in_storage,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
