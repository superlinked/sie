"""CLI utility to generate short and long descriptions for models.

Runs the same pipeline as the web UI Generate Descriptions buttons:
  1. Prepare 6K prompt (model metadata + README + MTEB summary)
  2. Generate 6K detailed description via OpenRouter
  3. Generate 2K long description from the 6K output
  4. Generate 200-char short description from the 6K output
  5. Save both descriptions to the database
  6. Upsert embedding into ChromaDB via SIE

Usage:
    python cli_generate.py <storage_id>                  # all models (default 20 parallel)
    python cli_generate.py <storage_id> <hf_id>          # single model
    python cli_generate.py <storage_id> --parallel 50    # 50 parallel pipelines
    python cli_generate.py <storage_id> --model google/gemini-2.5-flash
    python cli_generate.py <storage_id> --skip-existing   # skip models that already have descriptions
    python cli_generate.py <storage_id> --reindex-only    # rebuild Chroma from existing descriptions
"""

import argparse
import asyncio
import json
import logging
import sys
import time

from app.config import settings
from app.db.migrate import ensure_schema
from app.db.models import Model, Storage
from app.db.session import SessionLocal
from app.prompts import load_prompt
from app.services.chroma import reindex_all, upsert_embedding
from app.services.openrouter import generate_text, generate_text_async

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_model_json(model: Model) -> str:
    data = {
        "hf_id": model.hf_id,
        "author": model.author,
        "pipeline_tag": model.pipeline_tag,
        "library_name": model.library_name,
        "downloads_30d": model.downloads_30d,
        "likes": model.likes,
        "tags": model.tags,
        "config": model.config,
        "card_data": model.card_data,
    }
    return json.dumps(data, indent=2, default=str)


def _build_mteb_summary(model: Model) -> str:
    if not model.mteb_scores:
        return "(no MTEB results available)"
    lines: list[str] = []
    for r in model.mteb_scores:
        task_name = r.get("task_name", "unknown")
        main = r.get("main_score")
        if main is None:
            lines.append(f"- {task_name}: (no score)")
            continue
        try:
            score = float(main)
        except (TypeError, ValueError):
            lines.append(f"- {task_name}: (no score)")
            continue
        lines.append(f"- {task_name}: {score:.4f} ({score * 100:.2f}%)")
    return "\n".join(lines)


def _live_readme_for_prompt(hf_id: str, max_chars: int = 4000) -> str:
    """Live-fetch the HF README (truncated) — not stored in SQLite anymore."""
    from huggingface_hub import ModelCard

    try:
        text_value = ModelCard.load(hf_id).text or ""
    except Exception as exc:
        logger.info("Live README fetch failed for %s: %s", hf_id, exc)
        return ""
    return text_value[:max_chars]


def generate_for_model(
    db, model: Model, storage_id: str, llm_model: str | None, dry_run: bool = False
) -> bool:
    """Run the full description pipeline for a single model. Returns True on success."""
    hf_id = model.hf_id
    logger.info("--- Processing %s ---", hf_id)

    model_json = _build_model_json(model)
    mteb_summary = _build_mteb_summary(model)
    model_readme = _live_readme_for_prompt(hf_id)

    # Step 1: Render detailed prompt
    prompt_text = load_prompt(
        "detailed_description",
        model_id=hf_id,
        model_json=model_json,
        model_readme=model_readme,
        mteb_summary=mteb_summary,
    )
    logger.info("[1/6] Prepared 6K prompt (%d chars)", len(prompt_text))

    if dry_run:
        logger.info("[DRY RUN] Skipping LLM calls and save for %s", hf_id)
        return True

    # Step 2: Generate 6K detailed description
    logger.info("[2/6] Generating 6K detailed description …")
    t0 = time.time()
    detailed = generate_text(prompt_text, max_tokens=16384, model=llm_model)
    logger.info("[2/6] Done (%d chars, %.1fs)", len(detailed), time.time() - t0)

    # Step 3: Generate 2K long description
    long_prompt = load_prompt(
        "long_description",
        detailed_description=detailed,
        model_json=model_json,
        mteb_summary=mteb_summary,
    )
    logger.info("[3/6] Generating 2K long description …")
    t0 = time.time()
    long_desc = generate_text(long_prompt, max_tokens=8192, model=llm_model)
    logger.info("[3/6] Done (%d chars, %.1fs)", len(long_desc), time.time() - t0)

    # Step 4: Generate 200-char short description
    short_prompt = load_prompt(
        "short_description",
        detailed_description=detailed,
        model_json=model_json,
        mteb_summary=mteb_summary,
    )
    logger.info("[4/6] Generating 200-char short description …")
    t0 = time.time()
    short_desc = generate_text(short_prompt, max_tokens=4096, model=llm_model)
    logger.info("[4/6] Done (%d chars, %.1fs)", len(short_desc), time.time() - t0)

    # Step 5: Save to database
    model.long_description = long_desc
    model.short_description = short_desc
    db.commit()
    logger.info("[5/6] Saved descriptions for %s", hf_id)
    logger.info("  short (%d chars): %s", len(short_desc), short_desc[:120])
    logger.info("  long  (%d chars): %s…", len(long_desc), long_desc[:120])

    # Step 6: Upsert Chroma embedding
    try:
        upsert_embedding(storage_id, hf_id, short_desc, long_desc)
        logger.info("[6/6] Upserted Chroma embedding for %s", hf_id)
    except Exception as exc:
        logger.warning("Chroma upsert failed for %s: %s", hf_id, exc)

    return True


async def async_generate_for_model(
    model_pk: int,
    storage_id: str,
    llm_model: str | None,
    semaphore: asyncio.Semaphore,
    index: int,
    total: int,
) -> bool:
    """Async version of the per-model pipeline.

    DB sessions are opened only when needed (read and save) so that connections
    are not held during long LLM calls. LLM calls acquire the shared
    *semaphore* so that at most N calls are in-flight across all pipelines.
    """
    # --- Phase 1: read model data, then release the session ---
    db = SessionLocal()
    try:
        model = db.get(Model, model_pk)
        if model is None:
            logger.error("[%d/%d] Model pk=%s vanished from DB", index, total, model_pk)
            return False

        hf_id = model.hf_id
        model_json = _build_model_json(model)
        mteb_summary = _build_mteb_summary(model)
    finally:
        db.close()

    model_readme = _live_readme_for_prompt(hf_id)

    logger.info("[%d/%d] --- Processing %s ---", index, total, hf_id)

    prompt_text = load_prompt(
        "detailed_description",
        model_id=hf_id,
        model_json=model_json,
        model_readme=model_readme,
        mteb_summary=mteb_summary,
    )
    logger.info("[%d/%d] [1/6] Prepared 6K prompt (%d chars)", index, total, len(prompt_text))

    # --- Phase 2: LLM calls (no DB session held) ---

    # Step 2: 6K detailed description
    logger.info("[%d/%d] [2/6] Generating 6K detailed description …", index, total)
    t0 = time.time()
    detailed = await generate_text_async(prompt_text, max_tokens=16384, model=llm_model, semaphore=semaphore)
    logger.info("[%d/%d] [2/6] Done (%d chars, %.1fs)", index, total, len(detailed), time.time() - t0)

    # Step 3: 2K long description
    long_prompt = load_prompt(
        "long_description",
        detailed_description=detailed,
        model_json=model_json,
        mteb_summary=mteb_summary,
    )
    logger.info("[%d/%d] [3/6] Generating 2K long description …", index, total)
    t0 = time.time()
    long_desc = await generate_text_async(long_prompt, max_tokens=8192, model=llm_model, semaphore=semaphore)
    logger.info("[%d/%d] [3/6] Done (%d chars, %.1fs)", index, total, len(long_desc), time.time() - t0)

    # Step 4: 200-char short description
    short_prompt = load_prompt(
        "short_description",
        detailed_description=detailed,
        model_json=model_json,
        mteb_summary=mteb_summary,
    )
    logger.info("[%d/%d] [4/6] Generating 200-char short description …", index, total)
    t0 = time.time()
    short_desc = await generate_text_async(short_prompt, max_tokens=4096, model=llm_model, semaphore=semaphore)
    logger.info("[%d/%d] [4/6] Done (%d chars, %.1fs)", index, total, len(short_desc), time.time() - t0)

    # --- Phase 3: save results, then release the session ---

    # Step 5: DB save (blocking I/O via thread)
    def _save() -> None:
        save_db = SessionLocal()
        try:
            m = save_db.get(Model, model_pk)
            m.long_description = long_desc
            m.short_description = short_desc
            save_db.commit()
        finally:
            save_db.close()

    await asyncio.to_thread(_save)
    logger.info("[%d/%d] [5/6] Saved descriptions for %s", index, total, hf_id)
    logger.info("  short (%d chars): %s", len(short_desc), short_desc[:120])
    logger.info("  long  (%d chars): %s…", len(long_desc), long_desc[:120])

    # Step 6: Chroma upsert (blocking I/O via thread)
    try:
        await asyncio.to_thread(upsert_embedding, storage_id, hf_id, short_desc, long_desc)
        logger.info("[%d/%d] [6/6] Upserted Chroma embedding for %s", index, total, hf_id)
    except Exception as exc:
        logger.warning("Chroma upsert failed for %s: %s", hf_id, exc)

    return True


async def _run_parallel(
    model_pks: list[int],
    storage_id: str,
    llm_model: str | None,
    parallel: int,
) -> tuple[int, int]:
    """Launch all model pipelines with bounded concurrency. Returns (succeeded, failed)."""
    semaphore = asyncio.Semaphore(parallel)
    total = len(model_pks)
    logger.info("Launching %d model pipelines (max %d parallel LLM calls)", total, parallel)

    tasks = [
        asyncio.create_task(
            async_generate_for_model(pk, storage_id, llm_model, semaphore, i, total),
            name=f"model-{i}",
        )
        for i, pk in enumerate(model_pks, 1)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    succeeded = 0
    failed = 0
    for i, result in enumerate(results, 1):
        if isinstance(result, Exception):
            logger.exception("Pipeline %d failed: %s", i, result)
            failed += 1
        elif result:
            succeeded += 1
        else:
            failed += 1

    return succeeded, failed


def main():
    parser = argparse.ArgumentParser(
        description="Generate short and long descriptions for embedding models."
    )
    parser.add_argument(
        "storage_id",
        help="Logical storage id (e.g. test01)",
    )
    parser.add_argument(
        "hf_id",
        nargs="?",
        default=None,
        help="Single Hugging Face model id to process. If omitted, processes all models in the storage.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"OpenRouter model name (default: {settings.openrouter_model})",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=settings.llm_max_parallel,
        help=f"Max parallel LLM pipelines (default: {settings.llm_max_parallel})",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have both short and long descriptions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare prompts but don't call the LLM or save anything.",
    )
    parser.add_argument(
        "--reindex-only",
        action="store_true",
        help="Skip description generation; rebuild the Chroma collection from existing SQLite descriptions.",
    )
    args = parser.parse_args()

    ensure_schema()

    db = SessionLocal()
    try:
        storage = (
            db.query(Storage)
            .filter(Storage.storage_id == args.storage_id)
            .one_or_none()
        )
        if storage is None:
            logger.error("Storage id '%s' not found in database.", args.storage_id)
            sys.exit(1)

        if args.hf_id:
            models = (
                db.query(Model)
                .filter(Model.storage_id == storage.id, Model.hf_id == args.hf_id)
                .all()
            )
            if not models:
                logger.error(
                    "Model '%s' not found in storage '%s'.",
                    args.hf_id,
                    args.storage_id,
                )
                sys.exit(1)
        else:
            models = (
                db.query(Model)
                .filter(Model.storage_id == storage.id)
                .order_by(Model.hf_id)
                .all()
            )

        if args.reindex_only:
            logger.info(
                "Reindex-only mode: rebuilding Chroma collection for '%s' (%d models)",
                args.storage_id,
                len(models),
            )
            result = reindex_all(args.storage_id, models)
            logger.info(
                "Reindex complete: %d indexed, %d skipped out of %d total.",
                result["indexed"],
                result["skipped"],
                result["total"],
            )
            return

        if args.skip_existing:
            before = len(models)
            models = [
                m for m in models
                if not (m.short_description and m.long_description)
            ]
            skipped = before - len(models)
            if skipped:
                logger.info("Skipping %d model(s) that already have descriptions.", skipped)

        logger.info(
            "Processing %d model(s) in storage '%s' with model '%s' (parallel=%d)",
            len(models),
            args.storage_id,
            args.model or settings.openrouter_model,
            args.parallel,
        )

        if args.dry_run:
            for i, model in enumerate(models, 1):
                logger.info("=== Model %d / %d ===", i, len(models))
                generate_for_model(db, model, args.storage_id, args.model, dry_run=True)
            logger.info("Dry run complete for %d model(s).", len(models))
            return

        model_pks = [m.id for m in models]
        wall_t0 = time.time()
        succeeded, failed = asyncio.run(
            _run_parallel(model_pks, args.storage_id, args.model, args.parallel)
        )
        wall_elapsed = time.time() - wall_t0

        logger.info(
            "Done. %d succeeded, %d failed out of %d total (%.1fs wall clock).",
            succeeded,
            failed,
            len(models),
            wall_elapsed,
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
