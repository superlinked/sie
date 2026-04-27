"""Lightweight schema migration for SQLite.

We intentionally avoid Alembic here — the schema changes we make are
additive (``ALTER TABLE ... ADD COLUMN``) and easy enough to manage
in-line. The function is idempotent: it inspects the current columns
and only adds ones that are missing.

Called from:
  - FastAPI startup (``app/main.py``)
  - CLI entrypoints (``cli_download.py``, ``cli_generate.py``) so they
    don't blow up on fresh clones / older DB files.
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text

from app.db.session import Base, engine

logger = logging.getLogger(__name__)


def ensure_schema() -> None:
    """Create tables if missing, then apply additive column migrations."""
    Base.metadata.create_all(bind=engine)

    inspector = inspect(engine)
    if "models" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("models")}
    with engine.begin() as conn:
        if "mteb_scores" not in existing:
            conn.execute(text("ALTER TABLE models ADD COLUMN mteb_scores JSON"))
            logger.info("Migration: added mteb_scores column to models table")
        if "short_description" not in existing:
            conn.execute(
                text("ALTER TABLE models ADD COLUMN short_description VARCHAR(200)")
            )
            logger.info(
                "Migration: added short_description column to models table"
            )
        if "long_description" not in existing:
            conn.execute(
                text("ALTER TABLE models ADD COLUMN long_description VARCHAR(2048)")
            )
            logger.info(
                "Migration: added long_description column to models table"
            )
    # NOTE: SQLite cannot drop columns without a full table rebuild. Any fat
    # legacy fields (readme, siblings, safetensors, spaces, mteb_results) remain
    # in place on pre-migration databases; their rows just get cleared on
    # --overwrite re-ingest. Run ``VACUUM;`` afterwards to reclaim disk space.
