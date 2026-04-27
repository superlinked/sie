"""Seed a small local demo dataset so the app works without SIE or OpenRouter."""

from __future__ import annotations

import json
from pathlib import Path

from app.db import models as db_models
from app.db.migrate import ensure_schema
from app.db.session import SessionLocal

DEFAULT_STORAGE_ID = "demo"
DATA_PATH = Path(__file__).with_name("demo_models.json")


def main() -> None:
    ensure_schema()
    records = json.loads(DATA_PATH.read_text())

    with SessionLocal() as db:
        storage = (
            db.query(db_models.Storage)
            .filter(db_models.Storage.storage_id == DEFAULT_STORAGE_ID)
            .one_or_none()
        )
        if storage is None:
            storage = db_models.Storage(
                storage_id=DEFAULT_STORAGE_ID,
                description="Bundled demo catalog for local browsing and search",
            )
            db.add(storage)
            db.flush()

        db.query(db_models.Model).filter(
            db_models.Model.storage_id == storage.id
        ).delete()

        for record in records:
            db.add(
                db_models.Model(
                    storage_id=storage.id,
                    hf_id=record["hf_id"],
                    author=record.get("author"),
                    downloads_30d=record.get("downloads_30d"),
                    likes=record.get("likes"),
                    pipeline_tag=record.get("pipeline_tag"),
                    tags=record.get("tags"),
                    card_data={"demo_readme": record.get("demo_readme", "")},
                    mteb_scores=record.get("mteb_scores"),
                    short_description=record.get("short_description"),
                    long_description=record.get("long_description"),
                )
            )

        db.commit()

    print(f"Seeded {len(records)} demo models under storage id '{DEFAULT_STORAGE_ID}'.")


if __name__ == "__main__":
    main()
