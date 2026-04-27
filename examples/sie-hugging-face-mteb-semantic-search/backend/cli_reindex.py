"""CLI utility to (re)build the Chroma vector index for short descriptions.

Embeds every model's short_description via the SIE encode endpoint and
upserts the vectors into a persistent ChromaDB collection.

Usage:
    python cli_reindex.py <storage_id>
    python cli_reindex.py <storage_id> --batch-size 16
"""

import argparse
import logging
import sys

from app.config import settings
from app.db.models import Model, Storage
from app.db.session import SessionLocal
from app.services.chroma import reindex_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="(Re)build the Chroma vector index for model short descriptions."
    )
    parser.add_argument("storage_id", help="Logical storage id (e.g. test01)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Texts per SIE encode call (default: {settings.sie_embed_batch_size})",
    )
    args = parser.parse_args()

    if args.batch_size is not None:
        settings.sie_embed_batch_size = args.batch_size
        logger.info("Using SIE batch size: %d", args.batch_size)

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

        models = (
            db.query(Model)
            .filter(Model.storage_id == storage.id)
            .order_by(Model.hf_id)
            .all()
        )
        logger.info(
            "Found %d model(s) in storage '%s'", len(models), args.storage_id
        )

        result = reindex_all(args.storage_id, models)
        logger.info(
            "Reindex complete: %d short + %d long indexed, %d skipped (no short_description).",
            result["indexed"],
            result.get("long_indexed", 0),
            result["skipped"],
        )
    finally:
        db.close()


if __name__ == "__main__":
    main()
