"""ChromaDB vector-store helpers for model description embeddings.

A single collection ``models_{storage_id}`` holds **two entries per
model**:

* ``"{hf_id}::short"`` — embedding of ``short_description`` (used for
  the basic semantic search).
* ``"{hf_id}::long"``  — embedding of ``long_description`` (used as the
  rerank stage of the "Search with Reranking" feature).

The two entries are distinguished by a ``kind`` metadata field so
callers can filter to either set.
"""

from __future__ import annotations

import logging
from typing import Sequence

import chromadb

from app.config import settings
from app.db.models import Model
from app.services.sie_chroma import SIEEmbeddingFunction

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 64  # docs per ChromaDB upsert call

KIND_SHORT = "short"
KIND_LONG = "long"
_ID_SEP = "::"


def _entry_id(hf_id: str, kind: str) -> str:
    return f"{hf_id}{_ID_SEP}{kind}"


def _get_client() -> chromadb.ClientAPI:
    path = str(settings.chroma_path.resolve())
    return chromadb.PersistentClient(path=path)


def _collection_name(storage_id: str) -> str:
    return f"models_{storage_id}"


def _get_collection(
    client: chromadb.ClientAPI,
    storage_id: str,
) -> chromadb.Collection:
    ef = SIEEmbeddingFunction()
    return client.get_or_create_collection(
        name=_collection_name(storage_id),
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _base_metadata(m: Model) -> dict:
    return {
        "hf_id": m.hf_id,
        "author": m.author or "",
        "pipeline_tag": m.pipeline_tag or "",
        "downloads_30d": m.downloads_30d or 0,
        "likes": m.likes or 0,
    }


def reindex_all(storage_id: str, models: Sequence[Model]) -> dict:
    """(Re)build the Chroma collection for *storage_id*.

    Drops any existing collection first so stale entries (including
    pre-migration plain-``hf_id`` ids) are removed cleanly, then
    upserts both a ``::short`` and a ``::long`` entry for every model
    that has the corresponding description.
    """
    client = _get_client()
    name = _collection_name(storage_id)
    try:
        client.delete_collection(name=name)
        logger.info("Dropped existing collection '%s' for clean reindex", name)
    except Exception:
        # Collection may not exist yet — that's fine.
        pass

    collection = _get_collection(client, storage_id)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    short_count = 0
    long_count = 0
    skipped_no_short = 0

    for m in models:
        base_meta = _base_metadata(m)

        short_text = (m.short_description or "").strip()
        if short_text:
            ids.append(_entry_id(m.hf_id, KIND_SHORT))
            documents.append(short_text)
            metadatas.append({**base_meta, "kind": KIND_SHORT})
            short_count += 1
        else:
            logger.warning("Skipping %s — no short_description", m.hf_id)
            skipped_no_short += 1

        long_text = (m.long_description or "").strip()
        if long_text:
            ids.append(_entry_id(m.hf_id, KIND_LONG))
            documents.append(long_text)
            metadatas.append({**base_meta, "kind": KIND_LONG})
            long_count += 1

    total_entries = len(ids)
    logger.info(
        "Upserting %d entries (%d short, %d long) into collection '%s'",
        total_entries, short_count, long_count, name,
    )

    upserted = 0
    for start in range(0, total_entries, _UPSERT_BATCH):
        end = min(start + _UPSERT_BATCH, total_entries)
        batch_ids = ids[start:end]
        batch_docs = documents[start:end]
        batch_meta = metadatas[start:end]

        logger.info("Upsert batch %d-%d / %d", start + 1, end, total_entries)
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta,
        )
        upserted += len(batch_ids)

    logger.info(
        "Done — %d entries upserted (%d short, %d long).",
        upserted, short_count, long_count,
    )
    return {
        "indexed": short_count,
        "long_indexed": long_count,
        "skipped": skipped_no_short,
        "total": len(models),
    }


def upsert_embedding(
    storage_id: str, hf_id: str, short_desc: str | None, long_desc: str | None
) -> None:
    """Upsert a single model's short and long description embeddings.

    Called from ``/generate/save`` and ``cli_generate.py`` after every
    description save so the vector store stays in sync with SQLite.
    Empty/missing descriptions cause that ``kind`` to be deleted (so a
    cleared description doesn't leave a stale embedding behind).
    """
    client = _get_client()
    collection = _get_collection(client, storage_id)

    base_meta = {"hf_id": hf_id}

    short_text = (short_desc or "").strip()
    long_text = (long_desc or "").strip()

    upsert_ids: list[str] = []
    upsert_docs: list[str] = []
    upsert_meta: list[dict] = []
    delete_ids: list[str] = []

    if short_text:
        upsert_ids.append(_entry_id(hf_id, KIND_SHORT))
        upsert_docs.append(short_text)
        upsert_meta.append({**base_meta, "kind": KIND_SHORT})
    else:
        delete_ids.append(_entry_id(hf_id, KIND_SHORT))

    if long_text:
        upsert_ids.append(_entry_id(hf_id, KIND_LONG))
        upsert_docs.append(long_text)
        upsert_meta.append({**base_meta, "kind": KIND_LONG})
    else:
        delete_ids.append(_entry_id(hf_id, KIND_LONG))

    if upsert_ids:
        collection.upsert(
            ids=upsert_ids,
            documents=upsert_docs,
            metadatas=upsert_meta,
        )
    if delete_ids:
        try:
            collection.delete(ids=delete_ids)
        except Exception as exc:
            # Deleting non-existent ids is fine; log anything else.
            logger.debug("Chroma delete (%s) ignored: %s", delete_ids, exc)


def search(storage_id: str, query: str, n_results: int = 10) -> dict:
    """Semantic search over indexed short descriptions."""
    client = _get_client()
    collection = _get_collection(client, storage_id)
    return collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"kind": KIND_SHORT},
    )


def search_with_rerank(
    storage_id: str, query: str, n_results: int = 20
) -> dict:
    """Two-stage search: short-description kNN, then long-description rerank.

    1. Run a kNN query against the short-description vectors to get
       up to ``n_results`` candidate models.
    2. Re-query the same collection restricted to the long-description
       vectors of those candidates, so Chroma recomputes cosine
       distance between the query and each candidate's long vector.

    Returns a dict with ``results`` — a list of items in the reranked
    order, each containing ``hf_id``, ``rerank_distance``,
    ``short_distance`` (from the first pass) and ``short_description``.
    Models that have no long-description embedding fall back to their
    short-description distance and are appended after the reranked set.
    """
    client = _get_client()
    collection = _get_collection(client, storage_id)

    short = collection.query(
        query_texts=[query],
        n_results=n_results,
        where={"kind": KIND_SHORT},
    )

    short_ids = short.get("ids", [[]])[0]
    short_distances = short.get("distances", [[]])[0]
    short_documents = short.get("documents", [[]])[0]
    short_metadatas = short.get("metadatas", [[]])[0]

    by_hf: dict[str, dict] = {}
    for entry_id, dist, doc, meta in zip(
        short_ids, short_distances, short_documents, short_metadatas
    ):
        hf_id = (meta or {}).get("hf_id") or entry_id.split(_ID_SEP, 1)[0]
        by_hf[hf_id] = {
            "hf_id": hf_id,
            "short_distance": float(dist),
            "short_description": doc,
            "rerank_distance": None,
        }

    if by_hf:
        long_ids_filter = list(by_hf.keys())
        long = collection.query(
            query_texts=[query],
            n_results=len(long_ids_filter),
            where={
                "$and": [
                    {"kind": KIND_LONG},
                    {"hf_id": {"$in": long_ids_filter}},
                ]
            },
        )
        long_distances = long.get("distances", [[]])[0]
        long_metadatas = long.get("metadatas", [[]])[0]
        long_ids = long.get("ids", [[]])[0]

        for entry_id, dist, meta in zip(long_ids, long_distances, long_metadatas):
            hf_id = (meta or {}).get("hf_id") or entry_id.split(_ID_SEP, 1)[0]
            if hf_id in by_hf:
                by_hf[hf_id]["rerank_distance"] = float(dist)

    reranked = [v for v in by_hf.values() if v["rerank_distance"] is not None]
    reranked.sort(key=lambda v: v["rerank_distance"])

    fallback = [v for v in by_hf.values() if v["rerank_distance"] is None]
    fallback.sort(key=lambda v: v["short_distance"])

    return {"results": reranked + fallback}
