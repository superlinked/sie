from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from taxonomy_classification.taxonomy import (
    CategoryPath,
    category_path_from_string,
    full_path,
    leaf_name,
)

DEFAULT_INDEX_DIR = Path("data/chroma")
IndexVariant = Literal["leaf", "full-path"]


@dataclass(frozen=True)
class ChromaMatch:
    category: CategoryPath
    distance: float


def persistent_client(index_dir: Path = DEFAULT_INDEX_DIR) -> ClientAPI:
    return chromadb.PersistentClient(path=str(index_dir))


def collection_name(model: str, variant: IndexVariant) -> str:
    model_name = model.replace("/", "-").replace(":", "-").replace(" ", "-")
    return f"taxonomy-classification-{variant}-{model_name}"


def collection_text(category: CategoryPath, variant: IndexVariant) -> str:
    if variant == "leaf":
        return leaf_name(category)
    return full_path(category)


def has_collection(
    model: str, variant: IndexVariant, index_dir: Path = DEFAULT_INDEX_DIR
) -> bool:
    name = collection_name(model, variant)
    client = persistent_client(index_dir)
    return name in {collection.name for collection in client.list_collections()}


def get_collection(
    model: str, variant: IndexVariant, index_dir: Path = DEFAULT_INDEX_DIR
) -> Collection:
    client = persistent_client(index_dir)
    return client.get_collection(collection_name(model, variant))


def build_index(
    categories: Sequence[CategoryPath],
    embeddings: Sequence[Sequence[float]],
    model: str,
    variant: IndexVariant,
    index_dir: Path = DEFAULT_INDEX_DIR,
    rebuild: bool = False,
) -> Collection:
    client = persistent_client(index_dir)
    name = collection_name(model, variant)

    if rebuild and name in {
        collection.name for collection in client.list_collections()
    }:
        client.delete_collection(name)

    collection = client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=[full_path(category) for category in categories],
        documents=[collection_text(category, variant) for category in categories],
        metadatas=[
            {"path": full_path(category), "level": len(category)}
            for category in categories
        ],
        embeddings=[list(embedding) for embedding in embeddings],
    )
    return collection


def query_collection(
    collection: Collection, query_embedding: Sequence[float], top_k: int
) -> list[ChromaMatch]:
    result = collection.query(
        query_embeddings=[list(query_embedding)],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    matches: list[ChromaMatch] = []

    for metadata, distance in zip(
        result["metadatas"][0], result["distances"][0], strict=True
    ):
        matches.append(
            ChromaMatch(
                category=category_path_from_string(metadata["path"]),
                distance=float(distance),
            )
        )

    return matches


def query_index(
    model: str,
    variant: IndexVariant,
    query_embedding: Sequence[float],
    top_k: int,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> list[ChromaMatch]:
    collection = get_collection(model, variant, index_dir)
    return query_collection(collection, query_embedding, top_k)
