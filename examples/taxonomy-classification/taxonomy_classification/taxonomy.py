from __future__ import annotations

from pathlib import Path
from typing import Sequence

import polars as pl

CATEGORY_SEPARATOR = " > "
DEFAULT_TAXONOMY_PATH = Path("data/shopify-taxonomy-l3.parquet")
CategoryPath = tuple[str, ...]


def category_path(nodes: Sequence[str]) -> CategoryPath:
    return tuple(nodes)


def category_path_from_string(path: str) -> CategoryPath:
    return tuple(path.split(CATEGORY_SEPARATOR))


def load_taxonomy(path: Path = DEFAULT_TAXONOMY_PATH) -> list[CategoryPath]:
    frame = pl.read_parquet(path, columns=["category"])
    return [category_path(nodes) for nodes in frame["category"].to_list()]


def leaf_name(category: CategoryPath) -> str:
    return category[-1]


def full_path(category: CategoryPath) -> str:
    return CATEGORY_SEPARATOR.join(category)


def ancestor_set(category: CategoryPath) -> set[CategoryPath]:
    return {category[:level] for level in range(1, len(category) + 1)}


def level(category: CategoryPath) -> int:
    return len(category)


def level_lookup(categories: list[CategoryPath]) -> dict[CategoryPath, int]:
    return {category: level(category) for category in categories}
