from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl

from taxonomy_classification.taxonomy import CategoryPath, category_path

REQUIRED_COLUMNS = [
    "product_description",
    "product_title",
    "ground_truth_category",
    "potential_product_categories",
    "product_image",
]

DEFAULT_DATASET_PATH = Path("data/shopify-products-experiment-l3.parquet")


@dataclass(frozen=True)
class ProductImage:
    bytes: bytes
    path: str


@dataclass(frozen=True)
class ProductRecord:
    product_description: str
    product_title: str
    ground_truth_category: CategoryPath
    potential_product_categories: list[CategoryPath]
    product_image: ProductImage


def load_dataset(
    path: Path = DEFAULT_DATASET_PATH, limit: int | None = None
) -> list[ProductRecord]:
    frame = pl.read_parquet(path, columns=REQUIRED_COLUMNS, n_rows=limit)
    records: list[ProductRecord] = []

    for row in frame.iter_rows(named=True):
        image = cast(dict[str, object], row["product_image"])
        records.append(
            ProductRecord(
                product_description=cast(str, row["product_description"]),
                product_title=cast(str, row["product_title"]),
                ground_truth_category=category_path(
                    cast(list[str], row["ground_truth_category"])
                ),
                potential_product_categories=[
                    category_path(category)
                    for category in cast(
                        list[list[str]], row["potential_product_categories"]
                    )
                ],
                product_image=ProductImage(
                    bytes=cast(bytes, image["bytes"]),
                    path=cast(str, image["path"]),
                ),
            )
        )

    return records
