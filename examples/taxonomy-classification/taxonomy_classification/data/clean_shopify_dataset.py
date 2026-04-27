from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

REQUIRED_COLUMNS = [
    "product_description",
    "product_title",
    "ground_truth_category",
    "potential_product_categories",
    "product_image",
]
DEFAULT_INPUT = Path("data/train-00000-of-00015.parquet")
DEFAULT_TAXONOMY_FILE = Path("data/shopify-taxonomy-categories.txt")
DEFAULT_OUTPUT = Path("data/shopify-products-clean-full-depth.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean the Shopify product catalogue shard with Polars, drop rows "
            "missing required fields, and keep only rows whose categories are "
            "present in the Shopify taxonomy."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input Shopify shard parquet path.",
    )
    parser.add_argument(
        "--taxonomy-file",
        type=Path,
        default=DEFAULT_TAXONOMY_FILE,
        help="Path to Shopify taxonomy categories.txt.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet path for the cleaned, full-depth dataset.",
    )
    return parser.parse_args()


def load_valid_categories(path: Path) -> set[str]:
    valid_categories: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        _, category_path = line.split(" : ", maxsplit=1)
        valid_categories.add(category_path.strip())
    return valid_categories


def normalize_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        product_title=pl.col("product_title").str.strip_chars(),
        product_description=pl.col("product_description").str.strip_chars(),
        ground_truth_category=pl.col("ground_truth_category").str.strip_chars(),
        potential_product_categories=pl.col("potential_product_categories").list.eval(
            pl.element().str.strip_chars()
        ),
        product_image=pl.struct(
            pl.col("product_image").struct.field("bytes").alias("bytes"),
            pl.col("product_image")
            .struct.field("path")
            .str.strip_chars()
            .alias("path"),
        ),
    )


def required_row_mask() -> pl.Expr:
    return (
        pl.col("product_title").is_not_null()
        & (pl.col("product_title") != "")
        & pl.col("product_description").is_not_null()
        & (pl.col("product_description") != "")
        & pl.col("ground_truth_category").is_not_null()
        & (pl.col("ground_truth_category") != "")
        & pl.col("potential_product_categories").is_not_null()
        & (pl.col("potential_product_categories").list.len() > 0).fill_null(False)
        & ~pl.col("potential_product_categories")
        .list.eval(pl.element().is_null())
        .list.any()
        .fill_null(False)
        & ~pl.col("potential_product_categories")
        .list.eval(pl.element() == "")
        .list.any()
        .fill_null(False)
        & pl.col("product_image").is_not_null()
        & pl.col("product_image").struct.field("bytes").is_not_null()
        & (pl.col("product_image").struct.field("bytes").bin.size() > 0).fill_null(
            False
        )
        & pl.col("product_image").struct.field("path").is_not_null()
        & (pl.col("product_image").struct.field("path") != "").fill_null(False)
    )


def valid_taxonomy_row_mask(valid_categories: set[str]) -> pl.Expr:
    return pl.col("ground_truth_category").is_in(valid_categories) & pl.col(
        "potential_product_categories"
    ).list.eval(pl.element().is_in(valid_categories)).list.all().fill_null(False)


def main() -> None:
    args = parse_args()
    valid_categories = load_valid_categories(args.taxonomy_file)
    normalized = normalize_columns(pl.read_parquet(args.input)).select(REQUIRED_COLUMNS)
    required_rows = normalized.filter(required_row_mask())
    cleaned = required_rows.filter(valid_taxonomy_row_mask(valid_categories))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write_parquet(args.output)

    print(
        json.dumps(
            {
                "input_path": str(args.input),
                "taxonomy_file": str(args.taxonomy_file),
                "output_path": str(args.output),
                "input_rows": normalized.height,
                "rows_after_required_filter": required_rows.height,
                "output_rows": cleaned.height,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
