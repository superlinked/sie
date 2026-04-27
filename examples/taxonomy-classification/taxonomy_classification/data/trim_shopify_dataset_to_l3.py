from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from taxonomy_classification.data.clean_shopify_dataset import (
    REQUIRED_COLUMNS,
    load_valid_categories,
)

CATEGORY_SEPARATOR = " > "
MAX_LEVELS = 3
DEFAULT_INPUT = Path("data/shopify-products-clean-full-depth.parquet")
DEFAULT_TAXONOMY_FILE = Path("data/shopify-taxonomy-categories.txt")
DEFAULT_TAXONOMY_OUTPUT = Path("data/shopify-taxonomy-l3.parquet")
DEFAULT_OUTPUT = Path("data/shopify-products-experiment-l3.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim the cleaned Shopify dataset and taxonomy to L3 and store "
            "categories as node lists."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input cleaned full-depth parquet path.",
    )
    parser.add_argument(
        "--taxonomy-file",
        type=Path,
        default=DEFAULT_TAXONOMY_FILE,
        help="Path to Shopify taxonomy categories.txt.",
    )
    parser.add_argument(
        "--taxonomy-output",
        type=Path,
        default=DEFAULT_TAXONOMY_OUTPUT,
        help="Output parquet path for the trimmed L3 taxonomy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet path for the final experiment dataset.",
    )
    return parser.parse_args()


def trimmed_category_key(path: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in path.split(CATEGORY_SEPARATOR)[:MAX_LEVELS])


def trim_potential_categories(categories: list[str]) -> list[list[str]]:
    trimmed_categories: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    for category in categories:
        trimmed = trimmed_category_key(category)
        if trimmed in seen:
            continue
        seen.add(trimmed)
        trimmed_categories.append(list(trimmed))

    return trimmed_categories


def build_trimmed_taxonomy(
    valid_categories: set[str],
) -> tuple[pl.DataFrame, set[tuple[str, ...]]]:
    trimmed_categories = sorted(
        {trimmed_category_key(category) for category in valid_categories}
    )
    return (
        pl.DataFrame({"category": [list(category) for category in trimmed_categories]}),
        set(trimmed_categories),
    )


def main() -> None:
    args = parse_args()
    full_depth = pl.read_parquet(args.input).select(REQUIRED_COLUMNS)
    trimmed_taxonomy, valid_trimmed_categories = build_trimmed_taxonomy(
        load_valid_categories(args.taxonomy_file)
    )

    trimmed = full_depth.with_columns(
        ground_truth_category=pl.col("ground_truth_category").map_elements(
            lambda category: list(trimmed_category_key(category)),
            return_dtype=pl.List(pl.String),
        ),
        potential_product_categories=pl.col(
            "potential_product_categories"
        ).map_elements(
            trim_potential_categories,
            return_dtype=pl.List(pl.List(pl.String)),
        ),
    )

    invalid_ground_truth_rows = trimmed.filter(
        ~pl.col("ground_truth_category").map_elements(
            lambda category: tuple(category) in valid_trimmed_categories,
            return_dtype=pl.Boolean,
        )
    ).height
    invalid_potential_rows = trimmed.filter(
        pl.col("potential_product_categories").map_elements(
            lambda categories: any(
                tuple(category) not in valid_trimmed_categories
                for category in categories
            ),
            return_dtype=pl.Boolean,
        )
    ).height

    if invalid_ground_truth_rows or invalid_potential_rows:
        raise RuntimeError(
            "Trimmed dataset contains categories missing from the trimmed "
            f"taxonomy: invalid_ground_truth_rows={invalid_ground_truth_rows}, "
            f"invalid_potential_rows={invalid_potential_rows}"
        )

    args.taxonomy_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trimmed_taxonomy.write_parquet(args.taxonomy_output)
    trimmed.select(REQUIRED_COLUMNS).write_parquet(args.output)

    print(
        json.dumps(
            {
                "input_path": str(args.input),
                "taxonomy_file": str(args.taxonomy_file),
                "taxonomy_output_path": str(args.taxonomy_output),
                "output_path": str(args.output),
                "input_rows": full_depth.height,
                "output_rows": trimmed.height,
                "taxonomy_rows": trimmed_taxonomy.height,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
