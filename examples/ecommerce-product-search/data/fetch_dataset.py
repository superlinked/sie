"""Fetch and prepare a product subset from the Amazon Products 2023 dataset."""

import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset


def main():
    root = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((root / "config.yaml").read_text())
    ds_config = config["dataset"]

    print(f"Loading {ds_config['name']} from HuggingFace...")
    ds = load_dataset(ds_config["name"], split="train")
    print(f"  Total rows: {len(ds)}")

    # Filter: must have a description, title, and be in our target categories
    min_len = ds_config["min_description_length"]
    categories = set(ds_config["categories"])

    products = []
    for row in ds:
        desc = row.get("description") or ""
        title = row.get("title") or ""
        category = row.get("main_category") or ""

        if len(desc) < min_len:
            continue
        if not title:
            continue
        if category not in categories:
            continue

        products.append({
            "id": row["parent_asin"],
            "title": title,
            "description": desc,
            "category": category,
            "price": row.get("price"),
            "rating": row.get("average_rating"),
            "features": row.get("features") or [],
            "image": row.get("image"),
        })

    print(f"  After filtering: {len(products)} products")

    # Sample down to target size
    sample_size = min(ds_config["sample_size"], len(products))
    random.seed(42)
    products = random.sample(products, sample_size)

    # Print category distribution
    cat_counts = {}
    for p in products:
        cat_counts[p["category"]] = cat_counts.get(p["category"], 0) + 1
    print(f"\n  Sampled {len(products)} products:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    # Save
    out_path = Path(__file__).parent / "products.json"
    with open(out_path, "w") as f:
        json.dump(products, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
