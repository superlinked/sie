"""Ingest pipeline: extract attributes + encode embeddings for all products."""

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from sie_sdk import SIEClient
from sie_sdk.types import Item


def load_config():
    root = Path(__file__).resolve().parent.parent
    return yaml.safe_load((root / "config.yaml").read_text())


def load_products():
    path = Path(__file__).resolve().parent.parent / "data" / "products.json"
    with open(path) as f:
        return json.load(f)


def batch_iter(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def run_extraction(client, config, products):
    """Run extract() on all products to get structured attributes."""
    model = config["models"]["extractor"]
    labels = config["extract_labels"]
    gpu = config["cluster"]["gpu"]
    timeout = config["cluster"]["provision_timeout_s"]

    print(f"\n--- Extraction ({model}) ---")
    print(f"  Labels: {labels}")

    extracted = {}
    batch_size = config.get("ingest", {}).get("batch_size_extraction", 8)
    threshold = config.get("ingest", {}).get("confidence_threshold", 0.5)
    junk_len = config.get("ingest", {}).get("junk_text_max_len", 15)
    total = len(products)

    for i, batch in enumerate(batch_iter(products, batch_size)):
        items = [Item(id=p["id"], text=p["description"][:512]) for p in batch]
        start = time.time()

        results = client.extract(
            model,
            items,
            labels=labels,
            gpu=gpu,
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        )

        elapsed = time.time() - start
        done = min((i + 1) * batch_size, total)
        print(f"  [{done}/{total}] batch in {elapsed:.1f}s")

        for result in results:
            item_id = result.get("id", "")
            entities = result.get("entities", [])
            attrs = {}
            for e in entities:
                label = e["label"]
                text = e["text"].strip()
                if e["score"] < threshold:
                    continue
                if len(text) > junk_len and " " not in text:
                    continue
                if label not in attrs or e["score"] > attrs[label]["score"]:
                    attrs[label] = {"text": text, "score": e["score"]}
            extracted[item_id] = {k: v["text"] for k, v in attrs.items()}

    return extracted


def run_encoding(client, config, products):
    """Run encode() on all products to get dense embeddings."""
    model = config["models"]["embedding"]
    gpu = config["cluster"]["gpu"]
    timeout = config["cluster"]["provision_timeout_s"]

    print(f"\n--- Encoding ({model}) ---")

    embeddings = []
    batch_size = config.get("ingest", {}).get("batch_size_encoding", 32)
    total = len(products)

    for i, batch in enumerate(batch_iter(products, batch_size)):
        items = [Item(id=p["id"], text=f"{p['title']}. {p['description'][:512]}") for p in batch]
        start = time.time()

        results = client.encode(
            model,
            items,
            gpu=gpu,
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        )

        elapsed = time.time() - start
        done = min((i + 1) * batch_size, total)
        print(f"  [{done}/{total}] batch in {elapsed:.1f}s")

        # The server preserves input order (encode.py zips results to input by index),
        # so results[j] corresponds to batch[j], and the final embeddings array is
        # aligned positionally with the products list.
        if len(results) != len(batch):
            raise ValueError(
                f"encode() returned {len(results)} results for a batch of {len(batch)}"
            )
        for result in results:
            if "dense" not in result:
                raise ValueError(f"encode() returned no dense vector for item {result.get('id', '<unknown>')}")
            embeddings.append(result["dense"])

    if not embeddings:
        raise ValueError("No embeddings were generated; check products list and cluster connection")

    return np.stack(embeddings)


def main():
    config = load_config()
    products = load_products()
    print(f"Loaded {len(products)} products")

    cluster_url = os.environ.get("SIE_CLUSTER_URL", config["cluster"]["url"])
    api_key = os.environ.get("SIE_API_KEY", config["cluster"]["api_key"])

    with SIEClient(cluster_url, api_key=api_key) as client:
        # Step 1: Extract structured attributes
        extracted_attrs = run_extraction(client, config, products)

        # Step 2: Encode embeddings
        embeddings = run_encoding(client, config, products)
        print(f"\n  Embeddings shape: {embeddings.shape}")

    # Step 3: Build metadata (product info + extracted attributes)
    metadata = []
    for p in products:
        entry = {
            "id": p["id"],
            "title": p["title"],
            "description": p["description"][:500],
            "category": p["category"],
            "price": p["price"],
            "rating": p["rating"],
            "image": p["image"],
            "features": p.get("features", []),
            "attributes": extracted_attrs.get(p["id"], {}),
        }
        metadata.append(entry)

    # Save
    data_dir = Path(__file__).resolve().parent.parent / "data"
    np.save(data_dir / "embeddings.npy", embeddings)
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved embeddings to data/embeddings.npy")
    print(f"  Saved metadata to data/metadata.json")

    # Print a few examples
    print("\n--- Sample extractions ---")
    for p in products[:5]:
        attrs = extracted_attrs.get(p["id"], {})
        if attrs:
            print(f"  {p['title'][:60]}")
            print(f"    -> {attrs}")


if __name__ == "__main__":
    main()
