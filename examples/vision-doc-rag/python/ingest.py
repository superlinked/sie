"""Build the per-tenant visual index.

For every rendered PDF page PNG we ask SIE to encode the image with
vidore/colqwen2.5-v0.2, which returns a [tokens, 128] multivector. Each page's
multivector goes into a single .npz on disk, alongside a metadata.json that
keeps the client name, source PDF, page number, and source URL for routing,
filtering, and citation at query time.

There is no vector database here. MaxSim at the scale of one team's wiki
(hundreds to thousands of pages) is cheap and avoids the indexing step.
For larger corpora swap the .npz for a multivector store (LanceDB, Vespa,
Turbopuffer); the encode call is the same.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sie_sdk import SIEClient
from sie_sdk.types import Item

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAGES_MANIFEST = EXAMPLE_ROOT / "data" / "pages_manifest.json"


def load_config() -> dict[str, Any]:
    return yaml.safe_load((EXAMPLE_ROOT / "config.yaml").read_text())


def load_pages(pages_path: Path = DEFAULT_PAGES_MANIFEST) -> list[dict[str, Any]]:
    if not pages_path.exists():
        if pages_path == DEFAULT_PAGES_MANIFEST:
            hint = " Run `python data/fetch_pdfs.py` and `python data/render_pages.py` first."
        else:
            hint = ""
        raise FileNotFoundError(f"{pages_path} not found.{hint}")
    return json.loads(pages_path.read_text())


def encode_pages(
    client: SIEClient,
    model: str,
    pages: list[dict[str, Any]],
    gpu: str,
    timeout: float,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    multivectors: list[np.ndarray] = []
    metadata: list[dict] = []

    for i, page in enumerate(pages, 1):
        image_path = data_dir / page["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing page image: {image_path}. Run data/render_pages.py.")

        start = time.time()
        result = client.encode(
            model,
            Item(id=page["page_id"], images=[str(image_path)]),
            output_types=["multivector"],
            gpu=gpu,
            wait_for_capacity=True,
            provision_timeout_s=timeout,
        )
        elapsed = time.time() - start
        mv = result["multivector"].astype(np.float32)
        multivectors.append(mv)
        metadata.append(
            {
                "page_id": page["page_id"],
                "client": page["client"],
                "title": page["title"],
                "publisher": page["publisher"],
                "license": page["license"],
                "source_url": page["source_url"],
                "source_pdf": page["source_pdf"],
                "source_pdf_path": page["source_pdf_path"],
                "page_number": page["page_number"],
                "image_path": page["image_path"],
                "num_tokens": int(mv.shape[0]),
            }
        )
        citation = f"{page['source_pdf']} · p.{page['page_number']}"
        print(f"  [{i}/{len(pages)}] {page['client']:12s} {citation:44s} {mv.shape} in {elapsed:.1f}s")

    return multivectors, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pages-manifest",
        type=Path,
        default=DEFAULT_PAGES_MANIFEST,
        help="page manifest to ingest (defaults to data/pages_manifest.json)",
    )
    args = parser.parse_args()

    config = load_config()
    pages = load_pages(args.pages_manifest)
    print(f"Loaded {len(pages)} pages")

    cluster_url = os.environ.get("SIE_CLUSTER_URL", config["cluster"]["url"])
    api_key = os.environ.get("SIE_API_KEY", config["cluster"]["api_key"])
    gpu = config["cluster"]["gpu"]
    timeout = config["cluster"]["provision_timeout_s"]
    model = config["models"]["retriever"]

    print(f"\n--- Encoding pages with {model} ---")
    with SIEClient(cluster_url, api_key=api_key) as client:
        multivectors, metadata = encode_pages(client, model, pages, gpu, timeout)

    data_dir = Path(__file__).resolve().parent.parent / "data"
    # np.savez stores variable-length multivectors as one entry per array; we
    # key them by page_id so the search side can reload without an extra index.
    np.savez(
        data_dir / "multivectors.npz",
        **{m["page_id"]: mv for m, mv in zip(metadata, multivectors)},
    )
    (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    total_tokens = sum(m["num_tokens"] for m in metadata)
    by_client: dict[str, int] = {}
    for m in metadata:
        by_client[m["client"]] = by_client.get(m["client"], 0) + 1

    print(f"\n  Saved {len(metadata)} multivectors to data/multivectors.npz")
    print("  Saved metadata to data/metadata.json")
    print(f"  Total visual tokens: {total_tokens}")
    print("  Pages per tenant:")
    for client_name in sorted(by_client):
        print(f"    {client_name}: {by_client[client_name]}")


if __name__ == "__main__":
    main()
