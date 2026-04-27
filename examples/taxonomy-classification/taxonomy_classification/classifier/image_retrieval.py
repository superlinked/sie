from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sie_sdk import SIEClient

from taxonomy_classification.chroma_index import (
    DEFAULT_INDEX_DIR,
    IndexVariant,
    query_index,
)
from taxonomy_classification.classifier.latency import latency_summary
from taxonomy_classification.classifier.models import (
    EncodedItems,
    LatencySummary,
    PredictedCategories,
    RetrievedMatches,
)
from tqdm import tqdm

from taxonomy_classification.classifier.text_retrieval import ensure_index
from taxonomy_classification.data.dataset import (
    ProductImage,
    ProductRecord,
    load_dataset,
)
from taxonomy_classification.metrics import score_predictions
from taxonomy_classification.sie_client import create_sie_client
from taxonomy_classification.taxonomy import full_path

DEFAULT_BATCH_SIZE = 16
DEFAULT_PROVISION_TIMEOUT_S = 300.0
DEFAULT_TOP_K = 5


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image retrieval on product images."
    )
    parser.add_argument("--model", required=True, help="SIE vision model name.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["leaf", "full-path"],
        help="Category text form used in the index.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Persistent Chroma index directory.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the persistent index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of examples to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    return parser.parse_args()


def parse_predict_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict top taxonomy categories from an image."
    )
    parser.add_argument("--model", required=True, help="SIE vision model name.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=["leaf", "full-path"],
        help="Category text form used in the index.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Persistent Chroma index directory.",
    )
    parser.add_argument("--image-path", type=Path, required=True, help="Image file.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of categories to print.",
    )
    return parser.parse_args()


def image_format(path: str) -> str:
    suffix = Path(path).suffix.lower().removeprefix(".")
    if suffix == "jpg":
        return "jpeg"
    return suffix


def load_image(path: Path) -> ProductImage:
    return ProductImage(bytes=path.read_bytes(), path=str(path))


def encode_images(
    client: SIEClient,
    model: str,
    images: list[ProductImage],
) -> EncodedItems:
    embeddings: list[list[float]] = []
    latencies_ms: list[float] = []

    for start in tqdm(
        range(0, len(images), DEFAULT_BATCH_SIZE),
        total=(len(images) + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE,
        desc="Encoding images",
        unit="batch",
    ):
        batch_images = images[start : start + DEFAULT_BATCH_SIZE]
        start_time = time.perf_counter()
        results = client.encode(
            model,
            [
                {
                    "images": [
                        {
                            "data": image.bytes,
                            "format": image_format(image.path),
                        }
                    ]
                }
                for image in batch_images
            ],
            provision_timeout_s=DEFAULT_PROVISION_TIMEOUT_S,
        )
        latencies_ms.append((time.perf_counter() - start_time) * 1000)
        for result in results:
            embeddings.append(result["dense"].tolist())

    return EncodedItems(embeddings=embeddings, latencies_ms=latencies_ms)


def retrieve_matches(
    client: SIEClient,
    model: str,
    variant: IndexVariant,
    images: list[ProductImage],
    top_k: int,
    index_dir: Path,
    rebuild: bool = False,
) -> RetrievedMatches:
    ensure_index(client, model, variant, index_dir, rebuild)
    query_embeddings = encode_images(
        client,
        model,
        images,
    )
    return RetrievedMatches(
        matches=[
            query_index(model, variant, embedding, top_k, index_dir=index_dir)
            for embedding in query_embeddings.embeddings
        ],
        latencies_ms=query_embeddings.latencies_ms,
    )


def predict_categories(
    client: SIEClient,
    model: str,
    variant: IndexVariant,
    images: list[ProductImage],
    top_k: int,
    index_dir: Path,
    rebuild: bool = False,
) -> PredictedCategories:
    matches = retrieve_matches(
        client,
        model,
        variant,
        images,
        top_k,
        index_dir,
        rebuild=rebuild,
    )
    return PredictedCategories(
        categories=[item_matches[0].category for item_matches in matches.matches],
        latencies_ms=matches.latencies_ms,
    )


def evaluate_records(
    client: SIEClient,
    model: str,
    variant: IndexVariant,
    records: list[ProductRecord],
    index_dir: Path,
    rebuild: bool,
) -> tuple[dict[str, object], LatencySummary]:
    predictions = predict_categories(
        client,
        model,
        variant,
        [record.product_image for record in records],
        1,
        index_dir,
        rebuild=rebuild,
    )
    return score_predictions(
        predictions.categories,
        [record.ground_truth_category for record in records],
        [record.potential_product_categories for record in records],
    ), latency_summary(predictions.latencies_ms, batch_size=DEFAULT_BATCH_SIZE)


def eval_main() -> None:
    args = parse_eval_args()
    records = load_dataset(limit=args.limit)

    with create_sie_client(timeout_s=60) as client:
        metrics, latency_ms = evaluate_records(
            client,
            args.model,
            args.variant,
            records,
            args.index_dir,
            args.rebuild,
        )

    summary = {
        "model": args.model,
        "variant": args.variant,
        "index_dir": str(args.index_dir),
        "rebuild": args.rebuild,
        "limit": args.limit,
        "count": len(records),
        "strict": metrics["strict"],
        "lenient": metrics["lenient"],
        "latency_ms": latency_ms.as_dict(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def predict_main() -> None:
    args = parse_predict_args()

    with create_sie_client(timeout_s=60) as client:
        matches = retrieve_matches(
            client,
            args.model,
            args.variant,
            [load_image(args.image_path)],
            args.top_k,
            args.index_dir,
        )

    for match in matches.matches[0]:
        print(f"{full_path(match.category)}\t{match.distance}")
    print(f"latency_ms\t{matches.latencies_ms[0]:.3f}")


if __name__ == "__main__":
    eval_main()
