from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sie_sdk import SIEClient

from taxonomy_classification.chroma_index import (
    DEFAULT_INDEX_DIR,
    IndexVariant,
    build_index,
    collection_text,
    has_collection,
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

from taxonomy_classification.data.dataset import ProductRecord, load_dataset
from taxonomy_classification.metrics import score_predictions
from taxonomy_classification.sie_client import create_sie_client
from taxonomy_classification.taxonomy import full_path, load_taxonomy

DEFAULT_BATCH_SIZE = 8
DEFAULT_PROVISION_TIMEOUT_S = 300.0
DEFAULT_TOP_K = 5


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate text embedding retrieval on product descriptions."
    )
    parser.add_argument("--model", required=True, help="SIE embedding model name.")
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
        description="Predict top taxonomy categories from a product description."
    )
    parser.add_argument("--model", required=True, help="SIE embedding model name.")
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
    parser.add_argument("--description", required=True, help="Product description.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of categories to print.",
    )
    return parser.parse_args()


def encode_texts(
    client: SIEClient,
    model: str,
    texts: list[str],
    is_query: bool,
    progress_desc: str = "Encoding",
) -> EncodedItems:
    embeddings: list[list[float]] = []
    latencies_ms: list[float] = []

    for start in tqdm(
        range(0, len(texts), DEFAULT_BATCH_SIZE),
        total=(len(texts) + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE,
        desc=progress_desc,
        unit="batch",
    ):
        batch_texts = texts[start : start + DEFAULT_BATCH_SIZE]
        start_time = time.perf_counter()
        results = client.encode(
            model,
            [{"text": text} for text in batch_texts],
            is_query=is_query,
            provision_timeout_s=DEFAULT_PROVISION_TIMEOUT_S,
        )
        latencies_ms.append((time.perf_counter() - start_time) * 1000)
        for result in results:
            embeddings.append(result["dense"].tolist())

    return EncodedItems(embeddings=embeddings, latencies_ms=latencies_ms)


def ensure_index(
    client: SIEClient,
    model: str,
    variant: IndexVariant,
    index_dir: Path,
    rebuild: bool,
) -> None:
    if has_collection(model, variant, index_dir) and not rebuild:
        return

    categories = load_taxonomy()
    category_texts = [collection_text(category, variant) for category in categories]
    category_embeddings = encode_texts(
        client,
        model,
        category_texts,
        False,
        progress_desc="Encoding categories",
    )
    build_index(
        categories,
        category_embeddings.embeddings,
        model,
        variant,
        index_dir=index_dir,
        rebuild=rebuild,
    )


def retrieve_matches(
    client: SIEClient,
    model: str,
    variant: IndexVariant,
    descriptions: list[str],
    top_k: int,
    index_dir: Path,
    rebuild: bool = False,
) -> RetrievedMatches:
    ensure_index(client, model, variant, index_dir, rebuild)
    query_embeddings = encode_texts(
        client,
        model,
        descriptions,
        True,
        progress_desc="Encoding queries",
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
    descriptions: list[str],
    top_k: int,
    index_dir: Path,
    rebuild: bool = False,
) -> PredictedCategories:
    matches = retrieve_matches(
        client,
        model,
        variant,
        descriptions,
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
        [record.product_description for record in records],
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
            [args.description],
            args.top_k,
            args.index_dir,
        )

    for match in matches.matches[0]:
        print(f"{full_path(match.category)}\t{match.distance}")
    print(f"latency_ms\t{matches.latencies_ms[0]:.3f}")


if __name__ == "__main__":
    eval_main()
