from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sie_sdk import SIEClient

from taxonomy_classification.chroma_index import (
    DEFAULT_INDEX_DIR,
    ChromaMatch,
    IndexVariant,
    collection_text,
)
from taxonomy_classification.classifier.latency import latency_summary
from taxonomy_classification.classifier.models import (
    LatencySummary,
    RerankedDescriptions,
)
from tqdm import tqdm

from taxonomy_classification.classifier.text_retrieval import (
    DEFAULT_BATCH_SIZE as RETRIEVAL_BATCH_SIZE,
    retrieve_matches,
)
from taxonomy_classification.data.dataset import ProductRecord, load_dataset
from taxonomy_classification.metrics import score_predictions
from taxonomy_classification.sie_client import create_sie_client
from taxonomy_classification.taxonomy import CategoryPath, full_path

DEFAULT_PROVISION_TIMEOUT_S = 300.0
DEFAULT_TOP_K = 5


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval + reranking on product descriptions."
    )
    parser.add_argument(
        "--retrieval-model", required=True, help="SIE embedding model name."
    )
    parser.add_argument(
        "--reranker-model", required=True, help="SIE reranker model name."
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=["leaf", "full-path"],
        help="Category text form used for retrieval and reranking.",
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
        help="Rebuild the persistent retrieval index.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieval candidates to rerank.",
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
        description="Predict top taxonomy categories with retrieval + reranking."
    )
    parser.add_argument(
        "--retrieval-model", required=True, help="SIE embedding model name."
    )
    parser.add_argument(
        "--reranker-model", required=True, help="SIE reranker model name."
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=["leaf", "full-path"],
        help="Category text form used for retrieval and reranking.",
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


def rerank_matches(
    client: SIEClient,
    reranker_model: str,
    variant: IndexVariant,
    description: str,
    matches: list[ChromaMatch],
) -> tuple[list[tuple[CategoryPath, float]], float]:
    start_time = time.perf_counter()
    score_result = client.score(
        reranker_model,
        {"text": description},
        [
            {
                "id": full_path(match.category),
                "text": collection_text(match.category, variant),
            }
            for match in matches
        ],
        provision_timeout_s=DEFAULT_PROVISION_TIMEOUT_S,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    category_lookup = {full_path(match.category): match.category for match in matches}
    return [
        (category_lookup[score["item_id"]], score["score"])
        for score in score_result["scores"]
    ], latency_ms


def rerank_descriptions(
    client: SIEClient,
    retrieval_model: str,
    reranker_model: str,
    variant: IndexVariant,
    descriptions: list[str],
    top_k: int,
    index_dir: Path,
    rebuild: bool = False,
) -> RerankedDescriptions:
    retrieved_matches = retrieve_matches(
        client,
        retrieval_model,
        variant,
        descriptions,
        top_k,
        index_dir,
        rebuild=rebuild,
    )
    rankings: list[list[tuple[CategoryPath, float]]] = []
    reranking_latencies_ms: list[float] = []

    for description, matches in tqdm(
        zip(descriptions, retrieved_matches.matches, strict=True),
        total=len(descriptions),
        desc="Reranking",
        unit="item",
    ):
        ranking, latency_ms = rerank_matches(
            client,
            reranker_model,
            variant,
            description,
            matches,
        )
        rankings.append(ranking)
        reranking_latencies_ms.append(latency_ms)

    return RerankedDescriptions(
        rankings=rankings,
        retrieval_latencies_ms=retrieved_matches.latencies_ms,
        reranking_latencies_ms=reranking_latencies_ms,
    )


def evaluate_records(
    client: SIEClient,
    retrieval_model: str,
    reranker_model: str,
    variant: IndexVariant,
    records: list[ProductRecord],
    top_k: int,
    index_dir: Path,
    rebuild: bool,
) -> tuple[dict[str, object], LatencySummary, LatencySummary]:
    reranked = rerank_descriptions(
        client,
        retrieval_model,
        reranker_model,
        variant,
        [record.product_description for record in records],
        top_k,
        index_dir,
        rebuild=rebuild,
    )
    return (
        score_predictions(
            [ranking[0][0] for ranking in reranked.rankings],
            [record.ground_truth_category for record in records],
            [record.potential_product_categories for record in records],
        ),
        latency_summary(
            reranked.retrieval_latencies_ms, batch_size=RETRIEVAL_BATCH_SIZE
        ),
        latency_summary(reranked.reranking_latencies_ms, items_per_request=top_k),
    )


def eval_main() -> None:
    args = parse_eval_args()
    records = load_dataset(limit=args.limit)

    with create_sie_client(timeout_s=60) as client:
        metrics, retrieval_latency_ms, reranking_latency_ms = evaluate_records(
            client,
            args.retrieval_model,
            args.reranker_model,
            args.variant,
            records,
            args.top_k,
            args.index_dir,
            args.rebuild,
        )

    summary = {
        "retrieval_model": args.retrieval_model,
        "reranker_model": args.reranker_model,
        "variant": args.variant,
        "index_dir": str(args.index_dir),
        "rebuild": args.rebuild,
        "top_k": args.top_k,
        "limit": args.limit,
        "count": len(records),
        "strict": metrics["strict"],
        "lenient": metrics["lenient"],
        "retrieval_latency_ms": retrieval_latency_ms.as_dict(),
        "reranking_latency_ms": reranking_latency_ms.as_dict(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def predict_main() -> None:
    args = parse_predict_args()

    with create_sie_client(timeout_s=60) as client:
        reranked = rerank_descriptions(
            client,
            args.retrieval_model,
            args.reranker_model,
            args.variant,
            [args.description],
            args.top_k,
            args.index_dir,
        )

    for category, score in reranked.rankings[0]:
        print(f"{full_path(category)}\t{score}")
    print(f"retrieval_latency_ms\t{reranked.retrieval_latencies_ms[0]:.3f}")
    print(f"reranking_latency_ms\t{reranked.reranking_latencies_ms[0]:.3f}")


if __name__ == "__main__":
    eval_main()
