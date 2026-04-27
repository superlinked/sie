from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sie_sdk import SIEClient

from taxonomy_classification.classifier.latency import latency_summary
from taxonomy_classification.classifier.models import (
    ClassifiedDescriptions,
    LatencySummary,
    PredictedCategories,
)
from tqdm import tqdm

from taxonomy_classification.data.dataset import ProductRecord, load_dataset
from taxonomy_classification.metrics import score_predictions
from taxonomy_classification.sie_client import create_sie_client
from taxonomy_classification.taxonomy import CategoryPath, load_taxonomy

DEFAULT_BATCH_SIZE = 16
DEFAULT_PROVISION_TIMEOUT_S = 300.0
DEFAULT_TOP_K = 5


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate L1 zero-shot NLI classification on product descriptions."
    )
    parser.add_argument("--model", required=True, help="SIE NLI model name.")
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
        description="Predict top L1 labels from a product description."
    )
    parser.add_argument("--model", required=True, help="SIE NLI model name.")
    parser.add_argument("--description", required=True, help="Product description.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of labels to print.",
    )
    return parser.parse_args()


def l1_labels() -> list[str]:
    return sorted({category[0] for category in load_taxonomy()})


def l1_category(category: CategoryPath) -> CategoryPath:
    return (category[0],)


def classify_descriptions(
    client: SIEClient,
    model: str,
    descriptions: list[str],
    labels: list[str],
) -> ClassifiedDescriptions:
    rankings: list[list[tuple[str, float]]] = []
    latencies_ms: list[float] = []

    for start in tqdm(
        range(0, len(descriptions), DEFAULT_BATCH_SIZE),
        total=(len(descriptions) + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE,
        desc="Evaluating",
        unit="batch",
    ):
        batch_descriptions = descriptions[start : start + DEFAULT_BATCH_SIZE]
        start_time = time.perf_counter()
        results = client.extract(
            model,
            [{"text": description} for description in batch_descriptions],
            labels=labels,
            provision_timeout_s=DEFAULT_PROVISION_TIMEOUT_S,
        )
        latencies_ms.append((time.perf_counter() - start_time) * 1000)
        for result in results:
            rankings.append(
                [
                    (classification["label"], classification["score"])
                    for classification in result["classifications"]
                ]
            )

    return ClassifiedDescriptions(rankings=rankings, latencies_ms=latencies_ms)


def predict_l1_categories(
    client: SIEClient,
    model: str,
    descriptions: list[str],
    labels: list[str],
) -> PredictedCategories:
    classifications = classify_descriptions(
        client,
        model,
        descriptions,
        labels,
    )
    return PredictedCategories(
        categories=[(ranking[0][0],) for ranking in classifications.rankings],
        latencies_ms=classifications.latencies_ms,
    )


def evaluate_records(
    client: SIEClient, model: str, records: list[ProductRecord]
) -> tuple[dict[str, dict[str, float]], LatencySummary]:
    labels = l1_labels()
    predictions = predict_l1_categories(
        client,
        model,
        [record.product_description for record in records],
        labels,
    )
    scores = score_predictions(
        predictions.categories,
        [l1_category(record.ground_truth_category) for record in records],
        [
            [l1_category(category) for category in record.potential_product_categories]
            for record in records
        ],
    )
    return {
        "strict": scores["strict"]["macro"]["l1"],
        "lenient": scores["lenient"]["macro"]["l1"],
    }, latency_summary(predictions.latencies_ms, batch_size=DEFAULT_BATCH_SIZE)


def eval_main() -> None:
    args = parse_eval_args()
    records = load_dataset(limit=args.limit)

    with create_sie_client(timeout_s=60) as client:
        metrics, latency_ms = evaluate_records(client, args.model, records)

    summary = {
        "model": args.model,
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
    labels = l1_labels()

    with create_sie_client(timeout_s=60) as client:
        classifications = classify_descriptions(
            client,
            args.model,
            [args.description],
            labels,
        )

    for label, score in classifications.rankings[0][: args.top_k]:
        print(f"{label}\t{score}")
    print(f"latency_ms\t{classifications.latencies_ms[0]:.3f}")


if __name__ == "__main__":
    eval_main()
