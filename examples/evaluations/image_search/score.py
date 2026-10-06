"""Reconstruct every retained cosine, ranking and lifecycle forecast offline."""

import argparse
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from evidence import ARM_DIMENSIONS, DEFAULT_ROOT, inputs, read_json, verify


def vector(values: list[float], width: int) -> list[float]:
    if len(values) != width or any(not math.isfinite(value) for value in values):
        raise ValueError("Vector width or finite values differ from the requested model")
    if math.fsum(value * value for value in values) <= 0:
        raise ValueError("A zero vector cannot be ranked by cosine")
    return values


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cosine requires vectors of the same width")
    left = vector(left, len(left))
    right = vector(right, len(right))
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    return dot / math.sqrt(math.fsum(a * a for a in left) * math.fsum(b * b for b in right))


def rank(index: dict[str, list[float]], query: list[float], order: list[str]) -> tuple[list[str], dict[str, float]]:
    if set(index) != set(order) or len(set(order)) != len(order):
        raise ValueError("Ranking requires every candidate exactly once")
    scores = {image_id: cosine(query, index[image_id]) for image_id in order}
    # Equal scores retain the frozen catalogue order.
    return sorted(order, key=lambda image_id: -scores[image_id]), scores


def quote(call: dict, rates: dict) -> Decimal:
    usage = call["usage"]
    price = rates[call["reported_model"]]
    million = Decimal(10**6)
    if call["kind"] == "voyage":
        return Decimal(usage["text_tokens"]) * Decimal(price["text_per_million_usd"]) / million + Decimal(
            usage["image_pixels"]
        ) * Decimal(price["image_pixels_per_billion_usd"]) / Decimal(10**9)
    if call["kind"] == "caption":
        cached = usage["input_tokens_details"]["cached_tokens"]
        written = usage["input_tokens_details"]["cache_write_tokens"]
        return (
            Decimal(usage["input_tokens"] - cached - written) * Decimal(price["input_per_million_usd"])
            + Decimal(cached) * Decimal(price["cached_input_per_million_usd"])
            + Decimal(written) * Decimal(price["cache_write_per_million_usd"])
            + Decimal(usage["output_tokens"]) * Decimal(price["output_per_million_usd"])
        ) / million
    return Decimal(usage["total_tokens"]) * Decimal(price["input_per_million_usd"]) / million


def lifecycle(root: Path, companion: dict) -> dict[str, dict[str, str]]:
    rates = read_json(root, "public-tariffs.json")
    ledger = read_json(root, "usage-ledger.json")
    calls = read_json(root, "call-ledger.json")["calls"]
    priced = {row["call_id"]: row for row in ledger["calls"]}
    parts = defaultdict(lambda: defaultdict(Decimal))
    total = Decimal(0)
    for call in calls:
        if call["kind"] == "native":
            continue
        value = quote(call, rates)
        if value != Decimal(priced[call["call_id"]]["usage_priced_usd"]):
            raise ValueError("A provider call quote does not reconstruct from its retained usage")
        role = "caption" if call["kind"] == "caption" else call["role"]
        parts[call["arm"]][role] += value
        total += value
    if total != Decimal(companion["measured_provider_usage_quote_usd"]):
        raise ValueError("The complete provider usage quote differs from the companion")
    workload = companion["workload"]
    forecasts = {}
    for arm, values in parts.items():
        index = (values["index"] + values["caption"]) / Decimal(20) * Decimal(workload["initial_photos"])
        queries = values["query"] / Decimal(8) * Decimal(workload["search_queries"])
        forecasts[arm] = (index, queries)
    unit_rates = {}
    for row in companion["native_tariffs"]:
        unit = row["usd_per_unit"]
        unit_rates[(row["model"], row["unit"])] = Decimal(unit["numerator"]) / Decimal(unit["denominator"])
    for arm, model in (
        ("sie-siglip384", "google/siglip-so400m-patch14-384"),
        ("sie-siglip2-base224", "google/siglip2-base-patch16-224"),
    ):
        index = Decimal(workload["initial_photos"]) * unit_rates[(model, "images")]
        queries = (
            Decimal(workload["search_queries"])
            * Decimal(companion["native_query_billable_input_tokens_each"])
            * unit_rates[(model, "input_tokens")]
        )
        forecasts[arm] = (index, queries)
    result = {}
    for arm, (index, queries) in forecasts.items():
        fields = {
            "initial_100000_photo_index_usd": index,
            "million_queries_usd": queries,
            "total_usd": index + queries,
        }
        for key, value in fields.items():
            if value != Decimal(companion["forecast_usd"][arm][key]):
                raise ValueError(f"Lifecycle forecast did not reconstruct: {arm}/{key}")
        result[arm] = {key: str(value) for key, value in fields.items()}
    return result


def evaluate(root: Path) -> dict:
    companion = verify(root)
    images, queries, grades = inputs(root)
    image_ids = [row["image_id"] for row in images]
    query_ids = [row["query_id"] for row in queries]
    vectors = read_json(root, "vectors.json")
    observations = read_json(root, "cosines-and-ranks.json")["observations"]
    recorded = {(row["arm"], row["query_id"]): row for row in observations}
    if set(vectors) != set(ARM_DIMENSIONS) or len(recorded) != 40 or len(observations) != 40:
        raise ValueError("The recording must retain all five recipes and forty complete rankings")
    counts = {}
    maximum_delta = 0.0
    for arm, width in ARM_DIMENSIONS.items():
        values = vectors[arm]
        if set(values["index"]) != set(image_ids) or set(values["query"]) != set(query_ids):
            raise ValueError("Missing or extra image/query vector")
        for group in values.values():
            for row in group.values():
                vector(row, width)
        success = 0
        for query_id in query_ids:
            ranking, scores = rank(values["index"], values["query"][query_id], image_ids)
            original = recorded[(arm, query_id)]
            if ranking != original["ranking"]:
                raise ValueError(f"Recorded ranking did not reconstruct: {arm}/{query_id}")
            for image_id, score in scores.items():
                delta = abs(score - original["scores_by_image"][image_id])
                maximum_delta = max(maximum_delta, delta)
                if delta > 1e-12:
                    raise ValueError("Recorded cosine did not reconstruct")
            success += grades[query_id][ranking[0]] == 2
        counts[arm] = {"first_result_grade2": success, "independent_query_families": len(queries)}
    return {
        "quality": counts,
        "reconstructed_cosines": 800,
        "maximum_cosine_delta": maximum_delta,
        "forecast_usd": lifecycle(root, companion),
        "scope": companion["quality_scope"],
        "cost_scope": companion["cost_scope"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.evidence), indent=2))


if __name__ == "__main__":
    main()
