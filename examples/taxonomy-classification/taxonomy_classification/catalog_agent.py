from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image
from sie_sdk import SIEClient

from taxonomy_classification.sie_client import create_sie_client

DATASET = "Shopify/product-catalogue"
DATASET_REVISION = "d5c517c509f5aca99053897ef1de797d6d7e5aa5"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"

RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
VERIFIER_MODEL = "Qwen/Qwen3.6-27B"
PROVISION_TIMEOUT_S = 900.0
DESCRIPTION_CHARS = 512
TOP_K_PER_RANKING = 2

RERANK_INSTRUCTION = (
    "Rank Shopify taxonomy paths by which path should categorize the product "
    "listing for an online store. Use both the product image and listing text."
)
VERIFIER_SYSTEM_PROMPT = """You classify e-commerce listings into one supplied Shopify taxonomy path.
Choose the path for the item the merchant is selling. Distinguish a finished product from its accessory, replacement part, digital design, container, ingredient, or depicted subject. Use the listing image and copy together.
Return the zero-based index of exactly one supplied path. If the listing or taxonomy is contradictory, still choose the best path and mark needs_review true. Return only selected_index and needs_review."""


@dataclass(frozen=True)
class CatalogListing:
    row_idx: int
    title: str
    description: str
    image_bytes: bytes
    image_format: str
    image_sha256: str
    candidate_paths: list[str]
    ground_truth_path: str | None = None
    viewer_url: str | None = None


@dataclass(frozen=True)
class CatalogDecision:
    row_idx: int
    selected_path: str
    needs_review: bool
    candidate_union: list[str]
    text_scores: list[float]
    image_plus_copy_scores: list[float]
    verifier_response_id: str | None


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _image_format(image_bytes: bytes) -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        value = (image.format or "jpeg").lower()
    return "jpeg" if value == "jpg" else value


def _verifier_data_uri(image_bytes: bytes) -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.thumbnail((768, 768))
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=84, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(encoded.getvalue()).decode(
        "ascii"
    )


def _description_excerpt(description: str) -> str:
    value = description.strip()
    if len(value) <= DESCRIPTION_CHARS:
        return value
    return value[:DESCRIPTION_CHARS].rsplit(" ", 1)[0]


def listing_query(listing: CatalogListing) -> str:
    query = f"Product title: {listing.title}"
    description = _description_excerpt(listing.description)
    if description:
        query += f"\nProduct description: {description}"
    return query


def load_shopify_rows(
    *,
    offset: int,
    limit: int,
    cache_dir: Path,
) -> list[CatalogListing]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "offset": offset,
            "length": limit,
            "revision": DATASET_REVISION,
        }
    )
    payload = json.loads(_download(f"{DATASET_ROWS_URL}?{query}"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    listings: list[CatalogListing] = []
    for entry in payload["rows"]:
        row_idx = int(entry["row_idx"])
        row = entry["row"]
        image_path = cache_dir / f"shopify-train-{row_idx}.image"
        if image_path.exists():
            image_bytes = image_path.read_bytes()
        else:
            image_bytes = _download(row["product_image"]["src"])
            image_path.write_bytes(image_bytes)
        listings.append(
            CatalogListing(
                row_idx=row_idx,
                title=row["product_title"],
                description=row["product_description"],
                image_bytes=image_bytes,
                image_format=_image_format(image_bytes),
                image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                candidate_paths=list(row["potential_product_categories"]),
                ground_truth_path=row["ground_truth_category"],
                viewer_url=(
                    "https://huggingface.co/datasets/Shopify/product-catalogue/"
                    f"viewer/default/train?row={row_idx}"
                ),
            )
        )
    return listings


def _scores_in_request_order(
    result: dict[str, Any],
    candidate_count: int,
) -> list[float]:
    scores_by_index = {
        int(score["item_id"]): float(score["score"]) for score in result["scores"]
    }
    if set(scores_by_index) != set(range(candidate_count)):
        raise ValueError("SIE reranker response does not cover every candidate")
    return [scores_by_index[index] for index in range(candidate_count)]


def _top_indexes(scores: list[float], limit: int) -> list[int]:
    return sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:limit]


def candidate_union(
    paths: list[str],
    text_scores: list[float],
    image_plus_copy_scores: list[float],
    *,
    top_k: int = TOP_K_PER_RANKING,
) -> list[str]:
    indexes: list[int] = []
    for index in [
        *_top_indexes(image_plus_copy_scores, top_k),
        *_top_indexes(text_scores, top_k),
    ]:
        if index not in indexes:
            indexes.append(index)
    return [paths[index] for index in indexes]


def rerank_listing(
    client: SIEClient,
    listing: CatalogListing,
) -> tuple[list[float], list[float]]:
    query = listing_query(listing)
    candidates = [
        {"id": str(index), "text": path}
        for index, path in enumerate(listing.candidate_paths)
    ]
    text_result = client.score(
        RERANKER_MODEL,
        {"text": query},
        candidates,
        instruction=RERANK_INSTRUCTION,
        wait_for_capacity=True,
        provision_timeout_s=PROVISION_TIMEOUT_S,
    )
    image_result = client.score(
        RERANKER_MODEL,
        {
            "text": query,
            "images": [
                {
                    "data": listing.image_bytes,
                    "format": listing.image_format,
                }
            ],
        },
        candidates,
        instruction=RERANK_INSTRUCTION,
        wait_for_capacity=True,
        provision_timeout_s=PROVISION_TIMEOUT_S,
    )
    return (
        _scores_in_request_order(text_result, len(candidates)),
        _scores_in_request_order(image_result, len(candidates)),
    )


def verify_candidates(
    client: SIEClient,
    listing: CatalogListing,
    candidates: list[str],
) -> tuple[int, bool, str | None]:
    candidate_lines = "\n".join(
        f"{index}: {path}" for index, path in enumerate(candidates)
    )
    user_text = (
        f"TITLE\n{listing.title}\n\n"
        f"DESCRIPTION\n{listing.description.strip() or '(none)'}\n\n"
        f"CANDIDATE PATHS\n{candidate_lines}"
    )
    response = client.chat_completions(
        VERIFIER_MODEL,
        [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _verifier_data_uri(listing.image_bytes)},
                    },
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        temperature=0,
        max_tokens=512,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "taxonomy_selection",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "selected_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": len(candidates) - 1,
                        },
                        "needs_review": {"type": "boolean"},
                    },
                    "required": ["selected_index", "needs_review"],
                    "additionalProperties": False,
                },
            },
        },
        wait_for_capacity=True,
        provision_timeout_s=PROVISION_TIMEOUT_S,
    )
    content = response["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("SIE verifier returned non-text content")
    selection = json.loads(content)
    selected_index = selection["selected_index"]
    if not isinstance(selected_index, int) or not (
        0 <= selected_index < len(candidates)
    ):
        raise ValueError(f"Invalid selected_index: {selected_index!r}")
    return selected_index, bool(selection["needs_review"]), response.get("id")


def classify_listing(
    client: SIEClient,
    listing: CatalogListing,
) -> CatalogDecision:
    text_scores, image_plus_copy_scores = rerank_listing(client, listing)
    candidates = candidate_union(
        listing.candidate_paths,
        text_scores,
        image_plus_copy_scores,
    )
    selected_index, needs_review, response_id = verify_candidates(
        client,
        listing,
        candidates,
    )
    return CatalogDecision(
        row_idx=listing.row_idx,
        selected_path=candidates[selected_index],
        needs_review=needs_review,
        candidate_union=candidates,
        text_scores=text_scores,
        image_plus_copy_scores=image_plus_copy_scores,
        verifier_response_id=response_id,
    )


def _common_depth(prediction: str, reference: str) -> int:
    predicted_nodes = prediction.split(" > ")
    reference_nodes = reference.split(" > ")
    depth = 0
    while (
        depth < len(predicted_nodes)
        and depth < len(reference_nodes)
        and predicted_nodes[depth] == reference_nodes[depth]
    ):
        depth += 1
    return depth


def evaluation_metrics(
    listings: list[CatalogListing],
    decisions: list[CatalogDecision],
) -> dict[str, Any]:
    listing_by_row = {listing.row_idx: listing for listing in listings}
    exact = 0
    top_level = 0
    hierarchical_f1_total = 0.0
    for decision in decisions:
        listing = listing_by_row[decision.row_idx]
        reference = listing.ground_truth_path
        if reference is None:
            raise ValueError(f"Missing reference for row {listing.row_idx}")
        if decision.selected_path == reference:
            exact += 1
        if decision.selected_path.split(" > ")[0] == reference.split(" > ")[0]:
            top_level += 1
        common_depth = _common_depth(decision.selected_path, reference)
        precision = common_depth / len(decision.selected_path.split(" > "))
        recall = common_depth / len(reference.split(" > "))
        hierarchical_f1_total += (
            0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    count = len(decisions)
    return {
        "sample_size": count,
        "exact_path_correct": exact,
        "exact_path_accuracy": exact / count if count else 0.0,
        "top_level_correct": top_level,
        "top_level_accuracy": top_level / count if count else 0.0,
        "macro_hierarchical_f1": (hierarchical_f1_total / count if count else 0.0),
        "needs_review": sum(decision.needs_review for decision in decisions),
    }


def _common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/catalog-agent"),
    )
    return parser


def eval_main() -> None:
    parser = _common_parser(
        "Evaluate the focused multimodal catalog agent on Shopify listings."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    listings = load_shopify_rows(
        offset=args.offset,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )
    decisions: list[CatalogDecision] = []
    with create_sie_client(timeout_s=600) as client:
        for listing in listings:
            decision = classify_listing(client, listing)
            decisions.append(decision)
            print(
                f"{listing.row_idx:>5} "
                f"{'review' if decision.needs_review else 'accept':>6} "
                f"{decision.selected_path}"
            )

    output = {
        "record_type": "sie_catalog_agent_evaluation",
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "dataset": {
            "id": DATASET,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "row_window": [args.offset, args.offset + len(listings) - 1],
        },
        "models": {
            "reranker": RERANKER_MODEL,
            "verifier": VERIFIER_MODEL,
        },
        "candidate_rule": (
            "Union of the two highest-scoring paths from copy-only and "
            "image-plus-copy reranker calls."
        ),
        "response_schema": {
            "required": ["selected_index", "needs_review"],
        },
        "metrics": evaluation_metrics(listings, decisions),
        "results": [
            {
                **asdict(decision),
                "ground_truth_path": next(
                    listing.ground_truth_path
                    for listing in listings
                    if listing.row_idx == decision.row_idx
                ),
            }
            for decision in decisions
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["metrics"], indent=2))


def predict_main() -> None:
    parser = _common_parser("Run the focused catalog agent on one Shopify listing.")
    parser.set_defaults(limit=1)
    args = parser.parse_args()
    listings = load_shopify_rows(
        offset=args.offset,
        limit=1,
        cache_dir=args.cache_dir,
    )
    with create_sie_client(timeout_s=600) as client:
        decision = classify_listing(client, listings[0])
    print(json.dumps(asdict(decision), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    eval_main()
