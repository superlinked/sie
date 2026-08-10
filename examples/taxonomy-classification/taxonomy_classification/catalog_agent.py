from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shlex
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image
from sie_sdk import SIEClient

from taxonomy_classification.sie_client import create_sie_client, read_sie_settings

DATASET = "Shopify/product-catalogue"
DATASET_REVISION = "d5c517c509f5aca99053897ef1de797d6d7e5aa5"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"

RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
RERANKER_REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
VERIFIER_MODEL = "Qwen/Qwen3.6-27B"
VERIFIER_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
PROVISION_TIMEOUT_S = 900.0
DESCRIPTION_CHARS = 512
TOP_K_PER_RANKING = 2
EXPECTED_API_CALL_MODELS = {
    "copy_rerank": RERANKER_MODEL,
    "image_plus_copy_rerank": RERANKER_MODEL,
    "candidate_verification": VERIFIER_MODEL,
}
REQUIRED_PROVENANCE_FIELDS = (
    "request_id",
    "rate_book_version",
    "execution_identity_sha256",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")

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
    api_calls: list[dict[str, Any]] = field(default_factory=list)


def _api_call_record(
    *,
    stage: str,
    requested_model: str,
    response: dict[str, Any],
    timing_ms: float,
) -> dict[str, Any]:
    runtime_model = response.get("model")
    if runtime_model != requested_model:
        raise ValueError(
            f"SIE {stage} runtime model {runtime_model!r} differs from "
            f"requested model {requested_model!r}"
        )
    request = response.get("request")
    request_row = request if isinstance(request, dict) else {}
    request_id = request_row.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ValueError(f"SIE {stage} response has no request ID")
    rate_book_version = request_row.get("rate_book_version")
    if not isinstance(rate_book_version, str) or not rate_book_version:
        raise ValueError(f"SIE {stage} response has no rate-book version")
    execution_identity_sha256 = request_row.get("execution_identity_sha256")
    if not isinstance(execution_identity_sha256, str) or not SHA256_RE.fullmatch(
        execution_identity_sha256
    ):
        raise ValueError(f"SIE {stage} response has an invalid execution identity")
    credits_debited = request_row.get("credits_debited")
    if (
        not isinstance(credits_debited, int | float)
        or isinstance(credits_debited, bool)
        or credits_debited < 0
    ):
        raise ValueError(f"SIE {stage} response has invalid credits debited")
    return {
        "stage": stage,
        "requested_model": requested_model,
        "runtime_model": runtime_model,
        "request_id": request_id,
        "timing_ms": round(timing_ms, 1),
        "credits_debited": credits_debited,
        "rate_book_version": rate_book_version,
        "execution_identity_sha256": execution_identity_sha256,
    }


def _validate_api_calls(
    api_calls: object,
    *,
    row_idx: int,
) -> list[str]:
    if not isinstance(api_calls, list) or len(api_calls) != len(
        EXPECTED_API_CALL_MODELS
    ):
        raise ValueError(
            f"Row {row_idx} must record exactly "
            f"{len(EXPECTED_API_CALL_MODELS)} SIE API calls"
        )
    calls = [call for call in api_calls if isinstance(call, dict)]
    if len(calls) != len(api_calls):
        raise TypeError(f"Row {row_idx} has a non-object SIE API call")
    stages = [call.get("stage") for call in calls]
    if (
        any(not isinstance(stage, str) for stage in stages)
        or len(stages) != len(set(stages))
        or set(stages) != set(EXPECTED_API_CALL_MODELS)
    ):
        raise ValueError(f"Row {row_idx} has incomplete or duplicate SIE API stages")
    for call in calls:
        stage = call["stage"]
        expected_model = EXPECTED_API_CALL_MODELS[stage]
        if (
            call.get("requested_model") != expected_model
            or call.get("runtime_model") != expected_model
        ):
            raise ValueError(f"Row {row_idx} has the wrong model for {stage}")
        credits_debited = call.get("credits_debited")
        if (
            not isinstance(credits_debited, int | float)
            or isinstance(credits_debited, bool)
            or credits_debited < 0
        ):
            raise ValueError(f"Row {row_idx} {stage} has invalid credits debited")
        for field_name in REQUIRED_PROVENANCE_FIELDS:
            value = call.get(field_name)
            valid = isinstance(value, str) and bool(value)
            if field_name == "execution_identity_sha256":
                valid = isinstance(value, str) and bool(SHA256_RE.fullmatch(value))
            if not valid:
                raise ValueError(
                    f"Row {row_idx} {stage} has invalid {field_name.replace('_', ' ')}"
                )
    request_ids = [call["request_id"] for call in calls]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError(f"Row {row_idx} has duplicate SIE request IDs")
    return request_ids


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _image_format(image_bytes: bytes) -> str:
    with Image.open(io.BytesIO(image_bytes)) as image:
        value = (image.format or "jpeg").lower()
    return "jpeg" if value == "jpg" else value


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


def listing_sha256(listing: CatalogListing) -> str:
    payload = {
        "row_idx": listing.row_idx,
        "title": listing.title,
        "description": listing.description,
        "image_sha256": listing.image_sha256,
        "image_format": listing.image_format,
        "candidate_paths": listing.candidate_paths,
        "ground_truth_path": listing.ground_truth_path,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


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
        image_path = (
            cache_dir / f"shopify-train-{DATASET_REVISION[:12]}-{row_idx}.image"
        )
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
) -> tuple[list[float], list[float], list[dict[str, Any]]]:
    query = listing_query(listing)
    candidates = [
        {"id": str(index), "text": path}
        for index, path in enumerate(listing.candidate_paths)
    ]
    started = time.perf_counter()
    text_result = client.score(
        RERANKER_MODEL,
        {"text": query},
        candidates,
        instruction=RERANK_INSTRUCTION,
        wait_for_capacity=True,
        provision_timeout_s=PROVISION_TIMEOUT_S,
    )
    text_call = _api_call_record(
        stage="copy_rerank",
        requested_model=RERANKER_MODEL,
        response=text_result,
        timing_ms=(time.perf_counter() - started) * 1000,
    )
    started = time.perf_counter()
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
    image_call = _api_call_record(
        stage="image_plus_copy_rerank",
        requested_model=RERANKER_MODEL,
        response=image_result,
        timing_ms=(time.perf_counter() - started) * 1000,
    )
    return (
        _scores_in_request_order(text_result, len(candidates)),
        _scores_in_request_order(image_result, len(candidates)),
        [text_call, image_call],
    )


def verify_candidates(
    client: SIEClient,
    listing: CatalogListing,
    candidates: list[str],
) -> tuple[int, bool, dict[str, Any]]:
    if not candidates:
        raise ValueError(f"No candidate paths supplied for row {listing.row_idx}")
    candidate_lines = "\n".join(
        f"{index}: {path}" for index, path in enumerate(candidates)
    )
    user_text = (
        f"TITLE\n{listing.title}\n\n"
        f"DESCRIPTION\n{listing.description.strip() or '(none)'}\n\n"
        f"CANDIDATE PATHS\n{candidate_lines}"
    )
    selection_schema = {
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
    }
    started = time.perf_counter()
    response = client.generate(
        VERIFIER_MODEL,
        f"{VERIFIER_SYSTEM_PROMPT}\n\n{user_text}",
        images=[
            {
                "data": listing.image_bytes,
                "format": listing.image_format,
            },
        ],
        temperature=0,
        max_new_tokens=512,
        grammar={
            "json_schema": selection_schema,
            "label": "taxonomy_selection",
            "strict": True,
        },
        wait_for_capacity=True,
        provision_timeout_s=PROVISION_TIMEOUT_S,
    )
    content = response.get("text")
    if not isinstance(content, str):
        raise ValueError("SIE verifier returned non-text content")
    selection = json.loads(content)
    selected_index = selection["selected_index"]
    if type(selected_index) is not int or not (0 <= selected_index < len(candidates)):
        raise ValueError(f"Invalid selected_index: {selected_index!r}")
    needs_review = selection["needs_review"]
    if type(needs_review) is not bool:
        raise ValueError(f"Invalid needs_review: {needs_review!r}")
    call = _api_call_record(
        stage="candidate_verification",
        requested_model=VERIFIER_MODEL,
        response=response,
        timing_ms=(time.perf_counter() - started) * 1000,
    )
    return selected_index, needs_review, call


def classify_listing(
    client: SIEClient,
    listing: CatalogListing,
) -> CatalogDecision:
    text_scores, image_plus_copy_scores, rerank_calls = rerank_listing(client, listing)
    candidates = candidate_union(
        listing.candidate_paths,
        text_scores,
        image_plus_copy_scores,
    )
    selected_index, needs_review, verifier_call = verify_candidates(
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
        verifier_response_id=verifier_call["request_id"],
        api_calls=[*rerank_calls, verifier_call],
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


def _ranking_metrics(
    listings: list[CatalogListing],
    decisions: list[CatalogDecision],
    score_field: str,
) -> dict[str, Any]:
    ranked_decisions = [
        CatalogDecision(
            **{
                **asdict(decision),
                "selected_path": listing.candidate_paths[
                    max(
                        range(len(listing.candidate_paths)),
                        key=lambda index: getattr(decision, score_field)[index],
                    )
                ],
            }
        )
        for listing, decision in zip(listings, decisions, strict=True)
    ]
    return evaluation_metrics(listings, ranked_decisions)


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


def _evaluation_output(
    listings: list[CatalogListing],
    decisions_by_row: dict[int, CatalogDecision],
    *,
    offset: int,
    run_command: str | None = None,
) -> dict[str, Any]:
    completed_listings = [
        listing for listing in listings if listing.row_idx in decisions_by_row
    ]
    decisions = [decisions_by_row[listing.row_idx] for listing in completed_listings]
    request_ids = [
        request_id
        for decision in decisions
        for request_id in _validate_api_calls(
            decision.api_calls,
            row_idx=decision.row_idx,
        )
    ]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Catalog evaluation has duplicate SIE request IDs")
    agent_metrics = evaluation_metrics(completed_listings, decisions)
    copy_metrics = _ranking_metrics(completed_listings, decisions, "text_scores")
    image_metrics = _ranking_metrics(
        completed_listings, decisions, "image_plus_copy_scores"
    )
    endpoint, _api_key = read_sie_settings()
    return {
        "record_type": "sie_catalog_agent_evaluation",
        "run_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "endpoint": endpoint.rstrip("/"),
        "run_command": run_command,
        "timing_note": "Per-call timings are diagnostic, not benchmark results.",
        "dataset": {
            "id": DATASET,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "row_window": [offset, offset + len(listings) - 1],
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
            "type": "object",
            "properties": {
                "selected_index": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": TOP_K_PER_RANKING * 2 - 1,
                    "description": (
                        "The request schema tightens this bound to the number of "
                        "candidate-union entries minus one."
                    ),
                },
                "needs_review": {"type": "boolean"},
            },
            "required": ["selected_index", "needs_review"],
            "additionalProperties": False,
        },
        "metrics": {
            **agent_metrics,
            "copy_only_exact_path_correct": copy_metrics["exact_path_correct"],
            "copy_only_top_level_correct": copy_metrics["top_level_correct"],
            "copy_only_macro_hierarchical_f1": copy_metrics["macro_hierarchical_f1"],
            "image_plus_copy_exact_path_correct": image_metrics["exact_path_correct"],
            "image_plus_copy_top_level_correct": image_metrics["top_level_correct"],
            "image_plus_copy_macro_hierarchical_f1": image_metrics[
                "macro_hierarchical_f1"
            ],
            "agent_exact_path_correct": agent_metrics["exact_path_correct"],
            "agent_top_level_correct": agent_metrics["top_level_correct"],
            "agent_macro_hierarchical_f1": agent_metrics["macro_hierarchical_f1"],
        },
        "results": [
            {
                **asdict(decision),
                "ground_truth_path": listing.ground_truth_path,
                "source": {
                    "viewer_url": listing.viewer_url,
                    "image_sha256": listing.image_sha256,
                    "row_sha256": listing_sha256(listing),
                },
                "candidate_paths": listing.candidate_paths,
            }
            for listing, decision in zip(completed_listings, decisions, strict=True)
        ],
    }


def _write_evaluation_output(path: Path, output: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(output, temporary, indent=2, ensure_ascii=False)
            temporary.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_checkpoint(
    path: Path,
    listings: list[CatalogListing],
    *,
    offset: int,
) -> dict[int, CatalogDecision]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_dataset = {
        "id": DATASET,
        "revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "row_window": [offset, offset + len(listings) - 1],
    }
    expected_models = {
        "reranker": RERANKER_MODEL,
        "verifier": VERIFIER_MODEL,
    }
    if payload.get("record_type") != "sie_catalog_agent_evaluation":
        raise ValueError(f"Cannot resume from {path}: unexpected record type")
    endpoint, _api_key = read_sie_settings()
    if payload.get("endpoint") != endpoint.rstrip("/"):
        raise ValueError(f"Cannot resume from {path}: SIE endpoint changed")
    if (
        payload.get("dataset") != expected_dataset
        or payload.get("models") != expected_models
    ):
        raise ValueError(f"Cannot resume from {path}: dataset window or models changed")

    listings_by_row = {listing.row_idx: listing for listing in listings}
    decisions: dict[int, CatalogDecision] = {}
    request_ids: list[str] = []
    for result in payload.get("results", []):
        row_idx = int(result["row_idx"])
        listing = listings_by_row.get(row_idx)
        if listing is None:
            raise ValueError(f"Cannot resume from {path}: unexpected row {row_idx}")
        if row_idx in decisions:
            raise ValueError(f"Cannot resume from {path}: duplicate row {row_idx}")
        source = result.get("source")
        if not isinstance(source, dict) or source.get("row_sha256") != listing_sha256(
            listing
        ):
            raise ValueError(
                f"Cannot resume from {path}: source changed for row {row_idx}"
            )
        if result.get("ground_truth_path") != listing.ground_truth_path:
            raise ValueError(
                f"Cannot resume from {path}: reference changed for row {row_idx}"
            )
        for score_field in ("text_scores", "image_plus_copy_scores"):
            scores = result.get(score_field)
            if (
                not isinstance(scores, list)
                or len(scores) != len(listing.candidate_paths)
                or any(
                    not isinstance(score, int | float) or isinstance(score, bool)
                    for score in scores
                )
            ):
                raise ValueError(
                    f"Cannot resume from {path}: {score_field} changed for row {row_idx}"
                )
        text_scores = result["text_scores"]
        image_plus_copy_scores = result["image_plus_copy_scores"]
        expected_candidate_union = candidate_union(
            listing.candidate_paths,
            text_scores,
            image_plus_copy_scores,
        )
        if result.get("candidate_union") != expected_candidate_union:
            raise ValueError(
                f"Cannot resume from {path}: candidate union changed for row {row_idx}"
            )
        selected_path = result.get("selected_path")
        if (
            not isinstance(selected_path, str)
            or selected_path not in expected_candidate_union
        ):
            raise ValueError(
                f"Cannot resume from {path}: selected path changed for row {row_idx}"
            )
        needs_review = result.get("needs_review")
        if not isinstance(needs_review, bool):
            raise ValueError(
                f"Cannot resume from {path}: needs review changed for row {row_idx}"
            )
        verifier_response_id = result.get("verifier_response_id")
        if not isinstance(verifier_response_id, str) or not verifier_response_id:
            raise ValueError(
                f"Cannot resume from {path}: verifier_response_id missing for row "
                f"{row_idx}"
            )
        api_calls = result.get("api_calls")
        if not isinstance(api_calls, list):
            raise ValueError(
                f"Cannot resume from {path}: api_calls missing for row {row_idx}"
            )
        decision = CatalogDecision(
            row_idx=row_idx,
            selected_path=selected_path,
            needs_review=needs_review,
            candidate_union=expected_candidate_union,
            text_scores=text_scores,
            image_plus_copy_scores=image_plus_copy_scores,
            verifier_response_id=verifier_response_id,
            api_calls=api_calls,
        )
        row_request_ids = _validate_api_calls(decision.api_calls, row_idx=row_idx)
        verifier_call = next(
            call
            for call in decision.api_calls
            if call["stage"] == "candidate_verification"
        )
        if decision.verifier_response_id != verifier_call["request_id"]:
            raise ValueError(
                f"Cannot resume from {path}: verifier request changed for row {row_idx}"
            )
        request_ids.extend(row_request_ids)
        decisions[row_idx] = decision
    if len(request_ids) != len(set(request_ids)):
        raise ValueError(f"Cannot resume from {path}: duplicate SIE request IDs")
    return decisions


def _evaluation_run_command(args: argparse.Namespace) -> str:
    parts = [
        "eval-catalog-agent",
        "--offset",
        str(args.offset),
        "--limit",
        str(args.limit),
        "--cache-dir",
        args.cache_dir.as_posix(),
        "--output",
        args.output.as_posix(),
    ]
    if args.summary_output is not None:
        parts.extend(["--summary-output", args.summary_output.as_posix()])
    return shlex.join(parts)


def eval_main() -> None:
    parser = _common_parser(
        "Evaluate the focused multimodal catalog agent on Shopify listings."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    if args.summary_output is not None and args.summary_output.resolve() == (
        args.output.resolve()
    ):
        parser.error("--summary-output must differ from --output")
    run_command = _evaluation_run_command(args)
    listings = load_shopify_rows(
        offset=args.offset,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )
    decisions_by_row = _load_checkpoint(args.output, listings, offset=args.offset)
    if decisions_by_row:
        print(
            f"Resuming with {len(decisions_by_row)} completed rows from {args.output}"
        )
    pending_listings = [
        listing for listing in listings if listing.row_idx not in decisions_by_row
    ]
    if pending_listings:
        with create_sie_client(timeout_s=600) as client:
            for listing in pending_listings:
                decision = classify_listing(client, listing)
                decisions_by_row[listing.row_idx] = decision
                output = _evaluation_output(
                    listings,
                    decisions_by_row,
                    offset=args.offset,
                    run_command=run_command,
                )
                _write_evaluation_output(args.output, output)
                print(
                    f"{listing.row_idx:>5} "
                    f"{'review' if decision.needs_review else 'accept':>6} "
                    f"{decision.selected_path}"
                )

    output = _evaluation_output(
        listings,
        decisions_by_row,
        offset=args.offset,
        run_command=run_command,
    )
    _write_evaluation_output(args.output, output)
    if args.summary_output is not None:
        _write_evaluation_output(
            args.summary_output,
            {
                "record_type": "sie_catalog_agent_evaluation_summary",
                "run_at": output["run_at"],
                "endpoint": output["endpoint"],
                "run_command": output["run_command"],
                "timing_note": output["timing_note"],
                "dataset": output["dataset"],
                "models": {
                    "reranker": {
                        "id": output["models"]["reranker"],
                        "revision": RERANKER_REVISION,
                    },
                    "verifier": {
                        "id": output["models"]["verifier"],
                        "revision": VERIFIER_REVISION,
                    },
                },
                "candidate_rule": output["candidate_rule"],
                "response_schema": output["response_schema"],
                "metrics": output["metrics"],
                "evaluation": {
                    "path": args.output.as_posix(),
                    "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                },
            },
        )
    print(json.dumps(output["metrics"], indent=2))


def predict_main() -> None:
    parser = _common_parser("Run the focused catalog agent on one Shopify listing.")
    parser.set_defaults(limit=1)
    args = parser.parse_args()
    listings = load_shopify_rows(
        offset=args.offset,
        limit=args.limit,
        cache_dir=args.cache_dir,
    )
    with create_sie_client(timeout_s=600) as client:
        decision = classify_listing(client, listings[0])
    print(json.dumps(asdict(decision), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    eval_main()
