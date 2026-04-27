import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()


def _resolve_sie_connection(base_url=None):
    resolved_base_url = base_url or os.getenv("CLUSTER_URL")
    if not resolved_base_url:
        raise ValueError("Missing SIE base URL. Set CLUSTER_URL in the environment or pass base_url explicitly.")

    api_key = os.getenv("API_KEY") or None

    return resolved_base_url, api_key


def pull_wine_reviews(wine_row):
    reviews = wine_row.get("wine_reviews") or wine_row.get("reviews") or []
    normalized_reviews = []

    for review in reviews:
        if isinstance(review, str) and review.strip():
            normalized_reviews.append(review.strip())
        elif isinstance(review, dict):
            review_text = review.get("text") or review.get("review") or review.get("note")
            if review_text:
                normalized_reviews.append(str(review_text).strip())

    return normalized_reviews


def _structure_label(value):
    if value >= 0.8:
        return "high"
    if value >= 0.6:
        return "medium-high"
    if value >= 0.4:
        return "medium"
    if value >= 0.2:
        return "low"
    return "very low"


def build_standard_rerank_query_weights(
    user_preferences,
    wines,
    reference_row_indices,
    all_flavors,
    flavor_idf=None,
    alpha=None,
    max_terms=None,
):
    from .vectors import build_user_vector, build_wine_vector

    user_vector = build_user_vector(user_preferences, all_flavors, flavor_idf)
    reference_vectors = []

    for row_index in reference_row_indices or []:
        wine_vector = build_wine_vector(wines.iloc[row_index], all_flavors, flavor_idf)
        reference_vectors.append(wine_vector)

    if reference_vectors:
        average_reference_vector = np.mean(np.vstack(reference_vectors), axis=0)
    else:
        average_reference_vector = np.zeros_like(user_vector)

    final_query_vector = alpha * user_vector + (1 - alpha) * average_reference_vector

    structure_term_weights = {}
    flavor_term_weights = {}
    structure_names = ["acidity", "fizziness", "intensity", "sweetness", "tannin"]

    for i, structure_name in enumerate(structure_names):
        structure_weight = float(final_query_vector[i])
        if structure_weight > 0:
            structure_term_weights[f"{_structure_label(structure_weight)} {structure_name}"] = structure_weight

    for i, flavor_name in enumerate(all_flavors, start=5):
        flavor_weight = float(final_query_vector[i])
        if flavor_weight > 0:
            flavor_term_weights[flavor_name] = flavor_weight

    if max_terms is not None:
        sorted_flavor_terms = sorted(flavor_term_weights.items(), key=lambda item: item[1], reverse=True)
        flavor_term_weights = dict(sorted_flavor_terms[:max_terms])

    term_weights = {**structure_term_weights, **flavor_term_weights}

    return term_weights, final_query_vector


def rerank_wines_with_sie_reviews(
    wines,
    candidate_row_indices,
    user_preferences,
    all_flavors,
    flavor_idf=None,
    *,
    reference_row_indices=None,
    alpha=None,
    max_terms=12,
    base_url=None,
    model_name="BAAI/bge-reranker-v2-m3",
    gpu=None,
    provision_timeout_s=900,
):
    try:
        from sie_sdk import SIEClient
    except ImportError as exc:
        raise ImportError("sie_sdk is required for SIE reranking.") from exc

    base_url, api_key = _resolve_sie_connection(base_url)
    client = SIEClient(base_url, api_key=api_key)
    term_weights, final_query_vector = build_standard_rerank_query_weights(
        user_preferences,
        wines,
        reference_row_indices,
        all_flavors,
        flavor_idf,
        alpha=alpha,
        max_terms=max_terms,
    )

    wine_scores = []
    for row_index in candidate_row_indices:
        wine_row = wines.iloc[row_index]
        reviews = pull_wine_reviews(wine_row)

        if not reviews:
            wine_scores.append({
                "row_index": int(row_index),
                "rerank_score": 0.0,
                "review_count": 0,
            })
            continue

        review_items = [{"id": f"review-{review_index}", "text": review_text} for review_index, review_text in enumerate(reviews)]
        review_score_totals = np.zeros(len(reviews), dtype=float)

        for term_text, term_weight in term_weights.items():
            score_result = client.score(
                model_name,
                {"id": f"term-{term_text}", "text": term_text},
                review_items,
                gpu=gpu,
                wait_for_capacity=True,
                provision_timeout_s=provision_timeout_s,
            )

            for score_entry in score_result.get("scores", []):
                review_item_id = score_entry["item_id"]
                review_index = int(review_item_id.split("-")[-1])
                review_score_totals[review_index] += term_weight * float(score_entry["score"])

        final_wine_score = float(np.sum(review_score_totals) / len(reviews))
        wine_scores.append({
            "row_index": int(row_index),
            "rerank_score": final_wine_score,
            "review_count": len(reviews),
        })

    ranked_wines = sorted(wine_scores, key=lambda item: item["rerank_score"], reverse=True)
    for rank, wine_score in enumerate(ranked_wines):
        wine_score["rerank_rank"] = rank
        wine_score["query_vector_length"] = len(final_query_vector)

    return ranked_wines
