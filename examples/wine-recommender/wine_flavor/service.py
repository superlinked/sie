import pandas as pd

from . import engine, pretty_print, transforms


def load_catalog_assets(wines):
    unique_flavors = transforms.unique_flavors(wines)
    flavor_idf = engine.build_flavor_idf(wines)
    wine_matrix = engine.build_wine_matrix(wines, unique_flavors, flavor_idf)
    return {
        "unique_flavors": unique_flavors,
        "flavor_idf": flavor_idf,
        "wine_matrix": wine_matrix,
    }


def normalize_record(record):
    if isinstance(record, dict):
        return {key: normalize_record(value) for key, value in record.items()}

    if isinstance(record, list):
        return [normalize_record(value) for value in record]

    if isinstance(record, tuple):
        return [normalize_record(value) for value in record]

    try:
        if pd.isna(record):
            return None
    except (TypeError, ValueError):
        pass

    return record


def to_ui_structure_from_row(wine_row):
    def _scale(value):
        if value is None or pd.isna(value):
            return 0
        return int(round(float(value) * 20))

    return {
        "acidity": _scale(wine_row.get("taste_acidity")),
        "fizziness": _scale(wine_row.get("taste_fizziness")),
        "intensity": _scale(wine_row.get("taste_intensity")),
        "sweetness": _scale(wine_row.get("taste_sweetness")),
        "tannin": _scale(wine_row.get("taste_tannin")),
    }


def extract_wine_flavors(wine_row, max_flavors=6):
    flavor_counts = {}
    for flavor_group in wine_row.get("wine_flavors", []) or []:
        for keyword in flavor_group.get("primary_keywords") or []:
            name = keyword.get("name")
            count = keyword.get("count", 1) or 1
            if name:
                flavor_counts[name] = flavor_counts.get(name, 0) + count
        for keyword in flavor_group.get("secondary_keywords") or []:
            name = keyword.get("name")
            count = keyword.get("count", 1) or 1
            if name:
                flavor_counts[name] = flavor_counts.get(name, 0) + count

    ordered_flavors = sorted(
        flavor_counts.items(), key=lambda item: item[1], reverse=True
    )
    return [name for name, _ in ordered_flavors[:max_flavors]]


def wine_style(wine_row):
    return wine_row.get("style_name") or wine_row.get("style_varietal_name") or "Wine"


def to_catalog_wine_record(wine_row):
    wine_id = wine_row.get("wine_id")
    row_index = wine_row.get("row_index")
    if row_index is None:
        row_index = getattr(wine_row, "name", -1)
    return normalize_record(
        {
            "id": (
                str(int(wine_id))
                if wine_id is not None and not pd.isna(wine_id)
                else ""
            ),
            "row_index": int(row_index),
            "name": wine_row.get("wine_name"),
            "winery": wine_row.get("winery_name"),
            "vintage": wine_row.get("vintage_year"),
            "country": wine_row.get("country_name"),
            "region": wine_row.get("region_name"),
            "style": wine_style(wine_row),
            "price": wine_row.get("price_amount"),
            "structure": to_ui_structure_from_row(wine_row),
            "flavors": extract_wine_flavors(wine_row),
        }
    )


def build_flavor_tags(wines):
    grouped_flavors = {}

    for _, wine_row in wines.iterrows():
        for flavor_group in wine_row.get("wine_flavors", []) or []:
            category = (flavor_group.get("group") or "other").replace("_", " ").title()
            grouped_flavors.setdefault(category, set())
            for keyword in flavor_group.get("primary_keywords") or []:
                if keyword.get("name"):
                    grouped_flavors[category].add(keyword["name"])
            for keyword in flavor_group.get("secondary_keywords") or []:
                if keyword.get("name"):
                    grouped_flavors[category].add(keyword["name"])

    return [
        {"category": category, "flavors": sorted(list(flavors))}
        for category, flavors in sorted(grouped_flavors.items())
        if flavors
    ]


def build_catalog_response(wines):
    return {
        "wines": [to_catalog_wine_record(wine_row) for _, wine_row in wines.iterrows()],
        "flavorTags": build_flavor_tags(wines),
    }


def _to_recommendation_record(record):
    score = float(record.get("rerank_score") or 0.0)
    return {
        "id": str(record.get("wine_id")),
        "name": record.get("wine_name"),
        "winery": record.get("winery_name"),
        "vintage": record.get("vintage_year"),
        "country": record.get("country_name"),
        "region": record.get("region_name"),
        "style": record.get("style"),
        "price": record.get("price_amount"),
        "matchPercentage": max(0, min(100, int(round(score * 100)))),
        "flavorSummary": ", ".join(record.get("flavors", [])),
        "structure": record.get("structure"),
        "flavors": record.get("flavors", []),
        "reviewCount": record.get("review_count"),
        "rerankScore": score,
    }


def get_recommendations(
    wines,
    unique_flavors,
    flavor_idf,
    wine_matrix,
    user_preferences,
    *,
    top_k,
    reference_row_indices,
    rerank_method,
    base_url,
    rerank_model,
    embedding_model,
    gpu,
    provision_timeout_s,
    rerank_alpha,
    custom_rerank_a,
    custom_rerank_no_review_penalty,
    rerank_max_terms,
):
    user_vector = engine.build_user_vector(user_preferences, unique_flavors, flavor_idf)
    top_matches = engine.cosine_similarity_search(user_vector, wine_matrix, top_k=top_k)
    candidate_row_indices = [match["row_index"] for match in top_matches]

    if rerank_method == "custom":
        reranked_matches = engine.rerank_wines_with_custom_embeddings(
            wines,
            candidate_row_indices,
            user_preferences,
            reference_row_indices=reference_row_indices,
            base_url=base_url,
            model_name=embedding_model,
            gpu=gpu,
            provision_timeout_s=provision_timeout_s,
            a=custom_rerank_a,
            alpha=rerank_alpha,
            no_review_penalty=custom_rerank_no_review_penalty,
        )
    else:
        reranked_matches = engine.rerank_wines_with_sie_reviews(
            wines,
            candidate_row_indices,
            user_preferences,
            unique_flavors,
            flavor_idf,
            reference_row_indices=reference_row_indices,
            alpha=rerank_alpha,
            max_terms=rerank_max_terms,
            base_url=base_url,
            model_name=rerank_model,
            gpu=gpu,
            provision_timeout_s=provision_timeout_s,
        )

    result_frame = pretty_print.build_results_frame(wines, reranked_matches)
    result_records = []
    for record in result_frame.to_dict(orient="records"):
        wine_row = wines.iloc[record["row_index"]]
        enriched_record = {
            **record,
            "style": wine_style(wine_row),
            "structure": to_ui_structure_from_row(wine_row),
            "flavors": extract_wine_flavors(wine_row),
        }
        result_records.append(normalize_record(enriched_record))

    return {
        "rerank_method": rerank_method,
        "candidate_count": len(candidate_row_indices),
        "results": [_to_recommendation_record(record) for record in result_records],
    }
