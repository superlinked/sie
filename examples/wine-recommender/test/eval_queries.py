import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from wine_flavor import datasource, engine, pretty_print, transforms

load_dotenv()

SIE_BASE_URL = os.getenv("CLUSTER_URL")
SIE_API_KEY = os.getenv("API_KEY")
SIE_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
ALLOWED_SIE_RERANK_MODELS = {
    "BAAI/bge-reranker-v2-m3",
    "jinaai/jina-reranker-v2-base-multilingual",
}
COSINE_TOP_K = 3
REVIEWS_PER_WINE = 5
RERANK_MAX_TERMS = 12
REFERENCE_ROW_INDICES = []
FETCH_NUM_PAGES = 5

SCENARIOS = [
    {
        "name": "Fruity Red",
        "user_preferences": {
            "structure": {
                "acidity": 0.7,
                "fizziness": 0.0,
                "intensity": 0.8,
                "sweetness": 0.3,
                "tannin": 0.6,
            },
            "flavors": {
                "black cherry": 1.0,
                "plum": 0.9,
                "vanilla": 0.7,
                "oak": 0.6,
                "earthy": 0.8,
            },
        },
    },
    {
        "name": "Earthy Red",
        "user_preferences": {
            "structure": {
                "acidity": 0.6,
                "fizziness": 0.0,
                "intensity": 0.7,
                "sweetness": 0.2,
                "tannin": 0.7,
            },
            "flavors": {
                "earthy": 1.0,
                "leather": 0.8,
                "smoke": 0.7,
                "tobacco": 0.7,
                "minerals": 0.5,
            },
        },
    },
    {
        "name": "Crisp White",
        "user_preferences": {
            "structure": {
                "acidity": 0.8,
                "fizziness": 0.1,
                "intensity": 0.5,
                "sweetness": 0.2,
                "tannin": 0.1,
            },
            "flavors": {
                "citrus": 1.0,
                "orange zest": 0.8,
                "minerals": 0.7,
                "floral": 0.5,
            },
        },
    },
]


def validate_config():
    if not SIE_BASE_URL:
        raise ValueError("Missing CLUSTER_URL in the environment.")
    if not SIE_API_KEY:
        raise ValueError("Missing API_KEY in the environment.")
    if SIE_RERANK_MODEL not in ALLOWED_SIE_RERANK_MODELS:
        raise ValueError(
            f"Unsupported SIE reranker model '{SIE_RERANK_MODEL}'. "
            f"Choose one of: {sorted(ALLOWED_SIE_RERANK_MODELS)}"
        )


def build_cosine_results_frame(wines, top_matches):
    match_rows = []
    for rank, match in enumerate(top_matches):
        match_rows.append(
            {
                "row_index": match["row_index"],
                "cosine_score": match["similarity_score"],
                "cosine_rank": rank,
            }
        )

    match_frame = pd.DataFrame(match_rows)
    return match_frame.merge(
        wines.reset_index().rename(columns={"index": "row_index"})[
            [
                "row_index",
                "wine_id",
                "wine_name",
                "winery_name",
                "vintage_year",
                "rating_average",
                "country_name",
                "region_name",
                "price_amount",
                "price_currency",
                "review_count",
            ]
        ],
        on="row_index",
        how="left",
    ).sort_values(["cosine_rank", "cosine_score"], ascending=[True, False])


def main():
    validate_config()

    print("Fetching wines...", flush=True)
    wines = datasource.fetch_vivino_wines(num_pages=FETCH_NUM_PAGES)
    print("Fetching reviews...", flush=True)
    wines = datasource.attach_vivino_reviews(
        wines,
        review_pages=1,
        reviews_per_page=REVIEWS_PER_WINE,
        language="en",
    )

    print("Building shared vectors...", flush=True)
    unique_flavors = transforms.unique_flavors(wines)
    flavor_idf = engine.build_flavor_idf(wines)
    wine_matrix = engine.build_wine_matrix(wines, unique_flavors, flavor_idf)

    for scenario in SCENARIOS:
        scenario_name = scenario["name"]
        user_preferences = scenario["user_preferences"]
        user_vector = engine.build_user_vector(user_preferences, unique_flavors, flavor_idf)
        top_matches = engine.cosine_similarity_search(user_vector, wine_matrix, top_k=COSINE_TOP_K)
        candidate_row_indices = [match["row_index"] for match in top_matches]

        print(f"\n=== {scenario_name} ===", flush=True)
        cosine_results = build_cosine_results_frame(wines, top_matches)
        print("\n--- Cosine Top Results ---")
        print(cosine_results.round({"cosine_score": 4}).to_string(index=False))

        reranked_matches = engine.rerank_wines_with_sie_reviews(
            wines,
            candidate_row_indices,
            user_preferences,
            unique_flavors,
            flavor_idf,
            reference_row_indices=REFERENCE_ROW_INDICES,
            max_terms=RERANK_MAX_TERMS,
            base_url=SIE_BASE_URL,
            model_name=SIE_RERANK_MODEL,
        )
        top_results = pretty_print.build_results_frame(wines, reranked_matches)
        pretty_print.print_top_results(top_results)


if __name__ == "__main__":
    main()
