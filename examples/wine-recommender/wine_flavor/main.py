import os
from dotenv import load_dotenv
from . import datasource, engine, pretty_print, transforms

load_dotenv()


def _require_env(name):
    value = os.getenv(name)
    if value is None or value == "":
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _require_float_env(name, fallback_name=None):
    value = os.getenv(name)
    if (value is None or value == "") and fallback_name:
        value = os.getenv(fallback_name)
    if value is None or value == "":
        if fallback_name:
            raise ValueError(f"Missing required environment variable: {name} (or legacy {fallback_name})")
        raise ValueError(f"Missing required environment variable: {name}")
    return float(value)


try:
    SIE_BASE_URL = _require_env("CLUSTER_URL")
    SIE_API_KEY = _require_env("API_KEY")
    RERANK_METHOD = _require_env("RERANK_METHOD")
    SIE_RERANK_MODEL = _require_env("SIE_RERANK_MODEL")
    SIE_EMBEDDING_MODEL = _require_env("SIE_EMBEDDING_MODEL")
    RERANK_ALPHA = _require_float_env("RERANK_ALPHA")
    CUSTOM_RERANK_A = _require_float_env("CUSTOM_RERANK_A")
    CUSTOM_RERANK_NO_REVIEW_PENALTY = float(_require_env("CUSTOM_RERANK_NO_REVIEW_PENALTY"))
except ValueError as exc:
    raise ValueError(f"{exc}. Add it to your .env file.") from exc
ALLOWED_SIE_RERANK_MODELS = {
    "BAAI/bge-reranker-v2-m3",
    "jinaai/jina-reranker-v2-base-multilingual",
}
ALLOWED_RERANK_METHODS = {"standard", "custom"}
COSINE_TOP_K = 3
REVIEWS_PER_WINE = 5
RERANK_MAX_TERMS = 12
REFERENCE_ROW_INDICES = []


if RERANK_METHOD not in ALLOWED_RERANK_METHODS:
    raise ValueError(
        f"Unsupported rerank method '{RERANK_METHOD}'. "
        f"Choose one of: {sorted(ALLOWED_RERANK_METHODS)}"
    )

if SIE_RERANK_MODEL not in ALLOWED_SIE_RERANK_MODELS:
    raise ValueError(
        f"Unsupported SIE reranker model '{SIE_RERANK_MODEL}'. "
        f"Choose one of: {sorted(ALLOWED_SIE_RERANK_MODELS)}"
    )

print("Fetching wines...", flush=True)
wines = datasource.fetch_vivino_wines(num_pages=1)
print("Fetching reviews...", flush=True)
wines = datasource.attach_vivino_reviews(wines, review_pages=1, reviews_per_page=REVIEWS_PER_WINE, language="en")

# get all unique flavors from wine
print("Building vectors...", flush=True)
unique_flavors = transforms.unique_flavors(wines)
flavor_idf = engine.build_flavor_idf(wines)

# test example vector 
user_preferences = {
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
}

wine_matrix = engine.build_wine_matrix(wines, unique_flavors, flavor_idf)
user_vector = engine.build_user_vector(user_preferences, unique_flavors, flavor_idf)
top_matches = engine.cosine_similarity_search(user_vector, wine_matrix, top_k=COSINE_TOP_K)
candidate_row_indices = [match["row_index"] for match in top_matches]

print("Reranking candidates with SIE...", flush=True)
if RERANK_METHOD == "custom":
    reranked_matches = engine.rerank_wines_with_custom_embeddings(
        wines,
        candidate_row_indices,
        user_preferences,
        reference_row_indices=REFERENCE_ROW_INDICES,
        base_url=SIE_BASE_URL,
        model_name=SIE_EMBEDDING_MODEL,
        a=CUSTOM_RERANK_A,
        alpha=RERANK_ALPHA,
        no_review_penalty=CUSTOM_RERANK_NO_REVIEW_PENALTY,
    )
else:
    reranked_matches = engine.rerank_wines_with_sie_reviews(
        wines,
        candidate_row_indices,
        user_preferences,
        unique_flavors,
        flavor_idf,
        reference_row_indices=REFERENCE_ROW_INDICES,
        alpha=RERANK_ALPHA,
        max_terms=RERANK_MAX_TERMS,
        base_url=SIE_BASE_URL,
        model_name=SIE_RERANK_MODEL,
    )
top_results = pretty_print.build_results_frame(wines, reranked_matches)
pretty_print.print_top_results(top_results)
