import json
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
from wine_flavor import datasource, service
from wine_picture_detection import detect_wine_from_image_bytes

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
            raise ValueError(
                f"Missing required environment variable: {name} (or legacy {fallback_name})"
            )
        raise ValueError(f"Missing required environment variable: {name}")
    return float(value)


def _require_int_env(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return int(default)
    return int(value)


try:
    SIE_BASE_URL = _require_env("CLUSTER_URL")
    SIE_API_KEY = os.getenv("API_KEY") or None
    RERANK_METHOD = _require_env("RERANK_METHOD")
    SIE_RERANK_MODEL = _require_env("SIE_RERANK_MODEL")
    SIE_EMBEDDING_MODEL = _require_env("SIE_EMBEDDING_MODEL")
    RERANK_ALPHA = _require_float_env("RERANK_ALPHA")
    CUSTOM_RERANK_A = _require_float_env("CUSTOM_RERANK_A")
    CUSTOM_RERANK_NO_REVIEW_PENALTY = float(
        _require_env("CUSTOM_RERANK_NO_REVIEW_PENALTY")
    )
except ValueError as exc:
    raise ValueError(f"{exc}. Add it to your .env file.") from exc

ALLOWED_SIE_RERANK_MODELS = {
    "BAAI/bge-reranker-v2-m3",
    "jinaai/jina-reranker-v2-base-multilingual",
}
ALLOWED_RERANK_METHODS = {"standard", "custom"}
DEMO_NUM_PAGES = _require_int_env("DEMO_NUM_PAGES", 5)
REVIEWS_PER_WINE = _require_int_env("REVIEWS_PER_WINE", 5)
REVIEW_PAGES = _require_int_env("REVIEW_PAGES", 1)
COSINE_TOP_K = _require_int_env("COSINE_TOP_K", 20)
RERANK_MAX_TERMS = _require_int_env("RERANK_MAX_TERMS", 12)
SIE_GPU = os.getenv("SIE_GPU", "l4-spot")
SIE_PROVISION_TIMEOUT_S = _require_int_env("SIE_PROVISION_TIMEOUT_S", 900)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
DB_PATH = Path("wine_flavor.db")

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


class StructurePreferences(BaseModel):
    acidity: float = Field(ge=0.0, le=1.0)
    fizziness: float = Field(ge=0.0, le=1.0)
    intensity: float = Field(ge=0.0, le=1.0)
    sweetness: float = Field(ge=0.0, le=1.0)
    tannin: float = Field(ge=0.0, le=1.0)


class RecommendationRequest(BaseModel):
    structure: StructurePreferences
    flavors: dict[str, float] = Field(default_factory=dict)
    reference_row_indices: list[int] = Field(default_factory=list)
    top_k: int = Field(default=COSINE_TOP_K, ge=1, le=50)


class FlavorPromptRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)


class DemoCatalog:
    def __init__(self):
        self._lock = Lock()
        self._loaded = False
        self.wines = None
        self.unique_flavors = None
        self.flavor_idf = None
        self.wine_matrix = None

    def _load_from_sqlite(self):
        if not DB_PATH.exists():
            raise FileNotFoundError(f"Catalog database not found at {DB_PATH}")

        connection = sqlite3.connect(DB_PATH)
        try:
            wines = pd.read_sql_query(
                """
                SELECT
                    wine_id,
                    winery_name,
                    wine_name,
                    vintage_year,
                    rating_average,
                    ratings_count,
                    country_name,
                    region_name,
                    taste_acidity,
                    taste_fizziness,
                    taste_intensity,
                    taste_sweetness,
                    taste_tannin,
                    wine_flavors_json,
                    style_name,
                    style_varietal_name,
                    price_amount,
                    price_currency
                FROM wines
                ORDER BY wine_id
                """,
                connection,
            )
        finally:
            connection.close()

        if wines.empty:
            raise RuntimeError("Local SQLite catalog is empty.")

        wines["wine_flavors"] = wines["wine_flavors_json"].apply(
            lambda value: json.loads(value or "[]")
        )
        wines = wines.drop(columns=["wine_flavors_json"])
        wines["review_count"] = 0
        return wines

    def load(self, force=False):
        with self._lock:
            if self._loaded and not force:
                return

            if DB_PATH.exists():
                wines = self._load_from_sqlite()
            else:
                wines = datasource.fetch_vivino_wines(num_pages=DEMO_NUM_PAGES)

                wines = datasource.attach_vivino_reviews(
                    wines,
                    review_pages=REVIEW_PAGES,
                    reviews_per_page=REVIEWS_PER_WINE,
                    language="en",
                )

            catalog_assets = service.load_catalog_assets(wines)

            self.wines = wines
            self.unique_flavors = catalog_assets["unique_flavors"]
            self.flavor_idf = catalog_assets["flavor_idf"]
            self.wine_matrix = catalog_assets["wine_matrix"]
            self._loaded = True


catalog = DemoCatalog()


def _request_to_preferences(payload):
    return {
        "structure": payload.structure.model_dump(),
        "flavors": {name: float(value) for name, value in payload.flavors.items()},
    }


def _normalize_record(record):
    if isinstance(record, dict):
        return {key: _normalize_record(value) for key, value in record.items()}

    if isinstance(record, list):
        return [_normalize_record(value) for value in record]

    if isinstance(record, tuple):
        return [_normalize_record(value) for value in record]

    try:
        if pd.isna(record):
            return None
    except (TypeError, ValueError):
        pass

    return record


def _to_ui_structure_from_row(wine_row):
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


def _extract_wine_flavors(wine_row, max_flavors=6):
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


def _wine_style(wine_row):
    return wine_row.get("style_name") or wine_row.get("style_varietal_name") or "Wine"


def _catalog_wine_record(wine_row):
    wine_id = wine_row.get("wine_id")
    row_index = wine_row.get("row_index")
    if row_index is None:
        row_index = getattr(wine_row, "name", -1)
    return _normalize_record(
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
            "style": _wine_style(wine_row),
            "price": wine_row.get("price_amount"),
            "structure": _to_ui_structure_from_row(wine_row),
            "flavors": _extract_wine_flavors(wine_row),
        }
    )


def _load_detected_wine_from_db(wine_id):
    if wine_id is None or not DB_PATH.exists():
        return None

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        row = connection.execute(
            """
            SELECT
                wine_id,
                winery_name,
                wine_name,
                vintage_year,
                country_name,
                region_name,
                style_name,
                style_varietal_name,
                price_amount,
                wine_flavors_json,
                taste_acidity,
                taste_fizziness,
                taste_intensity,
                taste_sweetness,
                taste_tannin
            FROM wines
            WHERE wine_id = ?
            """,
            (int(wine_id),),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        return None

    return _catalog_wine_record(
        {
            "wine_id": row["wine_id"],
            "winery_name": row["winery_name"],
            "wine_name": row["wine_name"],
            "vintage_year": row["vintage_year"],
            "country_name": row["country_name"],
            "region_name": row["region_name"],
            "style_name": row["style_name"],
            "style_varietal_name": row["style_varietal_name"],
            "price_amount": row["price_amount"],
            "wine_flavors": json.loads(row["wine_flavors_json"] or "[]"),
            "taste_acidity": row["taste_acidity"],
            "taste_fizziness": row["taste_fizziness"],
            "taste_intensity": row["taste_intensity"],
            "taste_sweetness": row["taste_sweetness"],
            "taste_tannin": row["taste_tannin"],
        }
    )


def _select_detected_wine_record(wines, detected_wine_id=None):
    if detected_wine_id is None:
        return None

    if wines is not None and not wines.empty and detected_wine_id is not None:
        matches = wines[wines["wine_id"] == detected_wine_id]
        if not matches.empty:
            return _catalog_wine_record(matches.iloc[0])

    db_record = _load_detected_wine_from_db(detected_wine_id)
    if db_record is not None:
        return db_record

    return None


def _build_flavor_tags(wines):
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
        {
            "category": category,
            "flavors": sorted(list(flavors)),
        }
        for category, flavors in sorted(grouped_flavors.items())
        if flavors
    ]


def _prompt_flavor_adjustment(prompt: str, available_flavors: list[str]) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing required environment variable: OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4.1-mini",
        text={"format": {"type": "json_object"}},
        input=[
            {
                "role": "system",
                "content": (
                    "Return only JSON with keys structure and flavors. "
                    "Structure must contain acidity, fizziness, intensity, sweetness, tannin as integers from 0 to 100. "
                    "Flavors must be an array of up to 8 strings chosen only from the provided list."
                ),
            },
            {
                "role": "user",
                "content": f"Prompt: {prompt}\nAvailable flavors: {', '.join(available_flavors[:250])}",
            },
        ],
    )

    output_text = (response.output_text or "").strip()
    if not output_text:
        raise ValueError("OpenAI returned an empty response")

    try:
        return json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI returned invalid JSON: {output_text}") from exc


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


@asynccontextmanager
async def lifespan(_app):
    try:
        catalog.load()
    except Exception:
        # Keep the API bootable even if remote catalog fetches fail during startup.
        pass
    yield


app = FastAPI(title="Wine Flavor Demo API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "Wine Flavor Demo API",
        "status": "ok",
        "health": "/health",
        "recommendations": "/recommendations",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rerank_method": RERANK_METHOD,
        "catalog_loaded": catalog._loaded,
        "demo_num_pages": DEMO_NUM_PAGES,
    }


@app.get("/catalog")
def catalog_view():
    catalog.load()
    return jsonable_encoder(service.build_catalog_response(catalog.wines))


@app.post("/recommendations")
def recommendations(payload: RecommendationRequest):
    catalog.load()
    user_preferences = _request_to_preferences(payload)
    return jsonable_encoder(
        service.get_recommendations(
            catalog.wines,
            catalog.unique_flavors,
            catalog.flavor_idf,
            catalog.wine_matrix,
            user_preferences,
            top_k=payload.top_k,
            reference_row_indices=payload.reference_row_indices,
            rerank_method=RERANK_METHOD,
            base_url=SIE_BASE_URL,
            rerank_model=SIE_RERANK_MODEL,
            embedding_model=SIE_EMBEDDING_MODEL,
            gpu=SIE_GPU,
            provision_timeout_s=SIE_PROVISION_TIMEOUT_S,
            rerank_alpha=RERANK_ALPHA,
            custom_rerank_a=CUSTOM_RERANK_A,
            custom_rerank_no_review_penalty=CUSTOM_RERANK_NO_REVIEW_PENALTY,
            rerank_max_terms=RERANK_MAX_TERMS,
        )
    )


@app.post("/analyze-flavor-prompt")
def analyze_flavor_prompt(payload: FlavorPromptRequest):
    catalog.load()
    available_flavors = sorted(set(catalog.unique_flavors or []))
    available_flavors_set = set(available_flavors)

    try:
        result = _prompt_flavor_adjustment(payload.prompt, available_flavors)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Prompt analysis failed: {exc}"
        ) from exc

    structure = result.get("structure") or {}
    flavors = [
        flavor
        for flavor in (result.get("flavors") or [])
        if flavor in available_flavors_set
    ]
    return jsonable_encoder(
        {
            "structure": {
                "acidity": max(0, min(100, int(structure.get("acidity", 50)))),
                "fizziness": max(0, min(100, int(structure.get("fizziness", 0)))),
                "intensity": max(0, min(100, int(structure.get("intensity", 50)))),
                "sweetness": max(0, min(100, int(structure.get("sweetness", 10)))),
                "tannin": max(0, min(100, int(structure.get("tannin", 50)))),
            },
            "flavors": flavors[:8],
        }
    )


@app.post("/detect-wine-image")
async def detect_wine_image(file: UploadFile = File(...)):
    catalog.load()

    image_bytes = await file.read()
    uploads_dir = Path("uploads")
    original_name = file.filename or "upload.bin"
    suffix = Path(original_name).suffix or ".bin"
    saved_path = uploads_dir / f"{uuid4().hex}{suffix}"

    try:
        uploads_dir.mkdir(parents=True, exist_ok=True)
        saved_path.write_bytes(image_bytes)
        saved_path_value = str(saved_path)
    except OSError as exc:
        saved_path_value = f"save-failed:{exc.__class__.__name__}"

    print(
        "detect-wine-image upload:",
        {
            "filename": original_name,
            "content_type": file.content_type,
            "bytes": len(image_bytes),
            "saved_path": saved_path_value,
        },
    )

    detection = detect_wine_from_image_bytes(image_bytes)
    detected_wine = _select_detected_wine_record(catalog.wines, detection.wine_id)

    return jsonable_encoder(
        {
            "detected_wine": detected_wine,
            "match_score": detection.match_score,
            "ocr_text": detection.ocr_text,
        }
    )


@app.post("/reload")
def reload_catalog():
    catalog.load(force=True)
    return {
        "status": "reloaded",
        "wine_count": len(catalog.wines),
        "flavor_count": len(catalog.unique_flavors),
    }


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
