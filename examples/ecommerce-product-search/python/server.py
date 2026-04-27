"""FastAPI backend for product search."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sie_sdk import SIEClient

from search import load_index, search

config = None
embeddings = None
metadata = None
client = None
category_counts = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, embeddings, metadata, client, category_counts
    config = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    embeddings, metadata = load_index()
    cluster_url = os.environ.get("SIE_CLUSTER_URL", config["cluster"]["url"])
    api_key = os.environ.get("SIE_API_KEY", config["cluster"]["api_key"])
    client = SIEClient(cluster_url, api_key=api_key)
    cats = {}
    for m in metadata:
        cats[m["category"]] = cats.get(m["category"], 0) + 1
    category_counts = sorted(cats.items(), key=lambda x: -x[1])
    yield
    client.close()


app = FastAPI(title="SIE Product Search", lifespan=lifespan)

static_dir = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def index():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1),
    category: str | None = Query(None),
    brand: str | None = Query(None),
):
    filters = {}
    if category:
        filters["category"] = category
    if brand:
        filters["brand"] = brand

    results = search(client, config, embeddings, metadata, q, filters if filters else None)

    return {
        "query": q,
        "filters": filters,
        "results": [
            {
                "id": r["id"],
                "title": r["title"],
                "description": r["description"],
                "category": r["category"],
                "price": r.get("price"),
                "rating": r.get("rating"),
                "image": r.get("image"),
                "features": r.get("features", []),
                "attributes": r.get("attributes", {}),
                "scores": {
                    "vector": round(r["_vector_score"], 4),
                    "rerank": round(r["_rerank_score"], 4),
                },
            }
            for r in results
        ],
    }


@app.get("/api/categories")
def api_categories():
    return category_counts


@app.get("/api/stats")
def api_stats():
    return {
        "total_products": len(metadata),
        "embedding_dims": embeddings.shape[1],
        "categories": len(category_counts),
        "models": config["models"],
    }
