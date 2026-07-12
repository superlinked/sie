from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

SIE_URL = os.environ.get("SIE_URL", "http://localhost:8080").rstrip("/")
SIE_API_KEY = os.environ.get("SIE_API_KEY", "")
SIE_EMBED_MODEL = os.environ.get("SIE_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
STORE_PATH = Path(os.environ.get("PATRONUS_MEMORY_PATH", "data/memory.json"))
MAX_TEXT_CHARS = int(os.environ.get("PATRONUS_MAX_TEXT_CHARS", "12000"))

app = FastAPI(title="Patronus SIE Memory Bridge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    allow_credentials=False,
)


class IngestRequest(BaseModel):
    title: str = ""
    url: str = ""
    text: str = Field(min_length=1)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if SIE_API_KEY:
        headers["Authorization"] = f"Bearer {SIE_API_KEY}"
    return headers


def _clean_text(value: str, limit: int = MAX_TEXT_CHARS) -> str:
    return " ".join(str(value or "").split())[:limit]


def _memory_id(url: str, title: str) -> str:
    stable = f"{url.strip()}::{title.strip()}".encode("utf-8")
    return hashlib.sha256(stable).hexdigest()[:24]


def _load_store() -> list[dict[str, Any]]:
    if not STORE_PATH.exists():
        return []
    try:
        data = json.loads(STORE_PATH.read_text())
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid memory store: {STORE_PATH}")
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail=f"Memory store must be a list: {STORE_PATH}")
    return data


def _save_store(items: list[dict[str, Any]]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n")


async def _embed(input_texts: list[str]) -> list[list[float]]:
    payload = {"model": SIE_EMBED_MODEL, "input": input_texts}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{SIE_URL}/v1/embeddings", headers=_headers(), json=payload)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"SIE request failed: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"SIE returned {response.status_code}: {response.text[:300]}")

    body = response.json()
    data = body.get("data") or []
    vectors = [item.get("embedding") for item in data if isinstance(item, dict)]
    if len(vectors) != len(input_texts):
        raise HTTPException(status_code=502, detail="SIE embedding response did not match request size")
    return vectors


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb or 1.0)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"ok": "true", "sie_url": SIE_URL, "model": SIE_EMBED_MODEL}


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict[str, Any]:
    title = _clean_text(req.title, 240)
    url = _clean_text(req.url, 500)
    text = _clean_text(req.text)
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    memory_text = f"{title}\n{url}\n{text}".strip()
    vector = (await _embed([memory_text]))[0]
    item = {
        "id": _memory_id(url, title or text[:80]),
        "title": title or "Untitled page",
        "url": url,
        "text": text,
        "embedding": vector,
        "updated_at": int(time.time()),
    }

    items = _load_store()
    next_items = [old for old in items if old.get("id") != item["id"]]
    next_items.append(item)
    _save_store(next_items)
    return {"ok": True, "id": item["id"], "count": len(next_items)}


@app.post("/query")
async def query(req: QueryRequest) -> dict[str, Any]:
    items = [item for item in _load_store() if isinstance(item.get("embedding"), list)]
    if not items:
        return {"results": []}

    qvec = (await _embed([_clean_text(req.query, 1000)]))[0]
    scored = sorted(
        (
            {
                "title": item.get("title") or "Untitled page",
                "url": item.get("url") or "",
                "text": (item.get("text") or "")[:700],
                "score": round(_cosine(qvec, item["embedding"]), 4),
            }
            for item in items
        ),
        key=lambda row: row["score"],
        reverse=True,
    )
    return {"results": scored[: req.limit]}
