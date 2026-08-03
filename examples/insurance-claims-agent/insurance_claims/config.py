from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "sources"
PACKET_DIR = DATA_DIR / "claim-packet"
RUNS_DIR = ROOT / "runs"
FIXTURES_DIR = ROOT / "fixtures"


@dataclass(frozen=True)
class ClusterConfig:
    url: str
    generation_url: str
    api_key: str
    request_timeout_s: float
    provision_timeout_s: float


@dataclass(frozen=True)
class ModelsConfig:
    parse: str
    extract: str
    rerank: str
    vision: str
    review: str


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_chunks: int
    result_chunks: int
    chunk_characters: int


@dataclass(frozen=True)
class Source:
    slug: str
    title: str
    file_name: str
    media_type: str
    url: str
    source_page: str
    rights: str
    fixture_file: str | None = None

    @property
    def path(self) -> Path:
        return SOURCE_DIR / self.file_name

    @property
    def fixture_path(self) -> Path | None:
        return ROOT / self.fixture_file if self.fixture_file else None


@dataclass(frozen=True)
class AppConfig:
    cluster: ClusterConfig
    models: ModelsConfig
    retrieval: RetrievalConfig
    sources: tuple[Source, ...]


def _load_yaml() -> dict[str, Any]:
    value = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("config.yaml must contain an object")
    return value


def load_config() -> AppConfig:
    load_dotenv(ROOT / ".env")
    raw = _load_yaml()
    cluster_raw = raw["cluster"]
    url = os.environ.get("SIE_CLUSTER_URL", cluster_raw["url"]).rstrip("/")
    configured_generation_url = os.environ.get(
        "SIE_GENERATION_URL",
        cluster_raw.get("generation_url", ""),
    ).rstrip("/")
    model_raw = raw["models"]
    models = ModelsConfig(
        parse=os.environ.get("SIE_PARSE_MODEL", model_raw["parse"]),
        extract=os.environ.get("SIE_EXTRACT_MODEL", model_raw["extract"]),
        rerank=os.environ.get("SIE_RERANK_MODEL", model_raw["rerank"]),
        vision=os.environ.get("SIE_VISION_MODEL", model_raw["vision"]),
        review=os.environ.get("SIE_REVIEW_MODEL", model_raw["review"]),
    )
    retrieval_raw = raw["retrieval"]
    return AppConfig(
        cluster=ClusterConfig(
            url=url,
            generation_url=configured_generation_url or url,
            api_key=os.environ.get("SIE_API_KEY", cluster_raw.get("api_key", "")),
            request_timeout_s=float(cluster_raw.get("request_timeout_s", 900)),
            provision_timeout_s=float(cluster_raw.get("provision_timeout_s", 900)),
        ),
        models=models,
        retrieval=RetrievalConfig(
            candidate_chunks=int(retrieval_raw["candidate_chunks"]),
            result_chunks=int(retrieval_raw["result_chunks"]),
            chunk_characters=int(retrieval_raw["chunk_characters"]),
        ),
        sources=tuple(Source(**row) for row in raw["sources"]),
    )


def load_claim() -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / "claim.json").read_text(encoding="utf-8"))


def source_by_slug(config: AppConfig, slug: str) -> Source:
    for source in config.sources:
        if source.slug == slug:
            return source
    raise KeyError(slug)
