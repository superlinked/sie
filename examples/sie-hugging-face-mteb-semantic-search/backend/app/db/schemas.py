from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DownloadRequest(BaseModel):
    storage_id: str = Field(..., description="Logical storage id, e.g. test01")
    limit: int = Field(
        30,
        ge=1,
        le=200,
        description="How many MTEB-benchmarked models to import (max 200)",
    )


class ModelOut(BaseModel):
    # Core ids / times
    hf_id: str
    author: Optional[str] = None
    sha: Optional[str] = None
    created_at_hf: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    created_at: datetime

    # Flags
    private: Optional[bool] = None
    disabled: Optional[bool] = None

    # Metrics
    downloads: Optional[int] = None
    downloads_all_time: Optional[int] = None
    downloads_30d: Optional[int] = None
    likes: Optional[int] = None
    trending_score: Optional[float] = None

    # Taxonomy / tags
    tags: Optional[list[str]] = None
    pipeline_tag: Optional[str] = None
    library_name: Optional[str] = None
    mask_token: Optional[str] = None

    # Structured metadata kept in DB (small).
    config: Optional[dict[str, Any]] = None
    card_data: Optional[dict[str, Any]] = None

    # MTEB benchmark results (compact: [{task_name, main_score}, ...]).
    mteb_scores: Optional[list[dict[str, Any]]] = None

    # Descriptions
    short_description: Optional[str] = None
    long_description: Optional[str] = None

    class Config:
        from_attributes = True


class ModelSummary(BaseModel):
    """Lightweight projection for list / search results."""
    hf_id: str
    author: Optional[str] = None
    created_at: datetime
    downloads_30d: Optional[int] = None
    likes: Optional[int] = None
    pipeline_tag: Optional[str] = None
    short_description: Optional[str] = None

    class Config:
        from_attributes = True


class DownloadResponse(BaseModel):
    storage_id: str
    stored_count: int
    models: list[ModelOut]


class EnrichRequest(BaseModel):
    storage_id: str = Field(..., description="Logical storage id to enrich")


class EnrichResponse(BaseModel):
    storage_id: str
    total_models: int
    enriched_count: int
    skipped_count: int


class SemanticSearchRequest(BaseModel):
    storage_id: str = Field(..., description="Logical storage id, e.g. test01")
    query: str = Field(..., min_length=1, description="Natural-language search query")
    n_results: int = Field(
        20, ge=1, le=100, description="Number of results to return"
    )


class SemanticSearchResult(BaseModel):
    hf_id: str
    distance: float
    short_description: Optional[str] = None


class SemanticSearchResponse(BaseModel):
    storage_id: str
    query: str
    results: list[SemanticSearchResult]


class RerankSearchResult(BaseModel):
    hf_id: str
    rerank_distance: Optional[float] = None
    short_distance: float
    short_description: Optional[str] = None


class RerankSearchResponse(BaseModel):
    storage_id: str
    query: str
    results: list[RerankSearchResult]
