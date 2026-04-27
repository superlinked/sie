from __future__ import annotations

from dataclasses import dataclass

from taxonomy_classification.chroma_index import ChromaMatch
from taxonomy_classification.taxonomy import CategoryPath


@dataclass(frozen=True)
class LatencySummary:
    p50: float
    p95: float
    request_count: int
    batch_size: int | None = None
    items_per_request: int | None = None

    def as_dict(self) -> dict[str, float | int]:
        summary: dict[str, float | int] = {
            "p50": self.p50,
            "p95": self.p95,
            "request_count": self.request_count,
        }

        if self.batch_size is not None:
            summary["batch_size"] = self.batch_size

        if self.items_per_request is not None:
            summary["items_per_request"] = self.items_per_request

        return summary


@dataclass(frozen=True)
class ClassifiedDescriptions:
    rankings: list[list[tuple[str, float]]]
    latencies_ms: list[float]


@dataclass(frozen=True)
class EncodedItems:
    embeddings: list[list[float]]
    latencies_ms: list[float]


@dataclass(frozen=True)
class RetrievedMatches:
    matches: list[list[ChromaMatch]]
    latencies_ms: list[float]


@dataclass(frozen=True)
class PredictedCategories:
    categories: list[CategoryPath]
    latencies_ms: list[float]


@dataclass(frozen=True)
class RerankedDescriptions:
    rankings: list[list[tuple[CategoryPath, float]]]
    retrieval_latencies_ms: list[float]
    reranking_latencies_ms: list[float]
