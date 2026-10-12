"""The cost-routing study this example reproduces, its pinned evidence and its selected router.

Shared by fetch.py, score.py and run.py.

The router is `knn20` from the study's code/analyze.py (`router_scores`):
normalise each BAAI/bge-m3 dense vector, take the 20 most similar calibration
requests by cosine, weight each by max(cos, 0) and average their labels,
y = 1 - max(0, q_strong - q_cheap). A request goes to the cheap model when that
score is at least the pair's threshold, which the study chose on calibration.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"

DATASET = "superlinked/sie-task-evidence"
REVISION = "95d34b775b61bf1d584cada099bfed053710fc2b"
FOLDER = "evaluations/2026-10-11/request-routing-cost-results-v1"
MANIFEST = "MANIFEST.json"
# SHA-256 of MANIFEST.json at REVISION. The manifest lists the SHA-256 of every other file.
MANIFEST_SHA256 = "82f870f6a64916aa4e217eea03df8f17d3355f06f2c6e2718064b795c40de713"
FILES = (
    "results.json",
    "items-openai.jsonl",
    "items-anthropic.jsonl",
    "inputs/items.jsonl",
    "records/routers/bge.jsonl.gz",
)

EMBEDDER = "BAAI/bge-m3"
ROUTER = "knn20"
K = 20
SIE_USD_PER_CREDIT = 0.00001  # code/arms.py: 1 SIE credit = USD 0.00001


@dataclass(frozen=True)
class Pair:
    cheap: str
    strong: str
    cheap_name: str
    strong_name: str


# Model ids as code/arms.py sent them.
PAIRS = {
    "openai": Pair("gpt-6-luna", "gpt-6.1-sol", "GPT-6 Luna", "GPT-6.1 Sol"),
    "anthropic": Pair("claude-haiku-5-5", "claude-sonnet-5-5", "Claude Haiku 5.5", "Claude Sonnet 5.5"),
}


class EvidenceError(ValueError):
    """The evidence is missing, altered or not from the pinned revision."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_entries(body: bytes) -> dict[str, tuple[str, int]]:
    """Check MANIFEST.json against its pinned digest; return (sha256, bytes) for each file this example uses."""
    if sha256(body) != MANIFEST_SHA256:
        raise EvidenceError(f"{MANIFEST} does not match the pinned SHA-256 {MANIFEST_SHA256}")
    manifest = json.loads(body)
    listed = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(listed, dict):
        raise EvidenceError(f"{MANIFEST} has no files table")
    entries: dict[str, tuple[str, int]] = {}
    for name in FILES:
        entry = listed.get(name)
        if not isinstance(entry, dict):
            raise EvidenceError(f"{name}: not listed in {MANIFEST}")
        digest, size = entry.get("sha256"), entry.get("bytes")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise EvidenceError(f"{name}: {MANIFEST} lists an invalid SHA-256")
        if type(size) is not int or size < 0:
            raise EvidenceError(f"{name}: {MANIFEST} lists an invalid size")
        entries[name] = (digest, size)
    return entries


def check_file(name: str, data: bytes, entry: tuple[str, int]) -> None:
    digest, size = entry
    if len(data) != size:
        raise EvidenceError(f"{name}: {len(data)} bytes, {MANIFEST} lists {size}")
    if sha256(data) != digest:
        raise EvidenceError(f"{name}: SHA-256 differs from {MANIFEST}")


def verify_evidence(directory: Path) -> None:
    """Raise EvidenceError unless the directory holds the pinned manifest and every file matches it."""
    manifest = directory / MANIFEST
    if not manifest.is_file():
        raise EvidenceError(f"{directory} has no {MANIFEST}; run fetch.py first")
    for name, entry in manifest_entries(manifest.read_bytes()).items():
        path = directory / name
        if not path.is_file():
            raise EvidenceError(f"{name}: missing from {directory}; run fetch.py first")
        check_file(name, path.read_bytes(), entry)


def labels(q_cheap: np.ndarray, q_strong: np.ndarray) -> np.ndarray:
    """Router target: 1 when the cheap answer is as good as the strong one, lower by the quality it loses."""
    return 1.0 - np.maximum(0.0, q_strong - q_cheap)


def normalise(vectors: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        raise ValueError("vector has a non-finite value")
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("vector has zero length")
    return v / norms


def knn_score(similarities: np.ndarray, bank_labels: np.ndarray, k: int = K) -> float:
    """The kNN score of code/analyze.py for one request, from its cosine similarity to each bank request."""
    top = np.argsort(-similarities)[:k]
    weights = np.maximum(similarities[top], 0.0)
    if weights.sum() > 0:
        return float(np.sum(weights * bank_labels[top]) / np.sum(weights))
    return float(np.mean(bank_labels[top]))


def routes_cheap(score: float, tau: float | None) -> bool:
    """Cheap model when score >= tau. The study records a threshold of null as never routing cheap."""
    return tau is not None and score >= tau


@dataclass(frozen=True)
class Outcomes:
    """One pair's recorded per-request quality and list cost, aligned with Study.ids."""

    q_cheap: np.ndarray
    q_strong: np.ndarray
    usd_cheap: np.ndarray
    usd_strong: np.ndarray
    usd_router: np.ndarray
    score: np.ndarray
    route: list[str]


@dataclass(frozen=True)
class Study:
    ids: list[str]
    prompts: list[str]
    strata: np.ndarray
    calibration: np.ndarray
    raw_vectors: np.ndarray
    vectors: np.ndarray
    credits: np.ndarray
    pairs: dict[str, Outcomes]
    results: dict[str, Any]

    def index(self, item_id: str) -> int:
        try:
            return self.ids.index(item_id)
        except ValueError:
            raise KeyError(f"{item_id} is not a study request") from None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_study(directory: Path = EVIDENCE) -> Study:
    """Verify the evidence, then load it in the item order of inputs/items.jsonl, as analyze.py did."""
    verify_evidence(directory)
    items = read_jsonl(directory / "inputs/items.jsonl")
    ids = [item["id"] for item in items]
    if len(set(ids)) != len(ids):
        raise EvidenceError("inputs/items.jsonl repeats a request id")
    # analyze.py keeps the last bge row per id. The file also holds the pilot items the amendment swapped out.
    bge = {row["id"]: row for row in read_jsonl(directory / "records/routers/bge.jsonl.gz")}
    if any("vector" not in bge.get(i, {}) for i in ids):
        raise EvidenceError("records/routers/bge.jsonl.gz lacks a vector for a study request")
    raw = np.array([bge[i]["vector"] for i in ids], dtype=np.float64)
    credits = np.array([(bge[i].get("usage") or {}).get("credits_charged", 0) for i in ids], dtype=np.float64)
    pairs = {}
    for pair in PAIRS:
        rows = read_jsonl(directory / f"items-{pair}.jsonl")
        if [(r["id"], r["split"], r["stratum"]) for r in rows] != [(i["id"], i["split"], i["stratum"]) for i in items]:
            raise EvidenceError(f"items-{pair}.jsonl does not follow inputs/items.jsonl")
        pairs[pair] = Outcomes(
            q_cheap=np.array([r["q_cheap"] for r in rows], dtype=np.float64),
            q_strong=np.array([r["q_strong"] for r in rows], dtype=np.float64),
            usd_cheap=np.array([r["usd_cheap"] for r in rows], dtype=np.float64),
            usd_strong=np.array([r["usd_strong"] for r in rows], dtype=np.float64),
            usd_router=np.array([r["usd_router"] for r in rows], dtype=np.float64),
            score=np.array([r["score"] for r in rows], dtype=np.float64),
            route=[r["route"] for r in rows],
        )
    return Study(
        ids=ids,
        prompts=[item["prompt"] for item in items],
        strata=np.array([item["stratum"] for item in items]),
        calibration=np.array([item["split"] == "calibration" for item in items]),
        raw_vectors=raw,
        vectors=normalise(raw),
        credits=credits,
        pairs=pairs,
        results=json.loads((directory / "results.json").read_text(encoding="utf-8")),
    )


@dataclass(frozen=True)
class Bank:
    """What routing a new request needs: calibration vectors, their labels for one pair, and its threshold."""

    pair: str
    ids: list[str]
    vectors: np.ndarray
    labels: np.ndarray
    tau: float | None

    def score(self, vector: np.ndarray, exclude: str | None = None) -> float:
        """knn20 score of one dense vector. `exclude` drops a bank request, as the study did for its own."""
        similarities = self.vectors @ normalise(vector)
        keep = np.array([i != exclude for i in self.ids])
        return knn_score(similarities[keep], self.labels[keep])

    def route(self, vector: np.ndarray, exclude: str | None = None) -> dict[str, Any]:
        score = self.score(vector, exclude)
        cheap = routes_cheap(score, self.tau)
        models = PAIRS[self.pair]
        return {
            "pair": self.pair,
            "score": score,
            "tau": self.tau,
            "route": "cheap" if cheap else "strong",
            "model": models.cheap if cheap else models.strong,
        }


def bank(study: Study, pair: str) -> Bank:
    outcomes = study.pairs[pair]
    cal = study.calibration
    return Bank(
        pair=pair,
        ids=[i for i, is_cal in zip(study.ids, cal, strict=True) if is_cal],
        vectors=study.vectors[cal],
        labels=labels(outcomes.q_cheap, outcomes.q_strong)[cal],
        tau=study.results["calibration"][pair][ROUTER]["tau"],
    )
