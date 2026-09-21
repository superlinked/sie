"""Pinned inputs, the requests they build, and the ranking. Standard library only.

Both `run.py` (which calls SIE Cloud) and `score.py` (which never does) build
the requests from here, so the request the scorer rebuilds is the request the
runner sends.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
CORPUS_PATH = EVIDENCE / "inputs" / "corpus.json"
QUERIES_PATH = EVIDENCE / "inputs" / "queries.json"
CALLS_PATH = EVIDENCE / "calls.json"
MANIFEST_PATH = EVIDENCE / "manifest.json"

ENDPOINT = "https://api.superlinked.com"
MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
ENCODE_PATH = f"/v1/encode/{MODEL}"
# Passages per encode call. The corpus is sent in batches; the queries go in one.
CORPUS_BATCH = 24
# Documents use the model's document template. Queries set is_query, which is
# what makes this an asymmetric retrieval model rather than plain similarity.
CORPUS_PARAMS: dict[str, Any] = {"output_types": ["dense"]}
QUERY_PARAMS: dict[str, Any] = {"output_types": ["dense"], "options": {"is_query": True}}


class InputError(Exception):
    """A pinned input or recording this example refuses to use."""


def read_json(path: Path) -> Any:
    if not path.exists():
        if EVIDENCE not in path.parents and path != EVIDENCE:
            raise InputError(f"Missing {path}")
        raise InputError(f"Missing {path.relative_to(HERE)}. Run `python3 fetch.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compact_json(value: Any) -> bytes:
    """The encoding the manifest documents for response_sha256."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def allowed_revisions(manifest: dict[str, Any]) -> set[str]:
    """The served model revisions the manifest claims this run used.

    The runner writes a single string when one revision served the whole run
    and a list when more than one did, so both are accepted here.
    """
    value = manifest.get("model_revision")
    if isinstance(value, str):
        revisions = [value]
    elif isinstance(value, list):
        revisions = value
    else:
        # A dict would iterate as its keys and an int would raise TypeError, so
        # neither is coerced into an answer here.
        raise InputError(f"manifest model_revision is {type(value).__name__}, expected a string or a list of strings")
    if not revisions or not all(isinstance(item, str) and item for item in revisions):
        raise InputError("manifest does not name a served model revision")
    return set(revisions)


def check_revision(slug: str, entry: dict[str, Any], allowed: set[str]) -> str:
    """A recorded call has to say which served revision answered it."""
    revision = entry.get("model_revision")
    if not isinstance(revision, str) or not revision:
        raise InputError(f"{slug}: no served model revision recorded")
    if revision not in allowed:
        raise InputError(f"{slug}: served revision {revision} is not one the manifest names")
    return revision


def check_call_set(observed: set, expected: set, manifest: dict[str, Any], label: str) -> None:
    """Exactly the expected calls, no more and no fewer.

    A count alone would survive a swap or a duplicate, so the identities are
    compared and the manifest's own total is checked against them.
    """
    missing = sorted(expected - observed, key=str)
    extra = sorted(observed - expected, key=str)
    if missing:
        raise InputError(f"no recorded {label} for {missing[0]}")
    if extra:
        raise InputError(f"calls.json holds {len(extra)} {label} nothing pins, starting with {extra[0]}")
    recorded_total = manifest.get("calls_recorded")
    if recorded_total != len(observed):
        raise InputError(f"manifest says {recorded_total} calls were recorded, calls.json holds {len(observed)}")


def load_corpus() -> list[dict[str, Any]]:
    """The pinned passages, each checked against its own digest."""
    passages = read_json(CORPUS_PATH)["passages"]
    seen: set[str] = set()
    for passage in passages:
        if passage["id"] in seen:
            raise InputError(f"Duplicate passage {passage['id']}")
        seen.add(passage["id"])
        if sha256_bytes(passage["text"].encode("utf-8")) != passage["sha256"]:
            raise InputError(f"{passage['id']}: text does not match its pinned SHA-256")
    return passages


def load_queries(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The pinned questions, each naming passages that answer it.

    The answer ids were resolved when the corpus was built, from a verbatim
    phrase quoted in `answers`, before anything was encoded.
    """
    known = {passage["id"] for passage in passages}
    queries = read_json(QUERIES_PATH)["queries"]
    seen: set[str] = set()
    for query in queries:
        if query["id"] in seen:
            raise InputError(f"Duplicate query {query['id']}")
        seen.add(query["id"])
        if not query["answer_ids"]:
            raise InputError(f"{query['id']}: no answer passage named")
        unknown = [answer for answer in query["answer_ids"] if answer not in known]
        if unknown:
            raise InputError(f"{query['id']}: answer {unknown[0]} is not in the corpus")
    return queries


def corpus_bodies(passages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """One request body per corpus batch, named the way the recording names it."""
    bodies = []
    for index in range(0, len(passages), CORPUS_BATCH):
        batch = passages[index : index + CORPUS_BATCH]
        name = f"corpus-{index // CORPUS_BATCH:03d}"
        bodies.append((name, {"items": [{"id": p["id"], "text": p["text"]} for p in batch], "params": CORPUS_PARAMS}))
    return bodies


def query_body(queries: list[dict[str, Any]]) -> dict[str, Any]:
    """All the questions in one call, encoded as queries."""
    return {"items": [{"id": q["id"], "text": q["query"]} for q in queries], "params": QUERY_PARAMS}


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(x * y for x, y in zip(left, right))
    return dot / (math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(y * y for y in right)))


def rank_answers(
    queries: list[dict[str, Any]],
    passages: list[dict[str, Any]],
    query_vectors: dict[str, list[float]],
    passage_vectors: dict[str, list[float]],
) -> list[dict[str, Any]]:
    """For each question, where its answer lands among all the passages.

    Every passage is scored; nothing is filtered first. A question with more
    than one answer passage takes the best rank of them.
    """
    results = []
    for query in queries:
        vector = query_vectors[query["id"]]
        scored = sorted(
            ({"id": p["id"], "score": cosine(vector, passage_vectors[p["id"]])} for p in passages),
            key=lambda row: -row["score"],
        )
        position = {row["id"]: index + 1 for index, row in enumerate(scored)}
        rank = min(position[answer] for answer in query["answer_ids"])
        results.append(
            {
                "id": query["id"],
                "query": query["query"],
                "rank": rank,
                "top": scored[:3],
                "answer_ids": query["answer_ids"],
            }
        )
    return results
