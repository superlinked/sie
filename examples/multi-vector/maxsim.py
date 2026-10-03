"""Pinned inputs, the requests they build, and MaxSim. Standard library only.

Both `run.py` (which calls SIE Cloud) and `score.py` (which never does) build
the requests from here, so the request the scorer rebuilds is the request the
runner sends.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
CASES_PATH = EVIDENCE / "inputs" / "cases.json"
CALLS_PATH = EVIDENCE / "calls.json"
MANIFEST_PATH = EVIDENCE / "manifest.json"

ENDPOINT = "https://api.superlinked.com"
MODEL = "lightonai/GTE-ModernColBERT-v1"
ENCODE_PATH = f"/v1/encode/{urllib.parse.quote(MODEL, safe='')}"
KINDS = ("query", "passages")
# The server prepends [Q] to a query and [D] to a document, and drops
# bare-punctuation tokens from documents only. is_query is what picks which.
QUERY_PARAMS: dict[str, Any] = {"output_types": ["multivector"], "options": {"is_query": True}}
PASSAGE_PARAMS: dict[str, Any] = {"output_types": ["multivector"], "options": {"is_query": False}}


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


def load_cases() -> dict[str, Any]:
    """The pinned searches: a question and the four passages it chooses between."""
    doc = read_json(CASES_PATH)
    if doc["model"] != MODEL:
        raise InputError(f"cases.json pins {doc['model']!r}, this example scores {MODEL!r}")
    seen: set[str] = set()
    for case in doc["cases"]:
        slug = case["slug"]
        if slug in seen:
            raise InputError(f"Duplicate case {slug}")
        seen.add(slug)
        ids = [passage["id"] for passage in case["passages"]]
        if len(set(ids)) != len(ids):
            raise InputError(f"{slug}: two passages share an id")
        if case["answer_id"] not in ids:
            raise InputError(f"{slug}: answer {case['answer_id']} is not one of its passages")
    return doc


def query_body(case: dict[str, Any]) -> dict[str, Any]:
    """Call one: the question alone, encoded as a query."""
    return {"items": [{"id": f"{case['slug']}-query", "text": case["query"]}], "params": QUERY_PARAMS}


def passages_body(case: dict[str, Any]) -> dict[str, Any]:
    """Call two: all four passages in one call, encoded as documents."""
    return {
        "items": [{"id": passage["id"], "text": passage["text"]} for passage in case["passages"]],
        "params": PASSAGE_PARAMS,
    }


def maxsim(query_tokens: list[list[float]], passage_tokens: list[list[float]]) -> float:
    """Late interaction: every query token takes its best match, and they sum.

    The server returns L2-normalised vectors, so the dot product is the cosine
    and nothing is normalised here. The sum is not divided by the token count,
    so a longer question scores higher across the board; only the comparison
    between passages of one search means anything.
    """
    return sum(
        max(sum(a * b for a, b in zip(query_token, passage_token)) for passage_token in passage_tokens)
        for query_token in query_tokens
    )


def rank_passages(
    case: dict[str, Any], query: list[list[float]], passages: dict[str, list[list[float]]]
) -> dict[str, Any]:
    """Score the four passages of one search and say where the answer landed."""
    totals = {passage["id"]: maxsim(query, passages[passage["id"]]) for passage in case["passages"]}
    ordered = sorted(totals.items(), key=lambda item: -item[1])
    rank = next(index + 1 for index, (pid, _) in enumerate(ordered) if pid == case["answer_id"])
    closest_other = next((pid, total) for pid, total in ordered if pid != case["answer_id"])
    return {
        "slug": case["slug"],
        "query": case["query"],
        "rank": rank,
        "answer_id": case["answer_id"],
        "answer_maxsim": totals[case["answer_id"]],
        "closest_other_id": closest_other[0],
        "closest_other_maxsim": closest_other[1],
    }
