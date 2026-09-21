"""Pinned inputs and the two requests they build. Standard library only.

Both `run.py` (which calls SIE Cloud) and `score.py` (which never does) build
the request from here, so the request the scorer rebuilds is the request the
runner sends.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CANDIDATES_PATH = HERE / "data" / "candidates.json"
REVIEW_PATH = HERE / "data" / "review.json"
CALLS_PATH = HERE / "calls.json"

ENDPOINT = "https://api.superlinked.com"
KINDS = ("entities", "relations")
PAGE_ROLES = ("hero", "proof", "proof and playground", "not shown")
# The response fields one /v1/extract call returns for one item.
ITEM_KEYS = ("id", "entities", "relations", "classifications", "objects", "data")


class InputError(Exception):
    """A pinned input or recording this example refuses to use."""


def read_json(path: Path) -> Any:
    if not path.exists():
        raise InputError(f"Missing {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def compact_json(value: Any) -> bytes:
    """The encoding calls.json documents for response_sha256."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def extract_path(model: str) -> str:
    return f"/v1/extract/{urllib.parse.quote(model, safe='')}"


def load_candidates() -> dict[str, Any]:
    """Load data/candidates.json and confirm every paragraph is the pinned text."""
    doc = read_json(CANDIDATES_PATH)
    seen: set[str] = set()
    for candidate in doc["candidates"]:
        cid = candidate["id"]
        if cid in seen:
            raise InputError(f"Duplicate candidate {cid}")
        seen.add(cid)
        if sha256_bytes(candidate["text"].encode("utf-8")) != candidate["text_sha256"]:
            raise InputError(f"{cid}: text does not match its pinned SHA-256")
        if candidate["page_role"] not in PAGE_ROLES:
            raise InputError(f"{cid}: unknown page_role {candidate['page_role']!r}")
        if not candidate["entity_labels"] or not candidate["relation_labels"]:
            raise InputError(f"{cid}: entity and relation labels are both required")
    return doc


def entities_body(candidate: dict[str, Any]) -> dict[str, Any]:
    """Call one: the paragraph plus the entity types to look for."""
    return {
        "items": [{"text": candidate["text"]}],
        "params": {"labels": list(candidate["entity_labels"])},
    }


def relations_body(candidate: dict[str, Any], entities: list[dict[str, Any]]) -> dict[str, Any]:
    """Call two: the same paragraph, call one's entities, and the relation types."""
    return {
        "items": [{"text": candidate["text"], "metadata": {"entities": entities}}],
        "params": {"labels": list(candidate["relation_labels"])},
    }


def shown(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The candidates the task page renders: the hero, then the proof paragraphs.

    Proof candidates come back in the order they sit in candidates.json, which
    is not the order the page lays them out. Nothing counted here depends on it.
    """
    hero = [c for c in doc["candidates"] if c["page_role"] == "hero"]
    proof = [c for c in doc["candidates"] if c["page_role"].startswith("proof")]
    if len(hero) != 1:
        raise InputError(f"Expected exactly one hero candidate, found {len(hero)}")
    return hero + proof
