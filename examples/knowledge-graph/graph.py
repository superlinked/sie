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
EVIDENCE = HERE / "evidence"
CANDIDATES_PATH = EVIDENCE / "inputs" / "candidates.json"
REVIEW_PATH = EVIDENCE / "inputs" / "review.json"
CALLS_PATH = EVIDENCE / "calls.json"
MANIFEST_PATH = EVIDENCE / "manifest.json"

ENDPOINT = "https://api.superlinked.com"
KINDS = ("entities", "relations")
PAGE_ROLES = ("hero", "proof", "proof and playground", "not shown")
# The response fields one /v1/extract call returns for one item.
ITEM_KEYS = ("id", "entities", "relations", "classifications", "objects", "data")


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
    """The encoding calls.json documents for response_sha256."""
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
