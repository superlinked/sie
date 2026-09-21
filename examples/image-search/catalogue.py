"""Shared between run.py and score.py: the catalogue, the queries, the ranking.

Standard library only, on purpose. run.py imports this and adds the SDK;
score.py imports this and adds nothing. Because both read the request shape from
here, the body score.py rebuilds is the body run.py sent, and score.py can
refuse a recording it cannot rebuild.

The images are sent as raw file bytes. The SDK's images.py keeps already encoded
JPEG bytes byte-identical on the wire, so the SHA-256 of the file in inputs/ is
the digest of what SIE actually received. calls.json records that digest in
place of the bytes rather than a second base64 copy of a file sitting next to
it.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
INPUTS = EVIDENCE / "inputs"

ENDPOINT = "https://api.superlinked.com"
MODEL = "google/siglip-so400m-patch14-384"
ENCODE_PATH = f"/v1/encode/{MODEL}"
DIMS = 1152

# The HuggingFace revision of the weights, as GET /v1/models reports it. This is
# the checkpoint the published figures came from, and both run.py and score.py
# refuse to proceed against a different one.
#
# Not to be confused with the deployment revision carried on every response in
# X-SIE-Model-Revision. That digest identifies the SIE deployment, and models
# served together share it, so it can never stand in for the weights.
MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"

# SigLIP is trained with one contrastive objective over both towers, so a query
# and an image go through the same encoder with no is_query asymmetry. The
# ranking below is a plain cosine, which is what that training makes meaningful.
METRIC = "cosine similarity between the L2-normalized text vector and each L2-normalized image vector"

# Images per encode call. Small enough that one failure costs little, large
# enough that the run is a handful of calls rather than one per photograph.
BATCH = 12

# The four ways every request is written, from the bare category up to all three
# attributes. Scoring compares the target's rank across them, so "the whole
# request wins" is a measurement rather than an assertion.
FORMS = ("category", "colour-category", "material-category", "full")

VOWELS = "aeiou"


def article(word: str) -> str:
    return "an" if word[:1].lower() in VOWELS else "a"


def phrase(form: str, colour: str, material: str, category: str) -> str:
    """The query text for one target under one form."""
    if form == "category":
        words = [category]
    elif form == "colour-category":
        words = [colour, category]
    elif form == "material-category":
        words = [material, category]
    elif form == "full":
        words = [colour, material, category]
    else:
        raise SystemExit(f"Unknown query form: {form!r}")
    return f"{article(words[0])} {' '.join(words)}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_catalogue() -> list[dict[str, Any]]:
    """Every photograph, with the digest of the file as published."""
    path = INPUTS / "catalogue.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    records = json.loads(path.read_text(encoding="utf-8"))["images"]
    seen: set[str] = set()
    triples: set[tuple[str, str, str]] = set()
    for record in records:
        if record["id"] in seen:
            raise SystemExit(f"catalogue.json lists {record['id']!r} twice")
        seen.add(record["id"])
        triple = (record["colour"], record["material"], record["category"])
        if triple in triples:
            raise SystemExit(
                f"{record['id']}: a second photograph is {triple[0]} {triple[1]} {triple[2]}. "
                "With two, 'the image matching all three' is not a single image and no "
                "figure over it means anything."
            )
        triples.add(triple)
        file_path = INPUTS / record["file"]
        if not file_path.exists():
            raise SystemExit(f"{file_path} is missing. Run: python3 fetch.py")
        data = file_path.read_bytes()
        digest = sha256_bytes(data)
        if digest != record["sha256"]:
            raise SystemExit(f"{record['file']}: file digest {digest} is not the recorded {record['sha256']}")
        if len(data) != record["bytes"]:
            raise SystemExit(f"{record['file']}: {len(data)} bytes on disk, catalogue.json says {record['bytes']}")
    return records


def load_queries() -> list[dict[str, Any]]:
    path = INPUTS / "queries.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    queries = json.loads(path.read_text(encoding="utf-8"))["queries"]
    seen: set[str] = set()
    for query in queries:
        if query["id"] in seen:
            raise SystemExit(f"queries.json lists {query['id']!r} twice")
        seen.add(query["id"])
    return queries


def query_items(queries: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One item per (target, form) pair, in a fixed order."""
    items = []
    for query in queries:
        for form in FORMS:
            items.append(
                {
                    "id": f"{query['id']}/{form}",
                    "text": phrase(form, query["colour"], query["material"], query["category"]),
                }
            )
    return items


def image_batches(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [records[index : index + BATCH] for index in range(0, len(records), BATCH)]


def image_body(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": [
            {
                "id": record["id"],
                "images": [
                    {
                        "file": f"inputs/{record['file']}",
                        "format": record["format"],
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                    }
                ],
            }
            for record in batch
        ],
        "params": {"output_types": ["dense"]},
    }


def query_body(queries: list[dict[str, Any]]) -> dict[str, Any]:
    return {"items": query_items(queries), "params": {"output_types": ["dense"]}}


def bodies(records: list[dict[str, Any]], queries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Every call this example makes, keyed by the slug calls.json records."""
    out = {
        f"images-{index:02d}": image_body(batch)
        for index, batch in enumerate(image_batches(records), start=1)
    }
    out["queries"] = query_body(queries)
    return out


def normalize(values: list[float]) -> list[float]:
    total = sum(value * value for value in values) ** 0.5
    if not total or not math.isfinite(total):
        raise SystemExit("A returned vector has zero or non-finite length")
    return [value / total for value in values]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise SystemExit(f"Cannot compare a {len(left)}-dimensional vector with a {len(right)}-dimensional one")
    return sum(a * b for a, b in zip(normalize(left), normalize(right), strict=True))


def rank(query_vector: list[float], image_vectors: dict[str, list[float]]) -> list[tuple[str, float]]:
    """Every image, best first. Ties break by id, so the order is total."""
    scored = [(name, cosine(query_vector, vector)) for name, vector in image_vectors.items()]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def overlap(record: dict[str, Any], query: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Which of colour, material and category this photograph shares with the request."""
    return (
        record["colour"] == query["colour"],
        record["material"] == query["material"],
        record["category"] == query["category"],
    )


def competitor_kind(record: dict[str, Any], query: dict[str, Any]) -> str:
    """What kind of competitor a photograph is for one request.

    `full` is the single photograph matching all three. The three two-of-three
    kinds are named by the attribute they get WRONG, because that is what the
    request has to overrule.
    """
    colour, material, category = overlap(record, query)
    matched = sum((colour, material, category))
    if matched == 3:
        return "full"
    if matched == 2:
        if not colour:
            return "wrong-colour"
        if not material:
            return "wrong-material"
        return "wrong-category"
    if matched == 1:
        return "one-attribute"
    return "unrelated"
