"""Shared between run.py and score.py: request bodies in, ranking out.

Standard library only, on purpose. run.py imports this and adds the SDK;
score.py imports this and adds nothing. Because both read the request shape
from here, the body score.py rebuilds is the body run.py sent, and score.py can
refuse a recording it cannot rebuild.

The images are sent as raw file bytes. The SDK's images.py keeps already
encoded PNG/JPEG bytes byte-identical on the wire, so the SHA-256 of the file
in inputs/ is the digest of what SIE actually received. calls.json records that
digest in place of the bytes rather than a second base64 copy of a file already
sitting next to it.
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
# served together share it: arctic-embed-l-v2.0, siglip2-base, bge-m3 and this
# model all return the same value, so it can never stand in for the weights.
MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"

# The published ranking, in order. Pinned here in the committed source so
# score.py compares what it derives from the recording against something the
# recording cannot move. Editing evidence/ alone will not satisfy this.
EXPECTED_ORDER = (
    "red-leather-handbag",
    "black-handbag",
    "red-shoes",
    "green-backpack",
    "black-camera",
    "blue-running-sneaker",
)

# SigLIP is trained with one contrastive objective over both towers, so a query
# and an image go through the same encoder with no is_query asymmetry. The
# ranking below is a plain cosine, which is what that training makes meaningful.
METRIC = "cosine similarity between the L2-normalized text vector and each L2-normalized image vector"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def load_images() -> list[dict[str, Any]]:
    """The six photographs, with the digest of each file as committed."""
    records = json.loads((INPUTS / "images.json").read_text(encoding="utf-8"))["images"]
    seen: set[str] = set()
    for record in records:
        if record["id"] in seen:
            raise SystemExit(f"images.json lists {record['id']!r} twice")
        seen.add(record["id"])
        path = INPUTS / record["file"]
        if not path.exists():
            raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
        data = path.read_bytes()
        digest = sha256_bytes(data)
        if digest != record["sha256"]:
            raise SystemExit(f"{record['file']}: file digest {digest} is not the recorded {record['sha256']}")
        if len(data) != record["bytes"]:
            raise SystemExit(f"{record['file']}: {len(data)} bytes on disk, images.json says {record['bytes']}")
    return records


def load_query() -> dict[str, Any]:
    return json.loads((INPUTS / "query.json").read_text(encoding="utf-8"))


def image_body(records: list[dict[str, Any]]) -> dict[str, Any]:
    """One encode call carrying all six photographs."""
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
            for record in records
        ],
        "params": {"output_types": ["dense"]},
    }


def query_body(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": [{"id": query["id"], "text": query["text"]}],
        "params": {"output_types": ["dense"]},
    }


def bodies(records: list[dict[str, Any]], query: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {"images": image_body(records), "query": query_body(query)}


def normalize(values: list[float]) -> list[float]:
    total = sum(value * value for value in values) ** 0.5
    if not total or not math.isfinite(total):
        raise SystemExit("A returned vector has zero or non-finite length")
    return [value / total for value in values]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise SystemExit(f"Cannot compare a {len(left)}-dimensional vector with a {len(right)}-dimensional one")
    return sum(a * b for a, b in zip(normalize(left), normalize(right)))


def rank(query_vector: list[float], image_vectors: dict[str, list[float]]) -> list[tuple[str, float]]:
    scored = [(name, cosine(query_vector, vector)) for name, vector in image_vectors.items()]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored
