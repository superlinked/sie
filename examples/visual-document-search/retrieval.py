"""Pinned inputs, the requests they build, and both rankings. Standard library only.

run.py imports this and adds the SDK. score.py imports this and adds nothing.
Because both read the request shape and the scoring from here, the ranking the
scorer reproduces is the ranking the runner produced, and the scorer refuses a
recording it cannot rebuild.

Two rankings run over the same candidate set, which is every page of one source
document:

  text   BM25 over the ViDoRe `markdown` field, computed here, no model
  visual ColPali late interaction over the rendered page image

Late interaction is the reason the visual side wins. ColPali returns one vector
per image patch rather than one vector per page, so a figure occupying a tenth
of a page still has its own vectors to match against. MaxSim takes, for each
query vector, its strongest match anywhere on the page, and sums those.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"
INPUTS = EVIDENCE / "inputs"

# A self-hosted SIE server. Cloud does not serve a ColPali-family model, so this
# example points at one you start yourself; see the README for the docker run.
# SIE_BASE_URL overrides it.
ENDPOINT = "http://localhost:8080"
MODEL = "vidore/colpali-v1.3-hf"
ENCODE_PATH = f"/v1/encode/{MODEL}"
TOKEN_DIM = 128

# The HuggingFace revision of the weights, as GET /v1/models reports it. This is
# the checkpoint the published ranks came from, and both run.py and score.py
# refuse to proceed against a different one. SIE pins this revision for the
# model, so a server of the same version serves the same weights.
MODEL_REVISION = "133a9eb02947310513f52f8f0d39d622e0eab8dc"

# BM25 over the page's markdown, the text baseline the comparison is against.
K1 = 1.2
B = 0.75
TERM = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def encode_multivector(rows: list[list[float]]) -> str:
    """A [tokens, 128] float16 multivector as one base64 string.

    The server returns float16 when the request asks for it, so this carries the
    values it sent rather than a rounded copy of something wider.
    """
    flat = [value for row in rows for value in row]
    return base64.b64encode(struct.pack(f"<{len(flat)}e", *flat)).decode("ascii")


def decode_multivector(blob: str, dim: int = TOKEN_DIM) -> list[list[float]]:
    raw = base64.b64decode(blob)
    if len(raw) % (dim * 2):
        raise SystemExit(f"A multivector of {len(raw)} bytes is not a whole number of {dim}-wide float16 tokens")
    count = len(raw) // 2
    flat = struct.unpack(f"<{count}e", raw)
    return [list(flat[index : index + dim]) for index in range(0, count, dim)]


def load_comparisons() -> list[dict[str, Any]]:
    """The four query-and-document pairs the task page publishes."""
    records = json.loads((INPUTS / "comparisons.json").read_text(encoding="utf-8"))["comparisons"]
    seen: set[str] = set()
    for record in records:
        if record["id"] in seen:
            raise SystemExit(f"comparisons.json lists {record['id']!r} twice")
        seen.add(record["id"])
    return records


def load_pages() -> dict[str, list[dict[str, Any]]]:
    """Every candidate page, grouped by document, in page order."""
    records = json.loads((INPUTS / "pages.json").read_text(encoding="utf-8"))["pages"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for record in records:
        key = f"{record['dataset']}/{record['corpus_id']}"
        if key in seen:
            raise SystemExit(f"pages.json lists {key} twice")
        seen.add(key)
        grouped.setdefault(record["doc_id"], []).append(record)
    for pages in grouped.values():
        pages.sort(key=lambda page: page["page_number_in_doc"])
    return grouped


def check_page_bytes(page: dict[str, Any]) -> bytes:
    path = INPUTS / page["file"]
    if not path.exists():
        raise SystemExit(f"{path} is missing. Run: python3 fetch.py")
    data = path.read_bytes()
    digest = sha256_bytes(data)
    if digest != page["sha256"]:
        raise SystemExit(f"{page['file']}: file digest {digest} is not the recorded {page['sha256']}")
    if len(data) != page["bytes"]:
        raise SystemExit(f"{page['file']}: {len(data)} bytes on disk, pages.json says {page['bytes']}")
    return data


def terms(text: str) -> list[str]:
    return [term for term in TERM.findall((text or "").lower()) if term not in STOPWORDS]


def bm25_rank(query: str, pages: list[dict[str, Any]]) -> list[tuple[int, float]]:
    """Rank one document's pages by BM25 over their markdown. No model involved."""
    documents = {page["corpus_id"]: terms(page["markdown"]) for page in pages}
    count = len(documents)
    if not count:
        raise SystemExit("Cannot rank an empty candidate set")
    average_length = sum(len(tokens) for tokens in documents.values()) / count
    frequency: Counter[str] = Counter()
    for tokens in documents.values():
        frequency.update(set(tokens))
    query_terms = terms(query)
    scored = []
    for corpus_id, tokens in documents.items():
        counts = Counter(tokens)
        length = len(tokens)
        total = 0.0
        for term in query_terms:
            seen_in = frequency.get(term, 0)
            if not seen_in:
                continue
            idf = math.log(1 + (count - seen_in + 0.5) / (seen_in + 0.5))
            appearances = counts[term]
            total += idf * (appearances * (K1 + 1)) / (appearances + K1 * (1 - B + B * length / average_length))
        scored.append((corpus_id, total))
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def maxsim(query: list[list[float]], page: list[list[float]]) -> float:
    """Late interaction: each query vector's strongest page vector, summed.

    Both sides arrive L2-normalized from the ColPali adapter, so a dot product
    is already a cosine and no renormalization happens here.
    """
    if not query or not page:
        raise SystemExit("MaxSim needs a non-empty query and page multivector")
    width = len(query[0])
    total = 0.0
    for vector in query:
        if len(vector) != width:
            raise SystemExit("Query vectors disagree on width")
        best = None
        for candidate in page:
            if len(candidate) != width:
                raise SystemExit(f"A page vector is {len(candidate)} wide and the query is {width}")
            value = 0.0
            for left, right in zip(vector, candidate):
                value += left * right
            if best is None or value > best:
                best = value
        total += best
    if not math.isfinite(total):
        raise SystemExit("MaxSim produced a non-finite score")
    return total


def visual_rank(query: list[list[float]], pages: dict[int, list[list[float]]]) -> list[tuple[int, float]]:
    scored = [(corpus_id, maxsim(query, page)) for corpus_id, page in pages.items()]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def page_body(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """One encode call carrying a batch of page images.

    The images travel as raw file bytes. SIE's SDK keeps already encoded PNG
    unchanged on the wire, so the digest here is the digest of what SIE read.
    """
    return {
        "items": [
            {
                "id": str(page["corpus_id"]),
                "images": [
                    {
                        "file": page["file"],
                        "format": page["format"],
                        "bytes": page["bytes"],
                        "sha256": page["sha256"],
                    }
                ],
            }
            for page in pages
        ],
        "params": {"output_types": ["multivector"], "output_dtype": "float16"},
    }


def query_body(comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": [{"id": comparison["id"], "text": comparison["query"]}],
        "params": {"output_types": ["multivector"], "output_dtype": "float16"},
    }
