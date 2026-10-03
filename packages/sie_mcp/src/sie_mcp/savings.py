"""Token-savings metadata for ``docs_to_markdown``.

Two distinct figures travel in the metadata; do not conflate them:

- ``markdown_tokens_estimate`` — a LIVE, per-call estimate of *this* response's
  markdown, via a ~4 chars/token heuristic. Order-of-magnitude only.
- ``token_reduction`` — the committed measurement of markdown vs direct
  document ingestion, copied verbatim from the benchmark run. The measurement
  compares
  ``count_tokens`` on the document block (text plus one image per page) with the
  SIE markdown.
"""

import copy
import math
from typing import Any

_CHARS_PER_TOKEN = 4  # rough Claude/GPT-family heuristic

# Committed token-reduction measurement from run 20260610T144234Z.
# Every number below is copied verbatim from that committed run; re-run the
# benchmark to change them; never hand-edit or round performance claims.
# The dict is read-only; build_metadata deep-copies it so callers can't mutate it.
_OPUS = "claude-opus-4-8"
_SONNET = "claude-sonnet-4-6"
_TOKEN_REDUCTION: dict[str, Any] = {
    "source": "committed benchmark run 20260610T144234Z",
    # The response shape retains this provenance slot, but no public issue
    # tracks the measurement.
    "issue": None,
    "run": "20260610T144234Z",
    "method": (
        "count_tokens on the document block (text + one image per page, as "
        "Claude bills a document upload) vs the SIE markdown"
    ),
    # Token-weighted across the corpus. Opus is the conservative (lower-reduction)
    # profile; we report it as the headline so we never overstate.
    "blended_reduction_pct": {_OPUS: 82.9, _SONNET: 86.5},
    # Min/max reduction across every per-file-type row of BOTH profiles: the
    # corpus floor is Opus/HTML (63.3), the ceiling is Sonnet/PPTX (94.8).
    "per_file_type_reduction_pct": {"min": 63.3, "max": 94.8},
}

# Built from _TOKEN_REDUCTION so the prose can never drift from the data above.
_blended = _TOKEN_REDUCTION["blended_reduction_pct"]
_per_type = _TOKEN_REDUCTION["per_file_type_reduction_pct"]
_NOTE = (
    "markdown_tokens_estimate is a live ~4 chars/token estimate of this response, "
    "not a measured count. token_reduction is the committed measurement of "
    "markdown vs direct document ingestion: blended "
    f"{_blended[_OPUS]}% (Opus 4.8, the conservative profile) and "
    f"{_blended[_SONNET]}% (Sonnet 4.6); per-file-type reductions span "
    f"{_per_type['min']}%-{_per_type['max']}% across both profiles "
    f"(floor Opus/HTML, ceiling Sonnet/PPTX), measured in run {_TOKEN_REDUCTION['run']}."
)


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / _CHARS_PER_TOKEN) if text else 0


def _page_count(document: Any) -> int | None:
    if not isinstance(document, dict):
        return None
    pages = document.get("pages")
    if isinstance(pages, dict | list):
        return len(pages)
    return None


def build_metadata(
    *,
    markdown: str,
    document: Any = None,
    source_bytes: int,
    pages: int | None = None,
    engine: str | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "markdown_chars": len(markdown),
        "markdown_tokens_estimate": _estimate_tokens(markdown),
        "markdown_tokens_estimated": True,  # the token count above is a live estimate, not measured
        "source_bytes": source_bytes,
        "token_reduction": copy.deepcopy(_TOKEN_REDUCTION),
        "note": _NOTE,
    }
    if engine:
        meta["engine"] = engine
    # An explicit page count (VL-OCR render count) wins over Docling's document metadata.
    page_count = pages if pages is not None else _page_count(document)
    if page_count is not None:
        meta["source_pages"] = page_count
    return meta
