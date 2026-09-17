"""Environment-driven configuration for the SIE MCP edge service.

The edge holds one server-side cluster credential (`SIE_API_KEY`) and validates
per-user connector secrets at its own boundary. Issued per-user keys and metering
can integrate at that boundary later.
"""

import os
from dataclasses import dataclass

_DEFAULT_BASE_URL = "http://localhost:8080"
_DEFAULT_MODEL = "docling"
# answer_questions model ids: dense encoder and cross-encoder reranker.
# The grounded-answer generator reuses _DEFAULT_GENERATE_MODEL (defined below).
_DEFAULT_ENCODE_MODEL = "BAAI/bge-m3"
_DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# Passages fed to the generator, candidates kept before reranking, the character
# window / overlap the documents are chunked into (~4 chars/token), and the
# answer-length ceiling.
_DEFAULT_QA_TOP_K = 5
_DEFAULT_QA_RERANK_CANDIDATES = 20
_DEFAULT_QA_CHUNK_CHARS = 1200
_DEFAULT_QA_CHUNK_OVERLAP_CHARS = 200
_DEFAULT_QA_MAX_TOKENS = 512
# Input bounds for answer_questions. The edge holds every document's text in
# memory and sends all chunks in a single encode request, so unbounded inputs are
# a memory-exhaustion / model-fan-out vector (the same concern the document byte
# cap guards). Cap the total document size, the question count, and the number of
# chunks per call. Override via SIE_MCP_QA_MAX_*.
_DEFAULT_QA_MAX_DOCUMENT_CHARS = 2_000_000
_DEFAULT_QA_MAX_QUESTIONS = 50
_DEFAULT_QA_MAX_CHUNKS = 2000
_DEFAULT_CAPTION_MODEL = "microsoft/Florence-2-base-ft"
# CLIP is the zero-shot default: it is contrastively trained for cosine ranking,
# so client-side cosine/argmax separates labels cleanly. SigLIP's sigmoid-trained
# embeddings rank poorly under naive cosine — override
# via SIE_MCP_EMBED_MODEL if you supply your own scoring.
_DEFAULT_EMBED_MODEL = "openai/clip-vit-base-patch32"
_DEFAULT_IMAGE_TOP_K = 5
# General-purpose zero-shot label set; callers may override per request.
_DEFAULT_IMAGE_LABELS: tuple[str, ...] = (
    "a photo of a person",
    "an animal",
    "food",
    "a landscape or nature scene",
    "a building or architecture",
    "a vehicle",
    "a product photo",
    "a screenshot",
    "a chart or diagram",
    "a document or text",
    "artwork or illustration",
    "a logo or icon",
)
# claude.ai's documented OAuth callback for remote-MCP custom connectors.
_DEFAULT_OAUTH_REDIRECT_URIS = ("https://claude.ai/api/mcp/auth_callback",)
# Production generation profile for the structured-output tools (Outlines-backed
# → json_schema + regex; EBNF is rejected up front by sie_mcp.structured).
_DEFAULT_GENERATE_MODEL = "Qwen/Qwen3.5-4B"
# Default output-token ceiling for the structured tools. The gateway's chat
# default is 1024, which truncates larger extraction/generation JSON; structured
# output is bounded but can exceed that, so default higher. A ceiling, not a
# target — small outputs still stop at the grammar's natural end.
_DEFAULT_MAX_OUTPUT_TOKENS = 4096

# Decoded-payload ceiling for a single document. The edge holds the whole document
# in memory (decode + rasterize), so an unbounded payload is a memory-exhaustion
# vector; 50 MiB comfortably covers real PDFs/scans. Override via SIE_MCP_MAX_DOCUMENT_BYTES.
DEFAULT_MAX_DOCUMENT_BYTES = 50 * 1024 * 1024

# Decoded-payload ceiling for a single image. Same memory-exhaustion concern as
# documents: describe_image base64-decodes the image into memory before sending it
# to the cluster. 20 MiB comfortably covers real JPEG/PNG photos and high-res scans
# rendered to images. Override via SIE_MCP_MAX_IMAGE_BYTES.
DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024

# Provisional VL-OCR default for scanned and complex pages. Pin MinerU2.5-Pro
# here as the demo default; deployments can override it through configuration.
DEFAULT_VLOCR_MODEL = "opendatalab/MinerU2.5-Pro-2604-1.2B"

# GLiNER models for extract_entities / redact_pii. Env-overridable (SIE_MCP_EXTRACT_MODEL
# / SIE_MCP_PII_MODEL) for parity with the docling/generation model knobs, so a cluster
# that advertises different GLiNER ids can be targeted without a code change.
DEFAULT_EXTRACT_MODEL = "urchade/gliner_multi-v2.1"
DEFAULT_PII_MODEL = "urchade/gliner_multi_pii-v1"


def _parse_connector_secrets(raw: str) -> dict[str, str]:
    """Parse ``SIE_MCP_CONNECTOR_SECRETS`` into a ``{secret: user_id}`` map.

    Format: comma-separated entries, each ``secret`` or ``secret:user_id``. A bare
    secret maps to itself as a stable opaque identity, so per-user metering can
    attach later without changing the wire contract.
    """
    mapping: dict[str, str] = {}
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        secret, _, user_id = entry.partition(":")
        secret = secret.strip()
        if not secret:
            continue
        mapping[secret] = user_id.strip() or secret
    return mapping


def _parse_redirect_uris(raw: str) -> tuple[str, ...]:
    """Parse ``SIE_MCP_OAUTH_REDIRECT_URIS`` into the OAuth redirect allowlist.

    Comma-separated absolute URIs. An empty value keeps the default (claude.ai's
    callback) so the common case needs no configuration.
    """
    uris = tuple(entry.strip() for entry in raw.split(",") if entry.strip())
    return uris or _DEFAULT_OAUTH_REDIRECT_URIS


def _parse_csv(raw: str) -> list[str]:
    """Parse a comma-separated env value into a list of trimmed, non-empty items."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _env_int(name: str, *, default: int, min_value: int = 1) -> int:
    # ``min_value`` is the lowest accepted value; below it (and for missing/blank/
    # non-integer values) we fall back to ``default``. Size/token limits keep
    # min_value=1 (a 0 cap is nonsensical); top_k passes min_value=0 so
    # SIE_MCP_IMAGE_TOP_K=0 is honored as a caption-only request.
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= min_value else default


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, *, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_labels(raw: str, *, default: tuple[str, ...]) -> list[str]:
    """Parse a comma-separated label set, falling back to ``default`` when unset."""
    if not raw.strip():
        return list(default)
    labels = [chunk.strip() for chunk in raw.split(",")]
    return [label for label in labels if label]


@dataclass(frozen=True)
class MCPConfig:
    """Resolved edge configuration."""

    sie_base_url: str
    sie_api_key: str | None
    connector_secrets: dict[str, str]
    allow_anonymous: bool
    allowed_hosts: list[str]
    max_document_bytes: int
    max_image_bytes: int
    docling_model: str
    generate_model: str
    extract_model: str
    pii_model: str
    max_output_tokens: int
    vlocr_model: str
    encode_model: str
    rerank_model: str
    caption_model: str
    embed_model: str
    image_labels: list[str]
    image_top_k: int
    gpu: str | None
    docs_gpu: str | None
    extract_gpu: str | None
    generate_gpu: str | None
    image_gpu: str | None
    qa_gpu: str | None
    timeout_s: float
    qa_top_k: int
    qa_rerank_candidates: int
    qa_chunk_chars: int
    qa_chunk_overlap_chars: int
    qa_max_tokens: int
    qa_max_document_chars: int
    qa_max_questions: int
    qa_max_chunks: int
    # claude.ai connectors are OAuth-only (no pasteable Bearer); the edge bridges
    # the OAuth handshake onto the connector-secret shim. `public_base_url`
    # pins the externally reachable origin used to build OAuth metadata URLs; when
    # unset it is derived per-request from a loopback or SIE_MCP_ALLOWED_HOSTS Host header.
    oauth_enabled: bool
    public_base_url: str | None
    oauth_redirect_uris: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "MCPConfig":
        secrets = _parse_connector_secrets(os.getenv("SIE_MCP_CONNECTOR_SECRETS", ""))
        # Clamp the retrieval knobs so a misconfigured deployment can't produce
        # silently-wrong slicing (negative limits) or an opaque per-request
        # ValueError (overlap >= window); overlap is pinned below the window.
        chunk_chars = max(1, _env_int("SIE_MCP_QA_CHUNK_CHARS", default=_DEFAULT_QA_CHUNK_CHARS))
        chunk_overlap = min(
            max(0, _env_int("SIE_MCP_QA_CHUNK_OVERLAP_CHARS", default=_DEFAULT_QA_CHUNK_OVERLAP_CHARS)), chunk_chars - 1
        )
        # One global GPU lane (SIE_MCP_GPU); each per-tool override below falls back
        # to it, so a single-knob setup keeps working while a multi-lane cluster can
        # route each tool to its own lane (docling on l4, generation on rtx6000, ...).
        gpu = os.getenv("SIE_MCP_GPU") or None
        return cls(
            sie_base_url=os.getenv("SIE_BASE_URL", _DEFAULT_BASE_URL),
            sie_api_key=os.getenv("SIE_API_KEY") or None,
            connector_secrets=secrets,
            # Fail closed: anonymous access is opt-in only. A missing/misnamed secret
            # env var must not silently expose the server-side cluster credential.
            allow_anonymous=_env_flag("SIE_MCP_ALLOW_ANONYMOUS", default=False),
            # Host header allow-list for DNS-rebinding protection; empty disables it
            # (remote edge behind TLS ingress + connector-secret auth). See server.py.
            allowed_hosts=_parse_csv(os.getenv("SIE_MCP_ALLOWED_HOSTS", "")),
            max_document_bytes=_env_int("SIE_MCP_MAX_DOCUMENT_BYTES", default=DEFAULT_MAX_DOCUMENT_BYTES),
            max_image_bytes=_env_int("SIE_MCP_MAX_IMAGE_BYTES", default=DEFAULT_MAX_IMAGE_BYTES),
            docling_model=os.getenv("SIE_MCP_DOCLING_MODEL", _DEFAULT_MODEL),
            generate_model=os.getenv("SIE_MCP_GENERATE_MODEL", _DEFAULT_GENERATE_MODEL),
            extract_model=(os.getenv("SIE_MCP_EXTRACT_MODEL") or "").strip() or DEFAULT_EXTRACT_MODEL,
            pii_model=(os.getenv("SIE_MCP_PII_MODEL") or "").strip() or DEFAULT_PII_MODEL,
            max_output_tokens=_env_int("SIE_MCP_MAX_OUTPUT_TOKENS", default=_DEFAULT_MAX_OUTPUT_TOKENS),
            # Treat an empty/whitespace override as unset so OCR routing keeps a real model id.
            vlocr_model=(os.getenv("SIE_MCP_VLOCR_MODEL") or "").strip() or DEFAULT_VLOCR_MODEL,
            encode_model=os.getenv("SIE_MCP_ENCODE_MODEL", _DEFAULT_ENCODE_MODEL),
            rerank_model=os.getenv("SIE_MCP_RERANK_MODEL", _DEFAULT_RERANK_MODEL),
            # Treat an empty/whitespace override as unset so a real model id always stands.
            caption_model=(os.getenv("SIE_MCP_CAPTION_MODEL") or "").strip() or _DEFAULT_CAPTION_MODEL,
            embed_model=(os.getenv("SIE_MCP_EMBED_MODEL") or "").strip() or _DEFAULT_EMBED_MODEL,
            image_labels=_parse_labels(os.getenv("SIE_MCP_IMAGE_LABELS", ""), default=_DEFAULT_IMAGE_LABELS),
            # min_value=0 so a caption-only request (SIE_MCP_IMAGE_TOP_K=0) is honored.
            image_top_k=_env_int("SIE_MCP_IMAGE_TOP_K", default=_DEFAULT_IMAGE_TOP_K, min_value=0),
            gpu=gpu,
            docs_gpu=os.getenv("SIE_MCP_DOCS_GPU") or gpu,
            extract_gpu=os.getenv("SIE_MCP_EXTRACT_GPU") or gpu,
            generate_gpu=os.getenv("SIE_MCP_GENERATE_GPU") or gpu,
            image_gpu=os.getenv("SIE_MCP_IMAGE_GPU") or gpu,
            qa_gpu=os.getenv("SIE_MCP_QA_GPU") or gpu,
            # Document conversion can be slow on a cold model load; default generous.
            timeout_s=_env_float("SIE_MCP_TIMEOUT_S", default=300.0),
            qa_top_k=max(1, _env_int("SIE_MCP_QA_TOP_K", default=_DEFAULT_QA_TOP_K)),
            qa_rerank_candidates=max(
                1, _env_int("SIE_MCP_QA_RERANK_CANDIDATES", default=_DEFAULT_QA_RERANK_CANDIDATES)
            ),
            qa_chunk_chars=chunk_chars,
            qa_chunk_overlap_chars=chunk_overlap,
            qa_max_tokens=max(1, _env_int("SIE_MCP_QA_MAX_TOKENS", default=_DEFAULT_QA_MAX_TOKENS)),
            qa_max_document_chars=max(
                1, _env_int("SIE_MCP_QA_MAX_DOCUMENT_CHARS", default=_DEFAULT_QA_MAX_DOCUMENT_CHARS)
            ),
            qa_max_questions=max(1, _env_int("SIE_MCP_QA_MAX_QUESTIONS", default=_DEFAULT_QA_MAX_QUESTIONS)),
            qa_max_chunks=max(1, _env_int("SIE_MCP_QA_MAX_CHUNKS", default=_DEFAULT_QA_MAX_CHUNKS)),
            oauth_enabled=_env_flag("SIE_MCP_OAUTH_ENABLED", default=True),
            public_base_url=(os.getenv("SIE_MCP_PUBLIC_URL") or "").rstrip("/") or None,
            oauth_redirect_uris=_parse_redirect_uris(os.getenv("SIE_MCP_OAUTH_REDIRECT_URIS", "")),
        )
