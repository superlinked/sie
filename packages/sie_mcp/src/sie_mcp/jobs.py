"""The ``docs_to_markdown`` document-conversion job.

Stateless orchestration over the SIE ``extract`` primitive: a document's bytes go
to the cluster and come back as markdown. Docling is the front door for all
formats; scanned / complex-layout pages route to a VL-OCR model for better
quality (see :mod:`sie_mcp.ocr_routing`). Runs entirely client-side in the MCP
edge; the gateway and workers stay stateless per request.
"""

import base64
import binascii
from typing import Any, TypedDict

from sie_mcp import ocr_routing
from sie_mcp.config import DEFAULT_MAX_DOCUMENT_BYTES, DEFAULT_VLOCR_MODEL
from sie_mcp.documents import format_from_filename
from sie_mcp.ocr_routing import DocsToMarkdownError, EngineName, ExtractClient
from sie_mcp.savings import build_metadata


class DocsToMarkdownResult(TypedDict):
    markdown: str
    metadata: dict[str, Any]


def _decode_base64(document_base64: str, *, max_bytes: int) -> bytes:
    # Reject oversize payloads from the base64 length before allocating the decoded
    # buffer (decoded size ≈ 3/4 of the encoded length), then enforce the exact bound.
    if (len(document_base64) // 4) * 3 > max_bytes:
        msg = f"document exceeds the {max_bytes}-byte limit"
        raise DocsToMarkdownError(msg)
    try:
        data = base64.b64decode(document_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg = f"document_base64 is not valid base64: {exc}"
        raise DocsToMarkdownError(msg) from exc
    if len(data) > max_bytes:
        msg = f"document exceeds the {max_bytes}-byte limit"
        raise DocsToMarkdownError(msg)
    return data


async def docs_to_markdown(
    client: ExtractClient,
    *,
    document_base64: str,
    filename: str | None = None,
    ocr: bool = False,
    engine: EngineName = "auto",
    model: str = "docling",
    vlocr_model: str = DEFAULT_VLOCR_MODEL,
    gpu: str | None = None,
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
) -> DocsToMarkdownResult:
    """Convert a document to markdown on the SIE cluster, routing the engine per ``engine``."""
    data = _decode_base64(document_base64, max_bytes=max_document_bytes)
    fmt = format_from_filename(filename)

    result = await ocr_routing.route_to_markdown(
        client,
        data=data,
        fmt=fmt,
        ocr=ocr,
        engine=engine,
        docling_model=model,
        vlocr_model=vlocr_model,
        gpu=gpu,
    )

    return DocsToMarkdownResult(
        markdown=result.markdown,
        metadata=build_metadata(
            markdown=result.markdown,
            document=result.document,
            source_bytes=len(data),
            pages=result.pages,
            engine=result.engine,
        ),
    )
