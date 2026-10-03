"""VL-OCR quality routing for ``docs_to_markdown``.

Born-digital documents convert cleanly through Docling, but scanned / image-only
pages carry no text layer for Docling to read — its built-in OCR is the baseline
this path improves on. We rasterize such pages and route them to a vision-language
OCR model (MinerU2.5-Pro, the provisional default pinned in
:mod:`sie_mcp.config`), then stitch the per-page markdown back together in page
order.

Two return shapes have to be reconciled here: Docling fills
``ExtractResult["data"]["markdown"]``, whereas MinerU returns generated text in
``ExtractResult["entities"]`` (each ``{text, label: "mineru_…", score}``) — see
``sie_server.adapters.mineru_vl``. This module owns both the rendering mechanism
and the engine-selection policy; orchestration stays client-side in the MCP edge
so the gateway and workers remain stateless per request.
"""

import io
import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import pypdfium2 as pdfium
from PIL import Image
from sie_sdk import SIEError

logger = logging.getLogger(__name__)

EngineName = Literal["auto", "docling", "vl-ocr"]
ENGINES: tuple[EngineName, ...] = ("auto", "docling", "vl-ocr")

# PDF user space is 72 DPI; scale 2.0 → ~144 DPI, a quality/size balance that keeps
# rasterized pages legible for VL-OCR without ballooning the request payload.
_RENDER_SCALE = 2.0
# Page boundary in the stitched markdown — a blank line reads as a paragraph break,
# preserving document structure without injecting synthetic page markers.
_PAGE_SEPARATOR = "\n\n"
# Below this many non-whitespace chars, a renderable document is treated as
# imaged/scanned and (in ``auto``) re-run through VL-OCR. Born-digital pages clear
# this comfortably; a text-layer-less scan lands at ~0.
_NEAR_EMPTY_MARKDOWN_CHARS = 16
# Cap pages sent to VL-OCR. Each page is rasterized to a bitmap and JPEG-encoded
# into the request body, so an unbounded count would exhaust edge memory and the
# gateway request-size limit. Above this we fail loudly (no silent truncation) —
# windowed processing of very large scans is future work.
_MAX_VLOCR_PAGES = 100
# Cap rendered pixels per page so a single pathologically large page (page
# dimensions are attacker-controlled) can't exhaust edge memory. We downscale
# rather than fail. ~25 MP is generous — an A4 scan at 600 DPI is ~35 MP, and
# our ~144 DPI render lands far below this for normal pages.
_MAX_PAGE_PIXELS = 25_000_000
# Cap total rendered pixels across a document so the per-page cap doesn't
# multiply into a huge envelope: ``_MAX_VLOCR_PAGES`` × ``_MAX_PAGE_PIXELS`` is
# 2.5 Gpx (~7.5 GB of RGB) before encode/transport overhead, since every page is
# held in memory at once for the single batched ``extract`` call. ~500 Mpx caps
# the held RGB at ~1.5 GB while clearing a 100-page born-digital scan (~2 Mpx/page
# at our render scale) with wide headroom. We fail loudly (no silent truncation);
# windowed processing of very large scans is future work.
_MAX_TOTAL_PIXELS = 500_000_000

_PDF_MAGIC = b"%PDF-"
# Image magic bytes covering the formats Pillow can open for VL-OCR. TIFF in
# particular is a common scanned-document container, so it must route too.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")
_GIF_MAGIC = (b"GIF87a", b"GIF89a")
_BMP_MAGIC = b"BM"


class DocsToMarkdownError(Exception):
    """Raised when the cluster could not convert the document."""


class ExtractClient(Protocol):
    """The slice of ``SIEAsyncClient`` this module needs (keeps routing unit-testable)."""

    async def extract(self, model: str, items: Any, **kwargs: Any) -> Any:
        """Run an extract request against the SIE cluster."""


@dataclass(frozen=True)
class RouteResult:
    """Outcome of routing a document to markdown.

    ``document`` carries Docling's per-item document dict (page metadata) when the
    Docling path won; ``pages`` carries the rendered page count on the VL-OCR path.
    """

    markdown: str
    engine: Literal["docling", "vl-ocr"]
    document: Any = None
    pages: int | None = None


def entities_to_markdown(result: Any) -> str:
    """Map a MinerU ``ExtractResult``'s ``entities`` to one page's markdown.

    The default text-recognition task yields a single ``mineru_text`` entity per
    page; concatenating every entity's text in order also preserves table/equation
    entities should a layout task be used. Returns ``""`` for a page with no text.
    """
    entities = result.get("entities") if isinstance(result, dict) else None
    if not isinstance(entities, list):
        return ""
    parts = [
        entity["text"]
        for entity in entities
        if isinstance(entity, dict) and isinstance(entity.get("text"), str) and entity["text"].strip()
    ]
    return _PAGE_SEPARATOR.join(parts)


def _is_pdf(data: bytes) -> bool:
    return data[:5] == _PDF_MAGIC


def _is_image(data: bytes) -> bool:
    return (
        data[:8] == _PNG_MAGIC
        or data[:3] == _JPEG_MAGIC
        or data[:4] in _TIFF_MAGIC
        or data[:6] in _GIF_MAGIC
        or data[:2] == _BMP_MAGIC
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    )


def is_renderable(data: bytes) -> bool:
    """True when the bytes can be rasterized to page images for VL-OCR (PDF or image)."""
    return _is_pdf(data) or _is_image(data)


def _clamp_scale(width_pt: float, height_pt: float, scale: float) -> float:
    """Reduce ``scale`` so a page renders within ``_MAX_PAGE_PIXELS``; leave it as-is otherwise."""
    pixels = (width_pt * scale) * (height_pt * scale)
    if pixels <= _MAX_PAGE_PIXELS or pixels <= 0:
        return scale
    return scale * math.sqrt(_MAX_PAGE_PIXELS / pixels)


def _downscale_to_budget(image: Image.Image) -> Image.Image:
    """Return ``image`` shrunk to ``_MAX_PAGE_PIXELS`` if it exceeds the budget, else unchanged."""
    pixels = image.width * image.height
    if pixels <= _MAX_PAGE_PIXELS:
        return image
    factor = math.sqrt(_MAX_PAGE_PIXELS / pixels)
    size = (max(1, int(image.width * factor)), max(1, int(image.height * factor)))
    return image.resize(size)


def _render_pdf(data: bytes, *, scale: float) -> list[Image.Image]:
    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        msg = f"could not open PDF for VL-OCR rendering: {exc}"
        raise DocsToMarkdownError(msg) from exc
    try:
        n_pages = len(pdf)
        if n_pages > _MAX_VLOCR_PAGES:
            msg = (
                f"PDF has {n_pages} pages; VL-OCR is capped at {_MAX_VLOCR_PAGES} per request. "
                "Split the document or use engine='docling'."
            )
            raise DocsToMarkdownError(msg)
        images: list[Image.Image] = []
        total_pixels = 0
        for page in pdf:
            # Close every page even if rendering raises mid-loop (no handle leak),
            # and surface render failures as DocsToMarkdownError so ``auto`` can degrade.
            try:
                width_pt, height_pt = page.get_size()
                bitmap = page.render(scale=_clamp_scale(width_pt, height_pt, scale))
                image = bitmap.to_pil().convert("RGB")
            finally:
                page.close()
            # Bound total held memory: stop before the accumulated bitmaps blow the
            # budget (every page lives in ``images`` until the single batched extract).
            total_pixels += image.width * image.height
            if total_pixels > _MAX_TOTAL_PIXELS:
                msg = (
                    f"rendered pixels exceed the {_MAX_TOTAL_PIXELS:,}px VL-OCR budget "
                    f"by page {len(images) + 1}. Split the document or use engine='docling'."
                )
                raise DocsToMarkdownError(msg)
            images.append(image)
        return images
    except pdfium.PdfiumError as exc:
        msg = f"could not render PDF page for VL-OCR: {exc}"
        raise DocsToMarkdownError(msg) from exc
    finally:
        pdf.close()


def render_to_images(data: bytes, *, scale: float = _RENDER_SCALE) -> list[Image.Image]:
    """Rasterize a document to one RGB image per page (PDF) or a single image (PNG/JPEG)."""
    if _is_pdf(data):
        return _render_pdf(data, scale=scale)
    if _is_image(data):
        try:
            return [_downscale_to_budget(Image.open(io.BytesIO(data)).convert("RGB"))]
        except (OSError, ValueError) as exc:
            # PIL's UnidentifiedImageError/truncated-file errors subclass OSError;
            # wrap so a magic-byte false positive degrades cleanly in ``auto``.
            msg = f"could not open image for VL-OCR rendering: {exc}"
            raise DocsToMarkdownError(msg) from exc
    msg = "VL-OCR can only rasterize PDF or image inputs; use engine='docling' for this format."
    raise DocsToMarkdownError(msg)


async def _docling_markdown(
    client: ExtractClient,
    *,
    data: bytes,
    fmt: str | None,
    ocr: bool,
    model: str,
    gpu: str | None,
) -> tuple[str, Any]:
    """Run the Docling front door; return ``(markdown, document_metadata)``."""
    document: dict[str, Any] = {"data": data}
    if fmt:
        document["format"] = fmt
    options = {"ocr": True} if ocr else None
    result = await client.extract(model, {"document": document}, options=options, gpu=gpu)

    payload = _docling_payload(result)
    markdown = payload.get("markdown") or ""
    error = payload.get("error")
    if error and not markdown:
        raise DocsToMarkdownError(str(error))
    return markdown, payload.get("document")


def _docling_payload(result: Any) -> dict[str, Any]:
    """Pull the Docling per-item dict (``{text, markdown, document}``) out of an ExtractResult."""
    data = result.get("data") if isinstance(result, dict) else None
    if isinstance(data, list):
        data = data[0] if data else None
    return data if isinstance(data, dict) else {}


async def _vl_ocr_markdown(
    client: ExtractClient,
    *,
    data: bytes,
    model: str,
    gpu: str | None,
    scale: float,
    render: Any,
) -> tuple[str, int]:
    """Rasterize, OCR each page on the cluster, and stitch markdown in page order."""
    images = render(data, scale=scale)
    if not images:
        return "", 0
    items = [{"images": [image]} for image in images]
    results = await client.extract(model, items, gpu=gpu)
    if not isinstance(results, list):
        results = [results]
    page_markdowns = [entities_to_markdown(result) for result in results]
    return _PAGE_SEPARATOR.join(page_markdowns), len(images)


async def route_to_markdown(
    client: ExtractClient,
    *,
    data: bytes,
    fmt: str | None,
    ocr: bool,
    engine: str,
    docling_model: str,
    vlocr_model: str,
    gpu: str | None,
    render: Any = render_to_images,
    scale: float = _RENDER_SCALE,
) -> RouteResult:
    """Convert a document to markdown, choosing the engine per ``engine``.

    - ``docling``: Docling only (the current front door, all formats). The caller's
      ``ocr`` flag turns Docling's built-in OCR on here.
    - ``vl-ocr``: rasterize and run the VL-OCR model per page; stitch in order.
    - ``auto``: Docling first, falling back to VL-OCR when Docling yields empty /
      near-empty markdown for a rasterizable input (scanned PDFs, image-only docs).

    In ``auto`` the engine owns the OCR decision, so the caller's ``ocr`` flag is
    ignored: Docling's built-in OCR is the baseline VL-OCR supersedes, and running it
    would hand back garbled, run-together text that is both worse than VL-OCR *and*
    non-empty — masking the scan from the near-empty fallback trigger. Docling stays
    the text-layer front door (OCR off); scans land near-empty and fall back cleanly.

    Auto fallback is best-effort: a VL-OCR failure or an emptier VL-OCR result never
    discards a usable Docling result. Per-page hybrid routing (Docling for digital
    pages, VL-OCR only for scanned pages within one mixed doc) is a future thickening;
    today a mixed doc is best served by forcing ``engine='vl-ocr'``.
    """
    if engine not in ENGINES:
        msg = f"unknown engine {engine!r}; expected one of {', '.join(ENGINES)}"
        raise DocsToMarkdownError(msg)

    async def _vl_ocr() -> tuple[str, int]:
        return await _vl_ocr_markdown(client, data=data, model=vlocr_model, gpu=gpu, scale=scale, render=render)

    if engine == "vl-ocr":
        markdown, pages = await _vl_ocr()
        return RouteResult(markdown=markdown, engine="vl-ocr", pages=pages)

    # Honor ``ocr`` only when Docling is pinned; in ``auto`` VL-OCR is the OCR path.
    docling_ocr = ocr and engine == "docling"
    if ocr and not docling_ocr:
        # Surface the override so a caller who set ocr=True doesn't silently wonder
        # why Docling's built-in OCR never ran (VL-OCR fallback owns scans in auto).
        logger.debug("ocr=True ignored under engine=%r; VL-OCR fallback owns OCR in auto", engine)
    try:
        markdown, document = await _docling_markdown(
            client, data=data, fmt=fmt, ocr=docling_ocr, model=docling_model, gpu=gpu
        )
    except DocsToMarkdownError:
        # Docling refused (e.g. no text layer). For a rasterizable input, try VL-OCR
        # before giving up; otherwise the error stands.
        if engine == "auto" and is_renderable(data):
            vl_markdown, pages = await _vl_ocr()
            if vl_markdown.strip():
                return RouteResult(markdown=vl_markdown, engine="vl-ocr", pages=pages)
        raise

    if engine == "auto" and _looks_text_poor(markdown) and is_renderable(data):
        try:
            vl_markdown, pages = await _vl_ocr()
        except (DocsToMarkdownError, SIEError):
            # VL-OCR is best-effort here: a render error *or* an SDK/server failure
            # from the cluster must not discard the usable Docling markdown we hold.
            logger.warning("VL-OCR fallback failed; keeping Docling result", exc_info=True)
            vl_markdown, pages = "", 0
        # Only prefer VL-OCR when it actually recovered more text than Docling.
        if len(vl_markdown.strip()) > len(markdown.strip()):
            return RouteResult(markdown=vl_markdown, engine="vl-ocr", pages=pages)

    return RouteResult(markdown=markdown, engine="docling", document=document)


def _looks_text_poor(markdown: str) -> bool:
    """Heuristic: Docling produced too little text to be a born-digital extraction."""
    return len(markdown.strip()) < _NEAR_EMPTY_MARKDOWN_CHARS
