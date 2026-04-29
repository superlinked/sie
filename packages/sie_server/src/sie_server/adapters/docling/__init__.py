from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.core.inference_output import ExtractOutput
from sie_server.types.inputs import is_document_input

if TYPE_CHECKING:
    from sie_server.types.inputs import Item


logger = logging.getLogger(__name__)


_ERR_REQUIRES_DOCUMENT = "Docling requires Item.document with raw bytes"

# Minimal one-page PDF used to warm Docling's layout/table model downloads
# during load() so the first user request doesn't pay the download latency.
_TINY_PDF_BYTES = (
    b"%PDF-1.1\n%\xc2\xa5\xc2\xb1\xc3\xab\n\n1 0 obj\n  << /Type /Catalog\n     /Pages 2 0 R\n  >>\n"
    b"endobj\n\n2 0 obj\n  << /Type /Pages\n     /Kids [3 0 R]\n     /Count 1\n     /MediaBox [0 0 300 144]\n  >>\n"
    b"endobj\n\n3 0 obj\n  <<  /Type /Page\n      /Parent 2 0 R\n      /Resources\n       <<"
    b" /Font\n           <<\n             /F1\n              << /Type /Font\n                 /Subtype /Type1\n"
    b"                 /BaseFont /Times-Roman\n              >>\n           >>\n       >>\n      /Contents 4 0 R\n  >>\n"
    b"endobj\n\n4 0 obj\n  << /Length 55 >>\nstream\n  BT\n    /F1 18 Tf\n    0 0 Td\n    (Hello, world!) Tj\n  ET\nendstream\n"
    b"endobj\n\nxref\n0 5\n0000000000 65535 f \n0000000018 00000 n \n0000000077 00000 n \n0000000178 00000 n \n0000000457 00000 n \n"
    b"trailer\n  <<  /Root 1 0 R\n      /Size 5\n  >>\nstartxref\n565\n%%EOF\n"
)


class DoclingAdapter(BaseAdapter):
    """Composite-document parser backed by Docling's `DocumentConverter`.

    Supports PDF, DOCX, HTML, and other formats Docling auto-detects from bytes.
    The adapter is package-backed (no HF/local weights) — Docling ships its own
    layout / table-structure models which are downloaded on first use. We pre-warm
    those during ``load()`` so user requests don't pay that latency.

    Result shape (per item, in ``ExtractOutput.data``):

        {
            "text": "...",          # plain text rendering
            "markdown": "...",      # Markdown rendering (preserves tables, headings)
            "document": {...},      # full DoclingDocument JSON for downstream chunkers
        }

    OCR is disabled by default for speed and predictability. Pass
    ``options={"ocr": True}`` per request to enable it.

    Concurrency: a fresh ``DocumentConverter`` is built per item rather than
    sharing one across threads. Construction is ~10 ms once Docling's
    layout/table models have been pre-warmed (they cache globally), and this
    sidesteps thread-safety concerns reported upstream
    (https://github.com/docling-project/docling/issues/115).
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("document",),
        outputs=("json",),
        unload_fields=(),
    )

    def __init__(
        self,
        model_name_or_path: str | None = None,  # unused; Docling is package-backed
        *,
        compute_precision: str | None = None,  # unused; Docling runs on CPU
        **kwargs: Any,
    ) -> None:
        _ = (model_name_or_path, compute_precision, kwargs)
        self._loaded = False
        self._device: str | None = None

    def load(self, device: str) -> None:
        self._device = device
        # Pre-warm: triggers Docling's lazy download of layout/table models so
        # the first real request doesn't block on a multi-hundred-MB pull.
        # Models cache globally, so subsequent per-task converters are cheap.
        try:
            warm_converter = self._make_converter(ocr_enabled=False)
            self._convert_bytes(warm_converter, _TINY_PDF_BYTES, format_hint="pdf")
        except Exception:
            logger.exception("Docling pre-warm failed; first real request may be slow")
        self._loaded = True

    def extract(
        self,
        items: list[Item],
        *,
        labels: list[str] | None = None,
        output_schema: dict[str, Any] | None = None,
        instruction: str | None = None,
        options: dict[str, Any] | None = None,
        prepared_items: list[Any] | None = None,
    ) -> ExtractOutput:
        _ = (labels, output_schema, instruction, prepared_items)
        if not self._loaded:
            msg = "DoclingAdapter.load() must be called before extract()"
            raise RuntimeError(msg)

        ocr_enabled = bool(options and options.get("ocr"))
        results = self._run_extract(items, ocr_enabled=ocr_enabled)

        return ExtractOutput(
            entities=[[] for _ in items],
            data=results,
            batch_size=len(items),
        )

    def _run_extract(self, items: list[Item], *, ocr_enabled: bool) -> list[dict[str, Any]]:
        """Run extract per-item, parallelized across the batch.

        Each task gets its own DocumentConverter (see class docstring).
        """
        if len(items) <= 1:
            return [self._extract_one(item, ocr_enabled=ocr_enabled) for item in items]

        with ThreadPoolExecutor(max_workers=min(len(items), 4)) as pool:
            futures = [pool.submit(self._extract_one, item, ocr_enabled=ocr_enabled) for item in items]
            return [f.result() for f in futures]

    def _extract_one(self, item: Item, *, ocr_enabled: bool) -> dict[str, Any]:
        document = item.document
        if not is_document_input(document):
            return {"error": _ERR_REQUIRES_DOCUMENT}
        try:
            converter = self._make_converter(ocr_enabled=ocr_enabled)
            return self._convert_bytes(converter, document["data"], format_hint=document.get("format"))
        except Exception as e:  # noqa: BLE001 - per-item failure must not poison the batch
            logger.warning("Docling extract failed for item id=%s: %s", item.id, e)
            return {"error": str(e)}

    def _convert_bytes(self, converter: Any, data: bytes, *, format_hint: str | None) -> dict[str, Any]:
        from docling.datamodel.base_models import DocumentStream  # ty: ignore[unresolved-import]

        # Docling auto-detects format from bytes; the hint becomes the source name
        # (used for logging + extension-based fallback when sniffing is ambiguous).
        source_name = f"document.{format_hint}" if format_hint else "document"
        stream = DocumentStream(name=source_name, stream=io.BytesIO(data))
        result = converter.convert(stream)
        doc = result.document
        return {
            "text": doc.export_to_text(),
            "markdown": doc.export_to_markdown(),
            "document": doc.export_to_dict(),
        }

    def _make_converter(self, *, ocr_enabled: bool) -> Any:
        """Build a fresh DocumentConverter. One per task — see class docstring."""
        from docling.document_converter import DocumentConverter  # ty: ignore[unresolved-import]

        if not ocr_enabled:
            return DocumentConverter()

        from docling.datamodel.base_models import InputFormat  # ty: ignore[unresolved-import]
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # ty: ignore[unresolved-import]
        from docling.document_converter import PdfFormatOption  # ty: ignore[unresolved-import]

        pdf_opts = PdfPipelineOptions(do_ocr=True)
        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)})
