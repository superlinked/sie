from __future__ import annotations

import io
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from sie_server.adapters._base_adapter import BaseAdapter
from sie_server.adapters._spec import AdapterSpec
from sie_server.core.inference_output import ExtractItemError, ExtractOutput
from sie_server.types.inputs import is_document_input, is_image_input
from sie_server.types.responses import ErrorCode

if TYPE_CHECKING:
    from sie_server.types.inputs import Item


logger = logging.getLogger(__name__)


def _processed_page_count(result: Any) -> int:
    """Return the pages Docling actually processed for this conversion.

    ``ConversionResult.pages`` is authoritative for partial conversions,
    including an empty list. Older Docling result shapes without that field
    fall back to the exported document count. Metering is best-effort and must
    never fail a parse.
    """
    try:
        pages = getattr(result, "pages", None)
        if pages is not None:
            return max(len(pages), 0)
        doc = result.document
        num_pages = getattr(doc, "num_pages", None)
        if callable(num_pages):
            return max(int(num_pages()), 0)
        pages = getattr(doc, "pages", None)
        if pages:
            return max(len(pages), 0)
    except Exception:  # noqa: BLE001 — metering must never fail the parse
        return 0
    return 0


_ERR_REQUIRES_DOCUMENT = "Document or image input is required"
_ERR_EXACTLY_ONE_IMAGE = "Docling OCR requires exactly one image per item"
_ERR_CONVERSION_FAILED = "Document conversion failed"
_ERR_EXPORT_FAILED = "Document export failed"


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg)
    return value


def _positive_finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a positive finite number"
        raise ValueError(msg)
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        msg = f"{name} must be a positive finite number"
        raise ValueError(msg)
    return parsed


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
    Docling expects one artifact root containing its layout, table, and OCR
    assets. The adapter accepts that root through the same ``model_name_or_path``
    contract as other adapters: a local ``weights_path`` is used directly and an
    ``hf_id`` is resolved at its pinned revision. The legacy package-artifact
    root remains supported while existing configs migrate to the ordinary model
    source contract.

    Result shape (per item, in ``ExtractOutput.data``):

        {
            "text": "...",          # plain text rendering
            "markdown": "...",      # Markdown rendering (preserves tables, headings)
            "document": {...},      # full DoclingDocument JSON for downstream chunkers
        }

    OCR is disabled by default for speed and predictability. Pass
    ``options={"ocr": True}`` per request to enable it.

    Concurrency: one ``DocumentConverter`` is cached per ``ocr_enabled`` value
    on the adapter instance. ``self._device`` is set once in ``load()`` and is
    stable for the adapter's lifetime, so the effective cache key is
    ``(self._device, ocr_enabled)`` and at most two converters ever exist per
    adapter instance. Cross-request serialization is provided by
    ``ModelWorker._inference_executor`` (max_workers=1), so the cache itself
    does not need a lock. Items within one batch are processed serially
    (rather than via a per-item thread pool) to sidestep the converter's known
    thread-safety issue (https://github.com/docling-project/docling/issues/115);
    at GPU-bound concurrency the upstream worker is already saturating the
    device, so intra-batch parallelism does not buy real throughput.
    """

    spec: ClassVar[AdapterSpec] = AdapterSpec(
        inputs=("document", "image"),
        outputs=("json",),
        unload_fields=(),
    )

    def __init__(
        self,
        model_name_or_path: str | Path | None = None,
        *,
        revision: str | None = None,
        compute_precision: str | None = None,  # unused; device is threaded via load()
        max_num_pages: int = 100,
        max_file_size_bytes: int = 16 * 1024 * 1024,
        document_timeout_s: float = 90.0,
        package_artifact_root: str | Path | None = None,
        package_artifact_manifest_sha256: str | None = None,
        **kwargs: Any,
    ) -> None:
        _ = (compute_precision, kwargs)
        if model_name_or_path is not None and package_artifact_root is not None:
            raise ValueError("Docling requires exactly one ordinary or package artifact source")
        self._max_num_pages = _positive_int("max_num_pages", max_num_pages)
        self._max_file_size_bytes = _positive_int("max_file_size_bytes", max_file_size_bytes)
        self._document_timeout_s = _positive_finite_float("document_timeout_s", document_timeout_s)
        self._loaded = False
        self._device: str | None = None
        self._converters: dict[bool, Any] = {}
        self._model_name_or_path = model_name_or_path
        self._revision = revision
        self._package_artifact_root = Path(package_artifact_root) if package_artifact_root is not None else None
        self._package_artifact_manifest_sha256 = package_artifact_manifest_sha256
        self._artifacts_path: Path | None = None

    def load(self, device: str) -> None:
        self._device = device
        self._artifacts_path = self._resolve_artifacts_path()
        # Pre-warm: triggers Docling's lazy download of layout/table models so
        # the first real request doesn't block on a multi-hundred-MB pull.
        # Models cache globally, so subsequent per-task converters are cheap.
        try:
            warm_converter = self._get_converter(ocr_enabled=False)
            _payload, _pages, warm_error = self._convert_bytes(
                warm_converter,
                _TINY_PDF_BYTES,
                format_hint="pdf",
            )
            if warm_error is not None:
                raise RuntimeError(warm_error.message)
            # DocumentConverter construction is lazy, so exercise the OCR
            # converter too. This initializes its OCR assets during readiness
            # instead of deferring a missing staged file to the first request.
            ocr_converter = self._get_converter(ocr_enabled=True)
            _payload, _pages, ocr_warm_error = self._convert_bytes(
                ocr_converter,
                _TINY_PDF_BYTES,
                format_hint="pdf",
            )
            if ocr_warm_error is not None:
                raise RuntimeError(ocr_warm_error.message)
        except Exception as exc:
            if self._artifacts_path is not None:
                raise RuntimeError("Docling staged artifact initialization failed") from exc
            logger.exception("Docling pre-warm failed; first real request may be slow")
        self._loaded = True

    def _resolve_artifacts_path(self) -> Path | None:
        """Resolve an ordinary local or immutable Hub model source for Docling."""
        if self._package_artifact_root is not None:
            return self._package_artifact_root
        if self._model_name_or_path is None:
            return None
        if isinstance(self._model_name_or_path, Path):
            path = self._model_name_or_path.resolve()
            if not path.is_dir():
                raise ValueError("Docling local artifact source must be an existing directory")
            return path

        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=self._model_name_or_path,
                revision=self._revision,
            )
        ).resolve()

    def unload(self) -> None:
        self._converters.clear()
        self._artifacts_path = None
        self._loaded = False
        super().unload()

    def count_input_images(self, items: list[Item]) -> None:
        """OCR images are pages; avoid also emitting the generic image unit."""
        del items

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

        data = [result for result, _pages, _error in results]
        # Unit-meter seam (§7): the canonical parse/OCR billing dimension is
        # PAGES ("$ per 1k pages", design §7). Docling's document model knows
        # exactly how many pages it parsed per item, so surface that real count
        # to the queue executor (folded into ``ItemOutcome.units.pages``).
        # Zero is authoritative too: a terminal conversion failure before the
        # first page releases the reserve instead of charging a fallback unit.
        page_counts = [pages for _result, pages, _error in results]
        errors = [error for _result, _pages, error in results]
        emit_errors = errors if any(error is not None for error in errors) else None

        return ExtractOutput(
            entities=[[] for _ in items],
            data=data,
            errors=emit_errors,
            batch_size=len(items),
            pages=page_counts,
        )

    def _run_extract(
        self,
        items: list[Item],
        *,
        ocr_enabled: bool,
    ) -> list[tuple[dict[str, Any], int, ExtractItemError | None]]:
        """Run extract per-item, serially.

        Items are processed one at a time so we can share a single cached
        DocumentConverter (see class docstring). At GPU-bound concurrency the
        worker-level inference executor is already saturating the device, so
        intra-batch parallelism does not buy real throughput.

        Returns per item a ``(result, page_count, error)`` tuple. ``page_count``
        is the number of document pages Docling processed, including for partial
        conversions and export failures.
        """
        return [self._extract_one(item, ocr_enabled=ocr_enabled) for item in items]

    def _extract_one(
        self,
        item: Item,
        *,
        ocr_enabled: bool,
    ) -> tuple[dict[str, Any], int, ExtractItemError | None]:
        # Prefer document when both are provided: PDF/DOCX/HTML carry layout
        # that Docling's pipeline exploits; an image is the rasterized
        # fallback for callers that only have a page render.
        document = item.document
        if is_document_input(document):
            payload = document["data"]
            format_hint = document.get("format")
        else:
            images = item.images or []
            if not images:
                return {}, 0, ExtractItemError(code=ErrorCode.INVALID_INPUT.value, message=_ERR_REQUIRES_DOCUMENT)
            if len(images) != 1:
                return (
                    {},
                    0,
                    ExtractItemError(
                        code=ErrorCode.INVALID_INPUT.value,
                        message=_ERR_EXACTLY_ONE_IMAGE,
                    ),
                )
            first_image = images[0]
            if not is_image_input(first_image):
                return {}, 0, ExtractItemError(code=ErrorCode.INVALID_INPUT.value, message=_ERR_REQUIRES_DOCUMENT)
            payload = first_image["data"]
            format_hint = first_image.get("format") or "png"
        try:
            converter = self._get_converter(ocr_enabled=ocr_enabled)
            return self._convert_bytes(converter, payload, format_hint=format_hint)
        except Exception as exc:  # noqa: BLE001 - per-item failure must not poison the batch
            logger.warning("Docling conversion raised error_type=%s", type(exc).__name__)
            return {}, 0, ExtractItemError(code=ErrorCode.INFERENCE_ERROR.value, message=_ERR_CONVERSION_FAILED)

    def _get_converter(self, *, ocr_enabled: bool) -> Any:
        """Return the cached DocumentConverter for this ocr_enabled value, building lazily on first use."""
        cached = self._converters.get(ocr_enabled)
        if cached is not None:
            return cached
        converter = self._make_converter(ocr_enabled=ocr_enabled)
        self._converters[ocr_enabled] = converter
        return converter

    def _convert_bytes(
        self,
        converter: Any,
        data: bytes,
        *,
        format_hint: str | None,
    ) -> tuple[dict[str, Any], int, ExtractItemError | None]:
        from docling.datamodel.base_models import ConversionStatus, DocumentStream  # ty: ignore[unresolved-import]

        # Docling auto-detects format from bytes; the hint becomes the source name
        # (used for logging + extension-based fallback when sniffing is ambiguous).
        source_name = f"document.{format_hint}" if format_hint else "document"
        stream = DocumentStream(name=source_name, stream=io.BytesIO(data))
        result = converter.convert(
            stream,
            raises_on_error=False,
            max_num_pages=self._max_num_pages,
            max_file_size=self._max_file_size_bytes,
        )
        pages = _processed_page_count(result)
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            return (
                {},
                pages,
                ExtractItemError(
                    code=ErrorCode.INFERENCE_ERROR.value,
                    message=_ERR_CONVERSION_FAILED,
                ),
            )
        doc = result.document
        try:
            payload = {
                "text": doc.export_to_text(),
                "markdown": doc.export_to_markdown(),
                "document": doc.export_to_dict(),
            }
        except Exception as exc:  # noqa: BLE001 - preserve processed pages on export failure
            logger.warning("Docling export raised error_type=%s", type(exc).__name__)
            return (
                {},
                pages,
                ExtractItemError(
                    code=ErrorCode.INFERENCE_ERROR.value,
                    message=_ERR_EXPORT_FAILED,
                ),
            )
        return payload, pages, None

    def _make_converter(self, *, ocr_enabled: bool) -> Any:
        """Build a fresh DocumentConverter. Callers should usually go through _get_converter() for caching.

        Threads self._device through Docling's AcceleratorOptions so layout, table,
        and OCR models actually run on the configured device. Without this, Docling
        silently defaults to CPU regardless of how SIE was launched.
        """
        from docling.document_converter import DocumentConverter  # ty: ignore[unresolved-import]

        accelerator_options = self._build_accelerator_options()

        from docling.datamodel.base_models import InputFormat  # ty: ignore[unresolved-import]
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # ty: ignore[unresolved-import]
        from docling.document_converter import ImageFormatOption, PdfFormatOption  # ty: ignore[unresolved-import]

        # Pass do_ocr explicitly on both paths. Docling's PdfPipelineOptions defaults
        # do_ocr=True, so an unset default would silently OCR every PDF and make the
        # `ocr` profile a no-op vs. the default profile.
        pdf_kwargs: dict[str, Any] = {
            "do_ocr": ocr_enabled,
            "document_timeout": self._document_timeout_s,
        }
        artifacts_path = self._artifacts_path or self._package_artifact_root
        if artifacts_path is not None:
            pdf_kwargs["artifacts_path"] = artifacts_path
        if accelerator_options is not None:
            pdf_kwargs["accelerator_options"] = accelerator_options
        pdf_opts = PdfPipelineOptions(**pdf_kwargs)
        # Reuse the same PdfPipelineOptions for IMAGE input: Docling's image
        # pipeline shares the layout/OCR model stack with the PDF pipeline,
        # and re-using the option object keeps device/ocr settings in lock-step.
        # ImageFormatOption attaches the ImageDocumentBackend so Docling doesn't
        # auto-correct + deprecation-warn on every image request.
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_opts),
            }
        )

    def _build_accelerator_options(self) -> Any:
        """Translate self._device into a Docling AcceleratorOptions, or None."""
        if not self._device:
            return None
        from docling.datamodel.accelerator_options import AcceleratorOptions  # ty: ignore[unresolved-import]

        try:
            return AcceleratorOptions(device=str(self._device))
        except Exception as e:  # noqa: BLE001 - pydantic validation; fall back to auto
            logger.warning(
                "Docling: invalid device %r, falling back to 'auto' (%s)",
                self._device,
                e,
            )
            try:
                return AcceleratorOptions(device="auto")
            except Exception:
                logger.exception("Docling: failed to build AcceleratorOptions even with 'auto'")
                return None
