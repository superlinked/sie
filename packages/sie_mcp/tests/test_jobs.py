import base64
from collections.abc import Callable
from typing import Any

import pytest
from sie_mcp import jobs
from sie_mcp.jobs import DocsToMarkdownError


class _FakeClient:
    """Records extract() calls and returns a canned ExtractResult-shaped dict."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload
        self.calls: list[dict[str, Any]] = []

    async def extract(self, model: str, items: Any, **kwargs: Any) -> Any:
        self.calls.append({"model": model, "items": items, **kwargs})
        return {"data": self._payload}


class _VLOCRFakeClient:
    """Docling (dict item) returns ``{"data": ...}``; VL-OCR (list of items) returns
    a ``list[ExtractResult]`` with the same ``mineru_text`` entity per page.
    """

    def __init__(self, *, docling: Any = None, vl_text: str = "") -> None:
        self._docling = docling
        self._vl_text = vl_text
        self.calls: list[dict[str, Any]] = []

    async def extract(self, model: str, items: Any, **kwargs: Any) -> Any:
        self.calls.append({"model": model, "items": items, **kwargs})
        if isinstance(items, list):
            return [{"entities": [{"text": self._vl_text, "label": "mineru_text", "score": 1.0}]} for _ in items]
        return {"data": self._docling}


async def test_returns_markdown_and_metadata() -> None:
    client = _FakeClient({"markdown": "# Title\n\nbody", "document": {"pages": {"1": {}}}})

    result = await jobs.docs_to_markdown(client, document_base64="", filename="report.pdf")

    assert result["markdown"] == "# Title\n\nbody"
    assert result["metadata"]["source_pages"] == 1
    assert result["metadata"]["markdown_tokens_estimated"] is True
    assert result["metadata"]["markdown_chars"] == len("# Title\n\nbody")
    # Committed figures are wired in and trace back to the benchmark run.
    reduction = result["metadata"]["token_reduction"]
    assert reduction["source"] == "committed benchmark run 20260610T144234Z"
    assert reduction["issue"] is None
    assert reduction["run"] == "20260610T144234Z"
    assert reduction["blended_reduction_pct"] == {"claude-opus-4-8": 82.9, "claude-sonnet-4-6": 86.5}
    assert reduction["per_file_type_reduction_pct"] == {"min": 63.3, "max": 94.8}
    assert "count_tokens" in reduction["method"]


async def test_token_reduction_is_isolated_from_committed_figures() -> None:
    """Mutating one response's metadata must not corrupt the committed benchmark."""
    client = _FakeClient({"markdown": "x"})

    first = await jobs.docs_to_markdown(client, document_base64="", filename="a.pdf")
    first["metadata"]["token_reduction"]["blended_reduction_pct"]["claude-opus-4-8"] = 0.0

    second = await jobs.docs_to_markdown(client, document_base64="", filename="b.pdf")

    assert second["metadata"]["token_reduction"]["blended_reduction_pct"]["claude-opus-4-8"] == 82.9


async def test_derives_format_hint_from_filename() -> None:
    client = _FakeClient({"markdown": "x"})

    await jobs.docs_to_markdown(client, document_base64="", filename="report.pdf")

    assert client.calls[0]["items"]["document"]["format"] == "pdf"
    assert client.calls[0]["options"] is None  # no OCR by default


async def test_passes_ocr_option_when_requested() -> None:
    # ocr only enables Docling's built-in OCR under engine="docling"; under "auto"
    # the VL-OCR fallback owns scans (see test_ocr_routing).
    client = _FakeClient({"markdown": "x"})

    await jobs.docs_to_markdown(client, document_base64="", filename="scan.pdf", ocr=True, engine="docling")

    assert client.calls[0]["options"] == {"ocr": True}


async def test_handles_data_returned_as_list() -> None:
    client = _FakeClient([{"markdown": "from list"}])

    result = await jobs.docs_to_markdown(client, document_base64="", filename="a.docx")

    assert result["markdown"] == "from list"


async def test_raises_on_error_payload() -> None:
    client = _FakeClient({"error": "unsupported format"})

    with pytest.raises(DocsToMarkdownError):
        await jobs.docs_to_markdown(client, document_base64="", filename="a.bin")


async def test_rejects_invalid_base64() -> None:
    client = _FakeClient({"markdown": "x"})

    with pytest.raises(DocsToMarkdownError):
        await jobs.docs_to_markdown(client, document_base64="!!not base64!!", filename="a.pdf")


async def test_rejects_oversize_document() -> None:
    client = _FakeClient({"markdown": "x"})
    oversize = base64.b64encode(b"a" * 1024).decode()

    with pytest.raises(DocsToMarkdownError, match="exceeds"):
        await jobs.docs_to_markdown(client, document_base64=oversize, filename="a.pdf", max_document_bytes=512)
    # Rejected at the edge — no cluster call made.
    assert client.calls == []


async def test_metadata_reports_docling_engine() -> None:
    client = _FakeClient({"markdown": "# Title\n\nbody", "document": {"pages": {"1": {}}}})

    result = await jobs.docs_to_markdown(client, document_base64="", filename="report.pdf")

    assert result["metadata"]["engine"] == "docling"


async def test_engine_vl_ocr_renders_pdf_and_stitches_pages(make_pdf: Callable[[int], bytes]) -> None:
    pdf_b64 = base64.b64encode(make_pdf(2)).decode()
    client = _VLOCRFakeClient(vl_text="scanned page text")

    result = await jobs.docs_to_markdown(
        client,
        document_base64=pdf_b64,
        filename="scan.pdf",
        engine="vl-ocr",
        vlocr_model="opendatalab/MinerU2.5-Pro-2604-1.2B",
    )

    assert result["markdown"] == "scanned page text\n\nscanned page text"
    assert result["metadata"]["engine"] == "vl-ocr"
    assert result["metadata"]["source_pages"] == 2
    # One batched VL-OCR call, two image items (one per page), routed to the VL-OCR model.
    assert client.calls[0]["model"] == "opendatalab/MinerU2.5-Pro-2604-1.2B"
    assert len(client.calls[0]["items"]) == 2
