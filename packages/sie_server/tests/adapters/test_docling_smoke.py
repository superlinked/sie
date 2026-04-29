from __future__ import annotations

import io
from collections.abc import Callable

import pytest

pytest.importorskip("docling")

from sie_server.adapters.docling import DoclingAdapter
from sie_server.types.inputs import Item


@pytest.fixture(scope="module")
def loaded_adapter() -> DoclingAdapter:
    adapter = DoclingAdapter()
    adapter.load("cpu")
    return adapter


def _make_pdf_bytes() -> bytes:
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas  # ty: ignore[unresolved-import]

    _ = reportlab
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf)
    pdf.drawString(100, 750, "Smoke test heading")
    pdf.drawString(100, 720, "Hello from reportlab.")
    pdf.save()
    return buf.getvalue()


def _make_docx_bytes() -> bytes:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("Smoke test heading", level=1)
    document.add_paragraph("Hello from python-docx.")
    buf = io.BytesIO()
    document.save(buf)
    return buf.getvalue()


def _make_html_bytes() -> bytes:
    return b"<html><body><h1>Smoke test heading</h1><p>Hello from HTML.</p></body></html>"


@pytest.mark.parametrize(
    ("format_hint", "maker"),
    [
        ("pdf", _make_pdf_bytes),
        ("docx", _make_docx_bytes),
        ("html", _make_html_bytes),
    ],
)
def test_extract_real_document(loaded_adapter: DoclingAdapter, format_hint: str, maker: Callable[[], bytes]) -> None:
    data = maker()
    out = loaded_adapter.extract([Item(document={"data": data, "format": format_hint})])

    assert out.batch_size == 1
    assert out.data is not None
    item = out.data[0]
    assert "error" not in item, f"adapter reported error: {item.get('error')}"
    assert "Smoke test heading" in item["text"] or "Smoke test heading" in item["markdown"]
    assert "document" in item
