from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("docling")

from sie_server.adapters.docling.adapter import DoclingAdapter
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
        pytest.param("pdf", _make_pdf_bytes, marks=pytest.mark.model),
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


_PROSE = (
    "Alice was beginning to get very tired of sitting by her sister "
    "on the bank and of having nothing to do once or twice she had "
    "peeped into the book her sister was reading"
)

# Byte-identical to the catalog acceptance fixture. This compact,
# wide line caught the English-only PP-OCRv3 detector dropping every word after
# "SIE" while the larger prose fixture below still passed.
_COMPACT_OCR_PNG_B64 = (
    "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"
    "CUCU9jZ9MtgcKlsBIwAAAABJRU5ErkJggg=="
)
_COMPACT_OCR_PNG = base64.b64decode(_COMPACT_OCR_PNG_B64, validate=True)


def _make_prose_image_bytes() -> bytes:
    pil = pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageFont  # ty: ignore[unresolved-import]

    _ = pil
    image = Image.new("RGB", (1400, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    words = _PROSE.split()
    for row in range(0, len(words), 8):
        draw.text((40, 40 + (row // 8) * 70), " ".join(words[row : row + 8]), fill="black", font=font)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def artifact_backed_adapter() -> DoclingAdapter:
    yaml = pytest.importorskip("yaml")
    model_yaml = Path(__file__).resolve().parents[2] / "models" / "docling.yaml"
    pinned = yaml.safe_load(model_yaml.read_text())

    adapter = DoclingAdapter(
        model_name_or_path=pinned["hf_id"],
        revision=pinned["hf_revision"],
    )
    adapter.load("cpu")
    return adapter


@pytest.mark.integration
def test_served_ocr_recognizes_compact_catalog_fixture(artifact_backed_adapter: DoclingAdapter) -> None:
    assert hashlib.sha256(_COMPACT_OCR_PNG).hexdigest() == (
        "a9c82ded9169cdb69bbf3a2096e89a662cf93cb03dfd2cc01acf44acfc42cb9a"
    )
    out = artifact_backed_adapter.extract(
        [Item(images=[{"data": _COMPACT_OCR_PNG, "format": "png"}])],
        options={"ocr": True},
    )

    assert out.errors is None
    assert out.data is not None
    item = out.data[0]
    assert "error" not in item, f"adapter reported error: {item.get('error')}"
    words = {
        "".join(character for character in word if character.isalnum()) for word in item["markdown"].casefold().split()
    }
    assert {"sie", "catalog", "ready"} <= words, item["markdown"]
    assert out.pages == [1]


@pytest.mark.integration
def test_served_ocr_output_keeps_word_boundaries(artifact_backed_adapter: DoclingAdapter) -> None:
    """End-to-end sanity of the artifact-backed served OCR path.

    This exercises the real pinned artifact revision end to end: it proves the
    revision resolves a usable RapidOCR set on the served path and returns
    segmented text, which a bare adapter on docling's own cache cannot show.

    SCOPE — measured, not assumed. This floor does NOT discriminate the
    recogniser language. Mutating the adapter to ``lang=["ch"]`` provably
    reaches a different recogniser (``ch_PP-OCRv4_rec_mobile.onnx`` instead of
    ``en_PP-OCRv4_rec_mobile.onnx``) yet yields BYTE-IDENTICAL text here
    (density 0.2741 both ways): the Chinese set reads clean, large, synthetic
    renders of Latin script perfectly. Degrading the fixture does not rescue
    the separation — across six render variants the two languages are
    non-monotonic, and on a gray/noisy variant the Chinese set scored HIGHER
    (0.2296 vs 0.1290). Threshold-fitting a synthetic image would buy a flaky
    test, not a guard.

    The unit cases in ``test_docling.py::TestDoclingMakeConverter`` hold the
    language pin by asserting ``ocr_options.lang == ["en"]``. This served-path
    smoke only checks that the pinned artifact returns segmented text.

    So: keep this as a served-path smoke over the pinned revision. Do not read
    it as evidence that the language is correct.
    """
    out = artifact_backed_adapter.extract(
        [Item(images=[{"data": _make_prose_image_bytes(), "format": "png"}])],
        options={"ocr": True},
    )

    assert out.data is not None
    assert out.errors is None, f"adapter reported errors: {out.errors}"
    item = out.data[0]
    text = item["text"]
    alpha = sum(1 for ch in text if ch.isalpha())
    whitespace = sum(1 for ch in text if ch.isspace())
    assert alpha > 40, f"OCR recognised too little text to judge: {text!r}"
    density = whitespace / alpha
    assert density > 0.10, (
        f"served OCR output is word-joined (whitespace/alpha = {density:.3f}); "
        f"the pinned artifact revision resolves no recogniser able to segment "
        f"this page at all. This is a catastrophic-failure floor, not a "
        f"language check — see the docstring: {text!r}"
    )
