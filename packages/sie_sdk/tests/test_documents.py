from pathlib import Path

import pytest
from sie_sdk.documents import (
    convert_item_document,
    infer_document_format,
    to_document_bytes,
)


class TestInferDocumentFormat:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("report.pdf", "pdf"),
            ("REPORT.PDF", "pdf"),
            ("notes.docx", "docx"),
            ("page.html", "html"),
            ("page.htm", "html"),
            ("readme.md", "md"),
            ("data.csv", "csv"),
        ],
    )
    def test_known_suffixes(self, filename: str, expected: str) -> None:
        assert infer_document_format(filename) == expected
        assert infer_document_format(Path(filename)) == expected

    def test_unknown_suffix_returns_none(self) -> None:
        assert infer_document_format("blob.bin") is None
        assert infer_document_format("noextension") is None


class TestToDocumentBytes:
    def test_bytes_passthrough(self) -> None:
        payload = b"%PDF-1.4 fake"
        data, fmt = to_document_bytes(payload)
        assert data is payload
        assert fmt is None

    def test_path_input(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        data, fmt = to_document_bytes(path)
        assert data == b"%PDF-1.4 fake"
        assert fmt == "pdf"

    def test_string_path_input(self, tmp_path: Path) -> None:
        path = tmp_path / "page.html"
        path.write_bytes(b"<html></html>")
        data, fmt = to_document_bytes(str(path))
        assert data == b"<html></html>"
        assert fmt == "html"

    def test_unknown_suffix_returns_none_format(self, tmp_path: Path) -> None:
        path = tmp_path / "blob.bin"
        path.write_bytes(b"\x00\x01")
        _, fmt = to_document_bytes(path)
        assert fmt is None

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            to_document_bytes(tmp_path / "nope.pdf")

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError):
            to_document_bytes(123)  # type: ignore[arg-type]


class TestConvertItemDocument:
    def test_no_document_field_returns_unchanged(self) -> None:
        item = {"text": "hi"}
        assert convert_item_document(item) is item
        assert item == {"text": "hi"}

    def test_none_document_returns_unchanged(self) -> None:
        item = {"document": None}
        result = convert_item_document(item)
        assert result["document"] is None

    def test_bytes_input_normalised(self) -> None:
        item = {"document": b"%PDF-1.4"}
        convert_item_document(item)
        assert item["document"] == {"data": b"%PDF-1.4", "format": None}

    def test_path_input_infers_format(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.4")
        item = {"document": path}
        convert_item_document(item)
        assert item["document"] == {"data": b"%PDF-1.4", "format": "pdf"}

    def test_dict_input_with_explicit_format_wins(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.4")
        item = {"document": {"data": path, "format": "application/pdf"}}
        convert_item_document(item)
        assert item["document"] == {"data": b"%PDF-1.4", "format": "application/pdf"}

    def test_dict_input_with_bytes_passthrough(self) -> None:
        item = {"document": {"data": b"%PDF-1.4"}}
        convert_item_document(item)
        assert item["document"] == {"data": b"%PDF-1.4", "format": None}

    def test_explicit_none_format_is_preserved_over_inferred(self, tmp_path: Path) -> None:
        # Caller passes format=None to mean "no hint, don't infer from .pdf suffix".
        # Presence check must win over truthiness so the inferred 'pdf' is not silently used.
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.4")
        item = {"document": {"data": path, "format": None}}
        convert_item_document(item)
        assert item["document"] == {"data": b"%PDF-1.4", "format": None}

    def test_explicit_empty_format_is_preserved_over_inferred(self, tmp_path: Path) -> None:
        path = tmp_path / "doc.pdf"
        path.write_bytes(b"%PDF-1.4")
        item = {"document": {"data": path, "format": ""}}
        convert_item_document(item)
        assert item["document"] == {"data": b"%PDF-1.4", "format": ""}
