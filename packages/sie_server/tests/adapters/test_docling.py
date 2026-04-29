from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters.docling import DoclingAdapter
from sie_server.types.inputs import Item


def _make_adapter(*, ocr_factory: Any = None) -> tuple[DoclingAdapter, MagicMock]:
    """Build a loaded adapter whose `_make_converter` returns mocks.

    Returns the adapter and the patched factory so tests can inspect calls.
    Each invocation of the factory yields a *new* MagicMock-backed converter
    (mirroring the per-task instantiation contract documented on the adapter).
    """
    adapter = DoclingAdapter()
    adapter._loaded = True

    factory = MagicMock(name="_make_converter")

    def _new_converter(*, ocr_enabled: bool) -> MagicMock:
        if ocr_enabled and ocr_factory is not None:
            return ocr_factory(ocr_enabled=ocr_enabled)
        tag = "ocr" if ocr_enabled else "default"
        return _stub_converter(tag)

    factory.side_effect = _new_converter
    adapter._make_converter = factory  # type: ignore[method-assign]
    return adapter, factory


def _stub_converter(tag: str) -> MagicMock:
    """Return a MagicMock that mimics DocumentConverter.convert() return shape."""
    converter = MagicMock(name=f"DocumentConverter[{tag}]")

    def _convert(stream: Any) -> MagicMock:
        result = MagicMock()
        result.document.export_to_text.return_value = f"text:{tag}"
        result.document.export_to_markdown.return_value = f"# md:{tag}"
        result.document.export_to_dict.return_value = {"name": stream.name}
        return result

    converter.convert.side_effect = _convert
    return converter


class TestDoclingExtract:
    def test_returns_text_markdown_and_document(self) -> None:
        adapter, _ = _make_adapter()

        out = adapter.extract([Item(document={"data": b"%PDF-1.4 fake", "format": "pdf"})])

        assert out.batch_size == 1
        assert out.entities == [[]]
        assert out.data is not None
        assert out.data[0]["text"] == "text:default"
        assert out.data[0]["markdown"] == "# md:default"
        assert out.data[0]["document"] == {"name": "document.pdf"}

    def test_format_hint_drives_stream_name(self) -> None:
        adapter, _ = _make_adapter()

        out = adapter.extract([Item(document={"data": b"<html></html>", "format": "html"})])

        assert out.data is not None
        assert out.data[0]["document"] == {"name": "document.html"}

    def test_missing_format_falls_back_to_generic_name(self) -> None:
        adapter, _ = _make_adapter()

        out = adapter.extract([Item(document={"data": b"raw"})])

        assert out.data is not None
        assert out.data[0]["document"] == {"name": "document"}

    def test_non_document_item_yields_per_item_error(self) -> None:
        adapter, factory = _make_adapter()

        out = adapter.extract([Item(text="just text, no document")])

        assert out.data is not None
        assert "error" in out.data[0]
        assert "document" in out.data[0]["error"].lower()
        # No converter is constructed for a malformed item
        factory.assert_not_called()

    def test_per_item_failure_does_not_poison_batch(self) -> None:
        adapter = DoclingAdapter()
        adapter._loaded = True

        good_converter = MagicMock(name="DocumentConverter[good]")

        def _good_convert(stream: Any) -> MagicMock:
            r = MagicMock()
            r.document.export_to_text.return_value = "text:good"
            r.document.export_to_markdown.return_value = "# md:good"
            r.document.export_to_dict.return_value = {"name": stream.name}
            return r

        good_converter.convert.side_effect = _good_convert

        bad_converter = MagicMock(name="DocumentConverter[bad]")
        bad_converter.convert.side_effect = ValueError("corrupt PDF")

        # First call returns the good converter, second call returns the bad one.
        produced: list[MagicMock] = [good_converter, bad_converter]
        adapter._make_converter = MagicMock(side_effect=lambda **_: produced.pop(0))  # type: ignore[method-assign]

        out = adapter.extract(
            [
                Item(document={"data": b"a", "format": "pdf"}),
                Item(document={"data": b"b", "format": "pdf"}),
            ]
        )

        assert out.data is not None
        # Either order is valid because of the thread pool — assert both slots
        # individually.
        statuses = sorted(d.get("error", "ok") for d in out.data)
        assert statuses == ["corrupt PDF", "ok"]

    def test_extract_before_load_raises(self) -> None:
        adapter = DoclingAdapter()
        with pytest.raises(RuntimeError, match="load"):
            adapter.extract([Item(document={"data": b"x"})])

    def test_ocr_opt_in_passes_flag_to_factory(self) -> None:
        adapter, factory = _make_adapter()

        adapter.extract([Item(document={"data": b"x", "format": "pdf"})], options={"ocr": True})
        adapter.extract([Item(document={"data": b"x", "format": "pdf"})], options={"ocr": True})

        assert factory.call_args_list == [
            ({"ocr_enabled": True},),
            ({"ocr_enabled": True},),
        ] or all(call.kwargs == {"ocr_enabled": True} for call in factory.call_args_list)

    def test_ocr_default_off_passes_false(self) -> None:
        adapter, factory = _make_adapter()

        adapter.extract([Item(document={"data": b"x", "format": "pdf"})])

        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": False}

    def test_batch_constructs_converter_per_item(self) -> None:
        adapter, factory = _make_adapter()

        items = [Item(document={"data": b"x", "format": "pdf"}) for _ in range(3)]
        out = adapter.extract(items)

        assert out.batch_size == 3
        # One fresh converter per item — no sharing across the batch.
        assert factory.call_count == 3


class TestDoclingSpec:
    def test_capabilities(self) -> None:
        adapter = DoclingAdapter()
        assert adapter.capabilities.inputs == ["document"]
        assert adapter.capabilities.outputs == ["json"]

    def test_unload_is_no_op(self) -> None:
        # No converter state to clear — package-backed adapter holds no weights.
        adapter = DoclingAdapter()
        adapter.unload()  # must not raise


class TestDoclingMakeConverter:
    def test_make_converter_no_ocr_returns_default(self) -> None:
        adapter = DoclingAdapter()

        with patch("docling.document_converter.DocumentConverter") as mock_cls:
            adapter._make_converter(ocr_enabled=False)

        # No format_options when OCR is off
        mock_cls.assert_called_once_with()

    def test_make_converter_ocr_passes_pdf_pipeline_options(self) -> None:
        adapter = DoclingAdapter()

        with (
            patch("docling.document_converter.DocumentConverter") as mock_cls,
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.document_converter.PdfFormatOption") as mock_fmt_opt,
        ):
            adapter._make_converter(ocr_enabled=True)

        mock_opts.assert_called_once_with(do_ocr=True)
        mock_fmt_opt.assert_called_once()
        mock_cls.assert_called_once()
        assert "format_options" in mock_cls.call_args.kwargs
