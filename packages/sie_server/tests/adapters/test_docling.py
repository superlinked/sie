from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from docling.datamodel.base_models import ConversionStatus
from sie_server.adapters.docling.adapter import DoclingAdapter
from sie_server.types.inputs import Item


def _make_adapter(*, ocr_factory: Any = None) -> tuple[DoclingAdapter, MagicMock]:
    """Build a loaded adapter whose `_make_converter` returns mocks.

    Returns the adapter and the patched factory so tests can inspect calls.
    Each invocation of the factory yields a *new* MagicMock-backed converter;
    the adapter caches them via `_get_converter` so within a single
    `ocr_enabled` value only the first extract triggers a build.
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

    def _convert(stream: Any, **_kwargs: Any) -> MagicMock:
        result = MagicMock()
        result.status = ConversionStatus.SUCCESS
        result.document.export_to_text.return_value = f"text:{tag}"
        result.document.export_to_markdown.return_value = f"# md:{tag}"
        result.document.export_to_dict.return_value = {"name": stream.name}
        result.document.num_pages.return_value = 0
        result.pages = []
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

    def test_passes_supported_conversion_limits(self) -> None:
        adapter, _ = _make_adapter()

        adapter.extract([Item(document={"data": b"%PDF", "format": "pdf"})])

        converter = adapter._converters[False]
        kwargs = converter.convert.call_args.kwargs
        assert kwargs == {
            "raises_on_error": False,
            "max_num_pages": 100,
            "max_file_size": 16 * 1024 * 1024,
        }

    def test_surfaces_real_page_count_for_metering(self) -> None:
        # §7 parse dimension: the adapter reads the processed ConversionResult pages
        # and surfaces them on ExtractOutput.pages so the result
        # seam can bill "$ per 1k pages".
        adapter = DoclingAdapter()
        adapter._loaded = True

        def _paged_converter(*, ocr_enabled: bool) -> MagicMock:
            _ = ocr_enabled
            converter = MagicMock(name="DocumentConverter[paged]")

            def _convert(stream: Any, **_kwargs: Any) -> MagicMock:
                result = MagicMock()
                result.status = ConversionStatus.SUCCESS
                result.document.export_to_text.return_value = "text"
                result.document.export_to_markdown.return_value = "# md"
                result.document.export_to_dict.return_value = {"name": stream.name}
                result.document.num_pages.return_value = 3
                result.pages = [object(), object(), object()]
                return result

            converter.convert.side_effect = _convert
            return converter

        adapter._make_converter = MagicMock(side_effect=_paged_converter)  # type: ignore[method-assign]

        out = adapter.extract([Item(document={"data": b"%PDF-1.4 fake", "format": "pdf"})])

        assert out.pages == [3]

    def test_partial_conversion_uses_processed_pages_not_document_pages(self) -> None:
        adapter = DoclingAdapter()
        adapter._loaded = True
        converter = MagicMock()
        document = MagicMock()
        document.export_to_text.return_value = "text"
        document.export_to_markdown.return_value = "# md"
        document.export_to_dict.return_value = {}
        document.num_pages.return_value = 5
        converter.convert.return_value = SimpleNamespace(
            status=ConversionStatus.PARTIAL_SUCCESS,
            document=document,
            pages=[object(), object()],
        )
        adapter._make_converter = MagicMock(return_value=converter)  # type: ignore[method-assign]

        out = adapter.extract([Item(document={"data": b"%PDF", "format": "pdf"})])

        assert out.pages == [2]
        assert out.data is not None
        assert out.data[0]["text"] == "text"
        assert out.errors is None

    def test_failed_conversion_retains_processed_pages(self) -> None:
        adapter = DoclingAdapter()
        adapter._loaded = True
        converter = MagicMock()
        document = MagicMock()
        converter.convert.return_value = SimpleNamespace(
            status=ConversionStatus.FAILURE,
            document=document,
            pages=[object(), object()],
        )
        adapter._make_converter = MagicMock(return_value=converter)  # type: ignore[method-assign]

        out = adapter.extract([Item(document={"data": b"%PDF", "format": "pdf"})])

        assert out.data == [{}]
        assert out.pages == [2]
        assert out.errors is not None
        assert out.errors[0] is not None
        assert out.errors[0].code == "INFERENCE_ERROR"
        assert out.errors[0].message == "Document conversion failed"
        document.export_to_text.assert_not_called()

    def test_export_failure_retains_pages_and_sanitizes_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        adapter = DoclingAdapter()
        adapter._loaded = True
        converter = MagicMock()
        document = MagicMock()
        document.export_to_text.side_effect = RuntimeError("customer-42 secret")
        converter.convert.return_value = SimpleNamespace(
            status=ConversionStatus.SUCCESS,
            document=document,
            pages=[object(), object(), object()],
        )
        adapter._make_converter = MagicMock(return_value=converter)  # type: ignore[method-assign]

        out = adapter.extract([Item(id="private-id", document={"data": b"%PDF", "format": "pdf"})])

        assert out.data == [{}]
        assert out.pages == [3]
        assert out.errors is not None
        assert out.errors[0] is not None
        assert out.errors[0].message == "Document export failed"
        assert "customer-42" not in caplog.text
        assert "private-id" not in caplog.text

    def test_empty_processed_pages_do_not_fall_back_to_document_page_count(self) -> None:
        adapter = DoclingAdapter()
        adapter._loaded = True
        converter = MagicMock()
        document = MagicMock()
        document.export_to_text.return_value = "text"
        document.export_to_markdown.return_value = "# md"
        document.export_to_dict.return_value = {}
        document.num_pages.return_value = 4
        converter.convert.return_value = SimpleNamespace(
            status=ConversionStatus.SUCCESS,
            document=document,
            pages=[],
        )
        adapter._make_converter = MagicMock(return_value=converter)  # type: ignore[method-assign]

        out = adapter.extract([Item(document={"data": b"%PDF", "format": "pdf"})])

        assert out.pages == [0]

    def test_image_input_emits_pages_without_generic_image_units(self) -> None:
        adapter, _ = _make_adapter()
        item = Item(images=[{"data": b"png", "format": "png"}])

        assert adapter.count_input_images([item]) is None

    def test_per_item_error_reports_zero_pages(self) -> None:
        # A per-item failure before parsing surfaces authoritative zero pages,
        # allowing settlement to release the admission reserve.
        adapter, _ = _make_adapter()

        out = adapter.extract([Item(text="no document here")])

        assert out.data == [{}]
        assert out.errors is not None
        assert out.errors[0] is not None
        assert out.errors[0].code == "INVALID_INPUT"
        assert out.pages == [0]

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

        assert out.data == [{}]
        assert out.errors is not None
        assert out.errors[0] is not None
        assert out.errors[0].code == "INVALID_INPUT"
        # No converter is constructed for a malformed item
        factory.assert_not_called()

    def test_image_only_item_routes_to_converter(self) -> None:
        adapter, factory = _make_adapter()

        out = adapter.extract([Item(images=[{"data": b"\x89PNG\r\n\x1a\nfake", "format": "png"}])])

        assert out.data is not None
        assert out.data[0]["text"] == "text:default"
        # Stream name should reflect the image format hint
        assert out.data[0]["document"] == {"name": "document.png"}
        factory.assert_called_once_with(ocr_enabled=False)

    def test_image_without_format_defaults_to_png_name(self) -> None:
        adapter, _ = _make_adapter()

        out = adapter.extract([Item(images=[{"data": b"\x89PNG"}])])

        assert out.data is not None
        # When no format is supplied, the adapter defaults to "png" so Docling
        # has a usable extension hint.
        assert out.data[0]["document"] == {"name": "document.png"}

    def test_document_wins_when_both_document_and_images_present(self) -> None:
        adapter, _ = _make_adapter()

        out = adapter.extract(
            [
                Item(
                    document={"data": b"%PDF-1.4 real", "format": "pdf"},
                    images=[
                        {"data": b"\x89PNG ignored-1", "format": "png"},
                        {"data": b"\x89PNG ignored-2", "format": "png"},
                    ],
                )
            ]
        )

        assert out.data is not None
        # Document path uses ``document.pdf`` stream name; image path would
        # have produced ``document.png``.
        assert out.data[0]["document"] == {"name": "document.pdf"}

    def test_neither_document_nor_image_yields_per_item_error(self) -> None:
        adapter, factory = _make_adapter()

        out = adapter.extract([Item(text="just text")])

        assert out.data == [{}]
        assert out.errors is not None
        assert out.errors[0] is not None
        assert out.errors[0].code == "INVALID_INPUT"
        assert out.errors[0].message == "Document or image input is required"
        factory.assert_not_called()

    def test_empty_images_list_yields_per_item_error(self) -> None:
        adapter, factory = _make_adapter()

        out = adapter.extract([Item(images=[])])

        assert out.data == [{}]
        assert out.errors is not None
        assert out.errors[0] is not None
        assert out.errors[0].code == "INVALID_INPUT"
        assert out.errors[0].message == "Document or image input is required"
        factory.assert_not_called()

    def test_multiple_images_yield_typed_invalid_input(self) -> None:
        adapter, factory = _make_adapter()

        out = adapter.extract(
            [
                Item(
                    images=[
                        {"data": b"page-1", "format": "png"},
                        {"data": b"page-2", "format": "png"},
                    ]
                )
            ]
        )

        assert out.data == [{}]
        assert out.errors is not None
        assert out.errors[0] is not None
        assert out.errors[0].code == "INVALID_INPUT"
        assert out.errors[0].message == "Docling OCR requires exactly one image per item"
        factory.assert_not_called()

    def test_per_item_failure_does_not_poison_batch(self) -> None:
        adapter = DoclingAdapter()
        adapter._loaded = True

        converter = MagicMock(name="DocumentConverter[shared]")

        def _good_result(name: str, stream_name: str) -> MagicMock:
            r = MagicMock()
            r.status = ConversionStatus.SUCCESS
            r.document.export_to_text.return_value = f"text:{name}"
            r.document.export_to_markdown.return_value = f"# md:{name}"
            r.document.export_to_dict.return_value = {"name": stream_name}
            return r

        calls = {"n": 0}

        def _convert_side_effect(stream: Any, **_kwargs: Any) -> MagicMock:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("boom")
            return _good_result(f"doc{calls['n']}", stream.name)

        converter.convert.side_effect = _convert_side_effect
        factory = MagicMock(name="_make_converter", return_value=converter)
        adapter._make_converter = factory  # type: ignore[method-assign]

        out = adapter.extract(
            [
                Item(document={"data": b"a", "format": "pdf"}),
                Item(document={"data": b"b", "format": "pdf"}),
                Item(document={"data": b"c", "format": "pdf"}),
            ]
        )

        assert out.data is not None
        assert [d.get("text") for d in out.data] == ["text:doc1", None, "text:doc3"]
        assert out.errors is not None
        assert out.errors[0] is None
        assert out.errors[1] is not None
        assert out.errors[1].message == "Document conversion failed"
        assert out.errors[2] is None
        assert factory.call_count == 1  # converter built once, not three times

    def test_extract_before_load_raises(self) -> None:
        adapter = DoclingAdapter()
        with pytest.raises(RuntimeError, match="load"):
            adapter.extract([Item(document={"data": b"x"})])

    def test_ocr_opt_in_passes_flag_to_factory(self) -> None:
        adapter, factory = _make_adapter()

        adapter.extract([Item(document={"data": b"x", "format": "pdf"})], options={"ocr": True})
        adapter.extract([Item(document={"data": b"x", "format": "pdf"})], options={"ocr": True})

        # Cache means we built once even though we extracted twice.
        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": True}

    def test_ocr_default_off_passes_false(self) -> None:
        adapter, factory = _make_adapter()

        adapter.extract([Item(document={"data": b"x", "format": "pdf"})])

        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": False}

    def test_batch_reuses_cached_converter(self) -> None:
        adapter, factory = _make_adapter()

        items = [Item(document={"data": b"x", "format": "pdf"}) for _ in range(3)]
        out = adapter.extract(items)

        assert out.batch_size == 3
        # One cached converter shared across the whole batch.
        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": False}

    def test_extract_caches_converter_across_calls(self) -> None:
        adapter, factory = _make_adapter()
        adapter.extract([Item(document={"data": b"x", "format": "pdf"})])
        adapter.extract([Item(document={"data": b"y", "format": "pdf"})])
        assert factory.call_count == 1

    def test_extract_caches_per_ocr_key(self) -> None:
        adapter, factory = _make_adapter()
        adapter.extract([Item(document={"data": b"x", "format": "pdf"})])  # ocr=False
        adapter.extract([Item(document={"data": b"y", "format": "pdf"})], options={"ocr": True})  # ocr=True
        assert factory.call_count == 2
        adapter.extract([Item(document={"data": b"z", "format": "pdf"})])  # ocr=False (cached)
        assert factory.call_count == 2


class TestDoclingSpec:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_num_pages": 0},
            {"max_num_pages": True},
            {"max_num_pages": 1.5},
            {"max_file_size_bytes": -1},
            {"max_file_size_bytes": False},
            {"document_timeout_s": 0},
            {"document_timeout_s": float("nan")},
            {"document_timeout_s": float("inf")},
            {"document_timeout_s": True},
        ],
    )
    def test_conversion_limits_fail_closed(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="must be a positive"):
            DoclingAdapter(**kwargs)

    def test_capabilities(self) -> None:
        adapter = DoclingAdapter()
        assert adapter.capabilities.inputs == ["document", "image"]
        assert adapter.capabilities.outputs == ["json"]

    def test_unload_clears_converter_cache(self) -> None:
        adapter, _factory = _make_adapter()
        adapter.extract([Item(document={"data": b"x", "format": "pdf"})])
        assert adapter._converters  # populated
        assert adapter._loaded

        adapter.unload()  # must not raise
        assert adapter._converters == {}
        assert not adapter._loaded

        with pytest.raises(RuntimeError, match=r"load.* before extract"):
            adapter.extract([Item(document={"data": b"x", "format": "pdf"})])

    def test_staged_artifact_prewarm_fails_closed(self, tmp_path: Path) -> None:
        adapter = DoclingAdapter(package_artifact_root=tmp_path)
        adapter._make_converter = MagicMock(side_effect=RuntimeError("missing staged model"))  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="staged artifact initialization failed"):
            adapter.load("cpu")

        assert not adapter._loaded

    def test_staged_artifact_prewarm_exercises_ocr_converter(self, tmp_path: Path) -> None:
        adapter = DoclingAdapter(package_artifact_root=tmp_path)
        default_converter = _stub_converter("default")
        ocr_converter = _stub_converter("ocr")
        ocr_converter.convert.side_effect = FileNotFoundError("missing staged OCR model")
        adapter._make_converter = MagicMock(  # type: ignore[method-assign]
            side_effect=[default_converter, ocr_converter]
        )

        with pytest.raises(RuntimeError, match="staged artifact initialization failed"):
            adapter.load("cpu")

        assert default_converter.convert.call_count == 1
        assert ocr_converter.convert.call_count == 1
        assert not adapter._loaded

    def test_local_model_source_is_used_as_artifact_root(self, tmp_path: Path) -> None:
        adapter = DoclingAdapter(model_name_or_path=tmp_path)

        assert adapter._resolve_artifacts_path() == tmp_path.resolve()

    def test_hub_model_source_resolves_exact_revision(self, tmp_path: Path) -> None:
        adapter = DoclingAdapter(
            model_name_or_path="superlinked/docling-artifacts",
            revision="a" * 40,
        )

        with patch("huggingface_hub.snapshot_download", return_value=str(tmp_path)) as download:
            assert adapter._resolve_artifacts_path() == tmp_path.resolve()

        download.assert_called_once_with(
            repo_id="superlinked/docling-artifacts",
            revision="a" * 40,
        )

    def test_ordinary_model_source_prewarm_fails_closed(self, tmp_path: Path) -> None:
        adapter = DoclingAdapter(model_name_or_path=tmp_path)
        adapter._make_converter = MagicMock(  # type: ignore[method-assign]
            side_effect=RuntimeError("missing immutable model asset")
        )

        with pytest.raises(RuntimeError, match="staged artifact initialization failed"):
            adapter.load("cpu")

        assert not adapter._loaded

    def test_ordinary_and_package_artifact_sources_are_mutually_exclusive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            DoclingAdapter(model_name_or_path=tmp_path, package_artifact_root=tmp_path)


class TestDoclingMakeConverter:
    def test_make_converter_no_ocr_passes_do_ocr_false(self) -> None:
        # Docling's PdfPipelineOptions defaults do_ocr=True, so we must pass
        # do_ocr=False explicitly on the default path — otherwise the `ocr`
        # profile is a no-op vs the default profile.
        adapter = DoclingAdapter()
        assert adapter._device is None

        with (
            patch("docling.document_converter.DocumentConverter") as mock_cls,
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.document_converter.PdfFormatOption") as mock_fmt_opt,
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=False)

        mock_opts.assert_called_once_with(do_ocr=False, document_timeout=90.0)
        mock_fmt_opt.assert_called_once()
        mock_cls.assert_called_once()
        assert "format_options" in mock_cls.call_args.kwargs

    def test_make_converter_ocr_passes_pdf_pipeline_options(self) -> None:
        adapter = DoclingAdapter()

        with (
            patch("docling.document_converter.DocumentConverter") as mock_cls,
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.document_converter.PdfFormatOption") as mock_fmt_opt,
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=True)

        mock_opts.assert_called_once_with(do_ocr=True, document_timeout=90.0)
        mock_fmt_opt.assert_called_once()
        mock_cls.assert_called_once()
        assert "format_options" in mock_cls.call_args.kwargs

    def test_make_converter_passes_verified_staged_artifact_root(self, tmp_path: Path) -> None:
        adapter = DoclingAdapter(package_artifact_root=tmp_path)

        with (
            patch("docling.document_converter.DocumentConverter"),
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.document_converter.PdfFormatOption"),
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=False)

        assert mock_opts.call_args.kwargs["artifacts_path"] == tmp_path

    def test_make_converter_passes_resolved_ordinary_artifact_root(self, tmp_path: Path) -> None:
        adapter = DoclingAdapter(model_name_or_path=tmp_path)
        adapter._artifacts_path = adapter._resolve_artifacts_path()

        with (
            patch("docling.document_converter.DocumentConverter"),
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.document_converter.PdfFormatOption"),
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=True)

        assert mock_opts.call_args.kwargs["artifacts_path"] == tmp_path.resolve()

    def test_make_converter_no_ocr_uses_accelerator_options_when_device_set(self) -> None:
        adapter = DoclingAdapter()
        adapter._device = "cuda"

        with (
            patch("docling.document_converter.DocumentConverter") as mock_cls,
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.datamodel.accelerator_options.AcceleratorOptions") as mock_accel,
            patch("docling.document_converter.PdfFormatOption") as mock_fmt_opt,
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=False)

        mock_accel.assert_called_once_with(device="cuda")
        kwargs = mock_opts.call_args.kwargs
        assert kwargs["accelerator_options"] is mock_accel.return_value
        assert kwargs["do_ocr"] is False
        mock_fmt_opt.assert_called_once()
        mock_cls.assert_called_once()
        assert "format_options" in mock_cls.call_args.kwargs

    def test_make_converter_ocr_threads_accelerator_options(self) -> None:
        adapter = DoclingAdapter()
        adapter._device = "cuda:0"

        with (
            patch("docling.document_converter.DocumentConverter") as mock_cls,
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions") as mock_opts,
            patch("docling.datamodel.accelerator_options.AcceleratorOptions") as mock_accel,
            patch("docling.document_converter.PdfFormatOption"),
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=True)

        mock_accel.assert_called_once_with(device="cuda:0")
        kwargs = mock_opts.call_args.kwargs
        assert kwargs.get("do_ocr") is True
        assert kwargs.get("accelerator_options") is mock_accel.return_value
        mock_cls.assert_called_once()

    @pytest.mark.parametrize("device", ["cuda", "cuda:0", "mps", "cpu", "auto"])
    def test_make_converter_passes_known_device_strings_through(self, device: str) -> None:
        adapter = DoclingAdapter()
        adapter._device = device

        with (
            patch("docling.document_converter.DocumentConverter"),
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions"),
            patch("docling.datamodel.accelerator_options.AcceleratorOptions") as mock_accel,
            patch("docling.document_converter.PdfFormatOption"),
            patch("docling.document_converter.ImageFormatOption"),
        ):
            adapter._make_converter(ocr_enabled=True)

        mock_accel.assert_called_once_with(device=device)

    def test_make_converter_invalid_device_falls_back_to_auto(self) -> None:
        adapter = DoclingAdapter()
        adapter._device = "definitely-not-a-device"

        with (
            patch("docling.document_converter.DocumentConverter"),
            patch("docling.datamodel.pipeline_options.PdfPipelineOptions"),
            patch("docling.datamodel.accelerator_options.AcceleratorOptions") as mock_accel,
            patch("docling.document_converter.PdfFormatOption"),
            patch("docling.document_converter.ImageFormatOption"),
        ):
            mock_accel.side_effect = [ValueError("invalid device"), MagicMock()]
            adapter._make_converter(ocr_enabled=True)

        actual = [c.kwargs for c in mock_accel.call_args_list]
        assert actual == [
            {"device": "definitely-not-a-device"},
            {"device": "auto"},
        ]


class TestDoclingOcrPrecedence:
    """The DoclingAdapter only sees the merged ``options`` dict produced by
    ``resolve_runtime_options`` (api/options.py); it does not know about
    profiles. These tests pin down the contract that the merged dict drives
    ``ocr_enabled`` regardless of where the value came from.
    """

    def test_profile_runtime_ocr_true_makes_extract_ocr_default(self) -> None:
        """When profile resolution sets options={'ocr': True}, the adapter
        builds an OCR-enabled converter even with no per-request overrides.
        """
        adapter, factory = _make_adapter()

        # Simulates what the worker passes through after the 'ocr' profile is
        # resolved server-side: profile.runtime.ocr=true becomes options.ocr=true
        # before reaching the adapter.
        adapter.extract(
            [Item(document={"data": b"x", "format": "pdf"})],
            options={"ocr": True},
        )

        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": True}

    def test_request_ocr_false_overrides_profile_ocr_true(self) -> None:
        """Per-request override wins. ``resolve_runtime_options`` emits the
        already-merged dict, so the adapter sees options={'ocr': False}
        even if the profile would have defaulted ocr=True.
        """
        adapter, factory = _make_adapter()

        adapter.extract(
            [Item(document={"data": b"x", "format": "pdf"})],
            options={"ocr": False},
        )

        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": False}

    def test_request_ocr_true_works_without_profile_default(self) -> None:
        """Regression: per-request ocr=true keeps working when no profile
        default is set (i.e., the historical 'default' profile path).
        """
        adapter, factory = _make_adapter()

        adapter.extract(
            [Item(document={"data": b"x", "format": "pdf"})],
            options={"ocr": True},
        )

        assert factory.call_count == 1
        assert factory.call_args.kwargs == {"ocr_enabled": True}
