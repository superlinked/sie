import io
from collections.abc import Callable
from typing import Any

import pytest
from PIL import Image
from sie_mcp import ocr_routing
from sie_mcp.ocr_routing import DocsToMarkdownError
from sie_sdk import RequestError, SIEError


class _RoutingFakeClient:
    """Distinguishes Docling (dict item) from VL-OCR (list of items) calls.

    Docling calls return ``{"data": <docling_payload>}``; VL-OCR batch calls return
    a ``list[ExtractResult]`` with one ``entities`` list per page item (in order).
    Set ``vl_error`` to make the VL-OCR batch call raise (simulating a cluster fault).
    """

    def __init__(
        self,
        *,
        docling: Any = None,
        vl_pages: list[list[dict[str, Any]]] | None = None,
        vl_error: Exception | None = None,
        vl_page_errors: dict[int, Any] | None = None,
    ) -> None:
        self._docling = docling
        self._vl_pages = vl_pages or []
        self._vl_error = vl_error
        self._vl_page_errors = vl_page_errors or {}
        self.calls: list[dict[str, Any]] = []

    async def extract(self, model: str, items: Any, **kwargs: Any) -> Any:
        self.calls.append({"model": model, "items": items, **kwargs})
        if isinstance(items, list):
            if self._vl_error is not None:
                raise self._vl_error
            return [
                {"error": self._vl_page_errors[i]} if i in self._vl_page_errors else {"entities": self._vl_pages[i]}
                for i in range(len(items))
            ]
        return {"data": self._docling}


def _page(text: str) -> list[dict[str, Any]]:
    return [{"text": text, "label": "mineru_text", "score": 1.0}]


def _fake_render(n_pages: int) -> Callable[..., list[str]]:
    """An injectable renderer that returns ``n_pages`` opaque page sentinels."""

    def _render(_data: bytes, *, scale: float = 2.0) -> list[str]:
        return [f"page-image-{i}" for i in range(n_pages)]

    return _render


# --- entities_to_markdown ---------------------------------------------------


def test_entities_to_markdown_single_entity() -> None:
    result = {"entities": _page("# Heading\n\nbody")}
    assert ocr_routing.entities_to_markdown(result) == "# Heading\n\nbody"


def test_entities_to_markdown_concatenates_multiple_entities() -> None:
    result = {
        "entities": [
            {"text": "table md", "label": "mineru_table", "score": 1.0},
            {"text": "para md", "label": "mineru_text", "score": 1.0},
        ]
    }
    assert ocr_routing.entities_to_markdown(result) == "table md\n\npara md"


def test_entities_to_markdown_skips_blank_and_handles_missing() -> None:
    assert ocr_routing.entities_to_markdown({"entities": [{"text": "   ", "label": "x"}]}) == ""
    assert ocr_routing.entities_to_markdown({"entities": []}) == ""
    assert ocr_routing.entities_to_markdown({}) == ""
    assert ocr_routing.entities_to_markdown("not a dict") == ""


# --- renderability detection + rendering ------------------------------------


def test_is_renderable_by_magic_bytes(png_bytes: bytes, tiff_bytes: bytes) -> None:
    assert ocr_routing.is_renderable(b"%PDF-1.7\n...") is True
    assert ocr_routing.is_renderable(png_bytes) is True
    assert ocr_routing.is_renderable(b"\xff\xd8\xff\xe0jpeg") is True
    assert ocr_routing.is_renderable(tiff_bytes) is True  # scanned-doc TIFF
    assert ocr_routing.is_renderable(b"GIF89a....") is True
    assert ocr_routing.is_renderable(b"RIFF\x00\x00\x00\x00WEBPVP8 ") is True
    assert ocr_routing.is_renderable(b"PK\x03\x04 docx/zip") is False
    assert ocr_routing.is_renderable(b"") is False


def test_render_tiff_yields_single_image(tiff_bytes: bytes) -> None:
    images = ocr_routing.render_to_images(tiff_bytes)
    assert len(images) == 1
    assert images[0].mode == "RGB"


def test_render_caps_page_count(make_pdf: Callable[[int], bytes], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_routing, "_MAX_VLOCR_PAGES", 2)
    with pytest.raises(DocsToMarkdownError, match="capped at 2"):
        ocr_routing.render_to_images(make_pdf(3))


def test_render_wraps_image_open_failure() -> None:
    # Passes the BMP magic-byte gate but is not a decodable image.
    with pytest.raises(DocsToMarkdownError):
        ocr_routing.render_to_images(b"BMnot-a-real-bitmap")


def test_render_pdf_yields_one_rgb_image_per_page(make_pdf: Callable[[int], bytes]) -> None:
    images = ocr_routing.render_to_images(make_pdf(3))
    assert len(images) == 3
    assert all(isinstance(img, Image.Image) and img.mode == "RGB" for img in images)


def test_render_image_yields_single_image(png_bytes: bytes) -> None:
    images = ocr_routing.render_to_images(png_bytes)
    assert len(images) == 1
    assert images[0].mode == "RGB"


def test_render_rejects_non_renderable_format() -> None:
    with pytest.raises(DocsToMarkdownError):
        ocr_routing.render_to_images(b"PK\x03\x04 not a pdf")


def test_render_raises_on_corrupt_pdf() -> None:
    with pytest.raises(DocsToMarkdownError):
        ocr_routing.render_to_images(b"%PDF-garbage-not-a-real-pdf")


# --- route_to_markdown: explicit engines ------------------------------------


async def _route(client: _RoutingFakeClient, **kwargs: Any) -> ocr_routing.RouteResult:
    defaults: dict[str, Any] = {
        "data": b"%PDF-fake",
        "fmt": "pdf",
        "ocr": False,
        "engine": "auto",
        "docling_model": "docling",
        "vlocr_model": "mineru",
        "gpu": None,
    }
    defaults.update(kwargs)
    return await ocr_routing.route_to_markdown(client, **defaults)


async def test_engine_vl_ocr_stitches_pages_in_order() -> None:
    client = _RoutingFakeClient(vl_pages=[_page("PAGE ONE"), _page("PAGE TWO"), _page("PAGE THREE")])

    result = await _route(client, engine="vl-ocr", render=_fake_render(3))

    assert result.engine == "vl-ocr"
    assert result.pages == 3
    assert result.markdown == "PAGE ONE\n\nPAGE TWO\n\nPAGE THREE"
    # Single batched VL-OCR call, one image item per page, routed to the VL-OCR model.
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == "mineru"
    assert len(client.calls[0]["items"]) == 3


async def test_engine_docling_never_falls_back_even_when_empty() -> None:
    client = _RoutingFakeClient(docling={"markdown": ""}, vl_pages=[_page("recovered")])

    result = await _route(client, engine="docling", render=_fake_render(1))

    assert result.engine == "docling"
    assert result.markdown == ""
    assert len(client.calls) == 1  # no VL-OCR call


async def test_invalid_engine_raises() -> None:
    client = _RoutingFakeClient(docling={"markdown": "x"})
    with pytest.raises(DocsToMarkdownError):
        await _route(client, engine="nope")


# --- route_to_markdown: auto routing ----------------------------------------


async def test_auto_keeps_docling_for_born_digital() -> None:
    client = _RoutingFakeClient(
        docling={"markdown": "# Real digital page\n\nlots of extracted text", "document": {"pages": {"1": {}}}},
        vl_pages=[_page("should not be used")],
    )

    result = await _route(client, engine="auto", render=_fake_render(1))

    assert result.engine == "docling"
    assert result.markdown.startswith("# Real digital page")
    assert len(client.calls) == 1  # Docling only; no fallback


async def test_auto_falls_back_to_vl_ocr_for_scanned_pdf() -> None:
    client = _RoutingFakeClient(docling={"markdown": "   "}, vl_pages=[_page("OCR p1"), _page("OCR p2")])

    result = await _route(client, engine="auto", render=_fake_render(2))

    assert result.engine == "vl-ocr"
    assert result.markdown == "OCR p1\n\nOCR p2"
    assert result.pages == 2
    assert len(client.calls) == 2  # Docling first, then VL-OCR


async def test_auto_falls_back_when_docling_errors() -> None:
    client = _RoutingFakeClient(docling={"error": "no text layer"}, vl_pages=[_page("OCR recovered")])

    result = await _route(client, engine="auto", render=_fake_render(1))

    assert result.engine == "vl-ocr"
    assert result.markdown == "OCR recovered"


async def test_auto_no_fallback_for_non_renderable_format() -> None:
    # A near-empty docx (zip magic) is not rasterizable — Docling result stands.
    client = _RoutingFakeClient(docling={"markdown": ""}, vl_pages=[_page("unused")])

    result = await _route(client, data=b"PK\x03\x04docx", fmt="docx", engine="auto", render=_fake_render(1))

    assert result.engine == "docling"
    assert result.markdown == ""
    assert len(client.calls) == 1


async def test_auto_keeps_docling_when_vl_ocr_recovers_nothing() -> None:
    # Docling near-empty, but VL-OCR also yields nothing → don't discard Docling.
    client = _RoutingFakeClient(docling={"markdown": "tiny"}, vl_pages=[[]])

    result = await _route(client, engine="auto", render=_fake_render(1))

    assert result.engine == "docling"
    assert result.markdown == "tiny"


def _not_found() -> RequestError:
    return RequestError("no such model", code="MODEL_NOT_FOUND", status_code=404)


@pytest.mark.parametrize("docling_markdown", ["", "tiny"])
async def test_auto_raises_when_vl_ocr_model_is_not_found_on_text_poor_docling(docling_markdown: str) -> None:
    # Docling found (almost) no text, so this is a scan, and the VL-OCR model isn't
    # served. An empty "success" would hide that the document was never read, so the
    # error names the model, keeps the cluster's message and names the setting.
    client = _RoutingFakeClient(docling={"markdown": docling_markdown}, vl_error=_not_found())

    with pytest.raises(DocsToMarkdownError) as excinfo:
        await _route(client, engine="auto", render=_fake_render(1))
    message = str(excinfo.value)
    assert f"Docling returned only {len(docling_markdown)} characters" in message
    assert "'mineru'" in message
    assert "no such model" in message
    assert "SIE_MCP_VLOCR_MODEL" in message
    assert isinstance(excinfo.value.__cause__, RequestError)
    assert len(client.calls) == 2  # Docling, then the attempted VL-OCR fallback


async def test_auto_keeps_docling_errors_reason_when_vl_ocr_model_is_not_found() -> None:
    client = _RoutingFakeClient(docling={"error": "no text layer"}, vl_error=_not_found())

    with pytest.raises(DocsToMarkdownError, match=r"Docling failed \(.*no text layer.*SIE_MCP_VLOCR_MODEL"):
        await _route(client, engine="auto", render=_fake_render(1))


async def test_auto_vl_ocr_failure_other_than_not_found_has_no_setting_hint() -> None:
    # A cold start or a 5xx is not fixed by changing the model setting.
    client = _RoutingFakeClient(docling={"markdown": ""}, vl_error=SIEError("timed out while provisioning"))

    with pytest.raises(DocsToMarkdownError, match="timed out while provisioning") as excinfo:
        await _route(client, engine="auto", render=_fake_render(1))
    assert "SIE_MCP_VLOCR_MODEL" not in str(excinfo.value)


async def test_auto_render_failure_reports_the_render_error() -> None:
    def _too_many_pages(data: bytes, *, scale: float) -> list[Any]:
        raise DocsToMarkdownError("PDF has 150 pages; VL-OCR is capped at 100")

    client = _RoutingFakeClient(docling={"markdown": ""})

    with pytest.raises(DocsToMarkdownError, match="capped at 100") as excinfo:
        await _route(client, engine="auto", render=_too_many_pages)
    assert "SIE_MCP_VLOCR_MODEL" not in str(excinfo.value)


@pytest.mark.parametrize("docling_markdown", ["", "   "])
async def test_auto_raises_when_neither_engine_returns_text(docling_markdown: str) -> None:
    client = _RoutingFakeClient(docling={"markdown": docling_markdown}, vl_pages=[[]])

    with pytest.raises(DocsToMarkdownError, match="neither Docling nor the VL-OCR"):
        await _route(client, engine="auto", render=_fake_render(1))


async def test_auto_raises_when_docling_errors_and_vl_ocr_returns_no_text() -> None:
    client = _RoutingFakeClient(docling={"error": "no text layer"}, vl_pages=[[]])

    with pytest.raises(DocsToMarkdownError, match=r"Docling failed .*returned no text"):
        await _route(client, engine="auto", render=_fake_render(1))


async def test_vl_ocr_page_error_raises_instead_of_dropping_the_page() -> None:
    client = _RoutingFakeClient(
        docling={"markdown": ""},
        vl_pages=[_page("p1"), _page("unused"), _page("p3")],
        vl_page_errors={1: {"code": "INFERENCE_ERROR", "message": "out of memory"}},
    )

    with pytest.raises(DocsToMarkdownError, match=r"failed on 1 of 3 pages \(page 2: out of memory\)"):
        await _route(client, engine="vl-ocr", render=_fake_render(3))


async def test_engine_vl_ocr_raises_on_empty_result() -> None:
    client = _RoutingFakeClient(vl_pages=[[], []])

    with pytest.raises(DocsToMarkdownError, match=r"returned no text for 2 page"):
        await _route(client, engine="vl-ocr", render=_fake_render(2))


async def test_engine_vl_ocr_names_the_setting_when_the_model_is_not_found() -> None:
    client = _RoutingFakeClient(vl_error=_not_found())

    with pytest.raises(DocsToMarkdownError, match=r"'mineru'.*no such model.*SIE_MCP_VLOCR_MODEL"):
        await _route(client, engine="vl-ocr", render=_fake_render(1))


async def test_auto_threshold_keeps_docling_at_16_chars_without_calling_vl_ocr() -> None:
    client = _RoutingFakeClient(docling={"markdown": "x" * 16}, vl_error=_not_found())

    result = await _route(client, engine="auto", render=_fake_render(1))

    assert result.engine == "docling"
    assert len(client.calls) == 1


async def test_auto_threshold_tries_vl_ocr_below_16_chars() -> None:
    client = _RoutingFakeClient(docling={"markdown": "x" * 15}, vl_error=_not_found())

    with pytest.raises(DocsToMarkdownError, match="only 15 characters"):
        await _route(client, engine="auto", render=_fake_render(1))


async def test_auto_propagates_docling_error_when_not_renderable() -> None:
    client = _RoutingFakeClient(docling={"error": "unsupported format"})

    with pytest.raises(DocsToMarkdownError):
        await _route(client, data=b"PK\x03\x04docx", fmt="docx", engine="auto")


# --- route_to_markdown: ocr flag is engine-scoped ---------------------------


async def test_auto_ignores_ocr_flag_so_scanned_pdf_falls_back() -> None:
    # engine=auto + ocr=True must NOT run Docling's built-in OCR: doing so returns
    # garbled, non-empty text that masks the scan from the near-empty fallback. With
    # OCR forced off the scan lands empty and falls back to VL-OCR cleanly.
    client = _RoutingFakeClient(docling={"markdown": ""}, vl_pages=[_page("clean OCR")])

    result = await _route(client, engine="auto", ocr=True, render=_fake_render(1))

    assert result.engine == "vl-ocr"
    assert result.markdown == "clean OCR"
    # The Docling front-door pass ran with built-in OCR disabled (no options).
    assert client.calls[0]["options"] is None


async def test_docling_engine_honors_ocr_flag() -> None:
    # When Docling is pinned, the caller's ocr flag enables its built-in OCR.
    client = _RoutingFakeClient(docling={"markdown": "scanned text via docling ocr"})

    result = await _route(client, engine="docling", ocr=True, render=_fake_render(1))

    assert result.engine == "docling"
    assert client.calls[0]["options"] == {"ocr": True}


# --- rendered-pixel budget --------------------------------------------------


def test_clamp_scale_reduces_oversized_page() -> None:
    # A 5000x5000 pt page at scale 2.0 → 1e8 px, far above the budget; clamp it down.
    clamped = ocr_routing._clamp_scale(5000, 5000, 2.0)
    assert clamped < 2.0
    assert (5000 * clamped) * (5000 * clamped) <= ocr_routing._MAX_PAGE_PIXELS + 1


def test_clamp_scale_keeps_small_page_unchanged() -> None:
    assert ocr_routing._clamp_scale(200, 300, 2.0) == 2.0


def test_downscale_to_budget_shrinks_large_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_routing, "_MAX_PAGE_PIXELS", 1024)
    out = ocr_routing._downscale_to_budget(Image.new("RGB", (256, 256)))
    assert out.width * out.height <= 1024


def test_downscale_to_budget_leaves_small_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_routing, "_MAX_PAGE_PIXELS", 1024)
    out = ocr_routing._downscale_to_budget(Image.new("RGB", (16, 16)))
    assert out.size == (16, 16)


def test_render_downscales_oversized_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ocr_routing, "_MAX_PAGE_PIXELS", 1024)
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), "white").save(buf, format="PNG")  # 65536 px ≫ 1024
    images = ocr_routing.render_to_images(buf.getvalue())
    assert len(images) == 1
    assert images[0].width * images[0].height <= 1024


def test_render_caps_total_pixels(make_pdf: Callable[[int], bytes], monkeypatch: pytest.MonkeyPatch) -> None:
    # Each 200x300pt page renders to 400x600 = 240k px; two pages exceed a 300k cap,
    # so the total-budget guard fails loudly on the second page (no silent truncation).
    monkeypatch.setattr(ocr_routing, "_MAX_TOTAL_PIXELS", 300_000)
    with pytest.raises(DocsToMarkdownError, match="VL-OCR budget"):
        ocr_routing.render_to_images(make_pdf(2))


def test_render_allows_total_pixels_within_budget(make_pdf: Callable[[int], bytes]) -> None:
    # A 3-page doc (~720k px) stays well under the default budget and renders fully.
    images = ocr_routing.render_to_images(make_pdf(3))
    assert len(images) == 3
