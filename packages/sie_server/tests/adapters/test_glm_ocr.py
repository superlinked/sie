from __future__ import annotations

import pytest
from sie_server.types.inputs import Item


class TestGlmOcrAdapter:
    """Tests for GlmOcrAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> GlmOcrAdapter:
        """Create an adapter instance."""
        from sie_server.adapters.glm_ocr import GlmOcrAdapter

        return GlmOcrAdapter(
            "zai-org/GLM-OCR",
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: GlmOcrAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["image"]
        assert caps.outputs == ["json"]

    def test_dims(self, adapter: GlmOcrAdapter) -> None:
        """Adapter reports empty dimensions (extraction model)."""
        dims = adapter.dims
        assert dims.dense is None
        assert dims.sparse is None
        assert dims.multivector is None

    def test_encode_raises(self, adapter: GlmOcrAdapter) -> None:
        """Encode raises NotImplementedError."""
        items = [Item(text="hello")]
        with pytest.raises(NotImplementedError, match="does not support encode"):
            adapter.encode(items, output_types=["dense"])

    def test_extract_before_load(self, adapter: GlmOcrAdapter) -> None:
        """Extract before load raises RuntimeError."""
        from sie_server.types.inputs import ImageInput

        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.extract(items)

    def test_convert_output(self, adapter: GlmOcrAdapter) -> None:
        """Markdown text is wrapped in Entity with label 'markdown'."""
        entities = adapter._convert_output("# Title\n\nSome text")

        assert len(entities) == 1
        assert entities[0]["text"] == "# Title\n\nSome text"
        assert entities[0]["label"] == "markdown"
        assert entities[0]["score"] == 1.0

    def test_convert_output_strips_whitespace(self, adapter: GlmOcrAdapter) -> None:
        """Output text is stripped of leading/trailing whitespace."""
        entities = adapter._convert_output("  \n  # Title  \n  ")

        assert len(entities) == 1
        assert entities[0]["text"] == "# Title"

    def test_user_text_configurable(self) -> None:
        """Custom user_text is respected."""
        from sie_server.adapters.glm_ocr import GlmOcrAdapter

        custom = "Extract all text from the image."
        adapter = GlmOcrAdapter(
            "zai-org/GLM-OCR",
            user_text=custom,
        )

        assert adapter._user_text == custom

    def test_default_user_text(self) -> None:
        """Default user_text matches HF README example."""
        from sie_server.adapters.glm_ocr import GlmOcrAdapter

        adapter = GlmOcrAdapter("zai-org/GLM-OCR")
        assert adapter._user_text == "Text Recognition:"

    def test_float16_raises_on_cuda(self) -> None:
        """float16 compute_precision raises on CUDA devices."""
        from sie_server.adapters.glm_ocr import GlmOcrAdapter

        adapter = GlmOcrAdapter(
            "zai-org/GLM-OCR",
            compute_precision="float16",
        )
        with pytest.raises(ValueError, match="does not support float16"):
            adapter._resolve_dtype("cuda:0")
