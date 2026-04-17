from __future__ import annotations

import pytest
from sie_server.types.inputs import Item


class TestLightOnOCRAdapter:
    """Tests for LightOnOCRAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> LightOnOCRAdapter:
        """Create an adapter instance."""
        from sie_server.adapters.lighton_ocr import LightOnOCRAdapter

        return LightOnOCRAdapter(
            "lightonai/LightOnOCR-2-1B",
            compute_precision="bfloat16",
        )

    def test_capabilities(self, adapter: LightOnOCRAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["image"]
        assert caps.outputs == ["json"]

    def test_dims(self, adapter: LightOnOCRAdapter) -> None:
        """Adapter reports empty dimensions (extraction model)."""
        dims = adapter.dims
        assert dims.dense is None
        assert dims.sparse is None
        assert dims.multivector is None

    def test_encode_raises(self, adapter: LightOnOCRAdapter) -> None:
        """Encode raises NotImplementedError."""
        items = [Item(text="hello")]
        with pytest.raises(NotImplementedError, match="does not support encode"):
            adapter.encode(items, output_types=["dense"])

    def test_extract_before_load(self, adapter: LightOnOCRAdapter) -> None:
        """Extract before load raises RuntimeError."""
        from sie_server.types.inputs import ImageInput

        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.extract(items)

    def test_build_messages_default(self, adapter: LightOnOCRAdapter) -> None:
        """Default messages have system and user roles with image."""
        messages = adapter._build_messages(instruction=None)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are an OCR engine. Return the markdown representation of the document."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == [{"type": "image"}]

    def test_build_messages_with_instruction(self, adapter: LightOnOCRAdapter) -> None:
        """Instruction appends text content to user message."""
        messages = adapter._build_messages(instruction="Extract tables only")

        assert len(messages) == 2
        assert messages[1]["role"] == "user"
        content = messages[1]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "image"}
        assert content[1] == {"type": "text", "text": "Extract tables only"}

    def test_convert_output(self, adapter: LightOnOCRAdapter) -> None:
        """Markdown text is wrapped in Entity with label 'markdown'."""
        entities = adapter._convert_output("# Title\n\nSome text")

        assert len(entities) == 1
        assert entities[0]["text"] == "# Title\n\nSome text"
        assert entities[0]["label"] == "markdown"
        assert entities[0]["score"] == 1.0

    def test_convert_output_strips_whitespace(self, adapter: LightOnOCRAdapter) -> None:
        """Output text is stripped of leading/trailing whitespace."""
        entities = adapter._convert_output("  \n  # Title  \n  ")

        assert len(entities) == 1
        assert entities[0]["text"] == "# Title"

    def test_system_prompt_configurable(self) -> None:
        """Custom system_prompt is respected."""
        from sie_server.adapters.lighton_ocr import LightOnOCRAdapter

        custom_prompt = "Extract all text from the image."
        adapter = LightOnOCRAdapter(
            "lightonai/LightOnOCR-2-1B",
            system_prompt=custom_prompt,
        )

        messages = adapter._build_messages(instruction=None)
        assert messages[0]["content"] == custom_prompt
