from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from sie_server.types.inputs import Item


class TestDonutAdapter:
    """Tests for DonutAdapter with mocked model."""

    @pytest.fixture
    def mock_donut_model(self) -> MagicMock:
        """Create a mock Donut model."""
        mock = MagicMock()
        # Mock generate method - returns ModelOutput-like object
        mock_output = MagicMock()
        mock_output.sequences = MagicMock()
        mock.generate.return_value = mock_output
        mock.dtype = MagicMock()
        mock.decoder.config.max_position_embeddings = 2048
        return mock

    @pytest.fixture
    def mock_donut_processor(self) -> MagicMock:
        """Create a mock Donut processor."""
        mock = MagicMock()
        # Mock image processing - returns dict with pixel_values
        mock_image_result = MagicMock()
        mock_image_result.pixel_values = MagicMock()
        mock.return_value = mock_image_result
        # Mock tokenizer
        mock.tokenizer.return_value = MagicMock()
        mock.tokenizer.pad_token_id = 0
        mock.tokenizer.eos_token_id = 2
        mock.tokenizer.unk_token_id = 3
        mock.tokenizer.eos_token = "</s>"  # noqa: S105 — test mock tokenizer config
        mock.tokenizer.pad_token = "<pad>"  # noqa: S105 — test mock tokenizer config
        # Mock batch_decode
        mock.batch_decode.return_value = ["<s_cord-v2><s_menu><nm>Coffee</nm><price>5.00</price></s_menu></s>"]
        # Mock token2json
        mock.token2json.return_value = {"menu": {"nm": "Coffee", "price": "5.00"}}
        return mock

    @pytest.fixture
    def adapter(self) -> DonutAdapter:
        """Create an adapter instance."""
        from sie_server.adapters.donut import DonutAdapter

        return DonutAdapter(
            "naver-clova-ix/donut-base-finetuned-cord-v2",
            default_task="<s_cord-v2>",
            compute_precision="float16",
        )

    def test_capabilities(self, adapter: DonutAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["image"]
        assert caps.outputs == ["json"]

    def test_dims(self, adapter: DonutAdapter) -> None:
        """Adapter reports empty dimensions (extraction model)."""
        dims = adapter.dims
        assert dims.dense is None
        assert dims.sparse is None
        assert dims.multivector is None

    def test_encode_raises_not_implemented(self, adapter: DonutAdapter) -> None:
        """Encode raises NotImplementedError."""
        items = [Item(text="hello")]
        with pytest.raises(NotImplementedError, match="does not support encode"):
            adapter.encode(items, output_types=["dense"])

    def test_extract_before_load_raises(self, adapter: DonutAdapter) -> None:
        """Extract before load raises error."""
        from sie_server.types.inputs import ImageInput

        items = [Item(images=[ImageInput(data=b"fake", format="jpeg")])]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.extract(items)

    def test_build_prompt_basic_task(self, adapter: DonutAdapter) -> None:
        """Build prompt returns task token for basic tasks."""
        prompt = adapter._build_prompt("<s_cord-v2>", instruction=None)
        assert prompt == "<s_cord-v2>"

    def test_build_prompt_docvqa_with_question(self, adapter: DonutAdapter) -> None:
        """Build prompt formats DocVQA question correctly."""
        prompt = adapter._build_prompt("<s_docvqa>", instruction="What is the total?")
        assert prompt == "<s_docvqa><s_question>What is the total?</s_question><s_answer>"

    def test_try_parse_json_valid(self, adapter: DonutAdapter) -> None:
        """Try parse JSON handles valid JSON."""
        result = adapter._try_parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_try_parse_json_invalid(self, adapter: DonutAdapter) -> None:
        """Try parse JSON handles invalid JSON gracefully."""
        result = adapter._try_parse_json("not json at all")
        assert result == {"raw": "not json at all"}

    def test_try_parse_json_with_special_tokens(self, adapter: DonutAdapter) -> None:
        """Try parse JSON strips special tokens."""
        result = adapter._try_parse_json('<s_menu>{"key": "value"}</s_menu>')
        # After stripping tags: {"key": "value"}
        assert result == {"key": "value"}

    def test_convert_output_cord(self, adapter: DonutAdapter) -> None:
        """Convert output handles CORD format."""
        from sie_server.adapters.donut import TASK_CORD

        parsed = {"menu": {"nm": "Coffee", "price": "5.00"}}
        raw_text = "<s_menu><nm>Coffee</nm><price>5.00</price></s_menu>"

        entities = adapter._convert_output(parsed, TASK_CORD, raw_text)

        # Check entities extracted from nested structure
        assert len(entities) >= 2
        entity_labels = [e["label"] for e in entities]
        assert "menu.nm" in entity_labels
        assert "menu.price" in entity_labels

    def test_convert_output_docvqa(self, adapter: DonutAdapter) -> None:
        """Convert output handles DocVQA format."""
        from sie_server.adapters.donut import TASK_DOCVQA

        parsed = {"answer": "The total is $10.00"}
        raw_text = "<s_answer>The total is $10.00</s_answer>"

        entities = adapter._convert_output(parsed, TASK_DOCVQA, raw_text)

        assert len(entities) == 1
        assert entities[0]["text"] == "The total is $10.00"
        assert entities[0]["label"] == "answer"
        assert entities[0]["score"] == 1.0

    def test_convert_output_rvlcdip(self, adapter: DonutAdapter) -> None:
        """Convert output handles RVLCDIP format."""
        from sie_server.adapters.donut import TASK_RVLCDIP

        parsed = {"class": "invoice"}
        raw_text = "<s_class>invoice</s_class>"

        entities = adapter._convert_output(parsed, TASK_RVLCDIP, raw_text)

        assert len(entities) == 1
        assert entities[0]["text"] == "invoice"
        assert entities[0]["label"] == "document_class"

    def test_extract_cord_entities_nested(self, adapter: DonutAdapter) -> None:
        """Extract CORD entities handles nested structures."""
        parsed = {
            "menu": [
                {"nm": "Coffee", "price": "5.00"},
                {"nm": "Tea", "price": "3.00"},
            ],
            "total": {"total_price": "8.00"},
        }

        entities = adapter._extract_cord_entities(parsed)

        # Should have 5 entities: 2 items x 2 fields + 1 total
        assert len(entities) == 5
        # Check labels have correct prefixes
        labels = [e["label"] for e in entities]
        assert "menu[0].nm" in labels
        assert "menu[0].price" in labels
        assert "menu[1].nm" in labels
        assert "total.total_price" in labels
