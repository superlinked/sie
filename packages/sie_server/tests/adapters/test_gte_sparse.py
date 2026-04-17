from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters._utils import extract_texts, validate_output_types
from sie_server.adapters.gte_sparse_flash import GTESparseFlashAdapter
from sie_server.types.inputs import Item


class TestGTESparseFlashAdapter:
    """Tests for GTESparseFlashAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> GTESparseFlashAdapter:
        """Create an adapter instance."""
        return GTESparseFlashAdapter(
            "test-gte-sparse-model",
            max_seq_length=512,
            compute_precision="float16",
            trust_remote_code=True,
        )

    def test_capabilities(self, adapter: GTESparseFlashAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["sparse"]

    def test_dims_before_load(self, adapter: GTESparseFlashAdapter) -> None:
        """Dims returns None sparse before load (BaseAdapter reads from spec)."""
        dims = adapter.dims
        assert dims.sparse is None

    def test_encode_before_load_raises(self, adapter: GTESparseFlashAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["sparse"])

    def test_validate_output_types(self) -> None:
        """Only sparse output type is supported."""
        with pytest.raises(ValueError, match="Unsupported output types"):
            validate_output_types(["dense"], {"sparse"}, "GTESparseFlashAdapter")

        with pytest.raises(ValueError, match="Unsupported output types"):
            validate_output_types(["multivector"], {"sparse"}, "GTESparseFlashAdapter")

        # Should not raise for sparse
        validate_output_types(["sparse"], {"sparse"}, "GTESparseFlashAdapter")

    def test_extract_texts_basic(self) -> None:
        """Text extraction works without templates."""
        items = [Item(text="hello"), Item(text="world")]
        texts = extract_texts(items, instruction=None, is_query=False)
        assert texts == ["hello", "world"]

    def test_extract_texts_with_instruction(self) -> None:
        """Text extraction handles instruction."""
        items = [Item(text="hello")]
        texts = extract_texts(items, instruction="search:", is_query=True)
        assert texts == ["search: hello"]

    def test_extract_texts_with_query_template(self) -> None:
        """Text extraction uses query template when is_query=True."""
        items = [Item(text="hello")]
        texts = extract_texts(items, instruction=None, is_query=True, query_template="Query: {text}")
        assert texts == ["Query: hello"]

    def test_extract_texts_with_doc_template(self) -> None:
        """Text extraction uses doc template when is_query=False."""
        items = [Item(text="hello")]
        texts = extract_texts(items, instruction=None, is_query=False, doc_template="Document: {text}")
        assert texts == ["Document: hello"]

    def test_extract_texts_without_text_raises(self) -> None:
        """Text extraction raises if item has no text."""
        items = [Item()]  # No text
        with pytest.raises(ValueError, match="requires text input"):
            extract_texts(
                items,
                instruction=None,
                is_query=False,
                err_msg="GTESparseFlashAdapter requires text input",
            )

    @patch("sie_server.adapters.gte_sparse_flash.torch")
    def test_unload(self, mock_torch: MagicMock, adapter: GTESparseFlashAdapter) -> None:
        """Unload clears the model."""
        adapter._model = MagicMock()
        adapter._tokenizer = MagicMock()
        adapter._device = "cuda:0"
        adapter._vocab_size = 30522

        adapter.unload()

        assert adapter._model is None
        assert adapter.dims.sparse is None

    @patch("transformers.AutoModelForMaskedLM")
    @patch("transformers.AutoTokenizer")
    def test_load_wrong_architecture_raises(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        adapter: GTESparseFlashAdapter,
    ) -> None:
        """Load raises error for non-NewForMaskedLM architecture."""
        # Mock model without 'new' attribute (e.g., BERT)
        mock_model = MagicMock()
        mock_model.config.vocab_size = 30522
        # MagicMock has 'new' by default via __getattr__, so we need spec
        mock_model = MagicMock(spec=["config", "to", "eval"])
        mock_model.config.vocab_size = 30522
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()

        with pytest.raises(ValueError, match="NewForMaskedLM"):
            adapter.load("cuda:0")
