from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sie_server.adapters.colbert import ColBERTAdapter
from sie_server.types.inputs import Item

# Create a random generator for tests
_RNG = np.random.default_rng(42)


class TestColBERTAdapter:
    """Tests for ColBERTAdapter with mocked model."""

    @pytest.fixture
    def adapter(self) -> ColBERTAdapter:
        """Create an adapter instance."""
        return ColBERTAdapter(
            "test-colbert-model",
            token_dim=128,
            normalize=True,
            max_seq_length=512,
            query_max_length=32,
        )

    def test_capabilities(self, adapter: ColBERTAdapter) -> None:
        """Adapter reports correct capabilities."""
        caps = adapter.capabilities
        assert caps.inputs == ["text"]
        assert caps.outputs == ["multivector", "score"]

    def test_dims_before_load_returns_none(self, adapter: ColBERTAdapter) -> None:
        """Dims returns None values before load (BaseAdapter derives from spec)."""
        dims = adapter.dims
        assert dims.multivector is None

    def test_encode_before_load_raises(self, adapter: ColBERTAdapter) -> None:
        """Encode before load raises error."""
        items = [Item(text="hello")]
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.encode(items, output_types=["multivector"])

    def test_score_before_load_raises(self, adapter: ColBERTAdapter) -> None:
        """Score before load raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            adapter.score(Item(text="query"), [Item(text="doc")])

    @patch("transformers.AutoConfig")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_cpu_loads_with_eager_attention(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        mock_config_class: MagicMock,
        adapter: ColBERTAdapter,
    ) -> None:
        """CPU device loads with eager attention (no flash attention required)."""
        mock_config = MagicMock()
        mock_config.hidden_size = 384
        mock_config_class.from_pretrained.return_value = mock_config

        mock_model = MagicMock()
        mock_model.config.hidden_size = 384
        mock_model.named_modules.return_value = []
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token_id = 103
        mock_tokenizer.vocab = {}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        adapter.load("cpu")

        # Should use eager attention, not flash_attention_2
        mock_model_class.from_pretrained.assert_called_once()
        call_kwargs = mock_model_class.from_pretrained.call_args
        assert call_kwargs.kwargs["attn_implementation"] == "eager"

    def test_validate_output_types(self, adapter: ColBERTAdapter) -> None:
        """Only multivector output type is supported."""
        # This tests the validation logic without loading
        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter._validate_output_types(["dense"])

        with pytest.raises(ValueError, match="Unsupported output types"):
            adapter._validate_output_types(["sparse"])

        # Should not raise for multivector
        adapter._validate_output_types(["multivector"])

    def test_extract_texts_with_prefix(self, adapter: ColBERTAdapter) -> None:
        """Text extraction applies query/doc prefixes."""
        adapter._query_prefix = "[Q] "
        adapter._doc_prefix = "[D] "

        items = [Item(text="hello"), Item(text="world")]

        # Query mode
        texts = adapter._extract_texts(items, instruction=None, is_query=True)
        assert texts == ["[Q] hello", "[Q] world"]

        # Document mode
        texts = adapter._extract_texts(items, instruction=None, is_query=False)
        assert texts == ["[D] hello", "[D] world"]

    def test_extract_texts_with_instruction(self, adapter: ColBERTAdapter) -> None:
        """Text extraction handles instruction."""
        items = [Item(text="hello")]

        texts = adapter._extract_texts(items, instruction="search:", is_query=True)
        assert texts == ["search: hello"]

    def test_extract_texts_without_text_raises(self, adapter: ColBERTAdapter) -> None:
        """Text extraction raises if item has no text."""
        items = [Item()]  # No text
        with pytest.raises(ValueError, match="requires text input"):
            adapter._extract_texts(items, instruction=None, is_query=False)

    def test_run_embeddings_bert_architecture(self, adapter: ColBERTAdapter) -> None:
        """Test _run_embeddings with BERT-style embeddings (word_embeddings)."""
        import torch

        # Mock BERT-style embeddings
        mock_embeddings = MagicMock()
        mock_embeddings.word_embeddings = MagicMock(return_value=torch.randn(3, 768))
        mock_embeddings.position_embeddings = MagicMock(return_value=torch.randn(3, 768))
        mock_embeddings.token_type_embeddings = MagicMock(return_value=torch.randn(3, 768))
        mock_embeddings.LayerNorm = MagicMock(side_effect=lambda x: x)
        mock_embeddings.dropout = MagicMock(side_effect=lambda x: x)

        mock_model = MagicMock()
        mock_model.embeddings = mock_embeddings

        adapter._model = mock_model
        adapter._device = "cpu"

        input_ids = torch.tensor([1, 2, 3])
        position_ids = torch.tensor([0, 1, 2])

        result = adapter._run_embeddings(input_ids, position_ids)

        # Verify BERT path was taken
        mock_embeddings.word_embeddings.assert_called_once()
        mock_embeddings.position_embeddings.assert_called_once()
        assert result.shape == (3, 768)

    def test_run_embeddings_modernbert_architecture(self, adapter: ColBERTAdapter) -> None:
        """Test _run_embeddings handles ModernBERT architecture (tok_embeddings).

        ModernBERT uses different embedding attribute names:
        - tok_embeddings instead of word_embeddings
        - norm instead of LayerNorm
        - drop instead of dropout
        - No position embeddings (uses RoPE in attention)

        This test verifies the fallback path when ColBERTAdapter is used
        as a fallback for ColBERTModernBERTFlashAdapter on non-CUDA devices.
        """
        import torch

        # Mock ModernBERT-style embeddings (no word_embeddings, no position_embeddings)
        mock_embeddings = MagicMock(spec=[])  # Empty spec to avoid auto-attributes

        # Add only ModernBERT attributes
        mock_embeddings.tok_embeddings = MagicMock(return_value=torch.randn(3, 768))
        mock_embeddings.norm = MagicMock(side_effect=lambda x: x)
        mock_embeddings.drop = MagicMock(side_effect=lambda x: x)

        mock_model = MagicMock()
        mock_model.embeddings = mock_embeddings

        adapter._model = mock_model
        adapter._device = "cpu"

        input_ids = torch.tensor([1, 2, 3])
        position_ids = torch.tensor([0, 1, 2])

        result = adapter._run_embeddings(input_ids, position_ids)

        # Verify ModernBERT path was taken
        mock_embeddings.tok_embeddings.assert_called_once()
        mock_embeddings.norm.assert_called_once()
        mock_embeddings.drop.assert_called_once()
        assert result.shape == (3, 768)

    def test_run_embeddings_unsupported_architecture_raises(self, adapter: ColBERTAdapter) -> None:
        """Test _run_embeddings raises for unknown embedding architecture."""
        import torch

        # Mock embeddings without word_embeddings or tok_embeddings
        mock_embeddings = MagicMock(spec=[])  # No embedding methods

        mock_model = MagicMock()
        mock_model.embeddings = mock_embeddings

        adapter._model = mock_model
        adapter._device = "cpu"

        input_ids = torch.tensor([1, 2, 3])
        position_ids = torch.tensor([0, 1, 2])

        with pytest.raises(AttributeError, match=r"word_embeddings.*tok_embeddings"):
            adapter._run_embeddings(input_ids, position_ids)


class TestColBERTAdapterMultivectorOutput:
    """Tests for ColBERT multivector output format."""

    def test_multivector_shape_validation(self) -> None:
        """Multivector output should have shape [num_tokens, token_dim]."""
        # Create sample multivector output
        num_tokens = 10
        token_dim = 128
        multivector = _RNG.standard_normal((num_tokens, token_dim)).astype(np.float32)

        # Verify shape
        assert multivector.ndim == 2
        assert multivector.shape[0] == num_tokens
        assert multivector.shape[1] == token_dim

    def test_multivector_is_normalized(self) -> None:
        """Multivector tokens should be L2 normalized."""
        import torch
        from torch.nn import functional

        # Create random vectors
        num_tokens = 5
        token_dim = 128
        raw = torch.randn(num_tokens, token_dim)

        # Normalize
        normalized = functional.normalize(raw, p=2, dim=-1)

        # Check L2 norm is 1 for each token
        norms = torch.norm(normalized, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    def test_maxsim_computation(self) -> None:
        """MaxSim should compute correctly."""
        import torch

        # Create query and doc multivectors
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 2 query tokens
        doc = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])  # 3 doc tokens

        # Normalize
        query = torch.nn.functional.normalize(query, p=2, dim=-1)
        doc = torch.nn.functional.normalize(doc, p=2, dim=-1)

        # Compute MaxSim: sim[i,j] = cosine(query[i], doc[j])
        sim = torch.matmul(query, doc.T)

        # For each query token, find max similarity with any doc token
        max_sims, _ = sim.max(dim=-1)  # [num_query_tokens]

        # Sum over query tokens
        maxsim_score = max_sims.sum().item()

        # Query token 0: [1, 0] -> max sim with doc token 1 [1, 0] = 1.0
        # Query token 1: [0, 1] -> max sim with doc token 2 [0, 1] = 1.0
        # Total MaxSim = 2.0
        assert abs(maxsim_score - 2.0) < 1e-5


class TestColBERTScoreMethod:
    """Tests for ColBERT adapter's score() method (MaxSim scoring)."""

    def test_score_returns_correct_number_of_scores(self) -> None:
        """Score returns one score per document."""
        from sie_server.core.inference_output import EncodeOutput

        adapter = ColBERTAdapter("test-model")

        # Mock the encode method to return known multivectors via EncodeOutput
        query_mv = _RNG.standard_normal((5, 128)).astype(np.float32)  # 5 query tokens
        doc1_mv = _RNG.standard_normal((10, 128)).astype(np.float32)  # 10 doc tokens
        doc2_mv = _RNG.standard_normal((8, 128)).astype(np.float32)  # 8 doc tokens

        with patch.object(adapter, "encode") as mock_encode:
            # First call returns query multivector, second returns doc multivectors
            mock_encode.side_effect = [
                EncodeOutput(multivector=[query_mv], batch_size=1, is_query=True),
                EncodeOutput(multivector=[doc1_mv, doc2_mv], batch_size=2, is_query=False),
            ]

            # Mock that model is loaded
            adapter._model = MagicMock()
            adapter._device = "cpu"

            scores = adapter.score(
                Item(text="query"),
                [Item(text="doc1"), Item(text="doc2")],
            )

        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_maxsim_score_increases_with_similarity(self) -> None:
        """More similar documents get higher MaxSim scores."""
        from sie_server.core.inference_output import EncodeOutput

        adapter = ColBERTAdapter("test-model")

        # Create query with specific pattern
        query_mv = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Doc 1: similar to query (should get high score)
        doc1_mv = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        # Doc 2: orthogonal to query (should get lower score)
        doc2_mv = np.array([[0.707, 0.707], [-0.707, 0.707]], dtype=np.float32)

        with patch.object(adapter, "encode") as mock_encode:
            mock_encode.side_effect = [
                EncodeOutput(multivector=[query_mv], batch_size=1, is_query=True),
                EncodeOutput(multivector=[doc1_mv, doc2_mv], batch_size=2, is_query=False),
            ]

            adapter._model = MagicMock()
            adapter._device = "cpu"

            scores = adapter.score(
                Item(text="query"),
                [Item(text="similar"), Item(text="different")],
            )

        # Doc 1 (similar) should have higher score than Doc 2 (orthogonal)
        assert scores[0] > scores[1]
