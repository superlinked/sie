from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from sie_server.adapters.colbert import ColBERTAdapter
from sie_server.adapters.colbert_modernbert_flash import ColBERTModernBERTFlashAdapter
from sie_server.adapters.colbert_rotary_flash import ColBERTRotaryFlashAdapter
from sie_server.core.inference_output import ScoreOutput
from sie_server.types.inputs import Item

# Create a random generator for tests
_RNG = np.random.default_rng(42)


@pytest.fixture
def stub_chain_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep load-path unit tests offline: ColBERTAdapter.load() probes the pylate
    Dense chain, which would hf_hub_download modules.json for fake model IDs.
    """
    monkeypatch.setattr("sie_server.adapters.colbert.load_pylate_dense_chain", lambda *_args, **_kwargs: None)


def _write_chain_checkpoint(root: Path) -> str:
    """Write a local pylate-style checkpoint with a single Dense module 384->64."""
    root.mkdir(parents=True, exist_ok=True)
    modules = [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Dense", "type": "pylate.models.Dense.Dense"},
    ]
    (root / "modules.json").write_text(json.dumps(modules))
    dense_dir = root / "1_Dense"
    dense_dir.mkdir()
    config = {
        "in_features": 384,
        "out_features": 64,
        "bias": False,
        "activation_function": "torch.nn.modules.linear.Identity",
    }
    (dense_dir / "config.json").write_text(json.dumps(config))
    save_file({"linear.weight": torch.randn(64, 384)}, str(dense_dir / "model.safetensors"))
    return str(root)


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
        stub_chain_loader: None,
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

    def test_should_use_native_mode_modernbert(self, adapter: ColBERTAdapter) -> None:
        """ModernBERT auto-detects native mode (the manual flash loop is BERT-only)."""
        assert adapter._should_use_native_mode(SimpleNamespace(model_type="modernbert")) is True
        assert adapter._should_use_native_mode(SimpleNamespace(model_type="bert")) is False
        # Sharp edge: the explicit override check runs FIRST and wins over the
        # ModernBERT autodetection.
        adapter._use_native_attention = False
        assert adapter._should_use_native_mode(SimpleNamespace(model_type="modernbert")) is False

    @patch("transformers.AutoConfig")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_native_load_without_flash_attn_uses_sdpa(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        mock_config_class: MagicMock,
        adapter: ColBERTAdapter,
        stub_chain_loader: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Native mode on CUDA without flash-attn loads with sdpa instead of crashing."""
        mock_config = MagicMock()
        mock_config.hidden_size = 384
        mock_config.model_type = "modernbert"
        mock_config_class.from_pretrained.return_value = mock_config

        mock_model = MagicMock()
        mock_model.config.hidden_size = 384
        mock_model.named_modules.return_value = []
        mock_model.linear = torch.nn.Linear(384, 128)  # keep the projection probe offline
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token_id = 103
        mock_tokenizer.vocab = {}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        monkeypatch.setattr("sie_server.core.inference.is_flash_attention_available", lambda _device=None: False)

        adapter.load("cuda")

        call_kwargs = mock_model_class.from_pretrained.call_args
        assert call_kwargs.kwargs["attn_implementation"] == "sdpa"

    @patch("transformers.AutoConfig")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_native_load_with_flash_attn_uses_flash_attention_2(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        mock_config_class: MagicMock,
        adapter: ColBERTAdapter,
        stub_chain_loader: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Native mode with flash-attn available is bit-unchanged: flash_attention_2."""
        mock_config = MagicMock()
        mock_config.hidden_size = 384
        mock_config.model_type = "modernbert"
        mock_config_class.from_pretrained.return_value = mock_config

        mock_model = MagicMock()
        mock_model.config.hidden_size = 384
        mock_model.named_modules.return_value = []
        mock_model.linear = torch.nn.Linear(384, 128)  # keep the projection probe offline
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token_id = 103
        mock_tokenizer.vocab = {}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        monkeypatch.setattr("sie_server.core.inference.is_flash_attention_available", lambda _device=None: True)

        adapter.load("cuda")

        call_kwargs = mock_model_class.from_pretrained.call_args
        assert call_kwargs.kwargs["attn_implementation"] == "flash_attention_2"

    @patch("transformers.AutoConfig")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_chain_probe_wins_over_projection_probes(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        mock_config_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A checkpoint shipping a Dense chain uses it; the legacy probes must not run."""
        mock_config = MagicMock()
        mock_config.hidden_size = 384
        mock_config_class.from_pretrained.return_value = mock_config

        mock_model = MagicMock()
        mock_model.config.hidden_size = 384
        mock_model.named_modules.return_value = []
        mock_model.linear = torch.nn.Linear(384, 128)  # the legacy probe would pick this
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token_id = 103
        mock_tokenizer.vocab = {}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        adapter = ColBERTAdapter(_write_chain_checkpoint(tmp_path / "ckpt"), token_dim=64)
        adapter.load("cpu")

        assert adapter._dense_chain is not None
        assert [tuple(w.shape) for w in adapter._dense_chain] == [(64, 384)]
        assert adapter._linear is None
        assert adapter._actual_token_dim == 64

    @patch("transformers.AutoConfig")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_no_chain_falls_through_to_existing_probes(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        mock_config_class: MagicMock,
        adapter: ColBERTAdapter,
        stub_chain_loader: None,
    ) -> None:
        """Without a chain (no modules.json), _find_projection_layer runs as before."""
        mock_config = MagicMock()
        mock_config.hidden_size = 384
        mock_config_class.from_pretrained.return_value = mock_config

        linear = torch.nn.Linear(384, 128)
        mock_model = MagicMock()
        mock_model.config.hidden_size = 384
        mock_model.named_modules.return_value = []
        mock_model.linear = linear
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token_id = 103
        mock_tokenizer.vocab = {}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        adapter.load("cpu")

        assert adapter._dense_chain is None
        assert adapter._linear is linear
        assert adapter._actual_token_dim == 128

    def test_encode_native_nonexpanded_query_drops_pad_rows(self) -> None:
        """Without expansion, PAD rows must not leak into query multivectors; the
        expansion path still keeps ALL rows (MASK tokens are semantic).
        """
        adapter = ColBERTAdapter("test-model", skip_special_tokens=False, query_expansion=False)
        adapter._device = "cpu"
        adapter._actual_token_dim = 8

        hidden = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8) + 1.0

        class _FakeBatch(dict):
            def to(self, _device: str) -> _FakeBatch:
                return self

        batch = _FakeBatch(
            input_ids=torch.tensor([[1, 2, 3, 4], [1, 2, 0, 0]]),
            attention_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]),
        )
        adapter._tokenizer = MagicMock(return_value=batch)
        adapter._model = MagicMock(return_value=SimpleNamespace(last_hidden_state=hidden))

        out = adapter._encode_native(["q one two", "q"], max_length=4, use_expansion=False, is_query=True)
        assert out[0].shape[0] == 4  # all real tokens kept
        assert out[1].shape[0] == 2  # PAD rows dropped

        out_expanded = adapter._encode_native(["q one two", "q"], max_length=4, use_expansion=True, is_query=True)
        assert out_expanded[0].shape[0] == 4
        assert out_expanded[1].shape[0] == 4  # expansion keeps all rows

    def test_fallback_kwargs_overrides_disable_expansion_and_skiplist(self) -> None:
        """The flash adapter's fallback overrides align ColBERTAdapter with the flash
        path (no expansion, no doc skiplist, matching max length) without changing
        the defaults of directly-configured ColBERTAdapter models.
        """
        yaml_kwargs = {"model_name_or_path": "x", "token_dim": 64, "skip_special_tokens": False}
        merged = {**yaml_kwargs, **ColBERTModernBERTFlashAdapter.fallback_kwargs_overrides}
        fallback = ColBERTAdapter(**merged)
        assert fallback._query_expansion is False
        assert fallback._doc_punctuation_skiplist is False
        assert fallback._max_seq_length == 8192
        assert fallback._doc_max_length == 8192  # doc_max_length defaults to max_seq_length

        plain = ColBERTAdapter("x")
        assert plain._query_expansion is True
        assert plain._doc_punctuation_skiplist is True
        assert plain._max_seq_length == 512

    @patch("transformers.AutoConfig")
    @patch("transformers.AutoModel")
    @patch("transformers.AutoTokenizer")
    def test_skiplist_disabled_builds_empty_skiplist(
        self,
        mock_tokenizer_class: MagicMock,
        mock_model_class: MagicMock,
        mock_config_class: MagicMock,
        stub_chain_loader: None,
    ) -> None:
        """doc_punctuation_skiplist=False leaves the skiplist empty; default builds it."""
        mock_config = MagicMock()
        mock_config.hidden_size = 384
        mock_config_class.from_pretrained.return_value = mock_config

        mock_model = MagicMock()
        mock_model.config.hidden_size = 384
        mock_model.named_modules.return_value = []
        mock_model.linear = torch.nn.Linear(384, 128)  # keep the projection probe offline
        mock_model_class.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.mask_token_id = 103
        mock_tokenizer.vocab = {}
        mock_tokenizer.encode.return_value = [42]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        disabled = ColBERTAdapter("test-colbert-model", doc_punctuation_skiplist=False)
        disabled.load("cpu")
        assert disabled._doc_skiplist_ids == set()

        enabled = ColBERTAdapter("test-colbert-model")
        enabled.load("cpu")
        assert 42 in enabled._doc_skiplist_ids

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


# Encode-time runtime options resolved from the colbert/muvera profiles. These
# are irrelevant to MaxSim scoring; score_pairs() must accept-and-ignore them
# instead of inheriting BaseAdapter.score_pairs()'s NotImplementedError guard.
_ENCODE_OPTIONS = {
    "muvera": {},
    "output_types": ["dense"],
    "output_similarity": {"dense": "dot"},
}


class TestColBERTScorePairsOptions:
    """score_pairs() ignores encode-time options instead of raising (#1430)."""

    def test_score_pairs_with_nonempty_options_does_not_raise(self) -> None:
        """Non-empty encode-time options are accepted and ignored."""
        adapter = ColBERTAdapter("test-model")
        adapter._model = MagicMock()
        adapter._device = "cpu"

        # Both queries share text="q" -> grouped into one score() call over both
        # docs, so the patched return must cover both grouped docs.
        with patch.object(adapter, "score", return_value=[0.5, 0.3]) as mock_score:
            out = adapter.score_pairs(
                queries=[Item(text="q"), Item(text="q")],
                docs=[Item(text="d1"), Item(text="d2")],
                options=_ENCODE_OPTIONS,
            )

        assert isinstance(out, ScoreOutput)
        assert out.scores.shape == (2,)
        # Options must NOT be threaded into score().
        assert "options" not in mock_score.call_args.kwargs

    @pytest.mark.parametrize("options", [None, {}])
    def test_score_pairs_with_empty_options_returns_score_output(self, options: dict[str, object] | None) -> None:
        """options=None/{} behaves identically (parity with non-empty path)."""
        adapter = ColBERTAdapter("test-model")
        adapter._model = MagicMock()
        adapter._device = "cpu"

        with patch.object(adapter, "score", return_value=[0.5, 0.3]):
            out = adapter.score_pairs(
                queries=[Item(text="q"), Item(text="q")],
                docs=[Item(text="d1"), Item(text="d2")],
                options=options,
            )

        assert isinstance(out, ScoreOutput)
        assert out.scores.shape == (2,)

    def test_modernbert_flash_score_pairs_with_nonempty_options_does_not_raise(self) -> None:
        """Flash ColBERT adapter accepts-and-ignores encode-time options."""
        adapter = ColBERTModernBERTFlashAdapter("test-model")
        adapter._model = MagicMock()
        adapter._device = "cpu"

        with patch.object(adapter, "score", return_value=[0.5, 0.3]):
            out = adapter.score_pairs(
                queries=[Item(text="q"), Item(text="q")],
                docs=[Item(text="d1"), Item(text="d2")],
                options=_ENCODE_OPTIONS,
            )

        assert isinstance(out, ScoreOutput)
        assert out.scores.shape == (2,)

    def test_rotary_flash_score_pairs_with_nonempty_options_does_not_raise(self) -> None:
        """Rotary-flash ColBERT adapter (jina-colbert-v2) accepts-and-ignores options."""
        adapter = ColBERTRotaryFlashAdapter("test-model")
        adapter._model = MagicMock()
        adapter._device = "cpu"

        with patch.object(adapter, "score", return_value=[0.5, 0.3]) as mock_score:
            out = adapter.score_pairs(
                queries=[Item(text="q"), Item(text="q")],
                docs=[Item(text="d1"), Item(text="d2")],
                options=_ENCODE_OPTIONS,
            )

        assert isinstance(out, ScoreOutput)
        assert out.scores.shape == (2,)
        # Options must NOT be threaded into score().
        assert "options" not in mock_score.call_args.kwargs


class _ReentrancyDetectingTokenizer:
    """Emulates a HuggingFace fast tokenizer's non-re-entrancy.

    A real fast tokenizer wraps a Rust object behind a ``RefCell``; a second
    call that begins while a first is still in flight raises
    ``RuntimeError("Already borrowed")`` (issue #1800). This fake reproduces
    that exact failure mode: it raises if two threads are inside a tokenizer
    call at the same time, so a correct serialization guard makes it pass and a
    missing guard makes it fail.
    """

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.mask_token_id = 103
        self._active = 0
        self._active_lock = threading.Lock()

    def _enter(self) -> None:
        with self._active_lock:
            if self._active != 0:
                raise RuntimeError("Already borrowed")
            self._active += 1

    def _exit(self) -> None:
        with self._active_lock:
            self._active -= 1

    def _tokenize_one(self, text: str) -> list[int]:
        # Deterministic per-text length: [CLS] + one id per word + [SEP].
        return [1, *[42 for _ in text.split()], 2]

    def __call__(self, texts: Any, **kwargs: Any) -> Any:
        self._enter()
        try:
            # Simulate real tokenizer latency so overlapping threads collide.
            time.sleep(0.002)
            single = isinstance(texts, str)
            text_list = [texts] if single else list(texts)
            padding = kwargs.get("padding")
            ids = [self._tokenize_one(t) for t in text_list]
            if padding in (True, "max_length"):
                width = kwargs.get("max_length") if padding == "max_length" else max(len(x) for x in ids)
                assert width is not None
                pad_id = self.pad_token_id
                mask = [[1] * len(x) + [0] * (width - len(x)) for x in ids]
                ids = [x + [pad_id] * (width - len(x)) for x in ids]
            else:
                mask = [[1] * len(x) for x in ids]
            if kwargs.get("return_tensors") == "pt":
                return _FakeTensorBatch(
                    input_ids=torch.tensor(ids),
                    attention_mask=torch.tensor(mask),
                )
            return {"input_ids": ids, "attention_mask": mask}
        finally:
            self._exit()


class _FakeTensorBatch(dict):
    def to(self, _device: str) -> _FakeTensorBatch:
        return self


class TestColBERTTokenizerConcurrencyGuard:
    """Regression tests for issue #1800: colbert-small multivector encode
    desync surfaced as ``inference_error: Already borrowed`` under concurrency.

    The encode path tokenizes on the inference-executor thread while the §7.3
    metering fallback (``count_input_tokens``) tokenizes on a separate
    thread-pool thread for another concurrent request against the *same*
    tokenizer. ``_tokenizer_guard()`` must serialise both.
    """

    def _make_adapter(self) -> ColBERTAdapter:
        adapter = ColBERTAdapter("test-model", skip_special_tokens=False, query_expansion=False)
        adapter._device = "cpu"
        adapter._actual_token_dim = 8
        adapter._max_seq_length = 512
        tokenizer = _ReentrancyDetectingTokenizer()
        adapter._tokenizer = tokenizer  # ty: ignore[invalid-assignment]

        def _fake_forward(**batch: Any) -> Any:
            input_ids = batch["input_ids"]
            b, seq = input_ids.shape
            hidden = torch.ones(b, seq, 8, dtype=torch.float32)
            return SimpleNamespace(last_hidden_state=hidden)

        adapter._model = MagicMock(side_effect=_fake_forward)
        return adapter

    def test_encode_and_metering_do_not_collide_under_concurrency(self) -> None:
        """Encoding on one thread and metering-tokenizing on another must not
        raise ``Already borrowed`` — the guard serialises tokenizer access.
        """
        adapter = self._make_adapter()
        texts = ["a", "the quick brown fox", "x", "a considerably longer piece of text here"]
        items = [Item(text=t) for t in texts]

        errors: list[BaseException] = []
        stop = threading.Event()

        def encode_loop() -> None:
            try:
                for _ in range(40):
                    if stop.is_set():
                        return
                    out = adapter._encode_native(texts, max_length=64, use_expansion=False, is_query=False)
                    assert len(out) == len(texts)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        def metering_loop() -> None:
            try:
                for _ in range(40):
                    if stop.is_set():
                        return
                    counts = adapter.count_input_tokens(items)
                    assert counts is not None
                    assert len(counts) == len(items)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=encode_loop) for _ in range(2)]
        threads += [threading.Thread(target=metering_loop) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        stop.set()

        assert not errors, f"tokenizer collision under concurrency: {errors!r}"

    def test_reentrancy_detector_fires_without_guard(self) -> None:
        """Sanity check the harness: bare concurrent tokenizer calls (no guard)
        DO raise ``Already borrowed`` — otherwise the guard test is vacuous.
        """
        tokenizer = _ReentrancyDetectingTokenizer()
        errors: list[BaseException] = []

        def call_loop() -> None:
            try:
                for _ in range(40):
                    tokenizer(["a", "b c"], return_tensors="pt", padding=True)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=call_loop) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert any(isinstance(e, RuntimeError) and "Already borrowed" in str(e) for e in errors)


class TestColBERTRaggedMultivectorBatch:
    """Issue #1800: multivector encode must return exactly one vector-set per
    input, in worker-item order, for any batch/length mix.
    """

    def test_encode_native_ragged_batch_one_set_per_input_in_order(self) -> None:
        # normalize=False so the per-row hidden fill is preserved and we can
        # assert worker-item ordering directly.
        adapter = ColBERTAdapter("test-model", skip_special_tokens=False, query_expansion=False, normalize=False)
        adapter._device = "cpu"
        adapter._actual_token_dim = 8
        adapter._max_seq_length = 512
        tokenizer = _ReentrancyDetectingTokenizer()
        adapter._tokenizer = tokenizer  # ty: ignore[invalid-assignment]

        def _fake_forward(**batch: Any) -> Any:
            input_ids = batch["input_ids"]
            b, seq = input_ids.shape
            # Row-distinct hidden states so we can assert per-item ordering.
            base = torch.arange(b, dtype=torch.float32).reshape(b, 1, 1)
            hidden = torch.ones(b, seq, 8, dtype=torch.float32) + base
            return SimpleNamespace(last_hidden_state=hidden)

        adapter._model = MagicMock(side_effect=_fake_forward)

        # Ragged: very short and long inputs mixed.
        texts = ["a", "one two three four five", "x", "a b", "the fox ran quickly over"]
        out = adapter._encode_native(texts, max_length=64, use_expansion=False, is_query=False)

        # Exactly one vector-set per input, in order.
        assert len(out) == len(texts)
        for mv in out:
            assert mv.ndim == 2
            assert mv.shape[1] == 8
            assert mv.shape[0] > 0

        # Per-input token counts follow the (padded then attention-masked)
        # lengths: [CLS] + words + [SEP], with PAD rows dropped by the mask.
        expected_lens = [len(t.split()) + 2 for t in texts]
        assert [mv.shape[0] for mv in out] == expected_lens

        # Order preserved: row i's hidden fill == 1.0 + i (from _fake_forward).
        for i, mv in enumerate(out):
            assert float(mv[0, 0]) == 1.0 + i


class _RaisingExpansionTokenizer:
    """Fake tokenizer whose call raises, used to prove the query-expansion
    ``pad_token_id`` swap is restored even when tokenization fails.

    Exposes a mutable ``pad_token_id`` (the swap target) so the test can assert
    it is back to the original after the raising call propagates.
    """

    def __init__(self) -> None:
        self.pad_token_id = 7  # sentinel "original" pad id
        self.mask_token_id = 103

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        # Malformed-input / OOM analogue: the tokenizer raises mid-call, after
        # ``pad_token_id`` has already been swapped to the MASK id by the caller.
        raise RuntimeError("tokenizer boom")


class TestColBERTExpansionPadTokenRestore:
    """CodeRabbit #1800: the query-expansion branches swap ``pad_token_id`` to
    the MASK id before tokenizing and must restore it in ``finally``.

    Because the swap now runs under ``_tokenizer_guard()``, a tokenizer failure
    that skipped the restore would leave the instance's ``pad_token_id`` stuck at
    the MASK id and leak that corruption to every later guarded caller
    (including the §7.3 metering re-tokenize) — silently mis-padding and
    mis-counting until the model is reloaded. The ``finally`` must restore the
    original id even when the tokenizer raises.
    """

    def _adapter(self) -> ColBERTAdapter:
        adapter = ColBERTAdapter("test-model", skip_special_tokens=False, query_expansion=True)
        adapter._device = "cpu"
        adapter._actual_token_dim = 8
        adapter._max_seq_length = 512
        adapter._expansion_token_id = 103  # MASK id swapped in during expansion
        adapter._tokenizer = _RaisingExpansionTokenizer()  # ty: ignore[invalid-assignment]
        adapter._model = MagicMock()
        return adapter

    def test_encode_native_restores_pad_token_id_on_tokenizer_error(self) -> None:
        adapter = self._adapter()
        original = adapter._tokenizer.pad_token_id
        assert original == 7
        with pytest.raises(RuntimeError, match="tokenizer boom"):
            adapter._encode_native(["query one two"], max_length=8, use_expansion=True, is_query=True)
        # The finally restored the original id despite the raise (not left at
        # the MASK/expansion id 103).
        assert adapter._tokenizer.pad_token_id == original
        assert adapter._tokenizer.pad_token_id != adapter._expansion_token_id

    def test_encode_manual_flash_restores_pad_token_id_on_tokenizer_error(self) -> None:
        adapter = self._adapter()
        original = adapter._tokenizer.pad_token_id
        assert original == 7
        with pytest.raises(RuntimeError, match="tokenizer boom"):
            adapter._encode_manual_flash(["query one two"], max_length=8, use_expansion=True, is_query=True)
        assert adapter._tokenizer.pad_token_id == original
        assert adapter._tokenizer.pad_token_id != adapter._expansion_token_id
