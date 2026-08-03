from __future__ import annotations

from unittest.mock import MagicMock

import transformers
from sie_server.adapters.colbert_modernbert_flash.adapter import ColBERTModernBERTFlashAdapter


def test_load_tokenizer_happy_path_uses_auto_tokenizer(monkeypatch) -> None:
    """GTE/Reason path: AutoTokenizer succeeds, so its result is returned unchanged."""
    adapter = ColBERTModernBERTFlashAdapter("lightonai/GTE-ModernColBERT-v1")

    sentinel_auto = object()

    def fake_auto(*_args, **kwargs):
        assert kwargs["trust_remote_code"] is False
        return sentinel_auto

    def fail_fast(*_args, **_kwargs):
        raise AssertionError("PreTrainedTokenizerFast must not be called on the happy path")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_auto))
    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained", staticmethod(fail_fast))

    assert adapter._load_tokenizer() is sentinel_auto


def test_load_tokenizer_falls_back_when_auto_cannot_resolve_class(monkeypatch) -> None:
    """Iso-ModernColBERT path: transformers<5 cannot resolve the saved tokenizer_class.

    AutoTokenizer raises a ValueError ("Tokenizer class TokenizersBackend does not
    exist ...") and we must fall back to PreTrainedTokenizerFast.
    """
    adapter = ColBERTModernBERTFlashAdapter("topk-io/Iso-ModernColBERT")

    sentinel_fast = object()

    def fail_auto(*_args, **_kwargs):
        raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")

    def fake_fast(*_args, **_kwargs):
        return sentinel_fast

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(fail_auto))
    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained", staticmethod(fake_fast))

    assert adapter._load_tokenizer() is sentinel_fast


def test_load_disables_remote_repository_code(monkeypatch) -> None:
    adapter = ColBERTModernBERTFlashAdapter(
        "lightonai/GTE-ModernColBERT-v1",
        revision="cbbe53366e564450558f5e639dd499171f127538",
    )
    model = MagicMock()
    model.config.hidden_size = 768

    def fake_model(*_args, **kwargs):
        assert kwargs["trust_remote_code"] is False
        assert kwargs["revision"] == "cbbe53366e564450558f5e639dd499171f127538"
        return model

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", staticmethod(fake_model))
    monkeypatch.setattr(adapter, "_load_tokenizer", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(
        "sie_server.adapters.colbert_modernbert_flash.adapter.load_pylate_dense_chain",
        MagicMock(return_value=None),
    )

    adapter.load("cuda")

    model.to.assert_called_once_with("cuda")
    model.eval.assert_called_once_with()
