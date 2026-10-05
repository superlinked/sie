from __future__ import annotations

from pathlib import Path

import yaml
from sie_server.config.model import ModelConfig

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "topk-io__Iso-ModernColBERT.yaml"


def _config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(_MODEL_PATH.read_text()))


def test_default_and_muvera_follow_the_published_pylate_recipe() -> None:
    """The checkpoint's ``config_sentence_transformers.json`` (revision a43b93e).

    PyLate marks queries and documents with ``[Q] ``/``[D] `` after ``[CLS]``,
    cuts queries at 32 tokens and documents at 300, and drops the 32
    punctuation skiplist words from document vectors (the training loss never
    saw them, so their vectors are unconstrained).
    """
    config = _config()

    for name in ("default", "muvera"):
        profile = config.resolve_profile(name)
        assert profile.compute_precision == "bfloat16"
        assert profile.loadtime["query_prefix"] == "[Q] "
        assert profile.loadtime["doc_prefix"] == "[D] "
        assert profile.loadtime["doc_punctuation_skiplist"] is True
        assert profile.loadtime["skip_special_tokens"] is False
        assert profile.runtime["query_max_length"] == 32
        assert profile.runtime["max_seq_length"] == 300

    assert config.resolve_profile("muvera").runtime["output_similarity"] == {"dense": "dot"}


def test_long_context_changes_only_document_length() -> None:
    config = _config()
    default = config.resolve_profile("default")
    long_context = config.resolve_profile("long_context")

    assert long_context.loadtime == default.loadtime
    assert long_context.runtime == dict(default.runtime) | {"max_seq_length": 8192}


def test_candle_profile_keeps_its_own_options() -> None:
    """The Candle worker refuses the prefix and skiplist options, so its profile declares its own."""
    candle = _config().resolve_profile("candle")

    assert "query_prefix" not in candle.loadtime
    assert "doc_prefix" not in candle.loadtime
    assert "doc_punctuation_skiplist" not in candle.loadtime
    assert candle.loadtime["max_seq_length"] == 8192


def test_smve_profile_uses_topk_published_settings() -> None:
    """The model card evaluates SMVE at width 65536 and k=32; the smve profile returns that encoding."""
    config = _config()
    smve = config.resolve_profile("smve")

    assert smve.loadtime["smve_config"] == {"width": 65536, "k": 32}
    assert smve.runtime["output_types"] == ["sparse"]
    assert smve.runtime["output_similarity"] == {"sparse": "dot"}
    assert smve.runtime["max_seq_length"] == 300
    assert "smve_config" not in config.resolve_profile("candle").loadtime
