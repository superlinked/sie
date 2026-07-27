from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sie_server.adapters.colbert_modernbert_flash.adapter import ColBERTModernBERTFlashAdapter
from sie_server.config.model import ModelConfig

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "lightonai__GTE-ModernColBERT-v1.yaml"


def _config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(_MODEL_PATH.read_text()))


def test_default_and_muvera_preserve_published_retrieval_lengths() -> None:
    config = _config()

    for name in ("default", "muvera"):
        profile = config.resolve_profile(name)
        assert profile.loadtime["muvera_config"]["num_repetitions"] == 96
        assert profile.loadtime["query_prefix"] == "[Q] "
        assert profile.loadtime["doc_prefix"] == "[D] "
        assert profile.loadtime["doc_punctuation_skiplist"] is True
        assert profile.runtime["query_max_length"] == 48
        assert profile.runtime["max_seq_length"] == 300

    assert config.resolve_profile("muvera").runtime["output_similarity"] == {"dense": "dot"}


def test_long_context_changes_only_document_length() -> None:
    config = _config()
    default = config.resolve_profile("default")
    long_context = config.resolve_profile("long_context")

    assert long_context.loadtime == default.loadtime
    assert long_context.runtime == dict(default.runtime) | {"max_seq_length": 8192}


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("lightonai__GTE-ModernColBERT-v1.yaml", True),
        ("lightonai__Reason-ModernColBERT.yaml", False),
        ("topk-io__Iso-ModernColBERT.yaml", False),
    ],
)
def test_flash_and_fallback_share_explicit_punctuation_policy(filename: str, expected: bool) -> None:
    model_path = Path(__file__).resolve().parents[2] / "models" / filename
    config = ModelConfig.model_validate(yaml.safe_load(model_path.read_text()))
    loadtime = config.resolve_profile("default").loadtime

    assert loadtime["doc_punctuation_skiplist"] is expected
    fallback_kwargs = {**loadtime, **ColBERTModernBERTFlashAdapter.fallback_kwargs_overrides}
    assert fallback_kwargs["doc_punctuation_skiplist"] is expected
