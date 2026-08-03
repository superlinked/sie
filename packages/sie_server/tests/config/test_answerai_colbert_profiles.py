from __future__ import annotations

from pathlib import Path

import yaml
from sie_server.config.model import ModelConfig

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "answerdotai__answerai-colbert-small-v1.yaml"


def _config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(_MODEL_PATH.read_text()))


def test_default_and_muvera_preserve_published_retrieval_recipe() -> None:
    config = _config()

    assert config.max_sequence_length == 512
    for name in ("default", "muvera"):
        profile = config.resolve_profile(name)
        assert profile.loadtime["query_prefix"] == "[unused0]"
        assert profile.loadtime["doc_prefix"] == "[unused1]"
        assert profile.loadtime["doc_punctuation_skiplist"] is True
        assert profile.loadtime["query_expansion"] is False
        assert profile.runtime["query_max_length"] == 32
        assert "max_seq_length" not in profile.runtime
        assert profile.runtime.get("max_seq_length", config.max_sequence_length) == 512
