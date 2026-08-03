from __future__ import annotations

from pathlib import Path

import yaml
from sie_server.config.model import ModelConfig

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "mixedbread-ai__mxbai-colbert-large-v1.yaml"


def _config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(_MODEL_PATH.read_text()))


def test_default_uses_checkpoint_trained_markers_and_query_expansion() -> None:
    config = _config()
    profile = config.resolve_profile("default")

    assert config.hf_revision == "591f3d193b80b75aac38e4ff9e341d6d136f045b"
    assert profile.loadtime["query_prefix"] == "[unused0]"
    assert profile.loadtime["doc_prefix"] == "[unused1]"
    assert profile.loadtime["query_max_length"] == 128
    assert profile.loadtime["doc_max_length"] == 256
    assert profile.loadtime["query_expansion"] is True
    assert profile.loadtime["doc_punctuation_skiplist"] is True
    assert config.resolve_profile("muvera").runtime["output_similarity"] == {"dense": "cosine"}
