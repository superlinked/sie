from __future__ import annotations

from pathlib import Path

import yaml
from sie_server.config.model import ModelConfig

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "jinaai__jina-colbert-v2.yaml"
_MODEL_REVISION = "4552c4dc1ffd7d7a635b6a41a1077fe9c9cdd974"
_CODE_REVISION = "845308d0fd72a8406a3e378450e1a09522790419"


def _config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(_MODEL_PATH.read_text()))


def test_model_and_external_code_are_independently_pinned() -> None:
    config = _config()

    assert config.hf_revision == _MODEL_REVISION
    for name in ("default", "muvera"):
        assert config.resolve_profile(name).loadtime["code_revision"] == _CODE_REVISION


def test_default_and_muvera_preserve_published_retrieval_recipe() -> None:
    config = _config()

    assert config.max_sequence_length == 8192
    for name in ("default", "muvera"):
        profile = config.resolve_profile(name)
        assert profile.loadtime["query_prefix"] == "[QueryMarker]"
        assert profile.loadtime["doc_prefix"] == "[DocumentMarker]"
        assert profile.loadtime["doc_punctuation_skiplist"] is True
        assert profile.runtime["query_max_length"] == 32
        assert profile.runtime["max_seq_length"] == 300

    muvera = config.resolve_profile("muvera")
    assert muvera.runtime["output_similarity"] == {"dense": "dot"}
