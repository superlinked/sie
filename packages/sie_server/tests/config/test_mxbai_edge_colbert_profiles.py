from __future__ import annotations

from pathlib import Path

import yaml
from sie_server.config.model import ModelConfig

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "mixedbread-ai__mxbai-edge-colbert-v0-32m.yaml"


def _config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(_MODEL_PATH.read_text()))


def test_default_and_muvera_preserve_published_retrieval_recipe() -> None:
    config = _config()

    for name in ("default", "muvera"):
        profile = config.resolve_profile(name)
        assert profile.compute_precision == "bfloat16"
        assert (
            profile.adapter_path == "sie_server.adapters.colbert_modernbert_flash.adapter:ColBERTModernBERTFlashAdapter"
        )
        assert profile.loadtime["token_dim"] == 64
        assert profile.loadtime["query_max_length"] == 8192
        assert profile.loadtime["query_prefix"] == "[Q] "
        assert profile.loadtime["doc_prefix"] == "[D] "
        assert profile.loadtime["doc_punctuation_skiplist"] is True
        assert profile.runtime["query_max_length"] == 48
        assert profile.runtime["max_seq_length"] == 512
    assert config.resolve_profile("muvera").runtime["output_similarity"] == {"dense": "cosine"}
