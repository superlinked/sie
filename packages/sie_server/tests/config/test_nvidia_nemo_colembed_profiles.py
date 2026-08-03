from pathlib import Path

import yaml
from sie_server.config.model import ModelConfig

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "nvidia__llama-nemoretriever-colembed-3b-v1.yaml"


def test_v1_catalog_matches_emitted_llama_hidden_width() -> None:
    config = ModelConfig.model_validate(yaml.safe_load(_MODEL_PATH.read_text()))

    assert config.tasks.encode is not None
    assert config.tasks.encode.multivector is not None
    assert config.tasks.encode.multivector.dim == 3072
    for profile_name in ("default", "muvera"):
        profile = config.resolve_profile(profile_name)
        assert profile.loadtime["token_dim"] == 3072
