from pathlib import Path

import pytest
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.core.loader import load_model_configs, reject_unknown_loadtime_options, resolve_adapter_class

SERVER_ROOT = Path(__file__).resolve().parents[2]
MODELS = SERVER_ROOT / "models"
BUNDLES = SERVER_ROOT / "bundles"
ADAPTER = "sie_server.adapters.sequence_classification.adapter:SequenceClassificationAdapter"


@pytest.mark.parametrize(
    ("model_id", "revision", "capacity"),
    [
        ("ProsusAI/finbert", "4556d13015211d73dccd3fdd39d39232506f3e43", 512),
        ("protectai/deberta-v3-base-prompt-injection-v2", "90c9989b1a342275dd0d1a95aad283c04e075671", 512),
        ("unitary/toxic-bert", "4d6c22e74ba2fdd26bc4f7238f50766b045a0d94", 512),
        ("SamLowe/roberta-base-go_emotions", "d75048347613a25d77de8cf6412eaae9fa7b26be", 512),
        ("papluca/xlm-roberta-base-language-detection", "9865598389ca9d95637462f743f683b51d75b87b", 512),
        ("NousResearch/Minos-v1", "edb1e8b7cc80a86cc356534948cca0ef4c46f8e2", 8192),
    ],
)
def test_fixed_head_classifiers_are_pinned_extract_only_and_served_by_both_text_bundles(
    model_id: str, revision: str, capacity: int
) -> None:
    config = load_model_configs(MODELS)[model_id]
    assert config.hf_id == model_id
    assert config.hf_revision == revision
    assert config.max_sequence_length == capacity
    assert config.inputs.text is True
    assert config.tasks.extract is not None
    assert config.tasks.encode is None
    assert config.tasks.score is None
    profile = config.resolve_profile("default")
    assert profile.adapter_path == ADAPTER
    adapter_class = resolve_adapter_class(config, MODELS)
    assert adapter_class.__name__ == "SequenceClassificationAdapter"
    reject_unknown_loadtime_options(adapter_class, profile.loadtime, model_name=model_id)
    assert model_id in match_bundle_models(BUNDLES / "default.yaml", MODELS)
    assert model_id in match_bundle_models(BUNDLES / "transformers5.yaml", MODELS)
