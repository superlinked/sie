"""Pinned native checkpoint, bounded text contract and compatible bundle routing."""

from pathlib import Path

import yaml
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.api.systemone import _is_decision_model
from sie_server.config.model import ModelConfig

SERVER = Path(__file__).resolve().parents[2]
MODEL = "perplexity-ai/pplx-decider-v1.1-27b"


def test_perplexity_native_model_is_pinned_text_only_and_accepted_by_decision_routes():
    config = ModelConfig.model_validate(
        yaml.safe_load((SERVER / "models/perplexity-ai__pplx-decider-v1.1-27b.yaml").read_text())
    )
    assert config.hf_revision == "14c9d90137962d300f3de95633f7b6535e80c5ce"
    assert config.sie_id == config.hf_id == MODEL
    assert config.max_sequence_length == 8192
    assert config.inputs.text is True
    assert config.inputs.image is config.inputs.audio is config.inputs.video is False
    assert config.tasks.encode is config.tasks.score is None
    assert _is_decision_model(config)
    for bundle in (SERVER / "bundles").glob("*.yaml"):
        assert (MODEL in match_bundle_models(bundle, SERVER / "models")) == (bundle.name == "transformers5.yaml")
