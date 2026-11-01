"""Pinned flash v1.2 checkpoint (joint protocol), bounded text contract and compatible bundle routing."""

from pathlib import Path

import yaml
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.pplx_decider.adapter import _CHECKPOINT_FILES
from sie_server.api.systemone import _is_decision_model
from sie_server.config.model import ModelConfig

SERVER = Path(__file__).resolve().parents[2]
MODEL = "perplexity-ai/pplx-decider-flash-v1.2"


def test_flash_model_is_pinned_text_only_and_accepted_by_decision_routes():
    config = ModelConfig.model_validate(
        yaml.safe_load((SERVER / "models/perplexity-ai__pplx-decider-flash-v1.2.yaml").read_text())
    )
    assert config.hf_revision == "9ff7cf9838e59df461c276b7699d56b244d2af58"
    assert config.sie_id == config.hf_id == MODEL
    assert config.max_sequence_length == 8192
    assert config.inputs.text is True
    assert config.inputs.image is config.inputs.audio is config.inputs.video is False
    assert config.tasks.encode is config.tasks.score is None
    assert _is_decision_model(config)
    assert "joint_config.json" in _CHECKPOINT_FILES
    for bundle in (SERVER / "bundles").glob("*.yaml"):
        assert (MODEL in match_bundle_models(bundle, SERVER / "models")) == (bundle.name == "transformers5.yaml")
