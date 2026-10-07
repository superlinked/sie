"""SAM 3 model identity, shipped detection threshold and Transformers5 bundle routing."""

from pathlib import Path

import yaml
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.sam3.adapter import DEFAULT_SCORE_THRESHOLD
from sie_server.core.loader import load_model_configs, resolve_adapter_class

SERVER = Path(__file__).resolve().parents[2]
MODELS = SERVER / "models"
BUNDLES = SERVER / "bundles"
MODEL_ID = "facebook/sam3"
REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"


def test_sam3_descriptor_is_pinned_image_only_extract() -> None:
    config = load_model_configs(MODELS)[MODEL_ID]
    assert config.sie_id == config.hf_id == MODEL_ID
    assert config.hf_revision == REVISION
    assert config.inputs.image
    assert not config.inputs.text
    assert not config.inputs.audio
    assert not config.inputs.video
    assert config.tasks.extract is not None
    assert config.tasks.encode is None
    assert config.tasks.score is None
    profile = config.resolve_profile("default")
    assert profile.compute_precision == "bfloat16"
    assert profile.max_batch_tokens == 16384
    assert profile.adapter_path == "sie_server.adapters.sam3.adapter:Sam3Adapter"
    assert resolve_adapter_class(config, MODELS).__name__ == "Sam3Adapter"


def test_sam3_ships_the_transformers_processor_default_threshold() -> None:
    profile = load_model_configs(MODELS)[MODEL_ID].resolve_profile("default")
    assert DEFAULT_SCORE_THRESHOLD == 0.3
    assert profile.loadtime == {"score_threshold": DEFAULT_SCORE_THRESHOLD}
    assert dict(profile.runtime) == {"score_threshold": DEFAULT_SCORE_THRESHOLD}


def test_sam3_is_served_only_from_the_transformers5_bundle() -> None:
    bundle = yaml.safe_load((BUNDLES / "transformers5.yaml").read_text())
    assert "sie_server.adapters.sam3.adapter" in bundle["adapters"]
    assert MODEL_ID in match_bundle_models(BUNDLES / "transformers5.yaml", MODELS)
    default = yaml.safe_load((BUNDLES / "default.yaml").read_text())
    assert "sie_server.adapters.sam3.adapter" not in default["adapters"]
    assert MODEL_ID not in match_bundle_models(BUNDLES / "default.yaml", MODELS)
