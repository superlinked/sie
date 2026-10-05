"""Every model that declares an ``smve`` profile builds its SMVE postprocessor from its own config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sie_server.config.model import ModelConfig
from sie_server.core.loader import load_adapter
from sie_server.core.postprocessor import SmveConfig, SmvePostprocessor

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _config(model_file: str) -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load((_MODELS_DIR / model_file).read_text()))


_SMVE_MODELS = tuple(path.name for path in sorted(_MODELS_DIR.glob("*.yaml")) if "smve" in _config(path.name).profiles)


def test_model_catalog_contains_smve_profiles() -> None:
    assert _SMVE_MODELS


@pytest.mark.parametrize("model_file", _SMVE_MODELS)
def test_smve_profile_returns_dot_scored_sparse_vectors(model_file: str) -> None:
    profile = _config(model_file).resolve_profile("smve")

    assert profile.runtime["output_types"] == ["sparse"]
    assert profile.runtime["output_similarity"] == {"sparse": "dot"}
    assert profile.runtime["smve"] == {}


@pytest.mark.parametrize("model_file", _SMVE_MODELS)
def test_smve_profile_builds_the_configured_postprocessor(model_file: str) -> None:
    config = _config(model_file)
    default, smve_profile = config.resolve_profile("default"), config.resolve_profile("smve")
    # The profile shares the default profile's loaded adapter, which reads smve_config.
    assert smve_profile.loadtime == default.loadtime
    assert "smve_config" in default.loadtime

    smve = load_adapter(config, _MODELS_DIR, device="cpu").get_postprocessors()["smve"]

    assert isinstance(smve, SmvePostprocessor)
    assert smve.config == SmveConfig(**default.loadtime["smve_config"])
