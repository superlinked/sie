from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml
from sie_server.adapters.base import ModelAdapter
from sie_server.config.model import ModelConfig

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _muvera_profiles() -> tuple[tuple[str, str], ...]:
    profiles: list[tuple[str, str]] = []
    for model_path in sorted(_MODELS_DIR.glob("*.yaml")):
        config = ModelConfig.model_validate(yaml.safe_load(model_path.read_text()))
        if "muvera" not in config.profiles:
            continue
        profiles.append((model_path.name, config.resolve_profile("muvera").adapter_path))
    return tuple(profiles)


_MUVERA_PROFILES = _muvera_profiles()


def test_model_catalog_contains_muvera_profiles() -> None:
    assert _MUVERA_PROFILES


@pytest.mark.parametrize(("model_file", "adapter_path"), _MUVERA_PROFILES)
def test_muvera_profile_adapter_exposes_postprocessor_hook(model_file: str, adapter_path: str) -> None:
    module_name, class_name = adapter_path.split(":", maxsplit=1)
    adapter_cls = getattr(importlib.import_module(module_name), class_name)

    assert adapter_cls.get_postprocessors is not ModelAdapter.get_postprocessors, (
        f"{model_file} declares a muvera profile, but {adapter_path} inherits "
        "ModelAdapter.get_postprocessors() and cannot produce dense output"
    )
