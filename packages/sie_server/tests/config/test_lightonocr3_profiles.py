"""Public LightOnOCR-3 catalog pin, generation contract and bundle routing."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.lighton_ocr.adapter import LightOnOCR3Adapter
from sie_server.core.loader import load_adapter, load_model_config, validate_pinned_revision

_SERVER = Path(__file__).resolve().parents[2]
_MODELS = _SERVER / "models"
_BUNDLES = _SERVER / "bundles"
_ADAPTER = "sie_server.adapters.lighton_ocr.adapter"
_MODEL = "lightonai/LightOnOCR-3-4B"
_REVISION = "b71010f095dc3735de7ee6b9969bff8f8f50b39a"
_CONFIG = _MODELS / "lightonai__LightOnOCR-3-4B.yaml"
_GENERATION = {"max_new_tokens": 12288, "do_sample": True, "num_beams": 1, "temperature": 0.1, "top_p": 1.0}


def test_catalog_pins_checkpoint_and_upstream_generation() -> None:
    config = yaml.safe_load(_CONFIG.read_text())
    assert config["sie_id"] == config["hf_id"] == _MODEL
    assert config["hf_revision"] == _REVISION
    assert config["inputs"] == {"text": False, "image": True, "audio": False, "video": False}
    assert config["tasks"] == {"encode": None, "score": None, "extract": {}}
    assert config["max_sequence_length"] == 24576
    assert list(config["profiles"]) == ["default"]
    profile = config["profiles"]["default"]
    assert profile["adapter_path"] == f"{_ADAPTER}:LightOnOCR3Adapter"
    assert profile["compute_precision"] == "bfloat16"
    assert profile["adapter_options"] == {
        "loadtime": {"max_new_tokens": 12288, "temperature": 0.1, "top_p": 1.0},
        "runtime": {},
    }


def test_loader_builds_pinned_adapter_with_profile_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    native_load = Mock()
    monkeypatch.setattr(LightOnOCR3Adapter, "load", native_load)
    config = load_model_config(_CONFIG)
    validate_pinned_revision(config)
    adapter = load_adapter(config, _MODELS, device="cuda:0")
    assert isinstance(adapter, LightOnOCR3Adapter)
    assert adapter._model_name_or_path == _MODEL
    assert adapter._revision == _REVISION
    assert adapter._compute_precision == "bfloat16"
    assert adapter._generation_options({}) == _GENERATION
    assert adapter._generation_options({"max_new_tokens": 1024})["max_new_tokens"] == 1024
    with pytest.raises(ValueError, match="max_new_tokens"):
        adapter._generation_options({"max_new_tokens": 12289})
    native_load.assert_not_called()


def test_served_by_transformers5_bundle_only() -> None:
    assert _ADAPTER in yaml.safe_load((_BUNDLES / "transformers5.yaml").read_text())["adapters"]
    for bundle in sorted(_BUNDLES.glob("*.yaml")):
        assert (_MODEL in match_bundle_models(bundle, _MODELS)) == (bundle.name == "transformers5.yaml"), bundle.name
