"""Public LightOnOCR-3 catalog pin, generation contract and bundle routing."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.lighton_ocr.adapter import LightOnOCR3Adapter
from sie_server.adapters.sglang_vision_extract.adapter import SGLangVisionExtractAdapter
from sie_server.core.loader import load_adapter, load_model_config, validate_pinned_revision

_SERVER = Path(__file__).resolve().parents[2]
_MODELS = _SERVER / "models"
_BUNDLES = _SERVER / "bundles"
_ADAPTER = "sie_server.adapters.lighton_ocr.adapter"
_SGLANG_ADAPTER = "sie_server.adapters.sglang_vision_extract.adapter"
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
    assert list(config["profiles"]) == ["default", "transformers"]
    default = config["profiles"]["default"]
    assert default["adapter_path"] == f"{_SGLANG_ADAPTER}:SGLangVisionExtractAdapter"
    assert default["compute_precision"] == "bfloat16"
    loadtime = default["adapter_options"]["loadtime"]
    assert {key: loadtime[key] for key in ("max_new_tokens", "temperature", "top_p")} == {
        "max_new_tokens": 12288,
        "temperature": 0.1,
        "top_p": 1.0,
    }
    assert loadtime["chat_template_kwargs"] == {"enable_thinking": False}
    assert loadtime["allowed_instructions"] == ["grounding"]
    assert loadtime["trust_remote_code"] is False
    assert loadtime["meter_pages"] is True
    assert "system_prompt" not in loadtime
    assert "extra_env" not in loadtime
    assert default["adapter_options"]["runtime"] == {"max_new_tokens": 12288, "num_beams": 1}
    transformers = config["profiles"]["transformers"]
    assert transformers["adapter_path"] == f"{_ADAPTER}:LightOnOCR3Adapter"
    assert transformers["compute_precision"] == "bfloat16"
    assert transformers["adapter_options"] == {
        "loadtime": {"max_new_tokens": 12288, "temperature": 0.1, "top_p": 1.0},
        "runtime": {},
    }


def test_loader_builds_pinned_sglang_default(monkeypatch: pytest.MonkeyPatch) -> None:
    engine_load = Mock()
    monkeypatch.setattr(SGLangVisionExtractAdapter, "load", engine_load)
    config = load_model_config(_CONFIG)
    validate_pinned_revision(config)
    adapter = load_adapter(config, _MODELS, device="cuda:0")
    assert isinstance(adapter, SGLangVisionExtractAdapter)
    assert adapter._revision == _REVISION
    assert adapter._max_seq_length == 24576
    assert adapter._max_new_tokens == 12288
    assert (adapter._temperature, adapter._top_p) == (0.1, 1.0)
    assert adapter._system_prompt is None
    assert adapter._trust_remote_code is False
    engine_load.assert_not_called()


def test_transformers_profile_builds_profile_generation() -> None:
    config = load_model_config(_CONFIG)
    profile = config.resolve_profile("transformers")
    assert profile.adapter_path == f"{_ADAPTER}:LightOnOCR3Adapter"
    loadtime: dict[str, Any] = dict(profile.loadtime)
    adapter = LightOnOCR3Adapter(_MODEL, revision=config.hf_revision, **loadtime)
    assert adapter._model_name_or_path == _MODEL
    assert adapter._revision == _REVISION
    assert adapter._compute_precision == "bfloat16"
    assert adapter._generation_options({}) == _GENERATION
    assert adapter._generation_options({"max_new_tokens": 1024})["max_new_tokens"] == 1024
    with pytest.raises(ValueError, match="max_new_tokens"):
        adapter._generation_options({"max_new_tokens": 12289})


def test_default_routes_to_sglang_vision_extract_bundle_only() -> None:
    assert _ADAPTER in yaml.safe_load((_BUNDLES / "transformers5.yaml").read_text())["adapters"]
    for bundle in sorted(_BUNDLES.glob("*.yaml")):
        expected = bundle.name == "sglang-vision-extract.yaml"
        assert (_MODEL in match_bundle_models(bundle, _MODELS)) == expected, bundle.name
