"""LightOnOCR-3-1B catalog pin, generation contract and bundle routing.

The 1B keeps LightOnOCR-2-1B's Mistral3/Pixtral architecture, so it is served like
LightOnOCR-2-1B with the LightOnOCR-3 prompt and sampling contract.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.sglang_vision_extract.adapter import SGLangVisionExtractAdapter
from sie_server.core.loader import expand_profile_variants, load_adapter, load_model_config, validate_pinned_revision

_SERVER = Path(__file__).resolve().parents[2]
_MODELS = _SERVER / "models"
_BUNDLES = _SERVER / "bundles"
_MODEL = "lightonai/LightOnOCR-3-1B"
_REVISION = "23d544085c886bbdace647bcfab91203299417e2"
_CONFIG = _MODELS / "lightonai__LightOnOCR-3-1B.yaml"
_SGLANG_ADAPTER = "sie_server.adapters.sglang_vision_extract.adapter:SGLangVisionExtractAdapter"
# Model-card output cap, and the SIE->SGLang httpx pool ceiling (sglang/generation.py).
_MAX_NEW_TOKENS = 4096
_CONCURRENCY = 256
_CONTEXT = 16384
_LOADTIME = {
    "mem_fraction_static": 0.70,
    "served_model_name": _MODEL,
    "trust_remote_code": False,
    "max_concurrent_requests": _CONCURRENCY,
    "meter_pages": True,
    "max_new_tokens": _MAX_NEW_TOKENS,
    "temperature": 0.1,
    "top_p": 1.0,
    "allowed_instructions": ["grounding"],
    "extra_env": {"SIE_SGLANG_LIGHTON_OCR_COMPAT": "1"},
    "extra_launch_args": ["--disable-radix-cache"],
}


def test_catalog_pins_checkpoint_and_upstream_generation() -> None:
    config = yaml.safe_load(_CONFIG.read_text())
    assert config["sie_id"] == config["hf_id"] == _MODEL
    assert config["hf_revision"] == _REVISION
    assert config["inputs"] == {"text": False, "image": True, "audio": False, "video": False}
    assert config["tasks"] == {"encode": None, "score": None, "extract": {}}
    assert config["max_sequence_length"] == _CONTEXT
    assert list(config["profiles"]) == ["default", "markdown"]
    default = config["profiles"]["default"]
    assert default["adapter_path"] == _SGLANG_ADAPTER
    assert default["compute_precision"] == "bfloat16"
    assert default["max_batch_tokens"] == 16384
    assert default["adapter_options"]["loadtime"] == _LOADTIME
    assert default["adapter_options"]["runtime"] == {"max_new_tokens": _MAX_NEW_TOKENS, "num_beams": 1}
    markdown = config["profiles"]["markdown"]
    assert markdown["extends"] == "default"
    assert markdown["adapter_options"]["loadtime"] == {
        **_LOADTIME,
        "default_prompt": "grounding",
        "output_postprocess": "lightonocr3_markdown",
    }


def test_loader_builds_pinned_sglang_default(monkeypatch: pytest.MonkeyPatch) -> None:
    engine_load = Mock()
    monkeypatch.setattr(SGLangVisionExtractAdapter, "load", engine_load)
    config = load_model_config(_CONFIG)
    validate_pinned_revision(config)
    adapter = load_adapter(config, _MODELS, device="cuda:0")
    assert isinstance(adapter, SGLangVisionExtractAdapter)
    assert adapter._revision == _REVISION
    assert adapter._max_seq_length == _CONTEXT
    assert adapter._max_new_tokens == _MAX_NEW_TOKENS
    assert adapter._max_new_tokens < _CONTEXT
    assert (adapter._temperature, adapter._top_p) == (0.1, 1.0)
    assert adapter._system_prompt is None
    assert adapter._chat_template_kwargs == {}
    assert adapter._trust_remote_code is False
    assert adapter._allowed_instructions == frozenset({"grounding"})
    assert adapter._output_postprocess is None
    assert adapter._resolve_prompt_and_label(None, {}) == (None, "markdown")
    assert adapter._meter_pages is True
    assert adapter.max_concurrent_dispatch() == _CONCURRENCY
    assert adapter._extra_env == {"SIE_SGLANG_LIGHTON_OCR_COMPAT": "1"}
    assert adapter._extra_launch_args == ["--disable-radix-cache"]
    engine_load.assert_not_called()


def test_markdown_profile_retains_default_engine_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SGLangVisionExtractAdapter, "load", Mock())
    config = load_model_config(_CONFIG)
    default = config.resolve_profile("default")
    markdown = config.resolve_profile("markdown")
    assert markdown.adapter_path == default.adapter_path == _SGLANG_ADAPTER
    assert dict(markdown.runtime) == dict(default.runtime) == {"max_new_tokens": _MAX_NEW_TOKENS, "num_beams": 1}
    assert dict(markdown.loadtime) == {
        **default.loadtime,
        "default_prompt": "grounding",
        "output_postprocess": "lightonocr3_markdown",
    }
    variant = expand_profile_variants([config])[f"{_MODEL}:markdown"]
    validate_pinned_revision(variant)
    adapter = load_adapter(variant, _MODELS, device="cuda:0")
    assert isinstance(adapter, SGLangVisionExtractAdapter)
    assert adapter._revision == _REVISION
    assert adapter._output_postprocess == "lightonocr3_markdown"
    assert adapter._resolve_prompt_and_label(None, {}) == ("grounding", "markdown")
    assert (adapter._temperature, adapter._top_p) == (0.1, 1.0)
    assert adapter._system_prompt is None
    assert adapter.max_concurrent_dispatch() == _CONCURRENCY


def test_routes_to_sglang_vision_extract_bundle_only() -> None:
    for bundle in sorted(_BUNDLES.glob("*.yaml")):
        expected = bundle.name == "sglang-vision-extract.yaml"
        matches = match_bundle_models(bundle, _MODELS)
        assert (_MODEL in matches) == expected, bundle.name
        assert (f"{_MODEL}:markdown" in matches) == expected, bundle.name
        assert f"{_MODEL}:transformers" not in matches, bundle.name
