"""Public LightOnOCR-3 catalog pin, generation contract and bundle routing."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.lighton_ocr.adapter import LightOnOCR3Adapter
from sie_server.adapters.sglang_vision_extract.adapter import SGLangVisionExtractAdapter
from sie_server.core.loader import expand_profile_variants, load_adapter, load_model_config, validate_pinned_revision

_SERVER = Path(__file__).resolve().parents[2]
_MODELS = _SERVER / "models"
_BUNDLES = _SERVER / "bundles"
_ADAPTER = "sie_server.adapters.lighton_ocr.adapter"
_SGLANG_ADAPTER = "sie_server.adapters.sglang_vision_extract.adapter"
_GENERATION = {"do_sample": True, "num_beams": 1, "temperature": 0.1, "top_p": 1.0}
_PAGE_PIXELS = {"min_pixels": 65536, "max_pixels": 5242880}
_LAUNCH_ARGS = ["--disable-radix-cache", "--mm-process-config", '{"image":{"min_pixels":65536,"max_pixels":5242880}}']


@dataclass(frozen=True)
class _Checkpoint:
    test_id: str
    model: str
    revision: str
    max_new_tokens: int
    concurrency: int

    @property
    def config_path(self) -> Path:
        return _MODELS / f"{self.model.replace('/', '__')}.yaml"


# Shipped max_new_tokens and max_concurrent_requests. The 0.8B caps output at the model-card 4,096
# and dispatches 256 pages, the SIE->SGLang httpx pool ceiling (sglang/generation.py).
_CATALOG = (
    _Checkpoint("4B", "lightonai/LightOnOCR-3-4B", "b71010f095dc3735de7ee6b9969bff8f8f50b39a", 12288, 128),
    _Checkpoint("0.8B", "lightonai/LightOnOCR-3-0.8B", "36d8636925327c671bd8e479c517a004f7566bd9", 4096, 256),
)


@pytest.fixture(params=_CATALOG, ids=lambda checkpoint: checkpoint.test_id)
def catalog(request: pytest.FixtureRequest) -> _Checkpoint:
    return request.param


def test_catalog_pins_checkpoint_and_upstream_generation(catalog: _Checkpoint) -> None:
    model, revision, config_path = catalog.model, catalog.revision, catalog.config_path
    max_new_tokens, concurrency = catalog.max_new_tokens, catalog.concurrency
    config = yaml.safe_load(config_path.read_text())
    assert config["sie_id"] == config["hf_id"] == model
    assert config["hf_revision"] == revision
    assert config["inputs"] == {"text": False, "image": True, "audio": False, "video": False}
    assert config["tasks"] == {"encode": None, "score": None, "extract": {}}
    assert config["max_sequence_length"] == 24576
    assert list(config["profiles"]) == ["default", "markdown", "transformers"]
    default = config["profiles"]["default"]
    assert default["adapter_path"] == f"{_SGLANG_ADAPTER}:SGLangVisionExtractAdapter"
    assert default["compute_precision"] == "bfloat16"
    loadtime = default["adapter_options"]["loadtime"]
    assert {key: loadtime[key] for key in ("max_new_tokens", "temperature", "top_p")} == {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.1,
        "top_p": 1.0,
    }
    assert loadtime["served_model_name"] == model
    assert loadtime["chat_template_kwargs"] == {"enable_thinking": False}
    assert loadtime["allowed_instructions"] == ["grounding"]
    assert loadtime["trust_remote_code"] is False
    assert loadtime["meter_pages"] is True
    assert loadtime["max_concurrent_requests"] == concurrency
    assert loadtime["attention_backend"] == "flashinfer"
    assert loadtime["extra_launch_args"] == _LAUNCH_ARGS
    assert "system_prompt" not in loadtime
    assert "extra_env" not in loadtime
    assert "default_prompt" not in loadtime
    assert "output_postprocess" not in loadtime
    assert default["adapter_options"]["runtime"] == {"max_new_tokens": max_new_tokens, "num_beams": 1}
    transformers = config["profiles"]["transformers"]
    assert transformers["adapter_path"] == f"{_ADAPTER}:LightOnOCR3Adapter"
    assert transformers["compute_precision"] == "bfloat16"
    assert transformers["adapter_options"] == {
        "loadtime": {"max_new_tokens": max_new_tokens, "temperature": 0.1, "top_p": 1.0},
        "runtime": {},
    }


def test_loader_builds_pinned_sglang_default(catalog: _Checkpoint, monkeypatch: pytest.MonkeyPatch) -> None:
    revision, config_path = catalog.revision, catalog.config_path
    max_new_tokens, concurrency = catalog.max_new_tokens, catalog.concurrency
    engine_load = Mock()
    monkeypatch.setattr(SGLangVisionExtractAdapter, "load", engine_load)
    config = load_model_config(config_path)
    validate_pinned_revision(config)
    adapter = load_adapter(config, _MODELS, device="cuda:0")
    assert isinstance(adapter, SGLangVisionExtractAdapter)
    assert adapter._revision == revision
    assert adapter._max_seq_length == 24576
    assert adapter._max_new_tokens == max_new_tokens
    assert (adapter._temperature, adapter._top_p) == (0.1, 1.0)
    assert adapter._system_prompt is None
    assert adapter._trust_remote_code is False
    assert adapter.max_concurrent_dispatch() == concurrency
    assert adapter._extra_launch_args == _LAUNCH_ARGS
    engine_load.assert_not_called()


def test_transformers_profile_builds_profile_generation(catalog: _Checkpoint) -> None:
    model, revision, config_path = catalog.model, catalog.revision, catalog.config_path
    max_new_tokens = catalog.max_new_tokens
    config = load_model_config(config_path)
    profile = config.resolve_profile("transformers")
    assert profile.adapter_path == f"{_ADAPTER}:LightOnOCR3Adapter"
    loadtime: dict[str, Any] = dict(profile.loadtime)
    adapter = LightOnOCR3Adapter(model, revision=config.hf_revision, **loadtime)
    assert adapter._model_name_or_path == model
    assert adapter._revision == revision
    assert adapter._compute_precision == "bfloat16"
    assert adapter._output_postprocess is None
    assert adapter._generation_options({}) == {"max_new_tokens": max_new_tokens, **_GENERATION}
    assert adapter._generation_options({"max_new_tokens": 1024})["max_new_tokens"] == 1024
    with pytest.raises(ValueError, match="max_new_tokens"):
        adapter._generation_options({"max_new_tokens": max_new_tokens + 1})


def test_markdown_profile_retains_default_generation_context_and_engine_options(
    catalog: _Checkpoint,
) -> None:
    model, revision, config_path = catalog.model, catalog.revision, catalog.config_path
    max_new_tokens, concurrency = catalog.max_new_tokens, catalog.concurrency
    config = load_model_config(config_path)
    default = config.resolve_profile("default")
    markdown = config.resolve_profile("markdown")
    assert config.max_sequence_length == 24576
    assert markdown.max_batch_tokens == default.max_batch_tokens == 16384
    assert markdown.compute_precision == default.compute_precision == "bfloat16"
    assert markdown.adapter_path == default.adapter_path == f"{_SGLANG_ADAPTER}:SGLangVisionExtractAdapter"
    assert dict(markdown.runtime) == dict(default.runtime) == {"max_new_tokens": max_new_tokens, "num_beams": 1}
    assert dict(markdown.loadtime) == {
        **default.loadtime,
        "default_prompt": "grounding",
        "output_postprocess": "lightonocr3_markdown",
    }
    variants = expand_profile_variants([config])
    variant = variants[f"{model}:markdown"]
    validate_pinned_revision(variant)
    adapter = load_adapter(variant, _MODELS, device="cuda:0")
    assert isinstance(adapter, SGLangVisionExtractAdapter)
    assert adapter._revision == revision
    assert adapter._compute_precision == "bfloat16"
    assert adapter._max_seq_length == 24576
    assert adapter._output_postprocess == "lightonocr3_markdown"
    assert adapter._resolve_prompt_and_label(None, {}) == ("grounding", "markdown")
    assert adapter._max_new_tokens == max_new_tokens
    assert (adapter._temperature, adapter._top_p) == (0.1, 1.0)
    assert adapter._chat_template_kwargs == {"enable_thinking": False}
    assert adapter._allowed_instructions == frozenset({"grounding"})
    assert adapter._meter_pages is True
    assert adapter.max_concurrent_dispatch() == concurrency
    assert adapter._extra_launch_args == _LAUNCH_ARGS


def test_sglang_profiles_keep_their_l4_reservation_and_concurrency(catalog: _Checkpoint) -> None:
    profiles = yaml.safe_load(catalog.config_path.read_text())["profiles"]
    for name in ("default", "markdown"):
        loadtime = profiles[name]["adapter_options"]["loadtime"]
        assert loadtime["mem_fraction_static"] == 0.70
        assert loadtime["max_concurrent_requests"] == catalog.concurrency


def test_sglang_profiles_cap_a_page_at_5_mp_within_the_context(
    catalog: _Checkpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, config_path = catalog.model, catalog.config_path
    monkeypatch.setattr(SGLangVisionExtractAdapter, "load", Mock())
    config = load_model_config(config_path)
    variants = expand_profile_variants([config])
    for variant in (config, variants[f"{model}:markdown"]):
        adapter = load_adapter(variant, _MODELS, device="cuda:0")
        assert isinstance(adapter, SGLangVisionExtractAdapter)
        args = adapter._extra_launch_args
        assert args.count("--mm-process-config") == 1
        assert json.loads(args[args.index("--mm-process-config") + 1]) == {"image": _PAGE_PIXELS}
        assert adapter.image_token_budget == adapter.multi_image_token_budget == 5120
        assert adapter.image_token_budget + adapter._max_new_tokens < config.max_sequence_length


def test_default_routes_to_sglang_vision_extract_bundle_only(catalog: _Checkpoint) -> None:
    model = catalog.model
    assert _ADAPTER in yaml.safe_load((_BUNDLES / "transformers5.yaml").read_text())["adapters"]
    for bundle in sorted(_BUNDLES.glob("*.yaml")):
        expected = bundle.name == "sglang-vision-extract.yaml"
        assert (model in match_bundle_models(bundle, _MODELS)) == expected, bundle.name
        assert (f"{model}:markdown" in match_bundle_models(bundle, _MODELS)) == expected, bundle.name
        assert (f"{model}:transformers" in match_bundle_models(bundle, _MODELS)) == (
            bundle.name == "transformers5.yaml"
        )
