"""Declared sources, preload inheritance, and one resolved engine capacity."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from sie_server.config.model import ModelConfig
from sie_server.core.loader import expand_profile_variants, load_adapter
from sie_server.core.model_loader import ModelLoader


def config_values():
    return {
        "sie_id": "test/model",
        "hf_id": "test/model",
        "inputs": {"text": True},
        "tasks": {"generate": {"context_length": 1024, "max_output_tokens": 64}},
        "profiles": {
            "default": {
                "adapter_path": "sie_server.adapters.sglang.generation:SGLangGenerationAdapter",
                "max_batch_tokens": 1024,
                "kv_budget_tokens": 1024,
                "adapter_options": {
                    "loadtime": {
                        "lora_paths": {"a": {"id": "org/a", "revision": "a" * 40}, "other-a": "org/a"},
                        "preload_loras": [],
                        "mem_fraction_static": 0.5,
                        "max_lora_rank": 8,
                        "lora_target_modules": ["q_proj"],
                        "max_loras_per_batch": 1,
                    }
                },
            }
        },
    }


def test_loader_preserves_source_revision_and_uses_explicit_capacity(monkeypatch):
    config = ModelConfig.model_validate(config_values())
    monkeypatch.setenv("SIE_MAX_LORAS_PER_MODEL", "20")
    loader = ModelLoader(
        preprocessor_registry=Mock(), postprocessor_registry=Mock(), all_configs={}, max_loras_per_model=2
    )
    adapter = loader.instantiate_adapter("test/model", config, Path(), "cuda:0")
    assert adapter._lora.revisions == {"org/a": "a" * 40}
    assert adapter._lora.max_loaded == 2
    assert adapter._lora.engine_name("a") == adapter._lora.engine_name("other-a")
    args = adapter._lora.launch_args()
    assert args[args.index("--max-loaded-loras") + 1] == "2"


def test_preload_selector_uses_inherited_source_map():
    values = config_values()
    values["profiles"]["default"]["adapter_options"]["loadtime"]["preload_loras"] = ["a"]
    values["profiles"]["warm"] = {"extends": "default"}
    config = ModelConfig.model_validate(values)
    variant = expand_profile_variants([config])["test/model:warm"]
    adapter = load_adapter(variant, Path(), device="cuda:0")
    assert adapter._lora.preloaded == frozenset({"org/a"})
    assert adapter._lora.revisions["org/a"] == "a" * 40


@pytest.mark.parametrize("selector", [None, "a", ["missing"], ["a", "a"], [1], {}])
def test_invalid_preload_selector_fails_at_config_load(selector):
    values = config_values()
    values["profiles"]["default"]["adapter_options"]["loadtime"]["preload_loras"] = selector
    with pytest.raises(ValueError, match="preload_loras"):
        ModelConfig.model_validate(values)


def test_two_aliases_cannot_pin_one_source_to_different_revisions():
    values = config_values()
    paths = values["profiles"]["default"]["adapter_options"]["loadtime"]["lora_paths"]
    paths["other-a"] = {"id": "org/a", "revision": "b" * 40}
    with pytest.raises(ValueError, match="two different revisions"):
        ModelConfig.model_validate(values)
