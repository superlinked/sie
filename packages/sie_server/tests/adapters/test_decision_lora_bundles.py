"""Explicit LoRA profiles must route to a runtime that advertises their module."""

from pathlib import Path

import pytest
import yaml
from sie_config.model_registry import ModelRegistry as ConfigRegistry
from sie_server.api.ws import bundle_model_is_supported
from sie_server.config.model import ModelConfig
from sie_server.core.registry import ModelRegistry

BUNDLES = Path(__file__).resolve().parents[2] / "bundles"


@pytest.mark.parametrize(
    ("adapter_path", "bundle", "task"),
    [
        ("sie_server.adapters.laya.runtime_lora:LayaLoraAdapter", "default", "extract"),
        ("sie_server.adapters.gliclass.runtime_lora:GLiClassLoraAdapter", "default", "extract"),
        ("sie_server.adapters.gliner2.decide_lora:GLiNER2DecideLoraAdapter", "transformers5", "extract"),
        ("sie_server.adapters.pplx_decider.runtime_lora:PerplexityDecisionLoraAdapter", "transformers5", "extract"),
        ("sie_server.adapters.gliner2.entity_lora:GLiNER2EntityLoraAdapter", "default", "extract"),
        ("sie_server.adapters.cross_encoder.runtime_lora:CrossEncoderLoraAdapter", "default", "score"),
    ],
)
def test_runtime_lora_profile_routes_and_is_accepted_only_by_its_bundle(tmp_path, adapter_path, bundle, task):
    payload = {
        "sie_id": "local-lora",
        "hf_id": "local-model",
        "tasks": {task: {}},
        "profiles": {
            "default": {
                "adapter_path": adapter_path,
                "max_batch_tokens": 8192,
                "adapter_options": {"loadtime": {"lora_paths": {"a": "example/adapter"}}},
            }
        },
    }
    (tmp_path / "model.yaml").write_text(yaml.safe_dump(payload))
    configurations = ConfigRegistry(BUNDLES, tmp_path)
    assert configurations.unrouteable_models == {}
    assert configurations.resolve_bundle("local-lora") == bundle
    registry = ModelRegistry()
    registry._configs["local-lora"] = ModelConfig.model_validate(payload)
    assert bundle_model_is_supported(registry, bundle, "local-lora")
    other = "transformers5" if bundle == "default" else "default"
    assert not bundle_model_is_supported(registry, other, "local-lora")
