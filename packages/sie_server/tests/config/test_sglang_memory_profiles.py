from pathlib import Path

import pytest
from sie_server.core.loader import expand_profile_variants, load_adapter, load_model_config

MODELS = Path(__file__).resolve().parents[2] / "models"


@pytest.mark.parametrize(
    "name",
    [
        "Qwen__Qwen3Guard-Gen-4B",
        "Qwen__Qwen3Guard-Gen-0.6B",
        "ibm-granite__granite-guardian-3.0-2b",
        "Qwen__Qwen3-Embedding-4B",
        "lightonai__LightOnOCR-2-1B",
    ],
)
def test_default_profiles_use_model_sized_memory(name):
    config = load_model_config(MODELS / f"{name}.yaml")
    profile = config.resolve_profile("default")
    assert "mem_fraction_static" not in profile.loadtime
    adapter = load_adapter(config, MODELS, device="cuda:0")
    assert adapter._memory_budget.mem_fraction_static is None
    assert adapter._memory_budget.kv_budget_tokens == profile.kv_budget_tokens


def test_guard_calibrated_profile_retains_explicit_override():
    config = load_model_config(MODELS / "Qwen__Qwen3Guard-Gen-4B.yaml")
    assert config.resolve_profile("default").kv_budget_tokens == 32768
    variant = expand_profile_variants([config])[config.sie_id + ":a100-40gb"]
    adapter = load_adapter(variant, MODELS, device="cuda:0")
    assert adapter._memory_budget.mem_fraction_static == 0.85
    assert adapter._memory_budget.kv_budget_tokens == 45056


def test_extending_profile_forwards_its_resolved_kv_budget():
    config = load_model_config(MODELS / "Qwen__Qwen3Guard-Gen-4B.yaml")
    values = config.model_dump()
    values["profiles"]["larger"] = {"extends": "default", "kv_budget_tokens": 65536}
    config = type(config).model_validate(values)
    assert config.resolve_profile("default").kv_budget_tokens == 32768
    variant = expand_profile_variants([config])[config.sie_id + ":larger"]
    adapter = load_adapter(variant, MODELS, device="cuda:0")
    assert adapter._memory_budget.mem_fraction_static is None
    assert adapter._memory_budget.kv_budget_tokens == 65536


@pytest.mark.parametrize("name", ["Qwen__Qwen3.5-4B", "lightonai__LightOnOCR-3-4B", "Qwen__Qwen3-VL-8B-Instruct"])
def test_specialized_profiles_retain_calibrated_reservation(name):
    config = load_model_config(MODELS / f"{name}.yaml")
    assert config.resolve_profile("default").loadtime["mem_fraction_static"] == 0.85
