import json
import shutil
import struct
from pathlib import Path

import pytest
from sie_server.adapters.sglang import _memory
from sie_server.core.grammar_routing import resolve_grammar_serving_model
from sie_server.core.loader import expand_profile_variants, load_adapter, load_model_config
from sie_server.core.registry import ModelRegistry

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


# Shared-GPU profiles: model-sized instead of a fixed fraction of the card.
# Fixtures hold each pinned checkpoint's config and its safetensors headers
# aggregated into the categories the sizing distinguishes.
_FIXTURES = Path(__file__).resolve().parents[1] / "adapters" / "fixtures" / "sglang_memory"
_SHARED = {
    # name: (resident GiB, GiB free to load, KV budget, pinned pool flags)
    "Qwen__Qwen3.8-27B-FP8": (39.63, 0.0, 16384, ("--max-mamba-cache-size", "40")),
    "google__gemma-4-31B-it": (42.89, 62.16, 8192, ("--swa-full-tokens-ratio", "1.0")),
    "Qwen__Qwen3.6-27B": (59.97, 0.0, 16384, ("--max-mamba-cache-size", "16")),
}
_FIXED_FRACTIONS = {
    "Qwen__Qwen3.8-27B-FP8": {"default": 0.85, "no-spec": 0.85},
    "google__gemma-4-31B-it": {"default": 0.80, "no-spec": 0.80},
    "Qwen__Qwen3.6-27B": {"default": 0.95, "no-spec": 0.85},
}


def _checkpoint(name: str, directory: Path) -> Path:
    fixture = json.loads((_FIXTURES / f"{name}.json").read_text())
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(fixture["config"]))
    header, offset = {}, 0
    for tensor in fixture["tensors"]:
        header[tensor["name"]] = {
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
            "data_offsets": [offset, offset + tensor["bytes"]],
        }
        offset += tensor["bytes"]
    encoded = json.dumps(header).encode()
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded)
    return directory


@pytest.mark.parametrize("name", sorted(_SHARED))
def test_shared_profiles_size_from_the_pinned_checkpoint(name, tmp_path):
    resident, load, kv_budget, pool_flags = _SHARED[name]
    config = load_model_config(MODELS / f"{name}.yaml")
    fixture = json.loads((_FIXTURES / f"{name}.json").read_text())
    assert (fixture["hf_id"], fixture["hf_revision"]) == (config.hf_id, config.hf_revision)
    profile = config.resolve_profile("shared")
    assert "mem_fraction_static" not in profile.loadtime
    assert profile.kv_budget_tokens == kv_budget
    assert profile.loadtime["speculative"] == {"enabled": False}
    variant = expand_profile_variants([config])[f"{config.sie_id}:shared"]
    budget = load_adapter(variant, MODELS, device="cuda:0")._memory_budget
    assert budget.mem_fraction_static is None
    assert budget.kv_budget_tokens == kv_budget
    budget.model_name_or_path = str(_checkpoint(name, tmp_path / "checkpoint"))
    required = budget.required_bytes(device_type="cuda", device_total_bytes=80 * _memory.GIB)
    assert budget.reserved_bytes / _memory.GIB == pytest.approx(resident, abs=0.01)
    assert budget.reservation.load_bytes / _memory.GIB == pytest.approx(load, abs=0.01)
    assert required == max(budget.reserved_bytes, budget.reservation.load_bytes)
    assert budget.reservation.launch_args == pool_flags
    assert resident <= 60


@pytest.mark.parametrize("name", sorted(_SHARED))
def test_shared_profiles_leave_the_defaults_alone(name):
    config = load_model_config(MODELS / f"{name}.yaml")
    for profile, fraction in _FIXED_FRACTIONS[name].items():
        assert config.resolve_profile(profile).loadtime["mem_fraction_static"] == fraction
    alias = config.resolve_profile("shared-no-spec")
    assert alias.grammar_profile == "shared"
    assert alias.model_copy(update={"grammar_profile": None}) == config.resolve_profile("shared")


@pytest.mark.parametrize("name", sorted(_SHARED))
@pytest.mark.parametrize("model_filter", [None, "shared"])
def test_grammar_requests_stay_on_the_shared_engine(name, model_filter, tmp_path):
    shutil.copy(MODELS / f"{name}.yaml", tmp_path / f"{name}.yaml")
    sie_id = load_model_config(MODELS / f"{name}.yaml").sie_id
    registry = ModelRegistry(
        models_dir=tmp_path,
        model_filter=None if model_filter is None else [f"{sie_id}:{model_filter}"],
        enable_hot_reload=False,
    )
    assert resolve_grammar_serving_model(registry, f"{sie_id}:shared") == f"{sie_id}:shared"
