from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters.sglang import _memory
from sie_server.adapters.sglang._memory import GIB, SGLangMemoryBudget
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.adapters.sglang_vision_extract.adapter import SGLangVisionExtractAdapter
from sie_server.core.memory import MemoryManager, MemoryStats
from sie_server.core.registry import _adapter_load_required_bytes


def _write_weights(path: Path, count: int, dtype: str = "BF16", bytes_per_weight: int = 2) -> None:
    header = json.dumps(
        {"model.weight": {"dtype": dtype, "shape": [count], "data_offsets": [0, count * bytes_per_weight]}}
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header)


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128})
    )
    _write_weights(tmp_path / "model.safetensors", 4_000_000_000)
    return tmp_path


def _budget(path: Path, **kwargs) -> SGLangMemoryBudget:
    return SGLangMemoryBudget(str(path), None, "bfloat16", 32768, kv_budget_tokens=32768, **kwargs)


def test_cached_partial_snapshot_uses_pinned_config_without_requiring_repository_metadata(model_dir, tmp_path):
    revision = "a" * 40
    cache = tmp_path / "cache"
    snapshot = cache / "models--example--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for filename in ("config.json", "model.safetensors"):
        (snapshot / filename).symlink_to(model_dir / filename)
    with patch("sie_sdk.cache.get_cache_config", return_value=MagicMock(local_cache=cache)):
        assert _memory._model_directory("example/model", revision) == snapshot
        budget = SGLangMemoryBudget("example/model", revision, "bfloat16", 32768)
        assert budget.static_bytes == _budget(model_dir).static_bytes


def test_missing_cached_config_fails_before_launch(tmp_path):
    with patch("sie_sdk.cache.get_cache_config", return_value=MagicMock(local_cache=tmp_path)):
        with pytest.raises(ValueError, match="cached model config"):
            _memory._model_directory("example/model", "a" * 40)


def test_reservation_is_independent_of_device_size_and_tracks_precision(model_dir):
    budget = _budget(model_dir)
    expected = math.ceil(8_000_000_000 * 1.05) + (32768 + 512) * 147456 + 3 * GIB
    for total in (24 * GIB, 80 * GIB):
        assert budget.required_bytes(device_type="cuda", device_total_bytes=total) == expected
    assert budget.required_bytes(device_type="cpu", device_total_bytes=80 * GIB) is None
    fp32 = SGLangMemoryBudget(str(model_dir), None, "float32", 32768, kv_budget_tokens=32768)
    assert fp32.static_bytes - GIB == 2 * (budget.static_bytes - GIB)


def test_second_engine_uses_current_free_memory_without_shrinking_kv(model_dir, monkeypatch):
    budget = _budget(model_dir)
    fractions = []
    for free in (79 * GIB, 48 * GIB):
        monkeypatch.setattr(_memory, "device_memory", lambda _, free=free: (free, 80 * GIB))
        args = budget.launch_args(0)
        fraction = float(args[1])
        fractions.append(fraction)
        assert fraction * free == pytest.approx(budget.static_bytes)
        assert args[2:] == ["--max-total-tokens", "32768"]
    assert fractions[1] > fractions[0]


def test_insufficient_free_memory_fails_before_engine_launch(model_dir, monkeypatch):
    budget = _budget(model_dir)
    required = budget.required_bytes(device_type="cuda", device_total_bytes=80 * GIB)
    monkeypatch.setattr(_memory, "device_memory", lambda _: (required - 1, 80 * GIB))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        budget.launch_args(0)


def test_explicit_fraction_keeps_admission_and_unbounded_kv_behavior(monkeypatch):
    budget = _budget(Path("missing-model"), mem_fraction_static=0.85)
    monkeypatch.setattr(_memory, "device_memory", MagicMock(side_effect=AssertionError("must not query CUDA")))
    assert budget.required_bytes(device_type="cuda", device_total_bytes=80 * GIB) == 69 * GIB
    assert budget.launch_args(0) == ["--mem-fraction-static", "0.85"]


def test_sharded_weights_use_index_and_serving_precision(model_dir):
    (model_dir / "model.safetensors").unlink()
    _write_weights(model_dir / "model-1.safetensors", 1000, "F32", 4)
    _write_weights(model_dir / "model-2.safetensors", 2000)
    _write_weights(model_dir / "unused.safetensors", 9000)
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors", "c": "model-2.safetensors"}})
    )
    assert _memory._weight_bytes(model_dir, 2) == 6000


def test_kv_geometry_uses_explicit_head_dim_and_nested_text_config():
    config = {
        "text_config": {
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "hidden_size": 1024,
            "num_key_value_heads": 8,
            "head_dim": 128,
        }
    }
    assert _memory._kv_bytes_per_token(config, 2) == 114688
    del config["text_config"]["head_dim"]
    assert _memory._kv_bytes_per_token(config, 2) == 57344


@pytest.mark.parametrize(
    "options",
    [
        {"linear_num_key_heads": 8},
        {"kv_lora_rank": 512},
        {"quantization_config": {"quant_method": "fp8"}},
        {"layer_types": ["linear_attention"]},
    ],
)
def test_nonstandard_cache_layout_requires_explicit_override(options):
    with pytest.raises(ValueError, match="explicit mem_fraction_static"):
        _memory._kv_bytes_per_token(options, 2)


@pytest.mark.parametrize("kwargs", [{"tensor_parallel_size": 2}, {"has_auxiliary_weights": True}])
def test_auxiliary_allocations_require_explicit_override(model_dir, kwargs):
    with pytest.raises(ValueError, match="explicit mem_fraction_static"):
        _budget(model_dir, **kwargs).required_bytes(device_type="cuda", device_total_bytes=80 * GIB)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.1, True, float("nan"), float("inf")])
def test_invalid_fraction_rejected(fraction):
    with pytest.raises(ValueError, match="mem_fraction_static"):
        _budget(Path("missing"), mem_fraction_static=fraction)


@pytest.mark.parametrize("adapter_cls", [SGLangGenerationAdapter, SGLangEmbeddingAdapter, SGLangVisionExtractAdapter])
def test_adapters_use_the_same_budget_for_staging_and_residency(model_dir, adapter_cls):
    adapter = adapter_cls(str(model_dir), max_seq_length=32768, kv_budget_tokens=32768)
    assert adapter.memory_footprint() == 0
    required = adapter.load_required_memory_bytes(device_type="cuda", device_total_bytes=80 * GIB)
    adapter._process = MagicMock()
    assert adapter.memory_footprint() == required
    adapter._process = None
    assert adapter.memory_footprint() == 0


def test_real_pressure_decision_uses_the_sglang_reservation(model_dir):
    adapter = SGLangGenerationAdapter(str(model_dir), kv_budget_tokens=32768)
    manager = MemoryManager(device="cuda:0")
    with patch.object(manager, "get_stats", return_value=MemoryStats(30 * GIB, 80 * GIB, "cuda")):
        required = _adapter_load_required_bytes(adapter, manager)
        assert required is not None
        assert not manager.should_evict_for_load(required)
    with patch.object(manager, "get_stats", return_value=MemoryStats(80 * GIB - required + 1, 80 * GIB, "cuda")):
        assert manager.should_evict_for_load(required)


def test_embedding_and_extraction_default_to_context_budget(model_dir):
    for adapter_cls in (SGLangEmbeddingAdapter, SGLangVisionExtractAdapter):
        adapter = adapter_cls(str(model_dir), max_seq_length=8192)
        assert adapter._memory_budget.token_capacity == 8192


def test_non_aligned_budget_rounds_up():
    budget = SGLangMemoryBudget("model", None, "bfloat16", 8192, kv_budget_tokens=8193)
    assert budget.token_capacity == 8704


@pytest.mark.parametrize("flag", ["--max-total-tokens", "--mem-fraction-static=0.85"])
def test_raw_memory_flags_cannot_override_automatic_reservation(flag):
    with pytest.raises(ValueError, match="instead of extra_launch_args"):
        SGLangGenerationAdapter("model", extra_launch_args=[flag])
