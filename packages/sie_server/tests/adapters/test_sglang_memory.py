from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.adapters.sglang import _memory
from sie_server.adapters.sglang._memory import GIB, LoraCapacity, SGLangMemoryBudget
from sie_server.adapters.sglang.cuda13 import SGLangCuda13Adapter
from sie_server.adapters.sglang.embedding import SGLangEmbeddingAdapter
from sie_server.adapters.sglang.generation import SGLangGenerationAdapter
from sie_server.adapters.sglang_vision_extract.adapter import SGLangVisionExtractAdapter
from sie_server.core.memory import MemoryManager, MemoryStats
from sie_server.core.registry import _adapter_load_required_bytes

# Qwen3.5-family Gated DeltaNet hybrid geometry (Qwen3.6-27B, Qwen3.8-27B-FP8).
_GDN = {
    "num_hidden_layers": 64,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "mamba_ssm_dtype": "float32",
}
# Bytes of one recurrent-state slot over the 48 linear layers.
_GDN_SLOT_FP32 = 48 * (10240 * 3 * 2 + 48 * 128 * 128 * 4)
_GDN_SLOT_BF16 = 48 * (10240 * 3 * 2 + 48 * 128 * 128 * 2)
# Gemma 4 31B: 50 sliding-window layers (16 KV heads x 256) and 10 global
# layers (4 KV heads x 512).
_GEMMA4 = {
    "num_hidden_layers": 60,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    "num_attention_heads": 32,
    "num_key_value_heads": 16,
    "head_dim": 256,
    "num_global_key_value_heads": 4,
    "global_head_dim": 512,
    "sliding_window": 1024,
}
_GEMMA4["layer_types"] = _GEMMA4["layer_types"] * 10


def _write_tensors(path: Path, tensors: dict[str, tuple[str, list[int], int]]) -> None:
    header, offset = {}, 0
    for name, (dtype, shape, size) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)


def _write_weights(path: Path, count: int, dtype: str = "BF16", bytes_per_weight: int = 2) -> None:
    _write_tensors(path, {"model.weight": (dtype, [count], count * bytes_per_weight)})


def _model(tmp_path: Path, config: dict, tensors: dict[str, tuple[str, list[int], int]] | None = None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(config))
    _write_tensors(tmp_path / "model.safetensors", tensors or {"model.weight": ("BF16", [1000], 2000)})
    return tmp_path


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128})
    )
    _write_weights(tmp_path / "model.safetensors", 4_000_000_000)
    return tmp_path


def _budget(path: Path, **kwargs) -> SGLangMemoryBudget:
    return SGLangMemoryBudget(str(path), None, "bfloat16", 32768, kv_budget_tokens=32768, **kwargs)


def _kv(path: Path, **kwargs) -> int:
    """Reserved bytes beyond the weights and the startup allowance."""
    budget = _budget(path, **kwargs)
    weights, _ = _memory._weight_bytes(path, 2)
    return budget.static_bytes - math.ceil(weights * _memory.WEIGHT_ALLOWANCE) - GIB


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
    # 2,048 running requests by default, one int32 row of context + 4 each.
    request_table = 2049 * (32768 + 4) * 4
    expected = math.ceil(8_000_000_000 * 1.05) + (32768 + 512) * 147456 + 3 * GIB + request_table
    for total in (24 * GIB, 80 * GIB):
        assert budget.required_bytes(device_type="cuda", device_total_bytes=total) == expected
    assert budget.reserved_bytes == expected
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
    assert _memory._weight_bytes(model_dir, 2) == (6000, 6000)


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


def test_fp8_checkpoint_keeps_its_stored_bytes_and_float32_scales(tmp_path):
    config = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
    }
    path = _model(
        tmp_path,
        config,
        {
            "model.layers.0.mlp.down_proj.weight": ("F8_E4M3", [1000, 100], 100_000),
            "model.layers.0.mlp.down_proj.weight_scale_inv": ("BF16", [8, 1], 16),
            "model.embed_tokens.weight": ("BF16", [500, 100], 100_000),
            "mtp.layers.0.mlp.down_proj.weight": ("F8_E4M3", [1000, 100], 100_000),
        },
    )
    weights = 100_000 + 8 * 4 + 100_000
    assert _memory._weight_bytes(path, 2, serialized_quantization=True) == (weights, weights)
    budget = _budget(path)
    assert budget.reservation.load_bytes == 0
    # FP8 weights leave the KV cache at BF16: the checkpoint declares no KV scheme.
    assert budget.static_bytes == math.ceil(weights * 1.05) + (32768 + 512) * 2 * 2 * (64 + 64) * 2 + GIB


@pytest.mark.parametrize("method", ["awq", "gptq", "bitsandbytes", "compressed-tensors"])
def test_other_quantized_checkpoints_require_explicit_override(tmp_path, method):
    config = {"num_hidden_layers": 1, "num_attention_heads": 1, "head_dim": 8}
    path = _model(tmp_path, {**config, "quantization_config": {"quant_method": method}})
    with pytest.raises(ValueError, match=f"{method} checkpoints require an explicit mem_fraction_static"):
        _ = _budget(path).reservation


def test_online_fp8_reserves_quantized_weights_and_admits_the_unquantized_peak(tmp_path):
    config = {"num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64}
    path = _model(
        tmp_path,
        config,
        {
            "model.layers.0.mlp.down_proj.weight": ("BF16", [1000, 1000], 2_000_000),
            "model.layers.0.input_layernorm.weight": ("BF16", [1000], 2000),
            "model.embed_tokens.weight": ("BF16", [1000, 1000], 2_000_000),
            "model.visual.blocks.0.mlp.fc1.weight": ("BF16", [1000, 1000], 2_000_000),
        },
    )
    assert _memory._weight_bytes(path, 2, online_quantization=True) == (1_000_000 + 2000 + 4_000_000, 6_002_000)
    budget = _budget(path, launch_options=("--quantization", "fp8"))
    reservation = budget.reservation
    assert reservation.load_bytes == math.ceil(6_002_000 * 1.05) + GIB
    resident = reservation.static_bytes + reservation.runtime_bytes
    required = budget.required_bytes(device_type="cuda", device_total_bytes=80 * GIB)
    assert required == max(resident, reservation.load_bytes)
    # Residency accounting holds only the post-quantization reservation.
    assert budget.reserved_bytes == resident


def test_unsupported_online_quantization_requires_explicit_override(model_dir):
    with pytest.raises(ValueError, match="--quantization awq requires an explicit mem_fraction_static"):
        _ = _budget(model_dir, launch_options=("--quantization=awq",)).reservation


def test_mtp_weights_are_skipped_without_a_draft(tmp_path):
    path = _model(
        tmp_path,
        {"num_hidden_layers": 1, "num_attention_heads": 1, "head_dim": 8},
        {"model.weight": ("BF16", [100], 200), "mtp.layers.0.weight": ("BF16", [100], 200)},
    )
    assert _memory._weight_bytes(path, 2) == (200, 200)
    assert _memory._weight_bytes(path, 2, include_mtp=True) == (400, 400)


@pytest.mark.parametrize(
    ("args", "quantization_config", "element"),
    [
        ((), None, 2),
        (("--kv-cache-dtype", "fp8_e4m3"), None, 1),
        (("--kv-cache-dtype=fp8_e5m2",), None, 1),
        (("--kv-cache-dtype", "bfloat16"), None, 2),
        ((), {"quant_method": "fp8"}, 2),
        ((), {"quant_method": "fp8", "kv_cache_quant_algo": "FP8"}, 1),
    ],
)
def test_kv_cache_dtype_follows_the_flag_and_the_checkpoint(args, quantization_config, element):
    config = {"quantization_config": quantization_config} if quantization_config else {}
    assert _memory._kv_element_bytes(config, args, 2) == element


@pytest.mark.parametrize("dtype", ["fp4_e2m1", "mxfp8", "nvfp4"])
def test_scaled_kv_cache_formats_require_explicit_override(dtype):
    with pytest.raises(ValueError, match=f"--kv-cache-dtype {dtype} requires an explicit mem_fraction_static"):
        _memory._kv_element_bytes({}, ("--kv-cache-dtype", dtype), 2)


def test_hybrid_linear_attention_caches_kv_only_on_full_attention_layers():
    assert _memory._kv_bytes_per_token(_GDN, 2) == 16 * 4 * (256 + 256) * 2
    without_types = {key: value for key, value in _GDN.items() if key != "layer_types"}
    without_types["full_attention_interval"] = 4
    assert _memory._kv_bytes_per_token(without_types, 2) == 16 * 4 * (256 + 256) * 2


def test_gdn_state_slot_matches_the_engine_formula():
    assert _memory._linear_state_bytes(_GDN, 48, (), {}) == _GDN_SLOT_FP32 == 153_944_064
    assert _memory._linear_state_bytes(_GDN, 48, ("--mamba-ssm-dtype", "bfloat16"), {}) == _GDN_SLOT_BF16
    # The flag wins over the environment, which wins over the checkpoint.
    env = {"SGLANG_MAMBA_SSM_DTYPE": "bfloat16"}
    assert _memory._linear_state_bytes(_GDN, 48, (), env) == _GDN_SLOT_BF16
    assert _memory._linear_state_bytes(_GDN, 48, ("--mamba-ssm-dtype", "float32"), env) == _GDN_SLOT_FP32
    assert _memory._linear_state_bytes({**_GDN, "mamba_ssm_dtype": "bfloat16"}, 48, (), {}) == _GDN_SLOT_BF16


@pytest.mark.parametrize(
    ("args", "strategy_flag", "slots_per_request"),
    [
        ((), "--mamba-scheduler-strategy", 3),
        (("--mamba-scheduler-strategy", "extra_buffer"), "--mamba-scheduler-strategy", 5),
        (
            ("--mamba-scheduler-strategy", "extra_buffer", "--disable-overlap-schedule"),
            "--mamba-scheduler-strategy",
            4,
        ),
        ((), "--mamba-radix-cache-strategy", 5),
        (("--mamba-radix-cache-strategy", "extra_buffer_lazy"), "--mamba-radix-cache-strategy", 4),
        (("--mamba-radix-cache-strategy", "no_buffer"), "--mamba-radix-cache-strategy", 3),
        (("--disable-radix-cache",), "--mamba-radix-cache-strategy", 1),
    ],
)
def test_hybrid_state_pool_is_pinned_for_the_running_requests(tmp_path, args, strategy_flag, slots_per_request):
    path = _model(tmp_path, _GDN)
    budget = _budget(path, launch_options=(*args, "--max-running-requests", "8"), mamba_strategy_flag=strategy_flag)
    slots = 8 * slots_per_request
    assert budget.reservation.launch_args == ("--max-mamba-cache-size", str(slots))
    kv = (32768 + 512) * 16 * 4 * 512 * 2
    assert _kv(path, launch_options=(*args, "--max-running-requests", "8"), mamba_strategy_flag=strategy_flag) == (
        kv + (slots + 1) * _GDN_SLOT_FP32
    )


def test_hybrid_state_pool_honours_an_explicit_slot_count(tmp_path):
    path = _model(tmp_path, _GDN)
    budget = _budget(path, launch_options=("--max-mamba-cache-size", "10"))
    assert budget.reservation.launch_args == ()
    assert _kv(path, launch_options=("--max-mamba-cache-size", "10")) == (32768 + 512) * 65536 + 11 * _GDN_SLOT_FP32


def test_hybrid_state_pool_requires_a_running_request_bound(tmp_path):
    with pytest.raises(ValueError, match="--max-running-requests"):
        _ = _budget(_model(tmp_path, _GDN)).reservation


def test_unknown_mamba_strategy_requires_explicit_override(tmp_path):
    args = ("--mamba-radix-cache-strategy", "experimental", "--max-running-requests", "8")
    budget = _budget(_model(tmp_path, _GDN), launch_options=args, mamba_strategy_flag="--mamba-radix-cache-strategy")
    with pytest.raises(ValueError, match="experimental requires an explicit mem_fraction_static"):
        _ = budget.reservation


def test_sliding_window_hybrid_reserves_both_pools_and_pins_their_ratio(tmp_path):
    full = 10 * 4 * (512 + 512) * 2
    sliding = 50 * 16 * (256 + 256) * 2
    assert _memory._kv_bytes_per_token(_GEMMA4, 2) == full + sliding == 901_120
    assert _memory._kv_bytes_per_token(_GEMMA4, 1) == 450_560
    path = _model(tmp_path, {"text_config": _GEMMA4})
    assert _budget(path).reservation.launch_args == ("--swa-full-tokens-ratio", "1.0")
    assert _kv(path) == (32768 + 512) * 901_120
    explicit = _budget(path, launch_options=("--swa-full-tokens-ratio", "0.5"))
    assert explicit.reservation.launch_args == ()


def test_uniform_sliding_window_geometry_counts_every_layer():
    gemma3 = {
        "num_hidden_layers": 6,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
    }
    assert _memory._kv_bytes_per_token(gemma3, 2) == 6 * 4 * 512 * 2


def test_kv_shared_layers_still_get_their_own_buffers():
    assert _memory._kv_bytes_per_token({**_GEMMA4, "num_kv_shared_layers": 20}, 2) == 901_120


def test_latent_attention_caches_one_latent_per_layer():
    mla = {"num_hidden_layers": 61, "num_attention_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    assert _memory._kv_bytes_per_token(mla, 2) == (512 + 64) * 61 * 2


@pytest.mark.parametrize(
    "options",
    [
        {"mamba_d_state": 128},
        {"kv_lora_rank": 512, "qk_rope_head_dim": 64, "index_topk": 2048},
        {"layer_types": ["chunked_attention"]},
    ],
)
def test_nonstandard_cache_layout_requires_explicit_override(options):
    config = {"num_hidden_layers": 1, "num_attention_heads": 8, "head_dim": 64, **options}
    with pytest.raises(ValueError, match="explicit mem_fraction_static"):
        _memory._kv_bytes_per_token(config, 2)


@pytest.mark.parametrize("kwargs", [{"tensor_parallel_size": 2}, {"speculative": True}])
def test_auxiliary_allocations_require_explicit_override(model_dir, kwargs):
    with pytest.raises(ValueError, match="explicit mem_fraction_static"):
        _budget(model_dir, **kwargs).required_bytes(device_type="cuda", device_total_bytes=80 * GIB)


def _llama_projections(tmp_path: Path) -> Path:
    config = {"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}
    tensors = {}
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        for name, shape in {
            "self_attn.q_proj": [4096, 4096],
            "self_attn.k_proj": [1024, 4096],
            "self_attn.v_proj": [1024, 4096],
            "self_attn.o_proj": [4096, 4096],
            "mlp.gate_proj": [14336, 4096],
            "mlp.up_proj": [14336, 4096],
            "mlp.down_proj": [4096, 14336],
        }.items():
            tensors[f"{prefix}.{name}.weight"] = ("BF16", shape, 2)
    return _model(tmp_path, config, tensors)


def test_lora_pool_matches_the_engine_buffers(tmp_path):
    path = _llama_projections(tmp_path)
    modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    capacity = LoraCapacity(max_per_batch=8, max_rank=64, target_modules=modules)
    # A (slots, stacked * rank, in) plus B (slots, out, rank) per module and layer.
    widths = (3 * 4096 + 6144) + (4096 + 4096) + (2 * 4096 + 2 * 14336) + (14336 + 4096)
    assert _memory._lora_pool_bytes(path, 32, capacity, 2) == 2 * 8 * 64 * 32 * widths
    attention = LoraCapacity(max_per_batch=8, max_rank=64, target_modules=("q_proj", "v_proj"))
    assert _memory._lora_pool_bytes(path, 32, attention, 2) == 2 * 8 * 64 * 32 * (3 * 4096 + 6144)
    budget = _budget(path, lora=capacity)
    assert budget.static_bytes == _budget(path).static_bytes + 2 * 8 * 64 * 32 * widths


def test_lora_reuses_k_for_a_missing_v_projection(tmp_path):
    config = {"num_hidden_layers": 1, "num_attention_heads": 4, "head_dim": 64}
    path = _model(
        tmp_path,
        config,
        {
            "model.layers.0.self_attn.q_proj.weight": ("BF16", [2048, 1024], 2),
            "model.layers.0.self_attn.k_proj.weight": ("BF16", [512, 1024], 2),
        },
    )
    capacity = LoraCapacity(max_per_batch=1, max_rank=1, target_modules=("q_proj",))
    assert _memory._lora_pool_bytes(path, 1, capacity, 2) == 2 * (3 * 1024 + 2048 + 512 + 512)


@pytest.mark.parametrize(
    ("capacity", "match"),
    [
        (LoraCapacity(8, None, ("q_proj",)), "max_lora_rank"),
        (LoraCapacity(8, 16, None), "lora_target_modules"),
        (LoraCapacity(8, 16, ("all",)), "lora_target_modules"),
        (LoraCapacity(8, 16, ("embed_tokens",)), "embed_tokens"),
    ],
)
def test_undeclared_lora_capacity_requires_explicit_override(tmp_path, capacity, match):
    with pytest.raises(ValueError, match=match):
        _memory._lora_pool_bytes(_llama_projections(tmp_path), 32, capacity, 2)


def test_lora_on_experts_requires_explicit_override(tmp_path):
    path = _model(
        tmp_path,
        {"num_hidden_layers": 1, "num_attention_heads": 1, "head_dim": 8},
        {"model.layers.0.mlp.experts.0.down_proj.weight": ("BF16", [8, 8], 2)},
    )
    with pytest.raises(ValueError, match="mixture-of-experts"):
        _memory._lora_pool_bytes(path, 1, LoraCapacity(1, 1, ("down_proj",)), 2)


def test_request_table_follows_the_running_request_bound():
    assert _memory._request_table_bytes((), 16384, 8192) == 2049 * 8196 * 4
    assert _memory._request_table_bytes(("--max-running-requests", "8"), 16384, 8192) == 9 * 8196 * 4
    # Small pools admit at most half their tokens as requests.
    assert _memory._request_table_bytes((), 1024, 8192) == 513 * 8196 * 4


def test_launch_args_pin_the_reserved_pools(tmp_path, monkeypatch):
    path = _model(tmp_path, _GDN)
    budget = _budget(path, launch_options=("--max-running-requests", "2"))
    monkeypatch.setattr(_memory, "device_memory", lambda _: (79 * GIB, 80 * GIB))
    args = budget.launch_args(0)
    assert args[2:] == ["--max-total-tokens", "32768", "--max-mamba-cache-size", "6"]


def test_flag_value_reads_both_spellings_and_the_last_occurrence():
    args = ("--kv-cache-dtype", "bf16", "--quantization=fp8", "--kv-cache-dtype=fp8_e4m3")
    assert _memory.flag_value(args, "--kv-cache-dtype") == "fp8_e4m3"
    assert _memory.flag_value(args, "--quantization") == "fp8"
    assert _memory.flag_value(args, "--page-size") is None
    with pytest.raises(ValueError, match="positive integer --max-running-requests"):
        _memory._flag_int(("--max-running-requests", "many"), "--max-running-requests")


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


def test_generation_adapters_size_from_their_launch_shape(model_dir):
    args = ["--kv-cache-dtype", "fp8_e4m3", "--max-running-requests", "4"]
    adapter = SGLangCuda13Adapter(str(model_dir), extra_launch_args=args, extra_env={"A": "1"})
    budget = adapter._memory_budget
    assert budget.launch_options == tuple(args)
    assert budget.launch_env == {"A": "1"}
    assert budget.mamba_strategy_flag == "--mamba-radix-cache-strategy"
    assert SGLangGenerationAdapter(str(model_dir))._memory_budget.mamba_strategy_flag == "--mamba-scheduler-strategy"


def test_generation_lora_declares_its_pool(model_dir):
    adapter = SGLangGenerationAdapter(
        str(model_dir),
        lora_paths={"support": "acme/support-lora"},
        max_lora_rank=16,
        lora_target_modules=["q_proj", "v_proj"],
        max_loras_per_batch=2,
    )
    assert adapter._memory_budget.lora == LoraCapacity(2, 16, ("q_proj", "v_proj"))
    assert adapter._memory_budget.speculative is False
    spec = SGLangGenerationAdapter(str(model_dir), speculative={"enabled": True, "algorithm": "NEXTN"})
    assert spec._memory_budget.speculative is True


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
