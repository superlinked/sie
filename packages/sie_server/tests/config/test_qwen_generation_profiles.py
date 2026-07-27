from __future__ import annotations

from pathlib import Path

import yaml
from sie_server.config.model import ModelConfig, ResolvedProfile

_QWEN35_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "Qwen__Qwen3.5-4B.yaml"
_ADAPTER = "sie_server.adapters.sglang.generation:SGLangGenerationAdapter"
_MLX_REPO = "mlx-community/Qwen3.5-4B-4bit"
_QWEN36_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "Qwen__Qwen3.6-27B.yaml"


def _qwen35_config() -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(_QWEN35_MODEL_PATH.read_text()))


def _assert_common_generation_shape(profile: ResolvedProfile) -> None:
    assert profile.compute_precision == "bfloat16"
    assert profile.adapter_path == _ADAPTER
    assert profile.loadtime["served_model_name"] == "Qwen/Qwen3.5-4B"
    assert profile.loadtime["disable_cuda_graph"] is True
    assert profile.loadtime["attention_backend"] == "triton"
    assert profile.loadtime["grammar_backend"] == "outlines"
    assert profile.loadtime["reasoning_parser"] == "qwen3"
    assert profile.loadtime["tool_call_parser"] == "qwen3_coder"


def test_qwen35_default_resolves_to_measured_a100_shape() -> None:
    config = _qwen35_config()
    default = config.resolve_profile("default")
    assert config.tasks.generate is not None
    assert config.tasks.generate.context_length == 8192
    assert config.max_sequence_length == 8192

    _assert_common_generation_shape(default)
    assert default.max_batch_tokens == 32768
    assert default.kv_budget_tokens == 210944
    assert default.loadtime["mlx_repo"] == _MLX_REPO
    assert default.loadtime["mem_fraction_static"] == 0.85
    assert default.loadtime["speculative"] == {
        "enabled": True,
        "algorithm": "nextn",
        "num_steps": 3,
        "eagle_topk": 1,
        "num_draft_tokens": 4,
    }
    assert default.loadtime["extra_launch_args"] == [
        "--mamba-scheduler-strategy",
        "extra_buffer",
    ]
    assert default.runtime["inter_chunk_timeout_s"] == 15
    assert config.resolve_profile("a100-40gb") == default


def test_qwen35_l4_smoke_preserves_constrained_c1_shape() -> None:
    l4_smoke = _qwen35_config().resolve_profile("l4-smoke")

    _assert_common_generation_shape(l4_smoke)
    assert l4_smoke.max_batch_tokens == 8192
    assert l4_smoke.kv_budget_tokens == 7168
    assert l4_smoke.loadtime["mlx_repo"] == _MLX_REPO
    assert l4_smoke.loadtime["mem_fraction_static"] == 0.90
    assert l4_smoke.loadtime["speculative"]["enabled"] is True
    assert l4_smoke.loadtime["extra_launch_args"] == [
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--disable-overlap-schedule",
    ]


def test_qwen35_grammar_companion_is_non_speculative_a100() -> None:
    config = _qwen35_config()
    no_spec = config.resolve_profile("no-spec")
    default = config.resolve_profile("default")

    assert config.tasks.generate is not None
    assert config.tasks.generate.grammar_profile == "no-spec"
    _assert_common_generation_shape(no_spec)
    assert no_spec.max_batch_tokens == 32768
    assert no_spec.kv_budget_tokens == default.kv_budget_tokens == 210944
    assert no_spec.loadtime["mem_fraction_static"] == 0.85
    assert no_spec.loadtime["mlx_repo"] == _MLX_REPO
    assert {k: v for k, v in no_spec.loadtime.items() if k != "speculative"} == {
        k: v for k, v in default.loadtime.items() if k != "speculative"
    }
    assert no_spec.runtime == dict(default.runtime) | {
        "first_chunk_timeout_s": 180,
        "inter_chunk_timeout_s": 10,
    }
    assert no_spec.loadtime["speculative"] == {"enabled": False}


def test_qwen35_no_admission_variant_inherits_production_default() -> None:
    config = _qwen35_config()
    default = config.resolve_profile("default")
    no_admission = config.resolve_profile("default-no-admission")

    assert no_admission.max_batch_tokens == default.max_batch_tokens
    assert no_admission.kv_budget_tokens == default.kv_budget_tokens
    assert no_admission.loadtime == default.loadtime
    assert no_admission.runtime == default.runtime
    assert no_admission.admission_enabled is False


def test_qwen35_timeout_margin_changes_only_default_family() -> None:
    config = _qwen35_config()

    assert config.resolve_profile("default").runtime["inter_chunk_timeout_s"] == 15
    assert config.resolve_profile("a100-40gb").runtime["inter_chunk_timeout_s"] == 15
    assert config.resolve_profile("default-no-admission").runtime["inter_chunk_timeout_s"] == 15
    assert config.resolve_profile("l4-smoke").runtime["inter_chunk_timeout_s"] == 10
    assert config.resolve_profile("h100").runtime["inter_chunk_timeout_s"] == 10
    assert config.resolve_profile("no-spec").runtime["inter_chunk_timeout_s"] == 10


def test_qwen36_grammar_requests_route_to_no_spec() -> None:
    config = ModelConfig.model_validate(yaml.safe_load(_QWEN36_MODEL_PATH.read_text()))
    default = config.resolve_profile("default")
    no_spec = config.resolve_profile("no-spec")

    assert config.tasks.generate is not None
    assert config.tasks.generate.grammar_profile == "no-spec"
    assert default.adapter_path == no_spec.adapter_path == _ADAPTER
    assert default.loadtime["grammar_backend"] == no_spec.loadtime["grammar_backend"] == "outlines"
    assert default.loadtime["speculative"] == {
        "enabled": False,
        "algorithm": "nextn",
        "num_steps": 2,
        "eagle_topk": 1,
        "num_draft_tokens": 2,
    }
    assert no_spec.loadtime["speculative"] == {"enabled": False}


def test_qwen36_profiles_use_official_non_thinking_sampling_defaults() -> None:
    config = ModelConfig.model_validate(yaml.safe_load(_QWEN36_MODEL_PATH.read_text()))
    expected = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "min_new_tokens": 10,
    }

    for profile_name in ("default", "h100", "rtx-pro-6000", "batch", "no-spec"):
        assert config.resolve_profile(profile_name).runtime["default_sampling"] == expected


def test_qwen36_base_profiles_expose_8k_context() -> None:
    config = ModelConfig.model_validate(yaml.safe_load(_QWEN36_MODEL_PATH.read_text()))

    assert config.tasks.generate is not None
    assert config.tasks.generate.context_length == 8192
    assert config.max_sequence_length == 8192
    assert config.resolve_profile("default").kv_budget_tokens == 8192
    assert config.resolve_profile("h100").kv_budget_tokens == 32768
    assert config.resolve_profile("batch").kv_budget_tokens == 32768
    assert config.resolve_profile("no-spec").kv_budget_tokens == 65536
