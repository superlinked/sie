from __future__ import annotations

from pathlib import Path

import pytest
from sie_server.adapters._generation_base import thinking_blocks_must_be_hidden, thinking_mode_is_enabled
from sie_server.core.loader import expand_profile_variants, load_model_config

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

_THINKING_MODELS = (
    ("Qwen__Qwen3-0.6B.yaml", "Qwen/Qwen3-0.6B"),
    ("Qwen__Qwen3.5-4B.yaml", "Qwen/Qwen3.5-4B"),
    ("Qwen__Qwen3.6-27B.yaml", "Qwen/Qwen3.6-27B"),
    ("Qwen__Qwen3.6-35B-A3B.yaml", "Qwen/Qwen3.6-35B-A3B"),
    ("Qwen__Qwen3.8-27B-FP8.yaml", "Qwen/Qwen3.8-27B-FP8"),
    ("google__gemma-4-E2B-it.yaml", "google/gemma-4-E2B-it"),
    ("google__gemma-4-E4B-it.yaml", "google/gemma-4-E4B-it"),
    ("google__gemma-4-26B-A4B-it.yaml", "google/gemma-4-26B-A4B-it"),
    ("google__gemma-4-31B-it.yaml", "google/gemma-4-31B-it"),
)

_LONG_CONTEXT_MODELS = (
    ("Qwen__Qwen3-0.6B.yaml", "Qwen/Qwen3-0.6B", 32768),
    ("Qwen__Qwen3.5-4B.yaml", "Qwen/Qwen3.5-4B", 32768),
    ("Qwen__Qwen3.6-27B.yaml", "Qwen/Qwen3.6-27B", 262144),
    ("Qwen__Qwen3.6-35B-A3B.yaml", "Qwen/Qwen3.6-35B-A3B", 262144),
    ("google__gemma-4-E2B-it.yaml", "google/gemma-4-E2B-it", 32768),
    ("google__gemma-4-E4B-it.yaml", "google/gemma-4-E4B-it", 32768),
    ("google__gemma-4-26B-A4B-it.yaml", "google/gemma-4-26B-A4B-it", 32768),
    ("google__gemma-4-31B-it.yaml", "google/gemma-4-31B-it", 262144),
    ("defog__sqlcoder-7b-2.yaml", "defog/sqlcoder-7b-2", 16384),
)

_LONG_CONTEXT_THINKING_MODELS = (
    ("Qwen__Qwen3.6-27B.yaml", "Qwen/Qwen3.6-27B"),
    ("Qwen__Qwen3.6-35B-A3B.yaml", "Qwen/Qwen3.6-35B-A3B"),
    ("google__gemma-4-31B-it.yaml", "google/gemma-4-31B-it"),
)

_LONG_CONTEXT_THINKING_IDS = {model_id for _, model_id in _LONG_CONTEXT_THINKING_MODELS}

_HARDWARE_THINKING_PROFILES = {
    "Qwen/Qwen3.6-27B": ("h100-256k-thinking", "long-context-thinking-no-spec"),
    "Qwen/Qwen3.6-35B-A3B": ("h100-256k-thinking",),
    "Qwen/Qwen3.8-27B-FP8": ("h100-256k-thinking", "h100-256k-thinking-no-spec"),
    "google/gemma-4-31B-it": (
        "h100-96k-thinking",
        "h200-256k-thinking",
        "long-context-thinking-no-spec",
        "h100-96k-thinking-no-spec",
    ),
}


@pytest.mark.parametrize(("model_file", "model_id"), _THINKING_MODELS)
def test_thinking_profiles_are_explicit_and_other_variants_are_non_thinking(
    model_file: str,
    model_id: str,
) -> None:
    configs = expand_profile_variants([load_model_config(MODELS_DIR / model_file)])

    thinking_routes = {f"{model_id}:thinking"}
    if model_id in _LONG_CONTEXT_THINKING_IDS:
        thinking_routes.add(f"{model_id}:long-context-thinking")
    thinking_routes.update(f"{model_id}:{profile}" for profile in _HARDWARE_THINKING_PROFILES.get(model_id, ()))

    for route, config in configs.items():
        assert config.tasks.generate is not None
        expected = route in thinking_routes
        assert config.tasks.generate.chat_template_kwargs["enable_thinking"] is expected, route
        assert thinking_blocks_must_be_hidden(config) is True
        assert thinking_mode_is_enabled(config) is expected


def test_qwen_instruct_checkpoint_does_not_advertise_thinking_profile() -> None:
    """Qwen3-4B-Instruct-2507 is explicitly a non-thinking-only checkpoint."""
    config = load_model_config(MODELS_DIR / "Qwen__Qwen3-4B-Instruct-2507.yaml")

    assert "thinking" not in config.profiles
    assert config.tasks.generate is not None
    assert config.tasks.generate.chat_template_kwargs == {"enable_thinking": False}


@pytest.mark.parametrize(("model_file", "model_id", "expected_context"), _LONG_CONTEXT_MODELS)
def test_long_context_profile_promotes_request_and_adapter_limits(
    model_file: str,
    model_id: str,
    expected_context: int,
) -> None:
    configs = expand_profile_variants([load_model_config(MODELS_DIR / model_file)])

    base = configs[model_id]
    long_context = configs[f"{model_id}:long-context"]
    assert base.tasks.generate is not None
    assert long_context.tasks.generate is not None
    assert base.tasks.generate.context_length < expected_context
    assert long_context.tasks.generate.context_length == expected_context
    assert long_context.max_sequence_length == expected_context
    assert long_context.resolve_profile("default").loadtime["max_seq_length"] == expected_context


@pytest.mark.parametrize(("model_file", "model_id"), _LONG_CONTEXT_THINKING_MODELS)
def test_long_context_thinking_profile_preserves_256k_contract(
    model_file: str,
    model_id: str,
) -> None:
    configs = expand_profile_variants([load_model_config(MODELS_DIR / model_file)])

    non_thinking = configs[f"{model_id}:long-context"]
    thinking = configs[f"{model_id}:long-context-thinking"]
    assert non_thinking.tasks.generate is not None
    assert thinking.tasks.generate is not None
    assert non_thinking.tasks.generate.context_length == 262144
    assert thinking.tasks.generate.context_length == 262144
    expected_non_thinking_cap = 4096 if model_id == "google/gemma-4-31B-it" else 32768
    expected_thinking_cap = 32768 if model_id == "google/gemma-4-31B-it" else 81920
    assert non_thinking.tasks.generate.max_output_tokens == expected_non_thinking_cap
    assert thinking.tasks.generate.max_output_tokens == expected_thinking_cap
    assert non_thinking.max_sequence_length == thinking.max_sequence_length == 262144
    if model_id == "Qwen/Qwen3.6-27B":
        # Qwen thinking remains non-speculative while using its separately
        # measured CUDA-graph launch.
        assert non_thinking.resolve_profile("default").loadtime["speculative"]["enabled"] is True
        assert thinking.resolve_profile("default").loadtime["speculative"] == {"enabled": False}
        assert "disable_cuda_graph" not in thinking.resolve_profile("default").loadtime
    else:
        assert non_thinking.resolve_profile("default").loadtime == thinking.resolve_profile("default").loadtime
    assert non_thinking.resolve_profile("default").kv_budget_tokens == 262144
    assert thinking.resolve_profile("default").kv_budget_tokens == 262144
    assert non_thinking.tasks.generate.chat_template_kwargs == {"enable_thinking": False}
    assert thinking.tasks.generate.chat_template_kwargs == {"enable_thinking": True}


def test_generic_thinking_aliases_use_the_validated_hardware_context() -> None:
    expectations = (
        ("Qwen__Qwen3.6-27B.yaml", "Qwen/Qwen3.6-27B", "h100-256k-thinking", 262144, 81920),
        ("Qwen__Qwen3.6-35B-A3B.yaml", "Qwen/Qwen3.6-35B-A3B", "h100-256k-thinking", 262144, 81920),
        ("google__gemma-4-31B-it.yaml", "google/gemma-4-31B-it", "h100-96k-thinking", 98304, 32768),
    )

    for model_file, model_id, hardware_profile, context, output_cap in expectations:
        configs = expand_profile_variants([load_model_config(MODELS_DIR / model_file)])
        generic = configs[f"{model_id}:thinking"]
        hardware = configs[f"{model_id}:{hardware_profile}"]

        assert generic.tasks.generate is not None
        assert hardware.tasks.generate is not None
        assert generic.tasks.generate.context_length == hardware.tasks.generate.context_length == context
        assert generic.tasks.generate.max_output_tokens == output_cap
        assert generic.resolve_profile("default") == hardware.resolve_profile("default")


def test_gemma_31b_large_context_profiles_use_measured_kv_precision() -> None:
    configs = expand_profile_variants([load_model_config(MODELS_DIR / "google__gemma-4-31B-it.yaml")])

    for suffix in (
        "long-context",
        "long-context-thinking",
        "h100-96k",
        "h100-96k-thinking",
        "h200-256k",
        "h200-256k-thinking",
    ):
        loadtime = configs[f"google/gemma-4-31B-it:{suffix}"].resolve_profile("default").loadtime
        assert loadtime["extra_launch_args"] == [
            "--quantization",
            "fp8",
            "--kv-cache-dtype",
            "fp8_e4m3",
        ]


def test_gemma_31b_profiles_use_measured_mtp_shape_with_grammar_fallback() -> None:
    config = load_model_config(MODELS_DIR / "google__gemma-4-31B-it.yaml")
    assert config.tasks.generate is not None
    assert config.tasks.generate.grammar_profile == "no-spec"
    assert config.resolve_profile("no-spec").loadtime["speculative"] == {"enabled": False}

    expected = {
        "enabled": True,
        "algorithm": "nextn",
        "num_steps": 3,
        "eagle_topk": 1,
        "num_draft_tokens": 4,
        "draft_model": "google/gemma-4-31B-it-assistant",
        "draft_model_revision": "627c5ec1458b9086b841a91e0512fd31fd2fbbf1",
    }
    for profile_name in (
        "default",
        "h100-96k",
        "h100-96k-thinking",
        "thinking",
        "long-context",
        "long-context-thinking",
        "h200-256k-thinking",
    ):
        loadtime = config.resolve_profile(profile_name).loadtime
        assert loadtime["speculative"] == expected
        assert loadtime["speculative_needs_extra_buffer"] is False
        assert "disable_cuda_graph" not in loadtime

    for source_name, fallback_name, thinking in (
        ("long-context", "long-context-no-spec", False),
        ("h200-256k", "long-context-no-spec", False),
        ("long-context-thinking", "long-context-thinking-no-spec", True),
        ("h200-256k-thinking", "long-context-thinking-no-spec", True),
        ("h100-96k", "h100-96k-no-spec", False),
        ("h100-96k-thinking", "h100-96k-thinking-no-spec", True),
        ("thinking", "h100-96k-thinking-no-spec", True),
    ):
        source = config.resolve_profile(source_name)
        fallback = config.resolve_profile(fallback_name)
        assert source.grammar_profile == fallback_name
        assert fallback.grammar_profile is None
        assert source.adapter_path == fallback.adapter_path
        assert source.compute_precision == fallback.compute_precision
        assert source.max_batch_tokens == fallback.max_batch_tokens
        assert source.loadtime | {"speculative": None} == fallback.loadtime | {"speculative": None}
        assert fallback.loadtime["speculative"] == {"enabled": False}
        assert source.chat_template_kwargs == fallback.chat_template_kwargs
        effective_mode = source.chat_template_kwargs or config.tasks.generate.chat_template_kwargs
        assert effective_mode == {"enable_thinking": thinking}


@pytest.mark.parametrize(
    ("model_file", "model_id"),
    [
        ("Qwen__Qwen3.6-27B.yaml", "Qwen/Qwen3.6-27B"),
        ("Qwen__Qwen3.6-35B-A3B.yaml", "Qwen/Qwen3.6-35B-A3B"),
    ],
)
def test_qwen_h100_256k_aliases_match_long_context_profiles(model_file: str, model_id: str) -> None:
    configs = expand_profile_variants([load_model_config(MODELS_DIR / model_file)])

    for hardware_suffix, generic_suffix in (
        ("h100-256k", "long-context"),
        ("h100-256k-thinking", "long-context-thinking"),
    ):
        hardware = configs[f"{model_id}:{hardware_suffix}"]
        generic = configs[f"{model_id}:{generic_suffix}"]
        assert hardware.resolve_profile("default") == generic.resolve_profile("default")
        assert hardware.tasks.generate == generic.tasks.generate

    assert configs[f"{model_id}:h100-256k"].tasks.generate.max_output_tokens == 32768
    assert configs[f"{model_id}:h100-256k-thinking"].tasks.generate.max_output_tokens == 81920


def test_qwen_35b_profiles_use_runtime_compatible_grammar_backend() -> None:
    configs = expand_profile_variants([load_model_config(MODELS_DIR / "Qwen__Qwen3.6-35B-A3B.yaml")])

    for suffix in ("default", "h100-256k", "h100-256k-thinking"):
        model_id = "Qwen/Qwen3.6-35B-A3B"
        config = configs[model_id] if suffix == "default" else configs[f"{model_id}:{suffix}"]
        assert config.resolve_profile("default").loadtime["grammar_backend"] == "xgrammar"


def test_qwen_27b_profiles_use_runtime_compatible_grammar_backend() -> None:
    configs = expand_profile_variants([load_model_config(MODELS_DIR / "Qwen__Qwen3.6-27B.yaml")])
    model_id = "Qwen/Qwen3.6-27B"

    for suffix in ("default", "h100", "h100-fp8", "rtx-pro-6000", "h100-256k", "batch", "no-spec"):
        config = configs[model_id] if suffix == "default" else configs[f"{model_id}:{suffix}"]
        assert config.resolve_profile("default").loadtime["grammar_backend"] == "xgrammar"


def test_gemma_31b_hardware_profiles_expose_measured_context_caps() -> None:
    configs = expand_profile_variants([load_model_config(MODELS_DIR / "google__gemma-4-31B-it.yaml")])
    model_id = "google/gemma-4-31B-it"

    for hardware_suffix, generic_suffix in (
        ("h200-256k", "long-context"),
        ("h200-256k-thinking", "long-context-thinking"),
    ):
        hardware = configs[f"{model_id}:{hardware_suffix}"]
        generic = configs[f"{model_id}:{generic_suffix}"]
        assert hardware.resolve_profile("default") == generic.resolve_profile("default")
        assert hardware.tasks.generate == generic.tasks.generate
    for suffix in ("h100-96k", "h100-96k-thinking"):
        config = configs[f"{model_id}:{suffix}"]
        assert config.tasks.generate is not None
        assert config.tasks.generate.context_length == 98304
        assert config.max_sequence_length == 98304
        profile = config.resolve_profile("default")
        assert profile.max_batch_tokens == 98304
        assert profile.kv_budget_tokens == 98304
        assert profile.loadtime["max_seq_length"] == 98304
    assert configs[f"{model_id}:h100-96k"].tasks.generate.max_output_tokens == 4096
    assert configs[f"{model_id}:h100-96k-thinking"].tasks.generate.max_output_tokens == 32768


@pytest.mark.parametrize(
    ("model_file", "model_id", "temperature", "top_p", "top_k", "presence_penalty"),
    [
        ("Qwen__Qwen3-0.6B.yaml", "Qwen/Qwen3-0.6B", 0.6, 0.95, 20, None),
        ("Qwen__Qwen3.5-4B.yaml", "Qwen/Qwen3.5-4B", 1.0, 0.95, 20, 1.5),
        ("Qwen__Qwen3.6-27B.yaml", "Qwen/Qwen3.6-27B", 1.0, 0.95, 20, 0.0),
        ("Qwen__Qwen3.6-35B-A3B.yaml", "Qwen/Qwen3.6-35B-A3B", 1.0, 0.95, 20, 1.5),
    ],
)
def test_qwen_thinking_profiles_use_thinking_sampling_recipe(
    model_file: str,
    model_id: str,
    temperature: float,
    top_p: float,
    top_k: int,
    presence_penalty: float | None,
) -> None:
    configs = expand_profile_variants([load_model_config(MODELS_DIR / model_file)])
    sampling = configs[f"{model_id}:thinking"].resolve_profile("default").runtime["default_sampling"]

    assert sampling["temperature"] == temperature
    assert sampling["top_p"] == top_p
    assert sampling["top_k"] == top_k
    assert sampling.get("presence_penalty") == presence_penalty
