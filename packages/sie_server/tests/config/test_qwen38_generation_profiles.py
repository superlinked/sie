from __future__ import annotations

from pathlib import Path

from sie_server.core.loader import expand_profile_variants, load_model_config

_MODEL_ID = "Qwen/Qwen3.8-27B-FP8"
_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "Qwen__Qwen3.8-27B-FP8.yaml"
_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
_ADAPTER = "sie_server.adapters.sglang.cuda13:SGLangCuda13Adapter"
_NATIVE_CONTEXT = 262144
_NATIVE_OUTPUT_CAP = 32768
_SPECULATIVE = {
    "enabled": True,
    "algorithm": "eagle",
    "num_steps": 3,
    "eagle_topk": 1,
    "num_draft_tokens": 4,
}
_HARDWARE_PROFILES = {
    "h100-256k": "h100-256k-no-spec",
    "h200-256k": "h200-256k-no-spec",
    "rtx-pro-6000-256k": "rtx-pro-6000-256k-no-spec",
}


def test_qwen38_uses_the_pinned_official_fp8_checkpoint() -> None:
    config = load_model_config(_MODEL_PATH)

    assert config.sie_id == config.hf_id == _MODEL_ID
    assert config.hf_revision == _REVISION
    assert config.inputs.model_dump() == {
        "text": True,
        "document": False,
        "image": True,
        "audio": False,
        "video": False,
    }
    assert config.tasks.generate is not None
    assert config.tasks.generate.context_length == 8192
    assert config.tasks.generate.max_output_tokens == 4096
    assert config.tasks.generate.chat_template_kwargs == {"enable_thinking": False}
    assert set(config.profiles) == {
        "default",
        "no-spec",
        "h100-256k",
        "h100-256k-no-spec",
        "h200-256k",
        "h200-256k-no-spec",
        "rtx-pro-6000-256k",
        "rtx-pro-6000-256k-no-spec",
    }


def test_qwen38_default_is_a_conservative_non_speculative_route() -> None:
    config = load_model_config(_MODEL_PATH)
    default = config.resolve_profile("default")
    grammar = config.resolve_profile("no-spec")

    assert config.tasks.generate is not None
    assert config.tasks.generate.grammar_profile == "no-spec"
    assert grammar == default
    assert default.adapter_path == _ADAPTER
    assert default.max_batch_tokens == 16384
    assert default.kv_budget_tokens == 8192
    assert default.loadtime["mem_fraction_static"] == 0.85
    assert default.loadtime["disable_cuda_graph"] is True
    assert default.loadtime["speculative"] == {"enabled": False}
    assert default.loadtime["attention_backend"] == "flashinfer"
    assert "extra_env" not in default.loadtime
    default_args = default.loadtime["extra_launch_args"]
    assert default_args[default_args.index("--mamba-ssm-dtype") + 1] == "float32"
    # Qwen3 structured output is not constrained when thinking is disabled
    # while the reasoning parser waits for a think terminator. These profiles
    # are explicitly answer-only, so they must not launch that parser.
    assert "reasoning_parser" not in default.loadtime
    assert default.loadtime["tool_call_parser"] == "qwen3_coder"
    assert "--quantization" not in default.loadtime["extra_launch_args"]


def test_qwen38_hardware_profiles_materialize_the_native_context() -> None:
    configs = expand_profile_variants([load_model_config(_MODEL_PATH)])
    bare = configs[_MODEL_ID]

    assert bare.tasks.generate is not None
    assert bare.tasks.generate.context_length == 8192
    assert bare.max_sequence_length == 8192
    assert bare.resolve_profile("default").kv_budget_tokens == 8192

    for profile_name in (*_HARDWARE_PROFILES, *_HARDWARE_PROFILES.values()):
        variant = configs[f"{_MODEL_ID}:{profile_name}"]
        profile = variant.resolve_profile("default")

        assert variant.tasks.generate is not None
        assert variant.tasks.generate.context_length == _NATIVE_CONTEXT
        assert variant.tasks.generate.max_output_tokens == _NATIVE_OUTPUT_CAP
        assert variant.max_sequence_length == _NATIVE_CONTEXT
        assert variant.tasks.generate.chat_template_kwargs == {"enable_thinking": False}
        assert profile.adapter_path == _ADAPTER
        assert profile.max_batch_tokens == _NATIVE_CONTEXT
        assert profile.kv_budget_tokens == _NATIVE_CONTEXT
        assert profile.compute_precision == "bfloat16"
        assert profile.loadtime["max_seq_length"] == _NATIVE_CONTEXT
        assert profile.loadtime["mem_fraction_static"] == 0.85
        assert "reasoning_parser" not in profile.loadtime


def test_qwen38_hardware_profiles_use_mtp_with_exact_grammar_twins() -> None:
    config = load_model_config(_MODEL_PATH)

    for source_name, fallback_name in _HARDWARE_PROFILES.items():
        source = config.resolve_profile(source_name)
        fallback = config.resolve_profile(fallback_name)

        assert source.grammar_profile == fallback_name
        assert fallback.grammar_profile is None
        assert source.loadtime["speculative"] == _SPECULATIVE
        assert fallback.loadtime["speculative"] == {"enabled": False}
        assert source.loadtime | {"speculative": None} == fallback.loadtime | {"speculative": None}
        assert fallback.runtime == source.runtime
        assert fallback.runtime["first_chunk_timeout_s"] == 900
        assert fallback.runtime["inter_chunk_timeout_s"] == 30
        assert fallback.runtime["overall_timeout_s"] == 1800
        assert fallback.runtime["default_sampling"] == {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
        }
        assert fallback.runtime["stop_tokens"] == ["<|im_end|>"]
        assert source.adapter_path == fallback.adapter_path == _ADAPTER
        assert source.compute_precision == fallback.compute_precision == "bfloat16"
        assert source.max_batch_tokens == fallback.max_batch_tokens == _NATIVE_CONTEXT
        assert source.kv_budget_tokens == fallback.kv_budget_tokens == _NATIVE_CONTEXT


def test_qwen38_hardware_launches_keep_fp8_weights_with_tuned_state_precision() -> None:
    config = load_model_config(_MODEL_PATH)

    for profile_name in (*_HARDWARE_PROFILES, *_HARDWARE_PROFILES.values()):
        profile = config.resolve_profile(profile_name)
        args = profile.loadtime["extra_launch_args"]

        assert "--quantization" not in args
        assert args[args.index("--kv-cache-dtype") + 1] == "bfloat16"
        assert args[args.index("--mamba-ssm-dtype") + 1] == "bfloat16"
        assert args[args.index("--mamba-scheduler-strategy") + 1] == "extra_buffer"
        assert args.count("--disable-overlap-schedule") == 1
        assert args[args.index("--page-size") + 1] == "64"
        assert args[args.index("--max-running-requests") + 1] == "1"
        assert args[args.index("--cuda-graph-max-bs") + 1] == "1"
        assert profile.loadtime["extra_env"] == {"SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1"}

    for profile_name in (
        "h100-256k",
        "h100-256k-no-spec",
        "h200-256k",
        "h200-256k-no-spec",
    ):
        profile = config.resolve_profile(profile_name)
        args = profile.loadtime["extra_launch_args"]
        assert profile.loadtime["attention_backend"] == "fa3"
        assert args[args.index("--chunked-prefill-size") + 1] == "32768"
        assert args[args.index("--max-prefill-tokens") + 1] == "32768"
    for profile_name in ("rtx-pro-6000-256k", "rtx-pro-6000-256k-no-spec"):
        rtx_profile = config.resolve_profile(profile_name)
        rtx_args = rtx_profile.loadtime["extra_launch_args"]
        assert rtx_profile.loadtime["attention_backend"] == "flashinfer"
        assert rtx_args[rtx_args.index("--chunked-prefill-size") + 1] == "2048"
        assert "--max-prefill-tokens" not in rtx_args
