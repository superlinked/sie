"""Gemma 4 31B document-reading profiles: ``h100-96k`` with a 1,120 soft-token image budget."""

from __future__ import annotations

import json
from pathlib import Path

from sie_server.core.loader import expand_profile_variants, load_model_config
from sie_server.processors.streaming import _VISION_TOKENS_PER_IMAGE_ESTIMATE

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_FILE = MODELS_DIR / "google__gemma-4-31B-it.yaml"
MODEL_ID = "google/gemma-4-31B-it"

# Budgets accepted by the Gemma 4 image processor's ``max_soft_tokens``.
_GEMMA4_SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)
_HIRES_ARGS = ["--mm-process-config", '{"image":{"max_soft_tokens":1120}}']


def _mm_process_config(extra_launch_args: list[str]) -> dict:
    index = extra_launch_args.index("--mm-process-config")
    return json.loads(extra_launch_args[index + 1])


def test_hires_profiles_set_a_supported_image_budget_within_the_reservation() -> None:
    config = load_model_config(MODEL_FILE)
    for name in ("h100-96k-hires", "h100-96k-hires-no-spec"):
        mm_config = _mm_process_config(config.resolve_profile(name).loadtime["extra_launch_args"])
        assert mm_config == {"image": {"max_soft_tokens": 1120}}
        budget = mm_config["image"]["max_soft_tokens"]
        assert budget in _GEMMA4_SUPPORTED_SOFT_TOKENS
        # Two boundary tokens wrap the soft tokens; the worker reserves at least
        # the family-wide estimate per image, so admission needs no change.
        assert budget + 2 <= _VISION_TOKENS_PER_IMAGE_ESTIMATE


def test_hires_profiles_match_h100_96k_except_the_image_budget() -> None:
    config = load_model_config(MODEL_FILE)
    for hires_name, base_name in (("h100-96k-hires", "h100-96k"), ("h100-96k-hires-no-spec", "h100-96k-no-spec")):
        hires = config.resolve_profile(hires_name)
        base = config.resolve_profile(base_name)
        assert hires.loadtime["extra_launch_args"] == [*base.loadtime["extra_launch_args"], *_HIRES_ARGS]
        assert hires.loadtime | {"extra_launch_args": None} == base.loadtime | {"extra_launch_args": None}
        assert hires.runtime == base.runtime
        assert hires.adapter_path == base.adapter_path
        assert hires.compute_precision == base.compute_precision
        assert hires.max_batch_tokens == base.max_batch_tokens == 98304
        assert hires.kv_budget_tokens == base.kv_budget_tokens == 98304
        assert hires.chat_template_kwargs == base.chat_template_kwargs


def test_hires_profiles_keep_answer_only_contract_and_output_cap() -> None:
    configs = expand_profile_variants([load_model_config(MODEL_FILE)])
    for suffix in ("h100-96k-hires", "h100-96k-hires-no-spec"):
        config = configs[f"{MODEL_ID}:{suffix}"]
        generate = config.tasks.generate
        assert generate is not None
        assert generate.max_output_tokens == 4096
        assert generate.context_length == 98304
        assert config.max_sequence_length == 98304
        effective_mode = config.resolve_profile("default").chat_template_kwargs or generate.chat_template_kwargs
        assert effective_mode == {"enable_thinking": False}
    assert configs[f"{MODEL_ID}:h100-96k-hires"].tasks.generate == configs[f"{MODEL_ID}:h100-96k"].tasks.generate


def test_hires_grammar_requests_route_to_the_non_speculative_sibling() -> None:
    config = load_model_config(MODEL_FILE)
    source = config.resolve_profile("h100-96k-hires")
    fallback = config.resolve_profile("h100-96k-hires-no-spec")
    assert source.grammar_profile == "h100-96k-hires-no-spec"
    assert fallback.grammar_profile is None
    assert source.loadtime["speculative"] == config.resolve_profile("h100-96k").loadtime["speculative"]
    assert fallback.loadtime["speculative"] == {"enabled": False}
    assert source.loadtime | {"speculative": None} == fallback.loadtime | {"speculative": None}


def test_hires_out8k_profiles_only_raise_the_output_cap() -> None:
    config = load_model_config(MODEL_FILE)
    for out8k_name, base_name in (
        ("h100-96k-hires-out8k", "h100-96k-hires"),
        ("h100-96k-hires-out8k-no-spec", "h100-96k-hires-no-spec"),
    ):
        out8k = config.resolve_profile(out8k_name)
        base = config.resolve_profile(base_name)
        assert out8k.max_output_tokens == 8192
        assert base.max_output_tokens is None
        assert out8k.loadtime == base.loadtime
        assert out8k.runtime == base.runtime
        assert out8k.adapter_path == base.adapter_path
        assert out8k.compute_precision == base.compute_precision
        assert out8k.max_batch_tokens == base.max_batch_tokens == 98304
        assert out8k.kv_budget_tokens == base.kv_budget_tokens == 98304
        assert out8k.chat_template_kwargs == base.chat_template_kwargs
        assert _mm_process_config(out8k.loadtime["extra_launch_args"]) == {"image": {"max_soft_tokens": 1120}}


def test_hires_out8k_profiles_serve_answer_only_with_an_8192_token_cap() -> None:
    configs = expand_profile_variants([load_model_config(MODEL_FILE)])
    for suffix in ("h100-96k-hires-out8k", "h100-96k-hires-out8k-no-spec"):
        config = configs[f"{MODEL_ID}:{suffix}"]
        generate = config.tasks.generate
        assert generate is not None
        assert generate.max_output_tokens == 8192
        assert generate.context_length == 98304
        assert config.max_sequence_length == 98304
        effective_mode = config.resolve_profile("default").chat_template_kwargs or generate.chat_template_kwargs
        assert effective_mode == {"enable_thinking": False}
    # The existing hi-res and bare H100 routes keep the conservative 4,096-token cap.
    for suffix in ("h100-96k-hires", "h100-96k-hires-no-spec", "h100-96k", "h100-96k-no-spec"):
        assert configs[f"{MODEL_ID}:{suffix}"].tasks.generate.max_output_tokens == 4096


def test_hires_out8k_grammar_requests_route_to_the_non_speculative_sibling() -> None:
    config = load_model_config(MODEL_FILE)
    source = config.resolve_profile("h100-96k-hires-out8k")
    fallback = config.resolve_profile("h100-96k-hires-out8k-no-spec")
    assert source.grammar_profile == "h100-96k-hires-out8k-no-spec"
    assert fallback.grammar_profile is None
    assert source.max_output_tokens == fallback.max_output_tokens == 8192
    assert source.loadtime["speculative"] == config.resolve_profile("h100-96k").loadtime["speculative"]
    assert fallback.loadtime["speculative"] == {"enabled": False}
    assert source.loadtime | {"speculative": None} == fallback.loadtime | {"speculative": None}
