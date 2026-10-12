from __future__ import annotations

import json
from pathlib import Path

import pytest
from sie_server.adapters._generation_base import thinking_blocks_must_be_hidden, thinking_mode_is_enabled
from sie_server.config.model import ModelConfig
from sie_server.core.grammar_routing import resolve_grammar_serving_model
from sie_server.core.loader import expand_profile_variants, load_model_config
from sie_server.processors.schema_prompt import profile_schema_in_prompt

MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "google__gemma-4-31B-it.yaml"
MODEL_ID = "google/gemma-4-31B-it"
PROFILE = "h100-96k-hires-thinking-no-spec"
BUDGET_PROFILE = "h100-96k-hires-thinking-budget512-no-spec"
SCREENSHOT_PROFILES = (PROFILE, BUDGET_PROFILE)


def test_hires_thinking_profile_preserves_the_non_speculative_document_launch() -> None:
    config = load_model_config(MODEL_FILE)
    thinking = config.resolve_profile(PROFILE)
    answer_only = config.resolve_profile("h100-96k-hires-no-spec")

    thinking_launch = dict(thinking.loadtime)
    answer_only_launch = dict(answer_only.loadtime)
    # Only the static memory fraction differs: each was measured against its
    # own burst of concurrent image prefills (screenshots, full-page documents).
    assert thinking_launch.pop("mem_fraction_static") == 0.88
    assert answer_only_launch.pop("mem_fraction_static") == 0.85
    assert thinking_launch == answer_only_launch
    # The screenshot profile restates the document runtime block and adds one
    # key: it keeps a json_schema response format out of the prompt.
    thinking_runtime = dict(thinking.runtime)
    assert thinking_runtime.pop("schema_in_prompt") is False
    assert thinking_runtime == dict(answer_only.runtime)
    assert thinking.adapter_path == answer_only.adapter_path
    assert thinking.compute_precision == answer_only.compute_precision == "bfloat16"
    assert thinking.max_batch_tokens == thinking.kv_budget_tokens == 98304
    assert thinking.loadtime["speculative"] == {"enabled": False}
    assert thinking.loadtime["grammar_backend"] == "xgrammar"
    assert thinking.loadtime["json_number_max_digits"] == 19
    args = thinking.loadtime["extra_launch_args"]
    assert args[args.index("--quantization") + 1] == "fp8"
    assert args[args.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert json.loads(args[args.index("--mm-process-config") + 1]) == {"image": {"max_soft_tokens": 1120}}
    assert args.count("--constrained-json-disable-any-whitespace") == 1


def test_hires_thinking_variant_materializes_the_full_generation_contract() -> None:
    configs = expand_profile_variants([load_model_config(MODEL_FILE)])
    variant = configs[f"{MODEL_ID}:{PROFILE}"]
    generate = variant.tasks.generate
    assert generate is not None

    assert variant.hf_revision == "842da3794eaa0b77d5f08bae87a17459d91ff475"
    assert variant.max_sequence_length == generate.context_length == 98304
    assert generate.max_output_tokens == 32768
    assert generate.chat_template_kwargs == {"enable_thinking": True}
    assert thinking_mode_is_enabled(variant)
    assert thinking_blocks_must_be_hidden(variant)
    assert variant.resolve_profile("default").chat_template_kwargs == {"enable_thinking": True}


def test_hires_thinking_variant_keeps_stock_sampling_and_the_bare_contract() -> None:
    configs = expand_profile_variants([load_model_config(MODEL_FILE)])
    sampling = configs[f"{MODEL_ID}:{PROFILE}"].resolve_profile("default").runtime["default_sampling"]
    assert sampling == {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_new_tokens": 1}
    bare_generate = configs[MODEL_ID].tasks.generate
    assert bare_generate is not None
    assert bare_generate.max_output_tokens == 4096
    assert bare_generate.context_length == 8192
    assert bare_generate.chat_template_kwargs == {"enable_thinking": False}


def test_budget_profile_is_the_thinking_launch_with_a_512_token_reasoning_cap() -> None:
    config = load_model_config(MODEL_FILE)
    budget = config.resolve_profile(BUDGET_PROFILE)
    thinking = config.resolve_profile(PROFILE)

    budget_launch = dict(budget.loadtime)
    assert budget_launch.pop("thinking_budget") == 512
    assert "thinking_budget" not in thinking.loadtime
    assert budget_launch == thinking.loadtime
    assert budget.runtime == thinking.runtime
    assert budget.adapter_path == thinking.adapter_path
    assert budget.max_output_tokens == thinking.max_output_tokens == 32768
    assert budget.chat_template_kwargs == thinking.chat_template_kwargs == {"enable_thinking": True}
    assert budget.max_batch_tokens == budget.kv_budget_tokens == 98304


def test_budget_variant_materializes_the_thinking_generation_contract() -> None:
    configs = expand_profile_variants([load_model_config(MODEL_FILE)])
    budget = configs[f"{MODEL_ID}:{BUDGET_PROFILE}"]
    thinking = configs[f"{MODEL_ID}:{PROFILE}"]

    assert budget.tasks.generate == thinking.tasks.generate
    assert thinking_mode_is_enabled(budget)
    assert thinking_blocks_must_be_hidden(budget)


class _Served:
    """The model lookup grammar routing needs, over the expanded catalog."""

    def __init__(self, configs: dict[str, ModelConfig]) -> None:
        self._configs = configs

    def has_model(self, name: str) -> bool:
        return name in self._configs

    def get_config(self, name: str) -> ModelConfig:
        return self._configs[name]


@pytest.mark.parametrize("profile", SCREENSHOT_PROFILES)
def test_screenshot_variants_keep_a_json_schema_out_of_the_prompt(profile: str) -> None:
    configs = expand_profile_variants([load_model_config(MODEL_FILE)])
    served = f"{MODEL_ID}:{profile}"

    # A json_schema request stays on the profile it names, so the switch the
    # worker reads is this profile's own.
    assert resolve_grammar_serving_model(_Served(configs), served) == served
    assert configs[served].resolve_profile("default").runtime["schema_in_prompt"] is False
    assert not profile_schema_in_prompt(configs[served])


def test_only_the_screenshot_profiles_turn_the_schema_section_off() -> None:
    configs = expand_profile_variants([load_model_config(MODEL_FILE)])
    off = {model_id for model_id, config in configs.items() if not profile_schema_in_prompt(config)}
    assert off == {f"{MODEL_ID}:{profile}" for profile in SCREENSHOT_PROFILES}
    # The answer-only document profiles and the bare model keep showing it.
    assert profile_schema_in_prompt(configs[MODEL_ID])
    assert profile_schema_in_prompt(configs[f"{MODEL_ID}:h100-96k-hires-no-spec"])
