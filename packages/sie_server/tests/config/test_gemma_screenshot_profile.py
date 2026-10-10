from __future__ import annotations

import json
from pathlib import Path

from sie_server.adapters._generation_base import thinking_blocks_must_be_hidden, thinking_mode_is_enabled
from sie_server.core.loader import expand_profile_variants, load_model_config

MODEL_FILE = Path(__file__).resolve().parents[2] / "models" / "google__gemma-4-31B-it.yaml"
MODEL_ID = "google/gemma-4-31B-it"
PROFILE = "h100-96k-hires-thinking-no-spec"


def test_hires_thinking_profile_preserves_the_non_speculative_document_launch() -> None:
    config = load_model_config(MODEL_FILE)
    thinking = config.resolve_profile(PROFILE)
    answer_only = config.resolve_profile("h100-96k-hires-no-spec")

    assert thinking.loadtime == answer_only.loadtime
    assert thinking.runtime == answer_only.runtime
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
