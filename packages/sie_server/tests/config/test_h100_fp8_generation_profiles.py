from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sie_server.config.model import ModelConfig

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@pytest.mark.parametrize(
    ("model_file", "model_id", "adapter_path", "grammar_backend"),
    [
        (
            "Qwen__Qwen3.6-35B-A3B.yaml",
            "Qwen/Qwen3.6-35B-A3B",
            "sie_server.adapters.sglang.generation:SGLangGenerationAdapter",
            "outlines",
        ),
        (
            "google__gemma-4-31B-it.yaml",
            "google/gemma-4-31B-it",
            "sie_server.adapters.sglang.gemma:SGLangGemmaAdapter",
            "xgrammar",
        ),
    ],
)
def test_new_generation_models_resolve_h100_fp8_alias(
    model_file: str,
    model_id: str,
    adapter_path: str,
    grammar_backend: str,
) -> None:
    config = ModelConfig.model_validate(yaml.safe_load((MODELS_DIR / model_file).read_text()))
    default = config.resolve_profile("default")
    h100_fp8 = config.resolve_profile("h100-fp8")

    assert config.sie_id == model_id
    assert config.max_sequence_length == 8192
    assert config.tasks.generate is not None
    assert config.tasks.generate.context_length == 8192
    assert config.tasks.generate.max_output_tokens == 4096
    assert config.tasks.generate.grammar_profile is None

    assert h100_fp8 == default
    assert default.adapter_path == adapter_path
    assert default.compute_precision == "bfloat16"
    assert default.kv_budget_tokens == 8192
    assert default.loadtime["served_model_name"] == model_id
    assert default.loadtime["disable_cuda_graph"] is True
    assert default.loadtime["grammar_backend"] == grammar_backend
    assert default.loadtime["speculative"] == {"enabled": False}
    assert default.loadtime["extra_launch_args"] == ["--quantization", "fp8"]
