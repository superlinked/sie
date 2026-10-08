from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from sie_server.config.model import ModelConfig

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@pytest.mark.parametrize(
    (
        "model_file",
        "model_id",
        "adapter_path",
        "grammar_backend",
        "grammar_profile",
        "disable_cuda_graph",
        "speculative",
        "extra_launch_args",
    ),
    [
        (
            "Qwen__Qwen3.6-35B-A3B.yaml",
            "Qwen/Qwen3.6-35B-A3B",
            "sie_server.adapters.sglang.generation:SGLangGenerationAdapter",
            "xgrammar",
            None,
            True,
            {"enabled": False},
            [
                "--mm-process-config",
                '{"image":{"min_pixels":65536,"max_pixels":1003520}}',
                "--constrained-json-disable-any-whitespace",
                "--quantization",
                "fp8",
            ],
        ),
        (
            "google__gemma-4-31B-it.yaml",
            "google/gemma-4-31B-it",
            "sie_server.adapters.sglang.gemma:SGLangGemmaAdapter",
            "xgrammar",
            "no-spec",
            None,
            {
                "enabled": True,
                "algorithm": "nextn",
                "num_steps": 3,
                "eagle_topk": 1,
                "num_draft_tokens": 4,
                "draft_model": "google/gemma-4-31B-it-assistant",
                "draft_model_revision": "627c5ec1458b9086b841a91e0512fd31fd2fbbf1",
            },
            ["--quantization", "fp8"],
        ),
    ],
)
def test_new_generation_models_resolve_h100_fp8_alias(
    model_file: str,
    model_id: str,
    adapter_path: str,
    grammar_backend: str,
    grammar_profile: str | None,
    disable_cuda_graph: bool | None,
    speculative: dict[str, object],
    extra_launch_args: list[str],
) -> None:
    config = ModelConfig.model_validate(yaml.safe_load((MODELS_DIR / model_file).read_text()))
    default = config.resolve_profile("default")
    h100_fp8 = config.resolve_profile("h100-fp8")

    assert config.sie_id == model_id
    assert config.max_sequence_length == 8192
    assert config.tasks.generate is not None
    assert config.tasks.generate.context_length == 8192
    assert config.tasks.generate.max_output_tokens == 4096
    assert config.tasks.generate.grammar_profile == grammar_profile

    assert h100_fp8 == default
    assert default.adapter_path == adapter_path
    assert default.compute_precision == "bfloat16"
    assert default.kv_budget_tokens == 8192
    assert default.loadtime["served_model_name"] == model_id
    assert default.loadtime.get("disable_cuda_graph") is disable_cuda_graph
    assert default.loadtime["grammar_backend"] == grammar_backend
    assert default.loadtime["speculative"] == speculative
    assert default.loadtime["extra_launch_args"] == extra_launch_args
