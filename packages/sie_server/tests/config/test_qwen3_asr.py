"""Immutable Qwen3-ASR model identity and the supported dependency lane."""

from pathlib import Path

import yaml
from sie_server.config.model import ModelConfig


def test_qwen_asr_descriptor_and_transformers5_registration() -> None:
    server = Path(__file__).resolve().parents[2]
    config = ModelConfig.model_validate(yaml.safe_load((server / "models/Qwen__Qwen3-ASR-1.7B-hf.yaml").read_text()))
    assert config.sie_id == config.hf_id == "Qwen/Qwen3-ASR-1.7B-hf"
    assert config.hf_revision == "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
    assert config.inputs.audio
    assert not config.inputs.text
    assert not config.inputs.image
    assert not config.inputs.video
    assert config.tasks.extract is not None
    assert config.tasks.encode is None
    assert config.tasks.score is None
    assert config.max_sequence_length == 65536
    profile = config.profiles["default"]
    assert profile.compute_precision == "bfloat16"
    assert profile.max_batch_tokens == 720000
    assert profile.adapter_path == "sie_server.adapters.qwen3_asr.adapter:Qwen3ASRAdapter"
    assert profile.adapter_options.loadtime == {"max_new_tokens": 512, "inference_batch_size": 4}
    bundle = yaml.safe_load((server / "bundles/transformers5.yaml").read_text())
    assert "sie_server.adapters.qwen3_asr.adapter" in bundle["adapters"]
    assert bundle["deps"]["transformers"] == ">=5.14,<6"
    default = yaml.safe_load((server / "bundles/default.yaml").read_text())
    assert "sie_server.adapters.qwen3_asr.adapter" not in default["adapters"]
