"""Public native ZeRank catalog identity and default bundle routing."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from sie_server.adapters.native_causal_cross_encoder.adapter import NativeCausalCrossEncoderAdapter
from sie_server.core.loader import load_adapter, load_model_configs

_SERVER = Path(__file__).resolve().parents[2]
_MODEL = "zeroentropy/zerank-2-reranker"
_REVISION = "5eae30d5ee3c6b2df2ef6d723bde45172d761c4c"
_ADAPTER = "sie_server.adapters.native_causal_cross_encoder.adapter"


def test_native_catalog_pin_profile_and_bundle() -> None:
    config = yaml.safe_load((_SERVER / "models/zeroentropy__zerank-2-reranker.yaml").read_text())
    assert config["sie_id"] == config["hf_id"] == _MODEL
    assert config["hf_revision"] == _REVISION
    assert config["inputs"] == {"text": True, "image": False, "audio": False, "video": False}
    assert config["tasks"] == {"encode": None, "score": {}, "extract": None}
    assert config["max_sequence_length"] == 32768
    profile = config["profiles"]["default"]
    assert profile["compute_precision"] == "bfloat16"
    assert profile["max_batch_tokens"] == 32768
    assert profile["adapter_path"] == f"{_ADAPTER}:NativeCausalCrossEncoderAdapter"
    assert profile["adapter_options"] == {"loadtime": {}, "runtime": {}}
    bundle = yaml.safe_load((_SERVER / "bundles/default.yaml").read_text())
    assert _ADAPTER in bundle["adapters"]
    assert bundle["deps"]["sentence-transformers"] == ">=5.4.1,<6"
    assert bundle["deps"]["transformers"] == ">=4.57,<5"
    assert bundle["deps"]["gliner2"] == ">=1.3.1,<2"


def test_loader_forwards_native_context_precision_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    native_load = Mock()
    monkeypatch.setattr(NativeCausalCrossEncoderAdapter, "load", native_load)
    config = load_model_configs(_SERVER / "models")[_MODEL]
    adapter = load_adapter(config, _SERVER / "models", device="cpu")
    assert isinstance(adapter, NativeCausalCrossEncoderAdapter)
    assert adapter._max_seq_length == 32768
    assert adapter._model_name_or_path == _MODEL
    assert adapter._revision == _REVISION
    assert adapter._compute_precision == "bfloat16"
    native_load.assert_not_called()


@pytest.mark.parametrize("name", ["0.6B", "4B"])
def test_existing_qwen_routes_keep_their_specialized_modes(name: str) -> None:
    config = yaml.safe_load((_SERVER / f"models/Qwen__Qwen3-Reranker-{name}.yaml").read_text())
    profile = config["profiles"]["default"]
    assert (
        profile["adapter_path"] == "sie_server.adapters.qwen2_flash_cross_encoder.adapter:Qwen2FlashCrossEncoderAdapter"
    )
    options = profile["adapter_options"]["loadtime"]
    assert options["input_format"] == "qwen3"
    assert options["score_mode"] == "log_softmax"
    assert (options["yes_token"], options["no_token"]) == ("yes", "no")
