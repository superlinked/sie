"""Pinned fixed-head extraction catalog and unchanged dependency boundaries."""

from pathlib import Path

import pytest
import yaml
from sie_server.adapters.token_classification.adapter import OntoNotesTokenClassificationAdapter
from sie_server.core.loader import load_adapter, load_model_configs

SERVER = Path(__file__).resolve().parents[2]
MODEL = "learnrr/roberta-large-ontonotes5-ner"
REVISION = "696d6693ee55790dfc2d600e63d105d68a24a33e"
ADAPTER = "sie_server.adapters.token_classification.adapter"


def test_fixed_head_catalog_and_default_bundle():
    config = yaml.safe_load((SERVER / "models/learnrr__roberta-large-ontonotes5-ner.yaml").read_text())
    assert config["sie_id"] == config["hf_id"] == MODEL
    assert config["hf_revision"] == REVISION
    assert config["hf_tokenizer_dependencies"] == {MODEL: REVISION}
    assert config["inputs"] == {"text": True, "image": False, "audio": False, "video": False}
    assert config["tasks"] == {"encode": None, "score": None, "extract": {}}
    assert config["max_sequence_length"] == 512
    profile = config["profiles"]["default"]
    assert profile["max_batch_tokens"] == 16384
    assert profile["compute_precision"] == "float32"
    assert profile["adapter_path"] == f"{ADAPTER}:OntoNotesTokenClassificationAdapter"
    assert profile["adapter_options"] == {
        "loadtime": {"window_overlap": 128, "max_document_tokens": 16384},
        "runtime": {},
    }
    default = yaml.safe_load((SERVER / "bundles/default.yaml").read_text())
    assert ADAPTER in default["adapters"]
    assert default["deps"]["transformers"] == ">=4.57,<5"
    assert default["deps"]["torch"] == ">=2.9,<2.10"
    assert ADAPTER not in yaml.safe_load((SERVER / "bundles/transformers5.yaml").read_text())["adapters"]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_catalog_loader_keeps_fp32_exact_revision_and_window(device):
    config = load_model_configs(SERVER / "models")[MODEL]
    adapter = load_adapter(config, SERVER / "models", device=device, default_compute_precision="float16")
    assert isinstance(adapter, OntoNotesTokenClassificationAdapter)
    assert adapter._model_name_or_path == MODEL
    assert adapter._revision == REVISION
    assert adapter._compute_precision == "float32"
    assert adapter._max_seq_length == 512
    assert adapter._window_overlap == 128
    assert adapter._max_document_tokens == 16384
    assert adapter._model is adapter._tokenizer is None
