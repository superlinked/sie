"""Catalog identity and transformers5 routing for google/embeddinggemma-2."""

from pathlib import Path

import yaml
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.embedding_gemma2.adapter import (
    DOCUMENT_TEMPLATE,
    MATRYOSHKA_DIMS,
    MIN_TRANSFORMERS,
    NATIVE_DIM,
    QUERY_TEMPLATE,
)
from sie_server.core.loader import load_model_configs, resolve_adapter_class

SERVER = Path(__file__).resolve().parents[2]
MODELS = SERVER / "models"
BUNDLES = SERVER / "bundles"
MODEL_ID = "google/embeddinggemma-2"
REVISION = "f9972ac22e696a105c482202646979e33e4b5a1e"


def test_embeddinggemma2_descriptor_is_pinned_text_encode() -> None:
    config = load_model_configs(MODELS)[MODEL_ID]
    assert config.sie_id == config.hf_id == MODEL_ID
    assert config.hf_revision == REVISION
    assert config.inputs.text
    assert not config.inputs.image
    assert not config.inputs.audio
    assert not config.inputs.video
    assert config.max_sequence_length == 8192
    assert config.tasks.encode is not None
    assert config.tasks.encode.dense is not None
    assert config.tasks.encode.dense.dim == NATIVE_DIM
    assert config.tasks.score is None
    assert config.tasks.extract is None
    profile = config.resolve_profile("default")
    assert profile.compute_precision == "bfloat16"
    assert profile.max_batch_tokens == 16384
    assert profile.adapter_path == "sie_server.adapters.embedding_gemma2.adapter:EmbeddingGemma2Adapter"
    assert resolve_adapter_class(config, MODELS).__name__ == "EmbeddingGemma2Adapter"
    assert profile.loadtime == {}
    assert dict(profile.runtime) == {
        "normalize": True,
        "pooling": "mean",
        "query_template": QUERY_TEMPLATE,
        "doc_template": DOCUMENT_TEMPLATE,
    }


def test_embeddinggemma2_prompts_and_matryoshka_match_the_card() -> None:
    assert QUERY_TEMPLATE == "task: search result | query: {text}"
    assert DOCUMENT_TEMPLATE == "title: none | text: {text}"
    assert MATRYOSHKA_DIMS == (128, 256, 512, 768)
    assert MIN_TRANSFORMERS == "5.19.0"
    assert Version("5.18.1") < Version(MIN_TRANSFORMERS)
    assert Version("5.19.0") >= Version(MIN_TRANSFORMERS)


def test_embeddinggemma2_is_served_only_from_the_transformers5_bundle() -> None:
    bundle = yaml.safe_load((BUNDLES / "transformers5.yaml").read_text())
    assert "sie_server.adapters.embedding_gemma2.adapter" in bundle["adapters"]
    # Every transformers release the bundle admits ships EmbeddingGemma2Model.
    spec = SpecifierSet(bundle["deps"]["transformers"])
    assert Version("5.19.0") in spec
    assert Version("5.18.1") not in spec
    assert Version("4.57.6") not in spec
    assert MODEL_ID in match_bundle_models(BUNDLES / "transformers5.yaml", MODELS)

    default = yaml.safe_load((BUNDLES / "default.yaml").read_text())
    assert "sie_server.adapters.embedding_gemma2.adapter" not in default["adapters"]
    assert default["deps"]["transformers"] == "==4.57.6"
    assert MODEL_ID not in match_bundle_models(BUNDLES / "default.yaml", MODELS)

    for bundle_path in sorted(BUNDLES.glob("*.yaml")):
        if bundle_path.name == "transformers5.yaml":
            continue
        assert MODEL_ID not in match_bundle_models(bundle_path, MODELS)
