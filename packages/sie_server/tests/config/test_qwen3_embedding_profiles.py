from pathlib import Path

import pytest
from sie_sdk.bundle_utils import match_bundle_models
from sie_server.adapters.sentence_transformer import SentenceTransformerDenseAdapter
from sie_server.core.loader import expand_profile_variants, load_adapter, load_model_config

SERVER_ROOT = Path(__file__).resolve().parents[2]
MODELS = SERVER_ROOT / "models"
BUNDLES = SERVER_ROOT / "bundles"
SIZES = ["0.6B", "4B", "8B"]


@pytest.mark.parametrize("size", SIZES)
def test_qwen3_embedding_routes_are_available_in_their_separate_bundles(size: str) -> None:
    model_id = f"Qwen/Qwen3-Embedding-{size}"
    embedding_models = match_bundle_models(BUNDLES / "sglang-embedding.yaml", MODELS)
    flash_models = match_bundle_models(BUNDLES / "default.yaml", MODELS)
    generation_models = match_bundle_models(BUNDLES / "sglang.yaml", MODELS)

    assert model_id in embedding_models
    assert f"{model_id}:flash" not in embedding_models
    assert f"{model_id}:flash" in flash_models
    assert model_id not in flash_models
    assert model_id not in generation_models
    assert f"{model_id}:flash" not in generation_models


@pytest.mark.parametrize("size", SIZES)
def test_flash_variant_preserves_query_formatting_and_uses_its_own_pooling_name(size: str) -> None:
    config = load_model_config(MODELS / f"Qwen__Qwen3-Embedding-{size}.yaml")
    sglang = config.resolve_profile("default")
    flash = config.resolve_profile("flash")

    # Runtime overrides replace the whole mapping. A pooling-only override
    # would silently drop the query instruction and change the embedding space.
    assert sglang.runtime["pooling"] == "last_token"
    assert flash.runtime == {**sglang.runtime, "pooling": "last"}
    assert sglang.runtime["normalize"] is True
    assert "{instruction}" in sglang.runtime["query_template"]
    assert sglang.runtime["default_instruction"]
    assert flash.max_batch_tokens == sglang.max_batch_tokens
    assert flash.compute_precision == sglang.compute_precision


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_flash_variant_keeps_the_pinned_cpu_and_mac_adapter(size: str, device: str) -> None:
    config = load_model_config(MODELS / f"Qwen__Qwen3-Embedding-{size}.yaml")
    variant = expand_profile_variants([config])[f"{config.sie_id}:flash"]
    adapter = load_adapter(variant, MODELS, device=device)

    # Construct only: this test must not download weights or start a GPU engine.
    assert isinstance(adapter, SentenceTransformerDenseAdapter)
    assert adapter._model_name_or_path == config.hf_id
    assert adapter._revision == config.hf_revision
    assert adapter._max_seq_length == config.max_sequence_length
    assert adapter._model is None
