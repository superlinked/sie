from __future__ import annotations

from pathlib import Path

from sie_server.core.loader import load_model_configs

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
VISUAL_MODELS = (
    "vidore/colqwen2.5-v0.2",
    "vidore/colpali-v1.3-hf",
    "vidore/colSmol-256M",
    "TomoroAI/tomoro-colqwen3-embed-4b",
    "nvidia/llama-nemoretriever-colembed-3b-v1",
    "nvidia/nemotron-colembed-vl-4b-v2",
)


def test_implemented_visual_muvera_routes_are_advertised() -> None:
    configs = load_model_configs(MODELS_DIR)

    for model_id in VISUAL_MODELS:
        assert model_id in configs
        assert f"{model_id}:muvera" in configs
        assert "muvera" in configs[model_id].profiles
        default_profile = configs[model_id].resolve_profile("default")
        assert "muvera_config" in default_profile.loadtime
        muvera_profile = configs[model_id].resolve_profile("muvera")
        assert muvera_profile.runtime["muvera"] == {}
        assert muvera_profile.runtime["output_types"] == ["dense"]
        assert muvera_profile.runtime["output_similarity"] == {"dense": "dot"}


def test_visual_muvera_models_remain_available_as_multivector_defaults() -> None:
    configs = load_model_configs(MODELS_DIR)

    for model_id in VISUAL_MODELS:
        encode = configs[model_id].tasks.encode
        assert encode is not None
        assert encode.multivector is not None
        assert encode.dense is None
