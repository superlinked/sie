from __future__ import annotations

from pathlib import Path

from sie_server.core.loader import load_model_configs

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL = "TomoroAI/tomoro-colqwen3-embed-4b"


def test_compact_profile_caps_visual_tokens_and_keeps_the_default_load() -> None:
    configs = load_model_configs(MODELS_DIR)
    config = configs[MODEL]

    compact = config.resolve_profile("compact").loadtime
    default = config.resolve_profile("default").loadtime
    assert compact["max_num_visual_tokens"] == 768
    assert "max_num_visual_tokens" not in default
    # Everything else the default loads with is unchanged.
    assert {k: v for k, v in compact.items() if k != "max_num_visual_tokens"} == default
    # A load-time change is served as its own variant.
    assert f"{MODEL}:compact" in configs
