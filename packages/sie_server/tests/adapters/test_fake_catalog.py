"""Fake family catalog-path loading (#1847 acceptance).

Deliberately does NOT import ``sie_server.adapters.fake`` anywhere: the
adapters must be reachable purely through the normal catalog → config →
``load_adapter`` resolution, proving there is no test-only backdoor.

The whole family is ONE catalog file (``models/sie-fake.yaml``); the loader's
profile-variant expansion turns its non-default profiles into addressable
``sie-fake:<profile>`` models.
"""

from __future__ import annotations

from pathlib import Path

from sie_server.core.loader import load_adapter, load_model_configs
from sie_server.types.inputs import Item

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Every case the single YAML declares. The fault-bearing scenario variants
# are asserted without load() — loading sie-fake:slow-load would sleep 5 s.
FAKE_MODEL_IDS = (
    "sie-fake",
    "sie-fake:small-a",
    "sie-fake:small-b",
    "sie-fake:slow-load",
    "sie-fake:oom-3rd",
)
FAULTLESS_IDS = ("sie-fake", "sie-fake:small-a", "sie-fake:small-b")


def test_single_yaml_expands_to_all_cases() -> None:
    configs = load_model_configs(MODELS_DIR)
    for model_id in FAKE_MODEL_IDS:
        assert model_id in configs, f"{model_id} missing from catalog"
        assert configs[model_id].package_backed


def test_fakes_load_via_normal_catalog_path() -> None:
    configs = load_model_configs(MODELS_DIR)
    for model_id in FAULTLESS_IDS:
        adapter = load_adapter(configs[model_id], MODELS_DIR, device="cpu")
        adapter.load("cpu")
        try:
            # Declared footprint flows from YAML loadtime options.
            assert adapter.memory_footprint() > 0
            assert adapter.load_required_memory_bytes(device_type="cpu", device_total_bytes=0) == (
                adapter.memory_footprint()
            )
        finally:
            adapter.unload()


def test_all_surfaces_served_by_one_catalog_model() -> None:
    configs = load_model_configs(MODELS_DIR)
    config = configs["sie-fake"]
    assert set(config.outputs) == {"dense", "score", "tokens"}
    adapter = load_adapter(config, MODELS_DIR, device="cpu")
    adapter.load("cpu")
    try:
        out = adapter.encode([Item(text="t")], output_types=["dense"])
        assert config.tasks.encode is not None
        assert config.tasks.encode.dense is not None
        assert out.dense is not None
        assert out.dense.shape == (1, config.tasks.encode.dense.dim)
        scores = adapter.score(Item(text="q"), [Item(text="a"), Item(text="b")])
        assert len(scores) == 2
    finally:
        adapter.unload()


def test_variant_footprints_from_yaml() -> None:
    configs = load_model_configs(MODELS_DIR)
    base = load_adapter(configs["sie-fake"], MODELS_DIR, device="cpu")
    assert base.memory_footprint() == 134217728
    for variant in ("sie-fake:small-a", "sie-fake:small-b", "sie-fake:slow-load", "sie-fake:oom-3rd"):
        adapter = load_adapter(configs[variant], MODELS_DIR, device="cpu")
        assert adapter.memory_footprint() == 67108864
