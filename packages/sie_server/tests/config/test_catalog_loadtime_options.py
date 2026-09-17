"""Every shipped profile's load-time options must reach its adapter.

An adapter absorbs unrecognised keyword arguments, so a load-time option its
constructor does not name is dropped rather than applied. ``load_adapter``
refuses that silence, which turns the refusal into a release gate: a key the
loader consumes on the profile's behalf, or one an adapter takes through
``**kwargs`` by design, has to be declared as such or the model stops loading
entirely.

The catalog is read the way a worker reads it, so every non-default profile is
checked as the variant model the loader serves it as.
"""

from pathlib import Path

import pytest
from sie_server.config.model import ModelConfig
from sie_server.core.loader import (
    load_model_configs,
    reject_unknown_loadtime_options,
    resolve_adapter_class,
)

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# A catalog this much smaller than today's means collection silently stopped
# finding profiles, which would make every assertion below vacuous.
_MIN_PROFILES_CHECKED = 200


def _servable_catalog() -> list[tuple[str, ModelConfig]]:
    """Every config the Python loader can serve, variants included, by id.

    A config with no ``default`` profile is served by the Rust gateway rather
    than through ``load_adapter``, so it is outside this gate.
    """
    servable: list[tuple[str, ModelConfig]] = []
    for sie_id, config in sorted(load_model_configs(_MODELS_DIR).items()):
        try:
            config.resolve_profile("default")
        except ValueError:
            continue
        servable.append((sie_id, config))
    return servable


_CATALOG = _servable_catalog()


@pytest.mark.parametrize(("sie_id", "config"), _CATALOG, ids=[sie_id for sie_id, _ in _CATALOG])
def test_shipped_profile_loadtime_options_are_accepted(sie_id: str, config: ModelConfig) -> None:
    try:
        adapter_class = resolve_adapter_class(config, _MODELS_DIR)
    except ImportError as exc:  # optional serving dependency absent in this environment
        pytest.skip(f"adapter for {sie_id} not importable here: {exc}")

    reject_unknown_loadtime_options(
        adapter_class,
        config.resolve_profile("default").loadtime,
        model_name=config.sie_id,
    )


def test_catalog_coverage_is_not_vacuous() -> None:
    variants = [sie_id for sie_id, _ in _CATALOG if ":" in sie_id]

    assert len(_CATALOG) >= _MIN_PROFILES_CHECKED, (
        f"only {len(_CATALOG)} servable configs were collected from {_MODELS_DIR}; "
        "the per-profile assertions above are no longer covering the catalog"
    )
    assert variants, "no non-default profile variants were collected"


def test_derived_serving_artifact_profile_still_loads() -> None:
    """A derived serving artifact is declared in the profile and consumed by the loader.

    ``_build_adapter_kwargs`` replaces it with ``artifact_path`` and
    ``ct2_compute_type``, so the adapter never names the declared spelling.
    """
    declaring = [
        (sie_id, config)
        for sie_id, config in _CATALOG
        if "serving_artifact" in config.resolve_profile("default").loadtime
    ]
    assert declaring, "no shipped profile declares a derived serving artifact any more"
    for sie_id, config in declaring:
        try:
            adapter_class = resolve_adapter_class(config, _MODELS_DIR)
        except ImportError as exc:
            pytest.skip(f"adapter for {sie_id} not importable here: {exc}")
        reject_unknown_loadtime_options(
            adapter_class,
            config.resolve_profile("default").loadtime,
            model_name=config.sie_id,
        )
