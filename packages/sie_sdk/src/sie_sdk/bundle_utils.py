from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _scan_model_adapters(models_dir: Path) -> dict[str, set[str]]:
    """Scan model config YAMLs and return adapter modules per model.

    Args:
        models_dir: Path to the models directory containing *.yaml configs.

    Returns:
        Dict mapping model name to set of adapter module paths.
    """
    result: dict[str, set[str]] = {}
    if not models_dir.exists():
        return result

    for model_path in sorted(models_dir.glob("*.yaml")):
        try:
            model_data = yaml.safe_load(model_path.read_text()) or {}
        except Exception:
            logger.exception("Failed to parse model config %s", model_path.name)
            continue
        model_name = model_data.get("sie_id", model_path.stem.replace("__", "/"))
        modules: set[str] = set()
        for profile in model_data.get("profiles", {}).values():
            adapter_path = profile.get("adapter_path", "")
            module_path = adapter_path.split(":", maxsplit=1)[0]
            if module_path:
                modules.add(module_path)
        if modules:
            result[model_name] = modules

    return result


def match_bundle_models(bundle_path: Path, models_dir: Path) -> list[str]:
    """Match models to a bundle by adapter module paths.

    Loads the bundle YAML to get its adapter module list, then scans
    model config YAMLs to find models whose adapter_path module matches.

    Args:
        bundle_path: Path to the bundle YAML file.
        models_dir: Path to the models directory containing *.yaml configs.

    Returns:
        List of model names (sie_id or derived from filename) whose adapters
        match the bundle's adapter list.
    """
    with bundle_path.open() as f:
        data = yaml.safe_load(f) or {}

    adapter_modules = set(data.get("adapters", []))
    if not adapter_modules:
        return []

    model_adapters = _scan_model_adapters(models_dir)
    return [name for name, modules in model_adapters.items() if modules & adapter_modules]


def find_bundle_for_models(
    model_names: list[str],
    bundles_dir: Path,
    models_dir: Path,
) -> str | None:
    """Find the best bundle whose adapters cover the given models.

    Scans all bundle YAMLs in bundles_dir and returns the one whose adapter
    set covers all requested models with the fewest extra adapters (most
    specific match). Ties are broken by bundle priority (lower = higher
    priority).

    Args:
        model_names: List of model names to match.
        bundles_dir: Path to the bundles directory.
        models_dir: Path to the models directory containing *.yaml configs.

    Returns:
        Bundle name (without .yaml) of the best match, or None if no bundle
        covers all requested models.
    """
    if not model_names or not bundles_dir.exists() or not models_dir.exists():
        return None

    # Collect adapter modules needed by the requested models
    model_adapters = _scan_model_adapters(models_dir)
    needed_adapters: set[str] = set()
    for name in model_names:
        needed_adapters |= model_adapters.get(name, set())

    if not needed_adapters:
        return None

    # Score each bundle: must cover all needed adapters
    best_name: str | None = None
    best_extra = float("inf")
    best_priority = float("inf")

    for bundle_path in sorted(bundles_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(bundle_path.read_text()) or {}
        except Exception:
            logger.exception("Failed to parse bundle %s", bundle_path.name)
            continue
        bundle_adapters = set(data.get("adapters", []))
        if not needed_adapters <= bundle_adapters:
            continue  # doesn't cover all needed adapters
        extra = len(bundle_adapters - needed_adapters)
        priority = data.get("priority", 50)
        if extra < best_extra or (extra == best_extra and priority < best_priority):
            best_name = bundle_path.stem
            best_extra = extra
            best_priority = priority

    return best_name
