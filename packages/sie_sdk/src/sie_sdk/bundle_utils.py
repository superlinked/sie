from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ModelAdapterInventory = dict[str, tuple[set[str], str | None]]


def _adapter_module_from_path(adapter_path: object) -> str | None:
    if not isinstance(adapter_path, str) or not adapter_path:
        return None
    module_path = adapter_path.split(":", maxsplit=1)[0]
    return module_path or None


def _effective_adapter_path(profiles: dict, profile_name: str) -> str | None:
    current: str | None = profile_name
    seen: set[str] = set()
    while current:
        if current in seen:
            return None
        seen.add(current)
        profile = profiles.get(current)
        if not isinstance(profile, dict):
            return None
        adapter_path = profile.get("adapter_path")
        if isinstance(adapter_path, str) and adapter_path:
            return adapter_path
        parent = profile.get("extends")
        current = parent if isinstance(parent, str) and parent else None
    return None


def _profile_adapter_modules(profiles: dict, profile_name: str) -> set[str]:
    module_path = _adapter_module_from_path(_effective_adapter_path(profiles, profile_name))
    return {module_path} if module_path else set()


def _base_route_adapter_modules(profiles: dict) -> set[str]:
    if "default" not in profiles:
        return set()
    return _profile_adapter_modules(profiles, "default")


def _scan_model_adapters(models_dir: Path) -> ModelAdapterInventory:
    """Scan model config YAMLs and return adapter modules per route identity.

    Args:
        models_dir: Path to the models directory containing *.yaml configs.

    Returns:
        Dict mapping model/profile route name to ``(modules, pool)`` where
        ``modules`` is the set of effective adapter module paths for that route and
        ``pool`` is the optional configured pool name.
    """
    result: ModelAdapterInventory = {}
    if not models_dir.exists():
        return result

    for model_path in sorted(models_dir.glob("*.yaml")):
        try:
            model_data = yaml.safe_load(model_path.read_text()) or {}
        except Exception:
            logger.exception("Failed to parse model config %s", model_path.name)
            continue
        model_name = model_data.get("sie_id", model_path.stem.replace("__", "/"))
        profiles = model_data.get("profiles", {})
        if not isinstance(profiles, dict):
            continue

        raw_pool = model_data.get("pool")
        pool = raw_pool.strip().lower() if isinstance(raw_pool, str) else None
        route_pool = pool or None

        base_modules = _base_route_adapter_modules(profiles)
        if base_modules:
            result[model_name] = (base_modules, route_pool)

        for profile_name in sorted(profiles):
            if profile_name == "default":
                continue
            modules = _profile_adapter_modules(profiles, str(profile_name))
            if modules:
                result[f"{model_name}:{profile_name}"] = (modules, route_pool)

    return result


def match_bundle_model_adapters(
    bundle_path: Path,
    model_adapters: ModelAdapterInventory,
    *,
    pool_name: str | None = None,
) -> list[str]:
    """Match an in-memory model-adapter inventory to a bundle."""
    with bundle_path.open() as f:
        data = yaml.safe_load(f) or {}

    adapter_modules = set(data.get("adapters", []))
    if not adapter_modules:
        return []

    matches: list[str] = []
    normalized_pool_name = pool_name.strip().lower() if pool_name is not None else None
    for name, (modules, pool) in model_adapters.items():
        if normalized_pool_name is not None and (pool or "default") != normalized_pool_name:
            continue
        if modules & adapter_modules:
            matches.append(name)
    return matches


def match_bundle_models(bundle_path: Path, models_dir: Path, *, pool_name: str | None = None) -> list[str]:
    """Match models from a local catalog directory to a bundle by adapter module paths.

    Loads the bundle YAML to get its adapter module list, then scans
    model config YAMLs to find models whose adapter_path module matches.

    Args:
        bundle_path: Path to the bundle YAML file.
        models_dir: Path to the models directory containing *.yaml configs.

    Returns:
        List of model names (sie_id or derived from filename) whose adapters
        match the bundle's adapter list.
    """
    return match_bundle_model_adapters(bundle_path, _scan_model_adapters(models_dir), pool_name=pool_name)


def find_bundle_for_model_adapters(
    model_names: list[str],
    bundles_dir: Path,
    model_adapters: ModelAdapterInventory,
    *,
    pool_name: str | None = None,
) -> str | None:
    """Find the most specific bundle covering models in an adapter inventory."""
    if not model_names or not bundles_dir.exists():
        return None

    needed_adapters: set[str] = set()
    normalized_pool_name = pool_name.strip().lower() if pool_name is not None else None
    for name in model_names:
        modules, pool = model_adapters.get(name, (set(), None))
        if normalized_pool_name is not None and (pool or "default") != normalized_pool_name:
            continue
        needed_adapters |= modules

    if not needed_adapters:
        return None

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
            continue
        extra = len(bundle_adapters - needed_adapters)
        priority = data.get("priority", 50)
        if extra < best_extra or (extra == best_extra and priority < best_priority):
            best_name = bundle_path.stem
            best_extra = extra
            best_priority = priority

    return best_name


def find_bundle_for_models(
    model_names: list[str],
    bundles_dir: Path,
    models_dir: Path,
    *,
    pool_name: str | None = None,
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
        pool_name: Optional pool filter. When set, models whose declared
            pool does not match are excluded from the adapter-set used to
            select a bundle. Mirrors :func:`match_bundle_models`'s
            ``pool_name`` filter so pool isolation holds at the
            bundle-resolution layer too.

    Returns:
        Bundle name (without .yaml) of the best match, or None if no bundle
        covers all requested models.
    """
    if not model_names or not bundles_dir.exists() or not models_dir.exists():
        return None

    model_adapters = _scan_model_adapters(models_dir)
    return find_bundle_for_model_adapters(model_names, bundles_dir, model_adapters, pool_name=pool_name)
