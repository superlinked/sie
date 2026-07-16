from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import orjson
import yaml

logger = logging.getLogger(__name__)

# Routable profile fields included in bundle_config_hash.
# Both config service (model_registry) and server (ws.py) must use the same set.
_PROFILE_HASH_FIELDS = ("adapter_path", "max_batch_tokens", "compute_precision", "adapter_options")


def _canonical_profile_dict(profile: dict) -> dict:
    """Extract canonical routable fields from a raw profile dict for hashing."""
    return {
        k: _canonical_adapter_options(profile.get(k)) if k == "adapter_options" else profile.get(k)
        for k in _PROFILE_HASH_FIELDS
    }


def _is_hash_falsy(value: object) -> bool:
    """Falsy for bundle-hash canonicalization: ``None``, ``False``, ``0`` / ``0.0``,
    or an empty ``str``/``list``/``tuple``/``dict`` — mirrors the gateway's Rust
    ``canonicalize_adapter_options`` (see #1542).
    """
    if value is None or value is False:
        return True
    if isinstance(value, int | float) and value == 0:
        return True
    return isinstance(value, str | list | tuple | dict) and len(value) == 0


def _canonical_adapter_options(adapter_options: object) -> object | None:
    """Mirror the gateway (Rust) ``canonicalize_adapter_options`` for adapter_options
    in bundle hashes: drop keys whose value is an empty nested map, then collapse to
    ``None`` when every remaining value is falsy (null/false/0/0.0/``""``/``[]``/``{}``).
    Diverging from the gateway leaves workers stuck in ``pending_workers`` (#1542).
    """
    if not isinstance(adapter_options, dict):
        return adapter_options
    compact = {key: value for key, value in adapter_options.items() if not (isinstance(value, dict) and not value)}
    if not compact or all(_is_hash_falsy(value) for value in compact.values()):
        return None
    return compact


def _merge_hash_adapter_options(parent_options: object, child_options: object) -> object | None:
    """Merge inherited adapter_options using the worker's profile resolution rules."""
    if not isinstance(child_options, dict):
        return (
            _canonical_adapter_options(child_options)
            if child_options is not None
            else _canonical_adapter_options(parent_options)
        )

    merged: dict[str, object] = {}
    if isinstance(parent_options, dict):
        merged.update({str(key): copy.deepcopy(value) for key, value in parent_options.items()})
    for key_obj, value in child_options.items():
        key = str(key_obj)
        if key in {"loadtime", "runtime"} and isinstance(value, dict):
            if value:
                merged[key] = copy.deepcopy(value)
            continue
        merged[key] = copy.deepcopy(value)
    return _canonical_adapter_options(merged)


def _resolved_profile_hash_config(
    profiles: dict,
    profile_name: str,
    seen: set[str] | None = None,
) -> dict[str, object] | None:
    """Return hash-relevant profile fields after applying ``extends``."""
    if seen is None:
        seen = set()
    if profile_name in seen:
        return None
    seen.add(profile_name)

    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        return None

    parent_name = profile.get("extends")
    if isinstance(parent_name, str) and parent_name:
        resolved = _resolved_profile_hash_config(profiles, parent_name, seen)
        if resolved is None:
            return None
    else:
        resolved = {
            "adapter_path": None,
            "max_batch_tokens": None,
            "compute_precision": None,
            "adapter_options": None,
        }

    for key in ("adapter_path", "max_batch_tokens", "compute_precision"):
        value = profile.get(key)
        if value is not None:
            resolved[key] = value

    if "adapter_options" in profile:
        resolved["adapter_options"] = _merge_hash_adapter_options(
            resolved.get("adapter_options"),
            profile.get("adapter_options"),
        )
    else:
        resolved["adapter_options"] = _canonical_adapter_options(resolved.get("adapter_options"))

    return resolved


_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _adapter_module_from_path(adapter_path: object) -> str | None:
    """Extract ``module`` from ``module:Class`` adapter paths."""
    if not isinstance(adapter_path, str) or not adapter_path:
        return None
    module_path = adapter_path.split(":", maxsplit=1)[0]
    return module_path or None


def _validate_profile_name(profile_name: str) -> None:
    """Validate profile names that become ``base:profile`` route ids."""
    if not _PROFILE_NAME_RE.fullmatch(profile_name):
        msg = (
            f"Profile name '{profile_name}' must match [A-Za-z0-9._-]+ (no '/', ':', whitespace, or other punctuation)"
        )
        raise ValueError(msg)


def _effective_adapter_path(profiles: dict, profile_name: str) -> str | None:
    """Return the adapter path for a profile, following ``extends`` if needed."""
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


def _adapter_modules_for_profiles(profiles: dict, profile_names: set[str] | None = None) -> set[str]:
    """Collect effective adapter modules for selected profiles."""
    selected = profile_names if profile_names is not None else set(profiles.keys())
    modules: set[str] = set()
    for profile_name in selected:
        adapter_path = _effective_adapter_path(profiles, profile_name)
        module_path = _adapter_module_from_path(adapter_path)
        if module_path:
            modules.add(module_path)
    return modules


def _base_route_adapter_modules(profiles: dict) -> set[str]:
    """Adapter modules used when callers request the base model id."""
    if "default" not in profiles:
        return set()
    module_path = _adapter_module_from_path(_effective_adapter_path(profiles, "default"))
    return {module_path} if module_path else set()


def _matching_bundle_names(adapter_modules: set[str], bundles: dict[str, BundleInfo]) -> list[str]:
    """Return bundle ids compatible with the provided adapter modules."""
    if not adapter_modules:
        return []
    matching_bundles: list[tuple[int, str]] = []
    for bundle in bundles.values():
        if adapter_modules & set(bundle.adapters):
            matching_bundles.append((bundle.priority, bundle.name))
    matching_bundles.sort(key=lambda x: (x[0], x[1]))
    return [bundle_name for _, bundle_name in matching_bundles]


class ModelNotFoundError(Exception):
    """Model not found in any bundle (HTTP 404)."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Model not found: {model}")


class BundleConflictError(Exception):
    """Bundle override incompatible with model (HTTP 409)."""

    def __init__(self, model: str, bundle: str, compatible_bundles: list[str]) -> None:
        self.model = model
        self.bundle = bundle
        self.compatible_bundles = compatible_bundles
        super().__init__(
            f"Bundle '{bundle}' does not support model '{model}'. Compatible bundles: {compatible_bundles}"
        )


class ProfileConflictError(ValueError):
    """Existing profile cannot be changed through the append-only Config API."""

    def __init__(self, model: str, profiles: list[str]) -> None:
        self.model = model
        self.profiles = profiles
        super().__init__(
            f"Profile(s) {profiles} on model '{model}' already exist with different content. Config API is append-only."
        )


# Set of recognised execution engines for the ``engine`` bundle field.
# Locks in the disjoint-bundles convention discussed in the IPC/UDS audit
# (2026-04-26): a model that's served by two different engines declares
# two profiles, each pointing at the namespaced adapter, and the gateway
# routes to the bundle whose ``engine`` matches the worker image at hand.
#
# Add new engines here only when there's a real worker image speaking that
# backend — every value must round-trip through the gateway's bundle
# resolution and the worker's IPC contract. Drift is the easiest way to
# silently mis-route traffic, so we surface unknown values as a config
# load error rather than tolerating them.
KNOWN_ENGINES: frozenset[str] = frozenset({"pytorch", "candle"})
DEFAULT_ENGINE: str = "pytorch"
DEFAULT_MODEL_POOL: str = "default"
_MAX_POOL_NAME_LEN = 128

# Adapter-module prefix expected for each engine. The gateway's matcher
# is engine-agnostic (it intersects ``bundle.adapters`` with the model's
# ``adapter_path`` modules), so this constraint is enforced at config-
# load time rather than at the matcher level — a mismatch is a deploy
# bug, not a runtime fall-through. Extend this in lock-step with
# KNOWN_ENGINES (mirroring
# ``sie_gateway::types::bundle::engine_adapter_prefixes``).
_ENGINE_ADAPTER_PREFIXES: dict[str, tuple[str, ...]] = {
    "pytorch": ("sie_server.adapters.",),
    "candle": ("sie_server_rust.adapters.candle",),
}


def _adapter_mismatches_engine(adapter: str, engine: str) -> bool:
    expected_prefixes = _ENGINE_ADAPTER_PREFIXES.get(engine, ())
    if engine == DEFAULT_ENGINE:
        foreign_prefixes = tuple(
            prefix
            for other_engine, prefixes in _ENGINE_ADAPTER_PREFIXES.items()
            if other_engine != engine
            for prefix in prefixes
        )
        return adapter.startswith(foreign_prefixes)
    return bool(expected_prefixes) and not adapter.startswith(expected_prefixes)


def _normalize_pool_name(config: dict) -> None:
    """Validate and canonicalize an optional model-level pool name."""
    if "pool" not in config:
        return

    raw_pool = config.get("pool")
    if raw_pool is None:
        config.pop("pool", None)
        return
    if not isinstance(raw_pool, str):
        msg = "Field 'pool' must be a string"
        raise ValueError(msg)

    pool = raw_pool.strip().lower()
    if not pool:
        config.pop("pool", None)
        return

    if (
        len(pool) > _MAX_POOL_NAME_LEN
        or pool == "_default"
        or not all(c.isascii() and (c.isalnum() or c in "_-") for c in pool)
    ):
        msg = "Field 'pool' must match [A-Za-z0-9_-]{1,128} and must not be _default"
        raise ValueError(msg)

    config["pool"] = pool


def _model_pool_name(full_config: dict[str, object] | None) -> str:
    """Return the canonical routing pool for a model config snapshot."""
    if not full_config:
        return DEFAULT_MODEL_POOL
    raw_pool = full_config.get("pool")
    if not isinstance(raw_pool, str):
        return DEFAULT_MODEL_POOL
    pool = raw_pool.strip().lower()
    return pool or DEFAULT_MODEL_POOL


@dataclass
class BundleInfo:
    """Information about a bundle.

    Attributes:
        name: Bundle identifier (matches the YAML filename stem unless
            overridden by an explicit ``name:`` field).
        priority: Lower wins. The gateway's ``resolve_bundle`` returns
            the lowest-priority compatible bundle by default.
        adapters: Python module paths (e.g.
            ``sie_server.adapters.<family>``) that this bundle's
            worker image can serve.
        default: Whether this is the operator-blessed fallback bundle
            in the event of an ambiguous resolution. Surfaced as
            metadata; the priority field is what actually decides ties.
        engine: Execution engine the bundle's worker image speaks
            (``"pytorch"`` for the Python adapter image, ``"candle"``
            for the native Rust worker; see ``KNOWN_ENGINES``).
            Defaults to ``"pytorch"`` for back-compat with bundle YAMLs
            written before this field existed. The gateway uses
            ``engine`` to disambiguate when a model resolves to
            multiple bundles.
    """

    name: str
    priority: int
    adapters: list[str] = field(default_factory=list)
    default: bool = False
    engine: str = DEFAULT_ENGINE


@dataclass
class ModelInfo:
    """Information about a model and its compatible bundles."""

    name: str
    bundles: list[str] = field(default_factory=list)  # Ordered by priority (best first)


class ModelRegistry:
    """Source of truth for model->bundle mappings.

    Thread-safe registry that loads bundle and model configurations
    and provides bundle resolution for routing decisions.

    Attributes:
        bundles_dir: Path to bundles directory.
        models_dir: Path to models directory.
    """

    def __init__(
        self,
        bundles_dir: Path | str,
        models_dir: Path | str,
        *,
        auto_load: bool = True,
    ) -> None:
        """Initialize ModelRegistry.

        Args:
            bundles_dir: Path to directory containing bundle YAML files.
            models_dir: Path to directory containing model configs.
            auto_load: If True, load configs immediately. Set False for testing.
        """
        self._bundles_dir = Path(bundles_dir) if isinstance(bundles_dir, str) else bundles_dir
        self._models_dir = Path(models_dir) if isinstance(models_dir, str) else models_dir
        self._unrouteable_models: dict[str, set[str]] = {}

        # Protected by lock for thread-safe reload
        self._lock = threading.RLock()
        self._bundles: dict[str, BundleInfo] = {}
        self._models: dict[str, ModelInfo] = {}
        self._model_names_lower: dict[str, str] = {}  # lowercase -> canonical
        self._profile_variant_models: dict[str, ModelInfo] = {}
        self._profile_variant_names_lower: dict[str, str] = {}  # lowercase -> canonical variant
        self._profile_variant_base_models: dict[str, str] = {}  # variant -> base model
        self._model_adapter_modules: dict[str, set[str]] = {}  # model -> adapter modules
        self._model_profiles: dict[str, set[str]] = {}  # model -> profile names
        self._model_profile_configs: dict[str, dict[str, dict]] = {}  # model -> {profile_name: config_dict}
        # Full merged model config (including top-level metadata like
        # `description`, `default_bundle`). Populated on reload() from the
        # on-disk YAML and on add_model_config() via append-only merge.
        # This is the authoritative source for `/v1/configs/export` in
        # no-config-store deployments, where we otherwise have no way to
        # reconstruct the full YAML across multiple profile-append writes.
        self._model_full_configs: dict[str, dict] = {}
        self._bundle_hash_cache: dict[str, str] = {}

        if auto_load:
            self.reload()

    @property
    def bundles_dir(self) -> Path:
        """Path to bundles directory."""
        return self._bundles_dir

    @property
    def models_dir(self) -> Path:
        """Path to models directory."""
        return self._models_dir

    @property
    def unrouteable_models(self) -> dict[str, set[str]]:
        """Snapshot of models that loaded but have no matching bundle.

        Keyed by model name; value is the set of adapter modules referenced
        by that model's profiles which no bundle declares. Empty when the
        registry is consistent. Intended for readiness probes and metrics
        -- the registry itself does not block startup on a non-empty set.
        """
        with self._lock:
            return {m: set(mods) for m, mods in self._unrouteable_models.items()}

    def reload(self) -> None:
        """Reload all configs from disk.

        Thread-safe: builds new state in temp structures, then swaps atomically.
        """
        new_bundles: dict[str, BundleInfo] = {}
        new_models: dict[str, ModelInfo] = {}
        new_model_names_lower: dict[str, str] = {}
        new_profile_variant_models: dict[str, ModelInfo] = {}
        new_profile_variant_names_lower: dict[str, str] = {}
        new_profile_variant_base_models: dict[str, str] = {}
        new_model_adapter_modules: dict[str, set[str]] = {}
        new_model_route_adapter_modules: dict[str, set[str]] = {}
        new_model_profiles: dict[str, set[str]] = {}
        new_model_profile_configs: dict[str, dict[str, dict]] = {}
        new_model_full_configs: dict[str, dict] = {}

        # Load bundles
        if self._bundles_dir.exists():
            for bundle_path in self._bundles_dir.glob("*.yaml"):
                try:
                    with bundle_path.open() as f:
                        data = yaml.safe_load(f) or {}

                    name = data.get("name", bundle_path.stem)
                    priority = data.get("priority", 100)
                    adapters = data.get("adapters", [])
                    default = data.get("default", False)
                    # ``engine`` defaults to ``DEFAULT_ENGINE`` (== "pytorch")
                    # so existing bundle YAMLs keep working without edits.
                    # Unknown values are a hard error — silently coercing
                    # to "pytorch" would let a typo'd ``engine: pytroch``
                    # mis-route traffic and push diagnosis to the
                    # symptom side.
                    engine = data.get("engine", DEFAULT_ENGINE)
                    if engine not in KNOWN_ENGINES:
                        logger.error(
                            "Bundle %r at %s declares engine=%r which is not in "
                            "KNOWN_ENGINES=%s — skipping load. Update the bundle "
                            "YAML or add the engine to sie_config.model_registry.",
                            name,
                            bundle_path,
                            engine,
                            sorted(KNOWN_ENGINES),
                        )
                        continue

                    # Adapter-namespace consistency check. The gateway's
                    # matcher intersects ``bundle.adapters`` with each
                    # model's ``adapter_path`` modules, so a bundle
                    # that accidentally lists adapters outside this
                    # engine's namespace would produce
                    # ``UnsupportedModel`` errors at runtime. Catching
                    # this at config-load time makes the failure mode
                    # an obvious deploy-rejection rather than a stream
                    # of cryptic IPC NAKs.
                    bad_adapters = [a for a in adapters if _adapter_mismatches_engine(a, engine)]
                    if bad_adapters:
                        logger.error(
                            "Bundle %r (engine=%r) lists adapter(s) outside the "
                            "expected namespace(s) %s: %s — these will not be "
                            "servable by a worker speaking this engine; skipping load.",
                            name,
                            engine,
                            _ENGINE_ADAPTER_PREFIXES.get(engine, ()),
                            bad_adapters,
                        )
                        continue

                    new_bundles[name] = BundleInfo(
                        name=name,
                        priority=priority,
                        adapters=adapters,
                        default=default,
                        engine=engine,
                    )
                    logger.debug(
                        "Loaded bundle '%s': priority=%d, engine=%s, adapters=%d",
                        name,
                        priority,
                        engine,
                        len(adapters),
                    )

                except Exception:
                    logger.exception("Failed to load bundle: %s", bundle_path)
        else:
            logger.warning("Bundles directory not found: %s", self._bundles_dir)

        # Load models
        if self._models_dir.exists():
            for config_path in self._models_dir.glob("*.yaml"):
                if not config_path.is_file():
                    continue

                try:
                    with config_path.open() as f:
                        config = yaml.safe_load(f)

                    raw_model_name = config.get("sie_id") or config.get("name")
                    if raw_model_name:
                        model_name = str(raw_model_name)
                        profiles = config.get("profiles", {})
                        if not isinstance(profiles, dict):
                            msg = f"Model '{model_name}' profiles must be a mapping"
                            raise ValueError(msg)

                        profile_names = {str(profile_name) for profile_name in profiles}
                        for profile_name in profile_names:
                            if profile_name != "default":
                                _validate_profile_name(profile_name)

                        adapter_modules = _adapter_modules_for_profiles(profiles)
                        route_adapter_modules = _base_route_adapter_modules(profiles)

                        new_models[model_name] = ModelInfo(name=model_name)
                        new_model_adapter_modules[model_name] = adapter_modules
                        new_model_route_adapter_modules[model_name] = route_adapter_modules
                        new_model_names_lower[model_name.lower()] = model_name
                        logger.debug("Discovered model: %s (adapters: %s)", model_name, adapter_modules)
                        new_model_profiles[model_name] = profile_names
                        new_model_profile_configs[model_name] = {
                            str(pname): _canonical_profile_dict(pdata) for pname, pdata in profiles.items()
                        }
                        if isinstance(config, dict):
                            new_model_full_configs[model_name] = dict(config)

                        for profile_name in sorted(profiles):
                            if profile_name == "default":
                                continue
                            adapter_path = _effective_adapter_path(profiles, profile_name)
                            module_path = _adapter_module_from_path(adapter_path)
                            variant_modules = {module_path} if module_path else set()
                            variant_name = f"{model_name}:{profile_name}"
                            new_profile_variant_models[variant_name] = ModelInfo(
                                name=variant_name,
                                bundles=_matching_bundle_names(variant_modules, new_bundles),
                            )
                            new_profile_variant_names_lower[variant_name.lower()] = variant_name
                            new_profile_variant_base_models[variant_name] = model_name

                except Exception:
                    logger.exception("Failed to load model config: %s", config_path)
        else:
            logger.warning("Models directory not found: %s", self._models_dir)

        # Compute mappings
        for model_name, model_info in new_models.items():
            route_adapter_modules = new_model_route_adapter_modules.get(model_name, set())
            model_info.bundles = _matching_bundle_names(route_adapter_modules, new_bundles)

        # Detect baked-in inconsistency: any model profile whose adapter
        # module is not declared in *any* bundle. We report the missing
        # modules per model -- not just models where every profile is
        # broken. A model with one good profile and one orphan profile
        # still has ``ModelInfo.bundles`` non-empty (bundles are the union
        # across profiles) so the naive "bundles == []" check would hide
        # real drift. Mirrors ``add_model_config`` which rejects the same
        # shape at write time via ``new_adapter_modules - all_bundle_adapters``.
        #
        # Not fatal: routable profiles keep serving traffic, the log below
        # surfaces the bad profiles in sie-config logs without waiting for
        # every sie-gateway to hit "Adapter(s) not in any known bundle"
        # during bootstrap. The bundle-coverage regression test catches
        # this pre-merge in CI.
        all_bundle_adapters: set[str] = set()
        for bundle in new_bundles.values():
            all_bundle_adapters.update(bundle.adapters)

        unrouteable: dict[str, set[str]] = {}
        for model_name in new_models:
            adapter_modules = new_model_adapter_modules.get(model_name, set())
            missing = adapter_modules - all_bundle_adapters
            if missing:
                unrouteable[model_name] = missing

        if unrouteable:
            logger.error(
                "ModelRegistry: %d model(s) reference adapter modules not declared in any bundle "
                "(profiles pinned to those modules are unrouteable, other profiles keep working). "
                "Fix by adding the module to a bundle (packages/sie_server/bundles/*.yaml) or "
                "removing the profile. Missing modules per model: %s",
                len(unrouteable),
                {m: sorted(mods) for m, mods in sorted(unrouteable.items())},
            )

        # Atomic swap under lock. Replace each top-level dict with a new
        # one (rather than .clear() + .update()) so lock-free readers
        # can't observe a partially-populated state.
        with self._lock:
            self._bundles = new_bundles
            self._models = new_models
            self._model_names_lower = new_model_names_lower
            self._profile_variant_models = new_profile_variant_models
            self._profile_variant_names_lower = new_profile_variant_names_lower
            self._profile_variant_base_models = new_profile_variant_base_models
            self._model_adapter_modules = new_model_adapter_modules
            self._model_profiles = new_model_profiles
            self._model_profile_configs = new_model_profile_configs
            self._model_full_configs = new_model_full_configs
            self._unrouteable_models = unrouteable
            self._bundle_hash_cache = {}

        logger.info(
            "ModelRegistry loaded: %d bundles, %d models",
            len(new_bundles),
            len(new_models),
        )

    def resolve_bundle(self, model: str, bundle_override: str | None = None) -> str:
        """Resolve which bundle to use for a model.

        Args:
            model: Model name (e.g., "BAAI/bge-m3").
            bundle_override: Optional explicit bundle (e.g., "default").

        Returns:
            Bundle name to use.

        Raises:
            ModelNotFoundError: Model not in any bundle (404).
            BundleConflictError: Override bundle doesn't support model (409).
        """
        with self._lock:
            models = self._models
            model_names_lower = self._model_names_lower
            profile_variant_models = self._profile_variant_models
            profile_variant_names_lower = self._profile_variant_names_lower

            # Normalize model name (case-insensitive lookup)
            canonical_model = model_names_lower.get(model.lower())
            if canonical_model is None:
                canonical_variant = profile_variant_names_lower.get(model.lower())
                if canonical_variant is not None:
                    canonical_model = canonical_variant
                    model_info = profile_variant_models.get(canonical_model)
                else:
                    model_info = None
                # Try exact match
                if model_info is None and model in profile_variant_models:
                    canonical_model = model
                    model_info = profile_variant_models.get(canonical_model)
                elif model_info is None and model in models:
                    canonical_model = model
                    model_info = models.get(canonical_model)
                elif model_info is None:
                    raise ModelNotFoundError(model)
            else:
                model_info = models.get(canonical_model)
            if model_info is None or not model_info.bundles:
                raise ModelNotFoundError(model)

            if bundle_override is not None:
                # Validate override is compatible
                if bundle_override not in model_info.bundles:
                    raise BundleConflictError(
                        model=model,
                        bundle=bundle_override,
                        compatible_bundles=model_info.bundles,
                    )
                return bundle_override

            # Return highest priority (first in sorted list)
            return model_info.bundles[0]

    def get_model_info(self, model: str) -> ModelInfo | None:
        """Get model info including compatible bundles.

        Args:
            model: Model name.

        Returns:
            ModelInfo if found, None otherwise.
        """
        with self._lock:
            # Try case-insensitive lookup first
            canonical = self._model_names_lower.get(model.lower())
            if canonical:
                return self._models.get(canonical)
            variant = self._profile_variant_names_lower.get(model.lower())
            if variant:
                return self._profile_variant_models.get(variant)
            if model in self._profile_variant_models:
                return self._profile_variant_models.get(model)
            return self._models.get(model)

    def list_models(self) -> list[str]:
        """List all known model names.

        Returns:
            Sorted list of model names.
        """
        with self._lock:
            return sorted(self._models.keys())

    def list_serving_models(self) -> list[str]:
        """List routable model/profile identities.

        Unlike :meth:`list_models`, this is a serving surface: profile-only
        configs omit the bare base id and expose only their ``base:profile``
        variants. ``list_models`` intentionally remains catalog inventory so
        export/bootstrap can serialize the base YAML once.
        """
        with self._lock:
            serving: set[str] = {model_name for model_name, model_info in self._models.items() if model_info.bundles}
            serving.update(
                model_name for model_name, model_info in self._profile_variant_models.items() if model_info.bundles
            )
            return sorted(serving)

    def get_catalog_model_name(self, model_name: str) -> str:
        """Return the base catalog config id for a serving route identity."""
        with self._lock:
            canonical_variant = self._profile_variant_names_lower.get(model_name.lower())
            if canonical_variant is not None:
                return self._profile_variant_base_models.get(canonical_variant, canonical_variant)
            if model_name in self._profile_variant_models:
                return self._profile_variant_base_models.get(model_name, model_name)
            canonical = self._model_names_lower.get(model_name.lower())
            return canonical or model_name

    def get_route_profile_names(self, model_name: str) -> set[str]:
        """Return profile names represented by a serving route identity."""
        with self._lock:
            canonical_variant = self._profile_variant_names_lower.get(model_name.lower())
            if canonical_variant is not None:
                return {canonical_variant.rsplit(":", maxsplit=1)[1]}
            if model_name in self._profile_variant_models:
                return {model_name.rsplit(":", maxsplit=1)[1]}
            return self.get_model_profile_names(model_name)

    def get_model_profile_bundles(
        self,
        model_name: str,
        profile_names: set[str] | None = None,
    ) -> dict[str, list[str]]:
        """Return compatible bundles for each selected catalog profile."""
        with self._lock:
            canonical = self._model_names_lower.get(model_name.lower(), model_name)
            full_config = self._model_full_configs.get(canonical, {})
            profiles = full_config.get("profiles", {}) if isinstance(full_config, dict) else {}
            if not isinstance(profiles, dict):
                return {}
            selected = profile_names if profile_names is not None else self._model_profiles.get(canonical, set())
            out: dict[str, list[str]] = {}
            for profile_name in sorted(selected):
                if profile_name not in profiles:
                    continue
                module_path = _adapter_module_from_path(_effective_adapter_path(profiles, profile_name))
                modules = {module_path} if module_path else set()
                out[profile_name] = _matching_bundle_names(modules, self._bundles)
            return out

    def get_model_export_bundles(self, model_name: str) -> list[str]:
        """Return union of bundles targeted by any profile on a catalog model."""
        with self._lock:
            canonical = self._model_names_lower.get(model_name.lower(), model_name)
            full_config = self._model_full_configs.get(canonical, {})
            profiles = full_config.get("profiles", {}) if isinstance(full_config, dict) else {}
            if not isinstance(profiles, dict):
                return []
            modules = _adapter_modules_for_profiles(profiles, self._model_profiles.get(canonical, set()))
            return _matching_bundle_names(modules, self._bundles)

    def get_model_pool_name(self, model_name: str) -> str:
        """Return the canonical routing pool for a catalog model."""
        with self._lock:
            canonical = self._model_names_lower.get(model_name.lower(), model_name)
            return _model_pool_name(self._model_full_configs.get(canonical, {}))

    def list_bundles(self) -> list[str]:
        """List all bundle names.

        Returns:
            List of bundle names sorted by priority.
        """
        with self._lock:
            bundles_sorted = sorted(self._bundles.values(), key=lambda b: b.priority)
            return [b.name for b in bundles_sorted]

    def compute_bundles_hash(self) -> str:
        """Stable SHA-256 over the registry's full bundle surface.

        The gateway polls this via ``GET /v1/configs/epoch`` and re-fetches
        ``/v1/configs/bundles`` whenever the hash changes (see
        ``packages/sie_gateway/src/state/config_poller.rs``). Without a
        bundle-level change signal the gateway would only learn about a
        sie-config redeploy that introduced a new bundle when (a) the gateway
        itself restarts or (b) something happens to bump the model epoch — an
        unintuitive coupling that lets every worker for the new bundle get
        WebSocket-rejected with ``unknown_bundle_id`` until the next manual
        kick.

        Hash inputs are sorted at every level so two registries holding the
        same bundles in different load order produce the same hash. We hash
        the JSON representation rather than the on-disk YAML because YAML
        serialization is non-canonical (key order, quoting, line endings) and
        would produce spurious deltas on every redeploy.

        Returns the empty string when no bundles are loaded — the gateway
        treats an empty hash as "nothing to sync" rather than as a real value
        worth storing, which keeps fresh clusters from oscillating between
        empty and populated states during the bootstrap window.
        """
        with self._lock:
            bundles = list(self._bundles.values())
        if not bundles:
            return ""
        canonical = [
            {
                "name": b.name,
                "priority": b.priority,
                "adapters": sorted(b.adapters),
                "engine": b.engine,
            }
            for b in sorted(bundles, key=lambda x: x.name)
        ]
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def compute_bundle_config_hashes_hash(self) -> str:
        """Stable SHA-256 over per-bundle config hashes and pool ownership.

        The gateway polls this via ``GET /v1/configs/epoch`` to detect
        control-plane config drift even when the model epoch is unchanged (for
        example, no-store deployments where a sie-config image redeploy
        rebuilds the filesystem baseline at epoch 0). ``bundle_config_hash``
        intentionally stays worker-parity scoped to runtime profile fields, so
        this compact fingerprint also includes model-level pool ownership: a
        pure top-level ``pool:`` move changes routing/readiness even when the
        worker-applied config hash for each bundle is otherwise unchanged.
        """
        by_bundle = {}
        with self._lock:
            bundle_ids = sorted(self._bundles)
            if not bundle_ids:
                return ""
            for bundle_id in bundle_ids:
                model_pools = {}
                for model_name in sorted(self._models):
                    if bundle_id not in self.get_model_export_bundles(model_name):
                        continue
                    model_pools[model_name] = _model_pool_name(self._model_full_configs.get(model_name, {}))
                by_bundle[bundle_id] = {
                    "bundle_config_hash": self.compute_bundle_config_hash(bundle_id),
                    "model_pools": model_pools,
                }
        payload = json.dumps(by_bundle, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_bundle_info(self, bundle: str) -> BundleInfo | None:
        """Get bundle info.

        Args:
            bundle: Bundle name.

        Returns:
            BundleInfo if found, None otherwise.
        """
        with self._lock:
            return self._bundles.get(bundle)

    def get_models_for_bundle(self, bundle: str) -> list[str]:
        """Get all models that can be served by a bundle.

        Args:
            bundle: Bundle name.

        Returns:
            List of model names.
        """
        with self._lock:
            models = self._models
            profile_variant_models = self._profile_variant_models
            return [
                model_name
                for model_name, model_info in {**models, **profile_variant_models}.items()
                if bundle in model_info.bundles
            ]

    def model_exists(self, model: str) -> bool:
        """Check if a model exists in the registry.

        Args:
            model: Model name.

        Returns:
            True if model is known, False otherwise.
        """
        with self._lock:
            if model.lower() in self._model_names_lower:
                return True
            if model.lower() in self._profile_variant_names_lower:
                return True
            return model in self._models or model in self._profile_variant_models

    def validate_model_config(
        self,
        config: dict,
    ) -> tuple[list[str], list[str], list[str]]:
        """Validate a model config without mutating the registry.

        Returns the `(created, skipped, affected_bundles)` triple that
        `add_model_config` would produce, or raises on invalid input.
        Intended for callers that need to persist the config to disk
        *before* taking the registry mutation — on a subsequent
        `add_model_config` call, a validation error therefore cannot
        leave a half-mutated registry pointing at a non-existent file.

        Raises:
            ValueError: If validation fails (missing fields, unroutable
                adapter, append-only conflict).
        """
        with self._lock:
            return self._validate_config_locked(config)

    def _validate_config_locked(
        self,
        config: dict,
    ) -> tuple[list[str], list[str], list[str]]:
        """Run every input-level check without touching mutable state.

        Must be called with `self._lock` held. Returns the same triple
        `(created, skipped, affected_bundles)` that a successful
        `add_model_config` would produce so callers can make 409-conflict
        decisions before persisting.
        """
        sie_id = config.get("sie_id")
        if not sie_id:
            msg = "Missing required field: sie_id"
            raise ValueError(msg)

        _normalize_pool_name(config)
        incoming_has_pool = "pool" in config
        incoming_pool = config.get("pool") or "default"

        profiles = config.get("profiles", {})
        if not profiles:
            msg = "Missing required field: profiles"
            raise ValueError(msg)
        if not isinstance(profiles, dict):
            msg = "Field 'profiles' must be a mapping of profile_name -> profile_config"
            raise ValueError(msg)

        for profile_name, profile in profiles.items():
            _validate_profile_name(str(profile_name))
            if not isinstance(profile, dict):
                msg = f"Profile '{profile_name}' must be a mapping"
                raise ValueError(msg)
            adapter_path = _effective_adapter_path(profiles, str(profile_name))
            if not adapter_path:
                if not profile.get("extends"):
                    msg = f"Profile '{profile_name}' missing adapter_path"
                    raise ValueError(msg)
                continue

        existing_full_profiles = self._model_full_configs.get(sie_id, {}).get("profiles", {})
        merged_profiles_for_routing = dict(existing_full_profiles) if isinstance(existing_full_profiles, dict) else {}
        merged_profiles_for_routing.update(profiles)
        new_adapter_modules = _adapter_modules_for_profiles(
            merged_profiles_for_routing,
            {str(profile_name) for profile_name in profiles},
        )

        all_bundle_adapters: set[str] = set()
        for bundle in self._bundles.values():
            all_bundle_adapters.update(bundle.adapters)

        unroutable = new_adapter_modules - all_bundle_adapters
        if unroutable:
            msg = f"Adapter(s) not in any known bundle: {', '.join(sorted(unroutable))}"
            raise ValueError(msg)

        existing = self._models.get(sie_id)
        created_profiles: list[str] = []
        skipped_profiles: list[str] = []

        if existing:
            existing_pool = self._model_full_configs.get(sie_id, {}).get("pool") or "default"
            if incoming_has_pool and incoming_pool != existing_pool:
                msg = f"Pool on model '{sie_id}' already exists with different value. Config API is append-only."
                raise ValueError(msg)
            existing_full_profiles = self._model_full_configs.get(sie_id, {}).get("profiles", {})
            conflicting_profiles: list[str] = []
            for profile_name, profile in profiles.items():
                if profile_name in self.get_model_profile_names(sie_id):
                    stored_full = existing_full_profiles.get(profile_name)
                    if stored_full is not None and stored_full != profile:
                        conflicting_profiles.append(profile_name)
                        continue
                    stored = self._model_profile_configs.get(sie_id, {}).get(profile_name)
                    incoming = _canonical_profile_dict(profile)
                    if stored is not None and stored != incoming:
                        conflicting_profiles.append(profile_name)
                        continue
                    skipped_profiles.append(profile_name)
                else:
                    created_profiles.append(profile_name)
            if conflicting_profiles:
                raise ProfileConflictError(sie_id, conflicting_profiles)
        else:
            created_profiles = list(profiles.keys())  # type: ignore

        # Compute the post-apply adapter set so bundle mappings reflect
        # the hypothetical new state. This is a pure computation on a
        # local copy — no self._* mutation.
        post_adapter_modules = _adapter_modules_for_profiles(merged_profiles_for_routing)
        affected_bundles = _matching_bundle_names(post_adapter_modules, self._bundles)

        # Reject changes that land in zero bundles. `extends`-only
        # profiles skip the `adapter_path` collection step, so
        # `new_adapter_modules` can legally be empty for a brand-new
        # model — which yields `affected_bundles == []`. Persisting that
        # produces a model that is accepted by the registry but cannot
        # be routed to any worker bundle (and the NATS publish step is
        # skipped because there are no affected bundles).
        #
        # Pure replays (no created profiles) are allowed to return an
        # empty list; when we are adding profiles, the result must
        # resolve to at least one bundle.
        if created_profiles and not affected_bundles:
            if not existing:
                msg = (
                    f"Model '{sie_id}' does not resolve to any known bundle. "
                    "Every new model must contribute at least one profile whose "
                    "`adapter_path` module appears in a bundle's adapters list. "
                    "(Profiles that only `extends` another profile do not count.)"
                )
            else:
                msg = (
                    f"Appending profile(s) {created_profiles} to model '{sie_id}' "
                    "does not affect any bundle. Ensure at least one new profile "
                    "has an `adapter_path` that a known bundle owns."
                )
            raise ValueError(msg)

        return created_profiles, skipped_profiles, affected_bundles

    def add_model_config(
        self,
        config: dict,
    ) -> tuple[list[str], list[str], list[str]]:
        """Add a model config at runtime (from Config API or NATS notification).

        Validates adapter routability and adds the model/profiles to the registry.
        Append-only: existing profiles cannot be modified, only new ones added.

        Args:
            config: Parsed model config dict with sie_id, profiles, etc.

        Returns:
            Tuple of (created_profiles, skipped_profiles, affected_bundles).

        Raises:
            ValueError: If validation fails (missing fields, unroutable adapter).
            BundleConflictError: If profile already exists with different content.
        """
        with self._lock:
            # Re-run validation inside the same lock scope so concurrent
            # callers can't sneak a conflicting profile in between an
            # earlier validate_model_config() and this mutation.
            created_profiles, skipped_profiles, _ = self._validate_config_locked(config)

            sie_id = config["sie_id"]
            profiles = config["profiles"]
            existing = self._models.get(sie_id)

            # Build the post-mutation state in fresh containers and only
            # swap the top-level dict references at the very end. That way
            # the lock-free readers (`resolve_bundle`, `get_model_info`,
            # `list_models`, ...) either see the pre-state or the
            # post-state, never a half-mutated `ModelInfo.bundles` or
            # `_model_profiles[sie_id]` set. The atomic pointer-swap
            # semantics of CPython dict assignment is what keeps those
            # readers lock-free and correct — in-place mutation (set.add,
            # list.append, dict.update on a live value) would break that.
            new_profile_names = set(self._model_profiles.get(sie_id, set()))
            new_profile_names.update(created_profiles)

            new_profile_configs = dict(self._model_profile_configs.get(sie_id, {}))
            for pname in created_profiles:
                new_profile_configs[pname] = _canonical_profile_dict(profiles[pname])

            # Append-only merge of the full model config. Non-profile
            # top-level fields from the new body are added if absent and
            # preserved if already present; conflicts on existing
            # non-profile fields are rejected in config_api before we get
            # here, so a simple "incoming fills in gaps" semantic is safe.
            #
            # Profiles: only *newly created* profiles are inserted from
            # the incoming body (their full raw dict, so non-hash fields
            # like `extends`, `model_name`, `revision`, custom keys are
            # preserved). Existing profiles are left untouched in
            # `merged_profiles_full` — their original raw form was stored
            # on the first write and overwriting them from
            # `_model_profile_configs` would silently drop non-hash keys,
            # since that cache only holds the canonical 4-field subset.
            new_full_config: dict = dict(self._model_full_configs.get(sie_id, {}))
            for key, value in config.items():
                if key == "profiles":
                    continue
                if key not in new_full_config:
                    new_full_config[key] = value
            merged_profiles_full: dict = dict(new_full_config.get("profiles", {}))
            for pname in created_profiles:
                incoming = profiles.get(pname)
                if isinstance(incoming, dict):
                    merged_profiles_full[pname] = dict(incoming)
            new_full_config["profiles"] = merged_profiles_full
            if "sie_id" not in new_full_config:
                new_full_config["sie_id"] = sie_id

            new_adapter_modules_set = _adapter_modules_for_profiles(merged_profiles_full)
            route_adapter_modules = _base_route_adapter_modules(merged_profiles_full)
            affected_bundles = _matching_bundle_names(new_adapter_modules_set, self._bundles)
            new_model_info = ModelInfo(
                name=sie_id,
                bundles=_matching_bundle_names(route_adapter_modules, self._bundles),
            )

            new_profile_variant_models = {
                name: info
                for name, info in self._profile_variant_models.items()
                if self._profile_variant_base_models.get(name) != sie_id
            }
            new_profile_variant_names_lower = {
                lower: name
                for lower, name in self._profile_variant_names_lower.items()
                if self._profile_variant_base_models.get(name) != sie_id
            }
            new_profile_variant_base_models = {
                name: base for name, base in self._profile_variant_base_models.items() if base != sie_id
            }
            for profile_name in sorted(new_profile_names):
                if profile_name == "default":
                    continue
                adapter_path = _effective_adapter_path(merged_profiles_full, profile_name)
                module_path = _adapter_module_from_path(adapter_path)
                variant_modules = {module_path} if module_path else set()
                variant_name = f"{sie_id}:{profile_name}"
                new_profile_variant_models[variant_name] = ModelInfo(
                    name=variant_name,
                    bundles=_matching_bundle_names(variant_modules, self._bundles),
                )
                new_profile_variant_names_lower[variant_name.lower()] = variant_name
                new_profile_variant_base_models[variant_name] = sie_id
                affected_bundles.extend(new_profile_variant_models[variant_name].bundles)
            affected_bundles.sort()
            affected_bundles = list(dict.fromkeys(affected_bundles))

            # Atomic swaps (CPython guarantees single-pointer-store is
            # not interleaved with a concurrent pointer-load under the GIL).
            self._models = {**self._models, sie_id: new_model_info}
            if not existing:
                self._model_names_lower = {
                    **self._model_names_lower,
                    sie_id.lower(): sie_id,
                }
            self._profile_variant_models = new_profile_variant_models
            self._profile_variant_names_lower = new_profile_variant_names_lower
            self._profile_variant_base_models = new_profile_variant_base_models
            self._model_adapter_modules = {
                **self._model_adapter_modules,
                sie_id: new_adapter_modules_set,
            }
            self._model_profiles = {
                **self._model_profiles,
                sie_id: new_profile_names,
            }
            self._model_profile_configs = {
                **self._model_profile_configs,
                sie_id: new_profile_configs,
            }
            self._model_full_configs = {
                **self._model_full_configs,
                sie_id: new_full_config,
            }

            # Keep the unrouteable snapshot consistent with runtime writes.
            # `_validate_config_locked` already rejects adapters missing
            # from every bundle, so a successful write cannot *introduce*
            # new missing modules -- but a pre-existing stale entry (e.g.
            # registered at a prior reload before a bundle was added) must
            # be cleared now that the model's adapters are known routable.
            # Readiness probes / metrics reading ``unrouteable_models``
            # would otherwise lie until the next reload.
            all_bundle_adapters: set[str] = set()
            for bundle in self._bundles.values():
                all_bundle_adapters.update(bundle.adapters)
            missing_for_sie_id = new_adapter_modules_set - all_bundle_adapters
            new_unrouteable = {m: set(mods) for m, mods in self._unrouteable_models.items()}
            if missing_for_sie_id:
                new_unrouteable[sie_id] = missing_for_sie_id
            else:
                new_unrouteable.pop(sie_id, None)
            self._unrouteable_models = new_unrouteable

            # Hash cache invalidation: swap for an empty dict rather than
            # `.clear()` so any in-flight `compute_bundle_config_hash` call
            # that already captured the old dict reference doesn't see a
            # mid-iteration mutation.
            self._bundle_hash_cache = {}

            logger.info(
                "Added model config: %s (created=%s, skipped=%s, bundles=%s)",
                sie_id,
                created_profiles,
                skipped_profiles,
                affected_bundles,
            )

            return created_profiles, skipped_profiles, affected_bundles

    def _validate_structure_locked(self, config: dict) -> None:
        """Structural + routability checks WITHOUT the append-only conflict gate.

        Must be called with ``self._lock`` held. This is the subset of
        ``_validate_config_locked`` that does not consult the model's prior
        profiles: required ``sie_id``/``profiles``, per-profile shape +
        ``adapter_path`` resolution, and the "every adapter module lands in
        some bundle" routability check. ``replace_model_config`` uses it
        because a REPLACE deliberately overwrites existing profiles (the
        append-only conflict check would reject exactly the stale-entry heal
        it exists to perform), yet must still refuse an unroutable or
        malformed body.
        """
        sie_id = config.get("sie_id")
        if not sie_id:
            msg = "Missing required field: sie_id"
            raise ValueError(msg)

        _normalize_pool_name(config)

        profiles = config.get("profiles", {})
        if not profiles:
            msg = "Missing required field: profiles"
            raise ValueError(msg)
        if not isinstance(profiles, dict):
            msg = "Field 'profiles' must be a mapping of profile_name -> profile_config"
            raise ValueError(msg)

        for profile_name, profile in profiles.items():
            _validate_profile_name(str(profile_name))
            if not isinstance(profile, dict):
                msg = f"Profile '{profile_name}' must be a mapping"
                raise ValueError(msg)
            # A REPLACE discards prior profiles, so _effective_adapter_path
            # resolves ``extends`` against ONLY the incoming set (no merge with
            # existing state, unlike the append-only path). An unresolved value —
            # including a dangling ``extends`` whose parent is absent from this
            # body — must be rejected here: there is no prior state to rescue it,
            # so exempting a profile merely because it declares ``extends`` would
            # persist an unrouteable variant.
            adapter_path = _effective_adapter_path(profiles, str(profile_name))
            if not adapter_path:
                msg = f"Profile '{profile_name}' missing adapter_path"
                raise ValueError(msg)

        # A REPLACE resolves routability against ONLY the incoming profiles
        # (it discards the prior ones), so unlike the append-only path there
        # is no merge with existing profiles here.
        new_adapter_modules = _adapter_modules_for_profiles(profiles)
        all_bundle_adapters: set[str] = set()
        for bundle in self._bundles.values():
            all_bundle_adapters.update(bundle.adapters)
        unroutable = new_adapter_modules - all_bundle_adapters
        if unroutable:
            msg = f"Adapter(s) not in any known bundle: {', '.join(sorted(unroutable))}"
            raise ValueError(msg)

    def replace_model_config(self, config: dict) -> list[str]:
        """Replace a model's config wholesale (the catalog-convergence path).

        Unlike :meth:`add_model_config` (append-only: existing profiles are
        immutable), this REPLACES the model's stored profiles + top-level
        metadata with ``config`` exactly. It exists to heal a catalog entry
        whose stored content has drifted from the current source YAML — the
        append-only POST path 409s on a changed profile, so a differing entry
        can only converge through a deliberate replace.

        Rebuilds the model's registry state (profiles, adapter modules, route
        + profile-variant maps, unrouteable snapshot) from the incoming
        config alone. Structural + routability validation still applies (a
        malformed or unroutable body is rejected). Returns the sorted list of
        bundles the replaced model now affects.

        Raises:
            ValueError: If validation fails (missing fields, unroutable adapter).
        """
        with self._lock:
            self._validate_structure_locked(config)

            sie_id = config["sie_id"]
            profiles = config["profiles"]
            existing = self._models.get(sie_id)

            # Build the post-replace state in fresh containers; only the
            # top-level pointer swaps are visible to lock-free readers, so they
            # observe the pre- or post-state, never a half-rebuilt entry (same
            # atomic-pointer-store discipline as add_model_config).
            new_profile_names = {str(pname) for pname in profiles}
            new_profile_configs = {str(pname): _canonical_profile_dict(pdata) for pname, pdata in profiles.items()}

            new_full_config: dict = dict(config)
            new_full_config["profiles"] = {str(pname): dict(pdata) for pname, pdata in profiles.items()}
            new_full_config["sie_id"] = sie_id

            # Bundles the model belonged to BEFORE this replace (captured before
            # the _model_adapter_modules swap below). A replace that moves or
            # drops a profile changes which bundles route to the model, so its OLD
            # bundles must also be notified — otherwise they keep a stale
            # bundle_config_hash (still counting the now-departed model) until a
            # later full refresh, and workers on them NAK on the mismatch (#1771).
            old_affected_bundles = _matching_bundle_names(
                self._model_adapter_modules.get(sie_id, set()),
                self._bundles,
            )
            new_adapter_modules_set = _adapter_modules_for_profiles(new_full_config["profiles"])
            route_adapter_modules = _base_route_adapter_modules(new_full_config["profiles"])
            affected_bundles = old_affected_bundles + _matching_bundle_names(new_adapter_modules_set, self._bundles)
            new_model_info = ModelInfo(
                name=sie_id,
                bundles=_matching_bundle_names(route_adapter_modules, self._bundles),
            )

            # Rebuild this model's profile-variant entries from scratch: drop
            # every variant that belonged to sie_id (stale profiles must not
            # survive a replace) and re-derive from the incoming profiles.
            new_profile_variant_models = {
                name: info
                for name, info in self._profile_variant_models.items()
                if self._profile_variant_base_models.get(name) != sie_id
            }
            new_profile_variant_names_lower = {
                lower: name
                for lower, name in self._profile_variant_names_lower.items()
                if self._profile_variant_base_models.get(name) != sie_id
            }
            new_profile_variant_base_models = {
                name: base for name, base in self._profile_variant_base_models.items() if base != sie_id
            }
            for profile_name in sorted(new_profile_names):
                if profile_name == "default":
                    continue
                adapter_path = _effective_adapter_path(new_full_config["profiles"], profile_name)
                module_path = _adapter_module_from_path(adapter_path)
                variant_modules = {module_path} if module_path else set()
                variant_name = f"{sie_id}:{profile_name}"
                new_profile_variant_models[variant_name] = ModelInfo(
                    name=variant_name,
                    bundles=_matching_bundle_names(variant_modules, self._bundles),
                )
                new_profile_variant_names_lower[variant_name.lower()] = variant_name
                new_profile_variant_base_models[variant_name] = sie_id
                affected_bundles.extend(new_profile_variant_models[variant_name].bundles)
            affected_bundles.sort()
            affected_bundles = list(dict.fromkeys(affected_bundles))

            self._models = {**self._models, sie_id: new_model_info}
            if not existing:
                self._model_names_lower = {**self._model_names_lower, sie_id.lower(): sie_id}
            self._profile_variant_models = new_profile_variant_models
            self._profile_variant_names_lower = new_profile_variant_names_lower
            self._profile_variant_base_models = new_profile_variant_base_models
            self._model_adapter_modules = {**self._model_adapter_modules, sie_id: new_adapter_modules_set}
            self._model_profiles = {**self._model_profiles, sie_id: new_profile_names}
            self._model_profile_configs = {**self._model_profile_configs, sie_id: new_profile_configs}
            self._model_full_configs = {**self._model_full_configs, sie_id: new_full_config}

            all_bundle_adapters: set[str] = set()
            for bundle in self._bundles.values():
                all_bundle_adapters.update(bundle.adapters)
            missing_for_sie_id = new_adapter_modules_set - all_bundle_adapters
            new_unrouteable = {m: set(mods) for m, mods in self._unrouteable_models.items()}
            if missing_for_sie_id:
                new_unrouteable[sie_id] = missing_for_sie_id
            else:
                new_unrouteable.pop(sie_id, None)
            self._unrouteable_models = new_unrouteable

            self._bundle_hash_cache = {}

            logger.info(
                "Replaced model config: %s (profiles=%s, bundles=%s)",
                sie_id,
                sorted(new_profile_names),
                affected_bundles,
            )
            return affected_bundles

    def get_model_profile_names(self, model_name: str) -> set[str]:
        """Get known profile names for a model."""
        with self._lock:
            canonical_variant = self._profile_variant_names_lower.get(model_name.lower())
            if canonical_variant or model_name in self._profile_variant_models:
                return {"default"}
            return set(self._model_profiles.get(model_name, set()))

    def get_full_config(self, model_name: str) -> dict | None:
        """Return a deep-copied snapshot of the full merged model config.

        Preferred over reading raw YAML from the ConfigStore when the
        store is not enabled: the registry is always the authoritative
        in-memory state for both filesystem-seeded and API-added models.
        Returns ``None`` for unknown models.
        """
        with self._lock:
            full = self._model_full_configs.get(model_name)
            if full is None:
                return None
            # Full deep copy: the docstring promises a snapshot, and shallow-copying
            # only top-level keys leaves nested lists/dicts outside ``profiles``
            # aliased to registry-owned state so a mutating caller could corrupt it.
            return copy.deepcopy(full)

    def compute_bundle_config_hash(self, bundle_id: str) -> str:
        """Compute the config hash for a specific bundle.

        The hash covers all model configs/profiles whose adapter_path is
        in the bundle's adapter list. Bundle metadata is excluded (immutable).
        Results are cached and invalidated when model configs change.

        Args:
            bundle_id: Bundle identifier.

        Returns:
            Hex-encoded SHA-256 hash, or empty string if no models.
        """
        with self._lock:
            cached = self._bundle_hash_cache.get(bundle_id)
            if cached is not None:
                return cached

            result = self._compute_bundle_config_hash_locked(bundle_id)
            self._bundle_hash_cache[bundle_id] = result
            return result

    def compute_bundle_config_hash_for_pool(self, bundle_id: str, pool_name: str) -> str:
        """Compute the worker-parity config hash for a bundle within one pool."""
        normalized_pool = (pool_name or "").strip().lower() or DEFAULT_MODEL_POOL
        with self._lock:
            return self._compute_bundle_config_hash_locked(bundle_id, normalized_pool)

    def compute_bundle_pool_config_hashes(self) -> dict[str, dict[str, str]]:
        """Return worker-parity config hashes grouped by bundle and model pool."""
        with self._lock:
            hashes: dict[str, dict[str, str]] = {bundle_id: {} for bundle_id in self._bundles}
            pools_by_bundle: dict[str, set[str]] = {bundle_id: set() for bundle_id in self._bundles}
            for model_name in sorted(self._models):
                pool = _model_pool_name(self._model_full_configs.get(model_name, {}))
                for bundle_id in self.get_model_export_bundles(model_name):
                    if bundle_id in self._bundles:
                        pools_by_bundle.setdefault(bundle_id, set()).add(pool)

            for bundle_id, pools in pools_by_bundle.items():
                for pool in sorted(pools):
                    hash_value = self._compute_bundle_config_hash_locked(bundle_id, pool)
                    if hash_value:
                        hashes[bundle_id][pool] = hash_value
            return hashes

    def _compute_bundle_config_hash_locked(self, bundle_id: str, pool_name: str | None = None) -> str:
        bundle = self._bundles.get(bundle_id)
        if not bundle:
            return ""

        normalized_pool = (pool_name or "").strip().lower() or None
        bundle_adapter_set = set(bundle.adapters)
        items: list[dict] = []

        for model_name in sorted(self._models):
            full_config = self._model_full_configs.get(model_name, {})
            if normalized_pool is not None and _model_pool_name(full_config) != normalized_pool:
                continue

            adapter_modules = self._model_adapter_modules.get(model_name, set())
            # Only include if model has adapters in this bundle
            if adapter_modules & bundle_adapter_set:
                full_profiles = full_config.get("profiles", {}) if isinstance(full_config, dict) else {}
                if not isinstance(full_profiles, dict):
                    full_profiles = {}
                profiles_for_hash = []
                # Hash only the fields that affect inference behaviour,
                # matching the worker-side hash in
                # ``sie_server.api.ws._compute_bundle_config_hash``.
                for pname in sorted(self.get_model_profile_names(model_name)):
                    # Only include profiles whose adapter is routable in this bundle
                    filtered_cfg = _resolved_profile_hash_config(full_profiles, pname)
                    if filtered_cfg is None:
                        continue
                    p_adapter = filtered_cfg.get("adapter_path", "")
                    if isinstance(p_adapter, str) and p_adapter:
                        p_module = p_adapter.split(":", maxsplit=1)[0]
                        if p_module not in bundle_adapter_set:
                            continue
                    profiles_for_hash.append({"name": pname, "config": filtered_cfg})
                if not profiles_for_hash:
                    continue
                items.append(
                    {
                        "sie_id": model_name,
                        "profiles": profiles_for_hash,
                    }
                )

        if not items:
            return ""

        serialized = orjson.dumps(items, option=orjson.OPT_SORT_KEYS)
        return hashlib.sha256(serialized).hexdigest()


def parse_model_spec(model_spec: str) -> tuple[str | None, str]:
    """Parse model spec into (bundle_override, model_name).

    Format: [bundle:/]org/model[:variant]

    The separator is ":/" to distinguish bundle prefix from variant suffix.

    Examples:
        "BAAI/bge-m3" -> (None, "BAAI/bge-m3")
        "default:/BAAI/bge-m3" -> ("default", "BAAI/bge-m3")
        "BAAI/bge-m3:variant" -> (None, "BAAI/bge-m3:variant")

    Args:
        model_spec: Model specification string.

    Returns:
        Tuple of (bundle_override, model_name).
    """
    if not model_spec or not model_spec.strip():
        msg = "model_spec must not be empty"
        raise ValueError(msg)

    if ":/" in model_spec:
        parts = model_spec.split(":/", 1)
        bundle = parts[0].lower()
        model = parts[1]
        if not bundle:
            msg = "Bundle part of model_spec must not be empty"
            raise ValueError(msg)
        if not model:
            msg = "Model part of model_spec must not be empty"
            raise ValueError(msg)
        return bundle, model
    return None, model_spec
