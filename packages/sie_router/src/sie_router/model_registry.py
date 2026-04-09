from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import orjson
import yaml

logger = logging.getLogger(__name__)

# Routable profile fields included in bundle_config_hash.
# Both router (model_registry) and server (ws.py) must use the same set.
_PROFILE_HASH_FIELDS = ("adapter_path", "max_batch_tokens", "compute_precision", "adapter_options")


def _canonical_profile_dict(profile: dict) -> dict:
    """Extract canonical routable fields from a raw profile dict for hashing."""
    return {k: profile.get(k) for k in _PROFILE_HASH_FIELDS}


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


@dataclass
class BundleInfo:
    """Information about a bundle."""

    name: str
    priority: int
    adapters: list[str] = field(default_factory=list)
    default: bool = False


@dataclass
class ModelInfo:
    """Information about a model and its compatible bundles."""

    name: str
    bundles: list[str] = field(default_factory=list)  # Ordered by priority (best first)


class ModelRegistry:
    """Source of truth for model→bundle mappings.

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

        # Protected by lock for thread-safe reload
        self._lock = threading.RLock()
        self._bundles: dict[str, BundleInfo] = {}
        self._models: dict[str, ModelInfo] = {}
        self._model_names_lower: dict[str, str] = {}  # lowercase → canonical
        self._model_adapter_modules: dict[str, set[str]] = {}  # model → adapter modules
        self._model_profiles: dict[str, set[str]] = {}  # model → profile names
        self._model_profile_configs: dict[str, dict[str, dict]] = {}  # model → {profile_name: config_dict}
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

    def reload(self) -> None:
        """Reload all configs from disk.

        Thread-safe: builds new state in temp structures, then swaps atomically.
        """
        new_bundles: dict[str, BundleInfo] = {}
        new_models: dict[str, ModelInfo] = {}
        new_model_names_lower: dict[str, str] = {}
        new_model_adapter_modules: dict[str, set[str]] = {}
        new_model_profiles: dict[str, set[str]] = {}
        new_model_profile_configs: dict[str, dict[str, dict]] = {}

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

                    new_bundles[name] = BundleInfo(
                        name=name,
                        priority=priority,
                        adapters=adapters,
                        default=default,
                    )
                    logger.debug("Loaded bundle '%s': priority=%d, adapters=%d", name, priority, len(adapters))

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

                    model_name = config.get("sie_id") or config.get("name")
                    if model_name:
                        adapter_modules: set[str] = set()
                        profiles = config.get("profiles", {})
                        for profile in profiles.values():
                            adapter_path = profile.get("adapter_path", "")
                            if adapter_path:
                                module_path = adapter_path.split(":", maxsplit=1)[0]
                                adapter_modules.add(module_path)

                        new_models[model_name] = ModelInfo(name=model_name)
                        new_model_adapter_modules[model_name] = adapter_modules
                        new_model_names_lower[model_name.lower()] = model_name
                        logger.debug("Discovered model: %s (adapters: %s)", model_name, adapter_modules)
                        new_model_profiles[model_name] = set(profiles.keys())
                        new_model_profile_configs[model_name] = {
                            pname: _canonical_profile_dict(pdata) for pname, pdata in profiles.items()
                        }

                except Exception:
                    logger.exception("Failed to load model config: %s", config_path)
        else:
            logger.warning("Models directory not found: %s", self._models_dir)

        # Compute mappings
        for model_name, model_info in new_models.items():
            adapter_modules = new_model_adapter_modules.get(model_name, set())
            if not adapter_modules:
                continue

            matching_bundles: list[tuple[int, str]] = []
            for bundle in new_bundles.values():
                if adapter_modules & set(bundle.adapters):
                    matching_bundles.append((bundle.priority, bundle.name))

            if matching_bundles:
                matching_bundles.sort(key=lambda x: x[0])
                model_info.bundles = [b[1] for b in matching_bundles]

        # Atomic swap under lock
        with self._lock:
            self._bundles = new_bundles
            self._models = new_models
            self._model_names_lower = new_model_names_lower
            self._model_adapter_modules = new_model_adapter_modules
            self._model_profiles = new_model_profiles
            self._model_profile_configs = new_model_profile_configs
            self._bundle_hash_cache.clear()

        logger.info(
            "ModelRegistry loaded: %d bundles, %d models",
            len(new_bundles),
            len(new_models),
        )

    def resolve_bundle(self, model: str, bundle_override: str | None = None) -> str:
        """Resolve which bundle to use for a model.

        Lock-free: reads dict references that are atomically swapped by
        ``reload()``.  The GIL guarantees a single pointer read is atomic
        in CPython, so concurrent hot-reload cannot produce a torn read.

        Args:
            model: Model name (e.g., "BAAI/bge-m3").
            bundle_override: Optional explicit bundle (e.g., "default").

        Returns:
            Bundle name to use.

        Raises:
            ModelNotFoundError: Model not in any bundle (404).
            BundleConflictError: Override bundle doesn't support model (409).
        """
        # Snapshot references (atomic in CPython — GIL protects pointer reads)
        models = self._models
        model_names_lower = self._model_names_lower

        # Normalize model name (case-insensitive lookup)
        canonical_model = model_names_lower.get(model.lower())
        if canonical_model is None:
            # Try exact match
            if model not in models:
                raise ModelNotFoundError(model)
            canonical_model = model

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

        Lock-free: reads dict references atomically swapped by ``reload()``.

        Args:
            model: Model name.

        Returns:
            ModelInfo if found, None otherwise.
        """
        models = self._models
        model_names_lower = self._model_names_lower
        # Try case-insensitive lookup first
        canonical = model_names_lower.get(model.lower())
        if canonical:
            return models.get(canonical)
        return models.get(model)

    def list_models(self) -> list[str]:
        """List all known model names.

        Returns:
            Sorted list of model names.
        """
        return sorted(self._models.keys())

    def list_bundles(self) -> list[str]:
        """List all bundle names.

        Returns:
            List of bundle names sorted by priority.
        """
        bundles = self._bundles
        bundles_sorted = sorted(bundles.values(), key=lambda b: b.priority)
        return [b.name for b in bundles_sorted]

    def get_bundle_info(self, bundle: str) -> BundleInfo | None:
        """Get bundle info.

        Args:
            bundle: Bundle name.

        Returns:
            BundleInfo if found, None otherwise.
        """
        return self._bundles.get(bundle)

    def get_models_for_bundle(self, bundle: str) -> list[str]:
        """Get all models that can be served by a bundle.

        Args:
            bundle: Bundle name.

        Returns:
            List of model names.
        """
        models = self._models
        return [model_name for model_name, model_info in models.items() if bundle in model_info.bundles]

    def model_exists(self, model: str) -> bool:
        """Check if a model exists in the registry.

        Args:
            model: Model name.

        Returns:
            True if model is known, False otherwise.
        """
        if model.lower() in self._model_names_lower:
            return True
        return model in self._models

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
            sie_id = config.get("sie_id")
            if not sie_id:
                msg = "Missing required field: sie_id"
                raise ValueError(msg)

            profiles = config.get("profiles", {})
            if not profiles:
                msg = "Missing required field: profiles"
                raise ValueError(msg)

            # Collect adapter modules from new profiles
            new_adapter_modules: set[str] = set()
            for profile_name, profile in profiles.items():
                adapter_path = profile.get("adapter_path", "")
                if not adapter_path:
                    # Check if it extends another profile (inherits adapter_path)
                    if not profile.get("extends"):
                        msg = f"Profile '{profile_name}' missing adapter_path"
                        raise ValueError(msg)
                    continue
                module_path = adapter_path.split(":", maxsplit=1)[0]
                new_adapter_modules.add(module_path)

            # Validate all adapter modules are routable (exist in at least one known bundle)
            all_bundle_adapters: set[str] = set()
            for bundle in self._bundles.values():
                all_bundle_adapters.update(bundle.adapters)

            unroutable = new_adapter_modules - all_bundle_adapters
            if unroutable:
                msg = f"Adapter(s) not in any known bundle: {', '.join(sorted(unroutable))}"
                raise ValueError(msg)

            # Check if model already exists
            existing = self._models.get(sie_id)
            created_profiles: list[str] = []
            skipped_profiles: list[str] = []

            if existing:
                # Append-only: add new profiles, skip identical, reject conflicts
                existing_adapter_modules = self._model_adapter_modules.get(sie_id, set())
                for profile_name, profile in profiles.items():
                    if profile_name in self.get_model_profile_names(sie_id):
                        # Check if existing profile has identical config
                        stored = self._model_profile_configs.get(sie_id, {}).get(profile_name)
                        incoming = _canonical_profile_dict(profile)
                        if stored is not None and stored != incoming:
                            raise ValueError(
                                f"Profile '{profile_name}' on model '{sie_id}' already exists "
                                f"with different config (append-only — cannot modify)"
                            )
                        skipped_profiles.append(profile_name)
                    else:
                        created_profiles.append(profile_name)
                        adapter_path = profile.get("adapter_path", "")
                        if adapter_path:
                            module_path = adapter_path.split(":", maxsplit=1)[0]
                            existing_adapter_modules.add(module_path)

                self._model_adapter_modules[sie_id] = existing_adapter_modules
            else:
                # New model
                self._models[sie_id] = ModelInfo(name=sie_id)
                self._model_names_lower[sie_id.lower()] = sie_id
                self._model_adapter_modules[sie_id] = new_adapter_modules
                created_profiles = list(profiles.keys())

            # Store profile names and configs for future conflict detection + hashing
            if sie_id not in self._model_profiles:
                self._model_profiles[sie_id] = set()
            self._model_profiles[sie_id].update(created_profiles)
            if sie_id not in self._model_profile_configs:
                self._model_profile_configs[sie_id] = {}
            for pname in created_profiles:
                self._model_profile_configs[sie_id][pname] = _canonical_profile_dict(profiles[pname])

            # Recompute bundle mappings for this model
            adapter_modules = self._model_adapter_modules.get(sie_id, set())
            matching_bundles: list[tuple[int, str]] = []
            for bundle in self._bundles.values():
                if adapter_modules & set(bundle.adapters):
                    matching_bundles.append((bundle.priority, bundle.name))

            matching_bundles.sort(key=lambda x: x[0])
            self._models[sie_id].bundles = [b[1] for b in matching_bundles]

            affected_bundles = [b[1] for b in matching_bundles]

            self._bundle_hash_cache.clear()

            logger.info(
                "Added model config: %s (created=%s, skipped=%s, bundles=%s)",
                sie_id,
                created_profiles,
                skipped_profiles,
                affected_bundles,
            )

            return created_profiles, skipped_profiles, affected_bundles

    def get_model_profile_names(self, model_name: str) -> set[str]:
        """Get known profile names for a model."""
        return set(self._model_profiles.get(model_name, set()))

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

            bundle = self._bundles.get(bundle_id)
            if not bundle:
                return ""

            bundle_adapter_set = set(bundle.adapters)
            items: list[dict] = []

            for model_name, model_info in sorted(self._models.items()):
                if bundle_id not in model_info.bundles:
                    continue
                adapter_modules = self._model_adapter_modules.get(model_name, set())
                # Only include if model has adapters in this bundle
                if adapter_modules & bundle_adapter_set:
                    profile_configs = self._model_profile_configs.get(model_name, {})
                    profiles_for_hash = []
                    # Hash only the fields that affect inference behaviour,
                    # matching the worker-side hash in
                    # ``sie_server.api.ws._compute_bundle_config_hash``.
                    _hash_fields = ("adapter_path", "max_batch_tokens", "compute_precision", "adapter_options")
                    for pname in sorted(self.get_model_profile_names(model_name)):
                        # Only include profiles whose adapter is routable in this bundle
                        p_cfg = profile_configs.get(pname, {})
                        p_adapter = p_cfg.get("adapter_path", "")
                        if p_adapter:
                            p_module = p_adapter.split(":", maxsplit=1)[0]
                            if p_module not in bundle_adapter_set:
                                continue
                        # Normalize: only include hash-relevant fields.
                        # adapter_options: treat empty/default as None (matches worker).
                        adapter_opts = p_cfg.get("adapter_options")
                        if isinstance(adapter_opts, dict) and not any(adapter_opts.values()):
                            adapter_opts = None
                        filtered_cfg = {
                            k: (adapter_opts if k == "adapter_options" else p_cfg.get(k)) for k in _hash_fields
                        }
                        profiles_for_hash.append({"name": pname, "config": filtered_cfg})
                    items.append(
                        {
                            "sie_id": model_name,
                            "profiles": profiles_for_hash,
                        }
                    )

            if not items:
                return ""

            serialized = orjson.dumps(items, option=orjson.OPT_SORT_KEYS)
            result = hashlib.sha256(serialized).hexdigest()
            self._bundle_hash_cache[bundle_id] = result
            return result


def parse_model_spec(model_spec: str) -> tuple[str | None, str]:
    """Parse model spec into (bundle_override, model_name).

    Format: [bundle:/]org/model[:variant]

    The separator is ":/" to distinguish bundle prefix from variant suffix.

    Examples:
        "BAAI/bge-m3" → (None, "BAAI/bge-m3")
        "default:/BAAI/bge-m3" → ("default", "BAAI/bge-m3")
        "BAAI/bge-m3:variant" → (None, "BAAI/bge-m3:variant")

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
