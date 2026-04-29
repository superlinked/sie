from __future__ import annotations

import copy
import hashlib
import json
import logging
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
        new_model_adapter_modules: dict[str, set[str]] = {}
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
                        if isinstance(config, dict):
                            new_model_full_configs[model_name] = dict(config)

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
        # during bootstrap, and PR #704's test_bundle_coverage regression
        # test catches this pre-merge in CI.
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
        # Snapshot references (atomic in CPython -- GIL protects pointer reads)
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
            }
            for b in sorted(bundles, key=lambda x: x.name)
        ]
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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

        profiles = config.get("profiles", {})
        if not profiles:
            msg = "Missing required field: profiles"
            raise ValueError(msg)
        if not isinstance(profiles, dict):
            msg = "Field 'profiles' must be a mapping of profile_name -> profile_config"
            raise ValueError(msg)

        new_adapter_modules: set[str] = set()
        for profile_name, profile in profiles.items():
            if not isinstance(profile, dict):
                msg = f"Profile '{profile_name}' must be a mapping"
                raise ValueError(msg)
            adapter_path = profile.get("adapter_path", "")
            if not adapter_path:
                if not profile.get("extends"):
                    msg = f"Profile '{profile_name}' missing adapter_path"
                    raise ValueError(msg)
                continue
            module_path = adapter_path.split(":", maxsplit=1)[0]
            new_adapter_modules.add(module_path)

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
            for profile_name, profile in profiles.items():
                if profile_name in self.get_model_profile_names(sie_id):
                    stored = self._model_profile_configs.get(sie_id, {}).get(profile_name)
                    incoming = _canonical_profile_dict(profile)
                    if stored is not None and stored != incoming:
                        raise ValueError(
                            f"Profile '{profile_name}' on model '{sie_id}' already exists "
                            f"with different config (append-only -- cannot modify)"
                        )
                    skipped_profiles.append(profile_name)
                else:
                    created_profiles.append(profile_name)
        else:
            created_profiles = list(profiles.keys())  # type: ignore

        # Compute the post-apply adapter set so bundle mappings reflect
        # the hypothetical new state. This is a pure computation on a
        # local copy — no self._* mutation.
        post_adapter_modules: set[str] = set(self._model_adapter_modules.get(sie_id, set()))
        for profile_name in created_profiles:
            adapter_path = profiles[profile_name].get("adapter_path", "")
            if adapter_path:
                post_adapter_modules.add(adapter_path.split(":", maxsplit=1)[0])
        if not existing:
            post_adapter_modules = new_adapter_modules

        matching_bundles: list[tuple[int, str]] = []
        for bundle in self._bundles.values():
            if post_adapter_modules & set(bundle.adapters):
                matching_bundles.append((bundle.priority, bundle.name))
        matching_bundles.sort(key=lambda x: x[0])
        affected_bundles = [b[1] for b in matching_bundles]

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
            new_adapter_modules_set: set[str]
            if existing:
                base_modules = self._model_adapter_modules.get(sie_id, set())
                new_adapter_modules_set = set(base_modules)
                for profile_name in created_profiles:
                    adapter_path = profiles[profile_name].get("adapter_path", "")
                    if adapter_path:
                        module_path = adapter_path.split(":", maxsplit=1)[0]
                        new_adapter_modules_set.add(module_path)
            else:
                new_adapter_modules_set = set()
                for profile in profiles.values():
                    adapter_path = profile.get("adapter_path", "")
                    if adapter_path:
                        new_adapter_modules_set.add(adapter_path.split(":", maxsplit=1)[0])

            matching_bundles: list[tuple[int, str]] = []
            for bundle in self._bundles.values():
                if new_adapter_modules_set & set(bundle.adapters):
                    matching_bundles.append((bundle.priority, bundle.name))
            matching_bundles.sort(key=lambda x: x[0])
            affected_bundles = [b[1] for b in matching_bundles]

            new_model_info = ModelInfo(name=sie_id, bundles=list(affected_bundles))

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

            # Atomic swaps (CPython guarantees single-pointer-store is
            # not interleaved with a concurrent pointer-load under the GIL).
            self._models = {**self._models, sie_id: new_model_info}
            if not existing:
                self._model_names_lower = {
                    **self._model_names_lower,
                    sie_id.lower(): sie_id,
                }
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

    def get_model_profile_names(self, model_name: str) -> set[str]:
        """Get known profile names for a model."""
        return set(self._model_profiles.get(model_name, set()))

    def get_full_config(self, model_name: str) -> dict | None:
        """Return a deep-copied snapshot of the full merged model config.

        Preferred over reading raw YAML from the ConfigStore when the
        store is not enabled: the registry is always the authoritative
        in-memory state for both filesystem-seeded and API-added models.
        Returns ``None`` for unknown models.
        """
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
