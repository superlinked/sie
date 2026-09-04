import importlib
import importlib.util
import inspect
import logging
import os
import sys
import tempfile
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from sie_sdk.storage import (
    get_storage_backend,
    is_cloud_path,
    join_path,
)

from sie_server.adapters._generation_base import GenerationAdapter
from sie_server.adapters.base import ModelAdapter
from sie_server.config.engine import ComputePrecision
from sie_server.config.model import (
    AdapterOptions,
    ModelConfig,
    ProfileConfig,
    ResolvedProfile,
    is_immutable_revision,
    lora_entry_ref,
)
from sie_server.config.package_artifacts import (
    PACKAGE_ARTIFACT_MANIFEST_PATH_KEY,
    PACKAGE_ARTIFACT_MANIFEST_SHA256_KEY,
    PACKAGE_ARTIFACT_MODE_KEY,
    PackageArtifactMode,
    verify_staged_package_artifacts,
)
from sie_server.config.serving_artifacts import (
    SERVING_ARTIFACT_COMPUTE_TYPE_KEY,
    SERVING_ARTIFACT_KEY,
    SERVING_ARTIFACT_PATH_KEY,
    VerifiedServingArtifact,
)
from sie_server.core.inference import AttentionBackend

logger = logging.getLogger(__name__)

# Error messages
_ERR_CONFIG_NOT_FOUND = "Config file not found: {path}"

# The immutable-revision rule (40-hex commit SHA) moved to
# ``sie_server.config.model`` so the config schema can apply it to pinned LoRA
# refs (#2113) without a config -> core import cycle; ``is_immutable_revision``
# is re-exported here for existing adapter/tooling imports.


# Local cache for downloaded configs (used when models_dir is cloud)
_CONFIG_CACHE_DIR: Path | None = None


def _get_config_cache_dir() -> Path:
    """Get the local directory for caching downloaded configs."""
    global _CONFIG_CACHE_DIR
    if _CONFIG_CACHE_DIR is None:
        # Use SIE_LOCAL_CACHE if set, otherwise use temp directory
        local_cache = os.environ.get("SIE_LOCAL_CACHE")
        if local_cache:
            _CONFIG_CACHE_DIR = Path(local_cache) / "sie_configs"
        else:
            _CONFIG_CACHE_DIR = Path(tempfile.gettempdir()) / "sie_configs"
        _CONFIG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CONFIG_CACHE_DIR


def validate_pinned_revision(config: ModelConfig) -> None:
    """Enforce the served-model version-identity contract.

    A promoted/served, HF-backed model MUST pin ``hf_revision`` to an immutable
    commit SHA so its ``sie_id`` maps to identical weights forever. ``weights_path``
    models have no Hub revision to pin. Promoted package-backed models must name
    a digest-verified staged manifest; bundled packages and live external
    downloads do not by themselves establish immutable model identity. The
    package lock is not treated as a model revision. Raises ``ValueError`` naming
    the fix.
    """
    if config.package_backed:
        declaration = config.package_artifact_declaration
        if declaration.mode != PackageArtifactMode.STAGED:
            msg = (
                f"Served package-backed model '{config.sie_id}' does not declare a staged artifact manifest. "
                "Promoted serving requires an exact manifest and per-file digests; a package lock is not a "
                "model-weights revision."
            )
            raise ValueError(msg)
        return
    if config.hf_id is None:
        # weights_path: nothing on the Hub to pin.
        return
    revision = config.hf_revision
    if revision is None:
        msg = (
            f"Served model '{config.sie_id}' (hf_id '{config.hf_id}') has no 'hf_revision'. "
            "A promoted/served model must pin an immutable HF commit SHA so its id maps to "
            "identical weights forever. Resolve the current SHA and pin it."
        )
        raise ValueError(msg)
    if not is_immutable_revision(revision):
        msg = (
            f"Served model '{config.sie_id}' pins hf_revision={revision!r}, which is not an "
            "immutable 40-char commit SHA — a branch/tag name (e.g. 'main') drifts on the Hub. "
            "Pin the resolved commit SHA instead."
        )
        raise ValueError(msg)


def require_pinned_revisions(configs: Mapping[str, ModelConfig]) -> None:
    """Apply :func:`validate_pinned_revision` across a catalog; raise on the first violation."""
    for config in configs.values():
        validate_pinned_revision(config)


def load_model_configs(models_dir: Path | str, *, require_pinned_revision: bool = False) -> dict[str, ModelConfig]:
    """Load all model configs from a directory (local or cloud).

    Args:
        models_dir: Path to the models directory (local path, s3://, gs://, abfs(s)://, or oss://).
        require_pinned_revision: When True, reject any HF-backed config that does not pin an
            immutable ``hf_revision``. Off by default so the full dev catalog
            (which carries an unpinned long tail) still loads; the managed serving/staging
            path opts in.

    Returns:
        Dictionary mapping model names to their ModelConfig objects.

    Raises:
        FileNotFoundError: If models_dir doesn't exist (local only).
        ValueError: If a config file is invalid, or (when ``require_pinned_revision``)
            a served model is missing an immutable revision pin.
    """
    models_dir_str = str(models_dir)

    if is_cloud_path(models_dir_str):
        configs = _load_configs_from_cloud(models_dir_str)
    else:
        configs = _load_configs_from_local(Path(models_dir_str))
    if require_pinned_revision:
        require_pinned_revisions(configs)
    return configs


def _load_configs_from_local(models_dir: Path) -> dict[str, ModelConfig]:
    """Load model configs from a local directory.

    Scans for flat YAML files (e.g., baai-bge-m3.yaml) directly in models_dir.
    After loading, expands non-default standalone profiles into variant model entries
    (e.g., profile "bge_m3_flag" on "BAAI/bge-m3" becomes "BAAI/bge-m3:bge_m3_flag").
    """
    if not models_dir.exists():
        msg = f"Models directory not found: {models_dir}"
        raise FileNotFoundError(msg)

    configs: dict[str, ModelConfig] = {}

    for config_path in models_dir.glob("*.yaml"):
        if not config_path.is_file():
            continue

        try:
            config = load_model_config(config_path)
            configs[config.name] = config
            logger.info("Loaded config for model: %s", config.name)
        except Exception:
            logger.exception("Failed to load config from %s", config_path)
            raise

    return expand_profile_variants(configs)


def expand_profile_variants(configs: Iterable[ModelConfig] | Mapping[str, ModelConfig]) -> dict[str, ModelConfig]:
    """Return configs plus generated entries for non-default profiles."""
    if isinstance(configs, Mapping):
        expanded = dict(configs)
    else:
        expanded = {}
        for config in configs:
            if config.sie_id in expanded:
                msg = f"duplicate model config: {config.sie_id}"
                raise ValueError(msg)
            expanded[config.sie_id] = config
    _expand_profile_variants(expanded)
    return expanded


def _expand_profile_variants(configs: dict[str, ModelConfig]) -> None:
    """Expand non-default profiles into variant model entries.

    For each model config, non-default profiles become separate model entries
    named '{sie_id}:{profile_name}'. The variant entry is a copy of the base
    config with the variant profile promoted to 'default'. If a config omits
    a default profile, the bare base model id is removed after variant expansion
    so only explicit profile identities are routable.
    """
    variants: dict[str, ModelConfig] = {}
    remove_base_names: list[str] = []

    for base_name, config in configs.items():
        default_profile = config.profiles.get("default")
        if (
            default_profile is not None
            and config.tasks.generate is not None
            and (default_profile.chat_template_kwargs is not None or default_profile.max_output_tokens is not None)
        ):
            generate_updates: dict[str, Any] = {}
            if default_profile.chat_template_kwargs is not None:
                generate_updates["chat_template_kwargs"] = {
                    **config.tasks.generate.chat_template_kwargs,
                    **default_profile.chat_template_kwargs,
                }
            if default_profile.max_output_tokens is not None:
                generate_updates["max_output_tokens"] = default_profile.max_output_tokens
            bare_config = config.model_copy(
                update={
                    "tasks": config.tasks.model_copy(
                        update={
                            "generate": config.tasks.generate.model_copy(update=generate_updates),
                        },
                    ),
                },
            )
            bare_config._resolved_cache = {}
            bare_config._resolved_lock = threading.Lock()
            configs[base_name] = bare_config
        if "default" not in config.profiles:
            remove_base_names.append(base_name)
        for profile_name, profile in config.profiles.items():
            if profile_name == "default":
                continue
            resolved: ResolvedProfile | None = None
            # Resolve extending profiles, use standalone profiles as-is
            if profile.extends is not None:
                resolved = config.resolve_profile(profile_name)
                # Convert ResolvedProfile back to ProfileConfig
                variant_default = ProfileConfig(
                    extends=None,
                    max_batch_tokens=resolved.max_batch_tokens,
                    compute_precision=resolved.compute_precision,
                    adapter_path=resolved.adapter_path,
                    adapter_options=AdapterOptions(
                        loadtime=dict(resolved.loadtime),
                        runtime=dict(resolved.runtime),
                    ),
                    adaptive_batching=resolved.adaptive_batching,
                    kv_budget_tokens=resolved.kv_budget_tokens,
                    admission_enabled=resolved.admission_enabled,
                    max_output_tokens=resolved.max_output_tokens,
                    chat_template_kwargs=(
                        dict(resolved.chat_template_kwargs) if resolved.chat_template_kwargs is not None else None
                    ),
                    grammar_profile=resolved.grammar_profile,
                )
            else:
                variant_default = profile.model_copy(update={"extends": None})

            variant_id = f"{base_name}:{profile_name}"
            if variant_id in configs:
                logger.warning("Variant '%s' collides with existing config; skipping expansion", variant_id)
                continue
            variant_profiles = {"default": variant_default}
            variant_updates: dict[str, Any] = {
                "sie_id": variant_id,
                "profiles": variant_profiles,
            }
            loadtime = resolved.loadtime if resolved is not None else profile.adapter_options.loadtime
            profile_chat_template_kwargs = (
                resolved.chat_template_kwargs if resolved is not None else profile.chat_template_kwargs
            )
            generate_updates: dict[str, Any] = {}
            profile_max_seq = loadtime.get("max_seq_length")
            if isinstance(profile_max_seq, int) and profile_max_seq > 0:
                variant_updates["max_sequence_length"] = profile_max_seq
                if config.tasks.generate is not None:
                    generate_updates["context_length"] = profile_max_seq
            if profile_chat_template_kwargs is not None and config.tasks.generate is not None:
                generate_updates["chat_template_kwargs"] = {
                    **config.tasks.generate.chat_template_kwargs,
                    **dict(profile_chat_template_kwargs),
                }
            profile_max_output_tokens = (
                resolved.max_output_tokens if resolved is not None else profile.max_output_tokens
            )
            if profile_max_output_tokens is not None and config.tasks.generate is not None:
                generate_updates["max_output_tokens"] = profile_max_output_tokens
            if generate_updates and config.tasks.generate is not None:
                variant_updates["tasks"] = config.tasks.model_copy(
                    update={
                        "generate": config.tasks.generate.model_copy(update=generate_updates),
                    },
                )
            variant_config = config.model_copy(
                update=variant_updates,
            )
            # ``model_copy`` carries PrivateAttr values, including the resolved
            # profile cache. Variants must not share cached "default" profiles
            # with the base config or sibling variants.
            variant_config._resolved_cache = {}
            variant_config._resolved_lock = threading.Lock()
            variant_config._synthetic_profile_variant_source = (base_name, profile_name)
            variants[variant_id] = variant_config
            logger.info("Expanded profile '%s' as variant: %s", profile_name, variant_id)

    for base_name in remove_base_names:
        configs.pop(base_name, None)

    configs.update(variants)


def _load_configs_from_cloud(models_dir: str) -> dict[str, ModelConfig]:
    """Load model configs from cloud object storage.

    Discovers YAML files via LIST operation, downloads them to local cache, and parses them.
    Model configs are flat YAML files (e.g., gs://bucket/models/BAAI__bge-m3.yaml).
    """
    backend = get_storage_backend(models_dir)
    cache_dir = _get_config_cache_dir()
    configs: dict[str, ModelConfig] = {}

    logger.info("Loading model configs from %s", models_dir)

    # List YAML files in models directory
    for filename in backend.list_files(models_dir):
        if not filename.endswith(".yaml"):
            continue

        config_url = join_path(models_dir, filename)

        try:
            # Download config to local cache
            local_config_path = cache_dir / filename

            backend.download_file(config_url, local_config_path)

            # Parse config
            config = load_model_config(local_config_path)
            configs[config.name] = config
            logger.info("Loaded config for model: %s (from %s)", config.name, models_dir)
        except Exception:
            logger.exception("Failed to load config from %s", config_url)
            raise

    return expand_profile_variants(configs)


def load_model_config(config_path: Path) -> ModelConfig:
    """Load a single model config from a YAML file.

    Args:
        config_path: Path to the model config YAML file.

    Returns:
        ModelConfig instance.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is invalid.
    """
    if not config_path.exists():
        msg = _ERR_CONFIG_NOT_FOUND.format(path=config_path)
        raise FileNotFoundError(msg)

    with config_path.open() as f:
        raw_config = yaml.safe_load(f)

    return ModelConfig(**raw_config)


def resolve_adapter_path(
    config: ModelConfig,
    model_dir: Path,
) -> str:
    """Resolve the adapter path for a model config.

    Gets the adapter_path from the resolved default profile.

    Args:
        config: The model config to resolve adapter for.
        model_dir: Directory containing the model's config YAML file.

    Returns:
        Fully resolved adapter path (e.g., "sie_server.adapters.bge_m3:BGEM3Adapter").

    Raises:
        ValueError: If adapter cannot be resolved.
    """
    resolved = config.resolve_profile("default")
    adapter_path = resolved.adapter_path

    # If it's a built-in adapter (starts with sie_server.), return as-is
    if adapter_path.startswith("sie_server."):
        return adapter_path

    # Otherwise, it's a custom adapter file in the model directory
    # Format: "adapter.py:ClassName" -> "/full/path/to/adapter.py:ClassName"
    if ":" not in adapter_path:
        msg = f"Invalid adapter path '{adapter_path}': expected 'file.py:ClassName'"
        raise ValueError(msg)

    file_part, class_part = adapter_path.split(":", 1)
    full_path = model_dir / file_part
    return f"{full_path}:{class_part}"


def load_adapter(
    config: ModelConfig,
    model_dir: Path,
    *,
    device: str,
    default_compute_precision: ComputePrecision = "float16",
    attention_backend: AttentionBackend = "auto",
    verified_serving_artifact: VerifiedServingArtifact | None = None,
) -> ModelAdapter:
    """Load and instantiate a model adapter.

    Args:
        config: The model config.
        model_dir: Directory containing the model's config YAML file.
        device: Device for adapter selection (e.g., "cuda:0", "mps", "cpu").
        default_compute_precision: Default precision if model config doesn't specify.
        attention_backend: Default attention backend (applied when adapter supports it).

    Returns:
        Instantiated ModelAdapter ready to be loaded onto a device.

    Raises:
        ValueError: If adapter cannot be resolved or instantiated, or if adapter
            doesn't support the config's output types.
        ImportError: If adapter module/class not found.
    """
    adapter_path = resolve_adapter_path(config, model_dir)

    # Parse module:class
    if ":" not in adapter_path:
        msg = f"Invalid adapter path '{adapter_path}': expected 'module:ClassName'"
        raise ValueError(msg)

    module_path, class_name = adapter_path.rsplit(":", 1)

    # Load the adapter class
    if module_path.startswith("sie_server."):
        # Built-in adapter: import normally
        adapter_class = _import_builtin_adapter(module_path, class_name)
    else:
        # Custom adapter: load from file path
        adapter_class = _import_custom_adapter(Path(module_path), class_name)

    # Instantiate with config values. The engine-level default compute precision
    # is fp32 off-CUDA (numerically safe on CPU/MPS for models we have not verified
    # at lower precision); a model profile's explicit compute_precision still wins
    # (e.g. curated Mac models opt into fp16 on MPS). On CUDA the engine default
    # (typically fp16) is used unchanged. See adapters' dtype resolution for the
    # matching honor-explicit-precision logic.
    effective_default_precision: ComputePrecision = (
        default_compute_precision if device.startswith("cuda") else "float32"
    )
    adapter_kwargs = _build_adapter_kwargs(
        config,
        effective_default_precision,
        verified_serving_artifact=verified_serving_artifact,
    )

    # MPS bfloat16 support is incomplete in torch and can HANG model load (verified:
    # bge-m3 bf16 load wedges at ~1% CPU on Apple Silicon). fp16 is the MPS-verified
    # path (parity 1.0 vs CPU-fp32 in benchmarks), so coerce any resolved bf16 -> fp16
    # on MPS for every adapter. CUDA/CPU are untouched.
    if device == "mps" and adapter_kwargs.get("compute_precision") == "bfloat16":
        logger.info(
            "Coercing compute_precision bfloat16 -> float16 for '%s' on MPS (MPS bf16 is unreliable)", config.name
        )
        adapter_kwargs["compute_precision"] = "float16"

    # Apply engine-level attention backend only when adapter supports it
    if attention_backend != "auto" and "attn_implementation" not in adapter_kwargs:
        try:
            signature = inspect.signature(adapter_class.__init__)
            if "attn_implementation" in signature.parameters:
                adapter_kwargs["attn_implementation"] = attention_backend
        except (TypeError, ValueError):
            # If signature inspection fails, avoid passing unsupported kwargs
            pass

    # Wire the profile's ``runtime.default_sampling`` into the adapter when its
    # constructor accepts it. These defaults live under ``adapter_options.runtime``
    # (not ``loadtime``) and were previously dropped entirely — so a profile's
    # ``default_sampling`` silently no-op'd (e.g. a ``min_new_tokens`` floor for
    # the Qwen3.6 stop-token-as-first-token bug never reached SGLang). The adapter
    # merges this dict with ``setdefault`` semantics, so request-supplied sampling
    # values still win; this only fills params the caller omitted, which is the
    # documented intent of a ``default_sampling`` block. Guarded by signature
    # inspection so adapters that don't accept the kwarg are unaffected.
    if "default_sampling" not in adapter_kwargs:
        try:
            signature = inspect.signature(adapter_class.__init__)
            default_sampling = dict(config.resolve_profile("default").runtime).get("default_sampling")
            if "default_sampling" in signature.parameters and default_sampling:
                adapter_kwargs["default_sampling"] = default_sampling
        except (TypeError, ValueError):
            logger.debug(
                "Could not inspect adapter constructor for default_sampling support; skipping default_sampling wiring.",
                exc_info=True,
            )

    # Instantiate adapter using factory method for device-aware selection
    # All adapters inherit create_for_device() from ModelAdapter base class
    adapter = adapter_class.create_for_device(device=device, **adapter_kwargs)

    # Validate adapter supports config's output types
    adapter_outputs = set(adapter.capabilities.outputs)
    config_outputs = set(config.outputs)
    unsupported = config_outputs - adapter_outputs
    if unsupported:
        msg = (
            f"Adapter '{adapter_class.__name__}' does not support output types {unsupported}. "
            f"Adapter supports: {adapter_outputs}, config requires: {config_outputs}"
        )
        raise ValueError(msg)

    # If the config declares ``tasks.generate``, the resolved adapter MUST
    # be a ``GenerationAdapter`` subclass. The outputs check above would
    # catch any adapter that doesn't declare "tokens" in capabilities,
    # but an embedding adapter that mistakenly declared "tokens" would
    # slip through — this is the structural check. Surfacing the error
    # at adapter-load (worker boot) rather than first-request time
    # means misconfiguration is caught before any traffic lands.
    if config.tasks.generate is not None and not isinstance(adapter, GenerationAdapter):
        msg = (
            f"Model '{config.sie_id}' declares 'tasks.generate' but adapter "
            f"'{adapter_class.__name__}' is not a GenerationAdapter subclass. "
            "Generation requests require an adapter that inherits from "
            "sie_server.adapters._generation_base.GenerationAdapter."
        )
        raise ValueError(msg)

    return adapter


def _import_builtin_adapter(module_path: str, class_name: str) -> type[ModelAdapter]:
    """Import a built-in adapter class.

    Args:
        module_path: Module path (e.g., "sie_server.adapters.bge_m3").
        class_name: Class name (e.g., "BGEM3Adapter").

    Returns:
        The adapter class.

    Raises:
        ImportError: If module or class not found.
    """
    module_path = _resolve_legacy_adapter_module(module_path, class_name)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        msg = f"Could not import adapter module '{module_path}': {e}"
        raise ImportError(msg) from e

    if not hasattr(module, class_name):
        msg = f"Adapter class '{class_name}' not found in module '{module_path}'"
        raise ImportError(msg)

    return getattr(module, class_name)


def _resolve_legacy_adapter_module(module_path: str, class_name: str) -> str:
    if module_path == "sie_server.adapters.sglang":
        if class_name == "SGLangEmbeddingAdapter":
            return "sie_server.adapters.sglang.embedding"
        if class_name == "SGLangGenerationAdapter":
            return "sie_server.adapters.sglang.generation"
    return module_path


def _import_custom_adapter(file_path: Path, class_name: str) -> type[ModelAdapter]:
    """Import a custom adapter class from a file.

    Args:
        file_path: Path to the Python file containing the adapter.
        class_name: Class name to import.

    Returns:
        The adapter class.

    Raises:
        ImportError: If file or class not found.
    """
    if not file_path.exists():
        msg = f"Custom adapter file not found: {file_path}"
        raise ImportError(msg)

    # Generate a unique module name to avoid conflicts
    module_name = f"sie_custom_adapters.{file_path.stem}_{id(file_path)}"

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        msg = f"Could not load spec for {file_path}"
        raise ImportError(msg)

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        del sys.modules[module_name]
        msg = f"Error loading custom adapter from {file_path}: {e}"
        raise ImportError(msg) from e

    if not hasattr(module, class_name):
        del sys.modules[module_name]
        msg = f"Adapter class '{class_name}' not found in {file_path}"
        raise ImportError(msg)

    return getattr(module, class_name)


def _build_adapter_kwargs(
    config: ModelConfig,
    default_compute_precision: ComputePrecision,
    *,
    verified_serving_artifact: VerifiedServingArtifact | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for adapter instantiation.

    Maps ModelConfig fields to adapter constructor arguments.
    Uses the resolved default profile for adapter-specific options.

    Args:
        config: The model configuration.
        default_compute_precision: Default precision from engine config.

    Returns:
        Dictionary of kwargs for the adapter constructor.
    """
    # Determine model path: weights_path takes precedence over hf_id.
    # package_backed adapters (e.g., Docling) carry their own weights via the
    # installed package and intentionally have neither hf_id nor weights_path.
    model_name_or_path: str | Path | None
    if config.package_backed:
        model_name_or_path = None
    elif config.weights_path is not None:
        model_name_or_path = config.weights_path
    elif config.hf_id is not None:
        model_name_or_path = config.hf_id
    else:
        msg = f"Model '{config.name}' has no weights_path, hf_id, or package_backed flag"
        raise ValueError(msg)

    # Resolve default profile for adapter options and compute precision
    resolved = config.resolve_profile("default")

    # Resolve compute precision: profile override -> engine default
    compute_precision = resolved.compute_precision or default_compute_precision

    kwargs: dict[str, Any] = {
        "model_name_or_path": model_name_or_path,
        "max_seq_length": config.max_sequence_length,
        "compute_precision": compute_precision,
    }
    if config.tasks.encode is not None and config.tasks.encode.dense is not None:
        kwargs["dense_dim"] = config.tasks.encode.dense.dim

    # Pass HF revision if pinned in config
    if config.hf_revision is not None:
        kwargs["revision"] = config.hf_revision

    kwargs.update(resolved.loadtime)

    serving_artifact = config.serving_artifact_declaration()
    kwargs.pop(SERVING_ARTIFACT_KEY, None)
    if serving_artifact is not None:
        if verified_serving_artifact is None:
            raise ValueError(
                f"Model '{config.sie_id}' declares a derived serving artifact but the loader did not verify it"
            )
        if (
            verified_serving_artifact.repo_id != serving_artifact.repo_id
            or verified_serving_artifact.revision != serving_artifact.revision
            or verified_serving_artifact.manifest_sha256 != serving_artifact.manifest_sha256
            or verified_serving_artifact.compute_type != serving_artifact.compute_type
        ):
            raise ValueError(f"Model '{config.sie_id}' received a stale or mismatched verified serving artifact")
        kwargs[SERVING_ARTIFACT_PATH_KEY] = verified_serving_artifact.root
        kwargs[SERVING_ARTIFACT_COMPUTE_TYPE_KEY] = verified_serving_artifact.compute_type
        kwargs.pop("model_name_or_path", None)
        kwargs.pop("revision", None)

    # Adapter constructors take ``lora_paths`` as served-name -> bare path/id
    # (the sglang ``--lora-paths`` shape). A pinned ``{id, revision}`` value
    # (#2113) is a *config* spelling: collapse it to the bare id here so no
    # constructor ever sees the mapping. The revision itself is honored on the
    # hot-reload preload path (``ModelLoader``), which rejects pins on adapters
    # that cannot apply them.
    lora_paths = kwargs.get("lora_paths")
    if isinstance(lora_paths, Mapping):
        kwargs["lora_paths"] = {
            served: lora_entry_ref(value)[0] if value else value for served, value in lora_paths.items()
        }

    if config.package_backed:
        declaration = config.package_artifact_declaration
        offline = any(
            os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
        if declaration.mode == PackageArtifactMode.LIVE and offline:
            raise ValueError(
                f"Package-backed model '{config.sie_id}' permits live external downloads but offline mode is enabled"
            )
        for key in (
            PACKAGE_ARTIFACT_MODE_KEY,
            PACKAGE_ARTIFACT_MANIFEST_PATH_KEY,
            PACKAGE_ARTIFACT_MANIFEST_SHA256_KEY,
        ):
            kwargs.pop(key, None)
        if declaration.mode == PackageArtifactMode.STAGED:
            verified = verify_staged_package_artifacts(declaration)
            kwargs["package_artifact_root"] = verified.root
            kwargs[PACKAGE_ARTIFACT_MANIFEST_SHA256_KEY] = verified.manifest_sha256

    return kwargs
