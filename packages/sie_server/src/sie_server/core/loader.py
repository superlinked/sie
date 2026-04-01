import importlib
import importlib.util
import inspect
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from sie_sdk.storage import (
    get_storage_backend,
    is_cloud_path,
    join_path,
)

from sie_server.adapters.base import ModelAdapter
from sie_server.config.engine import ComputePrecision
from sie_server.config.model import AdapterOptions, ModelConfig, ProfileConfig
from sie_server.core.inference import AttentionBackend

logger = logging.getLogger(__name__)

# Error messages
_ERR_CONFIG_NOT_FOUND = "Config file not found: {path}"

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


def load_model_configs(models_dir: Path | str) -> dict[str, ModelConfig]:
    """Load all model configs from a directory (local or cloud).

    Args:
        models_dir: Path to the models directory (local path, s3://, or gs://).

    Returns:
        Dictionary mapping model names to their ModelConfig objects.

    Raises:
        FileNotFoundError: If models_dir doesn't exist (local only).
        ValueError: If a config file is invalid.
    """
    models_dir_str = str(models_dir)

    if is_cloud_path(models_dir_str):
        return _load_configs_from_cloud(models_dir_str)
    return _load_configs_from_local(Path(models_dir_str))


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

    _expand_profile_variants(configs)
    return configs


def _expand_profile_variants(configs: dict[str, ModelConfig]) -> None:
    """Expand non-default profiles into variant model entries.

    For each model config, non-default profiles become separate model entries
    named '{sie_id}:{profile_name}'. The variant entry is a copy of the base
    config with the variant profile promoted to 'default'.
    """
    variants: dict[str, ModelConfig] = {}

    for base_name, config in configs.items():
        for profile_name, profile in config.profiles.items():
            if profile_name == "default":
                continue
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
                )
            else:
                variant_default = profile.model_copy(update={"extends": None})

            variant_id = f"{base_name}:{profile_name}"
            if variant_id in configs:
                logger.warning("Variant '%s' collides with existing config; skipping expansion", variant_id)
                continue
            variant_profiles = {"default": variant_default}
            variant_config = config.model_copy(
                update={
                    "sie_id": variant_id,
                    "profiles": variant_profiles,
                },
            )
            variants[variant_id] = variant_config
            logger.info("Expanded profile '%s' as variant: %s", profile_name, variant_id)

    configs.update(variants)


def _load_configs_from_cloud(models_dir: str) -> dict[str, ModelConfig]:
    """Load model configs from S3/GCS.

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

    _expand_profile_variants(configs)
    return configs


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

    # Instantiate with config values
    adapter_kwargs = _build_adapter_kwargs(config, default_compute_precision)

    # Apply engine-level attention backend only when adapter supports it
    if attention_backend != "auto" and "attn_implementation" not in adapter_kwargs:
        try:
            signature = inspect.signature(adapter_class.__init__)
            if "attn_implementation" in signature.parameters:
                adapter_kwargs["attn_implementation"] = attention_backend
        except (TypeError, ValueError):
            # If signature inspection fails, avoid passing unsupported kwargs
            pass

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
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        msg = f"Could not import adapter module '{module_path}': {e}"
        raise ImportError(msg) from e

    if not hasattr(module, class_name):
        msg = f"Adapter class '{class_name}' not found in module '{module_path}'"
        raise ImportError(msg)

    return getattr(module, class_name)


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
    # Determine model path: weights_path takes precedence over hf_id
    model_name_or_path: str | Path
    if config.weights_path is not None:
        model_name_or_path = config.weights_path
    elif config.hf_id is not None:
        model_name_or_path = config.hf_id
    else:
        msg = f"Model '{config.name}' has no weights_path or hf_id"
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

    # Pass HF revision if pinned in config
    if config.hf_revision is not None:
        kwargs["revision"] = config.hf_revision

    kwargs.update(resolved.loadtime)

    return kwargs
