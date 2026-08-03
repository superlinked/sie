from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from sie_sdk.bundle_utils import match_bundle_models

from sie_server.bundle_requirements import resolve_bundle_requirements

logger = logging.getLogger(__name__)


def model_name_to_folder(model_name: str) -> str:
    """Convert a model name (org/model format) to folder name (org__model format).

    The folder naming convention uses '__' as separator because '/' is not allowed
    in directory names. Variant suffixes using ':' are also converted to '__'.

    Args:
        model_name: Model name in org/model or org/model:variant format.

    Returns:
        Folder name in org__model or org__model__variant format.

    Examples:
        >>> model_name_to_folder("BAAI/bge-m3")
        'BAAI__bge-m3'
        >>> model_name_to_folder("BAAI/bge-m3:FlagEmbedding")
        'BAAI__bge-m3__FlagEmbedding'
        >>> model_name_to_folder("simple-model")
        'simple-model'
    """
    return model_name.replace("/", "__").replace(":", "__")


def discover_model_configs(models_dir: Path) -> dict[str, Path]:
    """Discover all model configs by scanning YAML files in models directory.

    Models are stored as flat YAML files (e.g., baai-bge-m3.yaml) directly
    in the models directory. This function reads each YAML file and maps
    model names to their config file paths.

    Args:
        models_dir: Path to the models directory.

    Returns:
        Dict mapping model name (from config) to config file path.
    """
    model_configs: dict[str, Path] = {}

    if not models_dir.exists():
        return model_configs

    for config_path in models_dir.glob("*.yaml"):
        try:
            with config_path.open() as f:
                config = yaml.safe_load(f)
            model_name = config.get("sie_id")
            if model_name:
                model_configs[model_name] = config_path
        except (OSError, yaml.YAMLError, AttributeError):
            logger.debug("Failed to read config %s", config_path)
            continue

    return model_configs


# =============================================================================
# Bundle Dependency Resolution
# =============================================================================


@dataclass
class BundleDepResult:
    """Result of resolving bundle dependencies."""

    requirements: list[str]  # PEP 508 requirement strings
    conflicts: list[str]  # Human-readable conflict messages
    models: list[str]  # Model names in the bundle

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "requirements": self.requirements,
            "conflicts": self.conflicts,
            "models": self.models,
        }


def load_bundle(bundle_path: Path) -> dict:
    """Load a bundle YAML file.

    Args:
        bundle_path: Path to the bundle YAML file.

    Returns:
        Parsed bundle config dict with 'name' and 'adapters' keys.
    """
    with bundle_path.open() as f:
        data = yaml.safe_load(f) or {}

    return data


def collect_bundle_deps(
    bundle_name: str,
    bundles_dir: Path,
    models_dir: Path,
    *,
    exclude_cuda: bool = False,
    pool_name: str | None = None,
) -> BundleDepResult:
    """Collect all dependencies for a bundle from its deps section.

    Dependencies are declared directly in the bundle YAML under a `deps:` key.

    Args:
        bundle_name: Name of the bundle (without .yaml extension).
        bundles_dir: Path to the bundles directory.
        models_dir: Path to the models directory.
        exclude_cuda: If True, exclude CUDA-only packages (flash-attn, xformers).
            Use this when building CPU-only images.

    Returns:
        BundleDepResult with requirements, conflicts, and model names.
    """
    bundle_path = bundles_dir / f"{bundle_name}.yaml"
    if not bundle_path.exists():
        return BundleDepResult(
            requirements=[],
            conflicts=[f"Bundle '{bundle_name}' not found at {bundle_path}"],
            models=[],
        )

    bundle = load_bundle(bundle_path)
    bundle_deps = bundle.get("deps", {})

    requirements = resolve_bundle_requirements(bundle_deps, exclude_cuda=exclude_cuda)

    # Discover model names that match this bundle's adapters (for the result)
    model_names = match_bundle_models(bundle_path, models_dir, pool_name=pool_name)

    return BundleDepResult(
        requirements=requirements,
        conflicts=[],
        models=model_names,
    )
