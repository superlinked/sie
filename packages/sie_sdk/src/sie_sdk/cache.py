"""Model weight caching with hierarchical lookup.

Implements the caching hierarchy for model weights:
1. Local cache (HF_HOME/hub by default)
2. Cluster cache (S3/GCS bucket)
3. HuggingFace Hub fallback (if enabled)

The caching is transparent to adapters - they always see files in local cache.
This module pre-populates local cache from cluster cache if needed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sie_sdk.exceptions import GatedModelError
from sie_sdk.storage import get_storage_backend, join_path

if TYPE_CHECKING:
    from sie_sdk.storage import StorageBackend

logger = logging.getLogger(__name__)

HF_IGNORE_PATTERNS = [
    "*.md",
    "README*",
    "LICENSE*",
    "docs/*",
    ".github/*",
    "*.onnx",
    "onnx/*",
    # training-only
    "optimizer.*",
    "scheduler.*",
    "trainer_state.json",
    "training_args.bin",
    "checkpoint-*/*",
]
HF_LEGACY_WEIGHT_PATTERNS = ["*.bin", "*.ckpt", "*.msgpack", "*.h5", "*.ot"]


@dataclass
class CacheConfig:
    """Configuration for the weight cache."""

    local_cache: Path
    """Local cache directory (usually HF_HOME/hub)."""

    cluster_cache: str | None = None
    """Cluster cache URL (s3:// or gs://), or None if not configured."""

    hf_fallback: bool = True
    """Whether to fallback to HuggingFace Hub for downloads."""


def get_cache_config() -> CacheConfig:
    """Get cache configuration from environment variables.

    Reads:
        SIE_LOCAL_CACHE: Local cache directory (default: HF_HOME/hub)
        SIE_CLUSTER_CACHE: Cluster cache URL (s3:// or gs://)
        SIE_HF_FALLBACK: Whether to enable HF Hub fallback (default: true)

    Returns:
        CacheConfig with resolved paths.
    """
    # Local cache: explicit SIE_LOCAL_CACHE, or HF_HOME, or default
    local_cache_env = os.environ.get("SIE_LOCAL_CACHE")
    if local_cache_env:
        local_cache = Path(local_cache_env)
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            local_cache = Path(hf_home) / "hub"
        else:
            local_cache = Path.home() / ".cache" / "huggingface" / "hub"

    # Cluster cache: S3/GCS URL
    cluster_cache = os.environ.get("SIE_CLUSTER_CACHE")

    # HF fallback: default true
    hf_fallback_str = os.environ.get("SIE_HF_FALLBACK", "true").lower()
    hf_fallback = hf_fallback_str in ("true", "1", "yes")

    return CacheConfig(
        local_cache=local_cache,
        cluster_cache=cluster_cache,
        hf_fallback=hf_fallback,
    )


def is_model_cached(model_id: str, config: CacheConfig | None = None) -> bool:
    """Check if a model is already in local cache.

    Uses HuggingFace Hub's cache structure to check for the model.

    Args:
        model_id: HuggingFace model ID (e.g., "BAAI/bge-m3").
        config: Cache configuration. If None, reads from environment.

    Returns:
        True if model appears to be cached locally.
    """
    if config is None:
        config = get_cache_config()

    # HF Hub cache structure: models--{org}--{model}/snapshots/{revision}/
    cache_dir = config.local_cache / f"models--{model_id.replace('/', '--')}"
    if not cache_dir.exists():
        return False

    # Check for any snapshot directory with files
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return False

    return any(snapshot.is_dir() and any(snapshot.iterdir()) for snapshot in snapshots_dir.iterdir())


def ensure_model_cached(
    model_id: str,
    config: CacheConfig | None = None,
    revision: str | None = None,
) -> Path:
    """Ensure model weights are in local cache.

    Implements the caching hierarchy:
    1. Check local cache - return path if found
    2. Check cluster cache - download to local if found
    3. Download from HuggingFace Hub (if fallback enabled)

    Args:
        model_id: HuggingFace model ID (e.g., "BAAI/bge-m3").
        config: Cache configuration. If None, reads from environment.
        revision: Specific model revision (commit hash, branch, or tag).

    Returns:
        Path to cached model in local cache.

    Raises:
        GatedModelError: If model is gated and authentication fails.
        RuntimeError: If model not found in any cache tier and HF fallback disabled.
    """
    if config is None:
        config = get_cache_config()

    # Check local cache first
    if is_model_cached(model_id, config):
        logger.debug("Model %s found in local cache", model_id)
        return _get_model_cache_path(model_id, config)

    # Try cluster cache if configured
    if config.cluster_cache:
        if _download_from_cluster_cache(model_id, config):
            logger.info("Downloaded %s from cluster cache", model_id)
            return _get_model_cache_path(model_id, config)

    # Download from HuggingFace Hub if fallback enabled
    if config.hf_fallback:
        logger.info("Downloading %s from HuggingFace Hub", model_id)
        return _download_from_huggingface(model_id, config, revision)

    # Not in any cache and HF fallback disabled
    raise RuntimeError(
        f"Model '{model_id}' not found in local or cluster cache, "
        f"and HuggingFace fallback is disabled (SIE_HF_FALLBACK=false)"
    )


def _get_model_cache_path(model_id: str, config: CacheConfig) -> Path:
    """Get the local cache path for a model.

    Args:
        model_id: HuggingFace model ID.
        config: Cache configuration.

    Returns:
        Path to the model's cache directory.
    """
    return config.local_cache / f"models--{model_id.replace('/', '--')}"


def _download_from_cluster_cache(model_id: str, config: CacheConfig) -> bool:
    """Download model from cluster cache to local cache.

    Mirrors the HuggingFace Hub cache structure from cluster to local.

    Args:
        model_id: HuggingFace model ID.
        config: Cache configuration (must have cluster_cache set).

    Returns:
        True if successfully downloaded, False if not found in cluster cache.
    """
    if not config.cluster_cache:
        return False

    backend = get_storage_backend(config.cluster_cache)

    # Construct cluster cache path
    model_folder = f"models--{model_id.replace('/', '--')}"
    cluster_model_path = join_path(config.cluster_cache, model_folder)

    # Check if model exists in cluster cache
    if not backend.exists(join_path(cluster_model_path, "snapshots")):
        logger.debug("Model %s not found in cluster cache", model_id)
        return False

    # Create local cache directory
    local_model_path = config.local_cache / model_folder
    local_model_path.mkdir(parents=True, exist_ok=True)

    # Download the entire model directory structure
    _download_directory(backend, cluster_model_path, local_model_path)

    return True


def _download_directory(backend: StorageBackend, src_path: str, dst_path: Path) -> None:
    """Recursively download a directory from cloud storage.

    Args:
        backend: Storage backend instance.
        src_path: Source path in cloud storage.
        dst_path: Destination local path.
    """
    # Download files in current directory
    for filename in backend.list_files(src_path):
        src_file = join_path(src_path, filename)
        dst_file = dst_path / filename
        backend.download_file(src_file, dst_file)

    # Recursively download subdirectories
    for dirname in backend.list_dirs(src_path):
        src_subdir = join_path(src_path, dirname)
        dst_subdir = dst_path / dirname
        dst_subdir.mkdir(parents=True, exist_ok=True)
        _download_directory(backend, src_subdir, dst_subdir)


def _get_hf_ignore_patterns(
    model_id: str,
    revision: str | None,
    token: str | None,
) -> list[str]:
    """Check if a HF repo has safetensors files; This avoids downloading duplicate weight files (e.g., both .safetensors and .bin)."""
    from huggingface_hub import file_exists
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
    )

    try:
        has_safetensors = any(
            file_exists(model_id, filename, revision=revision, token=token)
            for filename in ("model.safetensors", "model.safetensors.index.json")
        )
        if has_safetensors:
            return HF_IGNORE_PATTERNS + HF_LEGACY_WEIGHT_PATTERNS
    except (HfHubHTTPError, GatedRepoError) as e:
        logger.warning(
            "Failed to list HF repo files; falling back to default ignore patterns",
            extra={
                "model_id": model_id,
                "revision": revision,
                "status_code": getattr(e.response, "status_code", None),
            },
        )
    return HF_IGNORE_PATTERNS


def _download_from_huggingface(
    model_id: str,
    config: CacheConfig,
    revision: str | None = None,
) -> Path:
    """Download model from HuggingFace Hub to local cache.

    Uses huggingface_hub's snapshot_download which automatically uses
    the standard HF cache structure (~/.cache/huggingface/hub/).

    Args:
        model_id: HuggingFace model ID.
        config: Cache configuration.
        revision: Specific model revision.

    Returns:
        Path to the model in local cache.

    Raises:
        GatedModelError: If model is gated and authentication fails.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    # Get token from environment
    token = os.environ.get("HF_TOKEN")

    try:
        # snapshot_download automatically uses HF cache structure
        # cache_dir should point to HF_HOME/hub where model dirs are created
        snapshot_download(
            model_id,
            revision=revision,
            token=token,
            cache_dir=str(config.local_cache),  # Points to HF_HOME/hub
            ignore_patterns=_get_hf_ignore_patterns(model_id, revision, token),
        )
        return _get_model_cache_path(model_id, config)

    except GatedRepoError as e:
        # User-friendly error for gated models
        raise GatedModelError(model_id, e) from e

    except RepositoryNotFoundError as e:
        # Check if this might be a gated model accessed without auth
        if token is None:
            msg = (
                f"Model '{model_id}' not found. This could mean:\n"
                f"  1. The model ID is incorrect\n"
                f"  2. The model is private/gated and requires authentication\n\n"
                f"If this is a gated model, set HF_TOKEN environment variable:\n"
                f"  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx\n\n"
                f"Original error: {e}"
            )
            raise RuntimeError(msg) from e
        raise

    except HfHubHTTPError as e:
        # Handle 401/403 errors that might indicate auth issues
        status_code = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
        if status_code in (401, 403):
            raise GatedModelError(model_id, e) from e
        raise


def populate_cluster_cache(
    model_id: str,
    config: CacheConfig | None = None,
) -> bool:
    """Upload model from local cache to cluster cache.

    Used by sie-admin to pre-populate cluster cache.

    Args:
        model_id: HuggingFace model ID.
        config: Cache configuration.

    Returns:
        True if successfully uploaded, False if model not in local cache.
    """
    if config is None:
        config = get_cache_config()

    if not config.cluster_cache:
        logger.warning("No cluster cache configured")
        return False

    if not is_model_cached(model_id, config):
        logger.warning("Model %s not in local cache, cannot populate cluster", model_id)
        return False

    # TODO: Implement upload to cluster cache
    # This requires adding upload methods to storage backends
    logger.info("populate_cluster_cache not yet implemented")
    return False
