"""Model weight caching with hierarchical lookup.

Implements the caching hierarchy for model weights:
1. Local cache (HF_HOME/hub by default)
2. Cluster cache (S3/GCS/Azure/Alibaba OSS object storage)
3. HuggingFace Hub fallback (if enabled)

The caching is transparent to adapters - they always see files in local cache.
This module pre-populates local cache from cluster cache if needed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
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
    """Cluster cache URL (s3://, gs://, abfs(s)://, or oss://), or None if not configured."""

    hf_fallback: bool = True
    """Whether to fallback to HuggingFace Hub for downloads."""


def get_cache_config() -> CacheConfig:
    """Get cache configuration from environment variables.

    Reads:
        SIE_LOCAL_CACHE: Local cache directory (default: HF_HOME/hub)
        SIE_CLUSTER_CACHE: Cluster cache URL (s3://, gs://, abfs(s)://, or oss://)
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

    # Cluster cache: S3/GCS/Azure/Alibaba OSS object-store URL
    cluster_cache = os.environ.get("SIE_CLUSTER_CACHE")

    # HF fallback: default true
    hf_fallback_str = os.environ.get("SIE_HF_FALLBACK", "true").lower()
    hf_fallback = hf_fallback_str in ("true", "1", "yes")

    return CacheConfig(
        local_cache=local_cache,
        cluster_cache=cluster_cache,
        hf_fallback=hf_fallback,
    )


def is_model_cached(
    model_id: str,
    config: CacheConfig | None = None,
    revision: str | None = None,
    *,
    required_files: list[str] | None = None,
) -> bool:
    """Check if a model is already in local cache.

    Uses HuggingFace Hub's cache structure to check for the model.

    Args:
        model_id: HuggingFace model ID (e.g., "BAAI/bge-m3").
        config: Cache configuration. If None, reads from environment.
        revision: Exact snapshot commit, branch, or tag to require. Branches
            and tags resolve through the local Hugging Face ``refs`` cache.
        required_files: Safe relative files required for a cache hit. By
            default, any file preserves the existing model-cache semantics.

    Returns:
        True if model appears to be cached locally.
    """
    if config is None:
        config = get_cache_config()

    if required_files is not None:
        for name in required_files:
            if (
                not isinstance(name, str)
                or any(character in name for character in ("\\", "\x00", ":"))
                or any(part in {"", ".", ".."} for part in name.split("/"))
            ):
                raise ValueError("Required cache files must be safe relative paths")

    # HF Hub cache structure: models--{org}--{model}/snapshots/{revision}/
    cache_dir = config.local_cache / f"models--{model_id.replace('/', '--')}"
    if not cache_dir.exists():
        return False

    # Check for a snapshot directory with files. When a revision is requested,
    # do not let an unrelated cached snapshot satisfy the immutable pin.
    snapshots_dir = cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return False

    if revision is not None:
        revision_path = Path(revision)
        if not revision_path.parts or revision == "." or revision_path.is_absolute() or ".." in revision_path.parts:
            return False
        resolved_revision = revision
        ref_path = cache_dir / "refs" / revision
        if ref_path.is_file():
            resolved_revision = ref_path.read_text().strip()
        resolved_path = Path(resolved_revision)
        if (
            not resolved_path.parts
            or resolved_revision == "."
            or resolved_path.is_absolute()
            or ".." in resolved_path.parts
        ):
            return False
        snapshot = snapshots_dir / resolved_revision
        return (
            snapshot.is_dir()
            and any(snapshot.iterdir())
            and (required_files is None or all((snapshot / name).is_file() for name in required_files))
        )

    return any(
        snapshot.is_dir()
        and any(snapshot.iterdir())
        and (required_files is None or all((snapshot / name).is_file() for name in required_files))
        for snapshot in snapshots_dir.iterdir()
    )


def ensure_model_cached(
    model_id: str,
    config: CacheConfig | None = None,
    revision: str | None = None,
    *,
    allow_patterns: list[str] | None = None,
    required_files: list[str] | None = None,
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
        allow_patterns: Optional file filter for Hub downloads only. Local and
            cluster cache lookup and population remain unchanged.
        required_files: Files required for a local or cluster hit. A partial
            snapshot continues through the existing fallback tiers.

    Returns:
        Path to cached model in local cache.

    Raises:
        GatedModelError: If model is gated and authentication fails.
        RuntimeError: If model not found in any cache tier and HF fallback disabled.
    """
    if config is None:
        config = get_cache_config()

    # Check local cache first
    if is_model_cached(model_id, config, revision=revision, required_files=required_files):
        logger.debug("Model %s found in local cache", model_id)
        return _get_model_cache_path(model_id, config)

    # Try cluster cache if configured
    if config.cluster_cache:
        if _download_from_cluster_cache(model_id, config) and is_model_cached(
            model_id, config, revision=revision, required_files=required_files
        ):
            logger.info("Downloaded %s from cluster cache", model_id)
            return _get_model_cache_path(model_id, config)

    # Download from HuggingFace Hub if fallback enabled
    if config.hf_fallback:
        logger.info("Downloading %s from HuggingFace Hub", model_id)
        return _download_from_huggingface(model_id, config, revision, allow_patterns=allow_patterns)

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

    # Check if model exists in cluster cache.
    # ``snapshots`` is a directory-like prefix; ``has_children`` does a
    # list-with-MaxKeys=1 instead of head_object, which would 404 on every
    # S3-compatible backend even when children are clearly present.
    if not backend.has_children(join_path(cluster_model_path, "snapshots")):
        logger.debug("Model %s not found in cluster cache", model_id)
        return False

    # Create local cache directory
    cache_root = config.local_cache.resolve()
    local_model_path = _cache_destination(cache_root, cache_root, model_folder, directory=True)
    local_model_path.mkdir(parents=True, exist_ok=True)

    # Download the entire model directory structure
    _download_directory(backend, cluster_model_path, local_model_path)

    return True


def _download_directory(backend: StorageBackend, src_path: str, dst_path: Path) -> None:
    """Recursively download a directory from cloud storage.

    The explicitly selected root may be a symlink. Descendant directories must
    not be symlinks, and file links must remain within that root. Downloads are
    staged privately and replaced atomically, preserving existing files on a
    failed transfer and never writing through an existing hardlink.

    Args:
        backend: Storage backend instance.
        src_path: Source path in cloud storage.
        dst_path: Destination local path.
    """
    root = dst_path.resolve()
    _download_cache_directory(backend, src_path, root, root)


def _cache_destination(root: Path, parent: Path, name: str, *, directory: bool) -> Path:
    """Validate an immediate backend child and its local destination."""
    if (
        not isinstance(name, str)
        or name in {"", ".", ".."}
        or any(character in name for character in ("/", "\\", "\x00"))
        or PureWindowsPath(name).drive
    ):
        raise ValueError("Invalid cluster cache child name")
    path = parent / name
    if not path.resolve().is_relative_to(root):
        raise ValueError("Cluster cache destination escapes its repository")
    # Do not let directory aliases redirect a repository or create cycles.
    directory_path = path if directory else parent
    for ancestor in (directory_path, *directory_path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError("Cluster cache descendant directories cannot be symlinks")
    return path


def _download_cache_directory(backend: StorageBackend, src_path: str, dst_path: Path, root: Path) -> None:
    """Walk beneath one admitted root without rebasing through child links."""
    for filename in backend.list_files(src_path):
        dst_file = _cache_destination(root, dst_path, filename, directory=False)
        src_file = join_path(src_path, filename)
        with TemporaryDirectory(prefix=".sie-cache-", dir=root) as temporary:
            staged_file = Path(temporary) / "download"
            backend.download_file(src_file, staged_file)
            _cache_destination(root, dst_path, filename, directory=False)
            staged_file.replace(dst_file)

    # Recursively download subdirectories
    for dirname in backend.list_dirs(src_path):
        dst_subdir = _cache_destination(root, dst_path, dirname, directory=True)
        dst_subdir.mkdir(parents=True, exist_ok=True)
        src_subdir = join_path(src_path, dirname)
        _download_cache_directory(backend, src_subdir, dst_subdir, root)


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
    *,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download model from HuggingFace Hub to local cache.

    Uses huggingface_hub's snapshot_download which automatically uses
    the standard HF cache structure (~/.cache/huggingface/hub/).

    Args:
        model_id: HuggingFace model ID.
        config: Cache configuration.
        revision: Specific model revision.
        allow_patterns: Optional file filter passed to snapshot_download.

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
            allow_patterns=allow_patterns,
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

    Used by cluster-management clients to pre-populate cluster cache. Inverse of
    :func:`_download_from_cluster_cache`: the model's HF-style cache
    directory (``models--{org}--{name}``) is mirrored recursively to the
    same folder name under ``config.cluster_cache``, so workers hydrating
    via :func:`ensure_model_cached` see an identical layout.

    Args:
        model_id: HuggingFace model ID.
        config: Cache configuration. If None, reads from environment.

    Returns:
        True if at least one file was uploaded, False if no cluster cache
        is configured, the model is not in local cache, or nothing was
        uploaded (per-file upload failures are logged by the backend and
        reflected in its returned count).
    """
    if config is None:
        config = get_cache_config()

    if not config.cluster_cache:
        logger.warning("No cluster cache configured")
        return False

    if not is_model_cached(model_id, config):
        logger.warning("Model %s not in local cache, cannot populate cluster", model_id)
        return False

    backend = get_storage_backend(config.cluster_cache)

    model_folder = f"models--{model_id.replace('/', '--')}"
    local_model_path = config.local_cache / model_folder
    cluster_model_path = join_path(config.cluster_cache, model_folder)

    file_count = backend.upload_directory(local_model_path, cluster_model_path)
    if file_count == 0:
        logger.warning("No files uploaded for %s to %s", model_id, cluster_model_path)
        return False

    logger.info("Uploaded %d file(s) for %s to cluster cache %s", file_count, model_id, cluster_model_path)
    return True
