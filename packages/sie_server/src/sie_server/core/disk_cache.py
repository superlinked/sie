"""Disk cache management for HuggingFace model weights.

Mirrors MemoryManager pattern but for disk storage.
Uses huggingface_hub's scan_cache_dir() for inspection and
delete_revisions() for eviction. LRU ordering is based on
blob_last_accessed timestamps.

Disk cache eviction behavior:
- Reactive LRU eviction when disk pressure exceeds threshold
- Check disk usage before downloading new models
- Evict least-recently-accessed models to make space
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HFCacheInfo

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DiskStats:
    """Disk usage statistics for a cache directory."""

    used_bytes: int
    total_bytes: int

    @property
    def usage_ratio(self) -> float:
        """Disk usage as a ratio (0.0 to 1.0)."""
        if self.total_bytes == 0:
            return 0.0
        return self.used_bytes / self.total_bytes

    @property
    def available_bytes(self) -> int:
        """Available disk space in bytes."""
        return max(0, self.total_bytes - self.used_bytes)

    @property
    def used_gb(self) -> float:
        """Used disk space in GB."""
        return self.used_bytes / (1024**3)

    @property
    def total_gb(self) -> float:
        """Total disk space in GB."""
        return self.total_bytes / (1024**3)

    @property
    def available_gb(self) -> float:
        """Available disk space in GB."""
        return self.available_bytes / (1024**3)


@dataclass
class DiskCacheConfig:
    """Configuration for disk cache management."""

    cache_dir: Path
    """Local cache directory (usually HF_HOME/hub)."""

    pressure_threshold: float = 0.85
    """Disk usage ratio that triggers LRU eviction (0.0 to 1.0)."""


@dataclass
class CachedModelInfo:
    """Information about a cached model."""

    repo_id: str
    """HuggingFace model ID (e.g., 'BAAI/bge-m3')."""

    size_bytes: int
    """Total size of the model on disk."""

    last_accessed: float
    """Timestamp of last blob access (for LRU ordering)."""

    commit_hashes: list[str] = field(default_factory=list)
    """Revision commit hashes (for delete_revisions())."""

    @property
    def size_gb(self) -> float:
        """Size in GB."""
        return self.size_bytes / (1024**3)


def _format_bytes(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size) < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


class ModelDiskCacheManager:
    """LRU disk cache manager for HuggingFace model weights.

    Mirrors MemoryManager pattern but for disk storage. Uses
    huggingface_hub's scan_cache_dir() for inspection and
    delete_revisions() for eviction.

    Usage:
        config = DiskCacheConfig(cache_dir=Path("~/.cache/huggingface/hub"))
        manager = ModelDiskCacheManager(config)

        # Before downloading a new model
        evicted = manager.ensure_space_before_download("BAAI/bge-m3")

        # After loading a model (update LRU)
        manager.touch("BAAI/bge-m3")
    """

    def __init__(
        self,
        config: DiskCacheConfig,
        pinned_provider: Callable[[], set[str]] | None = None,
    ) -> None:
        """Initialize the disk cache manager.

        Args:
            config: Disk cache configuration.
            pinned_provider: Optional callback returning the HF repo ids whose
                weights must never be evicted from disk (the pinned set).
                Evaluated lazily at eviction time so it tracks config changes.
        """
        self._config = config
        self._cache_dir = config.cache_dir
        self._threshold = config.pressure_threshold
        self._pinned_provider = pinned_provider

    @property
    def cache_dir(self) -> Path:
        """The cache directory being managed."""
        return self._cache_dir

    @property
    def pressure_threshold(self) -> float:
        """The disk usage threshold that triggers eviction."""
        return self._threshold

    def get_stats(self) -> DiskStats:
        """Get current disk usage statistics.

        Returns:
            DiskStats with used and total bytes for the cache directory's filesystem.
        """
        try:
            usage = shutil.disk_usage(self._cache_dir)
            return DiskStats(used_bytes=usage.used, total_bytes=usage.total)
        except OSError as e:
            logger.warning("Failed to get disk stats for %s: %s", self._cache_dir, e)
            return DiskStats(used_bytes=0, total_bytes=0)

    def check_pressure(self) -> bool:
        """Check if disk usage exceeds the pressure threshold.

        Returns:
            True if disk usage ratio exceeds the configured threshold.
        """
        stats = self.get_stats()
        return stats.usage_ratio > self._threshold

    def _scan_cache(self) -> HFCacheInfo | None:
        """Walk the whole cache directory once.

        This is the expensive operation in this module: it stats every blob of
        every repo, so callers should scan once and reuse the result.

        Returns:
            The scan result, or None if huggingface_hub is unavailable or the
            scan failed.
        """
        try:
            from huggingface_hub import scan_cache_dir
        except ImportError:
            logger.warning("huggingface_hub not available, cannot scan cache")
            return None

        try:
            return scan_cache_dir(self._cache_dir)
        except Exception as e:  # noqa: BLE001 — huggingface_hub scan errors are varied
            logger.warning("Failed to scan cache at %s: %s", self._cache_dir, e)
            return None

    def get_cached_models(self) -> list[CachedModelInfo]:
        """Get all cached models sorted by last access time (oldest first).

        Uses huggingface_hub's scan_cache_dir() to inspect the cache.

        Returns:
            List of CachedModelInfo sorted by last_accessed (LRU order).
        """
        cache_info = self._scan_cache()
        if cache_info is None:
            return []
        return self._models_from_scan(cache_info)

    @staticmethod
    def _models_from_scan(cache_info: HFCacheInfo) -> list[CachedModelInfo]:
        """Project a cache scan into model infos sorted LRU-first.

        Args:
            cache_info: A scan result from :meth:`_scan_cache`.

        Returns:
            List of CachedModelInfo sorted by last_accessed (LRU order).
        """
        models: list[CachedModelInfo] = []

        for repo in cache_info.repos:
            # Only consider model repos (not datasets or spaces)
            if repo.repo_type != "model":
                continue

            # Get the most recent blob access time across all revisions
            last_accessed = 0.0
            commit_hashes: list[str] = []

            for revision in repo.revisions:
                commit_hashes.append(revision.commit_hash)
                # Check each file's blob access time
                for cached_file in revision.files:
                    last_accessed = max(last_accessed, cached_file.blob_last_accessed)

            # Use repo's last_accessed if no file-level access time found
            if last_accessed == 0.0:
                last_accessed = repo.last_accessed

            models.append(
                CachedModelInfo(
                    repo_id=repo.repo_id,
                    size_bytes=repo.size_on_disk,
                    last_accessed=last_accessed,
                    commit_hashes=commit_hashes,
                )
            )

        # Sort by last_accessed (oldest first = LRU)
        models.sort(key=lambda m: m.last_accessed)

        return models

    def get_lru_model(self, exclude: set[str] | None = None) -> CachedModelInfo | None:
        """Get the least recently used model.

        Args:
            exclude: Set of model IDs to exclude from consideration.

        Returns:
            The LRU model info, or None if no models found.
        """
        exclude = exclude or set()
        models = self.get_cached_models()

        for model in models:
            if model.repo_id not in exclude:
                return model

        return None

    def evict_model(self, model: CachedModelInfo, cache_info: HFCacheInfo | None = None) -> int:
        """Delete a model from the disk cache.

        Uses huggingface_hub's delete_revisions() for safe deletion.

        Args:
            model: The model to evict.
            cache_info: Optional cache scan the delete strategy is computed
                from. Pass a scan that already covers ``model`` to avoid a
                second full cache walk; omit it to scan now. A scan that has
                gone stale is safe: unknown revisions are skipped with a
                warning and already-deleted paths are ignored.

        Returns:
            Number of bytes freed.
        """
        if not model.commit_hashes:
            logger.warning("No commit hashes for model %s, cannot evict", model.repo_id)
            return 0

        if cache_info is None:
            cache_info = self._scan_cache()
            if cache_info is None:
                return 0

        try:
            delete_strategy = cache_info.delete_revisions(*model.commit_hashes)
            if delete_strategy.expected_freed_size == 0:
                # The scan named revisions the cache no longer has (deleted
                # concurrently, or already evicted), so there is nothing to free.
                logger.debug("Nothing left to free for model %s, skipping", model.repo_id)
                return 0

            logger.info(
                "Evicting model '%s' from disk cache (will free %s)",
                model.repo_id,
                delete_strategy.expected_freed_size_str,
            )

            delete_strategy.execute()
            return model.size_bytes

        except Exception as e:  # noqa: BLE001 — cache eviction must not crash server
            logger.warning("Failed to evict model %s: %s", model.repo_id, e)
            return 0

    def ensure_space_before_download(
        self,
        model_id: str,
        estimated_bytes: int | None = None,
    ) -> list[str]:
        """Evict LRU models until disk pressure is below threshold.

        Called before downloading a new model to ensure there's space.

        Args:
            model_id: The model about to be downloaded (excluded from eviction).
            estimated_bytes: Optional estimate of download size (not currently used).

        Returns:
            List of evicted model IDs.
        """
        evicted: list[str] = []
        # Never evict the model being downloaded, nor any pinned model's weights:
        # re-downloading a pinned model on reload would reintroduce the cold start.
        exclude = {model_id}
        if self._pinned_provider is not None:
            exclude |= self._pinned_provider()

        # Scanning the cache walks and stats every blob, so it happens at most
        # once per call: victims are taken from that LRU-ordered snapshot and
        # only the cheap disk_usage check runs between evictions. Deletions are
        # reflected by that check, so freed space needs no rescan. The snapshot
        # going stale is safe — the cursor only moves forward (so the loop can
        # never spin), and a model that vanished concurrently frees nothing and
        # is skipped like any other failed eviction.
        candidates: list[CachedModelInfo] | None = None
        cache_info: HFCacheInfo | None = None
        cursor = 0

        while self.check_pressure():
            if candidates is None:
                cache_info = self._scan_cache()
                candidates = [] if cache_info is None else self._models_from_scan(cache_info)

            lru_model: CachedModelInfo | None = None
            while cursor < len(candidates):
                candidate = candidates[cursor]
                cursor += 1
                if candidate.repo_id not in exclude:
                    lru_model = candidate
                    break

            if lru_model is None:
                # No more models to evict
                stats = self.get_stats()
                logger.warning(
                    "Disk pressure still high (%.1f%%) but no models to evict",
                    stats.usage_ratio * 100,
                )
                break

            freed = self.evict_model(lru_model, cache_info=cache_info)
            if freed > 0:
                evicted.append(lru_model.repo_id)
                logger.info(
                    "Evicted '%s' from disk cache (freed %s)",
                    lru_model.repo_id,
                    _format_bytes(freed),
                )

        return evicted

    def touch(self, model_id: str) -> None:
        """Update access time for a model (for LRU tracking).

        Touches the model directory to update its mtime, which affects
        blob access times on next scan.

        Args:
            model_id: HuggingFace model ID (e.g., 'BAAI/bge-m3').
        """
        # Convert model ID to HF cache directory name
        cache_name = f"models--{model_id.replace('/', '--')}"
        model_dir = self._cache_dir / cache_name

        if model_dir.exists():
            try:
                # Touch the directory to update mtime
                os.utime(model_dir, None)
                logger.debug("Touched model directory: %s", model_dir)
            except OSError as e:
                logger.warning("Failed to touch model directory %s: %s", model_dir, e)
        else:
            logger.debug("Model directory does not exist: %s", model_dir)
