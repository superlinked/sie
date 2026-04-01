"""Tests for disk cache management."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from sie_server.core.disk_cache import (
    CachedModelInfo,
    DiskCacheConfig,
    DiskStats,
    ModelDiskCacheManager,
    _format_bytes,
)


def create_hf_cache_model(
    cache_dir: Path,
    repo_id: str,
    commit_hash: str = "abc123def456",
    files: dict[str, bytes] | None = None,
    access_time: float | None = None,
) -> Path:
    """Create a realistic HuggingFace cache structure for a model.

    Creates:
        models--org--name/
        ├── blobs/
        │   └── <hash>  (actual file content)
        ├── refs/
        │   └── main  (contains commit hash)
        └── snapshots/
            └── <commit_hash>/
                └── file.txt -> ../../blobs/<hash>

    Args:
        cache_dir: The cache directory (e.g., tmp_path / "hub")
        repo_id: Model ID like "BAAI/bge-m3"
        commit_hash: Fake commit hash for the snapshot
        files: Dict of filename -> content bytes
        access_time: Optional access time to set on blobs

    Returns:
        Path to the model directory
    """
    if files is None:
        files = {
            "config.json": b'{"model_type": "test"}',
            "model.safetensors": b"fake model weights " * 100,  # ~1.9KB
        }

    # Create directory structure
    model_name = f"models--{repo_id.replace('/', '--')}"
    model_dir = cache_dir / model_name
    blobs_dir = model_dir / "blobs"
    refs_dir = model_dir / "refs"
    snapshot_dir = model_dir / "snapshots" / commit_hash

    blobs_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Write refs/main with commit hash
    (refs_dir / "main").write_text(commit_hash)

    # Create blobs and symlinks
    for filename, content in files.items():
        # Create blob with hash-like name
        blob_hash = f"{hash(content) & 0xFFFFFFFFFFFF:012x}"
        blob_path = blobs_dir / blob_hash
        blob_path.write_bytes(content)

        # Set access time if specified
        if access_time is not None:
            os.utime(blob_path, (access_time, access_time))

        # Create symlink in snapshot
        symlink_path = snapshot_dir / filename
        symlink_path.symlink_to(f"../../blobs/{blob_hash}")

    return model_dir


class TestDiskStats:
    """Tests for DiskStats dataclass."""

    def test_basic_stats(self) -> None:
        """Test basic disk statistics calculations."""
        stats = DiskStats(
            used_bytes=80 * (1024**3),  # 80 GB
            total_bytes=100 * (1024**3),  # 100 GB
        )

        assert stats.used_gb == pytest.approx(80.0)
        assert stats.total_gb == pytest.approx(100.0)
        assert stats.available_gb == pytest.approx(20.0)
        assert stats.usage_ratio == pytest.approx(0.8)
        assert stats.available_bytes == 20 * (1024**3)

    def test_zero_total(self) -> None:
        """Test handling of zero total bytes."""
        stats = DiskStats(used_bytes=0, total_bytes=0)

        assert stats.usage_ratio == 0.0
        assert stats.available_bytes == 0

    def test_full_disk(self) -> None:
        """Test full disk statistics."""
        stats = DiskStats(
            used_bytes=100 * (1024**3),
            total_bytes=100 * (1024**3),
        )

        assert stats.usage_ratio == pytest.approx(1.0)
        assert stats.available_bytes == 0


class TestDiskCacheConfig:
    """Tests for DiskCacheConfig dataclass."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = DiskCacheConfig(cache_dir=Path("/tmp/cache"))  # noqa: S108 — test fixture

        assert config.cache_dir == Path("/tmp/cache")  # noqa: S108 — test fixture
        assert config.pressure_threshold == 0.85

    def test_custom_threshold(self) -> None:
        """Test custom threshold configuration."""
        config = DiskCacheConfig(
            cache_dir=Path("/tmp/cache"),  # noqa: S108 — test fixture
            pressure_threshold=0.90,
        )

        assert config.pressure_threshold == 0.90


class TestCachedModelInfo:
    """Tests for CachedModelInfo dataclass."""

    def test_creation(self) -> None:
        """Test creating cached model info."""
        model = CachedModelInfo(
            repo_id="BAAI/bge-m3",
            size_bytes=5 * (1024**3),  # 5 GB
            last_accessed=time.time(),
            commit_hashes=["abc123", "def456"],
        )

        assert model.repo_id == "BAAI/bge-m3"
        assert model.size_gb == pytest.approx(5.0)
        assert len(model.commit_hashes) == 2

    def test_default_commit_hashes(self) -> None:
        """Test default empty commit hashes."""
        model = CachedModelInfo(
            repo_id="test/model",
            size_bytes=1024,
            last_accessed=time.time(),
        )

        assert model.commit_hashes == []


class TestFormatBytes:
    """Tests for _format_bytes helper."""

    def test_bytes(self) -> None:
        """Test formatting small byte values."""
        assert _format_bytes(500) == "500.0 B"

    def test_kilobytes(self) -> None:
        """Test formatting kilobyte values."""
        assert _format_bytes(1024) == "1.0 KB"
        assert _format_bytes(1536) == "1.5 KB"

    def test_megabytes(self) -> None:
        """Test formatting megabyte values."""
        assert _format_bytes(1024 * 1024) == "1.0 MB"

    def test_gigabytes(self) -> None:
        """Test formatting gigabyte values."""
        assert _format_bytes(1024**3) == "1.0 GB"
        assert _format_bytes(int(2.5 * 1024**3)) == "2.5 GB"


class TestModelDiskCacheManager:
    """Tests for ModelDiskCacheManager with real HF cache structure."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create a temporary cache directory."""
        cache = tmp_path / "hub"
        cache.mkdir()
        return cache

    @pytest.fixture
    def manager(self, cache_dir: Path) -> ModelDiskCacheManager:
        """Create a disk cache manager for testing."""
        config = DiskCacheConfig(cache_dir=cache_dir, pressure_threshold=0.85)
        return ModelDiskCacheManager(config)

    def test_init(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test initialization."""
        assert manager.cache_dir == cache_dir
        assert manager.pressure_threshold == 0.85

    def test_get_stats_real(self, manager: ModelDiskCacheManager) -> None:
        """Test getting real disk statistics."""
        stats = manager.get_stats()

        # Should return valid stats for the tmp directory's filesystem
        assert stats.total_bytes > 0
        assert stats.used_bytes >= 0
        assert 0.0 <= stats.usage_ratio <= 1.0

    def test_get_stats_nonexistent_dir(self, tmp_path: Path) -> None:
        """Test handling of non-existent directory."""
        config = DiskCacheConfig(cache_dir=tmp_path / "nonexistent")
        manager = ModelDiskCacheManager(config)

        # Should return zero stats on error
        stats = manager.get_stats()
        assert stats.used_bytes == 0
        assert stats.total_bytes == 0

    def test_check_pressure_mocked(self, manager: ModelDiskCacheManager) -> None:
        """Test pressure check with mocked disk usage (only thing we need to mock)."""
        # Mock low usage
        mock_usage_low = MagicMock()
        mock_usage_low.used = 70 * (1024**3)
        mock_usage_low.total = 100 * (1024**3)

        with patch("shutil.disk_usage", return_value=mock_usage_low):
            assert manager.check_pressure() is False

        # Mock high usage
        mock_usage_high = MagicMock()
        mock_usage_high.used = 90 * (1024**3)
        mock_usage_high.total = 100 * (1024**3)

        with patch("shutil.disk_usage", return_value=mock_usage_high):
            assert manager.check_pressure() is True

    def test_get_cached_models_empty(self, manager: ModelDiskCacheManager) -> None:
        """Test getting cached models from empty cache - no mocking needed."""
        models = manager.get_cached_models()
        assert models == []

    def test_get_cached_models_with_real_structure(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test scanning real HF cache structure - no mocking."""
        # Create two models with different access times
        old_time = time.time() - 3600  # 1 hour ago
        new_time = time.time()

        create_hf_cache_model(
            cache_dir,
            "old-org/old-model",
            commit_hash="old123",
            access_time=old_time,
        )
        create_hf_cache_model(
            cache_dir,
            "new-org/new-model",
            commit_hash="new456",
            access_time=new_time,
        )

        models = manager.get_cached_models()

        assert len(models) == 2
        # Should be sorted by last_accessed (oldest first = LRU order)
        assert models[0].repo_id == "old-org/old-model"
        assert models[1].repo_id == "new-org/new-model"
        assert models[0].last_accessed < models[1].last_accessed
        # Should have commit hashes for deletion
        assert "old123" in models[0].commit_hashes
        assert "new456" in models[1].commit_hashes

    def test_get_cached_models_skips_datasets(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test that datasets are skipped - uses real scanning."""
        # Create a model
        create_hf_cache_model(cache_dir, "test/model", commit_hash="model123")

        # Create a dataset (different prefix)
        dataset_dir = cache_dir / "datasets--test--dataset"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "snapshots").mkdir()

        models = manager.get_cached_models()

        assert len(models) == 1
        assert models[0].repo_id == "test/model"

    def test_get_lru_model_real(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test getting LRU model with real cache - no mocking."""
        old_time = time.time() - 3600
        new_time = time.time()

        create_hf_cache_model(cache_dir, "model-a", commit_hash="aaa", access_time=old_time)
        create_hf_cache_model(cache_dir, "model-b", commit_hash="bbb", access_time=new_time)

        lru = manager.get_lru_model()

        assert lru is not None
        assert lru.repo_id == "model-a"  # Oldest model

    def test_get_lru_model_with_exclude(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test getting LRU model with exclusions - no mocking."""
        old_time = time.time() - 3600
        new_time = time.time()

        create_hf_cache_model(cache_dir, "model-a", commit_hash="aaa", access_time=old_time)
        create_hf_cache_model(cache_dir, "model-b", commit_hash="bbb", access_time=new_time)

        # Exclude the oldest, should get the next one
        lru = manager.get_lru_model(exclude={"model-a"})

        assert lru is not None
        assert lru.repo_id == "model-b"

    def test_get_lru_model_empty(self, manager: ModelDiskCacheManager) -> None:
        """Test getting LRU model from empty cache."""
        lru = manager.get_lru_model()
        assert lru is None

    def test_evict_model_real_deletion(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test evicting a model actually deletes files - no mocking."""
        model_dir = create_hf_cache_model(
            cache_dir,
            "test/model-to-evict",
            commit_hash="evict123",
        )

        # Verify model exists
        assert model_dir.exists()
        models_before = manager.get_cached_models()
        assert len(models_before) == 1

        # Evict it
        model_info = models_before[0]
        freed = manager.evict_model(model_info)

        # Verify it's gone
        assert freed > 0
        assert not model_dir.exists()
        models_after = manager.get_cached_models()
        assert len(models_after) == 0

    def test_evict_model_no_commit_hashes(self, manager: ModelDiskCacheManager) -> None:
        """Test evicting a model with no commit hashes does nothing."""
        model = CachedModelInfo(
            repo_id="test/model",
            size_bytes=1024,
            last_accessed=time.time(),
            commit_hashes=[],
        )

        freed = manager.evict_model(model)
        assert freed == 0

    def test_ensure_space_no_pressure(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test ensure_space when no pressure - only mock disk_usage."""
        create_hf_cache_model(cache_dir, "test/model", commit_hash="abc123")

        # Mock low disk pressure
        mock_usage = MagicMock()
        mock_usage.used = 50 * (1024**3)  # 50%
        mock_usage.total = 100 * (1024**3)

        with patch("shutil.disk_usage", return_value=mock_usage):
            evicted = manager.ensure_space_before_download("new/model")

        assert evicted == []
        # Original model should still exist
        assert (cache_dir / "models--test--model").exists()

    def test_ensure_space_with_pressure_real_eviction(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test ensure_space with pressure actually evicts - real deletion."""
        # Create a model that will be evicted
        old_time = time.time() - 3600
        model_dir = create_hf_cache_model(cache_dir, "old/model", commit_hash="old123", access_time=old_time)

        assert model_dir.exists()

        # Mock disk pressure: high -> low after eviction
        call_count = [0]

        def mock_disk_usage(_path: Path) -> MagicMock:
            call_count[0] += 1
            mock = MagicMock()
            if call_count[0] == 1:
                # First check: high pressure
                mock.used = 90 * (1024**3)
            else:
                # After eviction: low pressure
                mock.used = 70 * (1024**3)
            mock.total = 100 * (1024**3)
            return mock

        with patch("shutil.disk_usage", side_effect=mock_disk_usage):
            evicted = manager.ensure_space_before_download("new/model")

        assert evicted == ["old/model"]
        # Model should actually be deleted
        assert not model_dir.exists()

    def test_ensure_space_excludes_target_model(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test that the model being downloaded is never evicted."""
        # Create only the model we're about to download (already partially cached)
        model_dir = create_hf_cache_model(cache_dir, "new/model", commit_hash="new123")

        # Mock high pressure
        mock_usage = MagicMock()
        mock_usage.used = 90 * (1024**3)
        mock_usage.total = 100 * (1024**3)

        with patch("shutil.disk_usage", return_value=mock_usage):
            evicted = manager.ensure_space_before_download("new/model")

        # Should not evict the target model
        assert evicted == []
        assert model_dir.exists()

    def test_ensure_space_evicts_multiple_until_below_threshold(
        self, manager: ModelDiskCacheManager, cache_dir: Path
    ) -> None:
        """Test that multiple models can be evicted to get below threshold."""
        # Create three models with different ages
        base_time = time.time()
        model1 = create_hf_cache_model(cache_dir, "oldest/model", commit_hash="aaa", access_time=base_time - 3600)
        model2 = create_hf_cache_model(cache_dir, "middle/model", commit_hash="bbb", access_time=base_time - 1800)
        model3 = create_hf_cache_model(cache_dir, "newest/model", commit_hash="ccc", access_time=base_time)

        # Mock: pressure stays high until 2 models evicted
        call_count = [0]

        def mock_disk_usage(_path: Path) -> MagicMock:
            call_count[0] += 1
            mock = MagicMock()
            if call_count[0] <= 2:
                mock.used = 90 * (1024**3)  # High
            else:
                mock.used = 70 * (1024**3)  # Low after 2 evictions
            mock.total = 100 * (1024**3)
            return mock

        with patch("shutil.disk_usage", side_effect=mock_disk_usage):
            evicted = manager.ensure_space_before_download("brand-new/model")

        # Should evict oldest and middle (in LRU order)
        assert evicted == ["oldest/model", "middle/model"]
        assert not model1.exists()
        assert not model2.exists()
        assert model3.exists()  # Newest should remain

    def test_touch_updates_directory_mtime(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test touching updates model directory mtime."""
        model_dir = create_hf_cache_model(cache_dir, "BAAI/bge-m3", commit_hash="touch123")

        # Get initial mtime
        initial_mtime = model_dir.stat().st_mtime

        # Wait briefly and touch
        time.sleep(0.01)
        manager.touch("BAAI/bge-m3")

        # Verify mtime updated
        new_mtime = model_dir.stat().st_mtime
        assert new_mtime >= initial_mtime

    def test_touch_nonexistent_model_no_error(self, manager: ModelDiskCacheManager) -> None:
        """Test touching non-existent model doesn't raise."""
        # Should not raise, just log debug
        manager.touch("nonexistent/model")


class TestModelDiskCacheManagerErrorHandling:
    """Tests for error handling - these need mocking for error injection."""

    @pytest.fixture
    def cache_dir(self, tmp_path: Path) -> Path:
        """Create a temporary cache directory."""
        cache = tmp_path / "hub"
        cache.mkdir()
        return cache

    @pytest.fixture
    def manager(self, cache_dir: Path) -> ModelDiskCacheManager:
        """Create a disk cache manager for testing."""
        config = DiskCacheConfig(cache_dir=cache_dir)
        return ModelDiskCacheManager(config)

    def test_get_cached_models_scan_error(self, manager: ModelDiskCacheManager) -> None:
        """Test graceful handling of scan_cache_dir errors."""
        with patch("huggingface_hub.scan_cache_dir", side_effect=Exception("Scan failed")):
            models = manager.get_cached_models()
            assert models == []

    def test_evict_model_delete_error(self, manager: ModelDiskCacheManager, cache_dir: Path) -> None:
        """Test graceful handling of delete_revisions errors."""
        create_hf_cache_model(cache_dir, "test/model", commit_hash="err123")
        models = manager.get_cached_models()
        model = models[0]

        # Mock delete_revisions to fail
        with patch("huggingface_hub.scan_cache_dir") as mock_scan:
            mock_cache_info = MagicMock()
            mock_cache_info.delete_revisions.side_effect = Exception("Delete failed")
            mock_scan.return_value = mock_cache_info

            freed = manager.evict_model(model)
            assert freed == 0
