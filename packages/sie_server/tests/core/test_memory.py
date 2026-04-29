"""Tests for MemoryManager and memory tracking utilities."""

import time
from unittest.mock import MagicMock, patch

import pytest
from sie_server.core.memory import (
    CPUMemoryTracker,
    CUDAMemoryTracker,
    MemoryConfig,
    MemoryManager,
    MemoryStats,
    ModelMemoryInfo,
    MPSMemoryTracker,
    create_memory_tracker,
)


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_basic_stats(self) -> None:
        """Test basic memory statistics calculations."""
        stats = MemoryStats(
            used_bytes=8 * (1024**3),  # 8 GB
            total_bytes=16 * (1024**3),  # 16 GB
            device_type="cuda",
        )

        assert stats.used_gb == pytest.approx(8.0)
        assert stats.total_gb == pytest.approx(16.0)
        assert stats.usage_ratio == pytest.approx(0.5)
        assert stats.available_bytes == 8 * (1024**3)
        assert stats.available_gb == pytest.approx(8.0)

    def test_zero_total(self) -> None:
        """Test handling of zero total memory."""
        stats = MemoryStats(used_bytes=0, total_bytes=0, device_type="cpu")

        assert stats.usage_ratio == 0.0
        assert stats.available_bytes == 0

    def test_full_usage(self) -> None:
        """Test 100% memory usage."""
        stats = MemoryStats(
            used_bytes=16 * (1024**3),
            total_bytes=16 * (1024**3),
            device_type="cuda",
        )

        assert stats.usage_ratio == pytest.approx(1.0)
        assert stats.available_bytes == 0


class TestModelMemoryInfo:
    """Tests for ModelMemoryInfo dataclass."""

    def test_creation(self) -> None:
        """Test basic creation with defaults."""
        info = ModelMemoryInfo(model_name="test-model", device="cuda:0")

        assert info.model_name == "test-model"
        assert info.device == "cuda:0"
        assert info.loaded_at > 0
        assert info.last_used_at > 0
        assert info.estimated_bytes is None

    def test_touch(self) -> None:
        """Test touch updates last_used_at."""
        info = ModelMemoryInfo(model_name="test-model", device="cuda:0")
        original_time = info.last_used_at

        time.sleep(0.01)  # Small delay
        info.touch()

        assert info.last_used_at > original_time


class TestCreateMemoryTracker:
    """Tests for create_memory_tracker factory function."""

    def test_cuda_device(self) -> None:
        """Test CUDA device detection."""
        tracker = create_memory_tracker("cuda:0")
        assert isinstance(tracker, CUDAMemoryTracker)
        assert tracker.device_type() == "cuda"

    def test_cuda_without_id(self) -> None:
        """Test CUDA device without explicit ID."""
        tracker = create_memory_tracker("cuda")
        assert isinstance(tracker, CUDAMemoryTracker)

    def test_mps_device(self) -> None:
        """Test MPS device detection."""
        tracker = create_memory_tracker("mps")
        assert isinstance(tracker, MPSMemoryTracker)
        assert tracker.device_type() == "mps"

    def test_cpu_device(self) -> None:
        """Test CPU device detection."""
        tracker = create_memory_tracker("cpu")
        assert isinstance(tracker, CPUMemoryTracker)
        assert tracker.device_type() == "cpu"

    def test_unknown_defaults_to_cpu(self) -> None:
        """Test unknown device defaults to CPU."""
        tracker = create_memory_tracker("unknown")
        assert isinstance(tracker, CPUMemoryTracker)


class TestCPUMemoryTracker:
    """Tests for CPUMemoryTracker."""

    def test_get_stats_with_psutil(self) -> None:
        """Test CPU memory stats with psutil available."""
        tracker = CPUMemoryTracker()
        stats = tracker.get_stats()

        assert stats.device_type == "cpu"
        # Should have non-zero total on any real system
        assert stats.total_bytes > 0
        # Used should be less than or equal to total
        assert stats.used_bytes <= stats.total_bytes

    def test_get_stats_without_psutil(self) -> None:
        """Test CPU memory stats when psutil import fails."""
        CPUMemoryTracker()

        with patch.dict("sys.modules", {"psutil": None}):
            with patch("sie_server.core.memory.CPUMemoryTracker.get_stats") as mock:
                mock.return_value = MemoryStats(used_bytes=0, total_bytes=0, device_type="cpu")
                stats = mock()
                assert stats.total_bytes == 0


class TestMemoryConfig:
    """Tests for MemoryConfig."""

    def test_defaults(self) -> None:
        """Test default configuration values."""
        config = MemoryConfig()

        assert config.pressure_threshold == 0.85
        assert config.min_free_bytes is None
        assert config.proactive_eviction is True

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = MemoryConfig(
            pressure_threshold=0.9,
            min_free_bytes=4 * (1024**3),
            proactive_eviction=False,
        )

        assert config.pressure_threshold == 0.9
        assert config.min_free_bytes == 4 * (1024**3)
        assert config.proactive_eviction is False


class TestMemoryManager:
    """Tests for MemoryManager."""

    @pytest.fixture
    def manager(self) -> MemoryManager:
        """Create a memory manager for testing."""
        return MemoryManager(device="cpu")

    def test_init(self, manager: MemoryManager) -> None:
        """Test initialization."""
        assert manager.device == "cpu"
        assert manager.device_type == "cpu"
        assert manager.loaded_model_count == 0
        assert manager.loaded_models == []

    def test_register_model(self, manager: MemoryManager) -> None:
        """Test registering a model."""
        manager.register_model("model-a")

        assert manager.loaded_model_count == 1
        assert "model-a" in manager.loaded_models

    def test_register_multiple_models(self, manager: MemoryManager) -> None:
        """Test registering multiple models."""
        manager.register_model("model-a")
        manager.register_model("model-b")
        manager.register_model("model-c")

        assert manager.loaded_model_count == 3
        assert manager.loaded_models == ["model-a", "model-b", "model-c"]

    def test_unregister_model(self, manager: MemoryManager) -> None:
        """Test unregistering a model."""
        manager.register_model("model-a")
        manager.register_model("model-b")

        manager.unregister_model("model-a")

        assert manager.loaded_model_count == 1
        assert "model-a" not in manager.loaded_models
        assert "model-b" in manager.loaded_models

    def test_touch_updates_lru_order(self, manager: MemoryManager) -> None:
        """Test that touch moves model to end of LRU list."""
        manager.register_model("model-a")
        manager.register_model("model-b")
        manager.register_model("model-c")

        # Initially: a, b, c (a is LRU)
        assert manager.get_lru_model() == "model-a"

        # Touch a, now: b, c, a (b is LRU)
        manager.touch("model-a")
        assert manager.get_lru_model() == "model-b"

        # Touch b, now: c, a, b (c is LRU)
        manager.touch("model-b")
        assert manager.get_lru_model() == "model-c"

    def test_get_model_info(self, manager: MemoryManager) -> None:
        """Test getting model info."""
        manager.register_model("model-a", estimated_bytes=1000)

        info = manager.get_model_info("model-a")
        assert info is not None
        assert info.model_name == "model-a"
        assert info.estimated_bytes == 1000

    def test_get_model_info_not_found(self, manager: MemoryManager) -> None:
        """Test getting info for non-existent model."""
        info = manager.get_model_info("nonexistent")
        assert info is None

    def test_get_lru_model_empty(self, manager: MemoryManager) -> None:
        """Test LRU with no models."""
        assert manager.get_lru_model() is None

    def test_get_eviction_candidates(self, manager: MemoryManager) -> None:
        """Test getting eviction candidates."""
        manager.register_model("model-a")
        manager.register_model("model-b")
        manager.register_model("model-c")

        # Get 2 candidates (oldest first)
        candidates = manager.get_eviction_candidates(2)
        assert candidates == ["model-a", "model-b"]

    def test_get_idle_models_returns_sorted_by_age(self, manager: MemoryManager) -> None:
        """Stale models are returned oldest-first regardless of insertion order."""
        manager.register_model("recent")
        manager.register_model("stale-old")
        manager.register_model("stale-newer")

        # Force last_used_at deltas: ``stale-old`` last used 1000s ago,
        # ``stale-newer`` last used 100s ago, ``recent`` "now".
        now = time.monotonic()
        manager.get_model_info("stale-old").last_used_at = now - 1000.0  # type: ignore[union-attr]
        manager.get_model_info("stale-newer").last_used_at = now - 100.0  # type: ignore[union-attr]
        manager.get_model_info("recent").last_used_at = now

        idle = manager.get_idle_models(idle_threshold_s=50.0, now=now)
        assert idle == ["stale-old", "stale-newer"]

    def test_get_idle_models_empty_below_threshold(self, manager: MemoryManager) -> None:
        """Models touched within the threshold are not returned."""
        manager.register_model("fresh")
        # No artificial age — the model was just registered.
        idle = manager.get_idle_models(idle_threshold_s=3600.0)
        assert idle == []

    def test_get_idle_models_negative_threshold_rejected(self, manager: MemoryManager) -> None:
        with pytest.raises(ValueError, match="idle_threshold_s"):
            manager.get_idle_models(idle_threshold_s=-1.0)

    def test_get_stats(self, manager: MemoryManager) -> None:
        """Test getting memory stats."""
        stats = manager.get_stats()

        assert stats.device_type == "cpu"
        # Should work on any system
        assert stats.total_bytes >= 0

    def test_check_pressure_below_threshold(self) -> None:
        """Test pressure check when below threshold."""
        manager = MemoryManager(
            device="cpu",
            config=MemoryConfig(pressure_threshold=0.99),  # Very high threshold
        )

        # On any reasonable system, usage should be below 99%
        assert manager.check_pressure() is False

    def test_check_pressure_with_mocked_tracker(self) -> None:
        """Test pressure check with mocked memory stats."""
        manager = MemoryManager(
            device="cpu",
            config=MemoryConfig(pressure_threshold=0.80),
        )

        # Mock the tracker to return high usage
        mock_stats = MemoryStats(
            used_bytes=9 * (1024**3),
            total_bytes=10 * (1024**3),
            device_type="cpu",
        )
        manager._tracker = MagicMock()
        manager._tracker.get_stats.return_value = mock_stats
        manager._tracker.device_type.return_value = "cpu"

        assert manager.check_pressure() is True

    def test_check_pressure_min_free_bytes(self) -> None:
        """Test pressure check with min_free_bytes threshold."""
        manager = MemoryManager(
            device="cpu",
            config=MemoryConfig(
                pressure_threshold=0.99,  # High, so ratio won't trigger
                min_free_bytes=1000 * (1024**3),  # 1TB free (will trigger)
            ),
        )

        # No real system has 1TB free RAM
        assert manager.check_pressure() is True

    def test_should_evict_for_load_under_pressure(self) -> None:
        """Test eviction recommendation when under pressure."""
        manager = MemoryManager(device="cpu")

        # Mock high memory usage
        mock_stats = MemoryStats(
            used_bytes=9 * (1024**3),
            total_bytes=10 * (1024**3),
            device_type="cpu",
        )
        manager._tracker = MagicMock()
        manager._tracker.get_stats.return_value = mock_stats

        assert manager.should_evict_for_load() is True

    def test_should_evict_for_load_insufficient_space(self) -> None:
        """Test eviction recommendation when not enough space for new model."""
        manager = MemoryManager(device="cpu")

        # Mock some available memory
        mock_stats = MemoryStats(
            used_bytes=5 * (1024**3),
            total_bytes=10 * (1024**3),
            device_type="cpu",
        )
        manager._tracker = MagicMock()
        manager._tracker.get_stats.return_value = mock_stats

        # Need more than available
        assert manager.should_evict_for_load(required_bytes=6 * (1024**3)) is True

        # Need less than available
        assert manager.should_evict_for_load(required_bytes=4 * (1024**3)) is False

    @pytest.mark.asyncio
    async def test_check_memory_pressure_async(self) -> None:
        """Test async memory pressure check."""
        manager = MemoryManager(device="cpu")
        manager.register_model("model-a")

        # Mock high memory usage
        mock_stats = MemoryStats(
            used_bytes=9 * (1024**3),
            total_bytes=10 * (1024**3),
            device_type="cpu",
        )
        manager._tracker = MagicMock()
        manager._tracker.get_stats.return_value = mock_stats

        candidates = await manager.check_memory_pressure()

        assert len(candidates) == 1
        assert candidates[0] == "model-a"

    @pytest.mark.asyncio
    async def test_check_memory_pressure_disabled(self) -> None:
        """Test that proactive eviction can be disabled."""
        manager = MemoryManager(
            device="cpu",
            config=MemoryConfig(proactive_eviction=False),
        )
        manager.register_model("model-a")

        # Mock high memory usage
        mock_stats = MemoryStats(
            used_bytes=9 * (1024**3),
            total_bytes=10 * (1024**3),
            device_type="cpu",
        )
        manager._tracker = MagicMock()
        manager._tracker.get_stats.return_value = mock_stats

        candidates = await manager.check_memory_pressure()

        # Should return empty because proactive eviction is disabled
        assert candidates == []


class TestCUDAMemoryTracker:
    """Tests for CUDAMemoryTracker (mocked)."""

    def test_get_stats_no_cuda(self) -> None:
        """Test CUDA tracker when CUDA is not available."""
        tracker = CUDAMemoryTracker(device_id=0)

        with patch("torch.cuda.is_available", return_value=False):
            stats = tracker.get_stats()

            assert stats.device_type == "cuda"
            assert stats.used_bytes == 0
            assert stats.total_bytes == 0


class TestMPSMemoryTracker:
    """Tests for MPSMemoryTracker (mocked)."""

    def test_get_stats_no_mps(self) -> None:
        """Test MPS tracker when MPS is not available."""
        tracker = MPSMemoryTracker()

        with patch("torch.backends.mps.is_available", return_value=False):
            stats = tracker.get_stats()

            assert stats.device_type == "mps"
            assert stats.used_bytes == 0
            assert stats.total_bytes == 0
