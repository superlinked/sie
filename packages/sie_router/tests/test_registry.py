"""Tests for WorkerRegistry routing logic."""

import asyncio
from unittest.mock import patch

import pytest
from sie_router.registry import WorkerRegistry, _fill_first_score
from sie_router.types import WorkerState


class TestWorkerRegistry:
    """Test cases for WorkerRegistry."""

    def test_empty_registry_returns_none(self) -> None:
        """select_worker returns None when registry is empty."""
        registry = WorkerRegistry()
        assert registry.select_worker() is None
        assert registry.select_worker(gpu="l4") is None
        assert registry.select_worker(model="bge-m3") is None

    @pytest.mark.asyncio
    async def test_update_worker_adds_new_worker(self) -> None:
        """update_worker adds a new worker to the registry."""
        registry = WorkerRegistry()

        # Use WorkerStatusMessage format (gpus array with memory_*_bytes, models array with queue_depth)
        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "gpu_count": 1,
                "loaded_models": ["bge-m3"],
                "gpus": [
                    {
                        "device": "cuda:0",
                        "memory_used_bytes": int(10.0 * 1e9),
                        "memory_total_bytes": int(24.0 * 1e9),
                    }
                ],
                "models": [{"name": "bge-m3", "queue_depth": 5}],
            },
        )

        assert len(registry.workers) == 1
        worker = registry.get_worker("http://worker-0:8080")
        assert worker is not None
        assert worker.name == "worker-0"
        assert worker.machine_profile == "l4"
        assert worker.models == ["bge-m3"]
        assert worker.queue_depth == 5
        assert worker.healthy
        # Memory should be extracted from gpus array (kept in bytes, no conversion)
        assert worker.memory_used_bytes == int(10.0 * 1e9)
        assert worker.memory_total_bytes == int(24.0 * 1e9)

    @pytest.mark.asyncio
    async def test_machine_profile_filtering(self) -> None:
        """select_worker filters by machine profile."""
        registry = WorkerRegistry()

        # Add L4 worker
        await registry.update_worker(
            "http://worker-l4:8080",
            {"name": "worker-l4", "ready": True, "machine_profile": "l4"},
        )

        # Add A100 worker
        await registry.update_worker(
            "http://worker-a100:8080",
            {"name": "worker-a100", "ready": True, "machine_profile": "a100-80gb"},
        )

        # Select L4 worker
        worker = registry.select_worker(gpu="l4")
        assert worker is not None
        assert worker.machine_profile == "l4"

        # Select A100 worker
        worker = registry.select_worker(gpu="a100-80gb")
        assert worker is not None
        assert worker.machine_profile == "a100-80gb"

        # No H100 workers
        worker = registry.select_worker(gpu="h100")
        assert worker is None

    @pytest.mark.asyncio
    async def test_model_affinity(self) -> None:
        """select_worker prefers workers with model loaded."""
        registry = WorkerRegistry()

        # Worker with bge-m3 loaded (queue_depth from models array)
        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["bge-m3"],
                "models": [{"name": "bge-m3", "queue_depth": 10}],
            },
        )

        # Worker without model (no queue)
        await registry.update_worker(
            "http://worker-1:8080",
            {"name": "worker-1", "ready": True, "machine_profile": "l4", "loaded_models": [], "models": []},
        )

        # Should prefer worker-0 even with higher queue depth
        worker = registry.select_worker(model="bge-m3")
        assert worker is not None
        assert worker.name == "worker-0"

        # For unknown model, no workers have it loaded — falls back to lowest queue depth
        worker = registry.select_worker(model="gte-qwen2")
        assert worker is not None
        assert worker.name == "worker-1"  # Lowest queue depth (no fill-first for unknown models)

    @pytest.mark.asyncio
    async def test_load_balancing_fill_first(self) -> None:
        """select_worker with fill-first picks highest below-threshold queue."""
        registry = WorkerRegistry()

        # Worker with high queue (above 80% threshold: 100/64 = 156%)
        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 100}],
            },
        )

        # Worker with low queue (below threshold: 5/64 = 7.8%)
        await registry.update_worker(
            "http://worker-1:8080",
            {
                "name": "worker-1",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 5}],
            },
        )

        # Worker with medium queue (below threshold: 50/64 = 78%)
        await registry.update_worker(
            "http://worker-2:8080",
            {
                "name": "worker-2",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 50}],
            },
        )

        # Fill-first: prefers highest queue below threshold → worker-2 (50, below 80%)
        # worker-0 (100) is above threshold and penalized
        worker = registry.select_worker()
        assert worker is not None
        assert worker.name == "worker-2"

    @pytest.mark.asyncio
    async def test_load_balancing_legacy(self) -> None:
        """select_worker with fill-first disabled picks lowest queue depth."""
        registry = WorkerRegistry()

        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 100}],
            },
        )
        await registry.update_worker(
            "http://worker-1:8080",
            {
                "name": "worker-1",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 5}],
            },
        )

        with patch("sie_router.registry.FILL_FIRST_ENABLED", False):
            worker = registry.select_worker()
            assert worker is not None
            assert worker.name == "worker-1"  # Lowest queue depth

    @pytest.mark.asyncio
    async def test_unhealthy_workers_excluded(self) -> None:
        """select_worker skips unhealthy workers."""
        registry = WorkerRegistry()

        # Healthy worker with high queue
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )

        # Unhealthy worker with low queue
        await registry.update_worker(
            "http://worker-1:8080",
            {"name": "worker-1", "ready": True, "machine_profile": "l4"},
        )
        await registry.mark_unhealthy("http://worker-1:8080")

        # Should pick worker-0 (only healthy option)
        worker = registry.select_worker()
        assert worker is not None
        assert worker.name == "worker-0"

    @pytest.mark.asyncio
    async def test_has_capacity(self) -> None:
        """has_capacity returns True if GPU type is available."""
        registry = WorkerRegistry()

        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )

        assert registry.has_capacity("l4") is True
        assert registry.has_capacity("L4") is True  # Case insensitive
        assert registry.has_capacity("a100-80gb") is False

    @pytest.mark.asyncio
    async def test_heartbeat_timeout(self) -> None:
        """Workers are marked unhealthy after heartbeat timeout."""
        registry = WorkerRegistry(heartbeat_timeout_s=0.01)

        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )

        # Worker should be healthy initially
        worker = registry.get_worker("http://worker-0:8080")
        assert worker is not None
        assert worker.healthy

        # Wait for timeout
        await asyncio.sleep(0.02)

        # Check heartbeats - should mark as unhealthy
        unhealthy = await registry.check_heartbeats()
        assert "http://worker-0:8080" in unhealthy

        worker = registry.get_worker("http://worker-0:8080")
        assert worker is not None
        assert not worker.healthy

    @pytest.mark.asyncio
    async def test_remove_worker(self) -> None:
        """remove_worker removes worker from registry."""
        registry = WorkerRegistry()

        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )

        assert len(registry.workers) == 1

        await registry.remove_worker("http://worker-0:8080")

        assert len(registry.workers) == 0
        assert registry.get_worker("http://worker-0:8080") is None

    @pytest.mark.asyncio
    async def test_get_gpu_types(self) -> None:
        """get_gpu_types returns unique GPU types."""
        registry = WorkerRegistry()

        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )
        await registry.update_worker(
            "http://worker-1:8080",
            {"name": "worker-1", "ready": True, "machine_profile": "l4"},
        )
        await registry.update_worker(
            "http://worker-2:8080",
            {"name": "worker-2", "ready": True, "machine_profile": "a100-80gb"},
        )

        gpu_types = registry.get_gpu_types()
        assert set(gpu_types) == {"l4", "a100-80gb"}

    @pytest.mark.asyncio
    async def test_get_models(self) -> None:
        """get_models returns model to worker mapping."""
        registry = WorkerRegistry()

        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4", "loaded_models": ["bge-m3", "e5-large-v2"]},
        )
        await registry.update_worker(
            "http://worker-1:8080",
            {"name": "worker-1", "ready": True, "machine_profile": "l4", "loaded_models": ["bge-m3"]},
        )

        models = registry.get_models()
        assert "bge-m3" in models
        assert len(models["bge-m3"]) == 2
        assert "e5-large-v2" in models
        assert len(models["e5-large-v2"]) == 1

    @pytest.mark.asyncio
    async def test_cluster_status(self) -> None:
        """get_cluster_status returns aggregated status."""
        registry = WorkerRegistry()

        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "gpu_count": 2,
                "loaded_models": ["bge-m3"],
            },
        )
        await registry.update_worker(
            "http://worker-1:8080",
            {
                "name": "worker-1",
                "ready": True,
                "machine_profile": "a100-80gb",
                "gpu_count": 1,
                "loaded_models": ["bge-m3", "qwen3-8b"],
            },
        )

        status = registry.get_cluster_status()

        assert status.worker_count == 2
        assert status.gpu_count == 3  # 2 + 1
        assert status.models_loaded == 2  # bge-m3, qwen3-8b
        assert len(status.workers) == 2
        assert len(status.models) == 2

    @pytest.mark.asyncio
    async def test_gpu_match_with_model_affinity(self) -> None:
        """select_worker combines GPU match and model affinity."""
        registry = WorkerRegistry()

        # L4 worker with bge-m3
        await registry.update_worker(
            "http://worker-l4:8080",
            {"name": "worker-l4", "ready": True, "machine_profile": "l4", "loaded_models": ["bge-m3"]},
        )

        # A100 worker without model
        await registry.update_worker(
            "http://worker-a100:8080",
            {"name": "worker-a100", "ready": True, "machine_profile": "a100-80gb", "loaded_models": []},
        )

        # Request A100 for bge-m3 - should get A100 (profile match first)
        worker = registry.select_worker(gpu="a100-80gb", model="bge-m3")
        assert worker is not None
        assert worker.machine_profile == "a100-80gb"

        # Request L4 for bge-m3 - should get L4 with model
        worker = registry.select_worker(gpu="l4", model="bge-m3")
        assert worker is not None
        assert worker.machine_profile == "l4"
        assert "bge-m3" in worker.models


class TestWorkerReadiness:
    """Tests for worker readiness handling."""

    @pytest.mark.asyncio
    async def test_worker_not_ready_stays_unknown(self) -> None:
        """Worker with ready=False is not marked HEALTHY."""
        registry = WorkerRegistry()
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": False, "machine_profile": "l4"},
        )
        worker = registry.get_worker("http://worker-0:8080")
        assert worker is not None
        assert not worker.healthy  # UNKNOWN, not HEALTHY

    @pytest.mark.asyncio
    async def test_worker_ready_false_not_selected(self) -> None:
        """Worker with ready=False is not selected for routing."""
        registry = WorkerRegistry()
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": False, "machine_profile": "l4"},
        )
        # Should not select the not-ready worker
        worker = registry.select_worker(gpu="l4")
        assert worker is None

    @pytest.mark.asyncio
    async def test_worker_missing_ready_defaults_to_not_ready(self) -> None:
        """Worker without ready field defaults to not ready (backwards compat)."""
        registry = WorkerRegistry()
        # Old worker without ready field
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "machine_profile": "l4"},
        )
        worker = registry.get_worker("http://worker-0:8080")
        assert worker is not None
        # Should be UNKNOWN (not ready) by default
        assert not worker.healthy

    @pytest.mark.asyncio
    async def test_worker_becomes_ready(self) -> None:
        """Worker transitions from not-ready to ready."""
        registry = WorkerRegistry()

        # First message: not ready
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": False, "machine_profile": "l4"},
        )
        worker = registry.get_worker("http://worker-0:8080")
        assert worker is not None
        assert not worker.healthy

        # Second message: ready
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )
        worker = registry.get_worker("http://worker-0:8080")
        assert worker is not None
        assert worker.healthy

    @pytest.mark.asyncio
    async def test_callback_not_called_when_not_ready(self) -> None:
        """on_worker_healthy callback is NOT called when worker is not ready."""
        callback_invoked = False

        async def on_healthy(worker):
            nonlocal callback_invoked
            callback_invoked = True

        registry = WorkerRegistry(on_worker_healthy=on_healthy)

        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": False, "machine_profile": "l4"},
        )

        assert not callback_invoked  # Should NOT be called


class TestClusterStatusFormat:
    """Tests for cluster status wire format (consumed by sie-top)."""

    @pytest.mark.asyncio
    async def test_cluster_status_workers_have_memory_bytes(self) -> None:
        """Cluster status workers use memory_*_bytes (not GB)."""
        registry = WorkerRegistry()

        # Worker sends status in WorkerStatusMessage format
        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "gpu_count": 1,
                "loaded_models": ["bge-m3"],
                "gpus": [
                    {
                        "device": "cuda:0",
                        "memory_used_bytes": 10_000_000_000,  # 10 GB
                        "memory_total_bytes": 24_000_000_000,  # 24 GB
                    }
                ],
                "models": [],
            },
        )

        status = registry.get_cluster_status()

        # Verify workers have bytes fields (not GB)
        assert len(status.workers) == 1
        worker_info = status.workers[0]
        assert "memory_used_bytes" in worker_info
        assert "memory_total_bytes" in worker_info
        assert worker_info["memory_used_bytes"] == 10_000_000_000
        assert worker_info["memory_total_bytes"] == 24_000_000_000
        # Should NOT have the old GB fields
        assert "memory_used_gb" not in worker_info
        assert "memory_total_gb" not in worker_info
        assert "gpu_memory_used_gb" not in worker_info

    @pytest.mark.asyncio
    async def test_cluster_status_models_have_state(self) -> None:
        """Cluster status models include state field."""
        registry = WorkerRegistry()

        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["bge-m3", "e5-small"],
            },
        )

        status = registry.get_cluster_status()

        # Models should have state field
        assert len(status.models) == 2
        for model in status.models:
            assert "state" in model
            assert model["state"] == "loaded"  # Loaded models show as loaded
            assert "name" in model
            assert "worker_count" in model

    @pytest.mark.asyncio
    async def test_cluster_status_aggregates_multi_gpu_memory(self) -> None:
        """Memory is aggregated across multiple GPUs on a worker."""
        registry = WorkerRegistry()

        # Worker with 2 GPUs
        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "gpu_count": 2,
                "loaded_models": [],
                "gpus": [
                    {"device": "cuda:0", "memory_used_bytes": 5_000_000_000, "memory_total_bytes": 24_000_000_000},
                    {"device": "cuda:1", "memory_used_bytes": 8_000_000_000, "memory_total_bytes": 24_000_000_000},
                ],
                "models": [],
            },
        )

        status = registry.get_cluster_status()
        worker_info = status.workers[0]

        # Memory should be sum of both GPUs
        assert worker_info["memory_used_bytes"] == 13_000_000_000  # 5 + 8 GB
        assert worker_info["memory_total_bytes"] == 48_000_000_000  # 24 + 24 GB

    @pytest.mark.asyncio
    async def test_cluster_status_aggregates_queue_depth_from_models(self) -> None:
        """Queue depth is aggregated from models array."""
        registry = WorkerRegistry()

        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["bge-m3", "e5-small"],
                "models": [
                    {"name": "bge-m3", "queue_depth": 10},
                    {"name": "e5-small", "queue_depth": 5},
                ],
            },
        )

        worker = registry.get_worker("http://worker-0:8080")
        assert worker is not None
        assert worker.queue_depth == 15  # 10 + 5


class TestWorkerHealthyCallback:
    """Tests for the on_worker_healthy callback mechanism."""

    @pytest.mark.asyncio
    async def test_callback_called_on_new_worker(self) -> None:
        """Callback is invoked when a new worker is discovered."""
        callback_invoked = False
        callback_worker = None

        async def on_healthy(worker):
            nonlocal callback_invoked, callback_worker
            callback_invoked = True
            callback_worker = worker

        registry = WorkerRegistry(on_worker_healthy=on_healthy)

        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )

        assert callback_invoked
        assert callback_worker is not None
        assert callback_worker.name == "worker-0"

    @pytest.mark.asyncio
    async def test_callback_not_called_on_heartbeat(self) -> None:
        """Callback is NOT invoked on subsequent heartbeats (already healthy)."""
        call_count = 0

        async def on_healthy(worker):
            nonlocal call_count
            call_count += 1

        registry = WorkerRegistry(on_worker_healthy=on_healthy)

        # First update - should trigger callback
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )
        assert call_count == 1

        # Second update (heartbeat) - should NOT trigger callback
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )
        assert call_count == 1  # Still 1, not 2

    @pytest.mark.asyncio
    async def test_callback_called_on_recovery(self) -> None:
        """Callback is invoked when worker recovers from unhealthy state."""
        call_count = 0

        async def on_healthy(worker):
            nonlocal call_count
            call_count += 1

        registry = WorkerRegistry(on_worker_healthy=on_healthy)

        # First update - triggers callback
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )
        assert call_count == 1

        # Mark unhealthy
        await registry.mark_unhealthy("http://worker-0:8080")

        # Recovery update - should trigger callback again
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )
        assert call_count == 2  # Called again on recovery


class TestFillFirstScoring:
    """Tests for fill-first batch scoring strategy."""

    def test_score_prefers_higher_queue_depth_below_threshold(self) -> None:
        """Workers with more pending items score lower (better) below threshold."""
        empty = WorkerState(url="http://w0:8080", queue_depth=0, max_batch_requests=64)
        partial = WorkerState(url="http://w1:8080", queue_depth=30, max_batch_requests=64)
        half = WorkerState(url="http://w2:8080", queue_depth=50, max_batch_requests=64)

        # Higher queue depth = lower (better) score below 80% threshold
        assert _fill_first_score(partial) < _fill_first_score(empty)
        assert _fill_first_score(half) < _fill_first_score(partial)

    def test_score_penalizes_above_threshold(self) -> None:
        """Workers above 80% capacity score worse than any below-threshold worker."""
        below = WorkerState(url="http://w0:8080", queue_depth=1, max_batch_requests=64)
        at_threshold = WorkerState(url="http://w1:8080", queue_depth=52, max_batch_requests=64)  # 52/64 = 81%
        full = WorkerState(url="http://w2:8080", queue_depth=64, max_batch_requests=64)

        # Any below-threshold worker is better than any above-threshold worker
        assert _fill_first_score(below) < _fill_first_score(at_threshold)
        assert _fill_first_score(below) < _fill_first_score(full)

    def test_score_above_threshold_prefers_lower_queue(self) -> None:
        """Among above-threshold workers, lower queue depth is preferred."""
        w1 = WorkerState(url="http://w1:8080", queue_depth=52, max_batch_requests=64)
        w2 = WorkerState(url="http://w2:8080", queue_depth=60, max_batch_requests=64)

        assert _fill_first_score(w1) < _fill_first_score(w2)

    def test_score_handles_zero_capacity(self) -> None:
        """Workers with zero capacity don't cause division by zero."""
        w = WorkerState(url="http://w:8080", queue_depth=5, max_batch_requests=0)
        score = _fill_first_score(w)
        # max_batch_requests=0 triggers fallback: capacity = 0 or 64 = 64
        # fill_ratio = 5/64 ≈ 0.078 < 0.8 threshold → returns -queue_depth
        assert score == -5

    def test_score_uses_default_capacity(self) -> None:
        """Default max_batch_requests (64) is used when not set."""
        w = WorkerState(url="http://w:8080", queue_depth=10)
        assert w.max_batch_requests == 64
        score = _fill_first_score(w)
        assert score == -10

    @pytest.mark.asyncio
    async def test_fill_first_selects_partially_filled_worker(self) -> None:
        """select_worker with fill-first picks partially-filled over empty."""
        registry = WorkerRegistry()

        # Worker with some pending items (partially filled batch)
        await registry.update_worker(
            "http://worker-busy:8080",
            {
                "name": "worker-busy",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 30}],
                "max_batch_requests": 64,
            },
        )

        # Worker with empty queue
        await registry.update_worker(
            "http://worker-empty:8080",
            {
                "name": "worker-empty",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 0}],
                "max_batch_requests": 64,
            },
        )

        # With fill-first enabled, should prefer worker-busy (closer to full batch)
        with patch("sie_router.registry.FILL_FIRST_ENABLED", True):
            worker = registry.select_worker()
            assert worker is not None
            assert worker.name == "worker-busy"

    @pytest.mark.asyncio
    async def test_fill_first_avoids_near_full_worker(self) -> None:
        """select_worker with fill-first avoids near-capacity workers."""
        registry = WorkerRegistry()

        # Worker nearly full (> 80% of 64 = 51.2)
        await registry.update_worker(
            "http://worker-full:8080",
            {
                "name": "worker-full",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 55}],
                "max_batch_requests": 64,
            },
        )

        # Worker with moderate fill
        await registry.update_worker(
            "http://worker-moderate:8080",
            {
                "name": "worker-moderate",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 20}],
                "max_batch_requests": 64,
            },
        )

        with patch("sie_router.registry.FILL_FIRST_ENABLED", True):
            worker = registry.select_worker()
            assert worker is not None
            assert worker.name == "worker-moderate"

    @pytest.mark.asyncio
    async def test_legacy_scoring_when_fill_first_disabled(self) -> None:
        """select_worker uses lowest-queue-depth when fill-first is disabled."""
        registry = WorkerRegistry()

        # Worker with high queue
        await registry.update_worker(
            "http://worker-busy:8080",
            {
                "name": "worker-busy",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 30}],
            },
        )

        # Worker with low queue
        await registry.update_worker(
            "http://worker-idle:8080",
            {
                "name": "worker-idle",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["model-a"],
                "models": [{"name": "model-a", "queue_depth": 2}],
            },
        )

        # With fill-first disabled, should prefer lowest queue depth
        with patch("sie_router.registry.FILL_FIRST_ENABLED", False):
            worker = registry.select_worker()
            assert worker is not None
            assert worker.name == "worker-idle"

    @pytest.mark.asyncio
    async def test_max_batch_requests_extracted_from_status(self) -> None:
        """update_worker extracts max_batch_requests from status message."""
        registry = WorkerRegistry()

        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "max_batch_requests": 128,
            },
        )

        worker = registry.get_worker("http://worker-0:8080")
        assert worker is not None
        assert worker.max_batch_requests == 128

    @pytest.mark.asyncio
    async def test_max_batch_requests_defaults_to_64(self) -> None:
        """max_batch_requests defaults to 64 when not in status message."""
        registry = WorkerRegistry()

        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4"},
        )

        worker = registry.get_worker("http://worker-0:8080")
        assert worker is not None
        assert worker.max_batch_requests == 64
