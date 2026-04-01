"""Tests for Resource Pool management.

Per DESIGN.md Section 10.3, pools provide isolated capacity for benchmarking
and other use cases. These tests verify pool CRUD operations, worker assignment,
and routing with pools.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from sie_router.pools import (
    DEFAULT_POOL_NAME,
    AssignedWorker,
    DefaultPoolProtectedError,
    InvalidMachineProfileError,
    Pool,
    PoolManager,
    PoolSpec,
    PoolState,
    PoolStatus,
    parse_gpu_param,
)
from sie_router.types import MachineProfile


class TestParseGpuParam:
    """Test GPU parameter parsing."""

    def test_simple_gpu(self) -> None:
        """Parse simple GPU type without pool."""
        pool, gpu = parse_gpu_param("l4")
        assert pool is None
        assert gpu == "l4"

    def test_pool_prefixed_gpu(self) -> None:
        """Parse pool-prefixed GPU type."""
        pool, gpu = parse_gpu_param("eval-bench/l4")
        assert pool == "eval-bench"
        assert gpu == "l4"

    def test_pool_with_complex_gpu(self) -> None:
        """Parse pool with complex GPU type containing hyphens."""
        pool, gpu = parse_gpu_param("my-pool/a100-80gb")
        assert pool == "my-pool"
        assert gpu == "a100-80gb"

    def test_multiple_slashes(self) -> None:
        """Only first slash is used for splitting."""
        pool, gpu = parse_gpu_param("pool/gpu/extra")
        assert pool == "pool"
        assert gpu == "gpu/extra"


class TestPoolSpec:
    """Test PoolSpec dataclass."""

    def test_pool_spec_creation(self) -> None:
        """Create PoolSpec with gpus."""
        spec = PoolSpec(name="test-pool", gpus={"l4": 2, "a100-80gb": 1})
        assert spec.name == "test-pool"
        assert spec.gpus == {"l4": 2, "a100-80gb": 1}

    def test_pool_spec_empty_gpus(self) -> None:
        """Create PoolSpec without gpus (uses cluster default)."""
        spec = PoolSpec(name="test-pool", gpus={})
        assert spec.name == "test-pool"
        assert spec.gpus == {}


class TestPool:
    """Test Pool dataclass."""

    def test_pool_creation(self) -> None:
        """Create Pool with spec and status."""
        spec = PoolSpec(name="test-pool", gpus={"l4": 2})
        status = PoolStatus(
            state=PoolState.ACTIVE,
            assigned_workers=[
                AssignedWorker(name="worker-0", url="http://worker-0:8080", gpu="l4"),
                AssignedWorker(name="worker-1", url="http://worker-1:8080", gpu="l4"),
            ],
        )
        pool = Pool(spec=spec, status=status)

        assert pool.spec.name == "test-pool"
        assert pool.status.state == PoolState.ACTIVE
        assert len(pool.status.assigned_workers) == 2

    def test_pool_worker_urls(self) -> None:
        """Get worker URLs from pool."""
        spec = PoolSpec(name="test-pool", gpus={"l4": 2})
        status = PoolStatus(
            state=PoolState.ACTIVE,
            assigned_workers=[
                AssignedWorker(name="worker-0", url="http://worker-0:8080", gpu="l4"),
                AssignedWorker(name="worker-1", url="http://worker-1:8080", gpu="l4"),
            ],
        )
        pool = Pool(spec=spec, status=status)

        urls = {w.url for w in pool.status.assigned_workers}
        assert urls == {"http://worker-0:8080", "http://worker-1:8080"}


class TestPoolState:
    """Test PoolState enum."""

    def test_pool_states(self) -> None:
        """All pool states are defined."""
        assert PoolState.PENDING.value == "pending"
        assert PoolState.ACTIVE.value == "active"
        assert PoolState.EXPIRED.value == "expired"


class TestPoolManager:
    """Test PoolManager class."""

    def test_pool_manager_creation(self) -> None:
        """Create PoolManager."""
        manager = PoolManager(use_kubernetes=False)
        assert manager._use_kubernetes is False
        assert manager._pools == {}

    @pytest.mark.asyncio
    async def test_create_pool_local(self) -> None:
        """Create pool in local mode (no K8s)."""
        manager = PoolManager(use_kubernetes=False)

        pool = await manager.create_pool(
            name="test-pool",
            gpus={"l4": 1},
        )

        assert pool.spec.name == "test-pool"
        assert pool.spec.gpus == {"l4": 1}
        # In local mode without workers, pool starts as pending
        assert pool.status.state == PoolState.PENDING

    @pytest.mark.asyncio
    async def test_get_pool(self) -> None:
        """Get pool by name (sync method)."""
        manager = PoolManager(use_kubernetes=False)

        # Create pool
        await manager.create_pool(name="test-pool", gpus={"l4": 1})

        # Get pool (sync method)
        pool = manager.get_pool("test-pool")
        assert pool is not None
        assert pool.spec.name == "test-pool"

        # Get non-existent pool
        pool = manager.get_pool("non-existent")
        assert pool is None

    @pytest.mark.asyncio
    async def test_delete_pool(self) -> None:
        """Delete pool by name."""
        manager = PoolManager(use_kubernetes=False)

        # Create pool
        await manager.create_pool(name="test-pool", gpus={"l4": 1})

        # Delete pool
        deleted = await manager.delete_pool("test-pool")
        assert deleted is True

        # Pool should be gone
        pool = manager.get_pool("test-pool")
        assert pool is None

        # Delete non-existent pool
        deleted = await manager.delete_pool("non-existent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_pools_property(self) -> None:
        """Access all pools via pools property."""
        manager = PoolManager(use_kubernetes=False)

        # Create multiple pools
        await manager.create_pool(name="pool-1", gpus={"l4": 1})
        await manager.create_pool(name="pool-2", gpus={"a100-80gb": 2})

        # Access pools via property
        pools = manager.pools
        assert len(pools) == 2
        assert "pool-1" in pools
        assert "pool-2" in pools

    @pytest.mark.asyncio
    async def test_renew_lease(self) -> None:
        """Renew pool lease."""
        manager = PoolManager(use_kubernetes=False)

        # Create pool
        await manager.create_pool(name="test-pool", gpus={"l4": 1})

        # Renew lease
        renewed = await manager.renew_lease("test-pool")
        assert renewed is True

        # Renew non-existent pool
        renewed = await manager.renew_lease("non-existent")
        assert renewed is False

    @pytest.mark.asyncio
    async def test_pool_workers_via_status(self) -> None:
        """Get worker URLs for a pool via status."""
        manager = PoolManager(use_kubernetes=False)

        # Create pool with no workers (local mode)
        await manager.create_pool(name="test-pool", gpus={"l4": 1})

        # Get pool and check workers
        pool = manager.get_pool("test-pool")
        assert pool is not None
        assert pool.status.assigned_workers == []

    @pytest.mark.asyncio
    async def test_assign_workers_to_pool(self) -> None:
        """Assign workers to a pool."""
        manager = PoolManager(use_kubernetes=False)

        # Create pool
        await manager.create_pool(name="test-pool", gpus={"l4": 2})

        # Get pool and assign workers directly
        pool = manager.get_pool("test-pool")
        assert pool is not None

        pool.status.assigned_workers = [
            AssignedWorker(name="worker-0", url="http://worker-0:8080", gpu="l4"),
            AssignedWorker(name="worker-1", url="http://worker-1:8080", gpu="l4"),
        ]
        pool.status.state = PoolState.ACTIVE

        # Verify assignment
        assert len(pool.status.assigned_workers) == 2
        urls = {w.url for w in pool.status.assigned_workers}
        assert urls == {"http://worker-0:8080", "http://worker-1:8080"}


class TestPoolManagerK8s:
    """Test PoolManager with mocked K8s API."""

    def test_pool_manager_k8s_mode_creation(self) -> None:
        """Create PoolManager in K8s mode (no K8s calls until start)."""
        # Creating the manager doesn't call K8s, only start() does
        manager = PoolManager(
            use_kubernetes=True,
            k8s_namespace="sie-test",
        )
        assert manager._use_kubernetes is True
        assert manager._k8s_namespace == "sie-test"

    @pytest.mark.asyncio
    async def test_pool_manager_local_fallback(self) -> None:
        """PoolManager works in local mode without K8s."""
        # Local mode works without K8s
        manager = PoolManager(use_kubernetes=False)
        await manager.start()

        # Create pool
        pool = await manager.create_pool(name="test-pool", gpus={"l4": 1})
        assert pool.name == "test-pool"

        # Cleanup
        await manager.stop()


class TestMachineProfileValidation:
    """Test machine profile validation in PoolManager."""

    def test_pool_manager_with_profiles(self) -> None:
        """Create PoolManager with machine profiles."""
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
            "l4-spot": MachineProfile(name="l4-spot", gpu_type="nvidia-l4", spot=True),
            "a100-40gb": MachineProfile(name="a100-40gb", gpu_type="nvidia-a100"),
        }
        manager = PoolManager(use_kubernetes=False, machine_profiles=profiles)

        assert len(manager.machine_profiles) == 3
        assert "l4" in manager.machine_profiles
        assert "l4-spot" in manager.machine_profiles
        assert manager.machine_profiles["l4-spot"].spot is True

    @pytest.mark.asyncio
    async def test_create_pool_valid_profiles(self) -> None:
        """Create pool with valid machine profile keys."""
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
            "l4-spot": MachineProfile(name="l4-spot", gpu_type="nvidia-l4", spot=True),
        }
        manager = PoolManager(use_kubernetes=False, machine_profiles=profiles)

        # Valid profile keys should work
        pool = await manager.create_pool(name="test-pool", gpus={"l4": 1, "l4-spot": 2})
        assert pool.name == "test-pool"
        assert pool.spec.gpus == {"l4": 1, "l4-spot": 2}

    @pytest.mark.asyncio
    async def test_create_pool_invalid_profiles(self) -> None:
        """Create pool with invalid machine profile keys fails."""
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
        }
        manager = PoolManager(use_kubernetes=False, machine_profiles=profiles)

        # Invalid profile key should raise error
        with pytest.raises(InvalidMachineProfileError) as exc_info:
            await manager.create_pool(name="test-pool", gpus={"unknown-profile": 1})

        assert "unknown-profile" in exc_info.value.invalid_profiles
        assert "l4" in exc_info.value.valid_profiles

    @pytest.mark.asyncio
    async def test_create_pool_mixed_valid_invalid(self) -> None:
        """Create pool with mix of valid and invalid profiles fails."""
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
            "a100-40gb": MachineProfile(name="a100-40gb", gpu_type="nvidia-a100"),
        }
        manager = PoolManager(use_kubernetes=False, machine_profiles=profiles)

        # Mix of valid and invalid should fail
        with pytest.raises(InvalidMachineProfileError) as exc_info:
            await manager.create_pool(
                name="test-pool",
                gpus={"l4": 1, "invalid": 2, "also-invalid": 1},
            )

        assert "invalid" in exc_info.value.invalid_profiles
        assert "also-invalid" in exc_info.value.invalid_profiles
        assert "l4" not in exc_info.value.invalid_profiles

    @pytest.mark.asyncio
    async def test_create_pool_no_profiles_configured(self) -> None:
        """Create pool works without machine profiles (backward compat)."""
        # No profiles = no validation (backward compat)
        manager = PoolManager(use_kubernetes=False)

        # Should work with any GPU key
        pool = await manager.create_pool(name="test-pool", gpus={"anything": 1})
        assert pool.name == "test-pool"

    @pytest.mark.asyncio
    async def test_create_pool_empty_profiles(self) -> None:
        """Create pool with empty profiles dict skips validation."""
        manager = PoolManager(use_kubernetes=False, machine_profiles={})

        # Empty profiles = no validation
        pool = await manager.create_pool(name="test-pool", gpus={"anything": 1})
        assert pool.name == "test-pool"


class TestWorkerAssignmentLogic:
    """Test worker assignment algorithms."""

    @pytest.mark.asyncio
    async def test_worker_urls_filter(self) -> None:
        """Worker URLs filter in registry respects pool assignment."""
        from sie_router.registry import WorkerRegistry

        registry = WorkerRegistry()

        # Add workers
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4", "queue_depth": 0},
        )
        await registry.update_worker(
            "http://worker-1:8080",
            {"name": "worker-1", "ready": True, "machine_profile": "l4", "queue_depth": 0},
        )
        await registry.update_worker(
            "http://worker-2:8080",
            {"name": "worker-2", "ready": True, "machine_profile": "l4", "queue_depth": 0},
        )

        # Select from pool (only first two workers)
        pool_workers = {"http://worker-0:8080", "http://worker-1:8080"}
        worker = registry.select_worker(gpu="l4", worker_urls=pool_workers)

        assert worker is not None
        assert worker.url in pool_workers
        assert worker.url != "http://worker-2:8080"

    @pytest.mark.asyncio
    async def test_worker_urls_filter_empty(self) -> None:
        """Empty worker URLs filter returns None."""
        from sie_router.registry import WorkerRegistry

        registry = WorkerRegistry()

        # Add workers
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4", "queue_depth": 0},
        )

        # Select from empty pool
        worker = registry.select_worker(gpu="l4", worker_urls=set())
        assert worker is None

    @pytest.mark.asyncio
    async def test_worker_urls_filter_with_model_affinity(self) -> None:
        """Worker URLs filter works with model affinity."""
        from sie_router.registry import WorkerRegistry

        registry = WorkerRegistry()

        # Worker in pool with model loaded
        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4",
                "loaded_models": ["bge-m3"],
                "queue_depth": 10,
            },
        )
        # Worker in pool without model
        await registry.update_worker(
            "http://worker-1:8080",
            {"name": "worker-1", "ready": True, "machine_profile": "l4", "loaded_models": [], "queue_depth": 0},
        )
        # Worker NOT in pool with model
        await registry.update_worker(
            "http://worker-2:8080",
            {"name": "worker-2", "ready": True, "machine_profile": "l4", "loaded_models": ["bge-m3"], "queue_depth": 0},
        )

        pool_workers = {"http://worker-0:8080", "http://worker-1:8080"}

        # Should prefer worker-0 (in pool, has model)
        worker = registry.select_worker(gpu="l4", model="bge-m3", worker_urls=pool_workers)
        assert worker is not None
        assert worker.url == "http://worker-0:8080"


class TestBundleFiltering:
    """Test bundle-aware worker assignment."""

    @pytest.mark.asyncio
    async def test_assign_workers_filters_by_bundle(self) -> None:
        """assign_workers() only assigns workers matching pool's bundle."""
        manager = PoolManager(use_kubernetes=False)

        # Create pool with bundle filter
        pool = await manager.create_pool(name="test-pool", gpus={"l4": 1}, bundle="sglang")

        # Available workers: 2 default, 1 sglang
        available = [
            ("worker-0", "http://worker-0:8080", "l4", "default"),
            ("worker-1", "http://worker-1:8080", "l4", "default"),
            ("worker-2", "http://worker-2:8080", "l4", "sglang"),
        ]

        result = manager.assign_workers(pool, available)

        # Should assign only the sglang worker
        assert result is True
        assert len(pool.status.assigned_workers) == 1
        assert pool.status.assigned_workers[0].name == "worker-2"

    @pytest.mark.asyncio
    async def test_assign_workers_no_bundle_matches_all(self) -> None:
        """assign_workers() with no bundle filter considers all workers."""
        manager = PoolManager(use_kubernetes=False)

        # Create pool without bundle filter
        pool = await manager.create_pool(name="test-pool", gpus={"l4": 2})

        # Available workers with different bundles
        available = [
            ("worker-0", "http://worker-0:8080", "l4", "default"),
            ("worker-1", "http://worker-1:8080", "l4", "sglang"),
        ]

        result = manager.assign_workers(pool, available)

        # Should assign both workers regardless of bundle
        assert result is True
        assert len(pool.status.assigned_workers) == 2

    @pytest.mark.asyncio
    async def test_assign_workers_bundle_no_match_stays_pending(self) -> None:
        """Pool stays pending when no workers match its bundle."""
        manager = PoolManager(use_kubernetes=False)

        # Create pool needing sglang bundle
        pool = await manager.create_pool(name="test-pool", gpus={"l4": 1}, bundle="sglang")

        # Only default workers available
        available = [
            ("worker-0", "http://worker-0:8080", "l4", "default"),
            ("worker-1", "http://worker-1:8080", "l4", "default"),
        ]

        result = manager.assign_workers(pool, available)

        # No match - pool stays pending
        assert result is False
        assert pool.status.state == PoolState.PENDING
        assert len(pool.status.assigned_workers) == 0

    @pytest.mark.asyncio
    async def test_create_pool_stores_bundle(self) -> None:
        """create_pool() stores bundle in pool spec."""
        manager = PoolManager(use_kubernetes=False)

        pool = await manager.create_pool(name="test-pool", gpus={"l4": 1}, bundle="sglang")
        assert pool.spec.bundle == "sglang"

    @pytest.mark.asyncio
    async def test_create_pool_no_bundle_is_none(self) -> None:
        """create_pool() without bundle leaves spec.bundle as None."""
        manager = PoolManager(use_kubernetes=False)

        pool = await manager.create_pool(name="test-pool", gpus={"l4": 1})
        assert pool.spec.bundle is None


class TestMinimumWorkerCount:
    """Test minimum_worker_count pool parameter."""

    def test_pool_spec_default_minimum_worker_count(self) -> None:
        """PoolSpec defaults minimum_worker_count to 0."""
        spec = PoolSpec(name="test-pool", gpus={"l4": 1})
        assert spec.minimum_worker_count == 0

    def test_pool_spec_custom_minimum_worker_count(self) -> None:
        """PoolSpec stores custom minimum_worker_count."""
        spec = PoolSpec(name="test-pool", gpus={"l4": 1}, minimum_worker_count=2)
        assert spec.minimum_worker_count == 2

    @pytest.mark.asyncio
    async def test_create_pool_stores_minimum_worker_count(self) -> None:
        """create_pool() stores minimum_worker_count in pool spec."""
        manager = PoolManager(use_kubernetes=False)

        pool = await manager.create_pool(name="test-pool", gpus={"l4": 1}, minimum_worker_count=3)
        assert pool.spec.minimum_worker_count == 3

    @pytest.mark.asyncio
    async def test_create_pool_default_minimum_worker_count(self) -> None:
        """create_pool() without minimum_worker_count defaults to 0."""
        manager = PoolManager(use_kubernetes=False)

        pool = await manager.create_pool(name="test-pool", gpus={"l4": 1})
        assert pool.spec.minimum_worker_count == 0

    @pytest.mark.asyncio
    async def test_create_pool_with_bundle_and_minimum_worker_count(self) -> None:
        """create_pool() stores both bundle and minimum_worker_count."""
        manager = PoolManager(use_kubernetes=False)

        pool = await manager.create_pool(name="test-pool", gpus={"l4": 2}, bundle="sglang", minimum_worker_count=1)
        assert pool.spec.bundle == "sglang"
        assert pool.spec.minimum_worker_count == 1
        assert pool.spec.gpus == {"l4": 2}

    def test_configmap_roundtrip_with_minimum_worker_count(self) -> None:
        """minimum_worker_count survives ConfigMap serialization round-trip."""
        import json
        from types import SimpleNamespace

        manager = PoolManager(use_kubernetes=False)

        pool = Pool(
            spec=PoolSpec(name="warm-pool", gpus={"l4": 1}, bundle="sglang", minimum_worker_count=2),
            status=PoolStatus(state=PoolState.ACTIVE, created_at=1.0, last_renewed=2.0),
        )

        data = manager._pool_to_configmap_data(pool)
        spec_parsed = json.loads(data["spec"])
        assert spec_parsed["minimum_worker_count"] == 2
        assert spec_parsed["bundle"] == "sglang"

        cm = SimpleNamespace(metadata=SimpleNamespace(name="sie-pool-warm-pool"), data=data)
        restored = manager._configmap_to_pool(cm)
        assert restored is not None
        assert restored.spec.minimum_worker_count == 2
        assert restored.spec.bundle == "sglang"
        assert restored.spec.gpus == {"l4": 1}

    def test_configmap_roundtrip_zero_minimum_worker_count(self) -> None:
        """minimum_worker_count=0 is persisted and restored correctly."""
        import json
        from types import SimpleNamespace

        manager = PoolManager(use_kubernetes=False)

        pool = Pool(
            spec=PoolSpec(name="zero-pool", gpus={"l4": 1}, minimum_worker_count=0),
            status=PoolStatus(state=PoolState.PENDING, created_at=1.0, last_renewed=1.0),
        )

        data = manager._pool_to_configmap_data(pool)
        spec_parsed = json.loads(data["spec"])
        assert spec_parsed["minimum_worker_count"] == 0

        cm = SimpleNamespace(metadata=SimpleNamespace(name="sie-pool-zero-pool"), data=data)
        restored = manager._configmap_to_pool(cm)
        assert restored is not None
        assert restored.spec.minimum_worker_count == 0

    def test_configmap_backward_compat_missing_minimum_worker_count(self) -> None:
        """Old ConfigMaps without minimum_worker_count default to 0."""
        import json
        from types import SimpleNamespace

        manager = PoolManager(use_kubernetes=False)

        data = {
            "spec": json.dumps({"gpus": {"l4": 1}}),
            "status": json.dumps({"state": "active", "assigned_workers": [], "created_at": 1.0, "last_renewed": 2.0}),
        }
        cm = SimpleNamespace(metadata=SimpleNamespace(name="sie-pool-old-pool"), data=data)
        restored = manager._configmap_to_pool(cm)
        assert restored is not None
        assert restored.spec.minimum_worker_count == 0


class TestDefaultPool:
    """Test default pool functionality."""

    def test_default_pool_name_constant(self) -> None:
        """Default pool name constant is defined."""
        assert DEFAULT_POOL_NAME == "default"

    @pytest.mark.asyncio
    async def test_create_default_pool(self) -> None:
        """Create default pool with all machine profiles."""
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
            "l4-spot": MachineProfile(name="l4-spot", gpu_type="nvidia-l4", spot=True),
            "a100-40gb": MachineProfile(name="a100-40gb", gpu_type="nvidia-a100"),
        }
        manager = PoolManager(use_kubernetes=False, machine_profiles=profiles)

        # Create default pool
        await manager.create_default_pool()

        # Verify default pool exists
        pool = manager.get_pool(DEFAULT_POOL_NAME)
        assert pool is not None
        assert pool.name == DEFAULT_POOL_NAME

        # Verify it has all profiles with high limits
        assert pool.spec.gpus == {"l4": 999, "l4-spot": 999, "a100-40gb": 999}

    @pytest.mark.asyncio
    async def test_create_default_pool_no_profiles(self) -> None:
        """Create default pool skips when no profiles configured."""
        manager = PoolManager(use_kubernetes=False, machine_profiles={})

        # Create default pool (should be no-op)
        await manager.create_default_pool()

        # Default pool should not exist
        pool = manager.get_pool(DEFAULT_POOL_NAME)
        assert pool is None

    @pytest.mark.asyncio
    async def test_delete_default_pool_fails(self) -> None:
        """Deleting default pool raises DefaultPoolProtectedError."""
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
        }
        manager = PoolManager(use_kubernetes=False, machine_profiles=profiles)
        await manager.create_default_pool()

        # Verify default pool exists
        assert manager.get_pool(DEFAULT_POOL_NAME) is not None

        # Deleting default pool should raise error
        with pytest.raises(DefaultPoolProtectedError):
            await manager.delete_pool(DEFAULT_POOL_NAME)

        # Pool should still exist
        assert manager.get_pool(DEFAULT_POOL_NAME) is not None

    @pytest.mark.asyncio
    async def test_delete_other_pool_works(self) -> None:
        """Deleting non-default pool works normally."""
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
        }
        manager = PoolManager(use_kubernetes=False, machine_profiles=profiles)
        await manager.create_default_pool()
        await manager.create_pool("other-pool", {"l4": 1})

        # Delete other pool should work
        deleted = await manager.delete_pool("other-pool")
        assert deleted is True

        # Other pool gone, default still exists
        assert manager.get_pool("other-pool") is None
        assert manager.get_pool(DEFAULT_POOL_NAME) is not None

    @pytest.mark.asyncio
    async def test_default_pool_not_expired(self) -> None:
        """Default pool is never expired by check_expired_leases."""
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
        }
        # Use very short lease duration
        manager = PoolManager(
            use_kubernetes=False,
            machine_profiles=profiles,
            lease_duration_s=0.01,  # 10ms
        )
        await manager.create_default_pool()
        await manager.create_pool("other-pool", {"l4": 1})

        # Wait for leases to "expire"
        await asyncio.sleep(0.02)

        # Check expired leases
        expired = await manager.check_expired_leases()

        # Other pool should be expired, but not default
        assert "other-pool" in expired
        assert DEFAULT_POOL_NAME not in expired

        # Default pool still exists
        assert manager.get_pool(DEFAULT_POOL_NAME) is not None
        assert manager.get_pool("other-pool") is None

    @pytest.mark.asyncio
    async def test_default_pool_idempotent_creation(self) -> None:
        """Creating default pool multiple times is idempotent."""
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
        }
        manager = PoolManager(use_kubernetes=False, machine_profiles=profiles)

        # Create default pool twice
        await manager.create_default_pool()
        await manager.create_default_pool()

        # Should only have one default pool
        assert len([p for p in manager.pools.values() if p.name == DEFAULT_POOL_NAME]) == 1


class TestLeaseExpiry:
    """Test lease expiry timing for rolling upgrade safety."""

    def test_default_lease_duration_covers_rolling_upgrade(self) -> None:
        """Default lease duration is at least 15 min and equals 20 min."""
        assert PoolManager.DEFAULT_LEASE_DURATION_S >= 900
        assert PoolManager.DEFAULT_LEASE_DURATION_S == 1200

    @pytest.mark.asyncio
    async def test_pool_survives_15_minute_gap(self) -> None:
        """Pool is not expired when last renewed 15 minutes ago."""
        manager = PoolManager(use_kubernetes=False)
        await manager.create_pool(name="user-pool", gpus={"l4": 1})

        pool = manager.get_pool("user-pool")
        assert pool is not None

        # Simulate last renewal 15 minutes ago (within 20-min TTL)
        pool.status.last_renewed = time.time() - 900

        expired = await manager.check_expired_leases()
        assert "user-pool" not in expired
        assert manager.get_pool("user-pool") is not None

    @pytest.mark.asyncio
    async def test_pool_expires_after_ttl(self) -> None:
        """Pool is expired when last renewed more than 20 minutes ago."""
        manager = PoolManager(use_kubernetes=False)
        await manager.create_pool(name="user-pool", gpus={"l4": 1})

        pool = manager.get_pool("user-pool")
        assert pool is not None

        # Simulate last renewal 1201 seconds ago (past 20-min TTL)
        pool.status.last_renewed = time.time() - 1201

        expired = await manager.check_expired_leases()
        assert "user-pool" in expired
        assert manager.get_pool("user-pool") is None


class TestDefaultPoolRouting:
    """Test default pool routing in proxy."""

    @pytest.mark.asyncio
    async def test_delete_default_pool_returns_400(self) -> None:
        """DELETE /v1/pools/default returns 400 error."""
        from fastapi.testclient import TestClient
        from sie_router.app.app_factory import AppFactory
        from sie_router.app.app_state_config import AppStateConfig
        from sie_router.types import MachineProfile

        # Create app with pool management enabled
        config = AppStateConfig(worker_urls=[], enable_pools=True)
        app = AppFactory.create_app(config)

        # Set up machine profiles for default pool creation
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
        }

        with TestClient(app) as client:
            # Create default pool manually (since lifespan won't run with TestClient)
            pool_manager = app.state.pool_manager
            pool_manager._machine_profiles = profiles
            await pool_manager.create_default_pool()

            # Try to delete default pool
            response = client.delete("/v1/pools/default")

            assert response.status_code == 400
            assert "Cannot delete" in response.json()["detail"]["message"]

    @pytest.mark.asyncio
    async def test_default_pool_listed_in_pools(self) -> None:
        """GET /v1/pools includes default pool."""
        from fastapi.testclient import TestClient
        from sie_router.app.app_factory import AppFactory
        from sie_router.app.app_state_config import AppStateConfig
        from sie_router.types import MachineProfile

        config = AppStateConfig(worker_urls=[], enable_pools=True)
        app = AppFactory.create_app(config)
        profiles = {
            "l4": MachineProfile(name="l4", gpu_type="nvidia-l4"),
        }

        with TestClient(app) as client:
            pool_manager = app.state.pool_manager
            pool_manager._machine_profiles = profiles
            await pool_manager.create_default_pool()

            response = client.get("/v1/pools")

            assert response.status_code == 200
            pools = response.json()["pools"]
            pool_names = [p["name"] for p in pools]
            assert DEFAULT_POOL_NAME in pool_names
