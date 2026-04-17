"""Tests for router metrics, especially scale-from-zero support."""

import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sie_router.metrics import (
    ACTIVE_LEASE_GPUS,
    DEMAND_EXPIRY_SECONDS,
    PENDING_DEMAND,
    REJECTED_REQUESTS,
    _pool_demand_time,
    _pools_with_demand,
    _update_pending_demand,
    clear_pending_demand,
    record_pending_demand,
    record_rejected_request,
    update_worker_metrics,
)
from sie_router.metrics import (
    router as metrics_router,
)
from sie_router.pools import Pool, PoolSpec, PoolState, PoolStatus
from sie_router.registry import WorkerRegistry


class TestPendingDemandMetrics:
    """Test cases for pending demand metrics used by KEDA for scale-from-zero."""

    def setup_method(self) -> None:
        """Clear state before each test."""
        _pool_demand_time.clear()
        _pools_with_demand.clear()
        # Reset all PENDING_DEMAND labels
        PENDING_DEMAND._metrics.clear()

    def test_record_pending_demand_sets_gauge(self) -> None:
        """record_pending_demand sets the gauge to pool count (1 for first pool)."""
        record_pending_demand("l4", "default", "pool-0")

        # Check gauge value - now uses machine_profile, bundle labels, value is pool count
        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 1
        assert "l4:default:pool-0" in _pool_demand_time
        assert "pool-0" in _pools_with_demand.get("l4:default", set())

    def test_multiple_pools_increment_gauge(self) -> None:
        """Multiple pools with demand increment the gauge."""
        record_pending_demand("l4", "default", "pool-0")
        record_pending_demand("l4", "default", "pool-1")
        record_pending_demand("l4", "default", "pool-2")

        # Gauge should be 3 (number of pools with demand)
        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 3
        assert len(_pools_with_demand.get("l4:default", set())) == 3

    def test_same_pool_doesnt_increment_twice(self) -> None:
        """Same pool requesting demand multiple times doesn't increment."""
        record_pending_demand("l4", "default", "pool-0")
        record_pending_demand("l4", "default", "pool-0")
        record_pending_demand("l4", "default", "pool-0")

        # Gauge should still be 1 (same pool)
        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 1

    def test_clear_pending_demand_resets_gauge(self) -> None:
        """clear_pending_demand sets the gauge to 0."""
        record_pending_demand("l4", "default", "pool-0")
        record_pending_demand("l4", "default", "pool-1")
        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 2

        clear_pending_demand("l4", "default")

        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 0
        assert "l4:default" not in _pools_with_demand
        assert "l4:default:pool-0" not in _pool_demand_time
        assert "l4:default:pool-1" not in _pool_demand_time

    def test_multiple_machine_profiles_tracked_independently(self) -> None:
        """Each machine profile has its own pending demand gauge."""
        record_pending_demand("l4", "default", "pool-0")
        record_pending_demand("a100-80gb", "default", "pool-1")

        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 1
        assert PENDING_DEMAND.labels(machine_profile="a100-80gb", bundle="default")._value._value == 1

        clear_pending_demand("l4", "default")

        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 0
        assert PENDING_DEMAND.labels(machine_profile="a100-80gb", bundle="default")._value._value == 1

    def test_multiple_bundles_tracked_independently(self) -> None:
        """Each bundle has its own pending demand gauge."""
        record_pending_demand("l4", "default", "pool-0")
        record_pending_demand("l4", "sglang", "pool-1")

        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 1
        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="sglang")._value._value == 1

        clear_pending_demand("l4", "default")

        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 0
        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="sglang")._value._value == 1

    def test_demand_decays_after_expiry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pending demand is cleared after DEMAND_EXPIRY_SECONDS."""
        # Record demand from multiple pools
        record_pending_demand("l4", "default", "pool-0")
        record_pending_demand("l4", "default", "pool-1")
        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 2

        # Simulate time passing beyond expiry for pool-0 only
        pool_key_0 = "l4:default:pool-0"
        old_time = _pool_demand_time[pool_key_0]
        _pool_demand_time[pool_key_0] = old_time - DEMAND_EXPIRY_SECONDS - 1

        # Update should clear pool-0's demand but keep pool-1
        _update_pending_demand()

        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 1
        assert pool_key_0 not in _pool_demand_time
        assert "l4:default:pool-1" in _pool_demand_time

    @pytest.mark.asyncio
    async def test_demand_cleared_when_workers_available(self) -> None:
        """Pending demand is cleared when workers come online."""
        registry = WorkerRegistry()

        # Record demand for L4 with default bundle from multiple pools
        record_pending_demand("l4", "default", "pool-0")
        record_pending_demand("l4", "default", "pool-1")
        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 2

        # Add L4 worker with default bundle (machine_profile falls back to gpu)
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4", "bundle": "default", "queue_depth": 0},
        )

        # Update metrics should clear ALL demand for this WorkerGroup
        update_worker_metrics(registry)

        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 0

    @pytest.mark.asyncio
    async def test_demand_not_cleared_for_different_profile(self) -> None:
        """Pending demand for one machine profile is not cleared by workers of another."""
        registry = WorkerRegistry()

        # Record demand for A100
        record_pending_demand("a100-80gb", "default", "pool-0")
        assert PENDING_DEMAND.labels(machine_profile="a100-80gb", bundle="default")._value._value == 1

        # Add L4 worker (different machine profile)
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4", "bundle": "default", "queue_depth": 0},
        )

        # Update metrics should NOT clear A100 demand
        update_worker_metrics(registry)

        assert PENDING_DEMAND.labels(machine_profile="a100-80gb", bundle="default")._value._value == 1

    @pytest.mark.asyncio
    async def test_demand_not_cleared_for_different_bundle(self) -> None:
        """Pending demand for one bundle is not cleared by workers of another bundle."""
        registry = WorkerRegistry()

        # Record demand for sglang bundle
        record_pending_demand("l4", "sglang", "pool-0")
        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="sglang")._value._value == 1

        # Add L4 worker with default bundle (different bundle)
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4", "bundle": "default", "queue_depth": 0},
        )

        # Update metrics should NOT clear sglang demand
        update_worker_metrics(registry)

        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="sglang")._value._value == 1

    @pytest.mark.asyncio
    async def test_unhealthy_workers_dont_clear_demand(self) -> None:
        """Unhealthy workers don't clear pending demand."""
        registry = WorkerRegistry()

        # Record demand for L4
        record_pending_demand("l4", "default", "pool-0")

        # Add L4 worker then mark unhealthy
        await registry.update_worker(
            "http://worker-0:8080",
            {"name": "worker-0", "ready": True, "machine_profile": "l4", "bundle": "default", "queue_depth": 0},
        )
        await registry.mark_unhealthy("http://worker-0:8080")

        # Update metrics should NOT clear demand (worker is unhealthy)
        update_worker_metrics(registry)

        assert PENDING_DEMAND.labels(machine_profile="l4", bundle="default")._value._value == 1

    @pytest.mark.asyncio
    async def test_worker_with_explicit_machine_profile(self) -> None:
        """Workers with explicit machine_profile use that instead of gpu."""
        registry = WorkerRegistry()

        # Record demand for l4-spot profile
        record_pending_demand("l4-spot", "default", "pool-0")
        assert PENDING_DEMAND.labels(machine_profile="l4-spot", bundle="default")._value._value == 1

        # Add worker with explicit machine_profile
        await registry.update_worker(
            "http://worker-0:8080",
            {
                "name": "worker-0",
                "ready": True,
                "machine_profile": "l4-spot",
                "bundle": "default",
                "queue_depth": 0,
            },
        )

        # Update metrics should clear demand for l4-spot
        update_worker_metrics(registry)

        assert PENDING_DEMAND.labels(machine_profile="l4-spot", bundle="default")._value._value == 0


class TestActiveLeaseGPUsMetric:
    """Test cases for active lease GPU gauge used by KEDA to prevent premature scale-down."""

    def setup_method(self) -> None:
        """Clear state before each test."""
        from sie_router.metrics import ACTIVE_LEASE_GPUS, _previous_lease_gpu_keys

        ACTIVE_LEASE_GPUS._metrics.clear()
        _previous_lease_gpu_keys.clear()

    def test_clear_stale_lease_gpus_zeroes_removed_keys(self) -> None:
        """Stale label combinations are set to 0."""
        from sie_router.metrics import ACTIVE_LEASE_GPUS, _clear_stale_lease_gpus, _previous_lease_gpu_keys

        # Simulate previous scrape had l4:default active
        _previous_lease_gpu_keys.add(("l4", "default"))
        ACTIVE_LEASE_GPUS.labels(machine_profile="l4", bundle="default").set(2)

        # New scrape has no active leases
        _clear_stale_lease_gpus({})

        assert ACTIVE_LEASE_GPUS.labels(machine_profile="l4", bundle="default")._value._value == 0

    def test_clear_stale_lease_gpus_preserves_active(self) -> None:
        """Active label combinations are NOT zeroed."""
        from sie_router.metrics import ACTIVE_LEASE_GPUS, _clear_stale_lease_gpus, _previous_lease_gpu_keys

        _previous_lease_gpu_keys.add(("l4", "default"))
        ACTIVE_LEASE_GPUS.labels(machine_profile="l4", bundle="default").set(3)

        # Same key still active
        _clear_stale_lease_gpus({("l4", "default"): 3})

        # Should NOT be zeroed — value set by caller after this call
        assert ACTIVE_LEASE_GPUS.labels(machine_profile="l4", bundle="default")._value._value == 3

    def test_clear_stale_tracks_current_keys(self) -> None:
        """_previous_lease_gpu_keys is updated to current keys after clearing."""
        import sie_router.metrics as metrics_mod
        from sie_router.metrics import _clear_stale_lease_gpus

        metrics_mod._previous_lease_gpu_keys = {("old", "bundle")}

        _clear_stale_lease_gpus({("new", "bundle"): 1})

        assert metrics_mod._previous_lease_gpu_keys == {("new", "bundle")}


class TestActiveLeaseGPUsEndpoint:
    """Integration tests for ACTIVE_LEASE_GPUS population via the /metrics endpoint.

    These tests verify the full population logic in the metrics endpoint,
    including pool iteration, default-pool exclusion, lease expiry, and
    multi-pool accumulation.
    """

    def setup_method(self) -> None:
        """Clear metric state before each test."""
        import sie_router.metrics as metrics_mod

        ACTIVE_LEASE_GPUS._metrics.clear()
        metrics_mod._previous_lease_gpu_keys.clear()
        # Also clear pending demand state to avoid interference
        _pool_demand_time.clear()
        _pools_with_demand.clear()
        PENDING_DEMAND._metrics.clear()

    def _make_app(self, pool_manager: MagicMock) -> FastAPI:
        """Create a minimal FastAPI app wired for /metrics testing."""
        app = FastAPI()
        app.include_router(metrics_router)
        app.state.registry = WorkerRegistry()
        app.state.pool_manager = pool_manager
        return app

    def _make_pool(
        self,
        name: str,
        gpus: dict[str, int],
        bundle: str | None = None,
        last_renewed: float | None = None,
    ) -> Pool:
        """Create a Pool with the given parameters."""
        if last_renewed is None:
            last_renewed = time.time()
        return Pool(
            spec=PoolSpec(name=name, gpus=gpus, bundle=bundle),
            status=PoolStatus(
                state=PoolState.ACTIVE,
                last_renewed=last_renewed,
            ),
        )

    def _make_pool_manager(
        self,
        pools: dict[str, Pool],
        lease_duration_s: float = 300.0,
    ) -> MagicMock:
        """Create a mock pool_manager with the given pools."""
        pm = MagicMock()
        pm.pools = pools
        pm._lease_duration_s = lease_duration_s
        return pm

    def test_default_pool_excluded(self) -> None:
        """Default pool (name='default') does not contribute to ACTIVE_LEASE_GPUS."""
        default_pool = self._make_pool("default", gpus={"l4": 999}, bundle="default")
        pm = self._make_pool_manager({"default": default_pool})
        app = self._make_app(pm)

        with TestClient(app) as client:
            response = client.get("/metrics")

        assert response.status_code == 200
        # The gauge should never have been set for any label combo from the default pool
        # Verify by checking that no l4:default label was populated with 999
        assert ACTIVE_LEASE_GPUS.labels(machine_profile="l4", bundle="default")._value._value == 0

    def test_active_lease_reports_gpus(self) -> None:
        """An active non-default pool populates ACTIVE_LEASE_GPUS with its GPU count."""
        pool = self._make_pool("eval-pool", gpus={"l4": 2}, bundle="default")
        pm = self._make_pool_manager({"eval-pool": pool})
        app = self._make_app(pm)

        with TestClient(app) as client:
            response = client.get("/metrics")

        assert response.status_code == 200
        assert ACTIVE_LEASE_GPUS.labels(machine_profile="l4", bundle="default")._value._value == 2

    def test_expired_lease_excluded(self) -> None:
        """A pool whose lease has expired does not contribute to ACTIVE_LEASE_GPUS."""
        lease_duration = 300.0
        # last_renewed is well beyond the lease duration in the past
        expired_time = time.time() - lease_duration - 60
        pool = self._make_pool(
            "expired-pool",
            gpus={"l4": 4},
            bundle="default",
            last_renewed=expired_time,
        )
        pm = self._make_pool_manager({"expired-pool": pool}, lease_duration_s=lease_duration)
        app = self._make_app(pm)

        with TestClient(app) as client:
            response = client.get("/metrics")

        assert response.status_code == 200
        assert ACTIVE_LEASE_GPUS.labels(machine_profile="l4", bundle="default")._value._value == 0

    def test_multiple_pools_accumulate(self) -> None:
        """Multiple active pools requesting the same GPU type accumulate in the gauge."""
        pool_a = self._make_pool("pool-a", gpus={"l4": 2}, bundle="default")
        pool_b = self._make_pool("pool-b", gpus={"l4": 3}, bundle="default")
        pm = self._make_pool_manager(
            {
                "pool-a": pool_a,
                "pool-b": pool_b,
            }
        )
        app = self._make_app(pm)

        with TestClient(app) as client:
            response = client.get("/metrics")

        assert response.status_code == 200
        assert ACTIVE_LEASE_GPUS.labels(machine_profile="l4", bundle="default")._value._value == 5


class TestRejectedRequestsMetric:
    """Test cases for rejected requests counter used by KEDA as a supplementary scaling trigger."""

    def setup_method(self) -> None:
        """Clear state before each test."""
        _pool_demand_time.clear()
        _pools_with_demand.clear()
        REJECTED_REQUESTS._metrics.clear()
        PENDING_DEMAND._metrics.clear()

    def test_record_rejected_increments_counter(self) -> None:
        """record_rejected_request increments the counter by 1."""
        record_rejected_request("l4", "default", "no_healthy_workers")

        sample = REJECTED_REQUESTS.labels(machine_profile="l4", bundle="default", reason="no_healthy_workers")
        assert sample._value.get() == 1

    def test_record_rejected_multiple_reasons_tracked_independently(self) -> None:
        """Different reasons produce independent counter series."""
        record_rejected_request("l4", "default", "no_healthy_workers")
        record_rejected_request("l4", "default", "no_healthy_workers")
        record_rejected_request("l4", "default", "worker_503")

        no_healthy = REJECTED_REQUESTS.labels(machine_profile="l4", bundle="default", reason="no_healthy_workers")
        worker_503 = REJECTED_REQUESTS.labels(machine_profile="l4", bundle="default", reason="worker_503")
        assert no_healthy._value.get() == 2
        assert worker_503._value.get() == 1

    def test_record_rejected_refreshes_pending_demand_timestamp(self) -> None:
        """record_rejected_request refreshes _pool_demand_time when pending demand exists."""
        # Set up existing pending demand
        record_pending_demand("l4", "default", "pool-0")
        old_time = _pool_demand_time["l4:default:pool-0"]

        # Manually age the timestamp
        _pool_demand_time["l4:default:pool-0"] = old_time - 60.0

        # Record rejected request — should refresh the timestamp
        record_rejected_request("l4", "default", "no_capacity", pool_name="pool-0")

        new_time = _pool_demand_time["l4:default:pool-0"]
        assert new_time > old_time - 60.0

    def test_record_rejected_no_refresh_when_no_pending_demand(self) -> None:
        """record_rejected_request does not create demand entries if none exist."""
        record_rejected_request("l4", "default", "no_healthy_workers", pool_name="pool-0")

        # No demand tracking entries should be created
        assert "l4:default" not in _pools_with_demand
        assert "l4:default:pool-0" not in _pool_demand_time

    def test_record_rejected_empty_machine_profile_uses_unknown(self) -> None:
        """Empty machine_profile maps to 'unknown' label."""
        record_rejected_request("", "default", "no_healthy_workers")

        sample = REJECTED_REQUESTS.labels(machine_profile="unknown", bundle="default", reason="no_healthy_workers")
        assert sample._value.get() == 1
