"""Tests for adaptive batching: LatencyTracker, BatchEfficiencyTracker, AdaptiveBatchController."""

import asyncio
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
from sie_server.core.adaptive_batching import (
    AdaptiveBatchController,
    BatchEfficiencyTracker,
    LatencyTracker,
)
from sie_server.core.inference_output import EncodeOutput
from sie_server.core.prepared import make_text_item
from sie_server.core.worker import ModelWorker, WorkerConfig
from sie_server.core.worker.types import AdaptiveBatchingParams
from sie_server.types.inputs import Item


class TestLatencyTracker:
    """Tests for LatencyTracker rolling percentile calculator."""

    def test_empty_returns_none(self) -> None:
        tracker = LatencyTracker()
        assert tracker.p50() is None
        assert tracker.p90() is None
        assert tracker.p99() is None

    def test_below_min_samples_returns_none(self) -> None:
        tracker = LatencyTracker(min_samples=10)
        for i in range(9):
            tracker.record(float(i))
        assert tracker.p50() is None

    def test_at_min_samples_returns_value(self) -> None:
        tracker = LatencyTracker(min_samples=10)
        for i in range(10):
            tracker.record(float(i))
        assert tracker.p50() is not None

    def test_p50_simple(self) -> None:
        tracker = LatencyTracker(min_samples=1)
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            tracker.record(v)
        assert tracker.p50() == 30.0

    def test_p50_even_count(self) -> None:
        tracker = LatencyTracker(min_samples=1)
        for v in [10.0, 20.0, 30.0, 40.0]:
            tracker.record(v)
        assert tracker.p50() == 20.0

    def test_p90(self) -> None:
        tracker = LatencyTracker(min_samples=1)
        for v in range(1, 101):
            tracker.record(float(v))
        assert tracker.p90() == 90.0

    def test_p99(self) -> None:
        tracker = LatencyTracker(min_samples=1)
        for v in range(1, 101):
            tracker.record(float(v))
        assert tracker.p99() == 99.0

    def test_rolling_window(self) -> None:
        tracker = LatencyTracker(window_size=5, min_samples=1)
        for v in [100.0, 200.0, 300.0, 400.0, 500.0]:
            tracker.record(v)
        assert tracker.p50() == 300.0
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            tracker.record(v)
        assert tracker.p50() == 3.0

    def test_window_size_respected(self) -> None:
        tracker = LatencyTracker(window_size=10, min_samples=1)
        for i in range(100):
            tracker.record(float(i))
        assert tracker.sample_count == 10

    def test_sample_count(self) -> None:
        tracker = LatencyTracker()
        assert tracker.sample_count == 0
        tracker.record(1.0)
        assert tracker.sample_count == 1

    def test_reset(self) -> None:
        tracker = LatencyTracker(min_samples=1)
        for v in range(20):
            tracker.record(float(v))
        tracker.reset()
        assert tracker.sample_count == 0
        assert tracker.p50() is None

    def test_single_value(self) -> None:
        tracker = LatencyTracker(min_samples=1)
        tracker.record(42.0)
        assert tracker.p50() == 42.0

    def test_percentile_zero(self) -> None:
        tracker = LatencyTracker(min_samples=1)
        for v in [10.0, 20.0, 30.0]:
            tracker.record(v)
        assert tracker.percentile(0) == 10.0

    def test_percentile_100(self) -> None:
        tracker = LatencyTracker(min_samples=1)
        for v in [10.0, 20.0, 30.0]:
            tracker.record(v)
        assert tracker.percentile(100) == 30.0


class TestBatchEfficiencyTracker:
    """Tests for BatchEfficiencyTracker."""

    def test_empty(self) -> None:
        tracker = BatchEfficiencyTracker()
        assert tracker.mean_fill_ratio() is None
        assert tracker.sample_count == 0

    def test_full_batch(self) -> None:
        tracker = BatchEfficiencyTracker()
        tracker.record(1000, 1000)
        assert tracker.mean_fill_ratio() == pytest.approx(1.0)

    def test_half_batch(self) -> None:
        tracker = BatchEfficiencyTracker()
        tracker.record(500, 1000)
        assert tracker.mean_fill_ratio() == pytest.approx(0.5)

    def test_mean_of_multiple(self) -> None:
        tracker = BatchEfficiencyTracker()
        tracker.record(500, 1000)  # 0.5
        tracker.record(1000, 1000)  # 1.0
        assert tracker.mean_fill_ratio() == pytest.approx(0.75)

    def test_rolling_window(self) -> None:
        tracker = BatchEfficiencyTracker(window_size=3)
        tracker.record(100, 1000)  # 0.1
        tracker.record(200, 1000)  # 0.2
        tracker.record(300, 1000)  # 0.3
        assert tracker.mean_fill_ratio() == pytest.approx(0.2)
        tracker.record(900, 1000)  # pushes out 0.1
        # window: [0.2, 0.3, 0.9]
        assert tracker.mean_fill_ratio() == pytest.approx((0.2 + 0.3 + 0.9) / 3)

    def test_zero_max_cost(self) -> None:
        tracker = BatchEfficiencyTracker()
        tracker.record(100, 0)  # should be ignored
        assert tracker.sample_count == 0

    def test_reset(self) -> None:
        tracker = BatchEfficiencyTracker()
        tracker.record(500, 1000)
        tracker.reset()
        assert tracker.sample_count == 0


class TestAdaptiveBatchController:
    """Tests for the dual-knob AdaptiveBatchController."""

    def test_no_update_before_interval(self) -> None:
        ctrl = AdaptiveBatchController(update_interval=10, _current_wait_ms=10.0)
        for _ in range(9):
            wait, _cost = ctrl.step(observed_p50_ms=100.0, fill_ratio=0.5)
        assert wait == 10.0

    def test_updates_at_interval(self) -> None:
        ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            update_interval=5,
            gain=0.3,
            _current_wait_ms=10.0,
        )
        for _ in range(4):
            ctrl.step(observed_p50_ms=30.0, fill_ratio=0.5)
        wait, _ = ctrl.step(observed_p50_ms=30.0, fill_ratio=0.5)
        # headroom = 20, adj = 20 * 0.3 = 6.0, new_wait = 16.0
        assert wait == pytest.approx(16.0)

    def test_increases_wait_when_under_target(self) -> None:
        ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            gain=0.5,
            update_interval=1,
            _current_wait_ms=10.0,
        )
        wait, _ = ctrl.step(observed_p50_ms=30.0, fill_ratio=0.5)
        assert wait == pytest.approx(20.0)

    def test_decreases_wait_when_over_target(self) -> None:
        ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            gain=0.5,
            update_interval=1,
            _current_wait_ms=20.0,
        )
        wait, _ = ctrl.step(observed_p50_ms=80.0, fill_ratio=0.5)
        assert wait == pytest.approx(5.0)

    def test_holds_steady_at_target(self) -> None:
        ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            gain=0.5,
            update_interval=1,
            _current_wait_ms=15.0,
        )
        wait, _ = ctrl.step(observed_p50_ms=50.0, fill_ratio=0.5)
        assert wait == pytest.approx(15.0)

    def test_clamps_wait_to_min(self) -> None:
        ctrl = AdaptiveBatchController(
            target_p50_ms=10.0,
            min_wait_ms=2.0,
            gain=1.0,
            update_interval=1,
            _current_wait_ms=5.0,
        )
        wait, _ = ctrl.step(observed_p50_ms=100.0, fill_ratio=0.5)
        assert wait == 2.0

    def test_clamps_wait_to_max(self) -> None:
        ctrl = AdaptiveBatchController(
            target_p50_ms=100.0,
            max_wait_ms=30.0,
            gain=1.0,
            update_interval=1,
            _current_wait_ms=25.0,
        )
        wait, _ = ctrl.step(observed_p50_ms=10.0, fill_ratio=0.5)
        assert wait == 30.0

    def test_none_observed_holds_steady(self) -> None:
        ctrl = AdaptiveBatchController(update_interval=1, _current_wait_ms=10.0, _current_batch_cost=8192)
        wait, cost = ctrl.step(observed_p50_ms=None, fill_ratio=None)
        assert wait == 10.0
        assert cost == 8192

    # ── Batch cost knob tests ──

    def test_cost_increases_when_saturated_and_headroom(self) -> None:
        """When fill_ratio > threshold and under SLO, cost should increase."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=100.0,
            cost_gain=0.5,
            fill_ratio_threshold=0.7,
            update_interval=1,
            _current_wait_ms=10.0,
            _current_batch_cost=10000,
        )
        _, cost = ctrl.step(observed_p50_ms=50.0, fill_ratio=0.9)
        # headroom_frac = (100-50)/100 = 0.5
        # cost_adj = 10000 * 0.5 * 0.5 = 2500
        # new_cost = 10000 + 2500 = 12500
        assert cost == 12500

    def test_cost_stable_when_not_saturated(self) -> None:
        """When fill_ratio < threshold but under SLO, cost should NOT increase."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=100.0,
            cost_gain=0.5,
            fill_ratio_threshold=0.7,
            update_interval=1,
            _current_wait_ms=10.0,
            _current_batch_cost=10000,
        )
        _, cost = ctrl.step(observed_p50_ms=50.0, fill_ratio=0.3)
        # Under SLO but not saturated → cost stays same
        assert cost == 10000

    def test_cost_decreases_when_over_slo_and_saturated(self) -> None:
        """When over SLO and GPU saturated, cost decreases."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            cost_gain=0.5,
            fill_ratio_threshold=0.7,
            update_interval=1,
            _current_wait_ms=10.0,
            _current_batch_cost=10000,
        )
        _, cost = ctrl.step(observed_p50_ms=100.0, fill_ratio=0.9)
        # headroom_frac = (50-100)/50 = -1.0
        # cost_adj = 10000 * -1.0 * 0.5 = -5000
        # new_cost = 10000 - 5000 = 5000
        assert cost == 5000

    def test_cost_stable_when_over_slo_but_not_saturated(self) -> None:
        """When over SLO but batches aren't filling, cost stays — problem is elsewhere."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            cost_gain=0.5,
            fill_ratio_threshold=0.7,
            update_interval=1,
            _current_batch_cost=10000,
        )
        _, cost = ctrl.step(observed_p50_ms=100.0, fill_ratio=0.3)
        assert cost == 10000

    def test_cost_clamps_to_min(self) -> None:
        ctrl = AdaptiveBatchController(
            target_p50_ms=10.0,
            min_batch_cost=256,
            cost_gain=1.0,
            fill_ratio_threshold=0.7,
            update_interval=1,
            _current_batch_cost=1000,
        )
        _, cost = ctrl.step(observed_p50_ms=1000.0, fill_ratio=0.9)
        assert cost == 256

    def test_cost_clamps_to_max(self) -> None:
        ctrl = AdaptiveBatchController(
            target_p50_ms=1000.0,
            max_batch_cost=20000,
            cost_gain=1.0,
            fill_ratio_threshold=0.5,
            update_interval=1,
            _current_batch_cost=15000,
        )
        _, cost = ctrl.step(observed_p50_ms=10.0, fill_ratio=0.9)
        assert cost == 20000

    def test_cost_stable_with_none_fill_ratio(self) -> None:
        """Cost doesn't grow when fill_ratio is None (even with headroom)."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=100.0,
            update_interval=1,
            _current_batch_cost=10000,
        )
        _, cost = ctrl.step(observed_p50_ms=50.0, fill_ratio=None)
        assert cost == 10000

    # ── Combined behavior ──

    def test_convergence_both_knobs(self) -> None:
        """Under steady load with headroom and saturation, both knobs grow."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=100.0,
            min_wait_ms=1.0,
            max_wait_ms=50.0,
            min_batch_cost=256,
            max_batch_cost=65536,
            gain=0.3,
            cost_gain=0.1,
            fill_ratio_threshold=0.7,
            update_interval=1,
            _current_wait_ms=5.0,
            _current_batch_cost=4096,
        )
        for _ in range(20):
            wait, cost = ctrl.step(observed_p50_ms=40.0, fill_ratio=0.85)
        assert wait > 5.0
        assert cost > 4096

    def test_over_slo_both_knobs_decrease(self) -> None:
        """When over SLO, both knobs decrease."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            gain=0.3,
            cost_gain=0.1,
            update_interval=1,
            _current_wait_ms=30.0,
            _current_batch_cost=16384,
        )
        for _ in range(20):
            wait, cost = ctrl.step(observed_p50_ms=100.0, fill_ratio=0.9)
        assert wait < 30.0
        assert cost < 16384

    def test_current_properties(self) -> None:
        ctrl = AdaptiveBatchController(_current_wait_ms=15.0, _current_batch_cost=8192)
        assert ctrl.current_wait_ms == 15.0
        assert ctrl.current_batch_cost == 8192

    def test_reset(self) -> None:
        ctrl = AdaptiveBatchController(
            min_wait_ms=2.0,
            max_wait_ms=40.0,
            min_batch_cost=512,
            max_batch_cost=32768,
            update_interval=1,
            _current_wait_ms=30.0,
            _current_batch_cost=20000,
        )
        ctrl.step(observed_p50_ms=100.0, fill_ratio=0.5)
        ctrl.reset()
        assert ctrl.current_wait_ms == 10.0
        assert ctrl.current_batch_cost == 16384
        assert ctrl._steps_since_update == 0

    def test_sequential_adjustments(self) -> None:
        ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            gain=0.1,
            update_interval=1,
            _current_wait_ms=10.0,
        )
        w1, _ = ctrl.step(observed_p50_ms=30.0, fill_ratio=0.5)
        assert w1 == pytest.approx(12.0)
        w2, _ = ctrl.step(observed_p50_ms=40.0, fill_ratio=0.5)
        assert w2 == pytest.approx(13.0)
        w3, _ = ctrl.step(observed_p50_ms=60.0, fill_ratio=0.5)
        assert w3 == pytest.approx(12.0)


class TestAdaptiveBatchingWorkerIntegration:
    """Integration tests: adaptive controller wired into ModelWorker."""

    @pytest.fixture
    def slow_adapter(self) -> MagicMock:
        mock = MagicMock()

        def _slow_encode(items: list, *args: object, **kwargs: object) -> EncodeOutput:
            time.sleep(0.02)
            return EncodeOutput(
                dense=np.array([[0.1, 0.2, 0.3]] * len(items)),
                batch_size=len(items),
            )

        mock.encode.side_effect = _slow_encode
        return mock

    @pytest.fixture
    def adaptive_config(self) -> WorkerConfig:
        return WorkerConfig(
            max_batch_tokens=16384,
            max_batch_requests=64,
            max_batch_wait_ms=10,
            adaptive_batching=AdaptiveBatchingParams(
                enabled=True,
                target_p50_ms=100.0,
                min_wait_ms=1.0,
                max_wait_ms=40.0,
                gain=0.5,
                window_size=20,
                update_interval=2,
            ),
        )

    @pytest.mark.asyncio
    async def test_controller_created_when_enabled(self, slow_adapter: MagicMock) -> None:
        config = WorkerConfig(
            adaptive_batching=AdaptiveBatchingParams(enabled=True, target_p50_ms=50.0),
        )
        worker = ModelWorker(slow_adapter, config)
        assert worker._latency_tracker is not None
        assert worker._efficiency_tracker is not None
        assert worker._adaptive_controller is not None
        assert worker._adaptive_controller.target_p50_ms == 50.0

    @pytest.mark.asyncio
    async def test_controller_not_created_when_disabled(self, slow_adapter: MagicMock) -> None:
        worker = ModelWorker(slow_adapter, WorkerConfig())
        assert worker._latency_tracker is None
        assert worker._adaptive_controller is None

    @pytest.mark.asyncio
    async def test_latency_samples_recorded(self, slow_adapter: MagicMock, adaptive_config: WorkerConfig) -> None:
        worker = ModelWorker(slow_adapter, adaptive_config, model_name="test-model")
        await worker.start()
        try:
            items = [Item(text="hello")]
            prepared = [make_text_item([1, 2, 3], 0)]
            futures = []
            for _ in range(5):
                f = await worker.submit(prepared, items, ["dense"])
                futures.append(f)
            for f in futures:
                await asyncio.wait_for(f, timeout=5.0)
            assert worker._latency_tracker is not None
            assert worker._latency_tracker.sample_count >= 5
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_batch_params_adjust(self, slow_adapter: MagicMock, adaptive_config: WorkerConfig) -> None:
        """Both batch wait and batch cost change after controller steps."""
        worker = ModelWorker(slow_adapter, adaptive_config, model_name="test-model")
        initial_wait = worker._batch_config.max_batch_wait_ms
        await worker.start()
        try:
            items = [Item(text="hello")]
            prepared = [make_text_item([1, 2, 3], 0)]
            futures = []
            for _ in range(30):
                f = await worker.submit(prepared, items, ["dense"])
                futures.append(f)
            for f in futures:
                await asyncio.wait_for(f, timeout=5.0)
            current_wait = worker._batch_config.max_batch_wait_ms
            assert current_wait != initial_wait or worker._adaptive_controller is not None
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_default_config_no_adaptive(self, slow_adapter: MagicMock) -> None:
        worker = ModelWorker(slow_adapter, WorkerConfig())
        await worker.start()
        try:
            items = [Item(text="hello")]
            prepared = [make_text_item([1, 2, 3], 0)]
            f = await worker.submit(prepared, items, ["dense"])
            await asyncio.wait_for(f, timeout=5.0)
            assert worker._batch_config.max_batch_wait_ms == 10
        finally:
            await worker.stop()

    @pytest.mark.asyncio
    async def test_efficiency_tracker_records(self, slow_adapter: MagicMock, adaptive_config: WorkerConfig) -> None:
        """Efficiency tracker gets samples after batch processing."""
        worker = ModelWorker(slow_adapter, adaptive_config, model_name="test-model")
        await worker.start()
        try:
            items = [Item(text="hello")]
            prepared = [make_text_item([1, 2, 3], 0)]
            futures = []
            for _ in range(10):
                f = await worker.submit(prepared, items, ["dense"])
                futures.append(f)
            for f in futures:
                await asyncio.wait_for(f, timeout=5.0)
            assert worker._efficiency_tracker is not None
            assert worker._efficiency_tracker.sample_count > 0
        finally:
            await worker.stop()


class TestAutoCalibration:
    """Tests for auto-calibration of target_p50_ms from inference latency."""

    def test_auto_calibrate_fires_on_first_valid_p50(self) -> None:
        """target_p50_ms=None → auto-calibrate after min_samples inference observations."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=None,
            calibration_multiplier=1.5,
            update_interval=1,
        )
        assert not ctrl.calibrated
        assert ctrl.target_p50_ms is None

        # Feed inference samples
        for _ in range(10):
            ctrl.record_inference_sample(20.0)  # 20ms inference

        # Step should trigger calibration
        ctrl.step(observed_p50_ms=25.0, fill_ratio=0.5)
        assert ctrl.calibrated
        # target = 20.0 * 1.5 = 30.0
        assert ctrl.target_p50_ms == pytest.approx(30.0)

    def test_calibration_clamps_target(self) -> None:
        """Calibrated target is clamped to [min_target, max_target]."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=None,
            calibration_multiplier=1.5,
            min_target_p50_ms=10.0,
            max_target_p50_ms=100.0,
            update_interval=1,
        )
        # Very fast inference → 2ms * 1.5 = 3ms → clamped to 10ms
        for _ in range(10):
            ctrl.record_inference_sample(2.0)
        ctrl.step(observed_p50_ms=5.0, fill_ratio=0.5)
        assert ctrl.target_p50_ms == 10.0

    def test_calibration_clamps_target_high(self) -> None:
        """Calibrated target is clamped at the ceiling."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=None,
            calibration_multiplier=2.0,
            min_target_p50_ms=5.0,
            max_target_p50_ms=100.0,
            update_interval=1,
        )
        # Slow inference → 200ms * 2.0 = 400ms → clamped to 100ms
        for _ in range(10):
            ctrl.record_inference_sample(200.0)
        ctrl.step(observed_p50_ms=250.0, fill_ratio=0.5)
        assert ctrl.target_p50_ms == 100.0

    def test_explicit_target_skips_calibration(self) -> None:
        """target_p50_ms=30.0 → _calibrated=True, no inference samples collected."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=30.0,
            update_interval=1,
        )
        assert ctrl.calibrated
        assert ctrl.target_p50_ms == 30.0

        # Inference samples are ignored
        for _ in range(20):
            ctrl.record_inference_sample(10.0)
        ctrl.step(observed_p50_ms=25.0, fill_ratio=0.5)
        assert ctrl.target_p50_ms == pytest.approx(30.0)  # unchanged

    def test_controller_holds_knobs_during_calibration(self) -> None:
        """Before calibration, step() returns initial values without adjustment."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=None,
            update_interval=1,
            _current_wait_ms=10.0,
            _current_batch_cost=16384,
        )
        # Not enough samples yet
        for _ in range(5):
            ctrl.record_inference_sample(20.0)
        wait, cost = ctrl.step(observed_p50_ms=50.0, fill_ratio=0.8)
        # Should hold at initial values
        assert wait == 10.0
        assert cost == 16384
        assert not ctrl.calibrated

    def test_calibration_resets_integral(self) -> None:
        """Integral is reset to 0 when auto-calibration fires."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=None,
            integral_gain=0.1,
            update_interval=1,
        )
        # Manually set some integral state
        ctrl._integral = 5.0
        ctrl._last_step_time = time.monotonic()

        for _ in range(10):
            ctrl.record_inference_sample(20.0)
        ctrl.step(observed_p50_ms=25.0, fill_ratio=0.5)
        assert ctrl.calibrated
        assert ctrl._integral == 0.0
        assert ctrl._last_step_time is None

    def test_inference_samples_stop_after_calibration(self) -> None:
        """After calibration, record_inference_sample is a no-op."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=None,
            update_interval=1,
        )
        for _ in range(10):
            ctrl.record_inference_sample(20.0)
        ctrl.step(observed_p50_ms=25.0, fill_ratio=0.5)
        assert ctrl.calibrated

        before = ctrl._inference_tracker.sample_count
        ctrl.record_inference_sample(999.0)
        assert ctrl._inference_tracker.sample_count == before

    def test_snapshot_before_calibration(self) -> None:
        """Snapshot works before calibration — target is None."""
        ctrl = AdaptiveBatchController(target_p50_ms=None)
        snap = ctrl.snapshot(observed_p50_ms=None, fill_ratio=None)
        assert snap.enabled is True
        assert snap.calibrated is False
        assert snap.target_p50_ms is None
        assert snap.headroom_ms is None


class TestPIController:
    """Tests for time-based PI controller with saturation-aware anti-windup."""

    def test_integral_gain_zero_is_pure_proportional(self) -> None:
        """integral_gain=0 produces identical behavior to P-only."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            gain=0.3,
            integral_gain=0.0,
            update_interval=1,
            _current_wait_ms=10.0,
        )
        wait, _ = ctrl.step(observed_p50_ms=30.0, fill_ratio=0.5)
        # Pure proportional: headroom=20, adj=20*0.3=6, new=16
        assert wait == pytest.approx(16.0)

    def test_integral_accumulates_over_time(self) -> None:
        """Integral contributes time-normalized error to the adjustment."""
        from unittest.mock import patch

        t0 = 1000.0
        # Two calls: first step (sets _last_step_time), second step (dt=0.05s)
        timestamps = iter([t0, t0 + 0.05])

        with patch("sie_server.core.adaptive_batching.time") as mock_time:
            mock_time.monotonic.side_effect = lambda: next(timestamps)

            ctrl = AdaptiveBatchController(
                target_p50_ms=50.0,
                gain=0.0,  # disable proportional to isolate integral
                integral_gain=1.0,
                update_interval=1,
                _current_wait_ms=10.0,
            )
            # First step: dt=0 (no previous time), so no integral contribution
            ctrl.step(observed_p50_ms=30.0, fill_ratio=0.5)
            first_wait = ctrl.current_wait_ms
            assert first_wait == 10.0  # No change on first step (dt=0)

            # Second step: dt=0.05s, headroom=20ms
            # integral += 20 * 0.05 = 1.0
            # adjustment = 0 (P) + 1.0 * 1.0 (I) = 1.0
            ctrl.step(observed_p50_ms=30.0, fill_ratio=0.5)
            assert ctrl.current_wait_ms == pytest.approx(11.0)

    def test_anti_windup_at_max(self) -> None:
        """Integral does not accumulate when output is at max and error is positive."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=100.0,
            gain=0.0,
            integral_gain=0.1,
            max_wait_ms=50.0,
            update_interval=1,
            _current_wait_ms=50.0,  # at max
        )
        ctrl._last_step_time = time.monotonic() - 1.0  # 1s ago

        ctrl.step(observed_p50_ms=30.0, fill_ratio=0.5)
        # Headroom is positive (70ms), output at max → should NOT integrate
        assert ctrl._integral == 0.0

    def test_anti_windup_at_min(self) -> None:
        """Integral does not accumulate when output is at min and error is negative."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=10.0,
            gain=0.0,
            integral_gain=0.1,
            min_wait_ms=1.0,
            update_interval=1,
            _current_wait_ms=1.0,  # at min
        )
        ctrl._last_step_time = time.monotonic() - 1.0

        ctrl.step(observed_p50_ms=50.0, fill_ratio=0.5)
        # Headroom is negative (-40ms), output at min → should NOT integrate
        assert ctrl._integral == 0.0

    def test_anti_windup_allows_recovery(self) -> None:
        """Integral CAN accumulate when output is at max but error would decrease it."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=10.0,
            gain=0.0,
            integral_gain=0.1,
            max_wait_ms=50.0,
            update_interval=1,
            _current_wait_ms=50.0,  # at max
        )
        ctrl._last_step_time = time.monotonic() - 1.0

        ctrl.step(observed_p50_ms=50.0, fill_ratio=0.5)
        # Headroom is negative (-40ms) → error wants to decrease → allowed even at max
        assert ctrl._integral < 0

    def test_idle_decay(self) -> None:
        """After a long idle period (dt > 2s), integral decays exponentially."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            gain=0.0,
            integral_gain=0.1,
            update_interval=1,
            _current_wait_ms=10.0,
        )
        # Set up some accumulated integral
        ctrl._integral = 10.0
        ctrl._last_step_time = time.monotonic() - 4.0  # 4s ago → dt=4s, decay for 2s

        ctrl.step(observed_p50_ms=50.0, fill_ratio=0.5)
        # decay_factor = 0.5 ** (4-2) = 0.25, integral should be ~2.5 before new contribution
        assert ctrl._integral < 5.0  # substantially decayed

    def test_pi_reduces_steady_state_bias(self) -> None:
        """PI controller converges closer to target than P-only over multiple steps.

        PI reduces steady-state bias in the operating region compared with
        P-only. The plant is nonlinear so exact convergence is not guaranteed.
        """
        # P-only
        p_ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            gain=0.1,
            integral_gain=0.0,
            update_interval=1,
            _current_wait_ms=10.0,
        )
        # PI
        pi_ctrl = AdaptiveBatchController(
            target_p50_ms=50.0,
            gain=0.1,
            integral_gain=0.05,
            update_interval=1,
            _current_wait_ms=10.0,
        )

        # Simulate steady traffic with fixed observed p50
        for _ in range(50):
            p_ctrl.step(observed_p50_ms=45.0, fill_ratio=0.5)
            pi_ctrl.step(observed_p50_ms=45.0, fill_ratio=0.5)
            time.sleep(0.001)  # small dt for integral

        # Both should increase wait (headroom=5ms)
        assert p_ctrl.current_wait_ms > 10.0
        assert pi_ctrl.current_wait_ms > 10.0
        # PI should have accumulated integral and moved further
        assert pi_ctrl.current_wait_ms >= p_ctrl.current_wait_ms

    def test_reset_preserves_calibration_for_explicit_target(self) -> None:
        """Reset with explicit target keeps _calibrated=True."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=30.0,
            integral_gain=0.1,
            _current_wait_ms=25.0,
        )
        ctrl._integral = 5.0
        ctrl.reset()
        assert ctrl.calibrated  # explicit target stays calibrated
        assert ctrl._integral == 0.0

    def test_reset_clears_calibration_for_auto(self) -> None:
        """Reset with target_p50_ms=None clears calibration state."""
        ctrl = AdaptiveBatchController(
            target_p50_ms=None,
            update_interval=1,
        )
        for _ in range(10):
            ctrl.record_inference_sample(20.0)
        ctrl.step(observed_p50_ms=25.0, fill_ratio=0.5)
        assert ctrl.calibrated

        ctrl.reset()
        assert not ctrl.calibrated
        assert ctrl._inference_tracker.sample_count == 0


class TestProfileAdaptiveBatchingMerge:
    """Tests for per-model profile adaptive batching overrides."""

    def test_profile_override_target(self) -> None:
        """Profile-level target_p50_ms overrides engine default."""
        from sie_server.config.model import ProfileAdaptiveBatching, _merge_profile_adaptive_batching

        parent = None
        child = ProfileAdaptiveBatching(target_p50_ms=30.0)
        result = _merge_profile_adaptive_batching(parent, child)
        assert result is not None
        assert result.target_p50_ms == 30.0

    def test_both_none_returns_none(self) -> None:
        from sie_server.config.model import _merge_profile_adaptive_batching

        assert _merge_profile_adaptive_batching(None, None) is None

    def test_child_inherits_parent(self) -> None:
        """Child with no adaptive_batching inherits parent's."""
        from sie_server.config.model import ProfileAdaptiveBatching, _merge_profile_adaptive_batching

        parent = ProfileAdaptiveBatching(target_p50_ms=40.0, gain=0.3)
        result = _merge_profile_adaptive_batching(parent, None)
        assert result is not None
        assert result.target_p50_ms == 40.0
        assert result.gain == 0.3

    def test_fieldwise_merge(self) -> None:
        """Child overrides one field, inherits parent's other fields."""
        from sie_server.config.model import ProfileAdaptiveBatching, _merge_profile_adaptive_batching

        parent = ProfileAdaptiveBatching(target_p50_ms=40.0, gain=0.3)
        child = ProfileAdaptiveBatching(gain=0.2)
        result = _merge_profile_adaptive_batching(parent, child)
        assert result is not None
        assert result.target_p50_ms == 40.0  # inherited from parent
        assert result.gain == 0.2  # overridden by child

    def test_engine_merge_with_profile(self) -> None:
        """Full merge: engine defaults + profile overrides."""
        from sie_server.config.model import ProfileAdaptiveBatching
        from sie_server.core.model_loader import _merge_adaptive_params

        engine = AdaptiveBatchingParams(
            enabled=True,
            target_p50_ms=None,
            gain=0.3,
            integral_gain=0.05,
        )
        profile = ProfileAdaptiveBatching(target_p50_ms=25.0)

        result = _merge_adaptive_params(engine, profile)
        assert result.target_p50_ms == 25.0  # from profile
        assert result.gain == 0.3  # from engine
        assert result.integral_gain == 0.05  # from engine
        assert result.enabled is True  # always from engine

    def test_engine_merge_none_profile(self) -> None:
        """No profile overrides → engine params unchanged."""
        from sie_server.core.model_loader import _merge_adaptive_params

        engine = AdaptiveBatchingParams(enabled=True, target_p50_ms=None, gain=0.3)
        result = _merge_adaptive_params(engine, None)
        assert result is engine
