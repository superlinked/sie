from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LatencyTracker:
    """Rolling percentile tracker for request latencies.

    Maintains a fixed-size window of recent latency samples and computes
    exact percentiles by sorting. A deque of 200 samples is trivial to
    sort (~1us) and gives exact results without approximation structures.
    """

    window_size: int = 200
    min_samples: int = 10
    _samples: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._samples = deque(maxlen=self.window_size)

    def record(self, total_ms: float) -> None:
        """Record a latency sample in milliseconds."""
        self._samples.append(total_ms)

    def percentile(self, p: float) -> float | None:
        """Compute the p-th percentile (0–100) of recent samples.

        Returns None if fewer than min_samples have been recorded.
        """
        n = len(self._samples)
        if n < self.min_samples:
            return None
        sorted_samples = sorted(self._samples)
        # Use nearest-rank method
        idx = int(p / 100.0 * (n - 1))
        return sorted_samples[idx]

    def p50(self) -> float | None:
        """Return the median (p50) of recent samples."""
        return self.percentile(50)

    def p90(self) -> float | None:
        return self.percentile(90)

    def p99(self) -> float | None:
        return self.percentile(99)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        """Clear all samples."""
        self._samples.clear()


@dataclass(slots=True)
class BatchEfficiencyTracker:
    """Tracks batch fill ratios to measure GPU saturation.

    Records (actual_batch_cost / max_batch_cost) for recent batches.
    A fill ratio near 1.0 means the GPU is fully saturated.
    A fill ratio near 0 means batches are flushing nearly empty.
    """

    window_size: int = 50
    _fill_ratios: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._fill_ratios = deque(maxlen=self.window_size)

    def record(self, actual_cost: int, max_cost: int) -> None:
        """Record a batch fill ratio."""
        if max_cost > 0:
            self._fill_ratios.append(actual_cost / max_cost)

    def mean_fill_ratio(self) -> float | None:
        """Return the mean fill ratio, or None if no samples."""
        if not self._fill_ratios:
            return None
        return sum(self._fill_ratios) / len(self._fill_ratios)

    @property
    def sample_count(self) -> int:
        return len(self._fill_ratios)

    def reset(self) -> None:
        self._fill_ratios.clear()


@dataclass(frozen=True, slots=True)
class AdaptiveBatchState:
    """Immutable snapshot of adaptive controller state.

    Used by WebSocket status and router health to expose controller
    internals without coupling to private fields.
    """

    enabled: bool
    calibrated: bool
    target_p50_ms: float | None
    current_wait_ms: float
    current_batch_cost: int
    observed_p50_ms: float | None
    headroom_ms: float | None
    fill_ratio: float | None
    integral: float


@dataclass(slots=True)
class AdaptiveBatchController:
    """PI controller that adjusts batch wait and batch cost to maximize GPU
    saturation while respecting a latency SLO.

    Three knobs:
    1. **max_batch_wait_ms** — how long to wait for items before flushing.
       Adjusted by a PI (proportional-integral) controller based on the gap
       between target and observed p50 latency.
    2. **max_batch_cost** — token/cost limit per batch. Adjusted by a
       proportional-only controller gated on batch fill ratio.
    3. **target_p50_ms** — the latency SLO. Either set explicitly or
       auto-calibrated from observed inference latency.

    Auto-calibration:
        When ``target_p50_ms`` is None, the controller measures inference-only
        p50 (GPU forward pass, excluding queue/batch wait) during the first N
        requests and derives the target as ``inference_p50 × calibration_multiplier``.
        This avoids a feedback loop where conservative initial settings inflate
        early latency and poison the target.

    PI controller (wait knob):
        The integral term is time-normalized (``error × dt``) so Ki is stable
        across traffic rates. Saturation-aware anti-windup prevents integral
        accumulation when the output is clamped in the direction the error
        pushes. Idle decay prevents stale error from carrying into the next
        burst after a long idle period.

    Proportional-only controller (cost knob):
        Only adjusts when ``fill_ratio >= threshold`` (GPU is saturated at
        the current limit). This prevents wasteful increases when batches
        aren't filling. No integral term needed — the gating condition
        acts as a form of conditional integration.
    """

    # Latency target — None means auto-calibrate from inference latency
    target_p50_ms: float | None = None

    # Auto-calibration parameters
    calibration_multiplier: float = 1.5
    min_target_p50_ms: float = 5.0
    max_target_p50_ms: float = 500.0

    # Wait time bounds
    min_wait_ms: float = 1.0
    max_wait_ms: float = 50.0

    # Batch cost bounds (token limit)
    min_batch_cost: int = 256
    max_batch_cost: int = 65536

    # Controller tuning
    gain: float = 0.3
    integral_gain: float = 0.05
    cost_gain: float = 0.15
    update_interval: int = 10
    fill_ratio_threshold: float = 0.7

    # Current state
    _current_wait_ms: float = 10.0
    _current_batch_cost: int = 16384
    _steps_since_update: int = 0

    # Calibration state
    _auto_calibrate: bool = False  # True if target_p50_ms was originally None
    _calibrated: bool = False
    _inference_tracker: LatencyTracker = field(default_factory=lambda: LatencyTracker(window_size=50, min_samples=10))

    # PI integral state
    _integral: float = 0.0
    _integral_max: float = 20.0
    _last_step_time: float | None = None

    def __post_init__(self) -> None:
        if self.target_p50_ms is None:
            self._auto_calibrate = True
        else:
            self._calibrated = True

    def record_inference_sample(self, inference_ms: float) -> None:
        """Record an inference-only latency sample for auto-calibration.

        Only collects samples before calibration completes. After calibration,
        this is a no-op.

        Args:
            inference_ms: GPU forward pass time from RequestTiming.inference_ms.
        """
        if not self._calibrated:
            self._inference_tracker.record(inference_ms)

    def step(
        self,
        observed_p50_ms: float | None,
        fill_ratio: float | None,
    ) -> tuple[float, int]:
        """Advance the controller and return (new_wait_ms, new_batch_cost).

        Args:
            observed_p50_ms: Current rolling p50 latency (total_ms), or None
                if not enough samples yet.
            fill_ratio: Mean batch fill ratio (0.0–1.0), or None.

        Returns:
            Tuple of (max_batch_wait_ms, max_batch_cost).
        """
        self._steps_since_update += 1
        if self._steps_since_update < self.update_interval:
            return self._current_wait_ms, self._current_batch_cost

        self._steps_since_update = 0

        # --- Auto-calibration phase ---
        if not self._calibrated:
            inference_p50 = self._inference_tracker.p50()
            if inference_p50 is not None:
                self.target_p50_ms = _clamp(
                    inference_p50 * self.calibration_multiplier,
                    self.min_target_p50_ms,
                    self.max_target_p50_ms,
                )
                self._calibrated = True
                self._integral = 0.0
                self._last_step_time = None
                logger.info(
                    "Auto-calibrated: target_p50_ms=%.1fms (inference_p50=%.1fms x %.1f, clamped to [%.1f, %.1f])",
                    self.target_p50_ms,
                    inference_p50,
                    self.calibration_multiplier,
                    self.min_target_p50_ms,
                    self.max_target_p50_ms,
                )
            # Hold knobs at initial values until calibrated
            return self._current_wait_ms, self._current_batch_cost

        if observed_p50_ms is None or self.target_p50_ms is None:
            return self._current_wait_ms, self._current_batch_cost

        target = self.target_p50_ms
        headroom_ms = target - observed_p50_ms
        headroom_frac = headroom_ms / target  # normalized (-inf, 1)

        # --- Compute dt for time-normalized integral ---
        now = time.monotonic()
        if self._last_step_time is not None:
            dt_s = min(now - self._last_step_time, 5.0)  # cap idle gap
        else:
            dt_s = 0.0  # first step after calibration, no integral contribution
        self._last_step_time = now

        # --- Idle decay: prevent stale error from carrying into next burst ---
        if dt_s > 2.0:
            decay_factor = 0.5 ** (dt_s - 2.0)
            self._integral *= decay_factor

        # --- Saturation-aware anti-windup ---
        # Only integrate when the output is NOT saturated in the direction
        # the error would push it. This prevents overshoot after recovery.
        output_at_max = self._current_wait_ms >= self.max_wait_ms
        output_at_min = self._current_wait_ms <= self.min_wait_ms

        can_integrate = True
        if output_at_max and headroom_ms > 0:
            can_integrate = False
        if output_at_min and headroom_ms < 0:
            can_integrate = False

        if can_integrate and self.integral_gain > 0:
            self._integral += headroom_ms * dt_s
            self._integral = _clamp(self._integral, -self._integral_max, self._integral_max)

        # --- Knob 1: batch wait (PI controller) ---
        wait_adjustment = headroom_ms * self.gain + self._integral * self.integral_gain
        new_wait = self._current_wait_ms + wait_adjustment
        self._current_wait_ms = _clamp(new_wait, self.min_wait_ms, self.max_wait_ms)

        # --- Knob 2: batch cost (proportional-only, gated on fill ratio) ---
        if fill_ratio is not None and fill_ratio >= self.fill_ratio_threshold:
            cost_adjustment = self._current_batch_cost * headroom_frac * self.cost_gain
            new_cost = self._current_batch_cost + cost_adjustment
            self._current_batch_cost = int(_clamp(new_cost, self.min_batch_cost, self.max_batch_cost))

        logger.debug(
            "Adaptive batch: p50=%.1fms (target=%.1f), headroom=%.1fms, integral=%.2f, fill=%.2f, wait=%.1fms, cost=%d",
            observed_p50_ms,
            target,
            headroom_ms,
            self._integral,
            fill_ratio or 0.0,
            self._current_wait_ms,
            self._current_batch_cost,
        )

        return self._current_wait_ms, self._current_batch_cost

    @property
    def current_wait_ms(self) -> float:
        return self._current_wait_ms

    @property
    def current_batch_cost(self) -> int:
        return self._current_batch_cost

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def snapshot(
        self,
        observed_p50_ms: float | None = None,
        fill_ratio: float | None = None,
    ) -> AdaptiveBatchState:
        """Return an immutable snapshot of the controller state."""
        target = self.target_p50_ms
        headroom = None
        if target is not None and observed_p50_ms is not None:
            headroom = target - observed_p50_ms
        return AdaptiveBatchState(
            enabled=True,
            calibrated=self._calibrated,
            target_p50_ms=target,
            current_wait_ms=self._current_wait_ms,
            current_batch_cost=self._current_batch_cost,
            observed_p50_ms=observed_p50_ms,
            headroom_ms=headroom,
            fill_ratio=fill_ratio,
            integral=self._integral,
        )

    def reset(self) -> None:
        """Reset the controller to its initial state."""
        self._current_wait_ms = _clamp(10.0, self.min_wait_ms, self.max_wait_ms)
        self._current_batch_cost = int(_clamp(16384, self.min_batch_cost, self.max_batch_cost))
        self._steps_since_update = 0
        self._integral = 0.0
        self._last_step_time = None
        if self._auto_calibrate:
            self._calibrated = False
            self.target_p50_ms = None
            self._inference_tracker.reset()
        # If target was explicit, _calibrated stays True


# Keep backward compat alias
BatchWaitController = AdaptiveBatchController


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
