//! Adaptive batching: a rolling latency tracker and a fetch-timeout
//! controller for the NATS pull loop.
//!
//! The pull loop starts at `min` expiry. On empty/error fetches it
//! multiplicatively backs off by `backoff_growth`, capped at `max`. On
//! non-empty fetches it nudges the expiry toward `target_p50_ms`:
//!
//! ```text
//! headroom_ms = target_p50_ms - observed_p50
//! new = current + headroom_ms * gain / 1000
//! clamp(new, min, max)
//! ```
//!
//! - headroom > 0 → under SLO → grow expiry (bigger batches, more throughput)
//! - headroom < 0 → over SLO → shrink expiry (faster pickup, lower latency)
//! - too few samples → reset to `min`.
//!
//! The tracker is fed `queue_ms + inference_ms + postprocess_ms` per
//! successfully published result.

use std::collections::VecDeque;
use std::time::Duration;

/// Rolling latency tracker with a fixed-size window. `min_samples`
/// gates percentile queries so a cold start doesn't bias the controller.
#[derive(Debug, Clone)]
pub struct LatencyTracker {
    window_size: usize,
    min_samples: usize,
    samples: VecDeque<f64>,
}

impl LatencyTracker {
    pub fn new(window_size: usize, min_samples: usize) -> Self {
        Self {
            window_size,
            min_samples,
            samples: VecDeque::with_capacity(window_size),
        }
    }

    pub fn record(&mut self, total_ms: f64) {
        if self.samples.len() == self.window_size {
            self.samples.pop_front();
        }
        self.samples.push_back(total_ms);
    }

    /// Nearest-rank percentile over the current window, or `None` if
    /// fewer than `min_samples` are buffered. Index is integer-truncated.
    pub fn percentile(&self, p: f64) -> Option<f64> {
        let n = self.samples.len();
        if n < self.min_samples {
            return None;
        }
        let mut sorted: Vec<f64> = self.samples.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((p / 100.0) * ((n - 1) as f64)) as usize;
        sorted.get(idx).copied()
    }

    pub fn p50(&self) -> Option<f64> {
        self.percentile(50.0)
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Clear the sample window. Used by the adaptive scheduler's
    /// `reset()` path (see `crate::scheduler::AdaptiveBatchController`)
    /// when the auto-calibration state is being rebuilt from scratch
    /// — stale inference-p50 samples would otherwise bias the next
    /// calibration round.
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    /// p90 helper mirroring Python's `LatencyTracker.p90()`. Kept as
    /// a convenience for scheduler dashboards so we don't have to
    /// remember the 90.0 magic number everywhere.
    pub fn p90(&self) -> Option<f64> {
        self.percentile(90.0)
    }

    /// p99 helper mirroring Python's `LatencyTracker.p99()`.
    pub fn p99(&self) -> Option<f64> {
        self.percentile(99.0)
    }
}

/// Adaptive NATS-fetch expiry controller: multiplicative backoff on
/// empty batches, latency-driven nudging on loaded ones.
#[derive(Debug, Clone)]
pub struct FetchExpiryController {
    pub min: Duration,
    pub max: Duration,
    pub target_p50_ms: f64,
    pub gain: f64,
    pub backoff_growth: f64,
}

impl FetchExpiryController {
    /// Defaults: 2ms → 15ms, `target_p50_ms=50`, `gain=0.2`, backoff ×2.
    ///
    /// Since the pull loop switched to a long-lived stream,
    /// `min`/`max` now bound the **client-side batch-coalesce
    /// quantum**, not the server-side pull expiry. The semantics for
    /// operators are the same (bigger number = bigger batches / more
    /// throughput, smaller number = lower per-request latency), only
    /// the mechanism moved client-side.
    ///
    /// ## Default rationale
    ///
    /// The 2/15/50 defaults keep client-side coalescing bounded so the
    /// sidecar does not build oversized fetch batches before the adapter
    /// batcher can apply backpressure. Retune with a fresh validation
    /// baseline, not by restoring Python parity by inspection.
    ///
    /// Env knobs (all override the defaults above):
    ///
    /// * `SIE_ADAPTIVE_MIN_QUANTUM_MS` (default 2) — floor in ms.
    /// * `SIE_ADAPTIVE_MAX_QUANTUM_MS` (default 15) — ceiling in ms.
    /// * `SIE_ADAPTIVE_TARGET_P50_MS` (default 50) — target queue-path
    ///   p50 the controller nudges toward.
    ///
    /// **Tracker input is configurable** via
    /// `SIE_PULL_QUANTUM_INCLUDE_QUEUE_MS` (default off): with the
    /// flag set the per-record sample becomes
    /// `queue_ms + inference + postprocess` (whole-path feedback)
    /// instead of `inference + postprocess` only. See
    /// `crate::pull_quantum_includes_queue_ms` for the
    /// divergence rationale.
    pub fn from_env_or_default() -> Self {
        let min_ms = std::env::var("SIE_ADAPTIVE_MIN_QUANTUM_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(2);
        let max_ms = std::env::var("SIE_ADAPTIVE_MAX_QUANTUM_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&n| n >= min_ms)
            .unwrap_or_else(|| std::cmp::max(15, min_ms));
        let target_ms = std::env::var("SIE_ADAPTIVE_TARGET_P50_MS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(50.0);
        Self {
            min: Duration::from_millis(min_ms),
            max: Duration::from_millis(max_ms),
            target_p50_ms: target_ms,
            gain: 0.2,
            backoff_growth: 2.0,
        }
    }

    /// Empty/error path: multiplicatively back off, clamped to `max`.
    pub fn backoff(&self, current: Duration) -> Duration {
        let grown = current.mul_f64(self.backoff_growth);
        clamp_dur(grown, self.min, self.max)
    }

    /// Non-empty path: nudge toward `target_p50_ms`. Returns `min` when
    /// the tracker hasn't warmed up yet.
    pub fn adjust(&self, current: Duration, tracker: &LatencyTracker) -> Duration {
        let Some(observed) = tracker.p50() else {
            return self.min;
        };
        let headroom_ms = self.target_p50_ms - observed;
        let adjustment_s = headroom_ms * self.gain / 1000.0;
        let adjustment = Duration::try_from_secs_f64(adjustment_s.abs()).unwrap_or(Duration::MAX);
        let new = if adjustment_s >= 0.0 {
            current.saturating_add(adjustment)
        } else {
            current.saturating_sub(adjustment)
        };
        clamp_dur(new, self.min, self.max)
    }
}

fn clamp_dur(d: Duration, lo: Duration, hi: Duration) -> Duration {
    if d < lo {
        lo
    } else if d > hi {
        hi
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracker_returns_none_below_min_samples() {
        let mut t = LatencyTracker::new(200, 10);
        for i in 0..9 {
            t.record(i as f64);
        }
        assert_eq!(t.p50(), None);
    }

    #[test]
    fn tracker_computes_p50_after_min_samples() {
        let mut t = LatencyTracker::new(200, 10);
        for i in 1..=10 {
            t.record(i as f64);
        }
        // n=10 → idx = int(0.5 * 9) = 4 → sorted[4] = 5.0.
        assert_eq!(t.p50(), Some(5.0));
    }

    #[test]
    fn tracker_window_drops_oldest() {
        let mut t = LatencyTracker::new(5, 3);
        for i in 1..=10 {
            t.record(i as f64);
        }
        assert_eq!(t.sample_count(), 5);
        // window=[6..10], p50 idx=int(0.5*4)=2 → sorted[2]=8.
        assert_eq!(t.p50(), Some(8.0));
    }

    #[test]
    fn backoff_doubles_and_caps() {
        let c = FetchExpiryController::from_env_or_default();
        // With the worker defaults (2/15/50), backoff doubles 2 → 4 → 8
        // and then clamps at max=15.
        assert_eq!(
            c.backoff(Duration::from_millis(2)),
            Duration::from_millis(4)
        );
        assert_eq!(
            c.backoff(Duration::from_millis(4)),
            Duration::from_millis(8)
        );
        assert_eq!(
            c.backoff(Duration::from_millis(8)),
            Duration::from_millis(15)
        );
    }

    #[test]
    fn adjust_returns_min_when_not_enough_samples() {
        let c = FetchExpiryController::from_env_or_default();
        let t = LatencyTracker::new(200, 10);
        assert_eq!(c.adjust(Duration::from_millis(20), &t), c.min);
    }

    #[test]
    fn adjust_grows_timeout_when_below_target() {
        // Worker defaults: target=50, gain=0.2. p50=10, target=50
        // gives headroom=40 → +40*0.2/1000=+8ms → 5+8=13ms.
        let c = FetchExpiryController::from_env_or_default();
        let mut t = LatencyTracker::new(200, 10);
        for _ in 0..20 {
            t.record(10.0);
        }
        assert_eq!(
            c.adjust(Duration::from_millis(5), &t),
            Duration::from_millis(13)
        );
    }

    #[test]
    fn adjust_shrinks_timeout_when_over_target() {
        // Worker defaults: target=50. p50=100, target=50 gives
        // headroom=-50 → -10ms → 15-10=5ms.
        let c = FetchExpiryController::from_env_or_default();
        let mut t = LatencyTracker::new(200, 10);
        for _ in 0..20 {
            t.record(100.0);
        }
        assert_eq!(
            c.adjust(Duration::from_millis(15), &t),
            Duration::from_millis(5)
        );
    }

    #[test]
    fn adjust_stays_within_bounds_for_extreme_targets() {
        let mut t = LatencyTracker::new(200, 10);
        for _ in 0..20 {
            t.record(10.0);
        }
        let defaults = FetchExpiryController::from_env_or_default();
        for (target_p50_ms, expected) in [
            (f64::INFINITY, defaults.max),
            (1e300, defaults.max),
            (-1e300, defaults.min),
            (f64::NAN, defaults.min),
        ] {
            let c = FetchExpiryController {
                target_p50_ms,
                ..defaults.clone()
            };
            assert_eq!(
                c.adjust(Duration::from_millis(5), &t),
                expected,
                "{target_p50_ms}"
            );
        }
    }

    #[test]
    fn adjust_clamps_to_min_on_aggressive_shrink() {
        let c = FetchExpiryController::from_env_or_default();
        let mut t = LatencyTracker::new(200, 10);
        for _ in 0..20 {
            t.record(10_000.0); // way over target
        }
        let out = c.adjust(Duration::from_millis(20), &t);
        assert_eq!(out, c.min);
    }

    #[test]
    fn defaults_match_queue_mode_scheduler_values() {
        // Lock the queue-mode defaults that bound coalescing without
        // forcing the pull loop to its minimum under normal load.
        let c = FetchExpiryController::from_env_or_default();
        assert_eq!(c.min, Duration::from_millis(2));
        assert_eq!(c.max, Duration::from_millis(15));
        assert_eq!(c.target_p50_ms, 50.0);
    }
}
