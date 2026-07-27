//! [`AdaptiveBatchController`] — PI loop on `max_batch_wait_ms`,
//! proportional controller on `max_batch_cost`, auto-calibrated
//! `target_p50_ms`, and a starvation-recovery escape hatch.
//!
//! Ported from `sie_server/core/adaptive_batching.py` with **bit-exact
//! PI math**: both sides run f64 on the same formulas, so a given
//! input trace produces the same output trace ULP-for-ULP. The
//! validation harness exploits this property to diff Python vs Rust from
//! captured production traces.
//!
//! The controller is a pure synchronous state machine — it does not
//! own any async primitives and is safe to call under a mutex. Wall
//! clock reads go through [`std::time::Instant::now`]; tests pause
//! the clock via [`tokio::time::pause`] where they need determinism,
//! but the controller itself has no Tokio dependency.

use std::time::Instant;

use super::batch_config::BatchConfig;
use crate::latency::LatencyTracker;

/// Production floor for `min_wait_ms`. Python's config dataclass defaults
/// lower, but `ModelWorker.__init__` raises the runtime floor before
/// constructing the controller.
///
/// The wait floor and `PRODUCTION_MIN_COST_DIVISOR` jointly enforce a
/// "no sustained single-item waves" invariant under saturation: if both
/// floors collapse, the negative-headroom branch can keep shrinking batches
/// while latency rises, preventing recovery.
const PRODUCTION_MIN_WAIT_MS: f64 = 15.0;

/// Python-parity production multiplier for the `cost_gain` field —
/// `ModelWorker.__init__` sets `cost_gain = ab.gain * 0.5` so
/// the cost knob is half as aggressive as the wait knob. The
/// dataclass default 0.15 happens to coincide with `0.3 * 0.5` for
/// the default `gain`, but coupling them explicitly keeps parity
/// when an operator overrides `gain`.
const PRODUCTION_COST_GAIN_RATIO: f64 = 0.5;

/// Python-parity production multiplier for `max_batch_cost`'s ceiling.
/// `ModelWorker.__init__` sets `max_batch_cost = max(cost_floor,
/// max_batch_tokens * 4)` — gives the cost knob room to grow above
/// the model's nominal `max_batch_tokens` when saturated.
const PRODUCTION_MAX_COST_MULT: u64 = 4;

/// Production divisor for `min_batch_cost`. Mirrors
/// `ModelWorker.__init__`'s `max(256, max_batch_tokens // 4)` — the
/// cost knob never collapses below ~25 % of the model's nominal budget.
///
/// With `max_batch_tokens = 16384`, the floor is 4096 tokens —
/// roughly 10 items per forward at gte's ~400 tokens/item.
///
/// Why this matters: when the cost knob is allowed to fall to the bare
/// 256-token dataclass floor, the PI loop can form one-item waves under
/// negative headroom; `observed_p50` rises further and the loop diverges
/// instead of recovering.
///
/// Together with `PRODUCTION_MIN_WAIT_MS`, this floor is a
/// *stability feature*, not a tunable performance knob.
const PRODUCTION_MIN_COST_DIVISOR: u64 = 4;

/// Default latency window (inference-only tracker used for
/// auto-calibration). Matches Python's
/// `LatencyTracker(window_size=50, min_samples=10)`.
const INFERENCE_TRACKER_WINDOW: usize = 50;
const INFERENCE_TRACKER_MIN_SAMPLES: usize = 10;

/// Cap on `dt` between controller steps — prevents a huge idle gap
/// from dumping an enormous integral contribution into the next step.
/// Matches Python's `min(now - _last_step_time, 5.0)`.
const DT_CAP_SECS: f64 = 5.0;

/// Dispatch between the production wall-clock path and the
/// shadow-trace replay path in [`AdaptiveBatchController::step_impl`].
/// See [`AdaptiveBatchController::step_replay`] for why the replay
/// path is built on a separate f64 clock instead of [`Instant`].
#[derive(Debug, Clone, Copy)]
enum TimeSource {
    Wall,
    Replay,
}

/// When a step's `dt` exceeds this (seconds), decay the integrator
/// by `0.5 ^ (dt - threshold)`. Mirrors Python's idle-decay block:
/// `if dt_s > 2.0: decay = 0.5 ** (dt_s - 2.0)`.
const IDLE_DECAY_THRESHOLD_SECS: f64 = 2.0;

/// Immutable snapshot of controller state. Used by the WebSocket
/// status and gateway health probes to expose controller internals
/// without leaking the mutable struct.
///
/// Field-for-field parity with Python's `AdaptiveBatchState`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveBatchState {
    pub enabled: bool,
    pub calibrated: bool,
    pub target_p50_ms: Option<f64>,
    pub current_wait_ms: f64,
    pub current_batch_cost: u64,
    pub observed_p50_ms: Option<f64>,
    pub headroom_ms: Option<f64>,
    pub fill_ratio: Option<f64>,
    pub integral: f64,
    pub starvation_streak: u32,
    pub starvation_resets: u32,
}

/// PI controller plus companion proportional controller.
///
/// Constructed via [`AdaptiveBatchController::builder`] in tests /
/// production; the plain `new` path uses Python-parity defaults
/// everywhere.
///
/// ## Knobs
///
/// * `max_batch_wait_ms` — PI-controlled. Proportional on
///   `headroom_ms`, integrator in `ms·s` time-normalised, clamped
///   to `[min_wait_ms, max_wait_ms]`, saturation-aware anti-windup.
/// * `max_batch_cost` — proportional-only, gated on
///   `fill_ratio ≥ fill_ratio_threshold`. The gate behaves as a
///   conditional integrator: only adjusts when the GPU is saturated.
/// * `target_p50_ms` — either set explicitly (then `calibrated = true`
///   from construction) or auto-derived as
///   `inference_p50 × calibration_multiplier`, clamped to
///   `[min_target_p50_ms, max_target_p50_ms]`.
///
/// ## Starvation recovery
///
/// Python's docs call out a pathological attractor: sustained
/// above-target p50 can drive both knobs to their floors, at which
/// point each batch becomes a single item, p50 stays high, and the
/// loop is stuck. Recovery:
///
/// 1. Count consecutive batches where `batch_size ≤
///    starvation_batch_size`.
/// 2. If the streak hits `starvation_window` **and** both knobs are
///    at their floors, hard-reset to a midway point between floors
///    and the operator-provided initial values.
/// 3. Zero the integrator and the step-time so the PI loop restarts
///    from clean state.
///
/// The streak counter only advances when `step()` is called with
/// `Some(batch_size)`. Idle ticks or legacy test callers that pass
/// `None` are no-ops. Mirrors Python exactly.
#[derive(Debug, Clone)]
pub struct AdaptiveBatchController {
    // ── Configuration (mirror of Python's field defaults) ──────────
    target_p50_ms: Option<f64>,
    pub calibration_multiplier: f64,
    pub min_target_p50_ms: f64,
    pub max_target_p50_ms: f64,

    pub min_wait_ms: f64,
    pub max_wait_ms: f64,

    pub min_batch_cost: u64,
    pub max_batch_cost: u64,

    pub gain: f64,
    pub integral_gain: f64,
    pub cost_gain: f64,
    pub update_interval: u32,
    pub fill_ratio_threshold: f64,

    pub starvation_recovery_enabled: bool,
    pub starvation_window: u32,
    pub starvation_batch_size: u32,

    // ── Recovery anchors captured from the operator-provided start
    // values — Python does this in `__post_init__`. ────────────────
    initial_wait_ms: f64,
    initial_batch_cost: u64,

    // ── Runtime state ──────────────────────────────────────────────
    current_wait_ms: f64,
    current_batch_cost: u64,
    steps_since_update: u32,

    auto_calibrate: bool,
    calibrated: bool,
    inference_tracker: LatencyTracker,

    integral: f64,
    integral_max: f64,
    last_step_time: Option<Instant>,

    starvation_streak: u32,
    starvation_resets: u32,

    /// Virtual clock used only by [`Self::step_replay`]. Holds the
    /// total `dt_s` accumulated across replay calls (in seconds, f64)
    /// and the last-step snapshot that mirrors Python's
    /// `_last_step_time`. Using raw f64 here — instead of an
    /// [`Instant`] plus [`std::time::Duration`] — is deliberate:
    /// [`Duration`] truncates to nanoseconds, which breaks ULP
    /// identity with Python's `time.monotonic()` (a plain f64).
    /// Never read by the wall-clock [`Self::step`] path.
    replay_now_s: f64,
    replay_last_step_s: Option<f64>,
}

impl AdaptiveBatchController {
    /// Construct with Python-parity defaults, optionally overridden
    /// by env vars. Used by `Scheduler::builder()` when no explicit
    /// controller is supplied so every per-model scheduler picks up
    /// the same operator knobs.
    ///
    /// # Env vars
    ///
    /// All vars are `f64` (or `u64` for cost) and *override the
    /// matching field on top of the Python-parity default*. Unset =
    /// use default. Bad values (non-numeric, NaN, etc.) are logged
    /// and ignored — the controller falls back to defaults rather
    /// than crashing.
    ///
    /// | Var                                       | Field                    | Default | Production |
    /// |-------------------------------------------|--------------------------|---------|------------|
    /// | `SIE_ADAPTIVE_BATCH_MAX_WAIT_MS`          | `max_wait_ms`            | 50.0    | 50.0 |
    /// | `SIE_ADAPTIVE_BATCH_MIN_WAIT_MS`          | `min_wait_ms`            | 1.0     | 15.0 |
    /// | `SIE_ADAPTIVE_BATCH_INITIAL_WAIT_MS`      | `initial_wait_ms`        | 10.0    | 10.0 |
    /// | `SIE_ADAPTIVE_BATCH_TARGET_P50_MS`        | `target_p50_ms`          | None    | None |
    /// | `SIE_ADAPTIVE_BATCH_MIN_TARGET_P50_MS`    | `min_target_p50_ms`      | 5.0     | 5.0 |
    /// | `SIE_ADAPTIVE_BATCH_MAX_TARGET_P50_MS`    | `max_target_p50_ms`      | 500.0   | 500.0 |
    /// | `SIE_ADAPTIVE_BATCH_CALIBRATION_MULT`     | `calibration_multiplier` | 1.5     | 1.5 |
    /// | `SIE_ADAPTIVE_BATCH_MAX_COST`             | `max_batch_cost`         | 65536   | `max(cost_floor, max_batch_tokens * 4)` |
    /// | `SIE_ADAPTIVE_BATCH_MIN_COST`             | `min_batch_cost`         | 256     | `max(256, max_batch_tokens / 4)` |
    /// | `SIE_ADAPTIVE_BATCH_INITIAL_COST`         | `initial_batch_cost`     | 16384   | 16384 |
    ///
    /// The production floor values are intentionally higher than the
    /// dataclass-style defaults used in small unit tests. Lower wait/cost
    /// floors make the controller more eager to flush tiny batches; change
    /// them only with a fresh validation baseline.
    ///
    /// The `BATCH` infix distinguishes these from the existing
    /// `SIE_ADAPTIVE_*_QUANTUM_MS` family, which control the NATS
    /// fetch quantum (see [`crate::latency::FetchExpiryController`]).
    #[must_use]
    pub fn from_env_or_default() -> Self {
        let mut c = Self::default();

        fn get_f64(var: &str) -> Option<f64> {
            std::env::var(var).ok().and_then(|s| {
                s.parse::<f64>().ok().filter(|v| v.is_finite()).or_else(|| {
                    tracing::warn!(
                        var = var,
                        raw = s.as_str(),
                        "adaptive-batch env var: ignoring non-numeric/non-finite value"
                    );
                    None
                })
            })
        }
        fn get_u64(var: &str) -> Option<u64> {
            std::env::var(var).ok().and_then(|s| s.parse::<u64>().ok())
        }

        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_MAX_WAIT_MS") {
            c.max_wait_ms = v;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_MIN_WAIT_MS") {
            c.min_wait_ms = v;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_INITIAL_WAIT_MS") {
            c.initial_wait_ms = v;
            c.current_wait_ms = clamp(v, c.min_wait_ms, c.max_wait_ms);
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_TARGET_P50_MS") {
            c.target_p50_ms = Some(v);
            c.auto_calibrate = false;
            c.calibrated = true;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_MIN_TARGET_P50_MS") {
            c.min_target_p50_ms = v;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_MAX_TARGET_P50_MS") {
            c.max_target_p50_ms = v;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_CALIBRATION_MULT") {
            c.calibration_multiplier = v;
        }
        if let Some(v) = get_u64("SIE_ADAPTIVE_BATCH_MAX_COST") {
            c.max_batch_cost = v;
        }
        if let Some(v) = get_u64("SIE_ADAPTIVE_BATCH_MIN_COST") {
            c.min_batch_cost = v;
        }
        if let Some(v) = get_u64("SIE_ADAPTIVE_BATCH_INITIAL_COST") {
            c.initial_batch_cost = v;
            c.current_batch_cost = clamp_u64(v, c.min_batch_cost, c.max_batch_cost);
        }

        // Re-clamp current values in case env vars compressed the
        // bounds beneath the defaults.
        c.current_wait_ms = clamp(c.current_wait_ms, c.min_wait_ms, c.max_wait_ms);
        c.current_batch_cost = clamp_u64(c.current_batch_cost, c.min_batch_cost, c.max_batch_cost);

        tracing::info!(
            min_wait_ms = c.min_wait_ms,
            max_wait_ms = c.max_wait_ms,
            initial_wait_ms = c.initial_wait_ms,
            current_wait_ms = c.current_wait_ms,
            min_batch_cost = c.min_batch_cost,
            max_batch_cost = c.max_batch_cost,
            initial_batch_cost = c.initial_batch_cost,
            target_p50_ms = ?c.target_p50_ms,
            min_target_p50_ms = c.min_target_p50_ms,
            max_target_p50_ms = c.max_target_p50_ms,
            calibration_multiplier = c.calibration_multiplier,
            auto_calibrate = c.auto_calibrate,
            "adaptive-batch: controller constructed from env+defaults"
        );

        c
    }

    /// Construct with Python-parity defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Production-parity constructor: derive every field that
    /// `sie_server/core/worker/model_worker.py::ModelWorker.__init__`
    /// derives from `BatchConfig` + `AdaptiveBatchingConfig`, then
    /// apply env-var overrides on top.
    ///
    /// Why this exists distinct from [`Self::from_env_or_default`]:
    /// the bare dataclass defaults in [`Self::default`] mirror
    /// Python's *module-level* defaults
    /// (`adaptive_batching.AdaptiveBatchController` field defaults),
    /// which are intentionally test-only floors. Production deploys
    /// always go through `ModelWorker.__init__`, which substitutes:
    ///
    /// | Field             | Python production wiring                                         |
    /// |-------------------|------------------------------------------------------------------|
    /// | `min_wait_ms`     | `ab.min_wait_ms` (engine default **15.0**)                        |
    /// | `_current_wait_ms`| `cfg.max_batch_wait_ms` (default **15.0**)                        |
    /// | `min_batch_cost`  | `max(256, max_batch_tokens // 4)` — model-scaled                  |
    /// | `max_batch_cost`  | `max(cost_floor, max_batch_tokens * 4)` — model-scaled            |
    /// | `cost_gain`       | `gain * 0.5` — coupled, not free                                  |
    /// | `_current_batch_cost` | `max_batch_tokens` (default **16_384**)                       |
    ///
    /// These production floors matter: deployed Python does not run the
    /// low module-level defaults. If Rust falls back to those bare
    /// values, the PI loop flushes tiny batches too eagerly under
    /// saturation. Restoring the model-derived floors here is the
    /// equivalent of the Python production constructor and is what every
    /// deployed `ModelWorker` in `sie-internal` already runs.
    ///
    /// Env vars (`SIE_ADAPTIVE_BATCH_*`) are then applied on top of
    /// the production-derived defaults — operators can still pin
    /// individual knobs without recompiling. Production-parity is
    /// the *fallback*, env is the *override*.
    #[must_use]
    #[allow(
        clippy::field_reassign_with_default,
        reason = "Sequential derivations from cfg are clearer step-by-step than \
                  inline struct-update syntax — cost_floor depends on \
                  max_batch_tokens, cost_ceiling depends on cost_floor, etc."
    )]
    pub fn from_batch_config(cfg: &BatchConfig) -> Self {
        let mut c = Self::default();

        // Wait-knob floor + initial value follow Python production.
        c.min_wait_ms = PRODUCTION_MIN_WAIT_MS;
        c.initial_wait_ms = cfg.max_batch_wait_ms;

        // Cost-knob floor + ceiling are derived from the model's
        // `max_batch_tokens` budget (= cfg.max_batch_cost in Rust).
        // Python's expression captured here verbatim.
        let max_batch_tokens = cfg.max_batch_cost;
        let cost_floor = u64::max(256, max_batch_tokens / PRODUCTION_MIN_COST_DIVISOR);
        let cost_floor = u64::min(cost_floor, max_batch_tokens.max(1));
        let cost_ceiling = u64::max(
            cost_floor,
            max_batch_tokens.saturating_mul(PRODUCTION_MAX_COST_MULT),
        );
        c.min_batch_cost = cost_floor;
        c.max_batch_cost = cost_ceiling;
        c.initial_batch_cost = max_batch_tokens.max(cost_floor).min(cost_ceiling);

        // Cost-knob gain coupled to wait-knob gain (Python
        // `cost_gain = ab.gain * 0.5`).
        c.cost_gain = c.gain * PRODUCTION_COST_GAIN_RATIO;

        // Re-clamp current values into the new bounds before any
        // step runs.
        c.current_wait_ms = clamp(c.initial_wait_ms, c.min_wait_ms, c.max_wait_ms);
        c.current_batch_cost = clamp_u64(c.initial_batch_cost, c.min_batch_cost, c.max_batch_cost);

        c
    }

    /// Production-parity constructor with env-var overrides applied
    /// on top. This is what [`super::SchedulerBuilder::build`] uses
    /// when the operator hasn't supplied an explicit controller —
    /// every deployed Rust scheduler picks up Python production
    /// defaults plus operator env tuning in one shot.
    #[must_use]
    pub fn from_batch_config_and_env(cfg: &BatchConfig) -> Self {
        let mut c = Self::from_batch_config(cfg);
        Self::apply_env_overrides(&mut c);

        tracing::info!(
            min_wait_ms = c.min_wait_ms,
            max_wait_ms = c.max_wait_ms,
            initial_wait_ms = c.initial_wait_ms,
            current_wait_ms = c.current_wait_ms,
            min_batch_cost = c.min_batch_cost,
            max_batch_cost = c.max_batch_cost,
            initial_batch_cost = c.initial_batch_cost,
            cost_gain = c.cost_gain,
            target_p50_ms = ?c.target_p50_ms,
            "adaptive-batch: controller constructed from BatchConfig+env (production-parity)"
        );

        c
    }

    /// Apply `SIE_ADAPTIVE_BATCH_*` env-var overrides to an already
    /// constructed controller. Bad values log and fall through —
    /// never crash.
    fn apply_env_overrides(c: &mut Self) {
        fn get_f64(var: &str) -> Option<f64> {
            std::env::var(var).ok().and_then(|s| {
                s.parse::<f64>().ok().filter(|v| v.is_finite()).or_else(|| {
                    tracing::warn!(
                        var = var,
                        raw = s.as_str(),
                        "adaptive-batch env var: ignoring non-numeric/non-finite value"
                    );
                    None
                })
            })
        }
        fn get_u64(var: &str) -> Option<u64> {
            std::env::var(var).ok().and_then(|s| s.parse::<u64>().ok())
        }

        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_MAX_WAIT_MS") {
            c.max_wait_ms = v;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_MIN_WAIT_MS") {
            c.min_wait_ms = v;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_INITIAL_WAIT_MS") {
            c.initial_wait_ms = v;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_TARGET_P50_MS") {
            c.target_p50_ms = Some(v);
            c.auto_calibrate = false;
            c.calibrated = true;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_MIN_TARGET_P50_MS") {
            c.min_target_p50_ms = v;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_MAX_TARGET_P50_MS") {
            c.max_target_p50_ms = v;
        }
        if let Some(v) = get_f64("SIE_ADAPTIVE_BATCH_CALIBRATION_MULT") {
            c.calibration_multiplier = v;
        }
        if let Some(v) = get_u64("SIE_ADAPTIVE_BATCH_MAX_COST") {
            c.max_batch_cost = v;
        }
        if let Some(v) = get_u64("SIE_ADAPTIVE_BATCH_MIN_COST") {
            c.min_batch_cost = v;
        }
        if let Some(v) = get_u64("SIE_ADAPTIVE_BATCH_INITIAL_COST") {
            c.initial_batch_cost = v;
        }

        // Resolve floor/ceiling conflicts before clamping current
        // values. If the operator's env-supplied ceiling sits below
        // the production-parity floor (e.g.
        // `SIE_ADAPTIVE_BATCH_MAX_WAIT_MS=10` with the new default
        // `min_wait_ms = 15`), the more specific override wins:
        // drag the floor down to the ceiling so the PI loop has a
        // valid operating range. Without this guard `clamp(x, 15,
        // 10) == 15` and the wait knob silently exceeds its declared
        // cap.
        if c.min_wait_ms > c.max_wait_ms {
            tracing::warn!(
                min_wait_ms = c.min_wait_ms,
                max_wait_ms = c.max_wait_ms,
                "adaptive-batch: min_wait_ms > max_wait_ms after env overrides; \
                 lowering floor to match ceiling"
            );
            c.min_wait_ms = c.max_wait_ms;
        }
        if c.min_batch_cost > c.max_batch_cost {
            tracing::warn!(
                min_batch_cost = c.min_batch_cost,
                max_batch_cost = c.max_batch_cost,
                "adaptive-batch: min_batch_cost > max_batch_cost after env overrides; \
                 lowering floor to match ceiling"
            );
            c.min_batch_cost = c.max_batch_cost;
        }

        // Re-clamp current values in case env vars compressed bounds
        // beneath the production-derived defaults.
        c.current_wait_ms = clamp(c.initial_wait_ms, c.min_wait_ms, c.max_wait_ms);
        c.current_batch_cost = clamp_u64(c.initial_batch_cost, c.min_batch_cost, c.max_batch_cost);
    }

    /// Fluent builder. Prefer this over direct field assignment
    /// because `initial_wait_ms` / `initial_batch_cost` are anchored
    /// in `finish()` — bypassing the builder leaves the recovery
    /// anchors zeroed.
    #[must_use]
    pub fn builder() -> AdaptiveBatchControllerBuilder {
        AdaptiveBatchControllerBuilder::default()
    }

    // ---- Calibration / sampling ----

    /// Record an inference-only latency sample. Consumed by the
    /// auto-calibration path; once calibration completes, this is a
    /// no-op (Python parity).
    pub fn record_inference_sample(&mut self, inference_ms: f64) {
        if !self.calibrated {
            self.inference_tracker.record(inference_ms);
        }
    }

    // ---- Main step ----

    /// Advance the controller by one batch. Returns the caps the
    /// batcher should now honour `(wait_ms, batch_cost)`.
    ///
    /// `observed_p50_ms` is the **total** p50 (queue + inference +
    /// postprocess), fed from `LatencyTracker.record(total_ms)`
    /// upstream. `fill_ratio` is the mean of
    /// [`crate::scheduler::BatchEfficiencyTracker`]. `batch_size` is the size of the
    /// batch whose completion triggered this step — required for
    /// starvation tracking.
    pub fn step(
        &mut self,
        observed_p50_ms: Option<f64>,
        fill_ratio: Option<f64>,
        batch_size: Option<usize>,
    ) -> (f64, u64) {
        self.step_impl(TimeSource::Wall, observed_p50_ms, fill_ratio, batch_size)
    }

    /// Deterministic variant of [`Self::step`] that bypasses
    /// [`Instant::now`] by advancing an internal virtual f64-seconds
    /// clock by `dt_s` before running the controller. Intended
    /// **exclusively** for shadow-trace replay — see
    /// `examples/replay_controller_trace.rs` and
    /// `scripts/perf/replay-scheduler-trace.py`.
    ///
    /// The virtual clock advances on **every** call (including calls
    /// that early-return under `update_interval`), matching the Python
    /// semantics where `time.monotonic()` runs on every real-world
    /// tick even though `_last_step_time` is only updated in the
    /// branch that consumes `dt_s`. Using raw f64 seconds (rather than
    /// an [`Instant`] + [`std::time::Duration`]) is deliberate:
    /// `Duration` truncates to nanoseconds, which breaks ULP identity
    /// against Python's `float - float` arithmetic on `time.monotonic()`.
    ///
    /// Production code must use [`Self::step`]; using this entry
    /// point in the hot path would decouple the integrator from real
    /// elapsed time and silently break the PI tuning.
    #[doc(hidden)]
    pub fn step_replay(
        &mut self,
        dt_s: f64,
        observed_p50_ms: Option<f64>,
        fill_ratio: Option<f64>,
        batch_size: Option<usize>,
    ) -> (f64, u64) {
        self.replay_now_s += dt_s;
        self.step_impl(TimeSource::Replay, observed_p50_ms, fill_ratio, batch_size)
    }

    fn step_impl(
        &mut self,
        time_source: TimeSource,
        observed_p50_ms: Option<f64>,
        fill_ratio: Option<f64>,
        batch_size: Option<usize>,
    ) -> (f64, u64) {
        // Starvation counter runs on every call — it's a batch-rate
        // signal, not a controller-rate one.
        self.update_starvation_tracker(batch_size);

        self.steps_since_update += 1;
        if self.steps_since_update < self.update_interval {
            return (self.current_wait_ms, self.current_batch_cost);
        }
        self.steps_since_update = 0;

        // Auto-calibration branch: infer the target from observed
        // backend latency until an explicit env target is set.
        if !self.calibrated {
            if let Some(inference_p50) = self.inference_tracker.p50() {
                let t = clamp(
                    inference_p50 * self.calibration_multiplier,
                    self.min_target_p50_ms,
                    self.max_target_p50_ms,
                );
                self.target_p50_ms = Some(t);
                self.calibrated = true;
                self.integral = 0.0;
                self.last_step_time = None;
                self.replay_last_step_s = None;
                tracing::info!(
                    target_p50_ms = t,
                    inference_p50_ms = inference_p50,
                    calibration_multiplier = self.calibration_multiplier,
                    min_target_p50_ms = self.min_target_p50_ms,
                    max_target_p50_ms = self.max_target_p50_ms,
                    "adaptive: auto-calibrated latency target"
                );
            }
            // Hold knobs at initial values until calibrated.
            return (self.current_wait_ms, self.current_batch_cost);
        }

        let (Some(observed), Some(target)) = (observed_p50_ms, self.target_p50_ms) else {
            return (self.current_wait_ms, self.current_batch_cost);
        };

        // ── Starvation recovery runs *before* the PI update so a
        // reset this step doesn't fight the math on the same tick. ─
        if self.maybe_recover_from_starvation() {
            return (self.current_wait_ms, self.current_batch_cost);
        }

        let headroom_ms = target - observed;
        let headroom_frac = headroom_ms / target;

        // ── dt for time-normalised integral ───────────────────────
        // Two entry points: the production path sources `dt_s` from
        // `Instant::now() - last_step_time`; the `step_replay` path
        // uses a virtual f64-seconds clock so a Python and Rust run
        // of the same trace agree ULP-for-ULP. See
        // `step_replay`'s docstring for why f64 (not Duration) is
        // mandatory for the replay side.
        let dt_s = match time_source {
            TimeSource::Wall => {
                let now = Instant::now();
                let dt = match self.last_step_time {
                    Some(prev) => {
                        let gap = now.saturating_duration_since(prev).as_secs_f64();
                        f64::min(gap, DT_CAP_SECS)
                    }
                    None => 0.0, // first step after calibration
                };
                self.last_step_time = Some(now);
                dt
            }
            TimeSource::Replay => {
                let now_s = self.replay_now_s;
                let dt = match self.replay_last_step_s {
                    Some(prev) => f64::min(now_s - prev, DT_CAP_SECS),
                    None => 0.0,
                };
                self.replay_last_step_s = Some(now_s);
                dt
            }
        };

        // ── Idle decay (prevents stale error from bleeding into the
        // next burst after a long pause). ─────────────────────────
        if dt_s > IDLE_DECAY_THRESHOLD_SECS {
            let decay = 0.5f64.powf(dt_s - IDLE_DECAY_THRESHOLD_SECS);
            self.integral *= decay;
        }

        // ── Saturation-aware anti-windup on the wait knob ─────────
        let output_at_max = self.current_wait_ms >= self.max_wait_ms;
        let output_at_min = self.current_wait_ms <= self.min_wait_ms;

        let mut can_integrate = true;
        if output_at_max && headroom_ms > 0.0 {
            can_integrate = false;
        }
        if output_at_min && headroom_ms < 0.0 {
            can_integrate = false;
        }
        if can_integrate && self.integral_gain > 0.0 {
            self.integral += headroom_ms * dt_s;
            self.integral = clamp(self.integral, -self.integral_max, self.integral_max);
        }

        // ── Knob 1: wait ──────────────────────────────────────────
        let wait_adjustment = headroom_ms * self.gain + self.integral * self.integral_gain;
        let new_wait = self.current_wait_ms + wait_adjustment;
        self.current_wait_ms = clamp(new_wait, self.min_wait_ms, self.max_wait_ms);

        // ── Knob 2: cost (proportional, gated on fill_ratio) ──────
        // Anti-windup mirrors the wait knob: don't push further into
        // saturation in the direction the error wants. Without this,
        // the cost collapses to `min_batch_cost` once p50 > target;
        // see the regression doc referenced in the Python source.
        if let Some(fill) = fill_ratio {
            if fill >= self.fill_ratio_threshold {
                let cost_at_max = self.current_batch_cost >= self.max_batch_cost;
                let cost_at_min = self.current_batch_cost <= self.min_batch_cost;
                let mut can_adjust = true;
                if cost_at_max && headroom_ms > 0.0 {
                    can_adjust = false;
                }
                if cost_at_min && headroom_ms < 0.0 {
                    can_adjust = false;
                }
                if can_adjust {
                    let adj = self.current_batch_cost as f64 * headroom_frac * self.cost_gain;
                    let new_cost = self.current_batch_cost as f64 + adj;
                    let clamped = clamp(
                        new_cost,
                        self.min_batch_cost as f64,
                        self.max_batch_cost as f64,
                    );
                    // Python uses `int(clamp(...))` which truncates
                    // toward zero; f64 → u64 `as` conversion
                    // truncates identically for the non-negative
                    // range we clamp to.
                    self.current_batch_cost = clamped as u64;
                }
            }
        }

        tracing::debug!(
            observed_p50_ms = observed,
            target_p50_ms = target,
            headroom_ms,
            integral = self.integral,
            fill_ratio = fill_ratio.unwrap_or(0.0),
            current_wait_ms = self.current_wait_ms,
            current_batch_cost = self.current_batch_cost,
            starvation_streak = self.starvation_streak,
            "adaptive: step"
        );

        (self.current_wait_ms, self.current_batch_cost)
    }

    // ---- Starvation detector helpers ----

    fn update_starvation_tracker(&mut self, batch_size: Option<usize>) {
        let Some(size) = batch_size else {
            return;
        };
        if size > self.starvation_batch_size as usize {
            self.starvation_streak = 0;
        } else {
            self.starvation_streak = self.starvation_streak.saturating_add(1);
        }
    }

    fn maybe_recover_from_starvation(&mut self) -> bool {
        if !self.starvation_recovery_enabled {
            return false;
        }
        if self.starvation_streak < self.starvation_window {
            return false;
        }
        let at_wait_floor = self.current_wait_ms <= self.min_wait_ms;
        let at_cost_floor = self.current_batch_cost <= self.min_batch_cost;
        if !(at_wait_floor && at_cost_floor) {
            return false;
        }

        let recovery_wait = clamp(
            (self.min_wait_ms + self.initial_wait_ms) / 2.0,
            self.min_wait_ms,
            self.max_wait_ms,
        );
        // Mirrors Python: max(initial // 2, min_cost * 4), then clamp.
        // `initial_batch_cost / 2` is integer division in Python
        // (`//`); same in u64 here.
        let candidate = std::cmp::max(
            self.initial_batch_cost / 2,
            self.min_batch_cost.saturating_mul(4),
        );
        let recovery_cost = clamp(
            candidate as f64,
            self.min_batch_cost as f64,
            self.max_batch_cost as f64,
        ) as u64;

        tracing::warn!(
            streak = self.starvation_streak,
            starvation_batch_size = self.starvation_batch_size,
            old_wait_ms = self.current_wait_ms,
            old_cost = self.current_batch_cost,
            recovery_wait_ms = recovery_wait,
            recovery_cost,
            "adaptive: starvation recovery triggered"
        );

        self.current_wait_ms = recovery_wait;
        self.current_batch_cost = recovery_cost;
        self.integral = 0.0;
        self.last_step_time = None;
        self.replay_last_step_s = None;
        self.starvation_streak = 0;
        self.starvation_resets = self.starvation_resets.saturating_add(1);
        true
    }

    // ---- Snapshots / accessors ----

    #[must_use]
    pub fn current_wait_ms(&self) -> f64 {
        self.current_wait_ms
    }

    #[must_use]
    pub fn current_batch_cost(&self) -> u64 {
        self.current_batch_cost
    }

    #[must_use]
    pub fn calibrated(&self) -> bool {
        self.calibrated
    }

    #[must_use]
    pub fn target_p50_ms(&self) -> Option<f64> {
        self.target_p50_ms
    }

    #[must_use]
    pub fn starvation_streak(&self) -> u32 {
        self.starvation_streak
    }

    #[must_use]
    pub fn starvation_resets(&self) -> u32 {
        self.starvation_resets
    }

    #[must_use]
    pub fn snapshot(
        &self,
        observed_p50_ms: Option<f64>,
        fill_ratio: Option<f64>,
    ) -> AdaptiveBatchState {
        let headroom = match (self.target_p50_ms, observed_p50_ms) {
            (Some(t), Some(o)) => Some(t - o),
            _ => None,
        };
        AdaptiveBatchState {
            enabled: true,
            calibrated: self.calibrated,
            target_p50_ms: self.target_p50_ms,
            current_wait_ms: self.current_wait_ms,
            current_batch_cost: self.current_batch_cost,
            observed_p50_ms,
            headroom_ms: headroom,
            fill_ratio,
            integral: self.integral,
            starvation_streak: self.starvation_streak,
            starvation_resets: self.starvation_resets,
        }
    }

    /// Reset to operator-provided initial state.
    ///
    /// If the controller was constructed in auto-calibrate mode,
    /// `target_p50_ms` and `calibrated` are cleared so the next
    /// inference-sample stream can re-derive the target. If a
    /// target was set explicitly, calibration state is preserved.
    ///
    /// `starvation_resets` is deliberately *not* cleared — it's a
    /// monotonic operational counter and
    /// resetting it would erase evidence of incidents.
    pub fn reset(&mut self) {
        self.current_wait_ms = clamp(self.initial_wait_ms, self.min_wait_ms, self.max_wait_ms);
        self.current_batch_cost = clamp(
            self.initial_batch_cost as f64,
            self.min_batch_cost as f64,
            self.max_batch_cost as f64,
        ) as u64;
        self.steps_since_update = 0;
        self.integral = 0.0;
        self.last_step_time = None;
        self.replay_now_s = 0.0;
        self.replay_last_step_s = None;
        self.starvation_streak = 0;
        if self.auto_calibrate {
            self.calibrated = false;
            self.target_p50_ms = None;
            self.inference_tracker.reset();
        }
    }
}

impl Default for AdaptiveBatchController {
    fn default() -> Self {
        // Mirrors Python's dataclass field defaults exactly. Tests
        // pin every value so this can't drift silently.
        let initial_wait_ms = 10.0;
        let initial_batch_cost: u64 = 16_384;

        Self {
            target_p50_ms: None,
            calibration_multiplier: 1.5,
            min_target_p50_ms: 5.0,
            max_target_p50_ms: 500.0,

            min_wait_ms: 1.0,
            max_wait_ms: 50.0,

            min_batch_cost: 256,
            max_batch_cost: 65_536,

            gain: 0.3,
            integral_gain: 0.05,
            cost_gain: 0.15,
            update_interval: 10,
            fill_ratio_threshold: 0.7,

            starvation_recovery_enabled: true,
            starvation_window: 20,
            starvation_batch_size: 1,

            initial_wait_ms,
            initial_batch_cost,

            current_wait_ms: initial_wait_ms,
            current_batch_cost: initial_batch_cost,
            steps_since_update: 0,

            auto_calibrate: true, // because `target_p50_ms == None`
            calibrated: false,

            inference_tracker: LatencyTracker::new(
                INFERENCE_TRACKER_WINDOW,
                INFERENCE_TRACKER_MIN_SAMPLES,
            ),

            integral: 0.0,
            integral_max: 20.0,
            last_step_time: None,

            starvation_streak: 0,
            starvation_resets: 0,

            replay_now_s: 0.0,
            replay_last_step_s: None,
        }
    }
}

// ---- Builder ----

/// Builder for [`AdaptiveBatchController`].
///
/// All fields default to the same Python-parity values as
/// [`AdaptiveBatchController::default`]. Fields set via the builder
/// override those defaults; everything else stays at parity.
#[derive(Debug, Clone, Default)]
pub struct AdaptiveBatchControllerBuilder {
    target_p50_ms: Option<Option<f64>>, // double-Option: outer=unset, inner=explicit None
    calibration_multiplier: Option<f64>,
    min_target_p50_ms: Option<f64>,
    max_target_p50_ms: Option<f64>,
    min_wait_ms: Option<f64>,
    max_wait_ms: Option<f64>,
    min_batch_cost: Option<u64>,
    max_batch_cost: Option<u64>,
    gain: Option<f64>,
    integral_gain: Option<f64>,
    cost_gain: Option<f64>,
    update_interval: Option<u32>,
    fill_ratio_threshold: Option<f64>,
    starvation_recovery_enabled: Option<bool>,
    starvation_window: Option<u32>,
    starvation_batch_size: Option<u32>,
    initial_wait_ms: Option<f64>,
    initial_batch_cost: Option<u64>,
}

macro_rules! builder_setter {
    ($name:ident, $ty:ty) => {
        #[must_use]
        pub fn $name(mut self, v: $ty) -> Self {
            self.$name = Some(v);
            self
        }
    };
}

impl AdaptiveBatchControllerBuilder {
    pub fn target_p50_ms(mut self, v: Option<f64>) -> Self {
        self.target_p50_ms = Some(v);
        self
    }

    builder_setter!(calibration_multiplier, f64);
    builder_setter!(min_target_p50_ms, f64);
    builder_setter!(max_target_p50_ms, f64);
    builder_setter!(min_wait_ms, f64);
    builder_setter!(max_wait_ms, f64);
    builder_setter!(min_batch_cost, u64);
    builder_setter!(max_batch_cost, u64);
    builder_setter!(gain, f64);
    builder_setter!(integral_gain, f64);
    builder_setter!(cost_gain, f64);
    builder_setter!(update_interval, u32);
    builder_setter!(fill_ratio_threshold, f64);
    builder_setter!(starvation_recovery_enabled, bool);
    builder_setter!(starvation_window, u32);
    builder_setter!(starvation_batch_size, u32);
    builder_setter!(initial_wait_ms, f64);
    builder_setter!(initial_batch_cost, u64);

    #[must_use]
    pub fn build(self) -> AdaptiveBatchController {
        let mut c = AdaptiveBatchController::default();
        if let Some(v) = self.target_p50_ms {
            c.target_p50_ms = v;
            c.auto_calibrate = v.is_none();
            c.calibrated = v.is_some();
        }
        if let Some(v) = self.calibration_multiplier {
            c.calibration_multiplier = v;
        }
        if let Some(v) = self.min_target_p50_ms {
            c.min_target_p50_ms = v;
        }
        if let Some(v) = self.max_target_p50_ms {
            c.max_target_p50_ms = v;
        }
        if let Some(v) = self.min_wait_ms {
            c.min_wait_ms = v;
        }
        if let Some(v) = self.max_wait_ms {
            c.max_wait_ms = v;
        }
        if let Some(v) = self.min_batch_cost {
            c.min_batch_cost = v;
        }
        if let Some(v) = self.max_batch_cost {
            c.max_batch_cost = v;
        }
        if let Some(v) = self.gain {
            c.gain = v;
        }
        if let Some(v) = self.integral_gain {
            c.integral_gain = v;
        }
        if let Some(v) = self.cost_gain {
            c.cost_gain = v;
        }
        if let Some(v) = self.update_interval {
            c.update_interval = v;
        }
        if let Some(v) = self.fill_ratio_threshold {
            c.fill_ratio_threshold = v;
        }
        if let Some(v) = self.starvation_recovery_enabled {
            c.starvation_recovery_enabled = v;
        }
        if let Some(v) = self.starvation_window {
            c.starvation_window = v;
        }
        if let Some(v) = self.starvation_batch_size {
            c.starvation_batch_size = v;
        }
        if let Some(v) = self.initial_wait_ms {
            c.initial_wait_ms = v;
            c.current_wait_ms = v;
        }
        if let Some(v) = self.initial_batch_cost {
            c.initial_batch_cost = v;
            c.current_batch_cost = v;
        }
        c
    }
}

fn clamp(value: f64, lo: f64, hi: f64) -> f64 {
    f64::max(lo, f64::min(hi, value))
}

fn clamp_u64(value: u64, lo: u64, hi: u64) -> u64 {
    value.clamp(lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_explicit_target(target: f64) -> AdaptiveBatchController {
        AdaptiveBatchController::builder()
            .target_p50_ms(Some(target))
            .update_interval(1) // step on every call in tests
            .build()
    }

    #[test]
    fn default_uses_python_parity_values() {
        let c = AdaptiveBatchController::default();
        assert!((c.calibration_multiplier - 1.5).abs() < f64::EPSILON);
        assert!((c.min_target_p50_ms - 5.0).abs() < f64::EPSILON);
        assert!((c.max_target_p50_ms - 500.0).abs() < f64::EPSILON);
        assert_eq!(c.min_batch_cost, 256);
        assert_eq!(c.max_batch_cost, 65_536);
        assert!((c.gain - 0.3).abs() < f64::EPSILON);
        assert!((c.integral_gain - 0.05).abs() < f64::EPSILON);
        assert!((c.cost_gain - 0.15).abs() < f64::EPSILON);
        assert_eq!(c.update_interval, 10);
        assert_eq!(c.starvation_window, 20);
        assert_eq!(c.starvation_batch_size, 1);
        assert!(c.auto_calibrate);
        assert!(!c.calibrated);
        assert_eq!(c.current_wait_ms, c.initial_wait_ms);
        assert_eq!(c.current_batch_cost, c.initial_batch_cost);
    }

    /// Helper: run `f` with a stable, isolated env. Writes to a
    /// process-wide table so the body must not touch any other env
    /// vars that another test could be reading concurrently — here
    /// we serialise via a `parking_lot::Mutex` shared across env
    /// tests to keep `cargo test --jobs` deterministic.
    fn with_env_lock(f: impl FnOnce()) {
        // Use a static mutex so concurrent tests don't trample each
        // other's env. `std::sync::Mutex` is sufficient — the env
        // ops are sub-microsecond and we never hold across an
        // `await`.
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        f();
    }

    fn clear_env() {
        for v in [
            "SIE_ADAPTIVE_BATCH_MAX_WAIT_MS",
            "SIE_ADAPTIVE_BATCH_MIN_WAIT_MS",
            "SIE_ADAPTIVE_BATCH_INITIAL_WAIT_MS",
            "SIE_ADAPTIVE_BATCH_TARGET_P50_MS",
            "SIE_ADAPTIVE_BATCH_MIN_TARGET_P50_MS",
            "SIE_ADAPTIVE_BATCH_MAX_TARGET_P50_MS",
            "SIE_ADAPTIVE_BATCH_CALIBRATION_MULT",
            "SIE_ADAPTIVE_BATCH_MAX_COST",
            "SIE_ADAPTIVE_BATCH_MIN_COST",
            "SIE_ADAPTIVE_BATCH_INITIAL_COST",
        ] {
            // SAFETY: env mutation is unsafe in 2024-edition; the
            // surrounding `with_env_lock` serialises all callers.
            unsafe { std::env::remove_var(v) };
        }
    }

    #[test]
    fn from_env_or_default_no_env_matches_default() {
        with_env_lock(|| {
            clear_env();
            let c = AdaptiveBatchController::from_env_or_default();
            let d = AdaptiveBatchController::default();
            assert!((c.max_wait_ms - d.max_wait_ms).abs() < f64::EPSILON);
            assert!((c.min_wait_ms - d.min_wait_ms).abs() < f64::EPSILON);
            assert!((c.initial_wait_ms - d.initial_wait_ms).abs() < f64::EPSILON);
            assert_eq!(c.max_batch_cost, d.max_batch_cost);
            assert_eq!(c.target_p50_ms, d.target_p50_ms);
            assert_eq!(c.auto_calibrate, d.auto_calibrate);
        });
    }

    #[test]
    fn from_env_or_default_overrides_max_wait_ms() {
        // Cap the controller's wait knob at 10 ms and ensure the
        // current value is clamped with it.
        with_env_lock(|| {
            clear_env();
            unsafe { std::env::set_var("SIE_ADAPTIVE_BATCH_MAX_WAIT_MS", "10.0") };
            let c = AdaptiveBatchController::from_env_or_default();
            assert!((c.max_wait_ms - 10.0).abs() < f64::EPSILON);
            // current_wait_ms (init = 10.0) is at the new ceiling
            // post-clamp — fine, it just means the first PI step
            // can only push it down.
            assert!((c.current_wait_ms - 10.0).abs() < f64::EPSILON);
            clear_env();
        });
    }

    #[test]
    fn from_env_or_default_target_p50_disables_autocalibration() {
        // Setting an explicit target via env makes the controller
        // skip inference-tracker calibration entirely — useful when
        // we already know the steady-state inference time and want
        // deterministic startup.
        with_env_lock(|| {
            clear_env();
            unsafe { std::env::set_var("SIE_ADAPTIVE_BATCH_TARGET_P50_MS", "45.0") };
            let c = AdaptiveBatchController::from_env_or_default();
            assert_eq!(c.target_p50_ms, Some(45.0));
            assert!(!c.auto_calibrate);
            assert!(c.calibrated);
            clear_env();
        });
    }

    #[test]
    fn from_env_or_default_clamps_initial_to_new_max() {
        // If the operator drops max_wait below the default initial
        // value (10 ms), the constructor must clamp current_wait
        // down — otherwise the first batch sees a wait larger than
        // its declared ceiling and the PI loop fights it.
        with_env_lock(|| {
            clear_env();
            unsafe { std::env::set_var("SIE_ADAPTIVE_BATCH_MAX_WAIT_MS", "5.0") };
            let c = AdaptiveBatchController::from_env_or_default();
            assert!((c.max_wait_ms - 5.0).abs() < f64::EPSILON);
            assert!(c.current_wait_ms <= c.max_wait_ms);
            clear_env();
        });
    }

    #[test]
    fn from_env_or_default_ignores_garbage_values() {
        // Operator typo (eg. `SIE_ADAPTIVE_BATCH_MAX_WAIT_MS=ten`)
        // must not crash worker init — log + fall back to default.
        with_env_lock(|| {
            clear_env();
            unsafe { std::env::set_var("SIE_ADAPTIVE_BATCH_MAX_WAIT_MS", "not-a-number") };
            unsafe { std::env::set_var("SIE_ADAPTIVE_BATCH_INITIAL_WAIT_MS", "NaN") };
            let c = AdaptiveBatchController::from_env_or_default();
            // Both garbage env vars dropped → defaults preserved.
            assert!((c.max_wait_ms - 50.0).abs() < f64::EPSILON);
            assert!((c.initial_wait_ms - 10.0).abs() < f64::EPSILON);
            clear_env();
        });
    }

    #[test]
    fn from_env_or_default_overrides_calibration_knobs() {
        // Raise the auto-calibration floor / multiplier so cold-start
        // GPU samples do not pin `target_p50_ms` too low and force the
        // controller to permanently shrink batches under load.
        with_env_lock(|| {
            clear_env();
            unsafe { std::env::set_var("SIE_ADAPTIVE_BATCH_MIN_TARGET_P50_MS", "50.0") };
            unsafe { std::env::set_var("SIE_ADAPTIVE_BATCH_MAX_TARGET_P50_MS", "300.0") };
            unsafe { std::env::set_var("SIE_ADAPTIVE_BATCH_CALIBRATION_MULT", "2.5") };
            let c = AdaptiveBatchController::from_env_or_default();
            assert!((c.min_target_p50_ms - 50.0).abs() < f64::EPSILON);
            assert!((c.max_target_p50_ms - 300.0).abs() < f64::EPSILON);
            assert!((c.calibration_multiplier - 2.5).abs() < f64::EPSILON);
            // Auto-calibration still on (no explicit target set).
            assert!(c.auto_calibrate);
            clear_env();
        });
    }

    // ---- Production-parity wiring (`from_batch_config*`) ----

    #[test]
    fn from_batch_config_uses_python_production_floors() {
        // Mirror Python `ModelWorker.__init__` exactly:
        //   min_wait_ms = ab.min_wait_ms (engine default 15.0)
        //   _current_wait_ms = cfg.max_batch_wait_ms (default 15.0)
        //   min_batch_cost = max(256, max_batch_tokens // 4)
        //   max_batch_cost = max(cost_floor, max_batch_tokens * 4)
        //   cost_gain = ab.gain * 0.5
        //   _current_batch_cost = max_batch_tokens
        let cfg = BatchConfig::default(); // max_batch_cost = 16_384, max_batch_wait_ms = 15.0
        let c = AdaptiveBatchController::from_batch_config(&cfg);

        assert!(
            (c.min_wait_ms - 15.0).abs() < f64::EPSILON,
            "min_wait_ms should match Python production (15.0), got {}",
            c.min_wait_ms
        );
        assert!(
            (c.initial_wait_ms - 15.0).abs() < f64::EPSILON,
            "initial_wait_ms should equal cfg.max_batch_wait_ms (15.0)"
        );
        assert!(
            (c.current_wait_ms - 15.0).abs() < f64::EPSILON,
            "current_wait_ms should be initialised to initial_wait_ms"
        );
        assert_eq!(
            c.min_batch_cost, 4_096,
            "min_batch_cost should equal max_batch_tokens / 4 = 16384/4 = 4096"
        );
        assert_eq!(
            c.max_batch_cost, 65_536,
            "max_batch_cost should equal max_batch_tokens * 4 = 16384*4 = 65536"
        );
        assert_eq!(
            c.initial_batch_cost, 16_384,
            "initial_batch_cost should equal max_batch_tokens"
        );
        assert!(
            (c.cost_gain - (c.gain * 0.5)).abs() < f64::EPSILON,
            "cost_gain should be coupled to gain * 0.5 (Python production wiring)"
        );
    }

    #[test]
    fn from_batch_config_min_cost_floor_at_256_for_tiny_models() {
        // Tiny model with max_batch_tokens=512: 512/4 = 128, but the
        // Python floor is `max(256, max_batch_tokens // 4)` so the
        // floor must clamp to 256 here. Then the further `min(floor,
        // max_batch_tokens)` keeps it from exceeding the model
        // budget itself.
        let cfg = BatchConfig {
            max_batch_cost: 512,
            ..BatchConfig::default()
        };
        let c = AdaptiveBatchController::from_batch_config(&cfg);
        assert_eq!(c.min_batch_cost, 256, "tiny-model floor clamps up to 256");
        assert_eq!(
            c.max_batch_cost, 2_048,
            "max_batch_cost = max_batch_tokens * 4"
        );
    }

    #[test]
    fn from_batch_config_min_cost_clamps_to_budget_for_micro_models() {
        // Pathological model with max_batch_tokens=128: 128/4 = 32,
        // floor would clamp to 256 — but Python guards this with
        // `min(cost_floor, max_batch_tokens)` so the floor never
        // exceeds the model's own budget. Mirror that guard here.
        let cfg = BatchConfig {
            max_batch_cost: 128,
            ..BatchConfig::default()
        };
        let c = AdaptiveBatchController::from_batch_config(&cfg);
        assert_eq!(
            c.min_batch_cost, 128,
            "floor must not exceed max_batch_tokens for micro models"
        );
        assert_eq!(
            c.max_batch_cost, 512,
            "max_batch_cost = max_batch_tokens * 4 for micro models"
        );
    }

    #[test]
    fn from_batch_config_uses_cfg_max_batch_wait_ms() {
        // Operator who tunes `BatchConfig.max_batch_wait_ms` via
        // `SIE_BATCHER_MAX_BATCH_WAIT_MS` should see the controller's
        // initial wait pick that value up — Python's
        // `_current_wait_ms = self._config.max_batch_wait_ms`
        // captures the relationship.
        let cfg = BatchConfig {
            max_batch_wait_ms: 25.0,
            ..BatchConfig::default()
        };
        let c = AdaptiveBatchController::from_batch_config(&cfg);
        assert!((c.initial_wait_ms - 25.0).abs() < f64::EPSILON);
        assert!((c.current_wait_ms - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn from_batch_config_and_env_no_env_matches_from_batch_config() {
        // With no env vars set, env-overlay constructor and bare
        // production-parity constructor must agree exactly.
        with_env_lock(|| {
            clear_env();
            let cfg = BatchConfig::default();
            let bare = AdaptiveBatchController::from_batch_config(&cfg);
            let with_env = AdaptiveBatchController::from_batch_config_and_env(&cfg);

            assert!((bare.min_wait_ms - with_env.min_wait_ms).abs() < f64::EPSILON);
            assert!((bare.initial_wait_ms - with_env.initial_wait_ms).abs() < f64::EPSILON);
            assert!((bare.current_wait_ms - with_env.current_wait_ms).abs() < f64::EPSILON);
            assert_eq!(bare.min_batch_cost, with_env.min_batch_cost);
            assert_eq!(bare.max_batch_cost, with_env.max_batch_cost);
            assert!((bare.cost_gain - with_env.cost_gain).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn from_batch_config_and_env_max_wait_overrides_production_default() {
        // `SIE_ADAPTIVE_BATCH_MAX_WAIT_MS=10` must override the
        // production-parity default (50.0) and clamp `current_wait_ms`
        // down accordingly.
        with_env_lock(|| {
            clear_env();
            unsafe { std::env::set_var("SIE_ADAPTIVE_BATCH_MAX_WAIT_MS", "10.0") };
            let cfg = BatchConfig::default();
            let c = AdaptiveBatchController::from_batch_config_and_env(&cfg);
            assert!((c.max_wait_ms - 10.0).abs() < f64::EPSILON);
            assert!(
                c.current_wait_ms <= 10.0,
                "current_wait_ms must clamp to new ceiling"
            );
            clear_env();
        });
    }

    #[test]
    fn from_batch_config_and_env_min_wait_env_wins_over_production_default() {
        // Operator sets `SIE_ADAPTIVE_BATCH_MIN_WAIT_MS=5` to
        // experimentally drop the floor below Python production's
        // 15.0. Env must win.
        with_env_lock(|| {
            clear_env();
            unsafe { std::env::set_var("SIE_ADAPTIVE_BATCH_MIN_WAIT_MS", "5.0") };
            let cfg = BatchConfig::default();
            let c = AdaptiveBatchController::from_batch_config_and_env(&cfg);
            assert!((c.min_wait_ms - 5.0).abs() < f64::EPSILON);
            clear_env();
        });
    }

    #[test]
    fn explicit_target_skips_calibration() {
        let c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(25.0))
            .build();
        assert!(c.calibrated);
        assert_eq!(c.target_p50_ms(), Some(25.0));
        assert!(!c.auto_calibrate);
    }

    #[test]
    fn auto_calibration_derives_target_from_inference_p50() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(None) // auto-calibrate
            .calibration_multiplier(2.0)
            .update_interval(1)
            .build();
        assert!(!c.calibrated);

        // Seed the inference tracker with enough samples to cross
        // the min_samples threshold (10).
        for _ in 0..20 {
            c.record_inference_sample(10.0);
        }
        // First step with calibration pending: target gets set from
        // inference_p50 (=10ms) × multiplier (=2.0) = 20ms.
        let _ = c.step(None, None, Some(32));
        assert!(c.calibrated);
        assert_eq!(c.target_p50_ms(), Some(20.0));
    }

    #[test]
    fn auto_calibration_target_is_clamped() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(None)
            .calibration_multiplier(100.0) // absurd → will clamp high
            .min_target_p50_ms(5.0)
            .max_target_p50_ms(50.0)
            .update_interval(1)
            .build();
        for _ in 0..20 {
            c.record_inference_sample(5.0); // 5 * 100 = 500 → clamps to 50
        }
        c.step(None, None, Some(32));
        assert_eq!(c.target_p50_ms(), Some(50.0));
    }

    #[test]
    fn pre_calibration_holds_initial_knobs() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(None)
            .update_interval(1)
            .build();
        let (w0, b0) = (c.current_wait_ms(), c.current_batch_cost());
        // Step with no inference samples → no calibration → knobs held.
        let (w1, b1) = c.step(Some(999.0), Some(0.99), Some(1));
        assert_eq!(w0, w1);
        assert_eq!(b0, b1);
    }

    #[test]
    fn positive_headroom_grows_wait_knob() {
        let mut c = make_explicit_target(50.0);
        let (w0, _) = (c.current_wait_ms(), c.current_batch_cost());
        // observed=10ms < target=50ms → headroom=+40 → wait grows.
        let (w1, _) = c.step(Some(10.0), None, Some(32));
        assert!(w1 > w0, "positive headroom must grow wait: {w0} → {w1}");
    }

    #[test]
    fn negative_headroom_shrinks_wait_knob() {
        let mut c = make_explicit_target(10.0);
        let (w0, _) = (c.current_wait_ms(), c.current_batch_cost());
        // observed=50ms > target=10ms → headroom=-40 → wait shrinks.
        let (w1, _) = c.step(Some(50.0), None, Some(32));
        assert!(w1 < w0, "negative headroom must shrink wait: {w0} → {w1}");
    }

    #[test]
    fn wait_knob_clamped_to_bounds() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(50.0))
            .min_wait_ms(5.0)
            .max_wait_ms(30.0)
            .update_interval(1)
            .build();
        // Drive a huge positive headroom many times → must clamp at max.
        for _ in 0..100 {
            c.step(Some(0.0), None, Some(32));
        }
        assert!((c.current_wait_ms() - 30.0).abs() < 1e-9);

        // Flip: huge negative headroom → must clamp at min.
        for _ in 0..100 {
            c.step(Some(999.0), None, Some(32));
        }
        assert!((c.current_wait_ms() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn cost_knob_only_adjusts_when_fill_ratio_above_threshold() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(50.0))
            .fill_ratio_threshold(0.7)
            .update_interval(1)
            .build();
        let b0 = c.current_batch_cost();

        // fill_ratio below threshold → cost must not change.
        let (_, b1) = c.step(Some(10.0), Some(0.5), Some(32));
        assert_eq!(b1, b0);

        // fill_ratio above threshold → cost can move.
        let (_, b2) = c.step(Some(10.0), Some(0.9), Some(32));
        assert!(
            b2 > b0,
            "saturated + positive headroom must grow cost: {b0} → {b2}"
        );
    }

    #[test]
    fn cost_knob_clamped_to_bounds() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(50.0))
            .min_batch_cost(1000)
            .max_batch_cost(5000)
            .initial_batch_cost(2500)
            .fill_ratio_threshold(0.1) // always saturated
            .cost_gain(1.0) // big moves
            .update_interval(1)
            .build();

        for _ in 0..100 {
            c.step(Some(0.0), Some(0.9), Some(32));
        }
        assert!(c.current_batch_cost() <= 5000);
        assert!(c.current_batch_cost() >= 1000);
    }

    #[test]
    fn starvation_recovery_fires_at_floors_with_streak() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(10.0))
            .min_wait_ms(1.0)
            .max_wait_ms(50.0)
            .min_batch_cost(100)
            .max_batch_cost(10_000)
            .initial_wait_ms(20.0)
            .initial_batch_cost(4000)
            .starvation_window(5)
            .starvation_batch_size(1)
            .update_interval(1)
            .build();

        // Manually pin knobs at floors (would take many negative-headroom
        // steps to get there organically; we short-circuit for the test).
        c.current_wait_ms = c.min_wait_ms;
        c.current_batch_cost = c.min_batch_cost;

        // Feed 5 tiny batches — streak threshold met.
        for _ in 0..c.starvation_window {
            c.step(Some(100.0), Some(1.0), Some(1));
        }
        assert!(
            c.starvation_resets() >= 1,
            "at floors + streak should trigger recovery"
        );
        // Recovery wait = (1.0 + 20.0)/2 = 10.5
        assert!((c.current_wait_ms() - 10.5).abs() < 1e-9);
        // Recovery cost = max(4000/2=2000, 100*4=400) = 2000
        assert_eq!(c.current_batch_cost(), 2000);
        // Streak is cleared after reset.
        assert_eq!(c.starvation_streak(), 0);
    }

    #[test]
    fn starvation_not_triggered_if_not_at_floors() {
        // Feed observed == target so headroom is zero and the PI
        // loop never drives either knob off its initial value.
        // (Large negative headroom, as in the sibling test, crashes
        // both knobs into their floors on the first step, which
        // would trigger recovery — that path is covered by
        // `starvation_recovery_fires_at_floors_with_streak`.)
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(10.0))
            .starvation_window(3)
            .update_interval(1)
            .build();
        for _ in 0..10 {
            c.step(Some(10.0), Some(0.0 /* below fill threshold */), Some(1));
        }
        assert_eq!(c.starvation_resets(), 0);
        assert!(c.current_wait_ms() > c.min_wait_ms);
        assert!(c.current_batch_cost() > c.min_batch_cost);
    }

    #[test]
    fn starvation_counter_resets_on_healthy_batch() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(10.0))
            .starvation_window(5)
            .starvation_batch_size(1)
            .update_interval(1)
            .build();
        c.current_wait_ms = c.min_wait_ms;
        c.current_batch_cost = c.min_batch_cost;

        // 4 tiny batches — one short of recovery.
        for _ in 0..4 {
            c.step(Some(100.0), Some(1.0), Some(1));
        }
        assert_eq!(c.starvation_resets(), 0);

        // One healthy batch resets the streak.
        c.step(Some(100.0), Some(1.0), Some(32));
        assert_eq!(c.starvation_streak(), 0);

        // Next 4 tiny batches must not trigger — streak restarts.
        for _ in 0..4 {
            c.step(Some(100.0), Some(1.0), Some(1));
        }
        assert_eq!(c.starvation_resets(), 0);
    }

    #[test]
    fn starvation_recovery_can_be_disabled() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(10.0))
            .starvation_recovery_enabled(false)
            .starvation_window(2)
            .update_interval(1)
            .build();
        c.current_wait_ms = c.min_wait_ms;
        c.current_batch_cost = c.min_batch_cost;
        for _ in 0..50 {
            c.step(Some(100.0), Some(1.0), Some(1));
        }
        assert_eq!(c.starvation_resets(), 0);
    }

    #[test]
    fn none_batch_size_does_not_touch_streak() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(10.0))
            .starvation_window(3)
            .update_interval(1)
            .build();
        c.current_wait_ms = c.min_wait_ms;
        c.current_batch_cost = c.min_batch_cost;
        for _ in 0..10 {
            c.step(Some(100.0), Some(1.0), None);
        }
        assert_eq!(c.starvation_streak(), 0);
        assert_eq!(c.starvation_resets(), 0);
    }

    #[test]
    fn reset_restores_initial_state_and_recovers_autocal() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(None)
            .update_interval(1)
            .initial_wait_ms(15.0)
            .initial_batch_cost(2000)
            .build();
        // Calibrate manually.
        for _ in 0..20 {
            c.record_inference_sample(10.0);
        }
        c.step(None, None, Some(32));
        assert!(c.calibrated());
        // Drive the knobs around.
        for _ in 0..5 {
            c.step(Some(0.0), Some(0.9), Some(32));
        }
        // Bump the `starvation_resets` counter so we can verify it's
        // NOT reset.
        c.starvation_resets = 3;

        c.reset();
        assert_eq!(c.current_wait_ms(), 15.0);
        assert_eq!(c.current_batch_cost(), 2000);
        assert!(
            !c.calibrated(),
            "auto-calibrate controllers re-enter calibration"
        );
        assert_eq!(c.target_p50_ms(), None);
        assert_eq!(
            c.starvation_resets(),
            3,
            "starvation_resets is monotonic across reset"
        );
    }

    #[test]
    fn reset_preserves_calibrated_target_for_explicit_target() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(30.0))
            .update_interval(1)
            .build();
        assert!(c.calibrated());
        c.reset();
        assert!(
            c.calibrated(),
            "explicit-target controllers keep calibrated=true across reset"
        );
        assert_eq!(c.target_p50_ms(), Some(30.0));
    }

    #[test]
    fn update_interval_gates_controller_steps() {
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(50.0))
            .update_interval(5) // only every 5th call moves the knobs
            .build();
        let w0 = c.current_wait_ms();

        // 4 steps → no movement.
        for _ in 0..4 {
            c.step(Some(10.0), None, Some(32));
        }
        assert_eq!(c.current_wait_ms(), w0);

        // 5th step → move.
        c.step(Some(10.0), None, Some(32));
        assert!(c.current_wait_ms() > w0);
    }

    #[test]
    fn snapshot_reports_headroom_and_state() {
        let c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(20.0))
            .build();
        let s = c.snapshot(Some(15.0), Some(0.8));
        assert!(s.enabled);
        assert!(s.calibrated);
        assert_eq!(s.target_p50_ms, Some(20.0));
        assert_eq!(s.observed_p50_ms, Some(15.0));
        assert_eq!(s.fill_ratio, Some(0.8));
        assert_eq!(s.headroom_ms, Some(5.0));
    }

    #[test]
    fn integrator_saturation_antiwindup_blocks_at_max() {
        // Drive max headroom long enough to saturate wait at max.
        // Then confirm the integrator doesn't keep growing.
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(100.0))
            .min_wait_ms(1.0)
            .max_wait_ms(5.0) // tiny range, easy to saturate
            .gain(2.0)
            .integral_gain(1.0)
            .update_interval(1)
            .build();
        for _ in 0..50 {
            c.step(Some(0.0), None, Some(32));
        }
        let integ_saturated = c.integral;
        // Another batch of saturated steps — integral must not grow
        // further (saturation anti-windup).
        for _ in 0..50 {
            c.step(Some(0.0), None, Some(32));
        }
        assert!(
            (c.integral - integ_saturated).abs() < 1e-9,
            "integrator must stop growing while output is saturated at max"
        );
    }

    #[test]
    fn idle_decay_shrinks_integrator_after_long_gap() {
        // This is a timing-sensitive test; we synthesise the gap by
        // mutating `last_step_time` directly instead of sleeping.
        let mut c = AdaptiveBatchController::builder()
            .target_p50_ms(Some(50.0))
            .integral_gain(1.0)
            .update_interval(1)
            .build();
        // One step to seed `last_step_time`.
        c.step(Some(0.0), None, Some(32));
        // Manually build up integral without the decay firing.
        c.integral = 10.0;
        // Back-date the last_step_time by 5 seconds.
        c.last_step_time = Some(Instant::now() - Duration::from_secs(5));
        // Next step — dt ≈ 5s, decay should apply (0.5^(5-2) = 0.125).
        c.step(Some(50.0), None, Some(32)); // headroom=0 so no fresh integration
        assert!(
            c.integral < 10.0 * 0.5,
            "idle decay must shrink the integrator: got {}",
            c.integral
        );
    }
}
