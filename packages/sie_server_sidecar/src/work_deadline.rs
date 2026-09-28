//! Gateway-stamped work-item deadlines.
//!
//! `WorkItem::deadline` is an absolute Unix-epoch instant in seconds on the
//! gateway clock that also stamps `WorkItem::timestamp`. The sidecar compares
//! it with its own wall clock plus a bounded skew tolerance, so both hosts must
//! keep their clocks synchronised (for example with NTP). The comparison
//! drives three decisions:
//!
//! * dropping NATS deliveries nobody waits for before payload fetch or backend
//!   IPC. This is opt-in with `SIE_WORK_DEADLINE_ENFORCE=true`; by default the
//!   decision is only counted and logged;
//! * keeping a held delivery's JetStream lease alive with progress ACKs until
//!   it settles or its lease horizon passes;
//! * giving a backend `RunBatch` call up to the batch's latest deadline, within
//!   a configured ceiling, instead of only the fixed IPC request timeout.
//!
//! Items without a plausible deadline keep the previous behaviour on every
//! path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_nats::jetstream::{AckKind, Message};
use async_nats::Subject;
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tracing::debug;

use crate::nats_consumer::{work_cancel_tombstone_ttl, ACK_WAIT_SECS};
use crate::observability::metrics::SidecarTelemetry;
use crate::readiness::Readiness;

const ENFORCE_ENV: &str = "SIE_WORK_DEADLINE_ENFORCE";
const SKEW_TOLERANCE_ENV: &str = "SIE_WORK_DEADLINE_SKEW_TOLERANCE_MS";
const MAX_BUDGET_ENV: &str = "SIE_WORK_DEADLINE_MAX_BUDGET_S";
const DEFAULT_SKEW_TOLERANCE: Duration = Duration::from_secs(5);
const MAX_SKEW_TOLERANCE: Duration = Duration::from_secs(60);
/// The gateway's default 120 s request timeout plus a margin.
const DEFAULT_MAX_BUDGET: Duration = Duration::from_secs(180);
const WARN_INTERVAL: Duration = Duration::from_secs(30);

/// A held delivery is progress-ACKed at most two ticks apart, which must stay
/// below half of the pool consumer `ack_wait`.
pub(crate) const PROGRESS_TICK: Duration = Duration::from_secs(5);
const _: () = assert!(PROGRESS_TICK.as_secs() * 2 < ACK_WAIT_SECS / 2);

/// Where an item stands against its deadline, including the skew tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlineStatus {
    /// No usable deadline: the item predates deadlines, or its deadline is
    /// not a finite instant within the maximum budget of its timestamp.
    Unbounded,
    /// Time left until the deadline.
    Live(Duration),
    /// How long ago the deadline passed.
    Expired(Duration),
}

/// Evidence at intake that the gateway and worker clocks disagree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClockSkewSignal {
    /// The publish timestamp is ahead of the worker clock by more than the
    /// skew tolerance, so the worker clock is likely behind the gateway's.
    TimestampAhead { ahead_ms: u64 },
    /// A first delivery is already past its deadline: the item either waited
    /// in the stream longer than its whole budget, or the worker clock is
    /// ahead of the gateway's.
    FirstDeliveryExpired { overdue_ms: u64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkDeadlinePolicy {
    /// When false (the default), expired items are counted and logged but
    /// still executed.
    pub enforce: bool,
    /// Allowed wall-clock skew between gateway and worker hosts.
    pub skew_tolerance: Duration,
    /// Largest accepted `deadline - timestamp`, and the ceiling on any
    /// deadline-derived backend budget.
    pub max_budget: Duration,
    /// A queued item cannot outlive the work stream, so no lease is held
    /// longer than this.
    pub max_horizon: Duration,
}

impl WorkDeadlinePolicy {
    pub fn from_env() -> Self {
        Self::from_values(
            std::env::var(ENFORCE_ENV).ok().as_deref(),
            std::env::var(SKEW_TOLERANCE_ENV).ok().as_deref(),
            std::env::var(MAX_BUDGET_ENV).ok().as_deref(),
            work_cancel_tombstone_ttl(),
        )
    }

    fn from_values(
        enforce: Option<&str>,
        skew_tolerance_ms: Option<&str>,
        max_budget_s: Option<&str>,
        max_horizon: Duration,
    ) -> Self {
        let enforce = enforce.is_some_and(|raw| {
            matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        });
        let skew_tolerance = skew_tolerance_ms
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_SKEW_TOLERANCE)
            .min(MAX_SKEW_TOLERANCE);
        let max_budget = max_budget_s
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .filter(|seconds| *seconds > 0)
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_MAX_BUDGET)
            .min(max_horizon);
        Self {
            enforce,
            skew_tolerance,
            max_budget,
            max_horizon,
        }
    }

    fn plausible_deadline(&self, deadline: Option<f64>, timestamp: f64) -> Option<f64> {
        let deadline = deadline.filter(|value| value.is_finite() && *value > 0.0)?;
        if !timestamp.is_finite() || timestamp <= 0.0 {
            return None;
        }
        let budget = deadline - timestamp;
        (0.0..=self.max_budget.as_secs_f64())
            .contains(&budget)
            .then_some(deadline)
    }

    pub fn status(&self, deadline: Option<f64>, timestamp: f64, now_unix_s: f64) -> DeadlineStatus {
        let Some(deadline) = self.plausible_deadline(deadline, timestamp) else {
            return DeadlineStatus::Unbounded;
        };
        let remaining_s = deadline + self.skew_tolerance.as_secs_f64() - now_unix_s;
        let ceiling = self.max_budget + self.skew_tolerance;
        if remaining_s > 0.0 {
            DeadlineStatus::Live(
                Duration::try_from_secs_f64(remaining_s)
                    .unwrap_or(ceiling)
                    .min(ceiling),
            )
        } else {
            DeadlineStatus::Expired(
                Duration::try_from_secs_f64(-remaining_s).unwrap_or(Duration::MAX),
            )
        }
    }

    /// How long a held delivery's JetStream lease may be kept alive, or `None`
    /// for no lease. When enforcing, a redelivery after the deadline is
    /// dropped anyway, so the lease lapses there and a stuck holder can still
    /// be recovered. Without enforcement expired work still executes, so the
    /// lease lasts for one more maximum budget past the deadline.
    pub fn lease_horizon(
        &self,
        deadline: Option<f64>,
        timestamp: f64,
        now_unix_s: f64,
    ) -> Option<Duration> {
        let status = self.status(deadline, timestamp, now_unix_s);
        let horizon = match status {
            DeadlineStatus::Unbounded => return None,
            DeadlineStatus::Live(remaining) if self.enforce => remaining,
            DeadlineStatus::Expired(_) if self.enforce => return None,
            DeadlineStatus::Live(remaining) => remaining + self.max_budget,
            DeadlineStatus::Expired(overdue) => self.max_budget.checked_sub(overdue)?,
        };
        Some(horizon.min(self.max_horizon)).filter(|horizon| !horizon.is_zero())
    }

    /// Time until the latest live deadline among `(deadline, timestamp)`
    /// pairs, capped at the maximum budget, or `None` when no item carries a
    /// live deadline.
    pub fn run_batch_budget(
        &self,
        deadlines: impl IntoIterator<Item = (Option<f64>, f64)>,
        now_unix_s: f64,
    ) -> Option<Duration> {
        deadlines
            .into_iter()
            .filter_map(|(deadline, timestamp)| {
                match self.status(deadline, timestamp, now_unix_s) {
                    DeadlineStatus::Live(remaining) => Some(remaining.min(self.max_budget)),
                    DeadlineStatus::Unbounded | DeadlineStatus::Expired(_) => None,
                }
            })
            .max()
    }

    pub fn clock_skew_signal(
        &self,
        deadline: Option<f64>,
        timestamp: f64,
        first_delivery: bool,
        now_unix_s: f64,
    ) -> Option<ClockSkewSignal> {
        if matches!(
            self.status(deadline, timestamp, now_unix_s),
            DeadlineStatus::Unbounded
        ) {
            return None;
        }
        let ahead_s = timestamp - now_unix_s;
        if ahead_s > self.skew_tolerance.as_secs_f64() {
            return Some(ClockSkewSignal::TimestampAhead {
                ahead_ms: seconds_to_ms(ahead_s),
            });
        }
        match self.status(deadline, timestamp, now_unix_s) {
            DeadlineStatus::Expired(overdue) if first_delivery => {
                Some(ClockSkewSignal::FirstDeliveryExpired {
                    overdue_ms: u64::try_from(overdue.as_millis()).unwrap_or(u64::MAX),
                })
            }
            _ => None,
        }
    }
}

pub fn unix_now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or(0.0)
}

/// `now - timestamp` in milliseconds, clamped at zero, as a log field.
pub fn apparent_age_ms(timestamp: f64, now_unix_s: f64) -> u64 {
    if !timestamp.is_finite() || timestamp <= 0.0 {
        return 0;
    }
    seconds_to_ms(now_unix_s - timestamp)
}

fn seconds_to_ms(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * 1000.0).min(u64::MAX as f64) as u64
}

/// Allows one warning per interval and reports how many were suppressed
/// since the last one that was allowed.
pub struct WarnLimiter {
    interval_ms: u64,
    last_ms: AtomicU64,
    suppressed: AtomicU64,
}

const NEVER: u64 = u64::MAX;
static WARN_CLOCK_ORIGIN: LazyLock<Instant> = LazyLock::new(Instant::now);

impl WarnLimiter {
    pub const fn new(interval: Duration) -> Self {
        Self {
            interval_ms: interval.as_millis() as u64,
            last_ms: AtomicU64::new(NEVER),
            suppressed: AtomicU64::new(0),
        }
    }

    pub fn allow(&self) -> Option<u64> {
        self.allow_at(u64::try_from(WARN_CLOCK_ORIGIN.elapsed().as_millis()).unwrap_or(NEVER - 1))
    }

    fn allow_at(&self, now_ms: u64) -> Option<u64> {
        let last = self.last_ms.load(Ordering::Relaxed);
        let due = last == NEVER || now_ms.saturating_sub(last) >= self.interval_ms;
        if !due
            || self
                .last_ms
                .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
                .is_err()
        {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        Some(self.suppressed.swap(0, Ordering::Relaxed))
    }
}

pub static EXPIRED_DROP_WARNINGS: WarnLimiter = WarnLimiter::new(WARN_INTERVAL);
pub static EXPIRED_EXECUTE_WARNINGS: WarnLimiter = WarnLimiter::new(WARN_INTERVAL);
pub static CLOCK_SKEW_WARNINGS: WarnLimiter = WarnLimiter::new(WARN_INTERVAL);

/// Serialises a lease's progress ACKs with its settlement, so a progress ACK
/// is either sent before the settling ACK or NAK or not at all.
#[derive(Default)]
struct LeaseGate {
    settled: AtomicBool,
    publishing: tokio::sync::Mutex<()>,
}

/// JetStream leases kept alive by progress ACKs. A lease ends when its
/// delivery settles, when its [`ProgressLease`] token is dropped, or at its
/// horizon, after which JetStream redelivers as it would without a lease.
pub(crate) struct ProgressLeases<T> {
    leases: Mutex<HashMap<u64, HeldLease<T>>>,
    next_id: AtomicU64,
}

struct HeldLease<T> {
    target: T,
    gate: Arc<LeaseGate>,
    horizon: Instant,
    last_progress: Instant,
}

/// Owner of one held lease. Dropping it ends the lease, so a delivery that is
/// abandoned without an ACK or NAK stops being progress-ACKed.
pub struct ProgressLease<T> {
    id: u64,
    gate: Arc<LeaseGate>,
    leases: Arc<ProgressLeases<T>>,
}

impl<T> ProgressLease<T> {
    /// End the lease before its delivery is ACKed or NAKed. Waits for a
    /// progress ACK already being sent for it, so none follows the settlement.
    pub async fn settle(&self) {
        let publishing = self.gate.publishing.lock().await;
        self.gate.settled.store(true, Ordering::Release);
        drop(publishing);
        self.leases.release(self.id);
    }
}

impl<T> Drop for ProgressLease<T> {
    fn drop(&mut self) {
        self.gate.settled.store(true, Ordering::Release);
        self.leases.release(self.id);
    }
}

impl<T> ProgressLeases<T> {
    fn new() -> Self {
        Self {
            leases: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
        }
    }

    fn release(&self, id: u64) {
        self.leases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id);
    }
}

impl<T: Clone> ProgressLeases<T> {
    fn hold(self: &Arc<Self>, target: T, horizon: Instant, now: Instant) -> ProgressLease<T> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let gate = Arc::new(LeaseGate::default());
        self.leases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                id,
                HeldLease {
                    target,
                    gate: Arc::clone(&gate),
                    horizon,
                    last_progress: now,
                },
            );
        ProgressLease {
            id,
            gate,
            leases: Arc::clone(self),
        }
    }

    /// Leases whose last progress is at least one tick old, marked as
    /// progressed at `now`. Leases past their horizon are dropped.
    fn due(&self, now: Instant) -> Vec<(T, Arc<LeaseGate>)> {
        let mut leases = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        leases.retain(|_, lease| lease.horizon > now);
        leases
            .values_mut()
            .filter(|lease| now.saturating_duration_since(lease.last_progress) >= PROGRESS_TICK)
            .map(|lease| {
                lease.last_progress = now;
                (lease.target.clone(), Arc::clone(&lease.gate))
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct NatsProgressTarget {
    client: async_nats::Client,
    reply: Subject,
}

pub type NatsProgressLease = ProgressLease<NatsProgressTarget>;

/// Process-wide progress leases for held NATS deliveries.
pub(crate) struct NatsProgressLeases {
    leases: Arc<ProgressLeases<NatsProgressTarget>>,
    ticker: Mutex<Option<JoinHandle<()>>>,
    backend_readiness: OnceLock<Arc<Readiness>>,
}

static NATS_PROGRESS_LEASES: LazyLock<NatsProgressLeases> = LazyLock::new(NatsProgressLeases::new);

pub(crate) fn nats_progress_leases() -> &'static NatsProgressLeases {
    &NATS_PROGRESS_LEASES
}

impl NatsProgressLeases {
    fn new() -> Self {
        Self {
            leases: Arc::new(ProgressLeases::new()),
            ticker: Mutex::new(None),
            backend_readiness: OnceLock::new(),
        }
    }

    /// Only extend leases while the backend answers its heartbeat and the
    /// sidecar is not draining, so JetStream can move work off a worker whose
    /// backend stopped responding.
    pub(crate) fn gate_on_backend_readiness(&self, readiness: Arc<Readiness>) {
        let _ = self.backend_readiness.set(readiness);
    }

    fn backend_accepts_progress(&self) -> bool {
        self.backend_readiness
            .get()
            .is_none_or(|readiness| readiness.snapshot().is_ready())
    }

    pub(crate) fn hold(
        &self,
        msg: &Message,
        horizon: Duration,
        telemetry: &SidecarTelemetry,
    ) -> Option<NatsProgressLease> {
        let reply = msg.reply.clone()?;
        let now = Instant::now();
        let lease = self.leases.hold(
            NatsProgressTarget {
                client: msg.context.client(),
                reply,
            },
            now.checked_add(horizon)?,
            now,
        );
        self.ensure_ticker(telemetry);
        Some(lease)
    }

    fn ensure_ticker(&self, telemetry: &SidecarTelemetry) {
        let mut ticker = self.ticker.lock().unwrap_or_else(|e| e.into_inner());
        if ticker.as_ref().is_some_and(|handle| !handle.is_finished()) {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        *ticker = Some(runtime.spawn(run_progress_ticker(telemetry.clone())));
    }
}

async fn run_progress_ticker(telemetry: SidecarTelemetry) {
    let leases = nats_progress_leases();
    let mut tick = tokio::time::interval(PROGRESS_TICK);
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        if !leases.backend_accepts_progress() {
            continue;
        }
        for (target, gate) in leases.leases.due(Instant::now()) {
            let publishing = gate.publishing.lock().await;
            if gate.settled.load(Ordering::Acquire) {
                continue;
            }
            let result = target
                .client
                .publish(target.reply, AckKind::Progress.into())
                .await;
            drop(publishing);
            telemetry.nats_operation(
                "progress",
                if result.is_ok() { "success" } else { "error" },
                "none",
                1,
            );
            if let Err(error) = result {
                debug!(error = %error, "held delivery progress ACK failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl<T> ProgressLeases<T> {
        fn len(&self) -> usize {
            self.leases.lock().unwrap_or_else(|e| e.into_inner()).len()
        }
    }

    const HORIZON: Duration = Duration::from_secs(1_800);
    const NOW: f64 = 1_700_000_000.0;

    fn policy(enforce: Option<&str>, skew_ms: Option<&str>) -> WorkDeadlinePolicy {
        WorkDeadlinePolicy::from_values(enforce, skew_ms, None, HORIZON)
    }

    #[test]
    fn enforcement_is_opt_in_and_limits_are_bounded() {
        let default = policy(None, None);
        assert!(!default.enforce);
        assert_eq!(default.skew_tolerance, DEFAULT_SKEW_TOLERANCE);
        assert_eq!(default.max_budget, DEFAULT_MAX_BUDGET);
        assert_eq!(default.max_horizon, HORIZON);

        for on in ["true", "1", " YES ", "on"] {
            assert!(policy(Some(on), None).enforce, "{on}");
        }
        for off in ["false", "0", "no", "anything"] {
            assert!(!policy(Some(off), None).enforce, "{off}");
        }
        assert_eq!(
            policy(None, Some("250")).skew_tolerance,
            Duration::from_millis(250)
        );
        assert_eq!(
            policy(None, Some("-1")).skew_tolerance,
            DEFAULT_SKEW_TOLERANCE
        );
        assert_eq!(
            policy(None, Some("86400000")).skew_tolerance,
            MAX_SKEW_TOLERANCE
        );
        let custom = WorkDeadlinePolicy::from_values(None, None, Some("600"), HORIZON);
        assert_eq!(custom.max_budget, Duration::from_secs(600));
        let capped = WorkDeadlinePolicy::from_values(None, None, Some("99999"), HORIZON);
        assert_eq!(capped.max_budget, HORIZON);
        let invalid = WorkDeadlinePolicy::from_values(None, None, Some("0"), HORIZON);
        assert_eq!(invalid.max_budget, DEFAULT_MAX_BUDGET);
    }

    #[test]
    fn implausible_deadlines_are_treated_as_absent() {
        let policy = policy(Some("true"), None);
        let timestamp = NOW - 1.0;
        for deadline in [
            None,
            Some(0.0),
            Some(-5.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(timestamp - 1.0),
            Some(timestamp + DEFAULT_MAX_BUDGET.as_secs_f64() + 1.0),
        ] {
            assert_eq!(
                policy.status(deadline, timestamp, NOW),
                DeadlineStatus::Unbounded,
                "{deadline:?}"
            );
        }
        assert_eq!(
            policy.status(Some(NOW + 10.0), 0.0, NOW),
            DeadlineStatus::Unbounded,
            "a deadline without a publish timestamp has no checkable budget"
        );
    }

    #[test]
    fn deadline_expires_only_after_the_skew_tolerance() {
        let policy = policy(None, Some("2000"));
        let timestamp = NOW - 60.0;
        let deadline = Some(NOW);

        assert_eq!(
            policy.status(deadline, timestamp, NOW - 10.0),
            DeadlineStatus::Live(Duration::from_secs(12))
        );
        assert_eq!(
            policy.status(deadline, timestamp, NOW + 1.5),
            DeadlineStatus::Live(Duration::from_millis(500))
        );
        assert_eq!(
            policy.status(deadline, timestamp, NOW + 4.0),
            DeadlineStatus::Expired(Duration::from_secs(2))
        );
    }

    #[test]
    fn live_time_is_capped_by_the_maximum_budget() {
        let policy = policy(None, Some("0"));
        assert_eq!(
            policy.status(Some(NOW + 120.0), NOW, NOW - 3_600.0),
            DeadlineStatus::Live(DEFAULT_MAX_BUDGET)
        );
    }

    #[test]
    fn lease_lapses_at_the_deadline_only_while_enforcing() {
        let timestamp = NOW - 1.0;
        let enforcing = policy(Some("true"), Some("0"));
        assert_eq!(enforcing.lease_horizon(None, timestamp, NOW), None);
        assert_eq!(
            enforcing.lease_horizon(Some(NOW + 30.0), timestamp, NOW),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            enforcing.lease_horizon(Some(NOW - 0.5), timestamp - 60.0, NOW),
            None
        );

        let shadow = policy(None, Some("0"));
        assert_eq!(shadow.lease_horizon(None, timestamp, NOW), None);
        assert_eq!(
            shadow.lease_horizon(Some(NOW + 30.0), timestamp, NOW),
            Some(Duration::from_secs(30) + DEFAULT_MAX_BUDGET)
        );
        assert_eq!(
            shadow.lease_horizon(Some(NOW - 30.0), NOW - 90.0, NOW),
            Some(DEFAULT_MAX_BUDGET - Duration::from_secs(30))
        );
        assert_eq!(
            shadow.lease_horizon(Some(NOW - 400.0), NOW - 460.0, NOW),
            None,
            "a lease is never taken past one maximum budget beyond the deadline"
        );
    }

    #[test]
    fn run_batch_budget_follows_the_latest_live_deadline_within_the_ceiling() {
        let policy = WorkDeadlinePolicy::from_values(None, Some("0"), Some("100"), HORIZON);
        let timestamp = NOW - 1.0;

        assert_eq!(
            policy.run_batch_budget([(None, timestamp), (None, timestamp)], NOW),
            None
        );
        assert_eq!(
            policy.run_batch_budget([(Some(NOW - 1.0), timestamp - 5.0)], NOW),
            None
        );
        assert_eq!(
            policy.run_batch_budget(
                [
                    (None, timestamp),
                    (Some(NOW + 30.0), timestamp),
                    (Some(NOW + 90.0), timestamp),
                    (Some(NOW - 5.0), timestamp - 10.0),
                ],
                NOW
            ),
            Some(Duration::from_secs(90))
        );
        assert_eq!(
            policy.run_batch_budget([(Some(NOW + 99.0), NOW - 1.0)], NOW - 50.0),
            Some(Duration::from_secs(100)),
            "a worker clock running behind cannot stretch the budget past the ceiling"
        );
        assert_eq!(
            policy.run_batch_budget([(Some(NOW + 500.0), timestamp)], NOW),
            None,
            "a deadline beyond the maximum budget is ignored"
        );
    }

    #[test]
    fn clock_skew_is_reported_in_both_directions() {
        let policy = policy(None, Some("2000"));
        assert_eq!(
            policy.clock_skew_signal(Some(NOW + 130.0), NOW + 10.0, true, NOW),
            Some(ClockSkewSignal::TimestampAhead { ahead_ms: 10_000 })
        );
        assert_eq!(
            policy.clock_skew_signal(Some(NOW + 121.0), NOW + 1.0, true, NOW),
            None,
            "skew inside the tolerance is not reported"
        );
        assert_eq!(
            policy.clock_skew_signal(Some(NOW - 8.0), NOW - 128.0, true, NOW),
            Some(ClockSkewSignal::FirstDeliveryExpired { overdue_ms: 6_000 })
        );
        assert_eq!(
            policy.clock_skew_signal(Some(NOW - 8.0), NOW - 128.0, false, NOW),
            None,
            "a redelivery can legitimately be late"
        );
        assert_eq!(
            policy.clock_skew_signal(None, NOW + 60.0, true, NOW),
            None,
            "items without a deadline are not judged"
        );
    }

    #[test]
    fn apparent_age_is_non_negative_milliseconds() {
        assert_eq!(apparent_age_ms(NOW - 1.5, NOW), 1_500);
        assert_eq!(apparent_age_ms(NOW + 10.0, NOW), 0);
        assert_eq!(apparent_age_ms(0.0, NOW), 0);
    }

    #[test]
    fn warnings_are_rate_limited_and_report_suppressed_counts() {
        let limiter = WarnLimiter::new(Duration::from_secs(30));
        assert_eq!(limiter.allow_at(0), Some(0));
        assert_eq!(limiter.allow_at(1_000), None);
        assert_eq!(limiter.allow_at(29_999), None);
        assert_eq!(limiter.allow_at(30_000), Some(2));
        assert_eq!(limiter.allow_at(30_001), None);
        assert_eq!(limiter.allow_at(90_000), Some(1));
    }

    #[test]
    fn held_leases_are_progressed_on_each_tick_until_released_or_expired() {
        let leases = Arc::new(ProgressLeases::<u8>::new());
        let start = Instant::now();
        let a = leases.hold(1, start + Duration::from_secs(60), start);
        let _b = leases.hold(2, start + Duration::from_secs(12), start);

        assert!(leases.due(start + Duration::from_secs(4)).is_empty());

        let mut due: Vec<u8> = leases
            .due(start + PROGRESS_TICK)
            .into_iter()
            .map(|(target, _)| target)
            .collect();
        due.sort();
        assert_eq!(due, vec![1, 2]);
        assert!(
            leases
                .due(start + PROGRESS_TICK + Duration::from_secs(1))
                .is_empty(),
            "a lease progressed on this tick waits for the next one"
        );

        drop(a);
        let due: Vec<u8> = leases
            .due(start + PROGRESS_TICK * 2)
            .into_iter()
            .map(|(target, _)| target)
            .collect();
        assert_eq!(due, vec![2]);

        assert!(
            leases.due(start + Duration::from_secs(12)).is_empty(),
            "a lease stops at its horizon so JetStream can redeliver"
        );
        assert_eq!(leases.len(), 0);
    }

    #[test]
    fn dropping_a_lease_token_ends_the_lease() {
        let leases = Arc::new(ProgressLeases::<u8>::new());
        let start = Instant::now();
        let lease = leases.hold(7, start + Duration::from_secs(600), start);
        let gate = Arc::clone(&lease.gate);
        assert_eq!(leases.len(), 1);

        drop(lease);

        assert_eq!(leases.len(), 0);
        assert!(gate.settled.load(Ordering::Acquire));
        assert!(leases.due(start + PROGRESS_TICK).is_empty());
    }

    #[tokio::test]
    async fn settlement_waits_for_an_in_flight_progress_ack() {
        let leases = Arc::new(ProgressLeases::<u8>::new());
        let start = Instant::now();
        let lease = leases.hold(9, start + Duration::from_secs(600), start);
        let (_, gate) = leases
            .due(start + PROGRESS_TICK)
            .pop()
            .expect("the lease is due");
        let publishing = gate.publishing.lock().await;

        let settle = lease.settle();
        tokio::pin!(settle);
        assert!(
            tokio::time::timeout(Duration::from_millis(20), &mut settle)
                .await
                .is_err(),
            "settlement must wait while a progress ACK is being sent"
        );
        drop(publishing);
        settle.await;

        assert!(gate.settled.load(Ordering::Acquire));
        assert_eq!(leases.len(), 0);
    }

    #[test]
    fn a_lease_is_never_left_unprogressed_for_half_the_ack_wait() {
        let leases = Arc::new(ProgressLeases::<u8>::new());
        let start = Instant::now();
        let _lease = leases.hold(0, start + Duration::from_secs(600), start);
        let mut last_progress = start;
        let mut tick = start;
        for _ in 0..100 {
            tick += PROGRESS_TICK;
            if !leases.due(tick).is_empty() {
                assert!(
                    tick - last_progress < Duration::from_secs(ACK_WAIT_SECS / 2),
                    "lease went {:?} without a progress ACK",
                    tick - last_progress
                );
                last_progress = tick;
            }
        }
        assert!(last_progress > start);
    }

    #[test]
    fn progress_pauses_while_the_backend_is_not_ready() {
        let leases = NatsProgressLeases::new();
        assert!(leases.backend_accepts_progress());

        let readiness = Arc::new(Readiness::new(2_000, 3));
        leases.gate_on_backend_readiness(Arc::clone(&readiness));
        assert!(
            !leases.backend_accepts_progress(),
            "no successful backend ping yet"
        );

        readiness.record_ping_success();
        assert!(leases.backend_accepts_progress());

        readiness.mark_draining();
        assert!(!leases.backend_accepts_progress());
    }
}
