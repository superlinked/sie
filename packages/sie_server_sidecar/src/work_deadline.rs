//! Gateway-stamped work-item deadlines.
//!
//! `WorkItem::deadline` is an absolute Unix-epoch instant in seconds on the
//! gateway clock. The sidecar compares it with its own wall clock plus a
//! configured skew tolerance and uses the result for three decisions:
//!
//! * ACK-dropping NATS deliveries nobody waits for before payload fetch or
//!   backend IPC (`SIE_WORK_DEADLINE_ENFORCE=false` only logs the decision);
//! * keeping a held delivery's JetStream lease alive with progress ACKs until
//!   it settles or its deadline passes;
//! * giving a backend `RunBatch` call at least until the batch's latest
//!   deadline instead of only the fixed IPC request timeout.
//!
//! Items without a deadline keep the previous behaviour on every path.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_nats::jetstream::{AckKind, Message};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tracing::debug;

use crate::nats_consumer::{work_cancel_tombstone_ttl, ACK_WAIT_SECS};
use crate::observability::metrics::SidecarTelemetry;

const ENFORCE_ENV: &str = "SIE_WORK_DEADLINE_ENFORCE";
const SKEW_TOLERANCE_ENV: &str = "SIE_WORK_DEADLINE_SKEW_TOLERANCE_MS";
const DEFAULT_SKEW_TOLERANCE: Duration = Duration::from_secs(5);

/// A held delivery is progress-ACKed at most two ticks apart, which must stay
/// below half of the pool consumer `ack_wait`.
pub(crate) const PROGRESS_TICK: Duration = Duration::from_secs(5);
const _: () = assert!(PROGRESS_TICK.as_secs() * 2 < ACK_WAIT_SECS / 2);

/// Where an item stands against its deadline, including the skew tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadlineStatus {
    /// No usable deadline: the item predates deadlines or carries a
    /// non-finite or non-positive value.
    Unbounded,
    /// Time left until the deadline, capped at the policy horizon.
    Live(Duration),
    /// How long ago the deadline passed.
    Expired(Duration),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkDeadlinePolicy {
    /// When false, expired items are logged before IPC but still executed.
    pub enforce: bool,
    /// Allowed wall-clock skew between gateway and worker hosts.
    pub skew_tolerance: Duration,
    /// Upper bound on any deadline-derived hold or backend budget. A queued
    /// item cannot outlive the work stream, so nothing is held longer.
    pub max_horizon: Duration,
}

impl WorkDeadlinePolicy {
    pub fn from_env() -> Self {
        Self::from_values(
            std::env::var(ENFORCE_ENV).ok().as_deref(),
            std::env::var(SKEW_TOLERANCE_ENV).ok().as_deref(),
            work_cancel_tombstone_ttl(),
        )
    }

    fn from_values(
        enforce: Option<&str>,
        skew_tolerance_ms: Option<&str>,
        max_horizon: Duration,
    ) -> Self {
        let enforce = !enforce.is_some_and(|raw| {
            matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        });
        let skew_tolerance = skew_tolerance_ms
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .map(Duration::from_millis)
            .unwrap_or(DEFAULT_SKEW_TOLERANCE);
        Self {
            enforce,
            skew_tolerance,
            max_horizon,
        }
    }

    pub fn status(&self, deadline: Option<f64>, now_unix_s: f64) -> DeadlineStatus {
        let Some(deadline) = deadline.filter(|value| value.is_finite() && *value > 0.0) else {
            return DeadlineStatus::Unbounded;
        };
        let remaining_s = deadline + self.skew_tolerance.as_secs_f64() - now_unix_s;
        if remaining_s > 0.0 {
            DeadlineStatus::Live(
                Duration::try_from_secs_f64(remaining_s)
                    .unwrap_or(self.max_horizon)
                    .min(self.max_horizon),
            )
        } else {
            DeadlineStatus::Expired(
                Duration::try_from_secs_f64(-remaining_s).unwrap_or(Duration::MAX),
            )
        }
    }

    /// Time until the latest live deadline among `deadlines`, or `None` when
    /// no item carries one. A backend call for these items is given at least
    /// this long.
    pub fn run_batch_budget(
        &self,
        deadlines: impl IntoIterator<Item = Option<f64>>,
        now_unix_s: f64,
    ) -> Option<Duration> {
        deadlines
            .into_iter()
            .filter_map(|deadline| match self.status(deadline, now_unix_s) {
                DeadlineStatus::Live(remaining) => Some(remaining),
                DeadlineStatus::Unbounded | DeadlineStatus::Expired(_) => None,
            })
            .max()
    }
}

pub fn unix_now_s() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs_f64())
        .unwrap_or(0.0)
}

/// JetStream leases kept alive by progress ACKs, keyed by the delivery's ACK
/// reply subject. A lease ends when its delivery is ACKed, NAKed, or left
/// unacked on purpose, or at its horizon, after which JetStream redelivers as
/// it would without a lease.
pub(crate) struct ProgressLeases<C> {
    leases: Mutex<HashMap<String, ProgressLease<C>>>,
}

struct ProgressLease<C> {
    client: C,
    horizon: Instant,
    last_progress: Instant,
}

impl<C: Clone> ProgressLeases<C> {
    fn new() -> Self {
        Self {
            leases: Mutex::new(HashMap::new()),
        }
    }

    fn hold(&self, reply: &str, client: C, horizon: Instant, now: Instant) {
        self.leases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                reply.to_owned(),
                ProgressLease {
                    client,
                    horizon,
                    last_progress: now,
                },
            );
    }

    fn release(&self, reply: &str) {
        self.leases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(reply);
    }

    fn is_held(&self, reply: &str) -> bool {
        self.leases
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(reply)
    }

    /// Leases whose last progress is at least one tick old, marked as
    /// progressed at `now`. Leases past their horizon are dropped.
    fn due(&self, now: Instant) -> Vec<(String, C)> {
        let mut leases = self.leases.lock().unwrap_or_else(|e| e.into_inner());
        leases.retain(|_, lease| lease.horizon > now);
        leases
            .iter_mut()
            .filter(|(_, lease)| {
                now.saturating_duration_since(lease.last_progress) >= PROGRESS_TICK
            })
            .map(|(reply, lease)| {
                lease.last_progress = now;
                (reply.clone(), lease.client.clone())
            })
            .collect()
    }
}

/// Process-wide progress leases for NATS deliveries. Settlement lives on
/// [`crate::delivery::Delivery`], which releases the lease before it sends the
/// ACK or NAK.
pub(crate) struct NatsProgressLeases {
    leases: Arc<ProgressLeases<async_nats::Client>>,
    ticker: Mutex<Option<JoinHandle<()>>>,
}

static NATS_PROGRESS_LEASES: LazyLock<NatsProgressLeases> = LazyLock::new(|| NatsProgressLeases {
    leases: Arc::new(ProgressLeases::new()),
    ticker: Mutex::new(None),
});

pub(crate) fn nats_progress_leases() -> &'static NatsProgressLeases {
    &NATS_PROGRESS_LEASES
}

impl NatsProgressLeases {
    pub(crate) fn hold(&self, msg: &Message, horizon: Instant, telemetry: &SidecarTelemetry) {
        let Some(reply) = msg.reply.as_ref() else {
            return;
        };
        self.leases.hold(
            reply.as_str(),
            msg.context.client(),
            horizon,
            Instant::now(),
        );
        self.ensure_ticker(telemetry);
    }

    pub(crate) fn release(&self, msg: &Message) {
        if let Some(reply) = msg.reply.as_ref() {
            self.leases.release(reply.as_str());
        }
    }

    fn ensure_ticker(&self, telemetry: &SidecarTelemetry) {
        let mut ticker = self.ticker.lock().unwrap_or_else(|e| e.into_inner());
        if ticker.as_ref().is_some_and(|handle| !handle.is_finished()) {
            return;
        }
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        *ticker = Some(runtime.spawn(run_progress_ticker(
            Arc::clone(&self.leases),
            telemetry.clone(),
        )));
    }
}

async fn run_progress_ticker(
    leases: Arc<ProgressLeases<async_nats::Client>>,
    telemetry: SidecarTelemetry,
) {
    let mut tick = tokio::time::interval(PROGRESS_TICK);
    tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tick.tick().await;
        for (reply, client) in leases.due(Instant::now()) {
            if !leases.is_held(&reply) {
                continue;
            }
            let result = client.publish(reply, AckKind::Progress.into()).await;
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

    const HORIZON: Duration = Duration::from_secs(1_800);

    fn policy() -> WorkDeadlinePolicy {
        WorkDeadlinePolicy::from_values(None, None, HORIZON)
    }

    #[test]
    fn policy_defaults_to_enforcing_with_a_bounded_skew_tolerance() {
        let default = policy();
        assert!(default.enforce);
        assert_eq!(default.skew_tolerance, DEFAULT_SKEW_TOLERANCE);
        assert_eq!(default.max_horizon, HORIZON);

        for off in ["false", "0", " OFF ", "no"] {
            assert!(!WorkDeadlinePolicy::from_values(Some(off), None, HORIZON).enforce);
        }
        assert!(WorkDeadlinePolicy::from_values(Some("true"), None, HORIZON).enforce);
        assert_eq!(
            WorkDeadlinePolicy::from_values(None, Some("250"), HORIZON).skew_tolerance,
            Duration::from_millis(250)
        );
        assert_eq!(
            WorkDeadlinePolicy::from_values(None, Some("-1"), HORIZON).skew_tolerance,
            DEFAULT_SKEW_TOLERANCE
        );
    }

    #[test]
    fn items_without_a_usable_deadline_are_unbounded() {
        let policy = policy();
        for deadline in [
            None,
            Some(0.0),
            Some(-5.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            assert_eq!(
                policy.status(deadline, 1_700_000_000.0),
                DeadlineStatus::Unbounded,
                "{deadline:?}"
            );
        }
    }

    #[test]
    fn deadline_expires_only_after_the_skew_tolerance() {
        let policy = WorkDeadlinePolicy::from_values(None, Some("2000"), HORIZON);
        let deadline = Some(1_700_000_000.0);

        assert_eq!(
            policy.status(deadline, 1_699_999_990.0),
            DeadlineStatus::Live(Duration::from_secs(12))
        );
        assert_eq!(
            policy.status(deadline, 1_700_000_001.5),
            DeadlineStatus::Live(Duration::from_millis(500))
        );
        assert_eq!(
            policy.status(deadline, 1_700_000_004.0),
            DeadlineStatus::Expired(Duration::from_secs(2))
        );
    }

    #[test]
    fn live_time_is_capped_at_the_policy_horizon() {
        let policy = policy();
        assert_eq!(
            policy.status(Some(1.0e300), 1_700_000_000.0),
            DeadlineStatus::Live(HORIZON)
        );
    }

    #[test]
    fn run_batch_budget_follows_the_latest_live_deadline() {
        let policy = WorkDeadlinePolicy::from_values(None, Some("0"), HORIZON);
        let now = 1_700_000_000.0;

        assert_eq!(policy.run_batch_budget([None, None], now), None);
        assert_eq!(policy.run_batch_budget([Some(now - 1.0)], now), None);
        assert_eq!(
            policy.run_batch_budget(
                [None, Some(now + 30.0), Some(now + 90.0), Some(now - 5.0)],
                now
            ),
            Some(Duration::from_secs(90))
        );
        assert_eq!(
            policy.run_batch_budget([Some(now + 1.0e9)], now),
            Some(HORIZON)
        );
    }

    #[test]
    fn held_leases_are_progressed_on_each_tick_until_released_or_expired() {
        let leases = ProgressLeases::<u8>::new();
        let start = Instant::now();
        leases.hold("reply.a", 1, start + Duration::from_secs(60), start);
        leases.hold("reply.b", 2, start + Duration::from_secs(12), start);

        assert!(leases.due(start + Duration::from_secs(4)).is_empty());

        let mut due = leases.due(start + PROGRESS_TICK);
        due.sort();
        assert_eq!(
            due,
            vec![("reply.a".to_string(), 1), ("reply.b".to_string(), 2)]
        );
        assert!(
            leases
                .due(start + PROGRESS_TICK + Duration::from_secs(1))
                .is_empty(),
            "a lease progressed on this tick waits for the next one"
        );

        leases.release("reply.a");
        assert!(!leases.is_held("reply.a"));
        assert_eq!(
            leases.due(start + PROGRESS_TICK * 2),
            vec![("reply.b".to_string(), 2)]
        );

        assert!(
            leases.due(start + Duration::from_secs(12)).is_empty(),
            "a lease stops at its horizon so JetStream can redeliver"
        );
        assert!(!leases.is_held("reply.b"));
    }

    #[test]
    fn a_lease_is_never_left_unprogressed_for_half_the_ack_wait() {
        let leases = ProgressLeases::<u8>::new();
        let start = Instant::now();
        leases.hold("reply", 0, start + Duration::from_secs(600), start);
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
}
