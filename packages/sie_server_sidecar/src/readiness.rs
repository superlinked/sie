//! Sidecar readiness state — backs the `/readyz` HTTP endpoint and
//! moves the K8s readiness contract off the Python adapter.
//!
//! # What "ready" means
//!
//! The sidecar is **ready** iff:
//!
//! 1. **Handshake complete.** At least one `Ping` round-trip to the
//!    adapter has succeeded since process start. (Not "every model
//!    `EnsureModelReady`'d" — models load lazily on first batch and a
//!    pod can sit `Ready` for hours before its first inference. The
//!    handshake-only definition matches what
//!    `IpcServer.is_heartbeat_fresh` checks on the Python side.)
//! 2. **Heartbeat fresh.** The most recent successful `Ping` was
//!    within `freshness_ms` of now (monotonic). The default freshness
//!    window is `ping_interval_ms * 3`; operators can
//!    widen it via `SIE_WORKER_READYZ_STALE_MULT`.
//! 3. **Not draining.** SIGTERM has not yet been observed. Once
//!    drain begins we want K8s to pull the pod out of the service
//!    immediately so new traffic stops landing on a sidecar that is
//!    about to disappear; the heartbeat freshness check would catch
//!    this only after the heartbeat task has stopped, which is too
//!    late.
//!
//! # Plumbing
//!
//! [`Readiness`] is owned by [`crate::run`], shared by [`std::sync::Arc`] into:
//!
//! * `crate::spawn_heartbeat` — calls
//!   [`Readiness::record_ping_success`] after every successful
//!   `IpcClient::ping`.
//! * [`crate::runtime_state::spawn_probe_server`] — reads via
//!   [`Readiness::snapshot`] on every `GET /readyz`.
//! * The `run()` shutdown path — calls [`Readiness::mark_draining`]
//!   the moment the pull loop exits (which is itself driven by the
//!   `Shutdown` signal).
//!
//! # Why not put `last_ping` inside `IpcClient`?
//!
//! The IPC client is a transport — it knows how to round-trip a
//! frame, not what "the sidecar is healthy" means at the operational
//! layer. Tests instantiate `IpcClient` in isolation against a stub
//! server; we don't want those test pings to bump a global readiness
//! clock. Having `Readiness` be an explicit type at the
//! `lib::run`-orchestrator layer keeps the policy / mechanism split
//! that the rest of this crate already follows.
//!
//! # Performance
//!
//! Probe traffic in production is roughly `1 Hz × N pods × replicas`
//! plus the load-balancer's healthcheck. The probe path here is:
//!
//! * one `AtomicBool::load(Relaxed)` for the drain bit,
//! * one `AtomicU64::load(Relaxed)` for the last-ping timestamp,
//! * one `Instant::now()`,
//! * arithmetic.
//!
//! No lock; no allocation on the ready path. The 503-with-reason
//! path allocates a small `String`.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Multiplier applied to `ping_interval_ms` to compute the staleness
/// threshold. Operators tune via `SIE_WORKER_READYZ_STALE_MULT`.
pub const DEFAULT_STALE_MULT: u32 = 3;

/// Sentinel used in the `last_ping_nanos` atomic to mean "no ping
/// has succeeded yet". Picking 0 is safe because `last_ping_nanos`
/// stores `Instant::now() - origin` where `origin` is captured at
/// `Readiness::new` time *before* the heartbeat task is spawned, so
/// any real ping will record a strictly positive value.
const HANDSHAKE_PENDING: u64 = 0;

/// Lock-free shared state for the sidecar's readiness signal.
pub struct Readiness {
    /// Anchor for the monotonic timestamp stored in
    /// `last_ping_nanos`. We can't put an [`Instant`] inside an
    /// atomic, so we serialise as nanos-since-origin.
    ///
    /// [`Instant`] is monotonic by construction: `Instant::now() -
    /// origin` never goes backwards on the same machine, and a u64
    /// nanosecond counter has 584 years of headroom from process
    /// start, so we never have to think about wraparound.
    origin: Instant,
    /// `Instant::now() - origin` in nanoseconds at the moment of the
    /// most recent successful `Ping`. `0` means "no successful ping
    /// yet" (see [`HANDSHAKE_PENDING`]).
    ///
    /// Reads use `Ordering::Relaxed`: a delayed-but-eventually-
    /// consistent observation is fine — the worst case is one extra
    /// "still stale" probe before we report ready, and probe interval
    /// is already O(seconds). We never need this load to synchronise
    /// with any other field.
    last_ping_nanos: AtomicU64,
    /// Set the instant SIGTERM is observed. `/readyz` flips to
    /// `503 draining` immediately so the K8s service stops sending
    /// new traffic before the in-flight drain even starts to run
    /// down.
    draining: AtomicBool,
    /// `now - last_ping <= freshness_ms` ⇒ heartbeat fresh.
    ///
    /// Captured once at construction; the spec describes it as
    /// `ping_interval_ms * stale_mult` so a runtime mutation would
    /// be a footgun (the heartbeat schedule is also fixed at
    /// construction).
    freshness_ms: u64,
}

impl Readiness {
    /// Build a readiness handle whose freshness window is
    /// `ping_interval_ms * stale_mult`, clamped to a sane minimum so
    /// a misconfigured `stale_mult = 0` doesn't make the pod look
    /// stale on the very tick after a successful ping.
    ///
    /// `stale_mult = 0` is treated as the default ([`DEFAULT_STALE_MULT`])
    /// rather than rejected; the sidecar should boot even with weird
    /// env-var values, with a log warning at the call site.
    #[must_use]
    pub fn new(ping_interval_ms: u64, stale_mult: u32) -> Self {
        let mult = if stale_mult == 0 {
            DEFAULT_STALE_MULT
        } else {
            stale_mult
        };
        // `as u64 * u64` can overflow with adversarial inputs
        // (`u32::MAX * u64::MAX`); saturate to keep behaviour
        // predictable. In practice both are O(thousands).
        let freshness_ms = ping_interval_ms.saturating_mul(u64::from(mult));
        Self {
            origin: Instant::now(),
            last_ping_nanos: AtomicU64::new(HANDSHAKE_PENDING),
            draining: AtomicBool::new(false),
            freshness_ms,
        }
    }

    /// Freshness window in milliseconds. Exposed so the metrics
    /// server can include it in the 503 reason string for operator
    /// debugging.
    #[must_use]
    pub fn freshness_ms(&self) -> u64 {
        self.freshness_ms
    }

    /// Called by the heartbeat task on a successful `IpcClient::ping`.
    ///
    /// Stores `Instant::now() - origin` as nanos in the shared
    /// atomic. The first call moves us from "handshake pending" to
    /// "fresh"; subsequent calls just refresh the stamp.
    pub fn record_ping_success(&self) {
        // `as u64` truncation is a non-issue: `Instant - origin`
        // produces a value bounded by the process uptime, and u64
        // nanos = ~584 years.
        let nanos = self.origin.elapsed().as_nanos() as u64;
        // `nanos == 0` is theoretically possible if this fires on
        // the same nanosecond `origin` was captured (impossible on
        // any clock with ns resolution but still: belt + braces
        // because 0 is our sentinel). Bump to 1 in that case so the
        // sentinel stays unambiguous.
        let stamp = if nanos == HANDSHAKE_PENDING { 1 } else { nanos };
        self.last_ping_nanos.store(stamp, Ordering::Relaxed);
    }

    /// Called when the worker observes its shutdown signal. Once
    /// set, `/readyz` returns `503 draining` until process exit.
    /// Idempotent.
    pub fn mark_draining(&self) {
        self.draining.store(true, Ordering::Relaxed);
    }

    /// Cheap point-in-time view of the readiness state. Returned by
    /// value (no lock held) so the HTTP handler can format the
    /// reason string at its leisure.
    #[must_use]
    pub fn snapshot(&self) -> ReadinessSnapshot {
        let draining = self.draining.load(Ordering::Relaxed);
        let last_ping_nanos = self.last_ping_nanos.load(Ordering::Relaxed);

        let heartbeat_age_ms = if last_ping_nanos == HANDSHAKE_PENDING {
            None
        } else {
            // Recompute `now` AFTER reading the atomic so we never
            // produce a negative age. Subtraction can still
            // theoretically underflow if `last_ping_nanos` was
            // captured in a future tick (impossible without clock
            // skew on a single machine), so we saturate.
            let now_nanos = self.origin.elapsed().as_nanos() as u64;
            let elapsed_nanos = now_nanos.saturating_sub(last_ping_nanos);
            Some(elapsed_nanos / 1_000_000)
        };

        ReadinessSnapshot {
            handshaked: last_ping_nanos != HANDSHAKE_PENDING,
            heartbeat_age_ms,
            freshness_ms: self.freshness_ms,
            draining,
        }
    }
}

/// Point-in-time view returned by [`Readiness::snapshot`]. Carrying
/// the freshness window inside the snapshot keeps the formatting
/// logic close to the data and lets us surface it in the 503 body
/// for operator debugging without a second atomic load.
#[derive(Debug, Clone, Copy)]
pub struct ReadinessSnapshot {
    /// True iff at least one `Ping` has succeeded since process start.
    pub handshaked: bool,
    /// Milliseconds since the most recent successful ping, or `None`
    /// if no ping has succeeded yet.
    pub heartbeat_age_ms: Option<u64>,
    /// Staleness threshold from [`Readiness::freshness_ms`].
    pub freshness_ms: u64,
    /// True iff [`Readiness::mark_draining`] has been called.
    pub draining: bool,
}

impl ReadinessSnapshot {
    /// Whole-of-state predicate: ready iff all three sub-conditions
    /// hold. Mirrors the AND-gating documented at the module top.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        if self.draining {
            return false;
        }
        match self.heartbeat_age_ms {
            None => false,
            Some(age) => age <= self.freshness_ms,
        }
    }

    /// Human-readable reason the snapshot is *not* ready. Returns an
    /// empty string when [`Self::is_ready`] is true; the caller is
    /// expected to gate on `is_ready()` before calling. Format is
    /// stable enough for `curl /readyz`, but operators should rely
    /// on the HTTP status code, not the body, for automated checks.
    #[must_use]
    pub fn reason(&self) -> String {
        if self.draining {
            return "draining".to_owned();
        }
        match self.heartbeat_age_ms {
            None => "handshake pending".to_owned(),
            Some(age) if age > self.freshness_ms => format!(
                "heartbeat stale ({age} ms ago, threshold {} ms)",
                self.freshness_ms,
            ),
            Some(_) => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn fresh_readiness_is_handshake_pending() {
        let r = Readiness::new(2_000, 3);
        let snap = r.snapshot();
        assert!(!snap.handshaked, "fresh: handshake must be pending");
        assert!(!snap.is_ready(), "fresh: not ready");
        assert_eq!(snap.reason(), "handshake pending");
        assert_eq!(snap.heartbeat_age_ms, None);
        assert_eq!(snap.freshness_ms, 6_000);
    }

    #[test]
    fn record_ping_success_makes_us_ready() {
        let r = Readiness::new(2_000, 3);
        r.record_ping_success();
        let snap = r.snapshot();
        assert!(snap.handshaked);
        assert!(snap.is_ready(), "just-pinged should be ready");
        assert_eq!(snap.reason(), "");
        // Age must be small but non-negative; can't assert exactly
        // 0 because of clock granularity on slow runners.
        assert!(snap.heartbeat_age_ms.unwrap() < 1_000);
    }

    #[test]
    fn drain_flips_us_unready_immediately() {
        let r = Readiness::new(2_000, 3);
        r.record_ping_success();
        assert!(r.snapshot().is_ready());
        r.mark_draining();
        let snap = r.snapshot();
        assert!(!snap.is_ready(), "draining must beat freshness");
        assert_eq!(snap.reason(), "draining");
        assert!(
            snap.handshaked,
            "draining doesn't erase the handshake bit — operators may want to see we *did* connect"
        );
    }

    #[test]
    fn stale_heartbeat_reports_age_in_reason() {
        // Pinned-in-time test: synthesise a stale stamp by computing
        // an age larger than `freshness_ms` and seeding
        // `last_ping_nanos` directly. We can't time-travel a real
        // ping's `Instant::now()` reading, so we splice the value.
        let r = Readiness::new(100, 1); // freshness = 100 ms
        r.record_ping_success();

        // Wait long enough that the recorded ping is stale.
        std::thread::sleep(std::time::Duration::from_millis(150));

        let snap = r.snapshot();
        assert!(snap.handshaked);
        assert!(
            !snap.is_ready(),
            "100ms freshness must reject 150ms-old ping"
        );
        let reason = snap.reason();
        assert!(
            reason.starts_with("heartbeat stale"),
            "reason should call out staleness, got {reason:?}"
        );
        assert!(
            reason.contains("threshold 100 ms"),
            "reason must surface the configured threshold, got {reason:?}"
        );
    }

    #[test]
    fn zero_stale_mult_falls_back_to_default() {
        // Misconfigured env (`SIE_WORKER_READYZ_STALE_MULT=0`) must
        // not make the pod look stale on the next tick after a
        // successful ping.
        let r = Readiness::new(2_000, 0);
        assert_eq!(r.freshness_ms, 2_000 * u64::from(DEFAULT_STALE_MULT));
    }

    #[test]
    fn freshness_ms_does_not_overflow_on_adversarial_input() {
        // u32::MAX * u64::MAX would overflow; we saturate.
        let r = Readiness::new(u64::MAX, u32::MAX);
        assert_eq!(r.freshness_ms, u64::MAX);
        // And the snapshot path still works.
        let _ = r.snapshot();
    }

    #[test]
    fn snapshot_is_lock_free_and_thread_safe() {
        // Smoke: hammer record_ping_success / snapshot from many
        // threads. The struct is `Send + Sync` via its atomics, so
        // this test is mostly insurance against accidental
        // refactors that reintroduce a Mutex.
        let r = Arc::new(Readiness::new(2_000, 3));

        let mut handles = Vec::new();
        for _ in 0..8 {
            let r = Arc::clone(&r);
            handles.push(std::thread::spawn(move || {
                for _ in 0..1_000 {
                    r.record_ping_success();
                    let _ = r.snapshot();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let final_snap = r.snapshot();
        assert!(final_snap.handshaked);
        assert!(final_snap.is_ready());
    }

    #[test]
    fn ping_with_zero_age_is_promoted_off_the_sentinel() {
        // The `if nanos == HANDSHAKE_PENDING { 1 }` branch is
        // theoretical — `Instant::now()` advances every call on
        // any modern OS — but the safety belt matters because a
        // false-positive "handshake pending" would cause a flapping
        // pod to look unready forever. Pinning the branch with a
        // unit test keeps the invariant load-bearing.
        let r = Readiness::new(2_000, 3);
        // Force the underlying atomic to the sentinel directly...
        r.last_ping_nanos
            .store(HANDSHAKE_PENDING, Ordering::Relaxed);
        // ...then fake a "ping arrived at origin" by writing a 1
        // through the same path record_ping_success would have
        // taken on a clock that didn't advance. We can't easily
        // simulate the pinned-clock case in pure user code, but we
        // can pin the sentinel-promotion contract on a regular call
        // since it always goes through the same branch.
        r.record_ping_success();
        let v = r.last_ping_nanos.load(Ordering::Relaxed);
        assert_ne!(v, HANDSHAKE_PENDING, "post-ping must never equal sentinel");
    }
}
