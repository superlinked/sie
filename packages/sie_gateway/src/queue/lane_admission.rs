//! Per-lane admission accounting for queue publishes.
//!
//! Pool-granular backpressure sheds every model sharing a pool as soon as
//! *one* model's backlog crosses the pool threshold, because the number
//! [`crate::queue::publisher::WorkPublisher::check_backpressure`] compares is
//! the whole `WORK_POOL_{pool}` stream's pending count and a stream is per
//! pool, not per lane.
//!
//! The proposal names two realistic implementations. This is the first: a
//! gateway-local in-flight counter per lane. It is approximate — it does not
//! survive a gateway restart and it does not see the other replicas' work —
//! and it is deliberately the reversible one. Per-lane JetStream streams are
//! exact but are a topology migration and a soft one-way door.
//!
//! ## What is counted
//!
//! Work *items*, not requests. The proposal's correction to the review matters
//! here: the worker scheduler is not strictly FIFO within a lane (batchers are
//! per LoRA key with cross-key FCFS on head age, and a batcher cost-sorts its
//! pending queue ascending before every extract), so a cheap request does not
//! queue behind an expensive one on cost. What it queues behind is **depth**:
//! one 4096-item request contributes 4096 cheap items that fill every
//! subsequent batch. Counting items is what makes that visible; counting
//! requests would not.
//!
//! A reservation is taken for every publish that passes the check and is
//! released when the request's result collector is dropped — on success,
//! timeout, publish failure, or the expiry sweep alike, because every one of
//! those paths drops the collector.
//!
//! ## Shadow mode
//!
//! The decision is computed and recorded on every publish. Enforcement is a
//! separate flag. With enforcement off the outcome is
//! [`LaneAdmissionOutcome::ShedShadow`], the caller admits the work anyway,
//! and the only trace is telemetry, so the new decision can be evaluated
//! against production traffic before it sheds anything.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::state::demand_tracker::{PhysicalLane, PhysicalLaneCatalog};

/// Per-lane in-flight item ceiling used when the caller does not configure one.
///
/// Deliberately well below the pool-wide `SIE_GATEWAY_MAX_STREAM_PENDING`
/// default of 50000 — a per-lane threshold at or above the pool-wide one could
/// never fire first, which is the entire point. This is a shadow-mode starting
/// value, not a tuned one: read the shed rate per lane off
/// `sie.gateway.queue.lane_admission.decisions` before enabling enforcement.
pub const DEFAULT_MAX_LANE_IN_FLIGHT_ITEMS: u64 = 10_000;

/// The lane a work item is published onto: the `(pool, machine_profile,
/// bundle)` triple that names one physical worker lane.
///
/// This is the same triple the telemetry contract already treats as a bounded
/// catalog (`gateway_lane_admission_domain`, which reuses the KEDA domain's
/// `SIE_GATEWAY_CONFIGURED_PHYSICAL_LANES` source and fail-closed rules without
/// being a KEDA control signal). Model and tenant are deliberately
/// absent: they are unbounded, and the contract forbids them as metric labels.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LaneKey {
    pub pool: String,
    pub machine_profile: String,
    pub bundle: String,
}

impl LaneKey {
    pub fn new(pool: &str, machine_profile: &str, bundle: &str) -> Self {
        Self {
            pool: pool.to_string(),
            machine_profile: machine_profile.to_string(),
            bundle: bundle.to_string(),
        }
    }
}

/// What the per-lane check decided, and whether it was acted on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LaneAdmissionOutcome {
    /// The lane was below its ceiling.
    Admitted,
    /// The lane was over its ceiling but enforcement is off, so the work was
    /// admitted anyway. Recorded, never acted on.
    ShedShadow,
    /// The lane was over its ceiling and enforcement is on: the caller must
    /// reject this publish.
    ShedEnforced,
}

impl LaneAdmissionOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::ShedShadow => "shed_shadow",
            Self::ShedEnforced => "shed_enforced",
        }
    }

    /// True when the caller must refuse the publish.
    pub fn rejects(self) -> bool {
        matches!(self, Self::ShedEnforced)
    }
}

/// One lane's reservation, released on drop.
///
/// Held by the request's result collector so that every terminal path —
/// completion, timeout, publish failure, cancellation, the expiry sweep —
/// releases it without needing its own release call. An early return between
/// reservation and collector installation drops the guard and releases too.
#[derive(Debug)]
pub struct LaneReservation {
    counter: Arc<AtomicU64>,
    items: u64,
}

impl Drop for LaneReservation {
    fn drop(&mut self) {
        if self.items == 0 {
            return;
        }
        // Saturating: a double release would otherwise wrap the counter to
        // u64::MAX and shed the lane permanently.
        let _ = self
            .counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                Some(current.saturating_sub(self.items))
            });
    }
}

/// One evaluated admission: what was decided, and the reservation it took.
#[derive(Debug)]
pub struct LaneAdmission {
    pub outcome: LaneAdmissionOutcome,
    /// In-flight items on this lane *before* this request's reservation.
    pub in_flight: u64,
    pub threshold: u64,
    /// `None` only for [`LaneAdmissionOutcome::ShedEnforced`], where the work
    /// never enters the queue and therefore reserves nothing.
    pub reservation: Option<LaneReservation>,
}

/// Gateway-local per-lane in-flight accounting.
///
/// The counter map is keyed by lane. Its key space is bounded in practice by
/// the deployment's lane catalog; entries are retained rather than reaped
/// because a lane that has served traffic once will serve it again, and the
/// catalog caps the deployment at 1024 lanes.
pub struct LaneAdmissionControl {
    lanes: DashMap<LaneKey, Arc<AtomicU64>>,
    max_in_flight_items: u64,
    enforce: bool,
    /// Deployment-owned lane catalog. Only exact members carry lane labels on
    /// the decision metric; anything else is recorded without a lane rather
    /// than under a synthetic one, matching the contract's
    /// `synthetic_fallback_lane: forbidden`.
    catalog: PhysicalLaneCatalog,
}

impl Default for LaneAdmissionControl {
    /// Shadow mode at the default ceiling: the decision is computed and
    /// recorded, never enforced.
    fn default() -> Self {
        Self::new(
            DEFAULT_MAX_LANE_IN_FLIGHT_ITEMS,
            false,
            PhysicalLaneCatalog::default(),
        )
    }
}

impl LaneAdmissionControl {
    pub fn new(max_in_flight_items: u64, enforce: bool, catalog: PhysicalLaneCatalog) -> Self {
        Self {
            lanes: DashMap::new(),
            max_in_flight_items,
            enforce,
            catalog,
        }
    }

    /// Resolve `lane` against the deployment catalog, for bounded telemetry
    /// labels. `None` means the lane is not a configured physical lane.
    pub fn resolve(&self, lane: &LaneKey) -> Option<PhysicalLane> {
        self.catalog
            .resolve(&lane.pool, &lane.machine_profile, &lane.bundle)
    }

    /// Evaluate one publish against its lane's ceiling and, unless the
    /// decision is an enforced shed, reserve `items` on that lane.
    ///
    /// The comparison is against the depth already outstanding on the lane,
    /// not against depth-plus-this-request. That mirrors the pool-wide check
    /// it complements (`num_pending > threshold`) and, more importantly, means
    /// a single large request is never rejected outright on an idle lane: it
    /// is admitted, and it is the *next* arrival on that same lane that pays,
    /// which is exactly the depth-fairness behaviour B5 asks for.
    ///
    /// The check and the reservation are one atomic operation. A plain load
    /// followed by a separate `fetch_add` would let a concurrent burst on one
    /// lane all observe the same sub-threshold depth and all reserve against
    /// it — and a concurrent burst on one lane is precisely the saturation
    /// case this exists to catch, so the race is not a rare interleaving but
    /// the expected traffic shape. `fetch_update` retries its compare-exchange
    /// until the depth it observed is the depth it reserved against, which
    /// bounds the overshoot to the single request that crosses the ceiling.
    pub fn admit(&self, lane: &LaneKey, items: u64) -> LaneAdmission {
        let counter = self.counter_for(lane);
        // `Err` carries the observed value on abort, `Ok` the pre-update
        // value, so either way this yields the depth the decision was made
        // against. Shadow mode never aborts: the closure only refuses when
        // enforcement is on, so a shadow shed still reserves and the shadow
        // counter tracks real depth rather than a hypothetical one.
        let reserved = counter.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            if current > self.max_in_flight_items && self.enforce {
                None
            } else {
                // Saturating so a pathological counter cannot wrap to zero and
                // silently reopen a lane that is over its ceiling.
                Some(current.saturating_add(items))
            }
        });
        let (in_flight, reservation) = match reserved {
            Ok(previous) => (previous, Some(LaneReservation { counter, items })),
            Err(current) => (current, None),
        };
        let over = in_flight > self.max_in_flight_items;
        let outcome = match (over, self.enforce) {
            (false, _) => LaneAdmissionOutcome::Admitted,
            (true, false) => LaneAdmissionOutcome::ShedShadow,
            (true, true) => LaneAdmissionOutcome::ShedEnforced,
        };
        debug_assert_eq!(
            outcome.rejects(),
            reservation.is_none(),
            "an enforced shed must reserve nothing and every other outcome must reserve"
        );
        LaneAdmission {
            outcome,
            in_flight,
            threshold: self.max_in_flight_items,
            reservation,
        }
    }

    /// In-flight items currently reserved on `lane`.
    #[cfg(test)]
    pub fn in_flight(&self, lane: &LaneKey) -> u64 {
        self.lanes
            .get(lane)
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    fn counter_for(&self, lane: &LaneKey) -> Arc<AtomicU64> {
        if let Some(existing) = self.lanes.get(lane) {
            return Arc::clone(existing.value());
        }
        Arc::clone(
            self.lanes
                .entry(lane.clone())
                .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                .value(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control(enforce: bool) -> LaneAdmissionControl {
        LaneAdmissionControl::new(10, enforce, PhysicalLaneCatalog::default())
    }

    fn lane(pool: &str) -> LaneKey {
        LaneKey::new(pool, "a10g", "default")
    }

    #[test]
    fn admits_while_the_lane_is_under_its_ceiling() {
        let control = control(true);
        let admission = control.admit(&lane("hot"), 4);
        assert_eq!(admission.outcome, LaneAdmissionOutcome::Admitted);
        assert_eq!(admission.in_flight, 0);
        assert_eq!(control.in_flight(&lane("hot")), 4);
    }

    #[test]
    fn a_saturated_lane_sheds_and_a_cold_lane_on_the_same_pool_still_admits() {
        // Two lanes differing only in machine profile: the pool-wide check
        // cannot tell them apart, which is the defect this replaces.
        let control = control(true);
        let hot = LaneKey::new("shared", "h100", "default");
        let cold = LaneKey::new("shared", "a10g", "default");

        let _hot_reservation = control.admit(&hot, 11).reservation;
        assert_eq!(
            control.admit(&hot, 1).outcome,
            LaneAdmissionOutcome::ShedEnforced
        );
        assert_eq!(
            control.admit(&cold, 1).outcome,
            LaneAdmissionOutcome::Admitted
        );
    }

    #[test]
    fn an_enforced_shed_reserves_nothing() {
        let control = control(true);
        let _held = control.admit(&lane("hot"), 11).reservation;
        let shed = control.admit(&lane("hot"), 100);
        assert!(shed.reservation.is_none());
        assert_eq!(control.in_flight(&lane("hot")), 11);
    }

    #[test]
    fn shadow_mode_records_the_shed_without_changing_admission() {
        let control = control(false);
        let _held = control.admit(&lane("hot"), 11).reservation;
        let shadow = control.admit(&lane("hot"), 5);
        assert_eq!(shadow.outcome, LaneAdmissionOutcome::ShedShadow);
        assert!(!shadow.outcome.rejects());
        // The shadow admission still reserves, so the shadow counter tracks
        // real depth rather than a hypothetical one.
        assert!(shadow.reservation.is_some());
        assert_eq!(control.in_flight(&lane("hot")), 16);
    }

    #[test]
    fn dropping_the_reservation_releases_the_lane() {
        let control = control(true);
        {
            let _held = control.admit(&lane("hot"), 11).reservation;
            assert_eq!(
                control.admit(&lane("hot"), 1).outcome,
                LaneAdmissionOutcome::ShedEnforced
            );
        }
        assert_eq!(control.in_flight(&lane("hot")), 0);
        assert_eq!(
            control.admit(&lane("hot"), 1).outcome,
            LaneAdmissionOutcome::Admitted
        );
    }

    #[test]
    fn release_saturates_instead_of_wrapping() {
        let control = control(true);
        let lane = lane("hot");
        let counter = control.counter_for(&lane);
        let reservation = LaneReservation {
            counter: Arc::clone(&counter),
            items: 5,
        };
        drop(reservation);
        assert_eq!(control.in_flight(&lane), 0);
    }

    /// Exactly one request may cross the ceiling, however many arrive at once.
    ///
    /// Each round parks the lane at exactly the threshold and then releases a
    /// thundering herd at that boundary. The predicate is
    /// `depth-before-this-request > threshold`, so the single request that
    /// observes `threshold` is admitted and every request that observes the
    /// resulting `threshold + 1` is shed — one admission per round, no matter
    /// how the threads interleave.
    ///
    /// This is the property atomicity buys, and the boundary is where it is
    /// visible: with a load followed by a separate `fetch_add`, two threads
    /// that both read `threshold` before either adds will both admit, and the
    /// round records more than one. A concurrent burst on a lane sitting at
    /// its ceiling is not an exotic interleaving here — it is the saturation
    /// traffic the whole mechanism exists for, so the race would fire exactly
    /// when it matters most. Repeated because a thread race cannot be made
    /// deterministic; the rounds make an escape overwhelmingly unlikely.
    #[test]
    fn only_one_request_may_cross_the_ceiling_under_concurrency() {
        const THRESHOLD: u64 = 10;
        const THREADS: usize = 16;
        const ROUNDS: usize = 250;

        for round in 0..ROUNDS {
            let control = Arc::new(LaneAdmissionControl::new(
                THRESHOLD,
                true,
                PhysicalLaneCatalog::default(),
            ));
            let lane = lane("hot");

            // Park the lane at exactly the ceiling, held for the whole round.
            let _parked = control.admit(&lane, THRESHOLD).reservation;
            assert_eq!(control.in_flight(&lane), THRESHOLD);

            let barrier = Arc::new(std::sync::Barrier::new(THREADS));
            let admitted = Arc::new(AtomicU64::new(0));
            let workers: Vec<_> = (0..THREADS)
                .map(|_| {
                    let control = Arc::clone(&control);
                    let barrier = Arc::clone(&barrier);
                    let admitted = Arc::clone(&admitted);
                    let lane = lane.clone();
                    std::thread::spawn(move || {
                        barrier.wait();
                        let admission = control.admit(&lane, 1);
                        if admission.reservation.is_some() {
                            admitted.fetch_add(1, Ordering::Relaxed);
                        }
                        // Hold the reservation until the round is scored.
                        admission.reservation
                    })
                })
                .collect();

            let held: Vec<_> = workers
                .into_iter()
                .map(|worker| worker.join().expect("worker"))
                .collect();

            assert_eq!(
                admitted.load(Ordering::Relaxed),
                1,
                "round {round}: exactly one request may cross the ceiling"
            );
            assert_eq!(control.in_flight(&lane), THRESHOLD + 1, "round {round}");
            drop(held);
        }
    }

    #[test]
    fn unresolved_lanes_carry_no_catalog_identity() {
        let control = control(true);
        assert!(control.resolve(&lane("hot")).is_none());
    }

    #[test]
    fn configured_lanes_resolve_for_bounded_labels() {
        let catalog = PhysicalLaneCatalog::try_from_raw([(
            "hot".to_string(),
            "a10g".to_string(),
            "default".to_string(),
        )])
        .expect("catalog");
        let control = LaneAdmissionControl::new(10, true, catalog);
        let resolved = control.resolve(&lane("hot")).expect("resolved lane");
        assert_eq!(resolved.pool(), "hot");
        assert_eq!(resolved.machine_profile(), "a10g");
        assert_eq!(resolved.bundle(), "default");
    }
}
