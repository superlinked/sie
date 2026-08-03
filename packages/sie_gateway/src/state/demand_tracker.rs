use dashmap::DashMap;
use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use tokio::time::Instant;
use tracing::{info, warn};

const DEMAND_EXPIRY_SECS: u64 = 120;
const MAX_LANE_TOKEN_LEN: usize = 63;
pub const MAX_CONFIGURED_PHYSICAL_LANES: usize = 1024;

/// One Helm-owned physical worker lane that has an exact KEDA target.
///
/// Construction validates and canonicalizes all three tokens. Callers cannot
/// manufacture a scale-driving lane from request headers: they must resolve it
/// through [`PhysicalLaneCatalog::resolve`].
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct PhysicalLane {
    pool: String,
    machine_profile: String,
    bundle: String,
}

impl PhysicalLane {
    pub(crate) fn try_new(pool: &str, machine_profile: &str, bundle: &str) -> Result<Self, String> {
        Ok(Self {
            pool: canonical_lane_token("pool", pool)?,
            machine_profile: canonical_lane_token("machineProfile", machine_profile)?,
            bundle: canonical_lane_token("bundle", bundle)?,
        })
    }

    pub fn pool(&self) -> &str {
        &self.pool
    }

    pub fn machine_profile(&self) -> &str {
        &self.machine_profile
    }

    pub fn bundle(&self) -> &str {
        &self.bundle
    }
}

impl fmt::Display for PhysicalLane {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}|{}|{}",
            self.pool, self.machine_profile, self.bundle
        )
    }
}

fn canonical_lane_token(field: &str, value: &str) -> Result<String, String> {
    let canonical = value.trim().to_ascii_lowercase();
    if canonical.is_empty() {
        return Err(format!("{field} must not be empty or whitespace"));
    }
    if canonical.len() > MAX_LANE_TOKEN_LEN {
        return Err(format!(
            "{field} must be at most {MAX_LANE_TOKEN_LEN} characters"
        ));
    }
    if !canonical
        .chars()
        .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '-'))
    {
        return Err(format!(
            "{field} must match ^[A-Za-z0-9_-]+$ before lowercasing"
        ));
    }
    Ok(canonical)
}

/// Finite catalog rendered from the same Helm worker entries as KEDA.
#[derive(Clone, Debug, Default)]
pub struct PhysicalLaneCatalog {
    lanes: HashSet<PhysicalLane>,
}

impl PhysicalLaneCatalog {
    pub(crate) fn try_new(lanes: impl IntoIterator<Item = PhysicalLane>) -> Result<Self, String> {
        let mut catalog = HashSet::new();
        for lane in lanes {
            if !catalog.insert(lane.clone()) {
                return Err(format!("duplicate configured physical lane {lane}"));
            }
            if catalog.len() > MAX_CONFIGURED_PHYSICAL_LANES {
                return Err(format!(
                    "configured physical lane count exceeds hard limit {MAX_CONFIGURED_PHYSICAL_LANES}"
                ));
            }
        }
        Ok(Self { lanes: catalog })
    }

    /// Build a catalog directly from deployment-owned raw tuples. The
    /// resulting typed lanes can only be obtained again through exact catalog
    /// resolution; downstream crates cannot manufacture one independently.
    pub fn try_from_raw(
        lanes: impl IntoIterator<Item = (String, String, String)>,
    ) -> Result<Self, String> {
        let lanes = lanes
            .into_iter()
            .map(|(pool, machine_profile, bundle)| {
                PhysicalLane::try_new(&pool, &machine_profile, &bundle)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Self::try_new(lanes)
    }

    /// Union two already-validated deployment catalogs. Equivalent lanes
    /// appearing in multiple managed manifests are one physical target, while
    /// the global hard cap remains enforced.
    #[allow(dead_code)] // Public managed-composition API; unused by the standalone binary target.
    pub fn try_union(&self, other: &Self) -> Result<Self, String> {
        let mut lanes = self.lanes.clone();
        lanes.extend(other.lanes.iter().cloned());
        if lanes.len() > MAX_CONFIGURED_PHYSICAL_LANES {
            return Err(format!(
                "configured physical lane count exceeds hard limit {MAX_CONFIGURED_PHYSICAL_LANES}"
            ));
        }
        Ok(Self { lanes })
    }

    pub fn resolve(&self, pool: &str, machine_profile: &str, bundle: &str) -> Option<PhysicalLane> {
        let candidate = PhysicalLane::try_new(pool, machine_profile, bundle).ok()?;
        self.lanes.get(&candidate).cloned()
    }

    pub fn contains(&self, lane: &PhysicalLane) -> bool {
        self.lanes.contains(lane)
    }

    pub fn len(&self) -> usize {
        self.lanes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lanes.is_empty()
    }

    /// Return the complete deployment-owned lane catalog in stable order.
    ///
    /// KEDA reconciliation uses this finite view to query JetStream once per
    /// configured physical lane. Returning owned values prevents callers from
    /// holding a borrow across those asynchronous broker reads.
    pub(crate) fn lanes(&self) -> Vec<PhysicalLane> {
        let mut lanes: Vec<_> = self.lanes.iter().cloned().collect();
        lanes.sort();
        lanes
    }
}

/// Tracks pending demand only for configured physical KEDA lanes.
///
/// Entries store an expiry deadline rather than spawning one Tokio task per
/// lane. The five-second capacity reconciler prunes expired state while taking
/// its snapshot. The catalog bounds the key space, and the explicit cap is a
/// second line of defense against accidental construction outside that set.
pub struct DemandTracker {
    catalog: PhysicalLaneCatalog,
    /// One preallocated state cell per deployment-owned physical lane. Keeping
    /// pending markers and dispatch handoffs behind the same per-lane DashMap
    /// guard linearizes record/begin/ACK transitions without a global lock.
    lane_states: DashMap<PhysicalLane, LaneDemandState>,
    active_entries: AtomicUsize,
    max_active_lanes: usize,
}

#[derive(Debug)]
struct PendingDemandState {
    expires_at: Instant,
    epoch: u64,
}

#[derive(Debug, Default)]
struct LaneDemandState {
    record_epoch: u64,
    pending: Option<PendingDemandState>,
    in_flight: usize,
    failed_until: Option<Instant>,
}

/// Request-scoped boundary used to retire only demand observed before dispatch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DispatchHandoff {
    clear_pending_through: u64,
}

impl DemandTracker {
    pub fn new(catalog: PhysicalLaneCatalog) -> Self {
        Self::with_max_active_lanes(catalog.clone(), catalog.len())
    }

    fn with_max_active_lanes(catalog: PhysicalLaneCatalog, max_active_lanes: usize) -> Self {
        let lane_states = DashMap::new();
        for lane in catalog.lanes() {
            lane_states.insert(lane, LaneDemandState::default());
        }
        Self {
            catalog,
            lane_states,
            active_entries: AtomicUsize::new(0),
            max_active_lanes: max_active_lanes.min(MAX_CONFIGURED_PHYSICAL_LANES),
        }
    }

    pub fn catalog(&self) -> &PhysicalLaneCatalog {
        &self.catalog
    }

    pub fn resolve_lane(
        &self,
        pool: &str,
        machine_profile: &str,
        bundle: &str,
    ) -> Option<PhysicalLane> {
        self.catalog.resolve(pool, machine_profile, bundle)
    }

    /// Record or refresh demand for one catalog-resolved lane.
    ///
    /// Returns `false` if the lane belongs to another catalog or the defensive
    /// active-entry cap is reached. Neither case can create KEDA state.
    pub fn record(&self, lane: &PhysicalLane) -> bool {
        let Some(mut state) = self.lane_states.get_mut(lane) else {
            warn!(lane = %lane, "ignored pending demand outside configured physical lane catalog");
            return false;
        };

        let reserved_new_entry = state.pending.is_none();
        if reserved_new_entry
            && self
                .active_entries
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                    (current < self.max_active_lanes).then_some(current + 1)
                })
                .is_err()
        {
            warn!(
                lane = %lane,
                max_active_lanes = self.max_active_lanes,
                "ignored pending demand because the bounded tracker is full"
            );
            return false;
        }

        let Some(epoch) = state.record_epoch.checked_add(1) else {
            if reserved_new_entry {
                self.active_entries.fetch_sub(1, Ordering::AcqRel);
            }
            warn!(lane = %lane, "pending demand epoch exhausted; retaining existing marker");
            return false;
        };
        state.record_epoch = epoch;
        state.pending = Some(PendingDemandState {
            // Compute the deadline while holding the lane guard so a delayed
            // older call cannot overwrite a newer record with an earlier one.
            expires_at: Instant::now() + Duration::from_secs(DEMAND_EXPIRY_SECS),
            epoch,
        });
        true
    }

    #[cfg(test)]
    pub fn clear(&self, lane: &PhysicalLane) {
        if let Some(mut state) = self.lane_states.get_mut(lane) {
            if state.pending.take().is_some() {
                self.active_entries.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }

    /// Start one request-scoped durability handoff for a configured lane.
    ///
    /// The handoff itself is scale-worthy demand until the transport confirms
    /// durable acceptance. The per-lane counter prevents a successful request
    /// from hiding another concurrent request that is still awaiting an ACK.
    pub(crate) fn begin_dispatch_handoff(&self, lane: &PhysicalLane) -> Option<DispatchHandoff> {
        let Some(mut state) = self.lane_states.get_mut(lane) else {
            warn!(lane = %lane, "ignored dispatch handoff outside configured physical lane catalog");
            return None;
        };

        // A later successful ACK may retire an earlier cold/backpressure
        // marker for this lane because durable broker backlog has taken over
        // as the scale signal. A marker recorded after this boundary belongs
        // to newer demand and must survive that ACK.
        let handoff = DispatchHandoff {
            clear_pending_through: state.record_epoch,
        };

        let Some(next) = state.in_flight.checked_add(1) else {
            warn!(lane = %lane, "dispatch handoff counter overflow; retaining lane demand");
            extend_failure_deadline(&mut state, Instant::now());
            return None;
        };
        state.in_flight = next;
        Some(handoff)
    }

    /// Complete one request-scoped durability handoff.
    ///
    /// A successful completion releases only its own lease and retires an older
    /// same-lane cold/backpressure marker once broker backlog owns the signal.
    /// A failed or abandoned completion leaves a bounded 120-second demand
    /// marker so KEDA fails toward capacity even when JetStream never acquired
    /// backlog.
    pub(crate) fn finish_dispatch_handoff(
        &self,
        lane: &PhysicalLane,
        handoff: DispatchHandoff,
        durable: bool,
    ) {
        let Some(mut state) = self.lane_states.get_mut(lane) else {
            return;
        };
        state.in_flight = state.in_flight.saturating_sub(1);
        if durable {
            if state
                .pending
                .as_ref()
                .is_some_and(|pending| pending.epoch <= handoff.clear_pending_through)
            {
                state.pending = None;
                self.active_entries.fetch_sub(1, Ordering::AcqRel);
            }
        } else {
            extend_failure_deadline(&mut state, Instant::now());
        }
    }

    /// Snapshot active demand in deterministic order and prune expired state.
    pub fn active_lanes(&self) -> Vec<PhysicalLane> {
        let now = Instant::now();
        let mut lanes = Vec::new();
        for mut entry in self.lane_states.iter_mut() {
            let lane = entry.key().clone();
            let state = entry.value_mut();
            if state
                .pending
                .as_ref()
                .is_some_and(|pending| pending.expires_at <= now)
            {
                state.pending = None;
                self.active_entries.fetch_sub(1, Ordering::AcqRel);
                info!(
                    pool = lane.pool(),
                    machine_profile = lane.machine_profile(),
                    bundle = lane.bundle(),
                    "pending demand expired (no requests for 120s)"
                );
            }
            if state
                .failed_until
                .is_some_and(|expires_at| expires_at <= now)
            {
                state.failed_until = None;
                info!(
                    pool = lane.pool(),
                    machine_profile = lane.machine_profile(),
                    bundle = lane.bundle(),
                    "failed dispatch demand expired after 120s"
                );
            }
            if state.pending.is_some() || state.in_flight > 0 || state.failed_until.is_some() {
                lanes.push(lane);
            }
        }
        lanes.sort();
        lanes
    }
}

fn extend_failure_deadline(state: &mut LaneDemandState, now: Instant) {
    let deadline = now + Duration::from_secs(DEMAND_EXPIRY_SECS);
    state.failed_until = Some(
        state
            .failed_until
            .map_or(deadline, |current| current.max(deadline)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lane(pool: &str, machine_profile: &str, bundle: &str) -> PhysicalLane {
        PhysicalLane::try_new(pool, machine_profile, bundle).unwrap()
    }

    fn tracker(lanes: &[PhysicalLane]) -> DemandTracker {
        DemandTracker::new(PhysicalLaneCatalog::try_new(lanes.iter().cloned()).unwrap())
    }

    #[test]
    fn lane_catalog_canonicalizes_and_resolves_exact_tuple() {
        let configured = lane(" Customer_A ", " CPU_One ", " DEFAULT ");
        let catalog = PhysicalLaneCatalog::try_new([configured.clone()]).unwrap();

        assert_eq!(configured.to_string(), "customer_a|cpu_one|default");
        assert_eq!(
            catalog.resolve("CUSTOMER_A", "cpu_ONE", "default"),
            Some(configured)
        );
        assert!(catalog
            .resolve("customer_b", "cpu_one", "default")
            .is_none());
        assert!(catalog.resolve("customer_a", "other", "default").is_none());
    }

    #[test]
    fn lane_catalog_rejects_invalid_duplicate_and_oversized_input() {
        assert!(PhysicalLane::try_new("default", "", "default")
            .unwrap_err()
            .contains("machineProfile must not be empty"));
        assert!(PhysicalLane::try_new("default", "l4/spot", "default")
            .unwrap_err()
            .contains("must match"));
        assert!(PhysicalLane::try_new("default", &"x".repeat(64), "default")
            .unwrap_err()
            .contains("at most 63"));

        let duplicate = lane("default", "l4", "default");
        assert!(PhysicalLaneCatalog::try_new([duplicate.clone(), duplicate])
            .unwrap_err()
            .contains("duplicate configured physical lane"));

        let too_many = (0..=MAX_CONFIGURED_PHYSICAL_LANES)
            .map(|index| lane(&format!("pool-{index}"), "l4", "default"));
        assert!(PhysicalLaneCatalog::try_new(too_many)
            .unwrap_err()
            .contains("exceeds hard limit"));
    }

    #[tokio::test]
    async fn record_tracks_refreshes_and_clears_catalog_lane() {
        let configured = lane("default", "l4-spot", "default");
        let tracker = tracker(std::slice::from_ref(&configured));

        assert!(tracker.record(&configured));
        assert!(tracker.record(&configured));
        assert_eq!(tracker.active_lanes(), vec![configured.clone()]);
        tracker.clear(&configured);
        assert!(tracker.active_lanes().is_empty());
    }

    #[tokio::test]
    async fn clear_removes_only_the_exact_accepted_physical_lane() {
        let accepted = lane("default", "l4", "default");
        let unrelated = lane("default", "a100", "default");
        let tracker = tracker(&[accepted.clone(), unrelated.clone()]);
        assert!(tracker.record(&accepted));
        assert!(tracker.record(&unrelated));

        tracker.clear(&accepted);

        assert_eq!(tracker.active_lanes(), vec![unrelated]);
    }

    #[tokio::test(start_paused = true)]
    async fn concurrent_dispatch_handoffs_are_request_scoped_and_fail_safe() {
        let configured = lane("default", "l4", "default");
        let tracker = tracker(std::slice::from_ref(&configured));

        let first = tracker.begin_dispatch_handoff(&configured).unwrap();
        let second = tracker.begin_dispatch_handoff(&configured).unwrap();
        assert_eq!(tracker.active_lanes(), vec![configured.clone()]);

        // One durable request cannot hide the other request still awaiting an
        // ACK. If that second request fails, later successful requests cannot
        // clear its bounded failure marker either.
        tracker.finish_dispatch_handoff(&configured, first, true);
        assert_eq!(tracker.active_lanes(), vec![configured.clone()]);
        tracker.finish_dispatch_handoff(&configured, second, false);
        let third = tracker.begin_dispatch_handoff(&configured).unwrap();
        tracker.finish_dispatch_handoff(&configured, third, true);
        assert_eq!(tracker.active_lanes(), vec![configured.clone()]);

        tokio::time::advance(Duration::from_secs(DEMAND_EXPIRY_SECS + 1)).await;
        assert!(tracker.active_lanes().is_empty());
    }

    #[tokio::test]
    async fn durable_dispatch_clears_preexisting_lane_demand() {
        let configured = lane("default", "l4", "default");
        let tracker = tracker(std::slice::from_ref(&configured));

        assert!(tracker.record(&configured));
        let handoff = tracker.begin_dispatch_handoff(&configured).unwrap();
        tracker.finish_dispatch_handoff(&configured, handoff, true);

        assert!(tracker.active_lanes().is_empty());
    }

    #[tokio::test]
    async fn durable_dispatch_preserves_demand_recorded_after_handoff_started() {
        let configured = lane("default", "l4", "default");
        let tracker = tracker(std::slice::from_ref(&configured));

        let handoff = tracker.begin_dispatch_handoff(&configured).unwrap();
        assert!(tracker.record(&configured));
        tracker.finish_dispatch_handoff(&configured, handoff, true);

        assert_eq!(tracker.active_lanes(), vec![configured]);
    }

    #[tokio::test]
    async fn later_handoff_clears_only_the_marker_it_observed() {
        let configured = lane("default", "l4", "default");
        let tracker = tracker(std::slice::from_ref(&configured));

        assert!(tracker.record(&configured));
        let first = tracker.begin_dispatch_handoff(&configured).unwrap();
        assert!(tracker.record(&configured));
        let second = tracker.begin_dispatch_handoff(&configured).unwrap();

        tracker.finish_dispatch_handoff(&configured, first, true);
        assert!(tracker
            .lane_states
            .get(&configured)
            .unwrap()
            .pending
            .is_some());

        tracker.finish_dispatch_handoff(&configured, second, true);
        assert!(tracker.active_lanes().is_empty());
    }

    #[test]
    fn record_and_handoff_share_one_per_lane_transition_guard() {
        use std::sync::{mpsc, Arc};

        let configured = lane("default", "l4", "default");
        let tracker = Arc::new(tracker(std::slice::from_ref(&configured)));
        let lane_guard = tracker.lane_states.get_mut(&configured).unwrap();

        let (record_started_tx, record_started_rx) = mpsc::channel();
        let (record_done_tx, record_done_rx) = mpsc::channel();
        let record_tracker = Arc::clone(&tracker);
        let record_lane = configured.clone();
        let record_thread = std::thread::spawn(move || {
            record_started_tx.send(()).unwrap();
            record_done_tx
                .send(record_tracker.record(&record_lane))
                .unwrap();
        });
        record_started_rx.recv().unwrap();

        let (handoff_started_tx, handoff_started_rx) = mpsc::channel();
        let (handoff_done_tx, handoff_done_rx) = mpsc::channel();
        let handoff_tracker = Arc::clone(&tracker);
        let handoff_lane = configured.clone();
        let handoff_thread = std::thread::spawn(move || {
            handoff_started_tx.send(()).unwrap();
            handoff_done_tx
                .send(handoff_tracker.begin_dispatch_handoff(&handoff_lane))
                .unwrap();
        });
        handoff_started_rx.recv().unwrap();

        assert!(matches!(
            record_done_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        assert!(matches!(
            handoff_done_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));

        drop(lane_guard);
        assert!(record_done_rx.recv_timeout(Duration::from_secs(1)).unwrap());
        let handoff = handoff_done_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap();
        record_thread.join().unwrap();
        handoff_thread.join().unwrap();

        let pending_epoch = tracker
            .lane_states
            .get(&configured)
            .unwrap()
            .pending
            .as_ref()
            .unwrap()
            .epoch;
        tracker.finish_dispatch_handoff(&configured, handoff, true);
        let pending_survived = tracker
            .lane_states
            .get(&configured)
            .unwrap()
            .pending
            .is_some();
        assert_eq!(
            pending_survived,
            pending_epoch > handoff.clear_pending_through,
            "the durable ACK must clear exactly the records linearized before its handoff"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn refresh_extends_expiry_without_background_task() {
        let configured = lane("default", "l4", "default");
        let tracker = tracker(std::slice::from_ref(&configured));

        assert!(tracker.record(&configured));
        tokio::time::advance(Duration::from_secs(100)).await;
        assert!(tracker.record(&configured));
        tokio::time::advance(Duration::from_secs(100)).await;
        assert_eq!(tracker.active_lanes(), vec![configured.clone()]);
        tokio::time::advance(Duration::from_secs(21)).await;
        assert!(tracker.active_lanes().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn adversarial_unique_candidates_create_neither_entries_nor_tasks() {
        let configured = lane("default", "l4", "default");
        let tracker = tracker(std::slice::from_ref(&configured));
        let tasks_before = tokio::runtime::Handle::current()
            .metrics()
            .num_alive_tasks();

        for index in 0..10_000 {
            let profile = format!("caller-{index}");
            assert!(tracker
                .resolve_lane("default", &profile, "default")
                .is_none());
        }
        tokio::task::yield_now().await;

        assert!(tracker.active_lanes().is_empty());
        assert_eq!(
            tokio::runtime::Handle::current()
                .metrics()
                .num_alive_tasks(),
            tasks_before,
            "candidate resolution and demand recording must not spawn one task per lane"
        );
    }

    #[tokio::test]
    async fn defensive_active_entry_cap_rejects_new_catalog_lane() {
        let first = lane("default", "l4", "default");
        let second = lane("default", "a100", "default");
        let catalog = PhysicalLaneCatalog::try_new([first.clone(), second.clone()]).unwrap();
        let tracker = DemandTracker::with_max_active_lanes(catalog, 1);

        assert!(tracker.record(&first));
        assert!(!tracker.record(&second));
        assert_eq!(tracker.active_lanes(), vec![first]);
    }
}
