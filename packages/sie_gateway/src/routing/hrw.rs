//! Highest-Random-Weight (rendezvous) hashing for direct-dispatch routing.
//!
//! Given a request hash and a snapshot of candidate workers, pick the
//! least-pressured worker and use the HRW score as the deterministic
//! tie-breaker. This is `O(n)` per pick, stable across all callers without
//! coordination for equal-pressure workers, and still degrades gracefully
//! when workers come and go.
//!
//! Snapshots are immutable. The
//! [`crate::state::worker_registry::WorkerRegistry`] owns the source
//! of truth; this module just consumes a borrowed slice of
//! `(worker_id, worker_id_hash)` pairs.

use xxhash_rust::xxh3::xxh3_64;

use super::key::RoutingKeyResolved;

/// One entry on the direct-dispatch ring. Pre-hashing `worker_id` once at
/// snapshot build time keeps `pick_worker` cheap; pressure fields are copied
/// from worker heartbeats so gateway selection can avoid already-busy pods.
#[derive(Debug, Clone)]
pub struct RingEntry {
    pub worker_id: String,
    pub worker_id_hash: u64,
    pub ready_gpu_slots: i32,
    pub queue_depth: i32,
    pub pending_cost: i64,
    pub inflight_batches: i32,
}

impl RingEntry {
    #[cfg(test)]
    pub fn new(worker_id: impl Into<String>) -> Self {
        Self::with_pressure(worker_id, 1, 0, 0, 0)
    }

    pub fn with_pressure(
        worker_id: impl Into<String>,
        ready_gpu_slots: i32,
        queue_depth: i32,
        pending_cost: i64,
        inflight_batches: i32,
    ) -> Self {
        let worker_id = worker_id.into();
        let worker_id_hash = xxh3_64(worker_id.as_bytes());
        Self {
            worker_id,
            worker_id_hash,
            ready_gpu_slots: ready_gpu_slots.max(1),
            queue_depth: queue_depth.max(0),
            pending_cost: pending_cost.max(0),
            inflight_batches: inflight_batches.max(0),
        }
    }
}

/// Immutable worker ring for one `(model, pool, machine_profile, bundle)` lane.
/// `WorkerRegistry::ring_snapshot_for` rebuilds this per request from the
/// registry snapshot; the broader worker view is what lives behind `ArcSwap`,
/// so reads stay lock-free even though this lane ring is freshly constructed.
#[derive(Debug, Clone, Default)]
pub struct RingSnapshot {
    pub entries: Vec<RingEntry>,
}

impl RingSnapshot {
    #[cfg(test)]
    pub fn new(worker_ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            entries: worker_ids.into_iter().map(RingEntry::new).collect(),
        }
    }

    pub fn from_entries(entries: impl IntoIterator<Item = RingEntry>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
        }
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Clippy lints ``len`` without ``is_empty``; expose both even
    /// though :func:`pick` already early-returns on an empty ring.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Pressure-aware HRW pick.
///
/// Lower slot-normalized `pending_cost`, `queue_depth`, and
/// `inflight_batches` win first. HRW is then used as a stable tie-breaker so
/// equal-pressure workers still distribute cache/routing keys deterministically.
///
/// Returns `None` when the snapshot is empty (caller falls back to the
/// pool subject) or when the resolved key has no hash (caller may
/// round-robin or also fall back).
///
/// Ties on `combine` are broken lexicographically by `worker_id` so the
/// pick is identical across gateway replicas even when the snapshot's
/// source iteration order is nondeterministic (e.g. derived from a
/// `HashMap`).
pub fn pick_worker<'a>(snapshot: &'a RingSnapshot, key: &RoutingKeyResolved) -> Option<&'a str> {
    if snapshot.entries.is_empty() {
        return None;
    }
    let request_hash = key.hash?;
    snapshot
        .entries
        .iter()
        .min_by(|a, b| {
            pressure_key(a)
                .cmp(&pressure_key(b))
                .then_with(|| {
                    combine(request_hash, b.worker_id_hash)
                        .cmp(&combine(request_hash, a.worker_id_hash))
                })
                .then_with(|| b.worker_id.cmp(&a.worker_id))
        })
        .map(|e| e.worker_id.as_str())
}

fn pressure_key(entry: &RingEntry) -> (i64, i64, i64, i64, i64, i64) {
    let slots = i64::from(entry.ready_gpu_slots.max(1));
    (
        ceil_div_i64(entry.pending_cost.max(0), slots),
        ceil_div_i64(i64::from(entry.queue_depth.max(0)), slots),
        ceil_div_i64(i64::from(entry.inflight_batches.max(0)), slots),
        entry.pending_cost.max(0),
        i64::from(entry.queue_depth.max(0)),
        i64::from(entry.inflight_batches.max(0)),
    )
}

fn ceil_div_i64(value: i64, divisor: i64) -> i64 {
    if value <= 0 {
        0
    } else {
        (value + divisor.max(1) - 1) / divisor.max(1)
    }
}

/// Combine the request key hash with a worker-id hash.
///
/// Uses an asymmetric mix (`worker_id_hash.rotate_left(32) ^ request_hash`,
/// re-hashed) so the function is *not* invariant under the transformation
/// `(req, wid) → (req ⊕ x, wid ⊕ x)`. A plain `xxh3(req ^ wid)` mix gives
/// identical outputs for any pair with equal XOR — which means hash
/// collisions are not bounded by xxh3's collision resistance, only by the
/// symmetry of the input. Rotating one input before XOR breaks that
/// symmetry while keeping the function deterministic and branch-free.
#[inline]
fn combine(request_hash: u64, worker_id_hash: u64) -> u64 {
    let mixed = (worker_id_hash.rotate_left(32) ^ request_hash).to_le_bytes();
    xxh3_64(&mixed)
}

/// Test-only re-export of [`combine`] for regression tests that need to
/// assert the mix is not XOR-symmetric.
#[cfg(test)]
pub(crate) fn combine_for_test(request_hash: u64, worker_id_hash: u64) -> u64 {
    combine(request_hash, worker_id_hash)
}
