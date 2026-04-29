//! Monotonic epoch counter tracking how far forward this gateway's view of the
//! control plane has advanced.
//!
//! The value is the highest epoch we've either (a) fetched from
//! `GET /v1/configs/export` during bootstrap/recovery or (b) observed on an
//! incoming NATS config delta. The counter is lock-free and can be read from
//! any task.
//!
//! The epoch poller (`state::config_poller`) compares this value to
//! `GET /v1/configs/epoch` on `sie-config`. If `sie-config` is ahead, the
//! poller triggers a fresh export fetch to catch up — this is the only
//! mechanism that closes the staleness window when NATS deltas were silently
//! dropped (NATS Core pub/sub has no replay).
//!
//! Invariant: `set_max` only increases the counter. We never roll backward,
//! so if bootstrap and an inbound delta race, the larger wins.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Shared, monotonically non-decreasing epoch counter.
#[derive(Debug, Clone, Default)]
pub struct ConfigEpoch {
    inner: Arc<AtomicU64>,
}

impl ConfigEpoch {
    pub fn new() -> Self {
        Self::default()
    }

    /// Current best-known epoch. `0` means "never bootstrapped / no deltas
    /// seen yet".
    pub fn get(&self) -> u64 {
        self.inner.load(Ordering::Relaxed)
    }

    /// Set the epoch to `value` iff `value` is strictly greater than the
    /// current value. Returns `true` if the value moved forward. On a
    /// successful forward move, mirrors the new value into the
    /// `sie_gateway_config_epoch` Prometheus gauge.
    pub fn set_max(&self, value: u64) -> bool {
        let mut current = self.inner.load(Ordering::Relaxed);
        loop {
            if value <= current {
                return false;
            }
            match self.inner.compare_exchange_weak(
                current,
                value,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    // Mirror into Prometheus gauge. Use `i64` cast: the
                    // epoch is a small monotonic counter; overflow past
                    // `i64::MAX` would take centuries of writes.
                    crate::metrics::CONFIG_EPOCH.set(value as i64);
                    return true;
                }
                Err(observed) => current = observed,
            }
        }
    }

    /// Force the epoch to `value` unconditionally, bypassing the
    /// monotonic invariant. MUST ONLY be called when the caller has
    /// authoritative evidence that the local counter is ahead of the
    /// control plane (specifically: `state::config_poller` observed
    /// `remote < local` against `sie-config`). The caller is expected
    /// to follow this with a full export fetch so the registry and the
    /// counter land on the same snapshot. Also mirrored to the
    /// `sie_gateway_config_epoch` gauge so the exported value matches
    /// reality.
    pub fn force_set(&self, value: u64) {
        self.inner.store(value, Ordering::Relaxed);
        crate::metrics::CONFIG_EPOCH.set(value as i64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_at_zero() {
        let e = ConfigEpoch::new();
        assert_eq!(e.get(), 0);
    }

    #[test]
    fn set_max_only_advances() {
        let e = ConfigEpoch::new();
        assert!(e.set_max(5));
        assert_eq!(e.get(), 5);
        assert!(!e.set_max(3));
        assert_eq!(e.get(), 5);
        assert!(!e.set_max(5));
        assert_eq!(e.get(), 5);
        assert!(e.set_max(10));
        assert_eq!(e.get(), 10);
    }

    #[test]
    fn clones_share_state() {
        let a = ConfigEpoch::new();
        let b = a.clone();
        a.set_max(7);
        assert_eq!(b.get(), 7);
    }

    #[test]
    fn force_set_bypasses_monotonic_invariant() {
        let e = ConfigEpoch::new();
        e.set_max(42);
        assert_eq!(e.get(), 42);
        // Normal `set_max` refuses to go backward.
        assert!(!e.set_max(10));
        assert_eq!(e.get(), 42);
        // `force_set` overrides that and resets the counter to the
        // authoritative value. Used exclusively by the epoch poller's
        // local > remote recovery path.
        e.force_set(10);
        assert_eq!(e.get(), 10);
        // And after a force_set we can still advance normally.
        assert!(e.set_max(15));
        assert_eq!(e.get(), 15);
    }
}
