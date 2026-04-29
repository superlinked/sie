//! Last-known fingerprint of `sie-config`'s loaded bundle set.
//!
//! Parallel sibling of [`crate::state::config_epoch::ConfigEpoch`]: where
//! `ConfigEpoch` tracks "what model-write generation have we caught up to",
//! `BundlesHash` tracks "what bundle surface have we caught up to". The two
//! drift independently because bundles are filesystem artifacts inside the
//! `sie-config` image — their effective epoch is redeploy time, which the
//! model-write counter does not observe.
//!
//! `state::config_poller` polls `GET /v1/configs/epoch` (which now carries
//! both signals), and re-runs the bundle fetch whenever the remote
//! `bundles_hash` differs from the value stored here. After a successful
//! `BootstrapClient::bootstrap`, `bootstrap_once` records the fresh hash
//! into this container so the next poll tick observes "in sync".
//!
//! The empty string is the documented "nothing to sync" sentinel and is
//! what `sie-config` returns when its registry is unavailable. Storing an
//! empty string here is treated as "we have not caught up yet" and matches
//! against any non-empty remote value, so a fresh gateway always re-fetches
//! on its first successful poll.

use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Default)]
pub struct BundlesHash {
    inner: Arc<Mutex<String>>,
}

impl BundlesHash {
    pub fn new() -> Self {
        Self::default()
    }

    /// Current best-known hash. Empty string means "never installed bundles
    /// successfully" (initial state) or "sie-config is degraded and reported
    /// no hash" — both are treated as "out of sync".
    pub fn get(&self) -> String {
        self.inner
            .lock()
            .expect("BundlesHash mutex poisoned")
            .clone()
    }

    /// Replace the stored hash. Returns `true` if the value actually
    /// changed; the poller and bootstrap paths use this signal for log
    /// hygiene (don't shout "installed bundles" on every poll tick when
    /// nothing moved).
    pub fn store(&self, value: String) -> bool {
        let mut guard = self.inner.lock().expect("BundlesHash mutex poisoned");
        if *guard == value {
            return false;
        }
        *guard = value;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let h = BundlesHash::new();
        assert_eq!(h.get(), "");
    }

    #[test]
    fn store_returns_true_on_change_and_false_on_noop() {
        let h = BundlesHash::new();
        assert!(h.store("abc".to_string()));
        assert_eq!(h.get(), "abc");
        // Same value → no change reported. Avoids spurious "bundles
        // re-installed" log lines on every poll tick when sie-config has
        // not actually moved.
        assert!(!h.store("abc".to_string()));
        assert!(h.store("def".to_string()));
        assert_eq!(h.get(), "def");
    }

    #[test]
    fn clones_share_state() {
        let a = BundlesHash::new();
        let b = a.clone();
        a.store("xyz".to_string());
        assert_eq!(b.get(), "xyz");
    }
}
