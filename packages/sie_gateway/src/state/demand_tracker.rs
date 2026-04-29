use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tracing::info;

use crate::metrics;

const DEMAND_EXPIRY_SECS: u64 = 120;

/// Tracks pending demand per (gpu, bundle) with refreshable expiry deadlines.
/// Each active demand entry has a background task that clears the metric
/// after 120s of inactivity. Calling `record` resets the timer.
pub struct DemandTracker {
    /// Map from (gpu_lowercase, bundle_lowercase) to (original_gpu, original_bundle, notify).
    entries: DashMap<(String, String), (String, String, Arc<Notify>)>,
}

impl Default for DemandTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl DemandTracker {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
        }
    }

    /// Record pending demand for a (gpu, bundle) pair.
    /// Sets the PENDING_DEMAND gauge to 1.0 and starts/refreshes
    /// the 120s auto-expiry timer.
    pub fn record(self: &Arc<Self>, gpu: &str, bundle: &str) {
        let key = (gpu.to_lowercase(), bundle.to_lowercase());

        // Use entry() API for atomic get-or-insert to avoid TOCTOU race
        match self.entries.entry(key.clone()) {
            Entry::Occupied(entry) => {
                let (ref orig_gpu, ref orig_bundle, ref notify) = *entry.get();
                // Set gauge with the original-case labels to avoid creating duplicate series
                metrics::PENDING_DEMAND
                    .with_label_values(&[orig_gpu, orig_bundle])
                    .set(1.0);
                notify.notify_one();
            }
            Entry::Vacant(entry) => {
                let notify = Arc::new(Notify::new());
                entry.insert((gpu.to_string(), bundle.to_string(), Arc::clone(&notify)));

                // Set gauge with the canonical label casing (first caller defines it)
                metrics::PENDING_DEMAND
                    .with_label_values(&[gpu, bundle])
                    .set(1.0);

                let gpu_owned = gpu.to_string();
                let bundle_owned = bundle.to_string();
                let tracker = Arc::clone(self);

                tokio::spawn(async move {
                    let key = (gpu_owned.to_lowercase(), bundle_owned.to_lowercase());
                    loop {
                        if !tracker.entries.contains_key(&key) {
                            return;
                        }
                        tokio::select! {
                            _ = tokio::time::sleep(Duration::from_secs(DEMAND_EXPIRY_SECS)) => {
                                metrics::PENDING_DEMAND
                                    .with_label_values(&[&gpu_owned, &bundle_owned])
                                    .set(0.0);
                                tracker.entries.remove(&key);
                                info!(
                                    gpu = %gpu_owned,
                                    bundle = %bundle_owned,
                                    "pending demand expired (no requests for 120s)"
                                );
                                return;
                            }
                            _ = notify.notified() => {
                                continue;
                            }
                        }
                    }
                });
            }
        }
    }

    /// Explicitly clear demand for a (gpu, bundle) pair.
    /// Removes the entry and zeros the gauge.
    pub fn clear(&self, gpu: &str, bundle: &str) {
        let key = (gpu.to_lowercase(), bundle.to_lowercase());
        if let Some((_, (orig_gpu, orig_bundle, notify))) = self.entries.remove(&key) {
            notify.notify_one();
            metrics::PENDING_DEMAND
                .with_label_values(&[&orig_gpu, &orig_bundle])
                .set(0.0);
        } else {
            // No tracked entry, but clear the gauge anyway with the provided values
            metrics::PENDING_DEMAND
                .with_label_values(&[gpu, bundle])
                .set(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_record_sets_demand_gauge() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("l4-spot", "default");
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["l4-spot", "default"])
            .get();
        assert!((val - 1.0).abs() < f64::EPSILON);
        tracker.clear("l4-spot", "default");
    }

    #[tokio::test]
    async fn test_demand_entry_exists_after_record() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("test-gpu-expire", "test-bundle-expire");
        assert!(tracker.entries.contains_key(&(
            "test-gpu-expire".to_string(),
            "test-bundle-expire".to_string()
        )));
        tracker.clear("test-gpu-expire", "test-bundle-expire");
    }

    #[tokio::test]
    async fn test_record_refreshes_existing_timer() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("l4-refresh", "default");
        // Record again -- should not create a second entry
        tracker.record("l4-refresh", "default");
        assert!(tracker
            .entries
            .contains_key(&("l4-refresh".to_string(), "default".to_string())));
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["l4-refresh", "default"])
            .get();
        assert!((val - 1.0).abs() < f64::EPSILON);
        tracker.clear("l4-refresh", "default");
    }

    #[tokio::test]
    async fn test_clear_removes_entry_and_zeros_gauge() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("l4-clear", "default");
        tracker.clear("l4-clear", "default");
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["l4-clear", "default"])
            .get();
        assert!((val - 0.0).abs() < f64::EPSILON);
        assert!(!tracker
            .entries
            .contains_key(&("l4-clear".to_string(), "default".to_string())));
    }

    #[tokio::test]
    async fn test_clear_nonexistent_is_noop() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        // Should not panic
        tracker.clear("nonexistent", "nonexistent");
    }

    #[tokio::test]
    async fn test_case_insensitive_key_matching() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("L4-Spot", "Premium");
        // Second record with same case should refresh, not create new
        tracker.record("L4-Spot", "Premium");
        assert_eq!(tracker.entries.len(), 1);
        // Gauge was set with the original case labels
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["L4-Spot", "Premium"])
            .get();
        assert!((val - 1.0).abs() < f64::EPSILON);
        tracker.clear("L4-SPOT", "PREMIUM");
        assert_eq!(tracker.entries.len(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_timer_refresh_extends_expiry() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());

        // Record demand
        tracker.record("l4-timer", "default-timer");
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["l4-timer", "default-timer"])
            .get();
        assert!((val - 1.0).abs() < f64::EPSILON);

        // Advance 100s (< 120s expiry), then refresh.
        tokio::time::advance(Duration::from_secs(100)).await;
        tokio::task::yield_now().await;
        tracker.record("l4-timer", "default-timer");

        // Let the spawned task process the notify and loop back to start
        // a new sleep(120s) before we advance time further.
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Advance another 100s. The spawned task restarted its 120s sleep
        // after the notify wakeup, so this 100s advance does NOT expire it.
        tokio::time::advance(Duration::from_secs(100)).await;
        tokio::task::yield_now().await;

        // Gauge should STILL be 1.0 because the timer was refreshed
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["l4-timer", "default-timer"])
            .get();
        assert!(
            (val - 1.0).abs() < f64::EPSILON,
            "expected gauge=1.0 after refresh, got {val}"
        );

        // Advance 25s more (125s since last record, past the 120s expiry).
        tokio::time::advance(Duration::from_secs(25)).await;
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Gauge should now be 0.0 (expired)
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["l4-timer", "default-timer"])
            .get();
        assert!(
            (val - 0.0).abs() < f64::EPSILON,
            "expected gauge=0.0 after expiry, got {val}"
        );
    }

    #[tokio::test]
    async fn test_multiple_gpu_bundle_pairs_independent() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("l4-multi", "default");
        tracker.record("a100-multi", "premium");
        assert_eq!(tracker.entries.len(), 2);
        tracker.clear("l4-multi", "default");
        assert_eq!(tracker.entries.len(), 1);
        assert!(tracker
            .entries
            .contains_key(&("a100-multi".to_string(), "premium".to_string())));
        tracker.clear("a100-multi", "premium");
    }
}
