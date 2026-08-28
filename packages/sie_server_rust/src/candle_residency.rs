use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ResidencyPolicy {
    pub pinned: bool,
    pub preload: bool,
}

#[derive(Debug)]
pub(crate) struct CandleResidency<T> {
    entries: HashMap<String, ResidentEntry<T>>,
}

#[derive(Debug)]
struct ResidentEntry<T> {
    model: Arc<T>,
    loaded_at: Instant,
    last_used_at: Instant,
    active_forwards: usize,
    policy: ResidencyPolicy,
}

pub(crate) struct ResidencyUseGuard<T> {
    residency: Arc<Mutex<CandleResidency<T>>>,
    model_id: String,
}

impl<T> Drop for ResidencyUseGuard<T> {
    fn drop(&mut self) {
        if let Ok(mut residency) = self.residency.lock() {
            residency.end_use(&self.model_id, Instant::now());
        }
    }
}

impl<T> ResidencyUseGuard<T> {
    pub(crate) fn active(residency: Arc<Mutex<CandleResidency<T>>>, model_id: String) -> Self {
        Self {
            residency,
            model_id,
        }
    }
}

impl<T> Default for CandleResidency<T> {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

impl<T> CandleResidency<T> {
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub(crate) fn keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    pub(crate) fn get(&self, model_id: &str) -> Option<Arc<T>> {
        self.entries
            .get(model_id)
            .map(|entry| Arc::clone(&entry.model))
    }

    pub(crate) fn get_for_use(&mut self, model_id: &str, now: Instant) -> Option<Arc<T>> {
        let entry = self.entries.get_mut(model_id)?;
        entry.active_forwards += 1;
        entry.last_used_at = now;
        Some(Arc::clone(&entry.model))
    }

    pub(crate) fn insert(
        &mut self,
        model_id: String,
        model: Arc<T>,
        policy: ResidencyPolicy,
        now: Instant,
    ) {
        self.entries.insert(
            model_id,
            ResidentEntry {
                model,
                loaded_at: now,
                last_used_at: now,
                active_forwards: 0,
                policy,
            },
        );
    }

    pub(crate) fn remove(&mut self, model_id: &str) -> Option<Arc<T>> {
        self.entries.remove(model_id).map(|entry| entry.model)
    }

    pub(crate) fn retain<F>(&mut self, mut keep: F) -> usize
    where
        F: FnMut(&str) -> bool,
    {
        let before = self.entries.len();
        self.entries.retain(|model_id, _| keep(model_id));
        before.saturating_sub(self.entries.len())
    }

    pub(crate) fn update_policies<F>(&mut self, mut policy_for: F)
    where
        F: FnMut(&str) -> ResidencyPolicy,
    {
        for (model_id, entry) in &mut self.entries {
            entry.policy = policy_for(model_id);
        }
    }

    fn end_use(&mut self, model_id: &str, now: Instant) {
        let Some(entry) = self.entries.get_mut(model_id) else {
            return;
        };
        if entry.active_forwards > 0 {
            entry.active_forwards -= 1;
            entry.last_used_at = now;
        }
    }

    pub(crate) fn evict_lru_excluding(&mut self, exclude: Option<&str>) -> Option<String> {
        let candidate = self
            .entries
            .iter()
            .filter(|(model_id, entry)| {
                Some(model_id.as_str()) != exclude && entry.is_policy_evictable()
            })
            .min_by_key(|(_, entry)| (entry.last_used_at, entry.loaded_at))
            .map(|(model_id, _)| model_id.clone())?;
        self.entries.remove(&candidate);
        Some(candidate)
    }

    pub(crate) fn evict_idle(&mut self, idle_threshold: Duration, now: Instant) -> Option<String> {
        let candidate = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.is_policy_evictable())
            .filter(|(_, entry)| now.duration_since(entry.last_used_at) >= idle_threshold)
            .min_by_key(|(_, entry)| (entry.last_used_at, entry.loaded_at))
            .map(|(model_id, _)| model_id.clone())?;
        self.entries.remove(&candidate);
        Some(candidate)
    }
}

impl<T> ResidentEntry<T> {
    fn is_policy_evictable(&self) -> bool {
        !self.policy.pinned && self.active_forwards == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_eviction_skips_pinned_active_and_excluded_models() {
        let now = Instant::now();
        let mut residency = CandleResidency::default();
        residency.insert(
            "old".to_string(),
            Arc::new(()),
            ResidencyPolicy::default(),
            now - Duration::from_secs(30),
        );
        residency.insert(
            "pinned".to_string(),
            Arc::new(()),
            ResidencyPolicy {
                pinned: true,
                preload: false,
            },
            now - Duration::from_secs(20),
        );
        residency.insert(
            "active".to_string(),
            Arc::new(()),
            ResidencyPolicy::default(),
            now - Duration::from_secs(10),
        );
        residency.insert(
            "new".to_string(),
            Arc::new(()),
            ResidencyPolicy::default(),
            now,
        );
        let _active_model = residency.get_for_use("active", now).expect("mark active");

        assert_eq!(
            residency.evict_lru_excluding(Some("old")),
            Some("new".to_string())
        );
        assert!(residency.get("old").is_some());
        assert!(residency.get("pinned").is_some());
        assert!(residency.get("active").is_some());
        assert!(residency.get("new").is_none());
    }

    #[test]
    fn idle_eviction_removes_only_cold_evictable_models() {
        let now = Instant::now();
        let mut residency = CandleResidency::default();
        residency.insert(
            "cold".to_string(),
            Arc::new(()),
            ResidencyPolicy::default(),
            now - Duration::from_secs(120),
        );
        residency.insert(
            "warm".to_string(),
            Arc::new(()),
            ResidencyPolicy::default(),
            now - Duration::from_secs(10),
        );
        residency.insert(
            "pinned-cold".to_string(),
            Arc::new(()),
            ResidencyPolicy {
                pinned: true,
                preload: false,
            },
            now - Duration::from_secs(180),
        );

        assert_eq!(
            residency.evict_idle(Duration::from_secs(60), now),
            Some("cold".to_string())
        );
        assert!(residency.get("warm").is_some());
        assert!(residency.get("pinned-cold").is_some());
        assert_eq!(residency.evict_idle(Duration::from_secs(60), now), None);
    }

    #[test]
    fn use_guard_releases_active_marker_on_drop() {
        let now = Instant::now();
        let residency = Arc::new(Mutex::new(CandleResidency::default()));
        residency.lock().unwrap().insert(
            "model".to_string(),
            Arc::new(()),
            ResidencyPolicy::default(),
            now - Duration::from_secs(120),
        );

        let model = residency
            .lock()
            .unwrap()
            .get_for_use("model", now)
            .expect("model should be resident");
        drop(model);
        let guard = ResidencyUseGuard::active(Arc::clone(&residency), "model".to_string());
        assert_eq!(residency.lock().unwrap().evict_lru_excluding(None), None);
        drop(guard);
        assert_eq!(
            residency.lock().unwrap().evict_lru_excluding(None),
            Some("model".to_string())
        );
    }

    #[test]
    fn get_for_use_marks_model_active_before_returning() {
        let now = Instant::now();
        let mut residency = CandleResidency::default();
        residency.insert(
            "model".to_string(),
            Arc::new(()),
            ResidencyPolicy::default(),
            now - Duration::from_secs(120),
        );

        let model = residency
            .get_for_use("model", now)
            .expect("model should be resident");

        assert_eq!(residency.evict_lru_excluding(None), None);
        drop(model);
        residency.end_use("model", now);
        assert_eq!(
            residency.evict_lru_excluding(None),
            Some("model".to_string())
        );
    }

    #[test]
    fn end_use_refreshes_idle_timestamp_after_long_forward() {
        let now = Instant::now();
        let mut residency = CandleResidency::default();
        let request_start = now - Duration::from_secs(120);
        residency.insert(
            "model".to_string(),
            Arc::new(()),
            ResidencyPolicy::default(),
            request_start,
        );
        let model = residency
            .get_for_use("model", request_start)
            .expect("model should be resident");
        drop(model);

        residency.end_use("model", now);

        assert_eq!(residency.evict_idle(Duration::from_secs(60), now), None);
        assert_eq!(
            residency.evict_idle(Duration::from_secs(60), now + Duration::from_secs(60)),
            Some("model".to_string())
        );
    }

    #[test]
    fn retain_reports_removed_entry_count() {
        let now = Instant::now();
        let mut residency = CandleResidency::default();
        for model_id in ["keep", "drop-a", "drop-b"] {
            residency.insert(
                model_id.to_string(),
                Arc::new(()),
                ResidencyPolicy::default(),
                now,
            );
        }

        let removed = residency.retain(|model_id| model_id == "keep");

        assert_eq!(removed, 2);
        assert!(residency.get("keep").is_some());
        assert!(residency.get("drop-a").is_none());
        assert!(residency.get("drop-b").is_none());
    }
}
