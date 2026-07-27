//! Worker-local tombstones for request and direct-fallback cancellation.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::subject::normalize_model_id;

const BATCH_CANCEL_TTL: Duration = Duration::from_secs(120);
const REQUEST_CANCEL_MAX_ENTRIES: usize = 100_000;
const REQUEST_CANCEL_ORDER_COMPACTION_FACTOR: usize = 2;

#[derive(Clone, Default)]
pub struct BatchCancelState {
    inner: Arc<Mutex<HashMap<String, Instant>>>,
}

#[derive(Clone)]
pub struct RequestCancelState {
    inner: Arc<Mutex<RequestCancelEntries>>,
    ttl: Duration,
    max_entries: usize,
}

struct RequestCancelEntries {
    entries: HashMap<String, HashMap<String, RequestCancelEntry>>,
    order: VecDeque<RequestCancelOrderEntry>,
    len: usize,
    next_generation: u64,
}

#[derive(Clone, Copy)]
struct RequestCancelEntry {
    expires_at: Instant,
    generation: u64,
}

struct RequestCancelOrderEntry {
    router_id: String,
    request_id: String,
    expires_at: Instant,
    generation: u64,
}

impl BatchCancelState {
    pub(crate) fn cancel(&self, request_id: String) {
        let now = Instant::now();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        prune_expired(&mut guard, now);
        guard.insert(request_id, now + BATCH_CANCEL_TTL);
    }

    pub(crate) fn is_cancelled(&self, request_id: &str) -> bool {
        let now = Instant::now();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        prune_expired(&mut guard, now);
        guard.contains_key(request_id)
    }
}

impl RequestCancelState {
    pub(crate) fn new(ttl: Duration) -> Self {
        Self::with_limit(ttl, REQUEST_CANCEL_MAX_ENTRIES)
    }

    fn with_limit(ttl: Duration, max_entries: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(RequestCancelEntries {
                entries: HashMap::new(),
                order: VecDeque::new(),
                len: 0,
                next_generation: 1,
            })),
            ttl,
            max_entries,
        }
    }

    pub(crate) fn cancel(&self, router_id: String, request_id: String) {
        if self.max_entries == 0 {
            return;
        }
        let now = Instant::now();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        prune_request_cancel_order(&mut guard, now);
        let already_present = guard
            .entries
            .get(&router_id)
            .is_some_and(|requests| requests.contains_key(&request_id));
        if guard.len >= self.max_entries && !already_present {
            evict_oldest_request_cancel(&mut guard);
        }
        let generation = guard.next_generation;
        guard.next_generation = guard.next_generation.wrapping_add(1);
        let expires_at = now + self.ttl;
        guard.entries.entry(router_id.clone()).or_default().insert(
            request_id.clone(),
            RequestCancelEntry {
                expires_at,
                generation,
            },
        );
        guard.order.push_back(RequestCancelOrderEntry {
            router_id,
            request_id,
            expires_at,
            generation,
        });
        if !already_present {
            guard.len += 1;
        }
        compact_request_cancel_order_if_needed(&mut guard, self.max_entries);
    }

    pub(crate) fn is_cancelled(&self, router_id: &str, request_id: &str) -> bool {
        let now = Instant::now();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        match guard
            .entries
            .get(router_id)
            .and_then(|requests| requests.get(request_id))
            .copied()
        {
            Some(entry) if entry.expires_at > now => true,
            Some(_) => {
                remove_request_cancel(&mut guard, router_id, request_id);
                false
            }
            None => false,
        }
    }
}

fn request_cancel_entry_matches(
    state: &RequestCancelEntries,
    ordered: &RequestCancelOrderEntry,
) -> bool {
    state
        .entries
        .get(&ordered.router_id)
        .and_then(|requests| requests.get(&ordered.request_id))
        .is_some_and(|entry| entry.generation == ordered.generation)
}

fn remove_request_cancel(state: &mut RequestCancelEntries, router_id: &str, request_id: &str) {
    let (removed, remove_router) = match state.entries.get_mut(router_id) {
        Some(requests) => (requests.remove(request_id).is_some(), requests.is_empty()),
        None => (false, false),
    };
    if removed {
        state.len -= 1;
    }
    if remove_router {
        state.entries.remove(router_id);
    }
}

fn prune_request_cancel_order(state: &mut RequestCancelEntries, now: Instant) {
    loop {
        let Some(front) = state.order.front() else {
            return;
        };
        if request_cancel_entry_matches(state, front) && front.expires_at > now {
            return;
        }
        let ordered = state.order.pop_front().expect("front checked");
        if request_cancel_entry_matches(state, &ordered) {
            remove_request_cancel(state, &ordered.router_id, &ordered.request_id);
        }
    }
}

fn evict_oldest_request_cancel(state: &mut RequestCancelEntries) {
    while let Some(ordered) = state.order.pop_front() {
        if !request_cancel_entry_matches(state, &ordered) {
            continue;
        }
        remove_request_cancel(state, &ordered.router_id, &ordered.request_id);
        return;
    }
}

fn compact_request_cancel_order_if_needed(state: &mut RequestCancelEntries, max_entries: usize) {
    let max_order_entries = max_entries
        .saturating_mul(REQUEST_CANCEL_ORDER_COMPACTION_FACTOR)
        .max(1);
    if state.order.len() <= max_order_entries {
        return;
    }
    let mut active = state
        .entries
        .iter()
        .flat_map(|(router_id, requests)| {
            requests
                .iter()
                .map(move |(request_id, entry)| RequestCancelOrderEntry {
                    router_id: router_id.clone(),
                    request_id: request_id.clone(),
                    expires_at: entry.expires_at,
                    generation: entry.generation,
                })
        })
        .collect::<Vec<_>>();
    active.sort_unstable_by_key(|entry| (entry.expires_at, entry.generation));
    state.order = active.into();
}

fn prune_expired(entries: &mut HashMap<String, Instant>, now: Instant) {
    entries.retain(|_, expires_at| *expires_at > now);
}

pub(crate) fn request_id_from_batch_cancel_subject(
    subject: &str,
    local_worker_id: &str,
) -> Option<String> {
    let mut parts = subject.splitn(4, '.');
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some("batch_cancel"), Some(_router_id), Some(worker_id), Some(request_id))
            if worker_id == normalize_worker_id(local_worker_id) && !request_id.is_empty() =>
        {
            Some(request_id.to_string())
        }
        _ => None,
    }
}

pub(crate) fn request_id_from_work_cancel_subject(subject: &str) -> Option<(String, String)> {
    let mut parts = subject.splitn(3, '.');
    match (parts.next(), parts.next(), parts.next()) {
        (Some("work_cancel"), Some(router_id), Some(request_id))
            if !router_id.is_empty() && !request_id.is_empty() =>
        {
            Some((router_id.to_string(), request_id.to_string()))
        }
        _ => None,
    }
}

fn normalize_worker_id(worker_id: &str) -> String {
    normalize_model_id(worker_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_cancel_subject_parser_filters_worker_id() {
        assert_eq!(
            request_id_from_batch_cancel_subject("batch_cancel.gw.worker-1.req-1", "worker-1"),
            Some("req-1".to_string())
        );
        assert_eq!(
            request_id_from_batch_cancel_subject(
                "batch_cancel.gw.worker-1_dot_svc.req.with.dots",
                "worker-1.svc"
            ),
            Some("req.with.dots".to_string())
        );
        assert_eq!(
            request_id_from_batch_cancel_subject("batch_cancel.gw.worker-2.req-1", "worker-1"),
            None
        );
        assert_eq!(
            request_id_from_batch_cancel_subject("batch_cancel.gw.worker-1", "worker-1"),
            None
        );
    }

    #[test]
    fn work_cancel_subject_parser_keeps_full_request_id() {
        assert_eq!(
            request_id_from_work_cancel_subject("work_cancel.gw.req.with.dots"),
            Some(("gw".to_string(), "req.with.dots".to_string()))
        );
        assert_eq!(request_id_from_work_cancel_subject("work_cancel.gw"), None);
        assert_eq!(
            request_id_from_work_cancel_subject("work_cancel..req"),
            None
        );
        assert_eq!(request_id_from_work_cancel_subject("cancel.gw.req"), None);
    }

    #[test]
    fn request_cancel_state_expires_tombstones() {
        let state = RequestCancelState::new(Duration::ZERO);
        state.cancel("gw".to_string(), "req-expired".to_string());
        assert!(!state.is_cancelled("gw", "req-expired"));
    }

    #[test]
    fn request_cancel_state_preserves_router_namespace() {
        let state = RequestCancelState::new(Duration::from_secs(1));
        state.cancel("gw-a".to_string(), "req-1".to_string());
        assert!(state.is_cancelled("gw-a", "req-1"));
        assert!(!state.is_cancelled("gw-b", "req-1"));
    }
    #[test]
    fn request_cancel_state_bounds_retained_tombstones() {
        let state = RequestCancelState::with_limit(Duration::from_secs(60), 2);
        state.cancel("gw".into(), "req-1".into());
        state.cancel("gw".into(), "req-2".into());
        state.cancel("gw".into(), "req-3".into());
        let guard = state.inner.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.len, 2);
        assert!(guard.entries.get("gw").unwrap().contains_key("req-3"));
    }

    #[test]
    fn refreshed_tombstone_survives_stale_order_eviction() {
        let state = RequestCancelState::with_limit(Duration::from_secs(60), 2);
        state.cancel("gw".into(), "req-1".into());
        state.cancel("gw".into(), "req-2".into());
        state.cancel("gw".into(), "req-1".into());
        state.cancel("gw".into(), "req-3".into());

        assert!(state.is_cancelled("gw", "req-1"));
        assert!(!state.is_cancelled("gw", "req-2"));
        assert!(state.is_cancelled("gw", "req-3"));
    }

    #[test]
    fn refresh_only_traffic_keeps_order_storage_bounded() {
        let state = RequestCancelState::with_limit(Duration::from_secs(60), 2);
        for _ in 0..100 {
            state.cancel("gw".into(), "req-1".into());
        }
        let guard = state.inner.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.len, 1);
        assert!(guard.order.len() <= 4);
    }

    #[test]
    fn zero_capacity_retains_no_tombstones() {
        let state = RequestCancelState::with_limit(Duration::from_secs(60), 0);
        state.cancel("gw".into(), "req-1".into());
        assert!(!state.is_cancelled("gw", "req-1"));
    }

    #[test]
    fn batch_cancel_state_marks_request_ids() {
        let state = BatchCancelState::default();
        assert!(!state.is_cancelled("req-1"));
        state.cancel("req-1".to_string());
        assert!(state.is_cancelled("req-1"));
        assert!(!state.is_cancelled("req-2"));
    }
}
