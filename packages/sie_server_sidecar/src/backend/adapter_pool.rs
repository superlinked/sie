//! Adapter worker pool backend.
//!
//! This backend owns one IPC client per adapter worker child and keeps
//! model-to-child placement inside the sidecar. The dispatcher still sees a
//! single [`InferenceBackend`]; routing to a concrete adapter process happens
//! here on `EnsureModelReady` and is reused for subsequent batches.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures_util::future::join_all;
use tracing::{debug, info, warn};

use crate::backend::python_ipc::map_ipc_error;
use crate::backend::{BackendError, InferenceBackend};
use crate::ipc_client::{IpcClient, IpcError};
use crate::ipc_types::{
    ApplyModelConfigRequest, ApplyModelConfigResponse, BatchOutcome, DrainResponse,
    EnsureModelReadyResponse, GenerateEvent, PingResponse, ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest, ProcessGenerateRequest, ProcessScoreBatchRequest,
    ReplaceModelConfigsRequest, ReplaceModelConfigsResponse, RunBatchRequest,
    SetPinnedModelsResponse, SignalGenerateCancelResponse, WorkerCapabilitiesResponse,
};
use crate::runtime_state::RuntimeState;

struct AdapterWorkerChild {
    index: usize,
    socket_path: PathBuf,
    ipc: Arc<IpcClient>,
    ready: AtomicBool,
    inflight_batches: AtomicI64,
    pending_items: AtomicI64,
    pending_cost: AtomicI64,
    models: Mutex<HashSet<String>>,
}

impl AdapterWorkerChild {
    fn model_count(&self) -> usize {
        self.models
            .lock()
            .expect("adapter worker child model set poisoned")
            .len()
    }
}

/// Shared pool state used by the backend, heartbeat, config fanout, and
/// generation cancel fanout.
pub struct AdapterWorkerPool {
    children: Vec<Arc<AdapterWorkerChild>>,
    placements: Mutex<HashMap<String, usize>>,
    pinned_models: Mutex<HashSet<String>>,
    pinned_assignment_revision: AtomicU64,
    config_quarantined: AtomicBool,
    config_fanout_generation: AtomicU64,
    runtime_state: Arc<RuntimeState>,
}

impl AdapterWorkerPool {
    pub fn new(
        socket_paths: &[PathBuf],
        ipc_pool_size: usize,
        ipc_request_timeout_s: u64,
        model_ready_timeout_s: u64,
        runtime_state: Arc<RuntimeState>,
    ) -> Arc<Self> {
        let mut children = Vec::with_capacity(socket_paths.len().max(1));
        for (index, socket_path) in socket_paths.iter().enumerate() {
            let ipc = Arc::new(
                IpcClient::new_pool(socket_path, ipc_pool_size)
                    .with_timeout(Duration::from_secs(ipc_request_timeout_s))
                    .with_model_ready_timeout(Duration::from_secs(model_ready_timeout_s))
                    .with_telemetry(runtime_state.telemetry.clone()),
            );
            children.push(Arc::new(AdapterWorkerChild {
                index,
                socket_path: socket_path.clone(),
                ipc,
                ready: AtomicBool::new(false),
                inflight_batches: AtomicI64::new(0),
                pending_items: AtomicI64::new(0),
                pending_cost: AtomicI64::new(0),
                models: Mutex::new(HashSet::new()),
            }));
        }
        assert!(
            !children.is_empty(),
            "AdapterWorkerPool requires at least one IPC socket"
        );
        let pool = Arc::new(Self {
            children,
            placements: Mutex::new(HashMap::new()),
            pinned_models: Mutex::new(HashSet::new()),
            pinned_assignment_revision: AtomicU64::new(0),
            config_quarantined: AtomicBool::new(false),
            config_fanout_generation: AtomicU64::new(0),
            runtime_state,
        });
        pool.runtime_state
            .worker_gpu_slots_total
            .set(pool.children.len() as i64);
        pool.runtime_state.worker_gpu_slots_ready.set(0);
        pool
    }

    pub fn primary_ipc(&self) -> Arc<IpcClient> {
        Arc::clone(&self.children[0].ipc)
    }

    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    pub fn pinned_assignment_revision(&self) -> u64 {
        self.pinned_assignment_revision.load(Ordering::Acquire)
    }

    pub fn ready_child_count(&self) -> usize {
        self.children
            .iter()
            .filter(|child| child.ready.load(Ordering::Acquire))
            .count()
    }

    pub async fn ping_all(
        &self,
        timestamp_ms: f64,
    ) -> Vec<(usize, Result<PingResponse, IpcError>)> {
        let out = join_all(self.children.iter().map(|child| async move {
            let result = child.ipc.ping(timestamp_ms).await;
            if result.as_ref().is_ok_and(|resp| resp.ready) {
                self.mark_child_ready_from_health_success(child);
            } else {
                child.ready.store(false, Ordering::Release);
            }
            (child.index, result)
        }))
        .await;
        self.runtime_state
            .worker_gpu_slots_ready
            .set(self.ready_child_count() as i64);
        out
    }

    pub async fn worker_capabilities(&self) -> Result<WorkerCapabilitiesResponse, IpcError> {
        let mut combined = WorkerCapabilitiesResponse::default();
        let mut any_success = false;
        let mut last_err = None;
        for child in &self.children {
            match child.ipc.worker_capabilities().await {
                Ok(resp) => {
                    any_success = true;
                    self.mark_child_ready_from_health_success(child);
                    combined.has_generation_models |= resp.has_generation_models;
                    for model in resp.generation_models {
                        if !combined.generation_models.contains(&model) {
                            combined.generation_models.push(model);
                        }
                    }
                }
                Err(e) => {
                    child.ready.store(false, Ordering::Release);
                    last_err = Some(e);
                }
            }
        }
        self.runtime_state
            .worker_gpu_slots_ready
            .set(self.ready_child_count() as i64);
        if any_success {
            combined.generation_models.sort();
            Ok(combined)
        } else {
            Err(last_err.expect("last_err set when every capabilities probe failed"))
        }
    }

    pub async fn apply_model_config(
        &self,
        req: ApplyModelConfigRequest,
    ) -> Result<ApplyModelConfigResponse, IpcError> {
        let generation = self.begin_config_fanout("apply_model_config");
        let mut combined = None;
        let mut last_err = None;
        for child in &self.children {
            match child.ipc.apply_model_config(req.clone()).await {
                Ok(resp) => {
                    if !resp.applied {
                        warn!(
                            child_index = child.index,
                            "adapter-worker-pool: child rejected config apply"
                        );
                    }
                    merge_apply_model_config_response(&mut combined, resp);
                }
                Err(e) => {
                    child.ready.store(false, Ordering::Release);
                    last_err = Some(e);
                }
            }
        }
        if let Some(e) = last_err {
            self.quarantine_config_fanout("apply_model_config IPC failure", generation);
            return Err(e);
        }
        let resp = combined.expect("at least one child exists");
        if resp.applied {
            self.clear_config_quarantine_after_success("apply_model_config", generation);
        } else {
            self.quarantine_config_fanout("apply_model_config rejected or diverged", generation);
        }
        Ok(resp)
    }

    pub async fn replace_model_configs(
        &self,
        req: ReplaceModelConfigsRequest,
    ) -> Result<ReplaceModelConfigsResponse, IpcError> {
        let generation = self.begin_config_fanout("replace_model_configs");
        let mut combined = None;
        let mut last_err = None;
        for child in &self.children {
            match child.ipc.replace_model_configs(req.clone()).await {
                Ok(resp) => {
                    if !resp.applied {
                        warn!(
                            child_index = child.index,
                            "adapter-worker-pool: child rejected config replace"
                        );
                    }
                    merge_replace_model_configs_response(&mut combined, resp);
                }
                Err(e) => {
                    child.ready.store(false, Ordering::Release);
                    last_err = Some(e);
                }
            }
        }
        if let Some(e) = last_err {
            self.quarantine_config_fanout("replace_model_configs IPC failure", generation);
            return Err(e);
        }
        let resp = combined.expect("at least one child exists");
        if resp.applied {
            self.clear_config_quarantine_after_success("replace_model_configs", generation);
        } else {
            self.quarantine_config_fanout("replace_model_configs rejected or diverged", generation);
        }
        Ok(resp)
    }

    pub async fn signal_generate_cancel(
        &self,
        request_id: String,
    ) -> Result<SignalGenerateCancelResponse, IpcError> {
        let mut matched = false;
        let mut any_success = false;
        let mut last_err = None;
        for child in &self.children {
            match child.ipc.signal_generate_cancel(request_id.clone()).await {
                Ok(resp) => {
                    any_success = true;
                    matched |= resp.matched;
                }
                Err(e) => {
                    self.mark_child_call_failed(child);
                    last_err = Some(e);
                }
            }
        }
        if any_success {
            Ok(SignalGenerateCancelResponse { matched })
        } else {
            Err(last_err.expect("last_err set when every cancel fanout failed"))
        }
    }

    pub async fn set_pinned_models(
        &self,
        models: Vec<String>,
    ) -> Result<SetPinnedModelsResponse, IpcError> {
        let models = normalize_pinned_models(models);
        self.update_pinned_model_set(&models);
        let models_by_child = self.pinned_models_by_child(models);
        let mut applied = true;
        let mut pinned_count = 0u32;
        for child in &self.children {
            let child_models = models_by_child
                .get(child.index)
                .cloned()
                .unwrap_or_default();
            match child.ipc.set_pinned_models(child_models).await {
                Ok(resp) => {
                    self.mark_child_ready_from_health_success(child);
                    applied &= resp.applied;
                    pinned_count = pinned_count.saturating_add(resp.pinned_count);
                }
                Err(e) => {
                    self.mark_child_call_failed(child);
                    return Err(e);
                }
            }
        }
        self.runtime_state
            .worker_gpu_slots_ready
            .set(self.ready_child_count() as i64);
        Ok(SetPinnedModelsResponse {
            applied,
            pinned_count,
        })
    }

    fn pinned_models_by_child(&self, models: Vec<String>) -> Vec<Vec<String>> {
        let mut assigned = vec![BTreeSet::new(); self.children.len()];
        for model in normalize_pinned_models(models) {
            let child = self.child_for_model(&model);
            assigned[child.index].insert(model);
        }
        assigned
            .into_iter()
            .map(|models| models.into_iter().collect())
            .collect()
    }

    pub async fn process_generate<F, Fut>(
        &self,
        req: ProcessGenerateRequest,
        on_event: F,
    ) -> Result<(), IpcError>
    where
        F: FnMut(GenerateEvent) -> Fut,
        Fut: std::future::Future<Output = Result<(), IpcError>>,
    {
        self.ensure_not_config_quarantined()?;
        let model_id = req.model_id.clone();
        let child = self.child_for_model(&model_id);
        child.inflight_batches.fetch_add(1, Ordering::AcqRel);
        let result = child.ipc.process_generate(req, on_event).await;
        child.inflight_batches.fetch_sub(1, Ordering::AcqRel);
        match &result {
            Ok(()) => self.mark_child_call_succeeded(&child),
            Err(_) => {
                self.mark_child_call_failed(&child);
                self.clear_model_if_on_child(&model_id, child.index);
            }
        }
        result
    }

    pub fn record_model_pending_enqueue(&self, model_id: &str, cost: u64) -> usize {
        let child = self.child_for_model(model_id);
        child.pending_items.fetch_add(1, Ordering::AcqRel);
        child
            .pending_cost
            .fetch_add(clamp_u64_to_i64(cost), Ordering::AcqRel);
        child.index
    }

    pub fn record_child_pending_dequeue(&self, child_index: usize, item_count: usize, cost: u64) {
        let Some(child) = self.children.get(child_index).cloned() else {
            warn!(
                child_index,
                "adapter-worker-pool: ignoring pending dequeue for unknown child"
            );
            return;
        };
        atomic_add_floor_zero(&child.pending_items, -(item_count as i64));
        atomic_add_floor_zero(&child.pending_cost, -clamp_u64_to_i64(cost));
    }

    pub fn record_model_pending_dequeue(&self, model_id: &str, item_count: usize, cost: u64) {
        let index = self
            .placements
            .lock()
            .expect("adapter worker placement map poisoned")
            .get(model_id)
            .copied();
        let Some(index) = index else {
            return;
        };
        let child = Arc::clone(&self.children[index]);
        atomic_add_floor_zero(&child.pending_items, -(item_count as i64));
        atomic_add_floor_zero(&child.pending_cost, -clamp_u64_to_i64(cost));
    }

    pub async fn drain_all(
        &self,
        deadline_ms: u64,
    ) -> Vec<(usize, Result<DrainResponse, IpcError>)> {
        let mut out = Vec::with_capacity(self.children.len());
        for child in &self.children {
            out.push((child.index, child.ipc.drain(deadline_ms).await));
        }
        out
    }

    fn child_for_model(&self, model_id: &str) -> Arc<AdapterWorkerChild> {
        let mut placements = self
            .placements
            .lock()
            .expect("adapter worker placement map poisoned");
        if let Some(&index) = placements.get(model_id) {
            let child = Arc::clone(&self.children[index]);
            if child.ready.load(Ordering::Acquire) || self.ready_child_count() == 0 {
                return child;
            }
            placements.remove(model_id);
            self.mark_pinned_assignment_changed_if_needed(model_id);
            child
                .models
                .lock()
                .expect("adapter worker child model set poisoned")
                .remove(model_id);
        }

        let index = self.choose_child_index();
        placements.insert(model_id.to_string(), index);
        self.mark_pinned_assignment_changed_if_needed(model_id);
        let child = Arc::clone(&self.children[index]);
        child
            .models
            .lock()
            .expect("adapter worker child model set poisoned")
            .insert(model_id.to_string());
        info!(
            model = %model_id,
            child_index = index,
            socket = %child.socket_path.display(),
            "adapter-worker-pool: model placed on child"
        );
        child
    }

    fn update_pinned_model_set(&self, models: &[String]) {
        let next: HashSet<String> = models.iter().cloned().collect();
        let mut pinned_models = self
            .pinned_models
            .lock()
            .expect("adapter worker pinned model set poisoned");
        if *pinned_models != next {
            *pinned_models = next;
            self.pinned_assignment_revision
                .fetch_add(1, Ordering::AcqRel);
        }
    }

    fn mark_pinned_assignment_changed_if_needed(&self, model_id: &str) {
        let pinned = self
            .pinned_models
            .lock()
            .expect("adapter worker pinned model set poisoned")
            .contains(model_id);
        if pinned {
            self.pinned_assignment_revision
                .fetch_add(1, Ordering::AcqRel);
        }
    }

    fn choose_child_index(&self) -> usize {
        let any_ready = self.ready_child_count() > 0;
        self.children
            .iter()
            .min_by_key(|child| {
                let unready_penalty =
                    usize::from(any_ready && !child.ready.load(Ordering::Acquire));
                (
                    unready_penalty,
                    child.model_count(),
                    child.pending_cost.load(Ordering::Relaxed),
                    child.pending_items.load(Ordering::Relaxed),
                    child.inflight_batches.load(Ordering::Relaxed),
                    child.index,
                )
            })
            .expect("at least one child exists")
            .index
    }

    fn clear_model_if_on_child(&self, model_id: &str, child_index: usize) {
        let mut placements = self
            .placements
            .lock()
            .expect("adapter worker placement map poisoned");
        if placements.get(model_id).copied() == Some(child_index) {
            placements.remove(model_id);
            self.mark_pinned_assignment_changed_if_needed(model_id);
            self.children[child_index]
                .models
                .lock()
                .expect("adapter worker child model set poisoned")
                .remove(model_id);
            debug!(
                model = %model_id,
                child_index,
                "adapter-worker-pool: placement cleared after child failure"
            );
        }
    }

    fn mark_child_call_failed(&self, child: &AdapterWorkerChild) {
        child.ready.store(false, Ordering::Release);
        self.runtime_state
            .worker_gpu_slots_ready
            .set(self.ready_child_count() as i64);
    }

    fn mark_child_call_succeeded(&self, child: &AdapterWorkerChild) {
        self.mark_child_ready_from_health_success(child);
        self.runtime_state
            .worker_gpu_slots_ready
            .set(self.ready_child_count() as i64);
    }

    fn mark_child_ready_from_health_success(&self, child: &AdapterWorkerChild) {
        if self.config_quarantined.load(Ordering::Acquire) {
            child.ready.store(false, Ordering::Release);
            return;
        }
        child.ready.store(true, Ordering::Release);
        if self.config_quarantined.load(Ordering::Acquire) {
            child.ready.store(false, Ordering::Release);
        }
    }

    fn ensure_not_config_quarantined(&self) -> Result<(), IpcError> {
        if self.config_quarantined.load(Ordering::Acquire) {
            Err(IpcError::Server(
                "adapter worker pool config fanout incomplete; pool quarantined".to_string(),
            ))
        } else {
            Ok(())
        }
    }

    fn begin_config_fanout(&self, operation: &'static str) -> u64 {
        let generation = self.config_fanout_generation.fetch_add(1, Ordering::AcqRel) + 1;
        self.config_quarantined.store(true, Ordering::Release);
        for child in &self.children {
            child.ready.store(false, Ordering::Release);
        }
        self.runtime_state.worker_gpu_slots_ready.set(0);
        debug!(
            operation,
            generation,
            "adapter-worker-pool: config fanout started; children temporarily quarantined"
        );
        generation
    }

    fn quarantine_config_fanout(&self, reason: &'static str, generation: u64) {
        if self.config_fanout_generation.load(Ordering::Acquire) != generation {
            debug!(
                reason,
                generation, "adapter-worker-pool: ignoring stale config fanout failure"
            );
            return;
        }
        let was_quarantined = self.config_quarantined.swap(true, Ordering::AcqRel);
        if was_quarantined {
            warn!(
                reason,
                "adapter-worker-pool: keeping all children quarantined after config fanout failure"
            );
        } else {
            warn!(
                reason,
                "adapter-worker-pool: quarantining all children after config fanout failure"
            );
        }
        for child in &self.children {
            child.ready.store(false, Ordering::Release);
        }
        self.runtime_state.worker_gpu_slots_ready.set(0);
    }

    fn clear_config_quarantine_after_success(&self, operation: &'static str, generation: u64) {
        if self.config_fanout_generation.load(Ordering::Acquire) != generation {
            debug!(
                operation,
                generation, "adapter-worker-pool: ignoring stale config fanout success"
            );
            return;
        }
        let was_quarantined = self.config_quarantined.swap(false, Ordering::AcqRel);
        if was_quarantined {
            info!(
                operation,
                "adapter-worker-pool: config fanout succeeded; clearing child quarantine"
            );
        }
        for child in &self.children {
            child.ready.store(true, Ordering::Release);
        }
        self.runtime_state
            .worker_gpu_slots_ready
            .set(self.ready_child_count() as i64);
    }

    async fn run_child_batch<F, Fut>(
        &self,
        model_id: String,
        child: Arc<AdapterWorkerChild>,
        call: F,
    ) -> Result<BatchOutcome, IpcError>
    where
        F: FnOnce(Arc<IpcClient>) -> Fut,
        Fut: std::future::Future<Output = Result<BatchOutcome, IpcError>>,
    {
        self.ensure_not_config_quarantined()?;
        child.inflight_batches.fetch_add(1, Ordering::AcqRel);
        let result = call(Arc::clone(&child.ipc)).await;
        child.inflight_batches.fetch_sub(1, Ordering::AcqRel);
        match &result {
            Ok(_) => self.mark_child_call_succeeded(&child),
            Err(_) => {
                self.mark_child_call_failed(&child);
                self.clear_model_if_on_child(&model_id, child.index);
            }
        }
        result
    }

    pub async fn ensure_model_ready_on_placed_child(
        &self,
        model_id: &str,
    ) -> Result<EnsureModelReadyResponse, IpcError> {
        self.ensure_not_config_quarantined()?;
        let child = self.child_for_model(model_id);
        match child.ipc.ensure_model_ready(model_id).await {
            Ok(resp) => {
                self.mark_child_call_succeeded(&child);
                Ok(resp)
            }
            Err(e) => {
                self.mark_child_call_failed(&child);
                self.clear_model_if_on_child(model_id, child.index);
                Err(e)
            }
        }
    }
}

fn clamp_u64_to_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn atomic_add_floor_zero(value: &AtomicI64, delta: i64) {
    if delta >= 0 {
        value.fetch_add(delta, Ordering::AcqRel);
        return;
    }
    let _ = value.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
        Some(current.saturating_add(delta).max(0))
    });
}

fn normalize_pinned_models(models: Vec<String>) -> Vec<String> {
    models
        .into_iter()
        .map(|model| model.trim().to_string())
        .filter(|model| !model.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn merge_apply_model_config_response(
    combined: &mut Option<ApplyModelConfigResponse>,
    resp: ApplyModelConfigResponse,
) {
    let Some(existing) = combined.as_mut() else {
        *combined = Some(resp);
        return;
    };
    existing.applied &= resp.applied;
    merge_bundle_hash(
        &mut existing.applied,
        &mut existing.bundle_config_hash,
        resp.bundle_config_hash,
    );
    existing.config_version = existing.config_version.max(resp.config_version);
}

fn merge_replace_model_configs_response(
    combined: &mut Option<ReplaceModelConfigsResponse>,
    resp: ReplaceModelConfigsResponse,
) {
    let Some(existing) = combined.as_mut() else {
        *combined = Some(resp);
        return;
    };
    existing.applied &= resp.applied;
    merge_bundle_hash(
        &mut existing.applied,
        &mut existing.bundle_config_hash,
        resp.bundle_config_hash,
    );
    existing.config_version = existing.config_version.max(resp.config_version);

    let mut existing_models = existing.applied_models.clone();
    let mut new_models = resp.applied_models;
    existing_models.sort();
    new_models.sort();
    if existing_models != new_models {
        existing.applied = false;
        existing.applied_models = existing_models
            .into_iter()
            .chain(new_models)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        existing.applied_models.sort();
    }

    let mut existing_profiles = existing.applied_profiles.clone();
    let mut new_profiles = resp.applied_profiles;
    existing_profiles.sort();
    new_profiles.sort();
    if existing_profiles != new_profiles {
        existing.applied = false;
        existing.applied_profiles = existing_profiles
            .into_iter()
            .chain(new_profiles)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        existing.applied_profiles.sort();
    }
}

fn merge_bundle_hash(applied: &mut bool, existing_hash: &mut String, next_hash: String) {
    if existing_hash.is_empty() {
        *existing_hash = next_hash;
    } else if !next_hash.is_empty() && *existing_hash != next_hash {
        *applied = false;
    }
}

#[async_trait]
impl InferenceBackend for AdapterWorkerPool {
    fn name(&self) -> &'static str {
        "adapter-worker-pool"
    }

    fn supports(&self, _model_id: &str) -> bool {
        true
    }

    async fn ensure_model_ready(
        &self,
        model_id: &str,
    ) -> Result<EnsureModelReadyResponse, BackendError> {
        self.ensure_model_ready_on_placed_child(model_id)
            .await
            .map_err(map_ipc_error)
    }

    async fn process_encode_batch(
        &self,
        req: ProcessEncodeBatchRequest,
    ) -> Result<BatchOutcome, BackendError> {
        let model_id = req.model_id.clone();
        let child = self.child_for_model(&model_id);
        self.run_child_batch(model_id, child, |ipc| async move {
            ipc.process_encode_batch(req).await
        })
        .await
        .map_err(map_ipc_error)
    }

    async fn process_score_batch(
        &self,
        req: ProcessScoreBatchRequest,
    ) -> Result<BatchOutcome, BackendError> {
        let model_id = req.model_id.clone();
        let child = self.child_for_model(&model_id);
        self.run_child_batch(model_id, child, |ipc| async move {
            ipc.process_score_batch(req).await
        })
        .await
        .map_err(map_ipc_error)
    }

    async fn process_extract_batch(
        &self,
        req: ProcessExtractBatchRequest,
    ) -> Result<BatchOutcome, BackendError> {
        let model_id = req.model_id.clone();
        let child = self.child_for_model(&model_id);
        self.run_child_batch(model_id, child, |ipc| async move {
            ipc.process_extract_batch(req).await
        })
        .await
        .map_err(map_ipc_error)
    }

    async fn run_batch(&self, req: RunBatchRequest) -> Result<BatchOutcome, BackendError> {
        let model_id = req.model_id.clone();
        let child = self.child_for_model(&model_id);
        self.run_child_batch(
            model_id,
            child,
            |ipc| async move { ipc.run_batch(req).await },
        )
        .await
        .map_err(map_ipc_error)
    }

    async fn drain(&self, deadline_ms: u64) {
        for (index, result) in self.drain_all(deadline_ms).await {
            match result {
                Ok(resp) => debug!(
                    child_index = index,
                    acknowledged = resp.acknowledged,
                    "adapter-worker-pool drain acknowledged"
                ),
                Err(e) => warn!(
                    child_index = index,
                    error = %e,
                    "adapter-worker-pool drain RPC failed"
                ),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;

    async fn spawn_cancel_ok(path: PathBuf) -> tokio::task::JoinHandle<()> {
        let listener = UnixListener::bind(path).expect("bind cancel test socket");
        tokio::spawn(async move {
            loop {
                let (mut sock, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => return,
                };
                tokio::spawn(async move {
                    loop {
                        let mut len_buf = [0_u8; 4];
                        if sock.read_exact(&mut len_buf).await.is_err() {
                            return;
                        }
                        let n = u32::from_be_bytes(len_buf) as usize;
                        let mut buf = vec![0_u8; n];
                        if sock.read_exact(&mut buf).await.is_err() {
                            return;
                        }
                        let request: serde_json::Value =
                            rmp_serde::from_slice(&buf).expect("decode cancel request");
                        let request_id = request["request_id"]
                            .as_str()
                            .expect("cancel request envelope id")
                            .to_owned();
                        let resp = crate::ipc_types::ResponseEnvelope {
                            version: crate::ipc_types::IPC_VERSION,
                            request_id,
                            ok: true,
                            body: Some(SignalGenerateCancelResponse { matched: true }),
                            error: None,
                        };
                        let resp = rmp_serde::to_vec_named(&resp).expect("encode cancel response");
                        let len = (resp.len() as u32).to_be_bytes();
                        if sock.write_all(&len).await.is_err() {
                            return;
                        }
                        if sock.write_all(&resp).await.is_err() {
                            return;
                        }
                        if sock.flush().await.is_err() {
                            return;
                        }
                    }
                });
            }
        })
    }

    fn pool_with_children(count: usize) -> Arc<AdapterWorkerPool> {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths: Vec<PathBuf> = (0..count)
            .map(|i| dir.path().join(format!("ipc-{i}.sock")))
            .collect();
        let pool = AdapterWorkerPool::new(&paths, 1, 60, 900, Arc::new(RuntimeState::new()));
        for child in &pool.children {
            child.ready.store(true, Ordering::Release);
        }
        pool.runtime_state
            .worker_gpu_slots_ready
            .set(pool.ready_child_count() as i64);
        pool
    }

    fn pool_with_paths(paths: &[PathBuf]) -> Arc<AdapterWorkerPool> {
        let pool = AdapterWorkerPool::new(paths, 1, 1, 900, Arc::new(RuntimeState::new()));
        for child in &pool.children {
            child.ready.store(true, Ordering::Release);
        }
        pool.runtime_state
            .worker_gpu_slots_ready
            .set(pool.ready_child_count() as i64);
        pool
    }

    fn placement(pool: &AdapterWorkerPool, model_id: &str) -> Option<usize> {
        pool.placements
            .lock()
            .expect("placement map")
            .get(model_id)
            .copied()
    }

    #[test]
    fn keeps_existing_model_on_same_ready_child() {
        let pool = pool_with_children(2);

        let first = pool.child_for_model("model-a").index;
        let second = pool.child_for_model("model-a").index;

        assert_eq!(first, second);
        assert_eq!(placement(&pool, "model-a"), Some(first));
    }

    #[test]
    fn spreads_new_models_by_child_model_count() {
        let pool = pool_with_children(4);

        let placements: Vec<usize> = ["model-a", "model-b", "model-c", "model-d"]
            .into_iter()
            .map(|model| pool.child_for_model(model).index)
            .collect();

        assert_eq!(placements, vec![0, 1, 2, 3]);
    }

    #[test]
    fn partitions_pinned_models_by_placed_child_without_replication() {
        let pool = pool_with_children(3);

        let assigned = pool.pinned_models_by_child(vec![
            "model-a".to_string(),
            "model-b".to_string(),
            "model-c".to_string(),
            "model-a".to_string(),
            " ".to_string(),
        ]);

        assert_eq!(
            assigned,
            vec![
                vec!["model-a".to_string()],
                vec!["model-b".to_string()],
                vec!["model-c".to_string()],
            ]
        );
        assert_eq!(placement(&pool, "model-a"), Some(0));
        assert_eq!(placement(&pool, "model-b"), Some(1));
        assert_eq!(placement(&pool, "model-c"), Some(2));
    }

    #[test]
    fn pinned_assignment_revision_tracks_pinned_model_moves_only() {
        let pool = pool_with_children(2);
        pool.update_pinned_model_set(&["model-a".to_string()]);
        let initial_revision = pool.pinned_assignment_revision();

        let first = pool.child_for_model("model-a").index;
        assert_eq!(first, 0);
        let after_initial_place = pool.pinned_assignment_revision();
        assert!(after_initial_place > initial_revision);

        pool.children[first].ready.store(false, Ordering::Release);
        let moved = pool.child_for_model("model-a").index;
        assert_eq!(moved, 1);
        let after_move = pool.pinned_assignment_revision();
        assert!(after_move > after_initial_place);

        let non_pinned_revision = pool.pinned_assignment_revision();
        let non_pinned_first = pool.child_for_model("model-b").index;
        pool.children[non_pinned_first]
            .ready
            .store(false, Ordering::Release);
        let _ = pool.child_for_model("model-b");
        assert_eq!(pool.pinned_assignment_revision(), non_pinned_revision);
    }

    #[test]
    fn moves_model_when_placed_child_is_unready() {
        let pool = pool_with_children(2);

        let first = pool.child_for_model("model-a").index;
        pool.children[first].ready.store(false, Ordering::Release);
        pool.runtime_state
            .worker_gpu_slots_ready
            .set(pool.ready_child_count() as i64);
        let second = pool.child_for_model("model-a").index;

        assert_ne!(first, second);
        assert_eq!(placement(&pool, "model-a"), Some(second));
        assert!(!pool.children[first]
            .models
            .lock()
            .expect("model set")
            .contains("model-a"));
    }

    #[test]
    fn chooses_least_inflight_child_when_model_counts_match() {
        let pool = pool_with_children(2);
        pool.children[0]
            .inflight_batches
            .store(8, Ordering::Release);

        let chosen = pool.child_for_model("model-a").index;

        assert_eq!(chosen, 1);
    }

    #[test]
    fn chooses_least_pending_cost_when_model_counts_match() {
        let pool = pool_with_children(2);
        pool.children[0]
            .models
            .lock()
            .expect("model set")
            .insert("existing-a".to_string());
        pool.children[1]
            .models
            .lock()
            .expect("model set")
            .insert("existing-b".to_string());
        pool.children[0].pending_cost.store(128, Ordering::Release);
        pool.children[1].pending_cost.store(8, Ordering::Release);

        let chosen = pool.child_for_model("model-a").index;

        assert_eq!(chosen, 1);
    }

    #[test]
    fn records_pending_work_against_placed_child() {
        let pool = pool_with_children(2);

        let child_index = pool.record_model_pending_enqueue("model-a", 10);
        let child = pool.child_for_model("model-a");

        assert_eq!(child_index, child.index);
        assert_eq!(child.pending_items.load(Ordering::Acquire), 1);
        assert_eq!(child.pending_cost.load(Ordering::Acquire), 10);

        pool.record_child_pending_dequeue(child_index, 1, 10);

        assert_eq!(child.pending_items.load(Ordering::Acquire), 0);
        assert_eq!(child.pending_cost.load(Ordering::Acquire), 0);
    }

    #[test]
    fn pending_dequeue_uses_original_child_after_model_moves() {
        let pool = pool_with_children(2);

        let original_child = pool.record_model_pending_enqueue("model-a", 10);
        assert_eq!(original_child, 0);

        pool.children[0].ready.store(false, Ordering::Release);
        let moved_child = pool.child_for_model("model-a");
        assert_eq!(moved_child.index, 1);

        pool.record_child_pending_dequeue(original_child, 1, 10);

        assert_eq!(pool.children[0].pending_items.load(Ordering::Acquire), 0);
        assert_eq!(pool.children[0].pending_cost.load(Ordering::Acquire), 0);
        assert_eq!(pool.children[1].pending_items.load(Ordering::Acquire), 0);
        assert_eq!(pool.children[1].pending_cost.load(Ordering::Acquire), 0);
    }

    #[tokio::test]
    async fn cancel_fanout_failure_refreshes_aggregate_ready_slots() {
        let dir = tempfile::tempdir().expect("tempdir");
        let healthy_path = dir.path().join("healthy.sock");
        let missing_path = dir.path().join("missing.sock");
        let server = spawn_cancel_ok(healthy_path.clone()).await;
        let pool = pool_with_paths(&[healthy_path, missing_path]);

        assert_eq!(pool.runtime_state.worker_gpu_slots_ready.get(), 2);

        let resp = pool
            .signal_generate_cancel("req-cancel".to_string())
            .await
            .expect("one child should accept cancel");

        assert!(resp.matched);
        assert!(pool.children[0].ready.load(Ordering::Acquire));
        assert!(!pool.children[1].ready.load(Ordering::Acquire));
        assert_eq!(pool.runtime_state.worker_gpu_slots_ready.get(), 1);

        server.abort();
    }

    #[test]
    fn config_quarantine_blocks_child_readiness_until_full_success() {
        let pool = pool_with_children(2);

        let generation = pool.begin_config_fanout("test apply");

        assert!(pool.config_quarantined.load(Ordering::Acquire));
        assert_eq!(pool.ready_child_count(), 0);
        assert!(pool.ensure_not_config_quarantined().is_err());

        pool.mark_child_call_succeeded(&pool.children[0]);

        assert_eq!(pool.ready_child_count(), 0);
        assert!(!pool.children[0].ready.load(Ordering::Acquire));

        pool.clear_config_quarantine_after_success("test success", generation);

        assert!(!pool.config_quarantined.load(Ordering::Acquire));
        assert_eq!(pool.ready_child_count(), 2);
        pool.ensure_not_config_quarantined().unwrap();
    }

    #[test]
    fn stale_config_fanout_cannot_clear_newer_quarantine() {
        let pool = pool_with_children(2);

        let stale_generation = pool.begin_config_fanout("stale apply");
        let current_generation = pool.begin_config_fanout("current apply");

        pool.clear_config_quarantine_after_success("stale success", stale_generation);

        assert!(pool.config_quarantined.load(Ordering::Acquire));
        assert_eq!(pool.ready_child_count(), 0);

        pool.clear_config_quarantine_after_success("current success", current_generation);

        assert!(!pool.config_quarantined.load(Ordering::Acquire));
        assert_eq!(pool.ready_child_count(), 2);
    }

    #[test]
    fn stale_config_fanout_failure_cannot_requarantine_after_newer_success() {
        let pool = pool_with_children(2);

        let stale_generation = pool.begin_config_fanout("stale apply");
        let current_generation = pool.begin_config_fanout("current apply");
        pool.clear_config_quarantine_after_success("current success", current_generation);

        pool.quarantine_config_fanout("stale failure", stale_generation);

        assert!(!pool.config_quarantined.load(Ordering::Acquire));
        assert_eq!(pool.ready_child_count(), 2);
    }

    #[test]
    fn clearing_failed_pinned_placement_advances_assignment_revision() {
        let pool = pool_with_children(2);
        pool.update_pinned_model_set(&["model-a".to_string()]);
        let child = pool.child_for_model("model-a");
        let before_clear = pool.pinned_assignment_revision();

        pool.clear_model_if_on_child("model-a", child.index);

        assert!(pool.pinned_assignment_revision() > before_clear);
        assert_eq!(placement(&pool, "model-a"), None);
    }

    #[test]
    fn apply_config_merge_preserves_any_child_rejection() {
        let mut combined = None;

        merge_apply_model_config_response(
            &mut combined,
            ApplyModelConfigResponse {
                applied: false,
                bundle_config_hash: "h1".into(),
                config_version: 1,
            },
        );
        merge_apply_model_config_response(
            &mut combined,
            ApplyModelConfigResponse {
                applied: true,
                bundle_config_hash: "h1".into(),
                config_version: 2,
            },
        );

        let resp = combined.expect("combined response");
        assert!(!resp.applied);
        assert_eq!(resp.bundle_config_hash, "h1");
        assert_eq!(resp.config_version, 2);
    }

    #[test]
    fn apply_config_merge_rejects_child_hash_divergence() {
        let mut combined = None;

        merge_apply_model_config_response(
            &mut combined,
            ApplyModelConfigResponse {
                applied: true,
                bundle_config_hash: "h1".into(),
                config_version: 1,
            },
        );
        merge_apply_model_config_response(
            &mut combined,
            ApplyModelConfigResponse {
                applied: true,
                bundle_config_hash: "h2".into(),
                config_version: 1,
            },
        );

        assert!(!combined.expect("combined response").applied);
    }

    #[test]
    fn replace_config_merge_rejects_child_model_divergence() {
        let mut combined = None;

        merge_replace_model_configs_response(
            &mut combined,
            ReplaceModelConfigsResponse {
                applied: true,
                bundle_config_hash: "h1".into(),
                config_version: 1,
                applied_models: vec!["model-a".into()],
                applied_profiles: vec!["default".into()],
            },
        );
        merge_replace_model_configs_response(
            &mut combined,
            ReplaceModelConfigsResponse {
                applied: true,
                bundle_config_hash: "h1".into(),
                config_version: 1,
                applied_models: vec!["model-b".into()],
                applied_profiles: vec!["default".into()],
            },
        );

        let resp = combined.expect("combined response");
        assert!(!resp.applied);
        assert_eq!(
            resp.applied_models,
            vec!["model-a".to_string(), "model-b".to_string()]
        );
        assert_eq!(resp.applied_profiles, vec!["default".to_string()]);
    }

    #[test]
    fn replace_config_merge_rejects_child_profile_divergence() {
        let mut combined = None;

        for applied_profiles in [
            vec!["default".into()],
            vec!["default".into(), "fast".into()],
        ] {
            merge_replace_model_configs_response(
                &mut combined,
                ReplaceModelConfigsResponse {
                    applied: true,
                    bundle_config_hash: "h1".into(),
                    config_version: 1,
                    applied_models: vec!["model-a".into()],
                    applied_profiles,
                },
            );
        }

        let resp = combined.expect("combined response");
        assert!(!resp.applied);
        assert_eq!(resp.applied_models, vec!["model-a".to_string()]);
        assert_eq!(
            resp.applied_profiles,
            vec!["default".to_string(), "fast".to_string()]
        );
    }
}
