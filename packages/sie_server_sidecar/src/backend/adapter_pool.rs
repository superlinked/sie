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
use serde::Serialize;
use tracing::{debug, info, warn};

use crate::backend::python_ipc::map_ipc_error;
use crate::backend::{BackendError, InferenceBackend};
use crate::ipc_client::{IpcClient, IpcError};
use crate::ipc_types::{
    ApplyModelConfigRequest, ApplyModelConfigResponse, BatchOutcome, DrainResponse,
    EnsureModelReadyResponse, GenerateEvent, NumericalProfileSnapshotResponse, PingResponse,
    ProcessEncodeBatchRequest, ProcessExtractBatchRequest, ProcessGenerateRequest,
    ProcessScoreBatchRequest, ReplaceModelConfigsRequest, ReplaceModelConfigsResponse,
    RunBatchRequest, SetPinnedModelsResponse, SignalGenerateCancelResponse,
    WorkerCapabilitiesResponse,
};
use crate::runtime_state::RuntimeState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NumericalSnapshotStatus {
    Observed,
    Incomplete,
    Unavailable,
    Invalid,
}

#[derive(Debug, Clone, Serialize)]
pub struct NumericalProcessObservation {
    pub child_index: usize,
    pub status: NumericalSnapshotStatus,
    pub snapshot: Option<NumericalProfileSnapshotResponse>,
}

fn sha256_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value))
}

fn valid_numerical_snapshot(snapshot: &NumericalProfileSnapshotResponse) -> bool {
    let mut models = HashSet::new();
    snapshot
        .runtime_instance_id
        .as_deref()
        .is_none_or(sha256_digest)
        && snapshot.profiles.len() <= 1024
        && snapshot.profiles.iter().all(|profile| {
            !profile.model_id.is_empty()
                && profile.model_id.len() <= 1024
                && models.insert(&profile.model_id)
                && profile
                    .model_contract_sha256
                    .as_deref()
                    .is_none_or(sha256_digest)
                && profile.local_identity.as_deref().is_none_or(|identity| {
                    identity
                        .strip_prefix("v1:sha256:")
                        .is_some_and(sha256_digest)
                })
        })
}

struct AdapterWorkerChild {
    index: usize,
    socket_path: PathBuf,
    ipc: Arc<IpcClient>,
    // Diagnostics must not check out a serving/readiness IPC slot.
    numerical_ipc: IpcClient,
    ready: AtomicBool,
    inflight_batches: AtomicI64,
    pending_items: AtomicI64,
    pending_cost: AtomicI64,
    models: Mutex<HashSet<String>>,
}

struct ChildInflightGuard(Arc<AdapterWorkerChild>);

impl ChildInflightGuard {
    fn enter(child: Arc<AdapterWorkerChild>) -> Self {
        child.inflight_batches.fetch_add(1, Ordering::AcqRel);
        Self(child)
    }
}

impl Drop for ChildInflightGuard {
    fn drop(&mut self) {
        self.0.inflight_batches.fetch_sub(1, Ordering::AcqRel);
    }
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
    execution_authority_v1: Arc<AtomicBool>,
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
                numerical_ipc: IpcClient::new(socket_path)
                    .with_timeout(Duration::from_secs(ipc_request_timeout_s))
                    .with_telemetry(runtime_state.telemetry.clone()),
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
            execution_authority_v1: Arc::new(AtomicBool::new(false)),
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

    pub fn execution_authority_v1(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.execution_authority_v1)
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
        let results = join_all(self.children.iter().map(|child| async move {
            let result = child.ipc.ping(timestamp_ms).await;
            if result.as_ref().is_ok_and(|resp| resp.ready) {
                self.mark_child_ready_from_health_success(child);
            } else {
                child.ready.store(false, Ordering::Release);
            }
            let supports_authority = result.as_ref().is_ok_and(|resp| resp.ready)
                && tokio::time::timeout(Duration::from_secs(2), child.ipc.worker_capabilities())
                    .await
                    .ok()
                    .and_then(Result::ok)
                    .is_some_and(|resp| resp.supports_execution_authority_v1);
            (child.index, result, supports_authority)
        }))
        .await;
        self.execution_authority_v1.store(
            !self.config_quarantined.load(Ordering::Acquire)
                && results.iter().all(|(_, _, supports)| *supports),
            Ordering::Release,
        );
        let out = results
            .into_iter()
            .map(|(index, result, _)| (index, result))
            .collect();
        self.runtime_state
            .worker_gpu_slots_ready
            .set(self.ready_child_count() as i64);
        out
    }

    pub async fn worker_capabilities(&self) -> Result<WorkerCapabilitiesResponse, IpcError> {
        let mut combined = WorkerCapabilitiesResponse {
            supports_execution_authority_v1: true,
            ..Default::default()
        };
        let mut any_success = false;
        let mut last_err = None;
        for child in &self.children {
            match child.ipc.worker_capabilities().await {
                Ok(resp) => {
                    combined.supports_execution_authority_v1 &=
                        resp.supports_execution_authority_v1;
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
                    combined.supports_execution_authority_v1 = false;
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

    /// Diagnostic snapshots grant no routing authority or readiness capability.
    pub async fn numerical_process_inventory(&self) -> Vec<NumericalProcessObservation> {
        let mut observations = join_all(self.children.iter().map(|child| async move {
            match child.numerical_ipc.numerical_profile_snapshot().await {
                Ok(snapshot) if valid_numerical_snapshot(&snapshot) => {
                    NumericalProcessObservation {
                        child_index: child.index,
                        status: if snapshot.complete
                            && snapshot.runtime_instance_id.is_some()
                            && snapshot
                                .profiles
                                .iter()
                                .all(|profile| profile.model_contract_sha256.is_some())
                        {
                            NumericalSnapshotStatus::Observed
                        } else {
                            NumericalSnapshotStatus::Incomplete
                        },
                        snapshot: Some(snapshot),
                    }
                }
                Ok(_) => NumericalProcessObservation {
                    child_index: child.index,
                    status: NumericalSnapshotStatus::Invalid,
                    snapshot: None,
                },
                Err(_) => NumericalProcessObservation {
                    child_index: child.index,
                    status: NumericalSnapshotStatus::Unavailable,
                    snapshot: None,
                },
            }
        }))
        .await;
        let mut counts = HashMap::new();
        for observation in &observations {
            if let Some(instance) = observation
                .snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.runtime_instance_id.as_ref())
            {
                *counts.entry(instance.clone()).or_insert(0usize) += 1;
            }
        }
        for observation in &mut observations {
            if observation
                .snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.runtime_instance_id.as_ref())
                .is_some_and(|instance| counts[instance] > 1)
            {
                observation.status = NumericalSnapshotStatus::Invalid;
            }
        }
        observations
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
        self.process_generate_with_authority(req, on_event, false)
            .await
    }

    pub async fn process_generate_with_authority<F, Fut>(
        &self,
        req: ProcessGenerateRequest,
        on_event: F,
        require_authority: bool,
    ) -> Result<(), IpcError>
    where
        F: FnMut(GenerateEvent) -> Fut,
        Fut: std::future::Future<Output = Result<(), IpcError>>,
    {
        self.ensure_not_config_quarantined()?;
        let model_id = req.model_id.clone();
        let child = self.child_for_model(&model_id);
        let _inflight_guard = ChildInflightGuard::enter(Arc::clone(&child));
        let result = if require_authority {
            child
                .ipc
                .process_generate_with_execution_authority_v1(req, on_event)
                .await
        } else {
            child.ipc.process_generate(req, on_event).await
        };
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
    merge_unsupported_models(&mut existing.unsupported_models, resp.unsupported_models);
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
    merge_unsupported_models(&mut existing.unsupported_models, resp.unsupported_models);

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

/// A model is unsupported on the pod when any child cannot serve it.
fn merge_unsupported_models(existing: &mut Vec<String>, next: Vec<String>) {
    existing.extend(next);
    existing.sort();
    existing.dedup();
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
        self.run_batch_with_budget(req, None).await
    }

    async fn run_batch_with_budget(
        &self,
        req: RunBatchRequest,
        budget: Option<Duration>,
    ) -> Result<BatchOutcome, BackendError> {
        let model_id = req.model_id.clone();
        let child = self.child_for_model(&model_id);
        self.run_child_batch(model_id, child, |ipc| async move {
            ipc.run_batch_with_budget(req, budget).await
        })
        .await
        .map_err(map_ipc_error)
    }

    async fn run_batch_with_execution_authority_v1(
        &self,
        req: RunBatchRequest,
        budget: Option<Duration>,
    ) -> Result<BatchOutcome, BackendError> {
        let model_id = req.model_id.clone();
        let child = self.child_for_model(&model_id);
        self.run_child_batch(model_id, child, |ipc| async move {
            ipc.run_batch_with_execution_authority_v1(req, budget).await
        })
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

    #[derive(Default)]
    struct CapabilityProbeGate {
        enabled: AtomicBool,
        entered: tokio::sync::Notify,
        release: tokio::sync::Notify,
    }

    async fn spawn_capability_worker(
        path: PathBuf,
        supports: Arc<AtomicBool>,
        gate: Option<Arc<CapabilityProbeGate>>,
    ) -> tokio::task::JoinHandle<()> {
        let listener = UnixListener::bind(path).unwrap();
        tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let supports = Arc::clone(&supports);
                let gate = gate.clone();
                tokio::spawn(async move {
                    loop {
                        let mut length = [0_u8; 4];
                        if socket.read_exact(&mut length).await.is_err() {
                            return;
                        }
                        let mut bytes = vec![0; u32::from_be_bytes(length) as usize];
                        if socket.read_exact(&mut bytes).await.is_err() {
                            return;
                        }
                        let request: serde_json::Value = rmp_serde::from_slice(&bytes).unwrap();
                        let method = request["method"].as_str().unwrap();
                        if method == "WorkerCapabilities" {
                            if let Some(gate) = &gate {
                                if gate.enabled.load(Ordering::Acquire) {
                                    gate.entered.notify_one();
                                    gate.release.notified().await;
                                }
                            }
                        }
                        let capable = supports.load(Ordering::Acquire);
                        let body = match method {
                            "Ping" => Some(
                                serde_json::json!({"timestamp_ms": 0.0, "worker_id": "child", "ready": true}),
                            ),
                            "WorkerCapabilities" if capable => {
                                Some(serde_json::json!({"supports_execution_authority_v1": true}))
                            }
                            "WorkerCapabilities" => Some(serde_json::json!({})),
                            "RunBatchWithExecutionAuthorityV1" if capable => {
                                Some(serde_json::json!({"outcomes": []}))
                            }
                            _ => None,
                        };
                        let response = rmp_serde::to_vec_named(&serde_json::json!({
                            "version": crate::ipc_types::IPC_VERSION,
                            "request_id": request["request_id"],
                            "ok": body.is_some(),
                            "error": if body.is_none() { Some("unknown method") } else { None },
                            "body": body,
                        }))
                        .unwrap();
                        if socket
                            .write_all(&(response.len() as u32).to_be_bytes())
                            .await
                            .is_err()
                            || socket.write_all(&response).await.is_err()
                        {
                            return;
                        }
                    }
                });
            }
        })
    }

    #[tokio::test]
    async fn numerical_inventory_preserves_missing_legacy_and_replaced_processes() {
        async fn spawn_snapshot_worker(
            path: PathBuf,
            snapshot: Arc<Mutex<serde_json::Value>>,
        ) -> tokio::task::JoinHandle<()> {
            let listener = UnixListener::bind(path).unwrap();
            tokio::spawn(async move {
                loop {
                    let Ok((mut socket, _)) = listener.accept().await else {
                        return;
                    };
                    let snapshot = Arc::clone(&snapshot);
                    tokio::spawn(async move {
                        loop {
                            let mut length = [0_u8; 4];
                            if socket.read_exact(&mut length).await.is_err() {
                                return;
                            }
                            let mut bytes = vec![0; u32::from_be_bytes(length) as usize];
                            if socket.read_exact(&mut bytes).await.is_err() {
                                return;
                            }
                            let request: serde_json::Value = rmp_serde::from_slice(&bytes).unwrap();
                            assert_eq!(request["method"], "NumericalProfileSnapshot");
                            let body = snapshot.lock().unwrap().clone();
                            let response = rmp_serde::to_vec_named(&serde_json::json!({"version": crate::ipc_types::IPC_VERSION, "request_id": request["request_id"], "ok": true, "body": body})).unwrap();
                            if socket
                                .write_all(&(response.len() as u32).to_be_bytes())
                                .await
                                .is_err()
                                || socket.write_all(&response).await.is_err()
                            {
                                return;
                            }
                        }
                    });
                }
            })
        }
        let dir = tempfile::Builder::new()
            .prefix("sie-num-")
            .tempdir_in("/tmp")
            .unwrap();
        let first_path = dir.path().join("first.sock");
        let legacy_path = dir.path().join("legacy.sock");
        let first = Arc::new(Mutex::new(
            serde_json::json!({"runtime_instance_id": "a".repeat(64), "complete": true, "profiles": [{"model_id": "m", "local_identity": format!("v1:sha256:{}", "c".repeat(64)), "model_contract_sha256": "d".repeat(64)}]}),
        ));
        let legacy = Arc::new(Mutex::new(serde_json::json!({})));
        let first_server = spawn_snapshot_worker(first_path.clone(), Arc::clone(&first)).await;
        let legacy_server = spawn_snapshot_worker(legacy_path.clone(), Arc::clone(&legacy)).await;
        let pool = pool_with_paths(&[first_path, legacy_path, dir.path().join("missing.sock")]);
        let initial_ready = pool.ready_child_count();
        let observations = pool.numerical_process_inventory().await;
        assert_eq!(
            observations
                .iter()
                .map(|observation| observation.child_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            observations
                .iter()
                .map(|observation| observation.status)
                .collect::<Vec<_>>(),
            vec![
                NumericalSnapshotStatus::Observed,
                NumericalSnapshotStatus::Incomplete,
                NumericalSnapshotStatus::Unavailable
            ]
        );
        assert_eq!(observations[0].snapshot.as_ref().unwrap().profiles.len(), 1);
        assert!(!pool.execution_authority_v1().load(Ordering::Acquire));
        assert_eq!(pool.ready_child_count(), initial_ready);
        first.lock().unwrap()["profiles"][0]["model_contract_sha256"] = serde_json::Value::Null;
        let missing_contract = pool.numerical_process_inventory().await;
        assert_eq!(
            missing_contract[0].status,
            NumericalSnapshotStatus::Incomplete
        );
        first.lock().unwrap()["profiles"][0]["model_contract_sha256"] =
            serde_json::json!("d".repeat(64));
        first.lock().unwrap()["profiles"][0]["model_id"] = serde_json::json!("é".repeat(513));
        let oversized = pool.numerical_process_inventory().await;
        assert_eq!(oversized[0].status, NumericalSnapshotStatus::Invalid);
        assert!(oversized[0].snapshot.is_none());
        first.lock().unwrap()["profiles"][0]["model_id"] = serde_json::json!("m");
        let profile = first.lock().unwrap()["profiles"][0].clone();
        first.lock().unwrap()["profiles"] = serde_json::json!([profile.clone(), profile.clone()]);
        let duplicate_model = pool.numerical_process_inventory().await;
        assert_eq!(duplicate_model[0].status, NumericalSnapshotStatus::Invalid);
        assert!(duplicate_model[0].snapshot.is_none());
        first.lock().unwrap()["profiles"] = serde_json::json!([profile]);
        first.lock().unwrap()["runtime_instance_id"] = serde_json::json!("b".repeat(64));
        let replaced = pool.numerical_process_inventory().await;
        assert_eq!(
            replaced[0]
                .snapshot
                .as_ref()
                .unwrap()
                .runtime_instance_id
                .as_deref(),
            Some("b".repeat(64).as_str())
        );
        *legacy.lock().unwrap() = first.lock().unwrap().clone();
        let duplicated = pool.numerical_process_inventory().await;
        assert_eq!(duplicated[0].status, NumericalSnapshotStatus::Invalid);
        assert_eq!(duplicated[1].status, NumericalSnapshotStatus::Invalid);
        first.lock().unwrap()["profiles"][0]["local_identity"] =
            serde_json::json!("private-invalid-identity");
        let invalid = pool.numerical_process_inventory().await;
        assert_eq!(invalid[0].status, NumericalSnapshotStatus::Invalid);
        assert!(invalid[0].snapshot.is_none());
        assert!(!serde_json::to_string(&invalid)
            .unwrap()
            .contains("private-invalid"));
        let unavailable = pool_with_paths(&[dir.path().join("none.sock")]);
        let unavailable = unavailable.numerical_process_inventory().await;
        assert_eq!(unavailable.len(), 1);
        assert_eq!(unavailable[0].status, NumericalSnapshotStatus::Unavailable);
        first_server.abort();
        legacy_server.abort();
    }

    #[tokio::test]
    async fn every_child_must_support_authority_and_old_replacement_cannot_downgrade() {
        let dir = tempfile::Builder::new()
            .prefix("sie-a-")
            .tempdir_in("/tmp")
            .unwrap();
        let new_path = dir.path().join("new.sock");
        let old_path = dir.path().join("old.sock");
        let old_supports = Arc::new(AtomicBool::new(false));
        let new_server =
            spawn_capability_worker(new_path.clone(), Arc::new(AtomicBool::new(true)), None).await;
        let old_server =
            spawn_capability_worker(old_path.clone(), Arc::clone(&old_supports), None).await;
        let pool = pool_with_paths(&[new_path.clone(), old_path.clone()]);
        pool.ping_all(0.0).await;
        assert_eq!(pool.ready_child_count(), 2);
        assert!(!pool.execution_authority_v1().load(Ordering::Acquire));
        assert!(
            !pool
                .worker_capabilities()
                .await
                .unwrap()
                .supports_execution_authority_v1
        );
        old_supports.store(true, Ordering::Release);
        pool.ping_all(0.0).await;
        assert!(pool.execution_authority_v1().load(Ordering::Acquire));

        // Health is still positive when the placed child becomes an older
        // backend. The per-call discriminator must independently refuse it.
        pool.placements.lock().unwrap().insert("m".into(), 1);
        old_supports.store(false, Ordering::Release);
        let result = pool
            .run_batch_with_execution_authority_v1(
                RunBatchRequest {
                    model_id: "m".into(),
                    batch_id: 1,
                    lora_key: String::new(),
                    total_cost: 1,
                    items: Vec::new(),
                    accepts_batched_f16_multivectors: true,
                },
                None,
            )
            .await;
        assert!(matches!(result, Err(BackendError::Transient(_))));
        pool.ping_all(0.0).await;
        assert!(!pool.execution_authority_v1().load(Ordering::Acquire));

        let unavailable = pool_with_paths(&[new_path, dir.path().join("missing.sock")]);
        unavailable.ping_all(0.0).await;
        assert!(!unavailable.execution_authority_v1().load(Ordering::Acquire));
        assert!(
            !unavailable
                .worker_capabilities()
                .await
                .unwrap()
                .supports_execution_authority_v1
        );
        new_server.abort();
        old_server.abort();
    }

    #[tokio::test]
    async fn successful_authority_heartbeat_does_not_temporarily_revoke_support() {
        let dir = tempfile::Builder::new()
            .prefix("ipc-cap-refresh-")
            .tempdir_in("/tmp")
            .unwrap();
        let path = dir.path().join("child.sock");
        let gate = Arc::new(CapabilityProbeGate::default());
        let server = spawn_capability_worker(
            path.clone(),
            Arc::new(AtomicBool::new(true)),
            Some(gate.clone()),
        )
        .await;
        let pool = pool_with_paths(&[path]);
        pool.ping_all(0.0).await;
        assert!(pool.execution_authority_v1().load(Ordering::Acquire));
        gate.enabled.store(true, Ordering::Release);
        let task_pool = pool.clone();
        let refresh = tokio::spawn(async move { task_pool.ping_all(0.0).await });
        tokio::time::timeout(Duration::from_secs(1), gate.entered.notified())
            .await
            .unwrap();
        assert!(pool.execution_authority_v1().load(Ordering::Acquire));
        gate.release.notify_one();
        refresh.await.unwrap();
        assert!(pool.execution_authority_v1().load(Ordering::Acquire));
        server.abort();
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
    fn config_merges_union_unsupported_models_across_children() {
        let mut applied = None;
        for unsupported in [
            vec!["b".to_string(), "a".to_string()],
            vec!["a".to_string()],
        ] {
            merge_apply_model_config_response(
                &mut applied,
                ApplyModelConfigResponse {
                    unsupported_models: unsupported,
                    applied: true,
                    bundle_config_hash: "h1".into(),
                    config_version: 1,
                },
            );
        }
        let applied = applied.expect("combined apply");
        assert!(applied.applied);
        assert_eq!(applied.unsupported_models, ["a", "b"]);

        let mut replaced = None;
        for unsupported in [Vec::new(), vec!["c".to_string()]] {
            merge_replace_model_configs_response(
                &mut replaced,
                ReplaceModelConfigsResponse {
                    unsupported_models: unsupported,
                    applied: true,
                    bundle_config_hash: "h1".into(),
                    config_version: 1,
                    applied_models: vec!["c".into()],
                    applied_profiles: vec!["default".into()],
                },
            );
        }
        let replaced = replaced.expect("combined replace");
        assert!(replaced.applied);
        assert_eq!(replaced.unsupported_models, ["c"]);
    }

    #[test]
    fn apply_config_merge_preserves_any_child_rejection() {
        let mut combined = None;

        merge_apply_model_config_response(
            &mut combined,
            ApplyModelConfigResponse {
                unsupported_models: Vec::new(),
                applied: false,
                bundle_config_hash: "h1".into(),
                config_version: 1,
            },
        );
        merge_apply_model_config_response(
            &mut combined,
            ApplyModelConfigResponse {
                unsupported_models: Vec::new(),
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
                unsupported_models: Vec::new(),
                applied: true,
                bundle_config_hash: "h1".into(),
                config_version: 1,
            },
        );
        merge_apply_model_config_response(
            &mut combined,
            ApplyModelConfigResponse {
                unsupported_models: Vec::new(),
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
                unsupported_models: Vec::new(),
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
                unsupported_models: Vec::new(),
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
                    unsupported_models: Vec::new(),
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
