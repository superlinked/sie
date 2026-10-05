use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;
use utoipa::ToSchema;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerHealth {
    Unknown,
    Healthy,
    Unhealthy,
}

impl std::fmt::Display for WorkerHealth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerHealth::Healthy => write!(f, "healthy"),
            WorkerHealth::Unhealthy => write!(f, "unhealthy"),
            WorkerHealth::Unknown => write!(f, "unknown"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkerState {
    pub url: String,
    pub name: String,
    pub health: WorkerHealth,
    pub gpu_count: i32,
    pub ready_gpu_slots: i32,
    pub machine_profile: String,
    pub bundle: String,
    pub bundle_config_hash: String,
    /// Positive support for the versioned queue and backend execution fence.
    pub supports_execution_authority_v1: bool,
    /// Every backend child re-verifies a numerical admission before execution.
    pub supports_numerical_admission_v1: bool,
    /// Diagnostic observations only; never sufficient for numerical admission.
    pub numerical_process_inventory: Option<Arc<NumericalProcessInventory>>,
    pub models: Vec<String>,
    pub queue_depth: i32,
    pub pending_cost: i64,
    pub inflight_batches: i32,
    pub memory_used_bytes: i64,
    pub memory_total_bytes: i64,
    pub last_heartbeat: Instant,
    pub pool_name: String,
    /// Saturation flag, orthogonal to `health`. A saturated
    /// worker is *healthy* (live, heartbeating) but its admission
    /// capacity is full; the gateway must skip it for HRW direct
    /// dispatch and fall back to the pool. Hysteresis lives on the
    /// worker side (90/70 thresholds); the gateway just consumes the
    /// bool.
    pub saturated: bool,
    /// Routable model ids covered by `bundle_config_hash` that this worker
    /// reported it cannot serve. Empty for workers that predate the field.
    /// Shared so snapshot rebuilds do not copy it.
    pub unsupported_models: Arc<[String]>,
    /// The worker reported more than [`MAX_UNSUPPORTED_MODELS`] ids. It is
    /// then treated as unable to serve any model, because a dropped id would
    /// otherwise read as supported, so its whole lane is routed no model.
    pub unsupported_overflow: bool,
}

/// Upper bound on the `unsupported_models` one heartbeat may carry.
pub const MAX_UNSUPPORTED_MODELS: usize = 1024;

impl WorkerState {
    pub fn healthy(&self) -> bool {
        self.health == WorkerHealth::Healthy
    }

    /// Whether this worker can serve `model` under its advertised config.
    /// A worker that sends no list serves every model its hash covers.
    pub fn supports_model(&self, model: &str) -> bool {
        !self.unsupported_overflow
            && !self
                .unsupported_models
                .iter()
                .any(|unsupported| unsupported.eq_ignore_ascii_case(model))
    }

    /// Eligible for dispatch: healthy, with at least one ready slot, and not saturated.
    /// Saturated workers stay in the registry (so the pool fallback
    /// still drains to them via the bundle index) but are excluded
    /// from the per-`(model, pool)` HRW ring.
    pub fn eligible_for_dispatch(&self) -> bool {
        self.healthy() && self.ready_gpu_slots > 0 && !self.saturated
    }

    #[allow(dead_code)]
    pub fn memory_utilization(&self) -> f64 {
        if self.memory_total_bytes <= 0 {
            return 0.0;
        }
        self.memory_used_bytes as f64 / self.memory_total_bytes as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ClusterStatus {
    pub timestamp: f64,
    pub worker_count: i32,
    pub gpu_count: i32,
    pub models_loaded: i32,
    pub total_qps: f64,
    pub workers: Vec<WorkerInfo>,
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct WorkerInfo {
    pub name: String,
    pub url: String,
    pub gpu: String,
    pub gpu_count: i32,
    pub ready_gpu_slots: i32,
    pub loaded_models: Vec<String>,
    pub queue_depth: i32,
    pub pending_cost: i64,
    pub inflight_batches: i32,
    pub memory_used_bytes: i64,
    pub memory_total_bytes: i64,
    pub healthy: bool,
    pub bundle: String,
    pub bundle_config_hash: String,
    /// Models covered by `bundle_config_hash` that this worker cannot serve,
    /// for example during a rollout that adds adapters to its bundle. The
    /// gateway does not route these models to it. Omitted when empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unsupported_models: Vec<String>,
    /// The worker reported more unsupported models than the gateway accepts
    /// (1024), so no model is routed to its lane (pool, machine profile,
    /// bundle) while it overflows. Omitted when false.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub unsupported_models_overflow: bool,
    /// Most recent process diagnostics, including unavailable children.
    /// Health and observation time must be checked independently.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numerical_process_inventory: Option<NumericalProcessInventory>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelInfo {
    pub name: String,
    pub state: String,
    pub worker_count: i32,
    pub gpu_types: Vec<String>,
    pub total_queue_depth: i32,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineProfile {
    pub name: String,
    pub gpu_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub machine_type: String,
    #[serde(default)]
    pub spot: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub event: String,
    pub method: String,
    pub endpoint: String,
    pub status: u16,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub token_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub pool: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub gpu: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub worker: String,
    /// Wall-clock latency of the request in whole milliseconds.
    ///
    /// Emitted as an integer rather than `f64`: (a) the audit sink is
    /// a structured log, not a histogram — sub-millisecond precision
    /// has never been useful here, (b) `tracing`'s JSON formatter
    /// treats `u64` as an integer field which is ~3x cheaper to
    /// format than `f64`, and (c) downstream log parsers don't have
    /// to handle locale-dependent decimal separators.
    #[serde(default)]
    pub latency_ms: u64,
    #[serde(default)]
    pub body_bytes: i64,
}

// `Default` lets a synthetic census entry be built as a struct literal with
// `..Default::default()` instead of round-tripping through `serde_json`. Every
// field is already `#[serde(default)]`, so the derived value is exactly what a
// minimal heartbeat deserializes to — the two stay in step by construction.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct WorkerStatusMessage {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub ready: bool,
    #[serde(default)]
    pub gpu_count: i32,
    /// Total GPU slots visible to the worker pod. Defaults to
    /// `gpu_count` when absent.
    #[serde(default)]
    pub total_gpu_slots: Option<i32>,
    /// Slots ready to receive work. Defaults to all slots when
    /// `ready=true`, otherwise zero.
    #[serde(default)]
    pub ready_gpu_slots: Option<i32>,
    #[serde(default)]
    pub machine_profile: String,
    #[serde(default)]
    pub pool_name: String,
    #[serde(default)]
    pub bundle: String,
    #[serde(default)]
    pub bundle_config_hash: String,
    #[serde(default)]
    pub supports_execution_authority_v1: bool,
    #[serde(default)]
    pub supports_numerical_admission_v1: bool,
    #[serde(default)]
    pub loaded_models: Vec<String>,
    #[serde(default)]
    pub models: Vec<ModelStatus>,
    #[serde(default)]
    pub gpus: Vec<GpuStatus>,
    /// Compact top-level queue depth (fallback when models array is empty)
    #[serde(default)]
    pub queue_depth: Option<i32>,
    /// Compact top-level pending scheduler cost.
    #[serde(default)]
    pub pending_cost: Option<i64>,
    /// Compact top-level in-flight batch count.
    #[serde(default)]
    pub inflight_batches: Option<i32>,
    /// Compact top-level memory used (fallback when gpus array is empty)
    #[serde(default)]
    pub memory_used_bytes: Option<i64>,
    /// Compact top-level memory total (fallback when gpus array is empty)
    #[serde(default)]
    pub memory_total_bytes: Option<i64>,
    /// Saturation signal. `true` ⇒ the worker is at or above
    /// its admission high-water mark and the gateway should exclude it
    /// from the HRW ring until it drops below the low-water mark.
    /// Defaults to `false` for backward compatibility with workers
    /// running pre-routing builds.
    #[serde(default)]
    pub saturated: bool,
    /// Graceful worker shutdown tombstone. When true, the gateway should
    /// remove this worker from the live registry instead of marking it
    /// unhealthy. Defaults to false for older workers.
    #[serde(default)]
    pub terminated: bool,
    /// Routable model ids (`model` or `model:profile`) covered by
    /// `bundle_config_hash` that the worker cannot serve. Absent from
    /// workers that predate the field, which serve every model their hash
    /// covers.
    #[serde(default)]
    pub unsupported_models: Vec<String>,
    /// Optional bounded diagnostics. Invalid metadata must not drop health.
    #[serde(default, deserialize_with = "deserialize_numerical_inventory")]
    pub numerical_process_inventory: Option<NumericalProcessInventory>,
}

/// Wire budget shared with the sidecar publisher. Never truncate a fleet.
pub const MAX_NUMERICAL_INVENTORY_BYTES: usize = 64 * 1024;
pub const MAX_NUMERICAL_CHILDREN: usize = 256;

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct NumericalProcessInventory {
    pub observed_at_unix_ms: u64,
    pub children: Vec<NumericalProcessObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct NumericalProcessObservation {
    pub child_index: usize,
    pub status: NumericalSnapshotStatus,
    pub snapshot: Option<NumericalProfileSnapshot>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum NumericalSnapshotStatus {
    Observed,
    Incomplete,
    Unavailable,
    Invalid,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct NumericalProfileSnapshot {
    pub runtime_instance_id: Option<String>,
    pub complete: bool,
    pub profiles: Vec<NumericalProfileObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct NumericalProfileObservation {
    pub model_id: String,
    pub model_contract_sha256: Option<String>,
    pub local_identity: Option<String>,
    /// The contract of the model's remote profile on this process's upstream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_contract_sha256: Option<String>,
    /// The serving code that runs the remote profile in this process.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_execution_sha256: Option<String>,
    /// The local execution identities current evidence covers for the remote profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub admission: Option<NumericalAdmissionObservation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct NumericalAdmissionObservation {
    pub sha256: String,
    pub kind: String,
    pub local_identities: Vec<String>,
    pub model_contract_sha256: String,
    pub outputs: Vec<String>,
    pub expires_at_unix_ms: u64,
}

pub const MAX_ADMITTED_IDENTITIES: usize = 8;
const NUMERICAL_OUTPUTS: [&str; 4] = ["dense", "multivector", "score", "sparse"];

impl NumericalAdmissionObservation {
    fn valid(&self) -> bool {
        let mut identities = HashSet::new();
        let mut outputs = HashSet::new();
        sha256_digest(&self.sha256)
            && matches!(self.kind.as_str(), "openai" | "sie")
            && !self.local_identities.is_empty()
            && self.local_identities.len() <= MAX_ADMITTED_IDENTITIES
            && self
                .local_identities
                .iter()
                .all(|identity| local_identity_digest(identity) && identities.insert(identity))
            && sha256_digest(&self.model_contract_sha256)
            && !self.outputs.is_empty()
            && self.outputs.iter().all(|output| {
                NUMERICAL_OUTPUTS.contains(&output.as_str()) && outputs.insert(output)
            })
            && self.expires_at_unix_ms > 0
    }
}

fn sha256_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn local_identity_digest(identity: &str) -> bool {
    ["v1:sha256:", "v2:sha256:"]
        .iter()
        .any(|prefix| identity.strip_prefix(prefix).is_some_and(sha256_digest))
}

struct InventoryBudget(usize);

impl Write for InventoryBudget {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > MAX_NUMERICAL_INVENTORY_BYTES.saturating_sub(self.0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "numerical inventory exceeds wire budget",
            ));
        }
        self.0 += bytes.len();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn fits_inventory_budget(value: &impl Serialize) -> bool {
    serde_json::to_writer(&mut InventoryBudget(0), value).is_ok()
}

impl NumericalProcessInventory {
    pub fn valid(&self) -> bool {
        if self.observed_at_unix_ms == 0
            || self.children.is_empty()
            || self.children.len() > MAX_NUMERICAL_CHILDREN
            || !fits_inventory_budget(self)
        {
            return false;
        }
        let mut instances = HashMap::new();
        for (index, child) in self.children.iter().enumerate() {
            if child.child_index != index {
                return false;
            }
            let Some(snapshot) = &child.snapshot else {
                if !matches!(
                    child.status,
                    NumericalSnapshotStatus::Unavailable | NumericalSnapshotStatus::Invalid
                ) {
                    return false;
                }
                continue;
            };
            if child.status == NumericalSnapshotStatus::Unavailable
                || snapshot
                    .runtime_instance_id
                    .as_deref()
                    .is_some_and(|id| !sha256_digest(id))
                || snapshot.profiles.len() > 1024
            {
                return false;
            }
            let mut models = HashSet::new();
            for profile in &snapshot.profiles {
                if profile.model_id.is_empty()
                    || profile.model_id.len() > 1024
                    || !models.insert(&profile.model_id)
                    || profile
                        .model_contract_sha256
                        .as_deref()
                        .is_some_and(|id| !sha256_digest(id))
                    || profile
                        .local_identity
                        .as_deref()
                        .is_some_and(|id| !local_identity_digest(id))
                    || profile
                        .remote_contract_sha256
                        .as_deref()
                        .is_some_and(|id| !sha256_digest(id))
                    || profile
                        .remote_execution_sha256
                        .as_deref()
                        .is_some_and(|id| !sha256_digest(id))
                    || profile
                        .admission
                        .as_ref()
                        .is_some_and(|admission| !admission.valid())
                {
                    return false;
                }
            }
            let complete = snapshot.complete
                && snapshot.runtime_instance_id.is_some()
                && snapshot
                    .profiles
                    .iter()
                    .all(|profile| profile.model_contract_sha256.is_some());
            if (child.status == NumericalSnapshotStatus::Observed && !complete)
                || (child.status == NumericalSnapshotStatus::Incomplete && complete)
            {
                return false;
            }
            if let Some(instance) = &snapshot.runtime_instance_id {
                instances
                    .entry(instance)
                    .or_insert_with(Vec::new)
                    .push(child.status);
            }
        }
        // The pool marks every child sharing a process ID invalid. Preserve
        // those diagnostics, but never accept duplicate positive observations.
        instances.values().all(|statuses| {
            statuses.len() == 1
                || statuses
                    .iter()
                    .all(|status| *status == NumericalSnapshotStatus::Invalid)
        })
    }
}

fn deserialize_numerical_inventory<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<NumericalProcessInventory>, D::Error> {
    // MessagePack may carry non-JSON values (for example binary data). The
    // ignored alternative consumes those fully so optional diagnostics cannot
    // invalidate an otherwise usable health message.
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum InventoryWire {
        Json(serde_json::Value),
        Ignored(serde::de::IgnoredAny),
    }
    let InventoryWire::Json(value) = InventoryWire::deserialize(deserializer)? else {
        return Ok(None);
    };
    if !fits_inventory_budget(&value) {
        return Ok(None);
    }
    Ok(serde_json::from_value::<NumericalProcessInventory>(value)
        .ok()
        .filter(NumericalProcessInventory::valid))
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelStatus {
    #[serde(default)]
    pub queue_depth: i32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GpuStatus {
    #[serde(default)]
    pub memory_used_bytes: i64,
    #[serde(default)]
    pub memory_total_bytes: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inventory() -> serde_json::Value {
        serde_json::json!({
            "observed_at_unix_ms": 1,
            "children": [
                {"child_index": 0, "status": "observed", "snapshot": {
                    "runtime_instance_id": "a".repeat(64), "complete": true,
                    "profiles": [{"model_id": "model:default", "model_contract_sha256": "b".repeat(64), "local_identity": null}]
                }},
                {"child_index": 1, "status": "unavailable", "snapshot": null}
            ]
        })
    }

    #[test]
    fn numerical_diagnostics_preserve_missing_children_without_authority() {
        let message: WorkerStatusMessage = serde_json::from_value(serde_json::json!({
            "ready": true, "numerical_process_inventory": inventory()
        }))
        .unwrap();
        assert!(message.ready);
        assert!(!message.supports_execution_authority_v1);
        let inventory = message.numerical_process_inventory.unwrap();
        assert_eq!(inventory.children.len(), 2);
        assert_eq!(
            inventory.children[1].status,
            NumericalSnapshotStatus::Unavailable
        );
        assert!(inventory.children[0].snapshot.as_ref().unwrap().profiles[0]
            .local_identity
            .is_none());
    }

    #[test]
    fn invalid_numerical_metadata_never_discards_worker_health() {
        let mut cases = vec![
            serde_json::json!(null),
            serde_json::json!("bad"),
            inventory(),
        ];
        cases[2]["children"][0]["snapshot"]["runtime_instance_id"] = serde_json::json!("bad");
        let mut unknown = inventory();
        unknown["children"][0]["snapshot"]["caller_input"] =
            serde_json::json!("must not be retained");
        cases.push(unknown);
        let mut sparse = inventory();
        sparse["children"][1]["child_index"] = serde_json::json!(2);
        cases.push(sparse);
        let mut false_complete = inventory();
        false_complete["children"][0]["snapshot"]["complete"] = serde_json::json!(false);
        cases.push(false_complete);
        let mut too_large = inventory();
        too_large["padding"] = serde_json::json!("x".repeat(MAX_NUMERICAL_INVENTORY_BYTES));
        cases.push(too_large);
        let mut too_many = inventory();
        too_many["children"] = serde_json::Value::Array(
            (0..=MAX_NUMERICAL_CHILDREN)
                .map(|index| {
                    serde_json::json!({
                        "child_index": index, "status": "unavailable", "snapshot": null
                    })
                })
                .collect(),
        );
        cases.push(too_many);
        for inventory in cases {
            let message: WorkerStatusMessage = serde_json::from_value(serde_json::json!({
                "ready": true, "bundle_config_hash": "hash", "numerical_process_inventory": inventory
            })).unwrap();
            assert!(message.ready);
            assert_eq!(message.bundle_config_hash, "hash");
            assert!(message.numerical_process_inventory.is_none());
        }
    }

    #[test]
    fn binary_messagepack_diagnostics_do_not_discard_following_health_fields() {
        let mut bytes = rmp_serde::to_vec_named(&serde_json::json!({
            "numerical_process_inventory": null, "ready": true
        }))
        .unwrap();
        let nil = bytes.iter().position(|byte| *byte == 0xc0).unwrap();
        bytes.splice(nil..=nil, [0xc4, 3, 0, 1, 2]);
        let message: WorkerStatusMessage = rmp_serde::from_slice(&bytes).unwrap();
        assert!(message.ready);
        assert!(message.numerical_process_inventory.is_none());
    }

    #[test]
    fn duplicate_process_ids_are_only_retained_as_invalid_diagnostics() {
        let mut value = inventory();
        value["children"][1] = value["children"][0].clone();
        value["children"][1]["child_index"] = serde_json::json!(1);
        let message: WorkerStatusMessage = serde_json::from_value(serde_json::json!({
            "numerical_process_inventory": value.clone()
        }))
        .unwrap();
        assert!(message.numerical_process_inventory.is_none());
        for child in value["children"].as_array_mut().unwrap() {
            child["status"] = serde_json::json!("invalid");
        }
        let message: WorkerStatusMessage = serde_json::from_value(serde_json::json!({
            "numerical_process_inventory": value
        }))
        .unwrap();
        assert_eq!(
            message.numerical_process_inventory.unwrap().children.len(),
            2
        );
    }

    #[test]
    fn local_identities_of_both_versions_are_accepted_and_others_refused() {
        for (identity, accepted) in [
            (format!("v1:sha256:{}", "c".repeat(64)), true),
            (format!("v2:sha256:{}", "c".repeat(64)), true),
            (format!("v3:sha256:{}", "c".repeat(64)), false),
            (format!("v2:sha256:{}", "C".repeat(64)), false),
            ("v2:sha256:".to_string(), false),
        ] {
            let mut value = inventory();
            value["children"][0]["snapshot"]["profiles"][0]["local_identity"] =
                serde_json::json!(identity);
            let message: WorkerStatusMessage = serde_json::from_value(serde_json::json!({
                "numerical_process_inventory": value
            }))
            .unwrap();
            assert_eq!(
                message.numerical_process_inventory.is_some(),
                accepted,
                "{identity}"
            );
        }
    }

    fn inventory_with_admission(admission: serde_json::Value) -> serde_json::Value {
        let mut value = inventory();
        let profile = &mut value["children"][0]["snapshot"]["profiles"][0];
        profile["remote_contract_sha256"] = serde_json::json!("e".repeat(64));
        profile["remote_execution_sha256"] = serde_json::json!("f".repeat(64));
        profile["admission"] = admission;
        value
    }

    fn admission() -> serde_json::Value {
        serde_json::json!({
            "sha256": "1".repeat(64),
            "kind": "sie",
            "local_identities": [format!("v2:sha256:{}", "2".repeat(64))],
            "model_contract_sha256": "b".repeat(64),
            "outputs": ["score"],
            "expires_at_unix_ms": 1,
        })
    }

    fn parsed(value: serde_json::Value) -> Option<NumericalProcessInventory> {
        serde_json::from_value::<WorkerStatusMessage>(serde_json::json!({
            "numerical_process_inventory": value
        }))
        .unwrap()
        .numerical_process_inventory
    }

    #[test]
    fn remote_admissions_are_carried_and_malformed_ones_drop_the_inventory() {
        let inventory = parsed(inventory_with_admission(admission())).unwrap();
        let profile = &inventory.children[0].snapshot.as_ref().unwrap().profiles[0];
        let carried = profile.admission.as_ref().unwrap();
        assert_eq!(carried.kind, "sie");
        assert_eq!(carried.local_identities.len(), 1);
        assert_eq!(
            profile.remote_execution_sha256.as_deref(),
            Some("f".repeat(64).as_str())
        );
        let identity = format!("v2:sha256:{}", "2".repeat(64));
        for (field, value) in [
            ("sha256", serde_json::json!("x")),
            ("kind", serde_json::json!("vendor")),
            ("local_identities", serde_json::json!([])),
            ("local_identities", serde_json::json!([identity, identity])),
            (
                "local_identities",
                serde_json::json!((0..=MAX_ADMITTED_IDENTITIES)
                    .map(|index| format!("v2:sha256:{index:064x}"))
                    .collect::<Vec<_>>()),
            ),
            ("model_contract_sha256", serde_json::json!("x")),
            ("outputs", serde_json::json!(["tokens"])),
            ("outputs", serde_json::json!(["score", "score"])),
            ("expires_at_unix_ms", serde_json::json!(0)),
            ("unexpected", serde_json::json!(true)),
        ] {
            let mut invalid = admission();
            invalid[field] = value;
            assert!(
                parsed(inventory_with_admission(invalid)).is_none(),
                "{field}"
            );
        }
        let mut contract = inventory_with_admission(serde_json::Value::Null);
        contract["children"][0]["snapshot"]["profiles"][0]["remote_contract_sha256"] =
            serde_json::json!("x");
        assert!(parsed(contract).is_none());
    }

    #[test]
    fn an_observation_without_remote_facts_serializes_as_before() {
        let inventory = parsed(inventory()).unwrap();
        let encoded =
            serde_json::to_value(&inventory.children[0].snapshot.as_ref().unwrap().profiles[0])
                .unwrap();
        let mut keys: Vec<_> = encoded.as_object().unwrap().keys().cloned().collect();
        keys.sort();
        assert_eq!(
            keys,
            ["local_identity", "model_contract_sha256", "model_id"]
        );
    }

    fn make_worker(health: WorkerHealth, mem_used: i64, mem_total: i64) -> WorkerState {
        WorkerState {
            url: "http://w1:8080".into(),
            name: "w1".into(),
            health,
            gpu_count: 1,
            ready_gpu_slots: if health == WorkerHealth::Healthy {
                1
            } else {
                0
            },
            machine_profile: "l4".into(),
            bundle: "default".into(),
            bundle_config_hash: String::new(),
            supports_execution_authority_v1: false,
            supports_numerical_admission_v1: false,
            numerical_process_inventory: None,
            models: vec![],
            queue_depth: 0,
            pending_cost: 0,
            inflight_batches: 0,
            memory_used_bytes: mem_used,
            memory_total_bytes: mem_total,
            last_heartbeat: Instant::now(),
            pool_name: String::new(),
            saturated: false,
            unsupported_models: Arc::from([]),
            unsupported_overflow: false,
        }
    }

    #[test]
    fn test_eligible_for_dispatch_requires_healthy_ready_slot_and_not_saturated() {
        let mut w = make_worker(WorkerHealth::Healthy, 0, 0);
        assert!(w.eligible_for_dispatch());
        w.saturated = true;
        assert!(!w.eligible_for_dispatch());
        w.saturated = false;
        w.ready_gpu_slots = 0;
        assert!(!w.eligible_for_dispatch());
        w.ready_gpu_slots = 1;
        w.health = WorkerHealth::Unhealthy;
        assert!(!w.eligible_for_dispatch());
    }

    #[test]
    fn test_worker_status_message_deserialize_saturated_default_false() {
        let json = r#"{"ready": true}"#;
        let msg: WorkerStatusMessage = serde_json::from_str(json).unwrap();
        assert!(!msg.saturated);
        assert!(!msg.terminated);
    }

    #[test]
    fn test_worker_status_message_deserialize_saturated_true() {
        let json = r#"{"ready": true, "saturated": true}"#;
        let msg: WorkerStatusMessage = serde_json::from_str(json).unwrap();
        assert!(msg.saturated);
    }

    #[test]
    fn test_worker_status_message_matches_the_worker_status_wire_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../../../wire-fixtures/worker_status.json"))
                .expect("worker_status fixture parses");
        let example = &fixture["example"];
        let msg: WorkerStatusMessage =
            serde_json::from_value(example.clone()).expect("example parses");
        assert_eq!(msg.name, example["name"]);
        assert_eq!(msg.ready, example["ready"]);
        assert_eq!(msg.terminated, example["terminated"]);
        assert_eq!(msg.gpu_count, example["gpu_count"]);
        assert_eq!(
            msg.total_gpu_slots,
            example["total_gpu_slots"].as_i64().map(|v| v as i32)
        );
        assert_eq!(
            msg.ready_gpu_slots,
            example["ready_gpu_slots"].as_i64().map(|v| v as i32)
        );
        assert_eq!(msg.machine_profile, example["machine_profile"]);
        assert_eq!(msg.pool_name, example["pool_name"]);
        assert_eq!(msg.bundle, example["bundle"]);
        assert_eq!(msg.bundle_config_hash, example["bundle_config_hash"]);
        assert_eq!(
            serde_json::json!(msg.loaded_models),
            example["loaded_models"]
        );
        assert_eq!(
            msg.queue_depth,
            example["queue_depth"].as_i64().map(|v| v as i32)
        );
        assert_eq!(msg.pending_cost, example["pending_cost"].as_i64());
        assert_eq!(
            msg.inflight_batches,
            example["inflight_batches"].as_i64().map(|v| v as i32)
        );
        assert_eq!(msg.saturated, example["saturated"]);
        assert_eq!(
            serde_json::json!(msg.unsupported_models),
            example["unsupported_models"]
        );
        let fields: Vec<&str> = fixture["fields"]
            .as_array()
            .expect("fields")
            .iter()
            .map(|field| field.as_str().expect("field name"))
            .collect();
        assert_eq!(
            fields.len(),
            example.as_object().expect("example object").len(),
            "the example must carry every published field"
        );

        let mut older = example.clone();
        for field in fixture["omitted_when_empty"]
            .as_object()
            .expect("omitted")
            .keys()
        {
            older.as_object_mut().unwrap().remove(field);
        }
        let older: WorkerStatusMessage =
            serde_json::from_value(older).expect("older worker parses");
        assert!(older.unsupported_models.is_empty());
    }

    #[test]
    fn test_worker_supports_model_ignores_ascii_case_and_defaults_to_all() {
        let mut w = make_worker(WorkerHealth::Healthy, 0, 0);
        assert!(w.supports_model("org/new"));
        w.unsupported_models = vec!["Org/New".to_string(), "org/model:fast".to_string()].into();
        assert!(!w.supports_model("org/new"));
        assert!(!w.supports_model("org/model:fast"));
        assert!(w.supports_model("org/model"));
        w.unsupported_overflow = true;
        assert!(!w.supports_model("org/model"));
    }

    #[test]
    fn test_worker_status_message_deserialize_terminated_true() {
        let json = r#"{"ready": false, "terminated": true}"#;
        let msg: WorkerStatusMessage = serde_json::from_str(json).unwrap();
        assert!(!msg.ready);
        assert!(msg.terminated);
    }

    #[test]
    fn test_worker_healthy() {
        assert!(make_worker(WorkerHealth::Healthy, 0, 0).healthy());
        assert!(!make_worker(WorkerHealth::Unhealthy, 0, 0).healthy());
        assert!(!make_worker(WorkerHealth::Unknown, 0, 0).healthy());
    }

    #[test]
    fn test_memory_utilization() {
        let w = make_worker(WorkerHealth::Healthy, 3000, 4000);
        assert!((w.memory_utilization() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_utilization_zero_total() {
        let w = make_worker(WorkerHealth::Healthy, 0, 0);
        assert!((w.memory_utilization()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_memory_utilization_negative_total() {
        let w = make_worker(WorkerHealth::Healthy, 0, -1);
        assert!((w.memory_utilization()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_worker_health_display() {
        assert_eq!(WorkerHealth::Healthy.to_string(), "healthy");
        assert_eq!(WorkerHealth::Unhealthy.to_string(), "unhealthy");
        assert_eq!(WorkerHealth::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_worker_status_message_deserialize_defaults() {
        let json = r#"{"ready": true}"#;
        let msg: WorkerStatusMessage = serde_json::from_str(json).unwrap();
        assert!(msg.ready);
        assert!(msg.name.is_empty());
        assert_eq!(msg.gpu_count, 0);
        assert_eq!(msg.total_gpu_slots, None);
        assert_eq!(msg.ready_gpu_slots, None);
        assert!(msg.loaded_models.is_empty());
        assert!(msg.models.is_empty());
        assert!(msg.gpus.is_empty());
        assert_eq!(msg.pending_cost, None);
        assert_eq!(msg.inflight_batches, None);
        assert!(!msg.terminated);
    }

    #[test]
    fn test_worker_status_message_full() {
        let json = r#"{
            "name": "worker-1",
            "ready": true,
            "gpu_count": 2,
            "total_gpu_slots": 2,
            "ready_gpu_slots": 1,
            "machine_profile": "a100",
            "bundle": "premium",
            "bundle_config_hash": "abc",
            "loaded_models": ["model-a", "model-b"],
            "pending_cost": 99,
            "inflight_batches": 2,
            "models": [{"queue_depth": 3}],
            "gpus": [{"memory_used_bytes": 1000, "memory_total_bytes": 4000}]
        }"#;
        let msg: WorkerStatusMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.name, "worker-1");
        assert_eq!(msg.gpu_count, 2);
        assert_eq!(msg.total_gpu_slots, Some(2));
        assert_eq!(msg.ready_gpu_slots, Some(1));
        assert_eq!(msg.loaded_models.len(), 2);
        assert_eq!(msg.pending_cost, Some(99));
        assert_eq!(msg.inflight_batches, Some(2));
        assert_eq!(msg.models[0].queue_depth, 3);
        assert_eq!(msg.gpus[0].memory_used_bytes, 1000);
    }

    #[test]
    fn test_worker_status_message_compact_fields() {
        let json = r#"{
            "name": "w1",
            "ready": true,
            "gpu_count": 1,
            "machine_profile": "l4",
            "bundle": "default",
            "queue_depth": 5,
            "memory_used_bytes": 2000,
            "memory_total_bytes": 8000
        }"#;
        let msg: WorkerStatusMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.queue_depth, Some(5));
        assert_eq!(msg.memory_used_bytes, Some(2000));
        assert_eq!(msg.memory_total_bytes, Some(8000));
        assert!(msg.models.is_empty());
        assert!(msg.gpus.is_empty());
    }

    #[test]
    fn test_worker_status_message_compact_fields_absent() {
        let json = r#"{"ready": true}"#;
        let msg: WorkerStatusMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.queue_depth, None);
        assert_eq!(msg.memory_used_bytes, None);
        assert_eq!(msg.memory_total_bytes, None);
    }

    #[test]
    fn test_cluster_status_serialization() {
        let status = ClusterStatus {
            timestamp: 1234.5,
            worker_count: 2,
            gpu_count: 4,
            models_loaded: 3,
            total_qps: 100.0,
            workers: vec![],
            models: vec![],
        };
        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(json["worker_count"], 2);
        assert_eq!(json["gpu_count"], 4);
    }
}
