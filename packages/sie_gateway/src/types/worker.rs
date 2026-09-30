use serde::{Deserialize, Serialize};
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
