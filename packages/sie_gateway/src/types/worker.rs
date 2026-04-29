use serde::{Deserialize, Serialize};
use std::time::Instant;

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
    pub machine_profile: String,
    pub bundle: String,
    pub bundle_config_hash: String,
    pub models: Vec<String>,
    pub queue_depth: i32,
    pub memory_used_bytes: i64,
    pub memory_total_bytes: i64,
    pub last_heartbeat: Instant,
    pub pool_name: String,
}

impl WorkerState {
    pub fn healthy(&self) -> bool {
        self.health == WorkerHealth::Healthy
    }

    #[allow(dead_code)]
    pub fn memory_utilization(&self) -> f64 {
        if self.memory_total_bytes <= 0 {
            return 0.0;
        }
        self.memory_used_bytes as f64 / self.memory_total_bytes as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub timestamp: f64,
    pub worker_count: i32,
    pub gpu_count: i32,
    pub models_loaded: i32,
    pub total_qps: f64,
    pub workers: Vec<WorkerInfo>,
    pub models: Vec<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub name: String,
    pub url: String,
    pub gpu: String,
    pub gpu_count: i32,
    pub loaded_models: Vec<String>,
    pub queue_depth: i32,
    pub memory_used_bytes: i64,
    pub memory_total_bytes: i64,
    pub healthy: bool,
    pub bundle: String,
    pub bundle_config_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisioningResponse {
    pub status: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub gpu: String,
    pub estimated_wait_s: i32,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub message: String,
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

#[derive(Debug, Clone, Deserialize)]
pub struct WorkerStatusMessage {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub ready: bool,
    #[serde(default)]
    pub gpu_count: i32,
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
    /// Compact top-level memory used (fallback when gpus array is empty)
    #[serde(default)]
    pub memory_used_bytes: Option<i64>,
    /// Compact top-level memory total (fallback when gpus array is empty)
    #[serde(default)]
    pub memory_total_bytes: Option<i64>,
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
            machine_profile: "l4".into(),
            bundle: "default".into(),
            bundle_config_hash: String::new(),
            models: vec![],
            queue_depth: 0,
            memory_used_bytes: mem_used,
            memory_total_bytes: mem_total,
            last_heartbeat: Instant::now(),
            pool_name: String::new(),
        }
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
        assert!(msg.loaded_models.is_empty());
        assert!(msg.models.is_empty());
        assert!(msg.gpus.is_empty());
    }

    #[test]
    fn test_worker_status_message_full() {
        let json = r#"{
            "name": "worker-1",
            "ready": true,
            "gpu_count": 2,
            "machine_profile": "a100",
            "bundle": "premium",
            "bundle_config_hash": "abc",
            "loaded_models": ["model-a", "model-b"],
            "models": [{"queue_depth": 3}],
            "gpus": [{"memory_used_bytes": 1000, "memory_total_bytes": 4000}]
        }"#;
        let msg: WorkerStatusMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.name, "worker-1");
        assert_eq!(msg.gpu_count, 2);
        assert_eq!(msg.loaded_models.len(), 2);
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
