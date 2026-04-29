use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PoolState {
    Pending,
    Active,
    Expired,
}

impl std::fmt::Display for PoolState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolState::Pending => write!(f, "pending"),
            PoolState::Active => write!(f, "active"),
            PoolState::Expired => write!(f, "expired"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolSpec {
    pub name: String,
    #[serde(default)]
    pub bundle: Option<String>,
    #[serde(default)]
    pub gpus: HashMap<String, u32>,
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
    #[serde(default)]
    pub minimum_worker_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignedWorker {
    pub name: String,
    pub url: String,
    pub gpu: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolStatus {
    pub state: PoolState,
    #[serde(default)]
    pub assigned_workers: Vec<AssignedWorker>,
    #[serde(default)]
    pub created_at: f64,
    #[serde(default)]
    pub last_renewed: f64,
}

impl Default for PoolStatus {
    fn default() -> Self {
        Self {
            state: PoolState::Pending,
            assigned_workers: Vec::new(),
            created_at: 0.0,
            last_renewed: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pool {
    pub spec: PoolSpec,
    pub status: PoolStatus,
}

#[allow(dead_code)]
impl Pool {
    pub fn is_active(&self) -> bool {
        self.status.state == PoolState::Active
    }

    pub fn has_worker(&self, worker_url: &str) -> bool {
        self.status
            .assigned_workers
            .iter()
            .any(|w| w.url == worker_url)
    }

    pub fn worker_urls(&self) -> Vec<String> {
        self.status
            .assigned_workers
            .iter()
            .map(|w| w.url.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool(state: PoolState, workers: Vec<AssignedWorker>) -> Pool {
        Pool {
            spec: PoolSpec {
                name: "test".into(),
                bundle: None,
                gpus: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
            },
            status: PoolStatus {
                state,
                assigned_workers: workers,
                created_at: 0.0,
                last_renewed: 0.0,
            },
        }
    }

    #[test]
    fn test_is_active() {
        assert!(make_pool(PoolState::Active, vec![]).is_active());
        assert!(!make_pool(PoolState::Pending, vec![]).is_active());
        assert!(!make_pool(PoolState::Expired, vec![]).is_active());
    }

    #[test]
    fn test_has_worker() {
        let pool = make_pool(
            PoolState::Active,
            vec![AssignedWorker {
                name: "w1".into(),
                url: "http://w1:8080".into(),
                gpu: "l4".into(),
            }],
        );
        assert!(pool.has_worker("http://w1:8080"));
        assert!(!pool.has_worker("http://w2:8080"));
    }

    #[test]
    fn test_worker_urls() {
        let pool = make_pool(
            PoolState::Active,
            vec![
                AssignedWorker {
                    name: "w1".into(),
                    url: "http://w1:8080".into(),
                    gpu: "l4".into(),
                },
                AssignedWorker {
                    name: "w2".into(),
                    url: "http://w2:8080".into(),
                    gpu: "l4".into(),
                },
            ],
        );
        assert_eq!(pool.worker_urls(), vec!["http://w1:8080", "http://w2:8080"]);
    }

    #[test]
    fn test_pool_state_display() {
        assert_eq!(PoolState::Pending.to_string(), "pending");
        assert_eq!(PoolState::Active.to_string(), "active");
        assert_eq!(PoolState::Expired.to_string(), "expired");
    }

    #[test]
    fn test_pool_state_serde() {
        let json = serde_json::to_string(&PoolState::Active).unwrap();
        assert_eq!(json, "\"active\"");
        let state: PoolState = serde_json::from_str("\"pending\"").unwrap();
        assert_eq!(state, PoolState::Pending);
    }

    #[test]
    fn test_pool_status_default() {
        let status = PoolStatus::default();
        assert_eq!(status.state, PoolState::Pending);
        assert!(status.assigned_workers.is_empty());
        assert_eq!(status.created_at, 0.0);
    }

    #[test]
    fn test_pool_roundtrip_serialization() {
        let pool = make_pool(
            PoolState::Active,
            vec![AssignedWorker {
                name: "w1".into(),
                url: "http://w1:8080".into(),
                gpu: "l4".into(),
            }],
        );
        let json = serde_json::to_string(&pool).unwrap();
        let deserialized: Pool = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.spec.name, "test");
        assert_eq!(deserialized.status.state, PoolState::Active);
        assert_eq!(deserialized.status.assigned_workers.len(), 1);
    }
}
