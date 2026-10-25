use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashMap;
use utoipa::ToSchema;

/// The wire spelling of [`PoolTtl::Never`].
pub const POOL_TTL_NEVER: &str = "never";

/// How long a pool lives after its last renewal. On the wire it is a whole
/// number of seconds or the string `"never"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolTtl {
    /// The pool expires this many seconds after its last renewal.
    Seconds(u64),
    /// The pool has no lease: it lives until it is deleted.
    Never,
}

impl Serialize for PoolTtl {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            PoolTtl::Seconds(seconds) => serializer.serialize_u64(*seconds),
            PoolTtl::Never => serializer.serialize_str(POOL_TTL_NEVER),
        }
    }
}

impl<'de> Deserialize<'de> for PoolTtl {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct PoolTtlVisitor;

        impl serde::de::Visitor<'_> for PoolTtlVisitor {
            type Value = PoolTtl;

            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str(r#"a whole number of seconds or "never""#)
            }

            fn visit_u64<E: serde::de::Error>(self, seconds: u64) -> Result<PoolTtl, E> {
                Ok(PoolTtl::Seconds(seconds))
            }

            fn visit_i64<E: serde::de::Error>(self, seconds: i64) -> Result<PoolTtl, E> {
                u64::try_from(seconds)
                    .map(PoolTtl::Seconds)
                    .map_err(|_| E::invalid_value(serde::de::Unexpected::Signed(seconds), &self))
            }

            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<PoolTtl, E> {
                if value == POOL_TTL_NEVER {
                    Ok(PoolTtl::Never)
                } else {
                    Err(E::invalid_value(serde::de::Unexpected::Str(value), &self))
                }
            }
        }

        deserializer.deserialize_any(PoolTtlVisitor)
    }
}

impl utoipa::PartialSchema for PoolTtl {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::openapi::schema::{
            KnownFormat, ObjectBuilder, OneOfBuilder, SchemaFormat, SchemaType, Type,
        };
        OneOfBuilder::new()
            .item(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::Integer))
                    .format(Some(SchemaFormat::KnownFormat(KnownFormat::Int64)))
                    .minimum(Some(1))
                    .description(Some(
                        "The pool expires this many seconds after its last renewal.",
                    )),
            )
            .item(
                ObjectBuilder::new()
                    .schema_type(SchemaType::Type(Type::String))
                    .enum_values(Some([POOL_TTL_NEVER]))
                    .description(Some("The pool never expires; it lives until it is deleted.")),
            )
            .description(Some(
                "How long a pool lives after its last renewal: a whole number of seconds, or \"never\".",
            ))
            .into()
    }
}

impl ToSchema for PoolTtl {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct PoolSpec {
    pub name: String,
    /// Physical Helm/NATS queue namespace backing this logical API pool.
    ///
    /// API-created pools default to the Helm `default` queue pool. Pool `name`
    /// is user/dynamic; a non-default `queue_pool` is infra/admin and must
    /// match a Helm worker queue declared under `queueRouting.staticQueuePools`.
    #[serde(default = "default_queue_pool")]
    pub queue_pool: String,
    #[serde(default)]
    pub bundle: Option<String>,
    #[serde(default)]
    pub gpus: HashMap<String, u32>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub gpu_caps: HashMap<String, u32>,
    /// How long the pool lives after its last renewal: whole seconds, or
    /// `"never"` for a pool that lives until it is deleted. `null` on an
    /// API-created pool means the gateway's default lease. The `default` pool
    /// and static Helm queue pools report `"never"`.
    #[serde(default)]
    pub ttl_seconds: Option<PoolTtl>,
    /// Per-pool warm floor: the minimum number of machines the gateway keeps
    /// warm (via KEDA) so the first request to the pool never hits a cold VM.
    /// The gateway publishes it as `sie.gateway.pool.warm_floor`; the
    /// collector translates it for KEDA's Prometheus query.
    /// Default 0 leaves scale-from-zero unchanged.
    #[serde(default)]
    pub minimum_worker_count: u32,
    /// Per-pool pinned-model set: models the gateway keeps loaded so the first
    /// request to them pays no cold model-load. Chosen from the models the
    /// gateway already tracks (see `GET /v1/configs/models`); ids may be
    /// profile-qualified (`model-name:profile_name`). Default empty leaves
    /// lazy-loading unchanged.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pinned_models: Vec<String>,
}

pub fn default_queue_pool() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
pub struct AssignedWorker {
    pub name: String,
    pub url: String,
    pub gpu: String,
    #[serde(default = "default_bundle")]
    pub bundle: String,
}

fn default_bundle() -> String {
    "default".to_string()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, ToSchema)]
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
                queue_pool: default_queue_pool(),
                bundle: None,
                gpus: HashMap::new(),
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
                pinned_models: Vec::new(),
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
                bundle: "default".into(),
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
                    bundle: "default".into(),
                },
                AssignedWorker {
                    name: "w2".into(),
                    url: "http://w2:8080".into(),
                    gpu: "l4".into(),
                    bundle: "default".into(),
                },
            ],
        );
        assert_eq!(pool.worker_urls(), vec!["http://w1:8080", "http://w2:8080"]);
    }

    #[test]
    fn test_pool_ttl_wire_format() {
        assert_eq!(
            serde_json::to_value(PoolTtl::Seconds(86_400)).unwrap(),
            serde_json::json!(86_400)
        );
        assert_eq!(
            serde_json::to_value(PoolTtl::Never).unwrap(),
            serde_json::json!("never")
        );
        assert_eq!(
            serde_json::from_value::<PoolTtl>(serde_json::json!(60)).unwrap(),
            PoolTtl::Seconds(60)
        );
        assert_eq!(
            serde_json::from_value::<PoolTtl>(serde_json::json!("never")).unwrap(),
            PoolTtl::Never
        );
        for invalid in [
            serde_json::json!(-1),
            serde_json::json!(1.5),
            serde_json::json!("forever"),
            serde_json::json!("Never"),
            serde_json::json!(true),
        ] {
            let error = serde_json::from_value::<PoolTtl>(invalid.clone())
                .expect_err("only whole seconds or \"never\" are a pool TTL");
            assert!(
                error
                    .to_string()
                    .contains(r#"a whole number of seconds or "never""#),
                "{invalid}: {error}"
            );
        }
    }

    #[test]
    fn test_pool_spec_ttl_round_trips_through_stored_json() {
        let mut pool = make_pool(PoolState::Active, vec![]);
        for ttl in [None, Some(PoolTtl::Seconds(3600)), Some(PoolTtl::Never)] {
            pool.spec.ttl_seconds = ttl;
            let stored = serde_json::to_string(&pool).unwrap();
            let restored: Pool = serde_json::from_str(&stored).unwrap();
            assert_eq!(restored.spec.ttl_seconds, ttl, "{stored}");
        }
        // Pools stored before `"never"` existed still parse.
        let legacy = r#"{"spec":{"name":"legacy","gpus":{"l4":1},"ttl_seconds":600},"status":{"state":"active"}}"#;
        let restored: Pool = serde_json::from_str(legacy).unwrap();
        assert_eq!(restored.spec.ttl_seconds, Some(PoolTtl::Seconds(600)));
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
                bundle: "default".into(),
            }],
        );
        let json = serde_json::to_string(&pool).unwrap();
        let deserialized: Pool = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.spec.name, "test");
        assert_eq!(deserialized.status.state, PoolState::Active);
        assert_eq!(deserialized.status.assigned_workers.len(), 1);
    }

    #[test]
    fn test_pool_spec_minimum_worker_count_roundtrips() {
        let mut pool = make_pool(PoolState::Pending, vec![]);
        pool.spec.minimum_worker_count = 3;
        let json = serde_json::to_string(&pool).unwrap();
        assert!(json.contains("\"minimum_worker_count\":3"));
        let deserialized: Pool = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.spec.minimum_worker_count, 3);
    }

    #[test]
    fn test_pool_spec_minimum_worker_count_defaults_to_zero_when_absent() {
        // A spec JSON that omits `minimum_worker_count` must default to 0 so
        // older persisted ConfigMap state and bare create requests keep working.
        let json = r#"{"name":"test","gpus":{"l4":1}}"#;
        let spec: PoolSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.minimum_worker_count, 0);
    }

    #[test]
    fn test_pool_spec_queue_pool_defaults_to_default_when_absent() {
        // Older persisted ConfigMap state omits `queue_pool`; restore must keep
        // those logical pools backed by the default physical queue pool.
        let json = r#"{"name":"test","gpus":{"l4":1}}"#;
        let spec: PoolSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.queue_pool, default_queue_pool());
    }

    #[test]
    fn test_pool_spec_pinned_models_roundtrips() {
        let mut pool = make_pool(PoolState::Pending, vec![]);
        pool.spec.pinned_models = vec!["BAAI/bge-m3".to_string()];
        let json = serde_json::to_string(&pool).unwrap();
        assert!(json.contains("\"pinned_models\":[\"BAAI/bge-m3\"]"));
        let deserialized: Pool = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.spec.pinned_models, vec!["BAAI/bge-m3"]);
    }

    #[test]
    fn test_pool_spec_pinned_models_defaults_empty_when_absent() {
        // A spec JSON that omits `pinned_models` must default to empty so
        // older persisted ConfigMap state and bare create requests keep working.
        let json = r#"{"name":"test","gpus":{"l4":1}}"#;
        let spec: PoolSpec = serde_json::from_str(json).unwrap();
        assert!(spec.pinned_models.is_empty());
    }
}
