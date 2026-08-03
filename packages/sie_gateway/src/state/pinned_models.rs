use std::collections::{HashMap, HashSet};

use crate::observability::metrics::PinnedModelSnapshot;
use crate::types::pool::Pool;

/// Per-pool union of models loaded on healthy workers assigned to that
/// logical pool.
pub type PoolLoadedModels = HashMap<String, HashSet<String>>;

/// Build the complete current pinned-model readiness snapshot from gateway
/// business state. Emission and stale-series clearing stay inside the single
/// OpenTelemetry facade.
pub fn pinned_model_values(
    pools: &[Pool],
    loaded_models: &PoolLoadedModels,
) -> Vec<PinnedModelSnapshot> {
    let mut values = Vec::new();
    for pool in pools {
        let loaded_for_pool = loaded_models.get(&pool.spec.name);
        let mut seen = HashSet::new();
        for model in &pool.spec.pinned_models {
            if !seen.insert(model.as_str()) {
                continue;
            }
            let loaded = loaded_for_pool.is_some_and(|loaded| {
                loaded
                    .iter()
                    .any(|candidate| candidate.eq_ignore_ascii_case(model))
            });
            values.push(PinnedModelSnapshot {
                pool: pool.spec.name.clone(),
                model: model.clone(),
                loaded,
            });
        }
    }
    values.sort_by(|left, right| {
        left.pool
            .cmp(&right.pool)
            .then_with(|| left.model.cmp(&right.model))
    });
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::pool::{PoolSpec, PoolStatus};

    fn pool(name: &str, pinned_models: &[&str]) -> Pool {
        Pool {
            spec: PoolSpec {
                name: name.to_string(),
                queue_pool: "default".to_string(),
                bundle: None,
                gpus: HashMap::new(),
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
                pinned_models: pinned_models
                    .iter()
                    .map(|model| model.to_string())
                    .collect(),
            },
            status: PoolStatus::default(),
        }
    }

    fn loaded(entries: &[(&str, &[&str])]) -> PoolLoadedModels {
        entries
            .iter()
            .map(|(pool, models)| {
                (
                    pool.to_string(),
                    models.iter().map(|model| model.to_string()).collect(),
                )
            })
            .collect()
    }

    fn value(pool: &str, model: &str, loaded: bool) -> PinnedModelSnapshot {
        PinnedModelSnapshot {
            pool: pool.to_string(),
            model: model.to_string(),
            loaded,
        }
    }

    #[test]
    fn reports_loaded_and_unloaded_models_per_logical_pool() {
        assert_eq!(
            pinned_model_values(
                &[pool("tenant", &["BAAI/bge-m3", "intfloat/e5-base-v2"])],
                &loaded(&[("tenant", &["baai/BGE-M3"])]),
            ),
            vec![
                value("tenant", "BAAI/bge-m3", true),
                value("tenant", "intfloat/e5-base-v2", false),
            ]
        );
    }

    #[test]
    fn empty_or_duplicate_pins_do_not_create_extra_lanes() {
        assert!(pinned_model_values(&[pool("tenant", &[])], &loaded(&[])).is_empty());
        assert_eq!(
            pinned_model_values(
                &[pool("tenant", &["BAAI/bge-m3", "BAAI/bge-m3"])],
                &loaded(&[]),
            ),
            vec![value("tenant", "BAAI/bge-m3", false)]
        );
    }

    #[test]
    fn deleted_pool_is_absent_from_complete_snapshot() {
        assert!(pinned_model_values(&[], &loaded(&[("tenant", &["BAAI/bge-m3"])]),).is_empty());
    }
}
