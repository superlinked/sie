use std::collections::HashMap;

use tracing::warn;

use crate::state::pool_manager::CapacityPoolSnapshot;
#[cfg(test)]
use crate::types::pool::Pool;

/// One current warm-floor value for a physical queue lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmFloorValue {
    pub pool: String,
    pub machine_profile: String,
    pub bundle: String,
    pub value: u32,
}

/// Build the current warm-floor snapshot from pool business state.
///
/// Each pool with a nonzero floor contributes one value per physical
/// `(queue_pool, machine_profile, bundle)` lane. Logical pools sharing a lane
/// collapse to the maximum requested floor because KEDA scales that physical
/// lane. Removed lanes are deliberately absent: the telemetry facade compares
/// complete snapshots and emits the required explicit zero.
#[cfg(test)]
pub fn warm_floor_values(pools: &[Pool]) -> Vec<WarmFloorValue> {
    let pools: Vec<_> = pools.iter().map(CapacityPoolSnapshot::from_pool).collect();
    warm_floor_values_from_capacity(&pools)
}

/// Build the same warm-floor values from the compact capacity view used by the
/// five-second reconciler.
pub fn warm_floor_values_from_capacity(pools: &[CapacityPoolSnapshot]) -> Vec<WarmFloorValue> {
    let mut values: HashMap<(String, String, String), u32> = HashMap::new();
    for pool in pools {
        if pool.minimum_worker_count == 0 {
            continue;
        }
        if pool.machine_profiles.is_empty() {
            warn!(
                pool = %pool.name,
                minimum_worker_count = pool.minimum_worker_count,
                "warm floor requested but pool has no machine profile; floor is unenforceable"
            );
            continue;
        }
        for profile in &pool.machine_profiles {
            values
                .entry((
                    pool.queue_pool.clone(),
                    profile.clone(),
                    pool.bundle.clone(),
                ))
                .and_modify(|value| *value = (*value).max(pool.minimum_worker_count))
                .or_insert(pool.minimum_worker_count);
        }
    }

    let mut values: Vec<_> = values
        .into_iter()
        .map(|((pool, machine_profile, bundle), value)| WarmFloorValue {
            pool,
            machine_profile,
            bundle,
            value,
        })
        .collect();
    values.sort_by(|left, right| {
        left.pool
            .cmp(&right.pool)
            .then_with(|| left.machine_profile.cmp(&right.machine_profile))
            .then_with(|| left.bundle.cmp(&right.bundle))
    });
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::pool::{PoolSpec, PoolStatus};

    fn pool(name: &str, minimum_worker_count: u32, gpus: &[&str], bundle: Option<&str>) -> Pool {
        let gpus = gpus
            .iter()
            .map(|g| (g.to_string(), 0u32))
            .collect::<HashMap<_, _>>();
        Pool {
            spec: PoolSpec {
                name: name.to_string(),
                queue_pool: name.to_string(),
                bundle: bundle.map(str::to_string),
                gpus,
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count,
                pinned_models: Vec::new(),
            },
            status: PoolStatus::default(),
        }
    }

    fn logical_pool(
        name: &str,
        queue_pool: &str,
        minimum_worker_count: u32,
        gpus: &[&str],
        bundle: Option<&str>,
    ) -> Pool {
        let mut pool = pool(name, minimum_worker_count, gpus, bundle);
        pool.spec.queue_pool = queue_pool.to_string();
        pool
    }

    fn value(pool: &str, profile: &str, bundle: &str, value: u32) -> WarmFloorValue {
        WarmFloorValue {
            pool: pool.to_string(),
            machine_profile: profile.to_string(),
            bundle: bundle.to_string(),
            value,
        }
    }

    #[test]
    fn single_profile_produces_one_lane() {
        assert_eq!(
            warm_floor_values(&[pool("tenant", 2, &["l4"], None)]),
            vec![value("tenant", "l4", "default", 2)]
        );
    }

    #[test]
    fn multi_profile_produces_one_lane_each() {
        assert_eq!(
            warm_floor_values(&[pool("tenant", 2, &["l4", "a100"], None)]),
            vec![
                value("tenant", "a100", "default", 2),
                value("tenant", "l4", "default", 2),
            ]
        );
    }

    #[test]
    fn labels_are_lowercased() {
        assert_eq!(
            warm_floor_values(&[logical_pool(
                "tenant",
                "Customer-Queue",
                1,
                &["L4"],
                Some("SGLang"),
            )]),
            vec![value("customer-queue", "l4", "sglang", 1)]
        );
    }

    #[test]
    fn zero_floor_or_missing_profile_produces_no_lane() {
        assert!(warm_floor_values(&[pool("tenant", 0, &["l4"], None)]).is_empty());
        assert!(warm_floor_values(&[pool("tenant", 3, &[], None)]).is_empty());
    }

    #[test]
    fn snapshot_contains_only_current_profile() {
        assert_eq!(
            warm_floor_values(&[pool("tenant", 2, &["a100"], None)]),
            vec![value("tenant", "a100", "default", 2)]
        );
    }

    #[test]
    fn logical_pools_share_backing_queue_floor_without_summing() {
        assert_eq!(
            warm_floor_values(&[
                logical_pool("tenant-a", "default", 2, &["l4"], None),
                logical_pool("tenant-b", "default", 3, &["l4"], None),
            ]),
            vec![value("default", "l4", "default", 3)]
        );
    }

    #[test]
    fn empty_backing_queue_uses_default_lane() {
        assert_eq!(
            warm_floor_values(&[logical_pool("tenant", "", 2, &["l4"], None)]),
            vec![value("default", "l4", "default", 2)]
        );
    }
}
