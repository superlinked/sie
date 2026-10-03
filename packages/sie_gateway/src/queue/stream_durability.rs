//! Shared reconcile policy for the durability knobs on JetStream streams
//! this gateway owns: the `WORK_POOL_{pool}` work queues and `DEAD_LETTERS`.
//!
//! Both are created from `SIE_STREAM_STORAGE` / `SIE_STREAM_REPLICAS`, and
//! both face the same question on an *existing* stream: what happens when the
//! live stream's shape does not match the configured one? The answer differs
//! per field and is easy to get wrong in one place and not the other, so it
//! lives here once:
//!
//! - `num_replicas` IS changeable on a live stream (this is JetStream's
//!   supported R1 -> R3 scaling path), so it is reconciled.
//! - `storage` is NOT. NATS rejects the update with err_code 10052,
//!   "stream configuration update can not change storage type", because the
//!   message store is already materialized in RAM or on disk. The only way to
//!   "fix" it is to delete and recreate the stream — which destroys the
//!   queued work, or the forensic dead-letter record, that durability exists
//!   to protect. So a divergence is reported loudly and the stream is left
//!   intact for the operator to convert during a maintenance window.
//!
//! See the chart README's "Work-queue durability" section.

use async_nats::jetstream;
use tracing::{info, warn};

/// Reconcile an existing stream's replica count to `desired_replicas`.
///
/// Refreshes the caller's stream handle when an update is applied, so code
/// that keeps using it observes the new config.
///
/// Note that NATS does not police this against the actual topology: a
/// single-node server accepts `num_replicas: 3` and then reports R3, with no
/// cluster to back it (verified against nats-server 2.12.6). Guarding a
/// replica count against the deployed NATS cluster size is therefore the
/// chart's job, not the broker's — see the `streamReplicas` guards in
/// deploy/helm/sie-cluster/templates/gateway-deployment.yaml.
pub async fn reconcile_num_replicas(
    jetstream: &jetstream::Context,
    stream: &mut jetstream::stream::Stream,
    name: &str,
    desired_replicas: usize,
) -> Result<(), String> {
    let observed_replicas = stream.cached_info().config.num_replicas;
    if observed_replicas == desired_replicas {
        return Ok(());
    }

    let mut updated = stream.cached_info().config.clone();
    updated.num_replicas = desired_replicas;
    jetstream
        .update_stream(updated)
        .await
        .map_err(|e| format!("update stream {} num_replicas: {}", name, e))?;
    *stream = jetstream
        .get_stream(name.to_string())
        .await
        .map_err(|e| format!("refresh stream {} after num_replicas update: {}", name, e))?;
    info!(
        stream = %name,
        observed_replicas,
        desired_replicas,
        "reconciled JetStream stream num_replicas"
    );
    Ok(())
}

/// Report — but never repair — a storage-type divergence on an existing
/// stream. See the module docs for why this is deliberately not reconciled.
pub fn warn_on_storage_mismatch(
    stream: &jetstream::stream::Stream,
    name: &str,
    desired_storage: jetstream::stream::StorageType,
) {
    let observed_storage = stream.cached_info().config.storage;
    if observed_storage == desired_storage {
        return;
    }
    warn!(
        stream = %name,
        observed_storage = ?observed_storage,
        desired_storage = ?desired_storage,
        "JetStream stream storage type differs from SIE_STREAM_STORAGE and CANNOT be changed \
         in place; the stream keeps its existing storage until an operator drains it and \
         recreates it during a maintenance window"
    );
}
