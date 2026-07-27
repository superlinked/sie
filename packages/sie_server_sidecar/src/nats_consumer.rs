//! NATS JetStream pull consumer.
//!
//! This module owns the pool-level stream and durable pull consumer
//! matching the Python naming contract (see
//! `config::WorkerConfig::{stream_name,consumer_name,subject_filter}`),
//! then exposes a long-lived pull stream for the dispatcher. ACK/NAK,
//! batching, and adaptive fetch timing live above this layer.

use std::time::Duration;

use async_nats::jetstream::consumer::pull::{Config as ConsumerConfig, Stream as PullStream};
use async_nats::jetstream::consumer::{AckPolicy, PullConsumer};
use async_nats::jetstream::stream::{
    Config as StreamConfig, DiscardPolicy, RetentionPolicy, StorageType, Stream as JsStream,
};
use async_nats::jetstream::Context as JsContext;
use futures_util::TryStreamExt;
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::config::WorkerConfig;
use crate::subject::subjects_overlap;

/// Default consumer parameters. Must match the gateway's stream creator
/// and any other worker in the pool — when multiple creators share a
/// durable consumer by name, the first to create it wins, and subsequent
/// callers silently inherit whatever was there.
pub(crate) const ACK_WAIT_SECS: u64 = 30;
const GENERATION_ACK_WAIT_SECS: u64 = 300;
const DEFAULT_MAX_DELIVER: i64 = 20;
const DEFAULT_MAX_ACK_PENDING: i64 = 1000;
const DEFAULT_STREAM_MAX_AGE_SECS: u64 = 1_800;
const STREAM_MAX_MSGS: i64 = 100_000;

/// Env-overridable `max_deliver`, wired to the worker-sidecar by Helm
/// as `SIE_MAX_DELIVER` (default 20). With the default 30s ACK wait,
/// this gives a 600s retry envelope before a message hits the DLQ.
fn max_deliver() -> i64 {
    std::env::var("SIE_MAX_DELIVER")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_DELIVER)
}

/// The larger of the durable work lifetime and configured retry envelope is the
/// active-worker expiry horizon; the bounded state may evict the oldest entry
/// sooner. Core NATS cancellation remains best-effort: a worker that starts or
/// reconnects after the signal must still rely on payload cleanup/deadline
/// enforcement.
pub(crate) fn work_cancel_tombstone_ttl() -> Duration {
    Duration::from_secs(
        stream_max_age_secs().max(ACK_WAIT_SECS.saturating_mul(max_deliver() as u64)),
    )
}

/// Env-overridable stream `max_age`, wired to the worker-sidecar by
/// Helm as `SIE_STREAM_MAX_AGE_S` (default 1800). Should be >
/// `max_deliver * ack_wait` so messages remain inspectable after DLQ.
fn stream_max_age_secs() -> u64 {
    std::env::var("SIE_STREAM_MAX_AGE_S")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_STREAM_MAX_AGE_SECS)
}

/// Env-overridable `max_ack_pending`. Python hardcodes 1000 for now but
/// exposing the knob keeps the option open without a code change.
fn max_ack_pending() -> i64 {
    std::env::var("SIE_MAX_ACK_PENDING")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_MAX_ACK_PENDING)
}

fn generation_stream_max_age_secs() -> u64 {
    stream_max_age_secs().max(GENERATION_ACK_WAIT_SECS * max_deliver() as u64)
}

/// Opt-in: when `SIE_NATS_CONSERVATIVE_CLEANUP=1`, the stale-durable
/// sweep only deletes consumers that have **zero** pull waiters
/// (`num_waiting == 0`) and **zero** in-flight acks
/// (`num_ack_pending == 0`). Default behaviour (env unset / `0`)
/// keeps the historical aggressive delete because a concrete
/// `(pool, machine_profile, bundle)` lane must have one active durable
/// owner. Bundle peers can now share the same pool stream safely because
/// their lane filters do not overlap.
///
/// Flip this on if you ever introduce a topology where two consumers
/// legitimately share overlapping lane filters; we then refuse to delete
/// a healthy peer consumer and let the authoritative
/// `get_or_create_consumer` error surface so the operator can resolve
/// the conflict explicitly.
fn conservative_cleanup() -> bool {
    std::env::var("SIE_NATS_CONSERVATIVE_CLEANUP")
        .ok()
        .filter(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .is_some()
}

#[derive(Debug, Error)]
pub enum NatsSetupError {
    #[error("ensure stream {name}: {source}")]
    EnsureStream {
        name: String,
        #[source]
        source: async_nats::Error,
    },
    #[error("ensure consumer {name}: {source}")]
    EnsureConsumer {
        name: String,
        #[source]
        source: async_nats::Error,
    },
    #[error("connect NATS: {source}")]
    Connect {
        #[source]
        source: async_nats::ConnectError,
    },
}

/// Connect to NATS and return a JetStream context.
pub async fn connect(nats_url: &str) -> Result<(async_nats::Client, JsContext), NatsSetupError> {
    let client = async_nats::connect(nats_url)
        .await
        .map_err(|source| NatsSetupError::Connect { source })?;
    let js = async_nats::jetstream::new(client.clone());
    Ok((client, js))
}

fn canonical_stream_subjects(observed: Vec<String>, desired: &str) -> Option<Vec<String>> {
    if observed.len() == 1 && observed.first().is_some_and(|subject| subject == desired) {
        None
    } else {
        Some(vec![desired.to_string()])
    }
}

async fn reconcile_stream_subjects(
    js: &JsContext,
    stream: &mut JsStream,
    stream_name: &str,
    desired_subject: &str,
) -> Result<(), NatsSetupError> {
    let observed_subjects = stream.cached_info().config.subjects.clone();
    let Some(updated_subjects) =
        canonical_stream_subjects(observed_subjects.clone(), desired_subject)
    else {
        return Ok(());
    };

    let mut updated = stream.cached_info().config.clone();
    updated.subjects = updated_subjects;
    js.update_stream(updated)
        .await
        .map_err(|e| NatsSetupError::EnsureStream {
            name: stream_name.to_string(),
            source: e.into(),
        })?;
    *stream = js
        .get_stream(stream_name)
        .await
        .map_err(|e| NatsSetupError::EnsureStream {
            name: stream_name.to_string(),
            source: e.into(),
        })?;
    info!(
        stream = %stream_name,
        observed_subjects = ?observed_subjects,
        desired_subject = %desired_subject,
        "reconciled NATS stream subjects to canonical queue lane routing"
    );
    Ok(())
}

async fn reconcile_stream_max_age(
    js: &JsContext,
    stream: &mut JsStream,
    stream_name: &str,
    desired_max_age: Duration,
) -> Result<(), NatsSetupError> {
    let observed_max_age = stream.cached_info().config.max_age;
    if observed_max_age == desired_max_age {
        return Ok(());
    }

    let mut updated = stream.cached_info().config.clone();
    updated.max_age = desired_max_age;
    js.update_stream(updated)
        .await
        .map_err(|e| NatsSetupError::EnsureStream {
            name: stream_name.to_string(),
            source: e.into(),
        })?;
    *stream = js
        .get_stream(stream_name)
        .await
        .map_err(|e| NatsSetupError::EnsureStream {
            name: stream_name.to_string(),
            source: e.into(),
        })?;
    info!(
        stream = %stream_name,
        observed_max_age_s = observed_max_age.as_secs(),
        desired_max_age_s = desired_max_age.as_secs(),
        "reconciled NATS stream max_age"
    );
    Ok(())
}

/// Predicate used by [`cleanup_overlapping_consumers`]. Returns true
/// iff any of the consumer's configured filter(s) overlaps the
/// worker's intended `desired_subject` per NATS wildcard semantics.
///
/// Split out from `Info` so it can be unit-tested without constructing
/// the (large, time-stamped) `Info` value.
fn any_filter_overlaps(
    primary_filter: &str,
    multi_filters: &[String],
    desired_subject: &str,
) -> bool {
    if !primary_filter.is_empty() && subjects_overlap(primary_filter, desired_subject) {
        return true;
    }
    multi_filters
        .iter()
        .any(|s| subjects_overlap(s, desired_subject))
}

fn consumer_filter_matches_desired(
    primary_filter: &str,
    multi_filters: &[String],
    desired_subject: &str,
) -> bool {
    if primary_filter == desired_subject && multi_filters.is_empty() {
        return true;
    }
    primary_filter.is_empty()
        && multi_filters.len() == 1
        && multi_filters.first().is_some_and(|s| s == desired_subject)
}

async fn reconcile_same_name_consumer_filter(
    stream: &JsStream,
    mut consumer: PullConsumer,
    consumer_name: &str,
    desired_subject: &str,
    consumer_cfg: ConsumerConfig,
) -> Result<PullConsumer, NatsSetupError> {
    let Ok(info) = consumer.info().await else {
        return Ok(consumer);
    };
    if consumer_filter_matches_desired(
        &info.config.filter_subject,
        &info.config.filter_subjects,
        desired_subject,
    ) {
        return Ok(consumer);
    }

    warn!(
        stream = %stream.cached_info().config.name,
        consumer = %consumer_name,
        observed_filter_subject = %info.config.filter_subject,
        observed_filter_subjects = ?info.config.filter_subjects,
        desired_subject,
        "durable consumer filter drift: updating same-name consumer to the current queue lane"
    );

    match stream.update_consumer(consumer_cfg.clone()).await {
        Ok(updated) => Ok(updated),
        Err(update_error) => {
            warn!(
                stream = %stream.cached_info().config.name,
                consumer = %consumer_name,
                error = %update_error,
                "durable consumer update failed; deleting and recreating stale same-name consumer"
            );
            stream.delete_consumer(consumer_name).await.map_err(|e| {
                NatsSetupError::EnsureConsumer {
                    name: consumer_name.to_string(),
                    source: e.into(),
                }
            })?;
            consumer = stream
                .get_or_create_consumer(consumer_name, consumer_cfg)
                .await
                .map_err(|e| NatsSetupError::EnsureConsumer {
                    name: consumer_name.to_string(),
                    source: e.into(),
                })?;
            Ok(consumer)
        }
    }
}

/// Best-effort: remove any durable consumer on `stream` whose name is
/// not `desired_name` but whose filter subject(s) overlap
/// `desired_subject`. NATS WorkQueue forbids overlapping filters across
/// consumers on the same stream — without this, the worker can't bind
/// after a bundle/engine flip on a shared pool.
///
/// Failures (list errors, individual deletes) are logged but never
/// returned: we'd rather let `get_or_create_consumer` produce the
/// authoritative error a moment later than block startup on a transient
/// listing hiccup. In practice the only scenarios where listing fails
/// are also scenarios where the consumer create will fail loudly.
async fn cleanup_overlapping_consumers(
    stream: &JsStream,
    desired_name: &str,
    desired_subject: &str,
) {
    let conservative = conservative_cleanup();
    let mut listing = stream.consumers();
    let mut stale: Vec<(String, i64, i64)> = Vec::new();
    loop {
        match listing.try_next().await {
            Ok(Some(info)) => {
                if info.name == desired_name {
                    continue;
                }
                if any_filter_overlaps(
                    &info.config.filter_subject,
                    &info.config.filter_subjects,
                    desired_subject,
                ) {
                    stale.push((
                        info.name,
                        info.num_waiting as i64,
                        info.num_ack_pending as i64,
                    ));
                }
            }
            Ok(None) => break,
            Err(e) => {
                warn!(
                    error = %e,
                    desired_subject,
                    "listing consumers failed during stale-durable sweep; \
                     proceeding to consumer-create which will surface the \
                     authoritative error if a true overlap remains"
                );
                return;
            }
        }
    }

    for (name, num_waiting, num_ack_pending) in stale {
        if conservative && (num_waiting > 0 || num_ack_pending > 0) {
            warn!(
                stream = %stream.cached_info().config.name,
                stale_consumer = %name,
                desired_consumer = %desired_name,
                desired_subject,
                num_waiting,
                num_ack_pending,
                "skipping stale-durable delete: peer consumer looks active and \
                 SIE_NATS_CONSERVATIVE_CLEANUP=1; consumer-create will surface \
                 the overlap error and the operator must resolve it manually"
            );
            continue;
        }
        warn!(
            stream = %stream.cached_info().config.name,
            stale_consumer = %name,
            desired_consumer = %desired_name,
            desired_subject,
            num_waiting,
            num_ack_pending,
            conservative,
            "deleting stale durable that overlaps this worker's filter \
             (left over from a prior bundle/engine deploy on this pool)"
        );
        if let Err(e) = stream.delete_consumer(&name).await {
            warn!(
                stale_consumer = %name,
                error = %e,
                "failed to delete stale durable; consumer-create will likely fail next"
            );
        }
    }
}

/// Ensure the pool stream and the durable pull consumer exist. Called once
/// per worker startup; safe to call again for slow stream/durable reconcile.
pub async fn ensure_stream_and_consumer(
    js: &JsContext,
    config: &WorkerConfig,
) -> Result<PullConsumer, NatsSetupError> {
    ensure_stream_and_consumer_inner(js, config, true).await
}

/// Re-ensure the pool stream and durable consumer without rebuilding the
/// active pull stream. This is intentionally a slow control-plane repair path:
/// the hot fetch loop still owns pull-stream recovery on terminal stream errors.
pub async fn reconcile_stream_and_consumer(
    js: &JsContext,
    config: &WorkerConfig,
) -> Result<(), NatsSetupError> {
    ensure_stream_and_consumer_inner(js, config, false)
        .await
        .map(|_| ())
}

/// Ensure the worker-specific stream used by direct-dispatch.
///
/// The gateway publishes worker-directed work to
/// `sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`. That
/// subject intentionally does not match the pool stream
/// (`sie.work.{pool}.*.*.*`), so the sidecar must
/// bind this second stream for generation and capped logical batch dispatch.
pub async fn ensure_worker_stream_and_consumer(
    js: &JsContext,
    config: &WorkerConfig,
) -> Result<PullConsumer, NatsSetupError> {
    let stream_name = config.worker_stream_name();
    let subject = config.worker_subject_filter();
    let consumer_name = config.worker_consumer_name();
    let desired_max_age = Duration::from_secs(generation_stream_max_age_secs());

    let stream_cfg = StreamConfig {
        name: stream_name.clone(),
        subjects: vec![subject.clone()],
        retention: RetentionPolicy::WorkQueue,
        storage: StorageType::Memory,
        max_age: desired_max_age,
        max_messages: STREAM_MAX_MSGS,
        num_replicas: 1,
        discard: DiscardPolicy::New,
        ..Default::default()
    };

    let desired_ack_wait = Duration::from_secs(GENERATION_ACK_WAIT_SECS);
    let desired_max_deliver = max_deliver();
    let desired_max_ack_pending = max_ack_pending();
    let mut stream =
        js.get_or_create_stream(stream_cfg)
            .await
            .map_err(|e| NatsSetupError::EnsureStream {
                name: stream_name.clone(),
                source: e.into(),
            })?;
    reconcile_stream_subjects(js, &mut stream, &stream_name, &subject).await?;
    reconcile_stream_max_age(js, &mut stream, &stream_name, desired_max_age).await?;

    cleanup_overlapping_consumers(&stream, &consumer_name, &subject).await;

    let consumer_cfg = ConsumerConfig {
        durable_name: Some(consumer_name.clone()),
        filter_subject: subject.clone(),
        ack_policy: AckPolicy::Explicit,
        ack_wait: desired_ack_wait,
        max_deliver: desired_max_deliver,
        max_ack_pending: desired_max_ack_pending,
        ..Default::default()
    };
    let consumer: PullConsumer = stream
        .get_or_create_consumer(&consumer_name, consumer_cfg)
        .await
        .map_err(|e| NatsSetupError::EnsureConsumer {
            name: consumer_name.clone(),
            source: e.into(),
        })?;
    let consumer = reconcile_same_name_consumer_filter(
        &stream,
        consumer,
        &consumer_name,
        &subject,
        ConsumerConfig {
            durable_name: Some(consumer_name.clone()),
            filter_subject: subject.clone(),
            ack_policy: AckPolicy::Explicit,
            ack_wait: desired_ack_wait,
            max_deliver: desired_max_deliver,
            max_ack_pending: desired_max_ack_pending,
            ..Default::default()
        },
    )
    .await?;
    info!(
        stream = %stream_name,
        consumer = %consumer_name,
        subject = %subject,
        ack_wait_s = GENERATION_ACK_WAIT_SECS,
        "ensured worker direct-dispatch stream and pull consumer"
    );
    Ok(consumer)
}

pub async fn reconcile_worker_stream_and_consumer(
    js: &JsContext,
    config: &WorkerConfig,
) -> Result<(), NatsSetupError> {
    ensure_worker_stream_and_consumer(js, config)
        .await
        .map(|_| ())
}

async fn ensure_stream_and_consumer_inner(
    js: &JsContext,
    config: &WorkerConfig,
    log_success_at_info: bool,
) -> Result<PullConsumer, NatsSetupError> {
    let stream_name = config.stream_name();
    let stream_subject = config.stream_subject_filter();
    let subject = config.subject_filter();
    let consumer_name = config.consumer_name();

    let stream_cfg = StreamConfig {
        name: stream_name.clone(),
        subjects: vec![stream_subject.clone()],
        retention: RetentionPolicy::WorkQueue,
        storage: StorageType::Memory,
        max_age: Duration::from_secs(stream_max_age_secs()),
        max_messages: STREAM_MAX_MSGS,
        num_replicas: 1,
        discard: DiscardPolicy::New,
        ..Default::default()
    };

    let desired_max_age = stream_cfg.max_age;
    let desired_max_deliver = max_deliver();
    let desired_max_ack_pending = max_ack_pending();
    let desired_ack_wait = Duration::from_secs(ACK_WAIT_SECS);

    let mut stream =
        js.get_or_create_stream(stream_cfg)
            .await
            .map_err(|e| NatsSetupError::EnsureStream {
                name: stream_name.clone(),
                source: e.into(),
            })?;
    reconcile_stream_subjects(js, &mut stream, &stream_name, &stream_subject).await?;
    reconcile_stream_max_age(js, &mut stream, &stream_name, desired_max_age).await?;

    // Self-heal stale durables whose filters overlap this concrete
    // `(pool, machine_profile, bundle)` lane. Without this, NATS rejects
    // our consumer create with `consumer filter subject overlaps with X`
    // and the worker CrashLoops until an operator removes the stale
    // durable. Different bundles in the same pool no longer overlap
    // because the bundle is now a subject token.
    cleanup_overlapping_consumers(&stream, &consumer_name, &subject).await;

    let consumer_cfg = ConsumerConfig {
        durable_name: Some(consumer_name.clone()),
        filter_subject: subject.clone(),
        ack_policy: AckPolicy::Explicit,
        ack_wait: desired_ack_wait,
        max_deliver: desired_max_deliver,
        max_ack_pending: desired_max_ack_pending,
        ..Default::default()
    };

    let mut consumer: PullConsumer = stream
        .get_or_create_consumer(&consumer_name, consumer_cfg)
        .await
        .map_err(|e| NatsSetupError::EnsureConsumer {
            name: consumer_name.clone(),
            source: e.into(),
        })?;
    consumer = reconcile_same_name_consumer_filter(
        &stream,
        consumer,
        &consumer_name,
        &subject,
        ConsumerConfig {
            durable_name: Some(consumer_name.clone()),
            filter_subject: subject.clone(),
            ack_policy: AckPolicy::Explicit,
            ack_wait: desired_ack_wait,
            max_deliver: desired_max_deliver,
            max_ack_pending: desired_max_ack_pending,
            ..Default::default()
        },
    )
    .await?;

    // Same drift warning for the durable consumer. `get_or_create_consumer`
    // inherits the existing durable's config; new tuning here is silently
    // dropped until someone deletes the consumer.
    if let Ok(info) = consumer.info().await {
        if info.config.max_deliver != desired_max_deliver
            || info.config.ack_wait != desired_ack_wait
            || info.config.max_ack_pending != desired_max_ack_pending
        {
            warn!(
                consumer = %consumer_name,
                observed_max_deliver = info.config.max_deliver,
                desired_max_deliver,
                observed_ack_wait_s = info.config.ack_wait.as_secs(),
                desired_ack_wait_s = desired_ack_wait.as_secs(),
                observed_max_ack_pending = info.config.max_ack_pending,
                desired_max_ack_pending,
                "durable consumer config drift: existing durable does not match this worker's \
                 intended tuning. Delete the consumer or update the owning provisioner."
            );
        }
    }

    if log_success_at_info {
        info!(
            stream = %stream_name,
            consumer = %consumer_name,
            subject = %subject,
            "ensured NATS stream and pull consumer"
        );
    } else {
        debug!(
            stream = %stream_name,
            consumer = %consumer_name,
            subject = %subject,
            "reconciled NATS stream and pull consumer"
        );
    }
    Ok(consumer)
}

/// Lightweight wrapper around a pull consumer stream.
pub struct NatsConsumer {
    consumer: PullConsumer,
}

impl NatsConsumer {
    pub fn new(consumer: PullConsumer) -> Self {
        Self { consumer }
    }

    /// Begin a message stream with the given per-fetch batch size and
    /// expiry. Callers drive it via `futures_util::StreamExt::next` in the
    /// dispatcher loop.
    ///
    /// **Important:** the returned [`PullStream`] is designed to be held
    /// open for the process lifetime. Dropping & recreating it per tick
    /// is pathological — any messages the server has already routed to a
    /// not-yet-drained stream become `ack_pending` until `ack_wait`
    /// elapses (30 s by default), producing a catastrophic throughput
    /// collapse. Callers therefore construct one stream at startup and poll it
    /// with `.next().await` for as long as the worker runs.
    pub async fn messages(
        &self,
        batch: usize,
        expires: Duration,
    ) -> Result<PullStream, async_nats::Error> {
        let stream = self
            .consumer
            .stream()
            .max_messages_per_batch(batch)
            .expires(expires)
            .messages()
            .await?;
        Ok(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WorkerConfig;
    use std::path::PathBuf;

    #[test]
    fn consumer_config_matches_shared_defaults() {
        // Shared defaults (env-overridable where documented) across all
        // consumer creators in the pool:
        //   ack_wait = 30s
        //   max_deliver = 20            (env SIE_MAX_DELIVER)
        //   max_ack_pending = 1000      (env SIE_MAX_ACK_PENDING)
        //   stream max_age = 1800s      (env SIE_STREAM_MAX_AGE_S)
        //   stream max_msgs = 100_000
        assert_eq!(ACK_WAIT_SECS, 30);
        assert_eq!(DEFAULT_MAX_DELIVER, 20);
        assert_eq!(DEFAULT_MAX_ACK_PENDING, 1000);
        assert_eq!(DEFAULT_STREAM_MAX_AGE_SECS, 1_800);
        assert_eq!(STREAM_MAX_MSGS, 100_000);
    }

    #[test]
    fn canonical_stream_subjects_replaces_legacy_subject() {
        let observed = vec!["sie.work.*.default".to_string()];
        assert_eq!(
            canonical_stream_subjects(observed, "sie.work.default.*.*.*"),
            Some(vec!["sie.work.default.*.*.*".to_string()])
        );
    }

    #[test]
    fn canonical_stream_subjects_drops_extra_subjects() {
        let observed = vec![
            "sie.work.*.default".to_string(),
            "sie.work.other.*.*.*".to_string(),
        ];
        assert_eq!(
            canonical_stream_subjects(observed, "sie.work.default.*.*.*"),
            Some(vec!["sie.work.default.*.*.*".to_string()])
        );
    }

    #[test]
    fn canonical_stream_subjects_noops_when_exact() {
        let observed = vec!["sie.work.default.*.*.*".to_string()];
        assert_eq!(
            canonical_stream_subjects(observed, "sie.work.default.*.*.*"),
            None
        );
    }

    #[test]
    fn env_free_accessors_match_defaults() {
        // Env-free calls must return the hardcoded defaults. We do not
        // touch the environment (would be racy with other tests) — this
        // just guards against accidental typos in the env-var key.
        // If a developer happens to set these env vars locally, the
        // assertion may legitimately differ; skip-gate on that case.
        if std::env::var("SIE_MAX_DELIVER").is_err() {
            assert_eq!(max_deliver(), DEFAULT_MAX_DELIVER);
        }
        if std::env::var("SIE_STREAM_MAX_AGE_S").is_err() {
            assert_eq!(stream_max_age_secs(), DEFAULT_STREAM_MAX_AGE_SECS);
        }
        if std::env::var("SIE_MAX_ACK_PENDING").is_err() {
            assert_eq!(max_ack_pending(), DEFAULT_MAX_ACK_PENDING);
        }
    }

    #[test]
    fn config_produces_expected_names() {
        let cfg = WorkerConfig {
            nats_url: Some(String::new()),
            local_socket_path: None,
            pool: "default".into(),
            bundle: "b".into(),
            ipc_socket_path: PathBuf::from("/tmp/s.sock"),
            ipc_socket_paths: vec![PathBuf::from("/tmp/s.sock")],
            ipc_pool_size: 1,
            ipc_request_timeout_s: 60,
            model_ready_timeout_s: 900,
            payload_store_url: None,
            gateway_url: None,
            gateway_api_key: None,
            pool_admission_enabled: true,
            pool_admission_check_interval_ms: 5_000,
            pool_admission_pause_ms: 1_000,
            pool_admission_stale_after_ms: 30_000,
            probe_port: 9095,
            worker_id: "w".into(),
            ping_interval_ms: 2000,
            ready_stale_mult: 3,
            machine_profile: "default".into(),
            gpu_count: 1,
            bundle_config_hash: String::new(),
            config_service_url: None,
            config_service_token: None,
            config_poll_interval_ms: 30_000,
            config_full_export_interval_ms: 300_000,
            nats_config_trusted_producers: vec!["sie-config".into()],
            health_publish_interval_ms: 5_000,
        };
        assert_eq!(cfg.stream_name(), "WORK_POOL_default");
        assert_eq!(cfg.consumer_name(), "default_default_b");
        assert_eq!(cfg.stream_subject_filter(), "sie.work.default.*.*.*");
        assert_eq!(cfg.subject_filter(), "sie.work.default.default.b.*");
    }

    // Connect/ensure tests need a live NATS and are covered by integration
    // smoke tests. Here we just ensure the code compiles and the surface is usable.
    #[allow(dead_code)]
    fn _compile_check(_js: &JsContext, _cfg: &WorkerConfig) {}

    // ----- stale-durable predicate -------------------------------------------

    #[test]
    fn any_filter_overlaps_detects_same_lane_collision() {
        // Same pool + machine profile + bundle must overlap: replicas of
        // one StatefulSet intentionally share one durable consumer.
        let stale_filter = "sie.work.default.rtx6000.sglang.*";
        let desired = "sie.work.default.rtx6000.sglang.*";
        assert!(any_filter_overlaps(stale_filter, &[], desired));
    }

    #[test]
    fn any_filter_overlaps_allows_different_bundles_in_same_pool() {
        // The queue-pool fix: default and sglang can share logical pool
        // `default` and machine profile `rtx6000` without overlapping.
        assert!(!any_filter_overlaps(
            "sie.work.default.rtx6000.default.*",
            &[],
            "sie.work.default.rtx6000.sglang.*"
        ));
    }

    #[test]
    fn any_filter_overlaps_ignores_other_pools() {
        // A `default_h100` consumer on a shared NATS account must not
        // be deleted just because we're deploying `default_l4`.
        assert!(!any_filter_overlaps(
            "sie.work.default.h100.default.*",
            &[],
            "sie.work.default.l4.default.*"
        ));
    }

    #[test]
    fn any_filter_overlaps_handles_filter_subjects_list() {
        // Newer NATS consumers can use the multi-filter
        // `filter_subjects` list (with `filter_subject` blank).
        let multi = vec![
            "sie.work.foo.h100.default.*".to_string(),
            "sie.work.default.l4.default.*".to_string(),
        ];
        assert!(any_filter_overlaps(
            "",
            &multi,
            "sie.work.default.l4.default.*"
        ));

        let multi_disjoint = vec![
            "sie.work.foo.h100.default.*".to_string(),
            "sie.work.eval-l4.l4.default.*".to_string(),
        ];
        assert!(!any_filter_overlaps(
            "",
            &multi_disjoint,
            "sie.work.default.l4.default.*"
        ));
    }

    #[test]
    fn consumer_filter_matches_desired_primary_filter() {
        assert!(consumer_filter_matches_desired(
            "sie.work.default.rtx6000.sglang.*.worker-0",
            &[],
            "sie.work.default.rtx6000.sglang.*.worker-0"
        ));
    }

    #[test]
    fn consumer_filter_matches_desired_single_multi_filter() {
        let filters = vec!["sie.work.default.rtx6000.sglang.*.worker-0".to_string()];
        assert!(consumer_filter_matches_desired(
            "",
            &filters,
            "sie.work.default.rtx6000.sglang.*.worker-0"
        ));
    }

    #[test]
    fn consumer_filter_matches_desired_rejects_legacy_same_name_filter() {
        assert!(!consumer_filter_matches_desired(
            "sie.work.*.default.worker-0",
            &[],
            "sie.work.default.rtx6000.sglang.*.worker-0"
        ));
    }

    #[test]
    fn consumer_filter_matches_desired_rejects_extra_multi_filter() {
        let filters = vec![
            "sie.work.default.rtx6000.sglang.*.worker-0".to_string(),
            "sie.work.default.rtx6000.default.*.worker-0".to_string(),
        ];
        assert!(!consumer_filter_matches_desired(
            "",
            &filters,
            "sie.work.default.rtx6000.sglang.*.worker-0"
        ));
    }

    #[test]
    fn any_filter_overlaps_treats_empty_primary_as_unset() {
        // Empty `filter_subject` with empty `filter_subjects` means
        // a consumer with no filter — JetStream actually rejects
        // creating that on a WorkQueue stream, so it shouldn't
        // appear, but if it ever does we don't want to spuriously
        // delete it.
        assert!(!any_filter_overlaps(
            "",
            &[],
            "sie.work.default.l4.default.*"
        ));
    }
}
