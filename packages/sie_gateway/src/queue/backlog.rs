//! Authoritative, bounded JetStream backlog snapshots for KEDA.
//!
//! Applications emit one semantic `sie.gateway.lane.queue.depth` gauge. This
//! module supplies its value from durable broker state; Prometheus spelling
//! and routing remain collector concerns. The reader is deliberately separate
//! from [`super::dispatch::WorkDispatcher`], whose boundary is the request
//! publish/cancel path rather than control-plane reconciliation.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use async_nats::jetstream;
use async_trait::async_trait;
use futures_util::{stream, StreamExt, TryStreamExt};

use crate::state::demand_tracker::{
    PhysicalLane, PhysicalLaneCatalog, MAX_CONFIGURED_PHYSICAL_LANES,
};

const DEFAULT_QUERY_CONCURRENCY: usize = 32;
const DEFAULT_SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(4);

/// One bounded reconciliation result. Successful lanes carry an authoritative
/// value (including explicit zero); failed lanes carry only diagnostic text.
/// The two maps are disjoint and their union is the frozen deployment catalog.
///
/// Keeping failures lane-scoped is load-bearing for KEDA: a broken consumer
/// must age out only its own freshness series, never suppress unrelated lane
/// values or the gateway's global demand/lease/floor snapshot.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LaneBacklogSnapshot {
    values: BTreeMap<PhysicalLane, u64>,
    failures: BTreeMap<PhysicalLane, String>,
}

impl LaneBacklogSnapshot {
    pub(crate) fn complete(values: BTreeMap<PhysicalLane, u64>) -> Self {
        Self {
            values,
            failures: BTreeMap::new(),
        }
    }

    fn from_outcomes(
        values: BTreeMap<PhysicalLane, u64>,
        failures: BTreeMap<PhysicalLane, String>,
    ) -> Self {
        debug_assert!(values.keys().all(|lane| !failures.contains_key(lane)));
        Self { values, failures }
    }

    pub(crate) fn get(&self, lane: &PhysicalLane) -> Option<&u64> {
        self.values.get(lane)
    }

    pub(crate) fn values(&self) -> &BTreeMap<PhysicalLane, u64> {
        &self.values
    }

    pub(crate) fn failures(&self) -> &BTreeMap<PhysicalLane, String> {
        &self.failures
    }

    pub(crate) fn len(&self) -> usize {
        self.values.len() + self.failures.len()
    }
}

impl FromIterator<(PhysicalLane, u64)> for LaneBacklogSnapshot {
    fn from_iter<T: IntoIterator<Item = (PhysicalLane, u64)>>(iter: T) -> Self {
        Self::complete(iter.into_iter().collect())
    }
}

impl<const N: usize> From<[(PhysicalLane, u64); N]> for LaneBacklogSnapshot {
    fn from(values: [(PhysicalLane, u64); N]) -> Self {
        values.into_iter().collect()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LaneBacklogError {
    #[error("invalid physical-lane consumer catalog: {0}")]
    InvalidCatalog(String),
    #[error("JetStream backlog query failed for {lane}: {detail}")]
    Query { lane: PhysicalLane, detail: String },
}

/// Read the current backlog for every deployment-owned physical lane.
/// Implementations return every configured lane exactly once, either as an
/// authoritative value (including explicit zero) or as a lane-scoped failure.
#[async_trait]
pub trait LaneBacklogSource: Send + Sync {
    async fn snapshot(&self) -> Result<LaneBacklogSnapshot, LaneBacklogError>;
}

#[derive(Clone, Debug)]
struct LaneConsumerTarget {
    lane: PhysicalLane,
    stream_name: String,
    consumer_name: String,
    subject_filter: String,
}

/// JetStream-backed lane reader. Its target set is frozen from the same typed
/// catalog that Helm/KEDA render, so neither request data nor broker-created
/// names can manufacture metric labels.
pub struct JetStreamLaneBacklogSource {
    context: jetstream::Context,
    targets: Arc<[LaneConsumerTarget]>,
    query_concurrency: usize,
    snapshot_timeout: Duration,
}

impl JetStreamLaneBacklogSource {
    pub fn try_new(
        context: jetstream::Context,
        catalog: &PhysicalLaneCatalog,
    ) -> Result<Self, LaneBacklogError> {
        Ok(Self {
            context,
            targets: build_targets(catalog)?.into(),
            query_concurrency: DEFAULT_QUERY_CONCURRENCY,
            snapshot_timeout: DEFAULT_SNAPSHOT_TIMEOUT,
        })
    }

    async fn query_target(
        context: jetstream::Context,
        target: LaneConsumerTarget,
    ) -> Result<(PhysicalLane, u64), LaneBacklogError> {
        let stream = context
            .get_stream_no_info(&target.stream_name)
            .await
            .map_err(|error| query_error(&target.lane, error))?;

        let outstanding = match stream.consumer_info(&target.consumer_name).await {
            Ok(info) => {
                if info.stream_name != target.stream_name
                    || info.name != target.consumer_name
                    || !has_exact_filter(&info.config, &target.subject_filter)
                {
                    return Err(LaneBacklogError::Query {
                        lane: target.lane,
                        detail: format!(
                            "consumer identity/filter drift: stream={} consumer={} primary_filter={} multi_filters={:?}",
                            info.stream_name,
                            info.name,
                            info.config.filter_subject,
                            info.config.filter_subjects,
                        ),
                    });
                }
                checked_outstanding(info.num_pending, info.num_ack_pending).ok_or_else(|| {
                    LaneBacklogError::Query {
                        lane: target.lane.clone(),
                        detail: "pending + ack-pending counter overflow".to_string(),
                    }
                })?
            }
            Err(error)
                if matches!(
                    error.kind(),
                    jetstream::context::ConsumerInfoErrorKind::StreamNotFound
                ) =>
            {
                // No stream means no durable work has ever been published for
                // this pool. This is the only missing-resource state that is
                // unconditionally safe to report as zero.
                0
            }
            Err(error)
                if matches!(
                    error.kind(),
                    jetstream::context::ConsumerInfoErrorKind::NotFound
                ) =>
            {
                // Another lane's consumer can keep a pool stream publishable
                // while this lane has no durable yet. Count retained messages
                // for this exact subject rather than silently returning zero.
                exact_subject_backlog(&stream, &target).await?
            }
            Err(error) => return Err(query_error(&target.lane, error)),
        };

        Ok((target.lane, outstanding))
    }
}

#[async_trait]
impl LaneBacklogSource for JetStreamLaneBacklogSource {
    async fn snapshot(&self) -> Result<LaneBacklogSnapshot, LaneBacklogError> {
        let context = self.context.clone();
        let concurrency = self.query_concurrency.max(1);
        let mut query = stream::iter(self.targets.iter().cloned())
            .map(move |target| Self::query_target(context.clone(), target))
            .buffer_unordered(concurrency);
        let deadline = tokio::time::Instant::now() + self.snapshot_timeout;
        let mut values = BTreeMap::new();
        let mut failures = BTreeMap::new();

        loop {
            match tokio::time::timeout_at(deadline, query.next()).await {
                Ok(Some(Ok((lane, value)))) => {
                    values.insert(lane, value);
                }
                Ok(Some(Err(LaneBacklogError::Query { lane, detail }))) => {
                    failures.insert(lane, detail);
                }
                Ok(Some(Err(error))) => return Err(error),
                Ok(None) => break,
                Err(_) => {
                    let detail = format!(
                        "backlog query did not complete within {} ms",
                        self.snapshot_timeout.as_millis()
                    );
                    for target in self.targets.iter() {
                        if !values.contains_key(&target.lane)
                            && !failures.contains_key(&target.lane)
                        {
                            failures.insert(target.lane.clone(), detail.clone());
                        }
                    }
                    break;
                }
            }
        }

        Ok(LaneBacklogSnapshot::from_outcomes(values, failures))
    }
}

fn build_targets(
    catalog: &PhysicalLaneCatalog,
) -> Result<Vec<LaneConsumerTarget>, LaneBacklogError> {
    if catalog.len() > MAX_CONFIGURED_PHYSICAL_LANES {
        return Err(LaneBacklogError::InvalidCatalog(format!(
            "lane count exceeds {MAX_CONFIGURED_PHYSICAL_LANES}"
        )));
    }

    let mut consumer_owners: HashMap<(String, String), PhysicalLane> = HashMap::new();
    let mut targets = Vec::with_capacity(catalog.len());
    for lane in catalog.lanes() {
        let stream_name = format!("WORK_POOL_{}", lane.pool());
        let consumer_name = lane_consumer_name(&lane);
        if let Some(existing) =
            consumer_owners.insert((stream_name.clone(), consumer_name.clone()), lane.clone())
        {
            return Err(LaneBacklogError::InvalidCatalog(format!(
                "{existing} and {lane} map to the same durable consumer {consumer_name} in {stream_name}"
            )));
        }
        targets.push(LaneConsumerTarget {
            subject_filter: format!(
                "sie.work.{}.{}.{}.*",
                lane.pool(),
                lane.machine_profile(),
                lane.bundle()
            ),
            lane,
            stream_name,
            consumer_name,
        });
    }
    Ok(targets)
}

fn lane_consumer_name(lane: &PhysicalLane) -> String {
    // Keep this in lockstep with WorkerConfig::consumer_name and
    // sie_sdk.queue_types.work_consumer_name. Collision validation above is
    // mandatory because `_` is legal in all three physical-lane tokens.
    format!(
        "{}_{}_{}",
        lane.pool(),
        lane.machine_profile(),
        lane.bundle()
    )
}

fn has_exact_filter(config: &jetstream::consumer::Config, expected: &str) -> bool {
    let mut filters: Vec<&str> = config.filter_subjects.iter().map(String::as_str).collect();
    if !config.filter_subject.is_empty() {
        filters.push(&config.filter_subject);
    }
    filters.sort_unstable();
    filters.dedup();
    filters == [expected]
}

fn checked_outstanding(num_pending: u64, num_ack_pending: usize) -> Option<u64> {
    num_pending.checked_add(u64::try_from(num_ack_pending).ok()?)
}

async fn exact_subject_backlog<I>(
    stream: &jetstream::stream::Stream<I>,
    target: &LaneConsumerTarget,
) -> Result<u64, LaneBacklogError> {
    let mut subjects = stream
        .info_with_subjects(&target.subject_filter)
        .await
        .map_err(|error| query_error(&target.lane, error))?;
    let mut outstanding = 0_u64;
    while let Some((_subject, count)) = subjects
        .try_next()
        .await
        .map_err(|error| query_error(&target.lane, error))?
    {
        outstanding = outstanding
            .checked_add(u64::try_from(count).map_err(|_| LaneBacklogError::Query {
                lane: target.lane.clone(),
                detail: "exact-subject message count does not fit u64".to_string(),
            })?)
            .ok_or_else(|| LaneBacklogError::Query {
                lane: target.lane.clone(),
                detail: "exact-subject message count overflow".to_string(),
            })?;
    }
    Ok(outstanding)
}

fn query_error(lane: &PhysicalLane, error: impl std::fmt::Display) -> LaneBacklogError {
    LaneBacklogError::Query {
        lane: lane.clone(),
        detail: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog(lanes: &[(&str, &str, &str)]) -> PhysicalLaneCatalog {
        PhysicalLaneCatalog::try_new(
            lanes.iter().map(|(pool, profile, bundle)| {
                PhysicalLane::try_new(pool, profile, bundle).unwrap()
            }),
        )
        .unwrap()
    }

    async fn assert_backlog_eventually(
        source: &JetStreamLaneBacklogSource,
        lane: &PhysicalLane,
        expected: u64,
    ) {
        let mut last_observed = None;
        let converged = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let observed = source
                    .snapshot()
                    .await
                    .expect("read lane backlog")
                    .get(lane)
                    .copied();
                last_observed = observed;
                if observed == Some(expected) {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(
            converged.is_ok(),
            "lane backlog did not converge to {expected}; last observed {last_observed:?}"
        );
    }

    #[test]
    fn outstanding_counts_pending_and_unacked_only() {
        assert_eq!(checked_outstanding(7, 3), Some(10));
        assert_eq!(checked_outstanding(u64::MAX, 1), None);
    }

    #[test]
    fn consumer_filter_must_be_one_exact_lane() {
        let expected = "sie.work.default.l4.default.*";
        let mut config = jetstream::consumer::Config {
            filter_subject: expected.to_string(),
            ..Default::default()
        };
        assert!(has_exact_filter(&config, expected));

        config.filter_subjects = vec!["sie.work.default.h100.default.*".to_string()];
        assert!(!has_exact_filter(&config, expected));
        config.filter_subject.clear();
        assert!(!has_exact_filter(&config, expected));
    }

    #[test]
    fn ambiguous_durable_names_fail_catalog_construction() {
        let catalog = catalog(&[("shared", "a_b", "c"), ("shared", "a", "b_c")]);
        let error = build_targets(&catalog).unwrap_err().to_string();
        assert!(error.contains("same durable consumer shared_a_b_c"));
    }

    #[test]
    fn targets_preserve_exact_catalog_labels_and_zero_cardinality_bound() {
        let catalog = catalog(&[("default", "l4", "default"), ("shared", "h100", "sglang")]);
        let targets = build_targets(&catalog).unwrap();
        assert_eq!(targets.len(), 2);
        assert_eq!(
            targets
                .iter()
                .map(|target| target.subject_filter.as_str())
                .collect::<Vec<_>>(),
            vec![
                "sie.work.default.l4.default.*",
                "sie.work.shared.h100.sglang.*",
            ]
        );
    }

    /// Live JetStream contract test. Hermetic CI skips when `NATS_URL` is not
    /// configured; the observability/KEDA integration job supplies a broker.
    #[tokio::test]
    async fn real_jetstream_backlog_survives_delivery_until_ack() {
        let Ok(url) = std::env::var("NATS_URL") else {
            eprintln!("skipping: NATS_URL not set");
            return;
        };
        let client =
            match tokio::time::timeout(Duration::from_secs(2), async_nats::connect(&url)).await {
                Ok(Ok(client)) => client,
                _ => {
                    eprintln!("skipping: could not connect to NATS at {url}");
                    return;
                }
            };
        let context = jetstream::new(client);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let pool = format!("keda{nonce}");
        let catalog = catalog(&[(&pool, "l4", "default")]);
        let source = JetStreamLaneBacklogSource::try_new(context.clone(), &catalog).unwrap();
        let lane = catalog.lanes().into_iter().next().unwrap();

        assert_eq!(source.snapshot().await.unwrap().get(&lane), Some(&0));

        let stream_name = format!("WORK_POOL_{pool}");
        let subject_filter = format!("sie.work.{pool}.l4.default.*");
        let stream = context
            .get_or_create_stream(jetstream::stream::Config {
                name: stream_name.clone(),
                subjects: vec![format!("sie.work.{pool}.*.*.*")],
                retention: jetstream::stream::RetentionPolicy::WorkQueue,
                storage: jetstream::stream::StorageType::Memory,
                max_age: Duration::from_secs(60),
                ..Default::default()
            })
            .await
            .expect("create test work stream");

        // A pool stream may exist before this exact lane's worker has created
        // its durable. Stored work must not be misreported as zero.
        context
            .publish(format!("sie.work.{pool}.l4.default.model-a"), "one".into())
            .await
            .expect("publish first item")
            .await
            .expect("ack first publish");
        assert_backlog_eventually(&source, &lane, 1).await;

        let consumer_name = lane_consumer_name(&lane);
        let consumer = stream
            .create_consumer(jetstream::consumer::pull::Config {
                durable_name: Some(consumer_name),
                filter_subject: subject_filter,
                ack_policy: jetstream::consumer::AckPolicy::Explicit,
                ack_wait: Duration::from_secs(30),
                ..Default::default()
            })
            .await
            .expect("create exact lane consumer");
        for (model, payload) in [("model-b", "two"), ("model-c", "three")] {
            context
                .publish(
                    format!("sie.work.{pool}.l4.default.{model}"),
                    payload.into(),
                )
                .await
                .expect("publish item")
                .await
                .expect("ack publish");
        }
        assert_backlog_eventually(&source, &lane, 3).await;

        let mut messages = consumer.messages().await.expect("open pull stream");
        let first = tokio::time::timeout(Duration::from_secs(2), messages.next())
            .await
            .expect("first delivery timeout")
            .expect("first delivery ended")
            .expect("first delivery error");
        let second = tokio::time::timeout(Duration::from_secs(2), messages.next())
            .await
            .expect("second delivery timeout")
            .expect("second delivery ended")
            .expect("second delivery error");

        // One pending + two delivered-but-unacknowledged remains three.
        assert_backlog_eventually(&source, &lane, 3).await;
        first.ack().await.expect("ack first delivery");
        assert_backlog_eventually(&source, &lane, 2).await;
        second.ack().await.expect("ack second delivery");

        let third = tokio::time::timeout(Duration::from_secs(2), messages.next())
            .await
            .expect("third delivery timeout")
            .expect("third delivery ended")
            .expect("third delivery error");
        third.ack().await.expect("ack third delivery");
        assert_backlog_eventually(&source, &lane, 0).await;

        context
            .delete_stream(stream_name)
            .await
            .expect("delete test work stream");
    }

    #[tokio::test]
    async fn real_jetstream_consumer_drift_is_isolated_to_one_lane() {
        let Ok(url) = std::env::var("NATS_URL") else {
            eprintln!("skipping: NATS_URL not set");
            return;
        };
        let client =
            match tokio::time::timeout(Duration::from_secs(2), async_nats::connect(&url)).await {
                Ok(Ok(client)) => client,
                _ => {
                    eprintln!("skipping: could not connect to NATS at {url}");
                    return;
                }
            };
        let context = jetstream::new(client);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let healthy_pool = format!("healthy{nonce}");
        let corrupt_pool = format!("corrupt{nonce}");
        let catalog = catalog(&[
            (&healthy_pool, "l4", "default"),
            (&corrupt_pool, "h100", "default"),
        ]);
        let source = JetStreamLaneBacklogSource::try_new(context.clone(), &catalog).unwrap();
        let healthy = catalog.resolve(&healthy_pool, "l4", "default").unwrap();
        let corrupt = catalog.resolve(&corrupt_pool, "h100", "default").unwrap();
        let stream_name = format!("WORK_POOL_{corrupt_pool}");
        let stream = context
            .create_stream(jetstream::stream::Config {
                name: stream_name.clone(),
                subjects: vec![format!("sie.work.{corrupt_pool}.*.*.*")],
                retention: jetstream::stream::RetentionPolicy::WorkQueue,
                storage: jetstream::stream::StorageType::Memory,
                max_age: Duration::from_secs(60),
                ..Default::default()
            })
            .await
            .expect("create corrupt-lane stream");
        stream
            .create_consumer(jetstream::consumer::pull::Config {
                durable_name: Some(lane_consumer_name(&corrupt)),
                filter_subject: format!("sie.work.{corrupt_pool}.h100.other.*"),
                ..Default::default()
            })
            .await
            .expect("create drifted consumer");

        let snapshot = source.snapshot().await.expect("partial snapshot");
        assert_eq!(snapshot.get(&healthy), Some(&0));
        assert!(!snapshot.values().contains_key(&corrupt));
        assert!(snapshot
            .failures()
            .get(&corrupt)
            .is_some_and(|detail| detail.contains("identity/filter drift")));

        context
            .delete_stream(stream_name)
            .await
            .expect("delete corrupt-lane stream");
    }

    /// Opt-in live-broker benchmark for the enabled KEDA reconciliation cost
    /// at one, a representative 16, and the hard-limit 1,024 physical lanes.
    /// It additionally drives the Helm-enforced maximum of ten gateway
    /// replicas through concurrent 1,024-lane snapshots, and reports one
    /// 64-item durable publish-ACK batch so both the HA read load and the
    /// handoff from pending demand to broker backlog have explicit baselines.
    /// Every reported value is the median of three independently warmed
    /// samples; broker RTT remains intentionally included.
    ///
    /// Run against a disposable JetStream endpoint with:
    /// `SIE_RUN_TELEMETRY_BENCHMARK=1 NATS_URL=nats://127.0.0.1:4222 mise exec -- cargo test --manifest-path packages/sie_gateway/Cargo.toml --release --lib jetstream_lane_snapshot_live_microbenchmark -- --ignored --nocapture --test-threads=1`
    #[tokio::test]
    #[ignore = "opt-in release benchmark against a disposable JetStream broker"]
    async fn jetstream_lane_snapshot_live_microbenchmark() {
        const SAMPLES: usize = 3;

        assert_eq!(
            std::env::var("SIE_RUN_TELEMETRY_BENCHMARK").as_deref(),
            Ok("1"),
            "opt in with SIE_RUN_TELEMETRY_BENCHMARK=1"
        );
        let url = std::env::var("NATS_URL").expect("NATS_URL is required");
        let client = async_nats::connect(&url)
            .await
            .expect("connect benchmark NATS");
        let context = jetstream::new(client);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let pool = format!("kedabench{nonce}");
        let lanes: Vec<_> = (0..MAX_CONFIGURED_PHYSICAL_LANES)
            .map(|index| PhysicalLane::try_new(&pool, &format!("p{index:04}"), "default").unwrap())
            .collect();
        let stream_name = format!("WORK_POOL_{pool}");
        let work_stream = context
            .create_stream(jetstream::stream::Config {
                name: stream_name.clone(),
                subjects: vec![format!("sie.work.{pool}.*.*.*")],
                retention: jetstream::stream::RetentionPolicy::WorkQueue,
                storage: jetstream::stream::StorageType::Memory,
                max_age: Duration::from_secs(300),
                ..Default::default()
            })
            .await
            .expect("create benchmark stream");

        stream::iter(lanes.iter().cloned())
            .map(|lane| {
                let stream = work_stream.clone();
                async move {
                    stream
                        .create_consumer(jetstream::consumer::pull::Config {
                            durable_name: Some(lane_consumer_name(&lane)),
                            filter_subject: format!(
                                "sie.work.{}.{}.{}.*",
                                lane.pool(),
                                lane.machine_profile(),
                                lane.bundle()
                            ),
                            ack_policy: jetstream::consumer::AckPolicy::Explicit,
                            ..Default::default()
                        })
                        .await
                }
            })
            .buffer_unordered(DEFAULT_QUERY_CONCURRENCY)
            .try_collect::<Vec<_>>()
            .await
            .expect("create benchmark lane consumers");

        for lane_count in [1, 16, MAX_CONFIGURED_PHYSICAL_LANES] {
            let catalog =
                PhysicalLaneCatalog::try_new(lanes[..lane_count].iter().cloned()).unwrap();
            let source = JetStreamLaneBacklogSource::try_new(context.clone(), &catalog).unwrap();
            let mut samples = [0.0; SAMPLES];
            for sample in &mut samples {
                assert_eq!(source.snapshot().await.unwrap().len(), lane_count);
                let started = std::time::Instant::now();
                assert_eq!(source.snapshot().await.unwrap().len(), lane_count);
                *sample = started.elapsed().as_secs_f64() * 1_000.0;
            }
            let median_ms = crate::observability::metrics::telemetry_benchmark_median(samples);
            println!(
                "gateway_jetstream_lane_snapshot lanes={lane_count} concurrency={} samples_ms={samples:?} median_ms={median_ms:.3} median_us_per_lane={:.3}",
                DEFAULT_QUERY_CONCURRENCY,
                median_ms * 1_000.0 / lane_count as f64,
            );
        }

        const MAX_GATEWAY_REPLICAS: usize = 10;
        let max_catalog =
            PhysicalLaneCatalog::try_new(lanes.iter().cloned()).expect("build max lane catalog");
        let replica_sources: Vec<_> = (0..MAX_GATEWAY_REPLICAS)
            .map(|_| {
                JetStreamLaneBacklogSource::try_new(context.clone(), &max_catalog)
                    .expect("build replica backlog source")
            })
            .collect();

        async fn read_replica_snapshots(sources: &[JetStreamLaneBacklogSource]) {
            let snapshots =
                futures_util::future::try_join_all(sources.iter().map(LaneBacklogSource::snapshot))
                    .await
                    .expect("read concurrent replica snapshots");
            assert!(snapshots
                .iter()
                .all(|snapshot| snapshot.len() == MAX_CONFIGURED_PHYSICAL_LANES));
        }
        let mut ha_samples = [0.0; SAMPLES];
        for sample in &mut ha_samples {
            read_replica_snapshots(&replica_sources).await;
            let started = std::time::Instant::now();
            read_replica_snapshots(&replica_sources).await;
            *sample = started.elapsed().as_secs_f64() * 1_000.0;
        }
        let ha_median_ms = crate::observability::metrics::telemetry_benchmark_median(ha_samples);
        println!(
            "gateway_jetstream_ha_snapshot replicas={MAX_GATEWAY_REPLICAS} lanes_per_replica={} lookups_per_interval={} samples_ms={ha_samples:?} median_ms={ha_median_ms:.3}",
            MAX_CONFIGURED_PHYSICAL_LANES,
            MAX_GATEWAY_REPLICAS * MAX_CONFIGURED_PHYSICAL_LANES,
        );

        const ACK_BATCH: usize = 64;
        async fn publish_ack_batch(context: &jetstream::Context, subject: &str) {
            let mut ack_futures = Vec::with_capacity(ACK_BATCH);
            for index in 0..ACK_BATCH {
                ack_futures.push(
                    context
                        .publish(subject.to_string(), format!("benchmark-{index}").into())
                        .await
                        .expect("enqueue benchmark publish"),
                );
            }
            futures_util::future::try_join_all(
                ack_futures.into_iter().map(|ack| async move { ack.await }),
            )
            .await
            .expect("await benchmark publish ACKs");
        }
        let subject = format!("sie.work.{pool}.p0000.default.model");
        let mut ack_samples = [0.0; SAMPLES];
        for sample in &mut ack_samples {
            publish_ack_batch(&context, &subject).await;
            let started = std::time::Instant::now();
            publish_ack_batch(&context, &subject).await;
            *sample = started.elapsed().as_secs_f64() * 1_000.0;
        }
        let ack_median_ms = crate::observability::metrics::telemetry_benchmark_median(ack_samples);
        println!(
            "gateway_jetstream_publish_ack batch={ACK_BATCH} samples_ms={ack_samples:?} median_ms={ack_median_ms:.3} median_us_per_item_derived={:.3}",
            ack_median_ms * 1_000.0 / ACK_BATCH as f64,
        );

        context
            .delete_stream(stream_name)
            .await
            .expect("delete benchmark stream");
    }
}
