//! Shared demand accounting and lease-fenced threshold decisions.
//!
//! This prerequisite does not activate routing. Only validated, policy-eligible
//! requests may call `record_request`; neither inference payloads nor caller
//! identifiers enter the broker. A sampler measures counter deltas with its
//! monotonic clock, and consumers require its current broker lease revision.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

use async_nats::jetstream::{self, kv, stream};
use bytes::Bytes;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::types::model::{RoutingConfig, RoutingPolicy};

pub const MAX_THRESHOLD_MODELS: usize = 256;
pub const THRESHOLD_SAMPLE_INTERVAL: Duration = Duration::from_secs(1);
const MAX_SAMPLE_GAP: Duration = Duration::from_millis(2500);
const LEASE_TTL: Duration = Duration::from_secs(5);
const COUNTER_TTL: Duration = Duration::from_secs(86400);
const IO_TIMEOUT: Duration = Duration::from_secs(2);
const MAX_VALUE_BYTES: usize = 2048;
const MAX_BUCKET_BYTES: i64 = 1 << 20;
const CAS_ATTEMPTS: usize = 16;
const LEASE_KEY: &str = "sampler";

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ThresholdError {
    #[error("threshold configuration is invalid or exceeds its bounds")]
    Configuration,
    #[error("threshold broker state is unavailable")]
    Unavailable,
    #[error("threshold broker state is malformed or untrusted")]
    Untrusted,
    #[error("threshold configuration generation does not match")]
    Generation,
    #[error("threshold counter contention exceeded its bound")]
    Contended,
}

/// Exact snapshot authority, including the policy and local/remote execution
/// contracts. The generation must come from the authoritative config epoch.
#[derive(Clone, Debug)]
pub struct ThresholdTarget {
    key: String,
    generation: u64,
    contract: String,
    wake_above: f64,
    sleep_below: f64,
    window: Duration,
    cooldown: Duration,
}

impl ThresholdTarget {
    pub fn new(
        model: &str,
        generation: u64,
        execution_fingerprint: &str,
        routing: &RoutingConfig,
    ) -> Result<Self, ThresholdError> {
        if generation == 0
            || model.is_empty()
            || model.len() > 256
            || model.contains(':')
            || execution_fingerprint.len() != 64
            || !execution_fingerprint
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            || routing.policy != RoutingPolicy::Threshold
            || routing.validate().is_err()
        {
            return Err(ThresholdError::Configuration);
        }
        let window = routing.window_s.ok_or(ThresholdError::Configuration)?;
        let cooldown = routing.cooldown_s.ok_or(ThresholdError::Configuration)?;
        if window > 86400.0 || cooldown > 86400.0 {
            return Err(ThresholdError::Configuration);
        }
        let policy = serde_json::to_vec(routing).map_err(|_| ThresholdError::Configuration)?;
        let mut contract = Sha256::new();
        contract.update(execution_fingerprint.as_bytes());
        contract.update(policy);
        Ok(Self {
            key: digest_key(model),
            generation,
            contract: hex_digest(&contract.finalize()),
            wake_above: routing.wake_above.ok_or(ThresholdError::Configuration)?,
            sleep_below: routing.sleep_below.ok_or(ThresholdError::Configuration)?,
            window: Duration::from_secs_f64(window.max(1.0)),
            cooldown: Duration::from_secs_f64(cooldown.max(1.0)),
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ThresholdDecision {
    Undetermined,
    Remote,
    WakeLocal,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Counter {
    generation: u64,
    contract: String,
    total: u64,
    incarnation: String,
    #[serde(default)]
    continuity: u64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Decision {
    generation: u64,
    contract: String,
    owner: String,
    term: String,
    decision: ThresholdDecision,
}

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Lease {
    owner: String,
    generation: u64,
    contract: String,
    term: String,
}

/// One process owns one sampler token. Replacement/expired ownership loses all
/// timer evidence; the broker's TTL decides expiry, never a gateway wall clock.
pub struct ThresholdSampler {
    owner: String,
    lease_revision: u64,
    term: Option<String>,
    models: HashMap<String, SampleState>,
}

impl Default for ThresholdSampler {
    fn default() -> Self {
        Self {
            owner: Uuid::new_v4().to_string(),
            lease_revision: 0,
            term: None,
            models: HashMap::new(),
        }
    }
}

struct SamplingAttempt<'a> {
    sampler: &'a mut ThresholdSampler,
    succeeded: bool,
}

impl Drop for SamplingAttempt<'_> {
    fn drop(&mut self) {
        if !self.succeeded {
            self.sampler.lease_revision = 0;
            self.sampler.term = None;
            self.sampler.models.clear();
        }
    }
}

struct SampleState {
    contract: String,
    generation: u64,
    total: u64,
    continuity: u64,
    incarnation: Option<String>,
    sampled_at: Instant,
    decision: ThresholdDecision,
    crossing_since: Option<Instant>,
    crossing_target: Option<ThresholdDecision>,
}

impl SampleState {
    fn baseline(target: &ThresholdTarget, total: u64, now: Instant) -> Self {
        Self {
            contract: target.contract.clone(),
            generation: target.generation,
            total,
            continuity: 0,
            incarnation: None,
            sampled_at: now,
            decision: ThresholdDecision::Undetermined,
            crossing_since: None,
            crossing_target: None,
        }
    }

    fn sample(&mut self, target: &ThresholdTarget, total: u64, now: Instant) -> ThresholdDecision {
        let elapsed = now.saturating_duration_since(self.sampled_at);
        if self.contract != target.contract
            || self.generation != target.generation
            || total < self.total
            || elapsed > MAX_SAMPLE_GAP
        {
            *self = Self::baseline(target, total, now);
            return self.decision;
        }
        // Calling faster than the declared cadence must not discard counts or
        // manufacture consecutive evidence from the same sampling interval.
        if elapsed < THRESHOLD_SAMPLE_INTERVAL {
            return self.decision;
        }
        let rate = (total - self.total) as f64 / elapsed.as_secs_f64();
        self.total = total;
        self.sampled_at = now;
        let crossing = match self.decision {
            ThresholdDecision::Remote if rate > target.wake_above => {
                Some(ThresholdDecision::WakeLocal)
            }
            ThresholdDecision::WakeLocal if rate < target.sleep_below => {
                Some(ThresholdDecision::Remote)
            }
            ThresholdDecision::Undetermined if rate > target.wake_above => {
                Some(ThresholdDecision::WakeLocal)
            }
            ThresholdDecision::Undetermined if rate < target.sleep_below => {
                Some(ThresholdDecision::Remote)
            }
            _ => None,
        };
        if crossing != self.crossing_target {
            self.crossing_since = None;
            self.crossing_target = crossing;
        }
        if let Some(next) = crossing {
            let required = if next == ThresholdDecision::WakeLocal {
                target.window
            } else {
                target.cooldown
            };
            let since = *self.crossing_since.get_or_insert(now);
            if now.saturating_duration_since(since) >= required {
                self.decision = next;
                self.crossing_since = None;
                self.crossing_target = None;
            }
        }
        self.decision
    }
}

struct TargetState {
    target: ThresholdTarget,
    pending: AtomicU64,
    interrupted: AtomicBool,
    cached: RwLock<Option<CachedDecision>>,
}

struct DrainedDemand<'a> {
    state: &'a TargetState,
    delta: u64,
    interrupted: bool,
    committed: bool,
}

impl Drop for DrainedDemand<'_> {
    fn drop(&mut self) {
        if !self.committed {
            // Includes cancellation while a publish acknowledgement is in flight.
            self.state.interrupted.store(true, Ordering::Relaxed);
        }
    }
}

#[derive(Clone, Copy)]
struct CachedDecision {
    decision: ThresholdDecision,
    expires_at: Instant,
}

struct LeaseObservation {
    revision: u64,
    lease: Lease,
    read_started: Instant,
    expires_at: Option<Instant>,
}

struct TickState {
    last_started: Instant,
    observed: Option<LeaseObservation>,
}

/// Control-state connection for a separate NATS endpoint. Inference workers
/// and configuration publishers must not have credentials for that endpoint.
/// Do not pass the inference queue's JetStream context: subject ACLs cannot
/// establish control-record provenance inside a shared account.
pub struct ThresholdCoordinator {
    counters: kv::Store,
    decisions: kv::Store,
    leases: kv::Store,
    targets: HashMap<String, TargetState>,
    generation: u64,
    contract: String,
    tick: tokio::sync::Mutex<TickState>,
}

impl ThresholdCoordinator {
    pub async fn connect(
        context: &jetstream::Context,
        replicas: usize,
        targets: Vec<ThresholdTarget>,
    ) -> Result<Self, ThresholdError> {
        if targets.is_empty() || targets.len() > MAX_THRESHOLD_MODELS || !matches!(replicas, 1 | 3)
        {
            return Err(ThresholdError::Configuration);
        }
        let generation = targets[0].generation;
        if targets.iter().any(|target| target.generation != generation) {
            return Err(ThresholdError::Configuration);
        }
        let mut contracts = targets
            .iter()
            .map(|target| (&target.key, &target.contract))
            .collect::<Vec<_>>();
        contracts.sort_unstable();
        let mut digest = Sha256::new();
        for (key, contract) in contracts {
            digest.update(key.as_bytes());
            digest.update(contract.as_bytes());
        }
        let contract = hex_digest(&digest.finalize());
        let count = targets.len();
        let targets = targets
            .into_iter()
            .map(|target| {
                (
                    target.key.clone(),
                    TargetState {
                        target,
                        pending: AtomicU64::new(0),
                        interrupted: AtomicBool::new(false),
                        cached: RwLock::new(None),
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        if targets.len() != count {
            return Err(ThresholdError::Configuration);
        }
        timeout(async {
            Ok(Self {
                counters: bucket(context, "SIE_THRESHOLD_COUNTS", COUNTER_TTL, replicas).await?,
                decisions: bucket(context, "SIE_THRESHOLD_DECISIONS", LEASE_TTL * 2, replicas)
                    .await?,
                leases: bucket(context, "SIE_THRESHOLD_LEASE", LEASE_TTL, replicas).await?,
                targets,
                generation,
                contract,
                tick: tokio::sync::Mutex::new(TickState {
                    last_started: Instant::now(),
                    observed: None,
                }),
            })
        })
        .await
    }

    /// Count once after shared caller validation, before profile recursion.
    /// Request handling never waits for the control broker.
    pub fn record_request(&self, model: &str) -> Result<(), ThresholdError> {
        let state = self
            .targets
            .get(&digest_key(model))
            .ok_or(ThresholdError::Configuration)?;
        state
            .pending
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |total| {
                total.checked_add(1)
            })
            .map_err(|_| {
                state.interrupted.store(true, Ordering::Relaxed);
                ThresholdError::Untrusted
            })?;
        Ok(())
    }

    async fn counter_total(
        &self,
        target: &ThresholdTarget,
        delta: u64,
        interrupted: bool,
    ) -> Result<Counter, ThresholdError> {
        for _ in 0..CAS_ATTEMPTS {
            let (revision, existing) = read::<Counter>(&self.counters, &target.key).await?;
            let (mut counter, current) = match existing {
                Some(counter) => {
                    if counter.generation > target.generation
                        || (counter.generation == target.generation
                            && counter.contract != target.contract)
                    {
                        return Err(ThresholdError::Generation);
                    }
                    let current = counter.generation == target.generation;
                    let counter = if current {
                        counter
                    } else {
                        Counter {
                            generation: target.generation,
                            contract: target.contract.clone(),
                            total: 0,
                            incarnation: Uuid::new_v4().to_string(),
                            continuity: 0,
                        }
                    };
                    (counter, current)
                }
                None => (
                    Counter {
                        generation: target.generation,
                        contract: target.contract.clone(),
                        total: 0,
                        incarnation: Uuid::new_v4().to_string(),
                        continuity: 0,
                    },
                    false,
                ),
            };
            if Uuid::parse_str(&counter.incarnation).is_err() {
                return Err(ThresholdError::Untrusted);
            }
            if delta == 0 && !interrupted && current {
                return Ok(counter);
            }
            counter.total = counter
                .total
                .checked_add(delta)
                .ok_or(ThresholdError::Untrusted)?;
            if interrupted {
                counter.continuity = counter
                    .continuity
                    .checked_add(1)
                    .ok_or(ThresholdError::Untrusted)?;
            }
            match self
                .counters
                .update(&target.key, encode(&counter)?, revision)
                .await
            {
                Ok(_) => return Ok(counter),
                Err(error) if error.kind() == kv::UpdateErrorKind::WrongLastRevision => {}
                // An uncertain acknowledgement may follow a successful commit.
                // Never restore or retry its already-drained demand delta.
                Err(_) => return Err(ThresholdError::Unavailable),
            }
            tokio::task::yield_now().await;
        }
        Err(ThresholdError::Contended)
    }

    /// Every replica calls this once per second, including standby gateways.
    /// It flushes local demand before election and refreshes local decisions.
    pub async fn sample(&self, sampler: &mut ThresholdSampler) -> Result<(), ThresholdError> {
        let mut attempt = SamplingAttempt {
            sampler,
            succeeded: false,
        };
        let sampler = &mut *attempt.sampler;
        let mut standby_refreshed = false;
        let result = timeout(async {
            let mut tick = self.tick.lock().await;
            let now = Instant::now();
            let gap = now.saturating_duration_since(tick.last_started) > MAX_SAMPLE_GAP;
            tick.last_started = now;
            // Drain atomically before I/O. Requests racing this swap stay in the
            // next batch; a timed-out batch is never restored or counted twice.
            let deltas = self
                .targets
                .values()
                .map(|state| {
                    let delta = state.pending.swap(0, Ordering::Relaxed);
                    let interrupted = state.interrupted.swap(false, Ordering::Relaxed) || gap;
                    DrainedDemand {
                        state,
                        delta: if gap { 0 } else { delta },
                        interrupted,
                        committed: false,
                    }
                })
                .collect::<Vec<_>>();
            for mut demand in deltas {
                self.counter_total(&demand.state.target, demand.delta, demand.interrupted)
                    .await?;
                demand.committed = true;
            }
            let (previous_revision, previous) = read::<Lease>(&self.leases, LEASE_KEY).await?;
            let mut renewal_started = None;
            let term = match previous {
                Some(lease) => {
                    validate_lease(&lease)?;
                    if lease.generation > self.generation
                        || (lease.generation == self.generation && lease.contract != self.contract)
                    {
                        return Err(ThresholdError::Generation);
                    }
                    if lease.generation < self.generation {
                        sampler.models.clear();
                        Some(Uuid::new_v4().to_string())
                    } else if lease.owner == sampler.owner
                        && previous_revision == sampler.lease_revision
                        && sampler.term.as_deref() == Some(lease.term.as_str())
                    {
                        Some(lease.term)
                    } else {
                        None
                    }
                }
                None => {
                    sampler.models.clear();
                    Some(Uuid::new_v4().to_string())
                }
            };
            if let Some(term) = term {
                let started = Instant::now();
                sampler.lease_revision = self
                    .leases
                    .update(
                        LEASE_KEY,
                        encode(&Lease {
                            owner: sampler.owner.clone(),
                            generation: self.generation,
                            contract: self.contract.clone(),
                            term: term.clone(),
                        })?,
                        previous_revision,
                    )
                    .await
                    .map_err(|_| ThresholdError::Unavailable)?;
                sampler.term = Some(term.clone());
                renewal_started = Some(started);
                sampler
                    .models
                    .retain(|key, _| self.targets.contains_key(key));
                for state in self.targets.values() {
                    let target = &state.target;
                    let counter = self.counter_total(target, 0, false).await?;
                    let now = Instant::now();
                    let sampled = sampler
                        .models
                        .entry(target.key.clone())
                        .or_insert_with(|| SampleState::baseline(target, counter.total, now));
                    if gap
                        || sampled.continuity != counter.continuity
                        || sampled.incarnation.as_deref() != Some(counter.incarnation.as_str())
                    {
                        *sampled = SampleState::baseline(target, counter.total, now);
                        sampled.continuity = counter.continuity;
                        sampled.incarnation = Some(counter.incarnation.clone());
                    }
                    let decision = sampled.sample(target, counter.total, now);
                    self.decisions
                        .put(
                            &target.key,
                            encode(&Decision {
                                generation: target.generation,
                                contract: target.contract.clone(),
                                owner: sampler.owner.clone(),
                                term: term.clone(),
                                decision,
                            })?,
                        )
                        .await
                        .map_err(|_| ThresholdError::Unavailable)?;
                }
            }
            self.refresh_decisions(&mut tick, renewal_started).await?;
            if renewal_started.is_none() {
                standby_refreshed = true;
                Err(ThresholdError::Unavailable)
            } else {
                Ok(())
            }
        })
        .await;
        if result.is_err() && !standby_refreshed {
            for state in self.targets.values() {
                state.interrupted.store(true, Ordering::Relaxed);
                if let Ok(mut cached) = state.cached.write() {
                    *cached = None;
                }
            }
        }
        attempt.succeeded = result.is_ok();
        result
    }

    async fn refresh_decisions(
        &self,
        tick: &mut TickState,
        renewal_started: Option<Instant>,
    ) -> Result<(), ThresholdError> {
        let started = Instant::now();
        let (revision, lease) = read::<Lease>(&self.leases, LEASE_KEY).await?;
        let lease = lease.ok_or(ThresholdError::Unavailable)?;
        validate_lease(&lease)?;
        if lease.generation != self.generation || lease.contract != self.contract {
            return Err(ThresholdError::Generation);
        }
        let mut verified = Vec::with_capacity(self.targets.len());
        for state in self.targets.values() {
            let (_, decision) = read::<Decision>(&self.decisions, &state.target.key).await?;
            let decision = if let Some(decision) = decision {
                if decision.generation != state.target.generation
                    || decision.contract != state.target.contract
                {
                    return Err(ThresholdError::Generation);
                }
                if decision.owner != lease.owner || decision.term != lease.term {
                    return Err(ThresholdError::Unavailable);
                }
                (decision.decision != ThresholdDecision::Undetermined).then_some(decision.decision)
            } else {
                None
            };
            verified.push((state, decision));
        }
        let latest_started = Instant::now();
        let (latest_revision, latest) = read::<Lease>(&self.leases, LEASE_KEY).await?;
        if latest.as_ref() != Some(&lease) {
            return Err(ThresholdError::Unavailable);
        }
        // A reader cannot know a lease's age on first sight without shared
        // clocks. A locally-started renewal or an observed revision advance
        // proves a lower bound on its commit time on this monotonic clock.
        let mut expires_at = renewal_started.map(|at| at + LEASE_TTL);
        if latest_revision != revision {
            expires_at = Some(started + LEASE_TTL);
        }
        if let Some(previous) = &tick.observed {
            if previous.lease == lease {
                let bound = if previous.revision != latest_revision {
                    Some(previous.read_started + LEASE_TTL)
                } else {
                    previous.expires_at
                };
                expires_at = expires_at.max(bound);
            }
        }
        tick.observed = Some(LeaseObservation {
            revision: latest_revision,
            lease,
            read_started: latest_started,
            expires_at,
        });
        let deadline = expires_at.map(|bound| bound.min(started + THRESHOLD_SAMPLE_INTERVAL));
        for (state, decision) in verified {
            *state
                .cached
                .write()
                .map_err(|_| ThresholdError::Unavailable)? =
                deadline
                    .zip(decision)
                    .map(|(expires_at, decision)| CachedDecision {
                        decision,
                        expires_at,
                    });
        }
        Ok(())
    }

    /// Return only a recently verified local value. Expired or unavailable
    /// authority must preserve the ordinary local warm-up/fallback path.
    pub fn decision(&self, model: &str) -> Result<ThresholdDecision, ThresholdError> {
        let state = self
            .targets
            .get(&digest_key(model))
            .ok_or(ThresholdError::Configuration)?;
        let cached = state
            .cached
            .read()
            .map_err(|_| ThresholdError::Unavailable)?;
        match *cached {
            Some(cached) if Instant::now() < cached.expires_at => Ok(cached.decision),
            _ => Err(ThresholdError::Unavailable),
        }
    }
}

fn validate_lease(lease: &Lease) -> Result<(), ThresholdError> {
    if lease.generation == 0
        || lease.contract.len() != 64
        || !lease
            .contract
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        || Uuid::parse_str(&lease.owner).is_err()
        || Uuid::parse_str(&lease.term).is_err()
    {
        return Err(ThresholdError::Untrusted);
    }
    Ok(())
}

async fn timeout<T>(
    future: impl std::future::Future<Output = Result<T, ThresholdError>>,
) -> Result<T, ThresholdError> {
    tokio::time::timeout(IO_TIMEOUT, future)
        .await
        .map_err(|_| ThresholdError::Unavailable)?
}

fn encode(value: &impl Serialize) -> Result<Bytes, ThresholdError> {
    let bytes = serde_json::to_vec(value).map_err(|_| ThresholdError::Untrusted)?;
    if bytes.len() > MAX_VALUE_BYTES {
        return Err(ThresholdError::Untrusted);
    }
    Ok(bytes.into())
}

async fn read<T: DeserializeOwned>(
    store: &kv::Store,
    key: &str,
) -> Result<(u64, Option<T>), ThresholdError> {
    let subject = format!("$KV.{}.{key}", store.name);
    // The leader API preserves stored headers. Direct-get and ordinary KV
    // helpers lose the distinction between a gateway publish and a worker's
    // server-republished bytes, which must never authorize remote routing.
    let message = match store.stream.get_last_raw_message_by_subject(&subject).await {
        Ok(message) => message,
        Err(error) if error.kind() == stream::RawMessageErrorKind::NoMessageFound => {
            return Ok((0, None))
        }
        Err(_) => return Err(ThresholdError::Unavailable),
    };
    if message.subject.as_str() != subject || message.payload.len() > MAX_VALUE_BYTES {
        return Err(ThresholdError::Untrusted);
    }
    let mut seen = HashSet::new();
    let mut operation = None;
    let mut rollup = None;
    for (name, values) in message.headers.iter() {
        let name: &str = name.as_ref();
        if values.len() != 1 || !seen.insert(name.to_ascii_lowercase()) {
            return Err(ThresholdError::Untrusted);
        }
        if name.eq_ignore_ascii_case("Nats-Expected-Last-Subject-Sequence") {
            if values[0].as_str().parse::<u64>().is_err() {
                return Err(ThresholdError::Untrusted);
            }
        } else if name.eq_ignore_ascii_case("KV-Operation") {
            operation = Some(values[0].as_str());
        } else if name.eq_ignore_ascii_case("Nats-Rollup") {
            rollup = Some(values[0].as_str());
        } else {
            return Err(ThresholdError::Untrusted);
        }
    }
    if rollup.is_some() && (rollup != Some("sub") || operation != Some("PURGE")) {
        return Err(ThresholdError::Untrusted);
    }
    if let Some(operation) = operation {
        return if matches!(operation, "DEL" | "PURGE") {
            Ok((message.sequence, None))
        } else {
            Err(ThresholdError::Untrusted)
        };
    }
    let value = serde_json::from_slice(&message.payload).map_err(|_| ThresholdError::Untrusted)?;
    Ok((message.sequence, Some(value)))
}

async fn bucket(
    context: &jetstream::Context,
    name: &str,
    age: Duration,
    replicas: usize,
) -> Result<kv::Store, ThresholdError> {
    let store = context
        .create_key_value(kv::Config {
            bucket: name.to_string(),
            max_value_size: MAX_VALUE_BYTES as i32,
            max_bytes: MAX_BUCKET_BYTES,
            history: 1,
            max_age: age,
            storage: stream::StorageType::Memory,
            num_replicas: replicas,
            ..Default::default()
        })
        .await
        .map_err(|_| ThresholdError::Unavailable)?;
    let config = &store.stream.cached_info().config;
    if config.max_age != age
        || config.max_message_size != MAX_VALUE_BYTES as i32
        || config.max_bytes != MAX_BUCKET_BYTES
        || config.max_messages_per_subject != 1
        || config.num_replicas != replicas
        || config.storage != stream::StorageType::Memory
        || config.subjects != [format!("$KV.{name}.>")]
        || config.republish.is_some()
        || config.mirror.is_some()
        || config
            .sources
            .as_ref()
            .is_some_and(|sources| !sources.is_empty())
    {
        return Err(ThresholdError::Untrusted);
    }
    Ok(store)
}

fn digest_key(model: &str) -> String {
    hex_digest(&Sha256::digest(model.as_bytes()))
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn target() -> ThresholdTarget {
        let routing: RoutingConfig = serde_json::from_value(json!({
            "policy":"threshold", "fallback_profile":"remote", "wake_above":2,
            "sleep_below":0.5, "window_s":2, "cooldown_s":2
        }))
        .unwrap();
        ThresholdTarget::new("acme/chat", 1, &"a".repeat(64), &routing).unwrap()
    }

    #[test]
    fn sustained_shared_rate_wakes_and_idle_cooldown_returns_remote() {
        let target = target();
        let now = Instant::now();
        let mut state = SampleState::baseline(&target, 0, now);
        for second in 1..=2 {
            assert_eq!(
                state.sample(&target, second * 3, now + Duration::from_secs(second)),
                ThresholdDecision::Undetermined
            );
        }
        assert_eq!(
            state.sample(&target, 9, now + Duration::from_secs(3)),
            ThresholdDecision::WakeLocal
        );
        for second in 4..=5 {
            assert_eq!(
                state.sample(&target, 9, now + Duration::from_secs(second)),
                ThresholdDecision::WakeLocal
            );
        }
        assert_eq!(
            state.sample(&target, 9, now + Duration::from_secs(6)),
            ThresholdDecision::Remote
        );
    }

    #[test]
    fn interrupted_rates_and_timer_gaps_do_not_manufacture_sustained_demand() {
        let target = target();
        let now = Instant::now();
        let mut state = SampleState::baseline(&target, 0, now);
        assert_eq!(
            state.sample(&target, 3, now + Duration::from_secs(1)),
            ThresholdDecision::Undetermined
        );
        // Equality is not above the wake boundary.
        assert_eq!(
            state.sample(&target, 5, now + Duration::from_secs(2)),
            ThresholdDecision::Undetermined
        );
        assert_eq!(
            state.sample(&target, 8, now + Duration::from_secs(3)),
            ThresholdDecision::Undetermined
        );
        assert_eq!(
            state.sample(&target, 11, now + Duration::from_secs(4)),
            ThresholdDecision::Undetermined
        );
        assert_eq!(
            state.sample(&target, 14, now + Duration::from_secs(5)),
            ThresholdDecision::WakeLocal
        );
        // A suspended sampler discards its window evidence and rate baseline.
        assert_eq!(
            state.sample(&target, 999, now + Duration::from_secs(10)),
            ThresholdDecision::Undetermined
        );
        assert_eq!(
            state.sample(&target, 1002, now + Duration::from_secs(11)),
            ThresholdDecision::Undetermined
        );
        let mut changed = target.clone();
        changed.generation += 1;
        assert_eq!(
            state.sample(&changed, 1005, now + Duration::from_secs(12)),
            ThresholdDecision::Undetermined
        );
        assert_eq!(state.generation, changed.generation);
    }

    #[test]
    fn rapid_sampling_keeps_the_counter_baseline_and_binds_all_policy_fields() {
        let target = target();
        let now = Instant::now();
        let mut state = SampleState::baseline(&target, 0, now);
        state.sample(&target, 10, now + Duration::from_millis(10));
        assert_eq!(state.total, 0);
        assert_eq!(state.sampled_at, now);
        let routing: RoutingConfig = serde_json::from_value(json!({
            "policy":"threshold", "fallback_profile":"remote", "wake_above":3,
            "sleep_below":0.5, "window_s":2, "cooldown_s":2
        }))
        .unwrap();
        let changed = ThresholdTarget::new("acme/chat", 1, &"a".repeat(64), &routing).unwrap();
        assert_ne!(target.contract, changed.contract);
        assert_eq!(target.key, changed.key);
        assert!(ThresholdTarget::new("acme/chat:remote", 1, &"a".repeat(64), &routing).is_err());
        assert!(ThresholdTarget::new("acme/chat", 1, "unverified", &routing).is_err());
    }
}
