//! Shared demand accounting and lease-fenced threshold decisions.
//!
//! This prerequisite does not activate routing. Only validated, policy-eligible
//! requests may call `record_request`; neither inference payloads nor caller
//! identifiers enter the broker. A sampler measures counter deltas with its
//! monotonic clock, and consumers require its current broker lease revision.

use std::collections::{HashMap, HashSet};
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
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Decision {
    generation: u64,
    contract: String,
    owner: String,
    lease_revision: u64,
    decision: ThresholdDecision,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Lease {
    owner: String,
    generation: u64,
    contract: String,
}

/// One process owns one sampler token. Replacement/expired ownership loses all
/// timer evidence; the broker's TTL decides expiry, never a gateway wall clock.
pub struct ThresholdSampler {
    owner: String,
    lease_revision: u64,
    models: HashMap<String, SampleState>,
}

impl Default for ThresholdSampler {
    fn default() -> Self {
        Self {
            owner: Uuid::new_v4().to_string(),
            lease_revision: 0,
            models: HashMap::new(),
        }
    }
}

struct SampleState {
    contract: String,
    generation: u64,
    total: u64,
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

/// Control-state connection for a separate NATS endpoint. Inference workers
/// and configuration publishers must not have credentials for that endpoint.
/// Do not pass the inference queue's JetStream context: subject ACLs cannot
/// establish control-record provenance inside a shared account.
pub struct ThresholdCoordinator {
    counters: kv::Store,
    decisions: kv::Store,
    leases: kv::Store,
    targets: HashMap<String, ThresholdTarget>,
    generation: u64,
    contract: String,
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
        // Every replica binds leadership to the same complete configuration,
        // independent of the order in which the registry lists models.
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
            .map(|target| (target.key.clone(), target))
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
            })
        })
        .await
    }

    /// Count once after shared caller validation, before profile recursion.
    pub async fn record_request(&self, model: &str) -> Result<(), ThresholdError> {
        let key = digest_key(model);
        let target = self
            .targets
            .get(&key)
            .ok_or(ThresholdError::Configuration)?;
        timeout(self.counter_total(target, true)).await.map(|_| ())
    }

    // Generation replacement also runs during sampling, so an idle model
    // cannot stall the entire fleet waiting for another request to arrive.
    async fn counter_total(
        &self,
        target: &ThresholdTarget,
        increment: bool,
    ) -> Result<u64, ThresholdError> {
        for _ in 0..CAS_ATTEMPTS {
            let existing = read::<Counter>(&self.counters, &target.key).await?;
            let (revision, mut counter, current) = match existing {
                Some((revision, counter)) => {
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
                        }
                    };
                    (revision, counter, current)
                }
                None => (
                    0,
                    Counter {
                        generation: target.generation,
                        contract: target.contract.clone(),
                        total: 0,
                    },
                    false,
                ),
            };
            if !increment && current {
                return Ok(counter.total);
            }
            if increment {
                counter.total = counter
                    .total
                    .checked_add(1)
                    .ok_or(ThresholdError::Untrusted)?;
            }
            match self
                .counters
                .update(&target.key, encode(&counter)?, revision)
                .await
            {
                Ok(_) => return Ok(counter.total),
                Err(error) if error.kind() == kv::UpdateErrorKind::WrongLastRevision => {}
                Err(_) => return Err(ThresholdError::Unavailable),
            }
            tokio::task::yield_now().await;
        }
        Err(ThresholdError::Contended)
    }

    /// Sample all configured models. Call on the one-second cadence; standby
    /// gateways return unavailable while another valid lease owns the sampler.
    pub async fn sample(&self, sampler: &mut ThresholdSampler) -> Result<(), ThresholdError> {
        let result = timeout(async {
            let lease = read::<Lease>(&self.leases, LEASE_KEY).await?;
            let revision = match lease {
                Some((_, lease))
                    if lease.generation > self.generation
                        || (lease.generation == self.generation
                            && lease.contract != self.contract) =>
                {
                    return Err(ThresholdError::Generation);
                }
                Some((revision, lease)) if lease.generation < self.generation => {
                    sampler.models.clear();
                    revision
                }
                Some((revision, lease))
                    if lease.owner == sampler.owner && revision == sampler.lease_revision =>
                {
                    revision
                }
                Some(_) => return Err(ThresholdError::Unavailable),
                None => {
                    sampler.models.clear();
                    0
                }
            };
            let revision = self
                .leases
                .update(
                    LEASE_KEY,
                    encode(&Lease {
                        owner: sampler.owner.clone(),
                        generation: self.generation,
                        contract: self.contract.clone(),
                    })?,
                    revision,
                )
                .await
                .map_err(|_| ThresholdError::Unavailable)?;
            sampler.lease_revision = revision;
            sampler
                .models
                .retain(|key, _| self.targets.contains_key(key));
            for target in self.targets.values() {
                let total = self.counter_total(target, false).await?;
                let now = Instant::now();
                let state = sampler
                    .models
                    .entry(target.key.clone())
                    .or_insert_with(|| SampleState::baseline(target, total, now));
                let decision = state.sample(target, total, now);
                self.decisions
                    .put(
                        &target.key,
                        encode(&Decision {
                            generation: target.generation,
                            contract: target.contract.clone(),
                            owner: sampler.owner.clone(),
                            lease_revision: revision,
                            decision,
                        })?,
                    )
                    .await
                    .map_err(|_| ThresholdError::Unavailable)?;
            }
            Ok(())
        })
        .await;
        if result.is_err() {
            // Do not renew a partially sampled or incompatible lease again.
            // Both ownership and sustained-window evidence must be re-established.
            sampler.lease_revision = 0;
            sampler.models.clear();
        }
        result
    }

    /// Renewal invalidates decisions from the previous lease revision;
    /// stopped samplers lose authority after the broker's five-second TTL.
    /// Neither case compares wall clocks or different stream leaders.
    pub async fn decision(&self, model: &str) -> Result<ThresholdDecision, ThresholdError> {
        let key = digest_key(model);
        let target = self
            .targets
            .get(&key)
            .ok_or(ThresholdError::Configuration)?;
        timeout(async {
            let (revision, lease) = read::<Lease>(&self.leases, LEASE_KEY)
                .await?
                .ok_or(ThresholdError::Unavailable)?;
            if lease.generation != self.generation || lease.contract != self.contract {
                return Err(ThresholdError::Generation);
            }
            let (_, decision) = read::<Decision>(&self.decisions, &key)
                .await?
                .ok_or(ThresholdError::Unavailable)?;
            if decision.generation != target.generation || decision.contract != target.contract {
                return Err(ThresholdError::Generation);
            }
            if decision.owner != lease.owner || decision.lease_revision != revision {
                return Err(ThresholdError::Unavailable);
            }
            let (latest_revision, latest) = read::<Lease>(&self.leases, LEASE_KEY)
                .await?
                .ok_or(ThresholdError::Unavailable)?;
            if latest_revision != revision || latest.owner != lease.owner {
                return Err(ThresholdError::Unavailable);
            }
            if decision.decision == ThresholdDecision::Undetermined {
                return Err(ThresholdError::Unavailable);
            }
            Ok(decision.decision)
        })
        .await
    }
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
) -> Result<Option<(u64, T)>, ThresholdError> {
    let subject = format!("$KV.{}.{key}", store.name);
    // The leader API preserves stored headers. Direct-get and ordinary KV
    // helpers lose the distinction between a gateway publish and a worker's
    // server-republished bytes, which must never authorize remote routing.
    let message = match store.stream.get_last_raw_message_by_subject(&subject).await {
        Ok(message) => message,
        Err(error) if error.kind() == stream::RawMessageErrorKind::NoMessageFound => {
            return Ok(None)
        }
        Err(_) => return Err(ThresholdError::Unavailable),
    };
    if message.subject.as_str() != subject || message.payload.len() > MAX_VALUE_BYTES {
        return Err(ThresholdError::Untrusted);
    }
    let mut seen = HashSet::new();
    let mut operation = None;
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
        } else {
            return Err(ThresholdError::Untrusted);
        }
    }
    if let Some(operation) = operation {
        return if matches!(operation, "DEL" | "PURGE") {
            Ok(None)
        } else {
            Err(ThresholdError::Untrusted)
        };
    }
    let value = serde_json::from_slice(&message.payload).map_err(|_| ThresholdError::Untrusted)?;
    Ok(Some((message.sequence, value)))
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
