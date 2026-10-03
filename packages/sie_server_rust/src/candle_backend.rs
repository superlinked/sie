//! SIE-shaped native Candle executor for the Rust worker process.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value as Json;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::{error, info, warn};

use crate::candle_embedding::{
    CandleEmbeddingModel, CandleEmbeddingModelConfig, CandleEncodeRequest, CandleEncodeResult,
    CandleEncodeStageTimings, CandleF16MultivectorBatch, CandleForwardProfile,
    CandleMultivectorEmbedding, CandlePreparedEncodeRequest, CandleScoreResult,
};
use crate::candle_residency::{CandleResidency, ResidencyPolicy, ResidencyUseGuard};
use crate::candle_splade::CandleSparseEmbedding;
use crate::ipc_types::{
    ApplyModelConfigRequest, BatchOutcome, BatchedF16MultivectorItem, BatchedF16MultivectorOutput,
    DenseOutput, Disposition, EncodeBatchItem, EnsureModelReadyResponse, F16Values, ItemOutcome,
    ModelDescriptor, MultivectorOutput, PreparedTokens, ProcessEncodeBatchRequest,
    ProcessScoreBatchRequest, RawOutput, ReadinessState, ReplaceModelConfigsRequest,
    RunBatchRequest, ScoreBatchItem, ScoreOutputRaw, SetPinnedModelsRequest,
    SetPinnedModelsResponse, SparseOutput, UnitCounts,
};
use crate::observability::metrics::{
    self as managed_metrics, ForwardCompleted, ForwardInputSource, ForwardOutcome,
    ForwardOutputPath, ForwardStage, ForwardState, ModelEvictionReason, ModelLoadOutcome,
    ModelLoadStage, OomOutcome, OomStrategy,
};
use crate::text_prep::TextPrep;

const MODEL_LOAD_WAIT_INTERVAL: Duration = Duration::from_millis(50);
const DEFAULT_SLOW_FORWARD_LOG_MS: f64 = 1_000.0;
const OOM_ERROR_INDICATORS: &[&str] = &[
    "out of memory",
    "cannot allocate memory",
    "failed to allocate",
];

#[derive(Debug, Clone)]
pub struct CandleBackendConfig {
    pub batch_budget: u32,
    pub normalize: bool,
    pub max_concurrent_forwards: usize,
    pub idle_evict_s: Option<Duration>,
    pub oom_recovery: CandleOomRecoveryConfig,
}

impl CandleBackendConfig {
    pub fn new(batch_budget: u32, normalize: bool, max_concurrent_forwards: usize) -> Self {
        Self {
            batch_budget: batch_budget.max(1),
            normalize,
            max_concurrent_forwards: max_concurrent_forwards.max(1),
            idle_evict_s: None,
            oom_recovery: CandleOomRecoveryConfig::default(),
        }
    }

    pub fn with_idle_evict_s(mut self, idle_evict_s: Option<Duration>) -> Self {
        self.idle_evict_s = idle_evict_s;
        self
    }

    pub fn with_oom_recovery(mut self, oom_recovery: CandleOomRecoveryConfig) -> Self {
        self.oom_recovery = oom_recovery;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandleOomRecoveryAction {
    CacheClear,
    EvictLru,
    SplitBatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandleOomRecoveryConfig {
    enabled: bool,
    strategy: Vec<CandleOomRecoveryAction>,
    max_split_depth: usize,
    nak_delay_ms: u64,
}

impl Default for CandleOomRecoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: vec![
                CandleOomRecoveryAction::CacheClear,
                CandleOomRecoveryAction::EvictLru,
                CandleOomRecoveryAction::SplitBatch,
            ],
            max_split_depth: 4,
            nak_delay_ms: 10_000,
        }
    }
}

impl CandleOomRecoveryConfig {
    pub fn from_env() -> Result<Self> {
        let mut config = Self::default();
        if let Some(raw) = env_var("SIE_OOM_RECOVERY__ENABLED") {
            config.enabled = parse_env_bool("SIE_OOM_RECOVERY__ENABLED", &raw)?;
        }
        if let Some(raw) = env_var("SIE_OOM_RECOVERY__STRATEGY") {
            config.strategy = parse_oom_recovery_strategy(&raw)?;
        }
        if let Some(raw) = env_var("SIE_OOM_RECOVERY__MAX_SPLIT_DEPTH") {
            config.max_split_depth = parse_oom_max_split_depth(&raw)?;
        }
        if let Some(raw) = env_var("SIE_OOM_NAK_DELAY_S") {
            config.nak_delay_ms = parse_oom_nak_delay_ms(&raw)?;
        }
        let kill_switch = env_var("SIE_DISABLE_OOM_RECOVERY");
        apply_oom_recovery_kill_switch(&mut config, kill_switch.as_deref());
        Ok(config)
    }
}

#[derive(Clone)]
pub struct CandleBackend {
    config: CandleBackendConfig,
    catalog: Arc<RwLock<HashMap<String, ModelRuntimeConfig>>>,
    loaded_embeddings: Arc<Mutex<CandleResidency<LoadedEmbeddingModel>>>,
    loading_embeddings: Arc<Mutex<HashSet<String>>>,
    preload_models: Arc<RwLock<HashSet<String>>>,
    pinned_models: Arc<RwLock<HashSet<String>>>,
    idle_evictor: Arc<Mutex<Option<IdleEvictorState>>>,
}

struct IdleEvictorState {
    abort_handle: tokio::task::AbortHandle,
    finished: Arc<AtomicBool>,
    stop_requested: Arc<AtomicBool>,
}

impl IdleEvictorState {
    fn is_running(&self) -> bool {
        !self.finished.load(Ordering::Relaxed)
    }

    fn abort(self) -> bool {
        let was_running = self.is_running();
        self.stop_requested.store(true, Ordering::Relaxed);
        self.abort_handle.abort();
        was_running
    }
}

struct LoadedEmbeddingModel {
    model: Arc<CandleEmbeddingModel>,
    forward_permits: Arc<Semaphore>,
}

struct ModelExecution<T> {
    lock_wait_ms: f64,
    result: T,
}

struct ForwardSlot {
    _permit: OwnedSemaphorePermit,
    wait_ms: f64,
}

impl LoadedEmbeddingModel {
    fn new(model: CandleEmbeddingModel, max_concurrent_forwards: usize) -> Self {
        Self {
            model: Arc::new(model),
            forward_permits: Arc::new(Semaphore::new(max_concurrent_forwards.max(1))),
        }
    }

    async fn acquire_forward_slot(&self) -> Result<ForwardSlot> {
        let wait_start = Instant::now();
        let permit = Arc::clone(&self.forward_permits)
            .acquire_owned()
            .await
            .map_err(|_| anyhow::anyhow!("Candle forward semaphore closed"))?;
        let wait_ms = elapsed_ms(wait_start);
        Ok(ForwardSlot {
            _permit: permit,
            wait_ms,
        })
    }

    fn with_model<T>(
        &self,
        slot: ForwardSlot,
        f: impl FnOnce(&CandleEmbeddingModel) -> T,
    ) -> ModelExecution<T> {
        let result = f(&self.model);
        drop(slot._permit);
        ModelExecution {
            lock_wait_ms: slot.wait_ms,
            result,
        }
    }

    fn first(&self) -> Arc<CandleEmbeddingModel> {
        Arc::clone(&self.model)
    }
}

struct LoadedEmbeddingModelUse {
    model_id: String,
    model: Arc<LoadedEmbeddingModel>,
    _residency_use: ResidencyUseGuard<LoadedEmbeddingModel>,
}

struct EncodeGroupExecution {
    encoded: Result<CandleEncodeResult>,
    stats: EncodeGroupStats,
    lock_wait_ms: f64,
    encode_elapsed_ms: f64,
    encode_source: &'static str,
}

#[derive(Debug, Clone)]
struct PreparedScoreItem {
    query: CandleEncodeRequest,
    docs: Vec<CandleEncodeRequest>,
    item_ids: Vec<String>,
    normalize: bool,
    work_budget: usize,
    prepared_tokens: Option<PreparedTokens>,
    allow_prepared_tokens: bool,
}

struct ScoreItemExecution {
    scored: Result<CandleScoreResult>,
    lock_wait_ms: f64,
    wall_ms: f64,
}

#[derive(Debug, Clone)]
struct EncodeItemIdentity {
    position: usize,
    work_item_id: String,
    request_id: String,
    item_index: u32,
}

type PreparedEncodeGroup = Vec<(usize, PreparedEncodeItem)>;
type EncodeGroupKey = (EncodeOutputKind, String, bool, bool, String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EncodeOutputKind {
    Dense,
    Sparse,
    Multivector,
}

fn default_normalize(output_kind: EncodeOutputKind, configured_default: bool) -> bool {
    match output_kind {
        EncodeOutputKind::Sparse => false,
        EncodeOutputKind::Dense | EncodeOutputKind::Multivector => configured_default,
    }
}

#[derive(Clone, Copy)]
struct EncodeGroupContext<'a> {
    model_id: &'a str,
    output_kind: EncodeOutputKind,
    pooling: &'a str,
    normalize: bool,
    is_query: bool,
    output_dtype: &'a str,
    accepts_batched_f16_multivectors: bool,
}

struct EncodeGroupRecoveryContext<'a> {
    model_id: &'a str,
    telemetry_model_id: &'a str,
    active_model_id: &'a str,
    output_kind: EncodeOutputKind,
    pooling: String,
    normalize: bool,
    is_query: bool,
    output_dtype: String,
    accepts_batched_f16_multivectors: bool,
}

enum EncodeGroupAttempt {
    Executed {
        group: PreparedEncodeGroup,
        execution: Box<EncodeGroupExecution>,
    },
    FailedBeforeExecution {
        identities: Vec<EncodeItemIdentity>,
        message: String,
    },
}

enum EncodeGroupOutcome {
    Success {
        items: usize,
    },
    Oom {
        group: PreparedEncodeGroup,
        error: anyhow::Error,
    },
    Failed,
}

#[derive(Debug, PartialEq, Eq)]
enum OomRecoveryStep {
    RetryAfterCacheClear,
    RetryAfterEviction(String),
    SplitBatch,
    Terminal,
}

struct SlowEncodeLogContext<'a> {
    model_id: &'a str,
    pooling: &'a str,
    normalize: bool,
    stats: &'a EncodeGroupStats,
    first_item: Option<&'a PreparedEncodeItem>,
    ok: bool,
    lock_wait_ms: f64,
    encode_elapsed_ms: f64,
    encode_source: &'static str,
}

struct EncodeGroupModelContext {
    loaded: Arc<LoadedEmbeddingModel>,
    slot: ForwardSlot,
    output_kind: EncodeOutputKind,
    pooling: String,
    normalize: bool,
    is_query: bool,
    output_dtype: String,
}

fn maybe_log_slow_encode_group(ctx: SlowEncodeLogContext<'_>) {
    let Some(threshold_ms) = candle_slow_forward_log_threshold_ms() else {
        return;
    };
    let slow_encode = ctx.encode_elapsed_ms >= threshold_ms;
    let slow_lock = ctx.lock_wait_ms >= threshold_ms;
    if !(slow_encode || slow_lock) {
        return;
    }

    let first_work_item_id = ctx
        .first_item
        .map(|item| item.work_item_id.as_str())
        .unwrap_or("");
    let first_request_id = ctx
        .first_item
        .map(|item| item.request_id.as_str())
        .unwrap_or("");
    let first_item_index = ctx
        .first_item
        .map(|item| item.item_index)
        .unwrap_or_default();
    let first_item_is_query = ctx.first_item.is_some_and(|item| item.is_query);
    warn!(
        model_id = ctx.model_id,
        group_items = ctx.stats.items,
        pooling = ctx.pooling,
        normalize = ctx.normalize,
        ok = ctx.ok,
        threshold_ms,
        slow_encode,
        slow_lock,
        lock_wait_ms = ctx.lock_wait_ms,
        encode_elapsed_ms = ctx.encode_elapsed_ms,
        encode_source = ctx.encode_source,
        prepared_items = ctx.stats.prepared_items,
        prepared_sequences = ctx.stats.prepared_sequences,
        prepared_tokens_total = ctx.stats.prepared_tokens_total,
        prepared_tokens_min = ctx.stats.prepared_tokens_min,
        prepared_tokens_max = ctx.stats.prepared_tokens_max,
        prepared_max_seq_len = ctx.stats.prepared_max_seq_len,
        text_chars_total = ctx.stats.text_chars_total,
        text_chars_min = ctx.stats.text_chars_min,
        text_chars_max = ctx.stats.text_chars_max,
        first_work_item_id,
        first_request_id,
        first_item_index,
        first_item_is_query,
        "Slow Candle embedding encode group"
    );
}

fn encode_group_on_model(
    ctx: EncodeGroupModelContext,
    group: PreparedEncodeGroup,
) -> (PreparedEncodeGroup, EncodeGroupExecution) {
    let stats = EncodeGroupStats::from_group(&group);
    let requests: Vec<CandleEncodeRequest> = group
        .iter()
        .map(|(_, item)| CandleEncodeRequest {
            text: item.text.clone(),
        })
        .collect();
    let encode_start = Instant::now();
    let mut encode_source = "raw";
    let execution = ctx.loaded.with_model(ctx.slot, |model| {
        if let Some(prepared_requests) = prepared_requests_for_group(model, &group) {
            encode_source = "prepared";
            match ctx.output_kind {
                EncodeOutputKind::Dense => {
                    model.encode_prepared(&prepared_requests, &ctx.pooling, ctx.normalize)
                }
                EncodeOutputKind::Sparse => model.encode_prepared_sparse(&prepared_requests),
                EncodeOutputKind::Multivector => model
                    .encode_prepared_multivector_intermediate(
                        &prepared_requests,
                        ctx.normalize,
                        ctx.is_query,
                        &ctx.output_dtype,
                    )
                    .and_then(|intermediate| intermediate.finish()),
            }
        } else {
            match ctx.output_kind {
                EncodeOutputKind::Dense => model.encode(&requests, &ctx.pooling, ctx.normalize),
                EncodeOutputKind::Sparse => model.encode_sparse(&requests),
                EncodeOutputKind::Multivector => model
                    .encode_multivector_intermediate(
                        &requests,
                        ctx.normalize,
                        ctx.is_query,
                        &ctx.output_dtype,
                    )
                    .and_then(|intermediate| intermediate.finish()),
            }
        }
    });
    let encoded = execution.result;
    let encode_elapsed_ms = elapsed_ms(encode_start);
    let execution = EncodeGroupExecution {
        encoded,
        stats,
        lock_wait_ms: execution.lock_wait_ms,
        encode_elapsed_ms,
        encode_source,
    };
    (group, execution)
}

fn forward_output_path(context: EncodeGroupContext<'_>) -> ForwardOutputPath {
    match context.output_kind {
        EncodeOutputKind::Dense => ForwardOutputPath::Dense,
        EncodeOutputKind::Sparse => ForwardOutputPath::Sparse,
        EncodeOutputKind::Multivector if context.output_dtype == "float32" => {
            ForwardOutputPath::MultivectorF32
        }
        EncodeOutputKind::Multivector if context.accepts_batched_f16_multivectors => {
            ForwardOutputPath::MultivectorF16Batched
        }
        EncodeOutputKind::Multivector => ForwardOutputPath::MultivectorF16Individual,
    }
}

fn forward_input_source(source: &str) -> ForwardInputSource {
    match source {
        "prepared" => ForwardInputSource::Prepared,
        "raw" => ForwardInputSource::Raw,
        _ => ForwardInputSource::Other,
    }
}

fn forward_stage_durations(
    stages: CandleEncodeStageTimings,
    profile: Option<CandleForwardProfile>,
) -> Vec<(ForwardStage, f64)> {
    let mut values = vec![
        (ForwardStage::Forward, stages.forward_ms / 1_000.0),
        (ForwardStage::Pool, stages.pool_ms / 1_000.0),
        (ForwardStage::Normalize, stages.normalize_ms / 1_000.0),
        (ForwardStage::Conversion, stages.conversion_ms / 1_000.0),
        (
            ForwardStage::ConversionTensorReadback,
            stages.conversion_tensor_readback_ms / 1_000.0,
        ),
        (
            ForwardStage::ConversionHostPack,
            stages.conversion_host_pack_ms / 1_000.0,
        ),
        (ForwardStage::Inference, stages.inference_ms / 1_000.0),
    ];
    match profile {
        Some(CandleForwardProfile::XlmRoberta(profile)) => values.extend([
            (
                ForwardStage::XlmRobertaEmbedding,
                profile.embedding_ms / 1_000.0,
            ),
            (
                ForwardStage::XlmRobertaAttention,
                profile.attention_ms / 1_000.0,
            ),
            (
                ForwardStage::XlmRobertaAttentionQkv,
                profile.attention_qkv_ms / 1_000.0,
            ),
            (
                ForwardStage::XlmRobertaAttentionFlash,
                profile.attention_flash_ms / 1_000.0,
            ),
            (
                ForwardStage::XlmRobertaAttentionOutputDense,
                profile.attention_output_dense_ms / 1_000.0,
            ),
            (
                ForwardStage::XlmRobertaAttentionOutputLayernorm,
                profile.attention_output_layernorm_ms / 1_000.0,
            ),
            (ForwardStage::XlmRobertaFfn, profile.ffn_ms / 1_000.0),
            (
                ForwardStage::XlmRobertaFfnIntermediateDense,
                profile.ffn_intermediate_dense_ms / 1_000.0,
            ),
            (
                ForwardStage::XlmRobertaFfnActivation,
                profile.ffn_activation_ms / 1_000.0,
            ),
            (
                ForwardStage::XlmRobertaFfnOutputDense,
                profile.ffn_output_dense_ms / 1_000.0,
            ),
            (
                ForwardStage::XlmRobertaFfnOutputLayernorm,
                profile.ffn_output_layernorm_ms / 1_000.0,
            ),
        ]),
        Some(CandleForwardProfile::GteRope(profile)) => values.extend([
            (
                ForwardStage::GteRopeEmbedding,
                profile.embedding_ms / 1_000.0,
            ),
            (
                ForwardStage::GteRopeRopeSelect,
                profile.rope_select_ms / 1_000.0,
            ),
            (
                ForwardStage::GteRopeAttention,
                profile.attention_ms / 1_000.0,
            ),
            (
                ForwardStage::GteRopeAttentionQkv,
                profile.attention_qkv_ms / 1_000.0,
            ),
            (
                ForwardStage::GteRopeAttentionRotary,
                profile.attention_rotary_ms / 1_000.0,
            ),
            (
                ForwardStage::GteRopeAttentionFlash,
                profile.attention_flash_ms / 1_000.0,
            ),
            (
                ForwardStage::GteRopeAttentionOutputDense,
                profile.attention_output_dense_ms / 1_000.0,
            ),
            (
                ForwardStage::GteRopeAttentionOutputLayernorm,
                profile.attention_output_layernorm_ms / 1_000.0,
            ),
            (ForwardStage::GteRopeFfn, profile.ffn_ms / 1_000.0),
            (
                ForwardStage::GteRopeFfnUpGate,
                profile.ffn_up_gate_ms / 1_000.0,
            ),
            (
                ForwardStage::GteRopeFfnActivation,
                profile.ffn_activation_ms / 1_000.0,
            ),
            (ForwardStage::GteRopeFfnDown, profile.ffn_down_ms / 1_000.0),
            (
                ForwardStage::GteRopeFfnOutputLayernorm,
                profile.ffn_output_layernorm_ms / 1_000.0,
            ),
        ]),
        Some(CandleForwardProfile::ModernBert(profile)) => values.extend([
            (
                ForwardStage::ModernBertEmbedding,
                profile.embedding_ms / 1_000.0,
            ),
            (
                ForwardStage::ModernBertEmbeddingNorm,
                profile.embedding_norm_ms / 1_000.0,
            ),
            (
                ForwardStage::ModernBertRopeSelect,
                profile.rope_select_ms / 1_000.0,
            ),
            (
                ForwardStage::ModernBertAttention,
                profile.attention_ms / 1_000.0,
            ),
            (
                ForwardStage::ModernBertAttentionNorm,
                profile.attention_norm_ms / 1_000.0,
            ),
            (
                ForwardStage::ModernBertAttentionQkv,
                profile.attention_qkv_ms / 1_000.0,
            ),
            (
                ForwardStage::ModernBertAttentionRotary,
                profile.attention_rotary_ms / 1_000.0,
            ),
            (
                ForwardStage::ModernBertAttentionFlash,
                profile.attention_flash_ms / 1_000.0,
            ),
            (
                ForwardStage::ModernBertAttentionOutputDense,
                profile.attention_output_dense_ms / 1_000.0,
            ),
            (ForwardStage::ModernBertMlp, profile.mlp_ms / 1_000.0),
            (
                ForwardStage::ModernBertMlpNorm,
                profile.mlp_norm_ms / 1_000.0,
            ),
            (ForwardStage::ModernBertMlpWi, profile.mlp_wi_ms / 1_000.0),
            (
                ForwardStage::ModernBertMlpActivation,
                profile.mlp_activation_ms / 1_000.0,
            ),
            (ForwardStage::ModernBertMlpWo, profile.mlp_wo_ms / 1_000.0),
            (
                ForwardStage::ModernBertFinalNorm,
                profile.final_norm_ms / 1_000.0,
            ),
        ]),
        None => {}
    }
    values
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn idle_evict_check_interval(threshold: Duration) -> Duration {
    let secs = threshold.as_secs().saturating_div(2).clamp(1, 60);
    Duration::from_secs(secs)
}

fn env_var(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn parse_env_bool(name: &str, raw: &str) -> Result<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => anyhow::bail!("{name} must be a boolean; got {raw:?}"),
    }
}

fn parse_oom_recovery_strategy(raw: &str) -> Result<Vec<CandleOomRecoveryAction>> {
    let trimmed = raw.trim().trim_start_matches('[').trim_end_matches(']');
    let mut strategy = Vec::new();
    for value in trimmed.split(',') {
        let value = value.trim().trim_matches('"').trim_matches('\'').trim();
        if value.is_empty() {
            continue;
        }
        let action = match value {
            "cache_clear" => CandleOomRecoveryAction::CacheClear,
            "evict_lru" => CandleOomRecoveryAction::EvictLru,
            "split_batch" => CandleOomRecoveryAction::SplitBatch,
            _ => anyhow::bail!("unknown SIE_OOM_RECOVERY__STRATEGY action {value:?}"),
        };
        if !strategy.contains(&action) {
            strategy.push(action);
        }
    }
    if strategy.is_empty() {
        anyhow::bail!("SIE_OOM_RECOVERY__STRATEGY must include at least one action");
    }
    Ok(strategy)
}

fn parse_oom_max_split_depth(raw: &str) -> Result<usize> {
    let depth: usize = raw.trim().parse().with_context(|| {
        format!("SIE_OOM_RECOVERY__MAX_SPLIT_DEPTH must be an integer; got {raw:?}")
    })?;
    if depth > 8 {
        anyhow::bail!("SIE_OOM_RECOVERY__MAX_SPLIT_DEPTH must be <= 8; got {depth}");
    }
    Ok(depth)
}

fn parse_oom_nak_delay_ms(raw: &str) -> Result<u64> {
    let seconds: f64 = raw
        .trim()
        .parse()
        .with_context(|| format!("SIE_OOM_NAK_DELAY_S must be a number; got {raw:?}"))?;
    if !seconds.is_finite() || seconds <= 0.0 {
        anyhow::bail!("SIE_OOM_NAK_DELAY_S must be a positive finite number; got {raw:?}");
    }
    Ok((seconds * 1000.0).round() as u64)
}

fn apply_oom_recovery_kill_switch(config: &mut CandleOomRecoveryConfig, raw: Option<&str>) {
    if raw
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false)
    {
        config.enabled = false;
    }
}

fn next_oom_recovery_step(
    strategy: &[CandleOomRecoveryAction],
    cursor: &mut usize,
    mut evict_lru: impl FnMut() -> Option<String>,
) -> OomRecoveryStep {
    while *cursor < strategy.len() {
        let action = strategy[*cursor];
        *cursor += 1;
        match action {
            CandleOomRecoveryAction::CacheClear => {
                clear_candle_allocator_cache();
                return OomRecoveryStep::RetryAfterCacheClear;
            }
            CandleOomRecoveryAction::EvictLru => {
                if let Some(evicted_model_id) = evict_lru() {
                    return OomRecoveryStep::RetryAfterEviction(evicted_model_id);
                }
            }
            CandleOomRecoveryAction::SplitBatch => return OomRecoveryStep::SplitBatch,
        }
    }
    OomRecoveryStep::Terminal
}

fn clear_candle_allocator_cache() {
    // Candle does not currently expose a Torch-style allocator cache clear.
}

fn split_encode_group(
    mut group: PreparedEncodeGroup,
) -> (PreparedEncodeGroup, PreparedEncodeGroup) {
    let right = group.split_off(group.len() / 2);
    (group, right)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RuntimeDefaults {
    query_template: Option<String>,
    doc_template: Option<String>,
    default_instruction: Option<String>,
    normalize: Option<bool>,
    pooling: Option<String>,
    output_dtype: Option<String>,
    max_batch_tokens: Option<usize>,
    score_strategy: Option<CandleScoreStrategy>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelRuntimeConfig {
    hf_id: String,
    hf_revision: Option<String>,
    max_sequence_length: Option<usize>,
    query_max_length: Option<usize>,
    dense_dim: Option<usize>,
    sparse_dim: Option<usize>,
    multivector_dim: Option<usize>,
    compute_precision: Option<String>,
    profiles: HashMap<String, RuntimeDefaults>,
    routable_model_ids: Vec<String>,
    task_kind: CandleTaskKind,
}

impl ModelRuntimeConfig {
    fn native_sparse_dim(&self) -> Option<usize> {
        (self.dense_dim.is_none() && self.multivector_dim.is_none())
            .then_some(self.sparse_dim)
            .flatten()
    }

    fn native_multivector_dim(&self) -> Option<usize> {
        self.dense_dim
            .is_none()
            .then_some(self.multivector_dim)
            .flatten()
    }

    fn output_types(&self, model_id: &str, requested_profile: Option<&str>) -> Vec<String> {
        let mut outputs = Vec::new();
        if self.task_kind.supports_embedding() {
            if self.dense_dim.is_some() {
                outputs.push("dense".to_string());
            } else if self.native_sparse_dim().is_some() {
                outputs.push("sparse".to_string());
            } else if self.native_multivector_dim().is_some() {
                outputs.push("multivector".to_string());
            } else {
                outputs.push("dense".to_string());
            }
        }
        if self.supports_native_score(model_id, requested_profile) {
            outputs.push("score".to_string());
        }
        outputs
    }

    fn supports_native_score(&self, model_id: &str, requested_profile: Option<&str>) -> bool {
        self.task_kind.supports_score()
            && self.native_multivector_dim().is_some()
            && selected_runtime_defaults(model_id, requested_profile, &self.profiles).score_strategy
                == Some(CandleScoreStrategy::ColbertMaxsim)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandleScoreStrategy {
    ColbertMaxsim,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum CandleTaskKind {
    #[default]
    Embedding,
    Rerank,
    EmbeddingAndRerank,
}

impl CandleTaskKind {
    fn supports_embedding(self) -> bool {
        matches!(self, Self::Embedding | Self::EmbeddingAndRerank)
    }

    fn supports_score(self) -> bool {
        matches!(self, Self::Rerank | Self::EmbeddingAndRerank)
    }
}

#[derive(Debug, Deserialize)]
struct ModelConfigYaml {
    sie_id: String,
    #[serde(default)]
    hf_id: Option<String>,
    #[serde(default)]
    hf_revision: Option<String>,
    #[serde(default)]
    max_sequence_length: Option<usize>,
    #[serde(default)]
    tasks: TasksYaml,
    #[serde(default)]
    profiles: BTreeMap<String, ProfileConfigYaml>,
}

#[derive(Debug, Default, Deserialize)]
struct TasksYaml {
    #[serde(default)]
    encode: Option<Json>,
    #[serde(default)]
    score: Option<Json>,
}

#[derive(Debug, Default, Deserialize)]
struct ProfileConfigYaml {
    #[serde(default)]
    extends: Option<String>,
    #[serde(default)]
    max_batch_tokens: Option<usize>,
    #[serde(default)]
    compute_precision: Option<String>,
    #[serde(default)]
    adapter_path: Option<String>,
    #[serde(default)]
    adapter_options: AdapterOptionsYaml,
}

#[derive(Debug, Default, Deserialize)]
struct AdapterOptionsYaml {
    #[serde(default)]
    loadtime: BTreeMap<String, Json>,
    #[serde(default)]
    runtime: BTreeMap<String, Json>,
}

impl CandleBackend {
    pub fn new(config: CandleBackendConfig) -> Self {
        Self {
            config,
            catalog: Arc::new(RwLock::new(HashMap::new())),
            loaded_embeddings: Arc::new(Mutex::new(CandleResidency::default())),
            loading_embeddings: Arc::new(Mutex::new(HashSet::new())),
            preload_models: Arc::new(RwLock::new(HashSet::new())),
            pinned_models: Arc::new(RwLock::new(HashSet::new())),
            idle_evictor: Arc::new(Mutex::new(None)),
        }
    }

    pub fn supports(&self, model_id: &str) -> bool {
        let Some(model_id) = canonical_pinned_model_id(model_id) else {
            return false;
        };
        let guard = self.catalog.read().expect("Candle catalog lock poisoned");
        self.catalog_config_for(&model_id, &guard)
            .is_some_and(|config| config.routable_model_ids.contains(&model_id))
    }

    /// Resolve request-supplied routing values through the loaded release
    /// catalog before they may become metric attributes.
    pub(crate) fn telemetry_dimensions(
        &self,
        model_id: &str,
        requested_profile: Option<&str>,
    ) -> (String, String) {
        let Some(route) = effective_model_id(model_id, requested_profile) else {
            return ("other".to_string(), "other".to_string());
        };
        let catalog = self.catalog.read().expect("Candle catalog lock poisoned");
        telemetry_dimensions_for_route(&route, &catalog)
            .unwrap_or_else(|| ("other".to_string(), "other".to_string()))
    }

    fn record_model_residency_for_base(
        &self,
        base_model_id: &str,
        loaded: bool,
        eviction_reason: Option<ModelEvictionReason>,
    ) {
        if !managed_metrics::metrics_enabled() {
            return;
        }
        let dimensions = {
            let catalog = self.catalog.read().expect("Candle catalog lock poisoned");
            telemetry_dimensions_for_base_model(base_model_id, &catalog)
        };
        for (model, profile) in dimensions {
            managed_metrics::record_model_residency_changed(
                &model, &profile, loaded,
                // Candle does not expose authoritative resident bytes.
                None,
            );
            if let Some(reason) = eviction_reason {
                managed_metrics::record_model_evicted(&model, &profile, reason);
            }
        }
    }

    fn record_oom_recovery(&self, model_id: &str, strategy: OomStrategy, outcome: OomOutcome) {
        if !managed_metrics::metrics_enabled() {
            return;
        }
        let (model, profile) = self.telemetry_dimensions(model_id, None);
        managed_metrics::record_oom_recovery_completed(&model, &profile, strategy, outcome);
    }

    pub fn supported_models(&self) -> Vec<String> {
        let mut models: Vec<String> = self
            .catalog
            .read()
            .expect("Candle catalog lock poisoned")
            .values()
            .flat_map(|config| config.routable_model_ids.iter().cloned())
            .collect();
        models.sort();
        models.dedup();
        models
    }

    pub fn loaded_models(&self) -> Vec<String> {
        let loaded_model_ids: Vec<String> = self
            .loaded_embeddings
            .lock()
            .expect("Candle model lock poisoned")
            .keys();
        let catalog = self.catalog.read().expect("Candle catalog lock poisoned");
        loaded_model_routes_for_catalog(&loaded_model_ids, &catalog)
    }

    pub fn set_preload_models(&self, models: &[String]) -> u32 {
        let next = normalize_pinned_model_ids(models);
        let preload_count = u32::try_from(next.len()).unwrap_or(u32::MAX);
        {
            let mut preload = self
                .preload_models
                .write()
                .expect("Candle preload-model lock poisoned");
            *preload = next;
        }

        if preload_count > 0 {
            info!(preload_count, "Candle preload model set configured");
        }
        self.refresh_residency_policies();
        self.reconcile_warm_models();
        preload_count
    }

    pub fn set_pinned_models(&self, req: &SetPinnedModelsRequest) -> SetPinnedModelsResponse {
        let next = normalize_pinned_model_ids(&req.models);
        let pinned_count = u32::try_from(next.len()).unwrap_or(u32::MAX);
        let changed = {
            let mut pinned = self
                .pinned_models
                .write()
                .expect("Candle pinned-model lock poisoned");
            if *pinned == next {
                false
            } else {
                *pinned = next;
                true
            }
        };

        if changed {
            info!(pinned_count, "Candle pinned model set updated from sidecar");
        }
        self.refresh_residency_policies();
        self.reconcile_warm_models();
        SetPinnedModelsResponse {
            applied: true,
            pinned_count,
        }
    }

    pub async fn health_ready(&self) -> bool {
        let catalog = self.catalog.read().expect("Candle catalog lock poisoned");
        !catalog.is_empty()
    }

    pub fn start_idle_evictor(&self) -> bool {
        let Some(threshold) = self.config.idle_evict_s else {
            return false;
        };
        let mut idle_evictor = self
            .idle_evictor
            .lock()
            .expect("Candle idle-evictor lock poisoned");
        if idle_evictor
            .as_ref()
            .is_some_and(IdleEvictorState::is_running)
        {
            return false;
        }
        let interval = idle_evict_check_interval(threshold);
        let backend = self.clone();
        let finished = Arc::new(AtomicBool::new(false));
        let stop_requested = Arc::new(AtomicBool::new(false));
        let task_finished = Arc::clone(&finished);
        let task_stop_requested = Arc::clone(&stop_requested);
        let handle = tokio::spawn(async move {
            info!(
                idle_threshold_s = threshold.as_secs(),
                check_interval_s = interval.as_secs_f64(),
                "Candle idle evictor started"
            );
            loop {
                tokio::time::sleep(interval).await;
                if let Some(evicted_model_id) = backend.evict_idle_embedding_model(threshold) {
                    info!(
                        model_id = %evicted_model_id,
                        idle_threshold_s = threshold.as_secs(),
                        "Candle idle eviction unloaded resident embedding model"
                    );
                }
            }
        });
        let abort_handle = handle.abort_handle();
        tokio::spawn(async move {
            let result = handle.await;
            task_finished.store(true, Ordering::Relaxed);
            let stop_requested = task_stop_requested.load(Ordering::Relaxed);
            match result {
                Ok(()) if stop_requested => {}
                Ok(()) => error!("Candle idle evictor task exited unexpectedly"),
                Err(error) if stop_requested && error.is_cancelled() => {}
                Err(error) => error!(error = %error, "Candle idle evictor task terminated"),
            }
        });
        *idle_evictor = Some(IdleEvictorState {
            abort_handle,
            finished,
            stop_requested,
        });
        true
    }

    pub fn stop_idle_evictor(&self) -> bool {
        let handle = self
            .idle_evictor
            .lock()
            .expect("Candle idle-evictor lock poisoned")
            .take();
        if let Some(handle) = handle {
            return handle.abort();
        }
        false
    }

    pub fn apply_model_config(&self, req: &ApplyModelConfigRequest) -> Result<()> {
        if req.bundle_id.trim().is_empty() {
            anyhow::bail!("bundle_id is required");
        }
        let (sie_id, runtime_config) = parse_runtime_config(&req.model_id, &req.model_config)?;

        let config_changed = self
            .catalog
            .read()
            .expect("Candle catalog lock poisoned")
            .get(&sie_id)
            .is_some_and(|current| current != &runtime_config);
        if config_changed {
            self.evict_embedding_model(&sie_id, ModelEvictionReason::ConfigChange);
        }
        self.catalog
            .write()
            .expect("Candle catalog lock poisoned")
            .insert(sie_id, runtime_config);
        self.refresh_residency_policies();
        self.reconcile_warm_models();
        Ok(())
    }

    pub fn replace_model_configs(&self, req: &ReplaceModelConfigsRequest) -> Result<Vec<String>> {
        if req.bundle_id.trim().is_empty() {
            anyhow::bail!("bundle_id is required");
        }

        let mut next_catalog = HashMap::new();
        let mut applied_models = Vec::new();
        for entry in &req.models {
            let (sie_id, runtime_config) =
                parse_runtime_config(&entry.model_id, &entry.model_config)?;
            applied_models.extend(runtime_config.routable_model_ids.iter().cloned());
            next_catalog.insert(sie_id, runtime_config);
        }
        applied_models.sort();
        applied_models.dedup();

        let current_catalog = self
            .catalog
            .read()
            .expect("Candle catalog lock poisoned")
            .clone();
        let loaded_before = managed_metrics::metrics_enabled().then(|| {
            self.loaded_embeddings
                .lock()
                .expect("Candle model lock poisoned")
                .keys()
        });
        self.loaded_embeddings
            .lock()
            .expect("Candle model lock poisoned")
            .retain(|model_id| {
                catalog_config_unchanged_for_model(model_id, &current_catalog, &next_catalog)
            });
        self.loading_embeddings
            .lock()
            .expect("Candle model loading lock poisoned")
            .retain(|model_id| {
                catalog_config_unchanged_for_model(model_id, &current_catalog, &next_catalog)
            });
        if let Some(loaded_before) = loaded_before {
            let loaded_after: HashSet<_> = self
                .loaded_embeddings
                .lock()
                .expect("Candle model lock poisoned")
                .keys()
                .into_iter()
                .collect();
            for evicted_model_id in loaded_before {
                if !loaded_after.contains(&evicted_model_id) {
                    self.record_model_residency_for_base(
                        &evicted_model_id,
                        false,
                        Some(ModelEvictionReason::ConfigChange),
                    );
                }
            }
        }
        *self.catalog.write().expect("Candle catalog lock poisoned") = next_catalog;
        self.refresh_residency_policies();
        self.reconcile_warm_models();
        Ok(applied_models)
    }

    pub async fn ensure_model_ready(&self, model_id: &str) -> EnsureModelReadyResponse {
        if !self.supports(model_id) {
            return EnsureModelReadyResponse {
                state: ReadinessState::RetryLater,
                batch_budget: None,
                descriptor: None,
            };
        }

        let runtime_config = {
            let guard = self.catalog.read().expect("Candle catalog lock poisoned");
            self.catalog_config_for(model_id, &guard).cloned()
        };
        let Some(runtime_config) = runtime_config else {
            return EnsureModelReadyResponse {
                state: ReadinessState::RetryLater,
                batch_budget: Some(self.config.batch_budget),
                descriptor: None,
            };
        };

        let defaults = self.runtime_defaults_for(model_id, None);
        let task_kind = runtime_config.task_kind;
        if !task_kind.supports_embedding() {
            return EnsureModelReadyResponse {
                state: ReadinessState::RetryLater,
                batch_budget: None,
                descriptor: None,
            };
        }

        match self.embedding_load_state(model_id) {
            Ok(ReadinessState::Ready) => {}
            Ok(state) => {
                return EnsureModelReadyResponse {
                    state,
                    batch_budget: None,
                    descriptor: None,
                };
            }
            Err(error) => {
                warn!(
                    model = %model_id,
                    error = %error_chain(&error),
                    "Candle model readiness failed to start embedding load"
                );
                return EnsureModelReadyResponse {
                    state: ReadinessState::RetryLater,
                    batch_budget: None,
                    descriptor: None,
                };
            }
        }

        let tokenizer_descriptor = self.loaded_tokenizer_descriptor(model_id);
        let Some((tokenizer_path, tokenizer_id, max_seq_len)) = tokenizer_descriptor else {
            return EnsureModelReadyResponse {
                state: ReadinessState::LoadingInProgress,
                batch_budget: None,
                descriptor: None,
            };
        };

        EnsureModelReadyResponse {
            state: ReadinessState::Ready,
            batch_budget: Some(self.config.batch_budget),
            descriptor: Some(ModelDescriptor {
                tokenizer_path: Some(tokenizer_path),
                tokenizer_id: Some(tokenizer_id),
                output_types: runtime_config.output_types(model_id, None),
                supports_run_batch: true,
                // The loaded model has already clamped the catalog request to
                // the checkpoint's positional-embedding capacity.
                max_seq_len: u32::try_from(max_seq_len).ok(),
                default_query_template: if task_kind.supports_embedding() {
                    defaults.query_template
                } else {
                    None
                },
                default_doc_template: if task_kind.supports_embedding() {
                    defaults.doc_template
                } else {
                    None
                },
            }),
        }
    }

    pub async fn process_encode_batch(&self, req: ProcessEncodeBatchRequest) -> BatchOutcome {
        let ProcessEncodeBatchRequest {
            model_id,
            items,
            accepts_batched_f16_multivectors,
        } = req;
        self.encode_items(&model_id, items, accepts_batched_f16_multivectors)
            .await
    }

    pub async fn process_score_batch(&self, req: ProcessScoreBatchRequest) -> BatchOutcome {
        self.score_items(&req.model_id, req.items).await
    }

    pub async fn run_batch(&self, req: RunBatchRequest) -> BatchOutcome {
        let model_id = req.model_id;
        let accepts_batched_f16_multivectors = req.accepts_batched_f16_multivectors;
        let mut encode_items = Vec::with_capacity(req.items.len());
        let mut encode_positions = Vec::with_capacity(req.items.len());
        let mut score_items = Vec::with_capacity(req.items.len());
        let mut score_positions = Vec::with_capacity(req.items.len());
        let mut outcomes: Vec<Option<ItemOutcome>> = vec![None; req.items.len()];

        for (position, item) in req.items.into_iter().enumerate() {
            match item.op.as_str() {
                "encode" => match item.encode {
                    Some(encode) => {
                        encode_items.push(encode);
                        encode_positions.push(position);
                    }
                    None => {
                        outcomes[position] = Some(identity_error_outcome(
                            &item.work_item_id,
                            &item.request_id,
                            item.item_index,
                            "candle_invalid_batch_item",
                            "RunBatch encode item is missing encode payload",
                        ))
                    }
                },
                "score" => match item.score {
                    Some(score) => {
                        score_items.push(score);
                        score_positions.push(position);
                    }
                    None => {
                        outcomes[position] = Some(identity_error_outcome(
                            &item.work_item_id,
                            &item.request_id,
                            item.item_index,
                            "candle_invalid_batch_item",
                            "RunBatch score item is missing score payload",
                        ))
                    }
                },
                _ => {
                    outcomes[position] = Some(identity_error_outcome(
                        &item.work_item_id,
                        &item.request_id,
                        item.item_index,
                        "candle_unsupported_operation",
                        "Candle Rust worker supports encode and score batches",
                    ))
                }
            }
        }

        let mut batched_f16_multivectors = Vec::new();
        if !encode_items.is_empty() {
            let encoded = self
                .encode_items(&model_id, encode_items, accepts_batched_f16_multivectors)
                .await;
            batched_f16_multivectors = encoded.batched_f16_multivectors;
            for (position, outcome) in encode_positions.into_iter().zip(encoded.outcomes) {
                outcomes[position] = Some(outcome);
            }
        }
        if !score_items.is_empty() {
            let scored = self.score_items(&model_id, score_items).await.outcomes;
            for (position, outcome) in score_positions.into_iter().zip(scored) {
                outcomes[position] = Some(outcome);
            }
        }

        BatchOutcome {
            outcomes: outcomes
                .into_iter()
                .enumerate()
                .map(|(idx, outcome)| {
                    outcome.unwrap_or_else(|| {
                        identity_error_outcome(
                            "",
                            "",
                            idx as u32,
                            "candle_internal_error",
                            "missing Candle run_batch outcome",
                        )
                    })
                })
                .collect(),
            batched_f16_multivectors,
        }
    }

    async fn encode_items(
        &self,
        model_id: &str,
        items: Vec<EncodeBatchItem>,
        accepts_batched_f16_multivectors: bool,
    ) -> BatchOutcome {
        let mut outcomes: Vec<Option<ItemOutcome>> = vec![None; items.len()];
        let mut batched_f16_multivectors = Vec::new();
        let mut groups: BTreeMap<EncodeGroupKey, PreparedEncodeGroup> = BTreeMap::new();
        let mut load_model_id: Option<String> = None;
        for (position, item) in items.iter().enumerate() {
            if let Some(message) = profile_option_error(item.options.as_ref()) {
                outcomes[position] =
                    Some(error_outcome(item, "candle_unsupported_request", &message));
                continue;
            }
            let requested_profile =
                requested_item_profile(item.options.as_ref(), item.profile_id.as_deref());
            let Some(effective_model_id) = effective_model_id(model_id, requested_profile)
                .filter(|model_id| self.supports(model_id))
            else {
                outcomes[position] = Some(error_outcome(
                    item,
                    "candle_unsupported_model",
                    "model is not supported by this Rust Candle worker",
                ));
                continue;
            };
            let options = item.options.as_ref();
            if let Some(message) = output_dtype_type_error(options) {
                outcomes[position] = Some(error_outcome(
                    item,
                    "candle_unsupported_output_dtype",
                    &message,
                ));
                continue;
            }
            let option_output_types = option_output_types(options);
            let requested_output_types = option_output_types
                .as_deref()
                .or(item.output_types.as_deref());
            let Some(output_kind) =
                self.encode_output_kind(&effective_model_id, requested_output_types)
            else {
                outcomes[position] = Some(error_outcome(
                    item,
                    "candle_unsupported_request",
                    "Candle Rust worker supports native dense, sparse SPLADE, or ColBERT multivector text encode with SIE text-prep options only",
                ));
                continue;
            };
            let defaults = self.runtime_defaults_for(&effective_model_id, requested_profile);
            let output_dtype = option_str(options, "output_dtype")
                .map(str::to_string)
                .or(defaults.output_dtype.clone());
            if let Some(message) = output_dtype_error_for_kind(output_dtype.as_deref(), output_kind)
            {
                outcomes[position] = Some(error_outcome(
                    item,
                    "candle_unsupported_output_dtype",
                    &message,
                ));
                continue;
            }
            if !self.supports_encode_options(&effective_model_id, item) {
                outcomes[position] = Some(error_outcome(
                    item,
                    "candle_unsupported_request",
                    "Candle Rust worker supports native text encode with SIE text-prep options only",
                ));
                continue;
            }
            if let Err(message) = item_id(&item.item) {
                outcomes[position] = Some(error_outcome(item, "candle_invalid_item", message));
                continue;
            }
            let raw_text = match extract_text(&item.item) {
                Ok(text) => text,
                Err(message) => {
                    outcomes[position] = Some(error_outcome(item, "candle_invalid_item", message));
                    continue;
                }
            };

            let default_instruction = defaults.default_instruction.as_deref();
            let instruction =
                resolve_instruction(item.instruction.as_deref(), options, default_instruction);
            let query_template = option_str(options, "query_template")
                .map(str::to_string)
                .or(defaults.query_template);
            let doc_template = option_str(options, "doc_template")
                .map(str::to_string)
                .or(defaults.doc_template);
            let prepared_tokens = if instruction.is_none() {
                item.prepared_tokens.clone()
            } else {
                None
            };
            let pooling = option_str(options, "pooling")
                .map(str::to_string)
                .or(defaults.pooling)
                .unwrap_or_else(|| match output_kind {
                    EncodeOutputKind::Sparse => "splade".to_string(),
                    EncodeOutputKind::Dense | EncodeOutputKind::Multivector => "mean".to_string(),
                });
            let normalize = option_bool(options, "normalize")
                .or(defaults.normalize)
                .unwrap_or_else(|| default_normalize(output_kind, self.config.normalize));
            match output_kind {
                EncodeOutputKind::Sparse if pooling != "splade" => {
                    outcomes[position] = Some(error_outcome(
                        item,
                        "candle_unsupported_pooling",
                        "Candle Rust worker requires splade pooling for native sparse embeddings",
                    ));
                    continue;
                }
                EncodeOutputKind::Sparse if normalize => {
                    outcomes[position] = Some(error_outcome(
                        item,
                        "candle_unsupported_request",
                        "Candle Rust worker requires normalize=false for native sparse embeddings",
                    ));
                    continue;
                }
                EncodeOutputKind::Dense | EncodeOutputKind::Multivector
                    if !matches!(pooling.as_str(), "mean" | "cls") =>
                {
                    outcomes[position] = Some(error_outcome(
                        item,
                        "candle_unsupported_pooling",
                        "Candle Rust worker supports mean and cls pooling for native dense and multivector embeddings",
                    ));
                    continue;
                }
                _ => {}
            }

            load_model_id.get_or_insert(effective_model_id);
            let output_wire_dtype = match output_kind {
                EncodeOutputKind::Dense | EncodeOutputKind::Sparse => "float32",
                EncodeOutputKind::Multivector => multivector_wire_dtype(output_dtype.as_deref()),
            }
            .to_string();
            let text = TextPrep {
                instruction,
                is_query: item.is_query,
                query_template: query_template.as_deref(),
                doc_template: doc_template.as_deref(),
            }
            .apply(raw_text);

            groups
                .entry((
                    output_kind,
                    pooling,
                    normalize,
                    item.is_query,
                    output_wire_dtype,
                ))
                .or_default()
                .push((
                    position,
                    PreparedEncodeItem {
                        work_item_id: item.work_item_id.clone(),
                        request_id: item.request_id.clone(),
                        item_index: item.item_index,
                        text,
                        is_query: item.is_query,
                        output_dtype,
                        prepared_tokens,
                    },
                ));
        }

        let loaded_model = if groups.is_empty() {
            None
        } else {
            let load_model_id = load_model_id
                .as_deref()
                .expect("load_model_id is set when encode groups are non-empty");
            match self.embedding_model(load_model_id) {
                Ok(loaded_model) => Some(loaded_model),
                Err(error) => {
                    self.mark_model_load_error_outcomes(
                        load_model_id,
                        &groups,
                        &error,
                        &mut outcomes,
                    );
                    None
                }
            }
        };

        let Some(loaded_model_use) = loaded_model else {
            return BatchOutcome {
                outcomes: outcomes
                    .into_iter()
                    .enumerate()
                    .map(|(idx, outcome)| {
                        outcome.unwrap_or_else(|| {
                            identity_error_outcome(
                                "",
                                "",
                                idx as u32,
                                "candle_internal_error",
                                "missing Candle encode outcome",
                            )
                        })
                    })
                    .collect(),
                batched_f16_multivectors,
            };
        };
        let active_model_id = loaded_model_use.model_id.clone();
        let loaded_model = Arc::clone(&loaded_model_use.model);
        let telemetry_model_id = load_model_id.as_deref().unwrap_or(model_id);

        for ((output_kind, pooling, normalize, is_query, output_dtype), group) in groups {
            self.encode_group_with_oom_recovery(
                EncodeGroupRecoveryContext {
                    model_id,
                    telemetry_model_id,
                    active_model_id: &active_model_id,
                    output_kind,
                    pooling,
                    normalize,
                    is_query,
                    output_dtype,
                    accepts_batched_f16_multivectors,
                },
                Arc::clone(&loaded_model),
                group,
                &mut outcomes,
                &mut batched_f16_multivectors,
            )
            .await;
        }

        BatchOutcome {
            outcomes: outcomes
                .into_iter()
                .enumerate()
                .map(|(idx, outcome)| {
                    outcome.unwrap_or_else(|| {
                        identity_error_outcome(
                            "",
                            "",
                            idx as u32,
                            "candle_internal_error",
                            "missing Candle encode outcome",
                        )
                    })
                })
                .collect(),
            batched_f16_multivectors,
        }
    }

    fn mark_model_load_error_outcomes(
        &self,
        load_model_id: &str,
        groups: &BTreeMap<EncodeGroupKey, PreparedEncodeGroup>,
        error: &anyhow::Error,
        outcomes: &mut [Option<ItemOutcome>],
    ) {
        if is_oom_error(error) {
            let retry_outcomes = groups.values().map(Vec::len).sum::<usize>();
            warn!(
                model_id = load_model_id,
                retry_outcomes,
                nak_delay_ms = self.config.oom_recovery.nak_delay_ms,
                error = %error_chain(error),
                "Candle embedding model load exhausted OOM recovery; marking encode items for retry"
            );
        }
        for group in groups.values() {
            for (position, item) in group {
                outcomes[*position] = Some(model_load_error_outcome(
                    item,
                    error,
                    self.config.oom_recovery.nak_delay_ms,
                ));
            }
        }
    }

    async fn encode_group_with_oom_recovery(
        &self,
        recovery_context: EncodeGroupRecoveryContext<'_>,
        loaded_model: Arc<LoadedEmbeddingModel>,
        group: PreparedEncodeGroup,
        outcomes: &mut [Option<ItemOutcome>],
        batched_f16_multivectors: &mut Vec<BatchedF16MultivectorOutput>,
    ) {
        let context = EncodeGroupContext {
            model_id: recovery_context.model_id,
            output_kind: recovery_context.output_kind,
            pooling: &recovery_context.pooling,
            normalize: recovery_context.normalize,
            is_query: recovery_context.is_query,
            output_dtype: &recovery_context.output_dtype,
            accepts_batched_f16_multivectors: recovery_context.accepts_batched_f16_multivectors,
        };
        let (mut pending_group, mut last_error, recovery_config) = match self
            .run_and_apply_encode_group(
                context,
                Arc::clone(&loaded_model),
                group,
                outcomes,
                batched_f16_multivectors,
            )
            .await
        {
            EncodeGroupOutcome::Success { .. } | EncodeGroupOutcome::Failed => return,
            EncodeGroupOutcome::Oom { group, error } => {
                let config = self.config.oom_recovery.clone();
                if !config.enabled {
                    self.mark_oom_group_retry(context, group, error, config.nak_delay_ms, outcomes);
                    return;
                }

                warn!(
                    model_id = context.model_id,
                    group_items = group.len(),
                    pooling = context.pooling,
                    normalize = context.normalize,
                    is_query = context.is_query,
                    error = %error_chain(&error),
                    "Candle embedding inference hit OOM; attempting recovery"
                );
                (group, error, config)
            }
        };

        let mut recovery_action_cursor = 0usize;
        let mut pending_oom_strategy: Option<OomStrategy> = None;
        loop {
            let strategy = match next_oom_recovery_step(
                &recovery_config.strategy,
                &mut recovery_action_cursor,
                || {
                    let evicted = self.evict_lru_embedding_model_excluding_with_reason(
                        recovery_context.active_model_id,
                        ModelEvictionReason::OomRecovery,
                    );
                    if evicted.is_none() {
                        info!(
                            model_id = context.model_id,
                            active_model_id = recovery_context.active_model_id,
                            "Candle OOM recovery found no LRU sibling model to evict"
                        );
                    }
                    evicted
                },
            ) {
                OomRecoveryStep::RetryAfterCacheClear => {
                    if let Some(previous) = pending_oom_strategy.take() {
                        self.record_oom_recovery(
                            recovery_context.telemetry_model_id,
                            previous,
                            OomOutcome::Failed,
                        );
                    }
                    OomStrategy::CacheClear
                }
                OomRecoveryStep::RetryAfterEviction(evicted_model_id) => {
                    if let Some(previous) = pending_oom_strategy.take() {
                        self.record_oom_recovery(
                            recovery_context.telemetry_model_id,
                            previous,
                            OomOutcome::Failed,
                        );
                    }
                    warn!(
                        model_id = context.model_id,
                        active_model_id = recovery_context.active_model_id,
                        evicted_model_id = %evicted_model_id,
                        error = %error_chain(&last_error),
                        "Candle OOM recovery evicted LRU sibling model and retrying"
                    );
                    OomStrategy::EvictLru
                }
                OomRecoveryStep::SplitBatch => {
                    if let Some(previous) = pending_oom_strategy.take() {
                        self.record_oom_recovery(
                            recovery_context.telemetry_model_id,
                            previous,
                            OomOutcome::Failed,
                        );
                    }
                    let (succeeded, oom_failed) = self
                        .encode_group_split_recovery(
                            context,
                            Arc::clone(&loaded_model),
                            pending_group,
                            &recovery_config,
                            outcomes,
                            batched_f16_multivectors,
                        )
                        .await;
                    let outcome = if oom_failed > 0 {
                        OomOutcome::Terminal
                    } else if succeeded > 0 {
                        OomOutcome::Success
                    } else {
                        OomOutcome::Failed
                    };
                    self.record_oom_recovery(
                        recovery_context.telemetry_model_id,
                        OomStrategy::SplitBatch,
                        outcome,
                    );
                    return;
                }
                OomRecoveryStep::Terminal => {
                    self.record_oom_recovery(
                        recovery_context.telemetry_model_id,
                        pending_oom_strategy.unwrap_or(OomStrategy::Other),
                        OomOutcome::Terminal,
                    );
                    self.mark_oom_group_retry(
                        context,
                        pending_group,
                        last_error,
                        recovery_config.nak_delay_ms,
                        outcomes,
                    );
                    return;
                }
            };

            match self
                .run_and_apply_encode_group(
                    context,
                    Arc::clone(&loaded_model),
                    pending_group,
                    outcomes,
                    batched_f16_multivectors,
                )
                .await
            {
                EncodeGroupOutcome::Success { .. } => {
                    self.record_oom_recovery(
                        recovery_context.telemetry_model_id,
                        strategy,
                        OomOutcome::Success,
                    );
                    return;
                }
                EncodeGroupOutcome::Failed => {
                    self.record_oom_recovery(
                        recovery_context.telemetry_model_id,
                        strategy,
                        OomOutcome::Failed,
                    );
                    return;
                }
                EncodeGroupOutcome::Oom { group, error } => {
                    pending_group = group;
                    last_error = error;
                    pending_oom_strategy = Some(strategy);
                }
            };
        }
    }

    async fn encode_group_split_recovery(
        &self,
        context: EncodeGroupContext<'_>,
        loaded_model: Arc<LoadedEmbeddingModel>,
        group: PreparedEncodeGroup,
        config: &CandleOomRecoveryConfig,
        outcomes: &mut [Option<ItemOutcome>],
        batched_f16_multivectors: &mut Vec<BatchedF16MultivectorOutput>,
    ) -> (usize, usize) {
        let mut stack = vec![(group, 0usize)];
        let mut succeeded = 0usize;
        let mut oom_failed = 0usize;

        while let Some((group, depth)) = stack.pop() {
            let group_items = group.len();
            match self
                .run_and_apply_encode_group(
                    context,
                    Arc::clone(&loaded_model),
                    group,
                    outcomes,
                    batched_f16_multivectors,
                )
                .await
            {
                EncodeGroupOutcome::Success { items } => {
                    succeeded += items;
                }
                EncodeGroupOutcome::Failed => {}
                EncodeGroupOutcome::Oom { group, error } => {
                    if group_items <= 1 || depth >= config.max_split_depth {
                        oom_failed += group_items;
                        self.mark_oom_group_retry(
                            context,
                            group,
                            error,
                            config.nak_delay_ms,
                            outcomes,
                        );
                        continue;
                    }

                    let (left, right) = split_encode_group(group);
                    stack.push((right, depth + 1));
                    stack.push((left, depth + 1));
                }
            }
        }

        (succeeded, oom_failed)
    }

    async fn run_and_apply_encode_group(
        &self,
        context: EncodeGroupContext<'_>,
        loaded_model: Arc<LoadedEmbeddingModel>,
        group: PreparedEncodeGroup,
        outcomes: &mut [Option<ItemOutcome>],
        batched_f16_multivectors: &mut Vec<BatchedF16MultivectorOutput>,
    ) -> EncodeGroupOutcome {
        match self
            .run_encode_group_attempt(context, loaded_model, group)
            .await
        {
            EncodeGroupAttempt::Executed { group, execution } => self.apply_encode_group_execution(
                context,
                group,
                *execution,
                outcomes,
                batched_f16_multivectors,
            ),
            EncodeGroupAttempt::FailedBeforeExecution {
                identities,
                message,
            } => {
                for item in identities {
                    outcomes[item.position] = Some(identity_error_outcome(
                        &item.work_item_id,
                        &item.request_id,
                        item.item_index,
                        "candle_inference_failed",
                        &message,
                    ));
                }
                EncodeGroupOutcome::Failed
            }
        }
    }

    async fn run_encode_group_attempt(
        &self,
        context: EncodeGroupContext<'_>,
        loaded_model: Arc<LoadedEmbeddingModel>,
        group: PreparedEncodeGroup,
    ) -> EncodeGroupAttempt {
        let fallback_items = encode_item_identities(&group);
        let output_path = forward_output_path(context);
        let telemetry_dimensions = managed_metrics::metrics_enabled()
            .then(|| self.telemetry_dimensions(context.model_id, None));
        let permit_started = telemetry_dimensions.as_ref().map(|_| Instant::now());
        let waiting_guard = telemetry_dimensions.as_ref().map(|(model, profile)| {
            managed_metrics::begin_forward_activity(
                model,
                profile,
                ForwardState::Waiting,
                self.config.max_concurrent_forwards,
            )
        });
        let slot_result = loaded_model.acquire_forward_slot().await;
        drop(waiting_guard);
        if let (Some((model, profile)), Some(started)) =
            (telemetry_dimensions.as_ref(), permit_started)
        {
            managed_metrics::record_forward_permit_wait(
                model,
                profile,
                output_path,
                started.elapsed().as_secs_f64(),
            );
        }
        let slot = match slot_result {
            Ok(slot) => slot,
            Err(error) => {
                error!(
                    model_id = context.model_id,
                    pooling = context.pooling,
                    normalize = context.normalize,
                    error = %error,
                    "Candle embedding forward permit acquisition failed"
                );
                return EncodeGroupAttempt::FailedBeforeExecution {
                    identities: fallback_items,
                    message: format!("Candle embedding forward permit acquisition failed: {error}"),
                };
            }
        };
        let model_for_task = Arc::clone(&loaded_model);
        let task_context = EncodeGroupModelContext {
            loaded: model_for_task,
            slot,
            output_kind: context.output_kind,
            pooling: context.pooling.to_string(),
            normalize: context.normalize,
            is_query: context.is_query,
            output_dtype: context.output_dtype.to_string(),
        };
        let active_guard = telemetry_dimensions.as_ref().map(|(model, profile)| {
            managed_metrics::begin_forward_activity(
                model,
                profile,
                ForwardState::Active,
                self.config.max_concurrent_forwards,
            )
        });
        let joined =
            tokio::task::spawn_blocking(move || encode_group_on_model(task_context, group)).await;
        drop(active_guard);

        match joined {
            Ok((group, execution)) => EncodeGroupAttempt::Executed {
                group,
                execution: Box::new(execution),
            },
            Err(error) => {
                error!(
                    model_id = context.model_id,
                    pooling = context.pooling,
                    normalize = context.normalize,
                    error = %error,
                    "Candle embedding worker task failed"
                );
                EncodeGroupAttempt::FailedBeforeExecution {
                    identities: fallback_items,
                    message: format!("Candle embedding worker task failed: {error}"),
                }
            }
        }
    }

    fn apply_encode_group_execution(
        &self,
        context: EncodeGroupContext<'_>,
        group: PreparedEncodeGroup,
        execution: EncodeGroupExecution,
        outcomes: &mut [Option<ItemOutcome>],
        batched_f16_multivectors: &mut Vec<BatchedF16MultivectorOutput>,
    ) -> EncodeGroupOutcome {
        let group_stats = execution.stats;
        maybe_log_slow_encode_group(SlowEncodeLogContext {
            model_id: context.model_id,
            pooling: context.pooling,
            normalize: context.normalize,
            stats: &group_stats,
            first_item: group.first().map(|(_, item)| item),
            ok: execution.encoded.is_ok(),
            lock_wait_ms: execution.lock_wait_ms,
            encode_elapsed_ms: execution.encode_elapsed_ms,
            encode_source: execution.encode_source,
        });
        if candle_backend_diagnostics_enabled() {
            info!(
                model_id = context.model_id,
                group_items = group_stats.items,
                pooling = context.pooling,
                normalize = context.normalize,
                lock_wait_ms = execution.lock_wait_ms,
                encode_elapsed_ms = execution.encode_elapsed_ms,
                encode_source = execution.encode_source,
                prepared_items = group_stats.prepared_items,
                prepared_sequences = group_stats.prepared_sequences,
                prepared_tokens_total = group_stats.prepared_tokens_total,
                prepared_tokens_min = group_stats.prepared_tokens_min,
                prepared_tokens_max = group_stats.prepared_tokens_max,
                prepared_max_seq_len = group_stats.prepared_max_seq_len,
                text_chars_total = group_stats.text_chars_total,
                text_chars_min = group_stats.text_chars_min,
                text_chars_max = group_stats.text_chars_max,
                "Candle embedding backend diagnostics"
            );
        }
        if managed_metrics::metrics_enabled() {
            let (model, profile) = self.telemetry_dimensions(context.model_id, None);
            let outcome = if execution.encoded.is_ok() {
                ForwardOutcome::Success
            } else {
                ForwardOutcome::Error
            };
            let stages = execution
                .encoded
                .as_ref()
                .map(|encoded| {
                    forward_stage_durations(
                        encoded.stages,
                        encoded.forward_profile.as_deref().copied(),
                    )
                })
                .unwrap_or_default();
            managed_metrics::record_forward_completed(ForwardCompleted {
                model: &model,
                profile: &profile,
                outcome,
                input_source: forward_input_source(execution.encode_source),
                output_path: forward_output_path(context),
                duration_s: execution.encode_elapsed_ms / 1_000.0,
                stages: &stages,
            });
        }
        match execution.encoded {
            Ok(encoded) => {
                let items = group.len();
                let CandleEncodeResult {
                    embeddings,
                    sparse_embeddings,
                    multivectors,
                    multivectors_f16,
                    dim,
                    tokenization_ms,
                    inference_ms,
                    ..
                } = encoded;
                match context.output_kind {
                    EncodeOutputKind::Dense => {
                        for ((position, item), values) in group.into_iter().zip(embeddings) {
                            outcomes[position] = Some(success_dense_outcome(
                                &item,
                                values,
                                dim,
                                context.normalize,
                                inference_ms,
                                tokenization_ms,
                            ));
                        }
                    }
                    EncodeOutputKind::Sparse => {
                        let Some(sparse_embeddings) = sparse_embeddings else {
                            for (position, item) in group {
                                outcomes[position] = Some(identity_error_outcome(
                                    &item.work_item_id,
                                    &item.request_id,
                                    item.item_index,
                                    "candle_inference_failed",
                                    "Candle sparse encode returned no sparse embeddings",
                                ));
                            }
                            return EncodeGroupOutcome::Failed;
                        };
                        if sparse_embeddings.len() != group.len() {
                            for (position, item) in group {
                                outcomes[position] = Some(identity_error_outcome(
                                    &item.work_item_id,
                                    &item.request_id,
                                    item.item_index,
                                    "candle_inference_failed",
                                    "Candle sparse item count did not match encode group",
                                ));
                            }
                            return EncodeGroupOutcome::Failed;
                        }
                        for ((position, item), sparse) in group.into_iter().zip(sparse_embeddings) {
                            outcomes[position] = Some(success_sparse_outcome(
                                &item,
                                sparse,
                                dim,
                                inference_ms,
                                tokenization_ms,
                            ));
                        }
                    }
                    EncodeOutputKind::Multivector => {
                        if let Some(batch) = multivectors_f16 {
                            let CandleF16MultivectorBatch {
                                values_f16,
                                items: f16_items,
                            } = batch;
                            if f16_items.len() != group.len() {
                                for (position, item) in group {
                                    outcomes[position] = Some(identity_error_outcome(
                                        &item.work_item_id,
                                        &item.request_id,
                                        item.item_index,
                                        "candle_inference_failed",
                                        "Candle f16 multivector item count did not match encode group",
                                    ));
                                }
                                return EncodeGroupOutcome::Failed;
                            }

                            if context.accepts_batched_f16_multivectors {
                                let mut wire_items = Vec::with_capacity(f16_items.len());
                                for ((position, item), f16_item) in group.into_iter().zip(f16_items)
                                {
                                    outcomes[position] =
                                        Some(success_batched_f16_multivector_outcome(
                                            &item,
                                            inference_ms,
                                            tokenization_ms,
                                        ));
                                    let Ok(byte_offset) = u64::try_from(f16_item.byte_offset)
                                    else {
                                        outcomes[position] = Some(identity_error_outcome(
                                            &item.work_item_id,
                                            &item.request_id,
                                            item.item_index,
                                            "candle_inference_failed",
                                            "Candle f16 multivector offset exceeded wire range",
                                        ));
                                        continue;
                                    };
                                    let Ok(byte_len) = u64::try_from(f16_item.byte_len) else {
                                        outcomes[position] = Some(identity_error_outcome(
                                            &item.work_item_id,
                                            &item.request_id,
                                            item.item_index,
                                            "candle_inference_failed",
                                            "Candle f16 multivector length exceeded wire range",
                                        ));
                                        continue;
                                    };
                                    wire_items.push(BatchedF16MultivectorItem {
                                        work_item_id: item.work_item_id,
                                        byte_offset,
                                        byte_len,
                                        num_tokens: f16_item.num_tokens,
                                        token_dims: f16_item.token_dims,
                                    });
                                }
                                batched_f16_multivectors.push(BatchedF16MultivectorOutput {
                                    values_f16: F16Values(values_f16),
                                    items: wire_items,
                                });
                            } else {
                                let batch = CandleF16MultivectorBatch {
                                    values_f16,
                                    items: f16_items,
                                };
                                let legacy = match batch.into_individual() {
                                    Ok(legacy) => legacy,
                                    Err(error) => {
                                        for (position, item) in group {
                                            outcomes[position] = Some(identity_error_outcome(
                                                &item.work_item_id,
                                                &item.request_id,
                                                item.item_index,
                                                "candle_inference_failed",
                                                &format!("Candle f16 multivector conversion failed: {error}"),
                                            ));
                                        }
                                        return EncodeGroupOutcome::Failed;
                                    }
                                };
                                for ((position, item), multivector) in group.into_iter().zip(legacy)
                                {
                                    outcomes[position] = Some(success_multivector_outcome(
                                        &item,
                                        multivector,
                                        item.output_dtype.as_deref(),
                                        inference_ms,
                                        tokenization_ms,
                                    ));
                                }
                            }
                            return EncodeGroupOutcome::Success { items };
                        }

                        let Some(multivectors) = multivectors else {
                            for (position, item) in group {
                                outcomes[position] = Some(identity_error_outcome(
                                    &item.work_item_id,
                                    &item.request_id,
                                    item.item_index,
                                    "candle_inference_failed",
                                    "Candle multivector encode returned no multivectors",
                                ));
                            }
                            return EncodeGroupOutcome::Failed;
                        };
                        for ((position, item), multivector) in group.into_iter().zip(multivectors) {
                            outcomes[position] = Some(success_multivector_outcome(
                                &item,
                                multivector,
                                item.output_dtype.as_deref(),
                                inference_ms,
                                tokenization_ms,
                            ));
                        }
                    }
                }
                EncodeGroupOutcome::Success { items }
            }
            Err(error) if is_oom_error(&error) => EncodeGroupOutcome::Oom { group, error },
            Err(error) => {
                error!(
                    model_id = context.model_id,
                    group_items = group_stats.items,
                    pooling = context.pooling,
                    normalize = context.normalize,
                    lock_wait_ms = execution.lock_wait_ms,
                    encode_elapsed_ms = execution.encode_elapsed_ms,
                    encode_source = execution.encode_source,
                    prepared_items = group_stats.prepared_items,
                    prepared_sequences = group_stats.prepared_sequences,
                    prepared_tokens_total = group_stats.prepared_tokens_total,
                    prepared_tokens_min = group_stats.prepared_tokens_min,
                    prepared_tokens_max = group_stats.prepared_tokens_max,
                    prepared_max_seq_len = group_stats.prepared_max_seq_len,
                    text_chars_total = group_stats.text_chars_total,
                    text_chars_min = group_stats.text_chars_min,
                    text_chars_max = group_stats.text_chars_max,
                    error = %error_chain(&error),
                    "Candle embedding inference failed"
                );
                for (position, item) in group {
                    outcomes[position] = Some(identity_error_outcome(
                        &item.work_item_id,
                        &item.request_id,
                        item.item_index,
                        "candle_inference_failed",
                        &format!("Candle embedding inference failed: {error}"),
                    ));
                }
                EncodeGroupOutcome::Failed
            }
        }
    }

    fn mark_oom_group_retry(
        &self,
        context: EncodeGroupContext<'_>,
        group: PreparedEncodeGroup,
        error: anyhow::Error,
        nak_delay_ms: u64,
        outcomes: &mut [Option<ItemOutcome>],
    ) {
        warn!(
            model_id = context.model_id,
            group_items = group.len(),
            pooling = context.pooling,
            normalize = context.normalize,
            nak_delay_ms,
            error = %error_chain(&error),
            "Candle OOM recovery exhausted; marking encode items for retry"
        );
        for (position, item) in group {
            outcomes[position] = Some(oom_nak_outcome(&item, nak_delay_ms));
        }
    }

    async fn score_items(&self, model_id: &str, items: Vec<ScoreBatchItem>) -> BatchOutcome {
        let mut outcomes = Vec::with_capacity(items.len());
        for item in items {
            if let Some(message) = profile_option_error(item.options.as_ref()) {
                outcomes.push(score_error_outcome(
                    &item,
                    "candle_unsupported_request",
                    &message,
                ));
                continue;
            }
            let requested_profile =
                requested_item_profile(item.options.as_ref(), item.profile_id.as_deref());
            let Some(effective_model_id) = effective_model_id(model_id, requested_profile)
                .filter(|model_id| self.supports(model_id))
            else {
                outcomes.push(score_error_outcome(
                    &item,
                    "candle_unsupported_model",
                    "model is not supported by this Rust Candle worker",
                ));
                continue;
            };
            if !self.supports_score_item(&effective_model_id, &item) {
                outcomes.push(score_error_outcome(
                    &item,
                    "candle_unsupported_request",
                    "Candle Rust worker supports explicitly configured ColBERT MaxSim text score with Python-compatible encode runtime options",
                ));
                continue;
            }
            if let Err(message) = item_id(&item.query_item) {
                outcomes.push(score_error_outcome(
                    &item,
                    "candle_invalid_item",
                    &format!("Candle Rust worker score query {message}"),
                ));
                continue;
            }
            let query_text = match extract_text(&item.query_item) {
                Ok(text) => text,
                Err(message) => {
                    outcomes.push(score_error_outcome(
                        &item,
                        "candle_invalid_item",
                        &format!("Candle Rust worker score query {message}"),
                    ));
                    continue;
                }
            };
            let mut raw_docs = Vec::with_capacity(item.score_items.len());
            let mut item_ids = Vec::with_capacity(item.score_items.len());
            let mut invalid_doc = None;
            for (index, doc) in item.score_items.iter().enumerate() {
                let text = match extract_text(doc) {
                    Ok(text) => text,
                    Err(message) => {
                        invalid_doc = Some((index, message));
                        break;
                    }
                };
                let id = match score_item_id(doc, index) {
                    Ok(id) => id,
                    Err(message) => {
                        invalid_doc = Some((index, message));
                        break;
                    }
                };
                raw_docs.push(text);
                item_ids.push(id);
            }
            if let Some((index, message)) = invalid_doc {
                outcomes.push(score_error_outcome(
                    &item,
                    "candle_invalid_item",
                    &format!("Candle Rust worker score document {index} {message}"),
                ));
                continue;
            }

            let defaults = self.runtime_defaults_for(&effective_model_id, requested_profile);
            // Score ingress has already canonicalized request instruction
            // precedence into `ScoreBatchItem.instruction`. All encode-time
            // options, including a duplicate `options.instruction`, remain
            // accepted for rolling compatibility but are ignored here.
            let (query, docs, allow_prepared_tokens) =
                prepare_colbert_score_texts(query_text, &raw_docs, item.instruction.as_deref());
            // Late-interaction score is cosine MaxSim. Keep score-time
            // projection normalization invariant even if an encode caller
            // selects a profile that exposes raw multivectors.
            let normalize = true;
            let work_budget = defaults.max_batch_tokens.unwrap_or_else(|| {
                self.catalog
                    .read()
                    .expect("Candle catalog lock poisoned")
                    .get(base_model_id_for_residency(&effective_model_id))
                    .and_then(|config| config.max_sequence_length)
                    .unwrap_or(8192)
            });
            let prepared = Arc::new(PreparedScoreItem {
                query,
                docs,
                item_ids,
                normalize,
                work_budget: work_budget.max(1),
                prepared_tokens: item.prepared_tokens.clone(),
                allow_prepared_tokens,
            });

            if prepared.docs.is_empty() {
                outcomes.push(success_score_outcome(
                    &item,
                    &prepared.item_ids,
                    CandleScoreResult {
                        scores: Vec::new(),
                        query_tokens: 0,
                        doc_tokens: Vec::new(),
                        tokenization_ms: 0.0,
                        inference_ms: 0.0,
                        maxsim_ms: None,
                    },
                ));
                continue;
            }

            let loaded_model_use = match self.embedding_model(&effective_model_id) {
                Ok(loaded) => loaded,
                Err(error) => {
                    outcomes.push(score_model_load_error_outcome(
                        &item,
                        &error,
                        self.config.oom_recovery.nak_delay_ms,
                    ));
                    continue;
                }
            };
            outcomes.push(
                self.score_item_with_oom_recovery(
                    &effective_model_id,
                    &loaded_model_use.model_id,
                    &item,
                    Arc::clone(&loaded_model_use.model),
                    prepared,
                )
                .await,
            );
        }

        BatchOutcome {
            outcomes,
            batched_f16_multivectors: Vec::new(),
        }
    }

    async fn score_item_with_oom_recovery(
        &self,
        model_id: &str,
        active_model_id: &str,
        item: &ScoreBatchItem,
        loaded_model: Arc<LoadedEmbeddingModel>,
        prepared: Arc<PreparedScoreItem>,
    ) -> ItemOutcome {
        let mut work_budget = prepared.work_budget;
        let mut execution = match self
            .run_score_item_attempt(
                model_id,
                Arc::clone(&loaded_model),
                Arc::clone(&prepared),
                work_budget,
            )
            .await
        {
            Ok(execution) => execution,
            Err(error) => {
                return score_error_outcome(
                    item,
                    "candle_inference_failed",
                    &format!("Candle MaxSim worker task failed: {error}"),
                );
            }
        };
        if !execution.scored.as_ref().err().is_some_and(is_oom_error) {
            return score_execution_outcome(item, &prepared.item_ids, execution);
        }

        let recovery = self.config.oom_recovery.clone();
        if !recovery.enabled {
            return score_oom_nak_outcome(item, recovery.nak_delay_ms);
        }
        warn!(
            model_id,
            documents = prepared.docs.len(),
            work_budget,
            error = %execution.scored.as_ref().unwrap_err(),
            "Candle MaxSim score hit OOM; attempting recovery"
        );

        let mut recovery_action_cursor = 0usize;
        let mut pending_oom_strategy: Option<OomStrategy> = None;
        loop {
            let strategy = match next_oom_recovery_step(
                &recovery.strategy,
                &mut recovery_action_cursor,
                || {
                    self.evict_lru_embedding_model_excluding_with_reason(
                        active_model_id,
                        ModelEvictionReason::OomRecovery,
                    )
                },
            ) {
                OomRecoveryStep::RetryAfterCacheClear => {
                    if let Some(previous) = pending_oom_strategy.take() {
                        self.record_oom_recovery(model_id, previous, OomOutcome::Failed);
                    }
                    OomStrategy::CacheClear
                }
                OomRecoveryStep::RetryAfterEviction(evicted_model_id) => {
                    if let Some(previous) = pending_oom_strategy.take() {
                        self.record_oom_recovery(model_id, previous, OomOutcome::Failed);
                    }
                    warn!(
                        model_id,
                        active_model_id,
                        evicted_model_id = %evicted_model_id,
                        "Candle MaxSim OOM recovery evicted LRU sibling model"
                    );
                    OomStrategy::EvictLru
                }
                OomRecoveryStep::SplitBatch => {
                    if let Some(previous) = pending_oom_strategy.take() {
                        self.record_oom_recovery(model_id, previous, OomOutcome::Failed);
                    }
                    let mut split_depth = 0usize;
                    while split_depth < recovery.max_split_depth && work_budget > 1 {
                        split_depth += 1;
                        work_budget = (work_budget / 2).max(1);
                        execution = match self
                            .run_score_item_attempt(
                                model_id,
                                Arc::clone(&loaded_model),
                                Arc::clone(&prepared),
                                work_budget,
                            )
                            .await
                        {
                            Ok(execution) => execution,
                            Err(error) => {
                                self.record_oom_recovery(
                                    model_id,
                                    OomStrategy::SplitBatch,
                                    OomOutcome::Failed,
                                );
                                return score_error_outcome(
                                    item,
                                    "candle_inference_failed",
                                    &format!("Candle MaxSim worker task failed: {error}"),
                                );
                            }
                        };
                        match &execution.scored {
                            Ok(_) => {
                                self.record_oom_recovery(
                                    model_id,
                                    OomStrategy::SplitBatch,
                                    OomOutcome::Success,
                                );
                                return score_execution_outcome(
                                    item,
                                    &prepared.item_ids,
                                    execution,
                                );
                            }
                            Err(error) if is_oom_error(error) => continue,
                            Err(_) => {
                                self.record_oom_recovery(
                                    model_id,
                                    OomStrategy::SplitBatch,
                                    OomOutcome::Failed,
                                );
                                return score_execution_outcome(
                                    item,
                                    &prepared.item_ids,
                                    execution,
                                );
                            }
                        }
                    }
                    self.record_oom_recovery(
                        model_id,
                        OomStrategy::SplitBatch,
                        OomOutcome::Terminal,
                    );
                    return score_oom_nak_outcome(item, recovery.nak_delay_ms);
                }
                OomRecoveryStep::Terminal => {
                    self.record_oom_recovery(
                        model_id,
                        pending_oom_strategy.unwrap_or(OomStrategy::Other),
                        OomOutcome::Terminal,
                    );
                    return score_oom_nak_outcome(item, recovery.nak_delay_ms);
                }
            };

            execution = match self
                .run_score_item_attempt(
                    model_id,
                    Arc::clone(&loaded_model),
                    Arc::clone(&prepared),
                    work_budget,
                )
                .await
            {
                Ok(execution) => execution,
                Err(error) => {
                    self.record_oom_recovery(model_id, strategy, OomOutcome::Failed);
                    return score_error_outcome(
                        item,
                        "candle_inference_failed",
                        &format!("Candle MaxSim worker task failed: {error}"),
                    );
                }
            };
            match &execution.scored {
                Ok(_) => {
                    self.record_oom_recovery(model_id, strategy, OomOutcome::Success);
                    return score_execution_outcome(item, &prepared.item_ids, execution);
                }
                Err(error) if is_oom_error(error) => {
                    pending_oom_strategy = Some(strategy);
                }
                Err(_) => {
                    self.record_oom_recovery(model_id, strategy, OomOutcome::Failed);
                    return score_execution_outcome(item, &prepared.item_ids, execution);
                }
            }
        }
    }

    async fn run_score_item_attempt(
        &self,
        model_id: &str,
        loaded_model: Arc<LoadedEmbeddingModel>,
        prepared: Arc<PreparedScoreItem>,
        work_budget: usize,
    ) -> Result<ScoreItemExecution> {
        let output_path = ForwardOutputPath::Other;
        let telemetry_dimensions =
            managed_metrics::metrics_enabled().then(|| self.telemetry_dimensions(model_id, None));
        let permit_started = telemetry_dimensions.as_ref().map(|_| Instant::now());
        let waiting_guard = telemetry_dimensions.as_ref().map(|(model, profile)| {
            managed_metrics::begin_forward_activity(
                model,
                profile,
                ForwardState::Waiting,
                self.config.max_concurrent_forwards,
            )
        });
        let slot_result = loaded_model.acquire_forward_slot().await;
        drop(waiting_guard);
        if let (Some((model, profile)), Some(started)) =
            (telemetry_dimensions.as_ref(), permit_started)
        {
            managed_metrics::record_forward_permit_wait(
                model,
                profile,
                output_path,
                started.elapsed().as_secs_f64(),
            );
        }
        let slot = slot_result?;
        let active_guard = telemetry_dimensions.as_ref().map(|(model, profile)| {
            managed_metrics::begin_forward_activity(
                model,
                profile,
                ForwardState::Active,
                self.config.max_concurrent_forwards,
            )
        });
        let wall_start = Instant::now();
        let execution = tokio::task::spawn_blocking(move || {
            loaded_model.with_model(slot, |model| {
                score_prepared_item_on_model(model, &prepared, work_budget)
            })
        })
        .await
        .context("join Candle MaxSim worker task")?;
        drop(active_guard);
        let wall_ms = elapsed_ms(wall_start);
        if let Some((model, profile)) = telemetry_dimensions {
            let outcome = if execution.result.is_ok() {
                ForwardOutcome::Success
            } else {
                ForwardOutcome::Error
            };
            managed_metrics::record_forward_completed(ForwardCompleted {
                model: &model,
                profile: &profile,
                outcome,
                input_source: ForwardInputSource::Other,
                output_path,
                duration_s: wall_ms / 1_000.0,
                stages: &[],
            });
        }
        Ok(ScoreItemExecution {
            scored: execution.result,
            lock_wait_ms: execution.lock_wait_ms,
            wall_ms,
        })
    }

    fn embedding_model(&self, model_id: &str) -> Result<LoadedEmbeddingModelUse> {
        let base_model_id = self
            .base_model_for_supported_route(model_id)
            .with_context(|| format!("model {model_id:?} is not supported by this Candle worker"))?
            .to_string();
        let runtime_config = {
            let guard = self.catalog.read().expect("Candle catalog lock poisoned");
            self.catalog_config_for(model_id, &guard)
                .cloned()
                .unwrap_or_else(|| ModelRuntimeConfig {
                    hf_id: base_model_id.clone(),
                    hf_revision: None,
                    max_sequence_length: None,
                    query_max_length: None,
                    dense_dim: None,
                    sparse_dim: None,
                    multivector_dim: None,
                    compute_precision: None,
                    profiles: HashMap::new(),
                    routable_model_ids: vec![base_model_id.clone()],
                    task_kind: CandleTaskKind::Embedding,
                })
        };
        if !runtime_config.task_kind.supports_embedding() {
            anyhow::bail!("model {base_model_id:?} is not a Candle embedding model");
        }

        if let Some(model) = self.checkout_loaded_embedding_model(&base_model_id) {
            return Ok(model);
        }

        loop {
            if let Some(model) = self.checkout_loaded_embedding_model(&base_model_id) {
                return Ok(model);
            }
            if self.mark_embedding_loading(&base_model_id) {
                break;
            }
            std::thread::sleep(MODEL_LOAD_WAIT_INTERVAL);
        }
        let load_telemetry = managed_metrics::metrics_enabled().then(|| {
            let (model, profile) = self.telemetry_dimensions(model_id, None);
            (Instant::now(), model, profile)
        });

        let model_config = CandleEmbeddingModelConfig {
            model_id: base_model_id.clone(),
            hf_id: runtime_config.hf_id.clone(),
            hf_revision: runtime_config.hf_revision.clone(),
            max_seq_length: runtime_config.max_sequence_length.unwrap_or(512),
            query_max_length: runtime_config.query_max_length,
            dense_dim: runtime_config.dense_dim,
            sparse_dim: runtime_config.native_sparse_dim(),
            multivector_dim: runtime_config.native_multivector_dim(),
            compute_precision: runtime_config.compute_precision.clone(),
        };
        let load_once = || {
            let model = CandleEmbeddingModel::load(&model_config)
                .with_context(|| format!("load Candle embedding model {base_model_id:?}"))?;
            let loaded_model = Arc::new(LoadedEmbeddingModel::new(
                model,
                self.config.max_concurrent_forwards,
            ));
            {
                let guard = self.catalog.read().expect("Candle catalog lock poisoned");
                match self.catalog_config_for(&base_model_id, &guard) {
                    Some(current_config) if current_config == &runtime_config => {}
                    Some(_) => {
                        anyhow::bail!("model {base_model_id:?} changed while loading");
                    }
                    None => {
                        anyhow::bail!("model {base_model_id:?} was removed while loading");
                    }
                }
            }
            let policy = self.residency_policy_for_base_model(&base_model_id);
            let mut loaded = self
                .loaded_embeddings
                .lock()
                .expect("Candle model lock poisoned");
            if let Some(model) = loaded.get_for_use(&base_model_id, Instant::now()) {
                return Ok(self.loaded_embedding_model_use(&base_model_id, model));
            }

            info!(
                model_id = %base_model_id,
                max_concurrent_forwards = self.config.max_concurrent_forwards,
                "loaded Candle embedding model"
            );
            loaded.insert(
                base_model_id.clone(),
                Arc::clone(&loaded_model),
                policy,
                Instant::now(),
            );
            let model = loaded
                .get_for_use(&base_model_id, Instant::now())
                .expect("loaded Candle embedding model was just inserted");
            Ok(self.loaded_embedding_model_use(&base_model_id, model))
        };
        let load_result = match load_once() {
            Ok(model) => Ok(model),
            Err(error) if is_oom_error(&error) => {
                if let Some(evicted_model_id) = self
                    .evict_lru_embedding_model_excluding_with_reason(
                        &base_model_id,
                        ModelEvictionReason::LoadOom,
                    )
                {
                    warn!(
                        model_id = %base_model_id,
                        evicted_model_id = %evicted_model_id,
                        error = %error_chain(&error),
                        "Candle embedding model load hit OOM; evicted LRU resident model and retrying"
                    );
                    load_once()
                } else {
                    Err(error)
                }
            }
            Err(error) => Err(error),
        };
        self.unmark_embedding_loading(&base_model_id);
        if let Some((started, model, profile)) = load_telemetry {
            let outcome = if load_result.is_ok() {
                ModelLoadOutcome::Success
            } else {
                ModelLoadOutcome::Error
            };
            managed_metrics::record_model_load_completed(
                &model,
                &profile,
                outcome,
                ModelLoadStage::Total,
                started.elapsed().as_secs_f64(),
            );
            if load_result.is_ok() {
                self.record_model_residency_for_base(&base_model_id, true, None);
            }
        }
        load_result
    }

    fn reconcile_warm_models(&self) {
        let mut warm_models: Vec<String> = self
            .preload_models
            .read()
            .expect("Candle preload-model lock poisoned")
            .iter()
            .cloned()
            .collect();
        warm_models.extend(
            self.pinned_models
                .read()
                .expect("Candle pinned-model lock poisoned")
                .iter()
                .cloned(),
        );
        warm_models.sort();
        warm_models.dedup();
        if warm_models.is_empty() {
            return;
        }

        let mut load_started = 0_u32;
        let mut already_resident_or_loading = 0_u32;
        let mut skipped = 0_u32;
        for model_id in warm_models {
            let task_kind = {
                let guard = self.catalog.read().expect("Candle catalog lock poisoned");
                self.catalog_config_for(&model_id, &guard)
                    .map(|config| config.task_kind)
            };
            if !task_kind.is_some_and(CandleTaskKind::supports_embedding) {
                skipped += 1;
                continue;
            }

            match self.embedding_load_state(&model_id) {
                Ok(ReadinessState::LoadingStarted) => {
                    load_started += 1;
                }
                Ok(ReadinessState::Ready | ReadinessState::LoadingInProgress) => {
                    already_resident_or_loading += 1;
                }
                // `RetryLater` (unknown/not-ready) and `Failed` (terminal
                // load failure) are both "don't eager-load now": a terminally
                // failed model must not be re-driven by the warm reconcile.
                Ok(ReadinessState::RetryLater | ReadinessState::Failed) => {
                    skipped += 1;
                }
                Err(error) => {
                    skipped += 1;
                    warn!(
                        model_id = %model_id,
                        error = %error_chain(&error),
                        "Candle warm model eager-load reconcile failed"
                    );
                }
            }
        }

        if load_started > 0 || skipped > 0 {
            info!(
                load_started,
                already_resident_or_loading, skipped, "Candle warm model reconcile complete"
            );
        }
    }

    fn loaded_embedding_model(&self, base_model_id: &str) -> Option<Arc<LoadedEmbeddingModel>> {
        self.loaded_embeddings
            .lock()
            .expect("Candle model lock poisoned")
            .get(base_model_id)
    }

    fn checkout_loaded_embedding_model(
        &self,
        base_model_id: &str,
    ) -> Option<LoadedEmbeddingModelUse> {
        let model = self
            .loaded_embeddings
            .lock()
            .expect("Candle model lock poisoned")
            .get_for_use(base_model_id, Instant::now())?;
        Some(self.loaded_embedding_model_use(base_model_id, model))
    }

    fn loaded_embedding_model_use(
        &self,
        base_model_id: &str,
        model: Arc<LoadedEmbeddingModel>,
    ) -> LoadedEmbeddingModelUse {
        LoadedEmbeddingModelUse {
            model_id: base_model_id.to_string(),
            model,
            _residency_use: ResidencyUseGuard::active(
                Arc::clone(&self.loaded_embeddings),
                base_model_id.to_string(),
            ),
        }
    }

    pub fn evict_lru_embedding_model_excluding(&self, exclude_model_id: &str) -> Option<String> {
        self.evict_lru_embedding_model_excluding_with_reason(
            exclude_model_id,
            ModelEvictionReason::Lru,
        )
    }

    fn evict_lru_embedding_model_excluding_with_reason(
        &self,
        exclude_model_id: &str,
        reason: ModelEvictionReason,
    ) -> Option<String> {
        let exclude_base_model_id = canonical_pinned_model_id(exclude_model_id)
            .map(|model_id| base_model_id_for_residency(&model_id).to_string())
            .unwrap_or_else(|| base_model_id_for_residency(exclude_model_id).to_string());
        let evicted = self
            .loaded_embeddings
            .lock()
            .expect("Candle model lock poisoned")
            .evict_lru_excluding(Some(&exclude_base_model_id));
        if let Some(evicted_model_id) = evicted.as_deref() {
            self.record_model_residency_for_base(evicted_model_id, false, Some(reason));
        }
        evicted
    }

    fn evict_idle_embedding_model(&self, idle_threshold: Duration) -> Option<String> {
        let evicted = self
            .loaded_embeddings
            .lock()
            .expect("Candle model lock poisoned")
            .evict_idle(idle_threshold, Instant::now());
        if let Some(evicted_model_id) = evicted.as_deref() {
            self.record_model_residency_for_base(
                evicted_model_id,
                false,
                Some(ModelEvictionReason::Idle),
            );
        }
        evicted
    }

    fn refresh_residency_policies(&self) {
        let preload = self
            .preload_models
            .read()
            .expect("Candle preload-model lock poisoned")
            .clone();
        let pinned = self
            .pinned_models
            .read()
            .expect("Candle pinned-model lock poisoned")
            .clone();
        self.loaded_embeddings
            .lock()
            .expect("Candle model lock poisoned")
            .update_policies(|base_model_id| {
                residency_policy_for_base_model(base_model_id, &preload, &pinned)
            });
    }

    fn residency_policy_for_base_model(&self, base_model_id: &str) -> ResidencyPolicy {
        let preload = self
            .preload_models
            .read()
            .expect("Candle preload-model lock poisoned");
        let pinned = self
            .pinned_models
            .read()
            .expect("Candle pinned-model lock poisoned");
        residency_policy_for_base_model(base_model_id, &preload, &pinned)
    }

    fn embedding_load_route(&self, model_id: &str) -> Result<(String, String)> {
        let load_model_id = canonical_pinned_model_id(model_id).with_context(|| {
            format!("model {model_id:?} is not supported by this Candle worker")
        })?;
        let base_model_id = self
            .base_model_for_supported_route(&load_model_id)
            .with_context(|| format!("model {model_id:?} is not supported by this Candle worker"))?
            .to_string();
        Ok((load_model_id, base_model_id))
    }

    fn embedding_load_state(&self, model_id: &str) -> Result<ReadinessState> {
        let (load_model_id, base_model_id) = self.embedding_load_route(model_id)?;

        if self.loaded_embedding_model(&base_model_id).is_some() {
            return Ok(ReadinessState::Ready);
        }

        if self.is_embedding_loading(&base_model_id) {
            return Ok(ReadinessState::LoadingInProgress);
        }

        let backend = self.clone();
        tokio::task::spawn_blocking(move || {
            let result = backend.embedding_model(&load_model_id);
            match result {
                Ok(_) => {
                    info!(
                        model_id = %load_model_id,
                        "Candle background embedding load completed"
                    );
                }
                Err(error) => {
                    warn!(
                        model_id = %load_model_id,
                        error = %error_chain(&error),
                        "Candle background embedding load failed"
                    );
                }
            }
        });

        Ok(ReadinessState::LoadingStarted)
    }

    fn is_embedding_loading(&self, base_model_id: &str) -> bool {
        self.loading_embeddings
            .lock()
            .expect("Candle model loading lock poisoned")
            .contains(base_model_id)
    }

    fn mark_embedding_loading(&self, base_model_id: &str) -> bool {
        self.loading_embeddings
            .lock()
            .expect("Candle model loading lock poisoned")
            .insert(base_model_id.to_string())
    }

    fn unmark_embedding_loading(&self, base_model_id: &str) {
        self.loading_embeddings
            .lock()
            .expect("Candle model loading lock poisoned")
            .remove(base_model_id);
    }

    fn evict_embedding_model(&self, base_model_id: &str, reason: ModelEvictionReason) {
        let removed = self
            .loaded_embeddings
            .lock()
            .expect("Candle model lock poisoned")
            .remove(base_model_id)
            .is_some();
        self.loading_embeddings
            .lock()
            .expect("Candle model loading lock poisoned")
            .remove(base_model_id);
        if removed {
            self.record_model_residency_for_base(base_model_id, false, Some(reason));
        }
    }

    fn loaded_tokenizer_descriptor(&self, model_id: &str) -> Option<(String, String, usize)> {
        let base_model_id = self.base_model_for_supported_route(model_id)?;
        let model = self
            .loaded_embeddings
            .lock()
            .ok()?
            .get(base_model_id)?
            .first();
        Some((
            model.tokenizer_path().display().to_string(),
            model.tokenizer_id().to_string(),
            model.max_seq_length(),
        ))
    }

    fn supports_encode_options(&self, model_id: &str, item: &EncodeBatchItem) -> bool {
        let requested_profile =
            requested_item_profile(item.options.as_ref(), item.profile_id.as_deref());
        if !self.supports_operation(model_id, CandleTaskKind::Embedding)
            || !supported_options(item.options.as_ref(), ENCODE_OPTION_KEYS)
            || profile_option_error(item.options.as_ref()).is_some()
            || output_dtype_type_error(item.options.as_ref()).is_some()
            || !self.supports_pooling_override(model_id, requested_profile, item.options.as_ref())
        {
            return false;
        }

        let defaults = self.runtime_defaults_for(model_id, requested_profile);
        let output_dtype = option_str(item.options.as_ref(), "output_dtype")
            .map(str::to_string)
            .or(defaults.output_dtype);
        let option_output_types = option_output_types(item.options.as_ref());
        let requested_output_types = option_output_types
            .as_deref()
            .or(item.output_types.as_deref());
        self.encode_output_kind(model_id, requested_output_types)
            .map(|kind| output_dtype_error_for_kind(output_dtype.as_deref(), kind).is_none())
            .unwrap_or(false)
    }

    fn encode_output_kind(
        &self,
        model_id: &str,
        output_types: Option<&[String]>,
    ) -> Option<EncodeOutputKind> {
        let guard = self.catalog.read().expect("Candle catalog lock poisoned");
        let config = self.catalog_config_for(model_id, &guard);
        let supports_dense = config
            .map(|config| config.dense_dim.is_some())
            .unwrap_or(true);
        let supports_sparse = config
            .and_then(ModelRuntimeConfig::native_sparse_dim)
            .is_some();
        let supports_multivector = config
            .and_then(ModelRuntimeConfig::native_multivector_dim)
            .is_some();
        match output_types {
            None | Some([]) if supports_dense => Some(EncodeOutputKind::Dense),
            None | Some([]) if supports_multivector => Some(EncodeOutputKind::Multivector),
            None | Some([]) if supports_sparse => Some(EncodeOutputKind::Sparse),
            Some(types) if types.iter().all(|ty| ty == "dense") && supports_dense => {
                Some(EncodeOutputKind::Dense)
            }
            Some(types) if types.iter().all(|ty| ty == "multivector") && supports_multivector => {
                Some(EncodeOutputKind::Multivector)
            }
            Some(types) if types.iter().all(|ty| ty == "sparse") && supports_sparse => {
                Some(EncodeOutputKind::Sparse)
            }
            _ => None,
        }
    }

    fn supports_score_item(&self, model_id: &str, item: &ScoreBatchItem) -> bool {
        if !self.supports_operation(model_id, CandleTaskKind::Rerank)
            || !supported_options(item.options.as_ref(), SCORE_OPTION_KEYS)
            || profile_option_error(item.options.as_ref()).is_some()
        {
            return false;
        }
        let requested_profile =
            requested_item_profile(item.options.as_ref(), item.profile_id.as_deref());
        self.catalog
            .read()
            .expect("Candle catalog lock poisoned")
            .get(base_model_id_for_residency(model_id))
            .is_some_and(|config| config.supports_native_score(model_id, requested_profile))
    }

    fn runtime_defaults_for(
        &self,
        model_id: &str,
        requested_profile: Option<&str>,
    ) -> RuntimeDefaults {
        let guard = self.catalog.read().expect("Candle catalog lock poisoned");
        let Some(config) = self.catalog_config_for(model_id, &guard) else {
            return RuntimeDefaults::default();
        };

        selected_runtime_defaults(model_id, requested_profile, &config.profiles)
    }

    fn supports_pooling_override(
        &self,
        model_id: &str,
        requested_profile: Option<&str>,
        options: Option<&Json>,
    ) -> bool {
        let Some(pooling) = option_str(options, "pooling") else {
            return true;
        };
        self.runtime_defaults_for(model_id, requested_profile)
            .pooling
            .as_deref()
            == Some(pooling)
    }

    fn base_model_for_supported_route<'a>(&self, model_id: &'a str) -> Option<&'a str> {
        if !self.supports(model_id) {
            return None;
        }
        Some(
            synthetic_model_parts(model_id)
                .map(|(base, _)| base)
                .unwrap_or(model_id),
        )
    }

    fn catalog_config_for<'a>(
        &self,
        model_id: &'a str,
        catalog: &'a HashMap<String, ModelRuntimeConfig>,
    ) -> Option<&'a ModelRuntimeConfig> {
        catalog.get(model_id).or_else(|| {
            synthetic_model_parts(model_id)
                .map(|(base, _)| base)
                .and_then(|base| catalog.get(base))
        })
    }

    fn task_kind_for(&self, model_id: &str) -> CandleTaskKind {
        let guard = self.catalog.read().expect("Candle catalog lock poisoned");
        self.catalog_config_for(model_id, &guard)
            .map(|config| config.task_kind)
            .unwrap_or_default()
    }

    fn supports_operation(&self, model_id: &str, task_kind: CandleTaskKind) -> bool {
        if !self.supports(model_id) {
            return false;
        }
        let configured = self.task_kind_for(model_id);
        match task_kind {
            CandleTaskKind::Embedding => configured.supports_embedding(),
            CandleTaskKind::Rerank => configured.supports_score(),
            CandleTaskKind::EmbeddingAndRerank => {
                configured.supports_embedding() && configured.supports_score()
            }
        }
    }
}

const ENCODE_OPTION_KEYS: &[&str] = &[
    "default_instruction",
    "doc_template",
    "instruction",
    "is_query",
    "normalize",
    "output_dtype",
    "output_types",
    "pooling",
    "profile",
    "query_template",
];
// The Python ColBERT score path accepts and ignores encode-only runtime
// options. Keep those benign keys wire-compatible while continuing to reject
// unknown execution-changing options (for example LoRA).
const SCORE_OPTION_KEYS: &[&str] = &[
    "default_instruction",
    "doc_template",
    "instruction",
    "is_query",
    "muvera",
    "normalize",
    "output_dtype",
    "output_similarity",
    "output_types",
    "pooling",
    "profile",
    "query_template",
];
const CANDLE_ADAPTER_MODULE: &str = "sie_server_rust.adapters.candle";

struct PreparedEncodeItem {
    work_item_id: String,
    request_id: String,
    item_index: u32,
    text: String,
    is_query: bool,
    output_dtype: Option<String>,
    prepared_tokens: Option<PreparedTokens>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct EncodeGroupStats {
    items: usize,
    prepared_items: usize,
    prepared_sequences: usize,
    prepared_tokens_total: usize,
    prepared_tokens_min: usize,
    prepared_tokens_max: usize,
    prepared_max_seq_len: u32,
    text_chars_total: usize,
    text_chars_min: usize,
    text_chars_max: usize,
}

impl EncodeGroupStats {
    fn from_group(group: &[(usize, PreparedEncodeItem)]) -> Self {
        let mut stats = Self {
            items: group.len(),
            prepared_tokens_min: usize::MAX,
            text_chars_min: usize::MAX,
            ..Self::default()
        };

        for (_, item) in group {
            let text_chars = item.text.chars().count();
            stats.text_chars_total += text_chars;
            stats.text_chars_min = stats.text_chars_min.min(text_chars);
            stats.text_chars_max = stats.text_chars_max.max(text_chars);

            if let Some(prepared) = item.prepared_tokens.as_ref() {
                stats.prepared_items += 1;
                stats.prepared_sequences += prepared.input_ids.len();
                stats.prepared_max_seq_len = stats.prepared_max_seq_len.max(prepared.max_seq_len);
                let item_tokens = prepared.input_ids.iter().map(Vec::len).sum();
                stats.prepared_tokens_total += item_tokens;
                stats.prepared_tokens_min = stats.prepared_tokens_min.min(item_tokens);
                stats.prepared_tokens_max = stats.prepared_tokens_max.max(item_tokens);
            }
        }

        if group.is_empty() {
            stats.text_chars_min = 0;
        }
        if stats.prepared_items == 0 {
            stats.prepared_tokens_min = 0;
        }

        stats
    }
}

fn encode_item_identities(group: &[(usize, PreparedEncodeItem)]) -> Vec<EncodeItemIdentity> {
    group
        .iter()
        .map(|(position, item)| EncodeItemIdentity {
            position: *position,
            work_item_id: item.work_item_id.clone(),
            request_id: item.request_id.clone(),
            item_index: item.item_index,
        })
        .collect()
}

fn prepared_requests_for_group(
    model: &CandleEmbeddingModel,
    group: &[(usize, PreparedEncodeItem)],
) -> Option<Vec<CandlePreparedEncodeRequest>> {
    let mut prepared_requests = Vec::with_capacity(group.len());
    for (_, item) in group {
        let prepared = item.prepared_tokens.as_ref()?;
        if prepared.tokenizer_id != model.tokenizer_id() {
            return None;
        }
        let input_ids = single_required_row(&prepared.input_ids)?;
        if input_ids.is_empty() {
            return None;
        }
        prepared_requests.push(CandlePreparedEncodeRequest {
            input_ids,
            attention_mask: single_optional_row(&prepared.attention_mask)?,
            token_type_ids: single_optional_row(&prepared.token_type_ids)?,
        });
    }
    Some(prepared_requests)
}

fn score_prepared_item_on_model(
    model: &CandleEmbeddingModel,
    item: &PreparedScoreItem,
    work_budget: usize,
) -> Result<CandleScoreResult> {
    if item.allow_prepared_tokens {
        if let Some((query, docs)) = prepared_score_requests(model, item) {
            return model.score_prepared_multivector(&query, &docs, item.normalize, work_budget);
        }
    }
    model.score_multivector(&item.query, &item.docs, item.normalize, work_budget)
}

fn prepared_score_requests(
    model: &CandleEmbeddingModel,
    item: &PreparedScoreItem,
) -> Option<(
    CandlePreparedEncodeRequest,
    Vec<CandlePreparedEncodeRequest>,
)> {
    let prepared = item.prepared_tokens.as_ref()?;
    if prepared.tokenizer_id != model.tokenizer_id() {
        return None;
    }
    let expected_rows = item.docs.len().checked_add(1)?;
    if prepared.input_ids.len() != expected_rows
        || prepared.input_ids.iter().any(Vec::is_empty)
        || !(prepared.attention_mask.is_empty() || prepared.attention_mask.len() == expected_rows)
        || !(prepared.token_type_ids.is_empty() || prepared.token_type_ids.len() == expected_rows)
    {
        return None;
    }
    let mut requests = Vec::with_capacity(expected_rows);
    for index in 0..expected_rows {
        requests.push(CandlePreparedEncodeRequest {
            input_ids: prepared.input_ids[index].clone(),
            attention_mask: (!prepared.attention_mask.is_empty())
                .then(|| prepared.attention_mask[index].clone()),
            token_type_ids: (!prepared.token_type_ids.is_empty())
                .then(|| prepared.token_type_ids[index].clone()),
        });
    }
    let query = requests.remove(0);
    Some((query, requests))
}

fn single_required_row(rows: &[Vec<u32>]) -> Option<Vec<u32>> {
    match rows {
        [row] => Some(row.clone()),
        _ => None,
    }
}

fn single_optional_row(rows: &[Vec<u32>]) -> Option<Option<Vec<u32>>> {
    match rows {
        [] => Some(None),
        [row] => Some(Some(row.clone())),
        _ => None,
    }
}

fn supported_options(options: Option<&Json>, allowed_keys: &[&str]) -> bool {
    match options {
        None | Some(Json::Null) => true,
        Some(Json::Object(map)) => map.keys().all(|key| allowed_keys.contains(&key.as_str())),
        _ => false,
    }
}

fn raw_output_dtype<'a>(options: Option<&'a Json>, key: &str) -> Option<&'a Json> {
    options
        .and_then(Json::as_object)
        .and_then(|map| map.get(key))
        .filter(|value| !value.is_null())
}

fn output_dtype_error_for_kind(
    output_dtype: Option<&str>,
    output_kind: EncodeOutputKind,
) -> Option<String> {
    match (output_kind, output_dtype) {
        (_, None) | (_, Some("float32")) => None,
        (EncodeOutputKind::Multivector, Some("float16")) => None,
        (EncodeOutputKind::Dense, Some(dtype)) => Some(format!(
            "Candle Rust worker currently returns dense float32 only; requested output_dtype={dtype:?}"
        )),
        (EncodeOutputKind::Sparse, Some(dtype)) => Some(format!(
            "Candle Rust worker currently returns sparse float32 only; requested output_dtype={dtype:?}"
        )),
        (EncodeOutputKind::Multivector, Some(dtype)) => Some(format!(
            "Candle Rust worker currently returns multivector float16 or float32 only; requested output_dtype={dtype:?}"
        )),
    }
}

fn output_dtype_type_error(options: Option<&Json>) -> Option<String> {
    let value = raw_output_dtype(options, "output_dtype")?;
    match value {
        Json::String(_) => None,
        other => Some(format!(
            "Candle Rust worker output_dtype must be a string when provided; got {other}"
        )),
    }
}

fn profile_option_error(options: Option<&Json>) -> Option<String> {
    let value = options
        .and_then(Json::as_object)
        .and_then(|map| map.get("profile"))?;
    match value {
        Json::Null => None,
        Json::String(profile) if !profile.trim().is_empty() => None,
        Json::String(_) => {
            Some("Candle Rust worker profile must be a non-empty string when provided".to_string())
        }
        other => Some(format!(
            "Candle Rust worker profile must be a string when provided; got {other}"
        )),
    }
}

fn multivector_wire_dtype(output_dtype: Option<&str>) -> &'static str {
    match output_dtype {
        Some("float16") => "float16",
        _ => "float32",
    }
}

fn non_empty_str(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn synthetic_model_parts(model_id: &str) -> Option<(&str, &str)> {
    model_id
        .rsplit_once(':')
        .filter(|(base, profile)| !base.is_empty() && !profile.is_empty())
}

fn synthetic_profile_id(model_id: &str) -> Option<&str> {
    synthetic_model_parts(model_id).map(|(_, profile)| profile)
}

fn canonical_pinned_model_id(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let (base, profile) = match raw.split_once(':') {
        Some((base, profile)) => (base.trim(), profile.trim()),
        None => (raw, ""),
    };
    if base.is_empty() {
        return None;
    }
    if profile.is_empty() || profile.eq_ignore_ascii_case("default") {
        return Some(base.to_string());
    }
    Some(format!("{}:{}", base, profile.to_ascii_lowercase()))
}

fn base_model_id_for_residency(model_id: &str) -> &str {
    synthetic_model_parts(model_id)
        .map(|(base, _)| base)
        .unwrap_or(model_id)
}

fn warm_set_contains_base_model(models: &HashSet<String>, base_model_id: &str) -> bool {
    models
        .iter()
        .any(|model_id| base_model_id_for_residency(model_id) == base_model_id)
}

fn residency_policy_for_base_model(
    base_model_id: &str,
    preload: &HashSet<String>,
    pinned: &HashSet<String>,
) -> ResidencyPolicy {
    ResidencyPolicy {
        pinned: warm_set_contains_base_model(pinned, base_model_id),
        preload: warm_set_contains_base_model(preload, base_model_id),
    }
}

fn effective_model_id(model_id: &str, requested_profile: Option<&str>) -> Option<String> {
    let model_id = canonical_pinned_model_id(model_id)?;
    let Some(requested_profile) = requested_profile.and_then(non_empty_str) else {
        return Some(model_id);
    };
    if requested_profile.eq_ignore_ascii_case("default") {
        return Some(model_id);
    }
    if let Some((_, synthetic_profile)) = synthetic_model_parts(&model_id) {
        return synthetic_profile
            .eq_ignore_ascii_case(requested_profile)
            .then_some(model_id);
    }
    canonical_pinned_model_id(&format!("{model_id}:{requested_profile}"))
}

fn telemetry_dimensions_for_route(
    route: &str,
    catalog: &HashMap<String, ModelRuntimeConfig>,
) -> Option<(String, String)> {
    let canonical_route = canonical_pinned_model_id(route)?;
    let model = base_model_id_for_residency(&canonical_route);
    let config = catalog.get(model)?;
    if !config.routable_model_ids.contains(&canonical_route) {
        return None;
    }
    let profile = synthetic_profile_id(&canonical_route).unwrap_or("default");
    config
        .profiles
        .contains_key(profile)
        .then(|| (model.to_string(), profile.to_string()))
}

fn telemetry_dimensions_for_base_model(
    base_model_id: &str,
    catalog: &HashMap<String, ModelRuntimeConfig>,
) -> Vec<(String, String)> {
    let Some(config) = catalog.get(base_model_id) else {
        return Vec::new();
    };
    config
        .routable_model_ids
        .iter()
        .filter_map(|route| telemetry_dimensions_for_route(route, catalog))
        .collect()
}

fn catalog_config_unchanged_for_model(
    model_id: &str,
    current: &HashMap<String, ModelRuntimeConfig>,
    next: &HashMap<String, ModelRuntimeConfig>,
) -> bool {
    let base_model_id = synthetic_model_parts(model_id)
        .map(|(base, _)| base)
        .unwrap_or(model_id);
    match (current.get(base_model_id), next.get(base_model_id)) {
        (Some(current_config), Some(next_config)) => current_config == next_config,
        _ => false,
    }
}

fn loaded_model_routes_for_catalog(
    loaded_model_ids: &[String],
    catalog: &HashMap<String, ModelRuntimeConfig>,
) -> Vec<String> {
    let mut models: Vec<String> = loaded_model_ids
        .iter()
        .flat_map(|model_id| {
            catalog
                .get(model_id)
                .or_else(|| {
                    synthetic_model_parts(model_id)
                        .map(|(base, _)| base)
                        .and_then(|base| catalog.get(base))
                })
                .map(|config| config.routable_model_ids.clone())
                .unwrap_or_else(|| {
                    canonical_pinned_model_id(model_id)
                        .map(|id| vec![id])
                        .unwrap_or_default()
                })
        })
        .collect();
    models.sort();
    models.dedup();
    models
}

fn normalize_pinned_model_ids(raw_ids: &[String]) -> HashSet<String> {
    raw_ids
        .iter()
        .filter_map(|raw| canonical_pinned_model_id(raw))
        .collect()
}

fn error_chain(error: &anyhow::Error) -> String {
    error
        .chain()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(": ")
}

fn is_oom_error(error: &anyhow::Error) -> bool {
    let error = error_chain(error).to_ascii_lowercase();
    OOM_ERROR_INDICATORS
        .iter()
        .any(|indicator| error.contains(indicator))
}

fn option_str<'a>(options: Option<&'a Json>, key: &str) -> Option<&'a str> {
    option_string(options, key).and_then(non_empty_str)
}

fn requested_item_profile<'a>(
    options: Option<&'a Json>,
    profile_id: Option<&'a str>,
) -> Option<&'a str> {
    option_str(options, "profile").or_else(|| profile_id.and_then(non_empty_str))
}

fn option_string<'a>(options: Option<&'a Json>, key: &str) -> Option<&'a str> {
    options
        .and_then(Json::as_object)
        .and_then(|map| map.get(key))
        .and_then(Json::as_str)
}

fn option_bool(options: Option<&Json>, key: &str) -> Option<bool> {
    options
        .and_then(Json::as_object)
        .and_then(|map| map.get(key))
        .and_then(Json::as_bool)
}

fn option_output_types(options: Option<&Json>) -> Option<Vec<String>> {
    options
        .and_then(Json::as_object)
        .and_then(|map| map.get("output_types"))
        .and_then(Json::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Json::as_str)
                .filter_map(non_empty_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
}

fn candle_backend_diagnostics_enabled() -> bool {
    std::env::var("SIE_CANDLE_DIAGNOSTICS")
        .ok()
        .or_else(|| std::env::var("SIE_CANDLE_DEBUG_KERNELS").ok())
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "" | "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(false)
}

fn candle_slow_forward_log_threshold_ms() -> Option<f64> {
    static THRESHOLD: OnceLock<Option<f64>> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        let Ok(raw) = std::env::var("SIE_CANDLE_SLOW_FORWARD_LOG_MS") else {
            return Some(DEFAULT_SLOW_FORWARD_LOG_MS);
        };
        let trimmed = raw.trim();
        if matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "no" | "off"
        ) {
            return None;
        }
        match trimmed.parse::<f64>() {
            Ok(value) if value.is_finite() && value > 0.0 => Some(value),
            Err(_) | Ok(_) => Some(DEFAULT_SLOW_FORWARD_LOG_MS),
        }
    })
}

fn selected_runtime_defaults(
    model_id: &str,
    requested_profile: Option<&str>,
    profiles: &HashMap<String, RuntimeDefaults>,
) -> RuntimeDefaults {
    let requested_profile = requested_profile.and_then(non_empty_str);
    let synthetic_profile = synthetic_profile_id(model_id);
    for profile in [
        requested_profile.filter(|profile| *profile != "default"),
        synthetic_profile,
        requested_profile.filter(|profile| *profile == "default"),
        Some("default"),
    ]
    .into_iter()
    .flatten()
    {
        if let Some(defaults) = profiles.get(profile) {
            return defaults.clone();
        }
    }

    RuntimeDefaults::default()
}

fn parse_runtime_config(
    expected_model_id: &str,
    model_config: &str,
) -> Result<(String, ModelRuntimeConfig)> {
    if model_config.trim().is_empty() {
        anyhow::bail!("model_config is required");
    }
    let model_config: ModelConfigYaml = serde_yaml::from_str(model_config)
        .with_context(|| format!("invalid model_config for {expected_model_id}"))?;
    if !expected_model_id.is_empty() && model_config.sie_id != expected_model_id {
        anyhow::bail!(
            "model_id mismatch: notification={:?} config={:?}",
            expected_model_id,
            model_config.sie_id
        );
    }
    let routable_model_ids = candle_routable_model_ids(&model_config);
    if routable_model_ids.is_empty() {
        anyhow::bail!(
            "Candle model {} has no profiles using adapter module {CANDLE_ADAPTER_MODULE}",
            model_config.sie_id
        );
    }
    validate_candle_profile_loadtime(&model_config)?;
    let candle_profile_id = selected_candle_profile_id(&model_config)?;
    let compute_precision =
        effective_compute_precision(&candle_profile_id, &model_config.profiles, &mut Vec::new());
    let loadtime = resolved_loadtime(&candle_profile_id, &model_config.profiles, &mut Vec::new());
    let max_sequence_length =
        runtime_usize(&loadtime, "max_seq_length").or(model_config.max_sequence_length);
    let query_max_length = runtime_usize(&loadtime, "query_max_length");

    let mut profiles = HashMap::new();
    for profile_id in model_config.profiles.keys() {
        let runtime = resolved_runtime(profile_id, &model_config.profiles, &mut Vec::new());
        let mut defaults = RuntimeDefaults::from_runtime(&runtime);
        defaults.max_batch_tokens =
            effective_max_batch_tokens(profile_id, &model_config.profiles, &mut Vec::new());
        if candle_profile_uses_candle_adapter(profile_id, &model_config.profiles) {
            let loadtime = resolved_loadtime(profile_id, &model_config.profiles, &mut Vec::new());
            defaults.score_strategy = candle_score_strategy(loadtime.get("score_strategy"))?;
        }
        profiles.insert(profile_id.clone(), defaults);
    }
    let task_kind = CandleTaskKind::from_tasks(&model_config.tasks);
    let dense_dim = dense_dim_from_tasks(&model_config.tasks);
    let sparse_dim = sparse_dim_from_tasks(&model_config.tasks);
    let multivector_dim = multivector_dim_from_tasks(&model_config.tasks);
    let sie_id = model_config.sie_id;
    let hf_id = model_config
        .hf_id
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| sie_id.clone());
    let hf_revision = model_config
        .hf_revision
        .filter(|value| !value.trim().is_empty());
    Ok((
        sie_id,
        ModelRuntimeConfig {
            hf_id,
            hf_revision,
            max_sequence_length,
            query_max_length,
            dense_dim,
            sparse_dim,
            multivector_dim,
            compute_precision,
            profiles,
            routable_model_ids,
            task_kind,
        },
    ))
}

fn validate_candle_profile_loadtime(model_config: &ModelConfigYaml) -> Result<()> {
    let task_token_dim = multivector_dim_from_tasks(&model_config.tasks);
    for profile_id in model_config
        .profiles
        .keys()
        .filter(|profile_id| candle_profile_uses_candle_adapter(profile_id, &model_config.profiles))
    {
        let loadtime = resolved_loadtime(profile_id, &model_config.profiles, &mut Vec::new());
        for key in loadtime.keys() {
            if !matches!(
                key.as_str(),
                "max_seq_length"
                    | "query_max_length"
                    | "token_dim"
                    | "normalize"
                    | "score_strategy"
                    | "skip_special_tokens"
                    | "trust_remote_code"
            ) {
                anyhow::bail!(
                    "Candle model {} profile {profile_id} has unsupported loadtime option {key}",
                    model_config.sie_id
                );
            }
        }
        for key in ["max_seq_length", "query_max_length"] {
            if let Some(value) = loadtime.get(key) {
                let value = value.as_u64().with_context(|| {
                    format!(
                        "Candle model {} profile {profile_id} loadtime {key} must be a positive integer",
                        model_config.sie_id
                    )
                })?;
                if value == 0 || usize::try_from(value).is_err() {
                    anyhow::bail!(
                        "Candle model {} profile {profile_id} loadtime {key} must fit usize and be positive",
                        model_config.sie_id
                    );
                }
            }
        }
        if let Some(token_dim) = loadtime.get("token_dim") {
            let token_dim = token_dim.as_u64().with_context(|| {
                format!(
                    "Candle model {} profile {profile_id} loadtime token_dim must be a positive integer",
                    model_config.sie_id
                )
            })?;
            let token_dim = usize::try_from(token_dim).context("token_dim does not fit usize")?;
            if token_dim == 0 || Some(token_dim) != task_token_dim {
                anyhow::bail!(
                    "Candle model {} profile {profile_id} loadtime token_dim {token_dim} must equal tasks.encode.multivector.dim {:?}",
                    model_config.sie_id,
                    task_token_dim
                );
            }
        }
        let score_strategy =
            candle_score_strategy(loadtime.get("score_strategy")).with_context(|| {
                format!(
                    "Candle model {} profile {profile_id} loadtime score_strategy is invalid",
                    model_config.sie_id
                )
            })?;
        let loadtime_normalize = loadtime
            .get("normalize")
            .map(|loadtime_normalize| {
                loadtime_normalize.as_bool().with_context(|| {
                    format!(
                        "Candle model {} profile {profile_id} loadtime normalize must be a boolean",
                        model_config.sie_id
                    )
                })
            })
            .transpose()?;
        let runtime = resolved_runtime(profile_id, &model_config.profiles, &mut Vec::new());
        let runtime_normalize = runtime.get("normalize").and_then(Json::as_bool);
        if let Some(loadtime_normalize) = loadtime_normalize {
            if runtime_normalize != Some(loadtime_normalize) {
                anyhow::bail!(
                    "Candle model {} profile {profile_id} loadtime normalize={loadtime_normalize} must match explicit runtime normalize={runtime_normalize:?}",
                    model_config.sie_id
                );
            }
        }
        if score_strategy == Some(CandleScoreStrategy::ColbertMaxsim)
            && (loadtime_normalize != Some(true) || runtime_normalize != Some(true))
        {
            anyhow::bail!(
                "Candle model {} profile {profile_id} score_strategy=colbert_maxsim requires resolved loadtime normalize=true and runtime normalize=true",
                model_config.sie_id
            );
        }
        if let Some(skip_special_tokens) = loadtime.get("skip_special_tokens") {
            let skip_special_tokens = skip_special_tokens.as_bool().with_context(|| {
                format!(
                    "Candle model {} profile {profile_id} loadtime skip_special_tokens must be a boolean",
                    model_config.sie_id
                )
            })?;
            if skip_special_tokens {
                anyhow::bail!(
                    "Candle model {} profile {profile_id} requests skip_special_tokens=true, which the native Candle path does not support",
                    model_config.sie_id
                );
            }
        }
        if let Some(trust_remote_code) = loadtime.get("trust_remote_code") {
            let trust_remote_code = trust_remote_code.as_bool().with_context(|| {
                format!(
                    "Candle model {} profile {profile_id} loadtime trust_remote_code must be a boolean",
                    model_config.sie_id
                )
            })?;
            if trust_remote_code {
                anyhow::bail!(
                    "Candle model {} profile {profile_id} requests trust_remote_code=true, which the native Candle path does not support",
                    model_config.sie_id
                );
            }
        }
    }
    Ok(())
}

fn selected_candle_profile_id(model_config: &ModelConfigYaml) -> Result<String> {
    for profile_id in ["candle", "default"] {
        if candle_profile_uses_candle_adapter(profile_id, &model_config.profiles) {
            return Ok(profile_id.to_string());
        }
    }
    model_config
        .profiles
        .keys()
        .find(|profile_id| candle_profile_uses_candle_adapter(profile_id, &model_config.profiles))
        .cloned()
        .with_context(|| {
            format!(
                "Candle model {} has no profiles using adapter module {CANDLE_ADAPTER_MODULE}",
                model_config.sie_id
            )
        })
}

fn candle_routable_model_ids(model_config: &ModelConfigYaml) -> Vec<String> {
    let mut model_ids: Vec<String> = model_config
        .profiles
        .keys()
        .filter(|profile_id| candle_profile_uses_candle_adapter(profile_id, &model_config.profiles))
        .filter_map(|profile_id| model_id_for_profile(&model_config.sie_id, profile_id))
        .collect();
    model_ids.sort();
    model_ids.dedup();
    model_ids
}

fn candle_profile_uses_candle_adapter(
    profile_id: &str,
    profiles: &BTreeMap<String, ProfileConfigYaml>,
) -> bool {
    effective_adapter_path(profile_id, profiles, &mut Vec::new())
        .map(adapter_module_from_path)
        .is_some_and(|module| module == CANDLE_ADAPTER_MODULE)
}

fn model_id_for_profile(base_model_id: &str, profile_id: &str) -> Option<String> {
    if profile_id.eq_ignore_ascii_case("default") {
        return canonical_pinned_model_id(base_model_id);
    }
    canonical_pinned_model_id(&format!("{base_model_id}:{profile_id}"))
}

fn effective_adapter_path<'a>(
    profile_id: &str,
    profiles: &'a BTreeMap<String, ProfileConfigYaml>,
    stack: &mut Vec<String>,
) -> Option<&'a str> {
    if stack.iter().any(|seen| seen == profile_id) {
        return None;
    }
    let profile = profiles.get(profile_id)?;
    if let Some(adapter_path) = profile.adapter_path.as_deref().and_then(non_empty_str) {
        return Some(adapter_path);
    }
    stack.push(profile_id.to_string());
    let adapter_path = profile
        .extends
        .as_deref()
        .and_then(|parent| effective_adapter_path(parent, profiles, stack));
    stack.pop();
    adapter_path
}

fn effective_compute_precision(
    profile_id: &str,
    profiles: &BTreeMap<String, ProfileConfigYaml>,
    stack: &mut Vec<String>,
) -> Option<String> {
    if stack.iter().any(|seen| seen == profile_id) {
        return None;
    }
    let profile = profiles.get(profile_id)?;
    if let Some(compute_precision) = profile
        .compute_precision
        .as_deref()
        .and_then(non_empty_str)
        .map(str::to_string)
    {
        return Some(compute_precision);
    }
    stack.push(profile_id.to_string());
    let compute_precision = profile
        .extends
        .as_deref()
        .and_then(|parent| effective_compute_precision(parent, profiles, stack));
    stack.pop();
    compute_precision
}

fn effective_max_batch_tokens(
    profile_id: &str,
    profiles: &BTreeMap<String, ProfileConfigYaml>,
    stack: &mut Vec<String>,
) -> Option<usize> {
    if stack.iter().any(|seen| seen == profile_id) {
        return None;
    }
    let profile = profiles.get(profile_id)?;
    if let Some(max_batch_tokens) = profile.max_batch_tokens.filter(|value| *value > 0) {
        return Some(max_batch_tokens);
    }
    stack.push(profile_id.to_string());
    let max_batch_tokens = profile
        .extends
        .as_deref()
        .and_then(|parent| effective_max_batch_tokens(parent, profiles, stack));
    stack.pop();
    max_batch_tokens
}

fn adapter_module_from_path(adapter_path: &str) -> &str {
    adapter_path
        .split_once(':')
        .map(|(module, _)| module)
        .unwrap_or(adapter_path)
}

impl RuntimeDefaults {
    fn from_runtime(runtime: &BTreeMap<String, Json>) -> Self {
        Self {
            query_template: runtime_string(runtime, "query_template"),
            doc_template: runtime_string(runtime, "doc_template"),
            default_instruction: runtime_string(runtime, "default_instruction")
                .or_else(|| runtime_string(runtime, "instruction")),
            normalize: runtime_bool(runtime, "normalize"),
            pooling: runtime_string(runtime, "pooling"),
            output_dtype: runtime_string(runtime, "output_dtype"),
            max_batch_tokens: None,
            // Load-time and profile-specific; populated by
            // `parse_runtime_config` only for native Candle profiles.
            score_strategy: None,
        }
    }
}

impl CandleTaskKind {
    fn from_tasks(tasks: &TasksYaml) -> Self {
        let encode_dense = tasks.encode.as_ref().is_some_and(encode_has_dense);
        let encode_sparse = tasks.encode.as_ref().is_some_and(encode_has_sparse);
        let encode_multivector = tasks.encode.as_ref().is_some_and(encode_has_multivector);
        let score = tasks.score.is_some();
        match (encode_dense || encode_sparse || encode_multivector, score) {
            (true, true) if encode_multivector && !encode_dense => Self::EmbeddingAndRerank,
            (true, _) => Self::Embedding,
            (false, true) => Self::Rerank,
            (false, false) => Self::Embedding,
        }
    }
}

fn encode_has_dense(value: &Json) -> bool {
    value
        .as_object()
        .and_then(|map| map.get("dense"))
        .is_some_and(|dense| !dense.is_null())
}

fn encode_has_sparse(value: &Json) -> bool {
    value
        .as_object()
        .and_then(|map| map.get("sparse"))
        .is_some_and(|sparse| !sparse.is_null())
}

fn encode_has_multivector(value: &Json) -> bool {
    value
        .as_object()
        .and_then(|map| map.get("multivector"))
        .is_some_and(|multivector| !multivector.is_null())
}

fn dense_dim_from_tasks(tasks: &TasksYaml) -> Option<usize> {
    tasks
        .encode
        .as_ref()?
        .as_object()?
        .get("dense")?
        .as_object()?
        .get("dim")?
        .as_u64()
        .map(|dim| dim as usize)
}

fn sparse_dim_from_tasks(tasks: &TasksYaml) -> Option<usize> {
    tasks
        .encode
        .as_ref()?
        .as_object()?
        .get("sparse")?
        .as_object()?
        .get("dim")?
        .as_u64()
        .map(|dim| dim as usize)
}

fn multivector_dim_from_tasks(tasks: &TasksYaml) -> Option<usize> {
    tasks
        .encode
        .as_ref()?
        .as_object()?
        .get("multivector")?
        .as_object()?
        .get("dim")?
        .as_u64()
        .map(|dim| dim as usize)
}

fn resolved_runtime(
    profile_id: &str,
    profiles: &BTreeMap<String, ProfileConfigYaml>,
    stack: &mut Vec<String>,
) -> BTreeMap<String, Json> {
    if stack.iter().any(|seen| seen == profile_id) {
        return BTreeMap::new();
    }
    let Some(profile) = profiles.get(profile_id) else {
        return BTreeMap::new();
    };
    stack.push(profile_id.to_string());
    let inherited = profile
        .extends
        .as_deref()
        .map(|parent| resolved_runtime(parent, profiles, stack))
        .unwrap_or_default();
    let runtime = if profile.adapter_options.runtime.is_empty() {
        inherited
    } else {
        profile.adapter_options.runtime.clone()
    };
    stack.pop();
    runtime
}

fn resolved_loadtime(
    profile_id: &str,
    profiles: &BTreeMap<String, ProfileConfigYaml>,
    stack: &mut Vec<String>,
) -> BTreeMap<String, Json> {
    if stack.iter().any(|seen| seen == profile_id) {
        return BTreeMap::new();
    }
    let Some(profile) = profiles.get(profile_id) else {
        return BTreeMap::new();
    };
    stack.push(profile_id.to_string());
    let inherited = profile
        .extends
        .as_deref()
        .map(|parent| resolved_loadtime(parent, profiles, stack))
        .unwrap_or_default();
    let loadtime = if profile.adapter_options.loadtime.is_empty() {
        inherited
    } else {
        profile.adapter_options.loadtime.clone()
    };
    stack.pop();
    loadtime
}

fn runtime_string(runtime: &BTreeMap<String, Json>, key: &str) -> Option<String> {
    runtime
        .get(key)
        .and_then(Json::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn runtime_usize(runtime: &BTreeMap<String, Json>, key: &str) -> Option<usize> {
    runtime.get(key)?.as_u64().map(|value| value as usize)
}

fn runtime_bool(runtime: &BTreeMap<String, Json>, key: &str) -> Option<bool> {
    runtime.get(key).and_then(Json::as_bool)
}

fn candle_score_strategy(value: Option<&Json>) -> Result<Option<CandleScoreStrategy>> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value.as_str() {
        Some("colbert_maxsim") => Ok(Some(CandleScoreStrategy::ColbertMaxsim)),
        Some(other) => anyhow::bail!("score_strategy must be 'colbert_maxsim'; got {other:?}"),
        None => anyhow::bail!("score_strategy must be a string"),
    }
}

fn resolve_instruction<'a>(
    request_instruction: Option<&'a str>,
    options: Option<&'a Json>,
    default_instruction: Option<&'a str>,
) -> Option<&'a str> {
    request_instruction
        .or_else(|| option_string(options, "instruction"))
        .or_else(|| option_string(options, "default_instruction"))
        .or(default_instruction)
}

/// Match the Python ColBERT score entrypoint rather than encode-time text
/// preparation. Queue ingress owns instruction precedence and supplies the
/// canonical value here; profile defaults and encode-time options remain
/// benign but do not modify score text.
fn prepare_colbert_score_texts(
    query_text: &str,
    raw_docs: &[&str],
    request_instruction: Option<&str>,
) -> (CandleEncodeRequest, Vec<CandleEncodeRequest>, bool) {
    let instruction = request_instruction;
    let query = CandleEncodeRequest {
        text: TextPrep {
            instruction,
            is_query: true,
            query_template: None,
            doc_template: None,
        }
        .apply(query_text),
    };
    let docs = raw_docs
        .iter()
        .map(|text| CandleEncodeRequest {
            text: (*text).to_string(),
        })
        .collect();
    (query, docs, instruction.is_none())
}

fn extract_text(item: &Json) -> std::result::Result<&str, &'static str> {
    let object = item
        .as_object()
        .ok_or("item must be a map containing text")?;
    match object.get("text") {
        Some(Json::String(text)) => Ok(text),
        Some(Json::Null) => Err("item requires a non-null string 'text' field"),
        Some(_) => Err("item field 'text' must be a string or null"),
        None => match object.get("content") {
            Some(Json::String(content)) => Ok(content),
            Some(Json::Null) => Err("item requires a non-null string 'content' field"),
            Some(_) => Err("item field 'content' must be a string or null"),
            None => Err("item must contain a string 'text' or 'content' field"),
        },
    }
}

fn item_id(item: &Json) -> std::result::Result<Option<&str>, &'static str> {
    let object = item
        .as_object()
        .ok_or("item must be a map containing text")?;
    match object.get("id") {
        None | Some(Json::Null) => Ok(None),
        Some(Json::String(id)) => Ok(Some(id)),
        Some(_) => Err("item field 'id' must be a string or null"),
    }
}

fn score_item_id(item: &Json, index: usize) -> std::result::Result<String, &'static str> {
    Ok(item_id(item)?
        .map(str::to_string)
        .unwrap_or_else(|| format!("item-{index}")))
}

fn error_outcome(item: &EncodeBatchItem, code: &str, message: &str) -> ItemOutcome {
    identity_error_outcome(
        &item.work_item_id,
        &item.request_id,
        item.item_index,
        code,
        message,
    )
}

fn score_error_outcome(item: &ScoreBatchItem, code: &str, message: &str) -> ItemOutcome {
    identity_error_outcome(
        &item.work_item_id,
        &item.request_id,
        item.item_index,
        code,
        message,
    )
}

fn checked_score_input_tokens(query_tokens: u64, doc_tokens: &[u64]) -> Option<u64> {
    let document_tokens = doc_tokens
        .iter()
        .try_fold(0_u64, |total, tokens| total.checked_add(*tokens))?;
    let query_uses = query_tokens.checked_mul(u64::try_from(doc_tokens.len()).ok()?)?;
    document_tokens.checked_add(query_uses)
}

fn score_input_tokens(query_tokens: usize, doc_tokens: &[usize]) -> Option<u64> {
    let query_tokens = u64::try_from(query_tokens).ok()?;
    let doc_tokens = doc_tokens
        .iter()
        .map(|tokens| u64::try_from(*tokens))
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()?;
    checked_score_input_tokens(query_tokens, &doc_tokens)
}

fn success_score_outcome(
    item: &ScoreBatchItem,
    item_ids: &[String],
    result: CandleScoreResult,
) -> ItemOutcome {
    let Some(input_tokens) = score_input_tokens(result.query_tokens, &result.doc_tokens) else {
        return score_error_outcome(
            item,
            "candle_metering_overflow",
            "Candle score token count overflowed the authoritative unit counter",
        );
    };
    ItemOutcome {
        work_item_id: item.work_item_id.clone(),
        request_id: item.request_id.clone(),
        item_index: item.item_index,
        disposition: Disposition::PublishAndAck,
        nak_delay_ms: None,
        result_msgpack: Vec::new(),
        error: None,
        error_code: None,
        inference_ms: Some(result.inference_ms),
        tokenization_ms: Some(result.tokenization_ms),
        // The inclusive inference duration already contains MaxSim. Publishing
        // it as postprocessing as well would make the sidecar double-count it.
        postprocessing_ms: None,
        raw_output: Some(RawOutput {
            dense: None,
            score: Some(ScoreOutputRaw {
                scores: result.scores,
                item_ids: item_ids.to_vec(),
            }),
            sparse: None,
            multivector: None,
        }),
        units: Some(UnitCounts {
            input_tokens: Some(input_tokens),
            pages: None,
            images: None,
        }),
    }
}

fn score_execution_outcome(
    item: &ScoreBatchItem,
    item_ids: &[String],
    execution: ScoreItemExecution,
) -> ItemOutcome {
    let ScoreItemExecution {
        scored,
        lock_wait_ms,
        wall_ms,
    } = execution;
    match scored {
        Ok(result) => {
            if candle_backend_diagnostics_enabled() {
                info!(
                    work_item_id = item.work_item_id,
                    documents = result.doc_tokens.len(),
                    query_tokens = result.query_tokens,
                    document_tokens = result.doc_tokens.iter().sum::<usize>(),
                    tokenization_ms = result.tokenization_ms,
                    inference_ms = result.inference_ms,
                    maxsim_ms = result.maxsim_ms,
                    wall_ms,
                    lock_wait_ms,
                    "Candle MaxSim score diagnostics"
                );
            }
            success_score_outcome(item, item_ids, result)
        }
        Err(error) => {
            error!(
                work_item_id = item.work_item_id,
                documents = item_ids.len(),
                wall_ms,
                lock_wait_ms,
                error = %error_chain(&error),
                "Candle MaxSim score failed"
            );
            score_error_outcome(
                item,
                "candle_inference_failed",
                &format!("Candle MaxSim score failed: {error}"),
            )
        }
    }
}

fn score_oom_nak_outcome(item: &ScoreBatchItem, nak_delay_ms: u64) -> ItemOutcome {
    ItemOutcome {
        work_item_id: item.work_item_id.clone(),
        request_id: item.request_id.clone(),
        item_index: item.item_index,
        disposition: Disposition::NakRetry,
        nak_delay_ms: Some(nak_delay_ms),
        result_msgpack: Vec::new(),
        error: None,
        error_code: None,
        inference_ms: None,
        tokenization_ms: None,
        postprocessing_ms: None,
        raw_output: None,
        units: None,
    }
}

fn score_model_load_error_outcome(
    item: &ScoreBatchItem,
    error: &anyhow::Error,
    nak_delay_ms: u64,
) -> ItemOutcome {
    if is_oom_error(error) {
        score_oom_nak_outcome(item, nak_delay_ms)
    } else {
        score_error_outcome(
            item,
            "candle_model_load_failed",
            &format!("Candle model load failed: {error}"),
        )
    }
}

fn success_dense_outcome(
    item: &PreparedEncodeItem,
    values: Vec<f32>,
    dim: u32,
    normalize: bool,
    inference_ms: f64,
    tokenization_ms: f64,
) -> ItemOutcome {
    ItemOutcome {
        work_item_id: item.work_item_id.clone(),
        request_id: item.request_id.clone(),
        item_index: item.item_index,
        disposition: Disposition::PublishAndAck,
        nak_delay_ms: None,
        result_msgpack: Vec::new(),
        error: None,
        error_code: None,
        inference_ms: Some(inference_ms),
        tokenization_ms: Some(tokenization_ms),
        postprocessing_ms: None,
        raw_output: Some(RawOutput {
            dense: Some(DenseOutput {
                values,
                dim,
                normalize,
            }),
            score: None,
            sparse: None,
            multivector: None,
        }),
        units: None,
    }
}

fn success_sparse_outcome(
    item: &PreparedEncodeItem,
    sparse: CandleSparseEmbedding,
    dim: u32,
    inference_ms: f64,
    tokenization_ms: f64,
) -> ItemOutcome {
    ItemOutcome {
        work_item_id: item.work_item_id.clone(),
        request_id: item.request_id.clone(),
        item_index: item.item_index,
        disposition: Disposition::PublishAndAck,
        nak_delay_ms: None,
        result_msgpack: Vec::new(),
        error: None,
        error_code: None,
        inference_ms: Some(inference_ms),
        tokenization_ms: Some(tokenization_ms),
        postprocessing_ms: None,
        raw_output: Some(RawOutput {
            dense: None,
            score: None,
            sparse: Some(SparseOutput {
                indices: sparse.indices,
                values: sparse.values,
                dims: Some(dim),
            }),
            multivector: None,
        }),
        units: None,
    }
}

fn success_multivector_outcome(
    item: &PreparedEncodeItem,
    multivector: CandleMultivectorEmbedding,
    output_dtype: Option<&str>,
    inference_ms: f64,
    tokenization_ms: f64,
) -> ItemOutcome {
    ItemOutcome {
        work_item_id: item.work_item_id.clone(),
        request_id: item.request_id.clone(),
        item_index: item.item_index,
        disposition: Disposition::PublishAndAck,
        nak_delay_ms: None,
        result_msgpack: Vec::new(),
        error: None,
        error_code: None,
        inference_ms: Some(inference_ms),
        tokenization_ms: Some(tokenization_ms),
        postprocessing_ms: None,
        raw_output: Some(RawOutput {
            dense: None,
            score: None,
            sparse: None,
            multivector: Some(MultivectorOutput {
                values: multivector.values,
                values_f16: multivector.values_f16,
                num_tokens: multivector.num_tokens,
                token_dims: multivector.token_dims,
                dtype: Some(multivector_wire_dtype(output_dtype).to_string()),
            }),
        }),
        units: None,
    }
}

fn success_batched_f16_multivector_outcome(
    item: &PreparedEncodeItem,
    inference_ms: f64,
    tokenization_ms: f64,
) -> ItemOutcome {
    ItemOutcome {
        work_item_id: item.work_item_id.clone(),
        request_id: item.request_id.clone(),
        item_index: item.item_index,
        disposition: Disposition::PublishAndAck,
        nak_delay_ms: None,
        result_msgpack: Vec::new(),
        error: None,
        error_code: None,
        inference_ms: Some(inference_ms),
        tokenization_ms: Some(tokenization_ms),
        postprocessing_ms: None,
        // The batch-level buffer is authoritative for this outcome. The
        // sidecar validates its offset before producing the public payload.
        raw_output: None,
        units: None,
    }
}

fn identity_error_outcome(
    work_item_id: &str,
    request_id: &str,
    item_index: u32,
    code: &str,
    message: &str,
) -> ItemOutcome {
    ItemOutcome {
        work_item_id: work_item_id.to_string(),
        request_id: request_id.to_string(),
        item_index,
        disposition: Disposition::PublishErrorAndAck,
        nak_delay_ms: None,
        result_msgpack: Vec::new(),
        error: Some(message.to_string()),
        error_code: Some(code.to_string()),
        inference_ms: None,
        tokenization_ms: None,
        postprocessing_ms: None,
        raw_output: None::<RawOutput>,
        units: None,
    }
}

fn model_load_error_outcome(
    item: &PreparedEncodeItem,
    error: &anyhow::Error,
    nak_delay_ms: u64,
) -> ItemOutcome {
    if is_oom_error(error) {
        return oom_nak_outcome(item, nak_delay_ms);
    }

    identity_error_outcome(
        &item.work_item_id,
        &item.request_id,
        item.item_index,
        "candle_model_load_failed",
        &format!(
            "failed to load Candle embedding model: {}",
            error_chain(error)
        ),
    )
}

fn oom_nak_outcome(item: &PreparedEncodeItem, nak_delay_ms: u64) -> ItemOutcome {
    ItemOutcome {
        work_item_id: item.work_item_id.clone(),
        request_id: item.request_id.clone(),
        item_index: item.item_index,
        disposition: Disposition::NakRetry,
        nak_delay_ms: Some(nak_delay_ms),
        result_msgpack: Vec::new(),
        error: None,
        error_code: None,
        inference_ms: None,
        tokenization_ms: None,
        postprocessing_ms: None,
        raw_output: None::<RawOutput>,
        units: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prepared_encode_item(position: usize) -> PreparedEncodeItem {
        PreparedEncodeItem {
            work_item_id: format!("work-{position}"),
            request_id: "request".to_string(),
            item_index: position as u32,
            text: format!("text-{position}"),
            is_query: false,
            output_dtype: None,
            prepared_tokens: None,
        }
    }

    fn score_batch_item(options: Option<Json>) -> ScoreBatchItem {
        ScoreBatchItem {
            work_item_id: "work".to_string(),
            request_id: "request".to_string(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            query_item: serde_json::json!({"text": "query"}),
            score_items: Vec::new(),
            instruction: None,
            options,
            profile_id: None,
            payload_fetch_ms: 0.0,
            prepared_tokens: None,
        }
    }

    #[test]
    fn instruction_precedence_is_request_then_options_then_default() {
        let options = serde_json::json!({"instruction": "option"});

        assert_eq!(
            resolve_instruction(Some("request"), Some(&options), Some("default")),
            Some("request")
        );
        assert_eq!(
            resolve_instruction(None, Some(&options), Some("default")),
            Some("option")
        );
        assert_eq!(
            resolve_instruction(None, None, Some("default")),
            Some("default")
        );
        assert_eq!(
            resolve_instruction(Some(""), Some(&options), Some("default")),
            Some("")
        );
        let empty_option = serde_json::json!({"instruction": ""});
        assert_eq!(
            resolve_instruction(None, Some(&empty_option), Some("default")),
            Some("")
        );
        let request_default = serde_json::json!({"default_instruction": "request-default"});
        assert_eq!(
            resolve_instruction(None, Some(&request_default), Some("profile-default")),
            Some("request-default")
        );
    }

    #[test]
    fn colbert_score_uses_canonical_instruction_and_preserves_empty_override() {
        let (query, docs, allow_prepared) =
            prepare_colbert_score_texts("query", &["document"], Some("canonical"));
        assert_eq!(query.text, "canonical query");
        assert_eq!(docs[0].text, "document");
        assert!(!allow_prepared);

        let (query, docs, allow_prepared) =
            prepare_colbert_score_texts("query", &["document"], Some(""));
        assert_eq!(query.text, "query");
        assert_eq!(docs[0].text, "document");
        assert!(!allow_prepared);
    }

    #[test]
    fn colbert_score_options_remain_benign_but_do_not_prepare_text() {
        let options = serde_json::json!({
            "instruction": "ignored-option",
            "default_instruction": "ignored-default",
            "query_template": "ignored-query: {text}",
            "doc_template": "ignored-doc: {text}",
        });

        assert!(supported_options(Some(&options), SCORE_OPTION_KEYS));
        let (query, docs, allow_prepared) =
            prepare_colbert_score_texts("query", &["document"], None);
        assert_eq!(query.text, "query");
        assert_eq!(docs[0].text, "document");
        assert!(allow_prepared);
    }

    #[test]
    fn text_items_match_python_content_alias_and_id_contract() {
        assert_eq!(extract_text(&serde_json::json!({"text": ""})), Ok(""));
        assert_eq!(
            extract_text(&serde_json::json!({"content": "alias"})),
            Ok("alias")
        );
        assert_eq!(
            extract_text(&serde_json::json!({"text": "kept", "content": 42})),
            Ok("kept")
        );
        assert!(extract_text(&serde_json::json!({})).is_err());
        assert!(extract_text(&serde_json::json!({"text": null})).is_err());
        assert!(extract_text(&serde_json::json!({"content": null})).is_err());
        assert!(extract_text(&serde_json::json!({"text": 42})).is_err());
        assert!(extract_text(&serde_json::json!("not-a-map")).is_err());

        assert_eq!(item_id(&serde_json::json!({"id": ""})), Ok(Some("")));
        assert_eq!(item_id(&serde_json::json!({"id": null})), Ok(None));
        assert_eq!(item_id(&serde_json::json!({})), Ok(None));
        assert!(item_id(&serde_json::json!({"id": 42})).is_err());
    }

    #[test]
    fn score_options_accept_python_benign_encode_keys_but_reject_unknowns() {
        let benign = serde_json::json!({
            "muvera": {},
            "output_types": ["dense"],
            "output_similarity": {"dense": "dot"},
            "normalize": true,
            "pooling": "mean",
            "query_template": "query: {text}",
            "doc_template": "passage: {text}",
        });
        assert!(supported_options(Some(&benign), SCORE_OPTION_KEYS));
        assert!(!supported_options(
            Some(&serde_json::json!({"lora": "adapter"})),
            SCORE_OPTION_KEYS
        ));
        assert!(!supported_options(
            Some(&serde_json::json!({"unknown": true})),
            SCORE_OPTION_KEYS
        ));
    }

    #[test]
    fn score_unit_count_duplicates_query_work_per_document_and_checks_overflow() {
        assert_eq!(checked_score_input_tokens(3, &[5, 7]), Some(18));
        assert_eq!(checked_score_input_tokens(32, &[]), Some(0));
        assert_eq!(checked_score_input_tokens(u64::MAX, &[1]), None);
        assert_eq!(checked_score_input_tokens(1, &[u64::MAX]), None);
    }

    #[test]
    fn score_oom_outcome_retries_without_publishing_or_metering() {
        let outcome = score_oom_nak_outcome(&score_batch_item(None), 12_345);

        assert_eq!(outcome.disposition, Disposition::NakRetry);
        assert_eq!(outcome.nak_delay_ms, Some(12_345));
        assert!(outcome.error.is_none());
        assert!(outcome.error_code.is_none());
        assert!(outcome.raw_output.is_none());
        assert!(outcome.units.is_none());
    }

    #[test]
    fn config_normalizes_bounds() {
        let cfg = CandleBackendConfig::new(0, true, 1);
        assert_eq!(cfg.batch_budget, 1);
        assert!(cfg.normalize);
        assert_eq!(cfg.max_concurrent_forwards, 1);
    }

    #[test]
    fn oom_recovery_defaults_match_python_worker_contract() {
        let cfg = CandleOomRecoveryConfig::default();

        assert!(cfg.enabled);
        assert_eq!(
            cfg.strategy,
            vec![
                CandleOomRecoveryAction::CacheClear,
                CandleOomRecoveryAction::EvictLru,
                CandleOomRecoveryAction::SplitBatch,
            ]
        );
        assert_eq!(cfg.max_split_depth, 4);
        assert_eq!(cfg.nak_delay_ms, 10_000);
    }

    #[test]
    fn oom_recovery_kill_switch_matches_python_override() {
        let mut config = CandleOomRecoveryConfig {
            enabled: true,
            ..CandleOomRecoveryConfig::default()
        };

        apply_oom_recovery_kill_switch(&mut config, Some("true"));

        assert!(!config.enabled);

        let mut config = CandleOomRecoveryConfig {
            enabled: true,
            ..CandleOomRecoveryConfig::default()
        };

        apply_oom_recovery_kill_switch(&mut config, Some("false"));

        assert!(config.enabled);
    }

    #[test]
    fn oom_recovery_strategy_parse_dedups_and_accepts_jsonish_list() {
        let parsed = parse_oom_recovery_strategy(
            r#"["cache_clear", "evict_lru", "evict_lru", "split_batch"]"#,
        )
        .unwrap();

        assert_eq!(
            parsed,
            vec![
                CandleOomRecoveryAction::CacheClear,
                CandleOomRecoveryAction::EvictLru,
                CandleOomRecoveryAction::SplitBatch,
            ]
        );
    }

    #[test]
    fn oom_recovery_strategy_rejects_empty_or_unknown_actions() {
        assert!(parse_oom_recovery_strategy(" ").is_err());
        assert!(parse_oom_recovery_strategy("cache_clear,unknown").is_err());
    }

    #[test]
    fn oom_recovery_env_value_parsers_match_python_bounds() {
        assert!(parse_env_bool("TEST", "true").unwrap());
        assert!(!parse_env_bool("TEST", "off").unwrap());
        assert!(parse_env_bool("TEST", "wat").is_err());
        assert_eq!(parse_oom_max_split_depth("8").unwrap(), 8);
        assert!(parse_oom_max_split_depth("9").is_err());
        assert_eq!(parse_oom_nak_delay_ms("2.5").unwrap(), 2500);
        assert!(parse_oom_nak_delay_ms("0").is_err());
    }

    #[test]
    fn oom_recovery_steps_retry_before_eviction_and_split() {
        let strategy = vec![
            CandleOomRecoveryAction::CacheClear,
            CandleOomRecoveryAction::EvictLru,
            CandleOomRecoveryAction::SplitBatch,
        ];
        let mut cursor = 0usize;

        let cache_step = next_oom_recovery_step(&strategy, &mut cursor, || {
            panic!("cache_clear must not invoke LRU eviction")
        });
        let evict_step =
            next_oom_recovery_step(&strategy, &mut cursor, || Some("sibling".to_string()));
        let split_step = next_oom_recovery_step(&strategy, &mut cursor, || None);

        assert_eq!(cache_step, OomRecoveryStep::RetryAfterCacheClear);
        assert_eq!(
            evict_step,
            OomRecoveryStep::RetryAfterEviction("sibling".to_string())
        );
        assert_eq!(split_step, OomRecoveryStep::SplitBatch);
    }

    #[test]
    fn oom_recovery_steps_skip_empty_eviction_and_eventually_terminal() {
        let strategy = vec![
            CandleOomRecoveryAction::EvictLru,
            CandleOomRecoveryAction::SplitBatch,
        ];
        let mut cursor = 0usize;

        let split_step = next_oom_recovery_step(&strategy, &mut cursor, || None);
        let terminal_step = next_oom_recovery_step(&strategy, &mut cursor, || {
            panic!("terminal strategy must not keep evicting")
        });

        assert_eq!(split_step, OomRecoveryStep::SplitBatch);
        assert_eq!(terminal_step, OomRecoveryStep::Terminal);
    }

    #[test]
    fn output_dtype_policy_is_kind_aware() {
        assert!(output_dtype_error_for_kind(Some("float16"), EncodeOutputKind::Dense).is_some());
        assert!(
            output_dtype_error_for_kind(Some("float16"), EncodeOutputKind::Multivector).is_none()
        );
        assert_eq!(multivector_wire_dtype(None), "float32");
        assert_eq!(multivector_wire_dtype(Some("float16")), "float16");
        assert_eq!(multivector_wire_dtype(Some("float32")), "float32");
    }

    #[test]
    fn forward_output_path_is_closed_and_preserves_f16_wire_choice() {
        fn context(
            output_kind: EncodeOutputKind,
            output_dtype: &str,
            accepts_batched_f16_multivectors: bool,
        ) -> EncodeGroupContext<'_> {
            EncodeGroupContext {
                model_id: "test/model",
                output_kind,
                pooling: "mean",
                normalize: true,
                is_query: false,
                output_dtype,
                accepts_batched_f16_multivectors,
            }
        }

        assert_eq!(
            forward_output_path(context(EncodeOutputKind::Dense, "float32", false)),
            ForwardOutputPath::Dense
        );
        assert_eq!(
            forward_output_path(context(EncodeOutputKind::Multivector, "float32", false)),
            ForwardOutputPath::MultivectorF32
        );
        assert_eq!(
            forward_output_path(context(EncodeOutputKind::Multivector, "float16", true)),
            ForwardOutputPath::MultivectorF16Batched
        );
        assert_eq!(
            forward_output_path(context(EncodeOutputKind::Multivector, "float16", false)),
            ForwardOutputPath::MultivectorF16Individual
        );
    }

    #[test]
    fn oom_nak_outcome_matches_queue_retry_contract() {
        let item = PreparedEncodeItem {
            work_item_id: "work".to_string(),
            request_id: "request".to_string(),
            item_index: 7,
            text: "hello".to_string(),
            is_query: false,
            output_dtype: None,
            prepared_tokens: None,
        };

        let outcome = oom_nak_outcome(&item, 12_345);

        assert_eq!(outcome.work_item_id, "work");
        assert_eq!(outcome.request_id, "request");
        assert_eq!(outcome.item_index, 7);
        assert_eq!(outcome.disposition, Disposition::NakRetry);
        assert_eq!(outcome.nak_delay_ms, Some(12_345));
        assert!(outcome.error.is_none());
        assert!(outcome.error_code.is_none());
        assert!(outcome.raw_output.is_none());
    }

    #[test]
    fn model_load_oom_outcome_naks_without_publishing_error() {
        let item = PreparedEncodeItem {
            work_item_id: "work".to_string(),
            request_id: "request".to_string(),
            item_index: 7,
            text: "hello".to_string(),
            is_query: false,
            output_dtype: None,
            prepared_tokens: None,
        };
        let oom_error = anyhow::anyhow!("failed to allocate tensor");
        let non_oom_error = anyhow::anyhow!("missing config");

        let oom_outcome = model_load_error_outcome(&item, &oom_error, 12_345);
        assert_eq!(oom_outcome.disposition, Disposition::NakRetry);
        assert_eq!(oom_outcome.nak_delay_ms, Some(12_345));
        assert!(oom_outcome.error.is_none());
        assert!(oom_outcome.error_code.is_none());

        let non_oom_outcome = model_load_error_outcome(&item, &non_oom_error, 12_345);
        assert_eq!(non_oom_outcome.disposition, Disposition::PublishErrorAndAck);
        assert_eq!(
            non_oom_outcome.error_code.as_deref(),
            Some("candle_model_load_failed")
        );
        assert!(non_oom_outcome.nak_delay_ms.is_none());
    }

    #[test]
    fn model_load_error_fanout_naks_oom_groups() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1).with_oom_recovery(
            CandleOomRecoveryConfig {
                nak_delay_ms: 12_345,
                ..CandleOomRecoveryConfig::default()
            },
        ));
        let mut groups = BTreeMap::new();
        groups.insert(
            (
                EncodeOutputKind::Dense,
                "mean".to_string(),
                true,
                false,
                "float32".to_string(),
            ),
            vec![(2, prepared_encode_item(2)), (0, prepared_encode_item(0))],
        );
        groups.insert(
            (
                EncodeOutputKind::Dense,
                "cls".to_string(),
                false,
                false,
                "float32".to_string(),
            ),
            vec![(1, prepared_encode_item(1))],
        );
        let mut outcomes = vec![None, None, None];
        let oom_error = anyhow::anyhow!("CUDA out of memory");

        backend.mark_model_load_error_outcomes("test/model", &groups, &oom_error, &mut outcomes);

        for (position, outcome) in outcomes.iter().enumerate() {
            let outcome = outcome.as_ref().expect("outcome should be populated");
            assert_eq!(outcome.work_item_id, format!("work-{position}"));
            assert_eq!(outcome.request_id, "request");
            assert_eq!(outcome.item_index, position as u32);
            assert_eq!(outcome.disposition, Disposition::NakRetry);
            assert_eq!(outcome.nak_delay_ms, Some(12_345));
            assert!(outcome.error.is_none());
            assert!(outcome.error_code.is_none());
        }
    }

    #[test]
    fn split_encode_group_halves_without_reordering() {
        let group: Vec<(usize, PreparedEncodeItem)> = (0..5)
            .map(|position| {
                (
                    position,
                    PreparedEncodeItem {
                        work_item_id: format!("work-{position}"),
                        request_id: "request".to_string(),
                        item_index: position as u32,
                        text: format!("text-{position}"),
                        is_query: false,
                        output_dtype: None,
                        prepared_tokens: None,
                    },
                )
            })
            .collect();

        let (left, right) = split_encode_group(group);

        assert_eq!(
            left.iter()
                .map(|(position, _)| *position)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            right
                .iter()
                .map(|(position, _)| *position)
                .collect::<Vec<_>>(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn pinned_model_normalization_preserves_hf_case_and_folds_default() {
        let normalized = normalize_pinned_model_ids(&[
            " BAAI/bge-m3:default ".to_string(),
            "BAAI/bge-m3:CANDLE".to_string(),
            "Other/Model:Custom".to_string(),
            "".to_string(),
        ]);

        assert_eq!(normalized.len(), 3);
        assert!(normalized.contains("BAAI/bge-m3"));
        assert!(normalized.contains("BAAI/bge-m3:candle"));
        assert!(normalized.contains("Other/Model:custom"));
    }

    #[test]
    fn effective_model_id_merges_queue_profile_with_base_model() {
        assert_eq!(
            effective_model_id("BAAI/bge-m3", Some("CANDLE")).as_deref(),
            Some("BAAI/bge-m3:candle")
        );
        assert_eq!(
            effective_model_id("BAAI/bge-m3:candle", Some("candle")).as_deref(),
            Some("BAAI/bge-m3:candle")
        );
        assert_eq!(
            effective_model_id("BAAI/bge-m3:candle", Some("other")),
            None
        );
    }

    #[test]
    fn request_options_profile_overrides_queue_profile_placeholder() {
        let options = serde_json::json!({"profile": "candle"});

        assert_eq!(
            requested_item_profile(Some(&options), Some("default")),
            Some("candle")
        );
        assert_eq!(requested_item_profile(None, Some("candle")), Some("candle"));
    }

    #[test]
    fn profile_option_validation_rejects_ambiguous_values() {
        assert!(profile_option_error(Some(&serde_json::json!({"profile": 123}))).is_some());
        assert!(profile_option_error(Some(&serde_json::json!({"profile": "  "}))).is_some());
        assert!(profile_option_error(Some(&serde_json::json!({"profile": "unknown"}))).is_none());
        assert!(profile_option_error(Some(&serde_json::json!({"profile": null}))).is_none());
    }

    #[test]
    fn residency_policy_matches_profile_qualified_warm_sets_by_base_model() {
        let preload = normalize_pinned_model_ids(&["BAAI/bge-m3:CANDLE".to_string()]);
        let pinned = normalize_pinned_model_ids(&["Other/Model:default".to_string()]);

        assert_eq!(
            residency_policy_for_base_model("BAAI/bge-m3", &preload, &pinned),
            ResidencyPolicy {
                pinned: false,
                preload: true,
            }
        );
        assert_eq!(
            residency_policy_for_base_model("Other/Model", &preload, &pinned),
            ResidencyPolicy {
                pinned: true,
                preload: false,
            }
        );
    }

    #[test]
    fn idle_evict_check_interval_is_bounded() {
        assert_eq!(
            idle_evict_check_interval(Duration::from_secs(10)),
            Duration::from_secs(5)
        );
        assert_eq!(
            idle_evict_check_interval(Duration::from_secs(300)),
            Duration::from_secs(60)
        );
    }

    #[test]
    fn idle_evictor_start_is_disabled_without_ttl() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        assert!(!backend.start_idle_evictor());
        assert!(!backend.stop_idle_evictor());
    }

    #[tokio::test]
    async fn idle_evictor_start_is_idempotent_and_stoppable() {
        let backend = CandleBackend::new(
            CandleBackendConfig::new(64, true, 1).with_idle_evict_s(Some(Duration::from_secs(10))),
        );

        assert!(backend.start_idle_evictor());
        assert!(!backend.start_idle_evictor());
        assert!(backend.stop_idle_evictor());
        assert!(!backend.stop_idle_evictor());
        assert!(backend.start_idle_evictor());
        assert!(backend.stop_idle_evictor());
    }

    #[test]
    fn oom_error_detector_matches_python_indicators() {
        assert!(is_oom_error(&anyhow::anyhow!("CUDA out of memory")));
        let nested = Err::<(), anyhow::Error>(anyhow::anyhow!("failed to allocate tensor"))
            .context("load Candle embedding model")
            .unwrap_err();
        assert!(is_oom_error(&nested));
        assert!(!is_oom_error(&anyhow::anyhow!(
            "failed to load oom-classifier"
        )));
    }

    #[test]
    fn catalog_config_unchanged_requires_identical_base_config() {
        let (_, current_config) = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
hf_id: old/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#,
        )
        .expect("parse current config");
        let (_, unchanged_config) = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
hf_id: old/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#,
        )
        .expect("parse unchanged config");
        let (_, changed_config) = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
hf_id: new/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#,
        )
        .expect("parse changed config");

        let current = HashMap::from([("test/model".to_string(), current_config)]);
        let unchanged = HashMap::from([("test/model".to_string(), unchanged_config)]);
        let changed = HashMap::from([("test/model".to_string(), changed_config)]);

        assert!(catalog_config_unchanged_for_model(
            "test/model",
            &current,
            &unchanged
        ));
        assert!(catalog_config_unchanged_for_model(
            "test/model:candle",
            &current,
            &unchanged
        ));
        assert!(!catalog_config_unchanged_for_model(
            "test/model",
            &current,
            &changed
        ));
        assert!(!catalog_config_unchanged_for_model(
            "missing/model",
            &current,
            &unchanged
        ));
    }

    #[test]
    fn loaded_model_routes_expand_resident_base_to_catalog_profiles() {
        let (_, runtime_config) = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
profiles:
  default:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
  candle:
    extends: default
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
  pytorch:
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
"#,
        )
        .expect("parse Candle config");
        let catalog = HashMap::from([("test/model".to_string(), runtime_config)]);

        let routes = loaded_model_routes_for_catalog(&["test/model".to_string()], &catalog);

        assert_eq!(
            routes,
            vec!["test/model".to_string(), "test/model:candle".to_string()]
        );
        assert_eq!(
            telemetry_dimensions_for_route("test/model:candle", &catalog),
            Some(("test/model".to_string(), "candle".to_string()))
        );
        assert_eq!(
            telemetry_dimensions_for_base_model("test/model", &catalog),
            vec![
                ("test/model".to_string(), "default".to_string()),
                ("test/model".to_string(), "candle".to_string()),
            ]
        );
        assert_eq!(
            telemetry_dimensions_for_route("request/supplied", &catalog),
            None
        );
    }

    #[test]
    fn native_worker_advertises_applied_catalog_models() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        assert!(backend.supported_models().is_empty());
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply first Candle model config");
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "other/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: other/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply second Candle model config");

        assert_eq!(
            backend.supported_models(),
            vec!["other/model:candle", "test/model:candle"]
        );
        assert!(backend.supports("test/model:candle"));
        assert!(backend.supports("other/model:candle"));
        assert!(!backend.supports("test/model"));
        assert!(!backend.supports("missing/model:candle"));
    }

    #[test]
    fn set_pinned_models_stores_set_without_missing_catalog_load() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        let resp = backend.set_pinned_models(&SetPinnedModelsRequest {
            models: vec![
                " BAAI/bge-m3:default ".to_string(),
                "BAAI/bge-m3:CANDLE".to_string(),
                "".to_string(),
            ],
        });

        assert!(resp.applied);
        assert_eq!(resp.pinned_count, 2);
        let pinned = backend.pinned_models.read().expect("pinned lock");
        assert!(pinned.contains("BAAI/bge-m3"));
        assert!(pinned.contains("BAAI/bge-m3:candle"));
        assert!(backend.loading_embeddings.lock().unwrap().is_empty());
        assert!(backend.loaded_models().is_empty());
    }

    #[test]
    fn set_preload_models_keeps_startup_warm_set_separate_from_pins() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        let preload_count = backend.set_preload_models(&[" BAAI/bge-m3:CANDLE ".to_string()]);
        let resp = backend.set_pinned_models(&SetPinnedModelsRequest { models: vec![] });

        assert_eq!(preload_count, 1);
        assert!(resp.applied);
        assert_eq!(resp.pinned_count, 0);
        assert!(backend
            .pinned_models
            .read()
            .expect("pinned lock")
            .is_empty());
        let preload = backend.preload_models.read().expect("preload lock");
        assert!(preload.contains("BAAI/bge-m3:candle"));
        assert!(backend.loading_embeddings.lock().unwrap().is_empty());
        assert!(backend.loaded_models().is_empty());
    }

    #[test]
    fn f16_multivector_output_paths_preserve_wire_behavior() {
        fn group_and_execution() -> (PreparedEncodeGroup, EncodeGroupExecution) {
            let mut item = prepared_encode_item(0);
            item.output_dtype = Some("float16".to_string());
            let group = vec![(0, item)];
            let stats = EncodeGroupStats::from_group(&group);
            let execution = EncodeGroupExecution {
                encoded: Ok(CandleEncodeResult {
                    embeddings: Vec::new(),
                    sparse_embeddings: None,
                    multivectors: None,
                    multivectors_f16: Some(CandleF16MultivectorBatch {
                        values_f16: vec![half::f16::from_f32(1.0), half::f16::from_f32(2.0)],
                        items: vec![crate::candle_embedding::CandleF16MultivectorItem {
                            byte_offset: 0,
                            byte_len: 2 * std::mem::size_of::<half::f16>(),
                            num_tokens: 1,
                            token_dims: 2,
                        }],
                    }),
                    dim: 2,
                    tokenization_ms: 0.0,
                    inference_ms: 1.0,
                    stages: crate::candle_embedding::CandleEncodeStageTimings::default(),
                    forward_profile: None,
                }),
                stats,
                lock_wait_ms: 0.0,
                encode_elapsed_ms: 1.0,
                encode_source: "prepared",
            };
            (group, execution)
        }

        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));
        let mut outcomes = vec![None];
        let mut batched_outputs = Vec::new();
        let (group, execution) = group_and_execution();
        let result = backend.apply_encode_group_execution(
            EncodeGroupContext {
                model_id: "test/model",
                output_kind: EncodeOutputKind::Multivector,
                pooling: "mean",
                normalize: true,
                is_query: false,
                output_dtype: "float16",
                accepts_batched_f16_multivectors: true,
            },
            group,
            execution,
            &mut outcomes,
            &mut batched_outputs,
        );
        assert!(matches!(result, EncodeGroupOutcome::Success { items: 1 }));
        assert_eq!(batched_outputs.len(), 1);
        assert!(outcomes[0]
            .as_ref()
            .expect("batched outcome")
            .raw_output
            .is_none());

        let (group, execution) = group_and_execution();
        let result = backend.apply_encode_group_execution(
            EncodeGroupContext {
                model_id: "test/model",
                output_kind: EncodeOutputKind::Multivector,
                pooling: "mean",
                normalize: true,
                is_query: false,
                output_dtype: "float16",
                accepts_batched_f16_multivectors: false,
            },
            group,
            execution,
            &mut outcomes,
            &mut batched_outputs,
        );
        assert!(matches!(result, EncodeGroupOutcome::Success { items: 1 }));
        assert_eq!(batched_outputs.len(), 1);
        assert!(outcomes[0]
            .as_ref()
            .and_then(|outcome| outcome.raw_output.as_ref())
            .and_then(|output| output.multivector.as_ref())
            .is_some());
    }

    #[test]
    fn apply_model_config_uses_catalog_candle_profile_route() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
profiles:
  default:
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
    adapter_options:
      runtime:
        pooling: mean
        query_template: "default: {text}"
  candle:
    extends: default
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      runtime:
        pooling: cls
        query_template: "candle: {text}"
"#
                .to_string(),
            })
            .expect("apply model config");

        assert!(backend.supports("test/model:candle"));
        assert!(!backend.supports("test/model"));
        let defaults = backend.runtime_defaults_for("test/model:candle", None);
        assert_eq!(defaults.query_template.as_deref(), Some("candle: {text}"));
        assert_eq!(defaults.pooling.as_deref(), Some("cls"));
    }

    #[test]
    fn embedding_load_route_keeps_profile_qualified_candle_id() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
profiles:
  default:
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply model config");

        let (load_model_id, base_model_id) = backend
            .embedding_load_route(" test/model:CANDLE ")
            .expect("profile-qualified Candle route should resolve");

        assert_eq!(load_model_id, "test/model:candle");
        assert_eq!(base_model_id, "test/model");
        assert!(backend.embedding_load_route("test/model").is_err());
    }

    #[test]
    fn candle_profile_nonempty_runtime_replaces_parent_defaults() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
profiles:
  default:
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
    adapter_options:
      runtime:
        pooling: mean
        normalize: true
        query_template: "query: {text}"
        doc_template: "passage: {text}"
        output_dtype: float16
  candle:
    extends: default
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      runtime:
        pooling: cls
"#
                .to_string(),
            })
            .expect("apply model config");

        let defaults = backend.runtime_defaults_for("test/model:candle", None);
        assert_eq!(defaults.pooling.as_deref(), Some("cls"));
        assert_eq!(defaults.query_template, None);
        assert_eq!(defaults.doc_template, None);
        assert_eq!(defaults.normalize, None);
        assert_eq!(defaults.output_dtype, None);
    }

    #[test]
    fn parse_runtime_config_inherits_compute_precision_for_candle_profile() {
        let (_, config) = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
profiles:
  default:
    compute_precision: bfloat16
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
  candle:
    extends: default
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#,
        )
        .expect("parse Candle runtime config");

        assert_eq!(config.compute_precision.as_deref(), Some("bfloat16"));
    }

    #[test]
    fn bge_m3_candle_profile_inherits_bfloat16_precision() {
        let (_, config) = parse_runtime_config(
            "BAAI/bge-m3",
            include_str!("../../sie_server/models/BAAI__bge-m3.yaml"),
        )
        .expect("parse BGE-M3 Candle runtime config");

        assert_eq!(config.compute_precision.as_deref(), Some("bfloat16"));
    }

    #[test]
    fn snowflake_arctic_candle_profile_uses_float16_precision() {
        let (_, config) = parse_runtime_config(
            "Snowflake/snowflake-arctic-embed-l-v2.0",
            include_str!("../../sie_server/models/Snowflake__snowflake-arctic-embed-l-v2.0.yaml"),
        )
        .expect("parse Snowflake Arctic Candle runtime config");

        assert_eq!(config.compute_precision.as_deref(), Some("float16"));
    }

    #[test]
    fn parse_runtime_config_prefers_candle_compute_precision_override() {
        let (_, config) = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
profiles:
  default:
    compute_precision: bfloat16
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
  candle:
    extends: default
    compute_precision: float16
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#,
        )
        .expect("parse Candle runtime config");

        assert_eq!(config.compute_precision.as_deref(), Some("float16"));
    }

    #[test]
    fn runtime_defaults_accept_instruction_alias() {
        let runtime = serde_json::json!({
            "query_template": "Instruct: {instruction}\nQuery: {text}",
            "instruction": "retrieve relevant passages",
        });
        let runtime = runtime.as_object().unwrap().clone().into_iter().collect();

        let defaults = RuntimeDefaults::from_runtime(&runtime);

        assert_eq!(
            defaults.default_instruction.as_deref(),
            Some("retrieve relevant passages")
        );
    }

    #[test]
    fn apply_model_config_rejects_non_candle_adapter_profile() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        let error = backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
profiles:
  default:
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
  candle:
    extends: default
"#
                .to_string(),
            })
            .expect_err("non-Candle profile should be rejected");

        assert!(
            error
                .to_string()
                .contains("has no profiles using adapter module sie_server_rust.adapters.candle"),
            "{error:#}"
        );
    }

    #[tokio::test]
    async fn health_ready_waits_for_catalog_config() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        assert!(!backend.health_ready().await);

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply Candle model config");

        assert!(backend.health_ready().await);
    }

    #[tokio::test]
    async fn health_ready_resolves_profile_variant_from_base_catalog() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply Candle model config");

        assert!(backend.health_ready().await);
    }

    #[tokio::test]
    async fn health_ready_accepts_applied_catalog_model() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        assert!(!backend.health_ready().await);
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply Candle model config");

        assert!(backend.health_ready().await);
    }

    #[test]
    fn apply_model_config_registers_catalog_without_loading_weights() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
hf_id: not-a-real-hf-model-for-config-ack
max_sequence_length: 8192
tasks:
  encode:
    dense:
      dim: 1024
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply Candle model config");

        assert!(backend.supports("test/model:candle"));
        assert_eq!(backend.supported_models(), vec!["test/model:candle"]);
        assert!(backend.loading_embeddings.lock().unwrap().is_empty());
        assert!(backend.loaded_models().is_empty());
    }

    #[tokio::test]
    async fn ensure_model_ready_requires_catalog_before_loading_weights() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
hf_id: not-a-real-hf-model-for-readiness
max_sequence_length: 8192
tasks:
  encode:
    dense:
      dim: 1024
profiles:
  default:
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      runtime:
        query_template: "query: {text}"
        doc_template: "passage: {text}"
"#
                .to_string(),
            })
            .expect("apply Candle model config");

        let resp = backend.ensure_model_ready("other/model:candle").await;
        assert_eq!(resp.state, ReadinessState::RetryLater);
        assert_eq!(resp.batch_budget, None);
        assert!(resp.descriptor.is_none());
        assert!(backend.loaded_embeddings.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn ensure_model_ready_reports_loading_in_progress_without_descriptor() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
hf_id: not-a-real-hf-model-for-readiness
max_sequence_length: 8192
tasks:
  encode:
    dense:
      dim: 1024
profiles:
  default:
    adapter_path: sie_server.adapters.bert_flash:BertFlashAdapter
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply Candle model config");

        backend
            .loading_embeddings
            .lock()
            .unwrap()
            .insert("test/model".to_string());

        let resp = backend.ensure_model_ready("test/model:candle").await;
        assert_eq!(resp.state, ReadinessState::LoadingInProgress);
        assert_eq!(resp.batch_budget, None);
        assert!(resp.descriptor.is_none());
        assert!(backend.loaded_models().is_empty());
    }

    #[tokio::test]
    async fn ensure_model_ready_uses_applied_catalog() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
hf_id: not-a-real-hf-model-for-readiness
max_sequence_length: 8192
tasks:
  encode:
    dense:
      dim: 1024
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply Candle model config");

        backend
            .loading_embeddings
            .lock()
            .unwrap()
            .insert("test/model".to_string());

        let resp = backend.ensure_model_ready("test/model:candle").await;
        assert_eq!(resp.state, ReadinessState::LoadingInProgress);
        assert_eq!(resp.batch_budget, None);
        assert!(resp.descriptor.is_none());
    }

    #[tokio::test]
    async fn apply_model_config_accepts_default_only_candle_model() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["default".to_string()],
                model_config: r#"
sie_id: test/model
profiles:
  default:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      runtime:
        pooling: mean
"#
                .to_string(),
            })
            .expect("default-only Candle model should be accepted");

        assert!(backend.health_ready().await);
    }

    #[tokio::test]
    async fn replace_model_configs_refreshes_catalog_and_reports_applied_models() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        let applied = backend
            .replace_model_configs(&ReplaceModelConfigsRequest {
                bundle_id: "candle".to_string(),
                epoch: 2,
                bundle_config_hash: "hash-2".to_string(),
                models: vec![
                    crate::ipc_types::ReplaceModelConfigEntry {
                        model_id: "test/model".to_string(),
                        model_config: r#"
sie_id: test/model
profiles:
  default:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                        .to_string(),
                    },
                    crate::ipc_types::ReplaceModelConfigEntry {
                        model_id: "other/model".to_string(),
                        model_config: r#"
sie_id: other/model
profiles:
  default:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                        .to_string(),
                    },
                ],
            })
            .expect("replace Candle model configs");

        assert_eq!(
            applied,
            vec!["other/model".to_string(), "test/model".to_string()]
        );
        assert!(backend.health_ready().await);
        assert!(backend
            .catalog
            .read()
            .expect("catalog lock")
            .contains_key("other/model"));

        let cleared = backend
            .replace_model_configs(&ReplaceModelConfigsRequest {
                bundle_id: "candle".to_string(),
                epoch: 3,
                bundle_config_hash: "hash-3".to_string(),
                models: vec![],
            })
            .expect("replace with empty snapshot");

        assert!(cleared.is_empty());
        assert!(!backend.health_ready().await);
        assert!(backend.catalog.read().expect("catalog lock").is_empty());
    }

    #[tokio::test]
    async fn replace_model_configs_reports_all_applied_models() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        let applied = backend
            .replace_model_configs(&ReplaceModelConfigsRequest {
                bundle_id: "candle".to_string(),
                epoch: 2,
                bundle_config_hash: "hash-2".to_string(),
                models: vec![
                    crate::ipc_types::ReplaceModelConfigEntry {
                        model_id: "test/model".to_string(),
                        model_config: r#"
sie_id: test/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                        .to_string(),
                    },
                    crate::ipc_types::ReplaceModelConfigEntry {
                        model_id: "other/model".to_string(),
                        model_config: r#"
sie_id: other/model
profiles:
  default:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                        .to_string(),
                    },
                ],
            })
            .expect("replace Candle model configs");

        assert_eq!(
            applied,
            vec!["other/model".to_string(), "test/model:candle".to_string()]
        );
        assert!(backend.health_ready().await);
    }

    #[tokio::test]
    async fn replace_model_configs_drops_loading_model_when_config_changes() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        let old_config = r#"
sie_id: test/model
hf_id: old/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#;
        backend
            .replace_model_configs(&ReplaceModelConfigsRequest {
                bundle_id: "candle".to_string(),
                epoch: 1,
                bundle_config_hash: "hash-1".to_string(),
                models: vec![crate::ipc_types::ReplaceModelConfigEntry {
                    model_id: "test/model".to_string(),
                    model_config: old_config.to_string(),
                }],
            })
            .expect("replace initial Candle model config");
        backend
            .loading_embeddings
            .lock()
            .expect("loading lock")
            .insert("test/model".to_string());

        backend
            .replace_model_configs(&ReplaceModelConfigsRequest {
                bundle_id: "candle".to_string(),
                epoch: 2,
                bundle_config_hash: "hash-2".to_string(),
                models: vec![crate::ipc_types::ReplaceModelConfigEntry {
                    model_id: "test/model".to_string(),
                    model_config: old_config.to_string(),
                }],
            })
            .expect("replace unchanged Candle model config");
        assert!(
            backend
                .loading_embeddings
                .lock()
                .expect("loading lock")
                .contains("test/model"),
            "unchanged config should keep an in-flight load"
        );

        let applied = backend
            .replace_model_configs(&ReplaceModelConfigsRequest {
                bundle_id: "candle".to_string(),
                epoch: 3,
                bundle_config_hash: "hash-3".to_string(),
                models: vec![crate::ipc_types::ReplaceModelConfigEntry {
                    model_id: "test/model".to_string(),
                    model_config: r#"
sie_id: test/model
hf_id: new/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                    .to_string(),
                }],
            })
            .expect("replace changed Candle model config");

        assert_eq!(applied, vec!["test/model:candle".to_string()]);
        assert!(
            !backend
                .loading_embeddings
                .lock()
                .expect("loading lock")
                .contains("test/model"),
            "changed config must invalidate an in-flight load"
        );
    }

    #[tokio::test]
    async fn encode_rejects_non_float32_output_dtype_before_loading() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
tasks:
  encode:
    dense:
      dim: 384
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply Candle model config");

        let result = backend
            .process_encode_batch(ProcessEncodeBatchRequest {
                model_id: "test/model:candle".to_string(),
                items: vec![EncodeBatchItem {
                    work_item_id: "work".to_string(),
                    request_id: "req".to_string(),
                    item_index: 0,
                    total_items: 1,
                    timestamp: 0.0,
                    item: serde_json::json!({"text": "hello"}),
                    output_types: None,
                    instruction: None,
                    is_query: false,
                    options: Some(serde_json::json!({"output_dtype": "float16"})),
                    profile_id: None,
                    bundle_config_hash: None,
                    payload_fetch_ms: 0.0,
                    prepared_tokens: None,
                }],
                accepts_batched_f16_multivectors: false,
            })
            .await;

        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(
            result.outcomes[0].error_code.as_deref(),
            Some("candle_unsupported_output_dtype")
        );
        assert!(result.outcomes[0]
            .error
            .as_deref()
            .is_some_and(|message| message.contains("dense float32 only")));
        assert!(backend.loaded_embeddings.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn sparse_encode_rejects_normalization_before_loading() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "prithivida/Splade_PP_en_v2".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: include_str!(
                    "../../sie_server/models/prithivida__Splade_PP_en_v2.yaml"
                )
                .to_string(),
            })
            .expect("apply SPLADE Candle model config");

        let result = backend
            .process_encode_batch(ProcessEncodeBatchRequest {
                model_id: "prithivida/Splade_PP_en_v2:candle".to_string(),
                items: vec![EncodeBatchItem {
                    work_item_id: "work".to_string(),
                    request_id: "req".to_string(),
                    item_index: 0,
                    total_items: 1,
                    timestamp: 0.0,
                    item: serde_json::json!({"text": "hello"}),
                    output_types: None,
                    instruction: None,
                    is_query: false,
                    options: Some(serde_json::json!({"normalize": true})),
                    profile_id: None,
                    bundle_config_hash: None,
                    payload_fetch_ms: 0.0,
                    prepared_tokens: None,
                }],
                accepts_batched_f16_multivectors: false,
            })
            .await;

        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(
            result.outcomes[0].error_code.as_deref(),
            Some("candle_unsupported_request")
        );
        assert!(result.outcomes[0]
            .error
            .as_deref()
            .is_some_and(|message| message.contains("normalize=false")));
        assert!(backend.loaded_embeddings.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn encode_accepts_queue_profile_id_without_process_profile_env() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
tasks:
  encode:
    dense:
      dim: 384
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply Candle model config");

        let result = backend
            .process_encode_batch(ProcessEncodeBatchRequest {
                model_id: "test/model".to_string(),
                items: vec![EncodeBatchItem {
                    work_item_id: "work".to_string(),
                    request_id: "req".to_string(),
                    item_index: 0,
                    total_items: 1,
                    timestamp: 0.0,
                    item: serde_json::json!({"text": "hello"}),
                    output_types: None,
                    instruction: None,
                    is_query: false,
                    options: Some(serde_json::json!({"output_dtype": "float16"})),
                    profile_id: Some("candle".to_string()),
                    bundle_config_hash: None,
                    payload_fetch_ms: 0.0,
                    prepared_tokens: None,
                }],
                accepts_batched_f16_multivectors: false,
            })
            .await;

        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(
            result.outcomes[0].error_code.as_deref(),
            Some("candle_unsupported_output_dtype")
        );
        assert!(backend.loaded_embeddings.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn encode_prefers_options_profile_over_queue_placeholder() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));

        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
tasks:
  encode:
    dense:
      dim: 384
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#
                .to_string(),
            })
            .expect("apply Candle model config");

        let result = backend
            .process_encode_batch(ProcessEncodeBatchRequest {
                model_id: "test/model".to_string(),
                items: vec![EncodeBatchItem {
                    work_item_id: "work".to_string(),
                    request_id: "req".to_string(),
                    item_index: 0,
                    total_items: 1,
                    timestamp: 0.0,
                    item: serde_json::json!({"text": "hello"}),
                    output_types: None,
                    instruction: None,
                    is_query: false,
                    options: Some(serde_json::json!({
                        "profile": "candle",
                        "output_dtype": "float16"
                    })),
                    profile_id: Some("default".to_string()),
                    bundle_config_hash: None,
                    payload_fetch_ms: 0.0,
                    prepared_tokens: None,
                }],
                accepts_batched_f16_multivectors: false,
            })
            .await;

        assert_eq!(result.outcomes.len(), 1);
        assert_eq!(
            result.outcomes[0].error_code.as_deref(),
            Some("candle_unsupported_output_dtype")
        );
        assert!(backend.loaded_embeddings.lock().unwrap().is_empty());
    }

    #[test]
    fn parse_runtime_config_supports_multivector_encode_and_score_together() {
        let (_, config) = parse_runtime_config(
            "topk-io/Iso-ModernColBERT",
            include_str!("../../sie_server/models/topk-io__Iso-ModernColBERT.yaml"),
        )
        .expect("parse Iso-ModernColBERT Candle runtime config");

        assert_eq!(config.task_kind, CandleTaskKind::EmbeddingAndRerank);
        assert_eq!(config.dense_dim, None);
        assert_eq!(config.native_multivector_dim(), Some(128));
        assert_eq!(
            config.profiles["candle"].score_strategy,
            Some(CandleScoreStrategy::ColbertMaxsim)
        );
        assert_eq!(
            config.output_types("topk-io/Iso-ModernColBERT:candle", None),
            vec!["multivector".to_string(), "score".to_string()]
        );
        assert_eq!(config.compute_precision.as_deref(), Some("bfloat16"));
        assert_eq!(
            config.hf_revision.as_deref(),
            Some("a43b93e62b11ff205b7f935c1ce2207bfed2d283")
        );
        assert_eq!(config.max_sequence_length, Some(8192));
        assert_eq!(config.query_max_length, Some(32));
        assert_eq!(config.profiles["candle"].max_batch_tokens, Some(16384));
    }

    #[test]
    fn score_is_not_advertised_without_explicit_colbert_maxsim_strategy() {
        let (_, config) = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
tasks:
  encode:
    multivector:
      dim: 128
  score: {}
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        token_dim: 128
"#,
        )
        .expect("parse score-capable multivector model without a native strategy");

        assert_eq!(config.profiles["candle"].score_strategy, None);
        assert_eq!(
            config.output_types("test/model:candle", None),
            vec!["multivector".to_string()]
        );
        assert!(!config.supports_native_score("test/model:candle", None));
    }

    #[test]
    fn parse_runtime_config_rejects_unknown_score_strategy() {
        let error = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
tasks:
  encode:
    multivector:
      dim: 128
  score: {}
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        token_dim: 128
        score_strategy: arbitrary
"#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("score_strategy"));
        assert!(format!("{error:#}").contains("colbert_maxsim"));
    }

    #[test]
    fn colbert_score_strategy_scopes_descriptor_and_request_to_effective_profile() {
        let model_config = r#"
sie_id: test/model
tasks:
  encode:
    multivector:
      dim: 128
  score: {}
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        token_dim: 128
        normalize: true
        score_strategy: colbert_maxsim
      runtime:
        normalize: true
  candle_plain:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        token_dim: 128
        normalize: true
      runtime:
        normalize: true
"#;
        let (_, config) =
            parse_runtime_config("test/model", model_config).expect("parse two Candle profiles");

        assert_eq!(
            config.output_types("test/model:candle", None),
            vec!["multivector".to_string(), "score".to_string()]
        );
        assert_eq!(
            config.output_types("test/model:candle_plain", None),
            vec!["multivector".to_string()]
        );

        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string(), "candle_plain".to_string()],
                model_config: model_config.to_string(),
            })
            .expect("apply two Candle profiles");
        assert!(backend.supports_score_item("test/model:candle", &score_batch_item(None)));
        assert!(!backend.supports_score_item("test/model:candle_plain", &score_batch_item(None)));
    }

    #[test]
    fn candle_adapter_option_blocks_inherit_only_when_child_block_is_empty() {
        let model_config = r#"
sie_id: test/model
tasks:
  encode:
    multivector:
      dim: 128
  score: {}
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        token_dim: 128
        normalize: true
        score_strategy: colbert_maxsim
      runtime:
        normalize: true
        output_dtype: float16
  candle_empty:
    extends: candle
    adapter_options:
      loadtime: {}
      runtime: {}
  candle_replaced:
    extends: candle
    adapter_options:
      loadtime:
        token_dim: 128
        normalize: true
      runtime:
        normalize: true
"#;
        let (_, config) = parse_runtime_config("test/model", model_config)
            .expect("parse parent, empty child, and replacing child profiles");

        let parent = &config.profiles["candle"];
        let empty_child = &config.profiles["candle_empty"];
        let replaced_child = &config.profiles["candle_replaced"];
        assert_eq!(
            parent.score_strategy,
            Some(CandleScoreStrategy::ColbertMaxsim)
        );
        assert_eq!(
            empty_child.score_strategy,
            Some(CandleScoreStrategy::ColbertMaxsim)
        );
        assert_eq!(replaced_child.score_strategy, None);
        assert_eq!(parent.output_dtype.as_deref(), Some("float16"));
        assert_eq!(empty_child.output_dtype.as_deref(), Some("float16"));
        assert_eq!(replaced_child.output_dtype, None);

        assert_eq!(
            config.output_types("test/model:candle_empty", None),
            vec!["multivector".to_string(), "score".to_string()]
        );
        assert_eq!(
            config.output_types("test/model:candle_replaced", None),
            vec!["multivector".to_string()]
        );

        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec![
                    "candle".to_string(),
                    "candle_empty".to_string(),
                    "candle_replaced".to_string(),
                ],
                model_config: model_config.to_string(),
            })
            .expect("apply inherited and replaced Candle profiles");
        assert!(backend.supports_score_item("test/model:candle", &score_batch_item(None)));
        assert!(backend.supports_score_item("test/model:candle_empty", &score_batch_item(None)));
        assert!(!backend.supports_score_item("test/model:candle_replaced", &score_batch_item(None)));
    }

    #[test]
    fn colbert_score_strategy_requires_explicit_true_loadtime_normalize() {
        let error = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
tasks:
  encode:
    multivector:
      dim: 128
  score: {}
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        token_dim: 128
        score_strategy: colbert_maxsim
      runtime:
        normalize: true
"#,
        )
        .unwrap_err();

        assert!(format!("{error:#}").contains("requires resolved loadtime normalize=true"));
    }

    #[test]
    fn colbert_score_strategy_rejects_false_normalization() {
        let error = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
tasks:
  encode:
    multivector:
      dim: 128
  score: {}
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        token_dim: 128
        normalize: false
        score_strategy: colbert_maxsim
      runtime:
        normalize: false
"#,
        )
        .unwrap_err();

        assert!(format!("{error:#}").contains("normalize=true"));
    }

    #[test]
    fn colbert_score_strategy_rejects_runtime_normalize_mismatch() {
        let error = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
tasks:
  encode:
    multivector:
      dim: 128
  score: {}
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        token_dim: 128
        normalize: true
        score_strategy: colbert_maxsim
      runtime:
        normalize: false
"#,
        )
        .unwrap_err();

        assert!(format!("{error:#}").contains("must match explicit runtime normalize"));
    }

    #[test]
    fn score_request_is_rejected_without_explicit_colbert_maxsim_strategy() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "test/model".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: r#"
sie_id: test/model
tasks:
  encode:
    multivector:
      dim: 128
  score: {}
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        token_dim: 128
"#
                .to_string(),
            })
            .expect("apply multivector model without native score strategy");

        assert!(!backend.supports_score_item("test/model:candle", &score_batch_item(None)));
    }

    #[test]
    fn parse_runtime_config_rejects_unsupported_special_token_filtering() {
        let error = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        skip_special_tokens: true
"#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("skip_special_tokens=true"));
    }

    #[test]
    fn parse_runtime_config_rejects_special_token_filtering_on_any_candle_profile() {
        let error = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        skip_special_tokens: false
  candle_without_special_tokens:
    extends: candle
    adapter_options:
      loadtime:
        skip_special_tokens: true
"#,
        )
        .unwrap_err();

        let message = error.to_string();
        assert!(message.contains("candle_without_special_tokens"));
        assert!(message.contains("skip_special_tokens=true"));
    }

    #[test]
    fn parse_runtime_config_rejects_inconsistent_candle_loadtime_semantics() {
        for (loadtime, runtime, expected) in [
            ("token_dim: 64", "normalize: true", "token_dim 64"),
            ("normalize: true", "normalize: false", "must match"),
            ("query_prefix: '[Q] '", "normalize: true", "unsupported"),
            (
                "max_seq_length: zero",
                "normalize: true",
                "positive integer",
            ),
        ] {
            let yaml = format!(
                r#"
sie_id: test/model
tasks:
  encode:
    multivector:
      dim: 128
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
    adapter_options:
      loadtime:
        {loadtime}
      runtime:
        {runtime}
"#
            );
            let error = parse_runtime_config("test/model", &yaml).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn parse_runtime_config_accepts_existing_candle_false_remote_code_policy() {
        parse_runtime_config(
            "sentence-transformers/all-MiniLM-L6-v2",
            include_str!("../../sie_server/models/sentence-transformers__all-MiniLM-L6-v2.yaml"),
        )
        .expect("parse existing MiniLM Candle profile");
    }

    #[test]
    fn parse_runtime_config_threads_hugging_face_revision() {
        let (_, config) = parse_runtime_config(
            "test/model",
            r#"
sie_id: test/model
hf_id: upstream/model
hf_revision: 0123456789abcdef
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#,
        )
        .expect("parse pinned Candle model");

        assert_eq!(config.hf_id, "upstream/model");
        assert_eq!(config.hf_revision.as_deref(), Some("0123456789abcdef"));
    }

    #[test]
    fn candle_output_kind_keeps_bge_m3_dense_only() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "BAAI/bge-m3".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: include_str!("../../sie_server/models/BAAI__bge-m3.yaml").to_string(),
            })
            .expect("apply BGE-M3 Candle model config");

        let dense = vec!["dense".to_string()];
        let multivector = vec!["multivector".to_string()];

        assert_eq!(
            backend.encode_output_kind("BAAI/bge-m3:candle", Some(&dense)),
            Some(EncodeOutputKind::Dense)
        );
        assert_eq!(
            backend.encode_output_kind("BAAI/bge-m3:candle", Some(&multivector)),
            None
        );
        assert!(!backend.supports_operation("BAAI/bge-m3:candle", CandleTaskKind::Rerank));
    }

    #[test]
    fn candle_output_kind_accepts_iso_multivector_profile_default() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "topk-io/Iso-ModernColBERT".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: include_str!(
                    "../../sie_server/models/topk-io__Iso-ModernColBERT.yaml"
                )
                .to_string(),
            })
            .expect("apply Iso-ModernColBERT Candle model config");

        let multivector = vec!["multivector".to_string()];
        let dense = vec!["dense".to_string()];

        assert_eq!(
            backend.encode_output_kind("topk-io/Iso-ModernColBERT:candle", None),
            Some(EncodeOutputKind::Multivector)
        );
        assert_eq!(
            backend.encode_output_kind("topk-io/Iso-ModernColBERT:candle", Some(&multivector)),
            Some(EncodeOutputKind::Multivector)
        );
        assert_eq!(
            backend.encode_output_kind("topk-io/Iso-ModernColBERT:candle", Some(&dense)),
            None
        );
        assert_eq!(
            backend
                .runtime_defaults_for("topk-io/Iso-ModernColBERT:candle", None)
                .output_dtype
                .as_deref(),
            Some("float16")
        );
        assert!(backend.supports_operation(
            "topk-io/Iso-ModernColBERT:candle",
            CandleTaskKind::Embedding
        ));
        assert!(
            backend.supports_operation("topk-io/Iso-ModernColBERT:candle", CandleTaskKind::Rerank)
        );
        assert!(backend
            .supports_score_item("topk-io/Iso-ModernColBERT:candle", &score_batch_item(None)));
        assert_eq!(
            backend
                .runtime_defaults_for("topk-io/Iso-ModernColBERT:candle", None)
                .max_batch_tokens,
            Some(16384)
        );
    }

    #[test]
    fn candle_output_kind_accepts_splade_sparse_profile_default() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "prithivida/Splade_PP_en_v2".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: include_str!(
                    "../../sie_server/models/prithivida__Splade_PP_en_v2.yaml"
                )
                .to_string(),
            })
            .expect("apply SPLADE Candle model config");

        let sparse = vec!["sparse".to_string()];
        let dense = vec!["dense".to_string()];
        let multivector = vec!["multivector".to_string()];

        assert_eq!(
            backend.encode_output_kind("prithivida/Splade_PP_en_v2:candle", None),
            Some(EncodeOutputKind::Sparse)
        );
        assert_eq!(
            backend.encode_output_kind("prithivida/Splade_PP_en_v2:candle", Some(&sparse)),
            Some(EncodeOutputKind::Sparse)
        );
        assert_eq!(
            backend.encode_output_kind("prithivida/Splade_PP_en_v2:candle", Some(&dense)),
            None
        );
        assert_eq!(
            backend.encode_output_kind("prithivida/Splade_PP_en_v2:candle", Some(&multivector)),
            None
        );
        let defaults = backend.runtime_defaults_for("prithivida/Splade_PP_en_v2:candle", None);
        assert_eq!(defaults.pooling.as_deref(), Some("splade"));
        assert_eq!(defaults.normalize, Some(false));
    }

    #[test]
    fn encode_normalize_fallback_is_output_kind_aware() {
        assert!(!default_normalize(EncodeOutputKind::Sparse, true));
        assert!(default_normalize(EncodeOutputKind::Dense, true));
        assert!(default_normalize(EncodeOutputKind::Multivector, true));
        assert!(!default_normalize(EncodeOutputKind::Dense, false));
    }

    #[test]
    fn score_item_ids_preserve_strings_and_generate_stable_fallbacks() {
        assert_eq!(
            score_item_id(&serde_json::json!({"id": "doc-a"}), 0),
            Ok("doc-a".to_string())
        );
        assert_eq!(
            score_item_id(&serde_json::json!({}), 1),
            Ok("item-1".to_string())
        );
        assert!(score_item_id(&serde_json::json!({"id": 42}), 2).is_err());
        assert_eq!(
            score_item_id(&serde_json::json!({"id": ""}), 3),
            Ok(String::new())
        );
    }

    #[test]
    fn score_raw_output_preserves_document_order_for_sidecar_ranking() {
        let item = ScoreBatchItem {
            work_item_id: "work".to_string(),
            request_id: "request".to_string(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            query_item: serde_json::json!({"text": "query"}),
            score_items: Vec::new(),
            instruction: None,
            options: None,
            profile_id: None,
            payload_fetch_ms: 0.0,
            prepared_tokens: None,
        };
        let outcome = success_score_outcome(
            &item,
            &["doc-b".to_string(), "doc-a".to_string()],
            CandleScoreResult {
                scores: vec![0.1, 0.9],
                query_tokens: 2,
                doc_tokens: vec![3, 4],
                tokenization_ms: 1.0,
                inference_ms: 2.0,
                maxsim_ms: Some(0.5),
            },
        );

        assert_eq!(outcome.inference_ms, Some(2.0));
        assert_eq!(outcome.postprocessing_ms, None);
        assert_eq!(
            outcome.units,
            Some(UnitCounts {
                input_tokens: Some(11),
                pages: None,
                images: None,
            })
        );

        let score = outcome
            .raw_output
            .and_then(|raw| raw.score)
            .expect("raw score output");
        assert_eq!(score.item_ids, vec!["doc-b", "doc-a"]);
        assert_eq!(score.scores, vec![0.1, 0.9]);
    }

    #[test]
    fn sparse_raw_output_is_exclusive_and_preserves_ascending_indices() {
        let item = prepared_encode_item(0);
        let outcome = success_sparse_outcome(
            &item,
            CandleSparseEmbedding {
                indices: vec![7, 19, 30521],
                values: vec![0.25, 1.5, 0.75],
            },
            30522,
            2.0,
            1.0,
        );

        let raw = outcome.raw_output.expect("raw sparse output");
        assert!(raw.dense.is_none());
        assert!(raw.score.is_none());
        assert!(raw.multivector.is_none());
        let sparse = raw.sparse.expect("sparse payload");
        assert_eq!(sparse.indices, vec![7, 19, 30521]);
        assert_eq!(sparse.values, vec![0.25, 1.5, 0.75]);
        assert_eq!(sparse.dims, Some(30522));
    }

    #[tokio::test]
    async fn score_accepts_benign_python_encode_options_for_empty_document_set() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "topk-io/Iso-ModernColBERT".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: include_str!(
                    "../../sie_server/models/topk-io__Iso-ModernColBERT.yaml"
                )
                .to_string(),
            })
            .expect("apply Iso-ModernColBERT Candle model config");

        let result = backend
            .process_score_batch(ProcessScoreBatchRequest {
                model_id: "topk-io/Iso-ModernColBERT".to_string(),
                items: vec![ScoreBatchItem {
                    work_item_id: "work".to_string(),
                    request_id: "request".to_string(),
                    item_index: 0,
                    total_items: 1,
                    timestamp: 0.0,
                    query_item: serde_json::json!({"content": "query", "id": null}),
                    score_items: Vec::new(),
                    instruction: None,
                    options: Some(serde_json::json!({
                        "profile": "candle",
                        "output_dtype": "float16",
                        "muvera": {},
                        "output_types": ["dense"],
                        "output_similarity": {"dense": "dot"},
                    })),
                    profile_id: Some("default".to_string()),
                    payload_fetch_ms: 0.0,
                    prepared_tokens: None,
                }],
            })
            .await;

        assert_eq!(result.outcomes.len(), 1);
        let outcome = &result.outcomes[0];
        assert_eq!(outcome.disposition, Disposition::PublishAndAck);
        let score = outcome
            .raw_output
            .as_ref()
            .and_then(|raw| raw.score.as_ref())
            .expect("empty score output");
        assert!(score.scores.is_empty());
        assert!(score.item_ids.is_empty());
        assert_eq!(
            outcome.units.as_ref().and_then(|units| units.input_tokens),
            Some(0)
        );
        assert!(backend.loaded_embeddings.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn score_validates_missing_null_empty_and_malformed_text_items_before_load() {
        let backend = CandleBackend::new(CandleBackendConfig::new(64, true, 1));
        backend
            .apply_model_config(&ApplyModelConfigRequest {
                bundle_id: "candle".to_string(),
                model_id: "topk-io/Iso-ModernColBERT".to_string(),
                epoch: 1,
                bundle_config_hash: "hash".to_string(),
                profiles_added: vec!["candle".to_string()],
                model_config: include_str!(
                    "../../sie_server/models/topk-io__Iso-ModernColBERT.yaml"
                )
                .to_string(),
            })
            .expect("apply Iso-ModernColBERT Candle model config");

        let make_item = |index: u32, query_item: Json, score_items: Vec<Json>| ScoreBatchItem {
            work_item_id: format!("work-{index}"),
            request_id: "request".to_string(),
            item_index: index,
            total_items: 5,
            timestamp: 0.0,
            query_item,
            score_items,
            instruction: None,
            options: None,
            profile_id: None,
            payload_fetch_ms: 0.0,
            prepared_tokens: None,
        };
        let result = backend
            .process_score_batch(ProcessScoreBatchRequest {
                model_id: "topk-io/Iso-ModernColBERT:candle".to_string(),
                items: vec![
                    make_item(0, serde_json::json!({}), Vec::new()),
                    make_item(1, serde_json::json!({"text": null}), Vec::new()),
                    make_item(2, serde_json::json!({"text": ""}), Vec::new()),
                    make_item(
                        3,
                        serde_json::json!({"text": "query", "id": 42}),
                        Vec::new(),
                    ),
                    make_item(
                        4,
                        serde_json::json!({"text": "query"}),
                        vec![serde_json::json!({"text": "doc", "id": 42})],
                    ),
                ],
            })
            .await;

        assert_eq!(result.outcomes.len(), 5);
        for index in [0_usize, 1, 3, 4] {
            assert_eq!(
                result.outcomes[index].error_code.as_deref(),
                Some("candle_invalid_item")
            );
        }
        assert_eq!(result.outcomes[2].disposition, Disposition::PublishAndAck);
        assert_eq!(
            result.outcomes[2]
                .units
                .as_ref()
                .and_then(|units| units.input_tokens),
            Some(0)
        );
        assert!(backend.loaded_embeddings.lock().unwrap().is_empty());
    }
}
