//! Canonical single-emission metric facade for the Rust inference engine.
//!
//! Service code reports one typed semantic event here. If the OTLP metrics
//! provider is not enabled, every public recorder returns before allocating
//! attributes or touching an instrument. Prometheus spelling and fan-out are
//! collector concerns; this module never records a second compatibility copy.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use opentelemetry::metrics::{Counter, Gauge, Histogram, Meter, MeterProvider as _, UpDownCounter};
use opentelemetry::{global, KeyValue};
use opentelemetry_sdk::metrics::{Instrument, PeriodicReader, SdkMeterProvider, Stream};

use super::resource::{cleaned_env, telemetry_resource};
use super::transport::{build_metric_exporter, endpoint_origin_for_log, metric_export_config};

pub const REQUESTS_METRIC_NAME: &str = "sie.worker.requests";
pub const REQUEST_DURATION_METRIC_NAME: &str = "sie.worker.request.duration";
pub const INFERENCE_DURATION_METRIC_NAME: &str = "sie.worker.inference.duration";
pub const UNITS_METRIC_NAME: &str = "sie.worker.units";
pub const MODEL_LOADED_METRIC_NAME: &str = "sie.worker.model.loaded";
pub const MODEL_LOAD_DURATION_METRIC_NAME: &str = "sie.worker.model.load.duration";
pub const MODEL_MEMORY_METRIC_NAME: &str = "sie.worker.model.memory";
pub const OOM_RECOVERIES_METRIC_NAME: &str = "sie.worker.oom.recoveries";
pub const MODEL_EVICTIONS_METRIC_NAME: &str = "sie.worker.model.evictions";
pub const FORWARD_DURATION_METRIC_NAME: &str = "sie.worker.runtime.forward.duration";
pub const FORWARD_PERMIT_WAIT_METRIC_NAME: &str = "sie.worker.runtime.forward.permit.wait";
pub const FORWARD_CONCURRENT_METRIC_NAME: &str = "sie.worker.runtime.forward.concurrent";
pub const FORWARD_LIMIT_METRIC_NAME: &str = "sie.worker.runtime.forward.limit";

pub const REQUEST_DURATION_BUCKETS_S: &[f64] = &[
    0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
    5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
];
pub const MODEL_LOAD_DURATION_BUCKETS_S: &[f64] = &[
    0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0,
];

const METRICS_EXPORT_INTERVAL_MAX_MS: u64 = 5_000;
const METRICS_SHUTDOWN_TIMEOUT_MS: u64 = 3_000;
const MAX_DIMENSION_LENGTH: usize = 256;
const MAX_CATALOG_MODEL_PROFILE_PAIRS: usize = 256;
const MAX_REQUEST_DETAIL_MODEL_PROFILE_PAIRS: usize = 32;
const MAX_FORWARD_DETAIL_MODEL_PROFILE_PAIRS: usize = 4;
const OTHER: &str = "other";
const BACKEND: &str = "candle";

const OPERATION_SERIES: usize = 7;
const WORKER_OUTCOME_SERIES: usize = 5;
const INFERENCE_PHASE_SERIES: usize = 3;
const UNIT_TYPE_SERIES: usize = 3;
const MODEL_LOAD_OUTCOME_SERIES: usize = 4;
const MODEL_LOAD_STAGE_SERIES: usize = 5;
const OOM_STRATEGY_SERIES: usize = 4;
const OOM_OUTCOME_SERIES: usize = 4;
const MODEL_EVICTION_REASON_SERIES: usize = 10;
const FORWARD_OUTCOME_SERIES: usize = 2;
const FORWARD_INPUT_SOURCE_SERIES: usize = 3;
const FORWARD_OUTPUT_PATH_SERIES: usize = 6;
const FORWARD_STATE_SERIES: usize = 2;
const FORWARD_STAGE_SERIES: usize = 47;

pub(crate) const WORKER_CATALOG_PAIR_SERIES: usize = MAX_CATALOG_MODEL_PROFILE_PAIRS + 1;
pub(crate) const WORKER_REQUEST_DETAIL_CATALOG_PAIR_SERIES: usize =
    MAX_REQUEST_DETAIL_MODEL_PROFILE_PAIRS + 1;
pub(crate) const WORKER_FORWARD_DETAIL_CATALOG_PAIR_SERIES: usize =
    MAX_FORWARD_DETAIL_MODEL_PROFILE_PAIRS + 1;
pub(crate) const WORKER_REQUEST_CARDINALITY_LIMIT: usize =
    WORKER_REQUEST_DETAIL_CATALOG_PAIR_SERIES * OPERATION_SERIES * WORKER_OUTCOME_SERIES;
pub(crate) const WORKER_PHASE_CARDINALITY_LIMIT: usize =
    WORKER_REQUEST_CARDINALITY_LIMIT * INFERENCE_PHASE_SERIES;
pub(crate) const WORKER_UNITS_CARDINALITY_LIMIT: usize =
    WORKER_REQUEST_DETAIL_CATALOG_PAIR_SERIES * OPERATION_SERIES * UNIT_TYPE_SERIES;
pub(crate) const WORKER_MODEL_LOAD_CARDINALITY_LIMIT: usize =
    WORKER_CATALOG_PAIR_SERIES * MODEL_LOAD_OUTCOME_SERIES * MODEL_LOAD_STAGE_SERIES;
pub(crate) const WORKER_OOM_CARDINALITY_LIMIT: usize =
    WORKER_CATALOG_PAIR_SERIES * OOM_STRATEGY_SERIES * OOM_OUTCOME_SERIES;
pub(crate) const WORKER_EVICTION_CARDINALITY_LIMIT: usize =
    WORKER_CATALOG_PAIR_SERIES * MODEL_EVICTION_REASON_SERIES;
pub(crate) const WORKER_FORWARD_CARDINALITY_LIMIT: usize = WORKER_FORWARD_DETAIL_CATALOG_PAIR_SERIES
    * FORWARD_OUTCOME_SERIES
    * FORWARD_INPUT_SOURCE_SERIES
    * FORWARD_OUTPUT_PATH_SERIES
    * FORWARD_STAGE_SERIES;
pub(crate) const WORKER_FORWARD_PERMIT_CARDINALITY_LIMIT: usize =
    WORKER_CATALOG_PAIR_SERIES * FORWARD_OUTPUT_PATH_SERIES;
pub(crate) const WORKER_FORWARD_CONCURRENT_CARDINALITY_LIMIT: usize =
    WORKER_CATALOG_PAIR_SERIES * FORWARD_STATE_SERIES;

static METER_PROVIDER: OnceLock<SdkMeterProvider> = OnceLock::new();
static WORKER_METRICS: OnceLock<Option<WorkerMetrics>> = OnceLock::new();

/// Return the complete finite-domain ceiling for every Rust-engine metric.
/// Explicit views keep the SDK's 2,000-series default from becoming an
/// accidental second cardinality policy for valid catalog dimensions.
pub(crate) fn worker_metric_cardinality_limit(name: &str) -> Option<usize> {
    match name {
        REQUESTS_METRIC_NAME | REQUEST_DURATION_METRIC_NAME => {
            Some(WORKER_REQUEST_CARDINALITY_LIMIT)
        }
        INFERENCE_DURATION_METRIC_NAME => Some(WORKER_PHASE_CARDINALITY_LIMIT),
        UNITS_METRIC_NAME => Some(WORKER_UNITS_CARDINALITY_LIMIT),
        MODEL_LOADED_METRIC_NAME | MODEL_MEMORY_METRIC_NAME | FORWARD_LIMIT_METRIC_NAME => {
            Some(WORKER_CATALOG_PAIR_SERIES)
        }
        MODEL_LOAD_DURATION_METRIC_NAME => Some(WORKER_MODEL_LOAD_CARDINALITY_LIMIT),
        OOM_RECOVERIES_METRIC_NAME => Some(WORKER_OOM_CARDINALITY_LIMIT),
        MODEL_EVICTIONS_METRIC_NAME => Some(WORKER_EVICTION_CARDINALITY_LIMIT),
        FORWARD_DURATION_METRIC_NAME => Some(WORKER_FORWARD_CARDINALITY_LIMIT),
        FORWARD_PERMIT_WAIT_METRIC_NAME => Some(WORKER_FORWARD_PERMIT_CARDINALITY_LIMIT),
        FORWARD_CONCURRENT_METRIC_NAME => Some(WORKER_FORWARD_CONCURRENT_CARDINALITY_LIMIT),
        _ => None,
    }
}

fn worker_metric_cardinality_view(instrument: &Instrument) -> Option<Stream> {
    let limit = worker_metric_cardinality_limit(instrument.name())?;
    Some(
        Stream::builder()
            .with_cardinality_limit(limit)
            .build()
            .expect("constant worker cardinality limits must be valid"),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerOutcome {
    Success,
    Error,
    Retry,
    Cancelled,
    Other,
}

impl WorkerOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
            Self::Retry => "retry",
            Self::Cancelled => "cancelled",
            Self::Other => OTHER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelLoadOutcome {
    Success,
    Error,
    Timeout,
    Other,
}

impl ModelLoadOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
            Self::Timeout => "timeout",
            Self::Other => OTHER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelLoadStage {
    Total,
    Instantiate,
    Load,
    Resident,
    Other,
}

impl ModelLoadStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Total => "total",
            Self::Instantiate => "instantiate",
            Self::Load => "load",
            Self::Resident => "resident",
            Self::Other => OTHER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OomStrategy {
    CacheClear,
    EvictLru,
    SplitBatch,
    Other,
}

impl OomStrategy {
    fn as_str(self) -> &'static str {
        match self {
            Self::CacheClear => "cache_clear",
            Self::EvictLru => "evict_lru",
            Self::SplitBatch => "split_batch",
            Self::Other => OTHER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OomOutcome {
    Success,
    Failed,
    Terminal,
    Other,
}

impl OomOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failed => "failed",
            Self::Terminal => "terminal",
            Self::Other => OTHER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelEvictionReason {
    ConfigChange,
    Idle,
    Lru,
    LoadOom,
    OomRecovery,
    Other,
}

impl ModelEvictionReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::ConfigChange => "config_change",
            Self::Idle => "idle",
            Self::Lru => "lru",
            Self::LoadOom => "load_oom",
            Self::OomRecovery => "oom_recovery",
            Self::Other => OTHER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardState {
    Active,
    Waiting,
}

impl ForwardState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Waiting => "waiting",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardOutcome {
    Success,
    Error,
}

impl ForwardOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Error => "error",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardInputSource {
    Prepared,
    Raw,
    Other,
}

impl ForwardInputSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Prepared => "prepared",
            Self::Raw => "raw",
            Self::Other => OTHER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardOutputPath {
    Dense,
    Sparse,
    MultivectorF32,
    MultivectorF16Batched,
    MultivectorF16Individual,
    Other,
}

impl ForwardOutputPath {
    fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
            Self::MultivectorF32 => "multivector_f32",
            Self::MultivectorF16Batched => "multivector_f16_batched",
            Self::MultivectorF16Individual => "multivector_f16_individual",
            Self::Other => OTHER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardStage {
    Total,
    Forward,
    Pool,
    Normalize,
    Conversion,
    ConversionTensorReadback,
    ConversionHostPack,
    Inference,
    XlmRobertaEmbedding,
    XlmRobertaAttention,
    XlmRobertaAttentionQkv,
    XlmRobertaAttentionFlash,
    XlmRobertaAttentionOutputDense,
    XlmRobertaAttentionOutputLayernorm,
    XlmRobertaFfn,
    XlmRobertaFfnIntermediateDense,
    XlmRobertaFfnActivation,
    XlmRobertaFfnOutputDense,
    XlmRobertaFfnOutputLayernorm,
    GteRopeEmbedding,
    GteRopeRopeSelect,
    GteRopeAttention,
    GteRopeAttentionQkv,
    GteRopeAttentionRotary,
    GteRopeAttentionFlash,
    GteRopeAttentionOutputDense,
    GteRopeAttentionOutputLayernorm,
    GteRopeFfn,
    GteRopeFfnUpGate,
    GteRopeFfnActivation,
    GteRopeFfnDown,
    GteRopeFfnOutputLayernorm,
    ModernBertEmbedding,
    ModernBertEmbeddingNorm,
    ModernBertRopeSelect,
    ModernBertAttention,
    ModernBertAttentionNorm,
    ModernBertAttentionQkv,
    ModernBertAttentionRotary,
    ModernBertAttentionFlash,
    ModernBertAttentionOutputDense,
    ModernBertMlp,
    ModernBertMlpNorm,
    ModernBertMlpWi,
    ModernBertMlpActivation,
    ModernBertMlpWo,
    ModernBertFinalNorm,
}

impl ForwardStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Total => "total",
            Self::Forward => "forward",
            Self::Pool => "pool",
            Self::Normalize => "normalize",
            Self::Conversion => "conversion",
            Self::ConversionTensorReadback => "conversion_tensor_readback",
            Self::ConversionHostPack => "conversion_host_pack",
            Self::Inference => "inference",
            Self::XlmRobertaEmbedding => "xlm_roberta_embedding",
            Self::XlmRobertaAttention => "xlm_roberta_attention",
            Self::XlmRobertaAttentionQkv => "xlm_roberta_attention_qkv",
            Self::XlmRobertaAttentionFlash => "xlm_roberta_attention_flash",
            Self::XlmRobertaAttentionOutputDense => "xlm_roberta_attention_output_dense",
            Self::XlmRobertaAttentionOutputLayernorm => "xlm_roberta_attention_output_layernorm",
            Self::XlmRobertaFfn => "xlm_roberta_ffn",
            Self::XlmRobertaFfnIntermediateDense => "xlm_roberta_ffn_intermediate_dense",
            Self::XlmRobertaFfnActivation => "xlm_roberta_ffn_activation",
            Self::XlmRobertaFfnOutputDense => "xlm_roberta_ffn_output_dense",
            Self::XlmRobertaFfnOutputLayernorm => "xlm_roberta_ffn_output_layernorm",
            Self::GteRopeEmbedding => "gte_rope_embedding",
            Self::GteRopeRopeSelect => "gte_rope_rope_select",
            Self::GteRopeAttention => "gte_rope_attention",
            Self::GteRopeAttentionQkv => "gte_rope_attention_qkv",
            Self::GteRopeAttentionRotary => "gte_rope_attention_rotary",
            Self::GteRopeAttentionFlash => "gte_rope_attention_flash",
            Self::GteRopeAttentionOutputDense => "gte_rope_attention_output_dense",
            Self::GteRopeAttentionOutputLayernorm => "gte_rope_attention_output_layernorm",
            Self::GteRopeFfn => "gte_rope_ffn",
            Self::GteRopeFfnUpGate => "gte_rope_ffn_up_gate",
            Self::GteRopeFfnActivation => "gte_rope_ffn_activation",
            Self::GteRopeFfnDown => "gte_rope_ffn_down",
            Self::GteRopeFfnOutputLayernorm => "gte_rope_ffn_output_layernorm",
            Self::ModernBertEmbedding => "modernbert_embedding",
            Self::ModernBertEmbeddingNorm => "modernbert_embedding_norm",
            Self::ModernBertRopeSelect => "modernbert_rope_select",
            Self::ModernBertAttention => "modernbert_attention",
            Self::ModernBertAttentionNorm => "modernbert_attention_norm",
            Self::ModernBertAttentionQkv => "modernbert_attention_qkv",
            Self::ModernBertAttentionRotary => "modernbert_attention_rotary",
            Self::ModernBertAttentionFlash => "modernbert_attention_flash",
            Self::ModernBertAttentionOutputDense => "modernbert_attention_output_dense",
            Self::ModernBertMlp => "modernbert_mlp",
            Self::ModernBertMlpNorm => "modernbert_mlp_norm",
            Self::ModernBertMlpWi => "modernbert_mlp_wi",
            Self::ModernBertMlpActivation => "modernbert_mlp_activation",
            Self::ModernBertMlpWo => "modernbert_mlp_wo",
            Self::ModernBertFinalNorm => "modernbert_final_norm",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthoritativeUnits {
    pub input_tokens: Option<u64>,
    pub pages: Option<u64>,
    pub images: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PhaseDurations {
    pub tokenization_s: Option<f64>,
    pub inference_s: Option<f64>,
    pub postprocessing_s: Option<f64>,
}

pub struct ItemCompleted<'a> {
    pub operation: &'a str,
    pub outcome: WorkerOutcome,
    pub model: &'a str,
    pub profile: &'a str,
    pub duration_s: f64,
    pub phases: PhaseDurations,
    pub units: AuthoritativeUnits,
}

pub struct ForwardCompleted<'a> {
    pub model: &'a str,
    pub profile: &'a str,
    pub outcome: ForwardOutcome,
    pub input_source: ForwardInputSource,
    pub output_path: ForwardOutputPath,
    pub duration_s: f64,
    pub stages: &'a [(ForwardStage, f64)],
}

#[derive(Default)]
struct CatalogPairAdmission {
    // One table keeps the catalog and first-observed detail tiers atomic under
    // one lock; flags never reopen when later catalog pairs appear.
    admitted: HashMap<(String, String), CatalogPairDetailAdmission>,
    request_detail_count: usize,
    forward_detail_count: usize,
}

#[derive(Default)]
struct CatalogPairDetailAdmission {
    request: bool,
    forward: bool,
}

#[derive(Clone, Copy)]
enum CatalogPairScope {
    Catalog,
    RequestDetail,
    ForwardDetail,
}

struct WorkerMetrics {
    requests: Counter<u64>,
    request_duration: Histogram<f64>,
    inference_duration: Histogram<f64>,
    units: Counter<u64>,
    model_loaded: Gauge<u64>,
    model_load_duration: Histogram<f64>,
    model_memory: Gauge<u64>,
    oom_recoveries: Counter<u64>,
    model_evictions: Counter<u64>,
    forward_duration: Histogram<f64>,
    forward_permit_wait: Histogram<f64>,
    forward_concurrent: UpDownCounter<i64>,
    forward_limit: Gauge<u64>,
    lane: String,
    catalog_pairs: Mutex<CatalogPairAdmission>,
    catalog_warning_emitted: AtomicBool,
}

impl WorkerMetrics {
    fn new(meter: &Meter, lane: String) -> Self {
        Self {
            requests: meter
                .u64_counter(REQUESTS_METRIC_NAME)
                .with_description("Inference items completed by the engine.")
                .with_unit("{item}")
                .build(),
            request_duration: meter
                .f64_histogram(REQUEST_DURATION_METRIC_NAME)
                .with_description("End-to-end engine processing time per inference item.")
                .with_unit("s")
                .with_boundaries(REQUEST_DURATION_BUCKETS_S.to_vec())
                .build(),
            inference_duration: meter
                .f64_histogram(INFERENCE_DURATION_METRIC_NAME)
                .with_description("Engine processing time split into bounded phases.")
                .with_unit("s")
                .with_boundaries(REQUEST_DURATION_BUCKETS_S.to_vec())
                .build(),
            units: meter
                .u64_counter(UNITS_METRIC_NAME)
                .with_description("Authoritative non-generation units completed by the engine.")
                .with_unit("{unit}")
                .build(),
            model_loaded: meter
                .u64_gauge(MODEL_LOADED_METRIC_NAME)
                .with_description("Whether a catalog model is resident in the engine.")
                .with_unit("{model}")
                .build(),
            model_load_duration: meter
                .f64_histogram(MODEL_LOAD_DURATION_METRIC_NAME)
                .with_description("Model load duration.")
                .with_unit("s")
                .with_boundaries(MODEL_LOAD_DURATION_BUCKETS_S.to_vec())
                .build(),
            model_memory: meter
                .u64_gauge(MODEL_MEMORY_METRIC_NAME)
                .with_description("Engine-reported resident model memory.")
                .with_unit("By")
                .build(),
            oom_recoveries: meter
                .u64_counter(OOM_RECOVERIES_METRIC_NAME)
                .with_description("OOM recovery strategy outcomes.")
                .with_unit("{recovery}")
                .build(),
            model_evictions: meter
                .u64_counter(MODEL_EVICTIONS_METRIC_NAME)
                .with_description("Resident models evicted from the engine by bounded cause.")
                .with_unit("{model}")
                .build(),
            forward_duration: meter
                .f64_histogram(FORWARD_DURATION_METRIC_NAME)
                .with_description("Candle forward duration split by bounded execution stage.")
                .with_unit("s")
                .with_boundaries(REQUEST_DURATION_BUCKETS_S.to_vec())
                .build(),
            forward_permit_wait: meter
                .f64_histogram(FORWARD_PERMIT_WAIT_METRIC_NAME)
                .with_description("Time spent waiting for a Candle forward permit.")
                .with_unit("s")
                .with_boundaries(REQUEST_DURATION_BUCKETS_S.to_vec())
                .build(),
            forward_concurrent: meter
                .i64_up_down_counter(FORWARD_CONCURRENT_METRIC_NAME)
                .with_description("Candle forwards active or waiting for a permit.")
                .with_unit("{forward}")
                .build(),
            forward_limit: meter
                .u64_gauge(FORWARD_LIMIT_METRIC_NAME)
                .with_description("Configured concurrent Candle forward limit per loaded model.")
                .with_unit("{forward}")
                .build(),
            lane: bounded_release_value(&lane),
            catalog_pairs: Mutex::new(CatalogPairAdmission::default()),
            catalog_warning_emitted: AtomicBool::new(false),
        }
    }

    fn record_item_completed(&self, event: ItemCompleted<'_>) {
        let attributes =
            self.request_attributes(event.operation, event.outcome, event.model, event.profile);
        self.requests.add(1, &attributes);
        if let Some(duration_s) = nonnegative_finite(event.duration_s) {
            self.request_duration.record(duration_s, &attributes);
        }
        for (phase, duration_s) in [
            ("tokenization", event.phases.tokenization_s),
            ("inference", event.phases.inference_s),
            ("postprocessing", event.phases.postprocessing_s),
        ] {
            let Some(duration_s) = duration_s.and_then(nonnegative_finite) else {
                continue;
            };
            let mut phase_attributes = attributes.to_vec();
            phase_attributes.push(KeyValue::new("phase", phase));
            self.inference_duration
                .record(duration_s, &phase_attributes);
        }
        if event.outcome == WorkerOutcome::Success {
            for (unit_type, value) in [
                ("input_tokens", event.units.input_tokens),
                ("pages", event.units.pages),
                ("images", event.units.images),
            ] {
                let Some(value) = value.filter(|value| *value > 0) else {
                    continue;
                };
                let mut unit_attributes = vec![
                    attributes[0].clone(),
                    attributes[2].clone(),
                    attributes[3].clone(),
                    attributes[4].clone(),
                    attributes[5].clone(),
                ];
                unit_attributes.push(KeyValue::new("unit.type", unit_type));
                self.units.add(value, &unit_attributes);
            }
        }
    }

    fn record_model_residency_changed(
        &self,
        model: &str,
        profile: &str,
        loaded: bool,
        memory_bytes: Option<u64>,
    ) {
        let attributes = self.model_attributes(model, profile);
        self.model_loaded.record(u64::from(loaded), &attributes);
        if let Some(memory_bytes) = memory_bytes {
            self.model_memory.record(memory_bytes, &attributes);
        }
    }

    fn record_model_load_completed(
        &self,
        model: &str,
        profile: &str,
        outcome: ModelLoadOutcome,
        stage: ModelLoadStage,
        duration_s: f64,
    ) {
        let Some(duration_s) = nonnegative_finite(duration_s) else {
            return;
        };
        let mut attributes = Vec::with_capacity(6);
        attributes.push(KeyValue::new("outcome", outcome.as_str()));
        attributes.push(KeyValue::new("stage", stage.as_str()));
        attributes.extend(self.model_attributes(model, profile));
        self.model_load_duration.record(duration_s, &attributes);
    }

    fn record_oom_recovery_completed(
        &self,
        model: &str,
        profile: &str,
        strategy: OomStrategy,
        outcome: OomOutcome,
    ) {
        let mut attributes = Vec::with_capacity(6);
        attributes.push(KeyValue::new("strategy", strategy.as_str()));
        attributes.push(KeyValue::new("outcome", outcome.as_str()));
        attributes.extend(self.model_attributes(model, profile));
        self.oom_recoveries.add(1, &attributes);
    }

    fn record_model_evicted(&self, model: &str, profile: &str, reason: ModelEvictionReason) {
        let mut attributes = Vec::with_capacity(5);
        attributes.push(KeyValue::new("reason", reason.as_str()));
        attributes.extend(self.model_attributes(model, profile));
        self.model_evictions.add(1, &attributes);
    }

    fn record_forward_permit_wait(
        &self,
        model: &str,
        profile: &str,
        output_path: ForwardOutputPath,
        duration_s: f64,
    ) {
        let Some(duration_s) = nonnegative_finite(duration_s) else {
            return;
        };
        let mut attributes = Vec::with_capacity(5);
        attributes.push(KeyValue::new("output.path", output_path.as_str()));
        attributes.extend(self.model_attributes(model, profile));
        self.forward_permit_wait.record(duration_s, &attributes);
    }

    fn record_forward_completed(&self, event: ForwardCompleted<'_>) {
        let Some(duration_s) = nonnegative_finite(event.duration_s) else {
            return;
        };
        let base_attributes = self.forward_attributes(
            event.model,
            event.profile,
            event.outcome,
            event.input_source,
            event.output_path,
        );
        self.record_forward_stage(duration_s, ForwardStage::Total, &base_attributes);
        for (stage, raw_duration_s) in event.stages {
            if let Some(stage_duration_s) = nonnegative_finite(*raw_duration_s) {
                self.record_forward_stage(stage_duration_s, *stage, &base_attributes);
            }
        }
    }

    fn record_forward_stage(
        &self,
        duration_s: f64,
        stage: ForwardStage,
        base_attributes: &[KeyValue],
    ) {
        let mut attributes = Vec::with_capacity(base_attributes.len() + 1);
        attributes.extend_from_slice(base_attributes);
        attributes.push(KeyValue::new("stage", stage.as_str()));
        self.forward_duration.record(duration_s, &attributes);
    }

    fn begin_forward_activity(
        &self,
        model: &str,
        profile: &str,
        state: ForwardState,
        limit: usize,
    ) -> ForwardActivityGuard<'_> {
        let model_attributes = self.model_attributes(model, profile);
        let mut attributes = Vec::with_capacity(5);
        attributes.push(KeyValue::new("state", state.as_str()));
        attributes.extend(model_attributes.clone());
        self.forward_concurrent.add(1, &attributes);
        self.forward_limit
            .record(u64::try_from(limit).unwrap_or(u64::MAX), &model_attributes);
        ForwardActivityGuard {
            metrics: Some(self),
            attributes,
        }
    }

    fn request_attributes(
        &self,
        operation: &str,
        outcome: WorkerOutcome,
        model: &str,
        profile: &str,
    ) -> [KeyValue; 6] {
        let (model, profile, backend) = self.admit_request_model_profile(model, profile);
        [
            KeyValue::new("operation", bounded_operation(operation)),
            KeyValue::new("outcome", outcome.as_str()),
            KeyValue::new("backend", backend),
            KeyValue::new("lane", self.lane.clone()),
            KeyValue::new("model", model),
            KeyValue::new("profile", profile),
        ]
    }

    fn model_attributes(&self, model: &str, profile: &str) -> [KeyValue; 4] {
        let (model, profile, backend) = self.admit_model_profile(model, profile);
        [
            KeyValue::new("backend", backend),
            KeyValue::new("lane", self.lane.clone()),
            KeyValue::new("model", model),
            KeyValue::new("profile", profile),
        ]
    }

    /// Admit exact catalog dimensions for the process lifetime. The service
    /// call sites pass already catalog-resolved values; this final facade
    /// boundary prevents sequential catalog churn from retaining unbounded SDK
    /// series. Collapse affects telemetry only and never serving decisions.
    fn admit_model_profile(&self, model: &str, profile: &str) -> (String, String, &'static str) {
        self.admit_model_profile_for(model, profile, CatalogPairScope::Catalog)
    }

    fn admit_request_model_profile(
        &self,
        model: &str,
        profile: &str,
    ) -> (String, String, &'static str) {
        self.admit_model_profile_for(model, profile, CatalogPairScope::RequestDetail)
    }

    fn admit_forward_model_profile(
        &self,
        model: &str,
        profile: &str,
    ) -> (String, String, &'static str) {
        self.admit_model_profile_for(model, profile, CatalogPairScope::ForwardDetail)
    }

    fn admit_model_profile_for(
        &self,
        model: &str,
        profile: &str,
        scope: CatalogPairScope,
    ) -> (String, String, &'static str) {
        let pair = (bounded_release_value(model), bounded_release_value(profile));
        if pair.0 == OTHER || pair.1 == OTHER {
            return (OTHER.to_string(), OTHER.to_string(), OTHER);
        }

        let (exact, exhausted_scope, limit) = {
            let mut admission = self
                .catalog_pairs
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let existing_scope_admitted = admission.admitted.get(&pair).map(|detail| match scope {
                CatalogPairScope::Catalog => true,
                CatalogPairScope::RequestDetail => detail.request,
                CatalogPairScope::ForwardDetail => detail.forward,
            });
            if existing_scope_admitted == Some(true) {
                match scope {
                    CatalogPairScope::Catalog => (true, "catalog", MAX_CATALOG_MODEL_PROFILE_PAIRS),
                    CatalogPairScope::RequestDetail => (
                        true,
                        "request_detail",
                        MAX_REQUEST_DETAIL_MODEL_PROFILE_PAIRS,
                    ),
                    CatalogPairScope::ForwardDetail => (
                        true,
                        "forward_detail",
                        MAX_FORWARD_DETAIL_MODEL_PROFILE_PAIRS,
                    ),
                }
            } else if existing_scope_admitted.is_none()
                && admission.admitted.len() >= MAX_CATALOG_MODEL_PROFILE_PAIRS
            {
                (false, "catalog", MAX_CATALOG_MODEL_PROFILE_PAIRS)
            } else {
                if existing_scope_admitted.is_none() {
                    admission.admitted.insert(pair.clone(), Default::default());
                }
                match scope {
                    CatalogPairScope::Catalog => (true, "catalog", MAX_CATALOG_MODEL_PROFILE_PAIRS),
                    CatalogPairScope::RequestDetail => {
                        let exact =
                            admission.request_detail_count < MAX_REQUEST_DETAIL_MODEL_PROFILE_PAIRS;
                        if exact {
                            admission.request_detail_count += 1;
                            admission
                                .admitted
                                .get_mut(&pair)
                                .expect("catalog pair was admitted")
                                .request = true;
                        }
                        (
                            exact,
                            "request_detail",
                            MAX_REQUEST_DETAIL_MODEL_PROFILE_PAIRS,
                        )
                    }
                    CatalogPairScope::ForwardDetail => {
                        let exact =
                            admission.forward_detail_count < MAX_FORWARD_DETAIL_MODEL_PROFILE_PAIRS;
                        if exact {
                            admission.forward_detail_count += 1;
                            admission
                                .admitted
                                .get_mut(&pair)
                                .expect("catalog pair was admitted")
                                .forward = true;
                        }
                        (
                            exact,
                            "forward_detail",
                            MAX_FORWARD_DETAIL_MODEL_PROFILE_PAIRS,
                        )
                    }
                }
            }
        };
        if !exact && !self.catalog_warning_emitted.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                admission_scope = exhausted_scope,
                admitted_pairs = limit,
                "Rust worker telemetry model-profile budget exhausted; later pairs collapse to other"
            );
        }
        if exact {
            (pair.0, pair.1, BACKEND)
        } else {
            (OTHER.to_string(), OTHER.to_string(), OTHER)
        }
    }

    fn forward_attributes(
        &self,
        model: &str,
        profile: &str,
        outcome: ForwardOutcome,
        input_source: ForwardInputSource,
        output_path: ForwardOutputPath,
    ) -> Vec<KeyValue> {
        let (model, profile, backend) = self.admit_forward_model_profile(model, profile);
        vec![
            KeyValue::new("outcome", outcome.as_str()),
            KeyValue::new("input.source", input_source.as_str()),
            KeyValue::new("output.path", output_path.as_str()),
            KeyValue::new("backend", backend),
            KeyValue::new("lane", self.lane.clone()),
            KeyValue::new("model", model),
            KeyValue::new("profile", profile),
        ]
    }
}

pub struct ForwardActivityGuard<'a> {
    metrics: Option<&'a WorkerMetrics>,
    attributes: Vec<KeyValue>,
}

impl ForwardActivityGuard<'_> {
    fn disabled() -> Self {
        Self {
            metrics: None,
            attributes: Vec::new(),
        }
    }
}

impl Drop for ForwardActivityGuard<'_> {
    fn drop(&mut self) {
        if let Some(metrics) = self.metrics {
            metrics.forward_concurrent.add(-1, &self.attributes);
        }
    }
}

/// Initialise the sole canonical OTLP metric backend. Failures are fail-open.
pub fn init_metrics() {
    static INIT_GUARD: AtomicBool = AtomicBool::new(false);
    if INIT_GUARD.swap(true, Ordering::SeqCst) {
        return;
    }

    let enabled = positive_flag(cleaned_env("SIE_METRICS_ENABLED").as_deref());
    let export_config = match metric_export_config(enabled) {
        Ok(config) => config,
        Err(_) => {
            let _ = WORKER_METRICS.set(None);
            tracing::warn!("invalid OTLP metric transport configuration; metrics disabled");
            return;
        }
    };
    let Some(export_config) = export_config else {
        let _ = WORKER_METRICS.set(None);
        if enabled {
            tracing::warn!("SIE_METRICS_ENABLED set but no metric OTLP endpoint; metrics disabled");
        }
        return;
    };

    let exporter = match build_metric_exporter(&export_config) {
        Ok(exporter) => exporter,
        Err(_) => {
            let _ = WORKER_METRICS.set(None);
            tracing::warn!("failed to build OTLP metric exporter; metrics disabled");
            return;
        }
    };
    let reader = PeriodicReader::builder(exporter)
        .with_interval(metric_export_interval(
            cleaned_env("OTEL_METRIC_EXPORT_INTERVAL").as_deref(),
        ))
        .build();
    let provider = SdkMeterProvider::builder()
        .with_reader(reader)
        .with_resource(telemetry_resource())
        .with_view(worker_metric_cardinality_view)
        .build();
    let metrics = WorkerMetrics::new(&provider.meter("sie-worker"), lane_from_environment());

    global::set_meter_provider(provider.clone());
    let _ = METER_PROVIDER.set(provider);
    let _ = WORKER_METRICS.set(Some(metrics));
    tracing::info!(endpoint = %endpoint_origin_for_log(&export_config.endpoint), protocol = ?export_config.protocol, "OpenTelemetry worker metrics initialized");
}

pub fn shutdown_metrics() {
    if let Some(provider) = METER_PROVIDER.get() {
        let _ = provider.shutdown_with_timeout(Duration::from_millis(METRICS_SHUTDOWN_TIMEOUT_MS));
    }
}

#[inline]
pub fn metrics_enabled() -> bool {
    WORKER_METRICS.get().is_some_and(Option::is_some)
}

#[inline]
fn enabled_metrics() -> Option<&'static WorkerMetrics> {
    WORKER_METRICS.get().and_then(Option::as_ref)
}

#[inline]
pub fn record_item_completed(event: ItemCompleted<'_>) {
    let Some(metrics) = enabled_metrics() else {
        return;
    };
    metrics.record_item_completed(event);
}

#[inline]
pub fn record_model_residency_changed(
    model: &str,
    profile: &str,
    loaded: bool,
    memory_bytes: Option<u64>,
) {
    let Some(metrics) = enabled_metrics() else {
        return;
    };
    metrics.record_model_residency_changed(model, profile, loaded, memory_bytes);
}

#[inline]
pub fn record_model_load_completed(
    model: &str,
    profile: &str,
    outcome: ModelLoadOutcome,
    stage: ModelLoadStage,
    duration_s: f64,
) {
    let Some(metrics) = enabled_metrics() else {
        return;
    };
    metrics.record_model_load_completed(model, profile, outcome, stage, duration_s);
}

#[inline]
pub fn record_oom_recovery_completed(
    model: &str,
    profile: &str,
    strategy: OomStrategy,
    outcome: OomOutcome,
) {
    let Some(metrics) = enabled_metrics() else {
        return;
    };
    metrics.record_oom_recovery_completed(model, profile, strategy, outcome);
}

#[inline]
pub fn record_model_evicted(model: &str, profile: &str, reason: ModelEvictionReason) {
    let Some(metrics) = enabled_metrics() else {
        return;
    };
    metrics.record_model_evicted(model, profile, reason);
}

#[inline]
pub fn record_forward_permit_wait(
    model: &str,
    profile: &str,
    output_path: ForwardOutputPath,
    duration_s: f64,
) {
    let Some(metrics) = enabled_metrics() else {
        return;
    };
    metrics.record_forward_permit_wait(model, profile, output_path, duration_s);
}

#[inline]
pub fn record_forward_completed(event: ForwardCompleted<'_>) {
    let Some(metrics) = enabled_metrics() else {
        return;
    };
    metrics.record_forward_completed(event);
}

#[inline]
pub fn begin_forward_activity(
    model: &str,
    profile: &str,
    state: ForwardState,
    limit: usize,
) -> ForwardActivityGuard<'static> {
    let Some(metrics) = enabled_metrics() else {
        return ForwardActivityGuard::disabled();
    };
    metrics.begin_forward_activity(model, profile, state, limit)
}

fn positive_flag(raw: Option<&str>) -> bool {
    raw.is_some_and(|value| {
        let value = value.trim();
        value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("yes")
    })
}

fn metric_export_interval(raw: Option<&str>) -> Duration {
    let milliseconds = raw
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(METRICS_EXPORT_INTERVAL_MAX_MS)
        .min(METRICS_EXPORT_INTERVAL_MAX_MS);
    Duration::from_millis(milliseconds)
}

fn lane_from_environment() -> String {
    let pool = bounded_release_value(&cleaned_env("SIE_POOL").unwrap_or_else(|| "default".into()));
    let machine = bounded_release_value(
        &cleaned_env("SIE_MACHINE_PROFILE").unwrap_or_else(|| "default".into()),
    );
    let bundle =
        bounded_release_value(&cleaned_env("SIE_BUNDLE").unwrap_or_else(|| "default".into()));
    if [pool.as_str(), machine.as_str(), bundle.as_str()].contains(&OTHER) {
        OTHER.to_string()
    } else {
        bounded_release_value(&format!("{pool}|{machine}|{bundle}"))
    }
}

fn bounded_operation(value: &str) -> &'static str {
    match value.trim().to_ascii_lowercase().as_str() {
        "encode" => "encode",
        "score" => "score",
        "extract" => "extract",
        "embeddings" => "embeddings",
        "moderations" => "moderations",
        "generate" => "generate",
        _ => OTHER,
    }
}

fn bounded_release_value(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() || value.len() > MAX_DIMENSION_LENGTH || !value.is_ascii() {
        OTHER.to_string()
    } else {
        value.to_string()
    }
}

fn nonnegative_finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value.max(0.0))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use opentelemetry::KeyValue;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, Metric, MetricData};
    use opentelemetry_sdk::metrics::{
        InMemoryMetricExporter, InMemoryMetricExporterBuilder, Temporality,
    };

    use super::*;

    #[test]
    fn export_interval_cannot_exceed_the_contract_budget() {
        assert_eq!(metric_export_interval(None), Duration::from_secs(5));
        assert_eq!(metric_export_interval(Some("1000")), Duration::from_secs(1));
        assert_eq!(
            metric_export_interval(Some("30000")),
            Duration::from_secs(5)
        );
        assert_eq!(metric_export_interval(Some("0")), Duration::from_secs(5));
    }

    #[test]
    fn cancelled_worker_outcome_is_not_collapsed() {
        assert_eq!(WorkerOutcome::Cancelled.as_str(), "cancelled");
    }

    #[test]
    fn sparse_forward_output_path_has_a_stable_label() {
        assert_eq!(ForwardOutputPath::Sparse.as_str(), "sparse");
    }

    #[test]
    fn low_memory_wire_policy_resets_additive_metrics_but_preserves_state() {
        let exporter = InMemoryMetricExporterBuilder::new()
            .with_temporality(Temporality::LowMemory)
            .build();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let meter = provider.meter("temporality-contract-test");
        let counter = meter.u64_counter("test.counter").build();
        let histogram = meter.f64_histogram("test.histogram").build();
        let up_down = meter.i64_up_down_counter("test.up_down").build();
        let gauge = meter.u64_gauge("test.gauge").build();

        counter.add(1, &[]);
        histogram.record(0.1, &[]);
        up_down.add(2, &[]);
        gauge.record(7, &[]);
        provider.force_flush().expect("flush first interval");

        counter.add(3, &[]);
        histogram.record(0.2, &[]);
        up_down.add(-1, &[]);
        provider.force_flush().expect("flush second interval");

        let finished = exporter
            .get_finished_metrics()
            .expect("read metric intervals");
        assert_eq!(finished.len(), 2);
        let intervals: Vec<HashMap<_, _>> = finished
            .iter()
            .map(|resource| {
                resource
                    .scope_metrics()
                    .flat_map(|scope| scope.metrics())
                    .map(|metric| (metric.name(), metric))
                    .collect()
            })
            .collect();

        for (index, expected_counter) in [1, 3].into_iter().enumerate() {
            let AggregatedMetrics::U64(MetricData::Sum(sum)) =
                intervals[index]["test.counter"].data()
            else {
                panic!("counter must be a u64 sum");
            };
            assert_eq!(sum.temporality(), Temporality::Delta);
            assert_eq!(
                sum.data_points().next().expect("counter point").value(),
                expected_counter
            );

            let AggregatedMetrics::F64(MetricData::Histogram(histogram)) =
                intervals[index]["test.histogram"].data()
            else {
                panic!("histogram must be f64");
            };
            assert_eq!(histogram.temporality(), Temporality::Delta);
            assert_eq!(
                histogram
                    .data_points()
                    .next()
                    .expect("histogram point")
                    .count(),
                1
            );
        }

        for (index, expected_value) in [2, 1].into_iter().enumerate() {
            let AggregatedMetrics::I64(MetricData::Sum(sum)) =
                intervals[index]["test.up_down"].data()
            else {
                panic!("up/down counter must be an i64 sum");
            };
            assert_eq!(sum.temporality(), Temporality::Cumulative);
            assert_eq!(
                sum.data_points().next().expect("up/down point").value(),
                expected_value
            );

            let AggregatedMetrics::U64(MetricData::Gauge(gauge)) =
                intervals[index]["test.gauge"].data()
            else {
                panic!("gauge must remain current-value data");
            };
            assert_eq!(gauge.data_points().next().expect("gauge point").value(), 7);
        }
        provider.shutdown().expect("shutdown test provider");
    }

    #[test]
    fn one_facade_event_exports_the_canonical_worker_families_and_attributes() {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder().with_reader(reader).build();
        let metrics = WorkerMetrics::new(
            &provider.meter("sie-worker-test"),
            "default|gpu-a10|candle".to_string(),
        );

        metrics.record_item_completed(ItemCompleted {
            operation: "encode",
            outcome: WorkerOutcome::Success,
            model: "BAAI/bge-m3",
            profile: "candle",
            duration_s: 0.25,
            phases: PhaseDurations {
                tokenization_s: Some(0.01),
                inference_s: Some(0.2),
                postprocessing_s: Some(0.04),
            },
            units: AuthoritativeUnits {
                input_tokens: Some(17),
                ..AuthoritativeUnits::default()
            },
        });
        metrics.record_model_load_completed(
            "BAAI/bge-m3",
            "candle",
            ModelLoadOutcome::Success,
            ModelLoadStage::Total,
            2.0,
        );
        metrics.record_model_residency_changed("BAAI/bge-m3", "candle", true, Some(1024));
        metrics.record_oom_recovery_completed(
            "BAAI/bge-m3",
            "candle",
            OomStrategy::SplitBatch,
            OomOutcome::Success,
        );
        metrics.record_model_evicted("BAAI/bge-m3", "candle", ModelEvictionReason::Idle);
        metrics.record_forward_permit_wait("BAAI/bge-m3", "candle", ForwardOutputPath::Dense, 0.01);
        {
            let _waiting =
                metrics.begin_forward_activity("BAAI/bge-m3", "candle", ForwardState::Waiting, 2);
        }
        metrics.record_forward_completed(ForwardCompleted {
            model: "BAAI/bge-m3",
            profile: "candle",
            outcome: ForwardOutcome::Success,
            input_source: ForwardInputSource::Prepared,
            output_path: ForwardOutputPath::Dense,
            duration_s: 0.2,
            stages: &[(ForwardStage::Inference, 0.15)],
        });
        provider.force_flush().expect("flush test metrics");

        let finished = exporter.get_finished_metrics().expect("read test metrics");
        let all_metrics: Vec<_> = finished
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .collect();
        let names: BTreeSet<_> = all_metrics.iter().map(|metric| metric.name()).collect();
        assert_eq!(
            names,
            BTreeSet::from([
                REQUESTS_METRIC_NAME,
                REQUEST_DURATION_METRIC_NAME,
                INFERENCE_DURATION_METRIC_NAME,
                UNITS_METRIC_NAME,
                MODEL_LOADED_METRIC_NAME,
                MODEL_LOAD_DURATION_METRIC_NAME,
                MODEL_MEMORY_METRIC_NAME,
                OOM_RECOVERIES_METRIC_NAME,
                MODEL_EVICTIONS_METRIC_NAME,
                FORWARD_DURATION_METRIC_NAME,
                FORWARD_PERMIT_WAIT_METRIC_NAME,
                FORWARD_CONCURRENT_METRIC_NAME,
                FORWARD_LIMIT_METRIC_NAME,
            ])
        );

        let requests = all_metrics
            .iter()
            .find(|metric| metric.name() == REQUESTS_METRIC_NAME)
            .expect("request counter");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = requests.data() else {
            panic!("requests must be a u64 sum");
        };
        let point = sum.data_points().next().expect("request point");
        assert_eq!(point.value(), 1);
        assert_eq!(
            attribute_key_sets(requests),
            vec![string_set([
                "operation",
                "outcome",
                "backend",
                "lane",
                "model",
                "profile"
            ])]
        );

        let expected_attributes = [
            (
                REQUESTS_METRIC_NAME,
                string_set([
                    "operation",
                    "outcome",
                    "backend",
                    "lane",
                    "model",
                    "profile",
                ]),
            ),
            (
                REQUEST_DURATION_METRIC_NAME,
                string_set([
                    "operation",
                    "outcome",
                    "backend",
                    "lane",
                    "model",
                    "profile",
                ]),
            ),
            (
                INFERENCE_DURATION_METRIC_NAME,
                string_set([
                    "operation",
                    "outcome",
                    "phase",
                    "backend",
                    "lane",
                    "model",
                    "profile",
                ]),
            ),
            (
                UNITS_METRIC_NAME,
                string_set([
                    "operation",
                    "backend",
                    "lane",
                    "model",
                    "profile",
                    "unit.type",
                ]),
            ),
            (
                MODEL_LOADED_METRIC_NAME,
                string_set(["backend", "lane", "model", "profile"]),
            ),
            (
                MODEL_LOAD_DURATION_METRIC_NAME,
                string_set(["outcome", "stage", "backend", "lane", "model", "profile"]),
            ),
            (
                MODEL_MEMORY_METRIC_NAME,
                string_set(["backend", "lane", "model", "profile"]),
            ),
            (
                OOM_RECOVERIES_METRIC_NAME,
                string_set(["strategy", "outcome", "backend", "lane", "model", "profile"]),
            ),
            (
                MODEL_EVICTIONS_METRIC_NAME,
                string_set(["reason", "backend", "lane", "model", "profile"]),
            ),
            (
                FORWARD_DURATION_METRIC_NAME,
                string_set([
                    "outcome",
                    "input.source",
                    "output.path",
                    "stage",
                    "backend",
                    "lane",
                    "model",
                    "profile",
                ]),
            ),
            (
                FORWARD_PERMIT_WAIT_METRIC_NAME,
                string_set(["output.path", "backend", "lane", "model", "profile"]),
            ),
            (
                FORWARD_CONCURRENT_METRIC_NAME,
                string_set(["state", "backend", "lane", "model", "profile"]),
            ),
            (
                FORWARD_LIMIT_METRIC_NAME,
                string_set(["backend", "lane", "model", "profile"]),
            ),
        ];
        for (name, expected) in expected_attributes {
            let metric = all_metrics
                .iter()
                .find(|metric| metric.name() == name)
                .unwrap_or_else(|| panic!("missing {name}"));
            let point_keys = attribute_key_sets(metric);
            assert!(
                !point_keys.is_empty(),
                "{name} must export at least one point"
            );
            assert!(
                point_keys.iter().all(|keys| keys == &expected),
                "{name} attribute set drifted: {point_keys:?}"
            );
        }

        let request_duration = all_metrics
            .iter()
            .find(|metric| metric.name() == REQUEST_DURATION_METRIC_NAME)
            .expect("request duration histogram");
        let AggregatedMetrics::F64(MetricData::Histogram(histogram)) = request_duration.data()
        else {
            panic!("request duration must be an f64 histogram");
        };
        assert_eq!(
            histogram
                .data_points()
                .next()
                .expect("request duration point")
                .bounds()
                .collect::<Vec<_>>(),
            REQUEST_DURATION_BUCKETS_S
        );
    }

    fn string_set<const N: usize>(values: [&str; N]) -> BTreeSet<String> {
        values.into_iter().map(str::to_string).collect()
    }

    fn point_keys<'a>(attributes: impl Iterator<Item = &'a KeyValue>) -> BTreeSet<String> {
        attributes
            .map(|attribute| attribute.key.as_str().to_string())
            .collect()
    }

    fn string_attribute(attributes: &[KeyValue], key: &str) -> String {
        attributes
            .iter()
            .find(|attribute| attribute.key.as_str() == key)
            .unwrap_or_else(|| panic!("missing {key} attribute"))
            .value
            .as_str()
            .into_owned()
    }

    fn attribute_key_sets(metric: &Metric) -> Vec<BTreeSet<String>> {
        match metric.data() {
            AggregatedMetrics::U64(MetricData::Sum(sum)) => sum
                .data_points()
                .map(|point| point_keys(point.attributes()))
                .collect(),
            AggregatedMetrics::U64(MetricData::Gauge(gauge)) => gauge
                .data_points()
                .map(|point| point_keys(point.attributes()))
                .collect(),
            AggregatedMetrics::I64(MetricData::Sum(sum)) => sum
                .data_points()
                .map(|point| point_keys(point.attributes()))
                .collect(),
            AggregatedMetrics::F64(MetricData::Histogram(histogram)) => histogram
                .data_points()
                .map(|point| point_keys(point.attributes()))
                .collect(),
            other => panic!("unexpected aggregation for {}: {other:?}", metric.name()),
        }
    }

    #[test]
    fn dimensions_and_measurements_are_bounded_before_recording() {
        assert_eq!(bounded_operation("unknown"), OTHER);
        assert_eq!(bounded_release_value(""), OTHER);
        assert_eq!(bounded_release_value("naïve"), OTHER);
        assert_eq!(nonnegative_finite(f64::NAN), None);
        assert_eq!(nonnegative_finite(-1.0), Some(0.0));
    }

    #[test]
    fn catalog_pair_admission_is_process_lifetime_bounded_and_fail_open() {
        let provider = SdkMeterProvider::builder().build();
        let metrics = WorkerMetrics::new(
            &provider.meter("sie-worker-cardinality-admission-test"),
            "default|gpu-a10|candle".to_string(),
        );

        let invalid = metrics.model_attributes("naïve", "candle");
        assert_eq!(string_attribute(&invalid, "backend"), OTHER);
        assert_eq!(string_attribute(&invalid, "model"), OTHER);
        assert_eq!(string_attribute(&invalid, "profile"), OTHER);
        assert!(metrics
            .catalog_pairs
            .lock()
            .expect("catalog admission")
            .admitted
            .is_empty());

        for index in 0..MAX_CATALOG_MODEL_PROFILE_PAIRS {
            let model = format!("catalog/model-{index}");
            let attributes = metrics.model_attributes(&model, "candle");
            assert_eq!(string_attribute(&attributes, "model"), model);
            assert_eq!(string_attribute(&attributes, "profile"), "candle");
        }

        let overflow = metrics.model_attributes("catalog/overflow", "candle");
        assert_eq!(string_attribute(&overflow, "backend"), OTHER);
        assert_eq!(string_attribute(&overflow, "model"), OTHER);
        assert_eq!(string_attribute(&overflow, "profile"), OTHER);

        let retained = metrics.model_attributes("catalog/model-0", "candle");
        assert_eq!(string_attribute(&retained, "model"), "catalog/model-0");
        let admission = metrics
            .catalog_pairs
            .lock()
            .expect("catalog admission after overflow");
        assert_eq!(admission.admitted.len(), MAX_CATALOG_MODEL_PROFILE_PAIRS);
        drop(admission);
        assert!(metrics.catalog_warning_emitted.load(Ordering::Relaxed));
    }

    #[test]
    fn high_product_streams_have_smaller_telemetry_only_pair_admission() {
        let provider = SdkMeterProvider::builder().build();
        let metrics = WorkerMetrics::new(
            &provider.meter("sie-worker-detail-admission-test"),
            "default|gpu-a10|candle".to_string(),
        );

        for index in 0..MAX_REQUEST_DETAIL_MODEL_PROFILE_PAIRS {
            let model = format!("catalog/request-model-{index}");
            let attributes =
                metrics.request_attributes("encode", WorkerOutcome::Success, &model, "candle");
            assert_eq!(string_attribute(&attributes, "model"), model);
        }
        let request_overflow_model = "catalog/request-overflow";
        let request_overflow = metrics.request_attributes(
            "encode",
            WorkerOutcome::Success,
            request_overflow_model,
            "candle",
        );
        assert_eq!(string_attribute(&request_overflow, "backend"), OTHER);
        assert_eq!(string_attribute(&request_overflow, "model"), OTHER);
        assert_eq!(string_attribute(&request_overflow, "profile"), OTHER);
        assert_eq!(
            string_attribute(
                &metrics.model_attributes(request_overflow_model, "candle"),
                "model"
            ),
            request_overflow_model
        );

        for index in 0..MAX_FORWARD_DETAIL_MODEL_PROFILE_PAIRS {
            let model = format!("catalog/forward-model-{index}");
            let attributes = metrics.forward_attributes(
                &model,
                "candle",
                ForwardOutcome::Success,
                ForwardInputSource::Prepared,
                ForwardOutputPath::Dense,
            );
            assert_eq!(string_attribute(&attributes, "model"), model);
        }
        let forward_overflow_model = "catalog/forward-overflow";
        let forward_overflow = metrics.forward_attributes(
            forward_overflow_model,
            "candle",
            ForwardOutcome::Success,
            ForwardInputSource::Prepared,
            ForwardOutputPath::Dense,
        );
        assert_eq!(string_attribute(&forward_overflow, "backend"), OTHER);
        assert_eq!(string_attribute(&forward_overflow, "model"), OTHER);
        assert_eq!(string_attribute(&forward_overflow, "profile"), OTHER);
        assert_eq!(
            string_attribute(
                &metrics.model_attributes(forward_overflow_model, "candle"),
                "model"
            ),
            forward_overflow_model
        );

        let admission = metrics.catalog_pairs.lock().expect("detail admission");
        assert_eq!(
            admission.request_detail_count,
            MAX_REQUEST_DETAIL_MODEL_PROFILE_PAIRS
        );
        assert_eq!(
            admission.forward_detail_count,
            MAX_FORWARD_DETAIL_MODEL_PROFILE_PAIRS
        );
    }

    #[test]
    fn worker_cardinality_views_cover_every_owned_instrument() {
        let names = [
            REQUESTS_METRIC_NAME,
            REQUEST_DURATION_METRIC_NAME,
            INFERENCE_DURATION_METRIC_NAME,
            UNITS_METRIC_NAME,
            MODEL_LOADED_METRIC_NAME,
            MODEL_LOAD_DURATION_METRIC_NAME,
            MODEL_MEMORY_METRIC_NAME,
            OOM_RECOVERIES_METRIC_NAME,
            MODEL_EVICTIONS_METRIC_NAME,
            FORWARD_DURATION_METRIC_NAME,
            FORWARD_PERMIT_WAIT_METRIC_NAME,
            FORWARD_CONCURRENT_METRIC_NAME,
            FORWARD_LIMIT_METRIC_NAME,
        ];
        assert!(names
            .iter()
            .all(|name| worker_metric_cardinality_limit(name).is_some()));
        assert!(worker_metric_cardinality_limit("not-a-contract-metric").is_none());
        assert_eq!(WORKER_REQUEST_CARDINALITY_LIMIT, 33 * 7 * 5);
        assert_eq!(WORKER_PHASE_CARDINALITY_LIMIT, 33 * 7 * 5 * 3);
        assert_eq!(WORKER_UNITS_CARDINALITY_LIMIT, 33 * 7 * 3);
        assert_eq!(WORKER_FORWARD_CARDINALITY_LIMIT, 5 * 2 * 3 * 6 * 47);

        let limits: Vec<_> = names
            .iter()
            .map(|name| worker_metric_cardinality_limit(name).expect("owned instrument limit"))
            .collect();
        assert_eq!(limits.iter().sum::<usize>(), 29_577);
        assert_eq!(
            limits.iter().map(|limit| limit + 1).sum::<usize>(),
            29_590,
            "the pinned SDK requests one plus each view limit as initial tracker-map capacity"
        );

        let histogram_bucket_counter_cells = WORKER_REQUEST_CARDINALITY_LIMIT
            * (REQUEST_DURATION_BUCKETS_S.len() + 1)
            + WORKER_PHASE_CARDINALITY_LIMIT * (REQUEST_DURATION_BUCKETS_S.len() + 1)
            + WORKER_MODEL_LOAD_CARDINALITY_LIMIT * (MODEL_LOAD_DURATION_BUCKETS_S.len() + 1)
            + WORKER_FORWARD_CARDINALITY_LIMIT * (REQUEST_DURATION_BUCKETS_S.len() + 1)
            + WORKER_FORWARD_PERMIT_CARDINALITY_LIMIT * (REQUEST_DURATION_BUCKETS_S.len() + 1);
        assert_eq!(histogram_bucket_counter_cells, 379_022);
        assert_eq!(
            histogram_bucket_counter_cells * std::mem::size_of::<u64>(),
            3_032_176
        );
    }

    #[test]
    fn valid_worker_request_domain_never_uses_sdk_overflow_series() {
        let exporter = InMemoryMetricExporter::default();
        let reader = PeriodicReader::builder(exporter.clone()).build();
        let provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_view(worker_metric_cardinality_view)
            .build();
        let metrics = WorkerMetrics::new(
            &provider.meter("sie-worker-cardinality-view-test"),
            "default|gpu-a10|candle".to_string(),
        );
        let models: Vec<_> = (0..=MAX_CATALOG_MODEL_PROFILE_PAIRS)
            .map(|index| format!("catalog/model-{index}"))
            .collect();
        let operations = [
            "encode",
            "score",
            "extract",
            "embeddings",
            "moderations",
            "generate",
            "unknown-operation",
        ];
        let outcomes = [
            WorkerOutcome::Success,
            WorkerOutcome::Error,
            WorkerOutcome::Retry,
            WorkerOutcome::Cancelled,
            WorkerOutcome::Other,
        ];
        for model in &models {
            for operation in operations {
                for outcome in outcomes {
                    metrics.record_item_completed(ItemCompleted {
                        operation,
                        outcome,
                        model,
                        profile: "candle",
                        duration_s: f64::NAN,
                        phases: PhaseDurations::default(),
                        units: AuthoritativeUnits::default(),
                    });
                }
            }
        }

        provider
            .force_flush()
            .expect("flush cardinality test metrics");
        let finished = exporter
            .get_finished_metrics()
            .expect("read cardinality metrics");
        let requests = finished
            .iter()
            .flat_map(|resource| resource.scope_metrics())
            .flat_map(|scope| scope.metrics())
            .find(|metric| metric.name() == REQUESTS_METRIC_NAME)
            .expect("request counter");
        let AggregatedMetrics::U64(MetricData::Sum(sum)) = requests.data() else {
            panic!("requests must be a u64 sum");
        };
        assert_eq!(sum.data_points().count(), WORKER_REQUEST_CARDINALITY_LIMIT);
        assert!(sum.data_points().all(|point| point
            .attributes()
            .all(|attribute| attribute.key.as_str() != "otel.metric.overflow")));
    }

    /// Reproducible hot-path microbenchmark. One release-mode invocation
    /// collects three independently warmed samples and asserts their median:
    /// `SIE_RUN_TELEMETRY_BENCHMARK=1 cargo test --manifest-path packages/sie_server_rust/Cargo.toml --locked --release --lib benchmark_worker_item_paths -- --ignored --nocapture --test-threads=1`.
    #[cfg(not(debug_assertions))]
    #[test]
    #[ignore = "manual telemetry disabled/enabled microbenchmark"]
    fn benchmark_worker_item_paths() {
        const SAMPLES: usize = 3;
        const WARMUP_ITERATIONS: u32 = 10_000;
        const ITERATIONS: u32 = 200_000;
        assert!(
            !metrics_enabled(),
            "benchmark expects the global noop facade"
        );
        let budget_document: serde_json::Value = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../telemetry/performance-budgets.json"
        )))
        .expect("parse checked-in telemetry performance budgets");
        let budget = |name: &str| {
            budget_document["budgets"][name]
                .as_f64()
                .unwrap_or_else(|| panic!("missing numeric telemetry performance budget {name}"))
        };

        // Match the largest production stage set: seven common stages plus
        // all fifteen ModernBERT profile stages. `record_forward_completed`
        // adds the `total` observation, so this benchmarks 23 histogram
        // writes rather than a convenient smaller subset.
        let forward_stages = [
            (ForwardStage::Forward, 0.18),
            (ForwardStage::Pool, 0.01),
            (ForwardStage::Normalize, 0.005),
            (ForwardStage::Conversion, 0.005),
            (ForwardStage::ConversionTensorReadback, 0.002),
            (ForwardStage::ConversionHostPack, 0.003),
            (ForwardStage::Inference, 0.2),
            (ForwardStage::ModernBertEmbedding, 0.01),
            (ForwardStage::ModernBertEmbeddingNorm, 0.005),
            (ForwardStage::ModernBertRopeSelect, 0.002),
            (ForwardStage::ModernBertAttention, 0.08),
            (ForwardStage::ModernBertAttentionNorm, 0.005),
            (ForwardStage::ModernBertAttentionQkv, 0.02),
            (ForwardStage::ModernBertAttentionRotary, 0.005),
            (ForwardStage::ModernBertAttentionFlash, 0.04),
            (ForwardStage::ModernBertAttentionOutputDense, 0.01),
            (ForwardStage::ModernBertMlp, 0.06),
            (ForwardStage::ModernBertMlpNorm, 0.005),
            (ForwardStage::ModernBertMlpWi, 0.02),
            (ForwardStage::ModernBertMlpActivation, 0.005),
            (ForwardStage::ModernBertMlpWo, 0.02),
            (ForwardStage::ModernBertFinalNorm, 0.005),
        ];

        let exporter = InMemoryMetricExporterBuilder::new()
            .with_temporality(Temporality::LowMemory)
            .build();
        let reader = PeriodicReader::builder(exporter).build();
        let provider = SdkMeterProvider::builder()
            .with_reader(reader)
            .with_view(worker_metric_cardinality_view)
            .build();
        let metrics = WorkerMetrics::new(
            &provider.meter("sie-worker-benchmark"),
            "default|gpu-a10|candle".to_string(),
        );

        let exercise_disabled_item = |iterations: u32| {
            for _ in 0..iterations {
                record_item_completed(std::hint::black_box(ItemCompleted {
                    operation: "encode",
                    outcome: WorkerOutcome::Success,
                    model: "BAAI/bge-m3",
                    profile: "candle",
                    duration_s: 0.25,
                    phases: PhaseDurations::default(),
                    units: AuthoritativeUnits::default(),
                }));
            }
        };
        let exercise_disabled_forward = |iterations: u32| {
            for _ in 0..iterations {
                record_forward_completed(std::hint::black_box(ForwardCompleted {
                    model: "BAAI/bge-m3",
                    profile: "candle",
                    outcome: ForwardOutcome::Success,
                    input_source: ForwardInputSource::Prepared,
                    output_path: ForwardOutputPath::Dense,
                    duration_s: 0.25,
                    stages: &forward_stages,
                }));
            }
        };
        let exercise_enabled_item = |iterations: u32| {
            for _ in 0..iterations {
                metrics.record_item_completed(std::hint::black_box(ItemCompleted {
                    operation: "encode",
                    outcome: WorkerOutcome::Success,
                    model: "BAAI/bge-m3",
                    profile: "candle",
                    duration_s: 0.25,
                    phases: PhaseDurations::default(),
                    units: AuthoritativeUnits::default(),
                }));
            }
        };
        let exercise_enabled_phases = |iterations: u32| {
            for _ in 0..iterations {
                metrics.record_item_completed(std::hint::black_box(ItemCompleted {
                    operation: "encode",
                    outcome: WorkerOutcome::Success,
                    model: "BAAI/bge-m3",
                    profile: "candle",
                    duration_s: 0.25,
                    phases: PhaseDurations {
                        tokenization_s: Some(0.01),
                        inference_s: Some(0.2),
                        postprocessing_s: None,
                    },
                    units: AuthoritativeUnits::default(),
                }));
            }
        };
        let exercise_enabled_forward = |iterations: u32| {
            for _ in 0..iterations {
                metrics.record_forward_completed(std::hint::black_box(ForwardCompleted {
                    model: "BAAI/bge-m3",
                    profile: "candle",
                    outcome: ForwardOutcome::Success,
                    input_source: ForwardInputSource::Prepared,
                    output_path: ForwardOutputPath::Dense,
                    duration_s: 0.25,
                    stages: &forward_stages,
                }));
            }
        };

        let mut disabled_samples = [0.0; SAMPLES];
        let mut disabled_forward_samples = [0.0; SAMPLES];
        let mut enabled_samples = [0.0; SAMPLES];
        let mut enabled_phases_samples = [0.0; SAMPLES];
        let mut enabled_forward_samples = [0.0; SAMPLES];
        for sample_index in 0..SAMPLES {
            exercise_disabled_item(WARMUP_ITERATIONS);
            let started = std::time::Instant::now();
            exercise_disabled_item(ITERATIONS);
            disabled_samples[sample_index] =
                started.elapsed().as_nanos() as f64 / f64::from(ITERATIONS);

            exercise_disabled_forward(WARMUP_ITERATIONS);
            let started = std::time::Instant::now();
            exercise_disabled_forward(ITERATIONS);
            disabled_forward_samples[sample_index] =
                started.elapsed().as_nanos() as f64 / f64::from(ITERATIONS);

            exercise_enabled_item(WARMUP_ITERATIONS);
            let started = std::time::Instant::now();
            exercise_enabled_item(ITERATIONS);
            enabled_samples[sample_index] =
                started.elapsed().as_nanos() as f64 / f64::from(ITERATIONS);

            exercise_enabled_phases(WARMUP_ITERATIONS);
            let started = std::time::Instant::now();
            exercise_enabled_phases(ITERATIONS);
            enabled_phases_samples[sample_index] =
                started.elapsed().as_nanos() as f64 / f64::from(ITERATIONS);

            exercise_enabled_forward(WARMUP_ITERATIONS);
            let started = std::time::Instant::now();
            exercise_enabled_forward(ITERATIONS);
            enabled_forward_samples[sample_index] =
                started.elapsed().as_nanos() as f64 / f64::from(ITERATIONS);
        }
        provider.force_flush().expect("flush benchmark metrics");

        let median = |mut samples: [f64; SAMPLES]| {
            samples.sort_by(f64::total_cmp);
            samples[SAMPLES / 2]
        };
        let disabled_median_ns = median(disabled_samples);
        let disabled_forward_median_ns = median(disabled_forward_samples);
        let enabled_median_ns = median(enabled_samples);
        let enabled_phases_median_ns = median(enabled_phases_samples);
        let enabled_forward_median_ns = median(enabled_forward_samples);
        let incremental_median_ns = (enabled_median_ns - disabled_median_ns).max(0.0);
        let incremental_phases_median_ns = (enabled_phases_median_ns - disabled_median_ns).max(0.0);
        let incremental_forward_median_ns =
            (enabled_forward_median_ns - disabled_forward_median_ns).max(0.0);

        eprintln!(
            "worker facade: samples={SAMPLES} iterations_per_sample={ITERATIONS} disabled_item={disabled_samples:?} disabled_item_median={disabled_median_ns:.1} ns/event disabled_forward_23_histograms={disabled_forward_samples:?} disabled_forward_23_histograms_median={disabled_forward_median_ns:.1} ns/event enabled_item={enabled_samples:?} enabled_item_median={enabled_median_ns:.1} ns/event enabled_item_two_phases={enabled_phases_samples:?} enabled_item_two_phases_median={enabled_phases_median_ns:.1} ns/event enabled_forward_23_histograms={enabled_forward_samples:?} enabled_forward_23_histograms_median={enabled_forward_median_ns:.1} ns/event incremental_item_median={incremental_median_ns:.1} ns/event incremental_two_phases_median={incremental_phases_median_ns:.1} ns/event incremental_forward_23_histograms_median={incremental_forward_median_ns:.1} ns/event",
        );
        assert!(
            disabled_median_ns <= budget("rust_worker_disabled_ns_per_item"),
            "rust worker telemetry-disabled item median {disabled_median_ns:.1} ns exceeded its checked-in budget"
        );
        assert!(
            enabled_median_ns <= budget("rust_worker_enabled_ns_per_item"),
            "rust worker item telemetry median {enabled_median_ns:.1} ns exceeded its checked-in budget"
        );
        assert!(
            incremental_median_ns <= budget("rust_worker_incremental_ns_per_item"),
            "rust worker incremental item telemetry median {incremental_median_ns:.1} ns exceeded its checked-in budget"
        );
        assert!(
            enabled_phases_median_ns <= budget("rust_worker_two_phase_enabled_ns_per_item"),
            "rust worker two-phase telemetry median {enabled_phases_median_ns:.1} ns exceeded its checked-in budget"
        );
        assert!(
            incremental_phases_median_ns
                <= budget("rust_worker_two_phase_incremental_ns_per_item"),
            "rust worker incremental two-phase telemetry median {incremental_phases_median_ns:.1} ns exceeded its checked-in budget"
        );
        assert!(
            disabled_forward_median_ns <= budget("rust_worker_forward_disabled_ns_per_event"),
            "rust worker telemetry-disabled forward median {disabled_forward_median_ns:.1} ns exceeded its checked-in budget"
        );
        assert!(
            enabled_forward_median_ns <= budget("rust_worker_23_histogram_forward_ns_per_event"),
            "rust worker 23-histogram telemetry median {enabled_forward_median_ns:.1} ns exceeded its checked-in budget"
        );
        assert!(
            incremental_forward_median_ns
                <= budget("rust_worker_23_histogram_forward_incremental_ns_per_event"),
            "rust worker incremental 23-histogram telemetry median {incremental_forward_median_ns:.1} ns exceeded its checked-in budget"
        );
    }
}
