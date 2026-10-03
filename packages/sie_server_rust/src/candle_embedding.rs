//! Candle-backed dense embedding kernels for native text encoders.

use std::fs;
use std::mem::size_of;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use crate::candle_bert_flash::PackedBertModel;
use crate::candle_gte_rope::{Config as GteRopeConfig, GteRopeForwardProfile, GteRopeModel};
use crate::candle_layers;
use crate::candle_modernbert::{Config as ModernBertConfig, ModernBert, ModernBertForwardProfile};
use crate::candle_splade::{
    pool_packed_splade_activations_dispatch, pool_splade_activations, sparse_embeddings_from_dense,
    CandleBertSpladeHead, CandleSparseEmbedding,
};
use crate::candle_xlm_roberta::{
    Config as XlmRobertaConfig, XlmRobertaForwardProfile, XlmRobertaModel,
};
use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::{
    bert::{BertModel, Config as BertConfig},
    jina_bert::{BertModel as JinaBertModel, Config as JinaBertConfig},
    nomic_bert::{Config as NomicBertConfig, NomicBertModel},
};
use half::f16;
#[cfg(test)]
use hf_hub::Cache;
use hf_hub::{
    api::sync::{Api, ApiBuilder},
    Repo, RepoType,
};
use serde::Deserialize;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct CandleEmbeddingModelConfig {
    pub model_id: String,
    pub hf_id: String,
    pub hf_revision: Option<String>,
    pub max_seq_length: usize,
    pub query_max_length: Option<usize>,
    pub dense_dim: Option<usize>,
    pub sparse_dim: Option<usize>,
    pub multivector_dim: Option<usize>,
    pub compute_precision: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CandleEncodeRequest {
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct CandlePreparedEncodeRequest {
    pub input_ids: Vec<u32>,
    pub attention_mask: Option<Vec<u32>>,
    pub token_type_ids: Option<Vec<u32>>,
}

struct PreparedPackedTensors {
    input_ids: Tensor,
    token_type_ids: Tensor,
    position_ids: Tensor,
    seqlens: Tensor,
    seq_lengths: Vec<usize>,
    max_seqlen: usize,
}

#[derive(Debug, Clone)]
pub struct CandleEncodeResult {
    pub embeddings: Vec<Vec<f32>>,
    pub(crate) sparse_embeddings: Option<Vec<CandleSparseEmbedding>>,
    pub multivectors: Option<Vec<CandleMultivectorEmbedding>>,
    pub multivectors_f16: Option<CandleF16MultivectorBatch>,
    pub dim: u32,
    pub tokenization_ms: f64,
    pub inference_ms: f64,
    pub stages: CandleEncodeStageTimings,
    pub forward_profile: Option<Box<CandleForwardProfile>>,
}

#[derive(Debug, Clone)]
pub struct CandleScoreResult {
    pub scores: Vec<f32>,
    pub query_tokens: usize,
    pub doc_tokens: Vec<usize>,
    pub tokenization_ms: f64,
    pub inference_ms: f64,
    /// MaxSim-only timing is available when synchronized diagnostic timings
    /// are enabled. It is diagnostic metadata, not response postprocessing:
    /// `inference_ms` already covers query/doc forward plus MaxSim.
    pub maxsim_ms: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct CandleMultivectorEmbedding {
    pub values: Vec<f32>,
    pub values_f16: Vec<u8>,
    pub num_tokens: u32,
    pub token_dims: u32,
}

/// A single device-to-host f16 buffer with the layout of each multivector.
///
/// Keeping the offsets alongside the one buffer avoids allocating and copying
/// a separate buffer for every request in a packed ColBERT batch.
#[derive(Debug, Clone)]
pub struct CandleF16MultivectorBatch {
    pub values_f16: Vec<f16>,
    pub items: Vec<CandleF16MultivectorItem>,
}

#[derive(Debug, Clone)]
pub struct CandleF16MultivectorItem {
    pub byte_offset: usize,
    pub byte_len: usize,
    pub num_tokens: u32,
    pub token_dims: u32,
}

impl CandleF16MultivectorBatch {
    /// Legacy per-item representation used only when talking to an older
    /// sidecar that did not opt into the shared-buffer response.
    pub fn into_individual(self) -> Result<Vec<CandleMultivectorEmbedding>> {
        self.items
            .into_iter()
            .map(|item| {
                if !item.byte_offset.is_multiple_of(size_of::<f16>())
                    || !item.byte_len.is_multiple_of(size_of::<f16>())
                {
                    anyhow::bail!("f16 multivector byte range is not f16-aligned");
                }
                let value_offset = item.byte_offset / size_of::<f16>();
                let value_len = item.byte_len / size_of::<f16>();
                let end = item
                    .byte_offset
                    .checked_add(item.byte_len)
                    .context("f16 multivector byte range overflow")?;
                let value_end = value_offset
                    .checked_add(value_len)
                    .context("f16 multivector value range overflow")?;
                if end / size_of::<f16>() != value_end {
                    anyhow::bail!("f16 multivector byte range is not f16-aligned");
                }
                let values_f16: &[f16] = self
                    .values_f16
                    .get(value_offset..value_end)
                    .context("f16 multivector byte range exceeds batch buffer")?;
                Ok(CandleMultivectorEmbedding {
                    values: Vec::new(),
                    values_f16: f16_values_to_le_bytes(values_f16),
                    num_tokens: item.num_tokens,
                    token_dims: item.token_dims,
                })
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct F16MultivectorConversionTimings {
    tensor_readback_ms: f64,
    host_pack_ms: f64,
}

pub struct CandleMultivectorEncodeIntermediate {
    projected: Tensor,
    split: CandleMultivectorSplit,
    output_dtype: String,
    token_dim: usize,
    tokenization_ms: f64,
    inference_start: Instant,
    forward_ms: f64,
    pool_ms: f64,
    normalize_ms: f64,
    sync_timings: bool,
    forward_profile: Option<CandleForwardProfile>,
}

enum CandleMultivectorSplit {
    Masked { attention_mask: Tensor },
    Packed { seq_lengths: Vec<usize> },
}

impl CandleMultivectorEncodeIntermediate {
    pub fn finish(self) -> Result<CandleEncodeResult> {
        let conversion_start = Instant::now();
        let (multivectors, multivectors_f16, conversion_detail) = match self.split {
            CandleMultivectorSplit::Masked { attention_mask } => {
                if self.output_dtype == "float16" {
                    let (batch, detail) =
                        split_multivectors_f16(&self.projected, &attention_mask, self.token_dim)?;
                    (None, Some(batch), detail)
                } else {
                    (
                        Some(split_multivectors(
                            &self.projected,
                            &attention_mask,
                            self.token_dim,
                        )?),
                        None,
                        F16MultivectorConversionTimings::default(),
                    )
                }
            }
            CandleMultivectorSplit::Packed { seq_lengths } => {
                if self.output_dtype == "float16" {
                    let (batch, detail) = split_packed_multivectors_f16(
                        &self.projected,
                        &seq_lengths,
                        self.token_dim,
                    )?;
                    (None, Some(batch), detail)
                } else {
                    (
                        Some(split_packed_multivectors(
                            &self.projected,
                            &seq_lengths,
                            self.token_dim,
                        )?),
                        None,
                        F16MultivectorConversionTimings::default(),
                    )
                }
            }
        };
        if self.sync_timings {
            self.projected.device().synchronize()?;
        }
        let conversion_ms = elapsed_ms(conversion_start);
        let inference_ms = self.inference_start.elapsed().as_secs_f64() * 1000.0;

        Ok(CandleEncodeResult {
            embeddings: Vec::new(),
            sparse_embeddings: None,
            multivectors,
            multivectors_f16,
            dim: self.token_dim as u32,
            tokenization_ms: self.tokenization_ms,
            inference_ms,
            stages: CandleEncodeStageTimings {
                forward_ms: self.forward_ms,
                pool_ms: self.pool_ms,
                normalize_ms: self.normalize_ms,
                conversion_ms,
                conversion_tensor_readback_ms: conversion_detail.tensor_readback_ms,
                conversion_host_pack_ms: conversion_detail.host_pack_ms,
                inference_ms,
            },
            forward_profile: self.forward_profile.map(Box::new),
        })
    }

    /// Split projected token matrices without copying their values to the
    /// host. Only a padded attention mask may be read back to identify valid
    /// rows; projected multivectors remain on their original device.
    fn into_device_multivectors(self) -> Result<Vec<Tensor>> {
        match self.split {
            CandleMultivectorSplit::Packed { seq_lengths } => {
                let mut cursor = 0usize;
                let mut outputs = Vec::with_capacity(seq_lengths.len());
                for len in seq_lengths {
                    outputs.push(self.projected.narrow(0, cursor, len)?);
                    cursor = cursor
                        .checked_add(len)
                        .context("packed multivector device split overflow")?;
                }
                if cursor != self.projected.dim(0)? {
                    anyhow::bail!(
                        "packed multivector device split consumed {cursor} rows but projected has {}",
                        self.projected.dim(0)?
                    );
                }
                Ok(outputs)
            }
            CandleMultivectorSplit::Masked { attention_mask } => {
                let (batch_size, sequence_length, token_dim) = self.projected.dims3()?;
                let masks = attention_mask.to_vec2::<u32>()?;
                if masks.len() != batch_size {
                    anyhow::bail!(
                        "masked multivector batch mismatch: projected={batch_size} masks={}",
                        masks.len()
                    );
                }
                let mut outputs = Vec::with_capacity(batch_size);
                for (batch_index, mask) in masks.into_iter().enumerate() {
                    if mask.len() != sequence_length {
                        anyhow::bail!(
                            "masked multivector sequence mismatch: projected={sequence_length} mask={}",
                            mask.len()
                        );
                    }
                    let rows = self
                        .projected
                        .narrow(0, batch_index, 1)?
                        .reshape((sequence_length, token_dim))?;
                    let indices = mask
                        .into_iter()
                        .enumerate()
                        .filter_map(|(index, keep)| (keep != 0).then_some(index as u32))
                        .collect::<Vec<_>>();
                    if indices.is_empty() {
                        outputs.push(Tensor::zeros(
                            (0, token_dim),
                            self.projected.dtype(),
                            self.projected.device(),
                        )?);
                    } else {
                        let indices = Tensor::new(indices.as_slice(), self.projected.device())?;
                        outputs.push(candle_layers::index_select(&rows, &indices, 0)?);
                    }
                }
                Ok(outputs)
            }
        }
    }
}

pub struct CandleEmbeddingModel {
    model_id: String,
    hf_id: String,
    architecture: CandleEmbeddingArchitecture,
    model: CandleEmbeddingInner,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    max_seq_length: usize,
    query_max_length: Option<usize>,
    dense_dim: usize,
    sparse_dim: Option<usize>,
    splade_head: Option<CandleBertSpladeHead>,
    packed_bert: Option<PackedBertModel>,
    multivector_dim: Option<usize>,
    pylate_dense_chain: Option<PylateDenseChain>,
    pad_id: u32,
    tokenizer_path: PathBuf,
    tokenizer_id: String,
    diagnostics: CandleDiagnosticsConfig,
    diagnostic_batches: AtomicU64,
}

enum CandleEmbeddingInner {
    Bert(BertModel),
    JinaBert(JinaBertModel),
    XlmRoberta(XlmRobertaModel),
    GteRope(GteRopeModel),
    ModernBert(ModernBert),
    NomicBert(NomicBertModel),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandleEmbeddingArchitecture {
    Bert,
    JinaBert,
    XlmRoberta,
    GteRope,
    ModernBert,
    NomicBert,
}

#[derive(Debug, Deserialize)]
struct ModelTypeProbe {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    position_embedding_type: Option<String>,
    #[serde(default)]
    feed_forward_type: Option<String>,
}

struct LoadedCandleEmbedding {
    model: CandleEmbeddingInner,
    hidden_size: usize,
    max_position_embeddings: usize,
}

fn modernbert_rope_theta(config: &serde_json::Value, attention_kind: &str) -> Option<f64> {
    config
        .get("rope_parameters")
        .and_then(|params| params.get(attention_kind))
        .and_then(|params| params.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
}

fn parse_modernbert_config(raw_config: &str) -> Result<ModernBertConfig> {
    let mut config: serde_json::Value = serde_json::from_str(raw_config)?;
    if let (Some(norm_eps), Some(layer_norm_eps)) = (
        config.get("norm_eps").and_then(serde_json::Value::as_f64),
        config
            .get("layer_norm_eps")
            .and_then(serde_json::Value::as_f64),
    ) {
        if norm_eps != layer_norm_eps {
            anyhow::bail!(
                "ModernBERT norm_eps {norm_eps} conflicts with layer_norm_eps {layer_norm_eps}"
            );
        }
    }
    if config.get("norm_eps").is_none() {
        if let Some(layer_norm_eps) = config.get("layer_norm_eps").cloned() {
            config["norm_eps"] = layer_norm_eps;
        }
    }
    if config.get("global_rope_theta").is_none() {
        if let Some(theta) = modernbert_rope_theta(&config, "full_attention") {
            config["global_rope_theta"] = serde_json::Value::from(theta);
        }
    }
    if config.get("local_rope_theta").is_none() {
        if let Some(theta) = modernbert_rope_theta(&config, "sliding_attention") {
            config["local_rope_theta"] = serde_json::Value::from(theta);
        }
    }
    let config: ModernBertConfig =
        serde_json::from_value(config).context("deserialize normalized ModernBERT config")?;
    config.validate()?;
    Ok(config)
}

fn modernbert_var_builder<'a>(vb: VarBuilder<'a>) -> VarBuilder<'a> {
    if vb.contains_tensor("model.embeddings.tok_embeddings.weight")
        || !vb.contains_tensor("embeddings.tok_embeddings.weight")
    {
        return vb;
    }

    vb.rename_f(|name| name.strip_prefix("model.").unwrap_or(name).to_string())
}

#[derive(Debug)]
struct PylateDenseChain {
    weights: Vec<Tensor>,
    output_dim: usize,
}

enum CandleWeightFile {
    Safetensors(PathBuf),
    Pytorch(PathBuf),
}

#[derive(Debug, Clone)]
struct CandleDiagnosticsConfig {
    enabled: bool,
    sync_timings: bool,
    every_n: u64,
}

#[derive(Debug, Deserialize)]
struct SentenceTransformersModule {
    idx: usize,
    path: String,
    #[serde(rename = "type")]
    module_type: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PylateDenseConfig {
    in_features: usize,
    out_features: usize,
    bias: bool,
    use_residual: bool,
    activation_function: String,
}

#[derive(Debug, Clone, Copy)]
struct BatchTokenStats {
    items: usize,
    total_tokens: usize,
    squared_tokens: usize,
    min_tokens: usize,
    max_tokens: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CandleEncodeStageTimings {
    pub forward_ms: f64,
    pub pool_ms: f64,
    pub normalize_ms: f64,
    pub conversion_ms: f64,
    /// CUDA conversion plus the blocking tensor readback to host. It avoids
    /// claiming that either cast or PCIe transfer alone owns this time.
    pub conversion_tensor_readback_ms: f64,
    /// CPU packing from downloaded f16 values into the shared output buffer.
    pub conversion_host_pack_ms: f64,
    pub inference_ms: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum CandleForwardProfile {
    XlmRoberta(XlmRobertaForwardProfile),
    GteRope(GteRopeForwardProfile),
    ModernBert(ModernBertForwardProfile),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PackedPositionIdPolicy {
    ZeroBased,
    PaddingOffset,
}

#[derive(Debug, Clone, Copy)]
struct CandleGemmPrecisionConfig {
    f32: bool,
    f16: bool,
    bf16: bool,
}

impl CandleDiagnosticsConfig {
    fn from_env() -> Self {
        Self {
            enabled: env_bool_any(
                &["SIE_CANDLE_DIAGNOSTICS", "SIE_CANDLE_DEBUG_KERNELS"],
                false,
            ),
            sync_timings: env_bool_any(
                &["SIE_CANDLE_SYNC_DIAGNOSTICS", "SIE_CANDLE_DIAGNOSTICS_SYNC"],
                false,
            ),
            every_n: env_u64("SIE_CANDLE_DIAGNOSTICS_EVERY_N", 1).max(1),
        }
    }
}

impl CandleGemmPrecisionConfig {
    fn applies_to_dtype(self, dtype: DType) -> bool {
        match dtype {
            DType::F32 => self.f32,
            DType::F16 => self.f16,
            DType::BF16 => self.bf16,
            _ => false,
        }
    }

    fn from_env_for_device(device: &Device) -> Self {
        let config = Self {
            f32: env_bool_any(&["SIE_CANDLE_GEMM_REDUCED_PRECISION_F32"], false),
            f16: env_bool_any(&["SIE_CANDLE_GEMM_REDUCED_PRECISION_F16"], true),
            bf16: env_bool_any(
                &[
                    "SIE_CANDLE_GEMM_REDUCED_PRECISION_BF16",
                    "SIE_CANDLE_FAST_BF16_GEMM",
                ],
                true,
            ),
        };

        if device.is_cuda() {
            candle::cuda::set_gemm_reduced_precision_f32(config.f32);
            candle::cuda::set_gemm_reduced_precision_f16(config.f16);
            candle::cuda::set_gemm_reduced_precision_bf16(config.bf16);
        }

        config
    }
}

impl BatchTokenStats {
    fn from_lengths(lengths: impl IntoIterator<Item = usize>) -> Self {
        let mut items = 0usize;
        let mut total_tokens = 0usize;
        let mut squared_tokens = 0usize;
        let mut min_tokens = usize::MAX;
        let mut max_tokens = 0usize;

        for len in lengths {
            items += 1;
            total_tokens += len;
            squared_tokens = squared_tokens.saturating_add(len.saturating_mul(len));
            min_tokens = min_tokens.min(len);
            max_tokens = max_tokens.max(len);
        }

        if items == 0 {
            min_tokens = 0;
        }

        Self {
            items,
            total_tokens,
            squared_tokens,
            min_tokens,
            max_tokens,
        }
    }

    fn avg_tokens(self) -> f64 {
        if self.items == 0 {
            0.0
        } else {
            self.total_tokens as f64 / self.items as f64
        }
    }
}

fn attention_gflops(squared_tokens: usize, hidden_size: usize, layers: usize) -> f64 {
    4.0 * squared_tokens as f64 * hidden_size as f64 * layers as f64 / 1_000_000_000.0
}

fn tflops_per_second(gflops: f64, elapsed_ms: f64) -> f64 {
    if elapsed_ms <= 0.0 {
        0.0
    } else {
        gflops / elapsed_ms
    }
}

impl CandleEmbeddingArchitecture {
    fn from_probe(probe: &ModelTypeProbe) -> Result<Self> {
        let normalized = probe
            .model_type
            .as_deref()
            .unwrap_or("bert")
            .to_ascii_lowercase()
            .replace('-', "_");
        match normalized.as_str() {
            "bert" if probe.looks_like_jina_bert() => Ok(Self::JinaBert),
            "bert" => Ok(Self::Bert),
            "jina_bert" => Ok(Self::JinaBert),
            "xlm_roberta" => Ok(Self::XlmRoberta),
            "new" | "gte" if probe.uses_rope_position_embeddings() => Ok(Self::GteRope),
            "new" | "gte" => anyhow::bail!(
                "CandleEmbeddingAdapter supports model_type={normalized:?} only when position_embedding_type=rope"
            ),
            "modernbert" | "modern_bert" => Ok(Self::ModernBert),
            "nomic_bert" => Ok(Self::NomicBert),
            _ => anyhow::bail!(
                "CandleEmbeddingAdapter does not support model_type={:?}; supported model types are bert, jina-bert, xlm-roberta, new/gte rope, modernbert, and nomic_bert",
                probe.model_type
            ),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bert => "BERT",
            Self::JinaBert => "JinaBERT",
            Self::XlmRoberta => "XLM-RoBERTa",
            Self::GteRope => "GTE-RoPE",
            Self::ModernBert => "ModernBERT",
            Self::NomicBert => "NomicBERT",
        }
    }
}

impl ModelTypeProbe {
    fn looks_like_jina_bert(&self) -> bool {
        self.architectures
            .iter()
            .any(|architecture| architecture.to_ascii_lowercase().contains("jinabert"))
            || matches!(self.position_embedding_type.as_deref(), Some("alibi"))
                && matches!(self.feed_forward_type.as_deref(), Some("geglu"))
    }

    fn uses_rope_position_embeddings(&self) -> bool {
        self.position_embedding_type
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("rope"))
    }
}

impl CandleEmbeddingInner {
    fn load(
        architecture: CandleEmbeddingArchitecture,
        raw_config: &str,
        vb: VarBuilder,
        hf_id: &str,
    ) -> Result<LoadedCandleEmbedding> {
        match architecture {
            CandleEmbeddingArchitecture::Bert => {
                let model_config: BertConfig = serde_json::from_str(raw_config)
                    .with_context(|| format!("parse BERT config for {hf_id}"))?;
                let hidden_size = model_config.hidden_size;
                let max_position_embeddings = model_config.max_position_embeddings;
                let model = BertModel::load(vb, &model_config)
                    .with_context(|| format!("load Candle BERT model for {hf_id}"))?;
                Ok(LoadedCandleEmbedding {
                    model: Self::Bert(model),
                    hidden_size,
                    max_position_embeddings,
                })
            }
            CandleEmbeddingArchitecture::JinaBert => {
                let model_config: JinaBertConfig = serde_json::from_str(raw_config)
                    .with_context(|| format!("parse JinaBERT config for {hf_id}"))?;
                let hidden_size = model_config.hidden_size;
                let max_position_embeddings = model_config.max_position_embeddings;
                let model = JinaBertModel::new(vb, &model_config)
                    .with_context(|| format!("load Candle JinaBERT model for {hf_id}"))?;
                Ok(LoadedCandleEmbedding {
                    model: Self::JinaBert(model),
                    hidden_size,
                    max_position_embeddings,
                })
            }
            CandleEmbeddingArchitecture::XlmRoberta => {
                let model_config: XlmRobertaConfig = serde_json::from_str(raw_config)
                    .with_context(|| format!("parse XLM-RoBERTa config for {hf_id}"))?;
                let hidden_size = model_config.hidden_size;
                let max_position_embeddings = model_config.max_position_embeddings;
                let model = XlmRobertaModel::new(&model_config, vb)
                    .with_context(|| format!("load Candle XLM-RoBERTa model for {hf_id}"))?;
                Ok(LoadedCandleEmbedding {
                    model: Self::XlmRoberta(model),
                    hidden_size,
                    max_position_embeddings,
                })
            }
            CandleEmbeddingArchitecture::GteRope => {
                let model_config: GteRopeConfig = serde_json::from_str(raw_config)
                    .with_context(|| format!("parse GTE-RoPE config for {hf_id}"))?;
                let hidden_size = model_config.hidden_size;
                let max_position_embeddings = model_config.max_position_embeddings;
                let model = GteRopeModel::load(&model_config, vb)
                    .with_context(|| format!("load Candle GTE-RoPE model for {hf_id}"))?;
                Ok(LoadedCandleEmbedding {
                    model: Self::GteRope(model),
                    hidden_size,
                    max_position_embeddings,
                })
            }
            CandleEmbeddingArchitecture::ModernBert => {
                let model_config = parse_modernbert_config(raw_config)
                    .with_context(|| format!("parse ModernBERT config for {hf_id}"))?;
                let hidden_size = model_config.hidden_size;
                let max_position_embeddings = model_config.max_position_embeddings;
                let vb = modernbert_var_builder(vb);
                let model = ModernBert::load(vb, &model_config)
                    .with_context(|| format!("load Candle ModernBERT model for {hf_id}"))?;
                Ok(LoadedCandleEmbedding {
                    model: Self::ModernBert(model),
                    hidden_size,
                    max_position_embeddings,
                })
            }
            CandleEmbeddingArchitecture::NomicBert => {
                let model_config: NomicBertConfig = serde_json::from_str(raw_config)
                    .with_context(|| format!("parse NomicBERT config for {hf_id}"))?;
                let hidden_size = model_config.n_embd;
                let max_position_embeddings = model_config.n_positions;
                let model = NomicBertModel::load(vb, &model_config)
                    .with_context(|| format!("load Candle NomicBERT model for {hf_id}"))?;
                Ok(LoadedCandleEmbedding {
                    model: Self::NomicBert(model),
                    hidden_size,
                    max_position_embeddings,
                })
            }
        }
    }

    fn forward(
        &self,
        token_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
    ) -> candle::Result<Tensor> {
        match self {
            Self::Bert(model) => model.forward(token_ids, token_type_ids, Some(attention_mask)),
            Self::JinaBert(model) => model.forward(token_ids),
            Self::XlmRoberta(model) => model.forward(token_ids, attention_mask, token_type_ids),
            Self::GteRope(_) => {
                candle::bail!("Candle GTE-RoPE model requires packed FlashAttention forward")
            }
            Self::ModernBert(model) => model.forward(token_ids, attention_mask),
            Self::NomicBert(model) => {
                model.forward(token_ids, Some(token_type_ids), Some(attention_mask))
            }
        }
    }

    fn forward_packed(&self, packed: &PreparedPackedTensors) -> Option<candle::Result<Tensor>> {
        match self {
            Self::XlmRoberta(model) => Some(model.forward_packed(
                &packed.input_ids,
                &packed.position_ids,
                &packed.seqlens,
                packed.max_seqlen,
            )),
            Self::GteRope(model) => Some(model.forward_packed(
                &packed.input_ids,
                &packed.token_type_ids,
                &packed.position_ids,
                &packed.seqlens,
                packed.max_seqlen,
            )),
            Self::ModernBert(model) => Some(model.forward_packed(
                &packed.input_ids,
                &packed.position_ids,
                &packed.seqlens,
                packed.max_seqlen,
            )),
            _ => None,
        }
    }
}

impl CandleEmbeddingModel {
    pub fn load(config: &CandleEmbeddingModelConfig) -> Result<Self> {
        let device = candle_device()?;
        let gemm_precision = CandleGemmPrecisionConfig::from_env_for_device(&device);
        let api = hugging_face_api_from_env()?;
        let revision = config
            .hf_revision
            .clone()
            .unwrap_or_else(|| "main".to_string());
        let repo = api.repo(Repo::with_revision(
            config.hf_id.clone(),
            RepoType::Model,
            revision,
        ));
        let config_filename = repo
            .get("config.json")
            .with_context(|| format!("download config.json for {}", config.hf_id))?;
        let tokenizer_filename = repo
            .get("tokenizer.json")
            .with_context(|| format!("download tokenizer.json for {}", config.hf_id))?;
        let weights_file = download_weight_file(&repo, &config.hf_id)?;

        let raw_config = std::fs::read_to_string(&config_filename)
            .with_context(|| format!("read {}", config_filename.display()))?;
        let model_type_probe: ModelTypeProbe = serde_json::from_str(&raw_config)
            .with_context(|| format!("read model_type from config for {}", config.hf_id))?;
        let architecture = CandleEmbeddingArchitecture::from_probe(&model_type_probe)?;

        let dtype = dtype_from_compute_precision(config.compute_precision.as_deref(), &device)?;
        let vb = var_builder_from_weight_file(weights_file, &config.hf_id, &device, dtype)?;
        let (splade_head, packed_bert) = match config.sparse_dim {
            Some(sparse_dim) => {
                if architecture != CandleEmbeddingArchitecture::Bert {
                    anyhow::bail!(
                        "Candle sparse encode for {} requires a BERT masked-language-model checkpoint, got {}",
                        config.model_id,
                        architecture.name()
                    );
                }
                let bert_config: BertConfig = serde_json::from_str(&raw_config)
                    .with_context(|| format!("parse BERT SPLADE config for {}", config.hf_id))?;
                let head = CandleBertSpladeHead::load(vb.clone(), &bert_config)
                    .with_context(|| format!("load Candle SPLADE head for {}", config.hf_id))?;
                if head.vocab_size() != sparse_dim {
                    anyhow::bail!(
                        "Candle sparse dimension mismatch for {}: catalog sparse_dim={} model vocab_size={}",
                        config.model_id,
                        sparse_dim,
                        head.vocab_size()
                    );
                }
                let packed_bert = if device.is_cuda() && dtype == DType::F16 {
                    match PackedBertModel::load(vb.clone(), &bert_config) {
                        Ok(model) => Some(model),
                        Err(error) => {
                            warn!(
                                model_id = %config.model_id,
                                hf_id = %config.hf_id,
                                error = %error,
                                "packed BERT load failed; using padded SPLADE fallback"
                            );
                            None
                        }
                    }
                } else {
                    None
                };
                (Some(head), packed_bert)
            }
            None => (None, None),
        };
        let loaded = CandleEmbeddingInner::load(architecture, &raw_config, vb, &config.hf_id)?;
        if let Some(expected_dim) = config.dense_dim {
            if loaded.hidden_size != expected_dim {
                anyhow::bail!(
                    "Candle embedding dimension mismatch for {}: catalog dense_dim={} model hidden_size={}",
                    config.model_id,
                    expected_dim,
                    loaded.hidden_size
                );
            }
        }
        let pylate_dense_chain = match config.multivector_dim {
            Some(token_dim) => Some(load_pylate_dense_chain(
                &repo,
                &config.hf_id,
                loaded.hidden_size,
                token_dim,
                &device,
                dtype,
            )?),
            None => None,
        };

        let tokenizer_for_hash = Tokenizer::from_file(&tokenizer_filename)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("load tokenizer for {}", config.hf_id))?;
        let tokenizer_id = tokenizer_content_hash(&tokenizer_for_hash)
            .with_context(|| format!("hash tokenizer for {}", config.hf_id))?;
        let tokenizer_path =
            materialize_tokenizer(&config.model_id, &tokenizer_filename, &tokenizer_id)?;

        let mut tokenizer = tokenizer_for_hash;
        let pad_id = tokenizer
            .get_padding()
            .map(|padding| padding.pad_id)
            .unwrap_or(0);
        let max_seq_length = config
            .max_seq_length
            .max(1)
            .min(loaded.max_position_embeddings);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_seq_length,
                ..Default::default()
            }))
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("configure tokenizer truncation for {}", config.hf_id))?;
        if let Some(existing) = tokenizer.get_padding_mut() {
            existing.strategy = PaddingStrategy::BatchLongest;
        } else {
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                ..Default::default()
            }));
        }
        let diagnostics = CandleDiagnosticsConfig::from_env();

        info!(
            model_id = %config.model_id,
            hf_id = %config.hf_id,
            architecture = architecture.name(),
            device = ?device,
            configured_compute_precision = config.compute_precision.as_deref().unwrap_or("auto"),
            compute_dtype = ?dtype,
            dtype = ?dtype,
            hidden_size = loaded.hidden_size,
            sparse_dim = config.sparse_dim,
            packed_bert = packed_bert.is_some(),
            max_seq_length,
            gemm_reduced_precision_f32 = gemm_precision.f32,
            gemm_reduced_precision_f16 = gemm_precision.f16,
            gemm_reduced_precision_bf16 = gemm_precision.bf16,
            gemm_reduced_precision_applies_to_compute_dtype = gemm_precision.applies_to_dtype(dtype),
            tokenizer_id = %tokenizer_id,
            tokenizer_path = %tokenizer_path.display(),
            "loaded Candle embedding model"
        );
        if diagnostics.enabled {
            info!(
                model_id = %config.model_id,
                sync_timings = diagnostics.sync_timings,
                every_n = diagnostics.every_n,
                "Candle embedding diagnostics enabled"
            );
        }

        Ok(Self {
            model_id: config.model_id.clone(),
            hf_id: config.hf_id.clone(),
            architecture,
            model: loaded.model,
            tokenizer,
            device,
            dtype,
            max_seq_length,
            query_max_length: config
                .query_max_length
                .map(|max_length| max_length.max(1).min(max_seq_length)),
            dense_dim: loaded.hidden_size,
            sparse_dim: config.sparse_dim,
            splade_head,
            packed_bert,
            multivector_dim: config.multivector_dim,
            pylate_dense_chain,
            pad_id,
            tokenizer_path,
            tokenizer_id,
            diagnostics,
            diagnostic_batches: AtomicU64::new(0),
        })
    }

    pub fn encode(
        &self,
        requests: &[CandleEncodeRequest],
        pooling: &str,
        normalize: bool,
    ) -> Result<CandleEncodeResult> {
        if requests.is_empty() {
            return Ok(CandleEncodeResult {
                embeddings: Vec::new(),
                sparse_embeddings: None,
                multivectors: None,
                multivectors_f16: None,
                dim: self.dense_dim as u32,
                tokenization_ms: 0.0,
                inference_ms: 0.0,
                stages: CandleEncodeStageTimings::default(),
                forward_profile: None,
            });
        }
        if !matches!(pooling, "mean" | "cls") {
            anyhow::bail!("unsupported Candle pooling strategy {pooling:?}; expected mean or cls");
        }

        let tokenization_start = Instant::now();
        let texts: Vec<&str> = requests
            .iter()
            .map(|request| request.text.as_str())
            .collect();
        if self.should_use_packed_forward() {
            let prepared_requests = texts
                .iter()
                .map(|text| {
                    self.tokenizer
                        .encode(*text, true)
                        .map_err(anyhow::Error::msg)
                        .map(|encoding| CandlePreparedEncodeRequest {
                            input_ids: encoding.get_ids().to_vec(),
                            attention_mask: Some(encoding.get_attention_mask().to_vec()),
                            token_type_ids: Some(encoding.get_type_ids().to_vec()),
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let tokenization_ms = tokenization_start.elapsed().as_secs_f64() * 1000.0;
            let pack_start = Instant::now();
            let packed = self.prepared_packed_tensors(&prepared_requests)?;
            let pack_ms = elapsed_ms(pack_start);
            return self.encode_packed(
                &packed,
                pooling,
                normalize,
                tokenization_ms,
                pack_ms,
                "raw",
            );
        }

        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(anyhow::Error::msg)
            .context("tokenize Candle embedding batch")?;
        let token_stats = BatchTokenStats::from_lengths(encodings.iter().map(|encoding| {
            encoding
                .get_attention_mask()
                .iter()
                .filter(|&&mask| mask != 0)
                .count()
        }));
        let tensorize_start = Instant::now();
        let token_ids = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_ids(), &self.device))
            .collect::<candle::Result<Vec<_>>>()?;
        let attention_mask = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_attention_mask(), &self.device))
            .collect::<candle::Result<Vec<_>>>()?;
        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let tensorize_ms = elapsed_ms(tensorize_start);
        let tokenization_ms = tokenization_start.elapsed().as_secs_f64() * 1000.0;

        self.encode_tensors(
            &token_ids,
            &attention_mask,
            &token_type_ids,
            pooling,
            normalize,
            tokenization_ms,
            token_stats,
            tensorize_ms,
            "raw",
        )
    }

    pub fn encode_prepared(
        &self,
        requests: &[CandlePreparedEncodeRequest],
        pooling: &str,
        normalize: bool,
    ) -> Result<CandleEncodeResult> {
        if requests.is_empty() {
            return Ok(CandleEncodeResult {
                embeddings: Vec::new(),
                sparse_embeddings: None,
                multivectors: None,
                multivectors_f16: None,
                dim: self.dense_dim as u32,
                tokenization_ms: 0.0,
                inference_ms: 0.0,
                stages: CandleEncodeStageTimings::default(),
                forward_profile: None,
            });
        }
        if !matches!(pooling, "mean" | "cls") {
            anyhow::bail!("unsupported Candle pooling strategy {pooling:?}; expected mean or cls");
        }

        if self.should_use_packed_forward() {
            let pack_start = Instant::now();
            let packed = self.prepared_packed_tensors(requests)?;
            let pack_ms = elapsed_ms(pack_start);
            return self.encode_packed(&packed, pooling, normalize, 0.0, pack_ms, "prepared");
        }

        let token_stats = self.prepared_token_stats(requests);
        let tensorize_start = Instant::now();
        let (token_ids, attention_mask, token_type_ids) = self.prepared_tensors(requests)?;
        let tensorize_ms = elapsed_ms(tensorize_start);
        self.encode_tensors(
            &token_ids,
            &attention_mask,
            &token_type_ids,
            pooling,
            normalize,
            0.0,
            token_stats,
            tensorize_ms,
            "prepared",
        )
    }

    pub fn encode_sparse(&self, requests: &[CandleEncodeRequest]) -> Result<CandleEncodeResult> {
        let sparse_dim = self
            .sparse_dim
            .context("Candle sparse encode requested for model without sparse_dim")?;
        if requests.is_empty() {
            return Ok(CandleEncodeResult {
                embeddings: Vec::new(),
                sparse_embeddings: Some(Vec::new()),
                multivectors: None,
                multivectors_f16: None,
                dim: sparse_dim as u32,
                tokenization_ms: 0.0,
                inference_ms: 0.0,
                stages: CandleEncodeStageTimings::default(),
                forward_profile: None,
            });
        }

        let tokenization_start = Instant::now();
        let texts = requests
            .iter()
            .map(|request| request.text.as_str())
            .collect::<Vec<_>>();
        if self.should_use_packed_splade() {
            let prepared_requests = texts
                .iter()
                .map(|text| {
                    self.tokenizer
                        .encode(*text, true)
                        .map_err(anyhow::Error::msg)
                        .map(|encoding| CandlePreparedEncodeRequest {
                            input_ids: encoding.get_ids().to_vec(),
                            attention_mask: Some(encoding.get_attention_mask().to_vec()),
                            token_type_ids: Some(encoding.get_type_ids().to_vec()),
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let tokenization_ms = elapsed_ms(tokenization_start);
            let pack_start = Instant::now();
            let packed = self.prepared_packed_tensors(&prepared_requests)?;
            let pack_ms = elapsed_ms(pack_start);
            return self.encode_packed_sparse(&packed, tokenization_ms, pack_ms, "raw");
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts, true)
            .map_err(anyhow::Error::msg)
            .context("tokenize Candle SPLADE batch")?;
        let token_stats = BatchTokenStats::from_lengths(encodings.iter().map(|encoding| {
            encoding
                .get_attention_mask()
                .iter()
                .filter(|&&mask| mask != 0)
                .count()
        }));
        let tensorize_start = Instant::now();
        let token_ids = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_ids(), &self.device))
            .collect::<candle::Result<Vec<_>>>()?;
        let attention_mask = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_attention_mask(), &self.device))
            .collect::<candle::Result<Vec<_>>>()?;
        let token_type_ids = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_type_ids(), &self.device))
            .collect::<candle::Result<Vec<_>>>()?;
        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = Tensor::stack(&token_type_ids, 0)?;
        let tensorize_ms = elapsed_ms(tensorize_start);
        let tokenization_ms = elapsed_ms(tokenization_start);

        self.encode_sparse_tensors(
            &token_ids,
            &attention_mask,
            &token_type_ids,
            tokenization_ms,
            token_stats,
            tensorize_ms,
            "raw",
        )
    }

    pub fn encode_prepared_sparse(
        &self,
        requests: &[CandlePreparedEncodeRequest],
    ) -> Result<CandleEncodeResult> {
        let sparse_dim = self
            .sparse_dim
            .context("Candle sparse encode requested for model without sparse_dim")?;
        if requests.is_empty() {
            return Ok(CandleEncodeResult {
                embeddings: Vec::new(),
                sparse_embeddings: Some(Vec::new()),
                multivectors: None,
                multivectors_f16: None,
                dim: sparse_dim as u32,
                tokenization_ms: 0.0,
                inference_ms: 0.0,
                stages: CandleEncodeStageTimings::default(),
                forward_profile: None,
            });
        }

        if self.should_use_packed_splade()
            && Self::prepared_attention_masks_are_prefix(requests, self.max_seq_length)
        {
            let pack_start = Instant::now();
            let packed = self.prepared_packed_tensors(requests)?;
            let pack_ms = elapsed_ms(pack_start);
            return self.encode_packed_sparse(&packed, 0.0, pack_ms, "prepared");
        }

        let token_stats = self.prepared_token_stats(requests);
        let tensorize_start = Instant::now();
        let (token_ids, attention_mask, token_type_ids) = self.prepared_tensors(requests)?;
        let tensorize_ms = elapsed_ms(tensorize_start);
        self.encode_sparse_tensors(
            &token_ids,
            &attention_mask,
            &token_type_ids,
            0.0,
            token_stats,
            tensorize_ms,
            "prepared",
        )
    }

    pub fn encode_multivector(
        &self,
        requests: &[CandleEncodeRequest],
        normalize: bool,
        is_query: bool,
        output_dtype: &str,
    ) -> Result<CandleEncodeResult> {
        let token_dim = self
            .multivector_dim
            .context("Candle multivector encode requested for model without multivector_dim")?;
        if requests.is_empty() {
            return Ok(CandleEncodeResult {
                embeddings: Vec::new(),
                sparse_embeddings: None,
                multivectors: Some(Vec::new()),
                multivectors_f16: None,
                dim: token_dim as u32,
                tokenization_ms: 0.0,
                inference_ms: 0.0,
                stages: CandleEncodeStageTimings::default(),
                forward_profile: None,
            });
        }
        self.encode_multivector_intermediate(requests, normalize, is_query, output_dtype)?
            .finish()
    }

    pub fn encode_multivector_intermediate(
        &self,
        requests: &[CandleEncodeRequest],
        normalize: bool,
        is_query: bool,
        output_dtype: &str,
    ) -> Result<CandleMultivectorEncodeIntermediate> {
        if requests.is_empty() {
            anyhow::bail!("Candle multivector requests are empty");
        }

        let tokenization_start = Instant::now();
        let texts: Vec<&str> = requests
            .iter()
            .map(|request| request.text.as_str())
            .collect();
        let max_seq_length = self.multivector_raw_max_seq_length(is_query);
        if self.should_use_packed_forward() {
            let tokenizer = tokenizer_for_max_seq_length_without_padding(
                &self.tokenizer,
                max_seq_length,
                &self.hf_id,
            )?;
            let prepared_requests = tokenizer
                .encode_batch(texts, true)
                .map_err(anyhow::Error::msg)
                .context("tokenize Candle packed multivector batch")?
                .into_iter()
                .map(|encoding| CandlePreparedEncodeRequest {
                    input_ids: encoding.get_ids().to_vec(),
                    attention_mask: Some(encoding.get_attention_mask().to_vec()),
                    token_type_ids: Some(encoding.get_type_ids().to_vec()),
                })
                .collect::<Vec<_>>();
            let tokenization_ms = tokenization_start.elapsed().as_secs_f64() * 1000.0;
            let pack_start = Instant::now();
            let packed =
                self.prepared_packed_tensors_with_max(&prepared_requests, max_seq_length)?;
            let pack_ms = elapsed_ms(pack_start);
            return self.encode_packed_multivector_intermediate(
                &packed,
                normalize,
                output_dtype,
                tokenization_ms,
                pack_ms,
                "raw",
            );
        }
        let tokenizer = tokenizer_for_max_seq_length(&self.tokenizer, max_seq_length, &self.hf_id)?;
        let encodings = tokenizer
            .encode_batch(texts, true)
            .map_err(anyhow::Error::msg)
            .context("tokenize Candle multivector batch")?;
        let token_stats = BatchTokenStats::from_lengths(encodings.iter().map(|encoding| {
            encoding
                .get_attention_mask()
                .iter()
                .filter(|&&mask| mask != 0)
                .count()
        }));
        let tensorize_start = Instant::now();
        let token_ids = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_ids(), &self.device))
            .collect::<candle::Result<Vec<_>>>()?;
        let attention_mask = encodings
            .iter()
            .map(|encoding| Tensor::new(encoding.get_attention_mask(), &self.device))
            .collect::<candle::Result<Vec<_>>>()?;
        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        let tensorize_ms = elapsed_ms(tensorize_start);
        let tokenization_ms = tokenization_start.elapsed().as_secs_f64() * 1000.0;

        self.encode_multivector_tensors_intermediate(
            &token_ids,
            &attention_mask,
            &token_type_ids,
            normalize,
            output_dtype,
            tokenization_ms,
            token_stats,
            tensorize_ms,
            "raw",
        )
    }

    pub fn encode_prepared_multivector(
        &self,
        requests: &[CandlePreparedEncodeRequest],
        normalize: bool,
        is_query: bool,
        output_dtype: &str,
    ) -> Result<CandleEncodeResult> {
        let token_dim = self
            .multivector_dim
            .context("Candle multivector encode requested for model without multivector_dim")?;
        if requests.is_empty() {
            return Ok(CandleEncodeResult {
                embeddings: Vec::new(),
                sparse_embeddings: None,
                multivectors: Some(Vec::new()),
                multivectors_f16: None,
                dim: token_dim as u32,
                tokenization_ms: 0.0,
                inference_ms: 0.0,
                stages: CandleEncodeStageTimings::default(),
                forward_profile: None,
            });
        }
        self.encode_prepared_multivector_intermediate(requests, normalize, is_query, output_dtype)?
            .finish()
    }

    pub fn encode_prepared_multivector_intermediate(
        &self,
        requests: &[CandlePreparedEncodeRequest],
        normalize: bool,
        is_query: bool,
        output_dtype: &str,
    ) -> Result<CandleMultivectorEncodeIntermediate> {
        if requests.is_empty() {
            anyhow::bail!("prepared Candle multivector tokens are empty");
        }

        let max_seq_length = self.multivector_raw_max_seq_length(is_query);
        if self.should_use_packed_forward() {
            let pack_start = Instant::now();
            let packed = self.prepared_packed_tensors_with_max(requests, max_seq_length)?;
            let pack_ms = elapsed_ms(pack_start);
            return self.encode_packed_multivector_intermediate(
                &packed,
                normalize,
                output_dtype,
                0.0,
                pack_ms,
                "prepared",
            );
        }
        let token_stats = self.prepared_token_stats_with_max(requests, max_seq_length);
        let tensorize_start = Instant::now();
        let (token_ids, attention_mask, token_type_ids) =
            self.prepared_tensors_with_max(requests, max_seq_length)?;
        let tensorize_ms = elapsed_ms(tensorize_start);
        self.encode_multivector_tensors_intermediate(
            &token_ids,
            &attention_mask,
            &token_type_ids,
            normalize,
            output_dtype,
            0.0,
            token_stats,
            tensorize_ms,
            "prepared",
        )
    }

    /// Score one query against documents with late-interaction MaxSim.
    ///
    /// Query and document text are tokenized with their independent caps. The
    /// projected, normalized multivectors never leave the model device; only
    /// the final f32 score vector is copied to the host.
    pub fn score_multivector(
        &self,
        query: &CandleEncodeRequest,
        docs: &[CandleEncodeRequest],
        normalize: bool,
        doc_work_budget: usize,
    ) -> Result<CandleScoreResult> {
        if docs.is_empty() {
            return Ok(CandleScoreResult {
                scores: Vec::new(),
                query_tokens: 0,
                doc_tokens: Vec::new(),
                tokenization_ms: 0.0,
                inference_ms: 0.0,
                maxsim_ms: None,
            });
        }

        let tokenization_start = Instant::now();
        let mut query = self.tokenize_multivector_requests(std::slice::from_ref(query), true)?;
        let docs = self.tokenize_multivector_requests(docs, false)?;
        let tokenization_ms = elapsed_ms(tokenization_start);
        let query = query
            .pop()
            .context("Candle score query tokenization returned no query")?;
        self.score_prepared_multivector_impl(
            &query,
            &docs,
            normalize,
            doc_work_budget,
            tokenization_ms,
        )
    }

    /// Prepared-token form of [`Self::score_multivector`]. The rows must be
    /// supplied separately so query and document truncation caps remain
    /// distinct.
    pub fn score_prepared_multivector(
        &self,
        query: &CandlePreparedEncodeRequest,
        docs: &[CandlePreparedEncodeRequest],
        normalize: bool,
        doc_work_budget: usize,
    ) -> Result<CandleScoreResult> {
        self.score_prepared_multivector_impl(query, docs, normalize, doc_work_budget, 0.0)
    }

    fn score_prepared_multivector_impl(
        &self,
        query: &CandlePreparedEncodeRequest,
        docs: &[CandlePreparedEncodeRequest],
        normalize: bool,
        doc_work_budget: usize,
        tokenization_ms: f64,
    ) -> Result<CandleScoreResult> {
        self.multivector_dim
            .context("Candle MaxSim score requested for model without multivector_dim")?;
        if docs.is_empty() {
            return Ok(CandleScoreResult {
                scores: Vec::new(),
                query_tokens: 0,
                doc_tokens: Vec::new(),
                tokenization_ms,
                inference_ms: 0.0,
                maxsim_ms: None,
            });
        }

        let query_tokens =
            Self::prepared_effective_len(self.multivector_raw_max_seq_length(true), query);
        if query_tokens == 0 {
            anyhow::bail!("Candle MaxSim score query tokens are empty");
        }
        let doc_tokens = docs
            .iter()
            .map(|doc| {
                Self::prepared_effective_len(self.multivector_raw_max_seq_length(false), doc)
            })
            .collect::<Vec<_>>();
        if doc_tokens.contains(&0) {
            anyhow::bail!("Candle MaxSim score document tokens are empty");
        }

        self.synchronize_for_timing()?;
        let inference_start = Instant::now();
        let mut query_vectors = self
            .encode_prepared_multivector_intermediate(
                std::slice::from_ref(query),
                normalize,
                true,
                "float32",
            )?
            .into_device_multivectors()?;
        if query_vectors.len() != 1 {
            anyhow::bail!(
                "Candle MaxSim query encode returned {} vectors",
                query_vectors.len()
            );
        }
        let query_vectors = query_vectors
            .pop()
            .context("Candle MaxSim query encode returned no vector")?;

        let mut scores = Vec::with_capacity(docs.len());
        let mut maxsim_ms = self.diagnostics.sync_timings.then_some(0.0);
        for range in score_document_chunks(&doc_tokens, doc_work_budget.max(1)) {
            let doc_vectors = self
                .encode_prepared_multivector_intermediate(
                    &docs[range.clone()],
                    normalize,
                    false,
                    "float32",
                )?
                .into_device_multivectors()?;
            // Synchronize the document forward only for explicitly requested
            // diagnostic timings. The regular hot path keeps its existing
            // asynchronous execution and exposes only the inclusive inference
            // duration below.
            let score_start = if maxsim_ms.is_some() {
                self.synchronize_for_timing()?;
                Some(Instant::now())
            } else {
                None
            };
            scores.extend(maxsim_scores_device(&query_vectors, &doc_vectors)?);
            if let (Some(total), Some(start)) = (&mut maxsim_ms, score_start) {
                *total += elapsed_ms(start);
            }
        }
        self.synchronize_for_timing()?;
        let inference_ms = elapsed_ms(inference_start);
        if scores.len() != docs.len() {
            anyhow::bail!(
                "Candle MaxSim returned {} scores for {} documents",
                scores.len(),
                docs.len()
            );
        }

        Ok(CandleScoreResult {
            scores,
            query_tokens,
            doc_tokens,
            tokenization_ms,
            inference_ms,
            maxsim_ms,
        })
    }

    fn tokenize_multivector_requests(
        &self,
        requests: &[CandleEncodeRequest],
        is_query: bool,
    ) -> Result<Vec<CandlePreparedEncodeRequest>> {
        let tokenizer = tokenizer_for_max_seq_length_without_padding(
            &self.tokenizer,
            self.multivector_raw_max_seq_length(is_query),
            &self.hf_id,
        )?;
        let texts = requests
            .iter()
            .map(|request| request.text.as_str())
            .collect::<Vec<_>>();
        tokenizer
            .encode_batch(texts, true)
            .map_err(anyhow::Error::msg)
            .context("tokenize Candle MaxSim inputs")
            .map(|encodings| {
                encodings
                    .into_iter()
                    .map(|encoding| CandlePreparedEncodeRequest {
                        input_ids: encoding.get_ids().to_vec(),
                        attention_mask: Some(encoding.get_attention_mask().to_vec()),
                        token_type_ids: Some(encoding.get_type_ids().to_vec()),
                    })
                    .collect()
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_packed_sparse(
        &self,
        packed: &PreparedPackedTensors,
        tokenization_ms: f64,
        pack_ms: f64,
        source: &'static str,
    ) -> Result<CandleEncodeResult> {
        let sparse_dim = self
            .sparse_dim
            .context("Candle sparse encode requested for model without sparse_dim")?;
        let splade_head = self
            .splade_head
            .as_ref()
            .context("Candle sparse encode requested for model without a SPLADE head")?;
        let packed_bert = self
            .packed_bert
            .as_ref()
            .context("packed SPLADE encode requested without a packed BERT backbone")?;
        let token_stats = BatchTokenStats::from_lengths(packed.seq_lengths.iter().copied());

        self.synchronize_for_timing()?;
        let inference_start = Instant::now();
        let forward_start = Instant::now();
        let sequence_output = packed_bert
            .forward_packed(
                &packed.input_ids,
                &packed.token_type_ids,
                &packed.position_ids,
                &packed.seqlens,
                packed.max_seqlen,
            )
            .context("run packed Candle SPLADE BERT forward pass")?;
        self.synchronize_for_timing()?;
        let forward_ms = elapsed_ms(forward_start);

        let pool_start = Instant::now();
        let activated = splade_head
            .forward_activated(&sequence_output)
            .context("run packed Candle SPLADE MLM head")?;
        let max_weights = pool_packed_splade_activations_dispatch(
            &activated,
            &packed.seqlens,
            &packed.seq_lengths,
        )?;
        self.synchronize_for_timing()?;
        let pool_ms = elapsed_ms(pool_start);

        let conversion_start = Instant::now();
        let sparse_embeddings = sparse_embeddings_from_dense(&max_weights)?;
        self.synchronize_for_timing()?;
        let conversion_ms = elapsed_ms(conversion_start);
        let inference_ms = elapsed_ms(inference_start);
        let stages = CandleEncodeStageTimings {
            forward_ms,
            pool_ms,
            normalize_ms: 0.0,
            conversion_ms,
            inference_ms,
            ..Default::default()
        };
        self.log_encode_diagnostics(
            "packed_bert_splade",
            source,
            token_stats,
            "splade",
            false,
            tokenization_ms,
            pack_ms,
            stages,
            None,
        );

        Ok(CandleEncodeResult {
            embeddings: Vec::new(),
            sparse_embeddings: Some(sparse_embeddings),
            multivectors: None,
            multivectors_f16: None,
            dim: sparse_dim as u32,
            tokenization_ms,
            inference_ms,
            stages,
            forward_profile: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_sparse_tensors(
        &self,
        token_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
        tokenization_ms: f64,
        token_stats: BatchTokenStats,
        prepare_ms: f64,
        source: &'static str,
    ) -> Result<CandleEncodeResult> {
        let sparse_dim = self
            .sparse_dim
            .context("Candle sparse encode requested for model without sparse_dim")?;
        let splade_head = self
            .splade_head
            .as_ref()
            .context("Candle sparse encode requested for model without a SPLADE head")?;

        self.synchronize_for_timing()?;
        let inference_start = Instant::now();
        let forward_start = Instant::now();
        let sequence_output = self
            .model
            .forward(token_ids, attention_mask, token_type_ids)
            .context("run Candle SPLADE BERT forward pass")?;
        self.synchronize_for_timing()?;
        let forward_ms = elapsed_ms(forward_start);

        let pool_start = Instant::now();
        let activated = splade_head
            .forward_activated(&sequence_output)
            .context("run Candle SPLADE MLM head")?;
        let max_weights = pool_splade_activations(&activated, attention_mask)?;
        self.synchronize_for_timing()?;
        let pool_ms = elapsed_ms(pool_start);

        let conversion_start = Instant::now();
        let sparse_embeddings = sparse_embeddings_from_dense(&max_weights)?;
        self.synchronize_for_timing()?;
        let conversion_ms = elapsed_ms(conversion_start);
        let inference_ms = elapsed_ms(inference_start);
        let stages = CandleEncodeStageTimings {
            forward_ms,
            pool_ms,
            normalize_ms: 0.0,
            conversion_ms,
            inference_ms,
            ..Default::default()
        };
        self.log_encode_diagnostics(
            "padded",
            source,
            token_stats,
            "splade",
            false,
            tokenization_ms,
            prepare_ms,
            stages,
            None,
        );

        Ok(CandleEncodeResult {
            embeddings: Vec::new(),
            sparse_embeddings: Some(sparse_embeddings),
            multivectors: None,
            multivectors_f16: None,
            dim: sparse_dim as u32,
            tokenization_ms,
            inference_ms,
            stages,
            forward_profile: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_tensors(
        &self,
        token_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
        pooling: &str,
        normalize: bool,
        tokenization_ms: f64,
        token_stats: BatchTokenStats,
        prepare_ms: f64,
        source: &'static str,
    ) -> Result<CandleEncodeResult> {
        self.synchronize_for_timing()?;
        let inference_start = Instant::now();
        let forward_start = Instant::now();
        let sequence_output = self
            .model
            .forward(token_ids, attention_mask, token_type_ids)
            .context("run Candle embedding forward pass")?;
        self.synchronize_for_timing()?;
        let forward_ms = elapsed_ms(forward_start);

        let pool_start = Instant::now();
        let pooled = pool_embeddings(&sequence_output, attention_mask, pooling)?;
        self.synchronize_for_timing()?;
        let pool_ms = elapsed_ms(pool_start);

        let normalize_start = Instant::now();
        let pooled = if normalize {
            normalize_l2(&pooled)?
        } else {
            pooled
        };
        self.synchronize_for_timing()?;
        let normalize_ms = elapsed_ms(normalize_start);

        let conversion_start = Instant::now();
        let embeddings = pooled.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        self.synchronize_for_timing()?;
        let conversion_ms = elapsed_ms(conversion_start);
        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;
        self.log_encode_diagnostics(
            "padded",
            source,
            token_stats,
            pooling,
            normalize,
            tokenization_ms,
            prepare_ms,
            CandleEncodeStageTimings {
                forward_ms,
                pool_ms,
                normalize_ms,
                conversion_ms,
                inference_ms,
                ..Default::default()
            },
            None,
        );

        Ok(CandleEncodeResult {
            embeddings,
            sparse_embeddings: None,
            multivectors: None,
            multivectors_f16: None,
            dim: self.dense_dim as u32,
            tokenization_ms,
            inference_ms,
            stages: CandleEncodeStageTimings {
                forward_ms,
                pool_ms,
                normalize_ms,
                conversion_ms,
                inference_ms,
                ..Default::default()
            },
            forward_profile: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_multivector_tensors_intermediate(
        &self,
        token_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
        normalize: bool,
        output_dtype: &str,
        tokenization_ms: f64,
        _token_stats: BatchTokenStats,
        _prepare_ms: f64,
        _source: &'static str,
    ) -> Result<CandleMultivectorEncodeIntermediate> {
        let token_dim = self
            .multivector_dim
            .context("Candle multivector encode requested for model without multivector_dim")?;
        self.synchronize_for_timing()?;
        let inference_start = Instant::now();
        let forward_start = Instant::now();
        let sequence_output = self
            .model
            .forward(token_ids, attention_mask, token_type_ids)
            .context("run Candle multivector forward pass")?;
        self.synchronize_for_timing()?;
        let forward_ms = elapsed_ms(forward_start);

        let project_start = Instant::now();
        let projected = self.project_multivector_tokens(&sequence_output)?;
        self.synchronize_for_timing()?;
        let pool_ms = elapsed_ms(project_start);

        let normalize_start = Instant::now();
        let projected = if normalize {
            normalize_l2_last_dim(&projected)?
        } else {
            projected
        };
        self.synchronize_for_timing()?;
        let normalize_ms = elapsed_ms(normalize_start);

        Ok(CandleMultivectorEncodeIntermediate {
            projected,
            split: CandleMultivectorSplit::Masked {
                attention_mask: attention_mask.clone(),
            },
            output_dtype: output_dtype.to_string(),
            token_dim,
            tokenization_ms,
            inference_start,
            forward_ms,
            pool_ms,
            normalize_ms,
            sync_timings: self.diagnostics.sync_timings,
            forward_profile: None,
        })
    }

    fn encode_packed_multivector_intermediate(
        &self,
        packed: &PreparedPackedTensors,
        normalize: bool,
        output_dtype: &str,
        tokenization_ms: f64,
        _pack_ms: f64,
        _source: &'static str,
    ) -> Result<CandleMultivectorEncodeIntermediate> {
        let token_dim = self
            .multivector_dim
            .context("Candle multivector encode requested for model without multivector_dim")?;
        self.synchronize_for_timing()?;
        let inference_start = Instant::now();
        let forward_start = Instant::now();
        let (sequence_output, forward_profile) = if self.diagnostics.enabled {
            match &self.model {
                CandleEmbeddingInner::ModernBert(model) => {
                    let (output, profile) = model.forward_packed_profiled(
                        &packed.input_ids,
                        &packed.position_ids,
                        &packed.seqlens,
                        packed.max_seqlen,
                        self.diagnostics.sync_timings,
                    )?;
                    (output, Some(CandleForwardProfile::ModernBert(profile)))
                }
                CandleEmbeddingInner::XlmRoberta(model) => {
                    let (output, profile) = model.forward_packed_profiled(
                        &packed.input_ids,
                        &packed.position_ids,
                        &packed.seqlens,
                        packed.max_seqlen,
                        self.diagnostics.sync_timings,
                    )?;
                    (output, Some(CandleForwardProfile::XlmRoberta(profile)))
                }
                CandleEmbeddingInner::GteRope(model) => {
                    let (output, profile) = model.forward_packed_profiled(
                        &packed.input_ids,
                        &packed.token_type_ids,
                        &packed.position_ids,
                        &packed.seqlens,
                        packed.max_seqlen,
                        self.diagnostics.sync_timings,
                    )?;
                    (output, Some(CandleForwardProfile::GteRope(profile)))
                }
                _ => (
                    self.model
                        .forward_packed(packed)
                        .transpose()
                        .context("run packed Candle multivector forward pass")?
                        .context(
                            "packed Candle forward selected for model without packed support",
                        )?,
                    None,
                ),
            }
        } else {
            (
                self.model
                    .forward_packed(packed)
                    .transpose()
                    .context("run packed Candle multivector forward pass")?
                    .context("packed Candle forward selected for model without packed support")?,
                None,
            )
        };
        self.synchronize_for_timing()?;
        let forward_ms = elapsed_ms(forward_start);

        let project_start = Instant::now();
        let projected = self.project_multivector_tokens_packed(&sequence_output)?;
        self.synchronize_for_timing()?;
        let pool_ms = elapsed_ms(project_start);

        let normalize_start = Instant::now();
        let projected = if normalize {
            normalize_l2(&projected)?
        } else {
            projected
        };
        self.synchronize_for_timing()?;
        let normalize_ms = elapsed_ms(normalize_start);

        Ok(CandleMultivectorEncodeIntermediate {
            projected,
            split: CandleMultivectorSplit::Packed {
                seq_lengths: packed.seq_lengths.clone(),
            },
            output_dtype: output_dtype.to_string(),
            token_dim,
            tokenization_ms,
            inference_start,
            forward_ms,
            pool_ms,
            normalize_ms,
            sync_timings: self.diagnostics.sync_timings,
            forward_profile,
        })
    }

    fn encode_packed(
        &self,
        packed: &PreparedPackedTensors,
        pooling: &str,
        normalize: bool,
        tokenization_ms: f64,
        pack_ms: f64,
        source: &'static str,
    ) -> Result<CandleEncodeResult> {
        let token_stats = BatchTokenStats::from_lengths(packed.seq_lengths.iter().copied());
        self.synchronize_for_timing()?;
        let inference_start = Instant::now();
        let forward_start = Instant::now();
        let (sequence_output, forward_profile, output_is_pooled) = if self.diagnostics.enabled {
            match &self.model {
                CandleEmbeddingInner::XlmRoberta(model) => {
                    let (output, profile) = if pooling == "cls" {
                        model.forward_packed_cls_profiled(
                            &packed.input_ids,
                            &packed.position_ids,
                            &packed.seqlens,
                            packed.max_seqlen,
                            self.diagnostics.sync_timings,
                        )?
                    } else {
                        model.forward_packed_profiled(
                            &packed.input_ids,
                            &packed.position_ids,
                            &packed.seqlens,
                            packed.max_seqlen,
                            self.diagnostics.sync_timings,
                        )?
                    };
                    (
                        output,
                        Some(CandleForwardProfile::XlmRoberta(profile)),
                        pooling == "cls",
                    )
                }
                CandleEmbeddingInner::GteRope(model) => {
                    let (output, profile) = model.forward_packed_profiled(
                        &packed.input_ids,
                        &packed.token_type_ids,
                        &packed.position_ids,
                        &packed.seqlens,
                        packed.max_seqlen,
                        self.diagnostics.sync_timings,
                    )?;
                    (output, Some(CandleForwardProfile::GteRope(profile)), false)
                }
                CandleEmbeddingInner::ModernBert(model) => {
                    let (output, profile) = model.forward_packed_profiled(
                        &packed.input_ids,
                        &packed.position_ids,
                        &packed.seqlens,
                        packed.max_seqlen,
                        self.diagnostics.sync_timings,
                    )?;
                    (
                        output,
                        Some(CandleForwardProfile::ModernBert(profile)),
                        false,
                    )
                }
                _ => (
                    self.model
                        .forward_packed(packed)
                        .transpose()
                        .context("run packed Candle embedding forward pass")?
                        .context(
                            "packed Candle forward selected for model without packed support",
                        )?,
                    None,
                    false,
                ),
            }
        } else {
            match &self.model {
                CandleEmbeddingInner::XlmRoberta(model) if pooling == "cls" => (
                    model.forward_packed_cls(
                        &packed.input_ids,
                        &packed.position_ids,
                        &packed.seqlens,
                        packed.max_seqlen,
                    )?,
                    None,
                    true,
                ),
                _ => (
                    self.model
                        .forward_packed(packed)
                        .transpose()
                        .context("run packed Candle embedding forward pass")?
                        .context(
                            "packed Candle forward selected for model without packed support",
                        )?,
                    None,
                    false,
                ),
            }
        };
        self.synchronize_for_timing()?;
        let forward_ms = elapsed_ms(forward_start);

        let pool_start = Instant::now();
        let pooled = if output_is_pooled {
            sequence_output
        } else {
            pool_packed_embeddings(
                &sequence_output,
                &packed.seqlens,
                &packed.seq_lengths,
                pooling,
            )?
        };
        self.synchronize_for_timing()?;
        let pool_ms = elapsed_ms(pool_start);

        let normalize_start = Instant::now();
        let pooled = if normalize {
            normalize_l2(&pooled)?
        } else {
            pooled
        };
        self.synchronize_for_timing()?;
        let normalize_ms = elapsed_ms(normalize_start);

        let conversion_start = Instant::now();
        let embeddings = pooled.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        self.synchronize_for_timing()?;
        let conversion_ms = elapsed_ms(conversion_start);
        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;
        self.log_encode_diagnostics(
            self.packed_path_name(),
            source,
            token_stats,
            pooling,
            normalize,
            tokenization_ms,
            pack_ms,
            CandleEncodeStageTimings {
                forward_ms,
                pool_ms,
                normalize_ms,
                conversion_ms,
                inference_ms,
                ..Default::default()
            },
            forward_profile,
        );

        Ok(CandleEncodeResult {
            embeddings,
            sparse_embeddings: None,
            multivectors: None,
            multivectors_f16: None,
            dim: self.dense_dim as u32,
            tokenization_ms,
            inference_ms,
            stages: CandleEncodeStageTimings {
                forward_ms,
                pool_ms,
                normalize_ms,
                conversion_ms,
                inference_ms,
                ..Default::default()
            },
            forward_profile: forward_profile.map(Box::new),
        })
    }

    fn should_use_packed_forward(&self) -> bool {
        if !self.device.is_cuda() {
            return false;
        }
        match &self.model {
            CandleEmbeddingInner::XlmRoberta(_) => matches!(self.dtype, DType::F16 | DType::BF16),
            CandleEmbeddingInner::GteRope(_) => matches!(self.dtype, DType::F16),
            CandleEmbeddingInner::ModernBert(_) => matches!(self.dtype, DType::F16 | DType::BF16),
            _ => false,
        }
    }

    fn should_use_packed_splade(&self) -> bool {
        self.packed_bert.is_some() && self.device.is_cuda() && self.dtype == DType::F16
    }

    fn packed_path_name(&self) -> &'static str {
        match &self.model {
            CandleEmbeddingInner::XlmRoberta(_) => "packed_xlm_roberta",
            CandleEmbeddingInner::GteRope(_) => "packed_gte_rope",
            CandleEmbeddingInner::ModernBert(_) => "packed_modernbert",
            _ => "packed",
        }
    }

    fn packed_position_id(&self, idx: usize) -> u32 {
        packed_position_id(self.packed_position_id_policy(), self.pad_id, idx)
    }

    fn packed_position_id_policy(&self) -> PackedPositionIdPolicy {
        match &self.model {
            CandleEmbeddingInner::XlmRoberta(_) => PackedPositionIdPolicy::PaddingOffset,
            _ => PackedPositionIdPolicy::ZeroBased,
        }
    }

    fn packed_token_type_ids_ignored(&self) -> bool {
        match &self.model {
            CandleEmbeddingInner::GteRope(model) => model.ignores_token_type_ids(),
            _ => false,
        }
    }

    fn prepared_token_stats(&self, requests: &[CandlePreparedEncodeRequest]) -> BatchTokenStats {
        self.prepared_token_stats_with_max(requests, self.max_seq_length)
    }

    fn prepared_token_stats_with_max(
        &self,
        requests: &[CandlePreparedEncodeRequest],
        max_seq_length: usize,
    ) -> BatchTokenStats {
        BatchTokenStats::from_lengths(
            requests
                .iter()
                .map(|request| Self::prepared_effective_len(max_seq_length, request)),
        )
    }

    fn synchronize_for_timing(&self) -> Result<()> {
        if self.diagnostics.sync_timings {
            self.device.synchronize()?;
        }
        Ok(())
    }

    fn should_log_diagnostics(&self) -> bool {
        if !self.diagnostics.enabled {
            return false;
        }
        let batch = self.diagnostic_batches.fetch_add(1, Ordering::Relaxed) + 1;
        batch.is_multiple_of(self.diagnostics.every_n)
    }

    #[allow(clippy::too_many_arguments)]
    fn log_encode_diagnostics(
        &self,
        path: &'static str,
        source: &'static str,
        token_stats: BatchTokenStats,
        pooling: &str,
        normalize: bool,
        tokenization_ms: f64,
        prepare_ms: f64,
        timings: CandleEncodeStageTimings,
        forward_profile: Option<CandleForwardProfile>,
    ) {
        if !self.should_log_diagnostics() {
            return;
        }

        let cuda_compute_cap =
            std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "unset".to_string());
        let (xlm_profile, gte_profile, modernbert_profile) = match forward_profile {
            Some(CandleForwardProfile::XlmRoberta(profile)) => (
                profile,
                GteRopeForwardProfile::default(),
                ModernBertForwardProfile::default(),
            ),
            Some(CandleForwardProfile::GteRope(profile)) => (
                XlmRobertaForwardProfile::default(),
                profile,
                ModernBertForwardProfile::default(),
            ),
            Some(CandleForwardProfile::ModernBert(profile)) => (
                XlmRobertaForwardProfile::default(),
                GteRopeForwardProfile::default(),
                profile,
            ),
            None => (
                XlmRobertaForwardProfile::default(),
                GteRopeForwardProfile::default(),
                ModernBertForwardProfile::default(),
            ),
        };
        let (
            xlm_roberta_linear_backend,
            xlm_roberta_layernorm_backend,
            xlm_roberta_qkv_fused,
            xlm_roberta_ffn_activation_fused,
        ) = match &self.model {
            CandleEmbeddingInner::XlmRoberta(model) => model.kernel_backend(),
            _ => ("candle_transformers", "candle_transformers", false, false),
        };
        let (
            gte_rope_linear_backend,
            gte_rope_layernorm_backend,
            gte_rope_qkv_fused,
            gte_rope_rotary_inplace,
            gte_rope_ffn_activation_fused,
        ) = match &self.model {
            CandleEmbeddingInner::GteRope(model) => model.kernel_backend(),
            _ => (
                "candle_transformers",
                "candle_transformers",
                false,
                false,
                false,
            ),
        };
        let (modernbert_linear_backend, modernbert_layernorm_backend) = match &self.model {
            CandleEmbeddingInner::ModernBert(model) => model.kernel_backend(),
            _ => ("candle_transformers", "candle_transformers"),
        };
        let gemm_precision = candle_gemm_reduced_precision_state(&self.device);
        let xlm_roberta_attention_gflops = attention_gflops(
            token_stats.squared_tokens,
            xlm_profile.hidden_size,
            xlm_profile.layers,
        );
        let xlm_roberta_total_gflops = xlm_profile.linear_gflops + xlm_roberta_attention_gflops;
        let gte_rope_attention_gflops = attention_gflops(
            token_stats.squared_tokens,
            gte_profile.hidden_size,
            gte_profile.layers,
        );
        let gte_rope_total_gflops = gte_profile.linear_gflops + gte_rope_attention_gflops;
        let modernbert_attention_gflops = attention_gflops(
            token_stats.squared_tokens,
            modernbert_profile.hidden_size,
            modernbert_profile.layers,
        );
        let modernbert_total_gflops =
            modernbert_profile.linear_gflops + modernbert_attention_gflops;
        info!(
            model_id = %self.model_id,
            hf_id = %self.hf_id,
            architecture = self.architecture.name(),
            device = ?self.device,
            compute_dtype = ?self.dtype,
            dtype = ?self.dtype,
            cuda_compute_cap = %cuda_compute_cap,
            path,
            source,
            items = token_stats.items,
            total_tokens = token_stats.total_tokens,
            squared_tokens = token_stats.squared_tokens,
            avg_tokens = token_stats.avg_tokens(),
            min_tokens = token_stats.min_tokens,
            max_tokens = token_stats.max_tokens,
            max_seq_length = self.max_seq_length,
            pooling,
            normalize,
            sync_timings = self.diagnostics.sync_timings,
            gemm_reduced_precision_f32 = gemm_precision.f32,
            gemm_reduced_precision_f16 = gemm_precision.f16,
            gemm_reduced_precision_bf16 = gemm_precision.bf16,
            gemm_reduced_precision_applies_to_compute_dtype = gemm_precision.applies_to_dtype(self.dtype),
            tokenization_ms,
            prepare_ms,
            forward_ms = timings.forward_ms,
            pool_ms = timings.pool_ms,
            normalize_ms = timings.normalize_ms,
            conversion_ms = timings.conversion_ms,
            inference_ms = timings.inference_ms,
            xlm_roberta_embedding_ms = xlm_profile.embedding_ms,
            xlm_roberta_total_tokens = xlm_profile.total_tokens,
            xlm_roberta_max_seqlen = xlm_profile.max_seqlen,
            xlm_roberta_hidden_size = xlm_profile.hidden_size,
            xlm_roberta_intermediate_size = xlm_profile.intermediate_size,
            xlm_roberta_attention_heads = xlm_profile.attention_heads,
            xlm_roberta_attention_head_size = xlm_profile.attention_head_size,
            xlm_roberta_linear_gflops = xlm_profile.linear_gflops,
            xlm_roberta_attention_gflops,
            xlm_roberta_total_gflops,
            xlm_roberta_linear_tflops_s = tflops_per_second(xlm_profile.linear_gflops, timings.forward_ms),
            xlm_roberta_total_tflops_s = tflops_per_second(xlm_roberta_total_gflops, timings.forward_ms),
            xlm_roberta_attention_ms = xlm_profile.attention_ms,
            xlm_roberta_attention_qkv_ms = xlm_profile.attention_qkv_ms,
            xlm_roberta_attention_flash_ms = xlm_profile.attention_flash_ms,
            xlm_roberta_attention_output_dense_ms = xlm_profile.attention_output_dense_ms,
            xlm_roberta_attention_output_layernorm_ms = xlm_profile.attention_output_layernorm_ms,
            xlm_roberta_ffn_ms = xlm_profile.ffn_ms,
            xlm_roberta_ffn_intermediate_dense_ms = xlm_profile.ffn_intermediate_dense_ms,
            xlm_roberta_ffn_activation_ms = xlm_profile.ffn_activation_ms,
            xlm_roberta_ffn_output_dense_ms = xlm_profile.ffn_output_dense_ms,
            xlm_roberta_ffn_output_layernorm_ms = xlm_profile.ffn_output_layernorm_ms,
            xlm_roberta_layers = xlm_profile.layers,
            xlm_roberta_linear_backend,
            xlm_roberta_layernorm_backend,
            xlm_roberta_qkv_fused,
            xlm_roberta_ffn_activation_fused,
            gte_rope_embedding_ms = gte_profile.embedding_ms,
            gte_rope_select_ms = gte_profile.rope_select_ms,
            gte_rope_total_tokens = gte_profile.total_tokens,
            gte_rope_max_seqlen = gte_profile.max_seqlen,
            gte_rope_hidden_size = gte_profile.hidden_size,
            gte_rope_intermediate_size = gte_profile.intermediate_size,
            gte_rope_attention_heads = gte_profile.attention_heads,
            gte_rope_attention_head_size = gte_profile.attention_head_size,
            gte_rope_linear_gflops = gte_profile.linear_gflops,
            gte_rope_attention_gflops,
            gte_rope_total_gflops,
            gte_rope_linear_tflops_s = tflops_per_second(gte_profile.linear_gflops, timings.forward_ms),
            gte_rope_total_tflops_s = tflops_per_second(gte_rope_total_gflops, timings.forward_ms),
            gte_rope_attention_ms = gte_profile.attention_ms,
            gte_rope_attention_qkv_ms = gte_profile.attention_qkv_ms,
            gte_rope_attention_rotary_ms = gte_profile.attention_rotary_ms,
            gte_rope_attention_flash_ms = gte_profile.attention_flash_ms,
            gte_rope_attention_output_dense_ms = gte_profile.attention_output_dense_ms,
            gte_rope_attention_output_layernorm_ms = gte_profile.attention_output_layernorm_ms,
            gte_rope_ffn_ms = gte_profile.ffn_ms,
            gte_rope_ffn_up_gate_ms = gte_profile.ffn_up_gate_ms,
            gte_rope_ffn_activation_ms = gte_profile.ffn_activation_ms,
            gte_rope_ffn_down_ms = gte_profile.ffn_down_ms,
            gte_rope_ffn_output_layernorm_ms = gte_profile.ffn_output_layernorm_ms,
            gte_rope_layers = gte_profile.layers,
            gte_rope_linear_backend,
            gte_rope_layernorm_backend,
            gte_rope_qkv_fused,
            gte_rope_rotary_inplace,
            gte_rope_ffn_activation_fused,
            modernbert_embedding_ms = modernbert_profile.embedding_ms,
            modernbert_embedding_norm_ms = modernbert_profile.embedding_norm_ms,
            modernbert_rope_select_ms = modernbert_profile.rope_select_ms,
            modernbert_total_tokens = modernbert_profile.total_tokens,
            modernbert_max_seqlen = modernbert_profile.max_seqlen,
            modernbert_hidden_size = modernbert_profile.hidden_size,
            modernbert_intermediate_size = modernbert_profile.intermediate_size,
            modernbert_attention_heads = modernbert_profile.attention_heads,
            modernbert_attention_head_size = modernbert_profile.attention_head_size,
            modernbert_linear_gflops = modernbert_profile.linear_gflops,
            modernbert_attention_gflops,
            modernbert_total_gflops,
            modernbert_linear_tflops_s = tflops_per_second(modernbert_profile.linear_gflops, timings.forward_ms),
            modernbert_total_tflops_s = tflops_per_second(modernbert_total_gflops, timings.forward_ms),
            modernbert_attention_ms = modernbert_profile.attention_ms,
            modernbert_attention_norm_ms = modernbert_profile.attention_norm_ms,
            modernbert_attention_qkv_ms = modernbert_profile.attention_qkv_ms,
            modernbert_attention_rotary_ms = modernbert_profile.attention_rotary_ms,
            modernbert_attention_flash_ms = modernbert_profile.attention_flash_ms,
            modernbert_attention_output_dense_ms = modernbert_profile.attention_output_dense_ms,
            modernbert_mlp_ms = modernbert_profile.mlp_ms,
            modernbert_mlp_norm_ms = modernbert_profile.mlp_norm_ms,
            modernbert_mlp_wi_ms = modernbert_profile.mlp_wi_ms,
            modernbert_mlp_activation_ms = modernbert_profile.mlp_activation_ms,
            modernbert_mlp_wo_ms = modernbert_profile.mlp_wo_ms,
            modernbert_final_norm_ms = modernbert_profile.final_norm_ms,
            modernbert_layers = modernbert_profile.layers,
            modernbert_linear_backend,
            modernbert_layernorm_backend,
            "Candle embedding encode diagnostics"
        );
    }

    fn prepared_tensors(
        &self,
        requests: &[CandlePreparedEncodeRequest],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.prepared_tensors_with_max(requests, self.max_seq_length)
    }

    fn prepared_tensors_with_max(
        &self,
        requests: &[CandlePreparedEncodeRequest],
        max_seq_length: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let seq_len = requests
            .iter()
            .map(|request| request.input_ids.len().min(max_seq_length))
            .max()
            .unwrap_or(0);
        if seq_len == 0 {
            anyhow::bail!("prepared Candle tokens are empty");
        }

        let token_ids = requests
            .iter()
            .map(|request| {
                let len = request.input_ids.len().min(seq_len);
                let mut row = vec![self.pad_id; seq_len];
                row[..len].copy_from_slice(&request.input_ids[..len]);
                Tensor::new(row.as_slice(), &self.device)
            })
            .collect::<candle::Result<Vec<_>>>()?;
        let attention_mask = requests
            .iter()
            .map(|request| {
                let len = request.input_ids.len().min(seq_len);
                let mut row = vec![0u32; seq_len];
                match request.attention_mask.as_ref() {
                    Some(mask) if mask.len() >= len => row[..len].copy_from_slice(&mask[..len]),
                    Some(mask) => {
                        candle::bail!(
                            "prepared attention_mask length {} is shorter than input_ids length {}",
                            mask.len(),
                            len
                        );
                    }
                    None => row[..len].fill(1),
                }
                Tensor::new(row.as_slice(), &self.device)
            })
            .collect::<candle::Result<Vec<_>>>()?;
        let token_type_ids = requests
            .iter()
            .map(|request| {
                let len = request.input_ids.len().min(seq_len);
                let mut row = vec![0u32; seq_len];
                if let Some(types) = request.token_type_ids.as_ref() {
                    if types.len() < len {
                        candle::bail!(
                            "prepared token_type_ids length {} is shorter than input_ids length {}",
                            types.len(),
                            len
                        );
                    }
                    row[..len].copy_from_slice(&types[..len]);
                }
                Tensor::new(row.as_slice(), &self.device)
            })
            .collect::<candle::Result<Vec<_>>>()?;

        Ok((
            Tensor::stack(&token_ids, 0)?,
            Tensor::stack(&attention_mask, 0)?,
            Tensor::stack(&token_type_ids, 0)?,
        ))
    }

    fn prepared_packed_tensors(
        &self,
        requests: &[CandlePreparedEncodeRequest],
    ) -> Result<PreparedPackedTensors> {
        self.prepared_packed_tensors_with_max(requests, self.max_seq_length)
    }

    fn prepared_packed_tensors_with_max(
        &self,
        requests: &[CandlePreparedEncodeRequest],
        max_seq_length: usize,
    ) -> Result<PreparedPackedTensors> {
        let seq_lengths = requests
            .iter()
            .map(|request| Self::prepared_effective_len(max_seq_length, request))
            .collect::<Vec<_>>();
        if seq_lengths.contains(&0) {
            anyhow::bail!("prepared Candle tokens are empty");
        }

        let total_tokens = seq_lengths.iter().sum::<usize>();
        let max_seqlen = seq_lengths.iter().copied().max().unwrap_or(0);
        let mut input_ids = Vec::with_capacity(total_tokens);
        let mut token_type_ids = Vec::with_capacity(total_tokens);
        let mut position_ids = Vec::with_capacity(total_tokens);
        let mut seqlens = Vec::with_capacity(requests.len() + 1);
        let mut cursor = 0u32;
        seqlens.push(cursor);
        for (request, len) in requests.iter().zip(seq_lengths.iter().copied()) {
            Self::validate_prepared_attention_mask_len(max_seq_length, request)?;
            input_ids.extend_from_slice(&request.input_ids[..len]);
            Self::append_packed_token_type_ids(
                request,
                len,
                self.packed_token_type_ids_ignored(),
                &mut token_type_ids,
            )?;
            position_ids.extend((0..len).map(|idx| self.packed_position_id(idx)));
            cursor += len as u32;
            seqlens.push(cursor);
        }

        Ok(PreparedPackedTensors {
            input_ids: Tensor::new(input_ids.as_slice(), &self.device)?,
            token_type_ids: Tensor::new(token_type_ids.as_slice(), &self.device)?,
            position_ids: Tensor::new(position_ids.as_slice(), &self.device)?,
            seqlens: Tensor::new(seqlens.as_slice(), &self.device)?,
            seq_lengths,
            max_seqlen,
        })
    }

    fn validate_prepared_attention_mask_len(
        max_seq_length: usize,
        request: &CandlePreparedEncodeRequest,
    ) -> Result<()> {
        let input_len = request.input_ids.len().min(max_seq_length);
        if let Some(mask) = request.attention_mask.as_ref() {
            if mask.len() < input_len {
                anyhow::bail!(
                    "prepared attention_mask length {} is shorter than input_ids length {}",
                    mask.len(),
                    input_len
                );
            }
        }
        Ok(())
    }

    fn append_packed_token_type_ids(
        request: &CandlePreparedEncodeRequest,
        len: usize,
        ignore_token_type_ids: bool,
        out: &mut Vec<u32>,
    ) -> Result<()> {
        if let Some(types) = request.token_type_ids.as_ref() {
            if types.len() < len {
                anyhow::bail!(
                    "prepared token_type_ids length {} is shorter than packed input length {}",
                    types.len(),
                    len
                );
            }
            if ignore_token_type_ids {
                out.extend(std::iter::repeat_n(0, len));
            } else {
                out.extend_from_slice(&types[..len]);
            }
        } else {
            out.extend(std::iter::repeat_n(0, len));
        }
        Ok(())
    }

    fn project_multivector_flat(&self, sequence_output: &Tensor) -> Result<Tensor> {
        let token_dim = self
            .multivector_dim
            .context("Candle multivector encode requested for model without multivector_dim")?;
        let (_, hidden_size) = sequence_output.dims2()?;
        let mut projected = sequence_output.reshape(((), hidden_size))?;
        let chain = self.pylate_dense_chain.as_ref().with_context(|| {
            format!(
                "Candle multivector model {} has no supported PyLate Dense projection",
                self.model_id
            )
        })?;
        for weight in &chain.weights {
            projected = projected.matmul(&weight.t()?)?;
        }
        let (_, projected_dim) = projected.dims2()?;
        validate_multivector_projection_dim(&self.model_id, projected_dim, token_dim)?;
        if chain.output_dim != token_dim {
            anyhow::bail!(
                "Candle PyLate Dense chain for {} ended at dim {} but requested token_dim {}",
                self.model_id,
                chain.output_dim,
                token_dim
            );
        }
        Ok(projected)
    }

    fn project_multivector_tokens(&self, sequence_output: &Tensor) -> Result<Tensor> {
        let token_dim = self
            .multivector_dim
            .context("Candle multivector encode requested for model without multivector_dim")?;
        let (batch_size, seq_len, hidden_size) = sequence_output.dims3()?;
        let projected = self.project_multivector_flat(
            &sequence_output.reshape((batch_size * seq_len, hidden_size))?,
        )?;
        Ok(projected.reshape((batch_size, seq_len, token_dim))?)
    }

    fn project_multivector_tokens_packed(&self, sequence_output: &Tensor) -> Result<Tensor> {
        self.project_multivector_flat(sequence_output)
    }

    fn prepared_effective_len(
        max_seq_length: usize,
        request: &CandlePreparedEncodeRequest,
    ) -> usize {
        let mut len = request.input_ids.len().min(max_seq_length);
        if let Some(mask) = request.attention_mask.as_ref() {
            len = len.min(mask.len());
            while len > 0 && mask[len - 1] == 0 {
                len -= 1;
            }
        }
        len
    }

    /// FlashAttention represents padding solely through cumulative sequence
    /// lengths, so only a contiguous `1...10...0` mask can be packed without
    /// changing semantics. Arbitrary masks stay on the padded fallback.
    fn prepared_attention_masks_are_prefix(
        requests: &[CandlePreparedEncodeRequest],
        max_seq_length: usize,
    ) -> bool {
        requests.iter().all(|request| {
            let input_len = request.input_ids.len().min(max_seq_length);
            if input_len == 0 {
                return false;
            }
            let Some(mask) = request.attention_mask.as_ref() else {
                return true;
            };
            if mask.len() < input_len {
                return false;
            }
            let mut saw_padding = false;
            for value in &mask[..input_len] {
                match (*value, saw_padding) {
                    (1, false) => {}
                    (0, _) => saw_padding = true,
                    _ => return false,
                }
            }
            !mask[..input_len].iter().all(|value| *value == 0)
        })
    }

    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    fn multivector_raw_max_seq_length(&self, is_query: bool) -> usize {
        multivector_raw_max_seq_length(self.max_seq_length, self.query_max_length, is_query)
    }

    pub fn tokenizer_path(&self) -> &Path {
        &self.tokenizer_path
    }

    pub fn tokenizer_id(&self) -> &str {
        &self.tokenizer_id
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

fn validate_multivector_projection_dim(
    model_id: &str,
    projected_dim: usize,
    token_dim: usize,
) -> Result<()> {
    if projected_dim != token_dim {
        anyhow::bail!(
            "Candle multivector projection for {model_id} produced dim {projected_dim} but requested token_dim {token_dim}"
        );
    }
    Ok(())
}

fn tokenizer_for_max_seq_length(
    tokenizer: &Tokenizer,
    max_seq_length: usize,
    hf_id: &str,
) -> Result<Tokenizer> {
    let mut tokenizer = tokenizer.clone();
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_seq_length.max(1),
            ..Default::default()
        }))
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("configure tokenizer truncation for {hf_id}"))?;
    if let Some(existing) = tokenizer.get_padding_mut() {
        existing.strategy = PaddingStrategy::BatchLongest;
    } else {
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
    }
    Ok(tokenizer)
}

fn tokenizer_for_max_seq_length_without_padding(
    tokenizer: &Tokenizer,
    max_seq_length: usize,
    hf_id: &str,
) -> Result<Tokenizer> {
    let mut tokenizer = tokenizer.clone();
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: max_seq_length.max(1),
            ..Default::default()
        }))
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("configure tokenizer truncation for {hf_id}"))?;
    tokenizer.with_padding(None);
    Ok(tokenizer)
}

fn download_weight_file(
    repo: &hf_hub::api::sync::ApiRepo,
    hf_id: &str,
) -> Result<CandleWeightFile> {
    match repo.get("model.safetensors") {
        Ok(path) => Ok(CandleWeightFile::Safetensors(path)),
        Err(safetensors_error) => match repo.get("pytorch_model.bin") {
            Ok(path) => Ok(CandleWeightFile::Pytorch(path)),
            Err(pytorch_error) => anyhow::bail!(
                "download model.safetensors or pytorch_model.bin for {hf_id}: safetensors error: {safetensors_error}; pytorch error: {pytorch_error}"
            ),
        },
    }
}

fn hugging_face_api_from_env() -> Result<Api> {
    let mut builder = ApiBuilder::from_env();
    if let Some(token) = hugging_face_token_from_env_value(std::env::var("HF_TOKEN").ok()) {
        builder = builder.with_token(Some(token));
    }
    builder.build().context("create Hugging Face API client")
}

fn hugging_face_token_from_env_value(value: Option<String>) -> Option<String> {
    value.and_then(|token| {
        let token = token.trim();
        (!token.is_empty()).then(|| token.to_owned())
    })
}

#[cfg(test)]
fn hugging_face_cache_path_from_env() -> PathBuf {
    Cache::from_env().path().clone()
}

fn var_builder_from_weight_file(
    weights_file: CandleWeightFile,
    hf_id: &str,
    device: &Device,
    dtype: DType,
) -> Result<VarBuilder<'static>> {
    match weights_file {
        CandleWeightFile::Safetensors(path) => {
            // The safetensors file is downloaded into the HF cache and treated
            // as immutable while this process maps it.
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[path], dtype, device)
                    .with_context(|| format!("mmap safetensors for {hf_id}"))
            }
        }
        CandleWeightFile::Pytorch(path) => VarBuilder::from_pth(path, dtype, device)
            .with_context(|| format!("load PyTorch checkpoint for {hf_id}")),
    }
}

fn load_pylate_dense_chain(
    repo: &hf_hub::api::sync::ApiRepo,
    hf_id: &str,
    hidden_size: usize,
    token_dim: usize,
    device: &Device,
    dtype: DType,
) -> Result<PylateDenseChain> {
    if hidden_size == 0 || token_dim == 0 {
        anyhow::bail!(
            "Candle multivector model {hf_id} requires positive hidden_size and token_dim"
        );
    }
    let modules_path = match repo.get("modules.json") {
        Ok(path) => path,
        Err(error) => {
            anyhow::bail!(
                "Candle multivector model {hf_id} requires explicit supported PyLate projection metadata, but modules.json could not be loaded: {error}"
            );
        }
    };
    let raw_modules = fs::read_to_string(&modules_path)
        .with_context(|| format!("read {}", modules_path.display()))?;
    let mut modules: Vec<SentenceTransformersModule> = serde_json::from_str(&raw_modules)
        .with_context(|| format!("parse PyLate modules.json for {hf_id}"))?;
    let dense_modules = validate_pylate_modules(hf_id, &mut modules)?;
    if dense_modules.is_empty() {
        if hidden_size == token_dim {
            info!(
                hf_id,
                token_dim,
                "PyLate metadata declares no Dense modules; using exact-dimension identity projection"
            );
            return Ok(PylateDenseChain {
                weights: Vec::new(),
                output_dim: token_dim,
            });
        }
        anyhow::bail!(
            "PyLate modules.json for {hf_id} has no explicit Dense projection from hidden_size {hidden_size} to token_dim {token_dim}"
        );
    }

    let mut expected_in = hidden_size;
    let mut weights = Vec::with_capacity(dense_modules.len());
    for module in dense_modules {
        let config_rel = format!("{}/config.json", module.path);
        let config_path = repo
            .get(&config_rel)
            .with_context(|| format!("download PyLate Dense config {config_rel} for {hf_id}"))?;
        let raw_config = fs::read_to_string(&config_path)
            .with_context(|| format!("read {}", config_path.display()))?;
        let dense_config: PylateDenseConfig = serde_json::from_str(&raw_config)
            .with_context(|| format!("parse PyLate Dense config {config_rel} for {hf_id}"))?;
        validate_pylate_dense_config(hf_id, &module.path, &dense_config, expected_in)?;

        let safetensors_rel = format!("{}/model.safetensors", module.path);
        let pytorch_rel = format!("{}/pytorch_model.bin", module.path);
        let dense_vb = match repo.get(&safetensors_rel) {
            Ok(path) => unsafe {
                VarBuilder::from_mmaped_safetensors(&[path], dtype, device)
                    .with_context(|| format!("mmap {safetensors_rel} for {hf_id}"))?
            },
            Err(safetensors_error) => match repo.get(&pytorch_rel) {
                Ok(path) => VarBuilder::from_pth(path, dtype, device)
                    .with_context(|| format!("load {pytorch_rel} for {hf_id}"))?,
                Err(pytorch_error) => anyhow::bail!(
                    "download PyLate Dense weights for {hf_id} at {}: safetensors error: {safetensors_error}; pytorch error: {pytorch_error}",
                    module.path
                ),
            },
        };
        validate_pylate_dense_weights(hf_id, &module.path, &dense_vb)?;
        let weight = dense_vb
            .get(
                (dense_config.out_features, dense_config.in_features),
                "linear.weight",
            )
            .with_context(|| {
                format!(
                    "load PyLate Dense linear.weight for {hf_id} at {}",
                    module.path
                )
            })?;
        weights.push(weight);
        expected_in = dense_config.out_features;
    }

    if expected_in != token_dim {
        anyhow::bail!(
            "PyLate Dense chain for {hf_id} ended at dim {expected_in}, but requested token_dim is {token_dim}"
        );
    }

    info!(
        hf_id,
        token_dim,
        layers = weights.len(),
        "Loaded PyLate Dense chain for Candle multivector model"
    );
    Ok(PylateDenseChain {
        weights,
        output_dim: token_dim,
    })
}

fn validate_pylate_dense_config(
    hf_id: &str,
    path: &str,
    config: &PylateDenseConfig,
    expected_in: usize,
) -> Result<()> {
    if config.in_features == 0 || config.out_features == 0 {
        anyhow::bail!(
            "PyLate Dense config for {hf_id} at {path} requires positive in_features and out_features"
        );
    }
    if config.bias || config.use_residual {
        anyhow::bail!(
            "Unsupported PyLate Dense config for {hf_id} at {path}: bias={} use_residual={}; Candle supports only biasless, residualless projections",
            config.bias,
            config.use_residual
        );
    }
    if config.activation_function.rsplit('.').next() != Some("Identity") {
        anyhow::bail!(
            "Unsupported PyLate Dense activation {} for {hf_id} at {path}; Candle supports only Identity",
            config.activation_function
        );
    }
    if config.in_features != expected_in {
        anyhow::bail!(
            "PyLate Dense chain for {hf_id} is discontinuous at {path}: in_features={} expected={expected_in}",
            config.in_features
        );
    }
    Ok(())
}

fn validate_pylate_dense_weights(hf_id: &str, path: &str, vb: &VarBuilder<'_>) -> Result<()> {
    if vb.contains_tensor("linear.bias") {
        anyhow::bail!(
            "PyLate Dense weights for {hf_id} at {path} contain linear.bias although the supported projection is biasless"
        );
    }
    Ok(())
}

fn validate_pylate_modules<'a>(
    hf_id: &str,
    modules: &'a mut [SentenceTransformersModule],
) -> Result<Vec<&'a SentenceTransformersModule>> {
    if modules.is_empty() {
        anyhow::bail!("PyLate modules.json for {hf_id} is empty");
    }
    modules.sort_by_key(|module| module.idx);
    for (expected_idx, module) in modules.iter().enumerate() {
        if module.idx != expected_idx {
            anyhow::bail!(
                "PyLate modules.json for {hf_id} must have contiguous indices starting at zero; expected {expected_idx}, got {}",
                module.idx
            );
        }
    }

    let transformer = &modules[0];
    if transformer.module_type != "sentence_transformers.models.Transformer" {
        anyhow::bail!(
            "PyLate modules.json for {hf_id} must start with a Transformer, got {}",
            transformer.module_type
        );
    }
    if !transformer.path.is_empty() {
        anyhow::bail!(
            "PyLate Transformer for {hf_id} must use the repository root, got path {}",
            transformer.path
        );
    }

    let dense_modules = modules.iter().skip(1).collect::<Vec<_>>();
    for module in &dense_modules {
        if module.module_type != "pylate.models.Dense.Dense" {
            anyhow::bail!(
                "Unsupported PyLate module {} after Transformer for {hf_id}; Candle supports only biasless, residualless Identity Dense chains",
                module.module_type
            );
        }
        let path = Path::new(&module.path);
        if module.path.is_empty()
            || path.is_absolute()
            || path
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            anyhow::bail!(
                "PyLate Dense module path {:?} is not a safe repository-relative path for {hf_id}",
                module.path
            );
        }
    }
    Ok(dense_modules)
}

fn dtype_from_compute_precision(value: Option<&str>, device: &Device) -> Result<DType> {
    let dtype = match value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        None | Some("auto") | Some("float32") | Some("fp32") | Some("f32") => DType::F32,
        Some("bfloat16") | Some("bf16") => DType::BF16,
        Some("float16") | Some("fp16") | Some("f16") | Some("half") => DType::F16,
        Some(other) => anyhow::bail!(
            "unsupported Candle compute_precision {other:?}; expected float32, float16, bfloat16, or auto"
        ),
    };

    if matches!(device, Device::Cpu) && matches!(dtype, DType::BF16 | DType::F16) {
        warn!(
            requested_dtype = ?dtype,
            "Candle CPU backend does not reliably support reduced precision; using F32"
        );
        return Ok(DType::F32);
    }

    Ok(dtype)
}

fn tokenizer_content_hash(tokenizer: &Tokenizer) -> Result<String> {
    let canonical = tokenizer
        .to_string(false)
        .map_err(anyhow::Error::msg)
        .context("serialize tokenizer for content hash")?;
    let digest = blake3::hash(canonical.as_bytes());
    Ok(digest.to_hex().as_str()[..32].to_string())
}

fn materialize_tokenizer(model_id: &str, source: &Path, tokenizer_id: &str) -> Result<PathBuf> {
    let Some(root) =
        std::env::var_os("SIE_TOKENIZER_STAGING_DIR").filter(|value| !value.is_empty())
    else {
        return Ok(source.to_path_buf());
    };
    let root = PathBuf::from(root);
    fs::create_dir_all(&root)
        .with_context(|| format!("create tokenizer staging dir {}", root.display()))?;
    let filename = format!("{}-{tokenizer_id}.json", safe_filename(model_id));
    let staged = root.join(filename);
    if !staged.is_file() {
        fs::copy(source, &staged).with_context(|| {
            format!(
                "copy tokenizer {} to staged path {}",
                source.display(),
                staged.display()
            )
        })?;
    }
    Ok(staged)
}

fn safe_filename(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CandleDeviceSpec {
    Cpu,
    Cuda(usize),
    Metal(usize),
}

fn parse_candle_device_spec(raw: &str) -> Result<Option<CandleDeviceSpec>> {
    let value = raw.trim();
    if value.is_empty() {
        return Ok(None);
    }
    let (family, index) = value
        .split_once(':')
        .map_or((value, None), |(family, index)| (family, Some(index)));
    let index = index
        .map(|raw_index| {
            raw_index
                .parse::<usize>()
                .with_context(|| format!("invalid device index in {value:?}"))
        })
        .transpose()?
        .unwrap_or(0);
    match family.to_ascii_lowercase().as_str() {
        "cpu" => Ok(Some(CandleDeviceSpec::Cpu)),
        "cuda" => Ok(Some(CandleDeviceSpec::Cuda(index))),
        "metal" | "mps" => Ok(Some(CandleDeviceSpec::Metal(index))),
        other => anyhow::bail!("unsupported SIE device family {other:?} in {value:?}"),
    }
}

fn configured_candle_device_spec() -> Result<Option<CandleDeviceSpec>> {
    if let Ok(devices) = std::env::var("SIE_DEVICES") {
        if let Some(first) = devices
            .split(',')
            .map(str::trim)
            .find(|item| !item.is_empty())
        {
            return parse_candle_device_spec(first);
        }
    }
    if let Ok(device) = std::env::var("SIE_DEVICE") {
        return parse_candle_device_spec(&device);
    }
    Ok(None)
}

fn device_from_spec(spec: CandleDeviceSpec) -> Result<Device> {
    match spec {
        CandleDeviceSpec::Cpu => Ok(Device::Cpu),
        CandleDeviceSpec::Cuda(index) => Ok(Device::new_cuda(index)?),
        CandleDeviceSpec::Metal(index) => Ok(Device::new_metal(index)?),
    }
}

fn candle_device() -> Result<Device> {
    if let Some(spec) = configured_candle_device_spec()? {
        return device_from_spec(spec);
    }
    if candle::utils::cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if candle::utils::metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}

fn pool_embeddings(
    sequence_output: &Tensor,
    attention_mask: &Tensor,
    pooling: &str,
) -> Result<Tensor> {
    match pooling {
        "cls" => {
            let (batch_size, _seq_len, hidden_size) = sequence_output.dims3()?;
            Ok(sequence_output
                .narrow(1, 0, 1)?
                .reshape((batch_size, hidden_size))?)
        }
        "mean" => {
            let attention_mask = attention_mask
                .to_dtype(sequence_output.dtype())?
                .unsqueeze(2)?;
            let sum_mask = attention_mask.sum(1)?;
            let embeddings = sequence_output
                .broadcast_mul(&attention_mask)?
                .sum(1)?
                .broadcast_div(&sum_mask)?;
            Ok(embeddings)
        }
        _ => anyhow::bail!("unsupported Candle pooling strategy {pooling:?}; expected mean or cls"),
    }
}

fn pool_packed_embeddings(
    sequence_output: &Tensor,
    seqlens: &Tensor,
    seq_lengths: &[usize],
    pooling: &str,
) -> Result<Tensor> {
    if pooling == "cls" {
        let indices = seqlens.narrow(0, 0, seq_lengths.len())?;
        return Ok(candle_layers::index_select(sequence_output, &indices, 0)?);
    }

    let mut cursor = 0usize;
    let mut rows = Vec::with_capacity(seq_lengths.len());
    for len in seq_lengths {
        match pooling {
            "mean" => {
                let sum = sequence_output.narrow(0, cursor, *len)?.sum(0)?;
                rows.push((&sum / *len as f64)?);
            }
            _ => anyhow::bail!(
                "unsupported Candle pooling strategy {pooling:?}; expected mean or cls"
            ),
        }
        cursor += *len;
    }
    Ok(Tensor::stack(&rows, 0)?)
}

fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

fn normalize_l2_last_dim(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(2)?.sqrt()?)?)
}

/// Split documents into stable contiguous chunks bounded by token work. A
/// single long document is never truncated to satisfy the budget; it occupies
/// a chunk by itself and remains governed by the model's sequence cap.
fn score_document_chunks(doc_tokens: &[usize], work_budget: usize) -> Vec<Range<usize>> {
    let work_budget = work_budget.max(1);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut work = 0usize;
    for (index, tokens) in doc_tokens.iter().copied().enumerate() {
        if index > start && work.saturating_add(tokens) > work_budget {
            chunks.push(start..index);
            start = index;
            work = 0;
        }
        work = work.saturating_add(tokens);
    }
    if start < doc_tokens.len() {
        chunks.push(start..doc_tokens.len());
    }
    chunks
}

/// ColBERT late-interaction MaxSim in f32. Both query and documents stay on
/// their source device; only the stacked scalar scores are read back.
fn maxsim_scores_device(query: &Tensor, docs: &[Tensor]) -> Result<Vec<f32>> {
    if docs.is_empty() {
        return Ok(Vec::new());
    }
    let (query_tokens, query_dim) = query.dims2()?;
    if query_tokens == 0 {
        return Ok(vec![0.0; docs.len()]);
    }
    let query = query.to_dtype(DType::F32)?;
    let mut scores = Vec::with_capacity(docs.len());
    for doc in docs {
        let (doc_tokens, doc_dim) = doc.dims2()?;
        if doc_dim != query_dim {
            anyhow::bail!("MaxSim dimension mismatch: query={query_dim} document={doc_dim}");
        }
        if doc_tokens == 0 {
            scores.push(Tensor::new(f32::NEG_INFINITY, query.device())?);
            continue;
        }
        let doc = doc.to_dtype(DType::F32)?;
        let score = query.matmul(&doc.t()?)?.max(1)?.sum_all()?;
        scores.push(score);
    }
    Ok(Tensor::stack(&scores, 0)?.to_vec1::<f32>()?)
}

fn multivector_raw_max_seq_length(
    max_seq_length: usize,
    query_max_length: Option<usize>,
    is_query: bool,
) -> usize {
    if is_query {
        query_max_length.unwrap_or(max_seq_length)
    } else {
        max_seq_length
    }
}

fn packed_position_id(policy: PackedPositionIdPolicy, pad_id: u32, idx: usize) -> u32 {
    match policy {
        PackedPositionIdPolicy::ZeroBased => idx as u32,
        PackedPositionIdPolicy::PaddingOffset => pad_id + 1 + idx as u32,
    }
}

fn split_multivectors(
    projected: &Tensor,
    attention_mask: &Tensor,
    token_dim: usize,
) -> Result<Vec<CandleMultivectorEmbedding>> {
    let projected = projected.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let attention_mask = attention_mask.to_vec2::<u32>()?;
    let mut outputs = Vec::with_capacity(projected.len());
    for (rows, mask) in projected.into_iter().zip(attention_mask) {
        let mut values = Vec::with_capacity(rows.len() * token_dim);
        for (row, keep) in rows.into_iter().zip(mask) {
            if keep != 0 {
                values.extend_from_slice(&row);
            }
        }
        let num_tokens = values.len() / token_dim;
        outputs.push(CandleMultivectorEmbedding {
            values,
            values_f16: Vec::new(),
            num_tokens: num_tokens as u32,
            token_dims: token_dim as u32,
        });
    }
    Ok(outputs)
}

fn split_multivectors_f16(
    projected: &Tensor,
    attention_mask: &Tensor,
    token_dim: usize,
) -> Result<(CandleF16MultivectorBatch, F16MultivectorConversionTimings)> {
    if token_dim == 0 {
        anyhow::bail!("multivector token_dim must be positive");
    }
    let shape = projected.dims();
    let [batch_size, sequence_length, projected_dim] = shape else {
        anyhow::bail!("expected rank-3 masked multivector projection, got shape {shape:?}");
    };
    if *projected_dim != token_dim {
        anyhow::bail!(
            "multivector projected dim {projected_dim} does not match token_dim {token_dim}"
        );
    }

    let tensor_readback_start = Instant::now();
    let projected = projected
        .to_dtype(DType::F16)?
        .flatten_all()?
        .to_vec1::<f16>()?;
    let attention_mask = attention_mask.flatten_all()?.to_vec1::<u32>()?;
    let tensor_readback_ms = elapsed_ms(tensor_readback_start);
    let expected_mask_values = batch_size
        .checked_mul(*sequence_length)
        .context("masked multivector attention-mask size overflow")?;
    let expected_projected_values = expected_mask_values
        .checked_mul(token_dim)
        .context("masked multivector projection size overflow")?;
    if attention_mask.len() != expected_mask_values || projected.len() != expected_projected_values
    {
        anyhow::bail!(
            "masked multivector tensors do not match expected shape [{batch_size}, {sequence_length}, {token_dim}]"
        );
    }

    let host_pack_start = Instant::now();
    let mut values_f16 = Vec::with_capacity(projected.len());
    let mut items = Vec::with_capacity(*batch_size);
    for batch_idx in 0..*batch_size {
        let row_start = batch_idx * *sequence_length;
        let rows = &attention_mask[row_start..row_start + *sequence_length];
        let value_offset = values_f16.len();
        for (row_idx, keep) in rows.iter().enumerate() {
            if *keep != 0 {
                let value_start = (row_start + row_idx) * token_dim;
                let value_end = value_start + token_dim;
                let row = &projected[value_start..value_end];
                if row.len() != token_dim {
                    anyhow::bail!(
                        "multivector row dim {} does not match token_dim {}",
                        row.len(),
                        token_dim
                    );
                }
                values_f16.extend_from_slice(row);
            }
        }
        let value_len = values_f16
            .len()
            .checked_sub(value_offset)
            .context("masked f16 multivector value range underflow")?;
        let byte_offset = value_offset
            .checked_mul(size_of::<f16>())
            .context("masked f16 multivector byte offset overflow")?;
        let byte_len = value_len
            .checked_mul(size_of::<f16>())
            .context("masked f16 multivector byte length overflow")?;
        let num_tokens = value_len / token_dim;
        items.push(CandleF16MultivectorItem {
            byte_offset,
            byte_len,
            num_tokens: num_tokens as u32,
            token_dims: token_dim as u32,
        });
    }
    Ok((
        CandleF16MultivectorBatch { values_f16, items },
        F16MultivectorConversionTimings {
            tensor_readback_ms,
            host_pack_ms: elapsed_ms(host_pack_start),
        },
    ))
}

fn split_packed_multivectors(
    projected: &Tensor,
    seq_lengths: &[usize],
    token_dim: usize,
) -> Result<Vec<CandleMultivectorEmbedding>> {
    let projected = projected.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    let mut outputs = Vec::with_capacity(seq_lengths.len());
    let mut cursor = 0usize;
    for len in seq_lengths {
        let end = cursor + *len;
        if end > projected.len() {
            anyhow::bail!(
                "packed multivector split exceeded projected rows: end={} rows={}",
                end,
                projected.len()
            );
        }
        let mut values = Vec::with_capacity(*len * token_dim);
        for row in &projected[cursor..end] {
            if row.len() != token_dim {
                anyhow::bail!(
                    "packed multivector row dim {} does not match token_dim {}",
                    row.len(),
                    token_dim
                );
            }
            values.extend_from_slice(row);
        }
        outputs.push(CandleMultivectorEmbedding {
            values,
            values_f16: Vec::new(),
            num_tokens: *len as u32,
            token_dims: token_dim as u32,
        });
        cursor = end;
    }
    if cursor != projected.len() {
        anyhow::bail!(
            "packed multivector split consumed {} rows but projected has {} rows",
            cursor,
            projected.len()
        );
    }
    Ok(outputs)
}

fn split_packed_multivectors_f16(
    projected: &Tensor,
    seq_lengths: &[usize],
    token_dim: usize,
) -> Result<(CandleF16MultivectorBatch, F16MultivectorConversionTimings)> {
    if token_dim == 0 {
        anyhow::bail!("multivector token_dim must be positive");
    }
    let tensor_readback_start = Instant::now();
    let projected = projected
        .to_dtype(DType::F16)?
        .flatten_all()?
        .to_vec1::<f16>()?;
    let tensor_readback_ms = elapsed_ms(tensor_readback_start);
    let host_pack_start = Instant::now();
    let mut items = Vec::with_capacity(seq_lengths.len());
    let mut cursor = 0usize;
    for len in seq_lengths {
        let values_len = len
            .checked_mul(token_dim)
            .context("packed multivector f16 size overflow")?;
        let end = cursor
            .checked_add(values_len)
            .context("packed multivector f16 cursor overflow")?;
        if end > projected.len() {
            anyhow::bail!(
                "packed multivector split exceeded projected values: end={} values={}",
                end,
                projected.len()
            );
        }
        let byte_offset = cursor
            .checked_mul(size_of::<f16>())
            .context("packed multivector f16 byte offset overflow")?;
        let byte_len = values_len
            .checked_mul(size_of::<f16>())
            .context("packed multivector f16 byte length overflow")?;
        items.push(CandleF16MultivectorItem {
            byte_offset,
            byte_len,
            num_tokens: *len as u32,
            token_dims: token_dim as u32,
        });
        cursor = end;
    }
    if cursor != projected.len() {
        anyhow::bail!(
            "packed multivector split consumed {} rows but projected has {} rows",
            cursor,
            projected.len()
        );
    }
    Ok((
        CandleF16MultivectorBatch {
            values_f16: projected,
            items,
        },
        F16MultivectorConversionTimings {
            tensor_readback_ms,
            host_pack_ms: elapsed_ms(host_pack_start),
        },
    ))
}

fn extend_f16_le_bytes(out: &mut Vec<u8>, values: &[f16]) {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: f16 is repr(transparent) over u16, and a u8 view has no alignment requirement.
        // On little-endian targets its native representation is the required wire format.
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        out.extend_from_slice(bytes);
    }
    #[cfg(target_endian = "big")]
    for value in values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn f16_values_to_le_bytes(values: &[f16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    extend_f16_le_bytes(&mut bytes, values);
    bytes
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn env_bool_any(names: &[&str], default: bool) -> bool {
    for name in names {
        if let Ok(value) = std::env::var(name) {
            return env_bool_value(&value, default);
        }
    }
    default
}

fn env_bool_value(raw: &str, default: bool) -> bool {
    let trimmed = raw.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return default;
    }
    !matches!(trimmed.as_str(), "0" | "false" | "no" | "off")
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn candle_gemm_reduced_precision_state(device: &Device) -> CandleGemmPrecisionConfig {
    if !device.is_cuda() {
        return CandleGemmPrecisionConfig {
            f32: false,
            f16: false,
            bf16: false,
        };
    }

    CandleGemmPrecisionConfig {
        f32: candle::cuda::gemm_reduced_precision_f32(),
        f16: candle::cuda::gemm_reduced_precision_f16(),
        bf16: candle::cuda::gemm_reduced_precision_bf16(),
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::sync::{Mutex, MutexGuard};

    use super::*;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock() -> MutexGuard<'static, ()> {
        ENV_LOCK.lock().expect("env lock poisoned")
    }

    struct EnvVarGuard {
        key: &'static str,
        original: Option<OsString>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
            let original = std::env::var_os(key);
            std::env::set_var(key, value);
            Self { key, original }
        }

        fn unset(key: &'static str) -> Self {
            let original = std::env::var_os(key);
            std::env::remove_var(key);
            Self { key, original }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }

    #[test]
    fn hf_home_configures_hugging_face_cache_path() {
        let _env_lock = env_lock();
        let tempdir = tempfile::tempdir().unwrap();
        let _guard = EnvVarGuard::set("HF_HOME", tempdir.path());

        assert_eq!(
            hugging_face_cache_path_from_env(),
            tempdir.path().join("hub")
        );
    }

    #[test]
    fn parses_configured_candle_device_specs() {
        assert_eq!(
            parse_candle_device_spec("cuda:2").unwrap(),
            Some(CandleDeviceSpec::Cuda(2))
        );
        assert_eq!(
            parse_candle_device_spec("mps").unwrap(),
            Some(CandleDeviceSpec::Metal(0))
        );
        assert_eq!(
            parse_candle_device_spec("cpu").unwrap(),
            Some(CandleDeviceSpec::Cpu)
        );
        assert!(parse_candle_device_spec("cuda:nope").is_err());
    }

    #[test]
    fn sie_devices_overrides_scalar_device_for_candle() {
        let _env_lock = env_lock();
        let _devices = EnvVarGuard::set("SIE_DEVICES", " cuda:1, cuda:2 ");
        let _device = EnvVarGuard::set("SIE_DEVICE", "cpu");

        assert_eq!(
            configured_candle_device_spec().unwrap(),
            Some(CandleDeviceSpec::Cuda(1))
        );
    }

    #[test]
    fn scalar_device_is_used_when_sie_devices_is_empty() {
        let _env_lock = env_lock();
        let _devices = EnvVarGuard::set("SIE_DEVICES", " , ");
        let _device = EnvVarGuard::set("SIE_DEVICE", "mps:3");

        assert_eq!(
            configured_candle_device_spec().unwrap(),
            Some(CandleDeviceSpec::Metal(3))
        );
    }

    #[test]
    fn missing_device_env_leaves_candle_auto_detect_enabled() {
        let _env_lock = env_lock();
        let _devices = EnvVarGuard::unset("SIE_DEVICES");
        let _device = EnvVarGuard::unset("SIE_DEVICE");

        assert_eq!(configured_candle_device_spec().unwrap(), None);
    }

    #[test]
    fn parses_modernbert_transformers5_rope_parameters() {
        let config = parse_modernbert_config(
            r#"{
              "vocab_size": 50370,
              "hidden_size": 768,
              "num_hidden_layers": 22,
              "num_attention_heads": 12,
              "intermediate_size": 1152,
              "max_position_embeddings": 8192,
              "layer_norm_eps": 1e-5,
              "pad_token_id": 50283,
              "global_attn_every_n_layers": 3,
              "local_attention": 128,
              "rope_parameters": {
                "full_attention": {"rope_theta": 160000.0},
                "sliding_attention": {"rope_theta": 10000.0}
              }
            }"#,
        )
        .unwrap();

        assert_eq!(config.global_rope_theta, 160000.0);
        assert_eq!(config.local_rope_theta, 10000.0);
    }

    #[test]
    fn modernbert_var_builder_preserves_prefixed_checkpoint_layout() {
        let tensors = std::collections::HashMap::from([(
            "model.embeddings.tok_embeddings.weight".to_string(),
            Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap(),
        )]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let vb = modernbert_var_builder(vb);

        assert!(vb.contains_tensor("model.embeddings.tok_embeddings.weight"));
        assert!(!vb.contains_tensor("embeddings.tok_embeddings.weight"));
    }

    #[test]
    fn modernbert_var_builder_accepts_unprefixed_checkpoint_layout() {
        let tensors = std::collections::HashMap::from([(
            "embeddings.tok_embeddings.weight".to_string(),
            Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap(),
        )]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let vb = modernbert_var_builder(vb);

        assert!(vb.contains_tensor("model.embeddings.tok_embeddings.weight"));
        assert!(vb.contains_tensor("embeddings.tok_embeddings.weight"));
    }

    #[test]
    fn selects_supported_architectures_from_model_type() {
        assert_eq!(
            CandleEmbeddingArchitecture::from_probe(&ModelTypeProbe {
                model_type: None,
                architectures: Vec::new(),
                position_embedding_type: None,
                feed_forward_type: None,
            })
            .unwrap(),
            CandleEmbeddingArchitecture::Bert
        );
        assert_eq!(
            CandleEmbeddingArchitecture::from_probe(&probe_for_model_type("bert")).unwrap(),
            CandleEmbeddingArchitecture::Bert
        );
        assert_eq!(
            CandleEmbeddingArchitecture::from_probe(&probe_for_model_type("jina-bert")).unwrap(),
            CandleEmbeddingArchitecture::JinaBert
        );
        assert_eq!(
            CandleEmbeddingArchitecture::from_probe(&probe_for_model_type("xlm-roberta")).unwrap(),
            CandleEmbeddingArchitecture::XlmRoberta
        );
        assert_eq!(
            CandleEmbeddingArchitecture::from_probe(&probe_for_rope_model_type("new")).unwrap(),
            CandleEmbeddingArchitecture::GteRope
        );
        assert_eq!(
            CandleEmbeddingArchitecture::from_probe(&probe_for_rope_model_type("gte")).unwrap(),
            CandleEmbeddingArchitecture::GteRope
        );
        assert_eq!(
            CandleEmbeddingArchitecture::from_probe(&probe_for_model_type("modernbert")).unwrap(),
            CandleEmbeddingArchitecture::ModernBert
        );
        assert_eq!(
            CandleEmbeddingArchitecture::from_probe(&probe_for_model_type("nomic_bert")).unwrap(),
            CandleEmbeddingArchitecture::NomicBert
        );
    }

    #[test]
    fn selects_jina_bert_when_model_type_is_bert_but_architecture_is_jina() {
        let probe = ModelTypeProbe {
            model_type: Some("bert".to_string()),
            architectures: vec!["JinaBertForMaskedLM".to_string()],
            position_embedding_type: Some("alibi".to_string()),
            feed_forward_type: Some("geglu".to_string()),
        };

        assert_eq!(
            CandleEmbeddingArchitecture::from_probe(&probe).unwrap(),
            CandleEmbeddingArchitecture::JinaBert
        );
    }

    #[test]
    fn batch_token_stats_counts_each_sequence_once() {
        let stats = BatchTokenStats::from_lengths([2usize, 3, 5]);

        assert_eq!(stats.items, 3);
        assert_eq!(stats.total_tokens, 10);
        assert_eq!(stats.squared_tokens, 38);
        assert_eq!(stats.min_tokens, 2);
        assert_eq!(stats.max_tokens, 5);
    }

    #[test]
    fn multivector_raw_max_seq_length_uses_query_cap_only_for_queries() {
        assert_eq!(multivector_raw_max_seq_length(8192, Some(32), true), 32);
        assert_eq!(multivector_raw_max_seq_length(8192, Some(32), false), 8192);
        assert_eq!(multivector_raw_max_seq_length(8192, None, true), 8192);
    }

    #[test]
    fn score_document_chunks_respect_budget_without_truncating_long_docs() {
        assert_eq!(
            score_document_chunks(&[4, 5, 20, 3], 10),
            vec![0..2, 2..3, 3..4]
        );
        assert_eq!(score_document_chunks(&[20], 10), vec![0..1]);
        assert_eq!(score_document_chunks(&[512, 512], 16384), vec![0..2]);
        assert!(score_document_chunks(&[], 10).is_empty());
    }

    #[test]
    fn maxsim_scores_device_accumulates_in_f32_and_handles_empty_docs() -> Result<()> {
        let query =
            Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu)?.to_dtype(DType::F16)?;
        let exact =
            Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu)?.to_dtype(DType::F16)?;
        let opposite =
            Tensor::new(&[[-1.0f32, 0.0], [0.0, -1.0]], &Device::Cpu)?.to_dtype(DType::F16)?;
        let empty = Tensor::zeros((0, 2), DType::F16, &Device::Cpu)?;

        let scores = maxsim_scores_device(&query, &[exact, opposite, empty])?;

        assert_eq!(scores.len(), 3);
        assert!((scores[0] - 2.0).abs() < 1e-6);
        assert!(scores[1].abs() < 1e-6);
        assert_eq!(scores[2], f32::NEG_INFINITY);
        Ok(())
    }

    #[test]
    fn maxsim_scores_device_empty_query_is_zero_per_document() -> Result<()> {
        let query = Tensor::zeros((0, 2), DType::F32, &Device::Cpu)?;
        let doc = Tensor::new(&[[1.0f32, 0.0]], &Device::Cpu)?;
        assert_eq!(maxsim_scores_device(&query, &[doc])?, vec![0.0]);
        Ok(())
    }

    #[test]
    fn multivector_projection_dimension_must_match_without_truncation() {
        validate_multivector_projection_dim("test/model", 128, 128).unwrap();
        let error = validate_multivector_projection_dim("test/model", 768, 128).unwrap_err();
        assert!(error.to_string().contains("produced dim 768"));
    }

    #[test]
    fn pylate_modules_require_root_transformer_and_contiguous_dense_chain() {
        let mut valid = vec![
            SentenceTransformersModule {
                idx: 1,
                path: "1_Dense".to_string(),
                module_type: "pylate.models.Dense.Dense".to_string(),
            },
            SentenceTransformersModule {
                idx: 0,
                path: String::new(),
                module_type: "sentence_transformers.models.Transformer".to_string(),
            },
        ];
        let dense = validate_pylate_modules("test/model", &mut valid).unwrap();
        assert_eq!(dense.len(), 1);
        assert_eq!(dense[0].path, "1_Dense");

        let mut non_root = vec![SentenceTransformersModule {
            idx: 0,
            path: "0_Transformer".to_string(),
            module_type: "sentence_transformers.models.Transformer".to_string(),
        }];
        assert!(validate_pylate_modules("test/model", &mut non_root)
            .unwrap_err()
            .to_string()
            .contains("repository root"));

        let mut non_contiguous = vec![
            SentenceTransformersModule {
                idx: 0,
                path: String::new(),
                module_type: "sentence_transformers.models.Transformer".to_string(),
            },
            SentenceTransformersModule {
                idx: 2,
                path: "2_Dense".to_string(),
                module_type: "pylate.models.Dense.Dense".to_string(),
            },
        ];
        assert!(validate_pylate_modules("test/model", &mut non_contiguous)
            .unwrap_err()
            .to_string()
            .contains("contiguous indices"));
    }

    #[test]
    fn pylate_modules_reject_unsafe_dense_paths() {
        for path in ["", "../Dense", "/tmp/Dense", "nested/../Dense"] {
            let mut modules = vec![
                SentenceTransformersModule {
                    idx: 0,
                    path: String::new(),
                    module_type: "sentence_transformers.models.Transformer".to_string(),
                },
                SentenceTransformersModule {
                    idx: 1,
                    path: path.to_string(),
                    module_type: "pylate.models.Dense.Dense".to_string(),
                },
            ];
            assert!(validate_pylate_modules("test/model", &mut modules).is_err());
        }
    }

    #[test]
    fn pylate_dense_config_is_explicit_biasless_identity_projection() {
        let valid = PylateDenseConfig {
            in_features: 768,
            out_features: 128,
            bias: false,
            use_residual: false,
            activation_function: "torch.nn.modules.linear.Identity".to_string(),
        };
        validate_pylate_dense_config("test/model", "1_Dense", &valid, 768).unwrap();

        for (invalid, expected) in [
            (
                {
                    let mut config = valid.clone();
                    config.in_features = 0;
                    config
                },
                "positive",
            ),
            (
                {
                    let mut config = valid.clone();
                    config.out_features = 0;
                    config
                },
                "positive",
            ),
            (
                {
                    let mut config = valid.clone();
                    config.bias = true;
                    config
                },
                "bias=true",
            ),
            (
                {
                    let mut config = valid.clone();
                    config.use_residual = true;
                    config
                },
                "use_residual=true",
            ),
            (
                {
                    let mut config = valid.clone();
                    config.activation_function = "torch.nn.GELU".to_string();
                    config
                },
                "only Identity",
            ),
            (
                {
                    let mut config = valid.clone();
                    config.in_features = 384;
                    config
                },
                "discontinuous",
            ),
        ] {
            let error =
                validate_pylate_dense_config("test/model", "1_Dense", &invalid, 768).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn pylate_modules_require_explicit_index_path_and_type() {
        for raw in [
            r#"[{"path":"","type":"sentence_transformers.models.Transformer"}]"#,
            r#"[{"idx":0,"type":"sentence_transformers.models.Transformer"}]"#,
            r#"[{"idx":0,"path":""}]"#,
        ] {
            assert!(serde_json::from_str::<Vec<SentenceTransformersModule>>(raw).is_err());
        }

        let unknown_dense = r#"{
            "in_features": 768,
            "out_features": 128,
            "bias": false,
            "use_residual": false,
            "activation_function": "torch.nn.modules.linear.Identity",
            "scale": 0.5
        }"#;
        assert!(serde_json::from_str::<PylateDenseConfig>(unknown_dense).is_err());
    }

    #[test]
    fn pylate_modules_reject_unrelated_terminal_class_names() {
        let mut modules = vec![
            SentenceTransformersModule {
                idx: 0,
                path: String::new(),
                module_type: "other.models.Transformer".to_string(),
            },
            SentenceTransformersModule {
                idx: 1,
                path: "1_Dense".to_string(),
                module_type: "other.models.Dense".to_string(),
            },
        ];
        assert!(validate_pylate_modules("test/model", &mut modules).is_err());
    }

    #[test]
    fn pylate_dense_weights_reject_unconfigured_bias_tensor() {
        let tensors = std::collections::HashMap::from([
            (
                "linear.weight".to_string(),
                Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap(),
            ),
            (
                "linear.bias".to_string(),
                Tensor::zeros(2, DType::F32, &Device::Cpu).unwrap(),
            ),
        ]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        assert!(validate_pylate_dense_weights("test/model", "1_Dense", &vb)
            .unwrap_err()
            .to_string()
            .contains("linear.bias"));
    }

    #[test]
    fn rejects_unsupported_architecture_model_type() {
        let err = CandleEmbeddingArchitecture::from_probe(&probe_for_model_type("qwen2"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not support model_type"));
        assert!(err.contains("qwen2"));
    }

    fn probe_for_model_type(model_type: &str) -> ModelTypeProbe {
        ModelTypeProbe {
            model_type: Some(model_type.to_string()),
            architectures: Vec::new(),
            position_embedding_type: None,
            feed_forward_type: None,
        }
    }

    fn probe_for_rope_model_type(model_type: &str) -> ModelTypeProbe {
        ModelTypeProbe {
            model_type: Some(model_type.to_string()),
            architectures: Vec::new(),
            position_embedding_type: Some("rope".to_string()),
            feed_forward_type: None,
        }
    }

    #[test]
    fn rejects_unknown_pooling_without_loading_model() {
        let err = pool_embeddings(
            &Tensor::zeros((1, 1, 1), DType::F32, &Device::Cpu).unwrap(),
            &Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap(),
            "last_token",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("unsupported Candle pooling strategy"));
    }

    #[test]
    fn prepared_effective_len_strips_trailing_padding_mask() {
        let request = CandlePreparedEncodeRequest {
            input_ids: vec![0, 11, 12, 1, 1],
            attention_mask: Some(vec![1, 1, 1, 0, 0]),
            token_type_ids: None,
        };

        assert_eq!(
            CandleEmbeddingModel::prepared_effective_len(8192, &request),
            3
        );
        assert_eq!(CandleEmbeddingModel::prepared_effective_len(2, &request), 2);
    }

    #[test]
    fn packed_prepared_tokens_reject_short_attention_mask() {
        let request = CandlePreparedEncodeRequest {
            input_ids: vec![0, 11, 12],
            attention_mask: Some(vec![1, 1]),
            token_type_ids: None,
        };

        let err = CandleEmbeddingModel::validate_prepared_attention_mask_len(8192, &request)
            .unwrap_err()
            .to_string();
        assert!(err.contains("prepared attention_mask length 2 is shorter than input_ids length 3"));
    }

    #[test]
    fn packed_splade_accepts_only_prefix_attention_masks() {
        let prefix = CandlePreparedEncodeRequest {
            input_ids: vec![101, 11, 12, 0, 0],
            attention_mask: Some(vec![1, 1, 1, 0, 0]),
            token_type_ids: None,
        };
        let implicit = CandlePreparedEncodeRequest {
            input_ids: vec![101, 13, 102],
            attention_mask: None,
            token_type_ids: None,
        };
        assert!(CandleEmbeddingModel::prepared_attention_masks_are_prefix(
            &[prefix, implicit],
            512
        ));

        for mask in [vec![1, 0, 1], vec![0, 0, 0], vec![1, 2, 0], vec![1, 1]] {
            let request = CandlePreparedEncodeRequest {
                input_ids: vec![101, 11, 102],
                attention_mask: Some(mask),
                token_type_ids: None,
            };
            assert!(!CandleEmbeddingModel::prepared_attention_masks_are_prefix(
                &[request],
                512
            ));
        }
    }

    #[test]
    fn packed_prepared_tokens_zero_ignored_token_type_ids() {
        let request = CandlePreparedEncodeRequest {
            input_ids: vec![0, 11, 12],
            attention_mask: Some(vec![1, 1, 1]),
            token_type_ids: Some(vec![0, 1, 2]),
        };
        let mut out = Vec::new();

        CandleEmbeddingModel::append_packed_token_type_ids(&request, 3, true, &mut out).unwrap();

        assert_eq!(out, vec![0, 0, 0]);
    }

    #[test]
    fn packed_prepared_tokens_preserve_supported_token_type_ids() {
        let request = CandlePreparedEncodeRequest {
            input_ids: vec![0, 11, 12],
            attention_mask: Some(vec![1, 1, 1]),
            token_type_ids: Some(vec![0, 1, 0]),
        };
        let mut out = Vec::new();

        CandleEmbeddingModel::append_packed_token_type_ids(&request, 3, false, &mut out).unwrap();

        assert_eq!(out, vec![0, 1, 0]);
    }

    #[test]
    fn packed_position_ids_match_model_family_semantics() {
        assert_eq!(
            packed_position_id(PackedPositionIdPolicy::ZeroBased, 42, 7),
            7,
            "RoPE packed paths such as ModernBERT restart positions at zero"
        );
        assert_eq!(
            packed_position_id(PackedPositionIdPolicy::PaddingOffset, 1, 7),
            9,
            "XLM-R packed positions mirror padding_idx + 1 offset"
        );
    }

    #[test]
    fn pool_packed_embeddings_matches_cls_and_mean_semantics() -> Result<()> {
        let sequence_output = Tensor::new(
            &[[1f32, 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]],
            &Device::Cpu,
        )?;
        let seqlens = Tensor::new(&[0u32, 2, 5], &Device::Cpu)?;

        let cls = pool_packed_embeddings(&sequence_output, &seqlens, &[2, 3], "cls")?;
        assert_eq!(cls.to_vec2::<f32>()?, vec![vec![1., 2.], vec![5., 6.]]);

        let mean = pool_packed_embeddings(&sequence_output, &seqlens, &[2, 3], "mean")?;
        assert_eq!(mean.to_vec2::<f32>()?, vec![vec![2., 3.], vec![7., 8.]]);
        Ok(())
    }

    #[test]
    fn compute_precision_parser_uses_safe_cpu_fallback() {
        assert_eq!(
            dtype_from_compute_precision(Some("float32"), &Device::Cpu).unwrap(),
            DType::F32
        );
        assert_eq!(
            dtype_from_compute_precision(Some("bfloat16"), &Device::Cpu).unwrap(),
            DType::F32
        );
        assert!(dtype_from_compute_precision(Some("int8"), &Device::Cpu).is_err());
    }

    #[test]
    fn f16_wire_bytes_are_little_endian() {
        let values = [f16::from_bits(0x3c00), f16::from_bits(0x8001)];

        assert_eq!(
            f16_values_to_le_bytes(&values),
            vec![0x00, 0x3c, 0x01, 0x80]
        );
    }

    #[test]
    fn split_packed_multivectors_f16_uses_contiguous_item_slices() -> Result<()> {
        let projected = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &Device::Cpu)?;

        let (batch, timings) = split_packed_multivectors_f16(&projected, &[2, 1], 2)?;

        assert_eq!(batch.items.len(), 2);
        assert_eq!(batch.items[0].byte_offset, 0);
        assert_eq!(batch.items[0].byte_len, 8);
        assert_eq!(batch.items[0].num_tokens, 2);
        assert_eq!(batch.items[0].token_dims, 2);
        assert_eq!(
            f16_values_to_le_bytes(&batch.values_f16[0..4]),
            f16_values_to_le_bytes(&[
                f16::from_f32(1.),
                f16::from_f32(2.),
                f16::from_f32(3.),
                f16::from_f32(4.),
            ])
        );
        assert_eq!(batch.items[1].byte_offset, 8);
        assert_eq!(batch.items[1].byte_len, 4);
        assert_eq!(batch.items[1].num_tokens, 1);
        assert_eq!(
            f16_values_to_le_bytes(&batch.values_f16[4..6]),
            f16_values_to_le_bytes(&[f16::from_f32(5.), f16::from_f32(6.)])
        );
        let legacy = batch.clone().into_individual()?;
        assert_eq!(legacy.len(), 2);
        assert_eq!(
            legacy[0].values_f16,
            f16_values_to_le_bytes(&[
                f16::from_f32(1.),
                f16::from_f32(2.),
                f16::from_f32(3.),
                f16::from_f32(4.),
            ])
        );
        assert!(timings.tensor_readback_ms >= 0.0);
        assert!(timings.host_pack_ms >= 0.0);
        Ok(())
    }

    #[test]
    fn split_masked_multivectors_f16_preserves_selected_rows() -> Result<()> {
        let projected = Tensor::new(
            &[
                [[1f32, 2.], [3., 4.], [5., 6.]],
                [[7., 8.], [9., 10.], [11., 12.]],
            ],
            &Device::Cpu,
        )?;
        let attention_mask = Tensor::new(&[[1u32, 0, 1], [0, 1, 0]], &Device::Cpu)?;

        let (batch, timings) = split_multivectors_f16(&projected, &attention_mask, 2)?;

        assert_eq!(batch.items.len(), 2);
        assert_eq!(batch.items[0].byte_offset, 0);
        assert_eq!(batch.items[0].byte_len, 8);
        assert_eq!(batch.items[0].num_tokens, 2);
        assert_eq!(
            f16_values_to_le_bytes(&batch.values_f16[0..4]),
            f16_values_to_le_bytes(&[
                f16::from_f32(1.),
                f16::from_f32(2.),
                f16::from_f32(5.),
                f16::from_f32(6.),
            ])
        );
        assert_eq!(batch.items[1].byte_offset, 8);
        assert_eq!(batch.items[1].byte_len, 4);
        assert_eq!(batch.items[1].num_tokens, 1);
        assert_eq!(
            f16_values_to_le_bytes(&batch.values_f16[4..6]),
            f16_values_to_le_bytes(&[f16::from_f32(9.), f16::from_f32(10.)])
        );
        assert!(timings.tensor_readback_ms >= 0.0);
        assert!(timings.host_pack_ms >= 0.0);
        Ok(())
    }

    #[test]
    fn gemm_precision_flags_apply_only_to_matching_compute_dtype() {
        let config = CandleGemmPrecisionConfig {
            f32: false,
            f16: true,
            bf16: false,
        };

        assert!(config.applies_to_dtype(DType::F16));
        assert!(!config.applies_to_dtype(DType::BF16));
        assert!(!config.applies_to_dtype(DType::F32));
    }

    #[test]
    fn hugging_face_token_from_env_value_uses_only_nonempty_tokens() {
        assert_eq!(
            hugging_face_token_from_env_value(Some("  hf_read_token  ".to_string())),
            Some("hf_read_token".to_string())
        );
        assert_eq!(
            hugging_face_token_from_env_value(Some(" \n\t ".to_string())),
            None
        );
        assert_eq!(hugging_face_token_from_env_value(None), None);
    }

    #[test]
    #[ignore = "downloads a Hugging Face model and runs Candle inference"]
    fn live_encodes_all_minilm() -> Result<()> {
        let model = CandleEmbeddingModel::load(&CandleEmbeddingModelConfig {
            model_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            hf_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
            hf_revision: None,
            max_seq_length: 256,
            query_max_length: None,
            dense_dim: Some(384),
            sparse_dim: None,
            multivector_dim: None,
            compute_precision: None,
        })?;
        let encoded = model.encode(
            &[CandleEncodeRequest {
                text: "query: what is candle?".to_string(),
            }],
            "mean",
            true,
        )?;
        assert_eq!(encoded.dim, 384);
        assert_eq!(encoded.embeddings.len(), 1);
        assert_eq!(encoded.embeddings[0].len(), 384);
        assert!(encoded.embeddings[0].iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    #[ignore = "downloads pinned prithivida/Splade_PP_en_v2 PyTorch weights and runs Candle inference"]
    fn live_encodes_splade_pytorch_checkpoint() -> Result<()> {
        let model = CandleEmbeddingModel::load(&CandleEmbeddingModelConfig {
            model_id: "prithivida/Splade_PP_en_v2".to_string(),
            hf_id: "prithivida/Splade_PP_en_v2".to_string(),
            hf_revision: Some("f0d4aa214dcb60c274052a52c0497535e3aec64c".to_string()),
            max_seq_length: 512,
            query_max_length: None,
            dense_dim: None,
            sparse_dim: Some(30522),
            multivector_dim: None,
            compute_precision: Some("float32".to_string()),
        })?;
        let encoded = model.encode_sparse(&[
            CandleEncodeRequest {
                text: "test".to_string(),
            },
            CandleEncodeRequest {
                text: "A longer sentence keeps sparse batch ordering honest.".to_string(),
            },
        ])?;

        assert_eq!(encoded.dim, 30522);
        let sparse = encoded
            .sparse_embeddings
            .expect("SPLADE encode returns sparse rows");
        assert_eq!(sparse.len(), 2);
        let first = &sparse[0];
        assert!(first.indices.len() >= 3);
        assert_eq!(&first.indices[..3], &[2668, 2671, 2674]);
        for (actual, expected) in first.values[..3].iter().copied().map(f64::from).zip([
            0.1822509765625,
            0.1221923828125,
            0.061126708984375,
        ]) {
            assert!(
                (actual - expected).abs() <= 0.05,
                "SPLADE golden weight mismatch: actual={actual} expected={expected}"
            );
        }
        for row in sparse {
            assert!(!row.indices.is_empty());
            assert_eq!(row.indices.len(), row.values.len());
            assert!(row.indices.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(row
                .values
                .iter()
                .all(|value| value.is_finite() && *value > 0.0));
        }
        Ok(())
    }

    #[test]
    #[ignore = "downloads pinned SPLADE weights and compares CUDA F16 packed/padded forwards"]
    fn live_packed_splade_matches_padded_f16() -> Result<()> {
        let model = CandleEmbeddingModel::load(&CandleEmbeddingModelConfig {
            model_id: "prithivida/Splade_PP_en_v2".to_string(),
            hf_id: "prithivida/Splade_PP_en_v2".to_string(),
            hf_revision: Some("f0d4aa214dcb60c274052a52c0497535e3aec64c".to_string()),
            max_seq_length: 512,
            query_max_length: None,
            dense_dim: None,
            sparse_dim: Some(30522),
            multivector_dim: None,
            compute_precision: Some("float16".to_string()),
        })?;
        if !model.should_use_packed_splade() {
            anyhow::bail!("packed SPLADE parity test requires a CUDA F16 runtime")
        }

        let texts = [
            "how to sort a list in python",
            "segfault when freeing a pointer twice in C",
            "SPLADE max-pools lexical term weights over a mixed-length token batch.",
        ];
        let mut prepared = texts
            .iter()
            .map(|text| {
                model
                    .tokenizer
                    .encode(*text, true)
                    .map_err(anyhow::Error::msg)
                    .map(|encoding| CandlePreparedEncodeRequest {
                        input_ids: encoding.get_ids().to_vec(),
                        attention_mask: Some(encoding.get_attention_mask().to_vec()),
                        token_type_ids: Some(encoding.get_type_ids().to_vec()),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(token_types) = prepared[1].token_type_ids.as_mut() {
            let midpoint = token_types.len() / 2;
            token_types[midpoint..].fill(1);
        }

        let token_stats = model.prepared_token_stats(&prepared);
        let (token_ids, attention_mask, token_type_ids) = model.prepared_tensors(&prepared)?;
        let padded = model.encode_sparse_tensors(
            &token_ids,
            &attention_mask,
            &token_type_ids,
            0.0,
            token_stats,
            0.0,
            "prepared",
        )?;
        let packed_tensors = model.prepared_packed_tensors(&prepared)?;
        let packed = model.encode_packed_sparse(&packed_tensors, 0.0, 0.0, "prepared")?;
        let padded = padded
            .sparse_embeddings
            .context("padded SPLADE parity output is missing")?;
        let packed = packed
            .sparse_embeddings
            .context("packed SPLADE parity output is missing")?;
        assert_eq!(packed.len(), padded.len());

        let densify = |row: &CandleSparseEmbedding| {
            let mut dense = vec![0.0f32; 30522];
            for (&index, &value) in row.indices.iter().zip(&row.values) {
                dense[index as usize] = value;
            }
            dense
        };
        for (packed, padded) in packed.iter().zip(&padded) {
            let packed = densify(packed);
            let padded = densify(padded);
            let max_diff = packed
                .iter()
                .zip(&padded)
                .map(|(packed, padded)| (packed - padded).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 0.05,
                "packed/padded SPLADE max sparse-weight difference {max_diff} exceeds tolerance"
            );

            let top_indices = |values: &[f32]| {
                let mut indices = (0..values.len()).collect::<Vec<_>>();
                indices.sort_unstable_by(|left, right| values[*right].total_cmp(&values[*left]));
                indices
                    .into_iter()
                    .take(20)
                    .collect::<std::collections::HashSet<_>>()
            };
            let packed_top = top_indices(&packed);
            let padded_top = top_indices(&padded);
            let overlap = packed_top.intersection(&padded_top).count();
            assert!(
                overlap >= 18,
                "packed/padded SPLADE top-20 overlap {overlap}/20 is below tolerance"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "downloads BAAI/bge-m3 and runs Candle XLM-R dense inference"]
    fn live_encodes_bge_m3_dense() -> Result<()> {
        let model = CandleEmbeddingModel::load(&CandleEmbeddingModelConfig {
            model_id: "BAAI/bge-m3".to_string(),
            hf_id: "BAAI/bge-m3".to_string(),
            hf_revision: None,
            max_seq_length: 8192,
            query_max_length: None,
            dense_dim: Some(1024),
            sparse_dim: None,
            multivector_dim: None,
            compute_precision: Some("bfloat16".to_string()),
        })?;
        let encoded = model.encode(
            &[CandleEncodeRequest {
                text: "Represent this sentence for searching relevant passages.".to_string(),
            }],
            "cls",
            true,
        )?;
        assert_eq!(encoded.dim, 1024);
        assert_eq!(encoded.embeddings.len(), 1);
        assert_eq!(encoded.embeddings[0].len(), 1024);
        assert!(encoded.embeddings[0].iter().all(|value| value.is_finite()));
        Ok(())
    }

    #[test]
    #[ignore = "downloads Snowflake Arctic Embed L v2.0 and runs CUDA XLM-R FP16 inference"]
    fn live_encodes_snowflake_arctic_dense() -> Result<()> {
        let model = CandleEmbeddingModel::load(&CandleEmbeddingModelConfig {
            model_id: "Snowflake/snowflake-arctic-embed-l-v2.0".to_string(),
            hf_id: "Snowflake/snowflake-arctic-embed-l-v2.0".to_string(),
            hf_revision: Some("ac6544c8a46e00af67e330e85a9028c66b8cfd9a".to_string()),
            max_seq_length: 8192,
            query_max_length: None,
            dense_dim: Some(1024),
            sparse_dim: None,
            multivector_dim: None,
            compute_precision: Some("float16".to_string()),
        })?;
        let texts = [
            "query: what is Snowflake Arctic Embed?",
            "Snowflake Arctic Embed is a multilingual retrieval model.",
            "query: Wo findet man die besten Tacos?",
            "Die besten Tacos findet man in Mexiko-Stadt.",
        ];
        let requests = texts
            .iter()
            .map(|text| CandleEncodeRequest {
                text: (*text).to_string(),
            })
            .collect::<Vec<_>>();
        let encoded = model.encode(&requests, "cls", true)?;

        assert_eq!(encoded.dim, 1024);
        assert_eq!(encoded.embeddings.len(), texts.len());
        for embedding in &encoded.embeddings {
            assert_eq!(embedding.len(), 1024);
            assert!(embedding.iter().all(|value| value.is_finite()));
            let norm = embedding
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-3,
                "embedding norm {norm} is not unit length"
            );
        }

        let prepared = texts
            .iter()
            .map(|text| {
                model
                    .tokenizer
                    .encode(*text, true)
                    .map_err(anyhow::Error::msg)
                    .map(|encoding| CandlePreparedEncodeRequest {
                        input_ids: encoding.get_ids().to_vec(),
                        attention_mask: Some(encoding.get_attention_mask().to_vec()),
                        token_type_ids: Some(encoding.get_type_ids().to_vec()),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let packed = model.prepared_packed_tensors(&prepared)?;
        let xlm_roberta = match &model.model {
            CandleEmbeddingInner::XlmRoberta(model) => model,
            _ => anyhow::bail!("Snowflake Arctic did not load through XLM-R"),
        };
        let full_output = xlm_roberta.forward_packed(
            &packed.input_ids,
            &packed.position_ids,
            &packed.seqlens,
            packed.max_seqlen,
        )?;
        let full_cls =
            pool_packed_embeddings(&full_output, &packed.seqlens, &packed.seq_lengths, "cls")?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
        let fast_cls = xlm_roberta
            .forward_packed_cls(
                &packed.input_ids,
                &packed.position_ids,
                &packed.seqlens,
                packed.max_seqlen,
            )?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let (profiled_cls, profile) = xlm_roberta.forward_packed_cls_profiled(
            &packed.input_ids,
            &packed.position_ids,
            &packed.seqlens,
            packed.max_seqlen,
            false,
        )?;
        let profiled_cls = profiled_cls.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        assert_eq!(full_cls.len(), texts.len());
        assert_eq!(fast_cls.len(), full_cls.len());
        assert_eq!(profiled_cls.len(), full_cls.len());
        assert_eq!(profile.layers, 24);
        assert_eq!(
            profile.total_tokens,
            packed.seq_lengths.iter().sum::<usize>()
        );
        for ((full, fast), profiled) in full_cls.iter().zip(&fast_cls).zip(&profiled_cls) {
            assert!(
                cosine_similarity(full, fast) > 0.9999,
                "full and CLS-tail Snowflake Arctic forwards diverged"
            );
            assert!(
                cosine_similarity(fast, profiled) > 0.9999,
                "profiled and unprofiled CLS-tail Snowflake Arctic forwards diverged"
            );
        }

        for (text, batched) in texts.iter().zip(&encoded.embeddings) {
            let singleton = model.encode(
                &[CandleEncodeRequest {
                    text: (*text).to_string(),
                }],
                "cls",
                true,
            )?;
            assert!(
                cosine_similarity(batched, &singleton.embeddings[0]) > 0.999,
                "batched and singleton Snowflake Arctic embeddings diverged"
            );
        }

        Ok(())
    }

    #[test]
    #[ignore = "downloads Alibaba-NLP/gte-multilingual-base and runs CUDA GTE-RoPE dense inference"]
    fn live_encodes_gte_multilingual_base_rope_dense() -> Result<()> {
        let model = CandleEmbeddingModel::load(&CandleEmbeddingModelConfig {
            model_id: "Alibaba-NLP/gte-multilingual-base".to_string(),
            hf_id: "Alibaba-NLP/gte-multilingual-base".to_string(),
            hf_revision: None,
            max_seq_length: 8192,
            query_max_length: None,
            dense_dim: Some(768),
            sparse_dim: None,
            multivector_dim: None,
            compute_precision: Some("float16".to_string()),
        })?;
        let texts = [
            "Represent this sentence for searching relevant passages.",
            "A second sentence keeps the packed batch path honest.",
        ];
        let requests = texts
            .iter()
            .map(|text| CandleEncodeRequest {
                text: (*text).to_string(),
            })
            .collect::<Vec<_>>();
        let encoded = model.encode(&requests, "cls", true)?;
        assert_eq!(encoded.dim, 768);
        assert_eq!(encoded.embeddings.len(), texts.len());
        assert!(encoded
            .embeddings
            .iter()
            .all(|embedding| embedding.len() == 768
                && embedding.iter().all(|value| value.is_finite())));

        let prepared = texts
            .iter()
            .map(|text| {
                model
                    .tokenizer
                    .encode(*text, true)
                    .map_err(anyhow::Error::msg)
                    .map(|encoding| CandlePreparedEncodeRequest {
                        input_ids: encoding.get_ids().to_vec(),
                        attention_mask: Some(encoding.get_attention_mask().to_vec()),
                        token_type_ids: Some(encoding.get_type_ids().to_vec()),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let prepared_encoded = model.encode_prepared(&prepared, "cls", true)?;
        assert_eq!(prepared_encoded.dim, encoded.dim);
        assert_eq!(prepared_encoded.embeddings.len(), encoded.embeddings.len());
        for (raw, prepared) in encoded
            .embeddings
            .iter()
            .zip(prepared_encoded.embeddings.iter())
        {
            assert!(
                cosine_similarity(raw, prepared) > 0.999,
                "prepared and raw packed GTE embeddings diverged"
            );
        }
        Ok(())
    }

    fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut left_norm = 0.0f32;
        let mut right_norm = 0.0f32;
        for (left, right) in left.iter().zip(right.iter()) {
            dot += left * right;
            left_norm += left * left;
            right_norm += right * right;
        }
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }
}
