//! IPC wire types for the Rust worker.
//!
//! This mirrors the subset of `sie_server_sidecar::protocol::ipc_types`
//! needed by a Rust adapter process. The response-chunk v1 envelope and
//! limits are pinned across the native worker, sidecar, and Python worker by
//! `tools/check_response_chunk_protocol.py`. The wire format is named-map
//! msgpack framed as `[4-byte BE length][msgpack body]`.

use half::f16;
use serde::{Deserialize, Serialize};

pub const IPC_VERSION: u32 = 1;

pub const METHOD_PING: &str = "Ping";
pub const METHOD_ENSURE_MODEL_READY: &str = "EnsureModelReady";
pub const METHOD_PROCESS_ENCODE_BATCH: &str = "ProcessEncodeBatch";
pub const METHOD_PROCESS_SCORE_BATCH: &str = "ProcessScoreBatch";
pub const METHOD_PROCESS_EXTRACT_BATCH: &str = "ProcessExtractBatch";
pub const METHOD_PROCESS_GENERATE: &str = "ProcessGenerate";
pub const METHOD_WORKER_CAPABILITIES: &str = "WorkerCapabilities";
pub const METHOD_SIGNAL_GENERATE_CANCEL: &str = "SignalGenerateCancel";
pub const METHOD_RUN_BATCH: &str = "RunBatch";
pub const METHOD_APPLY_MODEL_CONFIG: &str = "ApplyModelConfig";
pub const METHOD_REPLACE_MODEL_CONFIGS: &str = "ReplaceModelConfigs";
pub const METHOD_SET_PINNED_MODELS: &str = "SetPinnedModels";
pub const METHOD_DRAIN: &str = "Drain";

#[derive(Debug, Clone, Deserialize)]
pub struct RequestEnvelope {
    pub version: u32,
    pub method: String,
    pub request_id: String,
    /// Explicit v1-only negotiation for oversized non-streaming response
    /// chunks. Missing/false preserves the legacy one-frame response.
    #[serde(default)]
    pub accepts_ipc_response_chunks_v1: bool,
    #[serde(default)]
    pub body: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseEnvelope<B> {
    pub version: u32,
    pub request_id: String,
    pub ok: bool,
    #[serde(default)]
    pub body: Option<B>,
    #[serde(default)]
    pub error: Option<String>,
}

/// One negotiated chunk of an oversized serialized [`ResponseEnvelope`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpcResponseChunkV1 {
    pub version: u32,
    pub request_id: String,
    #[serde(with = "serde_bytes")]
    pub transfer_digest: Vec<u8>,
    pub chunk_index: u32,
    pub chunk_count: u32,
    pub total_bytes: u64,
    #[serde(with = "serde_bytes")]
    pub payload: Vec<u8>,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingRequest {
    pub timestamp_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PingResponse {
    #[serde(default)]
    pub timestamp_ms: f64,
    #[serde(default)]
    pub worker_id: String,
    #[serde(default)]
    pub ready: bool,
    #[serde(default)]
    pub bundle_config_hash: String,
    #[serde(default)]
    pub loaded_models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsureModelReadyRequest {
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessState {
    Ready,
    LoadingStarted,
    LoadingInProgress,
    RetryLater,
    /// Terminal, non-retryable load failure (registry recorded a
    /// permanent `LoadFailure`, `cooldown=permanent`). The sidecar
    /// dead-letters the group as `MODEL_LOAD_FAILED` instead of
    /// re-driving `EnsureModelReady` forever. Wire string: `"failed"`.
    Failed,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelDescriptor {
    #[serde(default)]
    pub tokenizer_path: Option<String>,
    #[serde(default)]
    pub tokenizer_id: Option<String>,
    #[serde(default)]
    pub max_seq_len: Option<u32>,
    #[serde(default)]
    pub output_types: Vec<String>,
    #[serde(default)]
    pub supports_run_batch: bool,
    #[serde(default)]
    pub default_query_template: Option<String>,
    #[serde(default)]
    pub default_doc_template: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsureModelReadyResponse {
    pub state: ReadinessState,
    #[serde(default)]
    pub batch_budget: Option<u32>,
    #[serde(default)]
    pub descriptor: Option<ModelDescriptor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyModelConfigRequest {
    pub bundle_id: String,
    pub model_id: String,
    pub epoch: u64,
    pub bundle_config_hash: String,
    #[serde(default)]
    pub profiles_added: Vec<String>,
    pub model_config: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyModelConfigResponse {
    pub applied: bool,
    pub bundle_config_hash: String,
    #[serde(default)]
    pub config_version: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaceModelConfigEntry {
    pub model_id: String,
    pub model_config: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaceModelConfigsRequest {
    pub bundle_id: String,
    pub epoch: u64,
    pub bundle_config_hash: String,
    pub models: Vec<ReplaceModelConfigEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplaceModelConfigsResponse {
    pub applied: bool,
    pub bundle_config_hash: String,
    #[serde(default)]
    pub config_version: u64,
    #[serde(default)]
    pub applied_models: Vec<String>,
    #[serde(default)]
    pub applied_profiles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetPinnedModelsRequest {
    #[serde(default)]
    pub models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetPinnedModelsResponse {
    pub applied: bool,
    #[serde(default)]
    pub pinned_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCapabilitiesRequest {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCapabilitiesResponse {
    #[serde(default)]
    pub has_generation_models: bool,
    #[serde(default)]
    pub generation_models: Vec<String>,
    #[serde(default)]
    pub supported_models: Vec<String>,
    #[serde(default)]
    pub loaded_models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalGenerateCancelRequest {
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalGenerateCancelResponse {
    #[serde(default)]
    pub matched: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedTokens {
    pub input_ids: Vec<Vec<u32>>,
    #[serde(default)]
    pub attention_mask: Vec<Vec<u32>>,
    #[serde(default)]
    pub token_type_ids: Vec<Vec<u32>>,
    pub tokenizer_id: String,
    #[serde(default)]
    pub max_seq_len: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeBatchItem {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    pub total_items: u32,
    pub timestamp: f64,
    pub item: serde_json::Value,
    #[serde(default)]
    pub output_types: Option<Vec<String>>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub is_query: bool,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub bundle_config_hash: Option<String>,
    #[serde(default)]
    pub payload_fetch_ms: f64,
    #[serde(default)]
    pub prepared_tokens: Option<PreparedTokens>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessEncodeBatchRequest {
    pub model_id: String,
    pub items: Vec<EncodeBatchItem>,
    /// Opt-in for the additive shared-buffer multivector response. Keeping
    /// this false by default preserves a safe per-item fallback while a
    /// worker and sidecar are rolled independently.
    #[serde(default)]
    pub accepts_batched_f16_multivectors: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBatchItem {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    pub total_items: u32,
    pub timestamp: f64,
    pub query_item: serde_json::Value,
    pub score_items: Vec<serde_json::Value>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(default)]
    pub profile_id: Option<String>,
    #[serde(default)]
    pub payload_fetch_ms: f64,
    /// Optional sidecar-prepared tokens ordered as query followed by score
    /// documents. The current sidecar leaves this unset for score requests,
    /// but accepting it keeps the native score path wire-compatible with the
    /// existing prepared-token contract.
    #[serde(default)]
    pub prepared_tokens: Option<PreparedTokens>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessScoreBatchRequest {
    pub model_id: String,
    pub items: Vec<ScoreBatchItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractBatchItem {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    pub total_items: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessExtractBatchRequest {
    pub model_id: String,
    pub items: Vec<ExtractBatchItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Disposition {
    PublishAndAck,
    PublishErrorAndAck,
    NakRetry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemOutcome {
    pub work_item_id: String,
    pub request_id: String,
    pub item_index: u32,
    pub disposition: Disposition,
    #[serde(default)]
    pub nak_delay_ms: Option<u64>,
    #[serde(default, serialize_with = "serde_bytes::serialize")]
    pub result_msgpack: Vec<u8>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub error_code: Option<String>,
    #[serde(default)]
    pub inference_ms: Option<f64>,
    #[serde(default)]
    pub tokenization_ms: Option<f64>,
    #[serde(default)]
    pub postprocessing_ms: Option<f64>,
    #[serde(default)]
    pub raw_output: Option<RawOutput>,
    /// Authoritative billable-unit counts produced by the inference engine.
    /// A field is present only when the engine has an exact count.
    #[serde(default)]
    pub units: Option<UnitCounts>,
}

/// Mirror of `sie_server_sidecar::protocol::ipc_types::UnitCounts`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitCounts {
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub pages: Option<u64>,
    #[serde(default)]
    pub images: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOutcome {
    pub outcomes: Vec<ItemOutcome>,
    /// Native Candle f16 multivectors can share one contiguous buffer across
    /// a batch. The sidecar owns slicing and final public wire framing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub batched_f16_multivectors: Vec<BatchedF16MultivectorOutput>,
}

/// Contiguous f16 values encoded as one MessagePack binary value.
///
/// Native Candle receives this buffer directly from its device-to-host read.
/// Keeping it typed until the IPC serializer removes the otherwise redundant
/// host-side f16-to-byte copy. The wire representation remains byte-identical
/// to Python's `bytes` field: little-endian IEEE-754 binary16 values.
#[derive(Debug, Clone, Default)]
pub struct F16Values(pub Vec<f16>);

impl Serialize for F16Values {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[cfg(target_endian = "little")]
        {
            // SAFETY: `f16` is transparent over `u16`; a `u8` view has no
            // alignment requirement and little-endian memory is the wire form.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    self.0.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(self.0.as_slice()),
                )
            };
            serializer.serialize_bytes(bytes)
        }

        #[cfg(target_endian = "big")]
        {
            let mut bytes = Vec::with_capacity(std::mem::size_of_val(self.0.as_slice()));
            for value in &self.0 {
                bytes.extend_from_slice(&value.to_bits().to_le_bytes());
            }
            serializer.serialize_bytes(&bytes)
        }
    }
}

impl<'de> Deserialize<'de> for F16Values {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct F16ValuesVisitor;

        impl<'de> serde::de::Visitor<'de> for F16ValuesVisitor {
            type Value = F16Values;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a little-endian f16 byte sequence")
            }

            fn visit_borrowed_bytes<E>(self, bytes: &'de [u8]) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                f16_values_from_le_bytes(bytes).map_err(E::custom)
            }

            fn visit_bytes<E>(self, bytes: &[u8]) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                f16_values_from_le_bytes(bytes).map_err(E::custom)
            }

            fn visit_byte_buf<E>(self, bytes: Vec<u8>) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                f16_values_from_le_bytes(&bytes).map_err(E::custom)
            }
        }

        deserializer.deserialize_bytes(F16ValuesVisitor)
    }
}

fn f16_values_from_le_bytes(bytes: &[u8]) -> Result<F16Values, &'static str> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<f16>()) {
        return Err("f16 byte buffer length must be divisible by two");
    }

    #[cfg(target_endian = "little")]
    {
        let mut values = Vec::<f16>::with_capacity(bytes.len() / std::mem::size_of::<f16>());
        // SAFETY: every f16 bit pattern is valid. The allocation retains f16
        // alignment, unlike reinterpreting a Vec<u8> as an owning Vec<f16>.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                values.as_mut_ptr().cast::<u8>(),
                bytes.len(),
            );
            values.set_len(bytes.len() / std::mem::size_of::<f16>());
        }
        Ok(F16Values(values))
    }

    #[cfg(target_endian = "big")]
    {
        Ok(F16Values(
            bytes
                .chunks_exact(std::mem::size_of::<f16>())
                .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect(),
        ))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedF16MultivectorOutput {
    pub values_f16: F16Values,
    pub items: Vec<BatchedF16MultivectorItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedF16MultivectorItem {
    pub work_item_id: String,
    pub byte_offset: u64,
    pub byte_len: u64,
    pub num_tokens: u32,
    pub token_dims: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RawOutput {
    #[serde(default)]
    pub dense: Option<DenseOutput>,
    #[serde(default)]
    pub score: Option<ScoreOutputRaw>,
    #[serde(default)]
    pub sparse: Option<SparseOutput>,
    #[serde(default)]
    pub multivector: Option<MultivectorOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseOutput {
    pub values: Vec<f32>,
    pub dim: u32,
    #[serde(default)]
    pub normalize: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreOutputRaw {
    pub scores: Vec<f32>,
    pub item_ids: Vec<String>,
}

/// Sparse `(indices, values)` payload for one encode item.
///
/// The v1 native wire contract uses i32 indices and f32 values. `dims` is
/// the full sparse vocabulary size when the model exposes one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseOutput {
    pub indices: Vec<i32>,
    pub values: Vec<f32>,
    #[serde(default)]
    pub dims: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultivectorOutput {
    pub values: Vec<f32>,
    #[serde(
        default,
        skip_serializing_if = "Vec::is_empty",
        serialize_with = "serde_bytes::serialize",
        deserialize_with = "serde_bytes::deserialize"
    )]
    pub values_f16: Vec<u8>,
    pub num_tokens: u32,
    pub token_dims: u32,
    #[serde(default)]
    pub dtype: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBatchItem {
    pub op: String,
    #[serde(default)]
    pub work_item_id: String,
    #[serde(default)]
    pub request_id: String,
    #[serde(default)]
    pub item_index: u32,
    #[serde(default)]
    pub encode: Option<EncodeBatchItem>,
    #[serde(default)]
    pub score: Option<ScoreBatchItem>,
    #[serde(default)]
    pub extract: Option<ExtractBatchItem>,
    #[serde(default)]
    pub traceparent: Option<String>,
    #[serde(default)]
    pub tracestate: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunBatchRequest {
    pub model_id: String,
    pub batch_id: u64,
    pub lora_key: String,
    pub total_cost: u64,
    pub items: Vec<RunBatchItem>,
    /// Same capability gate as [`ProcessEncodeBatchRequest`], carried by the
    /// scheduler's production RPC path.
    #[serde(default)]
    pub accepts_batched_f16_multivectors: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainRequest {
    #[serde(default)]
    pub deadline_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrainResponse {
    #[serde(default)]
    pub acknowledged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_values_roundtrip_as_little_endian_msgpack_binary() {
        let values = F16Values(vec![f16::from_bits(0x3c00), f16::from_bits(0x8001)]);

        let encoded = rmp_serde::to_vec_named(&values).unwrap();
        let wire: serde_bytes::ByteBuf = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(wire.as_ref(), &[0x00, 0x3c, 0x01, 0x80]);

        let decoded: F16Values = rmp_serde::from_slice(&encoded).unwrap();
        assert_eq!(
            decoded
                .0
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            vec![0x3c00, 0x8001]
        );
    }

    #[test]
    fn run_batch_item_trace_context_roundtrips_msgpack() {
        let item = RunBatchItem {
            op: "encode".to_string(),
            work_item_id: "r.0".to_string(),
            request_id: "r".to_string(),
            item_index: 0,
            encode: None,
            score: None,
            extract: None,
            traceparent: Some(
                "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01".to_string(),
            ),
            tracestate: Some("vendor=opaque".to_string()),
        };

        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let back: RunBatchItem = rmp_serde::from_slice(&bytes).unwrap();

        assert_eq!(
            back.traceparent.as_deref(),
            Some("00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"),
        );
        assert_eq!(back.tracestate.as_deref(), Some("vendor=opaque"));
    }

    #[test]
    fn score_prepared_tokens_roundtrip_msgpack() {
        let item = ScoreBatchItem {
            work_item_id: "r.0".to_string(),
            request_id: "r".to_string(),
            item_index: 0,
            total_items: 1,
            timestamp: 0.0,
            query_item: serde_json::json!({"text": "query"}),
            score_items: vec![serde_json::json!({"text": "document"})],
            instruction: None,
            options: None,
            profile_id: Some("candle".to_string()),
            payload_fetch_ms: 0.0,
            prepared_tokens: Some(PreparedTokens {
                input_ids: vec![vec![1, 2], vec![3, 4, 5]],
                attention_mask: vec![vec![1, 1], vec![1, 1, 1]],
                token_type_ids: Vec::new(),
                tokenizer_id: "tokenizer".to_string(),
                max_seq_len: 8192,
            }),
        };

        let bytes = rmp_serde::to_vec_named(&item).unwrap();
        let back: ScoreBatchItem = rmp_serde::from_slice(&bytes).unwrap();

        let prepared = back.prepared_tokens.expect("prepared score tokens");
        assert_eq!(prepared.input_ids, vec![vec![1, 2], vec![3, 4, 5]]);
        assert_eq!(prepared.tokenizer_id, "tokenizer");
    }

    #[test]
    fn unit_counts_roundtrip_as_named_msgpack_fields() {
        let units = UnitCounts {
            input_tokens: Some(42),
            pages: None,
            images: Some(2),
        };

        let bytes = rmp_serde::to_vec_named(&units).unwrap();
        let map: std::collections::BTreeMap<String, serde_json::Value> =
            rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(map["input_tokens"].as_u64(), Some(42));
        assert!(map["pages"].is_null());

        let back: UnitCounts = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(back, units);
    }

    #[test]
    fn sparse_raw_output_roundtrips_named_msgpack_fields() {
        let output = RawOutput {
            sparse: Some(SparseOutput {
                indices: vec![7, 29_522],
                values: vec![0.25, 3.5],
                dims: Some(30_522),
            }),
            ..RawOutput::default()
        };

        let bytes = rmp_serde::to_vec_named(&output).unwrap();
        let map: std::collections::BTreeMap<String, serde_json::Value> =
            rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(map["sparse"]["indices"], serde_json::json!([7, 29_522]));
        assert_eq!(map["sparse"]["dims"].as_u64(), Some(30_522));

        let back: RawOutput = rmp_serde::from_slice(&bytes).unwrap();
        let sparse = back.sparse.expect("sparse output");
        assert_eq!(sparse.indices, vec![7, 29_522]);
        assert_eq!(sparse.values, vec![0.25, 3.5]);
        assert_eq!(sparse.dims, Some(30_522));
    }

    #[test]
    fn raw_output_without_sparse_field_remains_compatible() {
        let old_wire = rmp_serde::to_vec_named(&serde_json::json!({
            "dense": null,
            "score": null,
            "multivector": null,
        }))
        .unwrap();

        let output: RawOutput = rmp_serde::from_slice(&old_wire).unwrap();
        assert!(output.sparse.is_none());
    }
}
