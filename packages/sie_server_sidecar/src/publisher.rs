//! Encode an `ItemOutcome` into a `WorkResult` and publish it to the
//! gateway's reply subject.

use std::borrow::Cow;

use async_nats::Client;
use half::f16;
use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;
use tracing::{debug, warn};

use crate::ipc_types::{Disposition, ItemOutcome, RawOutput};
use crate::observability::metrics::{
    ResultTransportMode, ResultTransportOutcome, SidecarTelemetry,
};
use crate::output::{
    build_dense_payload_with_item_id, build_multivector_payload_from_f16_bytes_with_item_id,
    build_multivector_payload_with_dtype_and_item_id, build_score_payload,
    build_sparse_payload_with_item_id, l2_normalize_in_place, ShapeError,
};
use crate::work_types::WorkResult;

#[derive(Debug, Error)]
pub enum PublishError {
    #[error("nats publish: {0}")]
    Nats(#[from] async_nats::PublishError),
    #[error("nats flush after chunked result: {0}")]
    Flush(#[from] async_nats::client::FlushError),
    #[error("encode WorkResult: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("empty reply_subject — cannot publish")]
    EmptyReplySubject,
    /// A NATS delivery reached a dispatcher wired without a
    /// [`WorkPublisher`] (local-ingest mode, P2.10). Unreachable by
    /// construction; mapped to the skip-ACK path so the item stays
    /// redeliverable if the invariant ever breaks.
    #[error("no NATS publisher configured (local-ingest mode)")]
    NoPublisher,
    #[error(
        "encoded fallback WorkResult ({size} bytes) exceeds negotiated NATS max_payload ({max_payload} bytes)"
    )]
    ResultPayloadTooLarge { size: usize, max_payload: usize },
    #[error(
        "encoded ResultChunkV1 ({size} bytes) exceeds negotiated NATS max_payload ({max_payload} bytes)"
    )]
    ResultChunkPayloadTooLarge { size: usize, max_payload: usize },
}

const RESULT_PAYLOAD_TOO_LARGE_ERROR_CODE: &str = "PAYLOAD_TOO_LARGE";
const RESULT_CHUNK_KIND: &str = "result_chunk_v1";
const RESULT_CHUNK_PAYLOAD_TARGET_BYTES: usize = 768 * 1024;
const MAX_CHUNKED_RESULT_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_CHUNKS: usize = 64;

#[derive(Debug)]
enum EncodedWorkResult {
    Single {
        request_id: String,
        work_item_id: String,
        bytes: Vec<u8>,
        oversized_bytes: Option<usize>,
    },
    Chunked(ChunkedWorkResult),
}

/// Serialize-only view of [`WorkResult`] for the NATS path.
///
/// The owned type remains the local-ingest and decode contract. This view
/// keeps the same named-map field order while borrowing the often multi-MiB
/// `result_msgpack`, avoiding a full clone immediately before serialization.
#[derive(Clone, Serialize)]
struct WorkResultRef<'a> {
    work_item_id: &'a str,
    request_id: &'a str,
    item_index: u32,
    success: bool,
    #[serde(with = "serde_bytes")]
    result_msgpack: &'a [u8],
    error: Option<Cow<'a, str>>,
    error_code: Option<&'a str>,
    inference_ms: Option<f64>,
    queue_ms: Option<f64>,
    processing_ms: Option<f64>,
    worker_id: Option<&'a str>,
    tokenization_ms: Option<f64>,
    postprocessing_ms: Option<f64>,
    payload_fetch_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    units: Option<&'a crate::ipc_types::UnitCounts>,
    worker_direct: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    executed_bundle_config_hash: Option<&'a str>,
}

impl<'a> WorkResultRef<'a> {
    fn from_outcome(
        outcome: &'a ItemOutcome,
        worker_id: &'a str,
        timings: Option<Timings>,
        worker_direct: bool,
        executed_bundle_config_hash: Option<&'a str>,
    ) -> Self {
        let (success, error, error_code) = match outcome.disposition {
            Disposition::PublishAndAck => (true, None, None),
            Disposition::PublishErrorAndAck => (
                false,
                outcome.error.as_deref().map(Cow::Borrowed),
                outcome.error_code.as_deref(),
            ),
            Disposition::NakRetry => (
                false,
                Some(match outcome.error.as_deref() {
                    Some(error) => Cow::Borrowed(error),
                    None => Cow::Borrowed("nak_retry reached publisher"),
                }),
                outcome.error_code.as_deref(),
            ),
        };
        let (queue_ms, processing_ms, payload_fetch_ms) = timing_fields(timings);
        Self {
            work_item_id: &outcome.work_item_id,
            request_id: &outcome.request_id,
            item_index: outcome.item_index,
            success,
            result_msgpack: &outcome.result_msgpack,
            error,
            error_code,
            inference_ms: outcome.inference_ms,
            queue_ms,
            processing_ms,
            worker_id: Some(worker_id),
            tokenization_ms: outcome.tokenization_ms,
            postprocessing_ms: outcome.postprocessing_ms,
            payload_fetch_ms,
            units: outcome.units.as_ref(),
            worker_direct,
            executed_bundle_config_hash: executed_bundle_config_hash
                .filter(|hash| !hash.is_empty()),
        }
    }

    fn into_owned(self) -> WorkResult {
        WorkResult {
            work_item_id: self.work_item_id.to_owned(),
            request_id: self.request_id.to_owned(),
            item_index: self.item_index,
            success: self.success,
            result_msgpack: self.result_msgpack.to_vec(),
            error: self.error.map(Cow::into_owned),
            error_code: self.error_code.map(str::to_owned),
            inference_ms: self.inference_ms,
            queue_ms: self.queue_ms,
            processing_ms: self.processing_ms,
            worker_id: self.worker_id.map(str::to_owned),
            tokenization_ms: self.tokenization_ms,
            postprocessing_ms: self.postprocessing_ms,
            payload_fetch_ms: self.payload_fetch_ms,
            units: self.units.cloned(),
            worker_direct: self.worker_direct,
            executed_bundle_config_hash: self.executed_bundle_config_hash.map(str::to_owned),
        }
    }

    #[cfg(test)]
    fn from_owned(result: &'a WorkResult) -> Self {
        Self {
            work_item_id: &result.work_item_id,
            request_id: &result.request_id,
            item_index: result.item_index,
            success: result.success,
            result_msgpack: &result.result_msgpack,
            error: result.error.as_deref().map(Cow::Borrowed),
            error_code: result.error_code.as_deref(),
            inference_ms: result.inference_ms,
            queue_ms: result.queue_ms,
            processing_ms: result.processing_ms,
            worker_id: result.worker_id.as_deref(),
            tokenization_ms: result.tokenization_ms,
            postprocessing_ms: result.postprocessing_ms,
            payload_fetch_ms: result.payload_fetch_ms,
            units: result.units.as_ref(),
            worker_direct: result.worker_direct,
            executed_bundle_config_hash: result.executed_bundle_config_hash.as_deref(),
        }
    }

    fn compact_payload_too_large(&self, oversized_bytes: usize, max_payload: usize) -> Self {
        let mut compact = self.clone();
        compact.success = false;
        compact.result_msgpack = &[];
        compact.error = Some(Cow::Owned(format!(
            "Encoded result exceeds the transport limit ({oversized_bytes} > {max_payload} bytes); reduce the input length or requested output size."
        )));
        compact.error_code = Some(RESULT_PAYLOAD_TOO_LARGE_ERROR_CODE);
        compact
    }
}

#[derive(Debug)]
struct ChunkedWorkResult {
    work_item_id: String,
    request_id: String,
    item_index: u32,
    transfer_digest: [u8; 32],
    result_bytes: Vec<u8>,
    chunk_count: u32,
}

impl ChunkedWorkResult {
    fn total_bytes(&self) -> usize {
        self.result_bytes.len()
    }

    fn chunk_bounds(&self, chunk_index: u32) -> (usize, usize) {
        let total = self.result_bytes.len();
        let count = self.chunk_count as usize;
        let index = chunk_index as usize;
        let start = total * index / count;
        let end = total * (index + 1) / count;
        (start, end)
    }

    fn encode_chunk(&self, chunk_index: u32) -> Result<Vec<u8>, PublishError> {
        let (start, end) = self.chunk_bounds(chunk_index);
        let chunk = ResultChunkV1Ref {
            kind: RESULT_CHUNK_KIND,
            work_item_id: &self.work_item_id,
            request_id: &self.request_id,
            item_index: self.item_index,
            transfer_digest: &self.transfer_digest,
            chunk_index,
            chunk_count: self.chunk_count,
            total_bytes: self.result_bytes.len() as u64,
            payload: &self.result_bytes[start..end],
        };
        Ok(rmp_serde::to_vec_named(&chunk)?)
    }
}

/// Serialize-only view of [`ResultChunkV1`] that writes borrowed payload bytes
/// directly into the physical NATS frame. The gateway still decodes the owned
/// wire type; exact-wire tests keep both representations identical.
#[derive(Serialize)]
struct ResultChunkV1Ref<'a> {
    kind: &'a str,
    work_item_id: &'a str,
    request_id: &'a str,
    item_index: u32,
    #[serde(with = "serde_bytes")]
    transfer_digest: &'a [u8],
    chunk_index: u32,
    chunk_count: u32,
    total_bytes: u64,
    #[serde(with = "serde_bytes")]
    payload: &'a [u8],
}

/// Per-item timings stamped onto `WorkResult` by the worker.
///
/// * `queue_ms` — wall-clock from gateway publish (`WorkItem.timestamp`)
///   to worker result publish.
/// * `payload_fetch_ms` — time spent resolving `payload_ref` /
///   `query_payload_ref` against the payload store; `0.0` for inline items.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    pub queue_ms: f64,
    pub payload_fetch_ms: f64,
}

/// Delivery metadata kept separate from the backend outcome so adding a
/// negotiated wire capability does not turn `publish_result` into an
/// ever-growing positional argument list.
#[derive(Debug, Clone, Copy)]
pub struct PublishResultContext<'a> {
    pub caller_item_id: Option<&'a str>,
    pub timings: Option<Timings>,
    pub worker_direct: bool,
    pub executed_bundle_config_hash: Option<&'a str>,
    pub accepts_result_chunks: bool,
}

pub struct WorkPublisher {
    client: Client,
    worker_id: String,
    telemetry: SidecarTelemetry,
}

impl WorkPublisher {
    pub fn new(client: Client, worker_id: impl Into<String>, telemetry: SidecarTelemetry) -> Self {
        Self {
            client,
            worker_id: worker_id.into(),
            telemetry,
        }
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    /// Publish a `WorkResult` built from `outcome` + worker-side `timings`.
    /// Returns `PublishError::EmptyReplySubject` on empty subject so the
    /// caller can treat the publish (and ACK) as fire-and-forget.
    pub async fn publish_result(
        &self,
        reply_subject: &str,
        outcome: &ItemOutcome,
        context: PublishResultContext<'_>,
    ) -> Result<(), PublishError> {
        check_reply_subject(reply_subject)?;

        let max_payload = self.client.max_payload();
        let encoded = {
            // Keep a shaped payload alive only while it is borrowed for
            // serialization/planning. The encoded delivery owns everything
            // needed across the awaited NATS publishes below.
            let shaped;
            let effective = match shape_raw_output_for_wire(outcome, context.caller_item_id) {
                ShapeOutcome::Unchanged => outcome,
                ShapeOutcome::Shaped(value) => {
                    shaped = value;
                    &shaped
                }
            };
            let result = WorkResultRef::from_outcome(
                effective,
                &self.worker_id,
                context.timings,
                context.worker_direct,
                context.executed_bundle_config_hash,
            );
            match encode_work_result_delivery(result, max_payload, context.accepts_result_chunks) {
                Ok(encoded) => encoded,
                Err(error) => {
                    self.telemetry.result_transport_completed(
                        ResultTransportMode::Rejected,
                        ResultTransportOutcome::PlanningError,
                    );
                    return Err(error);
                }
            }
        };
        match encoded {
            EncodedWorkResult::Single {
                request_id,
                work_item_id,
                bytes,
                oversized_bytes,
            } => {
                if let Some(oversized_bytes) = oversized_bytes {
                    warn!(
                        request_id = %request_id,
                        work_item_id = %work_item_id,
                        oversized_bytes,
                        max_payload,
                        accepts_result_chunks = context.accepts_result_chunks,
                        "WorkResult could not use chunk transport; publishing typed compact error"
                    );
                }
                debug!(
                    reply = %reply_subject,
                    request_id = %request_id,
                    bytes = bytes.len(),
                    "publishing WorkResult"
                );
                let mode = if oversized_bytes.is_some() {
                    ResultTransportMode::CompactError
                } else {
                    ResultTransportMode::Single
                };
                match self
                    .client
                    .publish(reply_subject.to_string(), bytes.into())
                    .await
                {
                    Ok(()) => self
                        .telemetry
                        .result_transport_completed(mode, ResultTransportOutcome::Published),
                    Err(error) => {
                        self.telemetry
                            .result_transport_completed(mode, ResultTransportOutcome::PublishError);
                        return Err(error.into());
                    }
                }
            }
            EncodedWorkResult::Chunked(chunked) => {
                debug!(
                    reply = %reply_subject,
                    request_id = %chunked.request_id,
                    total_bytes = chunked.total_bytes(),
                    chunk_count = chunked.chunk_count,
                    "publishing chunked WorkResult"
                );
                for chunk_index in 0..chunked.chunk_count {
                    let bytes = match chunked.encode_chunk(chunk_index) {
                        Ok(bytes) => bytes,
                        Err(error) => {
                            self.telemetry.result_transport_completed(
                                ResultTransportMode::Chunked,
                                ResultTransportOutcome::PlanningError,
                            );
                            return Err(error);
                        }
                    };
                    if bytes.len() > max_payload {
                        self.telemetry.result_transport_completed(
                            ResultTransportMode::Chunked,
                            ResultTransportOutcome::PlanningError,
                        );
                        return Err(PublishError::ResultChunkPayloadTooLarge {
                            size: bytes.len(),
                            max_payload,
                        });
                    }
                    // Publish in ascending index order and await each call. A
                    // partial failure is returned to the dispatcher, which
                    // deliberately leaves the JetStream item unacked so the
                    // whole transfer is retried.
                    let envelope_bytes = bytes.len();
                    match self
                        .client
                        .publish(reply_subject.to_string(), bytes.into())
                        .await
                    {
                        Ok(()) => {
                            self.telemetry.result_chunk_published(envelope_bytes);
                        }
                        Err(error) => {
                            self.telemetry.result_transport_completed(
                                ResultTransportMode::Chunked,
                                ResultTransportOutcome::PublishError,
                            );
                            return Err(error.into());
                        }
                    }
                }
                // ``Client::publish`` queues each Core NATS message in the
                // async client. One flush after the complete ordered sequence
                // is the broker barrier that makes it safe for the dispatcher
                // to ACK the originating JetStream work item. A failed flush
                // propagates, leaving that work item unacked for redelivery.
                if let Err(error) = self.client.flush().await {
                    self.telemetry.result_transport_completed(
                        ResultTransportMode::Chunked,
                        ResultTransportOutcome::FlushError,
                    );
                    return Err(error.into());
                }
                self.telemetry.result_transport_completed(
                    ResultTransportMode::Chunked,
                    ResultTransportOutcome::Published,
                );
            }
        }
        Ok(())
    }

    /// Publish a pre-encoded generation chunk produced by Python's
    /// `StreamingProcessor`. The sidecar owns NATS I/O; Python owns the
    /// chunk envelope bytes.
    pub async fn publish_raw(
        &self,
        reply_subject: &str,
        payload: Vec<u8>,
    ) -> Result<(), PublishError> {
        check_reply_subject(reply_subject)?;
        self.client
            .publish(reply_subject.to_string(), payload.into())
            .await?;
        Ok(())
    }
}

/// Select the rolling-compatible NATS result transport.
///
/// A normal `WorkResult` is always left byte-identical. Only an oversized
/// result from a gateway that explicitly advertised chunk support may become
/// `ResultChunkV1` messages. If the bounded chunk plan cannot fit, this falls
/// back to the same compact typed error used by legacy gateways.
fn encode_work_result_delivery(
    result: WorkResultRef<'_>,
    max_payload: usize,
    accepts_result_chunks: bool,
) -> Result<EncodedWorkResult, PublishError> {
    let bytes = rmp_serde::to_vec_named(&result)?;
    if bytes.len() <= max_payload {
        return Ok(EncodedWorkResult::Single {
            request_id: result.request_id.to_string(),
            work_item_id: result.work_item_id.to_string(),
            bytes,
            oversized_bytes: None,
        });
    }

    let oversized_bytes = bytes.len();
    if accepts_result_chunks && oversized_bytes <= MAX_CHUNKED_RESULT_BYTES {
        if let Some(chunked) = plan_chunked_work_result(&result, bytes, max_payload)? {
            return Ok(EncodedWorkResult::Chunked(chunked));
        }
    }

    let bytes = compact_oversized_work_result(&result, oversized_bytes, max_payload)?;
    Ok(EncodedWorkResult::Single {
        request_id: result.request_id.to_string(),
        work_item_id: result.work_item_id.to_string(),
        bytes,
        oversized_bytes: Some(oversized_bytes),
    })
}

fn compact_oversized_work_result(
    result: &WorkResultRef<'_>,
    oversized_bytes: usize,
    max_payload: usize,
) -> Result<Vec<u8>, PublishError> {
    let compact = result.compact_payload_too_large(oversized_bytes, max_payload);
    let fallback = rmp_serde::to_vec_named(&compact)?;
    if fallback.len() > max_payload {
        return Err(PublishError::ResultPayloadTooLarge {
            size: fallback.len(),
            max_payload,
        });
    }
    Ok(fallback)
}

/// Find the smallest bounded chunk count whose largest fully serialized
/// envelope fits the negotiated NATS ceiling. Balanced ranges guarantee
/// exactly `chunk_count` non-empty payloads and keep every payload at or below
/// the 768 KiB target.
fn plan_chunked_work_result(
    result: &WorkResultRef<'_>,
    result_bytes: Vec<u8>,
    max_payload: usize,
) -> Result<Option<ChunkedWorkResult>, PublishError> {
    if result_bytes.is_empty() || result_bytes.len() > MAX_CHUNKED_RESULT_BYTES {
        return Ok(None);
    }

    let transfer_digest: [u8; 32] = Sha256::digest(&result_bytes).into();
    let min_chunks = result_bytes
        .len()
        .div_ceil(RESULT_CHUNK_PAYLOAD_TARGET_BYTES)
        .max(1);
    let max_chunks = MAX_RESULT_CHUNKS.min(result_bytes.len());

    for chunk_count in min_chunks..=max_chunks {
        let largest_payload_bytes = result_bytes.len().div_ceil(chunk_count);
        let probe = ResultChunkV1Ref {
            kind: RESULT_CHUNK_KIND,
            work_item_id: result.work_item_id,
            request_id: result.request_id,
            item_index: result.item_index,
            transfer_digest: &transfer_digest,
            chunk_index: (chunk_count - 1) as u32,
            chunk_count: chunk_count as u32,
            total_bytes: result_bytes.len() as u64,
            payload: &result_bytes[..largest_payload_bytes],
        };
        if rmp_serde::to_vec_named(&probe)?.len() <= max_payload {
            return Ok(Some(ChunkedWorkResult {
                work_item_id: result.work_item_id.to_string(),
                request_id: result.request_id.to_string(),
                item_index: result.item_index,
                transfer_digest,
                result_bytes,
                chunk_count: chunk_count as u32,
            }));
        }
    }

    Ok(None)
}

/// Result of inspecting an `ItemOutcome.raw_output` for Rust-side
/// shaping: either pass through unchanged, or emit a replacement
/// outcome with `result_msgpack` filled in (success or error).
///
/// `Shaped` carries a full `ItemOutcome` (~360 B) so the enum is
/// variant-heavy. Boxing the inner outcome to shrink the enum is a
/// mechanical refactor tracked for its own cleanup cycle — every
/// consumer site would have to thread a `Box` through, which is noise
/// for this hot path. Scoped `#[allow]` here rather than a
/// crate-wide one so the warning re-fires if anyone else builds a
/// heavyweight enum elsewhere.
#[allow(clippy::large_enum_variant)]
enum ShapeOutcome {
    Unchanged,
    Shaped(ItemOutcome),
}

/// Attempt to shape `outcome.raw_output` into `outcome.result_msgpack`
/// bytes using the Rust-side output shapers.
///
/// Rules:
///
/// * `raw_output == None` → `Unchanged`. Legacy Python path.
/// * `raw_output == Some(_)` but `result_msgpack` is non-empty →
///   `Unchanged`. Python gave us pre-framed bytes; trust them. The
///   Python side gates `raw_output` emission behind an env var, so
///   a mixed outcome is a wire artefact of a rolling deploy and
///   we take the safer, already-framed path.
/// * `raw_output == Some(_)` with empty `result_msgpack` and a
///   recognised variant → run the shaper; on success emit
///   `PublishAndAck` with the produced bytes, on failure emit
///   `PublishErrorAndAck` with the error message. Never silently
///   sinks the work item.
/// * `raw_output == Some(_)` with no recognised variant populated
///   (e.g. a future Rust build sees a variant it doesn't know) →
///   emit `PublishErrorAndAck` so we don't hand back an empty
///   `result_msgpack` that the SDK would misinterpret as "success
///   but no embeddings".
fn shape_raw_output_for_wire(outcome: &ItemOutcome, caller_item_id: Option<&str>) -> ShapeOutcome {
    let Some(raw) = outcome.raw_output.as_ref() else {
        return ShapeOutcome::Unchanged;
    };
    if !outcome.result_msgpack.is_empty() {
        return ShapeOutcome::Unchanged;
    }
    // Only shape on success outcomes. Error dispositions carry no
    // payload anyway, so fall through to legacy handling.
    if !matches!(outcome.disposition, Disposition::PublishAndAck) {
        return ShapeOutcome::Unchanged;
    }

    match shape(raw, caller_item_id) {
        Ok(bytes) => ShapeOutcome::Shaped(ItemOutcome {
            result_msgpack: bytes,
            // Drop the raw_output so we never pay the serialisation
            // cost for the inner arrays again in `rmp_serde::to_vec`.
            raw_output: None,
            ..outcome.clone()
        }),
        Err(e) => {
            tracing::warn!(
                work_item_id = %outcome.work_item_id,
                request_id = %outcome.request_id,
                error = %e,
                "raw_output shape failed; downgrading to error WorkResult"
            );
            ShapeOutcome::Shaped(ItemOutcome {
                disposition: Disposition::PublishErrorAndAck,
                result_msgpack: Vec::new(),
                error: Some(format!("raw_output shape: {e}")),
                error_code: Some("raw_output_shape_error".into()),
                raw_output: None,
                ..outcome.clone()
            })
        }
    }
}

/// Dispatch a [`RawOutput`] to the appropriate shaper. Returns a
/// `RawOutputUnknown` error if no recognised variant is populated;
/// callers turn that into a `PublishErrorAndAck` so unknown future
/// variants surface as a typed error rather than an empty success.
fn shape(raw: &RawOutput, caller_item_id: Option<&str>) -> Result<Vec<u8>, ShapeError> {
    // Each outcome carries exactly one variant. Dispatch order is
    // fixed so a test that accidentally populates two variants gets
    // a deterministic result: dense → score → sparse → multivector.
    // A legitimate multi-variant encode item (e.g. dense + sparse
    // for the same text) is framed via the Python-framed fallback path —
    // the Python gate refuses to mint a multi-variant `RawOutput`.
    if let Some(dense) = raw.dense.as_ref() {
        let mut values = dense.values.clone();
        if dense.normalize {
            l2_normalize_in_place(&mut values);
        }
        return build_dense_payload_with_item_id(&values, dense.dim as usize, caller_item_id);
    }
    if let Some(score) = raw.score.as_ref() {
        return build_score_payload(&score.scores, &score.item_ids);
    }
    if let Some(sparse) = raw.sparse.as_ref() {
        return build_sparse_payload_with_item_id(
            &sparse.indices,
            &sparse.values,
            sparse.dims,
            caller_item_id,
        );
    }
    if let Some(mv) = raw.multivector.as_ref() {
        let dtype = mv.dtype.as_deref().unwrap_or("float32");
        if !mv.values_f16.is_empty() {
            if dtype != "float16" {
                return Err(ShapeError::MsgpackWrite(format!(
                    "multivector values_f16 requires dtype=\"float16\", got {dtype:?}"
                )));
            }
            return build_multivector_payload_from_f16_bytes_with_item_id(
                &mv.values_f16,
                mv.num_tokens,
                mv.token_dims,
                caller_item_id,
            );
        }
        return build_multivector_payload_with_dtype_and_item_id(
            &mv.values,
            mv.num_tokens,
            mv.token_dims,
            dtype,
            caller_item_id,
        );
    }
    Err(ShapeError::MsgpackWrite(
        "raw_output had no recognised variant populated (forward-compat placeholder?)".into(),
    ))
}

/// Shape `raw_output` (if any) into `result_msgpack` bytes, then build the
/// gateway-facing `WorkResult`. This is the full outcome→wire conversion
/// [`WorkPublisher::publish_result`] performs before its NATS publish;
/// exposed as a pure function so the local-ingest path (P2.10, §4.6) —
/// which returns `WorkResult`s over a socket instead of publishing them —
/// produces byte-identical results.
pub fn shape_and_build_work_result(
    outcome: &ItemOutcome,
    caller_item_id: Option<&str>,
    worker_id: &str,
    timings: Option<Timings>,
    worker_direct: bool,
    executed_bundle_config_hash: Option<&str>,
) -> WorkResult {
    // If Python emitted `raw_output` (deferring wire framing to Rust),
    // shape it into `result_msgpack` here. Any shape error becomes an
    // error `WorkResult` so a misconfigured model cannot silently drop
    // a request.
    let owned;
    let effective = match shape_raw_output_for_wire(outcome, caller_item_id) {
        ShapeOutcome::Unchanged => outcome,
        ShapeOutcome::Shaped(o) => {
            owned = o;
            &owned
        }
    };
    build_work_result(
        effective,
        worker_id,
        timings,
        worker_direct,
        executed_bundle_config_hash,
    )
}

/// Frame one slice of a batch-level f16 multivector buffer. The batch protocol
/// is internal to the worker/sidecar boundary; this produces the identical
/// public payload as [`shape_raw_output_for_wire`] without first copying the
/// slice into an `ItemOutcome::raw_output` allocation.
pub fn shape_batched_f16_multivector_outcome(
    outcome: &ItemOutcome,
    values_f16: &[f16],
    num_tokens: u32,
    token_dims: u32,
    caller_item_id: Option<&str>,
) -> ItemOutcome {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: `f16` is transparent over `u16`; a `u8` view has no
        // alignment requirement and little-endian memory is the wire form.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                values_f16.as_ptr().cast::<u8>(),
                std::mem::size_of_val(values_f16),
            )
        };
        shape_batched_f16_multivector_bytes(outcome, bytes, num_tokens, token_dims, caller_item_id)
    }

    #[cfg(target_endian = "big")]
    {
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values_f16));
        for value in values_f16 {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        shape_batched_f16_multivector_bytes(outcome, &bytes, num_tokens, token_dims, caller_item_id)
    }
}

fn shape_batched_f16_multivector_bytes(
    outcome: &ItemOutcome,
    values_f16: &[u8],
    num_tokens: u32,
    token_dims: u32,
    caller_item_id: Option<&str>,
) -> ItemOutcome {
    if !outcome.result_msgpack.is_empty()
        || !matches!(outcome.disposition, Disposition::PublishAndAck)
    {
        return outcome.clone();
    }

    match build_multivector_payload_from_f16_bytes_with_item_id(
        values_f16,
        num_tokens,
        token_dims,
        caller_item_id,
    ) {
        Ok(result_msgpack) => ItemOutcome {
            result_msgpack,
            raw_output: None,
            ..outcome.clone()
        },
        Err(error) => {
            tracing::warn!(
                work_item_id = %outcome.work_item_id,
                request_id = %outcome.request_id,
                error = %error,
                "batched f16 multivector shape failed; downgrading to error WorkResult"
            );
            ItemOutcome {
                disposition: Disposition::PublishErrorAndAck,
                result_msgpack: Vec::new(),
                error: Some(format!("batched f16 multivector shape: {error}")),
                error_code: Some("raw_output_shape_error".into()),
                raw_output: None,
                ..outcome.clone()
            }
        }
    }
}

/// Convert an `ItemOutcome` into a `WorkResult` (gateway-facing wire
/// type). `NakRetry` should never reach this path — the caller filters.
/// `timings.is_some()` → queue / processing / payload_fetch are stamped;
/// error-only paths pass `None` and produce a bare error result.
pub fn build_work_result(
    outcome: &ItemOutcome,
    worker_id: &str,
    timings: Option<Timings>,
    worker_direct: bool,
    executed_bundle_config_hash: Option<&str>,
) -> WorkResult {
    WorkResultRef::from_outcome(
        outcome,
        worker_id,
        timings,
        worker_direct,
        executed_bundle_config_hash,
    )
    .into_owned()
}

/// Timing fields are stamped only on the success/publish path. Error results
/// omit them entirely. `processing_ms = 0.0` is a placeholder the gateway
/// ignores when `inference_ms` is set.
fn timing_fields(timings: Option<Timings>) -> (Option<f64>, Option<f64>, Option<f64>) {
    match timings {
        Some(t) => (
            Some(t.queue_ms),
            Some(0.0),
            (t.payload_fetch_ms > 0.0).then_some(t.payload_fetch_ms),
        ),
        None => (None, None, None),
    }
}

/// Decide whether a disposition should be published at all.
pub fn should_publish(disposition: &Disposition) -> bool {
    matches!(
        disposition,
        Disposition::PublishAndAck | Disposition::PublishErrorAndAck
    )
}

/// Reject empty / whitespace-only reply subjects so we never publish to
/// an empty NATS subject (which the client would silently drop).
pub(crate) fn check_reply_subject(subject: &str) -> Result<(), PublishError> {
    if subject.trim().is_empty() {
        return Err(PublishError::EmptyReplySubject);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc_types::{
        DenseOutput, MultivectorOutput, ScoreOutputRaw, SparseOutput, UnitCounts,
    };
    use crate::output::{
        build_dense_payload, build_multivector_payload, build_multivector_payload_from_f16_bytes,
        build_multivector_payload_with_dtype, build_sparse_payload,
    };
    use crate::work_types::ResultChunkV1;

    fn outcome() -> ItemOutcome {
        ItemOutcome {
            work_item_id: "req-1.0".into(),
            request_id: "req-1".into(),
            item_index: 0,
            disposition: Disposition::PublishAndAck,
            nak_delay_ms: None,
            result_msgpack: vec![0x81, 0xa5, b'h', b'e', b'l', b'l', b'o', 0x05],
            error: None,
            error_code: None,
            inference_ms: Some(17.5),
            tokenization_ms: Some(1.1),
            postprocessing_ms: Some(0.3),
            raw_output: None,
            units: None,
        }
    }

    fn single_delivery(
        result: WorkResult,
        max_payload: usize,
        accepts_result_chunks: bool,
    ) -> (WorkResult, Vec<u8>, Option<usize>) {
        let view = WorkResultRef::from_owned(&result);
        match encode_work_result_delivery(view, max_payload, accepts_result_chunks).unwrap() {
            EncodedWorkResult::Single {
                bytes,
                oversized_bytes,
                ..
            } => {
                let decoded = rmp_serde::from_slice(&bytes).unwrap();
                (decoded, bytes, oversized_bytes)
            }
            EncodedWorkResult::Chunked(_) => panic!("expected single-message result transport"),
        }
    }

    #[test]
    fn success_outcome_maps_to_success_result() {
        let r = build_work_result(&outcome(), "worker-1", None, true, Some("hash-a"));
        assert!(r.success);
        assert!(r.error.is_none());
        assert_eq!(r.request_id, "req-1");
        assert_eq!(r.item_index, 0);
        assert_eq!(r.worker_id.as_deref(), Some("worker-1"));
        assert!(r.worker_direct);
        assert_eq!(r.inference_ms, Some(17.5));
        assert_eq!(r.tokenization_ms, Some(1.1));
        assert_eq!(r.postprocessing_ms, Some(0.3));
        assert_eq!(r.result_msgpack.len(), 8);
        assert_eq!(r.executed_bundle_config_hash.as_deref(), Some("hash-a"));
    }

    #[test]
    fn publish_error_outcome_maps_to_failure_result() {
        let mut o = outcome();
        o.disposition = Disposition::PublishErrorAndAck;
        o.error = Some("kapow".into());
        o.error_code = Some("INTERNAL".into());
        o.units = Some(UnitCounts {
            input_tokens: None,
            pairs: None,
            pages: Some(3),
            images: None,
            audio_ms: None,
        });
        let r = build_work_result(&o, "w", None, false, None);
        assert!(!r.success);
        assert_eq!(r.error.as_deref(), Some("kapow"));
        assert_eq!(r.error_code.as_deref(), Some("INTERNAL"));
        assert_eq!(r.units.and_then(|units| units.pages), Some(3));
    }

    #[test]
    fn should_publish_rejects_nak_retry() {
        assert!(should_publish(&Disposition::PublishAndAck));
        assert!(should_publish(&Disposition::PublishErrorAndAck));
        assert!(!should_publish(&Disposition::NakRetry));
    }

    #[test]
    fn work_result_is_msgpack_round_trippable() {
        let mut outcome = outcome();
        outcome.units = Some(UnitCounts {
            input_tokens: None,
            pairs: None,
            pages: None,
            images: None,
            audio_ms: Some(1_001),
        });
        let r = build_work_result(&outcome, "w", None, false, None);
        let bytes = rmp_serde::to_vec_named(&r).unwrap();
        let back: WorkResult = rmp_serde::from_slice(&bytes).unwrap();
        assert!(back.success);
        assert_eq!(back.request_id, "req-1");
        assert_eq!(back.result_msgpack.len(), 8);
        assert_eq!(back.units.and_then(|units| units.audio_ms), Some(1_001));
    }

    #[test]
    fn borrowed_work_result_encoding_matches_owned_wire_at_bin_boundaries() {
        for payload_len in [255, 256, 65_535, 65_536] {
            let mut value = build_work_result(&outcome(), "w", None, false, Some("hash"));
            value.result_msgpack = vec![7; payload_len];

            assert_eq!(
                rmp_serde::to_vec_named(&WorkResultRef::from_owned(&value)).unwrap(),
                rmp_serde::to_vec_named(&value).unwrap(),
                "payload length {payload_len}"
            );
        }
    }

    #[test]
    fn borrowed_result_chunk_encoding_matches_owned_wire_at_bin_boundaries() {
        for payload_len in [255, 256, 65_535, 65_536] {
            let payload = vec![7; payload_len];
            let digest = [9_u8; 32];
            let borrowed = ResultChunkV1Ref {
                kind: RESULT_CHUNK_KIND,
                work_item_id: "req.0",
                request_id: "req",
                item_index: 0,
                transfer_digest: &digest,
                chunk_index: 0,
                chunk_count: 1,
                total_bytes: payload_len as u64,
                payload: &payload,
            };
            let borrowed_bytes = rmp_serde::to_vec_named(&borrowed).unwrap();
            let owned = ResultChunkV1 {
                kind: RESULT_CHUNK_KIND.to_string(),
                work_item_id: "req.0".to_string(),
                request_id: "req".to_string(),
                item_index: 0,
                transfer_digest: digest.to_vec(),
                chunk_index: 0,
                chunk_count: 1,
                total_bytes: payload_len as u64,
                payload,
            };

            assert_eq!(
                borrowed_bytes,
                rmp_serde::to_vec_named(&owned).unwrap(),
                "payload length {payload_len}"
            );
        }
    }

    #[test]
    fn result_at_exact_nats_limit_remains_successful() {
        let result = build_work_result(&outcome(), "w", None, false, None);
        let expected = rmp_serde::to_vec_named(&result).unwrap();
        let exact = expected.len();

        let (result, bytes, oversized) = single_delivery(result, exact, true);

        assert!(result.success);
        assert_eq!(bytes, expected);
        assert!(oversized.is_none());
    }

    #[test]
    fn oversized_nats_result_becomes_compact_typed_error() {
        let mut o = outcome();
        o.result_msgpack = vec![7; 2_048];
        let result = build_work_result(&o, "w", None, false, Some("hash-a"));

        let (result, bytes, oversized) = single_delivery(result, 512, false);

        assert!(!result.success);
        assert!(result.result_msgpack.is_empty());
        assert_eq!(
            result.error_code.as_deref(),
            Some(RESULT_PAYLOAD_TOO_LARGE_ERROR_CODE)
        );
        assert!(oversized.is_some_and(|size| size > 512));
        assert!(bytes.len() <= 512);
        let decoded: WorkResult = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(decoded.error_code.as_deref(), Some("PAYLOAD_TOO_LARGE"));
    }

    #[test]
    fn oversized_caller_item_id_is_removed_from_compact_error() {
        let mut o = outcome();
        o.result_msgpack.clear();
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![1.0],
                dim: 1,
                normalize: false,
            }),
            ..Default::default()
        });
        let caller_item_id = "private-caller-id-".repeat(256);
        let result = shape_and_build_work_result(&o, Some(&caller_item_id), "w", None, false, None);

        let (result, bytes, oversized) = single_delivery(result, 512, false);

        assert!(!result.success);
        assert_eq!(result.error_code.as_deref(), Some("PAYLOAD_TOO_LARGE"));
        assert!(oversized.is_some());
        assert!(!bytes
            .windows("private-caller-id".len())
            .any(|window| window == b"private-caller-id"));
    }

    #[test]
    fn impossibly_small_nats_limit_fails_before_publish() {
        let mut o = outcome();
        o.result_msgpack = vec![7; 2_048];
        let result = build_work_result(&o, "w", None, false, None);

        assert!(matches!(
            encode_work_result_delivery(WorkResultRef::from_owned(&result), 1, true),
            Err(PublishError::ResultPayloadTooLarge { max_payload: 1, .. })
        ));
    }

    #[test]
    fn negotiated_oversized_result_is_chunked_and_reassembles_exactly() {
        let mut o = outcome();
        // Representative of an 8K-token, 128-dimensional f16 multivector
        // result once its public result payload has been framed.
        o.result_msgpack = (0..2_200_000).map(|index| (index % 251) as u8).collect();
        let result = build_work_result(&o, "worker-8k", None, false, Some("hash-8k"));
        let expected = rmp_serde::to_vec_named(&result).unwrap();

        let EncodedWorkResult::Chunked(chunked) =
            encode_work_result_delivery(WorkResultRef::from_owned(&result), 1024 * 1024, true)
                .unwrap()
        else {
            panic!("negotiated oversized result should use chunks");
        };

        assert_eq!(chunked.chunk_count, 3);
        assert_eq!(chunked.total_bytes(), expected.len());
        assert_eq!(chunked.transfer_digest.len(), 32);
        let expected_digest: [u8; 32] = Sha256::digest(&expected).into();
        assert_eq!(chunked.transfer_digest, expected_digest);

        let mut reassembled = Vec::with_capacity(expected.len());
        for chunk_index in 0..chunked.chunk_count {
            let bytes = chunked.encode_chunk(chunk_index).unwrap();
            assert!(bytes.len() <= 1024 * 1024);
            let envelope: ResultChunkV1 = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(envelope.kind, RESULT_CHUNK_KIND);
            assert_eq!(envelope.work_item_id, "req-1.0");
            assert_eq!(envelope.request_id, "req-1");
            assert_eq!(envelope.item_index, 0);
            assert_eq!(envelope.chunk_index, chunk_index);
            assert_eq!(envelope.chunk_count, chunked.chunk_count);
            assert_eq!(envelope.total_bytes as usize, expected.len());
            assert_eq!(envelope.transfer_digest, chunked.transfer_digest);
            assert!(envelope.payload.len() <= RESULT_CHUNK_PAYLOAD_TARGET_BYTES);
            reassembled.extend_from_slice(&envelope.payload);
        }

        assert_eq!(reassembled, expected);
        assert_eq!(
            Sha256::digest(&reassembled).to_vec(),
            chunked.transfer_digest
        );
        let decoded: WorkResult = rmp_serde::from_slice(&reassembled).unwrap();
        assert!(decoded.success);
        assert_eq!(decoded.result_msgpack, o.result_msgpack);
        assert_eq!(
            decoded.executed_bundle_config_hash.as_deref(),
            Some("hash-8k")
        );
    }

    #[test]
    fn legacy_gateway_keeps_compact_error_instead_of_chunks() {
        let mut o = outcome();
        o.result_msgpack = vec![9; 2 * 1024 * 1024];
        let result = build_work_result(&o, "w", None, false, None);

        let (result, bytes, oversized) = single_delivery(result, 1024 * 1024, false);

        assert!(!result.success);
        assert_eq!(result.error_code.as_deref(), Some("PAYLOAD_TOO_LARGE"));
        assert!(oversized.is_some_and(|size| size > 1024 * 1024));
        assert!(bytes.len() <= 1024 * 1024);
    }

    #[test]
    fn negotiated_result_above_total_bound_becomes_compact_error() {
        let mut o = outcome();
        o.result_msgpack = vec![3; MAX_CHUNKED_RESULT_BYTES + 1];
        let result = build_work_result(&o, "w", None, false, None);

        let (result, bytes, oversized) = single_delivery(result, 1024 * 1024, true);

        assert!(!result.success);
        assert_eq!(result.error_code.as_deref(), Some("PAYLOAD_TOO_LARGE"));
        assert!(oversized.is_some_and(|size| size > MAX_CHUNKED_RESULT_BYTES));
        assert!(bytes.len() <= 1024 * 1024);
    }

    #[test]
    fn tiny_but_viable_nats_limit_uses_compact_error_when_64_chunks_cannot_fit() {
        let mut o = outcome();
        o.result_msgpack = vec![5; 100 * 1024];
        let result = build_work_result(&o, "w", None, false, None);

        let (result, bytes, oversized) = single_delivery(result, 512, true);

        assert!(!result.success);
        assert_eq!(result.error_code.as_deref(), Some("PAYLOAD_TOO_LARGE"));
        assert!(oversized.is_some_and(|size| size > 512));
        assert!(bytes.len() <= 512);
    }

    #[test]
    fn every_adaptive_chunk_envelope_respects_exact_nats_ceiling() {
        let mut o = outcome();
        o.result_msgpack = vec![11; 900 * 1024];
        let result = build_work_result(&o, "w", None, false, None);
        let original = rmp_serde::to_vec_named(&result).unwrap();

        let initial = plan_chunked_work_result(
            &WorkResultRef::from_owned(&result),
            original.clone(),
            usize::MAX,
        )
        .unwrap()
        .expect("unbounded plan");
        let first_envelope_size = initial.encode_chunk(0).unwrap().len();

        let EncodedWorkResult::Chunked(adapted) = encode_work_result_delivery(
            WorkResultRef::from_owned(&result),
            first_envelope_size - 1,
            true,
        )
        .unwrap() else {
            panic!("planner should adapt by adding a chunk");
        };
        assert!(adapted.chunk_count > initial.chunk_count);
        for chunk_index in 0..adapted.chunk_count {
            assert!(adapted.encode_chunk(chunk_index).unwrap().len() < first_envelope_size);
        }
    }

    #[test]
    fn local_result_building_is_not_capped_by_nats_limit() {
        let mut o = outcome();
        o.result_msgpack = vec![7; 2 * 1024 * 1024];

        let result = shape_and_build_work_result(&o, None, "w", None, false, None);

        assert!(result.success);
        assert_eq!(result.result_msgpack.len(), 2 * 1024 * 1024);
    }

    #[test]
    fn timings_populate_queue_and_processing_fields() {
        let timings = Timings {
            queue_ms: 12.5,
            payload_fetch_ms: 3.2,
        };
        let r = build_work_result(&outcome(), "w", Some(timings), false, None);
        assert_eq!(r.queue_ms, Some(12.5));
        // Python sets processing_ms = 0.0 as a placeholder; we match exactly.
        assert_eq!(r.processing_ms, Some(0.0));
        assert_eq!(r.payload_fetch_ms, Some(3.2));
    }

    #[test]
    fn zero_payload_fetch_ms_is_omitted() {
        // Matches Python: `if payload_fetch_ms > 0:` before inclusion.
        let timings = Timings {
            queue_ms: 5.0,
            payload_fetch_ms: 0.0,
        };
        let r = build_work_result(&outcome(), "w", Some(timings), false, None);
        assert_eq!(r.queue_ms, Some(5.0));
        assert!(r.payload_fetch_ms.is_none());
    }

    #[test]
    fn check_reply_subject_rejects_empty_and_whitespace() {
        assert!(matches!(
            check_reply_subject(""),
            Err(PublishError::EmptyReplySubject)
        ));
        assert!(matches!(
            check_reply_subject("   "),
            Err(PublishError::EmptyReplySubject)
        ));
        assert!(check_reply_subject("\t\n").is_err());
    }

    #[test]
    fn check_reply_subject_accepts_normal_inbox() {
        assert!(check_reply_subject("_INBOX.r1.abc").is_ok());
        assert!(check_reply_subject("sie.results.q1").is_ok());
    }

    #[test]
    fn error_path_omits_timings_fields() {
        // Error-only publishes (payload resolution failure, unknown op) go
        // through the `None` timings path and should not carry queue_ms etc.
        let mut o = outcome();
        o.disposition = Disposition::PublishErrorAndAck;
        o.error = Some("oops".into());
        o.error_code = Some("payload_error".into());
        let r = build_work_result(&o, "w", None, false, None);
        assert!(!r.success);
        assert!(r.queue_ms.is_none());
        assert!(r.processing_ms.is_none());
        assert!(r.payload_fetch_ms.is_none());
    }

    // ---- raw_output shaping ---------------------------------------

    #[test]
    fn raw_output_none_leaves_outcome_unchanged() {
        let o = outcome();
        // `outcome()` pre-fills legacy `result_msgpack` and no
        // `raw_output` — the shaper is a no-op.
        match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Unchanged => {}
            ShapeOutcome::Shaped(_) => {
                panic!("raw_output shaper must not touch legacy outcomes that already have result_msgpack");
            }
        }
    }

    #[test]
    fn raw_output_dense_is_shaped_into_msgpack() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![1.0, 2.0, 3.0],
                dim: 3,
                normalize: false,
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            ShapeOutcome::Unchanged => panic!("expected raw_output shaping"),
        };
        // Byte-for-byte identical to the Python
        // `_wrap_encode_output({"dense": np.float32([1,2,3])})` golden.
        let expected = build_dense_payload(&[1.0, 2.0, 3.0], 3).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
        assert!(matches!(shaped.disposition, Disposition::PublishAndAck));
        assert!(
            shaped.raw_output.is_none(),
            "shaper must strip raw_output to avoid double-encoding"
        );
        assert_eq!(shaped.work_item_id, "req-1.0");
        assert_eq!(
            shaped.inference_ms,
            Some(17.5),
            "timing metadata must round-trip"
        );
    }

    #[test]
    fn raw_output_dense_with_normalize_l2s_before_packing() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![3.0, 4.0],
                dim: 2,
                normalize: true,
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        // [3,4] L2-normalised is [0.6, 0.8].
        let expected = build_dense_payload(&[0.6, 0.8], 2).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
    }

    #[test]
    fn raw_output_score_is_shaped_and_sorted() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            score: Some(ScoreOutputRaw {
                scores: vec![0.5, 0.9],
                item_ids: vec!["b".into(), "a".into()],
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        // Rust shaper sorts desc: a(0.9,rank=0), b(0.5,rank=1).
        let expected =
            build_score_payload(&[0.5, 0.9], &["b".to_string(), "a".to_string()]).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
    }

    #[test]
    fn raw_output_shape_error_becomes_error_outcome() {
        // Mismatched dim → shaper returns ShapeError, publisher
        // must NOT panic; it downgrades to PublishErrorAndAck with
        // a typed error code so the gateway surfaces the incident.
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![1.0, 2.0],
                dim: 3,
                normalize: false,
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        assert!(matches!(
            shaped.disposition,
            Disposition::PublishErrorAndAck
        ));
        assert!(shaped.result_msgpack.is_empty());
        assert_eq!(shaped.error_code.as_deref(), Some("raw_output_shape_error"));
        assert!(shaped
            .error
            .as_deref()
            .unwrap_or("")
            .contains("raw_output shape"));
    }

    #[test]
    fn raw_output_with_preexisting_msgpack_is_not_reshaped() {
        // Mixed wire: Python sent both legacy and raw_output.
        // Trust the already-framed bytes — shaper treats this as
        // a rolling-deploy wire artefact.
        let mut o = outcome(); // legacy result_msgpack populated
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![9.0, 9.0, 9.0],
                dim: 3,
                normalize: false,
            }),
            ..Default::default()
        });
        match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Unchanged => {}
            ShapeOutcome::Shaped(_) => panic!("must prefer pre-framed result_msgpack"),
        }
    }

    #[test]
    fn raw_output_on_error_disposition_is_ignored() {
        // Defensive: Python shouldn't attach raw_output to an
        // error disposition, but if it does we don't shape.
        let mut o = outcome();
        o.disposition = Disposition::PublishErrorAndAck;
        o.result_msgpack = Vec::new();
        o.error = Some("upstream".into());
        o.raw_output = Some(RawOutput {
            dense: Some(DenseOutput {
                values: vec![1.0],
                dim: 1,
                normalize: false,
            }),
            ..Default::default()
        });
        match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Unchanged => {}
            ShapeOutcome::Shaped(_) => {
                panic!("must not run shaper on error outcomes; raw_output should be ignored")
            }
        }
    }

    #[test]
    fn raw_output_sparse_is_shaped_into_msgpack() {
        // End-to-end publisher hook for sparse: the Python side
        // sends only the typed `SparseOutput`; the publisher must
        // pack exactly the bytes the legacy `_wrap_encode_output`
        // path produces so SDK decoders see zero change.
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            sparse: Some(SparseOutput {
                indices: vec![3, 7, 42],
                values: vec![0.5, 1.5, 2.5],
                dims: Some(30522),
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            ShapeOutcome::Unchanged => panic!("expected raw_output sparse shaping"),
        };
        let expected = build_sparse_payload(&[3, 7, 42], &[0.5, 1.5, 2.5], Some(30522)).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
        assert!(shaped.raw_output.is_none());
        assert!(matches!(shaped.disposition, Disposition::PublishAndAck));
    }

    #[test]
    fn raw_output_sparse_length_mismatch_becomes_error() {
        // indices/values length parity is a hard invariant; if the
        // Python gate somehow lets a mismatched payload through
        // the publisher must surface a typed error rather than
        // crash or publish garbage.
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            sparse: Some(SparseOutput {
                indices: vec![1, 2, 3],
                values: vec![0.5], // mismatched
                dims: None,
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        assert!(matches!(
            shaped.disposition,
            Disposition::PublishErrorAndAck
        ));
        assert_eq!(shaped.error_code.as_deref(), Some("raw_output_shape_error"));
    }

    #[test]
    fn raw_output_multivector_is_shaped_into_msgpack() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        let values: Vec<f32> = (0..12).map(|i| i as f32).collect();
        o.raw_output = Some(RawOutput {
            multivector: Some(MultivectorOutput {
                values: values.clone(),
                values_f16: Vec::new(),
                num_tokens: 3,
                token_dims: 4,
                dtype: None,
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            ShapeOutcome::Unchanged => panic!("expected raw_output multivector shaping"),
        };
        let expected = build_multivector_payload(&values, 3, 4).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
    }

    #[test]
    fn raw_output_multivector_float16_is_shaped_into_msgpack() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        let values: Vec<f32> = (0..4).map(|i| i as f32).collect();
        o.raw_output = Some(RawOutput {
            multivector: Some(MultivectorOutput {
                values: values.clone(),
                values_f16: Vec::new(),
                num_tokens: 2,
                token_dims: 2,
                dtype: Some("float16".to_string()),
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            ShapeOutcome::Unchanged => panic!("expected raw_output multivector shaping"),
        };
        let expected = build_multivector_payload_with_dtype(&values, 2, 2, "float16").unwrap();
        assert_eq!(shaped.result_msgpack, expected);
    }

    #[test]
    fn raw_output_multivector_preencoded_float16_is_shaped_into_msgpack() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        let values_f16 = vec![0x00, 0x3c, 0x00, 0xc0, 0x00, 0x43, 0x00, 0x00];
        o.raw_output = Some(RawOutput {
            multivector: Some(MultivectorOutput {
                values: Vec::new(),
                values_f16: values_f16.clone(),
                num_tokens: 2,
                token_dims: 2,
                dtype: Some("float16".to_string()),
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            ShapeOutcome::Unchanged => panic!("expected raw_output multivector shaping"),
        };
        let expected = build_multivector_payload_from_f16_bytes(&values_f16, 2, 2).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
    }

    #[test]
    fn raw_output_echoes_caller_item_id_before_multivector() {
        let mut o = outcome();
        o.result_msgpack.clear();
        o.raw_output = Some(RawOutput {
            multivector: Some(MultivectorOutput {
                values: Vec::new(),
                values_f16: vec![0x00, 0x3c],
                num_tokens: 1,
                token_dims: 1,
                dtype: Some("float16".to_string()),
            }),
            ..Default::default()
        });

        let shaped = match shape_raw_output_for_wire(&o, Some("doc-42")) {
            ShapeOutcome::Shaped(s) => s,
            ShapeOutcome::Unchanged => panic!("expected raw_output multivector shaping"),
        };
        let decoded: rmpv::Value = rmp_serde::from_slice(&shaped.result_msgpack).unwrap();
        let fields = decoded.as_map().expect("encode result map");
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0].0.as_str(), Some("id"));
        assert_eq!(fields[0].1.as_str(), Some("doc-42"));
        assert_eq!(fields[1].0.as_str(), Some("multivector"));
    }

    #[test]
    fn batched_f16_multivector_slice_is_shaped_without_an_intermediate_copy() {
        let mut o = outcome();
        o.result_msgpack.clear();
        let values_f16 = [
            f16::from_bits(0x3c00),
            f16::from_bits(0xc000),
            f16::from_bits(0x4300),
            f16::from_bits(0),
        ];

        let shaped = shape_batched_f16_multivector_outcome(&o, &values_f16[1..3], 1, 2, None);

        let expected =
            build_multivector_payload_from_f16_bytes(&[0x00, 0xc0, 0x00, 0x43], 1, 2).unwrap();
        assert_eq!(shaped.result_msgpack, expected);
        assert!(shaped.raw_output.is_none());
    }

    #[test]
    fn batched_f16_multivector_echoes_caller_item_id() {
        let mut o = outcome();
        o.result_msgpack.clear();

        let shaped = shape_batched_f16_multivector_outcome(
            &o,
            &[f16::from_f32(1.0)],
            1,
            1,
            Some("doc-shared"),
        );

        let decoded: rmpv::Value = rmp_serde::from_slice(&shaped.result_msgpack).unwrap();
        let fields = decoded.as_map().expect("encode result map");
        assert_eq!(fields[0].0.as_str(), Some("id"));
        assert_eq!(fields[0].1.as_str(), Some("doc-shared"));
        assert_eq!(fields[1].0.as_str(), Some("multivector"));
    }

    #[test]
    fn malformed_batched_f16_multivector_is_a_typed_error() {
        let mut o = outcome();
        o.result_msgpack.clear();

        let shaped = shape_batched_f16_multivector_outcome(&o, &[f16::from_f32(1.0)], 1, 2, None);

        assert!(matches!(
            shaped.disposition,
            Disposition::PublishErrorAndAck
        ));
        assert_eq!(shaped.error_code.as_deref(), Some("raw_output_shape_error"));
    }

    #[test]
    fn raw_output_multivector_preencoded_float16_rejects_dtype_mismatch() {
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput {
            multivector: Some(MultivectorOutput {
                values: Vec::new(),
                values_f16: vec![0x00, 0x3c],
                num_tokens: 1,
                token_dims: 1,
                dtype: Some("float32".to_string()),
            }),
            ..Default::default()
        });
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        assert!(matches!(
            shaped.disposition,
            Disposition::PublishErrorAndAck
        ));
        assert_eq!(shaped.error_code.as_deref(), Some("raw_output_shape_error"));
    }

    #[test]
    fn raw_output_empty_variant_becomes_error_outcome() {
        // Forward-compat: a future Rust build sees RawOutput with
        // no recognised variant populated (all None). Must surface
        // as a typed error, not a silent empty success.
        let mut o = outcome();
        o.result_msgpack = Vec::new();
        o.raw_output = Some(RawOutput::default());
        let shaped = match shape_raw_output_for_wire(&o, None) {
            ShapeOutcome::Shaped(s) => s,
            _ => unreachable!(),
        };
        assert!(matches!(
            shaped.disposition,
            Disposition::PublishErrorAndAck
        ));
        assert_eq!(shaped.error_code.as_deref(), Some("raw_output_shape_error"));
    }
}
