//! Negotiated IPC response-chunk reassembly shared by pool and mux clients.
//!
//! The backend serializes the ordinary `ResponseEnvelope` exactly once and
//! splits only those bytes. This module accepts the explicit v1 envelope,
//! applies bounded-memory and identity checks, and returns the original bytes
//! only after length and SHA-256 verification.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use rmp::decode::read_str_from_slice;
use rmp::Marker;
use serde::Deserialize;
use sha2::{Digest, Sha256};

use super::ipc_types::{IpcResponseChunkV1, IPC_VERSION};
use crate::observability::metrics::{IpcResponseChunkOutcome, SidecarTelemetry};

pub(crate) const IPC_RESPONSE_CHUNK_KIND_V1: &str = "ipc_response_chunk_v1";
pub(crate) const LEGACY_IPC_RESPONSE_FRAME_BYTES: usize = 32 * 1024 * 1024;
pub(crate) const IPC_RESPONSE_CHUNK_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;
pub(crate) const MAX_CHUNKED_IPC_RESPONSE_BYTES: usize = 128 * 1024 * 1024;
pub(crate) const MAX_IPC_RESPONSE_CHUNKS: u32 = 64;
const RESPONSE_DECODE_HEADROOM_BYTES: usize = 4 * 1024;
pub(crate) const MAX_GLOBAL_IPC_RESPONSE_ASSEMBLY_BYTES: usize =
    2 * MAX_CHUNKED_IPC_RESPONSE_BYTES + RESPONSE_DECODE_HEADROOM_BYTES;
static PRODUCTION_RESPONSE_CHUNK_BUDGET: LazyLock<Arc<ResponseChunkBudget>> = LazyLock::new(|| {
    Arc::new(ResponseChunkBudget::new(
        MAX_GLOBAL_IPC_RESPONSE_ASSEMBLY_BYTES,
    ))
});

/// Bounds applied before reserving or allocating a response assembly buffer.
///
/// The payload ceiling is part of the v1 contract, not merely a producer hint:
/// together with the requirement that a chunked response exceed the 32 MiB
/// legacy frame ceiling, it ensures the `2 * total + headroom` reservation also
/// covers the raw physical frame and the decoded chunk payload held briefly
/// while appending it to the assembly buffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResponseChunkLimits {
    legacy_frame_bytes: usize,
    max_chunk_payload_bytes: usize,
    max_chunked_bytes: usize,
    max_chunks: u32,
}

impl ResponseChunkLimits {
    pub(crate) const fn production() -> Self {
        Self {
            legacy_frame_bytes: LEGACY_IPC_RESPONSE_FRAME_BYTES,
            max_chunk_payload_bytes: IPC_RESPONSE_CHUNK_PAYLOAD_BYTES,
            max_chunked_bytes: MAX_CHUNKED_IPC_RESPONSE_BYTES,
            max_chunks: MAX_IPC_RESPONSE_CHUNKS,
        }
    }

    #[cfg(test)]
    pub(crate) const fn relaxed_for_small_fixtures() -> Self {
        Self {
            legacy_frame_bytes: 0,
            max_chunk_payload_bytes: MAX_CHUNKED_IPC_RESPONSE_BYTES,
            max_chunked_bytes: MAX_CHUNKED_IPC_RESPONSE_BYTES,
            max_chunks: MAX_IPC_RESPONSE_CHUNKS,
        }
    }

    const fn max_chunk_frame_bytes(self) -> usize {
        self.max_chunk_payload_bytes
            .saturating_add(RESPONSE_DECODE_HEADROOM_BYTES)
    }
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum ResponseChunkError {
    #[error("malformed IPC response envelope: {0}")]
    MalformedEnvelope(rmp_serde::decode::Error),
    #[error("IPC response missing request_id")]
    MissingRequestId,
    #[error("IPC response request_id mismatch")]
    RequestIdMismatch,
    #[error("malformed IPC response chunk v1: {0}")]
    MalformedChunk(rmp_serde::decode::Error),
    #[error("IPC response chunk version mismatch")]
    VersionMismatch,
    #[error("IPC response chunk sequence must start at index zero")]
    MissingFirstChunk,
    #[error("IPC response chunk index is duplicate or out of order")]
    OutOfOrder,
    #[error("IPC response chunk layout changed during transfer")]
    LayoutMismatch,
    #[error("IPC response chunk bounds are invalid")]
    InvalidBounds,
    #[error("IPC response chunk digest is invalid")]
    InvalidDigest,
    #[error("IPC response chunk transfer exceeds the global assembly budget")]
    GlobalBudgetExceeded,
    #[error("IPC response changed from chunked to legacy mid-transfer")]
    LegacyDuringTransfer,
    #[error("IPC response allocation failed")]
    AllocationFailed,
}

#[derive(Debug, Deserialize)]
struct ResponseFrameHead<'a> {
    #[serde(default, borrow)]
    kind: Option<&'a str>,
    #[serde(default, borrow)]
    request_id: &'a str,
}

fn response_frame_head<'a>(
    frame: &'a [u8],
    limits: ResponseChunkLimits,
) -> Result<ResponseFrameHead<'a>, ResponseChunkError> {
    let head: ResponseFrameHead<'a> =
        rmp_serde::from_slice(frame).map_err(ResponseChunkError::MalformedEnvelope)?;
    if head.kind == Some(IPC_RESPONSE_CHUNK_KIND_V1) && frame.len() > limits.max_chunk_frame_bytes()
    {
        return Err(ResponseChunkError::InvalidBounds);
    }
    Ok(head)
}

fn canonical_response_frame_route(frame: &[u8]) -> Option<(&str, bool)> {
    let Marker::FixMap(entries) = rmp::decode::read_marker(&mut &frame[..]).ok()? else {
        return None;
    };
    if entries < 3 {
        return None;
    }

    let (version_key, rest) = read_str_from_slice(&frame[1..]).ok()?;
    if version_key != "version" || rest.first().copied() != Some(IPC_VERSION as u8) {
        return None;
    }
    let (request_id_key, rest) = read_str_from_slice(&rest[1..]).ok()?;
    if request_id_key != "request_id" {
        return None;
    }
    let (request_id, rest) = read_str_from_slice(rest).ok()?;
    let (third_key, _) = read_str_from_slice(rest).ok()?;
    match third_key {
        "ok" => Some((request_id, false)),
        "transfer_digest" => Some((request_id, true)),
        _ => None,
    }
}

pub(crate) struct ResponseFrameRoute {
    pub(crate) request_id: String,
    pub(crate) requires_chunk_parser: bool,
}

/// Parse only the routing identity from a response frame.
///
/// The mux uses this before selecting one per-request assembler. Unknown
/// `kind` values remain legacy-compatible; only the exact v1 discriminator is
/// interpreted as a chunk by [`ResponseAssembler`].
pub(crate) fn response_frame_route(
    frame: &[u8],
    limits: ResponseChunkLimits,
) -> Result<ResponseFrameRoute, ResponseChunkError> {
    if let Some((request_id, requires_chunk_parser)) = canonical_response_frame_route(frame) {
        if request_id.is_empty() {
            return Err(ResponseChunkError::MissingRequestId);
        }
        return Ok(ResponseFrameRoute {
            request_id: request_id.to_owned(),
            requires_chunk_parser,
        });
    }

    let head = response_frame_head(frame, limits)?;
    if head.request_id.is_empty() {
        return Err(ResponseChunkError::MissingRequestId);
    }
    Ok(ResponseFrameRoute {
        request_id: head.request_id.to_owned(),
        requires_chunk_parser: head.kind == Some(IPC_RESPONSE_CHUNK_KIND_V1),
    })
}

pub(crate) fn response_frame_is_exact_chunk(
    frame: &[u8],
    limits: ResponseChunkLimits,
) -> Result<bool, ResponseChunkError> {
    Ok(response_frame_head(frame, limits)?.kind == Some(IPC_RESPONSE_CHUNK_KIND_V1))
}

/// Process-wide byte reservation shared by every active pool and mux response
/// assembler in one sidecar process.
#[derive(Debug)]
pub(crate) struct ResponseChunkBudget {
    used: AtomicUsize,
    limit: usize,
    telemetry: Mutex<Option<SidecarTelemetry>>,
}

impl ResponseChunkBudget {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            used: AtomicUsize::new(0),
            limit,
            telemetry: Mutex::new(None),
        }
    }

    pub(crate) fn production() -> Arc<Self> {
        Arc::clone(&PRODUCTION_RESPONSE_CHUNK_BUDGET)
    }

    pub(crate) fn attach_telemetry(&self, telemetry: SidecarTelemetry) {
        if !telemetry.is_enabled() {
            return;
        }
        if let Ok(mut current) = self.telemetry.lock() {
            // Install and initialise under the same lock used by updates. If a
            // reservation races attachment, its updater either runs after this
            // assignment or this load observes its already-published atomic
            // value; the newly attached facade cannot start stale.
            telemetry.ipc_response_chunk_reserved_changed(self.used.load(Ordering::Acquire));
            *current = Some(telemetry);
        }
    }

    fn update_reserved_telemetry(&self) {
        if let Ok(current) = self.telemetry.lock() {
            if let Some(telemetry) = current.as_ref() {
                // Read while holding the telemetry lock so concurrent reserve/drop
                // operations cannot publish older CAS results out of order.
                telemetry.ipc_response_chunk_reserved_changed(self.used.load(Ordering::Acquire));
            }
        }
    }

    fn reserve(
        self: &Arc<Self>,
        bytes: usize,
    ) -> Result<ResponseChunkReservation, ResponseChunkError> {
        let mut observed = self.used.load(Ordering::Acquire);
        loop {
            let Some(next) = observed.checked_add(bytes) else {
                return Err(ResponseChunkError::GlobalBudgetExceeded);
            };
            if next > self.limit {
                return Err(ResponseChunkError::GlobalBudgetExceeded);
            }
            match self.used.compare_exchange_weak(
                observed,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.update_reserved_telemetry();
                    return Ok(ResponseChunkReservation {
                        budget: Arc::clone(self),
                        bytes,
                    });
                }
                Err(actual) => observed = actual,
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
}

#[derive(Debug)]
struct ResponseChunkReservation {
    budget: Arc<ResponseChunkBudget>,
    bytes: usize,
}

impl Drop for ResponseChunkReservation {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
        self.budget.update_reserved_telemetry();
    }
}

#[derive(Debug)]
struct TransferState {
    digest: [u8; 32],
    chunk_count: u32,
    total_bytes: usize,
    next_index: u32,
    bytes: Vec<u8>,
    hasher: Sha256,
    _reservation: ResponseChunkReservation,
}

/// A complete legacy or reconstructed response whose chunk reservation remains
/// charged until the typed caller finishes decoding it.
#[derive(Debug)]
pub(crate) struct AssembledResponse {
    bytes: Vec<u8>,
    _reservation: Option<ResponseChunkReservation>,
    chunk_count: Option<u32>,
}

impl AssembledResponse {
    pub(crate) fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }

    fn chunk_count(&self) -> Option<u32> {
        self.chunk_count
    }
}

#[derive(Debug)]
pub(crate) enum ResponseFrameStatus {
    Pending,
    Complete(AssembledResponse),
}

/// Reassembles at most one response for one expected request id.
pub(crate) struct ResponseAssembler {
    expected_request_id: String,
    budget: Arc<ResponseChunkBudget>,
    telemetry: Option<SidecarTelemetry>,
    limits: ResponseChunkLimits,
    transfer: Option<TransferState>,
}

impl ResponseAssembler {
    #[cfg(test)]
    pub(crate) fn new(
        expected_request_id: impl Into<String>,
        budget: Arc<ResponseChunkBudget>,
    ) -> Self {
        Self {
            expected_request_id: expected_request_id.into(),
            budget,
            telemetry: None,
            limits: ResponseChunkLimits::relaxed_for_small_fixtures(),
            transfer: None,
        }
    }

    pub(crate) fn new_with_telemetry_and_limits(
        expected_request_id: impl Into<String>,
        budget: Arc<ResponseChunkBudget>,
        telemetry: Option<SidecarTelemetry>,
        limits: ResponseChunkLimits,
    ) -> Self {
        Self {
            expected_request_id: expected_request_id.into(),
            budget,
            telemetry,
            limits,
            transfer: None,
        }
    }

    pub(crate) fn push(
        &mut self,
        frame: Vec<u8>,
    ) -> Result<ResponseFrameStatus, ResponseChunkError> {
        let result = self.push_inner(frame);
        if let Some(telemetry) = self.telemetry.as_ref() {
            match &result {
                Ok(ResponseFrameStatus::Complete(response)) => {
                    if let Some(chunk_count) = response.chunk_count() {
                        telemetry.ipc_response_chunk_transfer_completed(
                            IpcResponseChunkOutcome::Completed,
                            Some(response.len()),
                            Some(chunk_count),
                        );
                    }
                }
                Err(ResponseChunkError::GlobalBudgetExceeded) => {
                    telemetry.ipc_response_chunk_transfer_completed(
                        IpcResponseChunkOutcome::BudgetRejected,
                        None,
                        None,
                    );
                }
                Err(_) => telemetry.ipc_response_chunk_transfer_completed(
                    IpcResponseChunkOutcome::ProtocolError,
                    None,
                    None,
                ),
                Ok(ResponseFrameStatus::Pending) => {}
            }
        }
        result
    }

    pub(crate) fn push_legacy(
        &mut self,
        frame: Vec<u8>,
    ) -> Result<ResponseFrameStatus, ResponseChunkError> {
        if self.transfer.is_some() {
            if let Some(telemetry) = self.telemetry.as_ref() {
                telemetry.ipc_response_chunk_transfer_completed(
                    IpcResponseChunkOutcome::ProtocolError,
                    None,
                    None,
                );
            }
            return Err(ResponseChunkError::LegacyDuringTransfer);
        }
        Ok(ResponseFrameStatus::Complete(AssembledResponse {
            bytes: frame,
            _reservation: None,
            chunk_count: None,
        }))
    }

    fn push_inner(&mut self, frame: Vec<u8>) -> Result<ResponseFrameStatus, ResponseChunkError> {
        let head = response_frame_head(&frame, self.limits)?;
        if head.request_id.is_empty() {
            return Err(ResponseChunkError::MissingRequestId);
        }
        if head.request_id != self.expected_request_id {
            return Err(ResponseChunkError::RequestIdMismatch);
        }
        if head.kind != Some(IPC_RESPONSE_CHUNK_KIND_V1) {
            if self.transfer.is_some() {
                return Err(ResponseChunkError::LegacyDuringTransfer);
            }
            return Ok(ResponseFrameStatus::Complete(AssembledResponse {
                bytes: frame,
                _reservation: None,
                chunk_count: None,
            }));
        }

        let chunk: IpcResponseChunkV1 =
            rmp_serde::from_slice(&frame).map_err(ResponseChunkError::MalformedChunk)?;
        self.push_chunk(chunk)
    }

    fn push_chunk(
        &mut self,
        chunk: IpcResponseChunkV1,
    ) -> Result<ResponseFrameStatus, ResponseChunkError> {
        if chunk.version != IPC_VERSION {
            return Err(ResponseChunkError::VersionMismatch);
        }
        if chunk.request_id != self.expected_request_id {
            return Err(ResponseChunkError::RequestIdMismatch);
        }
        if chunk.chunk_count == 0
            || chunk.chunk_count > self.limits.max_chunks
            || chunk.total_bytes == 0
            || chunk.total_bytes <= self.limits.legacy_frame_bytes as u64
            || chunk.total_bytes > self.limits.max_chunked_bytes as u64
            || chunk.payload.is_empty()
            || chunk.payload.len() > self.limits.max_chunk_payload_bytes
            || chunk.payload.len() as u64 > chunk.total_bytes
            || chunk.chunk_count as u64 > chunk.total_bytes
            || chunk.transfer_digest.len() != 32
        {
            return Err(ResponseChunkError::InvalidBounds);
        }

        if self.transfer.is_none() {
            if chunk.chunk_index != 0 {
                return Err(ResponseChunkError::MissingFirstChunk);
            }
            let total_bytes = usize::try_from(chunk.total_bytes)
                .map_err(|_| ResponseChunkError::InvalidBounds)?;
            let reserved_bytes = total_bytes
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(RESPONSE_DECODE_HEADROOM_BYTES))
                .ok_or(ResponseChunkError::InvalidBounds)?;
            let reservation = self.budget.reserve(reserved_bytes)?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(total_bytes)
                .map_err(|_| ResponseChunkError::AllocationFailed)?;
            let digest: [u8; 32] = chunk
                .transfer_digest
                .as_slice()
                .try_into()
                .map_err(|_| ResponseChunkError::InvalidDigest)?;
            self.transfer = Some(TransferState {
                digest,
                chunk_count: chunk.chunk_count,
                total_bytes,
                next_index: 0,
                bytes,
                hasher: Sha256::new(),
                _reservation: reservation,
            });
        }

        let state = self
            .transfer
            .as_mut()
            .expect("response transfer was initialised above");
        let digest_matches = chunk.transfer_digest.as_slice() == state.digest;
        if chunk.chunk_count != state.chunk_count
            || chunk.total_bytes != state.total_bytes as u64
            || !digest_matches
        {
            return Err(ResponseChunkError::LayoutMismatch);
        }
        if chunk.chunk_index != state.next_index {
            return Err(ResponseChunkError::OutOfOrder);
        }
        let Some(new_len) = state.bytes.len().checked_add(chunk.payload.len()) else {
            return Err(ResponseChunkError::InvalidBounds);
        };
        if new_len > state.total_bytes {
            return Err(ResponseChunkError::InvalidBounds);
        }
        state.hasher.update(&chunk.payload);
        state.bytes.extend_from_slice(&chunk.payload);
        state.next_index += 1;

        if state.next_index < state.chunk_count {
            if state.bytes.len() == state.total_bytes {
                return Err(ResponseChunkError::InvalidBounds);
            }
            return Ok(ResponseFrameStatus::Pending);
        }
        if state.next_index != state.chunk_count || state.bytes.len() != state.total_bytes {
            return Err(ResponseChunkError::InvalidBounds);
        }
        let completed = self
            .transfer
            .take()
            .expect("completed response transfer is present");
        let actual_digest: [u8; 32] = completed.hasher.finalize().into();
        if actual_digest != completed.digest {
            return Err(ResponseChunkError::InvalidDigest);
        }
        Ok(ResponseFrameStatus::Complete(AssembledResponse {
            bytes: completed.bytes,
            _reservation: Some(completed._reservation),
            chunk_count: Some(completed.chunk_count),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(
        request_id: &str,
        bytes: &[u8],
        index: u32,
        count: u32,
        total: usize,
        digest: &[u8],
    ) -> Vec<u8> {
        rmp_serde::to_vec_named(&IpcResponseChunkV1 {
            version: IPC_VERSION,
            request_id: request_id.to_owned(),
            transfer_digest: digest.to_vec(),
            chunk_index: index,
            chunk_count: count,
            total_bytes: total as u64,
            payload: bytes.to_vec(),
            kind: IPC_RESPONSE_CHUNK_KIND_V1.to_owned(),
        })
        .unwrap()
    }

    #[test]
    fn reassembles_exact_bytes_and_releases_reservation() {
        let original = b"an exact serialized response";
        let digest = Sha256::digest(original);
        let budget = Arc::new(ResponseChunkBudget::new(8 * 1024));
        let mut assembler = ResponseAssembler::new("r1", Arc::clone(&budget));

        assert!(matches!(
            assembler
                .push(frame("r1", &original[..10], 0, 2, original.len(), &digest))
                .unwrap(),
            ResponseFrameStatus::Pending
        ));
        let reserved = original.len() * 2 + RESPONSE_DECODE_HEADROOM_BYTES;
        assert_eq!(budget.used(), reserved);
        let completed = assembler
            .push(frame("r1", &original[10..], 1, 2, original.len(), &digest))
            .unwrap();
        let ResponseFrameStatus::Complete(completed) = completed else {
            panic!("second chunk must complete the response");
        };
        assert_eq!(completed.as_slice(), original);
        assert_eq!(
            budget.used(),
            reserved,
            "decode reservation dropped too early"
        );
        drop(completed);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn rejects_duplicate_out_of_order_layout_digest_and_overbound() {
        let original = b"0123456789";
        let digest = Sha256::digest(original);

        let cases = [
            (
                vec![
                    frame("r", &original[..5], 0, 2, original.len(), &digest),
                    frame("r", &original[..5], 0, 2, original.len(), &digest),
                ],
                "duplicate",
            ),
            (
                vec![frame("r", &original[5..], 1, 2, original.len(), &digest)],
                "missing-first",
            ),
            (
                vec![
                    frame("r", &original[..5], 0, 2, original.len(), &digest),
                    frame("r", &original[5..], 1, 3, original.len(), &digest),
                ],
                "layout",
            ),
            (
                vec![frame("r", original, 0, 1, original.len(), &[0; 32])],
                "digest",
            ),
        ];

        for (frames, name) in cases {
            let budget = Arc::new(ResponseChunkBudget::new(8 * 1024));
            let mut assembler = ResponseAssembler::new("r", Arc::clone(&budget));
            let mut last = None;
            for frame in frames {
                match assembler.push(frame) {
                    Ok(status) => last = Some(Ok(status)),
                    Err(error) => {
                        last = Some(Err(error));
                        break;
                    }
                }
            }
            assert!(matches!(last, Some(Err(_))), "{name} must fail");
            drop(assembler);
            assert_eq!(budget.used(), 0, "{name} leaked its reservation");
        }

        let too_large = MAX_CHUNKED_IPC_RESPONSE_BYTES as u64 + 1;
        let overbound = rmp_serde::to_vec_named(&IpcResponseChunkV1 {
            version: IPC_VERSION,
            request_id: "r".to_owned(),
            transfer_digest: vec![0; 32],
            chunk_index: 0,
            chunk_count: 1,
            total_bytes: too_large,
            payload: vec![1],
            kind: IPC_RESPONSE_CHUNK_KIND_V1.to_owned(),
        })
        .unwrap();
        let budget = Arc::new(ResponseChunkBudget::new(1024));
        let mut assembler = ResponseAssembler::new("r", budget);
        assert!(matches!(
            assembler.push(overbound),
            Err(ResponseChunkError::InvalidBounds)
        ));

        let overcount = frame(
            "r",
            &[1],
            0,
            MAX_IPC_RESPONSE_CHUNKS + 1,
            (MAX_IPC_RESPONSE_CHUNKS + 1) as usize,
            &[0; 32],
        );
        let budget = Arc::new(ResponseChunkBudget::new(8 * 1024));
        let mut assembler = ResponseAssembler::new("r", budget);
        assert!(matches!(
            assembler.push(overcount),
            Err(ResponseChunkError::InvalidBounds)
        ));
    }

    #[test]
    fn production_limits_reject_legacy_sized_chunk_envelopes_before_reserving() {
        let budget = Arc::new(ResponseChunkBudget::new(
            MAX_GLOBAL_IPC_RESPONSE_ASSEMBLY_BYTES,
        ));
        let mut assembler = ResponseAssembler::new_with_telemetry_and_limits(
            "r",
            Arc::clone(&budget),
            None,
            ResponseChunkLimits::production(),
        );
        let malformed = frame("r", &[1], 0, 1, LEGACY_IPC_RESPONSE_FRAME_BYTES, &[0; 32]);

        assert!(matches!(
            assembler.push(malformed),
            Err(ResponseChunkError::InvalidBounds)
        ));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn production_limits_reject_payload_above_v1_ceiling_before_reserving() {
        let budget = Arc::new(ResponseChunkBudget::new(
            MAX_GLOBAL_IPC_RESPONSE_ASSEMBLY_BYTES,
        ));
        let mut assembler = ResponseAssembler::new_with_telemetry_and_limits(
            "r",
            Arc::clone(&budget),
            None,
            ResponseChunkLimits::production(),
        );
        let oversized_payload = vec![1; IPC_RESPONSE_CHUNK_PAYLOAD_BYTES + 1];
        let malformed = frame(
            "r",
            &oversized_payload,
            0,
            2,
            LEGACY_IPC_RESPONSE_FRAME_BYTES + 1,
            &[0; 32],
        );

        assert!(matches!(
            assembler.push(malformed),
            Err(ResponseChunkError::InvalidBounds)
        ));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn production_limits_reject_padded_chunk_frame_before_reserving() {
        #[derive(serde::Serialize)]
        struct PaddedChunk {
            kind: &'static str,
            version: u32,
            request_id: &'static str,
            #[serde(with = "serde_bytes")]
            transfer_digest: Vec<u8>,
            chunk_index: u32,
            chunk_count: u32,
            total_bytes: u64,
            #[serde(with = "serde_bytes")]
            payload: Vec<u8>,
            #[serde(with = "serde_bytes")]
            ignored_padding: Vec<u8>,
        }

        let limits = ResponseChunkLimits::production();
        let malformed = rmp_serde::to_vec_named(&PaddedChunk {
            kind: IPC_RESPONSE_CHUNK_KIND_V1,
            version: IPC_VERSION,
            request_id: "r",
            transfer_digest: vec![0; 32],
            chunk_index: 0,
            chunk_count: 2,
            total_bytes: (LEGACY_IPC_RESPONSE_FRAME_BYTES + 1) as u64,
            payload: vec![1],
            ignored_padding: vec![0; limits.max_chunk_frame_bytes()],
        })
        .unwrap();
        assert!(malformed.len() > limits.max_chunk_frame_bytes());

        let budget = Arc::new(ResponseChunkBudget::new(
            MAX_GLOBAL_IPC_RESPONSE_ASSEMBLY_BYTES,
        ));
        let mut assembler = ResponseAssembler::new_with_telemetry_and_limits(
            "r",
            Arc::clone(&budget),
            None,
            limits,
        );

        assert!(matches!(
            assembler.push(malformed),
            Err(ResponseChunkError::InvalidBounds)
        ));
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn global_budget_rejects_then_releases_on_cancel_drop() {
        let original = b"0123456789";
        let digest = Sha256::digest(original);
        let reservation_bytes = original.len() * 2 + RESPONSE_DECODE_HEADROOM_BYTES;
        let budget = Arc::new(ResponseChunkBudget::new(reservation_bytes));
        let mut first = ResponseAssembler::new("r1", Arc::clone(&budget));
        assert!(matches!(
            first
                .push(frame("r1", &original[..5], 0, 2, original.len(), &digest))
                .unwrap(),
            ResponseFrameStatus::Pending
        ));

        let mut second = ResponseAssembler::new("r2", Arc::clone(&budget));
        assert!(matches!(
            second.push(frame("r2", &original[..5], 0, 2, original.len(), &digest)),
            Err(ResponseChunkError::GlobalBudgetExceeded)
        ));

        drop(first);
        assert_eq!(budget.used(), 0);
        assert!(matches!(
            second
                .push(frame("r2", &original[..5], 0, 2, original.len(), &digest))
                .unwrap(),
            ResponseFrameStatus::Pending
        ));
    }

    #[test]
    fn unknown_kind_remains_a_legacy_response() {
        let legacy = rmp_serde::to_vec_named(&serde_json::json!({
            "kind": "future_response_v2",
            "version": IPC_VERSION,
            "request_id": "r",
            "ok": true,
            "body": {"value": 1},
            "error": null,
        }))
        .unwrap();
        let budget = ResponseChunkBudget::production();
        let mut assembler = ResponseAssembler::new("r", budget);
        let ResponseFrameStatus::Complete(completed) = assembler.push(legacy.clone()).unwrap()
        else {
            panic!("legacy response must complete immediately");
        };
        assert_eq!(completed.as_slice(), legacy);
    }

    #[test]
    fn production_budget_is_process_wide() {
        let first = ResponseChunkBudget::production();
        let second = ResponseChunkBudget::production();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn completed_transfer_with_telemetry_holds_reservation_until_drop() {
        let original = b"telemetry-bearing exact response";
        let digest = Sha256::digest(original);
        let budget = Arc::new(ResponseChunkBudget::new(8 * 1024));
        let telemetry = SidecarTelemetry::for_tests(&[]);
        budget.attach_telemetry(telemetry.clone());
        let mut assembler = ResponseAssembler::new_with_telemetry_and_limits(
            "r",
            Arc::clone(&budget),
            Some(telemetry),
            ResponseChunkLimits::relaxed_for_small_fixtures(),
        );

        let ResponseFrameStatus::Complete(completed) = assembler
            .push(frame("r", original, 0, 1, original.len(), &digest))
            .unwrap()
        else {
            panic!("single chunk must complete the transfer");
        };
        assert!(budget.used() > 0);

        drop(completed);
        assert_eq!(budget.used(), 0);
    }
}
