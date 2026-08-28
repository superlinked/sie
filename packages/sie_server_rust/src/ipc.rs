//! UDS IPC server for the Rust worker process.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use serde::de::DeserializeOwned;
use serde::Serialize;
use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, UnixListener, UnixStream};
use tokio::task::JoinSet;
use tracing::{debug, error, info, warn, Instrument};
use tracing_opentelemetry::OpenTelemetrySpanExt;

use crate::ipc_types::{
    ApplyModelConfigRequest, ApplyModelConfigResponse, BatchOutcome, DrainRequest, DrainResponse,
    EnsureModelReadyRequest, ProcessEncodeBatchRequest, ProcessExtractBatchRequest,
    ProcessScoreBatchRequest, ReplaceModelConfigsRequest, ReplaceModelConfigsResponse,
    RequestEnvelope, ResponseEnvelope, SetPinnedModelsRequest, SignalGenerateCancelResponse,
    WorkerCapabilitiesResponse, IPC_VERSION, METHOD_APPLY_MODEL_CONFIG, METHOD_DRAIN,
    METHOD_ENSURE_MODEL_READY, METHOD_PING, METHOD_PROCESS_ENCODE_BATCH,
    METHOD_PROCESS_EXTRACT_BATCH, METHOD_PROCESS_GENERATE, METHOD_PROCESS_SCORE_BATCH,
    METHOD_REPLACE_MODEL_CONFIGS, METHOD_RUN_BATCH, METHOD_SET_PINNED_MODELS,
    METHOD_SIGNAL_GENERATE_CANCEL, METHOD_WORKER_CAPABILITIES,
};
use crate::native_backend::NativeBackend;
use crate::observability::propagation::{extract_context_from_w3c, remote_span_context};

const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;
const IPC_RESPONSE_CHUNK_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;
const MAX_CHUNKED_IPC_RESPONSE_BYTES: usize = 128 * 1024 * 1024;
const MAX_IPC_RESPONSE_CHUNKS: u32 = 64;
const IPC_RESPONSE_CHUNK_KIND_V1: &str = "ipc_response_chunk_v1";

#[derive(Debug, Clone)]
pub struct IpcServerConfig {
    pub socket_path: PathBuf,
    pub worker_id: String,
    pub bundle: String,
    pub http_host: String,
    pub http_port: u16,
}

#[derive(Debug, Error)]
pub enum IpcServerError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("msgpack decode: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
    #[error("msgpack encode: {0}")]
    Encode(#[from] rmp_serde::encode::Error),
    #[error("json conversion: {0}")]
    Json(#[from] serde_json::Error),
    #[error("frame too large: {0} > {MAX_FRAME_BYTES}")]
    FrameTooLarge(u32),
}

#[derive(Clone)]
pub struct IpcServer {
    config: IpcServerConfig,
    backend: Arc<NativeBackend>,
    bundle_config_hash: Arc<RwLock<String>>,
}

impl IpcServer {
    pub fn new(config: IpcServerConfig, backend: NativeBackend) -> Self {
        Self {
            config,
            backend: Arc::new(backend),
            bundle_config_hash: Arc::new(RwLock::new(String::new())),
        }
    }

    pub async fn run(self) -> Result<(), IpcServerError> {
        if let Some(parent) = self.config.socket_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        unlink_stale_socket(&self.config.socket_path).await?;
        let listener = UnixListener::bind(&self.config.socket_path)?;
        let http_server = self.clone();
        let mut tasks = JoinSet::new();
        tasks.spawn(async move {
            if let Err(e) = http_server.run_http().await {
                error!(error = %e, "rust worker HTTP server stopped");
            }
        });

        info!(
            path = %self.config.socket_path.display(),
            worker_id = %self.config.worker_id,
            bundle = %self.config.bundle,
            "sie-server-rust IPC server listening"
        );

        let shutdown = shutdown_signal();
        tokio::pin!(shutdown);
        loop {
            tokio::select! {
                accepted = listener.accept() => {
                    let (stream, _) = accepted?;
                    let server = self.clone();
                    tasks.spawn(async move {
                        if let Err(e) = server.handle_connection(stream).await {
                            debug!(error = %e, "IPC connection closed with error");
                        }
                    });
                }
                _ = &mut shutdown => {
                    info!("shutdown signal received");
                    break;
                }
                Some(joined) = tasks.join_next() => {
                    if let Err(e) = joined {
                        warn!(error = %e, "background task failed");
                    }
                }
            }
        }

        Ok(())
    }

    async fn handle_connection(&self, mut stream: UnixStream) -> Result<(), IpcServerError> {
        loop {
            let frame = match read_frame(&mut stream).await {
                Ok(frame) => frame,
                Err(IpcServerError::Io(e))
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::UnexpectedEof
                            | std::io::ErrorKind::ConnectionReset
                            | std::io::ErrorKind::BrokenPipe
                    ) =>
                {
                    return Ok(());
                }
                Err(e) => return Err(e),
            };

            let response = self.handle_request_frame(&frame).await?;
            write_response(&mut stream, response).await?;
        }
    }

    async fn handle_request_frame(&self, frame: &[u8]) -> Result<HandledResponse, IpcServerError> {
        let envelope: RequestEnvelope = rmp_serde::from_slice(frame)?;
        let request_id = envelope.request_id.clone();
        let accepts_response_chunks_v1 =
            envelope.version == IPC_VERSION && envelope.accepts_ipc_response_chunks_v1;
        if envelope.version != IPC_VERSION {
            return Ok(HandledResponse {
                payload: response_error(
                    &envelope.request_id,
                    format!(
                        "IPC version mismatch: got {}, expected {}",
                        envelope.version, IPC_VERSION
                    ),
                )?,
                request_id,
                accepts_response_chunks_v1: false,
            });
        }

        let payload = match envelope.method.as_str() {
            METHOD_PING => {
                let req = body_as::<crate::ipc_types::PingRequest>(envelope.body)?;
                let ready = self.backend.health_ready().await;
                let bundle_config_hash = self
                    .bundle_config_hash
                    .read()
                    .expect("bundle hash lock poisoned")
                    .clone();
                response_ok(
                    &envelope.request_id,
                    crate::ipc_types::PingResponse {
                        timestamp_ms: req.timestamp_ms,
                        worker_id: self.config.worker_id.clone(),
                        ready,
                        bundle_config_hash,
                        loaded_models: self.backend.loaded_models(),
                    },
                )
            }
            METHOD_WORKER_CAPABILITIES => response_ok(&envelope.request_id, {
                let supported_models = self.backend.supported_models();
                let loaded_models = self.backend.loaded_models();
                WorkerCapabilitiesResponse {
                    has_generation_models: false,
                    generation_models: Vec::new(),
                    supported_models: supported_models.clone(),
                    loaded_models,
                }
            }),
            METHOD_ENSURE_MODEL_READY => {
                let req = body_as::<EnsureModelReadyRequest>(envelope.body)?;
                let resp = self.backend.ensure_model_ready(&req.model_id).await;
                response_ok(&envelope.request_id, resp)
            }
            METHOD_PROCESS_ENCODE_BATCH => {
                let req = body_as::<ProcessEncodeBatchRequest>(envelope.body)?;
                let resp = self.backend.process_encode_batch(req).await;
                response_ok(&envelope.request_id, resp)
            }
            METHOD_PROCESS_SCORE_BATCH => {
                let req = body_as::<ProcessScoreBatchRequest>(envelope.body)?;
                let resp = self.backend.process_score_batch(req).await;
                response_ok(&envelope.request_id, resp)
            }
            METHOD_PROCESS_EXTRACT_BATCH => {
                let req = body_as::<ProcessExtractBatchRequest>(envelope.body)?;
                response_ok(&envelope.request_id, unsupported_extract(req))
            }
            METHOD_RUN_BATCH => {
                let req = body_as::<crate::ipc_types::RunBatchRequest>(envelope.body)?;
                let op_label = req
                    .items
                    .first()
                    .map(|item| item.op.clone())
                    .unwrap_or_default();
                let model_id = req.model_id.clone();
                let batch_id = req.batch_id;
                let batch_size = req.items.len();
                let span = tracing::info_span!(
                    "worker.run_batch",
                    otel.name = "worker.run_batch",
                    sie.op = %op_label,
                    sie.model = %model_id,
                    sie.batch_id = batch_id,
                    sie.batch_size = batch_size,
                );
                {
                    let mut linked: HashSet<&str> = HashSet::new();
                    if let Some(parent) = req.items.iter().find(|item| {
                        remote_span_context(item.traceparent.as_deref(), item.tracestate.as_deref())
                            .is_some()
                    }) {
                        let tp = parent.traceparent.as_deref();
                        let ts = parent.tracestate.as_deref();
                        let _ = span.set_parent(extract_context_from_w3c(tp, ts));
                        if let Some(tp) = tp {
                            linked.insert(tp);
                        }
                    }
                    for item in &req.items {
                        let Some(tp) = item.traceparent.as_deref() else {
                            continue;
                        };
                        if !linked.insert(tp) {
                            continue;
                        }
                        if let Some(sc) = remote_span_context(Some(tp), item.tracestate.as_deref())
                        {
                            span.add_link(sc);
                        }
                    }
                }
                let resp = self.backend.run_batch(req).instrument(span).await;
                response_ok(&envelope.request_id, resp)
            }
            METHOD_APPLY_MODEL_CONFIG => {
                let req = body_as::<ApplyModelConfigRequest>(envelope.body)?;
                if let Err(error) = self.backend.apply_model_config(&req) {
                    return Ok(HandledResponse {
                        payload: response_error(&envelope.request_id, error.to_string())?,
                        request_id,
                        accepts_response_chunks_v1,
                    });
                }
                {
                    let mut hash = self
                        .bundle_config_hash
                        .write()
                        .expect("bundle hash lock poisoned");
                    *hash = req.bundle_config_hash.clone();
                }
                response_ok(
                    &envelope.request_id,
                    ApplyModelConfigResponse {
                        applied: true,
                        bundle_config_hash: req.bundle_config_hash,
                        config_version: req.epoch,
                    },
                )
            }
            METHOD_REPLACE_MODEL_CONFIGS => {
                let req = body_as::<ReplaceModelConfigsRequest>(envelope.body)?;
                let applied_models = match self.backend.replace_model_configs(&req) {
                    Ok(models) => models,
                    Err(error) => {
                        return Ok(HandledResponse {
                            payload: response_error(&envelope.request_id, error.to_string())?,
                            request_id,
                            accepts_response_chunks_v1,
                        });
                    }
                };
                let applied_profiles = applied_profiles(&applied_models);
                {
                    let mut hash = self
                        .bundle_config_hash
                        .write()
                        .expect("bundle hash lock poisoned");
                    *hash = req.bundle_config_hash.clone();
                }
                response_ok(
                    &envelope.request_id,
                    ReplaceModelConfigsResponse {
                        applied: true,
                        bundle_config_hash: req.bundle_config_hash,
                        config_version: req.epoch,
                        applied_models,
                        applied_profiles,
                    },
                )
            }
            METHOD_SET_PINNED_MODELS => {
                let req = body_as::<SetPinnedModelsRequest>(envelope.body)?;
                let resp = self.backend.set_pinned_models(&req);
                response_ok(&envelope.request_id, resp)
            }
            METHOD_DRAIN => {
                let _req = body_as::<DrainRequest>(envelope.body)?;
                response_ok(&envelope.request_id, DrainResponse { acknowledged: true })
            }
            METHOD_SIGNAL_GENERATE_CANCEL => response_ok(
                &envelope.request_id,
                SignalGenerateCancelResponse { matched: false },
            ),
            METHOD_PROCESS_GENERATE => response_error(
                &envelope.request_id,
                "sie-server-rust does not support generation".to_string(),
            ),
            other => response_error(&envelope.request_id, format!("unknown IPC method {other}")),
        }?;
        Ok(HandledResponse {
            payload,
            request_id,
            accepts_response_chunks_v1,
        })
    }

    #[cfg(test)]
    async fn handle_frame(&self, frame: &[u8]) -> Result<Vec<u8>, IpcServerError> {
        Ok(self.handle_request_frame(frame).await?.payload)
    }

    async fn run_http(self) -> Result<(), IpcServerError> {
        let listener =
            TcpListener::bind((self.config.http_host.as_str(), self.config.http_port)).await?;
        let addr = listener.local_addr()?;
        info!(addr = %addr, "sie-server-rust HTTP health server listening");

        loop {
            let (mut socket, _) = listener.accept().await?;
            let server = self.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 1024];
                let n = match socket.read(&mut buf).await {
                    Ok(n) => n,
                    Err(_) => return,
                };
                let request = String::from_utf8_lossy(&buf[..n]);
                let path = request
                    .lines()
                    .next()
                    .and_then(|line| line.split_whitespace().nth(1))
                    .unwrap_or("/");

                let (status, content_type, body) = match path {
                    "/healthz" | "/livez" => ("200 OK", "text/plain", "ok".to_string()),
                    "/readyz" => {
                        if server.backend.health_ready().await {
                            ("200 OK", "text/plain", "ready".to_string())
                        } else {
                            (
                                "503 Service Unavailable",
                                "text/plain",
                                "not ready".to_string(),
                            )
                        }
                    }
                    _ => ("404 Not Found", "text/plain", "not found".to_string()),
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = socket.write_all(response.as_bytes()).await;
            });
        }
    }
}

struct HandledResponse {
    payload: Vec<u8>,
    request_id: String,
    accepts_response_chunks_v1: bool,
}

enum ResponsePlan {
    Legacy(Vec<u8>),
    Chunked {
        payload: Vec<u8>,
        request_id: String,
        transfer_digest: [u8; 32],
        chunk_count: u32,
        chunk_payload_bytes: usize,
    },
}

/// Serialize-only response-chunk view. Borrowing the complete response slice
/// avoids allocating a second `Vec<u8>` for every physical chunk before
/// `rmp_serde` writes the UDS frame.
#[derive(Serialize)]
struct IpcResponseChunkV1Ref<'a> {
    version: u32,
    request_id: &'a str,
    #[serde(with = "serde_bytes")]
    transfer_digest: &'a [u8],
    chunk_index: u32,
    chunk_count: u32,
    total_bytes: u64,
    #[serde(with = "serde_bytes")]
    payload: &'a [u8],
    kind: &'a str,
}

fn plan_response(response: HandledResponse) -> Result<ResponsePlan, IpcServerError> {
    plan_response_with_limits(
        response,
        MAX_FRAME_BYTES,
        IPC_RESPONSE_CHUNK_PAYLOAD_BYTES,
        MAX_CHUNKED_IPC_RESPONSE_BYTES,
        MAX_IPC_RESPONSE_CHUNKS,
    )
}

fn plan_response_with_limits(
    response: HandledResponse,
    max_frame_bytes: usize,
    chunk_payload_bytes: usize,
    max_chunked_bytes: usize,
    max_chunks: u32,
) -> Result<ResponsePlan, IpcServerError> {
    if response.payload.len() <= max_frame_bytes {
        return Ok(ResponsePlan::Legacy(response.payload));
    }
    if !response.accepts_response_chunks_v1 {
        return Ok(ResponsePlan::Legacy(response_error(
            &response.request_id,
            "IPC response exceeds the legacy frame limit; response chunking v1 was not negotiated"
                .to_string(),
        )?));
    }

    if chunk_payload_bytes == 0 || max_chunks == 0 {
        return Ok(ResponsePlan::Legacy(response_error(
            &response.request_id,
            "IPC response exceeds the negotiated response chunking v1 bounds".to_string(),
        )?));
    }

    let chunk_count = response.payload.len().div_ceil(chunk_payload_bytes);
    if response.payload.len() > max_chunked_bytes || chunk_count > max_chunks as usize {
        return Ok(ResponsePlan::Legacy(response_error(
            &response.request_id,
            "IPC response exceeds the negotiated response chunking v1 bounds".to_string(),
        )?));
    }

    Ok(ResponsePlan::Chunked {
        transfer_digest: Sha256::digest(&response.payload).into(),
        payload: response.payload,
        request_id: response.request_id,
        chunk_count: chunk_count as u32,
        chunk_payload_bytes,
    })
}

fn encode_response_chunk(
    payload: &[u8],
    request_id: &str,
    transfer_digest: &[u8; 32],
    chunk_index: u32,
    chunk_count: u32,
    chunk_payload_bytes: usize,
) -> Result<Vec<u8>, IpcServerError> {
    let start = chunk_index as usize * chunk_payload_bytes;
    let end = (start + chunk_payload_bytes).min(payload.len());
    Ok(rmp_serde::to_vec_named(&IpcResponseChunkV1Ref {
        version: IPC_VERSION,
        request_id,
        transfer_digest,
        chunk_index,
        chunk_count,
        total_bytes: payload.len() as u64,
        payload: &payload[start..end],
        kind: IPC_RESPONSE_CHUNK_KIND_V1,
    })?)
}

async fn write_response(
    stream: &mut UnixStream,
    response: HandledResponse,
) -> Result<(), IpcServerError> {
    match plan_response(response)? {
        ResponsePlan::Legacy(payload) => write_frame(stream, &payload).await,
        ResponsePlan::Chunked {
            payload,
            request_id,
            transfer_digest,
            chunk_count,
            chunk_payload_bytes,
        } => {
            for chunk_index in 0..chunk_count {
                let frame = encode_response_chunk(
                    &payload,
                    &request_id,
                    &transfer_digest,
                    chunk_index,
                    chunk_count,
                    chunk_payload_bytes,
                )?;
                write_frame(stream, &frame).await?;
            }
            Ok(())
        }
    }
}

fn body_as<T: DeserializeOwned>(body: serde_json::Value) -> Result<T, IpcServerError> {
    Ok(serde_json::from_value(body)?)
}

fn applied_profiles(applied_models: &[String]) -> Vec<String> {
    let mut profiles = applied_models
        .iter()
        .map(|model_id| {
            model_id
                .rsplit_once(':')
                .map_or("default", |(_, profile)| profile)
                .to_string()
        })
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    profiles.sort();
    profiles
}

fn response_ok<T: Serialize>(request_id: &str, body: T) -> Result<Vec<u8>, IpcServerError> {
    let envelope = ResponseEnvelope {
        version: IPC_VERSION,
        request_id: request_id.to_string(),
        ok: true,
        body: Some(body),
        error: None,
    };
    Ok(rmp_serde::to_vec_named(&envelope)?)
}

fn response_error(request_id: &str, error: String) -> Result<Vec<u8>, IpcServerError> {
    let envelope: ResponseEnvelope<serde_json::Value> = ResponseEnvelope {
        version: IPC_VERSION,
        request_id: request_id.to_string(),
        ok: false,
        body: None,
        error: Some(error),
    };
    Ok(rmp_serde::to_vec_named(&envelope)?)
}

async fn read_frame(stream: &mut UnixStream) -> Result<Vec<u8>, IpcServerError> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf);
    if len as usize > MAX_FRAME_BYTES {
        return Err(IpcServerError::FrameTooLarge(len));
    }
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await?;
    Ok(buf)
}

async fn write_frame(stream: &mut UnixStream, payload: &[u8]) -> Result<(), IpcServerError> {
    if payload.len() > MAX_FRAME_BYTES {
        return Err(IpcServerError::FrameTooLarge(
            u32::try_from(payload.len()).unwrap_or(u32::MAX),
        ));
    }
    let len = u32::try_from(payload.len()).map_err(|_| IpcServerError::FrameTooLarge(u32::MAX))?;
    stream.write_all(&len.to_be_bytes()).await?;
    stream.write_all(payload).await?;
    Ok(())
}

async fn unlink_stale_socket(path: &Path) -> Result<(), std::io::Error> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(unix)]
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};

    let mut terminate = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate.recv() => {}
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn unsupported_extract(req: ProcessExtractBatchRequest) -> BatchOutcome {
    BatchOutcome {
        outcomes: req
            .items
            .into_iter()
            .map(|item| crate::ipc_types::ItemOutcome {
                work_item_id: item.work_item_id,
                request_id: item.request_id,
                item_index: item.item_index,
                disposition: crate::ipc_types::Disposition::PublishErrorAndAck,
                nak_delay_ms: None,
                result_msgpack: Vec::new(),
                error: Some("sie-server-rust native backend does not support extract".to_string()),
                error_code: Some("native_unsupported_operation".to_string()),
                inference_ms: None,
                tokenization_ms: None,
                postprocessing_ms: None,
                raw_output: None,
                units: None,
            })
            .collect(),
        batched_f16_multivectors: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde::Serialize;
    use serde_json::json;

    use super::*;
    use crate::candle_backend::{CandleBackend, CandleBackendConfig};
    use crate::ipc_types::{
        IpcResponseChunkV1, METHOD_APPLY_MODEL_CONFIG, METHOD_PING, METHOD_REPLACE_MODEL_CONFIGS,
        METHOD_SET_PINNED_MODELS, METHOD_WORKER_CAPABILITIES,
    };

    #[derive(Serialize)]
    struct TestRequestEnvelope {
        version: u32,
        method: String,
        request_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        accepts_ipc_response_chunks_v1: Option<bool>,
        body: serde_json::Value,
    }

    fn test_server() -> IpcServer {
        IpcServer::new(
            IpcServerConfig {
                socket_path: PathBuf::from("/tmp/sie-server-rust-test.sock"),
                worker_id: "worker-test".to_string(),
                bundle: "candle".to_string(),
                http_host: "127.0.0.1".to_string(),
                http_port: 0,
            },
            NativeBackend::Candle(CandleBackend::new(CandleBackendConfig::new(64, true, 1))),
        )
    }

    fn request_frame(method: &str, body: serde_json::Value) -> Vec<u8> {
        rmp_serde::to_vec_named(&TestRequestEnvelope {
            version: IPC_VERSION,
            method: method.to_string(),
            request_id: "req-1".to_string(),
            accepts_ipc_response_chunks_v1: None,
            body,
        })
        .expect("encode request")
    }

    #[test]
    fn applied_profile_ids_are_bounded_to_successful_routes() {
        assert_eq!(
            applied_profiles(&[
                "BAAI/bge-m3".to_string(),
                "Qwen/Qwen3.6-27B:candle".to_string(),
                "other/model:candle".to_string(),
            ]),
            vec!["candle".to_string(), "default".to_string()]
        );
    }

    #[tokio::test]
    async fn replace_model_configs_reports_applied_profiles() {
        let server = test_server();
        let response = server
            .handle_frame(&request_frame(
                METHOD_REPLACE_MODEL_CONFIGS,
                json!({
                    "bundle_id": "candle",
                    "epoch": 1,
                    "bundle_config_hash": "hash-candle",
                    "models": [{
                        "model_id": "BAAI/bge-m3",
                        "model_config": concat!(
                            "sie_id: BAAI/bge-m3\n",
                            "profiles:\n",
                            "  candle:\n",
                            "    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter\n",
                        ),
                    }],
                }),
            ))
            .await
            .expect("replace config");
        let envelope: ResponseEnvelope<serde_json::Value> =
            rmp_serde::from_slice(&response).expect("decode replace response");
        assert!(envelope.ok, "unexpected IPC error: {:?}", envelope.error);
        let body = envelope.body.expect("replace response body");
        assert_eq!(body.get("applied_profiles"), Some(&json!(["candle"])));
    }

    fn request_frame_with_response_chunks(method: &str, body: serde_json::Value) -> Vec<u8> {
        rmp_serde::to_vec_named(&TestRequestEnvelope {
            version: IPC_VERSION,
            method: method.to_string(),
            request_id: "req-chunked".to_string(),
            accepts_ipc_response_chunks_v1: Some(true),
            body,
        })
        .expect("encode chunk-capable request")
    }

    #[tokio::test]
    async fn request_chunk_capability_is_explicit_and_defaults_false() {
        let server = test_server();
        let legacy = server
            .handle_request_frame(&request_frame(METHOD_PING, json!({"timestamp_ms": 1.0})))
            .await
            .unwrap();
        assert!(!legacy.accepts_response_chunks_v1);

        let negotiated = server
            .handle_request_frame(&request_frame_with_response_chunks(
                METHOD_PING,
                json!({"timestamp_ms": 1.0}),
            ))
            .await
            .unwrap();
        assert!(negotiated.accepts_response_chunks_v1);
        assert_eq!(negotiated.request_id, "req-chunked");
    }

    #[test]
    fn small_negotiated_response_preserves_exact_legacy_bytes() {
        let payload = response_ok("r", json!({"value": [1, 2, 3]})).unwrap();
        let plan = plan_response_with_limits(
            HandledResponse {
                payload: payload.clone(),
                request_id: "r".to_string(),
                accepts_response_chunks_v1: true,
            },
            512,
            128,
            4096,
            64,
        )
        .unwrap();

        match plan {
            ResponsePlan::Legacy(actual) => assert_eq!(actual, payload),
            ResponsePlan::Chunked { .. } => panic!("small response must remain one legacy frame"),
        }
    }

    #[test]
    fn negotiated_response_chunks_reassemble_exact_serialized_bytes() {
        let payload = response_ok("r", json!({"blob": "x".repeat(1200)})).unwrap();
        let plan = plan_response_with_limits(
            HandledResponse {
                payload: payload.clone(),
                request_id: "r".to_string(),
                accepts_response_chunks_v1: true,
            },
            512,
            128,
            4096,
            64,
        )
        .unwrap();

        let ResponsePlan::Chunked {
            payload: planned_payload,
            request_id,
            transfer_digest,
            chunk_count,
            chunk_payload_bytes,
        } = plan
        else {
            panic!("oversized negotiated response must be chunked");
        };
        let mut reconstructed = Vec::new();
        for chunk_index in 0..chunk_count {
            let frame = encode_response_chunk(
                &planned_payload,
                &request_id,
                &transfer_digest,
                chunk_index,
                chunk_count,
                chunk_payload_bytes,
            )
            .unwrap();
            assert!(frame.len() <= 512);
            let chunk: IpcResponseChunkV1 = rmp_serde::from_slice(&frame).unwrap();
            assert_eq!(chunk.kind, IPC_RESPONSE_CHUNK_KIND_V1);
            assert_eq!(chunk.request_id, "r");
            assert_eq!(chunk.chunk_index, chunk_index);
            assert_eq!(chunk.chunk_count, chunk_count);
            assert_eq!(chunk.total_bytes, payload.len() as u64);
            assert_eq!(chunk.transfer_digest, transfer_digest);
            reconstructed.extend_from_slice(&chunk.payload);
        }
        assert_eq!(reconstructed, payload);
        let reconstructed_digest: [u8; 32] = Sha256::digest(&reconstructed).into();
        assert_eq!(reconstructed_digest, transfer_digest);
    }

    #[test]
    fn borrowed_response_chunk_encoding_matches_owned_wire_at_bin_boundaries() {
        for payload_len in [255, 256, 65_535, 65_536] {
            let payload = vec![7; payload_len];
            let digest = [9_u8; 32];
            let borrowed = IpcResponseChunkV1Ref {
                version: IPC_VERSION,
                request_id: "r",
                transfer_digest: &digest,
                chunk_index: 0,
                chunk_count: 1,
                total_bytes: payload_len as u64,
                payload: &payload,
                kind: IPC_RESPONSE_CHUNK_KIND_V1,
            };
            let borrowed_bytes = rmp_serde::to_vec_named(&borrowed).unwrap();
            let owned = IpcResponseChunkV1 {
                version: IPC_VERSION,
                request_id: "r".to_string(),
                transfer_digest: digest.to_vec(),
                chunk_index: 0,
                chunk_count: 1,
                total_bytes: payload_len as u64,
                payload,
                kind: IPC_RESPONSE_CHUNK_KIND_V1.to_string(),
            };

            assert_eq!(
                borrowed_bytes,
                rmp_serde::to_vec_named(&owned).unwrap(),
                "payload length {payload_len}"
            );
        }
    }

    #[test]
    fn absent_capability_and_overbound_response_return_compact_legacy_error() {
        for (accepts, max_chunked, max_chunks) in
            [(false, 4096, 64), (true, 800, 64), (true, 4096, 4)]
        {
            let payload = response_ok("r", json!({"blob": "x".repeat(1200)})).unwrap();
            let plan = plan_response_with_limits(
                HandledResponse {
                    payload,
                    request_id: "r".to_string(),
                    accepts_response_chunks_v1: accepts,
                },
                512,
                128,
                max_chunked,
                max_chunks,
            )
            .unwrap();
            let ResponsePlan::Legacy(error) = plan else {
                panic!("unsupported/overbound response must be a compact legacy error");
            };
            assert!(error.len() <= 512);
            let error: ResponseEnvelope<serde_json::Value> = rmp_serde::from_slice(&error).unwrap();
            assert!(!error.ok);
            assert_eq!(error.request_id, "r");
        }
    }

    #[test]
    fn invalid_chunk_planning_limits_return_compact_legacy_error() {
        for (chunk_payload_bytes, max_chunks) in [(0, 64), (128, 0)] {
            let payload = response_ok("r", json!({"blob": "x".repeat(1200)})).unwrap();
            let plan = plan_response_with_limits(
                HandledResponse {
                    payload,
                    request_id: "r".to_string(),
                    accepts_response_chunks_v1: true,
                },
                512,
                chunk_payload_bytes,
                4096,
                max_chunks,
            )
            .unwrap();
            let ResponsePlan::Legacy(error) = plan else {
                panic!("invalid planning limits must produce a compact legacy error");
            };
            assert!(error.len() <= 512);
            let error: ResponseEnvelope<serde_json::Value> = rmp_serde::from_slice(&error).unwrap();
            assert!(!error.ok);
            assert_eq!(error.request_id, "r");
        }
    }

    #[tokio::test]
    async fn ping_reports_candle_readiness_after_config_apply() {
        let server = test_server();

        let before = server
            .handle_frame(&request_frame(METHOD_PING, json!({"timestamp_ms": 123.0})))
            .await
            .expect("ping before config");
        let before: ResponseEnvelope<serde_json::Value> =
            rmp_serde::from_slice(&before).expect("decode ping before config");
        assert!(before.ok, "unexpected IPC error: {:?}", before.error);
        let body = before.body.expect("ping body");
        assert_eq!(body.get("worker_id"), Some(&json!("worker-test")));
        assert_eq!(body.get("ready"), Some(&json!(false)));

        let apply = server
            .handle_frame(&request_frame(
                METHOD_APPLY_MODEL_CONFIG,
                json!({
                    "bundle_id": "candle",
                    "model_id": "BAAI/bge-m3",
                    "epoch": 1,
                    "bundle_config_hash": "hash-candle",
                    "profiles_added": ["candle"],
                    "model_config": r#"
sie_id: BAAI/bge-m3
profiles:
  candle:
    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter
"#,
                }),
            ))
            .await
            .expect("apply config");
        let apply: ResponseEnvelope<serde_json::Value> =
            rmp_serde::from_slice(&apply).expect("decode apply response");
        assert!(apply.ok, "unexpected IPC error: {:?}", apply.error);

        let after = server
            .handle_frame(&request_frame(METHOD_PING, json!({"timestamp_ms": 456.0})))
            .await
            .expect("ping after config");
        let after: ResponseEnvelope<serde_json::Value> =
            rmp_serde::from_slice(&after).expect("decode ping after config");
        assert!(after.ok, "unexpected IPC error: {:?}", after.error);
        let body = after.body.expect("ping body");
        assert_eq!(body.get("ready"), Some(&json!(true)));
        assert_eq!(body.get("bundle_config_hash"), Some(&json!("hash-candle")));
    }

    #[tokio::test]
    async fn set_pinned_models_frame_is_dispatched() {
        let server = test_server();
        let frame = request_frame(
            METHOD_SET_PINNED_MODELS,
            json!({
                "models": [
                    " BAAI/bge-m3:default ",
                    "BAAI/bge-m3:CANDLE",
                    ""
                ]
            }),
        );

        let response = server.handle_frame(&frame).await.expect("handle frame");
        let envelope: ResponseEnvelope<serde_json::Value> =
            rmp_serde::from_slice(&response).expect("decode response");

        assert!(envelope.ok, "unexpected IPC error: {:?}", envelope.error);
        assert_eq!(envelope.request_id, "req-1");
        let body = envelope.body.expect("response body");
        assert_eq!(
            body.get("applied").and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            body.get("pinned_count").and_then(serde_json::Value::as_u64),
            Some(2)
        );
    }

    #[tokio::test]
    async fn worker_capabilities_still_reports_supported_models_after_pinned_rpc() {
        let server = test_server();
        let response = server
            .handle_frame(&request_frame(METHOD_WORKER_CAPABILITIES, json!({})))
            .await
            .expect("worker capabilities before config");
        let envelope: ResponseEnvelope<serde_json::Value> =
            rmp_serde::from_slice(&response).expect("decode response");

        assert!(envelope.ok, "unexpected IPC error: {:?}", envelope.error);
        let body = envelope.body.expect("response body");
        assert_eq!(body.get("supported_models"), Some(&json!([])));

        let apply_frame = request_frame(
            METHOD_APPLY_MODEL_CONFIG,
            json!({
                "bundle_id": "candle",
                "model_id": "BAAI/bge-m3",
                "epoch": 1,
                "bundle_config_hash": "hash",
                "profiles_added": ["candle"],
                "model_config": concat!(
                    "sie_id: BAAI/bge-m3\n",
                    "profiles:\n",
                    "  candle:\n",
                    "    adapter_path: sie_server_rust.adapters.candle:CandleEmbeddingAdapter\n",
                ),
            }),
        );
        let apply_response = server
            .handle_frame(&apply_frame)
            .await
            .expect("apply catalog config");
        let apply_envelope: ResponseEnvelope<serde_json::Value> =
            rmp_serde::from_slice(&apply_response).expect("decode apply response");
        assert!(
            apply_envelope.ok,
            "unexpected IPC error: {:?}",
            apply_envelope.error
        );

        let frame = request_frame(
            METHOD_SET_PINNED_MODELS,
            json!({"models": ["not/catalogued:candle"]}),
        );
        let _ = server.handle_frame(&frame).await.expect("set pinned");

        let response = server
            .handle_frame(&request_frame(METHOD_WORKER_CAPABILITIES, json!({})))
            .await
            .expect("worker capabilities");
        let envelope: ResponseEnvelope<serde_json::Value> =
            rmp_serde::from_slice(&response).expect("decode response");

        assert!(envelope.ok, "unexpected IPC error: {:?}", envelope.error);
        let body = envelope.body.expect("response body");
        assert_eq!(
            body.get("supported_models"),
            Some(&json!(["BAAI/bge-m3:candle"]))
        );
        assert_eq!(body.get("loaded_models"), Some(&json!([])));
    }
}
