//! Server-Sent Events (SSE) streaming response for the inference
//! endpoints.
//!
//! This module forwards per-chunk envelopes — emitted by the worker
//! through the streaming pipeline already documented in
//! :mod:`crate::queue::streaming` — to the HTTP client as
//! ``text/event-stream`` events as they arrive. It does **not**
//! introduce a second streaming pipeline; the chunks come off the
//! same broadcast tap installed by
//! :meth:`WorkPublisher::publish_generate_streaming_sse`, so:
//!
//! * The attempt-id stale-chunk drop logic in
//!   :meth:`StreamCollector::apply` still applies — stale chunks never
//!   reach the tap.
//! * The first-chunk-timeout pool-republish path in
//!   :func:`proxy::run_streaming_generate` is mirrored here under the
//!   same three-tier timeout taxonomy. Because the broadcast receiver
//!   is created before the work item is published, chunks from a
//!   second (post-republish) attempt land in the same stream the SSE
//!   handler is already consuming — no resubscribe needed.
//! * The :class:`StreamCancelGuard` drop-guard fires automatically
//!   when the SSE response future is dropped by axum (HTTP client
//!   disconnect), publishing the cancel signal and removing the
//!   collector exactly as on the non-streaming path.
//!
//! Wire shapes:
//!
//! * **Chat** (``/v1/chat/completions`` with ``stream: true``): emits
//!   OpenAI-compatible ``chat.completion.chunk`` events. Final chunk
//!   carries ``finish_reason`` and, when
//!   ``stream_options.include_usage == true``, is followed by a
//!   usage-only chunk. Always terminated by ``data: [DONE]\n\n``.
//! * **Completion** (``/v1/completions`` with ``stream: true``): emits
//!   ``text_completion`` events with the same ``stream_options.include_usage``
//!   contract as chat.
//! * **Generate** (``/v1/generate/{model}`` with ``stream: true``):
//!   emits the SIE-native shape
//!   ``{request_id, seq, text_delta, done, usage?, finish_reason?, timing?}``.
//!   Same ``[DONE]`` terminator; its terminal chunk carries ``usage`` inline,
//!   so it needs no separate usage-only chunk.
//!
//! A streamed response cannot carry usage in headers — they are flushed with
//! the first byte — so the final chunk is the only surface on which a metered
//! stream can report what it consumed.
//!
//! Error mid-stream: a worker-emitted error chunk (`ChunkEnvelope.error`)
//! is surfaced as a final event carrying an ``error`` block alongside
//! the normal chunk fields, followed by ``[DONE]`` and connection
//! close. Same shape for timeouts originating in the gateway.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use tokio::sync::broadcast;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, info, warn};

use crate::observability::metrics as telemetry;
use crate::queue::dispatch::{PendingDispatchKind, WorkDispatcher};
use crate::queue::publisher;
use crate::queue::streaming::{
    is_lower_sha256, ChunkEnvelope, ChunkError, StreamOutcome, StreamOutcomeOrigin,
};
use crate::server::AppState;
use crate::state::demand_tracker::{DemandTracker, PhysicalLane};

/// Wire shape selector — chat vs SIE-native generate.
#[derive(Debug, Clone, Copy)]
pub enum SseEndpoint {
    /// OpenAI-compatible chat completions chunk shape.
    Chat {
        /// Whether to emit a trailing usage-only chunk before ``[DONE]``.
        include_usage: bool,
    },
    /// SIE-native generate chunk shape.
    Generate,
    /// OpenAI legacy Completions chunk shape (`object: "text_completion"`,
    /// `choices[0].text`). Single-candidate (completions rejects `n>1`).
    Completion {
        /// Whether to emit a trailing usage-only chunk before ``[DONE]``.
        include_usage: bool,
    },
}

/// Parameters passed from the chat / generate handler to
/// :func:`build_sse_response`.
pub struct SseParams<'a> {
    pub state: &'a AppState,
    pub work_publisher: Arc<dyn WorkDispatcher>,
    pub physical_lane: PhysicalLane,
    /// DISPLAY id (the requested model) — surfaced in the streamed chunk
    /// ``model`` field and the routing/timeout metric labels.
    pub model: String,
    /// DISPATCH id — the grammar ``:no-spec`` variant when routing fired,
    /// otherwise equal to ``model``. Governs ONLY the NATS subject + work item
    /// (worker selection / publish target) so the request runs on the profile
    /// that enforces the grammar. See ``proxy::route_grammar_to_profile``.
    pub dispatch_model: String,
    pub bundle: String,
    pub engine: String,
    pub gpu: String,
    pub pool: String,
    pub admission_pool: String,
    pub bundle_config_hash: String,
    pub work_params: publisher::WorkParams,
    pub endpoint: SseEndpoint,
}

/// Build the SSE response. Publishes the work item, subscribes to
/// the per-chunk broadcast tap, and returns an axum SSE response
/// streaming events to the client.
///
/// Errors that occur **before** any chunk has been sent
/// (queue-publish failures, etc.) are surfaced as a regular JSON
/// error response (matching the non-streaming envelope). Errors
/// after the first byte goes out — timeouts, worker errors,
/// inter-chunk stalls — are surfaced **inside** the SSE stream as
/// a final error chunk + ``[DONE]``.
pub async fn build_sse_response(params: SseParams<'_>) -> Response {
    let SseParams {
        state,
        work_publisher,
        physical_lane,
        model,
        dispatch_model,
        bundle,
        engine,
        gpu,
        pool,
        admission_pool,
        bundle_config_hash,
        work_params,
        endpoint,
    } = params;

    // Resolve the routing key & target the same way
    // `run_streaming_generate` does. We replicate this here (rather
    // than extracting a shared helper) because the SSE path takes
    // ownership of the broadcast receiver returned by
    // `publish_generate_streaming_sse` and threads it through the
    // event stream — a shared helper would have to express both the
    // (no-tap, oneshot-only) and (with-tap, broadcast+oneshot)
    // returns, which clutters the type signature for no benefit.
    let resolved_key = match work_params.generate.as_ref() {
        Some(g) => crate::routing::key::resolve_from_generate(g),
        None => crate::routing::key::RoutingKeyResolved {
            hash: None,
            source: crate::routing::key::KeySource::None,
            #[cfg(feature = "raw-routing-logs")]
            raw_for_debug: None,
        },
    };
    let (target, pool_fallback_lane_worker_count) = if resolved_key.hash.is_none() {
        (
            publisher::PublishTarget::Pool {
                pool: pool.clone(),
                machine_profile: gpu.clone(),
                bundle: bundle.clone(),
                model: dispatch_model.clone(),
            },
            0,
        )
    } else {
        let admitted_worker_names = state
            .pool_manager
            .admitted_worker_names_for_capped_lane(&admission_pool, &gpu, &bundle)
            .await;
        let fallback_lane_worker_count = state.registry.pool_fallback_lane_worker_count(
            &pool,
            &gpu,
            &bundle,
            &bundle_config_hash,
            admitted_worker_names.as_ref(),
        );
        let ring = state.registry.ring_snapshot_for_admitted(
            &dispatch_model,
            &pool,
            &gpu,
            &bundle,
            &bundle_config_hash,
            admitted_worker_names.as_ref(),
        );
        match crate::routing::pick_worker(&ring, &resolved_key) {
            Some(worker_id) => (
                publisher::PublishTarget::Worker {
                    pool: pool.clone(),
                    machine_profile: gpu.clone(),
                    bundle: bundle.clone(),
                    model: dispatch_model.clone(),
                    worker_id: worker_id.to_string(),
                },
                fallback_lane_worker_count,
            ),
            None => (
                publisher::PublishTarget::Pool {
                    pool: pool.clone(),
                    machine_profile: gpu.clone(),
                    bundle: bundle.clone(),
                    model: dispatch_model.clone(),
                },
                fallback_lane_worker_count,
            ),
        }
    };
    let was_direct_dispatched = matches!(target, publisher::PublishTarget::Worker { .. });

    let (request_id, outcome_rx, chunk_rx, durability) = match work_publisher
        .publish_generate_streaming_sse(
            target,
            &model,
            &engine,
            &bundle_config_hash,
            &work_params,
            &admission_pool,
        )
        .await
    {
        Ok(triple) => triple,
        Err(e) => {
            // Pre-stream publish failure — surface as a regular JSON
            // error response (mirrors `build_streaming_error_response`
            // for the `PublishFailed` arm).
            let lower = e.to_lowercase();
            let retry_after = if lower.contains("no consumers") {
                telemetry::record_rejected_request(
                    state.demand_tracker.as_ref(),
                    &physical_lane,
                    "no_consumers",
                );
                Some(crate::handlers::proxy::PROVISIONING_RETRY_AFTER)
            } else {
                // Shared with the buffered and streaming paths rather than
                // re-derived here: this arm used to match only "backpressure"
                // and so missed the broker's own stream-full rejection, which
                // means the same thing.
                crate::handlers::proxy::record_publish_failure(state, &physical_lane, &lower)
            };
            return crate::handlers::proxy::build_streaming_publish_failed_for_sse(&e, retry_after);
        }
    };
    let durability_completion = crate::handlers::proxy::monitor_dispatch_durability(
        Arc::clone(&state.demand_tracker),
        physical_lane.clone(),
        durability,
        Arc::clone(&work_publisher),
        request_id.clone(),
        PendingDispatchKind::Stream,
    );

    // Choose timeouts via the same helper as the non-SSE path; we
    // copy `params` for the helper to inspect max_new_tokens.
    let max_new_tokens = work_params
        .generate
        .as_ref()
        .map(|g| g.max_new_tokens)
        .unwrap_or(512);
    let timeout_config = crate::handlers::proxy::generation_timeout_config(
        state,
        &dispatch_model,
        &work_params,
        max_new_tokens,
    );
    // Per ADR-0003: generation streaming uses the profile/runtime
    // overall_timeout_s as authority. The legacy SIE_GATEWAY_REQUEST_TIMEOUT
    // ceiling is not applied — it would clamp a 300s model-profile overall
    // to the default 30s and make the first-chunk policy unreachable on
    // cold loads.
    let effective_overall = timeout_config.overall;

    // Spawn the SSE driver task. We use an mpsc channel (rather than
    // wrapping the broadcast Receiver directly) so the driver can
    // synthesise terminator events (timeout error, [DONE]) without
    // entangling lifetimes with the broadcast::Receiver stream
    // adapter.
    //
    // Buffer size is sized for worker chunk-batch granularity (~32
    // tokens / 50 ms) plus a safety margin. The previous size of 16
    // filled in ~0.8 s of momentary client stall and then deadlocked
    // the driver's `event_tx.send().await` inside the chunk-recv arm —
    // the broadcast subscription drained into `Lagged` and the request
    // was misclassified as an `inter_chunk_timeout`. 256 gives the
    // client several seconds of slack before head-of-line blocking
    // begins; a still-slow client now gets a clean disconnect detected
    // through the new `closed()` branch in the select below rather
    // than a synthetic timeout.
    let (event_tx, event_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);

    let driver_publisher = Arc::clone(&work_publisher);
    let driver_request_id = request_id.clone();
    let driver_model = model.clone();
    let driver_pool = pool.clone();
    let driver_bundle = bundle.clone();
    let driver_gpu = gpu.clone();
    let driver_demand_tracker = Arc::clone(&state.demand_tracker);
    let driver_physical_lane = physical_lane;
    let stream_chat_id = format!("chatcmpl-{}", request_id);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    tokio::spawn(async move {
        run_sse_driver(SseDriverArgs {
            event_tx,
            chunk_rx,
            outcome_rx,
            durability_completion,
            publisher: driver_publisher,
            demand_tracker: driver_demand_tracker,
            physical_lane: driver_physical_lane,
            request_id: driver_request_id,
            model: driver_model.clone(),
            pool: driver_pool.clone(),
            bundle: driver_bundle,
            gpu: driver_gpu,
            endpoint,
            stream_chat_id,
            created,
            first_chunk_timeout: timeout_config.first_chunk,
            inter_chunk_timeout: timeout_config.inter_chunk,
            overall_timeout: effective_overall,
            was_direct_dispatched,
            pool_fallback_lane_worker_count,
        })
        .await;
    });

    let stream = ReceiverStream::new(event_rx);
    let sse = Sse::new(stream).keep_alive(KeepAlive::default());

    // Build the response and stamp the SIE-specific headers that
    // the non-SSE path also emits.
    let mut response = sse.into_response();
    let headers = response.headers_mut();
    headers.insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-cache"),
    );
    headers.insert(
        axum::http::HeaderName::from_static("x-accel-buffering"),
        axum::http::HeaderValue::from_static("no"),
    );
    let rid_header = match axum::http::HeaderValue::from_str(&request_id) {
        Ok(value) => value,
        Err(err) => {
            warn!(
                request_id = %request_id,
                error = %err,
                "non-ASCII request_id; falling back to empty x-sie-request-id header"
            );
            debug_assert!(false, "request_id must be ASCII");
            axum::http::HeaderValue::from_static("")
        }
    };
    headers.insert(
        axum::http::HeaderName::from_static("x-sie-request-id"),
        rid_header,
    );
    response
}

struct SseDriverArgs {
    event_tx: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    chunk_rx: broadcast::Receiver<ChunkEnvelope>,
    outcome_rx: tokio::sync::oneshot::Receiver<StreamOutcome>,
    durability_completion: tokio::sync::oneshot::Receiver<Result<(), String>>,
    publisher: Arc<dyn WorkDispatcher>,
    demand_tracker: Arc<DemandTracker>,
    physical_lane: PhysicalLane,
    request_id: String,
    model: String,
    pool: String,
    bundle: String,
    gpu: String,
    endpoint: SseEndpoint,
    stream_chat_id: String,
    created: u64,
    first_chunk_timeout: Duration,
    inter_chunk_timeout: Duration,
    overall_timeout: Duration,
    was_direct_dispatched: bool,
    pool_fallback_lane_worker_count: usize,
}

/// How long a disconnected client's stream stays registered waiting for the
/// worker's abort terminal.
///
/// A client disconnect cancels upstream, and the worker answers that cancel
/// with an abort terminal carrying a CONSISTENT count-so-far — evidence a
/// metered dispatcher bills exactly like a completed stream. Removing the
/// collector at `publish_cancel` time threw that terminal away: every ordinary
/// user stop billed zero AND raised the metered `lost_terminal_count` alert,
/// which exists to catch compute SIE could not bill and is worthless once
/// routine behaviour dominates it.
///
/// The window is the bound on how long a departed client's request-local state
/// (collector entry, and on the managed plane the credit hold behind it)
/// outlives the client. Five seconds covers the cancel round-trip plus the
/// worker's in-flight decode step with room to spare, while keeping that
/// lifetime short. On expiry the stream is torn down as a client disconnect —
/// released without charge and WITHOUT the lost-terminal fault alert.
const CLIENT_DISCONNECT_TERMINAL_GRACE: Duration = Duration::from_secs(5);

/// Tear down a stream whose HTTP client went away.
///
/// Cancels upstream, then keeps the collector REGISTERED for
/// [`CLIENT_DISCONNECT_TERMINAL_GRACE`] so the worker's abort terminal can
/// still land and be settled. `publish_cancel` only awaits the transport's ACK
/// of the cancel op, never the worker's terminal, so without this wait the
/// count-so-far raced a teardown it almost always lost.
///
/// `outcome_rx` is `None` when the terminal already resolved — there is nothing
/// left to wait for.
///
/// `stream_succeeded` follows the `#1602` discipline used by the `Lagged` arm:
/// a request whose terminal outcome already resolved `Ok` has completed
/// server-side, so there is nothing to cancel and the cancel would be a
/// redundant wire op. Teardown still runs.
async fn teardown_after_client_disconnect(
    publisher: &Arc<dyn WorkDispatcher>,
    request_id: &str,
    outcome_rx: Option<std::pin::Pin<&mut tokio::sync::oneshot::Receiver<StreamOutcome>>>,
    stream_succeeded: bool,
) {
    if !stream_succeeded {
        publisher.publish_cancel(request_id).await;
    }
    let Some(outcome_rx) = outcome_rx else {
        publisher.drop_pending_stream(request_id);
        return;
    };
    // `Ok(Ok(_))` ONLY. `timeout(..).is_ok()` is also true for `Ok(Err(RecvError))`
    // — the collector's sender dropped without ever sending a terminal — which
    // is the opposite billing signal: no count-so-far was settled, so taking the
    // fault-raising teardown there raises `lost_terminal_count` for exactly the
    // ordinary client stop this grace window exists to keep quiet.
    if matches!(
        tokio::time::timeout(CLIENT_DISCONNECT_TERMINAL_GRACE, outcome_rx).await,
        Ok(Ok(_))
    ) {
        // The terminal landed inside the window. The collector fired its
        // outcome, so a metered dispatcher has already settled the
        // worker-reported count-so-far; this removal is idempotent cleanup.
        debug!(
            request_id = %request_id,
            "cancelled stream reached its worker terminal; count-so-far settled"
        );
        publisher.drop_pending_stream(request_id);
        return;
    }
    // No terminal inside the window. Not a billing fault — the client left and
    // the worker never reported. Release through the disconnect-specific path so
    // metered dispatchers can drop the hold quietly instead of raising the
    // lost-terminal alert.
    info!(
        request_id = %request_id,
        grace_s = CLIENT_DISCONNECT_TERMINAL_GRACE.as_secs_f64(),
        "cancelled stream produced no worker terminal within the grace window; \
         releasing without charge"
    );
    publisher.drop_pending_stream_client_disconnect(request_id);
}

#[derive(Debug, Eq, PartialEq)]
enum TerminalDurabilityWait {
    Confirmed,
    Failed(String),
    MonitorStopped,
    ClientClosed,
    OverallTimeout,
}

/// Wait for a terminal result's initial dispatch ACK without losing the
/// request's outer lifecycle bounds. The durability arm is intentionally
/// biased: if ACK and shutdown/deadline become ready together, the completed
/// durability decision owns the terminal result.
async fn wait_for_terminal_durability(
    completion: &mut tokio::sync::oneshot::Receiver<Result<(), String>>,
    event_tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    overall_deadline: tokio::time::Instant,
) -> TerminalDurabilityWait {
    tokio::select! {
        biased;
        completion = completion => match completion {
            Ok(Ok(())) => TerminalDurabilityWait::Confirmed,
            Ok(Err(error)) => TerminalDurabilityWait::Failed(error),
            Err(_) => TerminalDurabilityWait::MonitorStopped,
        },
        _ = event_tx.closed() => TerminalDurabilityWait::ClientClosed,
        _ = tokio::time::sleep_until(overall_deadline) => {
            TerminalDurabilityWait::OverallTimeout
        }
    }
}

/// Internal SSE driver — loops on the broadcast tap, forwards
/// per-chunk events, and emits the synthetic terminator
/// (``[DONE]`` or error-chunk + ``[DONE]``) when the stream ends.
///
/// Mirrors :func:`proxy::run_streaming_generate`'s timeout taxonomy
/// (first_chunk / inter_chunk / overall) and the pool-republish
/// behaviour, but emits the failure mode **inside** the SSE stream
/// rather than as an HTTP error envelope, because by the time a
/// timeout fires the SSE response has already started (`200 OK` +
/// headers sent).
async fn run_sse_driver(args: SseDriverArgs) {
    let SseDriverArgs {
        event_tx,
        mut chunk_rx,
        outcome_rx,
        mut durability_completion,
        publisher,
        demand_tracker,
        physical_lane,
        request_id,
        model,
        pool,
        bundle,
        gpu,
        endpoint,
        stream_chat_id,
        created,
        first_chunk_timeout,
        inter_chunk_timeout,
        overall_timeout,
        was_direct_dispatched,
        pool_fallback_lane_worker_count,
    } = args;
    let wait_start = std::time::Instant::now();
    let record_wait = |outcome| {
        telemetry::record_queue_result_wait("generate", outcome, wait_start.elapsed());
    };

    // Install the cancel-on-drop guard. Mirrors
    // `run_streaming_generate`: a normal completion path defuses it;
    // a task abort (HTTP client disconnect) fires the cancel signal.
    let cancel_guard =
        crate::handlers::proxy::StreamCancelGuard::new(Arc::clone(&publisher), request_id.clone());

    let publish_instant = tokio::time::Instant::now();
    let mut first_chunk_deadline = publish_instant + first_chunk_timeout;
    let overall_deadline = publish_instant + overall_timeout;
    let mut last_chunk_at: Option<tokio::time::Instant> = None;
    let mut first_seen = false;
    // Per-``choice_index`` ``role_emitted`` set (H4). For n=1 (the default)
    // this only ever contains 0; for streaming n>1 each candidate's first
    // delta gets an ``assistant`` role emitted independently, matching
    // OpenAI's per-choice SSE contract.
    let mut role_emitted: std::collections::HashSet<u32> = std::collections::HashSet::new();
    // Latch ``true`` when any chunk with ``choice_index > 0`` or a
    // non-terminal per-choice ``finish_reason`` arrives — the markers
    // for streaming ``n>1``. On the global ``done=true`` terminal we
    // then skip emitting a (duplicate) chat chunk and go straight to
    // the usage / ``[DONE]`` finalisers: each candidate has already
    // received its own terminal closure with per-choice
    // ``finish_reason``/``logprobs``.
    let mut multi_candidate_stream = false;
    let mut republished_for_first_chunk = false;
    let supports_first_chunk_pool_republish = publisher.supports_first_chunk_pool_republish();
    let single_consumer_lane_at_dispatch = crate::routing::suppress_first_chunk_republish_for_lane(
        was_direct_dispatched,
        pool_fallback_lane_worker_count,
    );

    // The terminal-outcome oneshot. The per-chunk broadcast tap drives
    // the normal forwarding path, but the oneshot is the *only* carrier
    // of the synthesised terminal error produced by
    // `WorkPublisher::fail_pending_stream` (NAK + pool-republish failure
    // → typed `rate_limit_exceeded`, etc.). Without polling it, that
    // case surfaces to the client as a generic `transport_failure` /
    // "Result channel closed" once the collector is torn down and the
    // broadcast closes. We `select!` on it alongside the chunk tap so
    // the typed code/message reaches the client. Pinned so it can be
    // polled by `&mut` across loop iterations.
    tokio::pin!(outcome_rx);
    // A `oneshot::Receiver` PANICS ("called after complete") if polled
    // again after it resolves. The outcome arm's success / sender-dropped
    // branches `continue` the loop, so without this guard the next
    // (`biased`) iteration would re-poll the now-consumed receiver and
    // panic on the request path. Disable the branch once it has fired.
    let mut outcome_done = false;
    // Latched when the terminal outcome resolves Ok (the generation completed
    // server-side). The Lagged arm uses it to skip a pointless worker cancel
    // for an already-completed request. #1602
    let mut stream_succeeded = false;
    let mut durability_done = false;

    // Helper: send an SSE event onto the mpsc channel. Returns false
    // if the receiver is closed (HTTP client disconnect), which is
    // the signal to stop driving and let the cancel guard fire on
    // drop.
    async fn send_event(
        tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
        ev: Event,
    ) -> bool {
        tx.send(Ok(ev)).await.is_ok()
    }

    loop {
        // Cheap early-fire to mirror `run_streaming_generate`.
        let now = tokio::time::Instant::now();
        if now >= overall_deadline {
            send_error_chunk(
                &event_tx,
                &endpoint,
                &stream_chat_id,
                created,
                &model,
                &request_id,
                "overall_timeout",
                "Generation aborted: overall timeout",
            )
            .await;
            send_done(&event_tx).await;
            record_wait(telemetry::QueueResultOutcome::Timeout);
            cancel_guard.defuse();
            publisher.publish_cancel(&request_id).await;
            publisher.drop_pending_stream(&request_id);
            return;
        }
        if !first_seen && now >= first_chunk_deadline {
            // One-shot republish to pool (same predicate as
            // `run_streaming_generate`). The broadcast receiver is
            // already subscribed, so chunks from the republished
            // attempt flow into this same loop.
            let deadline_action = crate::handlers::proxy::first_chunk_deadline_action(
                supports_first_chunk_pool_republish,
                was_direct_dispatched,
                republished_for_first_chunk,
                single_consumer_lane_at_dispatch,
            );
            if deadline_action == crate::handlers::proxy::FirstChunkDeadlineAction::RepublishToPool
            {
                republished_for_first_chunk = true;
                // At-least-once-execution hazard (mirrors the non-SSE
                // `run_streaming_generate` path): a SLOW original
                // direct-dispatched worker that is still alive would
                // otherwise run to completion alongside the pool worker —
                // double execution / double billing and duplicate chunks
                // racing into the same collector. Cancel the original
                // attempt FIRST (keyed on `cancel.{router_id}.{request_id}`,
                // before the pool worker has started), THEN republish.
                publisher.publish_cancel(&request_id).await;
                match publisher
                    .republish_to_pool(&request_id, "first_chunk_timeout")
                    .await
                {
                    Ok(true) => {
                        first_chunk_deadline = tokio::time::Instant::now() + first_chunk_timeout;
                        continue;
                    }
                    Ok(false) => {
                        // NAK-driven republish already happened; the
                        // outcome path will surface whatever the
                        // second attempt produces.
                        first_chunk_deadline = tokio::time::Instant::now() + first_chunk_timeout;
                        continue;
                    }
                    Err(e) => {
                        telemetry::record_rejected_request(
                            demand_tracker.as_ref(),
                            &physical_lane,
                            "publish_ack_failed",
                        );
                        demand_tracker.record(&physical_lane);
                        warn!(
                            request_id = %request_id,
                            error = %e,
                            "SSE: first_chunk_timeout republish to pool failed"
                        );
                    }
                }
            }
            if deadline_action == crate::handlers::proxy::FirstChunkDeadlineAction::WaitForOverall {
                republished_for_first_chunk = true;
                first_chunk_deadline = overall_deadline;
                tracing::debug!(
                        request_id = %request_id,
                        pool = %pool,
                        machine_profile = %gpu,
                        bundle = %bundle,
                        supports_pool_republish = supports_first_chunk_pool_republish,
                        "SSE: first_chunk_timeout - pool republish unavailable; continuing on overall_timeout"
                );
                continue;
            }
            send_error_chunk(
                &event_tx,
                &endpoint,
                &stream_chat_id,
                created,
                &model,
                &request_id,
                "first_chunk_timeout",
                "Generation aborted: first_chunk timeout",
            )
            .await;
            send_done(&event_tx).await;
            record_wait(telemetry::QueueResultOutcome::Timeout);
            cancel_guard.defuse();
            publisher.publish_cancel(&request_id).await;
            publisher.drop_pending_stream(&request_id);
            return;
        }
        if let Some(la) = last_chunk_at {
            if la.elapsed() >= inter_chunk_timeout {
                send_error_chunk(
                    &event_tx,
                    &endpoint,
                    &stream_chat_id,
                    created,
                    &model,
                    &request_id,
                    "inter_chunk_timeout",
                    "Generation aborted: inter_chunk timeout",
                )
                .await;
                send_done(&event_tx).await;
                record_wait(telemetry::QueueResultOutcome::Timeout);
                cancel_guard.defuse();
                publisher.publish_cancel(&request_id).await;
                publisher.drop_pending_stream(&request_id);
                return;
            }
        }

        let inter_chunk_deadline = last_chunk_at.map(|la| {
            let elapsed = la.elapsed();
            if elapsed >= inter_chunk_timeout {
                now
            } else {
                now + (inter_chunk_timeout - elapsed)
            }
        });

        let chunk_or_timeout = tokio::select! {
            biased;
            completion = &mut durability_completion, if !durability_done => {
                durability_done = true;
                match completion {
                    Ok(Ok(())) => None,
                    Ok(Err(error)) => {
                        send_error_chunk(
                            &event_tx,
                            &endpoint,
                            &stream_chat_id,
                            created,
                            &model,
                            &request_id,
                            "transport_failure",
                            &format!("Queue durability confirmation failed: {error}"),
                        )
                        .await;
                        send_done(&event_tx).await;
                        record_wait(telemetry::QueueResultOutcome::DurabilityError);
                        cancel_guard.defuse();
                        return;
                    }
                    Err(_) => {
                        send_error_chunk(
                            &event_tx,
                            &endpoint,
                            &stream_chat_id,
                            created,
                            &model,
                            &request_id,
                            "transport_failure",
                            "Queue durability monitor stopped before completion",
                        )
                        .await;
                        send_done(&event_tx).await;
                        record_wait(telemetry::QueueResultOutcome::DurabilityError);
                        cancel_guard.defuse();
                        return;
                    }
                }
            }
            // Detect HTTP client disconnect while waiting between chunks.
            // Without this branch, a client that drops while the worker is
            // idle for `inter_chunk_timeout` would keep the broadcast
            // subscription + collector alive for that full window (up to
            // 300 s on `overall_timeout`); `event_tx.send()` failure on
            // the next chunk was the only signal. Tying the driver
            // explicitly to the receiver's lifecycle closes that leak.
            _ = event_tx.closed() => {
                debug!(request_id = %request_id, "SSE receiver dropped; tearing down driver");
                record_wait(telemetry::QueueResultOutcome::Cancelled);
                telemetry::record_generation_event(
                    telemetry::GenerationEvent::Cancellation,
                    if first_seen {
                        telemetry::GenerationEventReason::MidStream
                    } else {
                        telemetry::GenerationEventReason::BeforeFirstChunk
                    },
                    telemetry::GenerationEventOutcome::Cancelled,
                );
                cancel_guard.defuse();
                teardown_after_client_disconnect(
                    &publisher,
                    &request_id,
                    (!outcome_done).then_some(outcome_rx.as_mut()),
                    stream_succeeded,
                )
                .await;
                return;
            }
            // Terminal outcome arm. A worker outcome is only a completion
            // signal: its exact terminal was already queued on the ordered
            // chunk tap, so this arm must never bypass that tap. A gateway-
            // synthesized outcome has no tap terminal and must win over the
            // broadcast-`Closed` path that races collector teardown.
            outcome = &mut outcome_rx, if !outcome_done => {
                // One-shot: never poll the resolved receiver again (it
                // would panic). All branches below either return or
                // `continue` to the chunk tap.
                outcome_done = true;
                match outcome {
                    Ok(o) => {
                        if o.origin == StreamOutcomeOrigin::GatewaySynthetic {
                            let Some(err) = o.error else {
                                // Defensive fallback for an impossible
                                // synthetic success: let the closed tap emit
                                // its ordinary transport-failure diagnostic.
                                continue;
                            };
                            // Emit the typed code/message (e.g.
                            // rate_limit_exceeded → 429-equivalent inside
                            // the stream) instead of a generic
                            // transport_failure. Same error shape as the
                            // worker-error chunk path below.
                            send_synthetic_error_chunk(
                                &event_tx,
                                &endpoint,
                                &stream_chat_id,
                                created,
                                &model,
                                &request_id,
                                &err,
                            )
                            .await;
                            send_done(&event_tx).await;
                            record_wait(telemetry::QueueResultOutcome::WorkerError);
                            cancel_guard.defuse();
                            publisher.drop_pending_stream(&request_id);
                            return;
                        }
                        // Every worker outcome means its terminal is already
                        // queued on the tap (tap send precedes outcome build).
                        // Drain the tap in order; that preserves all buffered
                        // deltas plus the terminal's real seq/usage/TTFT and
                        // execution identity. The terminal arm owns `[DONE]`.
                        stream_succeeded = true;
                        continue;
                    }
                    Err(_) => {
                        // Sender dropped without sending (collector torn
                        // down by a racing path). Fall through to let the
                        // chunk-tap `Closed` arm classify it.
                        continue;
                    }
                }
            }
            recv = chunk_rx.recv() => Some(recv),
            _ = tokio::time::sleep_until(overall_deadline) => {
                continue; // top of loop re-evaluates overall_deadline
            }
            _ = tokio::time::sleep_until(first_chunk_deadline), if !first_seen => {
                continue;
            }
            _ = tokio::time::sleep_until(inter_chunk_deadline.unwrap_or(overall_deadline)),
                if first_seen => {
                continue;
            }
        };

        let Some(recv_result) = chunk_or_timeout else {
            continue;
        };
        let chunk = match recv_result {
            Ok(c) => c,
            Err(broadcast::error::RecvError::Lagged(n)) => {
                // A lagged consumer means the sender overwrote `n` of the
                // oldest still-UNforwarded chunks in the bounded broadcast ring
                // before this driver read them, so the client's response is
                // genuinely INCOMPLETE — real content was dropped. (If all
                // content had been forwarded, the cursor would be caught up and
                // recv() would return the terminal in order, not `Lagged`.)
                // Always surface that: a bare `[DONE]` would present a
                // silently-truncated stream as a complete success. See #1602.
                warn!(
                    request_id = %request_id,
                    lagged = n,
                    succeeded = stream_succeeded,
                    "SSE consumer lagged behind chunk tap; response is incomplete"
                );
                send_error_chunk(
                    &event_tx,
                    &endpoint,
                    &stream_chat_id,
                    created,
                    &model,
                    &request_id,
                    "inter_chunk_timeout",
                    "SSE consumer fell behind; response is incomplete",
                )
                .await;
                send_done(&event_tx).await;
                record_wait(telemetry::QueueResultOutcome::WorkerError);
                cancel_guard.defuse();
                // Only cancel the worker if the generation is still in flight.
                // A request whose terminal outcome already resolved Ok has
                // completed server-side — there is nothing to cancel, so the
                // publish_cancel would be a redundant wire op. See #1602.
                if !stream_succeeded {
                    publisher.publish_cancel(&request_id).await;
                }
                publisher.drop_pending_stream(&request_id);
                return;
            }
            Err(broadcast::error::RecvError::Closed) => {
                // The collector was removed (terminal chunk applied
                // and outcome fired, or drop_pending_stream raced).
                // If we never saw a chunk, surface a generic
                // result-channel-closed; otherwise the terminator
                // was already sent below on the terminal chunk
                // arm — guard against a double `[DONE]` by checking
                // `first_seen`.
                if !first_seen {
                    send_error_chunk(
                        &event_tx,
                        &endpoint,
                        &stream_chat_id,
                        created,
                        &model,
                        &request_id,
                        "transport_failure",
                        "Result channel closed",
                    )
                    .await;
                    send_done(&event_tx).await;
                }
                record_wait(if stream_succeeded {
                    telemetry::QueueResultOutcome::Success
                } else {
                    telemetry::QueueResultOutcome::ChannelClosed
                });
                cancel_guard.defuse();
                return;
            }
        };

        // Non-stale chunk arrived. Update timing trackers.
        first_seen = true;
        last_chunk_at = Some(tokio::time::Instant::now());

        // Forward as an SSE event, then handle terminal/error/usage.
        let is_terminal = chunk.done;

        // Never present a clean terminal event before the initial queue
        // submission crossed its durable boundary. Normally the ACK resolved
        // long before the first worker chunk, so this is a ready receiver and
        // adds no latency. If a result races ahead of a pathological late ACK,
        // hold only the terminal event; a rejection becomes an in-stream typed
        // transport failure instead of an apparent success.
        if is_terminal && !durability_done {
            durability_done = true;
            match wait_for_terminal_durability(
                &mut durability_completion,
                &event_tx,
                overall_deadline,
            )
            .await
            {
                TerminalDurabilityWait::Confirmed => {}
                TerminalDurabilityWait::Failed(error) => {
                    send_error_chunk(
                        &event_tx,
                        &endpoint,
                        &stream_chat_id,
                        created,
                        &model,
                        &request_id,
                        "transport_failure",
                        &format!("Queue durability confirmation failed: {error}"),
                    )
                    .await;
                    send_done(&event_tx).await;
                    record_wait(telemetry::QueueResultOutcome::DurabilityError);
                    cancel_guard.defuse();
                    // Tear the stream down like the sibling durability arm
                    // below. This exit writes no usage surface, so it never
                    // reaches the terminal hook that would otherwise release
                    // per-stream transport state; leaving it registered strands
                    // that state until the transport's own backstop reclaims it
                    // and reports a stranded stream that was in fact torn down
                    // deliberately.
                    publisher.drop_pending_stream(&request_id);
                    return;
                }
                TerminalDurabilityWait::MonitorStopped => {
                    send_error_chunk(
                        &event_tx,
                        &endpoint,
                        &stream_chat_id,
                        created,
                        &model,
                        &request_id,
                        "transport_failure",
                        "Queue durability monitor stopped before completion",
                    )
                    .await;
                    send_done(&event_tx).await;
                    record_wait(telemetry::QueueResultOutcome::DurabilityError);
                    cancel_guard.defuse();
                    publisher.publish_cancel(&request_id).await;
                    publisher.drop_pending_stream(&request_id);
                    return;
                }
                TerminalDurabilityWait::ClientClosed => {
                    debug!(request_id = %request_id, "SSE receiver dropped while awaiting terminal durability");
                    record_wait(telemetry::QueueResultOutcome::Cancelled);
                    telemetry::record_generation_event(
                        telemetry::GenerationEvent::Cancellation,
                        telemetry::GenerationEventReason::MidStream,
                        telemetry::GenerationEventOutcome::Cancelled,
                    );
                    cancel_guard.defuse();
                    teardown_after_client_disconnect(
                        &publisher,
                        &request_id,
                        (!outcome_done).then_some(outcome_rx.as_mut()),
                        stream_succeeded,
                    )
                    .await;
                    return;
                }
                TerminalDurabilityWait::OverallTimeout => {
                    send_error_chunk(
                        &event_tx,
                        &endpoint,
                        &stream_chat_id,
                        created,
                        &model,
                        &request_id,
                        "overall_timeout",
                        "Generation aborted: overall timeout",
                    )
                    .await;
                    send_done(&event_tx).await;
                    record_wait(telemetry::QueueResultOutcome::Timeout);
                    cancel_guard.defuse();
                    publisher.publish_cancel(&request_id).await;
                    publisher.drop_pending_stream(&request_id);
                    return;
                }
            }
        }

        // The transport's terminal-only usage members (see
        // `WorkDispatcher::stream_terminal_usage_extras`). Fetched HERE — once,
        // on the terminal, after the durability wait and BEFORE either usage
        // surface is built — so a transport that finalises per-request
        // accounting at stream end has finished doing so before the client is
        // told anything, and nothing mid-stream can ever carry these members.
        // An unmetered transport returns an empty vec and pays nothing.
        //
        // `carries_usage` is passed rather than used to skip the call: a stream
        // that will emit no usage surface must not WAIT for members it would
        // discard (this sits in front of the terminal content event and
        // `[DONE]`), but it must still reach the transport so per-stream state
        // is released here rather than by a leak backstop.
        let terminal_usage_extras = if is_terminal {
            publisher
                .stream_terminal_usage_extras(&request_id, terminal_carries_usage(endpoint, &chunk))
                .await
        } else {
            Vec::new()
        };

        // Detect streaming ``n>1``: any chunk past ``choice_index=0`` or any
        // non-terminal chunk with ``finish_reason`` set is a per-choice
        // marker. Latch the flag for the global-terminal suppression below.
        if chunk.choice_index != 0 || (!chunk.done && chunk.finish_reason.is_some()) {
            multi_candidate_stream = true;
        }

        // On the global ``done=true`` terminal of a multi-candidate stream
        // each candidate's per-choice closure was already forwarded; emit
        // the optional usage chunk and ``[DONE]`` directly. Don't forward
        // a second "choice 0 finishes stop" event — clients would see
        // contradictory finish reasons.
        let skip_forward = is_terminal && multi_candidate_stream && chunk.error.is_none();

        let emit_role_for_this_chunk = matches!(endpoint, SseEndpoint::Chat { .. })
            && !role_emitted.contains(&chunk.choice_index)
            && !skip_forward;
        let event_body = match endpoint {
            SseEndpoint::Chat { .. } => build_chat_chunk_event(
                &stream_chat_id,
                created,
                &model,
                &chunk,
                emit_role_for_this_chunk,
            ),
            SseEndpoint::Generate => build_generate_chunk_event(&chunk, &terminal_usage_extras),
            SseEndpoint::Completion { .. } => {
                build_text_completion_chunk_event(&stream_chat_id, created, &model, &chunk)
            }
        };
        if emit_role_for_this_chunk {
            role_emitted.insert(chunk.choice_index);
        }
        if !skip_forward {
            let ev = Event::default().data(event_body.to_string());
            if !send_event(&event_tx, ev).await {
                // Client disconnected — fire the cancel deterministically
                // here instead of relying on the guard's Drop (which spawns
                // a detached task and can race the outer return / next
                // request).
                debug!(request_id = %request_id, "SSE client disconnected mid-stream");
                record_wait(telemetry::QueueResultOutcome::Cancelled);
                telemetry::record_generation_event(
                    telemetry::GenerationEvent::Cancellation,
                    if first_seen {
                        telemetry::GenerationEventReason::MidStream
                    } else {
                        telemetry::GenerationEventReason::BeforeFirstChunk
                    },
                    telemetry::GenerationEventOutcome::Cancelled,
                );
                cancel_guard.defuse();
                teardown_after_client_disconnect(
                    &publisher,
                    &request_id,
                    (!outcome_done).then_some(outcome_rx.as_mut()),
                    stream_succeeded,
                )
                .await;
                return;
            }
        }

        if is_terminal {
            if let Some(body) = build_usage_only_chunk_event(
                endpoint,
                &stream_chat_id,
                created,
                &model,
                &chunk,
                &terminal_usage_extras,
            ) {
                let _ = send_event(&event_tx, Event::default().data(body.to_string())).await;
            }
            send_done(&event_tx).await;
            record_wait(if chunk.error.is_some() {
                telemetry::QueueResultOutcome::WorkerError
            } else {
                telemetry::QueueResultOutcome::Success
            });
            cancel_guard.defuse();
            return;
        }
    }
}

/// Build a single OpenAI-shaped chat.completion.chunk JSON value.
///
/// `emit_role` is true on the first chunk only; subsequent chunks
/// omit ``delta.role`` per the OpenAI spec. The terminal chunk
/// carries ``finish_reason``; non-terminal chunks have it null.
/// Worker-error chunks attach a top-level ``error`` block alongside
/// the normal envelope.
fn build_chat_chunk_event(
    id: &str,
    created: u64,
    model: &str,
    chunk: &ChunkEnvelope,
    emit_role: bool,
) -> Value {
    let mut delta = serde_json::Map::new();
    if emit_role {
        delta.insert("role".to_string(), json!("assistant"));
    }
    if !chunk.text_delta.is_empty() {
        delta.insert("content".to_string(), json!(chunk.text_delta));
    }
    // OpenAI tool-call delta: surface ``delta.tool_calls`` byte-for-
    // byte from the worker envelope. The worker emits one logical
    // delta per chunk (announcement or arguments body), already wrapped
    // in a single-element list to match OpenAI's wire shape exactly.
    if let Some(tcs) = chunk.tool_calls.as_ref() {
        if !tcs.is_empty() {
            let arr: Vec<Value> = tcs
                .iter()
                .map(|tc| {
                    let mut obj = serde_json::Map::new();
                    obj.insert("index".to_string(), json!(tc.index));
                    if let Some(id) = tc.id.as_ref() {
                        obj.insert("id".to_string(), json!(id));
                    }
                    obj.insert("type".to_string(), json!(tc.kind));
                    if let Some(func) = tc.function.as_ref() {
                        let mut fb = serde_json::Map::new();
                        if let Some(name) = func.name.as_ref() {
                            fb.insert("name".to_string(), json!(name));
                        }
                        fb.insert("arguments".to_string(), json!(func.arguments));
                        obj.insert("function".to_string(), Value::Object(fb));
                    }
                    Value::Object(obj)
                })
                .collect();
            delta.insert("tool_calls".to_string(), Value::Array(arr));
        }
    }
    // H4: per-choice ``finish_reason`` rides on non-terminal chunks too.
    // The worker emits a non-``done`` chunk with ``finish_reason`` set when
    // a specific candidate in a streaming ``n>1`` run completes; that chunk
    // carries the candidate's final delta + closure. The global ``done=true``
    // terminal still also surfaces ``finish_reason`` for the single-candidate
    // path. Either source produces the OpenAI per-choice ``finish_reason``
    // — clients receive one per ``choice_index``.
    let finish_reason = if chunk.done {
        if chunk.error.is_some() {
            // Don't claim a clean `stop` when the terminal carries an
            // error — that would let a client keying on `finish_reason`
            // read a failed generation as successful. The top-level
            // `error` object below is the authoritative failure signal.
            Value::Null
        } else {
            let raw = chunk.finish_reason.as_deref().unwrap_or("stop");
            Value::String(map_chat_finish_reason(raw).to_string())
        }
    } else if let Some(raw) = chunk.finish_reason.as_deref() {
        Value::String(map_chat_finish_reason(raw).to_string())
    } else {
        Value::Null
    };
    let mut body = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": crate::handlers::proxy::system_fingerprint(model),
        "choices": [{
            // Candidate ordinal for streaming n>1 (0 for the single-candidate
            // stream). Lets clients reassemble per-candidate streams.
            "index": chunk.choice_index,
            "delta": Value::Object(delta),
            // Per-chunk OpenAI logprobs: the worker attaches the
            // ``ChatCompletionTokenLogprob`` entries for the tokens in
            // this delta. Emit them in the ``{content: [...], refusal:
            // null}`` shape; null when logprobs weren't requested.
            "logprobs": match chunk.logprobs.as_ref() {
                Some(content) => json!({ "content": content, "refusal": Value::Null }),
                None => Value::Null,
            },
            "finish_reason": finish_reason,
        }],
    });
    if let Some(err) = chunk.error.as_ref() {
        if let Some(obj) = body.as_object_mut() {
            obj.insert("error".to_string(), worker_error_value(err, true));
            // Gateway request id, in-band on the ERROR chunk only (#3136):
            // a streamed response has no terminal headers, and the
            // ``chatcmpl-*`` id is not the correlation key server logs use.
            // Additive top-level member — named to match the SIE-native
            // generate error chunk — that OpenAI clients ignore.
            obj.insert("request_id".to_string(), json!(chunk.request_id));
        }
    }
    insert_terminal_execution_evidence(&mut body, chunk);
    body
}

/// OpenAI legacy Completions streaming chunk (`object: "text_completion"`).
/// Single-candidate (completions rejects `n>1`); the per-chunk text delta lands
/// on `choices[0].text`, with `finish_reason` on the terminal chunk.
fn build_text_completion_chunk_event(
    id: &str,
    created: u64,
    model: &str,
    chunk: &ChunkEnvelope,
) -> Value {
    let finish = if chunk.done && chunk.error.is_some() {
        Value::Null
    } else if chunk.done {
        Value::String(
            map_chat_finish_reason(chunk.finish_reason.as_deref().unwrap_or("stop")).to_string(),
        )
    } else {
        Value::Null
    };
    let mut body = json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "system_fingerprint": crate::handlers::proxy::system_fingerprint(model),
        // H3: ``logprobs`` is rejected at the /v1/completions input parser,
        // so streaming chunks no longer carry an always-null ``logprobs`` field.
        "choices": [{
            "text": chunk.text_delta,
            "index": 0,
            "finish_reason": finish,
        }],
    });
    if let Some(err) = chunk.error.as_ref() {
        if let Some(object) = body.as_object_mut() {
            object.insert("error".to_string(), worker_error_value(err, true));
            object.insert("request_id".to_string(), json!(chunk.request_id));
        }
    }
    insert_terminal_execution_evidence(&mut body, chunk);
    body
}

/// Build the optional trailing usage-only chunk (OpenAI
/// ``stream_options.include_usage``) emitted just before ``[DONE]``.
///
/// A streaming response cannot report usage in headers — they were flushed
/// with the first byte — so this chunk is the surface on which a metered
/// stream tells the caller what it actually consumed. Returns ``None`` when
/// the endpoint did not ask for it, or when the terminal chunk carried no
/// authoritative usage (never synthesise counts). ``Generate`` has no case
/// here: its native terminal chunk already carries ``usage`` inline.
///
/// `terminal_extras` are the transport's terminal-only members (see
/// [`WorkDispatcher::stream_terminal_usage_extras`]); they ride the same usage
/// object. They are merged into an EXISTING block only — a terminal that
/// reported no counts still emits no usage chunk, so the extras can never
/// conjure one out of a stream that reported nothing.
fn build_usage_only_chunk_event(
    endpoint: SseEndpoint,
    id: &str,
    created: u64,
    model: &str,
    chunk: &ChunkEnvelope,
    terminal_extras: &[(String, Value)],
) -> Option<Value> {
    let object = match endpoint {
        SseEndpoint::Chat { include_usage } if include_usage => "chat.completion.chunk",
        SseEndpoint::Completion { include_usage } if include_usage => "text_completion",
        _ => return None,
    };
    let usage = chunk.usage.as_ref()?;
    let mut usage_body = json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    });
    merge_terminal_usage_extras(&mut usage_body, terminal_extras);
    Some(json!({
        "id": id,
        "object": object,
        "created": created,
        "model": model,
        "system_fingerprint": crate::handlers::proxy::system_fingerprint(model),
        "choices": [],
        "usage": usage_body,
    }))
}

/// Whether this terminal will emit a usage surface at all.
///
/// Two independent reasons it will not, and both must be checked because the
/// transport is made to WAIT whenever this says yes:
///
/// * The OpenAI surfaces publish usage only under `stream_options.include_usage`
///   — omitted by default — and `build_usage_only_chunk_event` returns `None`
///   without ever reading the transport's members for the rest.
/// * The extras merge into an EXISTING usage block only, so a terminal whose
///   worker reported no counts (an error terminal, an aborted run) emits no
///   usage block on any surface — including native generate, whose inline
///   block is itself gated on `chunk.usage`.
fn terminal_carries_usage(endpoint: SseEndpoint, chunk: &ChunkEnvelope) -> bool {
    if chunk.usage.is_none() {
        return false;
    }
    match endpoint {
        SseEndpoint::Generate => true,
        SseEndpoint::Chat { include_usage } | SseEndpoint::Completion { include_usage } => {
            include_usage
        }
    }
}

/// Merge a transport's terminal-only members into a usage block.
///
/// Additive only: a member whose key the worker's own usage block already
/// carries is dropped rather than overwriting an authoritative count.
fn merge_terminal_usage_extras(usage: &mut Value, extras: &[(String, Value)]) {
    let Some(object) = usage.as_object_mut() else {
        return;
    };
    for (key, value) in extras {
        if !object.contains_key(key) {
            object.insert(key.clone(), value.clone());
        }
    }
}

fn insert_terminal_execution_evidence(body: &mut Value, chunk: &ChunkEnvelope) {
    if !chunk.done || chunk.error.is_some() {
        return;
    }
    let (Some(identity), Some(binding)) = (
        chunk.execution_identity_sha256.as_deref(),
        chunk.execution_binding_sha256.as_deref(),
    ) else {
        return;
    };
    if !is_lower_sha256(identity) || !is_lower_sha256(binding) {
        return;
    }
    if let Some(object) = body.as_object_mut() {
        object.insert("execution_identity_sha256".to_string(), json!(identity));
        object.insert("execution_binding_sha256".to_string(), json!(binding));
    }
}

/// SIE-native generate chunk shape.
fn build_generate_chunk_event(chunk: &ChunkEnvelope, terminal_extras: &[(String, Value)]) -> Value {
    let mut body = json!({
        "request_id": chunk.request_id,
        "seq": chunk.seq,
        "text_delta": chunk.text_delta,
        "done": chunk.done,
    });
    if let Some(obj) = body.as_object_mut() {
        if let Some(fr) = chunk.finish_reason.as_ref() {
            obj.insert("finish_reason".to_string(), json!(fr));
        }
        if let Some(u) = chunk.usage.as_ref() {
            let mut usage = json!({
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            });
            merge_terminal_usage_extras(&mut usage, terminal_extras);
            obj.insert("usage".to_string(), usage);
        }
        if let Some(t) = chunk.ttft_ms {
            obj.insert("ttft_ms".to_string(), json!(t));
        }
        if let Some(logprobs) = chunk.logprobs.as_ref() {
            obj.insert("logprobs".to_string(), json!(logprobs));
        }
        if let Some(err) = chunk.error.as_ref() {
            obj.insert("error".to_string(), worker_error_value(err, false));
        }
    }
    insert_terminal_execution_evidence(&mut body, chunk);
    body
}

fn worker_error_value(error: &ChunkError, include_openai_type: bool) -> Value {
    let code = error.client_safe_code();
    let mut object = serde_json::Map::new();
    object.insert("message".to_string(), json!(error.client_safe_message()));
    if include_openai_type {
        object.insert(
            "type".to_string(),
            json!(worker_error_openai_type_for(code)),
        );
    }
    object.insert("param".to_string(), json!(error.client_safe_param()));
    object.insert("code".to_string(), json!(code));
    if let Some(retry_after_s) = error.validated_retry_after_s() {
        object.insert("retry_after_s".to_string(), json!(retry_after_s));
    }
    Value::Object(object)
}

/// Emit a synthesized error chunk (gateway-side timeout or
/// transport failure) onto the SSE stream. Wraps the right shape
/// for each endpoint.
#[allow(clippy::too_many_arguments)]
async fn send_error_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    endpoint: &SseEndpoint,
    chat_id: &str,
    created: u64,
    model: &str,
    request_id: &str,
    code: &str,
    message: &str,
) {
    let body = match endpoint {
        SseEndpoint::Chat { .. } => json!({
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": crate::handlers::proxy::system_fingerprint(model),
            "choices": [{
                "index": 0,
                "delta": {},
                "logprobs": Value::Null,
                // Null, not "stop": this chunk carries an error, so we must
                // not let a client keying on `finish_reason` read an aborted
                // generation as a clean completion. The `error` object below
                // is the authoritative signal.
                "finish_reason": Value::Null,
            }],
            "error": {
                "message": message,
                "type": "server_error",
                "param": Value::Null,
                "code": code,
            },
            // Gateway request id, in-band on the ERROR chunk only (#3136) —
            // additive, mirrors the generate-shape member below so SDK
            // consumers can correlate stream failures with gateway logs.
            "request_id": request_id,
        }),
        SseEndpoint::Generate => json!({
            "request_id": request_id,
            "seq": 0,
            "text_delta": "",
            "done": true,
            "finish_reason": "error",
            "error": { "code": code, "message": message },
        }),
        SseEndpoint::Completion { .. } => json!({
            "id": chat_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "system_fingerprint": crate::handlers::proxy::system_fingerprint(model),
            "choices": [{"text": "", "index": 0, "finish_reason": Value::Null, "logprobs": Value::Null}],
            "error": { "message": message, "type": "server_error", "param": Value::Null, "code": code },
            "request_id": request_id,
        }),
    };
    let _ = tx.send(Ok(Event::default().data(body.to_string()))).await;
}

/// Emit a gateway-synthesized terminal error when no worker tap terminal exists.
#[allow(clippy::too_many_arguments)]
async fn send_synthetic_error_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    endpoint: &SseEndpoint,
    chat_id: &str,
    created: u64,
    model: &str,
    request_id: &str,
    error: &ChunkError,
) {
    let chunk = ChunkEnvelope {
        kind: "chunk".to_string(),
        request_id: request_id.to_string(),
        attempt_id: String::new(),
        seq: 0,
        text_delta: String::new(),
        done: true,
        is_first: false,
        finish_reason: Some("error".to_string()),
        usage: None,
        ttft_ms: None,
        error: Some(error.clone()),
        tool_calls: None,
        logprobs: None,
        candidates: Vec::new(),
        choice_index: 0,
        executed_bundle_config_hash: None,
        execution_identity_sha256: None,
        execution_binding_sha256: None,
    };
    let body = match endpoint {
        SseEndpoint::Chat { .. } => build_chat_chunk_event(chat_id, created, model, &chunk, false),
        SseEndpoint::Completion { .. } => {
            build_text_completion_chunk_event(chat_id, created, model, &chunk)
        }
        SseEndpoint::Generate => build_generate_chunk_event(&chunk, &[]),
    };
    let _ = tx.send(Ok(Event::default().data(body.to_string()))).await;
}

async fn send_done(tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>) {
    // The literal ``[DONE]`` terminator. axum's Event::data already
    // emits the trailing ``\n\n`` separator.
    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
}

/// Local mirror of `proxy::map_chat_finish_reason` — kept private
/// here to avoid widening that function's visibility just for SSE.
fn map_chat_finish_reason(sie: &str) -> &'static str {
    match sie {
        "length" => "length",
        "tool_calls" => "tool_calls",
        // ``content_filter`` / ``function_call`` are valid OpenAI finish
        // reasons the worker can emit; pass them through rather than
        // collapse to ``stop``. Kept in lockstep with
        // ``proxy::map_chat_finish_reason``.
        "content_filter" => "content_filter",
        "function_call" => "function_call",
        _ => "stop",
    }
}

/// Map a worker `error.code` onto the OpenAI `error.type`. Inlined
/// here instead of re-exporting `proxy::worker_error_openai_type`
/// (which already exists) so this module has zero coupling back to
/// the giant proxy.rs file beyond the small public surface listed
/// at the top of the file.
fn worker_error_openai_type_for(code: &str) -> &'static str {
    match code {
        "invalid_request" | "unsupported_field" => "invalid_request_error",
        "context_exceeded" => "context_length_exceeded",
        "rate_limit_exceeded" => "rate_limit_error",
        _ => "server_error",
    }
}

// `Sse::new` takes `S: Stream<Item = Result<Event, E>>` — the
// `ReceiverStream` wrapper from `tokio_stream` satisfies that bound
// when the channel item type is `Result<Event, Infallible>`. No
// extra plumbing required.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue::streaming::{
        ChunkApplied, ChunkEnvelope, ChunkError, ToolCallDeltaWire, ToolCallFunctionWire,
        UsageBlock,
    };

    mod client_disconnect_grace {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::time::Instant;

        use tokio::sync::{broadcast, oneshot};

        use super::super::{teardown_after_client_disconnect, CLIENT_DISCONNECT_TERMINAL_GRACE};
        use crate::queue::dispatch::WorkResult;
        use crate::queue::dispatch::{
            DispatchDurability, DispatchError, PendingGenerationSnapshot, WorkDispatcher,
        };
        use crate::queue::publisher::{PublishTarget, WorkParams};
        use crate::queue::streaming::{
            ChunkEnvelope, StreamOutcome, StreamOutcomeOrigin, UsageBlock,
        };

        /// Records exactly which teardown the SSE driver chose. The two are
        /// transport-identical and differ only in the billing signal they emit,
        /// so the choice IS the behaviour under test.
        #[derive(Default)]
        pub(super) struct TeardownRecorder {
            cancels: AtomicUsize,
            plain_drops: AtomicUsize,
            disconnect_drops: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl WorkDispatcher for TeardownRecorder {
            async fn publish_work(
                self: Arc<Self>,
                _target: PublishTarget,
                _admission_pool: &str,
                _endpoint: &str,
                _model: &str,
                _engine: &str,
                _bundle_config_hash: &str,
                _items: Vec<rmpv::Value>,
                _params: &WorkParams,
            ) -> Result<
                (
                    String,
                    oneshot::Receiver<Vec<WorkResult>>,
                    DispatchDurability,
                ),
                DispatchError,
            > {
                unreachable!("teardown test never publishes")
            }

            async fn publish_generate_streaming(
                &self,
                _target: PublishTarget,
                _display_model: &str,
                _engine: &str,
                _bundle_config_hash: &str,
                _params: &WorkParams,
                _admission_pool: &str,
            ) -> Result<
                (
                    String,
                    oneshot::Receiver<StreamOutcome>,
                    Arc<tokio::sync::Notify>,
                    DispatchDurability,
                ),
                String,
            > {
                unreachable!("teardown test never publishes")
            }

            async fn publish_generate_streaming_sse(
                &self,
                _target: PublishTarget,
                _display_model: &str,
                _engine: &str,
                _bundle_config_hash: &str,
                _params: &WorkParams,
                _admission_pool: &str,
            ) -> Result<
                (
                    String,
                    oneshot::Receiver<StreamOutcome>,
                    broadcast::Receiver<ChunkEnvelope>,
                    DispatchDurability,
                ),
                String,
            > {
                unreachable!("teardown test never publishes")
            }

            async fn publish_cancel(&self, _request_id: &str) {
                self.cancels.fetch_add(1, Ordering::Relaxed);
            }

            fn begin_work_abandonment(&self, _request_id: &str) -> bool {
                false
            }

            async fn finish_work_abandonment(&self, _request_id: &str) {}

            async fn republish_to_pool(
                &self,
                _request_id: &str,
                _reason: &'static str,
            ) -> Result<bool, String> {
                Ok(false)
            }

            async fn republish_pending_result_to_pool(
                &self,
                _request_id: &str,
                _reason: &'static str,
            ) -> Result<bool, String> {
                Ok(false)
            }

            fn drop_pending_stream(&self, _request_id: &str) {
                self.plain_drops.fetch_add(1, Ordering::Relaxed);
            }

            fn drop_pending_stream_client_disconnect(&self, _request_id: &str) {
                self.disconnect_drops.fetch_add(1, Ordering::Relaxed);
            }

            fn pending_generation_snapshot(&self) -> PendingGenerationSnapshot {
                PendingGenerationSnapshot::default()
            }

            fn pending_generation_for_model(&self, _model_id: &str) -> PendingGenerationSnapshot {
                PendingGenerationSnapshot::default()
            }

            fn stream_observed_first_chunk(&self, _request_id: &str) -> bool {
                false
            }

            fn stream_chunk_timing(
                &self,
                _request_id: &str,
            ) -> Option<(Option<Instant>, Option<Instant>)> {
                None
            }
        }

        fn abort_terminal(completion_tokens: u32) -> StreamOutcome {
            StreamOutcome {
                text: "partial".to_string(),
                // What the worker emits when it honours a cancel: a terminal
                // whose usage is the count-so-far.
                finish_reason: "cancelled".to_string(),
                usage: Some(UsageBlock {
                    prompt_tokens: 5,
                    completion_tokens,
                    total_tokens: 5 + completion_tokens,
                }),
                attempt_id: "attempt-1".to_string(),
                ttft_ms: None,
                tpot_ms: None,
                error: None,
                origin: StreamOutcomeOrigin::WorkerTerminal,
                tool_calls: None,
                logprobs: None,
                candidates: Vec::new(),
                executed_bundle_config_hash: None,
                execution_identity_sha256: None,
                execution_binding_sha256: None,
            }
        }

        /// (a) The worker's abort terminal lands inside the grace window.
        ///
        /// Regression: teardown used to remove the collector immediately after
        /// `publish_cancel` (which awaits only the transport ACK of the cancel
        /// op, never the worker), so the terminal arrived with nowhere to go and
        /// the count-so-far was discarded. The driver must instead keep the
        /// stream registered and let the terminal resolve — the ordinary
        /// teardown, whose metered settlement bills the partial output.
        #[tokio::test(start_paused = true)]
        async fn abort_terminal_inside_the_grace_window_reaches_settlement() {
            let recorder = Arc::new(TeardownRecorder::default());
            let publisher: Arc<dyn WorkDispatcher> = recorder.clone() as Arc<dyn WorkDispatcher>;
            let (outcome_tx, mut outcome_rx) = oneshot::channel::<StreamOutcome>();

            // The worker answers the cancel a beat later, well inside the window.
            let worker_delay = CLIENT_DISCONNECT_TERMINAL_GRACE / 5;
            tokio::spawn(async move {
                tokio::time::sleep(worker_delay).await;
                let _ = outcome_tx.send(abort_terminal(17));
            });

            let started = tokio::time::Instant::now();
            teardown_after_client_disconnect(
                &publisher,
                "req-disconnect-billed",
                Some(std::pin::Pin::new(&mut outcome_rx)),
                false,
            )
            .await;
            let waited = started.elapsed();

            assert!(
                waited >= worker_delay,
                "teardown must keep the stream registered until the worker's abort \
                 terminal lands rather than racing it (waited {waited:?})"
            );
            assert!(
                waited < CLIENT_DISCONNECT_TERMINAL_GRACE,
                "and must finish on the terminal, not by timing out"
            );
            assert!(
                matches!(
                    outcome_rx.try_recv(),
                    Err(oneshot::error::TryRecvError::Closed)
                ),
                "the abort terminal must have been consumed by the wait — i.e. \
                 delivered to the settlement path — not left to be dropped on the floor"
            );
            assert_eq!(
                recorder.cancels.load(Ordering::Relaxed),
                1,
                "the disconnect still cancels upstream"
            );
            assert_eq!(
                recorder.disconnect_drops.load(Ordering::Relaxed),
                0,
                "a terminal that arrived is not a terminal-less disconnect"
            );
            assert_eq!(
                recorder.plain_drops.load(Ordering::Relaxed),
                1,
                "the settled stream is torn down through the ordinary path"
            );
        }

        /// (b) No terminal inside the window.
        ///
        /// The hold must still be released — but through the disconnect-specific
        /// path, so a metered dispatcher does NOT raise the `lost_terminal_count`
        /// billing-fault alert for what is ordinary user behaviour.
        #[tokio::test(start_paused = true)]
        async fn silent_worker_releases_without_raising_a_billing_fault() {
            let recorder = Arc::new(TeardownRecorder::default());
            let publisher: Arc<dyn WorkDispatcher> = recorder.clone() as Arc<dyn WorkDispatcher>;
            // Held open for the whole test: the worker never reports.
            let (_outcome_tx, mut outcome_rx) = oneshot::channel::<StreamOutcome>();

            let started = tokio::time::Instant::now();
            teardown_after_client_disconnect(
                &publisher,
                "req-disconnect-silent",
                Some(std::pin::Pin::new(&mut outcome_rx)),
                false,
            )
            .await;

            assert!(
                started.elapsed() >= CLIENT_DISCONNECT_TERMINAL_GRACE,
                "the grace window is actually waited out"
            );
            assert_eq!(recorder.cancels.load(Ordering::Relaxed), 1);
            assert_eq!(
                recorder.disconnect_drops.load(Ordering::Relaxed),
                1,
                "a terminal-less disconnect releases through the quiet path"
            );
            assert_eq!(
                recorder.plain_drops.load(Ordering::Relaxed),
                0,
                "the fault-raising teardown must not fire on an ordinary stop"
            );
        }

        /// An already-resolved terminal needs no wait at all: tearing down is
        /// immediate, and the grace window never delays the next request's
        /// resources.
        #[tokio::test(start_paused = true)]
        async fn an_already_resolved_terminal_skips_the_wait() {
            let recorder = Arc::new(TeardownRecorder::default());
            let publisher: Arc<dyn WorkDispatcher> = recorder.clone() as Arc<dyn WorkDispatcher>;

            let started = tokio::time::Instant::now();
            teardown_after_client_disconnect(&publisher, "req-disconnect-done", None, false).await;

            assert!(started.elapsed() < CLIENT_DISCONNECT_TERMINAL_GRACE);
            assert_eq!(recorder.plain_drops.load(Ordering::Relaxed), 1);
            assert_eq!(recorder.disconnect_drops.load(Ordering::Relaxed), 0);
        }

        /// A generation that already completed server-side has nothing to
        /// cancel, so the disconnect teardown must not issue a redundant cancel
        /// wire op — the same `#1602` discipline the `Lagged` arm follows.
        #[tokio::test(start_paused = true)]
        async fn an_already_succeeded_stream_is_not_cancelled_again() {
            let recorder = Arc::new(TeardownRecorder::default());
            let publisher: Arc<dyn WorkDispatcher> = recorder.clone() as Arc<dyn WorkDispatcher>;

            teardown_after_client_disconnect(&publisher, "req-disconnect-late", None, true).await;

            assert_eq!(
                recorder.cancels.load(Ordering::Relaxed),
                0,
                "a request that already completed server-side must not be cancelled"
            );
            assert_eq!(
                recorder.plain_drops.load(Ordering::Relaxed),
                1,
                "teardown still runs"
            );
            assert_eq!(recorder.disconnect_drops.load(Ordering::Relaxed), 0);
        }
    }

    #[derive(Clone, Copy)]
    enum WorkerTerminalDelivery {
        BackloggedBeforeFirstPoll,
        AfterFirstDelta,
    }

    async fn run_driver_worker_error_race(
        endpoint: SseEndpoint,
        delivery: WorkerTerminalDelivery,
        error: ChunkError,
    ) -> Vec<String> {
        use crate::queue::streaming::StreamCollector;
        use crate::state::demand_tracker::PhysicalLaneCatalog;

        let recorder = Arc::new(client_disconnect_grace::TeardownRecorder::default());
        let publisher: Arc<dyn WorkDispatcher> = recorder as Arc<dyn WorkDispatcher>;
        let catalog = PhysicalLaneCatalog::try_from_raw([(
            "default".to_string(),
            "default".to_string(),
            "default".to_string(),
        )])
        .expect("catalog");
        let demand_tracker = Arc::new(DemandTracker::new(catalog));
        let physical_lane = demand_tracker
            .resolve_lane("default", "default", "default")
            .expect("lane");

        let (collector_tx, _collector_rx) = tokio::sync::oneshot::channel();
        let mut collector = StreamCollector::new(
            collector_tx,
            "test/model".to_string(),
            "default".to_string(),
        );
        let chunk_rx = collector.install_chunk_tap();
        let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel();
        let (durability_tx, durability_completion) = tokio::sync::oneshot::channel();
        durability_tx
            .send(Ok::<(), String>(()))
            .expect("durability receiver is live");
        let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(8);

        let args = SseDriverArgs {
            event_tx,
            chunk_rx,
            outcome_rx,
            durability_completion,
            publisher,
            demand_tracker,
            physical_lane,
            request_id: "req-test".to_string(),
            model: "test/model".to_string(),
            pool: "default".to_string(),
            bundle: "default".to_string(),
            gpu: "test".to_string(),
            endpoint,
            stream_chat_id: "chatcmpl-test".to_string(),
            created: 1,
            first_chunk_timeout: Duration::from_secs(30),
            inter_chunk_timeout: Duration::from_secs(30),
            overall_timeout: Duration::from_secs(60),
            was_direct_dispatched: false,
            pool_fallback_lane_worker_count: 1,
        };

        let mut payloads = Vec::new();
        if matches!(delivery, WorkerTerminalDelivery::AfterFirstDelta) {
            let driver = tokio::spawn(run_sse_driver(args));
            assert!(matches!(
                collector.apply(_delta_chunk(41, "partial")),
                ChunkApplied::Delta
            ));
            let event = tokio::time::timeout(Duration::from_secs(1), event_rx.recv())
                .await
                .expect("delta event timeout")
                .expect("delta event")
                .expect("infallible event");
            payloads.push(_event_data(event).await);

            let mut terminal = _terminal_chunk("error", None);
            terminal.seq = 42;
            terminal.usage = Some(UsageBlock {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            });
            terminal.ttft_ms = Some(17.5);
            terminal.executed_bundle_config_hash = Some("a".repeat(64));
            terminal.execution_identity_sha256 = Some("b".repeat(64));
            terminal.error = Some(error);
            assert!(matches!(collector.apply(terminal), ChunkApplied::Terminal));
            outcome_tx
                .send(collector.build_outcome().expect("terminal outcome"))
                .expect("outcome receiver is live");
            driver.await.expect("driver task");
        } else {
            assert!(matches!(
                collector.apply(_delta_chunk(41, "partial")),
                ChunkApplied::Delta
            ));
            let mut terminal = _terminal_chunk("error", None);
            terminal.seq = 42;
            terminal.usage = Some(UsageBlock {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
            });
            terminal.ttft_ms = Some(17.5);
            terminal.executed_bundle_config_hash = Some("a".repeat(64));
            terminal.execution_identity_sha256 = Some("b".repeat(64));
            terminal.error = Some(error);
            assert!(matches!(collector.apply(terminal), ChunkApplied::Terminal));
            outcome_tx
                .send(collector.build_outcome().expect("terminal outcome"))
                .expect("outcome receiver is live");
            run_sse_driver(args).await;
        }

        loop {
            let Some(event) = tokio::time::timeout(Duration::from_secs(1), event_rx.recv())
                .await
                .expect("driver event timeout")
            else {
                break;
            };
            payloads.push(_event_data(event.expect("infallible event")).await);
        }
        payloads
    }

    #[tokio::test]
    async fn live_driver_worker_error_outcome_preserves_full_contract_before_and_after_delta() {
        for delivery in [
            WorkerTerminalDelivery::BackloggedBeforeFirstPoll,
            WorkerTerminalDelivery::AfterFirstDelta,
        ] {
            for endpoint in [
                SseEndpoint::Chat {
                    include_usage: true,
                },
                SseEndpoint::Completion {
                    include_usage: true,
                },
                SseEndpoint::Generate,
            ] {
                let payloads = run_driver_worker_error_race(
                    endpoint,
                    delivery,
                    ChunkError {
                        code: "RESOURCE_EXHAUSTED".to_string(),
                        message: "scheduler full".to_string(),
                        param: Some("model".to_string()),
                        retry_after_s: Some(12),
                    },
                )
                .await;
                assert_eq!(
                    payloads
                        .iter()
                        .filter(|value| value.as_str() == "[DONE]")
                        .count(),
                    1
                );
                assert_eq!(payloads.last().expect("done"), "[DONE]");
                let json_payloads = payloads[..payloads.len() - 1]
                    .iter()
                    .map(|payload| {
                        serde_json::from_str::<Value>(payload).expect("worker event JSON")
                    })
                    .collect::<Vec<_>>();
                assert_eq!(
                    json_payloads.len(),
                    3 - usize::from(matches!(endpoint, SseEndpoint::Generate))
                );
                let delta = &json_payloads[0];
                match endpoint {
                    SseEndpoint::Chat { .. } => {
                        assert_eq!(delta["choices"][0]["delta"]["content"], "partial")
                    }
                    SseEndpoint::Completion { .. } => {
                        assert_eq!(delta["choices"][0]["text"], "partial")
                    }
                    SseEndpoint::Generate => {
                        assert_eq!(delta["seq"], 41);
                        assert_eq!(delta["text_delta"], "partial");
                    }
                }
                let error_payload = json_payloads
                    .iter()
                    .find(|payload| payload.get("error").is_some())
                    .expect("worker error event");
                assert_eq!(error_payload["error"]["code"], "RESOURCE_EXHAUSTED");
                assert!(error_payload["error"]["param"].is_null());
                assert_eq!(error_payload["error"]["retry_after_s"], 12);
                match endpoint {
                    SseEndpoint::Chat { .. } | SseEndpoint::Completion { .. } => {
                        assert_eq!(error_payload["error"]["type"], "server_error");
                    }
                    SseEndpoint::Generate => {
                        assert!(error_payload["error"].get("type").is_none());
                        assert_eq!(error_payload["seq"], 42);
                        assert_eq!(error_payload["usage"]["prompt_tokens"], 3);
                        assert_eq!(error_payload["usage"]["completion_tokens"], 2);
                        assert_eq!(error_payload["ttft_ms"], 17.5);
                    }
                }
                if !matches!(endpoint, SseEndpoint::Generate) {
                    let usage = json_payloads
                        .iter()
                        .find(|payload| payload["choices"] == json!([]))
                        .expect("usage-only event");
                    assert_eq!(usage["usage"]["prompt_tokens"], 3);
                    assert_eq!(usage["usage"]["completion_tokens"], 2);
                }
            }
        }
    }

    #[tokio::test]
    async fn live_driver_surfaces_synthetic_only_outcome_when_tap_closes() {
        use crate::state::demand_tracker::PhysicalLaneCatalog;

        for endpoint in [
            SseEndpoint::Chat {
                include_usage: false,
            },
            SseEndpoint::Completion {
                include_usage: false,
            },
            SseEndpoint::Generate,
        ] {
            let recorder = Arc::new(client_disconnect_grace::TeardownRecorder::default());
            let publisher: Arc<dyn WorkDispatcher> = recorder as Arc<dyn WorkDispatcher>;
            let catalog = PhysicalLaneCatalog::try_from_raw([(
                "default".to_string(),
                "default".to_string(),
                "default".to_string(),
            )])
            .expect("catalog");
            let demand_tracker = Arc::new(DemandTracker::new(catalog));
            let physical_lane = demand_tracker
                .resolve_lane("default", "default", "default")
                .expect("lane");
            let (chunk_tx, chunk_rx) = tokio::sync::broadcast::channel(4);
            drop(chunk_tx);
            let (outcome_tx, outcome_rx) = tokio::sync::oneshot::channel();
            outcome_tx
                .send(StreamOutcome {
                    text: String::new(),
                    finish_reason: "error".to_string(),
                    usage: None,
                    attempt_id: String::new(),
                    ttft_ms: None,
                    tpot_ms: None,
                    error: Some(ChunkError {
                        code: "rate_limit_exceeded".to_string(),
                        message: "pool republish failed".to_string(),
                        param: None,
                        retry_after_s: None,
                    }),
                    origin: StreamOutcomeOrigin::GatewaySynthetic,
                    tool_calls: None,
                    logprobs: None,
                    candidates: Vec::new(),
                    executed_bundle_config_hash: None,
                    execution_identity_sha256: None,
                    execution_binding_sha256: None,
                })
                .expect("outcome receiver is live");
            let (durability_tx, durability_completion) = tokio::sync::oneshot::channel();
            durability_tx
                .send(Ok::<(), String>(()))
                .expect("durability receiver is live");
            let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(4);

            run_sse_driver(SseDriverArgs {
                event_tx,
                chunk_rx,
                outcome_rx,
                durability_completion,
                publisher,
                demand_tracker,
                physical_lane,
                request_id: "req-synthetic".to_string(),
                model: "test/model".to_string(),
                pool: "default".to_string(),
                bundle: "default".to_string(),
                gpu: "test".to_string(),
                endpoint,
                stream_chat_id: "chatcmpl-synthetic".to_string(),
                created: 1,
                first_chunk_timeout: Duration::from_secs(30),
                inter_chunk_timeout: Duration::from_secs(30),
                overall_timeout: Duration::from_secs(60),
                was_direct_dispatched: false,
                pool_fallback_lane_worker_count: 1,
            })
            .await;

            let error = _event_data(
                event_rx
                    .recv()
                    .await
                    .expect("synthetic error event")
                    .expect("infallible event"),
            )
            .await;
            let done = _event_data(
                event_rx
                    .recv()
                    .await
                    .expect("done event")
                    .expect("infallible event"),
            )
            .await;
            let error: Value = serde_json::from_str(&error).expect("synthetic error JSON");
            assert_eq!(error["error"]["code"], "rate_limit_exceeded");
            assert_eq!(error["error"]["message"], "pool republish failed");
            assert_eq!(done, "[DONE]");
            assert!(event_rx.recv().await.is_none());
        }
    }

    #[tokio::test]
    async fn live_driver_worker_validation_error_keeps_openai_type_and_exact_param() {
        for endpoint in [
            SseEndpoint::Chat {
                include_usage: false,
            },
            SseEndpoint::Completion {
                include_usage: false,
            },
            SseEndpoint::Generate,
        ] {
            let payloads = run_driver_worker_error_race(
                endpoint,
                WorkerTerminalDelivery::BackloggedBeforeFirstPoll,
                ChunkError {
                    code: "unsupported_field".to_string(),
                    message: "top_k is unavailable".to_string(),
                    param: Some("top_k".to_string()),
                    retry_after_s: None,
                },
            )
            .await;
            let error_payload = payloads
                .iter()
                .filter_map(|payload| serde_json::from_str::<Value>(payload).ok())
                .find(|payload| payload.get("error").is_some())
                .expect("worker error JSON");
            assert_eq!(error_payload["error"]["code"], "unsupported_field");
            assert_eq!(error_payload["error"]["param"], "top_k");
            match endpoint {
                SseEndpoint::Chat { .. } | SseEndpoint::Completion { .. } => {
                    assert_eq!(error_payload["error"]["type"], "invalid_request_error");
                }
                SseEndpoint::Generate => {
                    assert!(error_payload["error"].get("type").is_none());
                }
            }
        }
    }

    #[tokio::test]
    async fn terminal_durability_ready_wins_over_closed_client_and_deadline() {
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(1);
        drop(event_rx);
        let (completion_tx, mut completion_rx) = tokio::sync::oneshot::channel();
        completion_tx.send(Ok(())).expect("send durability result");

        let result = wait_for_terminal_durability(
            &mut completion_rx,
            &event_tx,
            tokio::time::Instant::now(),
        )
        .await;

        assert_eq!(result, TerminalDurabilityWait::Confirmed);
    }

    #[tokio::test]
    async fn terminal_durability_wait_preserves_overall_deadline() {
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(1);
        let (_completion_tx, mut completion_rx) = tokio::sync::oneshot::channel();

        let result = wait_for_terminal_durability(
            &mut completion_rx,
            &event_tx,
            tokio::time::Instant::now(),
        )
        .await;

        assert_eq!(result, TerminalDurabilityWait::OverallTimeout);
    }

    #[tokio::test]
    async fn terminal_durability_wait_observes_client_disconnect() {
        let (event_tx, event_rx) = tokio::sync::mpsc::channel(1);
        drop(event_rx);
        let (_completion_tx, mut completion_rx) = tokio::sync::oneshot::channel();

        let result = wait_for_terminal_durability(
            &mut completion_rx,
            &event_tx,
            tokio::time::Instant::now() + Duration::from_secs(60),
        )
        .await;

        assert_eq!(result, TerminalDurabilityWait::ClientClosed);
    }

    /// Streaming path must preserve ``content_filter`` / ``function_call``
    /// (valid OpenAI finish reasons emitted by the worker) rather than
    /// collapse them to ``stop`` — kept in lockstep with
    /// ``proxy::map_chat_finish_reason``.
    #[test]
    fn test_map_chat_finish_reason_preserves_content_filter() {
        assert_eq!(map_chat_finish_reason("content_filter"), "content_filter");
        assert_eq!(map_chat_finish_reason("function_call"), "function_call");
    }

    /// A chunk carrying logprobs surfaces them per-chunk in the OpenAI
    /// ``{content: [...], refusal: null}`` shape on the streaming choice.
    #[test]
    fn test_build_chat_chunk_event_emits_logprobs() {
        let mut chunk = _delta_chunk(0, "Hi");
        chunk.logprobs = Some(vec![serde_json::json!({
            "token": "Hi",
            "logprob": -0.5,
            "bytes": [72, 105],
            "top_logprobs": [],
        })]);
        let ev = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, true);
        let lp = &ev["choices"][0]["logprobs"];
        assert_eq!(lp["content"][0]["token"], "Hi");
        assert!(lp["refusal"].is_null());
    }

    /// A chunk without logprobs keeps ``logprobs`` null (shape parity).
    #[test]
    fn test_build_chat_chunk_event_logprobs_null_when_absent() {
        let chunk = _delta_chunk(0, "Hi");
        let ev = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, true);
        assert!(ev["choices"][0]["logprobs"].is_null());
    }

    fn _delta_chunk(seq: u32, text: &str) -> ChunkEnvelope {
        ChunkEnvelope {
            kind: "chunk".to_string(),
            request_id: "req-test".to_string(),
            attempt_id: "att-1".to_string(),
            seq,
            text_delta: text.to_string(),
            done: false,
            is_first: seq == 0,
            finish_reason: None,
            usage: None,
            ttft_ms: None,
            error: None,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            choice_index: 0,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        }
    }

    fn _terminal_chunk(finish_reason: &str, usage: Option<UsageBlock>) -> ChunkEnvelope {
        ChunkEnvelope {
            kind: "chunk".to_string(),
            request_id: "req-test".to_string(),
            attempt_id: "att-1".to_string(),
            seq: 99,
            text_delta: String::new(),
            done: true,
            is_first: false,
            finish_reason: Some(finish_reason.to_string()),
            usage,
            ttft_ms: Some(12.5),
            error: None,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            choice_index: 0,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        }
    }

    // ── Chat chunk shape ───────────────────────────────────────────

    /// First chunk emits ``delta.role = "assistant"`` per the OpenAI
    /// streaming contract; subsequent chunks omit it.
    #[test]
    fn test_sse_chat_first_chunk_emits_role() {
        let chunk = _delta_chunk(0, "Hello");
        let v = build_chat_chunk_event("chatcmpl-1", 1_700_000_000, "m", &chunk, true);
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["id"], "chatcmpl-1");
        assert_eq!(v["model"], "m");
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(v["choices"][0]["delta"]["content"], "Hello");
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn test_sse_chat_chunk_carries_choice_index() {
        // Streaming n>1: a per-candidate delta with choice_index=2 surfaces as
        // choices[0].index = 2 so clients can reassemble per-candidate streams.
        let mut chunk = _delta_chunk(0, "B");
        chunk.choice_index = 2;
        let v = build_chat_chunk_event("chatcmpl-1", 1_700_000_000, "m", &chunk, false);
        assert_eq!(v["choices"][0]["index"], 2);
        assert_eq!(v["choices"][0]["delta"]["content"], "B");
    }

    /// H4: per-choice ``finish_reason`` rides on non-terminal chunks too —
    /// the worker emits a non-``done`` chunk with ``finish_reason`` set
    /// when a specific candidate in a streaming ``n>1`` run completes.
    /// The SSE builder must propagate it (not gate on ``done`` like the
    /// pre-fix shape).
    #[test]
    fn test_sse_chat_per_choice_finish_reason_on_non_done() {
        let mut chunk = _delta_chunk(5, "last");
        chunk.choice_index = 1;
        chunk.finish_reason = Some("length".to_string());
        // done=false — this is the per-choice closure, not the global terminal.
        chunk.done = false;
        let v = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false);
        assert_eq!(v["choices"][0]["index"], 1);
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert_eq!(v["choices"][0]["delta"]["content"], "last");
    }

    /// H4: per-choice logprobs surface on the per-candidate streaming chunk
    /// (not just on the single-candidate path). Each ``choice_index`` gets
    /// its own slice; the SSE encoder wraps each in the OpenAI
    /// ``{content: [...], refusal: null}`` envelope.
    #[test]
    fn test_sse_chat_per_choice_logprobs_attach() {
        let mut chunk = _delta_chunk(2, "tok");
        chunk.choice_index = 0;
        chunk.logprobs = Some(vec![serde_json::json!({
            "token": "tok",
            "logprob": -0.5,
            "bytes": [116, 111, 107],
            "top_logprobs": [],
        })]);
        let v = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false);
        let lps = &v["choices"][0]["logprobs"];
        assert!(!lps.is_null());
        assert_eq!(lps["content"][0]["token"], "tok");
        assert_eq!(lps["content"][0]["logprob"], -0.5);
    }

    #[test]
    fn test_build_text_completion_chunk_event_delta() {
        let chunk = _delta_chunk(0, "hi");
        let v = build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk);
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["text"], "hi");
        assert_eq!(v["choices"][0]["index"], 0);
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn test_build_text_completion_chunk_event_terminal_finish() {
        let chunk = _terminal_chunk("length", None);
        let v = build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk);
        assert_eq!(v["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn test_sse_text_completion_model_load_failed_uses_safe_public_message() {
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: crate::queue::streaming::MODEL_LOAD_FAILED_ERROR_CODE.to_string(),
            message: "SENSITIVE_WORKER_FAILURE_SENTINEL".to_string(),
            param: None,
            retry_after_s: None,
        });

        let value = build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk);
        assert!(value["choices"][0]["finish_reason"].is_null());
        assert_eq!(
            value["error"]["code"],
            crate::queue::streaming::MODEL_LOAD_FAILED_ERROR_CODE
        );
        assert_eq!(
            value["error"]["message"],
            crate::queue::streaming::MODEL_LOAD_FAILED_PUBLIC_MESSAGE
        );
        assert_eq!(value["error"]["type"], "server_error");
        assert!(value["error"]["param"].is_null());
        assert!(!value
            .to_string()
            .contains("SENSITIVE_WORKER_FAILURE_SENTINEL"));
    }

    #[test]
    fn test_build_text_completion_chunk_event_preserves_worker_error() {
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: "unsupported_field".to_string(),
            message: "top_k is unavailable".to_string(),
            param: Some("top_k".to_string()),
            retry_after_s: None,
        });

        let value = build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk);

        assert!(value["choices"][0]["finish_reason"].is_null());
        assert_eq!(value["error"]["code"], "unsupported_field");
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["param"], "top_k");
        assert_eq!(value["error"]["message"], "top_k is unavailable");
        assert_eq!(value["request_id"], "req-test");
    }

    #[test]
    fn test_sse_chat_subsequent_chunk_omits_role() {
        let chunk = _delta_chunk(1, " world");
        let v = build_chat_chunk_event("chatcmpl-1", 1_700_000_000, "m", &chunk, false);
        let delta = &v["choices"][0]["delta"];
        assert!(delta.get("role").is_none(), "role must not be re-emitted");
        assert_eq!(delta["content"], " world");
    }

    /// Terminal chunk carries OpenAI ``finish_reason``; SIE-native
    /// ``length`` is preserved, anything else collapses to ``stop``.
    #[test]
    fn test_sse_chat_terminal_finish_reason_stop() {
        let chunk = _terminal_chunk("stop", None);
        let v = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false);
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        // The terminal chunk has an empty delta — content must not
        // surface, but the delta object itself is still present.
        assert!(v["choices"][0]["delta"].get("content").is_none());
    }

    #[test]
    fn test_sse_chat_terminal_finish_reason_length_preserved() {
        let chunk = _terminal_chunk("length", None);
        let v = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false);
        assert_eq!(v["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn test_sse_successful_terminal_exposes_execution_evidence() {
        let identity = "a".repeat(64);
        let binding = "b".repeat(64);
        let mut chunk = _terminal_chunk("stop", None);
        chunk.execution_identity_sha256 = Some(identity.clone());
        chunk.execution_binding_sha256 = Some(binding.clone());

        for value in [
            build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false),
            build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk),
            build_generate_chunk_event(&chunk, &[]),
        ] {
            assert_eq!(value["execution_identity_sha256"], identity);
            assert_eq!(value["execution_binding_sha256"], binding);
        }
    }

    #[test]
    fn test_sse_execution_evidence_requires_complete_successful_terminal_pair() {
        let mut delta = _delta_chunk(0, "Hi");
        delta.execution_identity_sha256 = Some("a".repeat(64));
        delta.execution_binding_sha256 = Some("b".repeat(64));
        let mut partial = _terminal_chunk("stop", None);
        partial.execution_identity_sha256 = Some("a".repeat(64));
        let mut error = _terminal_chunk("error", None);
        error.execution_identity_sha256 = Some("a".repeat(64));
        error.execution_binding_sha256 = Some("b".repeat(64));
        error.error = Some(ChunkError {
            code: "transport_failure".to_string(),
            message: "upstream gone".to_string(),
            param: None,
            retry_after_s: None,
        });

        for value in [
            build_generate_chunk_event(&delta, &[]),
            build_generate_chunk_event(&partial, &[]),
            build_generate_chunk_event(&error, &[]),
        ] {
            assert!(value.get("execution_identity_sha256").is_none());
            assert!(value.get("execution_binding_sha256").is_none());
        }
    }

    /// Worker-error chunks surface an ``error`` block alongside the
    /// normal envelope and trigger the SDK error path.
    #[test]
    fn test_sse_chat_error_chunk_attaches_error_block() {
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: "unsupported_field".to_string(),
            message: "top_k is unavailable".to_string(),
            param: Some("top_k".to_string()),
            retry_after_s: None,
        });
        let v = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false);
        assert_eq!(v["error"]["code"], "unsupported_field");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "top_k");
        assert_eq!(v["error"]["message"], "top_k is unavailable");
        // #3136: the ERROR chunk carries the gateway request id in-band
        // (additive member; OpenAI clients ignore it) so SDK consumers can
        // correlate stream failures with gateway logs.
        assert_eq!(v["request_id"], "req-test");
    }

    #[test]
    fn test_sse_chat_model_load_failed_uses_safe_public_message() {
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: crate::queue::streaming::MODEL_LOAD_FAILED_ERROR_CODE.to_string(),
            message: "SENSITIVE_WORKER_FAILURE_SENTINEL".to_string(),
            param: None,
            retry_after_s: None,
        });

        let value = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false);
        assert_eq!(
            value["error"]["code"],
            crate::queue::streaming::MODEL_LOAD_FAILED_ERROR_CODE
        );
        assert_eq!(
            value["error"]["message"],
            crate::queue::streaming::MODEL_LOAD_FAILED_PUBLIC_MESSAGE
        );
        assert!(!value
            .to_string()
            .contains("SENSITIVE_WORKER_FAILURE_SENTINEL"));
    }

    /// #3136 scope guard: ``request_id`` rides ERROR chunks only — normal
    /// deltas and clean terminals keep the exact pre-existing chat shape.
    #[test]
    fn test_sse_chat_non_error_chunks_omit_request_id() {
        let delta = build_chat_chunk_event("chatcmpl-1", 0, "m", &_delta_chunk(0, "Hi"), true);
        assert!(delta.get("request_id").is_none());
        let terminal =
            build_chat_chunk_event("chatcmpl-1", 0, "m", &_terminal_chunk("stop", None), false);
        assert!(terminal.get("request_id").is_none());
    }

    // ── Generate (SIE-native) chunk shape ─────────────────────────

    #[test]
    fn test_sse_generate_delta_shape() {
        let chunk = _delta_chunk(3, "tok");
        let v = build_generate_chunk_event(&chunk, &[]);
        assert_eq!(v["request_id"], "req-test");
        assert_eq!(v["seq"], 3);
        assert_eq!(v["text_delta"], "tok");
        assert_eq!(v["done"], false);
        assert!(v.get("usage").is_none(), "usage absent on non-terminal");
    }

    #[test]
    fn test_sse_generate_delta_includes_requested_logprobs() {
        let mut chunk = _delta_chunk(0, "tok");
        chunk.logprobs = Some(vec![json!({
            "token": "tok",
            "logprob": -0.25,
            "bytes": [116, 111, 107],
            "top_logprobs": [],
        })]);
        let v = build_generate_chunk_event(&chunk, &[]);
        assert_eq!(v["logprobs"][0]["token"], "tok");
        assert_eq!(v["logprobs"][0]["logprob"], -0.25);
    }

    #[test]
    fn test_sse_generate_terminal_includes_usage_and_finish_reason() {
        let chunk = _terminal_chunk(
            "stop",
            Some(UsageBlock {
                prompt_tokens: 10,
                completion_tokens: 7,
                total_tokens: 17,
            }),
        );
        let v = build_generate_chunk_event(&chunk, &[]);
        assert_eq!(v["done"], true);
        assert_eq!(v["finish_reason"], "stop");
        assert_eq!(v["usage"]["prompt_tokens"], 10);
        assert_eq!(v["usage"]["completion_tokens"], 7);
        assert_eq!(v["usage"]["total_tokens"], 17);
        // TTFT is forwarded when the worker provides it.
        assert_eq!(v["ttft_ms"], 12.5);
    }

    #[test]
    fn test_sse_generate_error_chunk_attaches_error() {
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: "transport_failure".to_string(),
            message: "upstream gone".to_string(),
            param: None,
            retry_after_s: None,
        });
        let v = build_generate_chunk_event(&chunk, &[]);
        assert_eq!(v["error"]["code"], "transport_failure");
        assert_eq!(v["error"]["message"], "upstream gone");
    }

    #[test]
    fn test_sse_generate_model_load_failed_uses_safe_public_message() {
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: crate::queue::streaming::MODEL_LOAD_FAILED_ERROR_CODE.to_string(),
            message: "SENSITIVE_WORKER_FAILURE_SENTINEL".to_string(),
            param: None,
            retry_after_s: None,
        });

        let value = build_generate_chunk_event(&chunk, &[]);
        assert_eq!(
            value["error"]["code"],
            crate::queue::streaming::MODEL_LOAD_FAILED_ERROR_CODE
        );
        assert_eq!(
            value["error"]["message"],
            crate::queue::streaming::MODEL_LOAD_FAILED_PUBLIC_MESSAGE
        );
        assert!(!value
            .to_string()
            .contains("SENSITIVE_WORKER_FAILURE_SENTINEL"));
    }

    #[test]
    fn test_sse_unknown_worker_errors_fail_closed_on_all_generation_shapes() {
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: "SENSITIVE_WORKER_ERROR_CODE_SENTINEL".to_string(),
            message: "SENSITIVE_WORKER_FAILURE_SENTINEL".to_string(),
            param: None,
            retry_after_s: None,
        });

        let values = [
            build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false),
            build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk),
            build_generate_chunk_event(&chunk, &[]),
        ];
        for value in values {
            assert_eq!(value["error"]["code"], "inference_error");
            assert_eq!(
                value["error"]["message"],
                crate::queue::streaming::UNKNOWN_WORKER_ERROR_PUBLIC_MESSAGE
            );
            let serialized = value.to_string();
            assert!(!serialized.contains("SENSITIVE_WORKER_ERROR_CODE_SENTINEL"));
            assert!(!serialized.contains("SENSITIVE_WORKER_FAILURE_SENTINEL"));
        }
    }

    #[test]
    fn test_sse_payload_too_large_preserves_worker_code_on_all_generation_shapes() {
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: "PAYLOAD_TOO_LARGE".to_string(),
            message: "payload exceeds limit".to_string(),
            param: None,
            retry_after_s: None,
        });

        for value in [
            build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false),
            build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk),
            build_generate_chunk_event(&chunk, &[]),
        ] {
            assert_eq!(value["error"]["code"], "PAYLOAD_TOO_LARGE");
            assert_eq!(value["error"]["message"], "payload exceeds limit");
        }
    }

    #[test]
    fn worker_error_sse_shapes_preserve_validated_retry_after() {
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: "RESOURCE_EXHAUSTED".to_string(),
            message: "resource pressure".to_string(),
            param: Some("model".to_string()),
            retry_after_s: Some(12),
        });

        let chat = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false);
        let completion = build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk);
        let generate = build_generate_chunk_event(&chunk, &[]);

        for value in [&chat, &completion, &generate] {
            assert_eq!(value["error"]["retry_after_s"], 12);
            assert!(value["error"]["param"].is_null());
        }
    }

    #[test]
    fn worker_error_sse_shapes_omit_untrusted_or_wrong_code_retry_after() {
        for error in [
            ChunkError {
                code: "RESOURCE_EXHAUSTED".to_string(),
                message: "resource pressure".to_string(),
                param: None,
                retry_after_s: Some(61),
            },
            ChunkError {
                code: "MODEL_LOADING".to_string(),
                message: "draining".to_string(),
                param: None,
                retry_after_s: Some(12),
            },
        ] {
            let mut chunk = _terminal_chunk("error", None);
            chunk.error = Some(error);

            for value in [
                build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false),
                build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk),
                build_generate_chunk_event(&chunk, &[]),
            ] {
                assert!(value["error"].get("retry_after_s").is_none());
            }
        }
    }

    #[test]
    fn worker_inference_error_sse_shapes_sanitize_untrusted_message() {
        let sentinel = "SENSITIVE_BACKEND_SENTINEL";
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: "inference_error".to_string(),
            message: sentinel.to_string(),
            param: Some(sentinel.to_string()),
            retry_after_s: None,
        });

        let chat = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false);
        let completion = build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk);
        let generate = build_generate_chunk_event(&chunk, &[]);

        for value in [&chat, &completion, &generate] {
            assert_eq!(
                value["error"]["message"],
                "internal error during generation"
            );
            assert!(value["error"]["param"].is_null());
            assert!(!value.to_string().contains(sentinel));
        }
    }

    #[test]
    fn worker_grammar_compile_error_sanitizes_untrusted_message() {
        let sentinel = "SENSITIVE_GRAMMAR_SENTINEL";
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: "grammar_compile_failed".to_string(),
            message: sentinel.to_string(),
            param: Some(sentinel.to_string()),
            retry_after_s: None,
        });

        let value = build_generate_chunk_event(&chunk, &[]);

        assert_eq!(
            value["error"]["message"],
            "internal error compiling grammar"
        );
        assert!(value["error"]["param"].is_null());
        assert!(!value.to_string().contains(sentinel));
    }

    #[test]
    fn unknown_worker_error_sse_shapes_sanitize_untrusted_message() {
        let sentinel = "SENSITIVE_WORKER_ERROR_CODE_SENTINEL";
        let mut chunk = _terminal_chunk("error", None);
        chunk.error = Some(ChunkError {
            code: sentinel.to_string(),
            message: sentinel.to_string(),
            param: Some(sentinel.to_string()),
            retry_after_s: None,
        });

        let chat = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false);
        let completion = build_text_completion_chunk_event("cmpl-1", 0, "m", &chunk);
        let generate = build_generate_chunk_event(&chunk, &[]);

        for value in [&chat, &completion, &generate] {
            assert_eq!(
                value["error"]["message"],
                crate::queue::streaming::UNKNOWN_WORKER_ERROR_PUBLIC_MESSAGE
            );
            assert_eq!(value["error"]["code"], "inference_error");
            assert!(value["error"]["param"].is_null());
            assert!(!value.to_string().contains(sentinel));
        }
    }

    // ── Synthesized error chunks (gateway-side timeouts) ──────────

    #[tokio::test]
    async fn test_sse_send_error_chunk_chat_shape() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        send_error_chunk(
            &tx,
            &SseEndpoint::Chat {
                include_usage: false,
            },
            "chatcmpl-x",
            1,
            "m",
            "req-1",
            "first_chunk_timeout",
            "Generation aborted: first_chunk timeout",
        )
        .await;
        let evt = rx.recv().await.expect("event").expect("ok");
        let payload = _event_data(evt).await;
        let v: Value = serde_json::from_str(&payload).expect("json");
        assert_eq!(v["object"], "chat.completion.chunk");
        // An error chunk must not claim a clean `stop` finish_reason.
        assert!(v["choices"][0]["finish_reason"].is_null());
        assert_eq!(v["error"]["code"], "first_chunk_timeout");
        assert_eq!(v["error"]["type"], "server_error");
        // #3136: gateway-synthesized chat error chunks carry the gateway
        // request id in-band, mirroring the generate shape.
        assert_eq!(v["request_id"], "req-1");
    }

    #[tokio::test]
    async fn test_sse_send_error_chunk_generate_shape() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        send_error_chunk(
            &tx,
            &SseEndpoint::Generate,
            "unused",
            0,
            "m",
            "req-42",
            "overall_timeout",
            "Generation aborted: overall timeout",
        )
        .await;
        let evt = rx.recv().await.expect("event").expect("ok");
        let v: Value = serde_json::from_str(&_event_data(evt).await).expect("json");
        assert_eq!(v["request_id"], "req-42");
        assert_eq!(v["done"], true);
        assert_eq!(v["finish_reason"], "error");
        assert_eq!(v["error"]["code"], "overall_timeout");
    }

    #[tokio::test]
    async fn test_sse_outcome_worker_errors_use_safe_contract_on_all_endpoints() {
        let cases = [
            (
                crate::queue::streaming::MODEL_LOAD_FAILED_ERROR_CODE,
                crate::queue::streaming::MODEL_LOAD_FAILED_ERROR_CODE,
                crate::queue::streaming::MODEL_LOAD_FAILED_PUBLIC_MESSAGE,
            ),
            (
                "SENSITIVE_WORKER_ERROR_CODE_SENTINEL",
                "inference_error",
                crate::queue::streaming::UNKNOWN_WORKER_ERROR_PUBLIC_MESSAGE,
            ),
            (
                "model_load_failed",
                "inference_error",
                crate::queue::streaming::UNKNOWN_WORKER_ERROR_PUBLIC_MESSAGE,
            ),
        ];

        for (worker_code, expected_code, expected_message) in cases {
            for endpoint in [
                SseEndpoint::Generate,
                SseEndpoint::Chat {
                    include_usage: false,
                },
                SseEndpoint::Completion {
                    include_usage: false,
                },
            ] {
                let error = ChunkError {
                    code: worker_code.to_string(),
                    message: "SENSITIVE_WORKER_FAILURE_SENTINEL".to_string(),
                    param: None,
                    retry_after_s: None,
                };
                let (tx, mut rx) = tokio::sync::mpsc::channel(4);
                send_synthetic_error_chunk(&tx, &endpoint, "cmpl-1", 1, "m", "req-1", &error).await;
                send_done(&tx).await;

                let error_event = rx.recv().await.expect("error event").expect("valid event");
                let value: Value = serde_json::from_str(&_event_data(error_event).await)
                    .expect("valid worker error JSON");
                assert_eq!(value["error"]["code"], expected_code);
                assert_eq!(value["error"]["message"], expected_message);
                assert!(!value
                    .to_string()
                    .contains("SENSITIVE_WORKER_FAILURE_SENTINEL"));
                match endpoint {
                    SseEndpoint::Generate => {
                        assert_eq!(value["finish_reason"], "error");
                    }
                    SseEndpoint::Chat { .. } | SseEndpoint::Completion { .. } => {
                        assert!(value["choices"][0]["finish_reason"].is_null());
                        assert_eq!(value["error"]["type"], "server_error");
                        assert!(value["error"]["param"].is_null());
                        assert_eq!(value["request_id"], "req-1");
                    }
                }

                let done_event = rx.recv().await.expect("done event").expect("valid event");
                assert_eq!(_event_data(done_event).await, "[DONE]");
            }
        }
    }

    #[tokio::test]
    async fn test_sse_done_terminator_literal() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        send_done(&tx).await;
        let evt = rx.recv().await.expect("event").expect("ok");
        assert_eq!(_event_data(evt).await, "[DONE]");
    }

    // ── StreamCollector + broadcast tap integration ───────────────

    /// The chunk tap installed on a `StreamCollector` fans out every
    /// non-stale chunk applied by `apply()` to the subscriber. Stale
    /// chunks (mismatched attempt_id) must NOT reach the tap — the
    /// streaming drop-logic precedes the fan-out.
    #[tokio::test]
    async fn test_collector_tap_forwards_non_stale_chunks() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut collector =
            crate::queue::streaming::StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let mut tap = collector.install_chunk_tap();

        // First chunk latches attempt_id "A" and goes to the tap.
        collector.apply(_delta_chunk(0, "Hi"));
        let got = tap.recv().await.expect("recv");
        assert_eq!(got.text_delta, "Hi");
        assert_eq!(got.attempt_id, "att-1");

        // A stale chunk (different attempt_id) must NOT reach the tap.
        // Use a seq that *would* be contiguous on the live attempt so a
        // gap-rejection (H6) cannot mask the stale-attempt rejection
        // path this test is exercising.
        let mut stale = _delta_chunk(1, "ignored");
        stale.attempt_id = "att-B".to_string();
        collector.apply(stale);

        // The next legitimate chunk reaches the tap; the stale one
        // is silently absent from the broadcast stream. The seq must
        // be contiguous with the live attempt's watermark (last
        // accepted was seq 0, so the next legit seq is 1) — the H6
        // gap-rejection would otherwise drop a seq=2 chunk and leave
        // the tap empty.
        collector.apply(_delta_chunk(1, "next"));
        let got = tap.recv().await.expect("recv");
        assert_eq!(got.text_delta, "next");
        // No third event is pending.
        assert!(
            matches!(
                tap.try_recv(),
                Err(tokio::sync::broadcast::error::TryRecvError::Empty)
            ),
            "stale chunk leaked into the tap"
        );
    }

    /// End-to-end test of the per-chunk event builders driven against
    /// a real `StreamCollector` + broadcast tap. The chat handler's
    /// SSE loop builds the same sequence of events from the same tap.
    /// This test asserts the shapes the wire would carry without
    /// spinning up the (tokio-task-spawning) `run_sse_driver`.
    ///
    /// Sequence:
    ///   1. `delta(0, "Hello")` → first chat chunk with role
    ///   2. `delta(1, " world")` → chunk with content only
    ///   3. terminal with usage → final chunk with finish_reason="stop"
    ///      followed (when include_usage) by a usage chunk and `[DONE]`
    #[tokio::test]
    async fn test_sse_response_emits_chat_completion_chunks() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut collector =
            crate::queue::streaming::StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let mut tap = collector.install_chunk_tap();

        collector.apply(_delta_chunk(0, "Hello"));
        collector.apply(_delta_chunk(1, " world"));
        collector.apply(_terminal_chunk(
            "stop",
            Some(UsageBlock {
                prompt_tokens: 2,
                completion_tokens: 2,
                total_tokens: 4,
            }),
        ));

        // First chunk — must carry role.
        let c1 = tap.recv().await.unwrap();
        let v1 = build_chat_chunk_event("chatcmpl-1", 0, "m", &c1, true);
        assert_eq!(v1["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(v1["choices"][0]["delta"]["content"], "Hello");
        assert!(v1["choices"][0]["finish_reason"].is_null());

        // Second chunk — role omitted, content only.
        let c2 = tap.recv().await.unwrap();
        let v2 = build_chat_chunk_event("chatcmpl-1", 0, "m", &c2, false);
        assert!(v2["choices"][0]["delta"].get("role").is_none());
        assert_eq!(v2["choices"][0]["delta"]["content"], " world");

        // Terminal — finish_reason populated, content empty.
        let c3 = tap.recv().await.unwrap();
        assert!(c3.done);
        let v3 = build_chat_chunk_event("chatcmpl-1", 0, "m", &c3, false);
        assert_eq!(v3["choices"][0]["finish_reason"], "stop");
    }

    /// With ``stream_options.include_usage: true``, the SSE stream appends a
    /// usage-only chunk (``choices: []``) before ``[DONE]`` — on chat AND on
    /// legacy completions, which is the only place a streamed request can
    /// report what it consumed (headers are gone after the first byte).
    #[test]
    fn test_sse_response_emits_usage_chunk_when_requested() {
        let terminal = _terminal_chunk(
            "stop",
            Some(UsageBlock {
                prompt_tokens: 5,
                completion_tokens: 7,
                total_tokens: 12,
            }),
        );
        for (endpoint, object) in [
            (
                SseEndpoint::Chat {
                    include_usage: true,
                },
                "chat.completion.chunk",
            ),
            (
                SseEndpoint::Completion {
                    include_usage: true,
                },
                "text_completion",
            ),
        ] {
            let body = build_usage_only_chunk_event(endpoint, "cmpl-1", 1700, "m", &terminal, &[])
                .expect("include_usage with an authoritative terminal emits the usage-only chunk");
            assert_eq!(body["object"], object);
            assert!(body["choices"].as_array().unwrap().is_empty());
            assert_eq!(body["usage"]["prompt_tokens"], 5);
            assert_eq!(body["usage"]["completion_tokens"], 7);
            assert_eq!(body["usage"]["total_tokens"], 12);
            assert!(body["system_fingerprint"]
                .as_str()
                .unwrap()
                .starts_with("fp_"));
        }
    }

    /// No usage chunk without the opt-in, and never a synthesised one: a
    /// terminal that carried no authoritative usage reports nothing.
    #[test]
    fn test_sse_usage_chunk_is_opt_in_and_never_synthesised() {
        let terminal = _terminal_chunk(
            "stop",
            Some(UsageBlock {
                prompt_tokens: 5,
                completion_tokens: 7,
                total_tokens: 12,
            }),
        );
        for endpoint in [
            SseEndpoint::Chat {
                include_usage: false,
            },
            SseEndpoint::Completion {
                include_usage: false,
            },
            SseEndpoint::Generate,
        ] {
            assert!(
                build_usage_only_chunk_event(endpoint, "cmpl-1", 1700, "m", &terminal, &[])
                    .is_none()
            );
        }
        let countless = _terminal_chunk("error", None);
        assert!(build_usage_only_chunk_event(
            SseEndpoint::Completion {
                include_usage: true
            },
            "cmpl-1",
            1700,
            "m",
            &countless,
            &[],
        )
        .is_none());
    }

    /// A transport's terminal-only members ride the same usage block the
    /// worker's counts do — on the OpenAI usage-only chunk and on the native
    /// generate terminal alike.
    #[test]
    fn terminal_usage_extras_ride_the_usage_block_on_every_stream_surface() {
        let extras = vec![
            ("credits_charged".to_string(), json!(42)),
            ("rate_book_version".to_string(), json!("book-v1")),
        ];
        let terminal = _terminal_chunk(
            "stop",
            Some(UsageBlock {
                prompt_tokens: 5,
                completion_tokens: 7,
                total_tokens: 12,
            }),
        );
        for endpoint in [
            SseEndpoint::Chat {
                include_usage: true,
            },
            SseEndpoint::Completion {
                include_usage: true,
            },
        ] {
            let body =
                build_usage_only_chunk_event(endpoint, "cmpl-1", 1700, "m", &terminal, &extras)
                    .expect("usage chunk");
            assert_eq!(body["usage"]["credits_charged"], json!(42));
            assert_eq!(body["usage"]["rate_book_version"], json!("book-v1"));
            assert_eq!(body["usage"]["total_tokens"], json!(12));
        }
        let native = build_generate_chunk_event(&terminal, &extras);
        assert_eq!(native["usage"]["credits_charged"], json!(42));
        assert_eq!(native["usage"]["rate_book_version"], json!("book-v1"));
        assert_eq!(native["usage"]["completion_tokens"], json!(7));
    }

    /// The extras are additive only. They never conjure a usage block for a
    /// terminal that reported no counts, they never appear on a non-terminal
    /// chunk, and they never overwrite a worker-authoritative count.
    #[test]
    fn terminal_usage_extras_never_fabricate_or_overwrite() {
        let extras = vec![
            ("credits_charged".to_string(), json!(42)),
            ("prompt_tokens".to_string(), json!(999)),
        ];

        // No counts on the terminal → no usage surface at all, extras or not.
        let countless = _terminal_chunk("error", None);
        assert!(build_usage_only_chunk_event(
            SseEndpoint::Chat {
                include_usage: true
            },
            "cmpl-1",
            1700,
            "m",
            &countless,
            &extras,
        )
        .is_none());
        assert!(build_generate_chunk_event(&countless, &extras)
            .get("usage")
            .is_none());

        // A mid-stream delta carries no usage block, so nothing can ride it.
        let mut delta = _terminal_chunk("stop", None);
        delta.done = false;
        delta.text_delta = "hi".to_string();
        assert!(build_generate_chunk_event(&delta, &extras)
            .get("usage")
            .is_none());

        // A worker count always wins over a same-named extra.
        let terminal = _terminal_chunk(
            "stop",
            Some(UsageBlock {
                prompt_tokens: 5,
                completion_tokens: 7,
                total_tokens: 12,
            }),
        );
        let body = build_usage_only_chunk_event(
            SseEndpoint::Chat {
                include_usage: true,
            },
            "cmpl-1",
            1700,
            "m",
            &terminal,
            &extras,
        )
        .expect("usage chunk");
        assert_eq!(
            body["usage"]["prompt_tokens"],
            json!(5),
            "an extra must never overwrite an authoritative worker count"
        );
        assert_eq!(body["usage"]["credits_charged"], json!(42));
    }

    /// The transport is told whether its members can be published at all, so a
    /// stream that emits no usage surface never makes the client wait for one.
    /// This must agree exactly with `build_usage_only_chunk_event`'s own
    /// `include_usage` gate — a disagreement either stalls `[DONE]` for nothing
    /// or drops a charge the surface would have carried.
    #[test]
    fn terminal_carries_usage_agrees_with_the_surface_that_emits_it() {
        let terminal = _terminal_chunk(
            "stop",
            Some(UsageBlock {
                prompt_tokens: 5,
                completion_tokens: 7,
                total_tokens: 12,
            }),
        );
        for include_usage in [true, false] {
            for endpoint in [
                SseEndpoint::Chat { include_usage },
                SseEndpoint::Completion { include_usage },
            ] {
                assert_eq!(terminal_carries_usage(endpoint, &terminal), include_usage);
                assert_eq!(
                    build_usage_only_chunk_event(endpoint, "cmpl-1", 1700, "m", &terminal, &[])
                        .is_some(),
                    include_usage,
                    "the wait must be paid exactly where a usage surface exists"
                );
            }
        }
        // Native generate carries usage inline on its terminal chunk.
        assert!(terminal_carries_usage(SseEndpoint::Generate, &terminal));
        assert!(build_generate_chunk_event(&terminal, &[])
            .get("usage")
            .is_some());

        // A terminal whose worker reported NO counts emits no usage block on
        // any surface, so nothing may wait for members it cannot publish.
        let countless = _terminal_chunk("error", None);
        for endpoint in [
            SseEndpoint::Generate,
            SseEndpoint::Chat {
                include_usage: true,
            },
            SseEndpoint::Completion {
                include_usage: true,
            },
        ] {
            assert!(
                !terminal_carries_usage(endpoint, &countless),
                "a countless terminal has no usage surface to carry extras"
            );
        }
        assert!(build_generate_chunk_event(&countless, &[])
            .get("usage")
            .is_none());
        assert!(build_usage_only_chunk_event(
            SseEndpoint::Chat {
                include_usage: true
            },
            "cmpl-1",
            1700,
            "m",
            &countless,
            &[],
        )
        .is_none());
    }

    /// A worker-emitted error chunk lands in the SSE stream as a
    /// final event carrying both the normal envelope and an
    /// ``error`` block.
    #[tokio::test]
    async fn test_sse_response_emits_error_chunk_on_worker_error() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut collector =
            crate::queue::streaming::StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let mut tap = collector.install_chunk_tap();
        let mut err_chunk = _terminal_chunk("error", None);
        err_chunk.error = Some(ChunkError {
            code: "rate_limit_exceeded".to_string(),
            message: "saturated".to_string(),
            param: None,
            retry_after_s: None,
        });
        collector.apply(err_chunk);
        let got = tap.recv().await.unwrap();
        assert!(got.done);
        let v = build_chat_chunk_event("chatcmpl-1", 0, "m", &got, true);
        assert_eq!(v["error"]["code"], "rate_limit_exceeded");
        assert_eq!(v["error"]["type"], "rate_limit_error");
        // The SIE-native generate shape would carry it too.
        let g = build_generate_chunk_event(&got, &[]);
        assert_eq!(g["error"]["code"], "rate_limit_exceeded");
        assert_eq!(g["done"], true);
    }

    #[tokio::test]
    async fn text_completion_preserves_pre_first_and_mid_stream_worker_errors() {
        for preceding_delta in [false, true] {
            let (tx, _rx) = tokio::sync::oneshot::channel();
            let mut collector =
                crate::queue::streaming::StreamCollector::new(tx, "m".to_string(), "p".to_string());
            let mut tap = collector.install_chunk_tap();
            if preceding_delta {
                collector.apply(_delta_chunk(0, "partial"));
                let delta = tap.recv().await.unwrap();
                let value = build_text_completion_chunk_event("cmpl-1", 0, "m", &delta);
                assert_eq!(value["choices"][0]["text"], "partial");
                assert!(value.get("error").is_none());
            }

            let mut error = _terminal_chunk("error", None);
            error.seq = if preceding_delta { 1 } else { 0 };
            error.error = Some(ChunkError {
                code: "unsupported_field".to_string(),
                message: "top_k is unavailable".to_string(),
                param: Some("top_k".to_string()),
                retry_after_s: None,
            });
            collector.apply(error);

            let terminal = tap.recv().await.unwrap();
            let value = build_text_completion_chunk_event("cmpl-1", 0, "m", &terminal);
            assert_eq!(value["error"]["code"], "unsupported_field");
            assert_eq!(value["error"]["param"], "top_k");
            assert!(value["choices"][0]["finish_reason"].is_null());
            assert_eq!(value["request_id"], "req-test");
        }
    }

    /// `/v1/generate/{model}` (SIE-native) uses the simpler shape —
    /// no `chat.completion.chunk` wrapper, no `delta` block.
    #[tokio::test]
    async fn test_sse_response_for_generate_endpoint_uses_native_shape() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut collector =
            crate::queue::streaming::StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let mut tap = collector.install_chunk_tap();
        collector.apply(_delta_chunk(0, "Tok-1"));
        collector.apply(_terminal_chunk(
            "stop",
            Some(UsageBlock {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            }),
        ));
        let c1 = tap.recv().await.unwrap();
        let v1 = build_generate_chunk_event(&c1, &[]);
        assert_eq!(v1["text_delta"], "Tok-1");
        assert_eq!(v1["done"], false);
        assert!(
            v1.get("choices").is_none(),
            "no OpenAI envelope on native shape"
        );

        let c2 = tap.recv().await.unwrap();
        let v2 = build_generate_chunk_event(&c2, &[]);
        assert_eq!(v2["done"], true);
        assert_eq!(v2["finish_reason"], "stop");
        assert_eq!(v2["usage"]["total_tokens"], 2);
    }

    /// A chunk carrying a ``tool_calls`` delta is forwarded with the
    /// OpenAI streaming shape — ``delta.tool_calls[*]`` carries the
    /// flat ``{index, id?, type, function: {name?, arguments}}`` tree,
    /// and ``delta.content`` is omitted (or absent) when the chunk
    /// carries only a tool call.
    #[test]
    fn test_sse_emits_tool_call_delta() {
        let chunk = ChunkEnvelope {
            kind: "chunk".to_string(),
            request_id: "req-tc".to_string(),
            attempt_id: "att-1".to_string(),
            seq: 1,
            text_delta: String::new(),
            done: false,
            is_first: false,
            finish_reason: None,
            usage: None,
            ttft_ms: None,
            error: None,
            tool_calls: Some(vec![ToolCallDeltaWire {
                index: 0,
                id: Some("call_abc".to_string()),
                kind: "function".to_string(),
                function: Some(ToolCallFunctionWire {
                    name: Some("get_weather".to_string()),
                    arguments: String::new(),
                }),
            }]),
            logprobs: None,
            candidates: Vec::new(),
            choice_index: 0,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let v = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, true);
        let delta = &v["choices"][0]["delta"];
        // Tool-call announcement: id + function.name set, arguments empty.
        let tcs = delta["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0]["index"], 0);
        assert_eq!(tcs[0]["id"], "call_abc");
        assert_eq!(tcs[0]["type"], "function");
        assert_eq!(tcs[0]["function"]["name"], "get_weather");
        assert_eq!(tcs[0]["function"]["arguments"], "");
        // Content is not surfaced when the chunk had no text.
        assert!(delta.get("content").is_none());
        // Non-terminal — finish_reason is null.
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    /// The terminal chunk after a tool-call run uses
    /// ``finish_reason: "tool_calls"`` rather than the default
    /// ``"stop"`` so the SDK routes to its function-calling branch.
    #[test]
    fn test_sse_terminal_finish_reason_tool_calls_passthrough() {
        let chunk = _terminal_chunk("tool_calls", None);
        let v = build_chat_chunk_event("chatcmpl-1", 0, "m", &chunk, false);
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    }

    /// Terminal chunks also reach the tap so the SSE handler can
    /// emit the final ``finish_reason`` + ``[DONE]`` events.
    #[tokio::test]
    async fn test_collector_tap_forwards_terminal_chunk() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let mut collector =
            crate::queue::streaming::StreamCollector::new(tx, "m".to_string(), "p".to_string());
        let mut tap = collector.install_chunk_tap();
        collector.apply(_delta_chunk(0, "Hi"));
        let _ = tap.recv().await.expect("delta");
        collector.apply(_terminal_chunk("stop", None));
        let got = tap.recv().await.expect("terminal");
        assert!(got.done);
        assert_eq!(got.finish_reason.as_deref(), Some("stop"));
    }

    // ── Helpers ────────────────────────────────────────────────────

    /// Extract the `data:` payload from an `axum::response::sse::Event`
    /// by routing it through axum's actual SSE body. Axum's `Event`
    /// type is opaque (no accessors for `data` or `finalize`), so the
    /// test round-trips a one-event response through the public
    /// `IntoResponse` impl and reads the wire bytes. This is the
    /// canonical way to assert SSE event content — it also exercises
    /// the very `Sse::new` -> `into_response` plumbing the production
    /// handler uses.
    ///
    /// Strips the leading ``data: `` and trailing ``\n\n`` so callers
    /// can assert on the JSON payload alone.
    async fn _event_data(ev: Event) -> String {
        use axum::body::to_bytes;
        let stream = futures_util::stream::once(async move { Ok::<_, Infallible>(ev) });
        let resp = Sse::new(stream).into_response();
        let bytes = to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("collect");
        let s = std::str::from_utf8(&bytes).expect("utf8");
        let s = s.strip_prefix("data: ").unwrap_or(s);
        let s = s.strip_suffix("\n\n").unwrap_or(s);
        s.to_string()
    }
}
