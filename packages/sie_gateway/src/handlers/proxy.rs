use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine;
use dashmap::DashMap;
use percent_encoding::{percent_decode_str, utf8_percent_encode, NON_ALPHANUMERIC};
use rmp_serde;
use serde_json::{json, Map, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn, Instrument};

use crate::endpoint::InferenceEndpoint;
use crate::http_error::{
    code as err_code, embeddings_error, json_detail, json_detail_merge, json_openai_error,
    openai_code as oai_code, openai_type as oai_type, GatewayOwnedFault,
};
use crate::observability::metrics as telemetry;
use crate::queue::dispatch::{
    DispatchDurability, DispatchError, PendingDispatchKind, WorkDispatcher, WorkDispatcherExt,
};
use crate::queue::publisher;
use crate::queue::streaming::{
    client_safe_worker_error_code, client_safe_worker_error_message,
    client_safe_worker_error_param, is_lower_sha256, MODEL_LOAD_FAILED_ERROR_CODE,
    MODEL_LOAD_FAILED_PUBLIC_MESSAGE,
};

use crate::server::{
    AppState, GenerationRequestIntent, GovernedGenerationRoute, ModelAccessPolicy,
};
use crate::state::demand_tracker::PhysicalLane;
use crate::state::model_registry::{ModelRegistry, ResolveError};
use crate::state::pool_manager::{normalize_pool_name, PoolManager, DEFAULT_POOL_NAME};
use crate::state::worker_registry::{QueueRoute, WorkerRegistry};
use crate::types::AuditEntry;

use crate::middleware::auth::{extract_bearer_token, mask_token};

const GATEWAY_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The ONE ingress bound every JSON body-bearing route accepts.
///
/// 16 MiB: enough headroom for multimodal content parts (base64 images) while
/// closing the trivial OOM-under-concurrency vector the legacy 256 MiB cap left
/// open. `/v1/encode`, `/v1/score`, `/v1/embeddings`, `/v1/chat/completions`,
/// `/v1/completions` and `/v1/responses` all derive from it; only the two
/// routes that carry inline native MEDIA (`generate`'s images, `extract`'s
/// audio) raise it, and they say why below.
///
/// `pub` and singular on purpose: a gate, a probe, or a second surface that
/// buffers a body on a route's behalf must pin to the route's own number.
/// Smaller and it denies legitimate traffic the route would have served; larger
/// and it is a way around the rail. Both mistakes have shipped here before.
pub const MAX_JSON_BODY_BYTES: usize = 16 * 1024 * 1024;

const MAX_PROXY_BODY: usize = MAX_JSON_BODY_BYTES;
// Native generate accepts bounded inline images. A 16 MiB decoded image grows
// to ~21.4 MiB in base64; leave room for the prompt and JSON envelope while
// still bounding aggregate request memory at ingress.
const MAX_GENERATE_BODY: usize = 24 * 1024 * 1024;
const MAX_GENERATE_IMAGES: usize = 16;
const MAX_GENERATE_IMAGE_BYTES: usize = 16 * 1024 * 1024;
const MAX_GENERATE_IMAGE_BASE64_CHARS: usize = 4 * MAX_GENERATE_IMAGE_BYTES.div_ceil(3);
// The audio preprocessor accepts 24 MiB of encoded media. Base64 expands that
// to exactly 32 MiB; leave bounded room for the surrounding native JSON item.
const MAX_EXTRACT_BODY: usize = 34 * 1024 * 1024;

/// The ingress body cap for ONE native queue endpoint (`encode` / `score` /
/// `extract` / `generate`).
///
/// Public so a surface that carries a native body without being the native
/// route — the managed cost-estimate dry run's `{endpoint, request}` envelope
/// (#2435) — bounds that body at the SAME number the route itself does. The
/// caps are abuse rails on planner/ingress work; a second surface that accepts
/// a larger body than the route it prices is a way around the rail.
pub fn native_request_body_limit(endpoint: &str) -> usize {
    match endpoint {
        "generate" => MAX_GENERATE_BODY,
        "extract" => MAX_EXTRACT_BODY,
        _ => MAX_PROXY_BODY,
    }
}

/// The ingress body cap for ONE OpenAI-compatible route.
///
/// Public for the same reason as [`native_request_body_limit`], and separate
/// from it because the compat generation routes do NOT take the 24 MiB native
/// `generate` cap: their bodies carry no inline native media array, so they sit
/// on the shared [`MAX_JSON_BODY_BYTES`] bound. An estimate that bounded a chat
/// body at the native number would accept 1.5x what the route it prices
/// accepts — the exact way around the rail this helper exists to prevent
/// (#2435).
pub fn compat_request_body_limit(_path: &str) -> usize {
    MAX_JSON_BODY_BYTES
}

/// The bound on an inner NATIVE response an OpenAI-compatible wrapper must
/// buffer whole before it can reshape it.
///
/// # Why it is separate from the request cap
///
/// This bound used to be `compat_request_body_limit("/v1/embeddings")`. Reusing
/// a REQUEST cap for a RESPONSE is wrong regardless of the number: `/v1/embeddings`
/// is the one compat surface whose reply is an *amplification* of its request —
/// N short texts in, N × `dim` floats out — so the two quantities are unrelated,
/// and a future change to the ingress cap would silently move this bound.
/// Owning the constant separately is the actual fix; the value below is a second,
/// independent decision.
///
/// # Why the value is 16 MiB, and why it is CHOSEN rather than derived
///
/// An earlier revision of this constant set 64 MiB and claimed it was "anchored"
/// to `queue::publisher::MAX_RESULT_CHUNK_RESERVED_BYTES_PER_REQUEST`. That was
/// wrong twice over, and the corrected reasoning is recorded here so it is not
/// re-derived the same way:
///
/// 1. That constant bounds a RESERVATION, not a payload. `result_chunk_reservation_bytes`
///    computes `payload × 3 + 4 KiB` and checks the PRODUCT against the 64 MiB
///    limit, so the payload it authorises is ~21.3 MiB, not 64 MiB.
/// 2. The rail is not even on this path. Embedding item results are ~4 KiB, far
///    under `PAYLOAD_OFFLOAD_THRESHOLD` (1 MiB), so they never chunk. The
///    per-request number is also only safe because of a process-wide budget that
///    has no analogue here — nothing counts the compat wrapper's allocation.
///
/// There is no honest derivation available, because **no bound both admits every
/// reply this surface can produce AND fits the container.** At the queue maximum
/// (`MAX_QUEUE_REQUEST_ITEMS` = 4096) a float reply is ~90 MB at 1024 dims and
/// ~224 MB at 2560 dims (`Qwen/Qwen3-Embedding-4B`, shipped on this very surface
/// in `beta-launch-v1.yaml`). The handler holds roughly four copies at once (the
/// buffer, the parsed tree, the projected `data`, and the re-serialised body).
/// Four ~224 MB copies total ~896 MB, or about 1 GB for one request; concurrent
/// requests can therefore drive the process into several gigabytes. So this is
/// a CHOSEN bound with a stated rationale, not a derived one.
///
/// The choice is to hold today's effective bound exactly. 16 MiB is what the
/// surface enforces in production right now, so it is provably non-breaking in
/// BOTH directions: it refuses nothing that is served today, and it raises
/// per-request heap by nothing. Raising it would trade a clean, diagnosable 413
/// for an OOM kill that can also destroy unrelated concurrent requests —
/// strictly worse than the refusal it removes.
///
/// # What this deliberately does NOT fix
///
/// The 256-item ingress cap on `/v1/embeddings` prevents known-undeliverable
/// large batches from reaching this post-dispatch rail. This cap remains the
/// fail-closed backstop for output amplification, unexpected model dimensions,
/// and any future response-shape growth.
const MAX_COMPAT_RESPONSE_BODY: usize = 16 * 1024 * 1024;

/// Outcome of buffering an inner native response under
/// [`MAX_COMPAT_RESPONSE_BODY`].
///
/// The two failure arms are deliberately distinct because they earn different
/// caller-facing verdicts — but note that both are GATEWAY-OWNED faults for
/// billing: the worker has already run and reported its units by the time this
/// runs, and the customer receives no embeddings either way.
#[derive(Debug)]
pub(crate) enum CompatResponseBody {
    Buffered(axum::body::Bytes),
    /// The reply exceeded [`MAX_COMPAT_RESPONSE_BODY`]. Carries the observed
    /// length when the body announced one, so the caller can scale their batch
    /// by a known factor instead of guessing.
    TooLarge {
        observed: Option<u64>,
    },
    /// The body could not be read at all — a genuine stream failure, NOT a size
    /// condition.
    Unreadable,
}

/// Buffer an inner native response, classifying over-cap separately from
/// unreadable.
///
/// # Why this is not simply `to_bytes(body, CAP + 1)` and a length comparison
///
/// That was the first attempt and it was inoperative (#2617). `axum::body::to_bytes`
/// IS `Limited::new(body, limit).collect()`, and `Limited` yields
/// `LengthLimitError` the moment a frame would push the running total past the
/// limit — it does not collect the body and then check. So `Ok` can never carry
/// a length ABOVE the limit, and a `len() > CAP` guard over a `CAP + 1` limit
/// matches at exactly ONE size: `CAP + 1`. Every genuinely oversized reply took
/// the error arm and returned the opaque 500 the change existed to replace.
/// `compat_response_cap_classifies_sizes_past_the_boundary` pins `CAP + 2` and a
/// 4× oversize precisely because the old shape passed a `CAP`/`CAP + 1` test
/// while the branch was dead.
///
/// Two stages, so every body shape lands correctly:
///
/// 1. **Size hint first.** An in-process reply (which is what `proxy_request`
///    returns — a complete `Vec<u8>`) announces its exact length, so an oversized
///    body is refused WITHOUT allocating it. That is the whole point of checking
///    first: buffering 90 MB in order to discard it is the memory spike this cap
///    exists to prevent.
/// 2. **Downcast the limit error.** For a body that announces nothing, buffer to
///    the cap and recover the distinction from the error's source chain. This is
///    why `http-body-util` is a direct dependency.
pub(crate) async fn buffer_compat_response_body(body: Body) -> CompatResponseBody {
    use hyper::body::Body as _;

    // `lower()` is a guaranteed minimum, so exceeding the cap here is proof, not
    // an estimate. Unknown-length bodies report 0 and fall through to stage 2.
    let announced = body.size_hint().lower();
    if announced > MAX_COMPAT_RESPONSE_BODY as u64 {
        return CompatResponseBody::TooLarge {
            observed: Some(announced),
        };
    }
    match to_bytes(body, MAX_COMPAT_RESPONSE_BODY).await {
        Ok(bytes) => CompatResponseBody::Buffered(bytes),
        Err(error) if is_length_limit_error(&error) => {
            CompatResponseBody::TooLarge { observed: None }
        }
        Err(_) => CompatResponseBody::Unreadable,
    }
}

/// A terminal `/v1/embeddings` error for a POST-DISPATCH failure, marked as
/// gateway-owned so a metered composition releases the hold.
///
/// The worker already ran and the meter already holds its units by the time any
/// caller of this runs (the dispatcher records authoritative units *before*
/// fulfilling the result channel). Without the marker, the managed edge falls
/// through to `settle_planned_request`, which bills every recorded dispatch
/// regardless of the response status — so the customer pays in full for a
/// response carrying zero embeddings. That is the defect class of #2641, and it
/// is what [`GatewayOwnedFault`]'s own doc names: "a response body larger than
/// the read bound".
///
/// The status is the CALLER's verdict (retry or don't); the marker is the
/// BILLING verdict (charge or don't). A terminal 4xx and a gateway-owned fault
/// are not in tension: the caller must change their request, and they still owe
/// nothing for the batch that produced no output.
fn compat_translation_fault(
    stage: &'static str,
    status: StatusCode,
    code: &str,
    param: Option<&str>,
    message: String,
) -> Response {
    GatewayOwnedFault::new(stage)
        .mark((status, Json(embeddings_error(code, param, message))).into_response())
}

/// Is this `axum::Error` the length-limit condition rather than a broken stream?
///
/// `to_bytes` boxes `LengthLimitError` behind `axum::Error`, so the discriminator
/// is only reachable by walking the source chain. Matching on the Display string
/// would work today and rot silently on any upstream wording change; the downcast
/// cannot.
///
/// `pub` so the managed composition's own `to_bytes` call sites reuse this exact
/// discriminator rather than restating the downcast — the same reason the size
/// bounds themselves are imported rather than copied.
pub fn is_length_limit_error(error: &axum::Error) -> bool {
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(current) = source {
        if current
            .downcast_ref::<http_body_util::LengthLimitError>()
            .is_some()
        {
            return true;
        }
        source = std::error::Error::source(current);
    }
    false
}

/// The largest [`native_request_body_limit`] across every native endpoint —
/// the bound an envelope that can carry ANY of them must buffer to.
///
/// `dead_code`-allowed because the `sie-gateway` BINARY compiles this module
/// tree independently of the library (see the note in `lib.rs`), and only the
/// downstream composition crate reads this constant.
#[allow(dead_code)]
pub const MAX_NATIVE_REQUEST_BODY: usize = {
    let generate_or_proxy = if MAX_GENERATE_BODY > MAX_PROXY_BODY {
        MAX_GENERATE_BODY
    } else {
        MAX_PROXY_BODY
    };
    if MAX_EXTRACT_BODY > generate_or_proxy {
        MAX_EXTRACT_BODY
    } else {
        generate_or_proxy
    }
};
static GATEWAY_VERSION_MINOR: std::sync::LazyLock<u32> =
    std::sync::LazyLock::new(|| env!("CARGO_PKG_VERSION_MINOR").parse().unwrap_or(0));

/// Compute a stable, non-secret OpenAI `system_fingerprint` for a response.
///
/// OpenAI's contract is that `system_fingerprint` changes when the backend
/// config changes, so a client can detect that the determinism guarantee may
/// have shifted. This gateway-side derivation keys it on the served `model`
/// plus the gateway build version: stable for a fixed deployment, and it
/// changes when the model or the gateway (deployed in lockstep with the
/// worker engine bundle) changes. Per-request sampler params (`seed`,
/// `temperature`) are deliberately excluded — those are the caller's input,
/// not the backend's config.
///
/// Follow-up (worker-side, see `t3-determinism-seed-system-fingerprint`): fold
/// the worker's pinned engine versions + runtime config (carried on the result
/// envelope) into this hash for a finer-grained fingerprint. Replacing the
/// previous always-`null` value is backward compatible (a non-null string is
/// equally valid per OpenAI's nullable schema).
pub(crate) fn system_fingerprint(model: &str) -> String {
    let mut input = String::with_capacity(model.len() + 1 + GATEWAY_VERSION.len());
    input.push_str(model);
    input.push('\u{0}');
    input.push_str(GATEWAY_VERSION);
    format!("fp_{:016x}", xxhash_rust::xxh3::xxh3_64(input.as_bytes()))
}

/// The `Retry-After` hints (seconds, as the wire strings stamped directly onto
/// retryable capacity responses) the gateway returns for each cold-load /
/// provisioning condition. One typed home for what were six loose constants,
/// and the seam a future PR populates from config to make these an operator
/// dial without a rebuild (see #1574). The named `*_RETRY_AFTER` constants
/// below are thin aliases over [`RetryAfter::DEFAULT`], so existing call sites
/// are unchanged.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RetryAfter {
    pub provisioning: &'static str,
    pub backpressure: &'static str,
    pub gateway_timeout: &'static str,
    pub model_loading: &'static str,
    pub resource_exhausted: &'static str,
    pub lora_loading: &'static str,
}

impl RetryAfter {
    /// Compile-time defaults. `provisioning` is the longer pre-execution
    /// capacity-miss hint (kept within the range common OpenAI SDKs honor);
    /// the rest are short transient-retry hints.
    pub const DEFAULT: RetryAfter = RetryAfter {
        provisioning: "60",
        backpressure: "5",
        gateway_timeout: "5",
        model_loading: "5",
        resource_exhausted: "5",
        lora_loading: "5",
    };
}

/// Provisioning is a retryable pre-execution capacity miss, not an accepted
/// asynchronous job. Keep the Retry-After hint within the range common OpenAI
/// SDKs honor directly.
pub(crate) const PROVISIONING_RETRY_AFTER: &str = RetryAfter::DEFAULT.provisioning;
const BACKPRESSURE_RETRY_AFTER: &str = RetryAfter::DEFAULT.backpressure;
const GATEWAY_TIMEOUT_RETRY_AFTER: &str = RetryAfter::DEFAULT.gateway_timeout;
const MODEL_LOADING_RETRY_AFTER: &str = RetryAfter::DEFAULT.model_loading;
const MODEL_LOADING_ERROR_CODE: &str = "MODEL_LOADING";
/// Sealed cold-start wake quota refused (#2426): the org exceeded its
/// cold-starts-per-hour ceiling, so the dispatcher declined to buy a wake.
/// A deliberate quota refusal, surfaced as 429 — terminal, and deliberately
/// absent from the retryable lists: an hourly quota must not invite SDK
/// auto-retry storms the way MODEL_LOADING's short Retry-After does.
const COLD_START_RATE_LIMITED_ERROR_CODE: &str = "COLD_START_RATE_LIMITED";
/// Server-side OOM recovery exhausted. Workers stamp this on
/// ``WorkResult.error_code`` when the per-batch ``cache_clear → evict_lru
/// → split_batch`` strategy still runs out of GPU memory; the gateway
/// translates it into HTTP 503 + ``Retry-After`` so the SDK auto-retries
/// with bounded exponential backoff. The worker is **not** marked
/// unhealthy — it lost an allocation race, it isn't broken.
const RESOURCE_EXHAUSTED_ERROR_CODE: &str = "RESOURCE_EXHAUSTED";
const RESOURCE_EXHAUSTED_RETRY_AFTER: &str = RetryAfter::DEFAULT.resource_exhausted;
/// Worker is loading a LoRA adapter on demand. The SDK retries this with
/// the same ``provision_timeout_s`` budget it uses for ``MODEL_LOADING``;
/// see ``sie_sdk.client._shared.LORA_LOADING_*``.
const LORA_LOADING_ERROR_CODE: &str = "LORA_LOADING";
const LORA_LOADING_RETRY_AFTER: &str = RetryAfter::DEFAULT.lora_loading;
const INVALID_INPUT_ERROR_CODE: &str = "INVALID_INPUT";
const PAYLOAD_TOO_LARGE_ERROR_CODE: &str = err_code::PAYLOAD_TOO_LARGE;

/// Fallback `max_tokens` applied to a chat-completions request that
/// omits both `max_completion_tokens` and `max_tokens`.
///
/// OpenAI treats `max_tokens` as optional (the server picks a default),
/// so an OpenAI-compatible surface must not 400 when it is absent —
/// generic clients (Open WebUI, the `openai` SDK with no explicit cap)
/// routinely omit it. We default rather than reject. Operators can
/// override the value via `SIE_GATEWAY_DEFAULT_MAX_TOKENS`; see
/// [`default_max_tokens`].
const DEFAULT_MAX_TOKENS: u64 = 1024;

/// Resolve the chat-completions `max_tokens` default.
///
/// Reads `SIE_GATEWAY_DEFAULT_MAX_TOKENS` and falls back to
/// [`DEFAULT_MAX_TOKENS`] when the env var is unset or unparseable.
/// Centralised so the handler and its tests agree on one source of
/// truth (the test asserts the no-token path resolves to this value
/// rather than hard-coding the literal).
fn default_max_tokens() -> u64 {
    std::env::var("SIE_GATEWAY_DEFAULT_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

/// Track which SDK minor versions we've already warned about (to warn once per minor).
/// ``DashSet`` rather than ``Mutex<HashSet>`` so the version-skew path
/// doesn't acquire a Mutex on every request — mirrors the lock-free
/// pattern used by ``SDK_VERSION_CACHE`` immediately below.
static SDK_WARNED_MINORS: std::sync::LazyLock<dashmap::DashSet<u32>> =
    std::sync::LazyLock::new(dashmap::DashSet::new);
/// Hard cap on `SDK_WARNED_MINORS` so a caller walking unique minor
/// versions in `X-SIE-SDK-Version` cannot grow the set without bound.
/// Real deployments see a handful of client minor versions; 1024
/// matches `SDK_VERSION_CACHE_CAP`.
const SDK_WARNED_MINORS_CAP: usize = 1024;

/// Per-unique-SDK-version cache of the parsed minor-version number.
///
/// Every inference request runs through [`check_sdk_version`] which
/// previously allocated a fresh `Vec<&str>` via `split('.')` and ran
/// a `u32::parse` every call. A gateway sees at most a handful of
/// unique SDK version strings over its lifetime (one per client
/// release), so a DashMap keyed by the raw header value folds the
/// parse work down to a single lookup on the hot path. `Option<u32>`
/// caches "unparseable" so malformed headers don't re-parse either.
///
/// **Hard size cap.** `X-SIE-SDK-Version` is caller-supplied, so a
/// buggy or hostile client could otherwise walk the key space with
/// unique strings on every request. Once `SDK_VERSION_CACHE_CAP`
/// entries are populated we stop memoising and fall back to
/// parse-on-every-request (which is what `main` did before this
/// optimisation existed, so the worst case is still bounded).
/// 1024 entries is well above the real client-release count and
/// well below any memory worry — `Arc<str>` + `Option<u32>` costs
/// ~32 B per entry plus the version string itself.
static SDK_VERSION_CACHE: std::sync::LazyLock<DashMap<Arc<str>, Option<u32>>> =
    std::sync::LazyLock::new(DashMap::new);
const SDK_VERSION_CACHE_CAP: usize = 1024;

#[allow(dead_code)]
static HOP_BY_HOP_HEADERS: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
];

#[allow(dead_code)]
fn is_hop_by_hop(name: &str) -> bool {
    HOP_BY_HOP_HEADERS
        .iter()
        .any(|h| h.eq_ignore_ascii_case(name))
}

#[derive(Debug, PartialEq, Eq)]
enum EnginePinParse {
    None,
    Some(String),
    Unknown(String),
    InvalidUtf8,
}

fn parse_engine_pin(headers: &HeaderMap) -> EnginePinParse {
    let Some(raw) = headers.get("x-sie-engine") else {
        return EnginePinParse::None;
    };
    let s = match raw.to_str() {
        Ok(s) => s,
        Err(_) => return EnginePinParse::InvalidUtf8,
    };
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return EnginePinParse::None;
    }
    use crate::types::bundle::KNOWN_ENGINES;
    let normalised = trimmed.to_ascii_lowercase();
    if KNOWN_ENGINES.contains(&normalised.as_str()) {
        EnginePinParse::Some(normalised)
    } else {
        EnginePinParse::Unknown(trimmed.to_string())
    }
}

/// Outcome of resolving the JetStream pool to publish work for a request.
#[derive(Debug, PartialEq, Eq)]
enum PoolResolution {
    /// A healthy worker is registered and this is the pool to publish to.
    Route(QueueRoute),
    /// Caller pinned a logical pool that is not present in the pool manager.
    PoolNotFound(String),
    /// No healthy worker matches the requested `(pool, machine_profile,
    /// bundle)` lane — the caller should emit the surface-specific
    /// provisioning response and record pending demand so KEDA scales up.
    Provisioning,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProvisioningSurface {
    Native,
    OpenAiCompat,
}

/// Result of `resolve_effective_pool` — bundles the routing decision
/// with a flag telling the caller whether the registry had an exact
/// `(bundle, machine_profile)` worker match.
///
/// The gateway records pending demand (for KEDA auto-scale) whenever
/// the caller expressed a machine-profile preference but the exact tuple has no
/// healthy worker. By reporting `exact_gpu_match` here we can fold
/// that probe into the same registry load the routing decision already
/// does, instead of doing a separate route lookup on the hot path.
#[derive(Debug, PartialEq, Eq)]
struct PoolLookup {
    resolution: PoolResolution,
    /// Logical pool whose spec/status should be used for admission decisions.
    admission_pool: String,
    /// Physical queue pool label for pending-demand metrics/KEDA.
    demand_pool: String,
    /// `true` iff a healthy worker with a non-empty pool name and
    /// machine-profile existed for the exact `(bundle, gpu)` tuple at
    /// lookup time.
    ///
    /// Always `false` when `gpu.is_empty()` (no exact tuple to match).
    /// Caller-pinned pools still probe the registry when a GPU is
    /// present so demand tracking can avoid spurious scale-up signals.
    exact_gpu_match: bool,
    /// Concrete machine-profile labels that should receive pending demand
    /// even when the caller did not send `X-SIE-MACHINE-PROFILE`.
    ///
    /// KEDA queries are exact `pool/profile/bundle` matches, so recording
    /// `machine_profile=""` cannot scale a specific lane. When the caller
    /// pins a GPU we record that one profile; when a cold gpu-agnostic
    /// request cannot name one, we fan demand across every profile the pool
    /// can provision so each candidate cold lane can scale from zero (a
    /// capable one comes up and serves; the others reap on scale-down).
    /// Empty when a healthy worker already exists or no profile is known.
    pending_demand_profiles: Vec<String>,
}

/// Strict allowlist for caller-supplied pool names (`[A-Za-z0-9_-]`).
///
/// The pool flows verbatim into the JetStream work subject
/// `sie.work.{pool}.{machine_profile}.{bundle}.{model}` (see
/// `queue::publisher::work_subject`).
/// `model` / `worker_id` are scrubbed via `normalize_model_id`, but the
/// pool was not — a pool containing `.`, `*`, `>`, or whitespace would
/// produce an illegal / re-tokenised subject (subject injection). Unlike
/// the model scrub (which degrades wonky chars to deterministic
/// underscores), silently mangling the pool would mis-route the request
/// to a *different* pool's subject, so we REJECT (OpenAI-shaped 400)
/// rather than mangle. Length is bounded so an absurd pool can't bloat
/// the subject or a downstream metric label.
fn is_valid_pool_name(pool: &str) -> bool {
    !pool.is_empty()
        && pool.len() <= 128
        && !pool.eq_ignore_ascii_case("_default")
        && pool
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
}

async fn apply_model_pool_default(
    requested_pool: &mut String,
    model_pool: Option<&str>,
    pool_manager: Option<&PoolManager>,
) -> Result<(), String> {
    let model_pool = model_pool.and_then(|pool| {
        let pool = pool.trim();
        if pool.is_empty() || pool.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
            None
        } else {
            Some(normalize_pool_name(pool))
        }
    });
    let Some(model_pool) = model_pool else {
        if requested_pool.is_empty() {
            *requested_pool = DEFAULT_POOL_NAME.to_string();
        }
        return Ok(());
    };
    if requested_pool.is_empty() {
        *requested_pool = model_pool.clone();
        return Ok(());
    }
    if *requested_pool == model_pool {
        return Ok(());
    }
    if let Some(manager) = pool_manager {
        if manager
            .queue_pool_for_pool(requested_pool)
            .await
            .is_some_and(|queue_pool| queue_pool == model_pool)
        {
            return Ok(());
        }
    }
    Err(format!(
        "Model is assigned to pool '{model_pool}', but request targeted pool '{requested_pool}'"
    ))
}

/// Pure decision logic for the scale-from-zero branch of `proxy_request`.
/// Kept as a free function so it can be unit-tested without standing up
/// an `AppState` / `WorkPublisher`.
///
/// Rules:
/// - If the caller pinned a pool via `X-SIE-Pool` and supplied a GPU,
///   route only when a healthy worker exists in that exact lane. Otherwise
///   return `Provisioning` and record pending demand for the pinned lane. If
///   they pinned only a pool, use the pool's provisionable profiles for
///   pending-demand fan-out, probing a concrete cold lane only when the pool
///   has exactly one profile.
/// - Otherwise look up a healthy default-pool worker for `(bundle, gpu)`.
///   The worker must also report the gateway's expected bundle config hash.
///   If GPU was specified and the exact tuple has no worker, return
///   provisioning for that cold lane; the gateway must not silently route to
///   another machine profile.
/// - If nothing resolves, return `Provisioning` so the caller can emit
///   provisioning response — regardless of whether the caller sent
///   `X-SIE-MACHINE-PROFILE`. Before the fix this branch only fired when
///   `gpu` was non-empty, which turned a normal cold start into a queue
///   timeout for default-routing clients.
async fn resolve_effective_pool(
    registry: &WorkerRegistry,
    pool_manager: Option<&PoolManager>,
    bundle: &str,
    gpu: &str,
    pool_name: &str,
    bundle_config_hash: &str,
) -> PoolLookup {
    if !pool_name.is_empty() {
        let normalized_pool = normalize_pool_name(pool_name);
        let Some(queue_pool) = queue_pool_for_request(pool_manager, pool_name).await else {
            return PoolLookup {
                resolution: PoolResolution::PoolNotFound(normalized_pool.clone()),
                admission_pool: normalized_pool,
                demand_pool: DEFAULT_POOL_NAME.to_string(),
                exact_gpu_match: false,
                pending_demand_profiles: Vec::new(),
            };
        };
        // Caller-pinned cold demand may only name profiles from the resolved
        // pool catalog. A syntactically valid caller header is not evidence
        // that a KEDA target exists.
        if gpu.is_empty() {
            let profiles = demand_profiles_for_pool(pool_manager, pool_name).await;
            // Probe the exact cold lane when the pool has a single profile;
            // otherwise probe profile-agnostically and let demand fan out.
            let lookup_gpu = if profiles.len() == 1 {
                profiles[0].as_str()
            } else {
                ""
            };
            let route = registry
                .resolve_queue_route_in_pool(bundle, lookup_gpu, &queue_pool, bundle_config_hash)
                .await;
            let pending_demand_profiles = if route.is_none() {
                profiles
            } else {
                Vec::new()
            };
            return PoolLookup {
                resolution: match route {
                    Some(route) => PoolResolution::Route(route),
                    None => PoolResolution::Provisioning,
                },
                admission_pool: normalize_pool_name(pool_name),
                demand_pool: queue_pool,
                exact_gpu_match: false,
                pending_demand_profiles,
            };
        }

        let profiles = demand_profiles_for_pool(pool_manager, pool_name).await;
        let configured_profile = profiles
            .into_iter()
            .find(|profile| profile.eq_ignore_ascii_case(gpu));
        let route = registry
            .resolve_queue_route_in_pool(bundle, gpu, &queue_pool, bundle_config_hash)
            .await;
        let exact_gpu_match = route.is_some();
        return PoolLookup {
            resolution: match route {
                Some(route) => PoolResolution::Route(route),
                None => PoolResolution::Provisioning,
            },
            admission_pool: normalize_pool_name(pool_name),
            demand_pool: queue_pool,
            exact_gpu_match,
            pending_demand_profiles: if exact_gpu_match {
                Vec::new()
            } else {
                configured_profile.into_iter().collect()
            },
        };
    }

    // Primary lookup. Folds the "was the exact tuple routable?"
    // question into the same registry load we use to pick a pool.
    let primary = registry
        .resolve_queue_route_in_pool(bundle, gpu, DEFAULT_POOL_NAME, bundle_config_hash)
        .await;
    let exact_gpu_match = !gpu.is_empty() && primary.is_some();

    let resolution = match primary {
        Some(route) => PoolResolution::Route(route),
        None => PoolResolution::Provisioning,
    };
    let pending_demand_profiles = if !gpu.is_empty() && !exact_gpu_match {
        demand_profiles_for_pool(pool_manager, DEFAULT_POOL_NAME)
            .await
            .into_iter()
            .find(|profile| profile.eq_ignore_ascii_case(gpu))
            .into_iter()
            .collect()
    } else if matches!(resolution, PoolResolution::Provisioning) {
        demand_profiles_for_pool(pool_manager, DEFAULT_POOL_NAME).await
    } else {
        Vec::new()
    };
    PoolLookup {
        resolution,
        admission_pool: DEFAULT_POOL_NAME.to_string(),
        demand_pool: DEFAULT_POOL_NAME.to_string(),
        exact_gpu_match,
        pending_demand_profiles,
    }
}

async fn queue_pool_for_request(
    pool_manager: Option<&PoolManager>,
    pool_name: &str,
) -> Option<String> {
    let normalized = normalize_pool_name(pool_name);
    if normalized.is_empty() {
        return Some(DEFAULT_POOL_NAME.to_string());
    }
    let Some(manager) = pool_manager else {
        return Some(normalized);
    };
    manager.queue_pool_for_pool(&normalized).await
}

fn catalog_execution_hash_is_ready(bundle_config_hash: &str, uses_catalog_scope: bool) -> bool {
    !uses_catalog_scope || !bundle_config_hash.is_empty()
}

async fn demand_profiles_for_pool(
    pool_manager: Option<&PoolManager>,
    pool_name: &str,
) -> Vec<String> {
    match pool_manager {
        Some(manager) => manager.demand_profiles_for_pool(pool_name).await,
        None => Vec::new(),
    }
}

fn provisioning_message(gpu: &str, bundle: &str) -> String {
    if gpu.is_empty() {
        format!(
            "No worker available for bundle '{}'. Provisioning in progress.",
            bundle
        )
    } else {
        format!(
            "No worker available for GPU type '{}'. Provisioning in progress.",
            gpu
        )
    }
}

fn add_provisioning_headers(resp: &mut Response, retry_after: &'static str) {
    resp.headers_mut().insert(
        HeaderName::from_static("retry-after"),
        HeaderValue::from_static(retry_after),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
}

fn record_provisioning_response(surface: ProvisioningSurface, status: StatusCode) {
    let (surface_label, telemetry_surface) = match surface {
        ProvisioningSurface::Native => ("native", telemetry::ProvisioningSurface::Native),
        ProvisioningSurface::OpenAiCompat => ("openai", telemetry::ProvisioningSurface::OpenAi),
    };
    info!(
        surface = surface_label,
        http_status = status.as_u16(),
        "no queue worker available, returning provisioning response",
    );
    telemetry::record_provisioning_response(telemetry_surface, status.as_u16());
}

/// Record metrics for a failed publish attempt and, for backpressure, also
/// record pending demand so KEDA scales the saturated lane.
///
/// `backpressure` means a worker exists but its lane is saturated, so the
/// lane is exactly the one to scale up. Unlike the no-worker path (which
/// records demand upstream in `resolve_routing`), nothing else signals
/// demand for an already-provisioned-but-saturated lane — without this a
/// backpressure spike returns 503 but never triggers a scale-up, so KEDA
/// stays flat while the pool stays hot. See #1568.
///
/// Returns the `Retry-After` value the caller should surface (`None` for a
/// generic publish failure). The `no consumers` case is handled by the
/// caller — it needs a provisioning response / surface-specific retry.
///
/// A broker-side rejection at the stream's `max_messages` cap counts as
/// backpressure too. With `DiscardPolicy::New` (see `ensure_stream`) a full
/// work stream refuses the publish with JetStream's "maximum messages
/// exceeded" instead of silently dropping the oldest queued item, and that
/// rejection means the same thing as the gateway-local gate: the lane is
/// saturated and is exactly the one to scale up.
pub(crate) fn record_publish_failure(
    state: &AppState,
    physical_lane: &PhysicalLane,
    err_lower: &str,
) -> Option<&'static str> {
    if err_lower.contains("backpressure") || err_lower.contains("maximum messages exceeded") {
        telemetry::record_rejected_request(
            state.demand_tracker.as_ref(),
            physical_lane,
            "backpressure",
        );
        state.demand_tracker.record(physical_lane);
        Some(BACKPRESSURE_RETRY_AFTER)
    } else {
        None
    }
}

const DISPATCH_FAILURE_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

struct DispatchHandoffGuard {
    demand_tracker: Arc<crate::state::demand_tracker::DemandTracker>,
    physical_lane: PhysicalLane,
    handoff: crate::state::demand_tracker::DispatchHandoff,
    finished: bool,
}

impl DispatchHandoffGuard {
    fn new(
        demand_tracker: Arc<crate::state::demand_tracker::DemandTracker>,
        physical_lane: PhysicalLane,
    ) -> Option<Self> {
        demand_tracker
            .begin_dispatch_handoff(&physical_lane)
            .map(|handoff| Self {
                demand_tracker,
                physical_lane,
                handoff,
                finished: false,
            })
    }

    fn finish(mut self, durable: bool) {
        self.demand_tracker
            .finish_dispatch_handoff(&self.physical_lane, self.handoff, durable);
        self.finished = true;
    }
}

impl Drop for DispatchHandoffGuard {
    fn drop(&mut self) {
        if !self.finished {
            self.demand_tracker
                .finish_dispatch_handoff(&self.physical_lane, self.handoff, false);
        }
    }
}

/// Bridge one submitted request from transient exact-lane demand to the
/// transport's durable backlog without adding broker RTT to successful HTTP
/// responses.
///
/// Exactly one bounded task owns the ACK future(s). It records request-scoped
/// KEDA demand before it starts, clears only its own lease on success, retains
/// a 120-second failure marker on error/abort, and notifies the request driver
/// so a late rejection becomes a prompt typed transport failure instead of a
/// full result timeout. Failure cleanup is best-effort and separately bounded.
pub(crate) fn monitor_dispatch_durability(
    demand_tracker: Arc<crate::state::demand_tracker::DemandTracker>,
    physical_lane: PhysicalLane,
    durability: DispatchDurability,
    work_publisher: Arc<dyn WorkDispatcher>,
    request_id: String,
    kind: PendingDispatchKind,
) -> tokio::sync::oneshot::Receiver<Result<(), String>> {
    let handoff = DispatchHandoffGuard::new(Arc::clone(&demand_tracker), physical_lane.clone());
    let (completion_tx, completion_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let result = durability.wait().await;
        if let Some(handoff) = handoff {
            handoff.finish(result.is_ok());
        }

        if let Err(error) = &result {
            telemetry::record_rejected_request(
                demand_tracker.as_ref(),
                &physical_lane,
                "publish_ack_failed",
            );
            warn!(
                request_id = %request_id,
                lane = %physical_lane,
                error = %error,
                "dispatch durability was not confirmed; retaining pending demand and aborting request"
            );
        }

        // Notify the inline request/stream driver first. Cleanup must never
        // delay the client-visible transport failure.
        let _ = completion_tx.send(result.clone());

        if result.is_err() {
            // Poll teardown and worker cancellation concurrently under
            // independent bounds. `abort_pending_dispatch` removes the local
            // collector before its first payload-store await, so a stalled
            // Core-NATS cancel can never keep stream state alive indefinitely.
            let (cleanup, cancel) = tokio::join!(
                tokio::time::timeout(
                    DISPATCH_FAILURE_CLEANUP_TIMEOUT,
                    work_publisher.abort_pending_dispatch(&request_id, kind),
                ),
                tokio::time::timeout(
                    DISPATCH_FAILURE_CLEANUP_TIMEOUT,
                    work_publisher.publish_cancel(&request_id),
                ),
            );
            if cleanup.is_err() {
                warn!(
                    request_id = %request_id,
                    timeout_ms = DISPATCH_FAILURE_CLEANUP_TIMEOUT.as_millis(),
                    "dispatch durability failure payload cleanup timed out; periodic cleanup will retry"
                );
            }
            if cancel.is_err() {
                warn!(
                    request_id = %request_id,
                    timeout_ms = DISPATCH_FAILURE_CLEANUP_TIMEOUT.as_millis(),
                    "dispatch durability failure cancel timed out"
                );
            }
        }
    });
    completion_rx
}

fn build_provisioning_response_for_surface(
    gpu: &str,
    bundle: &str,
    surface: ProvisioningSurface,
) -> Response {
    let message = provisioning_message(gpu, bundle);
    match surface {
        ProvisioningSurface::Native => {
            record_provisioning_response(surface, StatusCode::SERVICE_UNAVAILABLE);
            let mut resp = (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "error": {
                        "code": err_code::PROVISIONING,
                        "message": message,
                    }
                })),
            )
                .into_response();
            add_provisioning_headers(&mut resp, PROVISIONING_RETRY_AFTER);
            resp.headers_mut().insert(
                HeaderName::from_static("x-sie-error-code"),
                HeaderValue::from_static(err_code::PROVISIONING),
            );
            resp
        }
        ProvisioningSurface::OpenAiCompat => {
            record_provisioning_response(surface, StatusCode::SERVICE_UNAVAILABLE);
            let mut resp = (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json_openai_error(
                    message,
                    oai_type::SERVER_ERROR,
                    None,
                    oai_code::PROVISIONING,
                )),
            )
                .into_response();
            add_provisioning_headers(&mut resp, PROVISIONING_RETRY_AFTER);
            resp.headers_mut().insert(
                HeaderName::from_static("x-sie-error-code"),
                HeaderValue::from_static(err_code::PROVISIONING),
            );
            resp
        }
    }
}

fn build_openai_provisioning_response(gpu: &str, bundle: &str) -> Response {
    build_provisioning_response_for_surface(gpu, bundle, ProvisioningSurface::OpenAiCompat)
}

fn pool_not_found_message(pool: &str) -> String {
    format!("Pool '{}' not found", pool)
}

fn build_pool_not_found_response_for_surface(pool: &str, surface: ProvisioningSurface) -> Response {
    let message = pool_not_found_message(pool);
    match surface {
        ProvisioningSurface::Native => (
            StatusCode::NOT_FOUND,
            Json(json_detail(err_code::POOL_NOT_FOUND, message)),
        )
            .into_response(),
        ProvisioningSurface::OpenAiCompat => (
            StatusCode::NOT_FOUND,
            Json(json_openai_error(
                message,
                oai_type::INVALID_REQUEST,
                Some("pool"),
                oai_code::INVALID_REQUEST,
            )),
        )
            .into_response(),
    }
}

fn provisioning_surface_for_endpoint(endpoint: &str) -> ProvisioningSurface {
    if endpoint_uses_openai_envelope(endpoint) {
        ProvisioningSurface::OpenAiCompat
    } else {
        ProvisioningSurface::Native
    }
}

async fn capped_lane_admission_response(
    state: &AppState,
    admission_pool: &str,
    demand_pool: &str,
    machine_profile: &str,
    bundle: &str,
    provisioning_surface: ProvisioningSurface,
) -> Option<Response> {
    let admission = state
        .pool_manager
        .capped_lane_status(admission_pool, machine_profile, bundle)
        .await?;

    if admission.cap == 0 {
        let message = format!(
            "Pool '{}' admits zero workers for GPU type '{}' and bundle '{}'.",
            admission_pool, machine_profile, bundle
        );
        let mut m = Map::new();
        m.insert("pool".to_string(), json!(admission_pool));
        m.insert("gpu".to_string(), json!(machine_profile));
        m.insert("bundle".to_string(), json!(bundle));
        m.insert("cap".to_string(), json!(admission.cap));
        let mut resp = match provisioning_surface {
            ProvisioningSurface::Native => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json_detail_merge(
                    err_code::POOL_CAPACITY_UNAVAILABLE,
                    message,
                    m,
                )),
            )
                .into_response(),
            ProvisioningSurface::OpenAiCompat => (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json_openai_error(
                    message,
                    oai_type::SERVER_ERROR,
                    None,
                    oai_code::TRANSPORT_FAILURE,
                )),
            )
                .into_response(),
        };
        resp.headers_mut().insert(
            HeaderName::from_static("x-sie-error-code"),
            HeaderValue::from_static(err_code::POOL_CAPACITY_UNAVAILABLE),
        );
        resp.headers_mut().insert(
            HeaderName::from_static("x-sie-version"),
            HeaderValue::from_static(GATEWAY_VERSION),
        );
        resp.headers_mut().insert(
            HeaderName::from_static("x-sie-server-version"),
            HeaderValue::from_static(GATEWAY_VERSION),
        );
        return Some(resp);
    }

    if admission.assigned_count == 0 {
        if let Some(lane) = state
            .demand_tracker
            .resolve_lane(demand_pool, machine_profile, bundle)
        {
            state.demand_tracker.record(&lane);
        }
        return Some(build_provisioning_response_for_surface(
            machine_profile,
            bundle,
            provisioning_surface,
        ));
    }

    None
}

async fn batch_publish_target(
    state: &AppState,
    pool: &str,
    admission_pool: &str,
    model: &str,
    machine_profile: &str,
    bundle: &str,
    bundle_config_hash: &str,
) -> Result<publisher::PublishTarget, ()> {
    let pool_target = publisher::PublishTarget::Pool {
        pool: pool.to_string(),
        machine_profile: machine_profile.to_string(),
        bundle: bundle.to_string(),
        model: model.to_string(),
    };
    let Some(admitted_worker_names) = state
        .pool_manager
        .admitted_worker_names_for_capped_lane(admission_pool, machine_profile, bundle)
        .await
    else {
        return Ok(pool_target);
    };
    if admitted_worker_names.is_empty() {
        return Err(());
    }

    let ring = state.registry.lazy_lane_ring_snapshot_for_admitted(
        model,
        pool,
        machine_profile,
        bundle,
        bundle_config_hash,
        &admitted_worker_names,
    );
    let seed = uuid::Uuid::now_v7().to_string();
    let key = crate::routing::key::RoutingKeyResolved {
        hash: Some(crate::routing::key::hash_bytes(&seed)),
        source: crate::routing::key::KeySource::RoutingKey,
        #[cfg(feature = "raw-routing-logs")]
        raw_for_debug: None,
    };
    let Some(worker_id) = crate::routing::pick_worker(&ring, &key) else {
        return Err(());
    };

    Ok(publisher::PublishTarget::Worker {
        pool: pool.to_string(),
        machine_profile: machine_profile.to_string(),
        bundle: bundle.to_string(),
        model: model.to_string(),
        worker_id: worker_id.to_string(),
    })
}

#[utoipa::path(
    post,
    path = "/v1/encode/{model}",
    tag = "inference",
    description = "Mixed-success batches return 200 with only successful items; the response carries no per-item error envelope. For per-item error visibility, send single-item batches.",
    params(
        ("model" = String, Path, description = "Model id; percent-encode slashes when using OpenAPI-generated clients"),
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    request_body = crate::openapi::EncodeRequest,
    responses(
        (status = 200, description = "Encode response", body = crate::openapi::EncodeResponse),
        (status = 400, description = "Invalid request", body = crate::openapi::StandardApiError),
        (status = 401, description = "Missing or invalid bearer token", body = crate::openapi::StandardApiError),
        (status = 404, description = "Model not found", body = crate::openapi::StandardApiError),
        (status = 409, description = "Bundle override conflicts with model routing", body = crate::openapi::BundleConflictResponse),
        (status = 413, description = "Request body too large", body = crate::openapi::StandardApiError),
        (status = 500, description = "All batch items failed or gateway internal error", body = crate::openapi::InferenceInternalServerErrorResponse),
        (status = 502, description = "Terminal model load failure (MODEL_LOAD_FAILED)", body = crate::openapi::GatewayModelLoadFailedResponse),
        (status = 503, description = "Provisioning in progress, queue unavailable, GPU not configured, model loading, or capacity exhausted", body = crate::openapi::InferenceServiceUnavailableResponse),
        (status = 504, description = "Result channel closed", body = crate::openapi::StandardApiError)
    )
)]
pub async fn proxy_encode(state: State<Arc<AppState>>, req: Request) -> impl IntoResponse {
    proxy_request(state, req, "encode").await
}

#[utoipa::path(
    post,
    path = "/v1/score/{model}",
    tag = "inference",
    description = "Mixed-success batches return 200 with only successful items; the response carries no per-item error envelope. For per-item error visibility, send single-item batches.",
    params(
        ("model" = String, Path, description = "Model id; percent-encode slashes when using OpenAPI-generated clients"),
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    request_body = crate::openapi::ScoreRequest,
    responses(
        (status = 200, description = "Score response", body = crate::openapi::ScoreResponse),
        (status = 400, description = "Invalid request", body = crate::openapi::StandardApiError),
        (status = 401, description = "Missing or invalid bearer token", body = crate::openapi::StandardApiError),
        (status = 404, description = "Model not found", body = crate::openapi::StandardApiError),
        (status = 409, description = "Bundle override conflicts with model routing", body = crate::openapi::BundleConflictResponse),
        (status = 413, description = "Request body too large", body = crate::openapi::StandardApiError),
        (status = 500, description = "All batch items failed or gateway internal error", body = crate::openapi::InferenceInternalServerErrorResponse),
        (status = 502, description = "Terminal model load failure (MODEL_LOAD_FAILED)", body = crate::openapi::GatewayModelLoadFailedResponse),
        (status = 503, description = "Provisioning in progress, queue unavailable, GPU not configured, model loading, or capacity exhausted", body = crate::openapi::InferenceServiceUnavailableResponse),
        (status = 504, description = "Result channel closed", body = crate::openapi::StandardApiError)
    )
)]
pub async fn proxy_score(state: State<Arc<AppState>>, req: Request) -> impl IntoResponse {
    proxy_request(state, req, "score").await
}

#[utoipa::path(
    post,
    path = "/v1/extract/{model}",
    tag = "inference",
    description = "Successful extract work results may carry an aligned per-item error alongside partial data. Transport-level mixed failures retain generic batch behavior: the 200 response includes only successful work results.",
    params(
        ("model" = String, Path, description = "Model id; percent-encode slashes when using OpenAPI-generated clients"),
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    request_body = crate::openapi::ExtractRequest,
    responses(
        (status = 200, description = "Extract response", body = crate::openapi::ExtractResponse),
        (status = 400, description = "Invalid request", body = crate::openapi::StandardApiError),
        (status = 401, description = "Missing or invalid bearer token", body = crate::openapi::StandardApiError),
        (status = 404, description = "Model not found", body = crate::openapi::StandardApiError),
        (status = 409, description = "Bundle override conflicts with model routing", body = crate::openapi::BundleConflictResponse),
        (status = 413, description = "Request body too large", body = crate::openapi::StandardApiError),
        (status = 500, description = "All batch items failed or gateway internal error", body = crate::openapi::InferenceInternalServerErrorResponse),
        (status = 502, description = "Terminal model load failure (MODEL_LOAD_FAILED)", body = crate::openapi::GatewayModelLoadFailedResponse),
        (status = 503, description = "Provisioning in progress, queue unavailable, GPU not configured, model loading, or capacity exhausted", body = crate::openapi::InferenceServiceUnavailableResponse),
        (status = 504, description = "Result channel closed", body = crate::openapi::StandardApiError)
    )
)]
pub async fn proxy_extract(state: State<Arc<AppState>>, req: Request) -> impl IntoResponse {
    proxy_request(state, req, "extract").await
}

#[utoipa::path(
    post,
    path = "/v1/generate/{model}",
    tag = "inference",
    description = "SIE-native text generation. Omit ``stream`` or set it to false for a blocking \
                   JSON response; set ``stream: true`` for Server-Sent Events terminated by \
                   ``data: [DONE]``. The model path parameter must use the SIE-safe ID \
                   (e.g. ``Qwen__Qwen3-4B-Instruct``); HF-style slashes are rejected with 400.",
    request_body = crate::openapi::GenerateRequest,
    params(
        ("model" = String, Path, description = "SIE-safe model id (double-underscore separator)"),
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    responses(
        (status = 200, description = "Generated text as blocking JSON or SIE-native SSE events", body = crate::openapi::GenerateResponse),
        (status = 400, description = "Invalid request", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 404, description = "Model not found", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 413, description = "Request body too large", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 500, description = "Worker emitted malformed response", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 502, description = "Terminal model load failure (MODEL_LOAD_FAILED)", body = crate::openapi::GatewayModelLoadFailedResponse),
        (status = 503, description = "Provisioning in progress, queue unavailable, or model loading", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 504, description = "Generation timeout", body = crate::openapi::OpenAIErrorEnvelope),
    )
)]
pub async fn proxy_generate(state: State<Arc<AppState>>, req: Request) -> impl IntoResponse {
    // Reject HF-style slashes explicitly with a helpful error. The wildcard
    // route ``/v1/generate/{*model}`` happily accepts ``Qwen/Qwen3-4B-Instruct``
    // — but the resolution registry keys on the SIE-safe ID, so we surface a
    // 400 with the rewrite suggestion rather than a confusing 404.
    let path = req.uri().path();
    let raw_model = path.strip_prefix("/v1/generate/").unwrap_or("");
    let decoded = match decode_model_path(raw_model) {
        Ok(model) => model,
        Err(message) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    message,
                    oai_type::INVALID_REQUEST,
                    Some("model"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response();
        }
    };
    if decoded.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                "model path is empty".to_string(),
                oai_type::INVALID_REQUEST,
                Some("model"),
                oai_code::INVALID_REQUEST,
            )),
        )
            .into_response();
    }
    if decoded.contains('/') {
        let sie_safe = decoded.replace('/', "__");
        return (
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                format!(
                    "model path '{decoded}' uses HuggingFace-style slashes; \
                     use the SIE-safe ID '{sie_safe}' instead \
                     (double-underscore separator)"
                ),
                oai_type::INVALID_REQUEST,
                Some("model"),
                oai_code::INVALID_REQUEST,
            )),
        )
            .into_response();
    }
    proxy_request(state, req, "generate").await
}

/// Resolve a request's model id to its serving bundle.
///
/// Extracted from `proxy_request` (#1543) so the model→bundle decision is one
/// testable unit instead of being inlined in the 2000-line handler. Behaviour
/// is identical to the block it replaces: on any failure it returns the exact
/// caller-facing error `Response` (boxed to keep the `Result` small) that the
/// inline code produced — model-not-found 404, bundle-routing-conflict 409, or
/// the unknown-model 404 when the registry is populated. With an empty registry
/// it falls back to the caller's bundle override (or `"default"`).
///
/// `requested` is the string the CALLER sent; `model_name` is what it resolved
/// to. The unknown-model 404 echoes `requested`, never `model_name` (#2542): a
/// resolved id can be a registry alias's target, and a 404 that names it hands
/// every caller a read of the alias table — including targets in another org's
/// reserved namespace. It also makes this 404 byte-identical to the #1841
/// cross-org one, which is the whole point of answering "not found" there.
/// The one native-surface `MODEL_NOT_FOUND` body.
///
/// Three native call sites answer "this model is not here": the #1841
/// visibility gate, the populated-registry unknown-model arm of
/// [`resolve_bundle_for_request`], and the #3441 pre-governed ordering check in
/// [`resolve_routing`]. They must stay **byte-identical** — the no-oracle
/// property (#2542) is that a hidden model, an absent model, and a model whose
/// governed route was never compiled are one indistinguishable response — so
/// they render through this one function instead of three copies that can drift.
///
/// `requested` is always the CALLER's string, never a resolved/governed id: a
/// resolved id can be an alias target in another org's reserved namespace.
fn unknown_model_response(endpoint: &str, requested: &str) -> Response {
    endpoint_error_response(
        endpoint,
        StatusCode::NOT_FOUND,
        err_code::MODEL_NOT_FOUND,
        oai_type::MODEL_NOT_FOUND,
        oai_code::MODEL_NOT_FOUND,
        Some("model"),
        format!("Model '{}' not found", requested),
    )
}

/// Is this id absent from a registry that has models?
///
/// The exact condition the populated-registry 404 arm below fires on, named so
/// the #3441 ordering check in [`resolve_routing`] can ask the same question
/// ahead of governed lookup without restating it. An EMPTY registry is
/// deliberately not "absent": that is the pre-bootstrap deployment path, which
/// falls back to the caller's bundle override rather than 404ing.
fn absent_from_populated_registry(registry: &ModelRegistry, model_name: &str) -> bool {
    !registry.model_exists(model_name) && registry.has_any_models()
}

fn resolve_bundle_for_request(
    registry: &ModelRegistry,
    model_name: &str,
    requested: &str,
    bundle_override: &str,
    engine_pin: Option<&str>,
    endpoint: &str,
) -> Result<String, Box<Response>> {
    let bundle_override_ref = if bundle_override.is_empty() {
        None
    } else {
        Some(bundle_override)
    };
    let bundle = if registry.model_exists(model_name) {
        match registry.resolve_bundle_with_engine(model_name, bundle_override_ref, engine_pin) {
            Ok(b) => b,
            Err(ResolveError::ModelNotFound(e)) => {
                return Err(Box::new(endpoint_error_response(
                    endpoint,
                    StatusCode::NOT_FOUND,
                    err_code::MODEL_NOT_FOUND,
                    oai_type::MODEL_NOT_FOUND,
                    oai_code::MODEL_NOT_FOUND,
                    Some("model"),
                    e.to_string(),
                )));
            }
            Err(ResolveError::BundleConflict(e)) => {
                // Bundle conflict is non-OpenAI shaped; keep legacy envelope
                // even for generate so the extra ``compatible_bundles`` array
                // stays accessible to existing clients.
                let mut m = Map::new();
                m.insert(
                    "compatible_bundles".to_string(),
                    json!(e.compatible_bundles),
                );
                return Err(Box::new(
                    (
                        StatusCode::CONFLICT,
                        Json(json_detail_merge(
                            err_code::BUNDLE_ROUTING_CONFLICT,
                            e.to_string(),
                            m,
                        )),
                    )
                        .into_response(),
                ));
            }
        }
    } else if registry.has_any_models() {
        return Err(Box::new(unknown_model_response(endpoint, requested)));
    } else if bundle_override.is_empty() {
        "default".to_string()
    } else {
        bundle_override.to_string()
    };
    Ok(bundle)
}

/// Normalized machine-profile + pool routing resolved from the request
/// headers. Shared by the primitive (`resolve_routing`) and generation
/// (`resolve_generation_route`) paths so the header parse → pool validate →
/// model-pool default → machine-profile resolution lives in one place.
struct ProfilePoolRoute {
    gpu: String,
    pool_name: String,
    gpu_configured: bool,
}

/// Failure modes of [`resolve_profile_and_pool`].
///
/// Error *rendering* stays with each caller so the native
/// (`endpoint_error_response`) and OpenAI (`json_openai_error`) surfaces keep
/// their exact status / code / param wire shape; this enum only carries the
/// labels both callers need for the rejection metric.
enum ProfilePoolError {
    /// A caller-supplied routing header is not valid UTF-8.
    InvalidHeader { header: &'static str },
    /// Caller-supplied pool failed the strict `[A-Za-z0-9_-]` allowlist.
    InvalidPool,
    /// Model is assigned to a different pool than the request targeted.
    PoolMismatch { message: String },
}

/// Parse only caller-supplied profile/pool overrides.
///
/// Deployment-governed generation uses this raw override view to validate a
/// caller pin against the exact compiled route. Registry-driven routing calls
/// [`resolve_profile_and_pool`] below, which additionally applies the model's
/// default pool. Keeping those steps separate prevents an absent header from
/// being mistaken for an explicit request that conflicts with desired state.
fn parse_profile_and_pool_overrides(
    state: &AppState,
    headers: &HeaderMap,
) -> Result<ProfilePoolRoute, ProfilePoolError> {
    let mut gpu = match headers.get("x-sie-machine-profile") {
        Some(value) => {
            value
                .to_str()
                .map(str::to_owned)
                .map_err(|_| ProfilePoolError::InvalidHeader {
                    header: "X-SIE-MACHINE-PROFILE",
                })?
        }
        None => String::new(),
    };
    let mut pool_name = match headers.get("x-sie-pool") {
        Some(value) => {
            value
                .to_str()
                .map(str::to_owned)
                .map_err(|_| ProfilePoolError::InvalidHeader {
                    header: "X-SIE-Pool",
                })?
        }
        None => String::new(),
    };

    // Parse pool from GPU param (e.g., "eval-l4/l4").
    if let Some((inline_pool, inline_gpu)) = gpu
        .split_once('/')
        .map(|(pool, profile)| (pool.to_string(), profile.to_string()))
    {
        if inline_pool.is_empty() || inline_gpu.is_empty() {
            return Err(ProfilePoolError::InvalidPool);
        }
        if !pool_name.is_empty() && !pool_name.eq_ignore_ascii_case(&inline_pool) {
            return Err(ProfilePoolError::PoolMismatch {
                message: format!(
                    "X-SIE-Pool '{}' conflicts with pool '{}' in X-SIE-MACHINE-PROFILE",
                    pool_name, inline_pool
                ),
            });
        }
        pool_name = inline_pool;
        gpu = inline_gpu;
    }

    if !pool_name.is_empty() && !is_valid_pool_name(&pool_name) {
        return Err(ProfilePoolError::InvalidPool);
    }
    if !pool_name.is_empty() {
        pool_name = normalize_pool_name(&pool_name);
    }

    if !gpu.is_empty() {
        gpu = resolve_machine_profile(&gpu, &state.config.gpu_profile_map);
    }

    let gpu_configured = gpu.is_empty()
        || state
            .config
            .configured_gpus
            .iter()
            .any(|configured| configured.eq_ignore_ascii_case(&gpu));

    Ok(ProfilePoolRoute {
        gpu,
        pool_name,
        gpu_configured,
    })
}

/// Resolve the machine-profile + pool routing carried by the request headers.
///
/// Parses `X-SIE-MACHINE-PROFILE` (optionally a combined
/// `pool/machine-profile`) and `X-SIE-Pool`, validates the pool against the
/// strict allowlist BEFORE it can flow into the JetStream work subject
/// `sie.work.{pool}.{machine_profile}.{bundle}.{model}` (an out-of-charset pool
/// would re-tokenise the subject — subject injection — and silently mis-route,
/// so reject rather than mangle), applies the model's default pool, resolves a
/// bare GPU to its spot variant, and reports whether the GPU is configured.
///
/// Pure resolution: it records no metrics and builds no response bodies, so
/// each caller keeps its own rejection metric and error envelope. Behaviour
/// mirrors the inline block it replaced in both callers exactly.
async fn resolve_profile_and_pool(
    state: &AppState,
    headers: &HeaderMap,
    model_name: &str,
) -> Result<ProfilePoolRoute, ProfilePoolError> {
    let ProfilePoolRoute {
        gpu,
        mut pool_name,
        gpu_configured,
    } = parse_profile_and_pool_overrides(state, headers)?;

    let model_pool = state.model_registry.get_model_pool_name(model_name);
    if let Err(message) = apply_model_pool_default(
        &mut pool_name,
        model_pool.as_deref(),
        Some(&state.pool_manager),
    )
    .await
    {
        return Err(ProfilePoolError::PoolMismatch { message });
    }

    Ok(ProfilePoolRoute {
        gpu,
        pool_name,
        gpu_configured,
    })
}

fn governed_generation_route_failure(
    customer_model: &str,
    intent: GenerationRequestIntent,
    detail: &str,
) -> Response {
    error!(
        customer_model,
        intent = ?intent,
        detail,
        "deployment-governed generation route is unavailable"
    );
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json_openai_error(
            "Generation route is unavailable for this deployment".to_string(),
            oai_type::SERVER_ERROR,
            None,
            oai_code::TRANSPORT_FAILURE,
        )),
    )
        .into_response()
}

/// Resolve the optional deployment-owned generation route.
///
/// `None` means the OSS composition installed no policy. Once a policy is
/// present, every model/intent pair must resolve and the returned route must be
/// self-consistent; any miss or drift fails closed instead of falling back to
/// registry insertion order or a sibling GPU lane.
#[allow(clippy::result_large_err)]
fn governed_generation_route(
    state: &AppState,
    customer_model: &str,
    dispatch_model: &str,
    intent: GenerationRequestIntent,
) -> Result<Option<GovernedGenerationRoute>, Response> {
    let Some(policy) = state
        .model_access_policy
        .as_deref()
        .and_then(ModelAccessPolicy::generation_route_policy)
    else {
        return Ok(None);
    };
    let Some(route) = policy.resolve(customer_model, intent) else {
        return Err(governed_generation_route_failure(
            customer_model,
            intent,
            "missing model/intent mapping",
        ));
    };
    if route.model != dispatch_model {
        return Err(governed_generation_route_failure(
            customer_model,
            intent,
            "compiled physical model disagrees with registry intent rewrite",
        ));
    }
    if route.bundle.trim().is_empty()
        || route.pool.trim().is_empty()
        || route.machine_profile.trim().is_empty()
    {
        return Err(governed_generation_route_failure(
            customer_model,
            intent,
            "route contains an empty physical coordinate",
        ));
    }
    if !state
        .config
        .configured_gpus
        .iter()
        .any(|configured| configured.eq_ignore_ascii_case(&route.machine_profile))
    {
        return Err(governed_generation_route_failure(
            customer_model,
            intent,
            "route machine profile is absent from the configured GPU catalog",
        ));
    }
    Ok(Some(route))
}

/// Validate caller overrides against one exact deployment-owned route.
///
/// An omitted header adopts the governed coordinate. An explicit, conflicting
/// header is a client error; it never causes a registry search that could pick
/// a sibling lane for the same model.
#[allow(clippy::result_large_err)]
fn governed_profile_and_pool(
    state: &AppState,
    headers: &HeaderMap,
    route: &GovernedGenerationRoute,
) -> Result<ProfilePoolRoute, Response> {
    let overrides = match parse_profile_and_pool_overrides(state, headers) {
        Ok(overrides) => overrides,
        Err(ProfilePoolError::InvalidHeader { header }) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    format!("{header} header must be valid UTF-8"),
                    oai_type::INVALID_REQUEST,
                    Some(header),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response());
        }
        Err(ProfilePoolError::InvalidPool) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    "Invalid pool name: only [A-Za-z0-9_-] are allowed (max 128 chars)".to_string(),
                    oai_type::INVALID_REQUEST,
                    Some("X-SIE-Pool"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response());
        }
        Err(ProfilePoolError::PoolMismatch { message }) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    message,
                    oai_type::INVALID_REQUEST,
                    Some("X-SIE-Pool"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response());
        }
    };

    if !overrides.pool_name.is_empty() && !overrides.pool_name.eq_ignore_ascii_case(&route.pool) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                format!(
                    "X-SIE-Pool '{}' conflicts with the deployment route for this model",
                    overrides.pool_name
                ),
                oai_type::INVALID_REQUEST,
                Some("X-SIE-Pool"),
                oai_code::INVALID_REQUEST,
            )),
        )
            .into_response());
    }
    if !overrides.gpu.is_empty() && !overrides.gpu.eq_ignore_ascii_case(&route.machine_profile) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                format!(
                    "X-SIE-MACHINE-PROFILE '{}' conflicts with the deployment route for this model",
                    overrides.gpu
                ),
                oai_type::INVALID_REQUEST,
                Some("X-SIE-MACHINE-PROFILE"),
                oai_code::INVALID_REQUEST,
            )),
        )
            .into_response());
    }

    Ok(ProfilePoolRoute {
        gpu: route.machine_profile.clone(),
        pool_name: route.pool.clone(),
        gpu_configured: true,
    })
}

/// The routing decision for a request: canonical model name, serving bundle,
/// and engine. Produced by [`resolve_routing`].
struct RoutingResult {
    model_name: String,
    dispatch_model: String,
    bundle: String,
    engine: String,
    gpu: String,
    pool_name: String,
    gpu_configured: bool,
}

/// Resolve a request into its routing decision (model → bundle → engine).
///
/// Extracted from `proxy_request` (#1543) so the routing decision is one unit,
/// composed of the already-tested `decode_model_path` / `resolve_model_spec_with_aliases`
/// / `parse_engine_pin` / [`resolve_bundle_for_request`] helpers. Behaviour is
/// identical to the inline block it replaces: on any failure it returns the
/// exact caller-facing error `Response` (boxed) the handler produced.
async fn resolve_routing(
    state: &AppState,
    headers: &HeaderMap,
    ext: &axum::http::Extensions,
    path: &str,
    endpoint: &str,
    generation_intent: Option<GenerationRequestIntent>,
) -> Result<RoutingResult, Box<Response>> {
    // Extract model from path: /v1/{endpoint}/{model...}
    let prefix = format!("/v1/{}/", endpoint);
    let raw_model = path.strip_prefix(&prefix).unwrap_or("");
    let model = match decode_model_path(raw_model) {
        Ok(model) => model,
        Err(message) => {
            return Err(Box::new(endpoint_error_response(
                endpoint,
                StatusCode::BAD_REQUEST,
                err_code::INVALID_REQUEST,
                oai_type::INVALID_REQUEST,
                oai_code::INVALID_REQUEST,
                Some("model"),
                message,
            )));
        }
    };

    if model.is_empty() {
        return Err(Box::new(endpoint_error_response(
            endpoint,
            StatusCode::BAD_REQUEST,
            err_code::INVALID_REQUEST,
            oai_type::INVALID_REQUEST,
            oai_code::INVALID_REQUEST,
            Some("model"),
            "model is required",
        )));
    }

    // Parse model spec: [bundle:/]org/model, expanding a job/friendly alias
    // whose target may carry a precision/profile bundle (see
    // resolve_model_spec_with_aliases).
    let (bundle_override, model_name) =
        resolve_model_spec_with_aliases(&state.config.model_aliases, &model, |m| {
            state.model_registry.resolve_canonical_model_name(m)
        });
    // #1841 org-scoped visibility: a custom model hidden from this caller is
    // treated as ABSENT — the identical MODEL_NOT_FOUND 404 the dispatcher emits
    // for an unknown model — so there is no cross-org existence oracle. Decided on
    // the RESOLVED id (alias/`__`/case already folded), so the gate can never
    // diverge from the dispatcher. No-op in OSS self-host (policy is None).
    //
    // The body echoes the CALLER's string and uses the unknown-model template
    // verbatim (#2542). Rendering `model_name` here was a naming oracle: an
    // operator alias whose target sits in another org's reserved namespace made
    // `POST /v1/encode/<alias>` answer with that org's custom-model id, to a
    // caller who owns nothing. It also gave the refusal its own wording, which
    // by itself distinguished "hidden from you" from "does not exist" — the one
    // thing this 404 exists to prevent.
    if state
        .model_access_policy
        .as_ref()
        .is_some_and(|p| !p.visible(&model_name, ext))
    {
        return Err(Box::new(unknown_model_response(endpoint, &model)));
    }
    // #2542 deployment serving policy on the RESOLVED id: the authoritative
    // "we do not serve this model here" verdict (the managed service's §6.5
    // licence exclusion). It runs HERE, not only at the edge, because an edge
    // gate compares the caller's raw string and a registry ALIAS can point that
    // string at a refused model — post-resolution is the only place the refusal
    // sees what will actually dispatch. Ordered after the visibility gate so the
    // no-oracle 404 still wins, and before the sealed branch so a refused model
    // cannot reach a lane. The policy owns the whole response (status, code,
    // envelope). No-op in OSS self-host (policy is None / default `None`).
    if let Some(refusal) = state
        .model_access_policy
        .as_ref()
        .and_then(|p| p.serving_refusal(&model_name, ext))
    {
        return Err(Box::new(refusal));
    }
    // #1841 bundle-less sealed dispatch: an org-registered custom model is served
    // from its own sealed sandbox, not a catalog bundle, so there is no bundle to
    // resolve — the normal path below would dead-end at MODEL_NOT_FOUND. The
    // visibility gate above already proved this caller OWNS the id, so the policy's
    // `sealed_route` decides purely on id shape; we return the synthetic sealed lane
    // tuple and SKIP bundle + profile resolution. `engine == "sealed"` is the sole
    // discriminator the dispatcher co-process routes on. A client can never reach
    // this via the `X-SIE-Engine` pin below — `"sealed"` is server-derived only and
    // is not in `KNOWN_ENGINES`. No-op in OSS self-host (policy is None).
    if let Some(route) = state
        .model_access_policy
        .as_ref()
        .and_then(|p| p.sealed_route(&model_name, ext))
    {
        debug_assert_eq!(route.engine, "sealed");
        return Ok(RoutingResult {
            dispatch_model: model_name.clone(),
            model_name,
            bundle: route.bundle,
            engine: route.engine,
            gpu: route.machine_profile,
            pool_name: route.pool,
            gpu_configured: true,
        });
    }
    // Hidden operator/debug header: normal clients should select explicit
    // model profile variants (for example ``model:candle``) or compatible
    // bundle overrides. Lowercase normalisation + UTF-8 validation lives in
    // ``parse_engine_pin`` (unit-tested below).
    let engine_pin = match parse_engine_pin(headers) {
        EnginePinParse::None => None,
        EnginePinParse::Some(eng) => Some(eng),
        EnginePinParse::Unknown(raw) => {
            use crate::types::bundle::KNOWN_ENGINES;
            // Route through endpoint_error_response so `generate` (an OpenAI-
            // envelope endpoint that also flows through here) gets the OpenAI
            // shape instead of the native `detail` one. See #1567.
            return Err(Box::new(endpoint_error_response(
                endpoint,
                StatusCode::BAD_REQUEST,
                err_code::INVALID_REQUEST,
                oai_type::INVALID_REQUEST,
                oai_code::INVALID_REQUEST,
                None,
                format!("X-SIE-Engine value {:?} is not in {:?}", raw, KNOWN_ENGINES),
            )));
        }
        EnginePinParse::InvalidUtf8 => {
            return Err(Box::new(endpoint_error_response(
                endpoint,
                StatusCode::BAD_REQUEST,
                err_code::INVALID_REQUEST,
                oai_type::INVALID_REQUEST,
                oai_code::INVALID_REQUEST,
                None,
                "X-SIE-Engine header must be valid UTF-8 (got non-printable bytes)",
            )));
        }
    };

    let mut dispatch_model = model_name.clone();
    if generation_intent == Some(GenerationRequestIntent::Grammar) {
        route_grammar_to_profile(&state.model_registry, &mut dispatch_model);
    }
    // #3441 ordering: a model this data plane does not have is answered by the
    // populated-registry `MODEL_NOT_FOUND` 404 BEFORE the governed lookup runs.
    //
    // Governed lookup is keyed on `(customer_model, intent)`, so an unknown id is
    // simply a policy miss, and a miss fails closed as a 503 — the correct verdict
    // for *deployment drift* (a catalog model whose route was not compiled), but
    // the wrong one for a model that does not resolve here at all. §7.5 of the
    // managed-service design separates the two: unresolvable → terminal 404,
    // routable-but-broken → retryable 503. Without this check a typo'd or retired
    // model id (prod-US, 2026-08-14: `Qwen__Qwen3-4B-Instruct-2507`) told the
    // caller to retry forever against a model that will never exist.
    //
    // Only the native generate path needs it: the OpenAI body-model routes reach
    // `resolve_model_and_bundle`'s identical 404 before `resolve_generation_route`
    // performs the same governed lookup. Scoped to `generation_intent` because
    // governed lookup is the only step that can precede the 404 — for every other
    // native endpoint `resolve_bundle_for_request` below is still the first
    // verdict, and reordering it there would change nothing.
    //
    // Sealed custom models (#1841) already returned above; they are legitimately
    // absent from the catalog registry and must not be caught here.
    if generation_intent.is_some()
        && absent_from_populated_registry(&state.model_registry, &model_name)
    {
        return Err(Box::new(unknown_model_response(endpoint, &model)));
    }
    let governed_route = if let Some(intent) = generation_intent {
        match governed_generation_route(state, &model_name, &dispatch_model, intent) {
            Ok(route) => route,
            Err(response) => return Err(Box::new(response)),
        }
    } else {
        None
    };

    if let Some(route) = governed_route.as_ref() {
        if !bundle_override.is_empty() && bundle_override != route.bundle {
            return Err(Box::new(
                (
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        format!(
                            "Bundle override '{}' conflicts with the deployment route for this model",
                            bundle_override
                        ),
                        oai_type::INVALID_REQUEST,
                        Some("model"),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response(),
                ));
        }
        match state
            .model_registry
            .resolve_bundle(&route.model, Some(&route.bundle))
        {
            Ok(resolved) if resolved == route.bundle => {}
            _ => {
                return Err(Box::new(governed_generation_route_failure(
                    &model_name,
                    generation_intent.expect("governed generation has request intent"),
                    "compiled bundle is incompatible with the physical model",
                )));
            }
        }
    }

    // Try model registry resolution. Three cases:
    //   1. Model is known        → resolve bundle (404 on BundleConflict, etc.)
    //   2. Model unknown, registry populated → 404 (fail fast; avoids queueing
    //      requests for typo'd model ids).
    //   3. Model unknown, registry empty     → fall back to caller's bundle
    //      override or "default". This is the pre-bootstrap / no-config
    //      deployment path; workers may still match on bundle+gpu alone.
    let bundle_model = governed_route
        .as_ref()
        .map_or(model_name.as_str(), |route| route.model.as_str());
    let effective_bundle_override = governed_route
        .as_ref()
        .map_or(bundle_override.as_str(), |route| route.bundle.as_str());
    let bundle = resolve_bundle_for_request(
        &state.model_registry,
        bundle_model,
        // The CALLER's string, for the not-found body only (#2542) — never the
        // resolved/governed id, which can be an alias target in another org's
        // reserved namespace.
        &model,
        effective_bundle_override,
        engine_pin.as_deref(),
        endpoint,
    )?;

    let engine = state
        .model_registry
        .get_bundle_info(&bundle)
        .map(|info| info.engine)
        .or_else(|| engine_pin.clone())
        .unwrap_or_else(|| crate::types::bundle::DEFAULT_ENGINE.to_string());

    // Machine-profile + pool resolution, shared with the generation path via
    // `resolve_profile_and_pool`. Errors are rendered here in this endpoint's
    // envelope (native `detail` or OpenAI, per `endpoint_error_response`).
    let ProfilePoolRoute {
        gpu,
        pool_name,
        gpu_configured,
    } = if let Some(route) = governed_route.as_ref() {
        match governed_profile_and_pool(state, headers, route) {
            Ok(route) => route,
            Err(response) => return Err(Box::new(response)),
        }
    } else {
        match resolve_profile_and_pool(state, headers, &model_name).await {
            Ok(route) => route,
            Err(ProfilePoolError::InvalidHeader { header }) => {
                return Err(Box::new(endpoint_error_response(
                    endpoint,
                    StatusCode::BAD_REQUEST,
                    err_code::INVALID_REQUEST,
                    oai_type::INVALID_REQUEST,
                    oai_code::INVALID_REQUEST,
                    Some(header),
                    format!("{header} header must be valid UTF-8"),
                )));
            }
            Err(ProfilePoolError::InvalidPool) => {
                return Err(Box::new(endpoint_error_response(
                    endpoint,
                    StatusCode::BAD_REQUEST,
                    err_code::INVALID_REQUEST,
                    oai_type::INVALID_REQUEST,
                    oai_code::INVALID_REQUEST,
                    Some("pool"),
                    "Invalid pool name: only [A-Za-z0-9_-] are allowed (max 128 chars)",
                )));
            }
            Err(ProfilePoolError::PoolMismatch { message }) => {
                return Err(Box::new(endpoint_error_response(
                    endpoint,
                    StatusCode::BAD_REQUEST,
                    err_code::INVALID_REQUEST,
                    oai_type::INVALID_REQUEST,
                    oai_code::INVALID_REQUEST,
                    Some("pool"),
                    message,
                )));
            }
        }
    };

    Ok(RoutingResult {
        model_name,
        dispatch_model,
        bundle,
        engine,
        gpu,
        pool_name,
        gpu_configured,
    })
}

pub(crate) async fn proxy_request(
    State(state): State<Arc<AppState>>,
    req: Request,
    endpoint: &str,
) -> Response {
    // SDK version skew detection
    check_sdk_version(req.headers());
    let provisioning_surface = provisioning_surface_for_endpoint(endpoint);

    // Keep the pre-generation queue hot path untouched for encode /
    // score / extract. Native `/v1/generate/{model}` is the only route
    // through this helper that needs gateway-side trace propagation;
    // OpenAI generation routes have their own tracing blocks.
    let proxy_span = if should_trace_proxy_request(endpoint) {
        let parent_cx = managed_request_parent(&req);
        let proxy_span = tracing::info_span!(
            "gateway.proxy",
            otel.name = "gateway.proxy_generate",
            sie.endpoint = endpoint,
            sie.request_id = tracing::field::Empty,
            sie.model = tracing::field::Empty,
        );
        {
            use tracing_opentelemetry::OpenTelemetrySpanExt;
            let _ = proxy_span.set_parent(parent_cx);
        }
        Some(proxy_span)
    } else {
        None
    };
    // #1500: non-generation endpoints (encode / score / extract /
    // embeddings) do not open a billed gateway span, but the
    // queue-publish path still needs the inbound W3C context so
    // `inject_current_context()` re-emits it into the work-item envelope
    // (otherwise the worker's `worker.run_batch` span roots a fresh trace
    // instead of continuing the client's). We capture the extracted
    // context here and scope it over the publish future via
    // `with_context` (Send- and thread-hop-safe, unlike attaching a
    // `!Send` `ContextGuard` across the handler's awaits). The generation
    // branch establishes the context through the poll-scoped proxy span
    // instrumentation below.
    let inbound_publish_cx = if proxy_span.is_some() {
        None
    } else {
        Some(managed_request_parent(&req))
    };
    let proxy_span = proxy_span.unwrap_or_else(tracing::Span::none);

    // `Span::enter()` guards must never live across `.await`: Tokio may resume
    // the task on another worker, stranding tracing-opentelemetry's context
    // guard in the first worker's TLS until thread teardown. Instrumenting the
    // future enters and exits the span around each poll instead.
    async move {
        proxy_request_inner(
            state,
            req,
            endpoint,
            provisioning_surface,
            inbound_publish_cx,
        )
        .await
    }
    .instrument(proxy_span)
    .await
}

async fn proxy_request_inner(
    state: Arc<AppState>,
    mut req: Request,
    endpoint: &str,
    provisioning_surface: ProvisioningSurface,
    inbound_publish_cx: Option<opentelemetry::Context>,
) -> Response {
    // Native generation routing depends on request intent (default vs grammar),
    // including in the no-policy OSS composition where grammar selects a
    // profile-qualified model. Inspect the bounded body once before worker
    // lookup and preserve the typed parse for the dispatch driver.
    let mut governed_generate_body = None;
    let mut governed_generate_parsed = None;
    let generation_intent = if endpoint == "generate" {
        let is_msgpack = req
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|content_type| content_type.contains("msgpack"));
        let (parts, body) = req.into_parts();
        let body_bytes = match axum::body::to_bytes(body, MAX_GENERATE_BODY).await {
            Ok(body) => body,
            Err(error) => {
                warn!(error = %error, limit = MAX_GENERATE_BODY, "request body too large or read error");
                return endpoint_error_response(
                    endpoint,
                    StatusCode::PAYLOAD_TOO_LARGE,
                    err_code::PAYLOAD_TOO_LARGE,
                    oai_type::INVALID_REQUEST,
                    oai_code::INVALID_REQUEST,
                    None,
                    format!("Request body too large (max {} bytes)", MAX_GENERATE_BODY),
                );
            }
        };
        req = Request::from_parts(parts, Body::empty());
        let (items, params) = match parse_queue_request(&body_bytes, is_msgpack, endpoint) {
            Ok(parsed) => parsed,
            Err(error) => return queue_parse_error_response(endpoint, error),
        };
        let intent = if params
            .generate
            .as_ref()
            .and_then(|generate| generate.grammar.as_ref())
            .is_some()
        {
            GenerationRequestIntent::Grammar
        } else {
            GenerationRequestIntent::Default
        };
        governed_generate_body = Some(body_bytes);
        governed_generate_parsed = Some((items, params));
        Some(intent)
    } else {
        None
    };

    let RoutingResult {
        model_name,
        dispatch_model,
        bundle,
        engine,
        gpu,
        pool_name,
        gpu_configured,
    } = match resolve_routing(
        &state,
        req.headers(),
        req.extensions(),
        req.uri().path(),
        endpoint,
        generation_intent,
    )
    .await
    {
        Ok(r) => r,
        Err(resp) => return *resp,
    };

    if let Some((items, params)) = governed_generate_parsed.as_ref() {
        if let Some(response) = validate_native_generate_pre_admission(
            &state,
            &model_name,
            &dispatch_model,
            items,
            params,
        ) {
            return response;
        }
    }

    // Publish the canonical `machine_profile` to the HTTP metrics
    // middleware via a request extension slot. The middleware reads
    // this AFTER the inner service responds, so every downstream
    // return path below — including the `gpu_not_configured` rejection
    // one block down and all the early exits inside `queue_mode_proxy`
    // — automatically gets the normalized label without each site
    // having to remember to tag its response. A fallback of
    // `"unknown"` kicks in at the middleware if we never set the slot
    // (e.g. the `model is required` exit above, which has no GPU).
    if let Some(slot) = req.extensions().get::<telemetry::MetricLabelsSlot>() {
        slot.set(telemetry::MetricLabels {
            machine_profile: if gpu.is_empty() {
                "unknown".to_string()
            } else if gpu_configured {
                gpu.clone()
            } else {
                // Known-invalid attacker input — bucket it so the label
                // space stays bounded.
                "invalid".to_string()
            },
            model: model_name.clone(),
        });
    }

    // `!gpu_configured` already implies a non-empty GPU and a non-empty
    // configured set (see how `gpu_configured` is computed), so no outer
    // guard is needed.
    if !gpu_configured {
        let mut m = Map::new();
        m.insert("gpu".to_string(), json!(&gpu));
        m.insert(
            "configured_gpu_types".to_string(),
            json!(&state.config.configured_gpus),
        );
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json_detail_merge(
                err_code::GPU_NOT_CONFIGURED,
                format!("GPU type '{}' is not configured in this cluster.", gpu),
                m,
            )),
        )
            .into_response();
    }

    let Some(work_publisher) = state.work_publisher.as_ref() else {
        return endpoint_error_response(
            endpoint,
            StatusCode::SERVICE_UNAVAILABLE,
            err_code::QUEUE_UNAVAILABLE,
            oai_type::SERVER_ERROR,
            oai_code::TRANSPORT_FAILURE,
            None,
            "Rust gateway is queue-only, but NATS JetStream is unavailable",
        );
    };

    let Some(hash_pool) = queue_pool_for_request(Some(&state.pool_manager), &pool_name).await
    else {
        let requested_pool = normalize_pool_name(&pool_name);
        return build_pool_not_found_response_for_surface(&requested_pool, provisioning_surface);
    };
    let (bundle_config_hash, model_revision, uses_catalog_scope) = state
        .model_registry
        .bundle_execution_evidence(&bundle, &hash_pool, &dispatch_model);
    if !catalog_execution_hash_is_ready(&bundle_config_hash, uses_catalog_scope) {
        return endpoint_error_response(
            endpoint,
            StatusCode::SERVICE_UNAVAILABLE,
            err_code::QUEUE_UNAVAILABLE,
            oai_type::SERVER_ERROR,
            oai_code::TRANSPORT_FAILURE,
            None,
            "Model execution evidence is unavailable while configuration converges",
        );
    }

    // Resolve the effective pool in one shot. `resolve_effective_pool`
    // folds the demand-tracking probe ("was there an exact
    // (bundle, machine_profile) match?") into the same registry load it uses to pick a
    // route, so we don't make two registry route-resolution calls on the
    // hot path. The `pending_demand_profiles` field is what drives KEDA:
    // it holds the caller's GPU preference when no exact-tuple worker was
    // registered, or, for a cold gpu-agnostic request, every machine profile
    // the pool can provision so each candidate lane can scale from zero.
    let lookup = resolve_effective_pool(
        &state.registry,
        Some(&state.pool_manager),
        &bundle,
        &gpu,
        &pool_name,
        &bundle_config_hash,
    )
    .await;
    let demand_pool = lookup.demand_pool.clone();
    let admission_pool = lookup.admission_pool.clone();
    let pending_demand_profiles = lookup.pending_demand_profiles.clone();
    for profile in &pending_demand_profiles {
        if let Some(lane) = state
            .demand_tracker
            .resolve_lane(&demand_pool, profile, &bundle)
        {
            state.demand_tracker.record(&lane);
        }
    }

    let effective_route = match lookup.resolution {
        PoolResolution::Route(route) => route,
        PoolResolution::PoolNotFound(pool) => {
            return build_pool_not_found_response_for_surface(&pool, provisioning_surface);
        }
        PoolResolution::Provisioning => {
            return build_provisioning_response_for_surface(&gpu, &bundle, provisioning_surface);
        }
    };
    let effective_pool = &effective_route.pool_name;
    let effective_machine_profile = &effective_route.machine_profile;
    let Some(physical_lane) =
        state
            .demand_tracker
            .resolve_lane(&demand_pool, effective_machine_profile, &bundle)
    else {
        error!(
            pool = %demand_pool,
            machine_profile = %effective_machine_profile,
            bundle = %bundle,
            "resolved worker route is absent from configured physical KEDA lane catalog"
        );
        return endpoint_error_response(
            endpoint,
            StatusCode::SERVICE_UNAVAILABLE,
            err_code::QUEUE_UNAVAILABLE,
            oai_type::SERVER_ERROR,
            oai_code::TRANSPORT_FAILURE,
            None,
            "Resolved worker lane is not configured for autoscaling",
        );
    };

    if let Some(resp) = capped_lane_admission_response(
        &state,
        &admission_pool,
        &demand_pool,
        effective_machine_profile,
        &bundle,
        provisioning_surface,
    )
    .await
    {
        return resp;
    }

    let batch_target = if endpoint == "generate" {
        None
    } else {
        match batch_publish_target(
            &state,
            effective_pool,
            &admission_pool,
            &model_name,
            effective_machine_profile,
            &bundle,
            &bundle_config_hash,
        )
        .await
        {
            Ok(target) => Some(target),
            Err(()) => {
                state.demand_tracker.record(&physical_lane);
                return build_provisioning_response_for_surface(
                    effective_machine_profile,
                    &bundle,
                    provisioning_surface,
                );
            }
        }
    };

    let token_id = extract_bearer_token(req.headers())
        .map(|t| mask_token(&t))
        .unwrap_or_default();
    let content_length = req
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(-1);

    // Extract the only two header-derived bits `queue_mode_proxy`
    // actually needs (request content-type and response negotiation)
    // *before* consuming the request. This lets us skip cloning the
    // entire `HeaderMap` just to read two flags on the hot path.
    let is_msgpack_in = req
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.contains("msgpack"))
        .unwrap_or(false);
    let use_msgpack_out = publisher::wants_msgpack(req.headers());

    // Per-endpoint body cap. `generate` allows one 16 MiB decoded inline
    // image after JSON base64 expansion, so its 24 MiB cap closes the
    // trivial-OOM-under-concurrency vector while admitting the maximum
    // legal image. The remaining endpoints routed here
    // (encode / score) get the same text-appropriate 16 MiB cap as the
    // chat / embeddings paths. Extract accepts bounded binary media, so
    // its cap covers the maximum legal audio after JSON base64 expansion.
    let body_limit = native_request_body_limit(endpoint);
    let body_bytes = if let Some(body) = governed_generate_body {
        body
    } else {
        match axum::body::to_bytes(req.into_body(), body_limit).await {
            Ok(b) => b,
            Err(e) => {
                warn!(error = %e, limit = body_limit, "request body too large or read error");
                return endpoint_error_response(
                    endpoint,
                    StatusCode::PAYLOAD_TOO_LARGE,
                    err_code::PAYLOAD_TOO_LARGE,
                    oai_type::INVALID_REQUEST,
                    oai_code::INVALID_REQUEST,
                    None,
                    format!("Request body too large (max {} bytes)", body_limit),
                );
            }
        }
    };

    let publish_fut = queue_mode_proxy(
        &state,
        Arc::clone(work_publisher),
        endpoint,
        &model_name,
        &dispatch_model,
        &bundle,
        &engine,
        effective_machine_profile,
        effective_pool,
        &admission_pool,
        &body_bytes,
        governed_generate_parsed,
        is_msgpack_in,
        use_msgpack_out,
        &token_id,
        content_length,
        Instant::now(),
        &bundle_config_hash,
        model_revision.as_deref(),
        batch_target,
        &physical_lane,
    );
    // Scope an OTel context over the publish so the work-item envelope
    // carries it (#1500): when the exporter is on we open a `gateway.publish`
    // span and scope *its* context (first arm, #1596); otherwise we scope the
    // raw inbound context (second arm). `with_context` re-attaches on every
    // poll, so `Context::current()` is correct across the publish's own
    // `.await`s and tokio thread hops — and the future stays `Send`, unlike
    // holding a `!Send` `ContextGuard` across awaits.
    use opentelemetry::trace::FutureExt;
    let cls = crate::endpoint::InferenceEndpoint::from_label(endpoint);
    match inbound_publish_cx {
        Some(inbound_cx)
            if cls.uses_publish_gateway_span()
                && crate::observability::tracing::exporter_enabled() =>
        {
            // #1596: give the non-streaming queue-publish path a sampled gateway
            // span so we get edge latency attribution + a stable trace root. We
            // scope the span's OWN context (not the raw inbound context) over the
            // publish so the worker attaches to `gateway.publish`.
            let span = tracing::info_span!(
                "gateway.publish",
                sie.endpoint = endpoint,
                sie.model = %model_name,
                sie.pool = %effective_pool,
                sie.publish_ms = tracing::field::Empty,
            );
            {
                use tracing_opentelemetry::OpenTelemetrySpanExt;
                let _ = span.set_parent(inbound_cx);
            }
            let publish_cx = {
                use tracing_opentelemetry::OpenTelemetrySpanExt;
                span.context()
            };
            publish_fut.with_context(publish_cx).instrument(span).await
        }
        Some(inbound_cx) => publish_fut.with_context(inbound_cx).await,
        None => publish_fut.await,
    }
}

fn should_trace_proxy_request(endpoint: &str) -> bool {
    InferenceEndpoint::from_label(endpoint).uses_generation_gateway_tracing()
}

/// Parent handler-local spans to the managed request span installed by the
/// metrics middleware. The header extraction fallback preserves direct handler
/// tests and any future router that intentionally omits that middleware.
fn managed_request_parent(req: &Request) -> opentelemetry::Context {
    req.extensions()
        .get::<telemetry::RequestTraceContext>()
        .map(|context| context.get().clone())
        .unwrap_or_else(|| {
            crate::observability::propagation::extract_context_from_headers(req.headers())
        })
}

/// Route request through the queue-only JetStream path.
///
/// `pool` is always pre-resolved by the caller via
/// [`resolve_effective_pool`] and is guaranteed non-empty: the
/// `PoolResolution::Provisioning` branch returns provisioning before we get
/// here, and every `PoolResolution::Route(_)` path produces a non-empty
/// string (either the caller-pinned `X-SIE-Pool`, or a `pool_name`
/// harvested from the registry snapshot — route resolution filters out
/// empty pool names and machine profiles). We therefore don't need to
/// re-query the registry inside this function.
fn queue_parse_error_response(endpoint: &str, error: QueueParseError) -> Response {
    match error {
        QueueParseError::Generic(message) => endpoint_error_response(
            endpoint,
            StatusCode::BAD_REQUEST,
            err_code::INVALID_REQUEST,
            oai_type::INVALID_REQUEST,
            oai_code::INVALID_REQUEST,
            None,
            format!("Failed to parse request body: {message}"),
        ),
        QueueParseError::PreBuilt(response) => response,
    }
}

fn unsupported_streaming_response(
    registry: &crate::state::model_registry::ModelRegistry,
    model: &str,
    display_model: &str,
    stream: bool,
) -> Option<Response> {
    if !stream {
        return None;
    }
    let streaming_supported = registry
        .get_model_info(model)
        .and_then(|info| info.info_extras.streaming_supported);
    if streaming_supported != Some(false) {
        return None;
    }
    Some(
        (
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                format!("Model '{display_model}' does not support streaming generation"),
                oai_type::INVALID_REQUEST,
                Some("stream"),
                oai_code::UNSUPPORTED_FIELD,
            )),
        )
            .into_response(),
    )
}

fn validate_native_generate_pre_admission(
    state: &AppState,
    display_model: &str,
    model: &str,
    items: &[rmpv::Value],
    params: &publisher::WorkParams,
) -> Option<Response> {
    if let Err(message) = publisher::validate_queue_request_item_count(items.len()) {
        return Some(endpoint_error_response(
            "generate",
            StatusCode::BAD_REQUEST,
            err_code::INVALID_REQUEST,
            oai_type::INVALID_REQUEST,
            oai_code::INVALID_REQUEST,
            None,
            message,
        ));
    }

    let Some(generate) = params.generate.as_ref() else {
        return Some(
            (
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    "generate request requires non-empty 'prompt' and positive integer 'max_new_tokens'",
                    oai_type::INVALID_REQUEST,
                    Some("prompt"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response(),
        );
    };

    if let Some(response) =
        unsupported_streaming_response(&state.model_registry, model, display_model, generate.stream)
    {
        return Some(response);
    }

    if let Some(grammar) = generate.grammar.as_ref() {
        let capabilities = state
            .model_registry
            .get_model_info(model)
            .as_ref()
            .and_then(|entry| entry.info_extras.grammar_capabilities.clone());
        if let Err(response) =
            super::grammar::check_capability(grammar, capabilities.as_deref(), display_model)
        {
            return Some(response);
        }
    }

    if let Some(requested_lora) = generate.lora_adapter.as_deref() {
        if let Some(info) = state.model_registry.get_model_info(model) {
            let profile_name = params
                .options
                .as_ref()
                .and_then(|options| options.get("profile"))
                .and_then(|value| value.as_str())
                .unwrap_or("default");
            match validate_lora_for_profile(&info, profile_name, requested_lora) {
                LoraValidation::Ok => {}
                LoraValidation::UnknownProfile => {
                    return Some(
                        (
                            StatusCode::BAD_REQUEST,
                            Json(json_openai_error(
                                format!(
                                    "unknown profile '{profile_name}' for model '{display_model}'"
                                ),
                                oai_type::INVALID_REQUEST,
                                Some("profile"),
                                oai_code::INVALID_REQUEST,
                            )),
                        )
                            .into_response(),
                    );
                }
                LoraValidation::UnknownAdapter => {
                    return Some(
                        (
                            StatusCode::BAD_REQUEST,
                            Json(json_openai_error(
                                format!(
                                    "unknown lora_adapter '{requested_lora}' for model '{display_model}'"
                                ),
                                oai_type::INVALID_REQUEST,
                                Some("lora_adapter"),
                                oai_code::UNKNOWN_LORA_ADAPTER,
                            )),
                        )
                            .into_response(),
                    );
                }
            }
        }
    }

    if generate_params_have_images(generate) {
        let image_supported = state
            .model_registry
            .get_model_info(model)
            .is_some_and(|info| info.info_extras.supports_vision_generation());
        if !native_generate_image_input_allowed(Some(generate), image_supported) {
            return Some(
                (
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        format!("Model '{display_model}' does not support image input"),
                        oai_type::INVALID_REQUEST,
                        Some("images"),
                        oai_code::UNSUPPORTED_FIELD,
                    )),
                )
                    .into_response(),
            );
        }
    }

    None
}

#[allow(clippy::too_many_arguments)]
async fn queue_mode_proxy(
    state: &AppState,
    work_publisher: Arc<dyn WorkDispatcher>,
    endpoint: &str,
    display_model: &str,
    model: &str,
    bundle: &str,
    engine: &str,
    gpu: &str,
    pool: &str,
    admission_pool: &str,
    body_bytes: &[u8],
    preparsed: Option<(Vec<rmpv::Value>, publisher::WorkParams)>,
    is_msgpack_in: bool,
    use_msgpack_out: bool,
    token_id: &str,
    content_length: i64,
    start: Instant,
    bundle_config_hash: &str,
    model_revision: Option<&str>,
    batch_target: Option<publisher::PublishTarget>,
    physical_lane: &PhysicalLane,
) -> Response {
    // Parse body once, extract items + params (avoids double parse)
    let (items, params) = match preparsed {
        Some(parsed) => parsed,
        None => match parse_queue_request(body_bytes, is_msgpack_in, endpoint) {
            Ok(parsed) => parsed,
            Err(error) => return queue_parse_error_response(endpoint, error),
        },
    };

    if items.is_empty() && endpoint != "score" && endpoint != "generate" {
        return endpoint_error_response(
            endpoint,
            StatusCode::BAD_REQUEST,
            err_code::INVALID_REQUEST,
            oai_type::INVALID_REQUEST,
            oai_code::INVALID_REQUEST,
            None,
            "No items found in request body",
        );
    }

    if let Err(message) = publisher::validate_queue_request_item_count(items.len()) {
        return endpoint_error_response(
            endpoint,
            StatusCode::BAD_REQUEST,
            err_code::INVALID_REQUEST,
            oai_type::INVALID_REQUEST,
            oai_code::INVALID_REQUEST,
            None,
            message,
        );
    }

    // Generate has its own publish + result-collection path
    // (streaming chunk aggregation instead of one-shot WorkResult fan-in).
    if endpoint == "generate" {
        // Take an Arc clone so the cancel-on-drop guard can outlive the
        // borrow checker without a 'static lifetime tangle.
        let Some(work_publisher_arc) = state.work_publisher.clone() else {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json_openai_error(
                    "Rust gateway is queue-only, but NATS JetStream is unavailable",
                    oai_type::SERVER_ERROR,
                    None,
                    oai_code::TRANSPORT_FAILURE,
                )),
            )
                .into_response();
        };
        if params.generate.as_ref().is_some_and(|params| params.stream) {
            return super::sse::build_sse_response(super::sse::SseParams {
                state,
                work_publisher: work_publisher_arc,
                physical_lane: physical_lane.clone(),
                model: display_model.to_string(),
                dispatch_model: model.to_string(),
                bundle: bundle.to_string(),
                engine: engine.to_string(),
                gpu: gpu.to_string(),
                pool: pool.to_string(),
                admission_pool: admission_pool.to_string(),
                bundle_config_hash: bundle_config_hash.to_string(),
                work_params: params,
                endpoint: super::sse::SseEndpoint::Generate,
            })
            .await;
        }
        return queue_mode_streaming_generate(
            state,
            work_publisher_arc,
            physical_lane,
            display_model,
            model,
            bundle,
            engine,
            gpu,
            pool,
            admission_pool,
            bundle_config_hash,
            model_revision,
            &params,
            use_msgpack_out,
            token_id,
            content_length,
            start,
        )
        .await;
    }

    let publish_start = Instant::now();
    let target = batch_target.unwrap_or_else(|| publisher::PublishTarget::Pool {
        pool: pool.to_string(),
        machine_profile: gpu.to_string(),
        bundle: bundle.to_string(),
        model: model.to_string(),
    });
    let direct_batch_fallback =
        matches!(target, publisher::PublishTarget::Worker { .. }) && endpoint != "generate";
    let (request_id, rx, durability) = match Arc::clone(&work_publisher)
        .publish_work(
            target,
            admission_pool,
            endpoint,
            model,
            engine,
            bundle_config_hash,
            items,
            &params,
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e, "failed to publish work");
            let lower = e.to_string().to_lowercase();
            if let Some(response) = dispatch_rejection_response(endpoint, &e) {
                if matches!(e, DispatchError::Backpressure(_)) {
                    telemetry::record_rejected_request(
                        state.demand_tracker.as_ref(),
                        physical_lane,
                        "backpressure",
                    );
                    state.demand_tracker.record(physical_lane);
                }
                return response;
            }
            if lower.contains("score request missing query item") {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json_detail(err_code::INVALID_REQUEST, e.to_string())),
                )
                    .into_response();
            }

            if lower.contains("no consumers") {
                telemetry::record_rejected_request(
                    state.demand_tracker.as_ref(),
                    physical_lane,
                    "no_consumers",
                );
                return build_provisioning_response_for_surface(
                    gpu,
                    bundle,
                    provisioning_surface_for_endpoint(endpoint),
                );
            }

            let mut response = (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json_detail(
                    err_code::QUEUE_UNAVAILABLE,
                    format!("Queue publish failed: {}", e),
                )),
            )
                .into_response();

            if let Some(retry_after) = record_publish_failure(state, physical_lane, &lower) {
                response.headers_mut().insert(
                    HeaderName::from_static("retry-after"),
                    HeaderValue::from_static(retry_after),
                );
            }

            response.headers_mut().insert(
                HeaderName::from_static("x-sie-version"),
                HeaderValue::from_static(GATEWAY_VERSION),
            );
            response.headers_mut().insert(
                HeaderName::from_static("x-sie-server-version"),
                HeaderValue::from_static(GATEWAY_VERSION),
            );
            return response;
        }
    };
    let mut abandonment_guard =
        PendingWorkAbandonGuard::new(Arc::clone(&work_publisher), request_id.clone());

    if direct_batch_fallback {
        if let Some(work_publisher_arc) = state.work_publisher.clone() {
            work_publisher_arc.spawn_batch_direct_fallback(
                request_id.clone(),
                batch_direct_fallback_delay(state.config.request_timeout),
            );
        }
    }
    let publish_elapsed = publish_start.elapsed();
    let mut durability_completion = monitor_dispatch_durability(
        Arc::clone(&state.demand_tracker),
        physical_lane.clone(),
        durability,
        Arc::clone(
            state
                .work_publisher
                .as_ref()
                .expect("queue-mode publisher checked before dispatch"),
        ),
        request_id.clone(),
        PendingDispatchKind::Result,
    );
    tracing::Span::current().record("sie.publish_ms", publish_elapsed.as_millis() as i64);

    // Wait for results (use configured request_timeout instead of hardcoded 300s).
    // Preserve fractional env values instead of truncating them through `as u64`.
    let timeout = Duration::from_secs_f64(state.config.request_timeout.max(0.001));
    let timeout_secs = timeout.as_secs_f64();
    let wait_start = Instant::now();
    let wait_deadline = tokio::time::Instant::now() + timeout;
    let mut result_rx = rx;
    let mut buffered_results = None;
    let mut durability_confirmed = false;
    let results = loop {
        tokio::select! {
            biased;
            completion = &mut durability_completion, if !durability_confirmed => {
                match completion {
                    Ok(Ok(())) => {
                        durability_confirmed = true;
                        if let Some(results) = buffered_results.take() {
                            break results;
                        }
                    }
                    Ok(Err(error)) => {
                        telemetry::record_queue_result_wait(
                            endpoint,
                            telemetry::QueueResultOutcome::DurabilityError,
                            wait_start.elapsed(),
                        );
                        return endpoint_error_response(
                            endpoint,
                            StatusCode::SERVICE_UNAVAILABLE,
                            err_code::QUEUE_UNAVAILABLE,
                            oai_type::SERVER_ERROR,
                            oai_code::TRANSPORT_FAILURE,
                            None,
                            format!("Queue durability confirmation failed: {error}"),
                        );
                    }
                    Err(_) => {
                        telemetry::record_queue_result_wait(
                            endpoint,
                            telemetry::QueueResultOutcome::DurabilityError,
                            wait_start.elapsed(),
                        );
                        return endpoint_error_response(
                            endpoint,
                            StatusCode::SERVICE_UNAVAILABLE,
                            err_code::QUEUE_UNAVAILABLE,
                            oai_type::SERVER_ERROR,
                            oai_code::TRANSPORT_FAILURE,
                            None,
                            "Queue durability monitor stopped before completion",
                        );
                    }
                }
            }
            result = &mut result_rx, if buffered_results.is_none() => {
                match result {
                    Ok(results) if durability_confirmed => break results,
                    Ok(results) => buffered_results = Some(results),
                    Err(_) => {
                        telemetry::record_queue_result_wait(
                            endpoint,
                            telemetry::QueueResultOutcome::ChannelClosed,
                            wait_start.elapsed(),
                        );
                        return (
                            StatusCode::GATEWAY_TIMEOUT,
                            Json(json_detail(
                                err_code::GATEWAY_TIMEOUT,
                                "Result channel closed",
                            )),
                        )
                            .into_response();
                    }
                }
            }
            _ = tokio::time::sleep_until(wait_deadline) => {
                telemetry::record_queue_result_wait(
                    endpoint,
                    telemetry::QueueResultOutcome::Timeout,
                    wait_start.elapsed(),
                );
                // The gateway accepted and published work, but no worker result
                // arrived before the request deadline. This is not the same signal
                // as worker-emitted MODEL_LOADING: under heavy load it usually
                // means queue/head-of-line delay or a lost NATS Core result, and
                // reporting it as MODEL_LOADING hides the real bottleneck.
                telemetry::record_rejected_request(
                    state.demand_tracker.as_ref(),
                    physical_lane,
                    "upstream_result_timeout",
                );
                return build_queue_result_timeout_response(model, timeout_secs);
            }
        }
    };
    abandonment_guard.defuse();
    let wait_elapsed = wait_start.elapsed();
    telemetry::record_queue_result_wait(
        endpoint,
        if results.iter().any(|result| result.success) {
            telemetry::QueueResultOutcome::Success
        } else {
            telemetry::QueueResultOutcome::WorkerError
        },
        wait_elapsed,
    );

    let elapsed = start.elapsed();
    let use_msgpack = use_msgpack_out;

    // A result that cannot fit the negotiated NATS payload ceiling means the
    // response is incomplete. Reject the whole request even when sibling items
    // succeeded; returning a shortened 200 would silently break positional and
    // id correspondence for multi-item encode calls.
    if let Some(oversized) = result_payload_too_large_error(&results) {
        let message = oversized
            .error
            .as_deref()
            .unwrap_or("Encoded worker result exceeds the transport limit");
        return build_result_payload_too_large_response(endpoint, message);
    }
    if result_transport_failure_error(&results).is_some() {
        return build_result_transport_failure_response(endpoint);
    }

    // Assemble response matching Python's envelope: {"model": "...", "items": [...]}
    let successful: Vec<&publisher::WorkResult> = results.iter().filter(|r| r.success).collect();
    let errors: Vec<&publisher::WorkResult> = results.iter().filter(|r| !r.success).collect();

    // Input validation is request-wide: returning a partial 200 after one
    // malformed item would hide the caller-fixable error and make native and
    // local execution disagree. Only a homogeneous INVALID_INPUT failure set
    // is translated; mixed worker failures retain the existing error handling.
    if let Some(message) = unanimous_worker_error_message(&errors, INVALID_INPUT_ERROR_CODE) {
        telemetry::record_rejected_request(
            state.demand_tracker.as_ref(),
            physical_lane,
            "invalid_input",
        );
        return build_invalid_input_response(&message);
    }

    if successful.is_empty() && !errors.is_empty() {
        if let Some(code) = unanimous_client_error_code(&errors) {
            let first_msg = errors
                .first()
                .and_then(|result| result.error.as_deref())
                .unwrap_or("Worker rejected invalid input");
            let openai_code = if code == "unsupported_field" {
                oai_code::UNSUPPORTED_FIELD
            } else {
                oai_code::INVALID_REQUEST
            };
            return endpoint_error_response(
                endpoint,
                StatusCode::BAD_REQUEST,
                err_code::INVALID_REQUEST,
                oai_type::INVALID_REQUEST,
                openai_code,
                None,
                first_msg,
            );
        }

        if errors
            .iter()
            .all(|r| r.error_code.as_deref() == Some(MODEL_LOAD_FAILED_ERROR_CODE))
        {
            return build_model_load_failed_response();
        }
        // Translate retryable worker error codes into the SDK-expected 503
        // contract. Without this every per-item failure surfaced as 500
        // ``all_items_failed`` and the SDK retry path never engaged. We
        // require a *unanimous* code across all failed items so we don't
        // mis-translate a mixed batch (e.g. one item OOM, another invalid
        // input) — only homogeneous, unambiguous cases get retried.
        if let Some(code) = unanimous_retryable_error_code(&errors) {
            let first_msg = errors
                .first()
                .and_then(|r| r.error.as_deref())
                .unwrap_or("Worker reported a retryable error");
            return build_retryable_error_response(code, first_msg);
        }
        if let Some((status, code)) = unanimous_terminal_client_error(&errors) {
            let first_msg = errors
                .first()
                .and_then(|r| r.error.as_deref())
                .unwrap_or("Worker rejected the request");
            return build_terminal_client_error_response(status, code, first_msg);
        }

        let error_details: Vec<serde_json::Value> = errors
            .iter()
            .map(|r| {
                let mut entry = json!({"item_index": r.item_index, "error": r.error});
                // Surface the per-item ``error_code`` for observability —
                // useful when a mixed batch lands here (some
                // RESOURCE_EXHAUSTED, some genuine inference errors).
                if let Some(code) = r.error_code.as_deref() {
                    entry["code"] = json!(code);
                }
                entry
            })
            .collect();
        // 2026-07-15 cold-start investigation: retryable codes (MODEL_LOADING
        // et al.) correctly 503 above, so reaching this 500 means a unanimous
        // NON-retryable worker code (or a mixed batch) that was previously
        // invisible in logs — name it so the next occurrence is diagnosable.
        // Log-only: worker-emitted codes are an unbounded set, so folding
        // them into the bounded reject-metric reason label would risk
        // cardinality; the label stays the static "all_items_failed".
        let unanimous_code = errors
            .first()
            .and_then(|r| r.error_code.as_deref())
            .filter(|first| {
                errors
                    .iter()
                    .all(|r| r.error_code.as_deref() == Some(first))
            });
        warn!(
            request_id = %request_id,
            model = %model,
            error_code = unanimous_code.unwrap_or("<mixed>"),
            failed_items = errors.len(),
            first_error = errors.first().and_then(|r| r.error.as_deref()).unwrap_or(""),
            "all items failed (500 all_items_failed)"
        );
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "all_items_failed", "details": error_details})),
        )
            .into_response();
    }

    let status: u16 = 200;

    let resp_body = build_queue_success_body(endpoint, model, &successful, use_msgpack);

    // §10 spine: emit at INFO so there is ONE greppable per-request success line
    // (carrying request_id) at the prod default RUST_LOG=info — the log anchor
    // that pairs with the linked gateway↔lane trace. Error/early-return paths
    // already log at info/warn; this closes the success path.
    info!(
        request_id = %request_id,
        endpoint = endpoint,
        model = %model,
        status = status,
        latency_ms = elapsed.as_millis(),
        "queue mode response"
    );

    // `REQUEST_COUNT` and `REQUEST_LATENCY` are now emitted by
    // `middleware::metrics::MetricsLayer` for *every* response on the
    // inference routes, including early returns (404, 413, 503, 504,
    // provisioning, ...). Do not re-emit them here or the success
    // path would be double-counted. The worker-registry per-request
    // bookkeeping stays — it is independent of Prometheus.
    state.registry.record_request("queue").await;

    emit_audit_log(AuditEntry {
        event: "proxy_request".to_string(),
        method: "POST".to_string(),
        endpoint: endpoint.to_string(),
        status,
        token_id: token_id.to_string(),
        model: model.to_string(),
        pool: pool.to_string(),
        gpu: gpu.to_string(),
        worker: format!("queue:{}", request_id),
        latency_ms: elapsed.as_millis() as u64,
        body_bytes: content_length,
    });

    let content_type = if use_msgpack {
        "application/x-msgpack"
    } else {
        "application/json"
    };

    let mut response = Response::builder()
        .status(StatusCode::from_u16(status).unwrap_or(StatusCode::OK))
        .body(Body::from(resp_body))
        .unwrap();
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static(content_type),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    insert_model_revision_header(
        response.headers_mut(),
        model_revision,
        bundle_config_hash,
        &successful,
    );
    insert_execution_identity_header(response.headers_mut(), &successful);
    insert_execution_binding_header(response.headers_mut(), &successful);
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-request-id"),
        HeaderValue::from_str(&request_id).unwrap_or_else(|_| HeaderValue::from_static("")),
    );
    insert_duration_header(
        response.headers_mut(),
        "x-queue-publish-time",
        publish_elapsed,
    );
    insert_duration_header(response.headers_mut(), "x-queue-wait-time", wait_elapsed);
    insert_queue_worker_timing_headers(response.headers_mut(), &successful);
    let worker_tag = format!("queue:{}", request_id);
    if let Ok(val) = HeaderValue::from_str(&worker_tag) {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-sie-worker"), val);
    }
    response
}

/// RAII guard for non-streaming queue work. A normal result defuses it; timeout
/// and result-channel failure call [`Self::abandon`] explicitly, while an HTTP
/// task drop reaches the same path through [`Drop`]. Collector removal is
/// synchronous so late results and the direct-fallback timer lose the race
/// immediately. The Core NATS tombstone and exact-key offload cleanup run in a
/// detached task so a storage delete cannot delay the timeout response.
struct PendingWorkAbandonGuard {
    publisher: Arc<dyn WorkDispatcher>,
    request_id: String,
    defused: bool,
}

impl PendingWorkAbandonGuard {
    fn new(publisher: Arc<dyn WorkDispatcher>, request_id: String) -> Self {
        Self {
            publisher,
            request_id,
            defused: false,
        }
    }

    fn defuse(&mut self) {
        self.defused = true;
    }

    fn abandon(&mut self) {
        if self.defused {
            return;
        }
        self.defused = true;
        if !self.publisher.begin_work_abandonment(&self.request_id) {
            return;
        }

        let publisher = Arc::clone(&self.publisher);
        let request_id = self.request_id.clone();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    publisher.finish_work_abandonment(&request_id).await;
                });
            }
            Err(_) => {
                // Collector/result-chunk memory is already released. Any
                // tracked offload remains eligible for the ordinary orphan
                // reconcile when a runtime is available again.
                debug!(
                    request_id = %request_id,
                    "runtime unavailable while finishing abandoned work"
                );
            }
        }
    }
}

impl Drop for PendingWorkAbandonGuard {
    fn drop(&mut self) {
        self.abandon();
    }
}

/// RAII guard that fires a streaming cancel signal when dropped without
/// being defused. The HTTP handler defuses it on every normal exit (terminal
/// chunk received, worker error surfaced, timeout fired) — leaving only
/// the client-disconnect / axum task-abort path to trigger the cancel
/// (axum drops the handler task when the connection closes).
pub(crate) struct StreamCancelGuard {
    publisher: Arc<dyn WorkDispatcher>,
    request_id: String,
    defused: bool,
}

impl StreamCancelGuard {
    pub(crate) fn new(publisher: Arc<dyn WorkDispatcher>, request_id: String) -> Self {
        Self {
            publisher,
            request_id,
            defused: false,
        }
    }

    pub(crate) fn defuse(mut self) {
        self.defused = true;
    }
}

impl Drop for StreamCancelGuard {
    fn drop(&mut self) {
        if self.defused {
            return;
        }
        let observed_first_chunk = self.publisher.stream_observed_first_chunk(&self.request_id);
        let reason = if observed_first_chunk {
            telemetry::GenerationEventReason::MidStream
        } else {
            telemetry::GenerationEventReason::BeforeFirstChunk
        };
        telemetry::record_generation_event(
            telemetry::GenerationEvent::Cancellation,
            reason,
            telemetry::GenerationEventOutcome::Cancelled,
        );
        // Drop is synchronous; spawn a detached task to emit the cancel
        // signal and clean up the collector. Both are best-effort.
        //
        // ``tokio::spawn`` panics if no runtime is set for the current
        // thread, which can happen when guards are dropped during
        // graceful shutdown (after the runtime has been torn down) or
        // when a guard escapes into a thread without a runtime handle.
        // Use ``Handle::try_current`` so the panic becomes a no-op +
        // synchronous cleanup instead of a process-killing unwind.
        let publisher = Arc::clone(&self.publisher);
        let request_id = self.request_id.clone();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    publisher.publish_cancel(&request_id).await;
                    publisher.drop_pending_stream(&request_id);
                });
            }
            Err(_) => {
                // No runtime — at minimum drop the collector entry so
                // the pending_streams map does not leak across reload.
                publisher.drop_pending_stream(&request_id);
            }
        }
    }
}

/// Streaming generate path: publish via streaming + await aggregated outcome.
///
/// Unlike the batch path, this function does not fan out items or wait on
/// a ``Vec<WorkResult>``: it installs a per-request streaming collector,
/// publishes one work envelope, and awaits a single ``StreamOutcome``
/// fired by the inbox handler when a terminal chunk arrives.
///
/// Phase E additions: a RAII [`StreamCancelGuard`] is installed for the
/// duration of the wait. If the future is dropped (axum aborts the task
/// when the HTTP client disconnects) the guard fires a cancel signal on
/// ``cancel.{router_id}.{request_id}`` and records the cancellation
/// metric. The guard is defused once a terminal outcome is received
/// (or an error is being returned) so a normal completion path does
/// not produce a spurious cancel.
/// Build an error response in the envelope appropriate for ``endpoint``.
///
/// Endpoints that use the OpenAI envelope (``generate``, ``chat``) get
/// the OpenAI-compatible ``{error:{message,type,param,code}}`` shape; everything
/// else keeps the legacy ``{detail:{code,message}}`` shape so existing
/// SDK error-parsing paths are unaffected. The
/// :func:`json_detail`-shaped sites that survive this slice live behind
/// :func:`endpoint_error_response`.
fn endpoint_uses_openai_envelope(endpoint: &str) -> bool {
    matches!(endpoint, "generate" | "chat")
}

fn endpoint_error_response(
    endpoint: &str,
    status: StatusCode,
    legacy_code: &'static str,
    openai_type: &'static str,
    openai_code: &'static str,
    param: Option<&str>,
    message: impl Into<String>,
) -> Response {
    let msg = message.into();
    if endpoint_uses_openai_envelope(endpoint) {
        (
            status,
            Json(json_openai_error(msg, openai_type, param, openai_code)),
        )
            .into_response()
    } else {
        (status, Json(json_detail(legacy_code, msg))).into_response()
    }
}

fn dispatch_rejection_response(endpoint: &str, error: &DispatchError) -> Option<Response> {
    let mut response = match error {
        DispatchError::PayloadTooLarge(error) => endpoint_error_response(
            endpoint,
            StatusCode::PAYLOAD_TOO_LARGE,
            err_code::PAYLOAD_TOO_LARGE,
            oai_type::INVALID_REQUEST,
            oai_code::INVALID_REQUEST,
            None,
            error.message(),
        ),
        DispatchError::InvalidInput(error) => endpoint_error_response(
            endpoint,
            StatusCode::BAD_REQUEST,
            err_code::INVALID_REQUEST,
            oai_type::INVALID_REQUEST,
            oai_code::INVALID_REQUEST,
            None,
            error.message(),
        ),
        DispatchError::Backpressure(error) => endpoint_error_response(
            endpoint,
            StatusCode::SERVICE_UNAVAILABLE,
            err_code::QUEUE_UNAVAILABLE,
            oai_type::SERVER_ERROR,
            oai_code::TRANSPORT_FAILURE,
            None,
            error.message(),
        ),
        DispatchError::Other(_) => return None,
    };
    if matches!(error, DispatchError::Backpressure(_)) {
        response.headers_mut().insert(
            HeaderName::from_static("retry-after"),
            HeaderValue::from_static(BACKPRESSURE_RETRY_AFTER),
        );
    }
    Some(response)
}

/// Result of a streaming-generate driver run, handed back to the
/// endpoint-specific caller.
///
/// Both ``/v1/generate/{model}`` and ``/v1/chat/completions`` go through
/// the same publish → timeout-supervised wait → cancel-guarded outcome
/// pipeline. They only differ in:
///
/// * the success-body envelope (SIE-native vs OpenAI ``chat.completion``)
/// * the audit-log ``endpoint`` field
///
/// — neither of which affects the streaming machinery itself. The
/// caller pattern-matches on this and produces the final
/// :class:`axum::response::Response`.
pub(crate) struct StreamingDriverOk {
    pub outcome: crate::queue::streaming::StreamOutcome,
    pub request_id: String,
    pub publish_elapsed: Duration,
    pub wait_elapsed: Duration,
}

/// Failure modes for :func:`run_streaming_generate`. Each carries the
/// metadata the caller needs to build an HTTP error in its own envelope
/// shape (SIE-native ``json_detail`` or OpenAI ``json_openai_error``).
pub(crate) enum StreamingDriverErr {
    /// JetStream publish failed before any work item left the gateway.
    /// ``error`` is the lowercased publisher message; ``retry_after`` is
    /// the recommended ``Retry-After`` header value when applicable
    /// (``no_consumers`` and ``backpressure`` cases).
    PublishFailed {
        message: String,
        retry_after: Option<&'static str>,
    },
    /// Initial publish submission returned, but the transport later rejected
    /// or abandoned its durable-acceptance acknowledgement.
    DurabilityFailed { message: String },
    /// The result channel was dropped before any chunk arrived (worker
    /// reset, gateway shutting down, …). Maps to 504 Gateway Timeout.
    ResultChannelClosed,
    /// One of the three streaming generation timeouts fired. ``kind`` is
    /// ``"first_chunk"`` | ``"inter_chunk"`` | ``"overall"``.
    Timeout { kind: &'static str },
    /// Worker emitted a terminal chunk with ``error`` populated. The
    /// caller chooses the wire status/code mapping; message, parameter,
    /// and retry metadata are bounded at the worker trust boundary.
    WorkerError {
        code: String,
        message: String,
        param: Option<String>,
        retry_after_s: Option<u16>,
        request_id: String,
        attempt_id: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FirstChunkDeadlineAction {
    RepublishToPool,
    WaitForOverall,
    SurfaceTimeout,
}

pub(crate) fn first_chunk_deadline_action(
    supports_pool_republish: bool,
    was_direct_dispatched: bool,
    republish_already_attempted: bool,
    single_consumer_lane_at_dispatch: bool,
) -> FirstChunkDeadlineAction {
    if !supports_pool_republish {
        return FirstChunkDeadlineAction::WaitForOverall;
    }
    if was_direct_dispatched && !republish_already_attempted && !single_consumer_lane_at_dispatch {
        return FirstChunkDeadlineAction::RepublishToPool;
    }
    if was_direct_dispatched && !republish_already_attempted && single_consumer_lane_at_dispatch {
        return FirstChunkDeadlineAction::WaitForOverall;
    }
    FirstChunkDeadlineAction::SurfaceTimeout
}

/// Streaming driver: publish a generate work item, wait
/// for a terminal aggregated :class:`StreamOutcome` under the three-tier
/// timeout taxonomy, and propagate cancel signals on client disconnect.
///
/// Endpoint-specific concerns — success-body shape, audit-log
/// labelling, error-envelope choice — live in the caller.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_streaming_generate(
    state: &AppState,
    work_publisher: Arc<dyn WorkDispatcher>,
    physical_lane: &PhysicalLane,
    // DISPLAY id (the requested model) — drives metric labels + the cancel
    // guard. ``dispatch_model`` below is the id actually published to NATS /
    // the work item (the grammar ``:no-spec`` variant when routing fired);
    // they are equal for non-routed requests. See ``route_grammar_to_profile``.
    model: &str,
    dispatch_model: &str,
    bundle: &str,
    engine: &str,
    gpu: &str,
    pool: &str,
    admission_pool: &str,
    bundle_config_hash: &str,
    params: &publisher::WorkParams,
) -> Result<StreamingDriverOk, StreamingDriverErr> {
    let publish_start = Instant::now();
    // Resolve the routing key, build the pressure-aware routing snapshot,
    // pick a worker. Falls back to pool publish when:
    // - generate params are absent (caller will surface a 400).
    // - the resolved key has no hash (no routing_key / prompt_cache_key
    //   / prompt — should be rare; pool round-robin is the right default).
    // - the ring is empty (no eligible workers loaded for this model/pool).
    //
    let resolved_key = match params.generate.as_ref() {
        Some(g) => crate::routing::key::resolve_from_generate(g),
        None => crate::routing::key::RoutingKeyResolved {
            hash: None,
            source: crate::routing::key::KeySource::None,
            #[cfg(feature = "raw-routing-logs")]
            raw_for_debug: None,
        },
    };
    // Short-circuit when the key resolution yielded nothing to hash
    // with. There's no point building the ring snapshot or invoking
    // HRW: the pick would always be `None` and the fallback reason
    // would be ambiguous. Label this case `no_key` so operators can
    // distinguish it from capacity/health-driven fallbacks. We also
    // skip the gauge update here — the ring isn't consulted, so
    // recording a size for it would be misleading.
    let (target, pool_fallback_lane_worker_count) = if resolved_key.hash.is_none() {
        (
            publisher::PublishTarget::Pool {
                pool: pool.to_string(),
                machine_profile: gpu.to_string(),
                bundle: bundle.to_string(),
                model: dispatch_model.to_string(),
            },
            0,
        )
    } else {
        let admitted_worker_names = state
            .pool_manager
            .admitted_worker_names_for_capped_lane(admission_pool, gpu, bundle)
            .await;
        let fallback_lane_worker_count = state.registry.pool_fallback_lane_worker_count(
            pool,
            gpu,
            bundle,
            bundle_config_hash,
            admitted_worker_names.as_ref(),
        );
        let ring = state.registry.ring_snapshot_for_admitted(
            dispatch_model,
            pool,
            gpu,
            bundle,
            bundle_config_hash,
            admitted_worker_names.as_ref(),
        );
        let picked = crate::routing::pick_worker(&ring, &resolved_key);
        match picked {
            Some(worker_id) => {
                tracing::debug!(
                    model = %model,
                    pool = %pool,
                    worker_id = %worker_id,
                    key = %crate::routing::fmt_key_hash(resolved_key.hash.unwrap_or(0)),
                    source = resolved_key.source.as_label(),
                    "pressure-aware worker pick"
                );
                (
                    publisher::PublishTarget::Worker {
                        pool: pool.to_string(),
                        machine_profile: gpu.to_string(),
                        bundle: bundle.to_string(),
                        model: dispatch_model.to_string(),
                        worker_id: worker_id.to_string(),
                    },
                    fallback_lane_worker_count,
                )
            }
            None => (
                publisher::PublishTarget::Pool {
                    pool: pool.to_string(),
                    machine_profile: gpu.to_string(),
                    bundle: bundle.to_string(),
                    model: dispatch_model.to_string(),
                },
                fallback_lane_worker_count,
            ),
        }
    };
    // Capture this before the move into `publish_generate_streaming`.
    let was_direct_dispatched = matches!(target, publisher::PublishTarget::Worker { .. });
    let (request_id, rx, activity, durability) = match work_publisher
        .publish_generate_streaming(
            target,
            model,
            engine,
            bundle_config_hash,
            params,
            admission_pool,
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!(error = %e, "failed to publish generate work");
            let lower = e.to_lowercase();
            let retry_after = if lower.contains("no consumers") {
                telemetry::record_rejected_request(
                    state.demand_tracker.as_ref(),
                    physical_lane,
                    "no_consumers",
                );
                Some(PROVISIONING_RETRY_AFTER)
            } else {
                // backpressure (records demand for KEDA) / generic failure. #1568
                record_publish_failure(state, physical_lane, &lower)
            };
            return Err(StreamingDriverErr::PublishFailed {
                message: e,
                retry_after,
            });
        }
    };
    let publish_elapsed = publish_start.elapsed();
    let mut durability_completion = monitor_dispatch_durability(
        Arc::clone(&state.demand_tracker),
        physical_lane.clone(),
        durability,
        Arc::clone(&work_publisher),
        request_id.clone(),
        PendingDispatchKind::Stream,
    );

    // Install the cancel-on-drop guard so an axum task abort (HTTP
    // client disconnect) fires a cancel signal to the worker. The
    // guard is defused on every normal completion path below.
    let cancel_guard = StreamCancelGuard::new(Arc::clone(&work_publisher), request_id.clone());

    // §4.4.3 three-tier timeout taxonomy. Defaults:
    //   first_chunk : 30s   (cold-start, grammar compile, queue depth)
    //   inter_chunk : 10s   (gap between chunks after streaming starts)
    //   overall     : max_new_tokens/10 + 30, capped at 5 min
    // Env overrides keep tuning ops-friendly. Per ADR-0003 the
    // profile/runtime overall_timeout_s is the sole authority for
    // generation; the legacy `SIE_GATEWAY_REQUEST_TIMEOUT` ceiling is
    // *not* applied here (it would collapse a 300s profile overall to
    // the default 30s and make the first-chunk policy unreachable).
    // That ceiling continues to govern encode/score/extract.
    let max_new_tokens = params
        .generate
        .as_ref()
        .map(|g| g.max_new_tokens)
        .unwrap_or(512);
    let timeout_config = generation_timeout_config(state, dispatch_model, params, max_new_tokens);
    let first_chunk_timeout = timeout_config.first_chunk;
    let inter_chunk_timeout = timeout_config.inter_chunk;
    let effective_overall = timeout_config.overall;

    let wait_start = Instant::now();
    // `activity` was returned alongside `request_id` above — the
    // collector's `Arc<Notify>` is cloned before the collector moves
    // into `pending_streams`, eliminating the lookup race that the
    // previous `.expect()` covered.

    let publish_tokio_instant = tokio::time::Instant::now();
    let mut first_chunk_deadline = publish_tokio_instant + first_chunk_timeout;
    let overall_deadline = publish_tokio_instant + effective_overall;

    // `was_direct_dispatched` was captured above (before the
    // `target` move into `publish_generate_streaming`). The
    // `republished_for_first_chunk` flag is a local idempotency guard
    // for the timeout-driven republish — the NAK-driven republish has
    // its own guard inside `WorkPublisher::republish_to_pool` (the
    // `republished` field on `StreamCollector`) so the two paths
    // cannot double-publish.
    let mut republished_for_first_chunk = false;
    let supports_first_chunk_pool_republish = work_publisher.supports_first_chunk_pool_republish();
    // Capture the exact pool-fallback lane size at dispatch time so
    // the H9 single-worker suppression below is scoped to
    // `(pool,machine_profile,bundle)` and still counts healthy cold
    // workers that can lazy-load after a pool republish. The old
    // JetStream pool-wide consumer count could be inflated by another
    // bundle or machine profile in the same logical pool, incorrectly
    // enabling a first-chunk fallback that had no alternate worker in
    // this lane.
    let single_consumer_lane_at_dispatch = crate::routing::suppress_first_chunk_republish_for_lane(
        was_direct_dispatched,
        pool_fallback_lane_worker_count,
    );

    // Drive the wait loop. We hold a single pinned receiver across
    // iterations and re-arm the inter-chunk timer on every chunk
    // arrival (signalled via ``activity``). The first-chunk and overall
    // deadlines are fixed at publish time; the inter-chunk deadline is
    // recomputed every iteration from ``last_chunk_at``.
    tokio::pin!(rx);
    let mut durability_confirmed = false;
    let mut buffered_outcome = None;
    let outcome_or_error: Result<crate::queue::streaming::StreamOutcome, &'static str> = loop {
        // A terminal outcome can race ahead of the initial dispatch ACK. Once
        // that happens the generation itself is complete: first/inter-chunk
        // progress deadlines are no longer meaningful, and the only valid
        // wait is durability confirmation bounded by the overall deadline.
        let terminal_outcome_buffered = buffered_outcome.is_some();
        let (first_at, last_at) = if terminal_outcome_buffered {
            (None, None)
        } else {
            work_publisher
                .stream_chunk_timing(&request_id)
                .unwrap_or((None, None))
        };

        // Cheap early-fire: if a deadline has already passed at this
        // iteration's start, surface it immediately. This catches the
        // case where the previous iteration was woken by activity but
        // a competing deadline had already fired in real time.
        let now = tokio::time::Instant::now();
        if now >= overall_deadline {
            break Err("overall");
        }
        if !terminal_outcome_buffered && first_at.is_none() && now >= first_chunk_deadline {
            // One-shot republish to pool if we direct-dispatched
            // and haven't already republished. Only extend the deadline
            // when a republish actually happened — `Ok(false)` means the
            // NAK path already republished (StreamCollector.republished
            // is true), in which case waiting another full
            // first_chunk_timeout would double the user-visible latency
            // for no benefit (no new work was published).
            //
            // Single-consumer-pool guard: in pools with exactly one
            // worker the H9 republish lands on the same worker that's
            // already running the original attempt, and the
            // ``publish_cancel`` we'd fire first arrives as a cancel-
            // tombstone on the pool-republished message — the worker
            // then refuses to decode with ``request cancelled before
            // this worker registered; pool-republished attempt is
            // authoritative``. There is no alternative worker to
            // failover *to*, so the fallback can only ever harm
            // throughput here. Extend the deadline to ``overall`` so
            // the original attempt gets the legitimate end-to-end
            // budget; if the worker really is dead the
            // overall_timeout path will surface that.
            let deadline_action = first_chunk_deadline_action(
                supports_first_chunk_pool_republish,
                was_direct_dispatched,
                republished_for_first_chunk,
                single_consumer_lane_at_dispatch,
            );
            if deadline_action == FirstChunkDeadlineAction::RepublishToPool {
                republished_for_first_chunk = true;
                // At-least-once-execution hazard: the original
                // direct-dispatched worker may simply be SLOW (cold
                // start, queue depth) rather than dead. If we republish
                // to the pool without first telling the original to stop,
                // both the original AND the pool worker can run the same
                // generation to completion — double execution and, for
                // metered models, double billing, plus duplicate chunks
                // racing into the same collector. Publish a cancel for
                // the original attempt FIRST so it stops driving the
                // adapter, THEN republish. Cancel is keyed on
                // `cancel.{router_id}.{request_id}`; the pool worker has
                // not started yet, so only the original observes it.
                work_publisher.publish_cancel(&request_id).await;
                match work_publisher
                    .republish_to_pool(&request_id, "first_chunk_timeout")
                    .await
                {
                    Ok(true) => {
                        first_chunk_deadline = tokio::time::Instant::now() + first_chunk_timeout;
                        continue;
                    }
                    Ok(false) => {
                        // NAK path beat us to the republish; there's no
                        // additional work in flight to wait for.
                        tracing::debug!(
                            request_id = %request_id,
                            "first_chunk_timeout — republish already performed by NAK path; surfacing timeout"
                        );
                        break Err("first_chunk");
                    }
                    Err(e) => {
                        telemetry::record_rejected_request(
                            state.demand_tracker.as_ref(),
                            physical_lane,
                            "publish_ack_failed",
                        );
                        state.demand_tracker.record(physical_lane);
                        tracing::warn!(
                            request_id = %request_id,
                            error = %e,
                            "first_chunk_timeout republish to pool failed"
                        );
                        break Err("first_chunk");
                    }
                }
            }
            if deadline_action == FirstChunkDeadlineAction::WaitForOverall {
                // A single-worker NATS lane has no alternate worker, and a
                // dispatcher may declare that it has no safe retained
                // payload / pool fallback at all. In either case, keep the
                // original attempt alive for its governed overall budget.
                // No cancel is sent at the first-chunk boundary.
                republished_for_first_chunk = true;
                first_chunk_deadline = overall_deadline;
                tracing::debug!(
                        request_id = %request_id,
                        pool = %pool,
                        machine_profile = %gpu,
                        bundle = %bundle,
                        supports_pool_republish = supports_first_chunk_pool_republish,
                        "first_chunk_timeout - pool republish unavailable; continuing on overall_timeout"
                );
                continue;
            }
            break Err("first_chunk");
        }
        if !terminal_outcome_buffered {
            if let Some(la) = last_at {
                if la.elapsed() >= inter_chunk_timeout {
                    break Err("inter_chunk");
                }
            }
        }

        let inter_chunk_deadline = if terminal_outcome_buffered {
            None
        } else {
            last_at.map(|la| {
                let elapsed = la.elapsed();
                if elapsed >= inter_chunk_timeout {
                    now
                } else {
                    now + (inter_chunk_timeout - elapsed)
                }
            })
        };

        tokio::select! {
            biased;
            completion = &mut durability_completion, if !durability_confirmed => {
                match completion {
                    Ok(Ok(())) => {
                        durability_confirmed = true;
                        if let Some(outcome) = buffered_outcome.take() {
                            break Ok(outcome);
                        }
                    }
                    Ok(Err(error)) => {
                        cancel_guard.defuse();
                        telemetry::record_queue_result_wait(
                            "generate",
                            telemetry::QueueResultOutcome::DurabilityError,
                            wait_start.elapsed(),
                        );
                        return Err(StreamingDriverErr::DurabilityFailed { message: error });
                    }
                    Err(_) => {
                        cancel_guard.defuse();
                        telemetry::record_queue_result_wait(
                            "generate",
                            telemetry::QueueResultOutcome::DurabilityError,
                            wait_start.elapsed(),
                        );
                        return Err(StreamingDriverErr::DurabilityFailed {
                            message: "dispatch durability monitor stopped before completion"
                                .to_string(),
                        });
                    }
                }
            }
            res = &mut rx, if buffered_outcome.is_none() => {
                match res {
                    Ok(outcome) if durability_confirmed => break Ok(outcome),
                    Ok(outcome) => {
                        buffered_outcome = Some(outcome);
                    }
                    Err(_) => break Err("result_channel_closed"),
                }
            },
            _ = tokio::time::sleep_until(overall_deadline) => {
                break Err("overall");
            }
            _ = tokio::time::sleep_until(first_chunk_deadline),
                if !terminal_outcome_buffered && first_at.is_none() => {
                // Deferred to next loop iteration where the
                // republish-once-on-first-chunk-timeout branch above
                // owns the decision. Breaking here would skip the
                // republish; falling through with `None` re-enters the
                // loop, the early-fire block runs, and the republish
                // decision happens with the freshest state.
            }
            _ = tokio::time::sleep_until(inter_chunk_deadline.unwrap_or(overall_deadline)),
                if !terminal_outcome_buffered && first_at.is_some() => {
                break Err("inter_chunk");
            }
            _ = activity.notified(), if !terminal_outcome_buffered => {},
        }
    };

    let outcome = match outcome_or_error {
        Ok(o) => {
            cancel_guard.defuse();
            o
        }
        Err("result_channel_closed") => {
            cancel_guard.defuse();
            telemetry::record_queue_result_wait(
                "generate",
                telemetry::QueueResultOutcome::ChannelClosed,
                wait_start.elapsed(),
            );
            work_publisher.drop_pending_stream(&request_id);
            return Err(StreamingDriverErr::ResultChannelClosed);
        }
        Err(kind) => {
            // One of the three generation timeouts fired. This is not a
            // client-disconnect cancellation, so defuse the Drop guard and
            // send the worker cancel explicitly.
            cancel_guard.defuse();
            telemetry::record_queue_result_wait(
                "generate",
                telemetry::QueueResultOutcome::Timeout,
                wait_start.elapsed(),
            );
            work_publisher.publish_cancel(&request_id).await;
            work_publisher.drop_pending_stream(&request_id);
            return Err(StreamingDriverErr::Timeout { kind });
        }
    };
    let wait_elapsed = wait_start.elapsed();
    telemetry::record_queue_result_wait(
        "generate",
        if outcome.error.is_some() {
            telemetry::QueueResultOutcome::WorkerError
        } else {
            telemetry::QueueResultOutcome::Success
        },
        wait_elapsed,
    );

    // If the worker emitted an error terminal, surface it as a typed
    // failure. The caller chooses the HTTP status / wire envelope.
    if let Some(err) = outcome.error.as_ref() {
        return Err(StreamingDriverErr::WorkerError {
            code: err.client_safe_code().to_string(),
            message: err.client_safe_message().to_string(),
            param: err.client_safe_param().map(str::to_string),
            retry_after_s: err.validated_retry_after_s(),
            request_id: request_id.clone(),
            attempt_id: outcome.attempt_id.clone(),
        });
    }

    Ok(StreamingDriverOk {
        outcome,
        request_id,
        publish_elapsed,
        wait_elapsed,
    })
}

/// HTTP status mapping for a worker-emitted terminal error code.
///
/// Shared between ``/v1/generate/{model}`` and ``/v1/chat/completions``
/// so the wire status is identical regardless of which entrypoint
/// surfaced the failure.
pub(crate) fn worker_error_http_status(code: &str) -> StatusCode {
    match code {
        "invalid_request" | "unsupported_field" => StatusCode::BAD_REQUEST,
        "context_exceeded" => StatusCode::BAD_REQUEST,
        PAYLOAD_TOO_LARGE_ERROR_CODE => StatusCode::PAYLOAD_TOO_LARGE,
        RESOURCE_EXHAUSTED_ERROR_CODE | MODEL_LOADING_ERROR_CODE | LORA_LOADING_ERROR_CODE => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        MODEL_LOAD_FAILED_ERROR_CODE => StatusCode::BAD_GATEWAY,
        "transport_failure" => StatusCode::SERVICE_UNAVAILABLE,
        "cancelled" => StatusCode::REQUEST_TIMEOUT,
        "rate_limit_exceeded" => StatusCode::TOO_MANY_REQUESTS,
        COLD_START_RATE_LIMITED_ERROR_CODE => StatusCode::TOO_MANY_REQUESTS,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// OpenAI ``error.type`` mapping for a worker-emitted terminal error
/// code. Pairs with :func:`worker_error_http_status`.
pub(crate) fn worker_error_openai_type(code: &str) -> &'static str {
    match code {
        "invalid_request" | "unsupported_field" | PAYLOAD_TOO_LARGE_ERROR_CODE => {
            oai_type::INVALID_REQUEST
        }
        "context_exceeded" => oai_type::CONTEXT_LENGTH_EXCEEDED,
        "rate_limit_exceeded" => oai_type::RATE_LIMIT,
        COLD_START_RATE_LIMITED_ERROR_CODE => oai_type::RATE_LIMIT,
        _ => oai_type::SERVER_ERROR,
    }
}

/// Retry semantics for a transient worker-emitted error code: the
/// `Retry-After` header value (seconds, as a string) paired with the
/// canonical SIE error code, or `None` when the code is not a known
/// retryable signal.
///
/// Single source of truth shared by the streaming (OpenAI-envelope) and unary
/// (SIE-native [`build_retryable_error_response`]) paths — the two surfaces
/// render the error body differently but agree on the retry semantics. Pairs
/// with [`worker_error_http_status`] / [`worker_error_openai_type`].
fn worker_error_retry_after(
    code: &str,
    resource_exhausted_retry_after_s: Option<u16>,
) -> Option<(String, &'static str)> {
    match code {
        RESOURCE_EXHAUSTED_ERROR_CODE => Some((
            resource_exhausted_retry_after_s
                .filter(|value| (1..=60).contains(value))
                .map(|value| value.to_string())
                .unwrap_or_else(|| RESOURCE_EXHAUSTED_RETRY_AFTER.to_string()),
            RESOURCE_EXHAUSTED_ERROR_CODE,
        )),
        MODEL_LOADING_ERROR_CODE => Some((
            MODEL_LOADING_RETRY_AFTER.to_string(),
            MODEL_LOADING_ERROR_CODE,
        )),
        LORA_LOADING_ERROR_CODE => Some((
            LORA_LOADING_RETRY_AFTER.to_string(),
            LORA_LOADING_ERROR_CODE,
        )),
        _ => None,
    }
}

/// Build the OpenAI-envelope error response for one of the
/// :class:`StreamingDriverErr` variants. Used by both
/// ``proxy_generate`` (via :func:`queue_mode_streaming_generate`) and
/// ``proxy_chat``.
///
/// Metrics for these failure paths are emitted inside
/// :func:`run_streaming_generate` before the typed error returns —
/// this helper is purely body+headers composition.
fn build_streaming_error_response(err: &StreamingDriverErr) -> Response {
    match err {
        StreamingDriverErr::PublishFailed {
            message,
            retry_after,
        } => {
            let is_no_consumers = message.to_lowercase().contains("no consumers");
            let openai_code = if is_no_consumers {
                oai_code::PROVISIONING
            } else {
                oai_code::TRANSPORT_FAILURE
            };
            let mut resp = (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json_openai_error(
                    format!("Queue publish failed: {message}"),
                    oai_type::SERVER_ERROR,
                    None,
                    openai_code,
                )),
            )
                .into_response();
            if let Some(ra) = retry_after {
                resp.headers_mut().insert(
                    HeaderName::from_static("retry-after"),
                    HeaderValue::from_static(ra),
                );
            }
            if is_no_consumers {
                resp.headers_mut().insert(
                    HeaderName::from_static("x-sie-error-code"),
                    HeaderValue::from_static(err_code::PROVISIONING),
                );
            }
            resp.headers_mut().insert(
                HeaderName::from_static("x-sie-version"),
                HeaderValue::from_static(GATEWAY_VERSION),
            );
            resp.headers_mut().insert(
                HeaderName::from_static("x-sie-server-version"),
                HeaderValue::from_static(GATEWAY_VERSION),
            );
            resp
        }
        StreamingDriverErr::DurabilityFailed { message } => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json_openai_error(
                format!("Queue durability confirmation failed: {message}"),
                oai_type::SERVER_ERROR,
                None,
                oai_code::TRANSPORT_FAILURE,
            )),
        )
            .into_response(),
        StreamingDriverErr::ResultChannelClosed => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(json_openai_error(
                "Result channel closed",
                oai_type::SERVER_ERROR,
                None,
                oai_code::TRANSPORT_FAILURE,
            )),
        )
            .into_response(),
        StreamingDriverErr::Timeout { kind } => {
            // Inter-chunk timeout returns 502 (partial response is
            // corrupt; SDK cannot retry); first-chunk and overall
            // return 504 (gateway/upstream timing).
            let status = if *kind == "inter_chunk" {
                StatusCode::BAD_GATEWAY
            } else {
                StatusCode::GATEWAY_TIMEOUT
            };
            let code = match *kind {
                "first_chunk" => oai_code::FIRST_CHUNK_TIMEOUT,
                "inter_chunk" => oai_code::INTER_CHUNK_TIMEOUT,
                _ => oai_code::OVERALL_TIMEOUT,
            };
            (
                status,
                Json(json_openai_error(
                    format!("Generation aborted: {kind} timeout"),
                    oai_type::SERVER_ERROR,
                    None,
                    code,
                )),
            )
                .into_response()
        }
        StreamingDriverErr::WorkerError {
            code,
            message,
            param,
            retry_after_s,
            request_id,
            attempt_id,
        } => {
            if code == MODEL_LOAD_FAILED_ERROR_CODE {
                let mut resp = build_model_load_failed_response();
                resp.headers_mut().insert(
                    HeaderName::from_static("x-sie-request-id"),
                    HeaderValue::from_str(request_id)
                        .unwrap_or_else(|_| HeaderValue::from_static("")),
                );
                return resp;
            }
            let stable_code = client_safe_worker_error_code(code);
            // OpenAI represents oversized request payloads as the canonical
            // invalid-request discriminator while the SIE code remains on the
            // response header. This is a compatibility rendering of an
            // already-classified safe code, not a second worker allowlist.
            let openai_code = match stable_code {
                PAYLOAD_TOO_LARGE_ERROR_CODE => oai_code::INVALID_REQUEST,
                _ => stable_code,
            };
            let status = worker_error_http_status(stable_code);
            let err_type = worker_error_openai_type(stable_code);
            let mut body = json_openai_error(
                client_safe_worker_error_message(code, message),
                err_type,
                client_safe_worker_error_param(code, param.as_deref()),
                openai_code,
            );
            // Surface the SIE-native ``attempt_id`` alongside the
            // OpenAI envelope so SIE-aware SDKs can correlate retries.
            if let Some(obj) = body.as_object_mut() {
                if let Some(err_obj) = obj.get_mut("error").and_then(|v| v.as_object_mut()) {
                    err_obj.insert(
                        "attempt_id".to_string(),
                        serde_json::Value::String(attempt_id.clone()),
                    );
                }
            }
            let mut resp = (status, Json(body)).into_response();
            resp.headers_mut().insert(
                HeaderName::from_static("x-sie-request-id"),
                HeaderValue::from_str(request_id).unwrap_or_else(|_| HeaderValue::from_static("")),
            );
            // 429 responses get a ``Retry-After: 1`` header per the
            // OpenAI contract so SDKs retry with bounded backoff instead of
            // hammering the still-saturated pool — EXCEPT the cold-start
            // quota refusal, whose whole point is an HOURLY ceiling: a
            // 1-second retry hint would re-create the auto-retry hot loop
            // the rail exists to prevent (review finding).
            if status == StatusCode::TOO_MANY_REQUESTS
                && stable_code != COLD_START_RATE_LIMITED_ERROR_CODE
            {
                resp.headers_mut().insert(
                    HeaderName::from_static("retry-after"),
                    HeaderValue::from_static("1"),
                );
            }
            let retryable = worker_error_retry_after(stable_code, *retry_after_s);
            if let Some((retry_after, error_code)) = retryable {
                resp.headers_mut().insert(
                    HeaderName::from_static("retry-after"),
                    HeaderValue::from_str(&retry_after).unwrap_or_else(|_| {
                        HeaderValue::from_static(RESOURCE_EXHAUSTED_RETRY_AFTER)
                    }),
                );
                resp.headers_mut().insert(
                    HeaderName::from_static("x-sie-error-code"),
                    HeaderValue::from_static(error_code),
                );
            } else if stable_code == PAYLOAD_TOO_LARGE_ERROR_CODE {
                resp.headers_mut().insert(
                    HeaderName::from_static("x-sie-error-code"),
                    HeaderValue::from_static(PAYLOAD_TOO_LARGE_ERROR_CODE),
                );
            }
            resp
        }
    }
}

/// SSE-side helper: build a JSON error response for a publish
/// failure that happened *before* any SSE bytes went out. Mirrors
/// the `PublishFailed` arm of :func:`build_streaming_error_response`
/// but is reachable from outside the SSE driver task (so SSE handlers
/// can return a regular HTTP error response instead of a malformed
/// half-open SSE stream when the queue publish itself fails).
pub(crate) fn build_streaming_publish_failed_for_sse(
    message: &str,
    retry_after: Option<&'static str>,
) -> Response {
    let is_no_consumers = message.to_lowercase().contains("no consumers");
    let openai_code = if is_no_consumers {
        oai_code::PROVISIONING
    } else {
        oai_code::TRANSPORT_FAILURE
    };
    let mut resp = (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json_openai_error(
            format!("Queue publish failed: {message}"),
            oai_type::SERVER_ERROR,
            None,
            openai_code,
        )),
    )
        .into_response();
    if let Some(ra) = retry_after {
        resp.headers_mut().insert(
            HeaderName::from_static("retry-after"),
            HeaderValue::from_static(ra),
        );
    }
    if is_no_consumers {
        resp.headers_mut().insert(
            HeaderName::from_static("x-sie-error-code"),
            HeaderValue::from_static(err_code::PROVISIONING),
        );
    }
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp
}

#[allow(clippy::too_many_arguments)]
async fn queue_mode_streaming_generate(
    state: &AppState,
    work_publisher: Arc<dyn WorkDispatcher>,
    physical_lane: &PhysicalLane,
    // DISPLAY id (requested model) — response body, success metrics, audit log.
    model: &str,
    // DISPATCH id (grammar ``:no-spec`` variant when routed) — NATS subject +
    // work item. Equal to ``model`` for non-routed requests.
    dispatch_model: &str,
    bundle: &str,
    engine: &str,
    gpu: &str,
    pool: &str,
    admission_pool: &str,
    bundle_config_hash: &str,
    model_revision: Option<&str>,
    params: &publisher::WorkParams,
    use_msgpack_out: bool,
    token_id: &str,
    content_length: i64,
    start: Instant,
) -> Response {
    let driver = run_streaming_generate(
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
        params,
    )
    .await;

    let StreamingDriverOk {
        outcome,
        request_id,
        publish_elapsed,
        wait_elapsed,
    } = match driver {
        Ok(ok) => ok,
        Err(err) => return build_streaming_error_response(&err),
    };
    let elapsed = start.elapsed();

    let body_bytes = build_generate_success_body_v2(model, &outcome, use_msgpack_out);

    state.registry.record_request("queue").await;

    emit_audit_log(AuditEntry {
        event: "proxy_request".to_string(),
        method: "POST".to_string(),
        endpoint: "generate".to_string(),
        status: 200,
        token_id: token_id.to_string(),
        model: model.to_string(),
        pool: pool.to_string(),
        gpu: gpu.to_string(),
        worker: format!("queue:{}", request_id),
        latency_ms: elapsed.as_millis() as u64,
        body_bytes: content_length,
    });

    let content_type = if use_msgpack_out {
        "application/x-msgpack"
    } else {
        "application/json"
    };
    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(Body::from(body_bytes))
        .unwrap();
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static(content_type),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    insert_stream_model_revision_header(
        response.headers_mut(),
        model_revision,
        bundle_config_hash,
        &outcome,
    );
    insert_stream_execution_identity_header(response.headers_mut(), &outcome);
    insert_stream_execution_binding_header(response.headers_mut(), &outcome);
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-request-id"),
        HeaderValue::from_str(&request_id).unwrap_or_else(|_| HeaderValue::from_static("")),
    );
    insert_duration_header(
        response.headers_mut(),
        "x-queue-publish-time",
        publish_elapsed,
    );
    insert_duration_header(response.headers_mut(), "x-queue-wait-time", wait_elapsed);
    if let Ok(val) = HeaderValue::from_str(&format!("queue:{}", request_id)) {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-sie-worker"), val);
    }
    response
}

// ── OpenAI-compatible /v1/chat/completions ───────────────────────────

/// Parsed and validated chat-completion request body.
#[derive(Debug)]
struct ChatRequestParams {
    model: String,
    messages: Vec<publisher::ChatMessage>,
    max_new_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
    /// OpenAI ``frequency_penalty`` (range ``[-2.0, 2.0]``). Forwarded
    /// verbatim to the worker's sampling block; absent → worker uses
    /// the sampler default (typically 0.0).
    frequency_penalty: Option<f64>,
    /// OpenAI ``presence_penalty`` (range ``[-2.0, 2.0]``). Same
    /// semantics as :attr:`frequency_penalty`.
    presence_penalty: Option<f64>,
    /// Non-OpenAI sampling knob (Together / Fireworks / vLLM extension).
    /// Integer ``>= 1``; forwarded verbatim to the worker's sampling
    /// block. Absent → model/sampler default (top-k disabled).
    top_k: Option<u32>,
    /// Non-OpenAI sampling knob. Float in ``(0.0, 2.0]`` (``1.0`` = no
    /// penalty); forwarded to SGLang's ``repetition_penalty``. Absent →
    /// sampler default.
    repetition_penalty: Option<f64>,
    /// SGLang ``sampling_params.min_new_tokens``: minimum tokens to emit
    /// before any stop condition can end the response. Integer ``>= 0``;
    /// absent → no minimum. See the field doc on ``ChatCompletionRequest``
    /// in :mod:`openapi.rs` for the canonical use case (anti-stop-first
    /// fix on Qwen3.6 greedy decoding).
    min_tokens: Option<u32>,
    /// Per-request chat-template kwargs from the bounded public contract;
    /// validated before publication and revalidated by the worker.
    chat_template_kwargs: Option<serde_json::Value>,
    /// Routing hints plumbed through to the work envelope.
    routing_key: Option<String>,
    prompt_cache_key: Option<String>,
    /// Translated from OpenAI ``response_format.json_schema`` by
    /// :func:`chat_params_from_json`. ``None`` when the request omits
    /// ``response_format``.
    grammar: Option<publisher::GrammarSpec>,
    /// OpenAI ``stream`` flag. ``true`` switches the handler to the
    /// SSE response branch; ``false`` (default) uses the existing
    /// aggregating path. Parsed by :func:`chat_params_from_json`;
    /// validation rejects non-boolean values with 400.
    stream: bool,
    /// OpenAI ``stream_options.include_usage`` — when ``true`` the
    /// SSE stream appends a usage-only chunk (``choices: []``) before
    /// the terminating ``[DONE]`` event. Ignored when
    /// :attr:`stream` is ``false``. Unknown keys inside
    /// ``stream_options`` are rejected by the parser.
    stream_include_usage: bool,
    /// OpenAI ``tools``: validated non-empty array. ``None`` when the
    /// request omitted the field. Each entry is
    /// ``{type: "function", function: {name, description?, parameters?}}``
    /// with the ``function.parameters`` (when present) already
    /// run through the JSON-Schema safety walker.
    tools: Option<Vec<serde_json::Value>>,
    /// OpenAI ``tool_choice``: one of the literal strings ``"auto"`` /
    /// ``"none"`` / ``"required"`` or
    /// ``{type:"function", function:{name}}``.
    tool_choice: Option<serde_json::Value>,
    /// OpenAI ``parallel_tool_calls`` (default ``true``).
    parallel_tool_calls: Option<bool>,
    /// OpenAI ``seed``: optional signed 64-bit per-request sampling seed.
    /// Forwarded unchanged; reproducibility semantics depend on the active
    /// generation backend and deployment configuration.
    seed: Option<i64>,
    /// OpenAI ``logit_bias``: ``{token_id: bias_float}`` map. Gateway
    /// validates per-value range ``[-100.0, 100.0]`` and caps map size.
    logit_bias: Option<std::collections::BTreeMap<String, f64>>,
    /// OpenAI ``logprobs``: ``true`` requests per-token log-probabilities
    /// on each chunk. Worker forwards as SGLang's ``return_logprob``.
    logprobs: Option<bool>,
    /// OpenAI ``top_logprobs``: how many alternates per position
    /// (``[0, 20]`` per the OpenAI spec; gateway clamps). Implies
    /// ``logprobs: true`` when ``>0``.
    top_logprobs: Option<u32>,
    /// OpenAI ``n``: multi-candidate count. ``1`` is the default.
    /// ``n>1 && stream:true`` rejects with 400. Worker passes through
    /// to SGLang ``sampling_params.n``.
    n: Option<u32>,
    /// OpenAI ``best_of``: generate this many candidates, return the top ``n``
    /// by cumulative logprob. ``best_of >= n``; non-streaming only.
    best_of: Option<u32>,
    /// Multi-LoRA: served-name of a LoRA adapter to apply (SIE extension).
    /// Forwarded to the worker as SGLang ``sampling_params.lora_path``.
    lora_adapter: Option<String>,
}

impl ChatRequestParams {
    /// The EXACT `WorkParams` a validated chat body dispatches with.
    ///
    /// Extracted from the handler body (not copied) so the managed
    /// cost-estimate dry run (#2435) can derive a chat request's dispatch
    /// params through the same mapping the live route uses. Anything that
    /// influences the reservation ceiling — the messages, the output cap, the
    /// `n`/`best_of` fan-out, the tool schemas, the template kwargs — is
    /// therefore produced ONCE, and an estimate cannot drift from the price the
    /// same body is admitted at.
    ///
    /// Thread the routing hints into both `GenerateParams` (where the HRW
    /// resolver reads them) and `WorkParams` (where they were already carried
    /// since the chat/grammar surfaces for the worker's inert hint path). Clone
    /// is one allocation each and avoids either reader having to fall back on
    /// the other.
    fn into_work_params(self) -> publisher::WorkParams {
        publisher::WorkParams {
            generate: Some(publisher::GenerateParams {
                input: publisher::GenerateInput::Messages {
                    messages: self.messages,
                },
                max_new_tokens: self.max_new_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                stop: self.stop,
                frequency_penalty: self.frequency_penalty,
                presence_penalty: self.presence_penalty,
                top_k: self.top_k,
                repetition_penalty: self.repetition_penalty,
                min_tokens: self.min_tokens,
                chat_template_kwargs: self.chat_template_kwargs,
                grammar: self.grammar,
                routing_key: self.routing_key.clone(),
                prompt_cache_key: self.prompt_cache_key.clone(),
                tools: self.tools,
                tool_choice: self.tool_choice,
                parallel_tool_calls: self.parallel_tool_calls,
                seed: self.seed,
                logit_bias: self.logit_bias,
                logprobs: self.logprobs,
                top_logprobs: self.top_logprobs,
                n: self.n,
                best_of: self.best_of,
                stream: self.stream,
                lora_adapter: self.lora_adapter.clone(),
            }),
            routing_key: self.routing_key,
            prompt_cache_key: self.prompt_cache_key,
            ..Default::default()
        }
    }
}

/// Outcome of :func:`chat_params_from_json`. Errors carry an
/// already-built OpenAI-envelope response so the caller can return
/// directly.
#[allow(clippy::large_enum_variant)]
enum ChatParamsResult {
    Ok(ChatRequestParams),
    Err(Response),
}

/// The `(model, WorkParams)` a `/v1/chat/completions` body dispatches with —
/// the read-only seam the managed cost-estimate dry run (#2435) prices from.
///
/// A composition of the two pieces the live handler itself runs
/// ([`chat_params_from_json`] then [`ChatRequestParams::into_work_params`]), so
/// there is no second parser to keep in sync: a body the estimate prices is
/// validated and normalized by exactly the code that would admit it. Errors are
/// the handler's own pre-built OpenAI envelopes, so a malformed body estimates
/// with the same status/`param`/`code` it would be rejected with.
///
/// `dead_code`-allowed because the `sie-gateway` BINARY compiles this module
/// tree independently of the library (see the note in `lib.rs`), and only the
/// downstream composition crate calls this seam.
#[allow(clippy::result_large_err, dead_code)]
pub fn chat_completions_dispatch_inputs(
    body: &serde_json::Value,
) -> Result<(String, publisher::WorkParams), Response> {
    match chat_params_from_json(body) {
        ChatParamsResult::Ok(params) => {
            let model = params.model.clone();
            Ok((model, params.into_work_params()))
        }
        ChatParamsResult::Err(response) => Err(response),
    }
}

/// Extract an inline OpenAI ``image_url`` value into ``(base64, format)``.
///
/// Only base64 ``data:`` URIs are accepted — the gateway never fetches
/// remote ``http(s)`` URLs (SSRF surface + a blocking network call on the
/// request path; clients inline images as data URIs, which the SDK does).
/// The ``url`` may be a bare string or the ``{"url": "data:..."}`` object
/// shape.
///
/// Returns the **base64 payload string** (validated to decode and be
/// non-empty) plus the format hint parsed from the MIME subtype. We keep
/// the base64 *string* rather than raw bytes because the work item travels
/// through the sidecar's ``serde_json::Value``, which can't carry a msgpack
/// ``bin`` — see :struct:`ChatImage`. The worker base64-decodes it.
fn decode_image_data_uri(value: &serde_json::Value) -> Result<(String, Option<String>), String> {
    use base64::Engine as _;

    let url = match value {
        serde_json::Value::String(s) => s.as_str(),
        serde_json::Value::Object(o) => o
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "image_url.url must be a non-empty string".to_string())?,
        _ => return Err("image_url must be a string or an object with a 'url'".to_string()),
    };
    if url.is_empty() {
        return Err("image_url.url must be a non-empty string".to_string());
    }
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| "image content must be an inline base64 'data:' URI; remote URL fetching is not supported".to_string())?;
    let (header, payload) = rest
        .split_once(',')
        .ok_or_else(|| "malformed image data URI (missing ',')".to_string())?;
    let mut params = header.split(';');
    let mime = params.next().unwrap_or("");
    if !mime.starts_with("image/") {
        // Reject ``data:text/...`` and ``data:;base64,...`` — only image media types.
        return Err("image data URI must have an image/* media type".to_string());
    }
    if !params.any(|p| p == "base64") {
        return Err("image data URI must be base64-encoded".to_string());
    }
    let format = mime
        .split_once('/')
        .map(|(_, sub)| sub.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty());
    let payload = payload.trim();
    // Validate at the edge (reject malformed base64 / empty image) so the
    // client gets a clean 400, but transport the base64 string itself.
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|e| format!("invalid base64 image data: {e}"))?;
    if decoded.is_empty() {
        return Err("image data URI decoded to empty bytes".to_string());
    }
    Ok((payload.to_string(), format))
}

/// Validate an OpenAI ``/v1/chat/completions`` request body against
/// the chat-completions supported subset.
///
/// Accepted fields:
///
/// * ``model`` (required, string)
/// * ``messages`` (required, non-empty array of ``{role, content}``)
/// * ``max_completion_tokens`` (preferred) or ``max_tokens``
/// * ``temperature``, ``top_p``, ``stop``
/// * ``frequency_penalty`` / ``presence_penalty`` — forwarded to the
///   worker; validated against OpenAI's ``[-2.0, 2.0]`` range.
/// * ``response_format`` — ``json_schema`` translates to a SIE grammar;
///   ``json_object`` (loose JSON) translates to a built-in generic
///   schema (``{"type": "object", ...}``).
/// * ``tools`` / ``tool_choice`` / ``parallel_tool_calls`` — validated and
///   forwarded; ``function.parameters`` is run through the JSON-Schema
///   safety walker (see the `ChatRequestParams` field docs above).
/// * ``logprobs`` / ``top_logprobs`` — validated and forwarded; the worker
///   emits per-token log-probabilities via SGLang ``return_logprob`` and
///   the gateway plumbs them onto each chunk / the aggregate body.
/// * ``seed`` — validated as a signed 64-bit integer and forwarded unchanged;
///   reproducibility semantics depend on the active generation backend and
///   deployment configuration.
/// * ``logit_bias`` (range/size-validated, forwarded), ``n`` (default
///   ``1``; both ``n>1`` non-streaming and streaming paths supported),
///   ``stream`` / ``stream_options``.
/// * ``user`` (debug-log only), ``safety_identifier`` (silent),
///   ``prompt_cache_key`` (plumbed), ``routing_key`` (plumbed)
///
/// Any field outside this set surfaces as 400 with
/// ``code: "unsupported_field"`` and ``param`` set to the offending field
/// name.
fn chat_params_from_json(body: &serde_json::Value) -> ChatParamsResult {
    let bad = |msg: &str, param: Option<&str>, code: &'static str| -> ChatParamsResult {
        ChatParamsResult::Err(
            (
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    msg.to_string(),
                    oai_type::INVALID_REQUEST,
                    param,
                    code,
                )),
            )
                .into_response(),
        )
    };
    let unsupported = |param: &str| -> ChatParamsResult {
        bad(
            &format!("'{param}' is not supported by this endpoint"),
            Some(param),
            oai_code::UNSUPPORTED_FIELD,
        )
    };

    let Some(obj) = body.as_object() else {
        return bad(
            "request body must be a JSON object",
            None,
            oai_code::INVALID_REQUEST,
        );
    };

    // -- required: model
    let model = match obj.get("model").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => {
            return bad(
                "field 'model' is required",
                Some("model"),
                oai_code::INVALID_REQUEST,
            );
        }
    };

    // -- required: messages
    let messages_val = match obj.get("messages") {
        Some(v) => v,
        None => {
            return bad(
                "field 'messages' is required",
                Some("messages"),
                oai_code::INVALID_REQUEST,
            );
        }
    };
    let messages_arr = match messages_val.as_array() {
        Some(a) if !a.is_empty() => a,
        Some(_) => {
            return bad(
                "'messages' must be a non-empty array",
                Some("messages"),
                oai_code::INVALID_REQUEST,
            );
        }
        None => {
            return bad(
                "'messages' must be a non-empty array",
                Some("messages"),
                oai_code::INVALID_REQUEST,
            );
        }
    };
    let mut messages: Vec<publisher::ChatMessage> = Vec::with_capacity(messages_arr.len());
    // Cap total images per request so an oversized vision request fails here —
    // before decoding every part and publishing the work — rather than at the
    // worker. Mirrors the worker's ``_MAX_IMAGES_PER_REQUEST``.
    const MAX_IMAGES_PER_REQUEST: usize = 16;
    let mut total_images: usize = 0;
    // ``tool`` is allowed so the multi-turn tool-use loop works: the
    // caller replays the assistant's tool_call request and the tool
    // result back into ``messages`` for the model's final answer.
    // ``developer`` is OpenAI's newer name for ``system``; accept it and fold
    // it to ``system`` below so the wire stays on the 4-role set the worker's
    // chat template understands (Qwen has no ``developer`` slot).
    const ALLOWED_ROLES: &[&str] = &["system", "user", "assistant", "tool", "developer"];
    for (idx, item) in messages_arr.iter().enumerate() {
        let Some(item_obj) = item.as_object() else {
            return bad(
                &format!("messages[{idx}] must be an object"),
                Some(&format!("messages[{idx}]")),
                oai_code::INVALID_REQUEST,
            );
        };
        let role = match item_obj.get("role").and_then(|v| v.as_str()) {
            // Normalize ``developer`` → ``system`` at the gateway.
            Some("developer") => "system".to_string(),
            Some(r) if ALLOWED_ROLES.contains(&r) => r.to_string(),
            Some(other) => {
                return bad(
                    &format!(
                        "messages[{idx}].role must be one of {:?}, got {other:?}",
                        ALLOWED_ROLES
                    ),
                    Some(&format!("messages[{idx}].role")),
                    oai_code::INVALID_REQUEST,
                );
            }
            None => {
                return bad(
                    &format!("messages[{idx}].role is required and must be a string"),
                    Some(&format!("messages[{idx}].role")),
                    oai_code::INVALID_REQUEST,
                );
            }
        };
        // Assistant tool-call requests carry ``tool_calls`` (OpenAI shape).
        // M13: validate each entry's inner shape rather than accepting any
        // JSON array — SDKs that drop the ``id`` / ``function.name`` /
        // ``function.arguments`` fields would otherwise produce wire frames
        // the worker has no way to interpret.
        let tool_calls: Option<Vec<serde_json::Value>> = match item_obj.get("tool_calls") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::Array(arr)) => {
                for (tci, tc) in arr.iter().enumerate() {
                    let Some(tc_obj) = tc.as_object() else {
                        return bad(
                            &format!("messages[{idx}].tool_calls[{tci}] must be an object"),
                            Some(&format!("messages[{idx}].tool_calls[{tci}]")),
                            oai_code::INVALID_REQUEST,
                        );
                    };
                    // ``id`` — non-empty string
                    match tc_obj.get("id").and_then(|v| v.as_str()) {
                        Some(s) if !s.is_empty() => {}
                        _ => {
                            return bad(
                                &format!(
                                    "messages[{idx}].tool_calls[{tci}].id is required and must be a non-empty string"
                                ),
                                Some(&format!("messages[{idx}].tool_calls[{tci}].id")),
                                oai_code::INVALID_REQUEST,
                            );
                        }
                    }
                    // ``type`` — must be the literal string "function"
                    match tc_obj.get("type").and_then(|v| v.as_str()) {
                        Some("function") => {}
                        Some(other) => {
                            return bad(
                                &format!(
                                    "messages[{idx}].tool_calls[{tci}].type must be \"function\", got {other:?}"
                                ),
                                Some(&format!("messages[{idx}].tool_calls[{tci}].type")),
                                oai_code::INVALID_REQUEST,
                            );
                        }
                        None => {
                            return bad(
                                &format!(
                                    "messages[{idx}].tool_calls[{tci}].type is required and must be \"function\""
                                ),
                                Some(&format!("messages[{idx}].tool_calls[{tci}].type")),
                                oai_code::INVALID_REQUEST,
                            );
                        }
                    }
                    // ``function`` — object with name (string) + arguments (string)
                    let Some(func) = tc_obj.get("function").and_then(|v| v.as_object()) else {
                        return bad(
                            &format!(
                                "messages[{idx}].tool_calls[{tci}].function is required and must be an object"
                            ),
                            Some(&format!("messages[{idx}].tool_calls[{tci}].function")),
                            oai_code::INVALID_REQUEST,
                        );
                    };
                    match func.get("name").and_then(|v| v.as_str()) {
                        Some(s) if !s.is_empty() => {}
                        _ => {
                            return bad(
                                &format!(
                                    "messages[{idx}].tool_calls[{tci}].function.name is required and must be a non-empty string"
                                ),
                                Some(&format!(
                                    "messages[{idx}].tool_calls[{tci}].function.name"
                                )),
                                oai_code::INVALID_REQUEST,
                            );
                        }
                    }
                    // OpenAI ships ``arguments`` as a JSON-string (not an
                    // object). Require a string; the worker is responsible
                    // for further parsing.
                    match func.get("arguments") {
                        Some(serde_json::Value::String(_)) => {}
                        _ => {
                            return bad(
                                &format!(
                                    "messages[{idx}].tool_calls[{tci}].function.arguments is required and must be a JSON-string"
                                ),
                                Some(&format!(
                                    "messages[{idx}].tool_calls[{tci}].function.arguments"
                                )),
                                oai_code::INVALID_REQUEST,
                            );
                        }
                    }
                }
                Some(arr.clone())
            }
            Some(_) => {
                return bad(
                    &format!("messages[{idx}].tool_calls must be an array"),
                    Some(&format!("messages[{idx}].tool_calls")),
                    oai_code::INVALID_REQUEST,
                );
            }
        };
        // M13: ``tool_call_id`` is required when role == "tool" (it ties
        // the result back to the prior assistant tool_call), and is
        // forbidden on every other role. Today the parser silently
        // accepted it on any role, which let malformed tool-loop replays
        // through.
        let tool_call_id = match item_obj.get("tool_call_id") {
            None | Some(serde_json::Value::Null) => {
                if role == "tool" {
                    return bad(
                        &format!(
                            "messages[{idx}].tool_call_id is required on role:\"tool\" messages"
                        ),
                        Some(&format!("messages[{idx}].tool_call_id")),
                        oai_code::INVALID_REQUEST,
                    );
                }
                None
            }
            Some(serde_json::Value::String(s)) => {
                if role != "tool" {
                    return bad(
                        &format!(
                            "messages[{idx}].tool_call_id is only valid on role:\"tool\" messages (got role:{role:?})"
                        ),
                        Some(&format!("messages[{idx}].tool_call_id")),
                        oai_code::INVALID_REQUEST,
                    );
                }
                if s.is_empty() {
                    return bad(
                        &format!("messages[{idx}].tool_call_id must be a non-empty string"),
                        Some(&format!("messages[{idx}].tool_call_id")),
                        oai_code::INVALID_REQUEST,
                    );
                }
                Some(s.clone())
            }
            Some(_) => {
                return bad(
                    &format!("messages[{idx}].tool_call_id must be a string"),
                    Some(&format!("messages[{idx}].tool_call_id")),
                    oai_code::INVALID_REQUEST,
                );
            }
        };
        // ``content`` is required EXCEPT on an assistant message that
        // carries tool_calls (OpenAI sends content:null there). Tool and
        // normal messages still require a string content.
        // Images decoded from this message's ``image_url`` content parts.
        // Whether the model may actually accept them is gated after model
        // resolution (mirrors the grammar/tools capability gates) — parsing
        // here is capability-agnostic because the model isn't resolved yet.
        let mut message_images: Vec<publisher::ChatImage> = Vec::new();
        // Ordered text↔image layout, preserving the parts' original order so
        // the worker can interleave placeholders (vs. images-first). Only the
        // placeholder positions depend on this; bytes still ride
        // ``message_images`` in order. Populated only for the Array branch and
        // only forwarded when the message has ≥1 image (#1294).
        let mut content_parts: Vec<publisher::ContentPart> = Vec::new();
        let content = match item_obj.get("content") {
            Some(serde_json::Value::String(c)) => c.clone(),
            // OpenAI multimodal content-parts. Concatenate ``text`` parts and
            // decode ``image_url`` parts into ``message_images`` (the vision
            // capability gate runs after model resolution); record the ordered
            // layout in ``content_parts`` so interleaving survives to the worker.
            //
            // M13: parts with missing/non-string ``text`` (or no ``type``)
            // previously slipped through via ``filter_map`` — they now
            // reject as ``invalid_request`` so SDK bugs surface early.
            Some(serde_json::Value::Array(parts)) => {
                let mut text = String::new();
                for (pi, part) in parts.iter().enumerate() {
                    let Some(part_obj) = part.as_object() else {
                        return bad(
                            &format!("messages[{idx}].content[{pi}] must be an object"),
                            Some(&format!("messages[{idx}].content[{pi}]")),
                            oai_code::INVALID_REQUEST,
                        );
                    };
                    let ptype = match part_obj.get("type").and_then(|v| v.as_str()) {
                        Some(t) => t,
                        None => {
                            return bad(
                                &format!(
                                    "messages[{idx}].content[{pi}].type is required and must be a string"
                                ),
                                Some(&format!("messages[{idx}].content[{pi}].type")),
                                oai_code::INVALID_REQUEST,
                            );
                        }
                    };
                    match ptype {
                        "text" | "input_text" => match part_obj.get("text") {
                            Some(serde_json::Value::String(t)) => {
                                text.push_str(t);
                                content_parts
                                    .push(publisher::ContentPart::Text { text: t.clone() });
                            }
                            Some(_) => {
                                return bad(
                                    &format!("messages[{idx}].content[{pi}].text must be a string"),
                                    Some(&format!("messages[{idx}].content[{pi}].text")),
                                    oai_code::INVALID_REQUEST,
                                );
                            }
                            None => {
                                return bad(
                                    &format!(
                                        "messages[{idx}].content[{pi}].text is required for text content parts"
                                    ),
                                    Some(&format!("messages[{idx}].content[{pi}].text")),
                                    oai_code::INVALID_REQUEST,
                                );
                            }
                        },
                        "image_url" | "input_image" => {
                            let Some(image_url) = part_obj.get("image_url") else {
                                return bad(
                                    &format!(
                                        "messages[{idx}].content[{pi}].image_url is required for image content parts"
                                    ),
                                    Some(&format!("messages[{idx}].content[{pi}].image_url")),
                                    oai_code::INVALID_REQUEST,
                                );
                            };
                            total_images += 1;
                            if total_images > MAX_IMAGES_PER_REQUEST {
                                return bad(
                                    &format!(
                                        "too many images ({total_images}); maximum is {MAX_IMAGES_PER_REQUEST} per request"
                                    ),
                                    Some(&format!("messages[{idx}].content[{pi}].image_url")),
                                    oai_code::INVALID_REQUEST,
                                );
                            }
                            match decode_image_data_uri(image_url) {
                                Ok((data, format)) => {
                                    // Large work items (multi-MB images) are offloaded to
                                    // the object store by ``publish_generate`` and bounded
                                    // by the request body cap, so no inline-size gate here.
                                    message_images.push(publisher::ChatImage { data, format });
                                    content_parts.push(publisher::ContentPart::Image);
                                }
                                Err(reason) => {
                                    return bad(
                                        &format!("messages[{idx}].content[{pi}]: {reason}"),
                                        Some(&format!("messages[{idx}].content[{pi}].image_url")),
                                        oai_code::INVALID_REQUEST,
                                    );
                                }
                            }
                        }
                        other => {
                            return bad(
                                &format!(
                                    "messages[{idx}].content[{pi}]: unsupported content part type '{other}'"
                                ),
                                Some(&format!("messages[{idx}].content[{pi}].type")),
                                oai_code::UNSUPPORTED_FIELD,
                            );
                        }
                    }
                }
                text
            }
            // OpenAI sends content:null on an assistant message that carries
            // tool_calls; ``get`` returns Some(Null) (not None), so match both.
            None | Some(serde_json::Value::Null) if role == "assistant" && tool_calls.is_some() => {
                String::new()
            }
            _ => {
                return bad(
                    &format!("messages[{idx}].content must be a string or content-part array"),
                    Some(&format!("messages[{idx}].content")),
                    oai_code::UNSUPPORTED_FIELD,
                );
            }
        };
        let has_images = !message_images.is_empty();
        messages.push(publisher::ChatMessage {
            role,
            content,
            tool_calls,
            tool_call_id,
            images: if has_images {
                Some(message_images)
            } else {
                None
            },
            // Forward the ordered layout only for multimodal messages — a
            // text-only message keeps ``content_parts: None`` and renders from
            // ``content`` as before (no wire bloat, no behavior change).
            content_parts: if has_images {
                Some(content_parts)
            } else {
                None
            },
        });
    }

    // -- token budget: max_completion_tokens preferred, fall back to max_tokens.
    // OpenAI treats max_tokens as optional (defaults server-side), so an
    // OpenAI-compatible surface must not hard-require it — generic clients
    // like Open WebUI omit it. When absent, default to a sane cap rather
    // than 400. Override the default via SIE_GATEWAY_DEFAULT_MAX_TOKENS.
    //
    // M1 hardening: present-but-invalid values (e.g. ``"16"``, ``-1``,
    // ``1.5``) must reject rather than silently falling back to the
    // default. Only the truly-absent / explicit-null case uses the
    // default. Inlined per-field rather than via a closure to avoid
    // ``clippy::result_large_err`` on a closure returning the large
    // ``ChatParamsResult`` error variant (mirrors the pattern used by
    // the penalty validation block below).
    let max_completion: Option<u32> = match obj.get("max_completion_tokens") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_u64() {
            Some(n) if (1..=u32::MAX as u64).contains(&n) => Some(n as u32),
            _ => {
                return bad(
                    "'max_completion_tokens' must be a positive integer",
                    Some("max_completion_tokens"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    let max_tokens_legacy: Option<u32> = match obj.get("max_tokens") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_u64() {
            Some(n) if (1..=u32::MAX as u64).contains(&n) => Some(n as u32),
            _ => {
                return bad(
                    "'max_tokens' must be a positive integer",
                    Some("max_tokens"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    let max_new_tokens = match (max_completion, max_tokens_legacy) {
        (Some(n), _) => n,
        (None, Some(n)) => n,
        (None, None) => {
            // The compile-time default is already a small positive u32;
            // the conversion below is total in practice.
            u32::try_from(default_max_tokens()).unwrap_or(u32::MAX)
        }
    };

    // -- sampling. Range-validate so NaN / inf / out-of-range samplers
    // never reach the worker.
    //
    // M1 hardening: present-but-non-numeric values (e.g. ``"0.5"``, bool,
    // object) must reject rather than being silently dropped via
    // ``and_then(as_f64)``. Only the truly-absent / explicit-null case
    // is accepted as "no override".
    let temperature = match obj.get("temperature") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && f >= 0.0 && f <= f32::MAX as f64 => Some(f as f32),
            _ => {
                return bad(
                    "'temperature' must be a finite number >= 0",
                    Some("temperature"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    let top_p = match obj.get("top_p") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && f > 0.0 && f <= 1.0 => Some(f as f32),
            _ => {
                return bad(
                    "'top_p' must be a number in (0, 1]",
                    Some("top_p"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    let stop = match obj.get("stop") {
        None => None,
        Some(v) => {
            // OpenAI accepts a string or array of strings. We accept
            // both; non-string scalars are rejected.
            if let Some(s) = v.as_str() {
                Some(vec![s.to_string()])
            } else if let Some(arr) = v.as_array() {
                let mut out: Vec<String> = Vec::with_capacity(arr.len());
                for entry in arr {
                    let Some(s) = entry.as_str() else {
                        return bad(
                            "'stop' must be a string or array of strings",
                            Some("stop"),
                            oai_code::INVALID_REQUEST,
                        );
                    };
                    out.push(s.to_string());
                }
                if out.is_empty() {
                    None
                } else {
                    Some(out)
                }
            } else {
                return bad(
                    "'stop' must be a string or array of strings",
                    Some("stop"),
                    oai_code::INVALID_REQUEST,
                );
            }
        }
    };

    // -- penalties: validate range [-2.0, 2.0] per OpenAI spec, then
    //    forward to the worker via :class:`ChatRequestParams`. NaN /
    //    null / non-numeric inputs reject as 400; absent → None →
    //    worker uses its default (typically 0.0 for penalties).
    //
    // Inlined per-field rather than via a closure to avoid
    // ``clippy::result_large_err`` on a closure returning the
    // large ``ChatParamsResult`` error variant. The branch logic is
    // simple enough that the duplication is cheaper than wrapping the
    // helper in ``Box<ChatParamsResult>``.
    let mut frequency_penalty: Option<f64> = None;
    if let Some(val) = obj.get("frequency_penalty") {
        if !val.is_null() {
            match val.as_f64() {
                Some(f) if f.is_finite() && (-2.0..=2.0).contains(&f) => {
                    frequency_penalty = Some(f);
                }
                _ => {
                    return bad(
                        "'frequency_penalty' must be a number in [-2.0, 2.0]",
                        Some("frequency_penalty"),
                        oai_code::INVALID_REQUEST,
                    );
                }
            }
        }
    }
    let mut presence_penalty: Option<f64> = None;
    if let Some(val) = obj.get("presence_penalty") {
        if !val.is_null() {
            match val.as_f64() {
                Some(f) if f.is_finite() && (-2.0..=2.0).contains(&f) => {
                    presence_penalty = Some(f);
                }
                _ => {
                    return bad(
                        "'presence_penalty' must be a number in [-2.0, 2.0]",
                        Some("presence_penalty"),
                        oai_code::INVALID_REQUEST,
                    );
                }
            }
        }
    }

    // -- top_k: non-OpenAI sampling knob (Together / Fireworks / vLLM).
    //    Integer >= 1; absent → model default (top-k disabled). Sent as
    //    a JSON integer; floats / non-integers / values < 1 reject.
    let mut top_k: Option<u32> = None;
    if let Some(val) = obj.get("top_k") {
        if !val.is_null() {
            match val.as_i64() {
                Some(k) if (1..=u32::MAX as i64).contains(&k) => {
                    top_k = Some(k as u32);
                }
                _ => {
                    return bad(
                        "'top_k' must be an integer >= 1",
                        Some("top_k"),
                        oai_code::INVALID_REQUEST,
                    );
                }
            }
        }
    }
    // -- repetition_penalty: non-OpenAI multiplicative penalty. Float in
    //    (0.0, 2.0] (1.0 = no penalty); absent → sampler default.
    let mut repetition_penalty: Option<f64> = None;
    if let Some(val) = obj.get("repetition_penalty") {
        if !val.is_null() {
            match val.as_f64() {
                Some(f) if f.is_finite() && f > 0.0 && f <= 2.0 => {
                    repetition_penalty = Some(f);
                }
                _ => {
                    return bad(
                        "'repetition_penalty' must be a number in (0.0, 2.0]",
                        Some("repetition_penalty"),
                        oai_code::INVALID_REQUEST,
                    );
                }
            }
        }
    }
    // -- min_tokens: SGLang ``sampling_params.min_new_tokens``. Integer
    //    >= 0; values >= u32::MAX or negative reject. Absent → no
    //    minimum. Workaround for stop-first artefacts (e.g. Qwen3.6
    //    greedy decoding occasionally emits the stop token first).
    let mut min_tokens: Option<u32> = None;
    if let Some(val) = obj.get("min_tokens") {
        if !val.is_null() {
            match val.as_i64() {
                Some(k) if (0..=u32::MAX as i64).contains(&k) => {
                    min_tokens = Some(k as u32);
                }
                _ => {
                    return bad(
                        "'min_tokens' must be an integer >= 0",
                        Some("min_tokens"),
                        oai_code::INVALID_REQUEST,
                    );
                }
            }
        }
    }
    // -- chat_template_kwargs: bounded kwargs for the tokenizer's
    //    ``apply_chat_template``. The worker merges them under the model
    //    YAML defaults at render time. Unknown keys reject here rather than
    //    crossing the queue as arbitrary tokenizer arguments.
    let mut chat_template_kwargs: Option<serde_json::Value> = None;
    if let Some(val) = obj.get("chat_template_kwargs") {
        if !val.is_null() {
            let Some(kwargs) = val.as_object() else {
                return bad(
                    "'chat_template_kwargs' must be an object",
                    Some("chat_template_kwargs"),
                    oai_code::INVALID_REQUEST,
                );
            };
            for key in kwargs.keys() {
                if !matches!(key.as_str(), "enable_thinking" | "guardian_config") {
                    return bad(
                        &format!("unsupported chat_template_kwargs key: {key}"),
                        Some(&format!("chat_template_kwargs.{key}")),
                        oai_code::UNSUPPORTED_FIELD,
                    );
                }
            }
            if let Some(enable_thinking) = kwargs.get("enable_thinking") {
                if !enable_thinking.is_boolean() {
                    return bad(
                        "'chat_template_kwargs.enable_thinking' must be a boolean",
                        Some("chat_template_kwargs.enable_thinking"),
                        oai_code::INVALID_REQUEST,
                    );
                }
            }
            if let Some(guardian_config) = kwargs.get("guardian_config") {
                let Some(guardian) = guardian_config.as_object() else {
                    return bad(
                        "'chat_template_kwargs.guardian_config' must be an object",
                        Some("chat_template_kwargs.guardian_config"),
                        oai_code::INVALID_REQUEST,
                    );
                };
                for key in guardian.keys() {
                    if key != "risk_name" {
                        return bad(
                            &format!("unsupported chat_template_kwargs.guardian_config key: {key}"),
                            Some(&format!("chat_template_kwargs.guardian_config.{key}")),
                            oai_code::UNSUPPORTED_FIELD,
                        );
                    }
                }
                match guardian
                    .get("risk_name")
                    .and_then(|risk_name| risk_name.as_str())
                {
                    Some(risk_name)
                        if !risk_name.is_empty() && risk_name.chars().count() <= 128 => {}
                    _ => {
                        return bad(
                            "'chat_template_kwargs.guardian_config.risk_name' must be a non-empty string of at most 128 characters",
                            Some("chat_template_kwargs.guardian_config.risk_name"),
                            oai_code::INVALID_REQUEST,
                        );
                    }
                }
            }
            chat_template_kwargs = Some(val.clone());
        }
    }

    // -- n: parsed and surfaced on :class:`ChatRequestParams.n` so the
    //    wire envelope (``GenerateParams.n``) drives the worker's
    //    candidate fan-out. ``n=1`` (or absent) is the implicit default;
    //    ``n>1`` is supported on both the non-streaming and streaming
    //    paths (see :func:`build_chat_completion_body` and the SSE
    //    per-``choice_index`` interleave).
    let n: Option<u32> = match obj.get("n") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_u64() {
            Some(0) => {
                return bad(
                    "'n' must be a positive integer",
                    Some("n"),
                    oai_code::INVALID_REQUEST,
                );
            }
            Some(n) if n <= 128 => Some(n as u32),
            Some(_) => {
                return bad(
                    "'n' must be in [1, 128]",
                    Some("n"),
                    oai_code::INVALID_REQUEST,
                );
            }
            None => {
                return bad(
                    "'n' must be an integer",
                    Some("n"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };

    // OpenAI ``best_of``: generate N candidates, return the top ``n``. Range
    // [1, 128]; cross-field rules (best_of >= n, non-streaming) checked below.
    let best_of: Option<u32> = match obj.get("best_of") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_u64() {
            Some(b) if (1..=128).contains(&b) => Some(b as u32),
            _ => {
                return bad(
                    "'best_of' must be an integer in [1, 128]",
                    Some("best_of"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };

    // Multi-LoRA: optional served-name of a LoRA adapter (SIE extension).
    // Non-empty string; the worker resolves it against the model's loaded
    // adapters (unknown name → error chunk).
    let lora_adapter: Option<String> = match obj.get("lora_adapter") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(_) => {
            return bad(
                "'lora_adapter' must be a non-empty string",
                Some("lora_adapter"),
                oai_code::INVALID_REQUEST,
            );
        }
    };

    // -- inert / accept-and-drop
    let _ = obj.get("user");
    // ``safety_identifier`` is intentionally NOT logged at any level
    // (decision 0.2 in plan §0). We just acknowledge its presence by
    // not rejecting it.
    let _ = obj.get("safety_identifier");
    // ``seed`` — validate once across every generation route and preserve the
    // signed value on the worker wire. ``null`` and absent are both no-ops.
    let seed = match parse_seed_field(obj.get("seed")) {
        Ok(seed) => seed,
        Err(resp) => return ChatParamsResult::Err(resp),
    };
    // ``logprobs`` / ``top_logprobs`` — when ``logprobs: true`` the
    // worker forwards SGLang's ``return_logprob`` flag. ``top_logprobs``
    // ranges over ``[0, 20]`` per OpenAI's spec.
    let logprobs: Option<bool> = match obj.get("logprobs") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Bool(b)) => Some(*b),
        Some(_) => {
            return bad(
                "'logprobs' must be a boolean",
                Some("logprobs"),
                oai_code::INVALID_REQUEST,
            );
        }
    };
    let top_logprobs: Option<u32> = match obj.get("top_logprobs") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_u64() {
            Some(n) if n <= 20 => Some(n as u32),
            _ => {
                return bad(
                    "'top_logprobs' must be an integer in [0, 20]",
                    Some("top_logprobs"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    // OpenAI rule: ``top_logprobs`` requires ``logprobs: true``.
    if matches!(top_logprobs, Some(n) if n > 0) && !matches!(logprobs, Some(true)) {
        return bad(
            "'top_logprobs' requires 'logprobs: true'",
            Some("top_logprobs"),
            oai_code::INVALID_REQUEST,
        );
    }
    // ``logit_bias`` — ``{token_id: bias_float}`` map. Gateway caps
    // the map size and per-value range so a giant or out-of-range
    // payload cannot DoS the worker's sampler.
    const MAX_LOGIT_BIAS_KEYS: usize = 1024;
    let logit_bias: Option<std::collections::BTreeMap<String, f64>> = match obj.get("logit_bias") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => {
            let Some(map) = v.as_object() else {
                return bad(
                    "'logit_bias' must be an object",
                    Some("logit_bias"),
                    oai_code::INVALID_REQUEST,
                );
            };
            if map.len() > MAX_LOGIT_BIAS_KEYS {
                return bad(
                    &format!("'logit_bias' has too many entries (max {MAX_LOGIT_BIAS_KEYS})"),
                    Some("logit_bias"),
                    oai_code::INVALID_REQUEST,
                );
            }
            let mut out = std::collections::BTreeMap::new();
            for (k, val) in map.iter() {
                if k.parse::<i64>().is_err() {
                    return bad(
                        &format!(
                            "'logit_bias' keys must be token-id integers as strings (got {k:?})"
                        ),
                        Some("logit_bias"),
                        oai_code::INVALID_REQUEST,
                    );
                }
                let f = val.as_f64().filter(|f| f.is_finite());
                let Some(f) = f else {
                    return bad(
                        "'logit_bias' values must be finite numbers",
                        Some("logit_bias"),
                        oai_code::INVALID_REQUEST,
                    );
                };
                if !(-100.0..=100.0).contains(&f) {
                    return bad(
                        "'logit_bias' values must be in [-100.0, 100.0]",
                        Some("logit_bias"),
                        oai_code::INVALID_REQUEST,
                    );
                }
                out.insert(k.clone(), f);
            }
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
    };
    let routing_key = obj
        .get("routing_key")
        .and_then(|v| v.as_str())
        .map(String::from);
    let prompt_cache_key = obj
        .get("prompt_cache_key")
        .and_then(|v| v.as_str())
        .map(String::from);

    // -- streaming (SSE). ``stream`` and ``stream_options`` are
    //    accepted; ``stream_options.include_usage`` is the only
    //    sub-key currently in the OpenAI spec and anything else
    //    inside that object is rejected explicitly so unknown future
    //    knobs surface as ``unsupported_field`` instead of being
    //    silently ignored.
    let stream = match obj.get("stream") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(_) => {
            return bad(
                "'stream' must be a boolean",
                Some("stream"),
                oai_code::INVALID_REQUEST,
            );
        }
    };
    let mut stream_include_usage = false;
    if let Some(opts) = obj.get("stream_options") {
        if !opts.is_null() {
            let Some(opts_obj) = opts.as_object() else {
                return bad(
                    "'stream_options' must be a JSON object",
                    Some("stream_options"),
                    oai_code::INVALID_REQUEST,
                );
            };
            for key in opts_obj.keys() {
                if key != "include_usage" {
                    return bad(
                        &format!("'stream_options.{key}' is not supported by this endpoint"),
                        Some(&format!("stream_options.{key}")),
                        oai_code::UNSUPPORTED_FIELD,
                    );
                }
            }
            if let Some(iu) = opts_obj.get("include_usage") {
                match iu {
                    serde_json::Value::Bool(b) => stream_include_usage = *b,
                    serde_json::Value::Null => {}
                    _ => {
                        return bad(
                            "'stream_options.include_usage' must be a boolean",
                            Some("stream_options.include_usage"),
                            oai_code::INVALID_REQUEST,
                        );
                    }
                }
            }
        }
    }
    // ``stream_options`` with ``stream:false`` is OpenAI-legal (the
    // options are simply ignored). We mirror that behaviour rather
    // than rejecting, so SDKs that always set both knobs together
    // still parse on the non-streaming path.

    // -- explicit rejects: unsupported features.
    //
    // ``response_format`` is handled separately below. ``tools``,
    // ``tool_choice``, ``parallel_tool_calls`` are validated further
    // below and plumbed through. The legacy OpenAI ``functions`` /
    // ``function_call`` keys remain rejected with a deprecation hint —
    // OpenAI itself recommends migrating to ``tools`` / ``tool_choice``.
    for &fname in &[
        "functions",
        "function_call",
        "modalities",
        "audio",
        "metadata",
        "store",
        "service_tier",
        "prediction",
        "reasoning_effort",
        "verbosity",
    ] {
        if obj.contains_key(fname) {
            let message = if fname == "functions" || fname == "function_call" {
                format!("'{fname}' is deprecated by OpenAI; use 'tools' / 'tool_choice' instead")
            } else {
                format!("'{fname}' is not supported by this endpoint")
            };
            return ChatParamsResult::Err(
                (
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        message,
                        oai_type::INVALID_REQUEST,
                        Some(fname),
                        oai_code::UNSUPPORTED_FIELD,
                    )),
                )
                    .into_response(),
            );
        }
    }

    // -- tools / tool_choice / parallel_tool_calls (OpenAI tool calling).
    let tools = match obj.get("tools") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match validate_chat_tools(v) {
            Ok(t) => Some(t),
            Err(resp) => return ChatParamsResult::Err(resp),
        },
    };
    let tool_choice = match obj.get("tool_choice") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match validate_chat_tool_choice(v) {
            Ok(c) => Some(c),
            Err(resp) => return ChatParamsResult::Err(resp),
        },
    };
    let parallel_tool_calls = match obj.get("parallel_tool_calls") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Bool(b)) => Some(*b),
        Some(_) => {
            return bad(
                "'parallel_tool_calls' must be a boolean",
                Some("parallel_tool_calls"),
                oai_code::INVALID_REQUEST,
            );
        }
    };
    // ``tool_choice`` without ``tools`` is meaningless; OpenAI itself
    // returns a 400 in that case. We mirror the behaviour so SDKs that
    // accidentally send ``tool_choice: "auto"`` on a plain chat call
    // see the rejection early rather than at a fictitious worker
    // capability check.
    if tool_choice.is_some() && tools.is_none() {
        return bad(
            "'tool_choice' requires 'tools' to be set",
            Some("tool_choice"),
            oai_code::INVALID_REQUEST,
        );
    }

    // Translate ``response_format`` → :class:`GrammarSpec`.
    // The translator runs after the reject-list so a future
    // ``response_format`` interaction with one of the rejected fields
    // (e.g. ``response_format`` + ``tools``) is still caught by the
    // simpler reject path above.
    let grammar = match obj.get("response_format") {
        None => None,
        Some(v) => match translate_response_format(v) {
            Ok(g) => g,
            Err(resp) => return ChatParamsResult::Err(resp),
        },
    };

    // Forced ``tool_choice`` ("required" / named function) and
    // ``response_format`` both drive constrained decoding on the worker;
    // they would compile two competing grammars onto one request. Reject
    // the combination early with a clear 400 rather than letting the
    // worker silently drop one. ``"auto"`` / ``"none"`` don't force a
    // grammar, so they remain compatible with ``response_format``.
    let tool_choice_forces = match &tool_choice {
        Some(serde_json::Value::String(s)) => s == "required",
        Some(serde_json::Value::Object(_)) => true, // named {type:function,function:{name}}
        _ => false,
    };
    if tool_choice_forces && grammar.is_some() {
        return bad(
            "'tool_choice' (\"required\" or a named function) cannot be combined with \
             'response_format' — both constrain decoding",
            Some("tool_choice"),
            oai_code::INVALID_REQUEST,
        );
    }

    // -- accept-list: any other key not in the known set is rejected.
    const ACCEPTED: &[&str] = &[
        "model",
        "messages",
        "max_completion_tokens",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "top_k",
        "repetition_penalty",
        "min_tokens",
        "chat_template_kwargs",
        "n",
        "best_of",
        "lora_adapter",
        "user",
        "safety_identifier",
        "seed",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "prompt_cache_key",
        "routing_key",
        "response_format",
        "stream",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream_options",
    ];
    for key in obj.keys() {
        if !ACCEPTED.contains(&key.as_str()) {
            return unsupported(key);
        }
    }

    // ``n>1`` is supported both non-streaming (terminal ``candidates[]`` → a
    // multi-entry ``choices`` array) and streaming (the worker fans the
    // candidates out as per-``choice_index`` SSE delta chunks). No reject.

    // ``best_of``: must be >= n, and (mirroring OpenAI) is non-streaming only.
    if let Some(b) = best_of {
        if b > 1 && stream {
            return bad(
                "'best_of' > 1 is not supported with stream:true (mirrors OpenAI)",
                Some("best_of"),
                oai_code::UNSUPPORTED_FIELD,
            );
        }
        if b < n.unwrap_or(1) {
            return bad(
                "'best_of' must be >= 'n'",
                Some("best_of"),
                oai_code::INVALID_REQUEST,
            );
        }
    }

    ChatParamsResult::Ok(ChatRequestParams {
        model,
        messages,
        max_new_tokens,
        temperature,
        top_p,
        stop,
        frequency_penalty,
        presence_penalty,
        top_k,
        repetition_penalty,
        min_tokens,
        chat_template_kwargs,
        routing_key,
        prompt_cache_key,
        grammar,
        stream,
        stream_include_usage,
        tools,
        tool_choice,
        parallel_tool_calls,
        seed,
        logit_bias,
        logprobs,
        top_logprobs,
        n,
        best_of,
        lora_adapter,
    })
}

/// Validate the OpenAI ``tools`` array.
///
/// Spec contract:
///
/// * Must be a non-empty JSON array.
/// * Each entry must be ``{type: "function", function: {name: string,
///   description?: string, parameters?: object}}``.
/// * ``function.parameters`` (when present) is run through the shared
///   JSON-Schema safety walker so the same depth / size / unsupported-
///   keyword caps that apply to ``response_format.json_schema`` also
///   apply here.
///
/// Returns the original `Vec<Value>` (one entry per tool) on success
/// so the caller can plumb it verbatim through the wire envelope —
/// the worker doesn't need a richer Rust-side type today, and a
/// `Vec<Value>` keeps the JSON-shape boundary minimal.
#[allow(clippy::result_large_err)]
fn validate_chat_tools(v: &serde_json::Value) -> Result<Vec<serde_json::Value>, Response> {
    let bad = |msg: String, param: &str, code: &'static str| -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                msg,
                oai_type::INVALID_REQUEST,
                Some(param),
                code,
            )),
        )
            .into_response()
    };
    let Some(arr) = v.as_array() else {
        return Err(bad(
            "'tools' must be an array".to_string(),
            "tools",
            oai_code::INVALID_REQUEST,
        ));
    };
    if arr.is_empty() {
        return Err(bad(
            "'tools' must contain at least one entry".to_string(),
            "tools",
            oai_code::INVALID_REQUEST,
        ));
    }
    // Sanity-cap the array size before walking; an attacker that sneaks
    // past ``MAX_PROXY_BODY`` with a thousand tiny tools would still
    // cost a thousand schema walks. 64 is plenty in practice.
    if arr.len() > 64 {
        return Err(bad(
            format!("'tools' length {} exceeds limit (64)", arr.len()),
            "tools",
            oai_code::INVALID_REQUEST,
        ));
    }
    for (i, tool) in arr.iter().enumerate() {
        let Some(obj) = tool.as_object() else {
            return Err(bad(
                format!("tools[{i}] must be an object"),
                &format!("tools[{i}]"),
                oai_code::INVALID_REQUEST,
            ));
        };
        let kind = obj.get("type").and_then(|v| v.as_str());
        if kind != Some("function") {
            return Err(bad(
                format!("tools[{i}].type must be \"function\""),
                &format!("tools[{i}].type"),
                oai_code::INVALID_REQUEST,
            ));
        }
        let Some(func) = obj.get("function").and_then(|v| v.as_object()) else {
            return Err(bad(
                format!("tools[{i}].function must be an object"),
                &format!("tools[{i}].function"),
                oai_code::INVALID_REQUEST,
            ));
        };
        let Some(name) = func.get("name").and_then(|v| v.as_str()) else {
            return Err(bad(
                format!("tools[{i}].function.name must be a string"),
                &format!("tools[{i}].function.name"),
                oai_code::INVALID_REQUEST,
            ));
        };
        if name.is_empty() {
            return Err(bad(
                format!("tools[{i}].function.name must be non-empty"),
                &format!("tools[{i}].function.name"),
                oai_code::INVALID_REQUEST,
            ));
        }
        if let Some(desc) = func.get("description") {
            if !desc.is_string() && !desc.is_null() {
                return Err(bad(
                    format!("tools[{i}].function.description must be a string"),
                    &format!("tools[{i}].function.description"),
                    oai_code::INVALID_REQUEST,
                ));
            }
        }
        if let Some(params) = func.get("parameters") {
            if !params.is_null() {
                // Re-use the same JSON-Schema safety walker that
                // ``response_format.json_schema`` runs. We wrap the
                // schema in the ``{json_schema: <schema>}`` envelope
                // ``parse_grammar`` expects.
                let wrapped = serde_json::json!({"json_schema": params});
                match super::grammar::parse_grammar(&wrapped) {
                    super::grammar::GrammarParseResult::Ok(_) => {}
                    super::grammar::GrammarParseResult::Err(_) => {
                        return Err(bad(
                            format!(
                                "tools[{i}].function.parameters failed JSON-Schema safety caps"
                            ),
                            &format!("tools[{i}].function.parameters"),
                            oai_code::INVALID_REQUEST,
                        ));
                    }
                }
            }
        }
    }
    Ok(arr.clone())
}

/// Validate OpenAI ``tool_choice``: ``"auto"`` / ``"none"`` /
/// ``"required"`` or ``{type:"function", function:{name}}``. Returns
/// the value unchanged on success so the caller can plumb it verbatim.
#[allow(clippy::result_large_err)]
fn validate_chat_tool_choice(v: &serde_json::Value) -> Result<serde_json::Value, Response> {
    let bad = |msg: String| -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                msg,
                oai_type::INVALID_REQUEST,
                Some("tool_choice"),
                oai_code::INVALID_REQUEST,
            )),
        )
            .into_response()
    };
    if let Some(s) = v.as_str() {
        return match s {
            "auto" | "none" | "required" => Ok(v.clone()),
            other => Err(bad(format!(
                "'tool_choice' string must be one of \"auto\", \"none\", \"required\" — got {other:?}"
            ))),
        };
    }
    let Some(obj) = v.as_object() else {
        return Err(bad("'tool_choice' must be a string or object".to_string()));
    };
    if obj.get("type").and_then(|v| v.as_str()) != Some("function") {
        return Err(bad("'tool_choice.type' must be \"function\"".to_string()));
    }
    let func = match obj.get("function").and_then(|v| v.as_object()) {
        Some(f) => f,
        None => {
            return Err(bad(
                "'tool_choice.function' must be an object with a 'name'".to_string(),
            ));
        }
    };
    let Some(name) = func.get("name").and_then(|v| v.as_str()) else {
        return Err(bad(
            "'tool_choice.function.name' must be a string".to_string()
        ));
    };
    if name.is_empty() {
        return Err(bad(
            "'tool_choice.function.name' must be non-empty".to_string()
        ));
    }
    Ok(v.clone())
}

/// Translate the OpenAI ``response_format`` field into a
/// :class:`GrammarSpec`. Returns ``Ok(None)`` only for the (unusual)
/// case of an explicit JSON ``null``; otherwise the function either
/// produces a grammar or surfaces a 400.
///
/// Supported shape (matches OpenAI's August-2024 Structured Outputs
/// release):
///
/// ```jsonc
/// {
///   "type": "json_schema",
///   "json_schema": {
///     "name": "...",
///     "strict": true,
///     "schema": { ... }
///   }
/// }
/// ```
///
/// ``response_format.type == "json_object"`` (loose JSON without a
/// schema) is accepted and translated to a built-in generic JSON
/// schema (``{"type": "object", "additionalProperties": true}``). This
/// matches OpenAI's documented behaviour: any syntactically-valid JSON
/// object output. The grammar is tagged with ``label = "json_object"``
/// so cache/observability can distinguish it from caller-supplied
/// schemas.
#[allow(clippy::result_large_err)]
fn translate_response_format(
    v: &serde_json::Value,
) -> Result<Option<crate::queue::publisher::GrammarSpec>, Response> {
    let bad = |msg: String, param: &str, code: &'static str| -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                msg,
                oai_type::INVALID_REQUEST,
                Some(param),
                code,
            )),
        )
            .into_response()
    };

    if v.is_null() {
        return Ok(None);
    }
    let Some(obj) = v.as_object() else {
        return Err(bad(
            "'response_format' must be a JSON object".to_string(),
            "response_format",
            oai_code::INVALID_REQUEST,
        ));
    };
    let type_str = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match type_str {
        "json_schema" => {}
        "json_object" => {
            // OpenAI ``json_object`` mode: constrain output to any
            // syntactically-valid JSON object. We back this with a
            // built-in generic schema and the existing ``json_schema``
            // worker path; no schema walker traversal is needed because
            // the schema is trusted-internal. Tag it with
            // ``label = "json_object"`` so cache observability surfaces
            // the loose mode distinctly.
            let generic_schema = json!({
                "type": "object",
                "additionalProperties": true,
            });
            return Ok(Some(crate::queue::publisher::GrammarSpec::JsonSchema {
                value: generic_schema,
                label: Some("json_object".to_string()),
                strict: None,
            }));
        }
        "regex" => {
            // SIE/vLLM-style extension: constrain output to a regex over the
            // vocabulary. Reuse `parse_grammar` (length caps + reject metrics)
            // by handing it the `{regex: ...}` wrapper it already understands.
            let Some(regex) = obj.get("regex").and_then(|r| r.as_str()) else {
                return Err(bad(
                    "'response_format.regex' is required and must be a string".to_string(),
                    "response_format.regex",
                    oai_code::INVALID_REQUEST,
                ));
            };
            let mut wrapped = serde_json::Map::new();
            wrapped.insert("regex".to_string(), json!(regex));
            return match super::grammar::parse_grammar(&serde_json::Value::Object(wrapped)) {
                super::grammar::GrammarParseResult::Ok(g) => Ok(Some(g)),
                super::grammar::GrammarParseResult::Err(resp) => Err(resp),
            };
        }
        "grammar" => {
            // SIE/vLLM-style extension: a context-free grammar. `syntax`
            // selects the surface dialect (`ebnf` default, or `lark`); both
            // map to the `ebnf` sampling param (SGLang's backends accept Lark
            // there too), so we forward the source verbatim via the existing
            // `{ebnf: ...}` wrapper.
            let Some(grammar) = obj.get("grammar").and_then(|g| g.as_str()) else {
                return Err(bad(
                    "'response_format.grammar' is required and must be a string".to_string(),
                    "response_format.grammar",
                    oai_code::INVALID_REQUEST,
                ));
            };
            if let Some(syntax) = obj.get("syntax") {
                let ok = syntax.as_str().is_some_and(|s| s == "ebnf" || s == "lark");
                if !ok {
                    return Err(bad(
                        "'response_format.syntax' must be \"ebnf\" or \"lark\"".to_string(),
                        "response_format.syntax",
                        oai_code::INVALID_REQUEST,
                    ));
                }
            }
            let mut wrapped = serde_json::Map::new();
            wrapped.insert("ebnf".to_string(), json!(grammar));
            return match super::grammar::parse_grammar(&serde_json::Value::Object(wrapped)) {
                super::grammar::GrammarParseResult::Ok(g) => Ok(Some(g)),
                super::grammar::GrammarParseResult::Err(resp) => Err(resp),
            };
        }
        "" => {
            return Err(bad(
                "'response_format.type' is required".to_string(),
                "response_format.type",
                oai_code::INVALID_REQUEST,
            ));
        }
        other => {
            return Err(bad(
                format!("'response_format.type' = {other:?} is not supported"),
                "response_format.type",
                oai_code::UNSUPPORTED_FIELD,
            ));
        }
    }

    let Some(js) = obj.get("json_schema").and_then(|v| v.as_object()) else {
        return Err(bad(
            "'response_format.json_schema' must be an object".to_string(),
            "response_format.json_schema",
            oai_code::INVALID_REQUEST,
        ));
    };
    let Some(schema) = js.get("schema") else {
        return Err(bad(
            "'response_format.json_schema.schema' is required".to_string(),
            "response_format.json_schema.schema",
            oai_code::INVALID_REQUEST,
        ));
    };
    let label = js.get("name").and_then(|v| v.as_str()).map(String::from);
    let strict = js.get("strict").and_then(|v| v.as_bool());

    // Run the schema through the shared safety-cap walker so chat
    // requests cannot bypass depth / size / reject-list limits. The
    // input shape is ``{json_schema: <schema>}`` — same wrapper the
    // SIE-native grammar parser sees — so we synthesise that shape
    // here.
    let mut wrapped = serde_json::Map::new();
    wrapped.insert("json_schema".to_string(), schema.clone());
    if let Some(name) = js.get("name") {
        wrapped.insert("label".to_string(), name.clone());
    }
    if let Some(s) = js.get("strict") {
        wrapped.insert("strict".to_string(), s.clone());
    }
    match super::grammar::parse_grammar(&serde_json::Value::Object(wrapped)) {
        super::grammar::GrammarParseResult::Ok(g) => {
            // ``parse_grammar`` builds a :class:`GrammarSpec` from the
            // wrapped form. Re-attach the OpenAI label/strict in case
            // the wrapper above dropped them (it shouldn't, but
            // defence-in-depth — the strict flag has wire effect).
            match g {
                crate::queue::publisher::GrammarSpec::JsonSchema { value, .. } => {
                    Ok(Some(crate::queue::publisher::GrammarSpec::JsonSchema {
                        value,
                        label,
                        strict,
                    }))
                }
                other => Ok(Some(other)),
            }
        }
        super::grammar::GrammarParseResult::Err(resp) => Err(resp),
    }
}

/// Compose the OpenAI ``chat.completion`` response body from an
/// aggregated :class:`StreamOutcome`.
///
/// Returns ``Err(Response)`` with a 500 ``malformed_worker_response``
/// envelope when the worker's terminal chunk omitted the ``usage``
/// block — the OpenAI envelope requires it, and synthesising one would
/// silently lose accounting accuracy.
#[allow(clippy::result_large_err)]
fn build_chat_completion_body(
    model: &str,
    request_id: &str,
    outcome: &crate::queue::streaming::StreamOutcome,
) -> Result<Vec<u8>, Response> {
    let Some(usage) = outcome.usage.as_ref() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json_openai_error(
                "worker terminal chunk omitted required 'usage' block",
                oai_type::SERVER_ERROR,
                None,
                oai_code::MALFORMED_WORKER_RESPONSE,
            )),
        )
            .into_response());
    };
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // OpenAI logprobs envelope (`{content: [...], refusal: null}`) or null.
    let logprobs_value = |lps: Option<&Vec<Value>>| match lps {
        Some(content) => json!({ "content": content, "refusal": Value::Null }),
        None => Value::Null,
    };
    let choices: Vec<Value> = if outcome.candidates.is_empty() {
        // Single-candidate path (`n == 1`, the default). When the worker
        // collected tool calls, surface them on ``message.tool_calls`` with
        // ``content: null`` (per the OpenAI non-streaming contract); else the
        // assistant text.
        let mut message = serde_json::Map::new();
        message.insert("role".to_string(), json!("assistant"));
        if let Some(tcs) = outcome.tool_calls.as_ref() {
            message.insert("content".to_string(), Value::Null);
            let arr: Vec<Value> = tcs
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": tc.kind,
                        "function": { "name": tc.name, "arguments": tc.arguments }
                    })
                })
                .collect();
            message.insert("tool_calls".to_string(), Value::Array(arr));
        } else {
            message.insert("content".to_string(), json!(outcome.text));
        }
        vec![json!({
            "index": 0,
            "message": Value::Object(message),
            "finish_reason": map_chat_finish_reason(&outcome.finish_reason),
            "logprobs": logprobs_value(outcome.logprobs.as_ref()),
        })]
    } else {
        // Multi-candidate path (`n > 1`): one ``choices[]`` entry per
        // candidate, each with its own text, finish_reason, logprobs, and
        // (H5 non-streaming) tool_calls. ``usage`` is the worker's
        // aggregate (prompt counted once, completion summed). When a
        // candidate's ``tool_calls`` is populated, the OpenAI non-
        // streaming contract says ``message.content`` is ``null`` and
        // ``message.tool_calls`` carries the array.
        outcome
            .candidates
            .iter()
            .enumerate()
            .map(|(i, cand)| {
                let mut message = serde_json::Map::new();
                message.insert("role".to_string(), json!("assistant"));
                if let Some(tcs) = cand.tool_calls.as_ref() {
                    if !tcs.is_empty() {
                        message.insert("content".to_string(), Value::Null);
                        message.insert("tool_calls".to_string(), Value::Array(tcs.clone()));
                    } else {
                        message.insert("content".to_string(), json!(cand.text));
                    }
                } else {
                    message.insert("content".to_string(), json!(cand.text));
                }
                json!({
                    "index": i,
                    "message": Value::Object(message),
                    "finish_reason": map_chat_finish_reason(
                        cand.finish_reason.as_deref().unwrap_or(&outcome.finish_reason),
                    ),
                    "logprobs": logprobs_value(cand.logprobs.as_ref()),
                })
            })
            .collect()
    };
    let body = json!({
        "id": format!("chatcmpl-{}", request_id),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": choices,
        // OpenAI `system_fingerprint`: a stable per-(model, gateway-build)
        // backend-config identifier (see `system_fingerprint`). Present and
        // identical in shape on both the blocking and streaming responses.
        "system_fingerprint": system_fingerprint(model),
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    });
    Ok(serde_json::to_vec(&body).unwrap_or_default())
}

/// Map a SIE-native ``finish_reason`` onto the OpenAI vocabulary.
///
/// SIE-native values include ``stop``, ``length``, ``error``,
/// ``cancelled``. OpenAI accepts ``stop`` | ``length`` | ``tool_calls``
/// | ``content_filter`` | ``function_call``. Unknown SIE values map to
/// ``stop`` so SDKs that strictly validate the enum still parse the
/// response.
fn map_chat_finish_reason(sie: &str) -> &'static str {
    match sie {
        "length" => "length",
        // ``tool_calls`` is a first-class OpenAI finish reason emitted
        // when the model produced a tool-call delta stream; pass it
        // through unchanged so SDKs route to their function-calling
        // branch.
        "tool_calls" => "tool_calls",
        // ``content_filter`` and ``function_call`` are valid OpenAI
        // finish reasons the worker can emit (see
        // ``queue::streaming::is_known_finish_reason``). Pass them
        // through so a safety-stopped completion is not silently
        // reported as a clean ``stop``.
        "content_filter" => "content_filter",
        "function_call" => "function_call",
        // ``stop``, ``cancelled``, ``error`` all collapse to ``stop`` —
        // OpenAI does not have a distinct error/cancelled finish
        // reason on the success body; non-success outcomes never reach
        // this branch.
        _ => "stop",
    }
}

/// Resolve a request's ``model`` string into the canonical model id + its
/// bundle, applying the registry rules and OpenAI-shaped error envelopes.
/// Shared by `/v1/chat/completions` and `/v1/completions`.
// The Err variant is a fully-built HTTP error Response (large by nature); these
// resolution helpers return it for the caller to `return` directly, mirroring
// the handlers' own pattern.
#[allow(clippy::result_large_err)]
pub(crate) fn resolve_model_and_bundle(
    state: &AppState,
    model_spec: &str,
    ext: &axum::http::Extensions,
) -> Result<(String, String), Response> {
    let (bundle_override, model_name) =
        resolve_model_spec_with_aliases(&state.config.model_aliases, model_spec, |m| {
            state.model_registry.resolve_canonical_model_name(m)
        });
    let bundle_override_ref = if bundle_override.is_empty() {
        None
    } else {
        Some(bundle_override.as_str())
    };
    // #1841 org-scoped visibility (mirrors resolve_routing for the OpenAI body
    // routes): a custom model hidden from this caller is treated as absent — the
    // identical MODEL_NOT_FOUND 404 the block below emits, echoing the CALLER's
    // string under the same template so neither the id nor the wording can tell
    // "hidden from you" apart from "does not exist" (#2542). Gates the RESOLVED
    // id. No-op in OSS (policy None).
    if state
        .model_access_policy
        .as_ref()
        .is_some_and(|p| !p.visible(&model_name, ext))
    {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json_openai_error(
                format!("Model '{model_spec}' not found"),
                oai_type::MODEL_NOT_FOUND,
                Some("model"),
                oai_code::MODEL_NOT_FOUND,
            )),
        )
            .into_response());
    }
    // #2542 deployment serving policy on the RESOLVED id (mirrors `resolve_routing`
    // for the OpenAI body routes): the authoritative refusal for a model this
    // deployment does not serve. Decided post-alias/`__`/case, so a registry alias
    // pointing at a refused model cannot slip past an edge gate that only ever saw
    // the caller's raw string. After `visible` (the no-oracle 404 wins), before the
    // sealed branch. No-op in OSS (policy None).
    if let Some(refusal) = state
        .model_access_policy
        .as_ref()
        .and_then(|p| p.serving_refusal(&model_name, ext))
    {
        return Err(refusal);
    }
    // #1841 sealed dispatch (OpenAI body routes): an org-registered custom model
    // has no catalog bundle, so short-circuit with the synthetic sealed bundle
    // rather than dead-ending at MODEL_NOT_FOUND below. The engine/pool override
    // happens in `resolve_generation_route`, which re-consults the same policy on
    // the same id. Ownership is already proven by `visible()` above.
    if let Some(route) = state
        .model_access_policy
        .as_ref()
        .and_then(|p| p.sealed_route(&model_name, ext))
    {
        return Ok((model_name, route.bundle));
    }
    let bundle = if state.model_registry.model_exists(&model_name) {
        match state
            .model_registry
            .resolve_bundle(&model_name, bundle_override_ref)
        {
            Ok(b) => b,
            Err(ResolveError::ModelNotFound(e)) => {
                return Err((
                    StatusCode::NOT_FOUND,
                    Json(json_openai_error(
                        e.to_string(),
                        oai_type::MODEL_NOT_FOUND,
                        Some("model"),
                        oai_code::MODEL_NOT_FOUND,
                    )),
                )
                    .into_response());
            }
            Err(ResolveError::BundleConflict(e)) => {
                return Err((
                    StatusCode::CONFLICT,
                    Json(json_openai_error(
                        e.to_string(),
                        oai_type::INVALID_REQUEST,
                        Some("model"),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response());
            }
        }
    } else if state.model_registry.has_any_models() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json_openai_error(
                // The caller's string, never the resolved one — see the
                // visibility gate above (#2542).
                format!("Model '{model_spec}' not found"),
                oai_type::MODEL_NOT_FOUND,
                Some("model"),
                oai_code::MODEL_NOT_FOUND,
            )),
        )
            .into_response());
    } else if bundle_override.is_empty() {
        "default".to_string()
    } else {
        bundle_override.clone()
    };
    Ok((model_name, bundle))
}

/// The routing context resolved from request headers + the model's bundle:
/// machine profile, effective pool, bundle config hash, the bound work
/// publisher, plus audit fields. Produced by [`resolve_generation_route`].
struct ResolvedRoute {
    physical_lane: PhysicalLane,
    bundle: String,
    gpu: String,
    engine: String,
    admission_pool: String,
    effective_pool: String,
    effective_machine_profile: String,
    bundle_config_hash: String,
    model_revision: Option<String>,
    work_publisher: Arc<dyn WorkDispatcher>,
    token_id: String,
    content_length: i64,
}

/// Resolve GPU/pool routing from headers, validate them, pick the effective
/// pool (or surface an OpenAI-shaped 503 provisioning response), and bind the
/// work publisher. Shared by the chat + completions handlers; the OpenAI-shaped
/// errors are identical across both.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
async fn resolve_generation_route(
    state: &AppState,
    hdr: &HeaderMap,
    bundle: &str,
    customer_model: &str,
    dispatch_model: &str,
    request_intent: GenerationRequestIntent,
    explicit_bundle_override: &str,
    ext: &axum::http::Extensions,
    metric_labels_slot: Option<&telemetry::MetricLabelsSlot>,
) -> Result<ResolvedRoute, Response> {
    // #1841 sealed dispatch: an org-registered custom model routes to its own
    // sealed sandbox, not a catalog bundle. Force the synthetic (gpu, pool) =
    // ("sealed", "sealed") so `resolve_effective_pool` below matches the registered
    // sealed worker lane (header-based profile/pool resolution would pick a catalog
    // pool), and pin `engine = "sealed"` (the dispatcher's sole routing
    // discriminator) instead of the bundle-derived engine — a bundle-less "sealed"
    // would otherwise fall back to DEFAULT_ENGINE and mis-route to the shared
    // catalog lane, a silent isolation failure. Ownership is already proven by the
    // visibility gate in `resolve_model_and_bundle`; re-consulting the same policy
    // on the same id is a pure, cheap id-shape check.
    let sealed = state
        .model_access_policy
        .as_ref()
        .and_then(|p| p.sealed_route(customer_model, ext));
    let governed = if sealed.is_none() {
        governed_generation_route(state, customer_model, dispatch_model, request_intent)?
    } else {
        None
    };
    if let Some(route) = governed.as_ref() {
        if !explicit_bundle_override.is_empty() && explicit_bundle_override != route.bundle {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    format!(
                        "Bundle override '{}' conflicts with the deployment route for this model",
                        explicit_bundle_override
                    ),
                    oai_type::INVALID_REQUEST,
                    Some("model"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response());
        }
        match state
            .model_registry
            .resolve_bundle(&route.model, Some(&route.bundle))
        {
            Ok(resolved) if resolved == route.bundle => {}
            _ => {
                return Err(governed_generation_route_failure(
                    customer_model,
                    request_intent,
                    "compiled bundle is incompatible with the physical model",
                ));
            }
        }
    }
    let bundle = governed
        .as_ref()
        .map_or_else(|| bundle.to_string(), |route| route.bundle.clone());
    // Machine-profile + pool resolution, shared with the primitive path via
    // `resolve_profile_and_pool`. Errors are rendered here in the OpenAI
    // envelope (`json_openai_error`), preserving the compat surface.
    // #1841: a sealed custom-model route carries its own machine profile + pool,
    // so it short-circuits catalog profile/pool resolution entirely.
    let (gpu, pool_name, gpu_configured) = if let Some(route) = &sealed {
        (route.machine_profile.clone(), route.pool.clone(), true)
    } else if let Some(route) = governed.as_ref() {
        let route = governed_profile_and_pool(state, hdr, route)?;
        (route.gpu, route.pool_name, route.gpu_configured)
    } else {
        match resolve_profile_and_pool(state, hdr, customer_model).await {
            Ok(ProfilePoolRoute {
                gpu,
                pool_name,
                gpu_configured,
            }) => (gpu, pool_name, gpu_configured),
            Err(ProfilePoolError::InvalidHeader { header }) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        format!("{header} header must be valid UTF-8"),
                        oai_type::INVALID_REQUEST,
                        Some(header),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response());
            }
            Err(ProfilePoolError::InvalidPool) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        "Invalid pool name: only [A-Za-z0-9_-] are allowed (max 128 chars)"
                            .to_string(),
                        oai_type::INVALID_REQUEST,
                        Some("X-SIE-Pool"),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response());
            }
            Err(ProfilePoolError::PoolMismatch { message }) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        message,
                        oai_type::INVALID_REQUEST,
                        Some("X-SIE-Pool"),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response());
            }
        }
    };

    if !gpu_configured {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json_openai_error(
                format!("GPU type '{gpu}' is not configured in this cluster."),
                oai_type::SERVER_ERROR,
                Some("X-SIE-MACHINE-PROFILE"),
                oai_code::TRANSPORT_FAILURE,
            )),
        )
            .into_response());
    }

    let Some(work_publisher_arc) = state.work_publisher.clone() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json_openai_error(
                "Rust gateway is queue-only, but NATS JetStream is unavailable",
                oai_type::SERVER_ERROR,
                None,
                oai_code::TRANSPORT_FAILURE,
            )),
        )
            .into_response());
    };

    let Some(hash_pool) = queue_pool_for_request(Some(&state.pool_manager), &pool_name).await
    else {
        let requested_pool = normalize_pool_name(&pool_name);
        return Err(build_pool_not_found_response_for_surface(
            &requested_pool,
            ProvisioningSurface::OpenAiCompat,
        ));
    };
    let (bundle_config_hash, model_revision, uses_catalog_scope) = state
        .model_registry
        .bundle_execution_evidence(&bundle, &hash_pool, dispatch_model);
    if !catalog_execution_hash_is_ready(&bundle_config_hash, uses_catalog_scope) {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json_openai_error(
                "Model execution evidence is unavailable while configuration converges",
                oai_type::SERVER_ERROR,
                None,
                oai_code::TRANSPORT_FAILURE,
            )),
        )
            .into_response());
    }
    // #1841: pin the sealed engine marker (see the top-of-fn note); a bundle-less
    // "sealed" bundle has no BundleInfo, so the else-branch would yield DEFAULT_ENGINE.
    let engine = match &sealed {
        Some(route) => route.engine.clone(),
        None => state
            .model_registry
            .get_bundle_info(&bundle)
            .map(|info| info.engine)
            .unwrap_or_else(|| crate::types::bundle::DEFAULT_ENGINE.to_string()),
    };
    let lookup = resolve_effective_pool(
        &state.registry,
        Some(&state.pool_manager),
        &bundle,
        &gpu,
        &pool_name,
        &bundle_config_hash,
    )
    .await;
    let demand_pool = lookup.demand_pool.clone();
    let admission_pool = lookup.admission_pool.clone();
    let pending_demand_profiles = lookup.pending_demand_profiles.clone();
    for profile in &pending_demand_profiles {
        if let Some(lane) = state
            .demand_tracker
            .resolve_lane(&demand_pool, profile, &bundle)
        {
            state.demand_tracker.record(&lane);
        }
    }
    let effective_route = match lookup.resolution {
        PoolResolution::Route(route) => route,
        PoolResolution::PoolNotFound(pool) => {
            return Err(build_pool_not_found_response_for_surface(
                &pool,
                ProvisioningSurface::OpenAiCompat,
            ));
        }
        PoolResolution::Provisioning => {
            return Err(build_openai_provisioning_response(&gpu, &bundle));
        }
    };
    let effective_pool = effective_route.pool_name;
    let effective_machine_profile = effective_route.machine_profile;
    let Some(physical_lane) =
        state
            .demand_tracker
            .resolve_lane(&demand_pool, &effective_machine_profile, &bundle)
    else {
        error!(
            pool = %demand_pool,
            machine_profile = %effective_machine_profile,
        bundle = %bundle,
            "resolved generation route is absent from configured physical KEDA lane catalog"
        );
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json_openai_error(
                "Resolved worker lane is not configured for autoscaling".to_string(),
                oai_type::SERVER_ERROR,
                None,
                oai_code::TRANSPORT_FAILURE,
            )),
        )
            .into_response());
    };

    // The HTTP metrics middleware installed this shared slot before the handler
    // consumed the request body. Fill it after canonical routing so all three
    // OpenAI generation surfaces report the profile actually used. Rejections
    // before a route exists intentionally remain `unknown`.
    if let Some(slot) = metric_labels_slot {
        slot.set(telemetry::MetricLabels {
            machine_profile: effective_machine_profile.clone(),
            model: customer_model.to_string(),
        });
    }

    if let Some(resp) = capped_lane_admission_response(
        state,
        &admission_pool,
        &demand_pool,
        &effective_machine_profile,
        &bundle,
        ProvisioningSurface::OpenAiCompat,
    )
    .await
    {
        return Err(resp);
    }

    let token_id = extract_bearer_token(hdr)
        .map(|t| mask_token(&t))
        .unwrap_or_default();
    let content_length = hdr
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(-1);

    Ok(ResolvedRoute {
        physical_lane,
        bundle,
        gpu,
        engine,
        admission_pool,
        effective_pool,
        effective_machine_profile,
        bundle_config_hash,
        model_revision,
        work_publisher: work_publisher_arc,
        token_id,
        content_length,
    })
}

/// Rewrite a grammar-constrained request's dispatch model id to the model's
/// declared ``grammar_profile`` variant (e.g. ``no-spec``), in place.
///
/// Shared by every grammar-capable surface (``/v1/chat/completions`` and the
/// native ``/v1/generate`` path) so they route identically. NEXTN/MTP
/// speculative decoding bypasses SGLang's Outlines grammar FSM (leaks
/// out-of-schema keys, truncates mid-JSON), so a model whose default profile is
/// speculative points ``grammar_profile`` at a non-speculative profile. The
/// target is resolved off the request's *base* model
/// (``ModelRegistry::grammar_route_variant``). An explicit sibling variant may
/// name a profile-scoped non-speculative twin that preserves its context,
/// hardware launch shape, and thinking mode; otherwise it uses the model-wide
/// fallback. The grammar-safe variant itself is a no-op (same bundle/pool/lane
/// — only the worker-loaded model changes). Call this BEFORE the capability/LoRA
/// gates and resolve those gates against the routed variant: the variant
/// preserves the base capabilities but narrows the profile-scoped LoRA
/// allow-list, so a gate run against the base would accept an adapter the
/// dispatched profile does not serve. The rewritten (DISPATCH) id governs only
/// the NATS subject + work item; the caller keeps the requested (DISPLAY) id for
/// the response body, success metrics, and audit log. Invoke ONLY when a grammar
/// is present. Degrades to the requested id when no profile is declared or the
/// variant is absent (the cluster still serves on the requested profile — never
/// hang/5xx).
///
/// Both grammar-capable surfaces (``/v1/generate`` and ``/v1/chat/completions``)
/// now route *before* their capability/LoRA gates so those validate against the
/// routed variant. This is safe because grammar capabilities are model/task-level
/// (identical across a model's profiles), so the gate verdict is the same on the
/// base and the routed variant.
/// `pub` so the managed cost-estimate dry run (#2435) applies the SAME rewrite
/// before rating. `profile` is a rate-key dimension and this rewrite decides it
/// for every grammar-constrained generation, so an estimate that skipped it
/// would quote `profile="default"` for a request admission holds under
/// `profile="{grammar_profile}"` — a different price, or a confident 200 for a
/// request the live path finds unpriced.
pub fn route_grammar_to_profile(
    registry: &crate::state::model_registry::ModelRegistry,
    model: &mut String,
) {
    use crate::state::model_registry::GrammarRoute;
    match registry.grammar_route_variant(model) {
        GrammarRoute::Rewrite(variant) => {
            tracing::info!(
                base = %model,
                variant = %variant,
                "routing grammar-constrained request to non-speculative profile variant"
            );
            *model = variant;
        }
        GrammarRoute::Keep => {}
        GrammarRoute::MissingVariant(grammar_profile) => {
            tracing::warn!(
                base = %model,
                grammar_profile = %grammar_profile,
                "declared grammar_profile variant not in registry; serving grammar on the requested profile"
            );
        }
    }
}

#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    tag = "inference",
    description = "OpenAI-compatible chat completions. This surface supports the blocking, \
                   non-streaming subset of the OpenAI Chat Completions API. \
                   See the ``messages``-shaped /v1/generate work-item shape for the \
                   underlying contract.",
    request_body = crate::openapi::ChatCompletionRequest,
    params(
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    responses(
        (status = 200, description = "Chat completion response", body = crate::openapi::ChatCompletionResponse),
        (status = 400, description = "Invalid or unsupported request", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 404, description = "Model not found", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 413, description = "Request body is too large", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 500, description = "Worker emitted malformed response", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 503, description = "Provisioning in progress, queue unavailable, or model loading", body = crate::openapi::OpenAIErrorEnvelope),
    )
)]
pub async fn proxy_chat(State(state): State<Arc<AppState>>, req: Request) -> Response {
    check_sdk_version(req.headers());
    let metric_labels_slot = req
        .extensions()
        .get::<telemetry::MetricLabelsSlot>()
        .cloned();

    // M5: extract the inbound W3C trace context and open a
    // gateway-side span as its child. Poll-scoped instrumentation keeps
    // the span current while the handler runs so `publish_generate_streaming`
    // picks up *this* span's context when it calls
    // `inject_current_context` to populate the work envelope.
    //
    // When no `traceparent` header is present the extracted context
    // is empty and the span we open becomes a new trace root. Either
    // way the envelope carries a valid traceparent the worker can
    // continue. We attach via `tracing-opentelemetry`'s set_parent
    // so the structured logs emitted on the existing `tracing::*`
    // call-sites become part of the same OTel span tree.
    let parent_cx = managed_request_parent(&req);
    let chat_span = tracing::info_span!(
        "gateway.proxy_chat",
        http.route = "/v1/chat/completions",
        sie.routing_key_kind = tracing::field::Empty,
        sie.model = tracing::field::Empty,
        sie.request_id = tracing::field::Empty,
    );
    {
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        let _ = chat_span.set_parent(parent_cx);
    }
    // Poll-scoped instrumentation is thread-hop-safe. A guard returned by
    // `Span::enter()` must not cross the handler's awaits because Tokio may
    // resume the future on a different worker thread.
    async move { proxy_chat_inner(state, req, metric_labels_slot).await }
        .instrument(chat_span)
        .await
}

async fn proxy_chat_inner(
    state: Arc<AppState>,
    req: Request,
    metric_labels_slot: Option<telemetry::MetricLabelsSlot>,
) -> Response {
    let max_chat_body = compat_request_body_limit("/v1/chat/completions");
    let hdr = req.headers().clone();
    let (parts, body) = req.into_parts(); // keep extensions (caller identity) for the #1841 gate
    let body_bytes = match to_bytes(body, max_chat_body).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(json_openai_error(
                    format!("request body too large: {e}"),
                    oai_type::INVALID_REQUEST,
                    None,
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response();
        }
    };

    let body_json: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    format!("invalid JSON: {e}"),
                    oai_type::INVALID_REQUEST,
                    None,
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response();
        }
    };

    let params = match chat_params_from_json(&body_json) {
        ChatParamsResult::Ok(p) => p,
        ChatParamsResult::Err(resp) => {
            return resp;
        }
    };

    // -- model registry resolution (mirrors proxy_request, but the
    //    model comes from the body rather than the path). Shared with
    //    /v1/completions via resolve_model_and_bundle.
    let (model_name, bundle) =
        match resolve_model_and_bundle(&state, &params.model, &parts.extensions) {
            Ok(mb) => mb,
            Err(resp) => return resp,
        };
    let (explicit_bundle_override, _) = parse_model_spec(&params.model);

    // Grammar routing (follow-up: chat-path gate ordering). Resolve the model's
    // declared ``grammar_profile`` variant and compute the DISPATCH id BEFORE the
    // capability/LoRA gates below, so the gates validate against — and the work
    // item dispatches to — the profile the request actually runs on. The
    // profile-scoped LoRA allow-list is narrowed per variant, so gating the base
    // would accept an adapter the ``:no-spec`` profile does not serve. The
    // DISPLAY id (``model_name``) stays the requested model for the response
    // body, success metrics, and audit log. Equal to ``model_name`` unless a
    // grammar is present and a variant exists. See ``route_grammar_to_profile``.
    let mut dispatch_model = model_name.clone();
    if params.grammar.is_some() {
        route_grammar_to_profile(&state.model_registry, &mut dispatch_model);
    }

    // -- per-request max_output_tokens cap from model config + grammar capability.
    //    Gates resolve against the routed DISPATCH model so the profile-scoped
    //    LoRA allow-list and capabilities match the profile that will serve.
    if let Some(info) = state.model_registry.get_model_info(&dispatch_model) {
        if let Some(response) = unsupported_streaming_response(
            &state.model_registry,
            &dispatch_model,
            &model_name,
            params.stream,
        ) {
            return response;
        }
        // Multi-LoRA: pre-validate the requested adapter against the
        // *selected profile's* advertised served-names — not the union
        // across profiles. The chat path has no explicit profile
        // parameter; profile selection happens via the model spec
        // (``model:profile`` resolves to a variant entry whose
        // ``profile_configs`` only holds the resolved profile under
        // ``"default"``), so looking up ``"default"`` here works for
        // both the base entry (default profile) and any variant entry
        // (which was narrowed at construction time). Closes M10:
        // previously a chat request for model ``M`` (profile
        // ``default``) could pass an adapter that was only configured
        // for profile ``a100``, then fail opaquely on the worker.
        if let Some(req_lora) = params.lora_adapter.as_deref() {
            match validate_lora_for_profile(&info, "default", req_lora) {
                LoraValidation::Ok => {}
                // ``"default"`` is always considered a known profile by
                // ``validate_lora_for_profile`` (workers synthesize a
                // default at load time), so ``UnknownProfile`` cannot
                // be returned on the chat path — collapse both
                // rejection arms into the same ``unknown_lora_adapter``
                // response that's been the chat-gate contract since A.
                LoraValidation::UnknownProfile | LoraValidation::UnknownAdapter => {
                    return (
                        StatusCode::BAD_REQUEST,
                        Json(json_openai_error(
                            format!("unknown lora_adapter '{req_lora}' for model '{model_name}'"),
                            oai_type::INVALID_REQUEST,
                            Some("lora_adapter"),
                            oai_code::UNKNOWN_LORA_ADAPTER,
                        )),
                    )
                        .into_response();
                }
            }
        }
        if let Some(cap) = info.effective_max_output_tokens() {
            if cap > 0 && params.max_new_tokens > cap {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        format!(
                            "max_completion_tokens ({}) exceeds model max_output_tokens ({cap})",
                            params.max_new_tokens
                        ),
                        oai_type::INVALID_REQUEST,
                        Some("max_completion_tokens"),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response();
            }
        }
        if let Some(g) = params.grammar.as_ref() {
            if let Err(resp) = super::grammar::check_capability(
                g,
                info.info_extras.grammar_capabilities.as_deref(),
                &model_name,
            ) {
                return resp;
            }
        }
        // Tool-calling capability gate. ``tools`` requires
        // ``tasks.generate.capabilities.tools: true`` on the model
        // YAML; older models without that flag (or non-generation
        // models) reject with 400 ``unsupported_field`` so SDKs see a
        // clear failure rather than a silent passthrough that
        // produces garbled output.
        if params.tools.is_some() {
            let supported = info.info_extras.tools_supported.unwrap_or(false);
            if !supported {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        format!("Model '{model_name}' does not support tool calling"),
                        oai_type::INVALID_REQUEST,
                        Some("tools"),
                        oai_code::UNSUPPORTED_FIELD,
                    )),
                )
                    .into_response();
            }
        }
        // Vision capability gate. ``image_url`` content parts were decoded
        // into ``ChatMessage.images`` capability-agnostically (the model
        // wasn't resolved yet); reject here unless the model YAML declares
        // ``inputs.image: true``. Mirrors the grammar validation gates.
        let has_images = params
            .messages
            .iter()
            .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()));
        if has_images {
            // Vision *generation* requires both image input AND a generation
            // task — ``inputs.image`` alone is also set by encode-only image
            // models (CLIP/SigLIP), which must not accept a chat image request.
            let image_supported = info.info_extras.supports_vision_generation();
            if !image_supported {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        format!("Model '{model_name}' does not support image input"),
                        oai_type::INVALID_REQUEST,
                        Some("messages"),
                        oai_code::UNSUPPORTED_FIELD,
                    )),
                )
                    .into_response();
            }
        }
    } else if params.grammar.is_some() {
        // No model info means we cannot determine grammar capabilities;
        // safer to reject than to publish work the model cannot honour.
        return (
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                format!("Model '{model_name}' does not support grammar (no model info)"),
                oai_type::INVALID_REQUEST,
                Some("response_format"),
                oai_code::UNSUPPORTED_FIELD,
            )),
        )
            .into_response();
    } else if params.tools.is_some() {
        // Same defensive rejection for ``tools`` when the model is
        // unknown — safer than publishing work the model cannot
        // honour.
        return (
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                format!("Model '{model_name}' does not support tool calling (no model info)"),
                oai_type::INVALID_REQUEST,
                Some("tools"),
                oai_code::UNSUPPORTED_FIELD,
            )),
        )
            .into_response();
    } else if params
        .messages
        .iter()
        .any(|m| m.images.as_ref().is_some_and(|imgs| !imgs.is_empty()))
    {
        // Same defensive rejection for image input when the model is
        // unknown — safer than publishing vision work the model can't honour.
        return (
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                format!("Model '{model_name}' does not support image input (no model info)"),
                oai_type::INVALID_REQUEST,
                Some("messages"),
                oai_code::UNSUPPORTED_FIELD,
            )),
        )
            .into_response();
    }

    // -- headers → GPU/pool routing, effective-pool selection, publisher bind.
    //    Shared with /v1/completions via resolve_generation_route.
    let ResolvedRoute {
        physical_lane,
        bundle,
        gpu,
        engine,
        admission_pool,
        effective_pool,
        effective_machine_profile,
        bundle_config_hash,
        model_revision,
        work_publisher: work_publisher_arc,
        token_id,
        content_length,
    } = match resolve_generation_route(
        &state,
        &hdr,
        &bundle,
        &model_name,
        &dispatch_model,
        if params.grammar.is_some() {
            GenerationRequestIntent::Grammar
        } else {
            GenerationRequestIntent::Default
        },
        &explicit_bundle_override,
        &parts.extensions,
        metric_labels_slot.as_ref(),
    )
    .await
    {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    let start = Instant::now();

    // Copied out BEFORE the params are consumed into `WorkParams` below.
    let stream = params.stream;
    let stream_include_usage = params.stream_include_usage;
    let work_params = params.into_work_params();

    // SSE branch — when `stream: true` we hand off to the SSE
    // response builder. The non-streaming aggregating path below is
    // untouched. The SSE builder uses the same streaming pipeline
    // (chunk envelopes on `_INBOX.{router_id}.{request_id}`,
    // `StreamCollector`, cancel guard, first-chunk-timeout
    // republish-to-pool) but installs a broadcast tap on the
    // collector so each chunk is forwarded to the HTTP client as it
    // arrives instead of being aggregated.
    if stream {
        return super::sse::build_sse_response(super::sse::SseParams {
            state: state.as_ref(),
            work_publisher: work_publisher_arc,
            physical_lane: physical_lane.clone(),
            model: model_name.clone(),
            dispatch_model: dispatch_model.clone(),
            bundle: bundle.clone(),
            engine: engine.clone(),
            gpu: effective_machine_profile.clone(),
            pool: effective_pool.clone(),
            admission_pool: admission_pool.clone(),
            bundle_config_hash: bundle_config_hash.clone(),
            work_params,
            endpoint: super::sse::SseEndpoint::Chat {
                include_usage: stream_include_usage,
            },
        })
        .await;
    }

    let driver = run_streaming_generate(
        &state,
        work_publisher_arc,
        &physical_lane,
        &model_name,
        &dispatch_model,
        &bundle,
        &engine,
        &effective_machine_profile,
        &effective_pool,
        &admission_pool,
        &bundle_config_hash,
        &work_params,
    )
    .await;

    let StreamingDriverOk {
        outcome,
        request_id,
        publish_elapsed,
        wait_elapsed,
    } = match driver {
        Ok(ok) => ok,
        Err(err) => return build_streaming_error_response(&err),
    };
    let elapsed = start.elapsed();

    let body_bytes = match build_chat_completion_body(&model_name, &request_id, &outcome) {
        Ok(b) => b,
        Err(resp) => return resp,
    };

    state.registry.record_request("queue").await;
    emit_audit_log(AuditEntry {
        event: "proxy_request".to_string(),
        method: "POST".to_string(),
        endpoint: "chat".to_string(),
        status: 200,
        token_id,
        model: model_name.clone(),
        pool: effective_pool.clone(),
        gpu: gpu.clone(),
        worker: format!("queue:{request_id}"),
        latency_ms: elapsed.as_millis() as u64,
        body_bytes: content_length,
    });

    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(Body::from(body_bytes))
        .unwrap();
    response.headers_mut().insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    insert_stream_model_revision_header(
        response.headers_mut(),
        model_revision.as_deref(),
        &bundle_config_hash,
        &outcome,
    );
    insert_stream_execution_identity_header(response.headers_mut(), &outcome);
    insert_stream_execution_binding_header(response.headers_mut(), &outcome);
    response.headers_mut().insert(
        HeaderName::from_static("x-sie-request-id"),
        HeaderValue::from_str(&request_id).unwrap_or_else(|_| HeaderValue::from_static("")),
    );
    insert_duration_header(
        response.headers_mut(),
        "x-queue-publish-time",
        publish_elapsed,
    );
    insert_duration_header(response.headers_mut(), "x-queue-wait-time", wait_elapsed);
    if let Ok(val) = HeaderValue::from_str(&format!("queue:{request_id}")) {
        response
            .headers_mut()
            .insert(HeaderName::from_static("x-sie-worker"), val);
    }
    response
}

/// Parsed, validated `/v1/completions` (legacy OpenAI Completions) request.
/// Raw-prompt continuation — no chat template — via `GenerateInput::Prompt`.
struct CompletionsParams {
    model: String,
    prompt: String,
    max_new_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stop: Option<Vec<String>>,
    frequency_penalty: Option<f64>,
    presence_penalty: Option<f64>,
    seed: Option<i64>,
    stream: bool,
    /// OpenAI ``stream_options.include_usage`` — when ``true`` the SSE stream
    /// emits a trailing usage-only ``text_completion`` chunk before ``[DONE]``.
    /// Unknown / non-boolean ``stream_options`` are rejected by the parser.
    stream_include_usage: bool,
}

impl CompletionsParams {
    /// The EXACT `WorkParams` a validated `/v1/completions` body dispatches
    /// with. Extracted (not copied) for the same reason as
    /// [`ChatRequestParams::into_work_params`] — see #2435.
    fn into_work_params(self) -> publisher::WorkParams {
        publisher::WorkParams {
            generate: Some(publisher::GenerateParams {
                input: publisher::GenerateInput::Prompt {
                    prompt: self.prompt,
                },
                max_new_tokens: self.max_new_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                stop: self.stop,
                frequency_penalty: self.frequency_penalty,
                presence_penalty: self.presence_penalty,
                seed: self.seed,
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

enum CompletionsParamsResult {
    Ok(CompletionsParams),
    Err(Response),
}

/// The `(model, WorkParams)` a `/v1/completions` body dispatches with. See
/// [`chat_completions_dispatch_inputs`], including the `dead_code` note.
#[allow(clippy::result_large_err, dead_code)]
pub fn completions_dispatch_inputs(
    body: &serde_json::Value,
) -> Result<(String, publisher::WorkParams), Response> {
    match completions_params_from_json(body) {
        CompletionsParamsResult::Ok(params) => {
            let model = params.model.clone();
            Ok((model, params.into_work_params()))
        }
        CompletionsParamsResult::Err(response) => Err(response),
    }
}

/// Parse + validate a `/v1/completions` body. MVP scope: a single `prompt`
/// string (or 1-element array), non-streaming, single-candidate; `echo`,
/// `suffix`, `logprobs`, `best_of` are out of scope. Errors use the OpenAI
/// envelope so SDKs surface a precise `param`/`code`.
///
/// Validation is strict (H3 hardening): unrecognized top-level fields,
/// known-out-of-scope fields (``echo`` / ``suffix`` / ``logprobs`` /
/// ``best_of``), and present-but-wrong-type sampler / token-cap values
/// all surface as 400 rather than being silently coerced.
fn completions_params_from_json(body: &serde_json::Value) -> CompletionsParamsResult {
    let bad = |msg: &str, param: Option<&str>, code: &'static str| {
        CompletionsParamsResult::Err(
            (
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    msg.to_string(),
                    oai_type::INVALID_REQUEST,
                    param,
                    code,
                )),
            )
                .into_response(),
        )
    };
    let Some(obj) = body.as_object() else {
        return bad(
            "request body must be a JSON object",
            None,
            oai_code::INVALID_REQUEST,
        );
    };

    // -- explicit reject-list: known-out-of-scope OpenAI fields. Surface
    //    these as ``unsupported_field`` so SDKs route on a stable code
    //    rather than discovering silent drops empirically.
    for &fname in &["echo", "suffix", "logprobs", "best_of"] {
        if obj.contains_key(fname) {
            return bad(
                &format!("'{fname}' is not supported by this endpoint"),
                Some(fname),
                oai_code::UNSUPPORTED_FIELD,
            );
        }
    }

    let model = match obj.get("model").and_then(|v| v.as_str()) {
        Some(m) if !m.is_empty() => m.to_string(),
        _ => {
            return bad(
                "field \"model\" is required",
                Some("model"),
                oai_code::INVALID_REQUEST,
            )
        }
    };

    let prompt = match obj.get("prompt") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Array(a)) if a.len() == 1 => match a[0].as_str() {
            Some(s) => s.to_string(),
            None => {
                return bad(
                    "'prompt' array entries must be strings",
                    Some("prompt"),
                    oai_code::INVALID_REQUEST,
                )
            }
        },
        Some(serde_json::Value::Array(_)) => {
            return bad(
                "batched array prompts are not supported; send one prompt string",
                Some("prompt"),
                oai_code::UNSUPPORTED_FIELD,
            );
        }
        _ => {
            return bad(
                "field \"prompt\" is required and must be a string",
                Some("prompt"),
                oai_code::INVALID_REQUEST,
            )
        }
    };

    // ``stream`` must be a boolean when present; non-bool values must not
    // silently coerce to ``false`` (H3).
    let stream = match obj.get("stream") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(_) => {
            return bad(
                "'stream' must be a boolean",
                Some("stream"),
                oai_code::INVALID_REQUEST,
            );
        }
    };
    // ``stream_options.include_usage`` — same contract as chat: the only
    // recognised key, boolean, and OpenAI-legal (ignored) with ``stream:false``.
    let mut stream_include_usage = false;
    if let Some(opts) = obj.get("stream_options") {
        if !opts.is_null() {
            let Some(opts_obj) = opts.as_object() else {
                return bad(
                    "'stream_options' must be a JSON object",
                    Some("stream_options"),
                    oai_code::INVALID_REQUEST,
                );
            };
            for key in opts_obj.keys() {
                if key != "include_usage" {
                    return bad(
                        &format!("'stream_options.{key}' is not supported by this endpoint"),
                        Some(&format!("stream_options.{key}")),
                        oai_code::UNSUPPORTED_FIELD,
                    );
                }
            }
            match opts_obj.get("include_usage") {
                None | Some(serde_json::Value::Null) => {}
                Some(serde_json::Value::Bool(b)) => stream_include_usage = *b,
                Some(_) => {
                    return bad(
                        "'stream_options.include_usage' must be a boolean",
                        Some("stream_options.include_usage"),
                        oai_code::INVALID_REQUEST,
                    );
                }
            }
        }
    }
    // ``n``: single-candidate only on this endpoint. Reject any non-integer
    // and any value outside ``n == 1``. ``n=1`` is accepted as a no-op
    // (it is the implicit default).
    match obj.get("n") {
        None | Some(serde_json::Value::Null) => {}
        Some(v) => match v.as_u64() {
            Some(1) => {}
            Some(0) => {
                return bad(
                    "'n' must be a positive integer",
                    Some("n"),
                    oai_code::INVALID_REQUEST,
                );
            }
            Some(_) => {
                return bad(
                    "'n' > 1 is not yet supported on /v1/completions",
                    Some("n"),
                    oai_code::UNSUPPORTED_FIELD,
                );
            }
            None => {
                return bad(
                    "'n' must be an integer",
                    Some("n"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    }

    // OpenAI's documented default for completions is 16. ``max_tokens``
    // must be a positive integer when present — silently falling back to
    // the default on a present-but-invalid value would hide caller bugs.
    let max_new_tokens = match obj.get("max_tokens") {
        None | Some(serde_json::Value::Null) => 16u32,
        Some(v) => match v.as_u64() {
            Some(n) if (1..=u32::MAX as u64).contains(&n) => n as u32,
            _ => {
                return bad(
                    "'max_tokens' must be a positive integer",
                    Some("max_tokens"),
                    oai_code::INVALID_REQUEST,
                )
            }
        },
    };

    // -- samplers: reject present-but-invalid values rather than silently
    //    dropping them on the floor.
    let temperature = match obj.get("temperature") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && f >= 0.0 && f <= f32::MAX as f64 => Some(f as f32),
            _ => {
                return bad(
                    "'temperature' must be a finite number >= 0",
                    Some("temperature"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    let top_p = match obj.get("top_p") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && f > 0.0 && f <= 1.0 => Some(f as f32),
            _ => {
                return bad(
                    "'top_p' must be a number in (0, 1]",
                    Some("top_p"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    let frequency_penalty = match obj.get("frequency_penalty") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && (-2.0..=2.0).contains(&f) => Some(f),
            _ => {
                return bad(
                    "'frequency_penalty' must be a number in [-2.0, 2.0]",
                    Some("frequency_penalty"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    let presence_penalty = match obj.get("presence_penalty") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && (-2.0..=2.0).contains(&f) => Some(f),
            _ => {
                return bad(
                    "'presence_penalty' must be a number in [-2.0, 2.0]",
                    Some("presence_penalty"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    let seed = match parse_seed_field(obj.get("seed")) {
        Ok(seed) => seed,
        Err(resp) => return CompletionsParamsResult::Err(resp),
    };
    // ``stop`` accepts a single string or an array of strings; any
    // mixed-type array (e.g. ``["x", 1]``) rejects rather than silently
    // dropping the bad entry.
    let stop = match obj.get("stop") {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(vec![s.clone()]),
        Some(serde_json::Value::Array(a)) => {
            let mut out = Vec::with_capacity(a.len());
            for entry in a {
                let Some(s) = entry.as_str() else {
                    return bad(
                        "'stop' must be a string or array of strings",
                        Some("stop"),
                        oai_code::INVALID_REQUEST,
                    );
                };
                out.push(s.to_string());
            }
            if out.is_empty() {
                None
            } else {
                Some(out)
            }
        }
        Some(_) => {
            return bad(
                "'stop' must be a string or array of strings",
                Some("stop"),
                oai_code::INVALID_REQUEST,
            )
        }
    };

    // -- allow-list: any other top-level key surfaces as
    //    ``unsupported_field`` so unknown OpenAI extensions / typos
    //    fail loudly.
    const ACCEPTED: &[&str] = &[
        "model",
        "prompt",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "stream",
        "stream_options",
        "n",
    ];
    for key in obj.keys() {
        if !ACCEPTED.contains(&key.as_str()) {
            return bad(
                &format!("'{key}' is not supported by this endpoint"),
                Some(key),
                oai_code::UNSUPPORTED_FIELD,
            );
        }
    }

    CompletionsParamsResult::Ok(CompletionsParams {
        model,
        prompt,
        max_new_tokens,
        temperature,
        top_p,
        stop,
        frequency_penalty,
        presence_penalty,
        seed,
        stream,
        stream_include_usage,
    })
}

/// Build the OpenAI `text_completion` body from the aggregated outcome.
#[allow(clippy::result_large_err)]
fn build_text_completion_body(
    model: &str,
    request_id: &str,
    outcome: &crate::queue::streaming::StreamOutcome,
) -> Result<Vec<u8>, Response> {
    let Some(usage) = outcome.usage.as_ref() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json_openai_error(
                "worker terminal chunk omitted required 'usage' block",
                oai_type::SERVER_ERROR,
                None,
                oai_code::MALFORMED_WORKER_RESPONSE,
            )),
        )
            .into_response());
    };
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // H3: ``logprobs`` is now rejected at the input parser, so the
    // response body no longer carries the always-null ``logprobs`` field.
    let body = json!({
        "id": format!("cmpl-{}", request_id),
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": outcome.text,
            "index": 0,
            "finish_reason": map_chat_finish_reason(&outcome.finish_reason),
        }],
        "system_fingerprint": system_fingerprint(model),
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    });
    Ok(serde_json::to_vec(&body).unwrap_or_default())
}

#[utoipa::path(
    post,
    path = "/v1/completions",
    tag = "inference",
    description = "OpenAI-compatible legacy Completions. Raw-prompt continuation \
                   (no chat template). Non-streaming, single-candidate subset.",
    responses(
        (status = 200, description = "Text completion response"),
        (status = 400, description = "Invalid or unsupported request", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 404, description = "Model not found", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 500, description = "Worker emitted malformed response", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 503, description = "Provisioning in progress, queue unavailable, or model loading", body = crate::openapi::OpenAIErrorEnvelope),
    )
)]
/// `/v1/completions` — legacy OpenAI Completions. Reuses the shared model/route
/// resolution + generation driver; differs from chat only in the request parse
/// (raw `prompt` → `GenerateInput::Prompt`) and the `text_completion` body.
pub async fn proxy_completions(State(state): State<Arc<AppState>>, req: Request) -> Response {
    check_sdk_version(req.headers());
    let metric_labels_slot = req
        .extensions()
        .get::<telemetry::MetricLabelsSlot>()
        .cloned();

    let max_completions_body = compat_request_body_limit("/v1/completions");
    let hdr = req.headers().clone();
    let (parts, body) = req.into_parts(); // keep extensions (caller identity) for the #1841 gate
    let body_bytes = match to_bytes(body, max_completions_body).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(json_openai_error(
                    format!("request body too large: {e}"),
                    oai_type::INVALID_REQUEST,
                    None,
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response();
        }
    };
    let body_json: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    format!("invalid JSON: {e}"),
                    oai_type::INVALID_REQUEST,
                    None,
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response();
        }
    };

    let params = match completions_params_from_json(&body_json) {
        CompletionsParamsResult::Ok(p) => p,
        CompletionsParamsResult::Err(resp) => return resp,
    };

    let (model_name, bundle) =
        match resolve_model_and_bundle(&state, &params.model, &parts.extensions) {
            Ok(mb) => mb,
            Err(resp) => return resp,
        };
    let (explicit_bundle_override, _) = parse_model_spec(&params.model);

    if let Some(response) = unsupported_streaming_response(
        &state.model_registry,
        &model_name,
        &model_name,
        params.stream,
    ) {
        return response;
    }

    let ResolvedRoute {
        physical_lane,
        bundle,
        gpu,
        engine,
        admission_pool,
        effective_pool,
        effective_machine_profile,
        bundle_config_hash,
        model_revision,
        work_publisher: work_publisher_arc,
        token_id,
        content_length,
    } = match resolve_generation_route(
        &state,
        &hdr,
        &bundle,
        &model_name,
        &model_name,
        GenerationRequestIntent::Default,
        &explicit_bundle_override,
        &parts.extensions,
        metric_labels_slot.as_ref(),
    )
    .await
    {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    let start = Instant::now();

    // Copied out BEFORE the params are consumed into `WorkParams` below.
    let stream = params.stream;
    let stream_include_usage = params.stream_include_usage;
    let work_params = params.into_work_params();

    // SSE streaming → emit `text_completion` chunks. Single-candidate
    // (completions rejects n>1), so no per-candidate interleave.
    if stream {
        return super::sse::build_sse_response(super::sse::SseParams {
            state: state.as_ref(),
            work_publisher: work_publisher_arc,
            physical_lane: physical_lane.clone(),
            model: model_name.clone(),
            // /v1/completions does not accept grammar, so it never routes:
            // dispatch == display.
            dispatch_model: model_name.clone(),
            bundle: bundle.clone(),
            engine: engine.clone(),
            gpu: effective_machine_profile.clone(),
            pool: effective_pool.clone(),
            admission_pool: admission_pool.clone(),
            bundle_config_hash: bundle_config_hash.clone(),
            work_params,
            endpoint: super::sse::SseEndpoint::Completion {
                include_usage: stream_include_usage,
            },
        })
        .await;
    }

    let driver = run_streaming_generate(
        &state,
        work_publisher_arc,
        &physical_lane,
        &model_name,
        &model_name,
        &bundle,
        &engine,
        &effective_machine_profile,
        &effective_pool,
        &admission_pool,
        &bundle_config_hash,
        &work_params,
    )
    .await;

    let StreamingDriverOk {
        outcome,
        request_id,
        publish_elapsed,
        wait_elapsed,
    } = match driver {
        Ok(ok) => ok,
        Err(err) => return build_streaming_error_response(&err),
    };
    let elapsed = start.elapsed();

    let body_bytes = match build_text_completion_body(&model_name, &request_id, &outcome) {
        Ok(b) => b,
        Err(resp) => return resp,
    };

    state.registry.record_request("queue").await;
    emit_audit_log(AuditEntry {
        event: "proxy_request".to_string(),
        method: "POST".to_string(),
        endpoint: "completions".to_string(),
        status: 200,
        token_id,
        model: model_name.clone(),
        pool: effective_pool.clone(),
        gpu: gpu.clone(),
        worker: format!("queue:{request_id}"),
        latency_ms: elapsed.as_millis() as u64,
        body_bytes: content_length,
    });

    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(Body::from(body_bytes))
        .unwrap();
    let h = response.headers_mut();
    h.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    h.insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    insert_stream_model_revision_header(
        h,
        model_revision.as_deref(),
        &bundle_config_hash,
        &outcome,
    );
    insert_stream_execution_identity_header(h, &outcome);
    insert_stream_execution_binding_header(h, &outcome);
    if let Ok(val) = HeaderValue::from_str(&request_id) {
        h.insert(HeaderName::from_static("x-sie-request-id"), val);
    }
    insert_duration_header(h, "x-queue-publish-time", publish_elapsed);
    insert_duration_header(h, "x-queue-wait-time", wait_elapsed);
    response
}

/// Parsed, validated `/v1/responses` (OpenAI Responses API) request. MVP scope:
/// a single string `input` (raw prompt continuation), non-streaming.
struct ResponsesParams {
    model: String,
    /// Either a raw string prompt or a list of role/content messages.
    input: publisher::GenerateInput,
    max_new_tokens: u32,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<i64>,
}

impl ResponsesParams {
    /// The EXACT `WorkParams` a validated `/v1/responses` body dispatches with.
    /// Extracted (not copied) for the same reason as
    /// [`ChatRequestParams::into_work_params`] — see #2435.
    fn into_work_params(self) -> publisher::WorkParams {
        publisher::WorkParams {
            generate: Some(publisher::GenerateParams {
                input: self.input,
                max_new_tokens: self.max_new_tokens,
                temperature: self.temperature,
                top_p: self.top_p,
                seed: self.seed,
                ..Default::default()
            }),
            ..Default::default()
        }
    }
}

enum ResponsesParamsResult {
    Ok(ResponsesParams),
    Err(Response),
}

/// The `(model, WorkParams)` a `/v1/responses` body dispatches with. See
/// [`chat_completions_dispatch_inputs`], including the `dead_code` note.
#[allow(clippy::result_large_err, dead_code)]
pub fn responses_dispatch_inputs(
    body: &serde_json::Value,
) -> Result<(String, publisher::WorkParams), Response> {
    match responses_params_from_json(body) {
        ResponsesParamsResult::Ok(params) => {
            let model = params.model.clone();
            Ok((model, params.into_work_params()))
        }
        ResponsesParamsResult::Err(response) => Err(response),
    }
}

/// Parse + validate a `/v1/responses` body. MVP: a string `input` or a
/// list of role/content messages, non-streaming. Out of scope per
/// ADR-0001: tools, stateful threading (`previous_response_id`), reasoning,
/// background, metadata, instructions, multimodal content parts.
///
/// Validation is strict (H2 hardening): unrecognized top-level fields,
/// known-out-of-scope fields, missing roles, non-text content parts,
/// and present-but-wrong-type sampler / token-cap values all surface as
/// 400 rather than being silently coerced or dropped.
fn responses_params_from_json(body: &serde_json::Value) -> ResponsesParamsResult {
    let bad = |msg: &str, param: Option<&str>, code: &'static str| {
        ResponsesParamsResult::Err(
            (
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    msg.to_string(),
                    oai_type::INVALID_REQUEST,
                    param,
                    code,
                )),
            )
                .into_response(),
        )
    };
    let Some(obj) = body.as_object() else {
        return bad(
            "request body must be a JSON object",
            None,
            oai_code::INVALID_REQUEST,
        );
    };

    // -- explicit reject-list: known-out-of-scope Responses fields. ADR-0001
    //    scopes the Responses MVP to a stateless, single-turn, text-only
    //    surface. Anything else surfaces as ``unsupported_field`` so SDKs
    //    that target the wider OpenAI Responses spec fail loudly rather
    //    than silently losing context.
    for &fname in &[
        "previous_response_id",
        "tools",
        "tool_choice",
        "reasoning",
        "background",
        "metadata",
        "instructions",
    ] {
        if obj.contains_key(fname) {
            return bad(
                &format!("'{fname}' is not supported by this endpoint"),
                Some(fname),
                oai_code::UNSUPPORTED_FIELD,
            );
        }
    }

    let model = match obj.get("model").and_then(|v| v.as_str()) {
        Some(m) if !m.is_empty() => m.to_string(),
        _ => {
            return bad(
                "field \"model\" is required",
                Some("model"),
                oai_code::INVALID_REQUEST,
            )
        }
    };

    // The stateless Responses MVP accepts ordinary text messages only.
    // Tool outputs require the wider Responses item/tool contract, which is
    // explicitly out of scope here. ``developer`` normalizes to ``system``.
    const ALLOWED_ROLES: &[&str] = &["system", "user", "assistant", "developer"];

    let input = match obj.get("input") {
        Some(serde_json::Value::String(s)) => {
            publisher::GenerateInput::Prompt { prompt: s.clone() }
        }
        // Responses array input: a list of {role, content} items. content
        // is a string or a list of text parts; non-text parts reject.
        Some(serde_json::Value::Array(items)) => {
            if items.is_empty() {
                return bad(
                    "'input' array must not be empty",
                    Some("input"),
                    oai_code::INVALID_REQUEST,
                );
            }
            let mut messages: Vec<publisher::ChatMessage> = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                let Some(o) = item.as_object() else {
                    return bad(
                        &format!("input[{i}] must be an object"),
                        Some("input"),
                        oai_code::INVALID_REQUEST,
                    );
                };
                for key in o.keys() {
                    if !matches!(key.as_str(), "role" | "content") {
                        let param = format!("input[{i}].{key}");
                        return bad(
                            &format!("'{param}' is not supported by this endpoint"),
                            Some(&param),
                            oai_code::UNSUPPORTED_FIELD,
                        );
                    }
                }
                let role = match o.get("role").and_then(|v| v.as_str()) {
                    Some("developer") => "system".to_string(),
                    Some(r) if ALLOWED_ROLES.contains(&r) => r.to_string(),
                    Some(other) => {
                        return bad(
                            &format!(
                                "input[{i}].role must be one of {ALLOWED_ROLES:?}, got {other:?}"
                            ),
                            Some(&format!("input[{i}].role")),
                            oai_code::INVALID_REQUEST,
                        );
                    }
                    None => {
                        return bad(
                            &format!("input[{i}].role is required and must be a string"),
                            Some(&format!("input[{i}].role")),
                            oai_code::INVALID_REQUEST,
                        );
                    }
                };
                let content = match o.get("content") {
                    Some(serde_json::Value::String(s)) => s.clone(),
                    Some(serde_json::Value::Array(parts)) => {
                        // Strict validation: each part must be a text part
                        // with a string ``text`` field. Non-text parts
                        // (image_url / input_image / etc.) reject;
                        // missing-or-non-string ``text`` rejects.
                        let mut text = String::new();
                        for (j, p) in parts.iter().enumerate() {
                            let Some(part_obj) = p.as_object() else {
                                return bad(
                                    &format!("input[{i}].content[{j}] must be an object"),
                                    Some(&format!("input[{i}].content[{j}]")),
                                    oai_code::INVALID_REQUEST,
                                );
                            };
                            for key in part_obj.keys() {
                                if !matches!(key.as_str(), "type" | "text") {
                                    let param = format!("input[{i}].content[{j}].{key}");
                                    return bad(
                                        &format!("'{param}' is not supported by this endpoint"),
                                        Some(&param),
                                        oai_code::UNSUPPORTED_FIELD,
                                    );
                                }
                            }
                            let ptype = match part_obj.get("type").and_then(|v| v.as_str()) {
                                Some(t) => t,
                                None => {
                                    return bad(
                                        &format!(
                                            "input[{i}].content[{j}].type is required and must be a string"
                                        ),
                                        Some(&format!("input[{i}].content[{j}].type")),
                                        oai_code::INVALID_REQUEST,
                                    );
                                }
                            };
                            if !matches!(ptype, "text" | "input_text" | "output_text") {
                                return bad(
                                    &format!(
                                        "input[{i}].content[{j}]: unsupported content part type '{ptype}'"
                                    ),
                                    Some(&format!("input[{i}].content[{j}].type")),
                                    oai_code::UNSUPPORTED_FIELD,
                                );
                            }
                            match part_obj.get("text") {
                                Some(serde_json::Value::String(t)) => text.push_str(t),
                                Some(_) => {
                                    return bad(
                                        &format!("input[{i}].content[{j}].text must be a string"),
                                        Some(&format!("input[{i}].content[{j}].text")),
                                        oai_code::INVALID_REQUEST,
                                    );
                                }
                                None => {
                                    return bad(
                                        &format!(
                                            "input[{i}].content[{j}].text is required for text content parts"
                                        ),
                                        Some(&format!("input[{i}].content[{j}].text")),
                                        oai_code::INVALID_REQUEST,
                                    );
                                }
                            }
                        }
                        text
                    }
                    _ => {
                        return bad(
                            &format!("input[{i}].content must be a string or text parts"),
                            Some(&format!("input[{i}].content")),
                            oai_code::INVALID_REQUEST,
                        );
                    }
                };
                messages.push(publisher::ChatMessage {
                    role,
                    content,
                    tool_calls: None,
                    tool_call_id: None,
                    // Responses-API path is text-only for now; vision arrives
                    // via /v1/chat/completions (the cut-your-bill skill surface).
                    images: None,
                    content_parts: None,
                });
            }
            publisher::GenerateInput::Messages { messages }
        }
        _ => {
            return bad(
                "field \"input\" is required (a string or a list of messages)",
                Some("input"),
                oai_code::INVALID_REQUEST,
            );
        }
    };

    // ``stream`` must be a boolean when present. ``true`` is rejected
    // (streaming responses unimplemented); non-bool values reject as
    // a type error rather than silently coercing.
    match obj.get("stream") {
        None | Some(serde_json::Value::Null) | Some(serde_json::Value::Bool(false)) => {}
        Some(serde_json::Value::Bool(true)) => {
            return bad(
                "streaming is not yet supported on /v1/responses; use stream:false",
                Some("stream"),
                oai_code::UNSUPPORTED_FIELD,
            );
        }
        Some(_) => {
            return bad(
                "'stream' must be a boolean",
                Some("stream"),
                oai_code::INVALID_REQUEST,
            );
        }
    }

    // Responses uses ``max_output_tokens``; default to 16 to mirror completions.
    let max_new_tokens = match obj.get("max_output_tokens") {
        None | Some(serde_json::Value::Null) => 16u32,
        Some(v) => match v.as_u64() {
            Some(n) if (1..=u32::MAX as u64).contains(&n) => n as u32,
            _ => {
                return bad(
                    "'max_output_tokens' must be a positive integer",
                    Some("max_output_tokens"),
                    oai_code::INVALID_REQUEST,
                )
            }
        },
    };

    // -- samplers: reject present-but-invalid values rather than silently
    //    dropping them on the floor.
    let temperature = match obj.get("temperature") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && f >= 0.0 && f <= f32::MAX as f64 => Some(f as f32),
            _ => {
                return bad(
                    "'temperature' must be a finite number >= 0",
                    Some("temperature"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    let top_p = match obj.get("top_p") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && f > 0.0 && f <= 1.0 => Some(f as f32),
            _ => {
                return bad(
                    "'top_p' must be a number in (0, 1]",
                    Some("top_p"),
                    oai_code::INVALID_REQUEST,
                );
            }
        },
    };
    let seed = match parse_seed_field(obj.get("seed")) {
        Ok(seed) => seed,
        Err(resp) => return ResponsesParamsResult::Err(resp),
    };

    // -- allow-list: anything else surfaces as ``unsupported_field``.
    const ACCEPTED: &[&str] = &[
        "model",
        "input",
        "max_output_tokens",
        "temperature",
        "top_p",
        "seed",
        "stream",
    ];
    for key in obj.keys() {
        if !ACCEPTED.contains(&key.as_str()) {
            return bad(
                &format!("'{key}' is not supported by this endpoint"),
                Some(key),
                oai_code::UNSUPPORTED_FIELD,
            );
        }
    }

    ResponsesParamsResult::Ok(ResponsesParams {
        model,
        input,
        max_new_tokens,
        temperature,
        top_p,
        seed,
    })
}

/// Build the OpenAI Responses `response` body from the aggregated outcome.
#[allow(clippy::result_large_err)]
fn build_responses_body(
    model: &str,
    request_id: &str,
    outcome: &crate::queue::streaming::StreamOutcome,
) -> Result<Vec<u8>, Response> {
    let Some(usage) = outcome.usage.as_ref() else {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json_openai_error(
                "worker terminal chunk omitted required 'usage' block",
                oai_type::SERVER_ERROR,
                None,
                oai_code::MALFORMED_WORKER_RESPONSE,
            )),
        )
            .into_response());
    };
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = json!({
        "id": format!("resp-{}", request_id),
        "object": "response",
        "created_at": created,
        "model": model,
        "status": "completed",
        "output": [{
            "type": "message",
            "id": format!("msg-{}", request_id),
            "role": "assistant",
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": outcome.text,
                "annotations": [],
            }],
        }],
        "usage": {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
    });
    Ok(serde_json::to_vec(&body).unwrap_or_default())
}

#[utoipa::path(
    post,
    path = "/v1/responses",
    tag = "inference",
    description = "OpenAI Responses API (MVP): string input, non-streaming.",
    responses(
        (status = 200, description = "Response object"),
        (status = 400, description = "Invalid or unsupported request", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 404, description = "Model not found", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 413, description = "Request body is too large", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 500, description = "Worker emitted malformed response", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 503, description = "Provisioning in progress, queue unavailable, or model loading", body = crate::openapi::OpenAIErrorEnvelope),
    )
)]
/// `/v1/responses` — OpenAI Responses API (MVP). String `input` → raw-prompt
/// generation via the shared resolve+drive helpers; `response`-shaped body.
pub async fn proxy_responses(State(state): State<Arc<AppState>>, req: Request) -> Response {
    check_sdk_version(req.headers());
    let metric_labels_slot = req
        .extensions()
        .get::<telemetry::MetricLabelsSlot>()
        .cloned();

    let max_responses_body = compat_request_body_limit("/v1/responses");
    let hdr = req.headers().clone();
    let (parts, body) = req.into_parts(); // keep extensions (caller identity) for the #1841 gate
    let body_bytes = match to_bytes(body, max_responses_body).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(json_openai_error(
                    format!("request body too large: {e}"),
                    oai_type::INVALID_REQUEST,
                    None,
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response();
        }
    };
    let body_json: serde_json::Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    format!("invalid JSON: {e}"),
                    oai_type::INVALID_REQUEST,
                    None,
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response();
        }
    };

    let params = match responses_params_from_json(&body_json) {
        ResponsesParamsResult::Ok(p) => p,
        ResponsesParamsResult::Err(resp) => return resp,
    };

    let (model_name, bundle) =
        match resolve_model_and_bundle(&state, &params.model, &parts.extensions) {
            Ok(mb) => mb,
            Err(resp) => return resp,
        };
    let (explicit_bundle_override, _) = parse_model_spec(&params.model);
    let ResolvedRoute {
        physical_lane,
        bundle,
        gpu,
        engine,
        admission_pool,
        effective_pool,
        effective_machine_profile,
        bundle_config_hash,
        model_revision,
        work_publisher: work_publisher_arc,
        token_id,
        content_length,
    } = match resolve_generation_route(
        &state,
        &hdr,
        &bundle,
        &model_name,
        &model_name,
        GenerationRequestIntent::Default,
        &explicit_bundle_override,
        &parts.extensions,
        metric_labels_slot.as_ref(),
    )
    .await
    {
        Ok(r) => r,
        Err(resp) => return resp,
    };
    let start = Instant::now();

    let work_params = params.into_work_params();

    let driver = run_streaming_generate(
        &state,
        work_publisher_arc,
        &physical_lane,
        &model_name,
        // /v1/responses does not accept grammar, so it never routes:
        // dispatch == display.
        &model_name,
        &bundle,
        &engine,
        &effective_machine_profile,
        &effective_pool,
        &admission_pool,
        &bundle_config_hash,
        &work_params,
    )
    .await;
    let StreamingDriverOk {
        outcome,
        request_id,
        publish_elapsed,
        wait_elapsed,
    } = match driver {
        Ok(ok) => ok,
        Err(err) => return build_streaming_error_response(&err),
    };
    let elapsed = start.elapsed();

    let body_bytes = match build_responses_body(&model_name, &request_id, &outcome) {
        Ok(b) => b,
        Err(resp) => return resp,
    };

    state.registry.record_request("queue").await;
    emit_audit_log(AuditEntry {
        event: "proxy_request".to_string(),
        method: "POST".to_string(),
        endpoint: "responses".to_string(),
        status: 200,
        token_id,
        model: model_name.clone(),
        pool: effective_pool.clone(),
        gpu: gpu.clone(),
        worker: format!("queue:{request_id}"),
        latency_ms: elapsed.as_millis() as u64,
        body_bytes: content_length,
    });

    let mut response = Response::builder()
        .status(StatusCode::OK)
        .body(Body::from(body_bytes))
        .unwrap();
    let h = response.headers_mut();
    h.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    h.insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    insert_stream_model_revision_header(
        h,
        model_revision.as_deref(),
        &bundle_config_hash,
        &outcome,
    );
    insert_stream_execution_identity_header(h, &outcome);
    insert_stream_execution_binding_header(h, &outcome);
    if let Ok(val) = HeaderValue::from_str(&request_id) {
        h.insert(HeaderName::from_static("x-sie-request-id"), val);
    }
    insert_duration_header(h, "x-queue-publish-time", publish_elapsed);
    insert_duration_header(h, "x-queue-wait-time", wait_elapsed);
    response
}

/// Read a duration-in-seconds env var, falling back to ``default``.
/// Used by the streaming timeout taxonomy. Invalid values are silently
/// treated as missing so a typo in an env var doesn't fail-closed.
#[cfg(test)]
fn env_seconds_or(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(default)
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GenerationTimeoutConfig {
    pub first_chunk: Duration,
    pub inter_chunk: Duration,
    pub overall: Duration,
}

/// Per-process snapshot of the streaming-timeout environment overrides.
///
/// These three env vars are read once at first use rather than on every
/// request: `generation_timeout_config` runs on the generation hot path
/// and previously did three `std::env::var` syscalls per request (each an
/// allocation plus a libc lookup). Env vars don't change after process
/// start, so a `LazyLock` is both correct and cheaper. An invalid or
/// non-positive value is treated as "unset" (matches the prior
/// `.filter(|v| *v > 0.0)` semantics) so a typo fails open to the
/// profile-runtime or hard default rather than fail-closed.
struct TimeoutEnvOverrides {
    first_chunk_s: Option<f64>,
    inter_chunk_s: Option<f64>,
    overall_s: Option<f64>,
}

static TIMEOUT_ENV_OVERRIDES: std::sync::LazyLock<TimeoutEnvOverrides> =
    std::sync::LazyLock::new(|| {
        let read = |key: &str| -> Option<f64> {
            std::env::var(key)
                .ok()
                .and_then(|s| s.parse::<f64>().ok())
                .filter(|v| *v > 0.0)
        };
        TimeoutEnvOverrides {
            first_chunk_s: read("SIE_GATEWAY_FIRST_CHUNK_TIMEOUT_S"),
            inter_chunk_s: read("SIE_GATEWAY_INTER_CHUNK_TIMEOUT_S"),
            overall_s: read("SIE_GATEWAY_OVERALL_TIMEOUT_S"),
        }
    });

pub(crate) fn generation_timeout_config(
    state: &AppState,
    model: &str,
    params: &publisher::WorkParams,
    max_new_tokens: u32,
) -> GenerationTimeoutConfig {
    let derived_overall = ((max_new_tokens as u64) / 10).saturating_add(30).min(300) as f64;
    let profile_name = params
        .options
        .as_ref()
        .and_then(|opts| opts.get("profile"))
        .and_then(Value::as_str)
        .unwrap_or("default");

    let runtime = state
        .model_registry
        .get_model_info(model)
        .and_then(|entry| entry.profile_configs.get(profile_name).cloned())
        .and_then(|profile| profile.adapter_options)
        .and_then(|opts| opts.get("runtime").cloned())
        .and_then(|runtime| match runtime {
            Value::Object(map) => Some(map),
            _ => None,
        });

    let request_seconds = |key: &str| -> Option<f64> {
        params
            .options
            .as_ref()
            .and_then(|options| options.get(key))
            .and_then(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_u64().map(|value| value as f64))
            })
            .filter(|value| *value > 0.0)
    };

    let runtime_seconds = |key: &str| -> Option<f64> {
        runtime
            .as_ref()
            .and_then(|rt| rt.get(key))
            .and_then(|v| v.as_f64().or_else(|| v.as_u64().map(|u| u as f64)))
            .filter(|v| *v > 0.0)
    };

    // Precedence: environment override > request option > profile runtime >
    // hard default. Request options are already structurally validated.
    // Env overrides are read once at process start (see
    // `TIMEOUT_ENV_OVERRIDES`) instead of three `std::env::var` syscalls
    // per request.
    let env = &*TIMEOUT_ENV_OVERRIDES;
    let first_chunk = env
        .first_chunk_s
        .or_else(|| request_seconds("first_chunk_timeout_s"))
        .or_else(|| runtime_seconds("first_chunk_timeout_s"))
        .unwrap_or(30.0);
    let inter_chunk = env
        .inter_chunk_s
        .or_else(|| request_seconds("inter_chunk_timeout_s"))
        .or_else(|| runtime_seconds("inter_chunk_timeout_s"))
        .unwrap_or(10.0);
    let overall = env
        .overall_s
        .or_else(|| request_seconds("overall_timeout_s"))
        .or_else(|| runtime_seconds("overall_timeout_s"))
        .unwrap_or(derived_overall);

    let (first_chunk, overall) =
        enforce_first_chunk_invariant(first_chunk, overall, model, profile_name);

    GenerationTimeoutConfig {
        first_chunk: Duration::from_secs_f64(first_chunk),
        inter_chunk: Duration::from_secs_f64(inter_chunk),
        overall: Duration::from_secs_f64(overall),
    }
}

/// ADR-0003 invariant: `overall >= first_chunk`. If a misconfiguration
/// declares the inverse, the first-chunk policy would be dead code (the
/// overall deadline would fire before first-chunk could). The
/// pure-function shape keeps the runtime guard testable without
/// constructing an `AppState`; a profile-time check belongs on the
/// worker, where the YAML is loaded.
///
/// Returns `(clamped_first_chunk, overall)`. On violation, `first_chunk`
/// is clamped down to `overall` so the request still makes progress, and
/// a warning is logged so the misconfiguration is visible.
pub(crate) fn enforce_first_chunk_invariant(
    first_chunk: f64,
    overall: f64,
    model: &str,
    profile: &str,
) -> (f64, f64) {
    if first_chunk > overall {
        tracing::warn!(
            model = %model,
            profile = %profile,
            first_chunk_s = first_chunk,
            overall_s = overall,
            "generation timeout misconfigured: first_chunk > overall; clamping first_chunk \
             to overall. Adjust the profile or env override so overall_timeout_s >= \
             first_chunk_timeout_s (ADR-0003)."
        );
        (overall, overall)
    } else {
        (first_chunk, overall)
    }
}

fn build_model_load_failed_response() -> Response {
    let mut resp = (
        StatusCode::BAD_GATEWAY,
        Json(json!({
            "error": {
                "code": MODEL_LOAD_FAILED_ERROR_CODE,
                "message": MODEL_LOAD_FAILED_PUBLIC_MESSAGE,
                "error_class": MODEL_LOAD_FAILED_ERROR_CODE,
                "attempts": 1,
                "permanent": true,
            }
        })),
    )
        .into_response();
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-error-code"),
        HeaderValue::from_static(MODEL_LOAD_FAILED_ERROR_CODE),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-error-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp
}

fn build_result_payload_too_large_response(endpoint: &str, message: &str) -> Response {
    let mut resp = endpoint_error_response(
        endpoint,
        StatusCode::PAYLOAD_TOO_LARGE,
        err_code::PAYLOAD_TOO_LARGE,
        oai_type::INVALID_REQUEST,
        oai_code::INVALID_REQUEST,
        None,
        message,
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-error-code"),
        HeaderValue::from_static(err_code::PAYLOAD_TOO_LARGE),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp
}

fn result_payload_too_large_error(
    results: &[publisher::WorkResult],
) -> Option<&publisher::WorkResult> {
    results
        .iter()
        .find(|result| result.error_code.as_deref() == Some(PAYLOAD_TOO_LARGE_ERROR_CODE))
}

fn result_transport_failure_error(
    results: &[publisher::WorkResult],
) -> Option<&publisher::WorkResult> {
    results
        .iter()
        .find(|result| result.error_code.as_deref() == Some(oai_code::TRANSPORT_FAILURE))
}

fn build_result_transport_failure_response(endpoint: &str) -> Response {
    let mut resp = endpoint_error_response(
        endpoint,
        StatusCode::SERVICE_UNAVAILABLE,
        oai_code::TRANSPORT_FAILURE,
        oai_type::SERVER_ERROR,
        oai_code::TRANSPORT_FAILURE,
        None,
        "Worker result transport validation failed",
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-error-code"),
        HeaderValue::from_static(oai_code::TRANSPORT_FAILURE),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp
}

/// Return the retryable error code shared by every failed item, or
/// ``None`` if the batch is mixed / non-retryable.
///
/// Mixed batches go through the legacy ``all_items_failed`` 500 path so
/// callers can inspect per-item codes; unanimous retryable batches get
/// the dedicated 503 + ``Retry-After`` contract that the SDK already
/// understands. The set of codes that count as retryable here mirrors
/// the SDK's auto-retry table — see ``sie_sdk.client._shared``.
fn unanimous_retryable_error_code(errors: &[&publisher::WorkResult]) -> Option<&'static str> {
    let first = errors.first()?.error_code.as_deref()?;
    let canonical = match first {
        RESOURCE_EXHAUSTED_ERROR_CODE => RESOURCE_EXHAUSTED_ERROR_CODE,
        MODEL_LOADING_ERROR_CODE => MODEL_LOADING_ERROR_CODE,
        LORA_LOADING_ERROR_CODE => LORA_LOADING_ERROR_CODE,
        _ => return None,
    };
    if errors
        .iter()
        .all(|r| r.error_code.as_deref() == Some(canonical))
    {
        Some(canonical)
    } else {
        None
    }
}

/// Return the caller-fixable terminal code shared by every failed item. Mixed
/// batches keep their per-item 500 envelope; only homogeneous failures can be
/// translated without hiding a different sibling error.
fn unanimous_terminal_client_error(
    errors: &[&publisher::WorkResult],
) -> Option<(StatusCode, &'static str)> {
    let first = errors.first()?.error_code.as_deref()?;
    let (status, canonical) = match first {
        INVALID_INPUT_ERROR_CODE => (StatusCode::BAD_REQUEST, INVALID_INPUT_ERROR_CODE),
        PAYLOAD_TOO_LARGE_ERROR_CODE => {
            (StatusCode::PAYLOAD_TOO_LARGE, PAYLOAD_TOO_LARGE_ERROR_CODE)
        }
        COLD_START_RATE_LIMITED_ERROR_CODE => (
            StatusCode::TOO_MANY_REQUESTS,
            COLD_START_RATE_LIMITED_ERROR_CODE,
        ),
        _ => return None,
    };
    errors
        .iter()
        .all(|result| result.error_code.as_deref() == Some(canonical))
        .then_some((status, canonical))
}

fn unanimous_worker_error_message(errors: &[&publisher::WorkResult], code: &str) -> Option<String> {
    let first = errors.first()?;
    if first.error_code.as_deref() != Some(code)
        || !errors
            .iter()
            .all(|result| result.error_code.as_deref() == Some(code))
    {
        return None;
    }
    Some(
        first
            .error
            .as_deref()
            .unwrap_or("Invalid input")
            .to_string(),
    )
}

fn build_invalid_input_response(message: &str) -> Response {
    build_terminal_client_error_response(StatusCode::BAD_REQUEST, INVALID_INPUT_ERROR_CODE, message)
}

fn unanimous_client_error_code(errors: &[&publisher::WorkResult]) -> Option<&'static str> {
    let first = errors.first()?.error_code.as_deref()?;
    let canonical = match first {
        "invalid_request" => "invalid_request",
        "unsupported_field" => "unsupported_field",
        _ => return None,
    };
    if errors
        .iter()
        .all(|result| result.error_code.as_deref() == Some(first))
    {
        Some(canonical)
    } else {
        None
    }
}

fn batch_direct_fallback_delay(request_timeout_s: f64) -> Duration {
    let timeout_s = if request_timeout_s.is_finite() && request_timeout_s > 0.0 {
        request_timeout_s
    } else {
        30.0
    };
    Duration::from_secs_f64((timeout_s / 3.0).clamp(1.0, 10.0))
}

/// Build a native error response for a unanimous caller-fixable worker
/// rejection. The body and version headers match gateway-owned 400/413 errors,
/// preserving the local HTTP and queued-sidecar contract.
fn build_terminal_client_error_response(
    status: StatusCode,
    code: &'static str,
    message: &str,
) -> Response {
    let mut resp = (status, Json(json_detail(code, message))).into_response();
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-error-code"),
        HeaderValue::from_static(code),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-error-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp
}

/// Build a ``503 + <code>`` response that mirrors the worker-side HTTP
/// contract (see ``packages/sie_server/src/sie_server/api/helpers.py``).
///
///   * status:  503 Service Unavailable
///   * body:    ``{"error": {"code": <code>, "message": <upstream message>}}``
///   * headers: ``Retry-After: 5``, ``X-SIE-Error-Code: <code>``, plus the
///     standard ``X-SIE-*`` version pair.
///
/// The worker is **not** marked unhealthy — these codes are transient
/// per-request signals, not worker-health signals.
fn build_retryable_error_response(code: &'static str, message: &str) -> Response {
    // Retry hint via the shared `worker_error_retry_after` classifier — the
    // same source of truth the streaming path uses.
    let retry_after = worker_error_retry_after(code, None)
        .map(|(retry_after, _)| retry_after)
        .unwrap_or_else(|| {
            // Defensive default. Should be unreachable given the
            // ``unanimous_retryable_error_code`` allow-list (the only caller
            // pathway), but if a future code is added there without here, fall
            // back to the most conservative retry hint we know rather than
            // panicking in production. The ``debug_assert!`` ensures any such
            // mismatch fails loudly in tests / dev builds.
            debug_assert!(
                false,
                "build_retryable_error_response called with unmapped code: {code}"
            );
            MODEL_LOADING_RETRY_AFTER.to_string()
        });
    let mut resp = (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": {
                "code": code,
                "message": message,
            },
        })),
    )
        .into_response();
    resp.headers_mut().insert(
        HeaderName::from_static("retry-after"),
        HeaderValue::from_str(&retry_after).unwrap_or_else(|_| HeaderValue::from_static("5")),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-error-code"),
        HeaderValue::from_str(code).unwrap_or_else(|_| HeaderValue::from_static("ERROR")),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp
}

/// Build a gateway-owned timeout response for a queued request whose worker
/// result did not arrive within the configured request deadline.
///
/// Worker-emitted `MODEL_LOADING` still uses the retryable 503 worker contract.
/// This path is different: the gateway already accepted/published the request
/// and timed out waiting for the NATS Core result, so surface an explicit
/// gateway timeout instead of pretending the model is still loading.
///
/// * status:  504 Gateway Timeout
/// * body:    `{"detail": {"code": "GATEWAY_TIMEOUT", "message": ...}}`
/// * headers: `Retry-After: 5`, `X-SIE-Error-Code: GATEWAY_TIMEOUT`, plus the
///   standard `X-SIE-*` version pair.
fn build_queue_result_timeout_response(model: &str, timeout_secs: f64) -> Response {
    let mut resp = (
        StatusCode::GATEWAY_TIMEOUT,
        Json(json_detail(
            err_code::GATEWAY_TIMEOUT,
            format!(
                "Timed out waiting {:.1}s for queued result from model '{}'. The request was published, but no worker result reached the gateway before the deadline.",
                timeout_secs, model
            ),
        )),
    )
        .into_response();
    resp.headers_mut().insert(
        HeaderName::from_static("retry-after"),
        HeaderValue::from_static(GATEWAY_TIMEOUT_RETRY_AFTER),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-error-code"),
        HeaderValue::from_static(err_code::GATEWAY_TIMEOUT),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp.headers_mut().insert(
        HeaderName::from_static("x-sie-server-version"),
        HeaderValue::from_static(GATEWAY_VERSION),
    );
    resp
}

/// Outcome of the per-profile LoRA adapter validation gate.
///
/// Extracted from the chat and generate handlers so the decision logic
/// is unit-testable without spinning up a router. Reflects the M10
/// scoping discipline: validation MUST run against the selected
/// profile, never the union across profiles.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum LoraValidation {
    /// Adapter is advertised on the selected profile — request passes
    /// the gate. Worker still validates redundantly.
    Ok,
    /// The selected ``profile_name`` is not declared on the model.
    /// Returns ``400 invalid_request`` with ``param: "profile"``.
    UnknownProfile,
    /// The selected profile exists but doesn't advertise the requested
    /// adapter (or the model declares no adapters at all). Returns
    /// ``400 unknown_lora_adapter`` with ``param: "lora_adapter"``.
    UnknownAdapter,
}

/// Decide whether a request carrying ``lora_adapter = Some(req)`` for
/// the given ``profile_name`` should be admitted to the queue. Pure
/// function over the resolved model entry; the calling site is
/// responsible for turning the outcome into an HTTP response and
/// emitting the right metric.
///
/// ``"default"`` is always treated as a valid profile name even when
/// it's not explicitly declared on the model, because workers
/// synthesize a default profile at load time (matches the gateway's
/// ``generation_timeout_config`` behavior).
pub(crate) fn validate_lora_for_profile(
    entry: &crate::types::model::ModelEntry,
    profile_name: &str,
    req: &str,
) -> LoraValidation {
    let profile_known = profile_name == "default" || entry.profile_names.contains(profile_name);
    if !profile_known {
        return LoraValidation::UnknownProfile;
    }
    let allowed = entry
        .lora_adapters_for_profile(profile_name)
        .is_some_and(|names| names.iter().any(|n| n == req));
    if allowed {
        LoraValidation::Ok
    } else {
        LoraValidation::UnknownAdapter
    }
}

fn parse_model_spec(spec: &str) -> (String, String) {
    if let Some(idx) = spec.find(":/") {
        (spec[..idx].to_string(), spec[idx + 2..].to_string())
    } else {
        (String::new(), spec.to_string())
    }
}

/// Resolve a model id from a URL path back to the registry id.
///
/// HTTP path segments cannot contain unescaped `/`, so clients (and the
/// generated SDK) refer to a model like `Qwen/Qwen3-4B-Instruct-2507` as
/// `Qwen__Qwen3-4B-Instruct-2507` in `POST /v1/generate/{id}`. The registry,
/// however, is keyed on the slash form. Without this fallback the gateway
/// reports 404 on what is actually a valid model.
///
/// Lookup order: (1) the id as given (so models that genuinely contain `__`
/// in their canonical name still work), (2) `__` → `/`. If neither hits the
/// registry, return the original so the downstream 404 path produces a
/// useful error referencing the id the caller actually sent.
///
/// `canonicalize` returns the registry's canonical stored name for an id
/// (case-insensitively) or `None` if unknown. Returning the canonical name
/// — rather than the as-given casing — is what collapses `Org/Model`,
/// `org/model`, and `ORG/MODEL` to a single downstream identity (one
/// Prometheus label series, one HRW dispatch key). Applied at the request
/// boundary so every downstream consumer sees the canonical name.
fn resolve_path_model_id(path_id: &str, canonicalize: impl Fn(&str) -> Option<String>) -> String {
    if let Some(canonical) = canonicalize(path_id) {
        return canonical;
    }
    if path_id.contains("__") {
        let slash_id = path_id.replace("__", "/");
        if let Some(canonical) = canonicalize(&slash_id) {
            return canonical;
        }
    }
    path_id.to_string()
}

/// Canonicalise `id`, falling back to a configured job/friendly alias.
///
/// A real model id resolves through `canonicalize` directly. Otherwise, if
/// `id` (case-insensitive) is a configured alias (e.g. `"code"`), its target
/// is substituted and canonicalised; when the target itself is not in the
/// registry the target string is returned verbatim (so the downstream
/// model-not-found path references the resolved model, not the alias).
/// Mirrors the GPU-alias mechanism (`SIE_GATEWAY_GPU_ALIASES`).
fn canonicalize_with_aliases(
    aliases: &std::collections::HashMap<String, String>,
    id: &str,
    canonicalize: impl Fn(&str) -> Option<String>,
) -> Option<String> {
    if let Some(canonical) = canonicalize(id) {
        return Some(canonical);
    }
    lookup_alias(aliases, id).map(|alias| {
        // The alias target may be `bundle:/model`; the bundle is routed
        // separately (resolve_model_spec_with_aliases), so resolve only the
        // model portion here. A bare target leaves the model unchanged.
        // Resolve it like any path id: handles the SIE-safe "__"->"/" form and
        // canonical case-folding, falling back to the target model verbatim (so
        // model-not-found references the resolved model).
        let model = alias_model_target(alias.target, alias.requested_profile);
        resolve_path_model_id(&model, &canonicalize)
    })
}

struct AliasLookup<'a, 'b> {
    target: &'a String,
    requested_profile: Option<&'b str>,
}

/// Look up a job/friendly alias by its name, tolerating a `:profile` suffix.
///
/// Tries the full lowercased id first, then the base before the first `:`.
/// When the base alias matches, the suffix is returned so callers can apply it
/// to the alias target unless that target already names a concrete profile.
/// A real model id with a `:` resolves through `canonicalize` before this is
/// reached, so alias suffix handling never shadows a registered variant.
fn lookup_alias<'a, 'b>(
    aliases: &'a std::collections::HashMap<String, String>,
    id: &'b str,
) -> Option<AliasLookup<'a, 'b>> {
    let key = id.to_ascii_lowercase();
    if let Some(target) = aliases.get(&key) {
        return Some(AliasLookup {
            target,
            requested_profile: None,
        });
    }
    id.split_once(':')
        .filter(|(_, profile)| !profile.is_empty())
        .and_then(|(base, profile)| {
            let base_key = base.to_ascii_lowercase();
            aliases.get(&base_key).map(|target| AliasLookup {
                target,
                requested_profile: Some(profile),
            })
        })
}

fn alias_model_target(target: &str, requested_profile: Option<&str>) -> String {
    let (_bundle, model) = parse_model_spec(target);
    if model.contains(':') {
        return model;
    }
    requested_profile
        .filter(|profile| !profile.is_empty())
        .map(|profile| format!("{model}:{profile}"))
        .unwrap_or(model)
}

/// Resolve a request's model spec to `(bundle_override, canonical_model_name)`,
/// expanding a job/friendly alias whose target may itself carry a bundle.
///
/// Replaces the `parse_model_spec` + `canonicalize_with_aliases` pair at the
/// routing sites. The model-name resolution is unchanged (real id, else alias
/// target, else the `__`->`/` form, else verbatim). The addition: an alias
/// *target* may be a full `bundle:/org/model` spec, so an operator can pin a
/// precision/profile bundle for a job alias — e.g. map `sql` to a BF16 bundle
/// that avoids the FP8 SQL-accuracy regression (ADR 0001). An explicit caller
/// bundle (`somebundle:/sql`) always wins over the alias's bundle.
/// `pub` because it is THE definition of "which model does this string denote"
/// for this gateway, and two surfaces outside the route need that definition:
///
/// - the managed cost-estimate dry run (#2435) rates a caller-supplied id, and
///   model is a rate-key dimension, so an estimate that skipped the `__`->`/`,
///   case, and alias folding would quote a different identity than admission
///   holds (or, as it did, 503 on the very path spelling the SDKs emit);
/// - a surface that must DECIDE something about a model with no routing seam of
///   its own (the managed async jobs/batches ingress, #2542) has to reach the
///   same identity this function produces, or its decision is about a spelling
///   rather than about a model.
pub fn resolve_model_spec_with_aliases(
    aliases: &std::collections::HashMap<String, String>,
    spec: &str,
    canonicalize: impl Fn(&str) -> Option<String>,
) -> (String, String) {
    let (req_bundle, req_model) = parse_model_spec(spec);
    let model_name = resolve_path_model_id(&req_model, |id| {
        canonicalize_with_aliases(aliases, id, &canonicalize)
    });
    // Adopt the alias target's bundle only when the caller pinned none AND the
    // bare model resolved through an alias (not a real model id). Uses the same
    // `:profile`-tolerant lookup as the model-name resolution above, so a
    // `sql:profile` request adopts the `sql` alias bundle too.
    if req_bundle.is_empty() && canonicalize(&req_model).is_none() {
        if let Some(alias) = lookup_alias(aliases, &req_model) {
            let (alias_bundle, _) = parse_model_spec(alias.target);
            return (alias_bundle, model_name);
        }
    }
    (req_bundle, model_name)
}

/// `pub` for the same reason as [`resolve_model_spec_with_aliases`]: the
/// estimate receives a target PATH and must decode its model segment the way
/// the route does before rating it.
pub fn decode_model_path(raw: &str) -> Result<String, String> {
    percent_decode_str(raw)
        .decode_utf8()
        .map(|decoded| decoded.into_owned())
        .map_err(|_| "model path is not valid UTF-8 after percent decoding".to_string())
}

fn resolve_machine_profile(
    gpu: &str,
    gpu_profile_map: &std::collections::HashMap<String, String>,
) -> String {
    // Lowercase the input once and reuse it for both the exact and the
    // `-spot` variant lookup. Previously we paid two `to_lowercase()`
    // heap allocations per request even in the common "already
    // canonical, not in the map" case.
    let gpu_lower = gpu.to_ascii_lowercase();
    if let Some(val) = gpu_profile_map.get(&gpu_lower) {
        return val.clone();
    }

    let mut spot_key = gpu_lower;
    spot_key.push_str("-spot");
    if let Some(val) = gpu_profile_map.get(&spot_key) {
        info!(from = gpu, to = %val, "resolved machine_profile");
        return val.clone();
    }

    gpu.to_string()
}

fn insert_duration_header(headers: &mut HeaderMap, name: &'static str, duration: Duration) {
    if let Ok(value) = HeaderValue::from_str(&format!("{:.1}", duration.as_secs_f64() * 1000.0)) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

fn insert_timing_header(headers: &mut HeaderMap, name: &'static str, value_ms: f64) {
    if let Ok(value) = HeaderValue::from_str(&format!("{value_ms:.1}")) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

fn max_result_timing<F>(results: &[&publisher::WorkResult], field: F) -> Option<f64>
where
    F: Fn(&publisher::WorkResult) -> Option<f64>,
{
    results
        .iter()
        .filter_map(|result| field(result))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

fn insert_optional_timing_header(
    headers: &mut HeaderMap,
    name: &'static str,
    value_ms: Option<f64>,
) {
    let Some(value_ms) = value_ms else {
        return;
    };
    if value_ms <= 0.0 {
        return;
    }
    if let Ok(value) = HeaderValue::from_str(&format!("{value_ms:.1}")) {
        headers.insert(HeaderName::from_static(name), value);
    }
}

fn insert_model_revision_header(
    headers: &mut HeaderMap,
    model_revision: Option<&str>,
    expected_bundle_config_hash: &str,
    successful: &[&publisher::WorkResult],
) {
    if expected_bundle_config_hash.is_empty()
        || successful.is_empty()
        || successful.iter().any(|result| {
            result.executed_bundle_config_hash.as_deref() != Some(expected_bundle_config_hash)
        })
    {
        return;
    }
    if model_revision.is_none() {
        return;
    }
    if let Ok(value) = HeaderValue::from_str(expected_bundle_config_hash) {
        headers.insert(HeaderName::from_static("x-sie-model-revision"), value);
    }
}

fn insert_stream_model_revision_header(
    headers: &mut HeaderMap,
    model_revision: Option<&str>,
    expected_bundle_config_hash: &str,
    outcome: &crate::queue::streaming::StreamOutcome,
) {
    if expected_bundle_config_hash.is_empty()
        || outcome.executed_bundle_config_hash.as_deref() != Some(expected_bundle_config_hash)
    {
        return;
    }
    if model_revision.is_none() {
        return;
    }
    if let Ok(value) = HeaderValue::from_str(expected_bundle_config_hash) {
        headers.insert(HeaderName::from_static("x-sie-model-revision"), value);
    }
}

fn insert_execution_identity_header(
    headers: &mut HeaderMap,
    successful: &[&publisher::WorkResult],
) {
    insert_unanimous_sha256_header(
        headers,
        successful,
        "x-sie-execution-identity-sha256",
        |result| result.execution_identity_sha256.as_deref(),
    );
}

fn insert_execution_binding_header(headers: &mut HeaderMap, successful: &[&publisher::WorkResult]) {
    insert_unanimous_sha256_header(
        headers,
        successful,
        "x-sie-execution-binding-sha256",
        |result| result.execution_binding_sha256.as_deref(),
    );
}

fn insert_unanimous_sha256_header<'a>(
    headers: &mut HeaderMap,
    successful: &[&'a publisher::WorkResult],
    header_name: &'static str,
    field: impl Fn(&'a publisher::WorkResult) -> Option<&'a str>,
) {
    let Some(first) = successful.first().and_then(|result| field(result)) else {
        return;
    };
    if !is_lower_sha256(first) || successful.iter().any(|result| field(result) != Some(first)) {
        return;
    }
    if let Ok(value) = HeaderValue::from_str(first) {
        headers.insert(HeaderName::from_static(header_name), value);
    }
}

fn insert_stream_execution_identity_header(
    headers: &mut HeaderMap,
    outcome: &crate::queue::streaming::StreamOutcome,
) {
    insert_stream_sha256_header(
        headers,
        "x-sie-execution-identity-sha256",
        outcome.execution_identity_sha256.as_deref(),
    );
}

fn insert_stream_execution_binding_header(
    headers: &mut HeaderMap,
    outcome: &crate::queue::streaming::StreamOutcome,
) {
    insert_stream_sha256_header(
        headers,
        "x-sie-execution-binding-sha256",
        outcome.execution_binding_sha256.as_deref(),
    );
}

fn insert_stream_sha256_header(
    headers: &mut HeaderMap,
    header_name: &'static str,
    value: Option<&str>,
) {
    let Some(value) = value.filter(|value| is_lower_sha256(value)) else {
        return;
    };
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(HeaderName::from_static(header_name), value);
    }
}

fn insert_queue_worker_timing_headers(
    headers: &mut HeaderMap,
    successful: &[&publisher::WorkResult],
) {
    if successful.is_empty() {
        return;
    }

    insert_timing_header(
        headers,
        "x-queue-time",
        max_result_timing(successful, |result| result.queue_ms).unwrap_or(0.0),
    );
    insert_timing_header(
        headers,
        "x-inference-time",
        max_result_timing(successful, |result| result.inference_ms).unwrap_or(0.0),
    );
    insert_optional_timing_header(
        headers,
        "x-tokenization-time",
        max_result_timing(successful, |result| result.tokenization_ms),
    );
    insert_optional_timing_header(
        headers,
        "x-postprocessing-time",
        max_result_timing(successful, |result| result.postprocessing_ms),
    );
    insert_optional_timing_header(
        headers,
        "x-payload-fetch-time",
        max_result_timing(successful, |result| result.payload_fetch_ms),
    );
}

pub(crate) fn is_valid_compat_model_id(model: &str) -> bool {
    let mut chars = model.chars();
    chars
        .next()
        .is_some_and(|value| value.is_ascii_alphanumeric())
        && !model.contains("..")
        && !model.contains('\\')
        && chars
            .all(|value| value.is_ascii_alphanumeric() || matches!(value, '.' | '_' | '/' | '-'))
}

pub(crate) fn is_openai_compat_forwarded_header(name: &str) -> bool {
    [
        "x-sie-request-id",
        "x-sie-version",
        "x-sie-server-version",
        "x-sie-worker",
        "x-queue-publish-time",
        "x-queue-wait-time",
        "x-queue-time",
        "x-sie-model-revision",
        "x-sie-execution-identity-sha256",
        "x-sie-execution-binding-sha256",
        "x-inference-time",
        "x-tokenization-time",
        "x-postprocessing-time",
        "x-payload-fetch-time",
    ]
    .iter()
    .any(|allowed| name.eq_ignore_ascii_case(allowed))
}

/// Headers copied from an inbound OpenAI-compatible request onto the
/// internal SIE-native request it is rewritten into. Auth/routing
/// hints plus the W3C trace headers (`traceparent`/`tracestate`) so the
/// worker span continues the client's trace instead of rooting a fresh
/// one — every other endpoint already injects queue trace context, and
/// without this embeddings would be the lone disconnected path.
pub(crate) fn is_openai_compat_inner_request_header(name: &str) -> bool {
    [
        "authorization",
        "x-sie-machine-profile",
        "x-sie-pool",
        "x-sie-engine",
        "x-sie-sdk-version",
        "traceparent",
        "tracestate",
    ]
    .iter()
    .any(|allowed| name.eq_ignore_ascii_case(allowed))
}

/// Re-surface a SIE-native error response from an inner primitive call
/// (used to service an OpenAI-compatible endpoint) in the OpenAI ``{error:{…}}``
/// envelope. The inner path emits ``{detail:{code,message}}`` (and, for the
/// 502/503 SDK-stable shapes, ``{error:{…}}``); a client that wraps
/// ``client.embeddings.create(...)`` in ``except openai.APIError`` cannot
/// parse those, so we map the inner ``detail.code`` / ``error.code`` (or,
/// absent both, the HTTP status) through
/// [`crate::http_error::openai_error_from_detail_code`] and rebuild the body.
/// The upstream status, allowlisted ``x-…`` headers, and
/// ``Retry-After`` (so 429/503 auto-retry still works) are preserved. The
/// inner ``message`` is already gateway-sanitized (field names + validation
/// text, never request content or raw upstream internals); unparseable
/// bodies fall back to a generic message, never the raw bytes.
pub(crate) async fn translate_inner_compat_error(resp: Response) -> Response {
    // Deliberately NOT [`MAX_COMPAT_RESPONSE_BODY`]. This buffers an ERROR
    // envelope — a code and a sanitized message, kilobytes at the outside — so
    // 16 MiB is already orders of magnitude above any real body here and the
    // amplification that forced the embeddings cap up does not exist. Raising
    // it by symmetry would only widen the buffer an unhealthy upstream can make
    // the gateway hold.
    const MAX: usize = 16 * 1024 * 1024;
    let status = resp.status();
    let headers = resp.headers().clone();
    let parsed: Value = match to_bytes(resp.into_body(), MAX).await {
        Ok(b) => serde_json::from_slice(&b).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    // SIE-native `detail.code` / SDK-stable `error.code` is the precise
    // discriminator; fall back to the HTTP status only when the body is
    // unparseable or lacks a structured code.
    let sie_code = parsed
        .get("detail")
        .or_else(|| parsed.get("error"))
        .and_then(|d| d.get("code"))
        .and_then(|c| c.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| sie_code_from_status(status));
    // Prefer the already-sanitized inner message from either shape.
    let message = parsed
        .get("detail")
        .and_then(|d| d.get("message"))
        .or_else(|| parsed.get("error").and_then(|e| e.get("message")))
        .and_then(|m| m.as_str())
        .unwrap_or("internal server error")
        .to_string();
    let mut out = (status, Json(embeddings_error(&sie_code, None, message))).into_response();
    for (k, v) in headers.iter() {
        let n = k.as_str();
        if is_openai_compat_forwarded_header(n) || n.eq_ignore_ascii_case("retry-after") {
            out.headers_mut().insert(k.clone(), v.clone());
        }
    }
    if let Ok(value) = HeaderValue::from_str(&sie_code) {
        out.headers_mut()
            .insert(HeaderName::from_static("x-sie-error-code"), value);
    }
    out
}

/// Map an HTTP status to the SIE-native ``code`` used as a fallback when an
/// inner error body carries no ``detail.code`` / ``error.code``. Mirrors the
/// status-to-code pairing in [`crate::http_error::openai_error_from_detail_code`].
fn sie_code_from_status(status: StatusCode) -> String {
    match status {
        StatusCode::BAD_REQUEST => err_code::INVALID_REQUEST,
        StatusCode::NOT_FOUND => err_code::MODEL_NOT_FOUND,
        StatusCode::PAYLOAD_TOO_LARGE => err_code::PAYLOAD_TOO_LARGE,
        StatusCode::TOO_MANY_REQUESTS => "RATE_LIMIT",
        StatusCode::SERVICE_UNAVAILABLE => err_code::QUEUE_UNAVAILABLE,
        StatusCode::GATEWAY_TIMEOUT => err_code::GATEWAY_TIMEOUT,
        _ => err_code::INTERNAL_ERROR,
    }
    .to_string()
}

fn result_decode_error_value(r: &publisher::WorkResult, message: String) -> serde_json::Value {
    json!({
        "item_index": r.item_index,
        "work_item_id": r.work_item_id,
        "error": {
            "code": "RESULT_DECODE_FAILED",
            "message": message,
        },
    })
}

/// The `usage` block for an items-shaped queue response, summed from the
/// workers' own authoritative [`publisher::UnitCounts`].
///
/// Serves `score` and `encode` alike: the two envelopes report the same shape
/// from the same numbers, so a caller reading `usage` on either endpoint reads
/// the post-tokenization count the worker measured — never a character
/// estimate. `None` when ANY successful result lacks the dimension: a partial
/// sum is not a measurement of the request, and omitting is the only honest
/// rendering of "this path could not count". A unit of `0` is a measurement and
/// is reported as such.
fn aggregate_result_usage(successful: &[&publisher::WorkResult]) -> Option<Value> {
    if successful.is_empty() {
        return None;
    }
    let mut input_tokens = 0_u64;
    let mut images = 0_u64;
    let mut all_have_images = true;
    for result in successful {
        let units = result.units.as_ref()?;
        input_tokens = input_tokens.checked_add(units.input_tokens?)?;
        match units.images {
            Some(value) => images = images.checked_add(value)?,
            None => all_have_images = false,
        }
    }

    let mut usage = Map::new();
    usage.insert("input_tokens".to_string(), json!(input_tokens));
    if all_have_images {
        usage.insert("images".to_string(), json!(images));
    }
    Some(Value::Object(usage))
}

fn build_queue_success_body(
    endpoint: &str,
    model: &str,
    successful: &[&publisher::WorkResult],
    use_msgpack: bool,
) -> Vec<u8> {
    // Generate has its own envelope shape: ``{model, text, finish_reason, usage}``
    // composed from the worker's single result blob. Items/score envelopes
    // continue to wrap a list, even when length==1.
    if endpoint == "generate" {
        return build_generate_success_body(model, successful, use_msgpack);
    }

    let content_key = if endpoint == "score" {
        "scores"
    } else {
        "items"
    };

    // `encode` joins `score` here so the queue ingress reports the same
    // authoritative counts the direct ingress puts on `EncodeResponse.usage`,
    // and so the `/v1/embeddings` compatibility layer has a real number to
    // render on both paths instead of a character estimate.
    let usage = if endpoint == "score" || endpoint == "encode" {
        aggregate_result_usage(successful)
    } else {
        None
    };

    if use_msgpack {
        // Msgpack: build {"model": ..., "items"|"scores": [raw_blobs...]}
        // at byte level. The worker already produced msgpack bytes, so the
        // success path trusts and appends them directly instead of decoding and
        // cloning every blob. JSON responses still decode defensively below.
        let payload_len = successful
            .iter()
            .map(|r| r.result_msgpack.len())
            .sum::<usize>();
        let mut packer = rmp::encode::buffer::ByteBuf::new();
        rmp::encode::write_map_len(&mut packer, if usage.is_some() { 3 } else { 2 }).unwrap();
        rmp::encode::write_str(&mut packer, "model").unwrap();
        rmp::encode::write_str(&mut packer, model).unwrap();
        rmp::encode::write_str(&mut packer, content_key).unwrap();
        let mut parts = if endpoint == "score" && successful.len() == 1 {
            let mut parts = packer.into_vec();
            parts.reserve(payload_len);
            parts.extend_from_slice(&successful[0].result_msgpack);
            parts
        } else {
            rmp::encode::write_array_len(&mut packer, successful.len() as u32).unwrap();
            let mut parts = packer.into_vec();
            parts.reserve(payload_len);
            for result in successful {
                parts.extend_from_slice(&result.result_msgpack);
            }
            parts
        };
        if let Some(usage) = usage {
            let mut usage_packer = rmp::encode::buffer::ByteBuf::new();
            rmp::encode::write_str(&mut usage_packer, "usage").unwrap();
            parts.extend_from_slice(&usage_packer.into_vec());
            parts.extend_from_slice(
                &rmp_serde::to_vec_named(&usage).expect("score usage JSON value must serialize"),
            );
        }
        parts
    } else {
        // JSON: decode each blob, convert numpy arrays, wrap in the server
        // envelope. The result blobs typically carry msgpack_numpy-encoded
        // arrays; `rmpv_to_response_json` decodes those without bouncing large
        // binary buffers through serde_json byte arrays.
        let mut result_items: Vec<serde_json::Value> = successful
            .iter()
            .map(
                |r| match rmp_serde::from_slice::<rmpv::Value>(&r.result_msgpack) {
                    Ok(rmpv_val) => rmpv_to_response_json(rmpv_val),
                    Err(err) => result_decode_error_value(
                        r,
                        format!("failed to decode result_msgpack: {err}"),
                    ),
                },
            )
            .collect();
        let items_val = if endpoint == "score" && result_items.len() == 1 {
            match result_items.pop() {
                Some(serde_json::Value::Array(arr)) => serde_json::Value::Array(arr),
                Some(other) => serde_json::Value::Array(vec![other]),
                None => serde_json::Value::Array(Vec::new()),
            }
        } else {
            serde_json::Value::Array(result_items)
        };
        let mut response = Map::new();
        response.insert("model".to_string(), json!(model));
        response.insert(content_key.to_string(), items_val);
        if let Some(usage) = usage {
            response.insert("usage".to_string(), usage);
        }
        serde_json::to_vec(&response).unwrap_or_default()
    }
}

/// Streaming response body. Aggregates a [`StreamOutcome`] into
/// the v1 HTTP envelope, keeping the walking-skeleton's top-level field names
/// (``model``, ``text``, ``finish_reason``, ``usage``) and adding the
/// SIE-native ``attempt_id``, ``ttft_ms``, ``tpot_ms``.
pub(crate) fn build_generate_success_body_v2(
    model: &str,
    outcome: &crate::queue::streaming::StreamOutcome,
    use_msgpack: bool,
) -> Vec<u8> {
    let usage_value = outcome.usage.as_ref().map(|u| {
        json!({
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        })
    });
    let mut body = serde_json::Map::new();
    body.insert("model".to_string(), json!(model));
    body.insert("text".to_string(), json!(outcome.text));
    body.insert("finish_reason".to_string(), json!(outcome.finish_reason));
    if let Some(u) = usage_value {
        body.insert("usage".to_string(), u);
    }
    body.insert("attempt_id".to_string(), json!(outcome.attempt_id));
    if let Some(t) = outcome.ttft_ms {
        body.insert("ttft_ms".to_string(), json!(t));
    }
    if let Some(t) = outcome.tpot_ms {
        body.insert("tpot_ms".to_string(), json!(t));
    }
    let value = serde_json::Value::Object(body);
    if use_msgpack {
        rmp_serde::to_vec_named(&value).unwrap_or_default()
    } else {
        serde_json::to_vec(&value).unwrap_or_default()
    }
}

/// Compose the walking-skeleton's generate response from the worker's single result.
///
/// The worker publishes ``{text, finish_reason, usage}`` as the msgpack blob
/// in :class:`WorkResult.result_msgpack`. The gateway response shape adds
/// ``model`` and keeps everything else flat — no ``items`` list, no
/// per-item wrapper, since generate always produces exactly one result.
#[allow(dead_code)]
fn build_generate_success_body(
    model: &str,
    successful: &[&publisher::WorkResult],
    use_msgpack: bool,
) -> Vec<u8> {
    // Defensive: the queue path should have exactly one successful result
    // (``total_items == 1`` is set for generate in ``publish_work``). An
    // empty list here is a bug; emit a usable error rather than panic.
    let blob = match successful.first() {
        Some(r) => &r.result_msgpack,
        None => {
            return serde_json::to_vec(&json!({
                "error": "generate request produced no successful result",
            }))
            .unwrap_or_default();
        }
    };

    let decoded: rmpv::Value = match rmp_serde::from_slice(blob) {
        Ok(v) => v,
        Err(err) => {
            return serde_json::to_vec(&json!({
                "error": format!("failed to decode generate result: {err}"),
            }))
            .unwrap_or_default();
        }
    };
    let mut value = rmpv_to_response_json(decoded);
    if let serde_json::Value::Object(ref mut map) = value {
        map.insert("model".to_string(), json!(model));
    }

    if use_msgpack {
        rmp_serde::to_vec_named(&value).unwrap_or_default()
    } else {
        serde_json::to_vec(&value).unwrap_or_default()
    }
}

/// Check client SDK version skew.
/// Warns once per minor version if client SDK differs by >1 minor version.
fn check_sdk_version(headers: &HeaderMap) {
    let Some(sdk_version) = headers
        .get("x-sie-sdk-version")
        .and_then(|v| v.to_str().ok())
    else {
        return;
    };

    // Fast path: this SDK version was already parsed on a previous
    // request. A successful hit avoids the `split('.')` allocation
    // and `u32::parse` on every subsequent request.
    let cached = SDK_VERSION_CACHE.get(sdk_version).map(|v| *v);
    let sdk_minor = match cached {
        Some(Some(m)) => Some(m),
        Some(None) => return, // header is malformed; stop re-parsing it
        None => {
            // First request for this version string — parse once and
            // memoise. Parse semver-like string (e.g. "0.2.3" → 2).
            let parsed = sdk_version
                .split('.')
                .nth(1)
                .and_then(|p| p.parse::<u32>().ok());
            // Size-capped insert: once the cache is full we stop
            // memoising so a hostile client can't walk unique
            // header values and grow the map forever. `len()` is
            // a snapshot so two racing inserts can push us one or
            // two entries over the cap — that's fine, the point
            // is bounded growth, not a strict bound.
            if SDK_VERSION_CACHE.len() < SDK_VERSION_CACHE_CAP {
                SDK_VERSION_CACHE.insert(Arc::<str>::from(sdk_version), parsed);
            }
            parsed
        }
    };

    let Some(sdk_minor) = sdk_minor else { return };

    // Size-capped insert (mirrors `SDK_VERSION_CACHE`): `sdk_minor` is
    // parsed from a caller-supplied header, so a hostile client could
    // otherwise walk unique minor numbers (`0.<n>.0`) and grow this set
    // without bound. Past the cap we stop tracking "already warned" and
    // fall back to warning-on-every-skewed-request — noisier logs, but
    // bounded memory. `len()` is a racy snapshot; a couple of entries of
    // slop over the cap is fine, the point is bounded growth.
    if sdk_minor.abs_diff(*GATEWAY_VERSION_MINOR) > 1
        && SDK_WARNED_MINORS.len() < SDK_WARNED_MINORS_CAP
        && SDK_WARNED_MINORS.insert(sdk_minor)
    {
        warn!(
            sdk_version = %sdk_version,
            gateway_version = GATEWAY_VERSION,
            "client SDK version skew detected (>1 minor version difference)"
        );
    }
}

/// Parse request body once, extract both raw items and work params for queue mode.
///
/// `is_msgpack` is computed by the caller from the `content-type`
/// header before the request body is consumed, which lets us avoid
/// holding on to a full `HeaderMap` clone just to read two flags.
///
/// Items are returned as `rmpv::Value`. This lets msgpack request
/// bodies pass straight through to the worker without the old
/// `msgpack → rmpv::Value → serde_json::Value → msgpack` detour
/// (which in particular blew every `bin` field up into a
/// `Vec<serde_json::Value::Number>` — ~16 MiB of allocations per
/// 1 MiB of binary input). JSON bodies are converted to `rmpv` once
/// via [`json_item_to_rmpv`]. Known native media byte fields are
/// base64-decoded into msgpack `bin` to mirror the worker's typed JSON
/// decode; the rest of the JSON item shape is converted cheaply and
/// losslessly.
/// Failure modes for :func:`parse_queue_request`. ``Generic`` keeps
/// the walking-skeleton's string-based wire-error behaviour (caller wraps in a
/// generic 400). ``PreBuilt`` carries an already-shaped OpenAI error
/// envelope so grammar safety violations can surface with
/// their precise ``param`` / ``code`` rather than being re-wrapped.
#[derive(Debug)]
pub enum QueueParseError {
    /// Caller surfaces as a generic 400 with the message body.
    Generic(String),
    /// Caller returns the response verbatim (already 400 with the
    /// correct OpenAI envelope).
    PreBuilt(Response),
}

impl From<String> for QueueParseError {
    fn from(s: String) -> Self {
        QueueParseError::Generic(s)
    }
}

// The body-parse helpers below all carry an ``Err`` arm shaped as a
// fully-built ``axum::response::Response`` so grammar
// validation can surface the precise OpenAI envelope (``param`` /
// ``code``) without re-wrapping. ``Response`` is large (>128 B) so
// clippy::result_large_err fires; we accept the cost — boxing
// would force every caller to dereference at the recovery site and
// match the pre-existing ``ChatParamsResult::Err(Response)`` shape.
#[allow(clippy::result_large_err)]
pub fn parse_queue_request(
    body: &[u8],
    is_msgpack: bool,
    endpoint: &str,
) -> Result<(Vec<rmpv::Value>, publisher::WorkParams), QueueParseError> {
    let (items, params) = if is_msgpack {
        parse_queue_request_msgpack(body, endpoint)?
    } else {
        parse_queue_request_json(body, endpoint)?
    };
    validate_queue_item_shapes(endpoint, &items, &params)?;
    Ok((items, params))
}

pub(crate) const MAX_SCORE_ITEMS: usize = 1000;
pub(crate) const MAX_EMBEDDING_INPUTS: usize = 256;

/// Reject queue-request bodies whose per-item / query shapes the worker
/// cannot consume, at ingress, instead of forwarding them to a GPU lane
/// where they NAK and burn ~15s of in-lane retries before surfacing a 500.
///
/// The worker's typed contract (ScoreRequest.query: Item, {Encode,Score,
/// Extract}Request.items: list[Item]) requires every query and every work
/// item to be a map/object. Well-formed SDK traffic (msgpack AND correctly
/// shaped JSON) always satisfies this, so this only rejects malformed
/// bodies and leaves the happy path byte-identical.
///
/// Runs on the shared rmpv representation both wire paths converge on, so a
/// single call from `parse_queue_request` closes the class for JSON and
/// msgpack alike.
#[allow(clippy::result_large_err)]
fn validate_queue_item_shapes(
    endpoint: &str,
    items: &[rmpv::Value],
    params: &publisher::WorkParams,
) -> Result<(), QueueParseError> {
    // `generate` carries no items array (prompt/sampling live under
    // params.generate); skip it entirely.
    if endpoint == "generate" {
        return Ok(());
    }

    // score: a PRESENT query must be a map. A missing query defaults to an
    // empty map upstream (work_params_from_json / _from_rmpv), which is a
    // Map and passes, so this only rejects an explicit non-map query
    // (string / number / array / null).
    if endpoint == "score" {
        if items.is_empty() {
            return Err("'items' must contain at least one candidate"
                .to_string()
                .into());
        }
        if items.len() > MAX_SCORE_ITEMS {
            return Err(
                format!("'items' must contain at most {MAX_SCORE_ITEMS} candidates").into(),
            );
        }
        if let Some(query) = params.query_item.as_ref() {
            if !matches!(query, rmpv::Value::Map(_)) {
                return Err("'query' must be an object".to_string().into());
            }
        }
    }

    // Every work item must be a map/object.
    for (index, item) in items.iter().enumerate() {
        if !matches!(item, rmpv::Value::Map(_)) {
            return Err(format!("item at index {index} must be an object").into());
        }
    }

    Ok(())
}

#[allow(clippy::result_large_err)]
fn parse_queue_request_json(
    body: &[u8],
    endpoint: &str,
) -> Result<(Vec<rmpv::Value>, publisher::WorkParams), QueueParseError> {
    let mut parsed: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| QueueParseError::Generic(format!("json decode: {}", e)))?;

    if !parsed.is_object() {
        return Err(QueueParseError::Generic(
            "Request body must be a JSON object".to_string(),
        ));
    }

    let params = work_params_from_json(&parsed, endpoint)?;

    // Generate requests have no ``items`` array — the prompt and sampling
    // params travel under ``WorkParams.generate`` instead.
    if endpoint == "generate" {
        return Ok((Vec::new(), params));
    }

    let items_json = if let Some(map) = parsed.as_object_mut() {
        if endpoint == "score" {
            match map.remove("items") {
                Some(serde_json::Value::Array(arr)) => arr,
                Some(_) => return Err("'items' must be an array".to_string().into()),
                None => Vec::new(),
            }
        } else if let Some(value) = map.remove("items") {
            match value {
                serde_json::Value::Array(arr) => arr,
                _ => return Err("'items' must be an array".to_string().into()),
            }
        } else if let Some(val) = map.remove("input") {
            match val {
                serde_json::Value::Array(arr) => arr,
                other => vec![other],
            }
        } else if let Some(value) = map.remove("inputs") {
            match value {
                serde_json::Value::Array(arr) => arr,
                _ => return Err("'inputs' must be an array".to_string().into()),
            }
        } else {
            vec![parsed]
        }
    } else {
        vec![parsed]
    };

    let items = items_json
        .into_iter()
        .map(json_item_to_rmpv)
        .collect::<Result<Vec<_>, _>>()?;
    Ok((items, params))
}

#[allow(clippy::result_large_err)]
fn parse_queue_request_msgpack(
    body: &[u8],
    endpoint: &str,
) -> Result<(Vec<rmpv::Value>, publisher::WorkParams), QueueParseError> {
    let parsed: rmpv::Value = rmp_serde::from_slice(body)
        .map_err(|e| QueueParseError::Generic(format!("msgpack decode: {}", e)))?;

    // Parity with `parse_queue_request_json`: top-level must be a
    // map. The JSON path rejects scalars/arrays with 400 before we
    // ever reach a fallback, so reject the same shapes on the
    // msgpack path instead of silently turning e.g. a top-level
    // array into `items = vec![<array>]`, which would only fail
    // later in worker-specific ways.
    let mut map = match parsed {
        rmpv::Value::Map(m) => m,
        _ => {
            return Err(QueueParseError::Generic(
                "Request body must be a msgpack map".to_string(),
            ));
        }
    };

    let params = work_params_from_rmpv(&map, endpoint)?;

    if endpoint == "generate" {
        return Ok((Vec::new(), params));
    }

    let items: Vec<rmpv::Value> = if endpoint == "score" {
        match rmpv_map_remove(&mut map, "items") {
            Some(rmpv::Value::Array(arr)) => arr,
            Some(_) => return Err("'items' must be an array".to_string().into()),
            None => Vec::new(),
        }
    } else if let Some(value) = rmpv_map_remove(&mut map, "items") {
        match value {
            rmpv::Value::Array(arr) => arr,
            _ => return Err("'items' must be an array".to_string().into()),
        }
    } else if let Some(val) = rmpv_map_remove(&mut map, "input") {
        match val {
            rmpv::Value::Array(arr) => arr,
            other => vec![other],
        }
    } else if let Some(value) = rmpv_map_remove(&mut map, "inputs") {
        match value {
            rmpv::Value::Array(arr) => arr,
            _ => return Err("'inputs' must be an array".to_string().into()),
        }
    } else {
        // No recognised items key — treat the remaining map as a
        // single item. Rebuild the map value (the original was
        // consumed above — `work_params_from_rmpv` only borrowed it
        // but the items-lookup calls used `rmpv_map_remove`, which
        // mutates; fields that the params extractor cares about are
        // still present because the remove helpers only strip the
        // items-related keys).
        vec![rmpv::Value::Map(map)]
    };

    Ok((items, params))
}

/// Remove the first entry whose key (string or binary UTF-8) matches
/// `key`, returning its value. Lookup is O(n) but n is the number of
/// top-level request fields (≤ ~10), so the cost is negligible and we
/// avoid allocating an intermediate map.
fn rmpv_map_remove(map: &mut Vec<(rmpv::Value, rmpv::Value)>, key: &str) -> Option<rmpv::Value> {
    let pos = map.iter().position(|(k, _)| rmpv_key_eq(k, key))?;
    Some(map.swap_remove(pos).1)
}

fn rmpv_map_get<'a>(map: &'a [(rmpv::Value, rmpv::Value)], key: &str) -> Option<&'a rmpv::Value> {
    map.iter()
        .find(|(k, _)| rmpv_key_eq(k, key))
        .map(|(_, v)| v)
}

fn rmpv_key_eq(key: &rmpv::Value, expected: &str) -> bool {
    match key {
        rmpv::Value::String(s) => s.as_str() == Some(expected),
        // Python msgpack without strict_map_key=True emits bin keys.
        rmpv::Value::Binary(b) => std::str::from_utf8(b).ok() == Some(expected),
        _ => false,
    }
}

fn rmpv_as_str(value: &rmpv::Value) -> Option<&str> {
    match value {
        rmpv::Value::String(s) => s.as_str(),
        _ => None,
    }
}

fn rmpv_as_bool(value: &rmpv::Value) -> Option<bool> {
    match value {
        rmpv::Value::Boolean(b) => Some(*b),
        _ => None,
    }
}

fn rmpv_as_array(value: &rmpv::Value) -> Option<&[rmpv::Value]> {
    match value {
        rmpv::Value::Array(a) => Some(a),
        _ => None,
    }
}

fn score_grammar_error(field: &str, expected: &str) -> String {
    format!("'{field}' must be {expected}")
}

fn validate_score_rmpv_map_keys(
    entries: &[(rmpv::Value, rmpv::Value)],
    path: &str,
) -> Result<(), String> {
    let mut seen = std::collections::HashSet::with_capacity(entries.len());
    for (key, _) in entries {
        let rmpv::Value::String(key) = key else {
            return Err(if path.is_empty() {
                "score request field names must be MessagePack strings".to_string()
            } else {
                format!("'{path}' field names must be MessagePack strings")
            });
        };
        let Some(key) = key.as_str() else {
            return Err(if path.is_empty() {
                "score request field names must be valid UTF-8 MessagePack strings".to_string()
            } else {
                format!("'{path}' field names must be valid UTF-8 MessagePack strings")
            });
        };
        if !seen.insert(key) {
            let field = if path.is_empty() {
                key.to_string()
            } else {
                format!("{path}.{key}")
            };
            return Err(format!("duplicate score request field '{field}'"));
        }
    }
    Ok(())
}

fn validate_generate_rmpv_map_keys(
    entries: &[(rmpv::Value, rmpv::Value)],
    path: &str,
) -> Result<(), (String, String)> {
    let mut seen = std::collections::HashSet::with_capacity(entries.len());
    for (key, _) in entries {
        let rmpv::Value::String(key) = key else {
            let label = if path.is_empty() { "request" } else { path };
            return Err((
                format!("'{label}' field names must be MessagePack strings"),
                label.to_string(),
            ));
        };
        let Some(key) = key.as_str() else {
            let label = if path.is_empty() { "request" } else { path };
            return Err((
                format!("'{label}' field names must be valid UTF-8 MessagePack strings"),
                label.to_string(),
            ));
        };
        if !seen.insert(key) {
            let field = if path.is_empty() {
                key.to_string()
            } else {
                format!("{path}.{key}")
            };
            return Err((format!("duplicate generate request field '{field}'"), field));
        }
    }
    Ok(())
}

fn validate_generate_rmpv_json_value(
    value: &rmpv::Value,
    path: &str,
) -> Result<(), (String, String)> {
    match value {
        rmpv::Value::Map(entries) => {
            validate_generate_rmpv_map_keys(entries, path)?;
            for (key, value) in entries {
                let rmpv::Value::String(key) = key else {
                    unreachable!("map keys were validated above")
                };
                let key = key.as_str().expect("map key UTF-8 was validated above");
                let child_path = if path.is_empty() {
                    key.to_string()
                } else {
                    format!("{path}.{key}")
                };
                validate_generate_rmpv_json_value(value, &child_path)?;
            }
            Ok(())
        }
        rmpv::Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                validate_generate_rmpv_json_value(value, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn validate_score_grammar_json(parsed: &Value) -> Result<(), String> {
    if let Some(instruction) = parsed.get("instruction") {
        if !instruction.is_null() && !instruction.is_string() {
            return Err(score_grammar_error("instruction", "a string or null"));
        }
    }

    match parsed.get("options") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(options)) => {
            if let Some(instruction) = options.get("instruction") {
                if !instruction.is_null() && !instruction.is_string() {
                    return Err(score_grammar_error(
                        "options.instruction",
                        "a string or null",
                    ));
                }
            }
            Ok(())
        }
        Some(_) => Err(score_grammar_error("options", "an object or null")),
    }
}

fn validate_score_grammar_rmpv(parsed: &[(rmpv::Value, rmpv::Value)]) -> Result<(), String> {
    validate_score_rmpv_map_keys(parsed, "")?;
    if let Some(instruction) = rmpv_map_get(parsed, "instruction") {
        if !matches!(instruction, rmpv::Value::Nil) && rmpv_as_str(instruction).is_none() {
            return Err(score_grammar_error("instruction", "a string or null"));
        }
    }

    match rmpv_map_get(parsed, "options") {
        None | Some(rmpv::Value::Nil) => Ok(()),
        Some(rmpv::Value::Map(options)) => {
            validate_score_rmpv_map_keys(options, "options")?;
            if let Some(instruction) = rmpv_map_get(options, "instruction") {
                if !matches!(instruction, rmpv::Value::Nil) && rmpv_as_str(instruction).is_none() {
                    return Err(score_grammar_error(
                        "options.instruction",
                        "a string or null",
                    ));
                }
            }
            Ok(())
        }
        Some(_) => Err(score_grammar_error("options", "an object or null")),
    }
}

fn rmpv_string_array(value: &rmpv::Value) -> Option<Vec<String>> {
    rmpv_as_array(value).map(|arr| {
        arr.iter()
            .filter_map(|v| rmpv_as_str(v).map(String::from))
            .collect()
    })
}

/// Parse an OpenAI ``stop`` field from a JSON value: a string (→ a
/// single-element vec) OR an array of strings. A non-string scalar or a
/// non-string array entry is rejected with the same OpenAI-shaped 400 the
/// chat path uses. ``None`` / JSON ``null`` → ``Ok(None)`` (worker
/// default). An empty array → ``Ok(None)`` (nothing to stop on),
/// matching the chat path. Shared by the ``/v1/generate`` JSON parser and
/// reused for the msgpack parser via a small rmpv→json bridge.
#[allow(clippy::result_large_err)]
fn parse_json_stop(value: Option<&serde_json::Value>) -> Result<Option<Vec<String>>, Response> {
    let invalid = || -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                "'stop' must be a string or array of strings",
                oai_type::INVALID_REQUEST,
                Some("stop"),
                oai_code::INVALID_REQUEST,
            )),
        )
            .into_response()
    };
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => {
            if let Some(s) = v.as_str() {
                Ok(Some(vec![s.to_string()]))
            } else if let Some(arr) = v.as_array() {
                let mut out: Vec<String> = Vec::with_capacity(arr.len());
                for entry in arr {
                    let Some(s) = entry.as_str() else {
                        return Err(invalid());
                    };
                    out.push(s.to_string());
                }
                if out.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(out))
                }
            } else {
                Err(invalid())
            }
        }
    }
}

/// Convert a small config-style `rmpv::Value` back to
/// `serde_json::Value` for the `options` / `output_schema` fields.
/// These are always tiny (a handful of flags/strings) and never
/// carry binary, so the conversion cost is negligible — and keeping
/// the `WorkParams` type stable here avoids a cascade of changes
/// into rest of the gateway that only cares about their structural
/// shape.
fn rmpv_to_json_owned(value: &rmpv::Value) -> serde_json::Value {
    rmpv_to_json(value.clone())
}

/// Carry the typed encode ``params.output_dtype`` through the queue's existing
/// runtime-options field. Queue workers already resolve
/// ``request options > profile runtime > default`` from this object, so
/// overlaying the typed parameter here preserves the HTTP encode contract:
/// ``params.output_dtype > options.output_dtype > profile > default``.
///
/// A non-object ``options`` value is already outside the typed request
/// contract. Preserve it so this compatibility bridge does not silently turn
/// a malformed request into a different valid one; the existing worker-side
/// validation remains authoritative for that case.
fn encode_options_with_output_dtype(
    options: Option<Value>,
    output_dtype: Option<Value>,
) -> Option<Value> {
    let Some(output_dtype) = output_dtype.filter(|value| !value.is_null()) else {
        return options;
    };

    match options {
        Some(Value::Object(mut options)) => {
            options.insert("output_dtype".to_string(), output_dtype);
            Some(Value::Object(options))
        }
        None | Some(Value::Null) => {
            let mut options = Map::new();
            options.insert("output_dtype".to_string(), output_dtype);
            Some(Value::Object(options))
        }
        malformed => malformed,
    }
}

/// Resolve the score instruction once at queue ingress so every worker engine
/// consumes the same canonical field. A present top-level string wins even
/// when it is empty; an absent or null top-level field promotes
/// `options.instruction` to match the typed Python API.
fn score_instruction_from_json(parsed: &Value) -> Option<String> {
    match parsed.get("instruction") {
        Some(Value::Null) | None => parsed
            .get("options")
            .and_then(|options| options.get("instruction"))
            .and_then(Value::as_str)
            .map(String::from),
        Some(instruction) => instruction.as_str().map(String::from),
    }
}

/// Msgpack twin of [`score_instruction_from_json`].
fn score_instruction_from_rmpv(parsed: &[(rmpv::Value, rmpv::Value)]) -> Option<String> {
    match rmpv_map_get(parsed, "instruction") {
        Some(rmpv::Value::Nil) | None => rmpv_map_get(parsed, "options")
            .and_then(|options| match options {
                rmpv::Value::Map(options) => rmpv_map_get(options, "instruction"),
                _ => None,
            })
            .and_then(rmpv_as_str)
            .map(String::from),
        Some(instruction) => rmpv_as_str(instruction).map(String::from),
    }
}

/// Build the per-request :class:`WorkParams` from a JSON body.
///
/// Score grammar violations use the generic parse-error path, which the
/// caller maps to a 400 before dispatch. Generate grammar violations carry
/// an already-shaped OpenAI 400 response through ``PreBuilt``. Other generate
/// parse misses still flow through ``Ok(WorkParams)`` with
/// ``params.generate == None`` for existing downstream validation.
#[allow(clippy::result_large_err)]
fn work_params_from_json(
    parsed: &serde_json::Value,
    endpoint: &str,
) -> Result<publisher::WorkParams, QueueParseError> {
    if endpoint == "score" {
        validate_score_grammar_json(parsed)?;
        return Ok(publisher::WorkParams {
            output_types: None,
            instruction: score_instruction_from_json(parsed),
            is_query: false,
            options: parsed.get("options").cloned(),
            labels: None,
            output_schema: None,
            query_item: Some(
                parsed
                    .get("query")
                    .cloned()
                    .map(json_item_to_rmpv)
                    .transpose()?
                    .unwrap_or_else(|| rmpv::Value::Map(Vec::new())),
            ),
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
        });
    }

    if endpoint == "generate" {
        return Ok(publisher::WorkParams {
            options: parse_generate_options_field(parsed.get("options"))
                .map_err(QueueParseError::PreBuilt)?,
            generate: generate_params_from_json(parsed).map_err(QueueParseError::PreBuilt)?,
            ..Default::default()
        });
    }

    let nested_params = parsed.get("params");
    let field = |key: &str| nested_params.and_then(|params| params.get(key));
    let options = if endpoint == "encode" {
        encode_options_with_output_dtype(field("options").cloned(), field("output_dtype").cloned())
    } else {
        field("options").cloned()
    };

    Ok(publisher::WorkParams {
        output_types: field("output_types").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        }),
        instruction: field("instruction")
            .and_then(|v| v.as_str())
            .map(String::from),
        is_query: field("is_query")
            .and_then(|v| v.as_bool())
            .or_else(|| {
                options
                    .as_ref()
                    .and_then(|value| value.get("is_query"))
                    .and_then(|v| v.as_bool())
            })
            .unwrap_or(false),
        options,
        labels: field("labels").and_then(|v| v.as_array()).map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        }),
        output_schema: field("output_schema").cloned(),
        query_item: None,
        generate: None,
        routing_key: None,
        prompt_cache_key: None,
    })
}

/// Maximum number of entries in a ``logit_bias`` map. Mirrors the cap
/// enforced inline by :func:`chat_params_from_json` so the two routes
/// share an identical DoS-prevention budget.
const MAX_LOGIT_BIAS_KEYS_GENERATE: usize = 1024;

/// Shared OpenAI-shaped 400 builder for the pure-function sampler
/// helpers below. Returns a fully-rendered ``Response`` so callers can
/// short-circuit with ``?``.
fn sampler_bad_request(message: String, param: &str, code: &'static str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json_openai_error(
            message,
            oai_type::INVALID_REQUEST,
            Some(param),
            code,
        )),
    )
        .into_response()
}

/// Parse OpenAI ``seed`` as a signed 64-bit per-request sampling seed.
/// ``null`` / absent → ``Ok(None)``. Values outside the signed 64-bit range
/// and non-integers reject with 400 ``invalid_request``.
///
/// Shared by every generation route so validation and wire semantics cannot
/// drift between the OpenAI-compatible and SIE-native surfaces.
#[allow(clippy::result_large_err)]
fn parse_seed_field(value: Option<&serde_json::Value>) -> Result<Option<i64>, Response> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => v.as_i64().map(Some).ok_or_else(|| {
            sampler_bad_request(
                "'seed' must be a signed 64-bit integer".to_string(),
                "seed",
                oai_code::INVALID_REQUEST,
            )
        }),
    }
}

/// Msgpack twin of [`parse_seed_field`]. Parse the native value directly so
/// non-JSON msgpack scalars (notably NaN / infinity and unsigned integers
/// above ``i64::MAX``) cannot be collapsed to ``null`` by the JSON bridge.
#[allow(clippy::result_large_err)]
fn parse_rmpv_seed_field(value: Option<&rmpv::Value>) -> Result<Option<i64>, Response> {
    match value {
        None | Some(rmpv::Value::Nil) => Ok(None),
        Some(rmpv::Value::Integer(value)) => value.as_i64().map(Some).ok_or_else(|| {
            sampler_bad_request(
                "'seed' must be a signed 64-bit integer".to_string(),
                "seed",
                oai_code::INVALID_REQUEST,
            )
        }),
        Some(_) => Err(sampler_bad_request(
            "'seed' must be a signed 64-bit integer".to_string(),
            "seed",
            oai_code::INVALID_REQUEST,
        )),
    }
}

/// Parse OpenAI ``logprobs``: a boolean opting into per-token log-
/// probabilities. ``null`` / absent → ``Ok(None)``. Anything else
/// (including strings like ``"true"``) → 400 ``invalid_request``.
#[allow(clippy::result_large_err)]
fn parse_logprobs_field(value: Option<&serde_json::Value>) -> Result<Option<bool>, Response> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(sampler_bad_request(
            "'logprobs' must be a boolean".to_string(),
            "logprobs",
            oai_code::INVALID_REQUEST,
        )),
    }
}

/// Parse OpenAI ``top_logprobs``: number of alternates per position.
/// Range ``[0, 20]`` per OpenAI spec. Non-integer / out-of-range → 400.
#[allow(clippy::result_large_err)]
fn parse_top_logprobs_field(value: Option<&serde_json::Value>) -> Result<Option<u32>, Response> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) if n <= 20 => Ok(Some(n as u32)),
            _ => Err(sampler_bad_request(
                "'top_logprobs' must be an integer in [0, 20]".to_string(),
                "top_logprobs",
                oai_code::INVALID_REQUEST,
            )),
        },
    }
}

/// Cross-field rule: ``top_logprobs > 0`` requires ``logprobs: true``.
/// OpenAI rejects this combination (``logprobs: false`` with non-zero
/// ``top_logprobs``) so we mirror the same 400. Callers pass the
/// already-parsed values so this stays a pure function.
#[allow(clippy::result_large_err)]
fn check_logprobs_consistency(
    logprobs: Option<bool>,
    top_logprobs: Option<u32>,
) -> Result<(), Response> {
    if matches!(top_logprobs, Some(n) if n > 0) && !matches!(logprobs, Some(true)) {
        return Err(sampler_bad_request(
            "'top_logprobs' requires 'logprobs: true'".to_string(),
            "top_logprobs",
            oai_code::INVALID_REQUEST,
        ));
    }
    Ok(())
}

/// Parse OpenAI ``logit_bias``: ``{token_id: bias_float}`` map. Gateway
/// caps the map size (DoS) and per-value range (sampler safety).
///
/// Mirrors :func:`chat_params_from_json`'s inline parser (line ~2940) —
/// same caps, same error envelopes. An empty map collapses to
/// ``Ok(None)`` matching the chat path's behaviour.
#[allow(clippy::result_large_err)]
fn parse_logit_bias_field(
    value: Option<&serde_json::Value>,
) -> Result<Option<std::collections::BTreeMap<String, f64>>, Response> {
    let Some(v) = value else {
        return Ok(None);
    };
    if v.is_null() {
        return Ok(None);
    }
    let Some(map) = v.as_object() else {
        return Err(sampler_bad_request(
            "'logit_bias' must be an object".to_string(),
            "logit_bias",
            oai_code::INVALID_REQUEST,
        ));
    };
    if map.len() > MAX_LOGIT_BIAS_KEYS_GENERATE {
        return Err(sampler_bad_request(
            format!("'logit_bias' has too many entries (max {MAX_LOGIT_BIAS_KEYS_GENERATE})"),
            "logit_bias",
            oai_code::INVALID_REQUEST,
        ));
    }
    let mut out = std::collections::BTreeMap::new();
    for (k, val) in map.iter() {
        if k.parse::<i64>().is_err() {
            return Err(sampler_bad_request(
                format!("'logit_bias' keys must be token-id integers as strings (got {k:?})"),
                "logit_bias",
                oai_code::INVALID_REQUEST,
            ));
        }
        let f = val.as_f64().filter(|f| f.is_finite());
        let Some(f) = f else {
            return Err(sampler_bad_request(
                "'logit_bias' values must be finite numbers".to_string(),
                "logit_bias",
                oai_code::INVALID_REQUEST,
            ));
        };
        if !(-100.0..=100.0).contains(&f) {
            return Err(sampler_bad_request(
                "'logit_bias' values must be in [-100.0, 100.0]".to_string(),
                "logit_bias",
                oai_code::INVALID_REQUEST,
            ));
        }
        out.insert(k.clone(), f);
    }
    if out.is_empty() {
        Ok(None)
    } else {
        Ok(Some(out))
    }
}

#[allow(clippy::result_large_err)]
fn parse_optional_string_field(
    value: Option<&serde_json::Value>,
    field: &'static str,
) -> Result<Option<String>, Response> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(sampler_bad_request(
            format!("'{field}' must be a string"),
            field,
            oai_code::INVALID_REQUEST,
        )),
    }
}

#[allow(clippy::result_large_err)]
fn validate_generate_options_map(
    options: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), Response> {
    const ACCEPTED: &[&str] = &[
        "default_sampling",
        "stop_tokens",
        "first_chunk_timeout_s",
        "inter_chunk_timeout_s",
        "overall_timeout_s",
    ];
    for key in options.keys() {
        if !ACCEPTED.contains(&key.as_str()) {
            let field = format!("options.{key}");
            return Err(sampler_bad_request(
                format!("'{field}' is not supported by this endpoint"),
                &field,
                oai_code::UNSUPPORTED_FIELD,
            ));
        }
    }

    if let Some(value) = options.get("default_sampling") {
        let sampling = match value {
            serde_json::Value::Object(sampling) => sampling,
            _ => {
                return Err(sampler_bad_request(
                    "'options.default_sampling' must be an object".to_string(),
                    "options.default_sampling",
                    oai_code::INVALID_REQUEST,
                ));
            }
        };
        const SAMPLERS: &[&str] = &[
            "temperature",
            "top_p",
            "presence_penalty",
            "top_k",
            "min_new_tokens",
        ];
        for (key, value) in sampling {
            let field = format!("options.default_sampling.{key}");
            if !SAMPLERS.contains(&key.as_str()) {
                return Err(sampler_bad_request(
                    format!("'{field}' is not supported"),
                    &field,
                    oai_code::UNSUPPORTED_FIELD,
                ));
            }
            let valid = match key.as_str() {
                "temperature" => value
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && value >= 0.0),
                "top_p" => value
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && value > 0.0 && value <= 1.0),
                "presence_penalty" => value
                    .as_f64()
                    .is_some_and(|value| value.is_finite() && (-2.0..=2.0).contains(&value)),
                "top_k" => value.as_u64().is_some_and(|value| value >= 1),
                "min_new_tokens" => value.as_u64().is_some(),
                _ => unreachable!("sampling key was checked above"),
            };
            if !valid {
                return Err(sampler_bad_request(
                    format!("'{field}' has an invalid value"),
                    &field,
                    oai_code::INVALID_REQUEST,
                ));
            }
        }
    }

    if let Some(value) = options.get("stop_tokens") {
        let valid = value.as_array().is_some_and(|values| {
            values
                .iter()
                .all(|value| value.as_str().is_some_and(|value| !value.is_empty()))
        });
        if !valid {
            return Err(sampler_bad_request(
                "'options.stop_tokens' must be an array of non-empty strings".to_string(),
                "options.stop_tokens",
                oai_code::INVALID_REQUEST,
            ));
        }
    }

    for key in [
        "first_chunk_timeout_s",
        "inter_chunk_timeout_s",
        "overall_timeout_s",
    ] {
        if let Some(value) = options.get(key) {
            let valid = value
                .as_f64()
                .is_some_and(|value| value.is_finite() && value > 0.0);
            if !valid {
                let field = format!("options.{key}");
                return Err(sampler_bad_request(
                    format!("'{field}' must be a positive number"),
                    &field,
                    oai_code::INVALID_REQUEST,
                ));
            }
        }
    }
    Ok(())
}

#[allow(clippy::result_large_err)]
fn parse_generate_options_field(
    value: Option<&serde_json::Value>,
) -> Result<Option<serde_json::Value>, Response> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(options) = value.as_object() else {
        return Err(sampler_bad_request(
            "'options' must be an object".to_string(),
            "options",
            oai_code::INVALID_REQUEST,
        ));
    };

    let mut options = options.clone();
    match options.remove("profile") {
        None | Some(serde_json::Value::Null) => {}
        Some(serde_json::Value::String(profile)) if profile == "default" => {}
        Some(serde_json::Value::String(profile)) => {
            return Err(sampler_bad_request(
                format!(
                    "non-default options.profile '{profile}' cannot select a routed model variant; use the 'model:profile' identity"
                ),
                "options.profile",
                oai_code::INVALID_REQUEST,
            ));
        }
        Some(_) => {
            return Err(sampler_bad_request(
                "'options.profile' must be a string".to_string(),
                "options.profile",
                oai_code::INVALID_REQUEST,
            ));
        }
    }
    validate_generate_options_map(&options)?;
    Ok(Some(serde_json::Value::Object(options)))
}

/// Parse SIE-extension ``lora_adapter``: optional served-name. The
/// gateway only enforces "non-empty string" here; cross-checking
/// against the model's advertised adapters happens after model
/// resolution in :func:`queue_mode_proxy` (mirrors chat's gate at
/// line ~4110).
#[allow(clippy::result_large_err)]
fn parse_lora_adapter_field(value: Option<&serde_json::Value>) -> Result<Option<String>, Response> {
    match value {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) if !s.is_empty() => Ok(Some(s.clone())),
        Some(_) => Err(sampler_bad_request(
            "'lora_adapter' must be a non-empty string".to_string(),
            "lora_adapter",
            oai_code::INVALID_REQUEST,
        )),
    }
}

/// Accept-list of top-level keys recognised on ``/v1/generate``. Any
/// key outside this set surfaces as ``400 unsupported_field`` per A's
/// strict allow-list discipline (M8) — the same pattern as chat's
/// ``ACCEPTED`` block at line ~3170.
///
/// Native generate intentionally remains single-candidate. Compatibility
/// fields whose output cannot be represented by the native response
/// (``n`` / ``best_of`` / ``stream_options``) stay on the OpenAI adapters.
const GENERATE_ACCEPTED_FIELDS: &[&str] = &[
    "prompt",
    "images",
    "max_new_tokens",
    "temperature",
    "top_p",
    "stop",
    "frequency_penalty",
    "presence_penalty",
    "grammar",
    "routing_key",
    "prompt_cache_key",
    "safety_identifier",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "lora_adapter",
    "options",
    "stream",
];

/// Parse the SIE-native JSON image envelope used by ``/v1/generate``.
///
/// The gateway never fetches URLs. Each entry is an exact
/// ``{data: <standard-base64>, format?: <short token>}`` object and remains
/// base64 on the generate work wire so it can round-trip through the
/// sidecar's ``serde_json::Value`` representation.
#[allow(clippy::result_large_err)]
fn parse_native_generate_images(
    value: Option<&serde_json::Value>,
) -> Result<Option<Vec<publisher::ChatImage>>, Response> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(entries) = value.as_array() else {
        return Err(sampler_bad_request(
            "'images' must be a non-empty array".to_string(),
            "images",
            oai_code::INVALID_REQUEST,
        ));
    };
    if entries.is_empty() || entries.len() > MAX_GENERATE_IMAGES {
        return Err(sampler_bad_request(
            format!("'images' must contain between 1 and {MAX_GENERATE_IMAGES} entries"),
            "images",
            oai_code::INVALID_REQUEST,
        ));
    }
    let mut images = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let param = format!("images[{index}]");
        let Some(object) = entry.as_object() else {
            return Err(sampler_bad_request(
                format!("'{param}' must be an object"),
                &param,
                oai_code::INVALID_REQUEST,
            ));
        };
        if let Some(unknown) = object
            .keys()
            .find(|key| !matches!(key.as_str(), "data" | "format"))
        {
            let unknown_param = format!("{param}.{unknown}");
            return Err(sampler_bad_request(
                format!("'{unknown_param}' is not supported"),
                &unknown_param,
                oai_code::UNSUPPORTED_FIELD,
            ));
        }
        let data_param = format!("{param}.data");
        let Some(data) = object.get("data").and_then(serde_json::Value::as_str) else {
            return Err(sampler_bad_request(
                format!("'{data_param}' must be a non-empty base64 string"),
                &data_param,
                oai_code::INVALID_REQUEST,
            ));
        };
        if !native_generate_image_encoded_size_allowed(data.len()) {
            return Err(sampler_bad_request(
                format!(
                    "'{data_param}' must decode to between 1 byte and {MAX_GENERATE_IMAGE_BYTES} bytes"
                ),
                &data_param,
                oai_code::INVALID_REQUEST,
            ));
        }
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|_| {
                sampler_bad_request(
                    format!("'{data_param}' must be valid standard base64"),
                    &data_param,
                    oai_code::INVALID_REQUEST,
                )
            })?;
        if decoded.is_empty() || decoded.len() > MAX_GENERATE_IMAGE_BYTES {
            return Err(sampler_bad_request(
                format!(
                    "'{data_param}' must decode to between 1 byte and {MAX_GENERATE_IMAGE_BYTES} bytes"
                ),
                &data_param,
                oai_code::INVALID_REQUEST,
            ));
        }
        let format = match object.get("format") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(format))
                if !format.is_empty()
                    && format.len() <= 32
                    && format.bytes().all(|byte| {
                        byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'+' | b'-')
                    }) =>
            {
                Some(format.to_ascii_lowercase())
            }
            Some(_) => {
                let format_param = format!("{param}.format");
                return Err(sampler_bad_request(
                    format!("'{format_param}' must be a short media-format token"),
                    &format_param,
                    oai_code::INVALID_REQUEST,
                ));
            }
        };
        images.push(publisher::ChatImage {
            data: data.to_string(),
            format,
        });
    }
    Ok(Some(images))
}

fn native_generate_image_encoded_size_allowed(length: usize) -> bool {
    (1..=MAX_GENERATE_IMAGE_BASE64_CHARS).contains(&length)
}

fn generate_params_have_images(params: &publisher::GenerateParams) -> bool {
    matches!(
        &params.input,
        publisher::GenerateInput::Messages { messages }
            if messages.iter().any(|message| {
                message.images.as_ref().is_some_and(|images| !images.is_empty())
            })
    )
}

fn native_generate_image_input_allowed(
    params: Option<&publisher::GenerateParams>,
    model_supports_vision_generation: bool,
) -> bool {
    !params.is_some_and(generate_params_have_images) || model_supports_vision_generation
}

/// Parse the walking-skeleton ``/v1/generate/{model}`` JSON body shape into a
/// :class:`GenerateParams`. Returns ``Ok(None)`` when required fields are
/// missing or malformed — the caller surfaces a generic 400 in that case.
/// Returns ``Err(Response)`` when grammar validation fails;
/// that response is the precise OpenAI envelope and is returned
/// verbatim.
///
/// Text-only requests preserve the ``Prompt`` arm. Native image requests and
/// chat requests assemble the existing ``Messages`` arm so both reach the
/// worker's model-native template renderer.
#[allow(clippy::result_large_err)]
fn generate_params_from_json(
    parsed: &serde_json::Value,
) -> Result<Option<publisher::GenerateParams>, Response> {
    let prompt = match parsed.get("prompt") {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(serde_json::Value::String(prompt)) if !prompt.is_empty() => prompt.clone(),
        Some(serde_json::Value::String(_)) => return Ok(None),
        Some(_) => {
            return Err(sampler_bad_request(
                "'prompt' must be a string".to_string(),
                "prompt",
                oai_code::INVALID_REQUEST,
            ))
        }
    };
    let images = parse_native_generate_images(parsed.get("images"))?;
    // Granular `max_new_tokens` rejection so OpenAI-shaped error
    // envelopes carry the correct `param` field. Previously every
    // failure mode collapsed to `Ok(None)` and the caller emitted
    // `param: "prompt"` for all of them, which broke SDKs that branch
    // on `error.param`.
    let max_new_tokens = match parsed.get("max_new_tokens") {
        None | Some(serde_json::Value::Null) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    "'max_new_tokens' is required".to_string(),
                    oai_type::INVALID_REQUEST,
                    Some("max_new_tokens"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response())
        }
        Some(v) => match v.as_u64().and_then(|n| u32::try_from(n).ok()) {
            Some(n) => n,
            None => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        "'max_new_tokens' must be a positive integer fitting in u32".to_string(),
                        oai_type::INVALID_REQUEST,
                        Some("max_new_tokens"),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response())
            }
        },
    };
    if max_new_tokens == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                "'max_new_tokens' must be > 0".to_string(),
                oai_type::INVALID_REQUEST,
                Some("max_new_tokens"),
                oai_code::INVALID_REQUEST,
            )),
        )
            .into_response());
    }
    // Preserve absent-vs-explicit sampler semantics so profile defaults can
    // apply, while rejecting values that cannot execute faithfully.
    let temperature = match parsed.get("temperature") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && f >= 0.0 && f <= f32::MAX as f64 => Some(f as f32),
            _ => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        "'temperature' must be a finite number >= 0",
                        oai_type::INVALID_REQUEST,
                        Some("temperature"),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response());
            }
        },
    };
    let top_p = match parsed.get("top_p") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match v.as_f64() {
            Some(f) if f.is_finite() && f > 0.0 && f <= 1.0 => Some(f as f32),
            _ => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        "'top_p' must be in (0, 1]",
                        oai_type::INVALID_REQUEST,
                        Some("top_p"),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response());
            }
        },
    };
    // OpenAI accepts ``stop`` as a string OR an array of strings; a
    // non-string scalar / non-string array entry is a 400. The prior
    // code took ``as_array()`` only (silently dropping a string ``stop``)
    // and ``filter_map(as_str)`` (silently dropping non-string entries),
    // which both diverged from the chat path. Reuse the shared helper so
    // the two routes stay in lockstep.
    let stop = parse_json_stop(parsed.get("stop"))?;
    // SIE-native ``/v1/generate`` now advertises the same penalty
    // surface as the OpenAI chat path: range-validated in ``[-2.0,
    // 2.0]``, absent → worker sampler default. Out-of-range values
    // reject with the same 400 envelope shape so SDK error handling
    // stays uniform across both routes.
    #[allow(clippy::result_large_err)]
    fn parse_penalty(
        parsed: &serde_json::Value,
        field: &'static str,
    ) -> Result<Option<f64>, Response> {
        match parsed.get(field) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(v) => match v.as_f64() {
                Some(f) if f.is_finite() && (-2.0..=2.0).contains(&f) => Ok(Some(f)),
                _ => Err((
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        format!("'{field}' must be a number in [-2.0, 2.0]"),
                        oai_type::INVALID_REQUEST,
                        Some(field),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response()),
            },
        }
    }
    let frequency_penalty = parse_penalty(parsed, "frequency_penalty")?;
    let presence_penalty = parse_penalty(parsed, "presence_penalty")?;
    let grammar = match parsed.get("grammar") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => match super::grammar::parse_grammar(v) {
            super::grammar::GrammarParseResult::Ok(g) => Some(g),
            super::grammar::GrammarParseResult::Err(resp) => return Err(resp),
        },
    };
    let routing_key = parse_optional_string_field(parsed.get("routing_key"), "routing_key")?
        .filter(|value| !value.is_empty());
    let prompt_cache_key =
        parse_optional_string_field(parsed.get("prompt_cache_key"), "prompt_cache_key")?
            .filter(|value| !value.is_empty());
    // Privacy contract: `safety_identifier` is parsed and
    // immediately discarded — never placed on the JetStream wire,
    // never logged with its raw value. The trace log records only
    // its presence so we can confirm at runtime that clients are
    // sending it without exposing the value itself.
    let safety_identifier =
        parse_optional_string_field(parsed.get("safety_identifier"), "safety_identifier")?;
    if safety_identifier.is_some() {
        tracing::trace!("safety_identifier acknowledged and dropped (JSON)");
    }

    parse_generate_options_field(parsed.get("options"))?;

    // Native generate keeps sampler controls that have faithful native
    // semantics. Multi-candidate / best-of controls remain compatibility-only
    // until the native response types can represent every candidate.
    let seed = parse_seed_field(parsed.get("seed"))?;
    let logit_bias = parse_logit_bias_field(parsed.get("logit_bias"))?;
    let lora_adapter = parse_lora_adapter_field(parsed.get("lora_adapter"))?;

    // ``stream`` parsing is duplicated with :func:`stream_flag_from_body`
    // (which the SSE branch in ``queue_mode_proxy`` consults). Parsing it
    // here also gates logprobs, whose native shape exists only on SSE.
    let stream = match parsed.get("stream") {
        None | Some(serde_json::Value::Null) => false,
        Some(serde_json::Value::Bool(b)) => *b,
        Some(_) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    "'stream' must be a boolean".to_string(),
                    oai_type::INVALID_REQUEST,
                    Some("stream"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response());
        }
    };
    // Strict accept-list. Anything outside the supported set
    // surfaces as ``400 unsupported_field`` (A's discipline) so SDK
    // typos / unknown future knobs fail fast instead of being
    // silently dropped. Tool-calling fields (``tools`` /
    // ``tool_choice`` / ``parallel_tool_calls``) stay chat-only and
    // are rejected here.
    if let Some(obj) = parsed.as_object() {
        for key in obj.keys() {
            if !GENERATE_ACCEPTED_FIELDS.contains(&key.as_str()) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        format!("'{key}' is not supported by this endpoint"),
                        oai_type::INVALID_REQUEST,
                        Some(key.as_str()),
                        oai_code::UNSUPPORTED_FIELD,
                    )),
                )
                    .into_response());
            }
        }
    }

    for field in ["logprobs", "top_logprobs"] {
        if !stream && parsed.get(field).is_some_and(|value| !value.is_null()) {
            return Err(sampler_bad_request(
                format!("'{field}' is supported only with 'stream: true' on the native endpoint"),
                field,
                oai_code::UNSUPPORTED_FIELD,
            ));
        }
    }
    let logprobs = parse_logprobs_field(parsed.get("logprobs"))?;
    let top_logprobs = parse_top_logprobs_field(parsed.get("top_logprobs"))?;
    check_logprobs_consistency(logprobs, top_logprobs)?;

    let input = match images {
        Some(images) => publisher::GenerateInput::Messages {
            messages: vec![publisher::ChatMessage {
                role: "user".to_string(),
                content: prompt,
                tool_calls: None,
                tool_call_id: None,
                images: Some(images),
                content_parts: None,
            }],
        },
        None => publisher::GenerateInput::Prompt { prompt },
    };
    Ok(Some(publisher::GenerateParams {
        input,
        max_new_tokens,
        temperature,
        top_p,
        stop,
        frequency_penalty,
        presence_penalty,
        // top_k / repetition_penalty / min_tokens / chat_template_kwargs
        // are exposed only via the chat route today; SIE-native
        // /v1/generate stays a thin prompt wrapper and ignores them.
        top_k: None,
        repetition_penalty: None,
        min_tokens: None,
        chat_template_kwargs: None,
        grammar,
        routing_key,
        prompt_cache_key,
        // OpenAI tool-calling fields are exposed only via the chat
        // route — SIE-native /v1/generate stays a thin wrapper over
        // the raw prompt API.
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        seed,
        logit_bias,
        logprobs,
        top_logprobs,
        n: None,
        best_of: None,
        stream,
        lora_adapter,
    }))
}

#[allow(clippy::result_large_err)]
fn work_params_from_rmpv(
    parsed: &[(rmpv::Value, rmpv::Value)],
    endpoint: &str,
) -> Result<publisher::WorkParams, QueueParseError> {
    if endpoint == "score" {
        validate_score_grammar_rmpv(parsed)?;
        return Ok(publisher::WorkParams {
            output_types: None,
            instruction: score_instruction_from_rmpv(parsed),
            is_query: false,
            options: rmpv_map_get(parsed, "options").map(rmpv_to_json_owned),
            labels: None,
            output_schema: None,
            query_item: Some(
                rmpv_map_get(parsed, "query")
                    .cloned()
                    .unwrap_or_else(|| rmpv::Value::Map(Vec::new())),
            ),
            generate: None,
            routing_key: None,
            prompt_cache_key: None,
        });
    }

    if endpoint == "generate" {
        return Ok(publisher::WorkParams {
            options: parse_generate_options_field(
                rmpv_map_get(parsed, "options")
                    .map(rmpv_to_json_owned)
                    .as_ref(),
            )
            .map_err(QueueParseError::PreBuilt)?,
            generate: generate_params_from_rmpv(parsed).map_err(QueueParseError::PreBuilt)?,
            ..Default::default()
        });
    }

    // For `encode`/`extract`, match ``sie_server`` / msgspec: tuning fields live
    // only under the ``params`` object (no top-level merge).
    let nested = rmpv_map_get(parsed, "params").and_then(|v| match v {
        rmpv::Value::Map(m) => Some(m.as_slice()),
        _ => None,
    });
    let field = |key: &str| -> Option<&rmpv::Value> { nested.and_then(|m| rmpv_map_get(m, key)) };
    let options_rmpv = field("options");
    let options = if endpoint == "encode" {
        encode_options_with_output_dtype(
            options_rmpv.map(rmpv_to_json_owned),
            field("output_dtype").map(rmpv_to_json_owned),
        )
    } else {
        options_rmpv.map(rmpv_to_json_owned)
    };
    let is_query = field("is_query")
        .and_then(rmpv_as_bool)
        .or_else(|| {
            options_rmpv
                .and_then(|v| match v {
                    rmpv::Value::Map(m) => rmpv_map_get(m, "is_query"),
                    _ => None,
                })
                .and_then(rmpv_as_bool)
        })
        .unwrap_or(false);

    Ok(publisher::WorkParams {
        output_types: field("output_types").and_then(rmpv_string_array),
        instruction: field("instruction").and_then(rmpv_as_str).map(String::from),
        is_query,
        options,
        labels: field("labels").and_then(rmpv_string_array),
        output_schema: field("output_schema").map(rmpv_to_json_owned),
        query_item: None,
        generate: None,
        routing_key: None,
        prompt_cache_key: None,
    })
}

#[allow(clippy::result_large_err)]
fn generate_params_from_rmpv(
    parsed: &[(rmpv::Value, rmpv::Value)],
) -> Result<Option<publisher::GenerateParams>, Response> {
    let validation_response = |(message, parameter): (String, String)| {
        sampler_bad_request(message, &parameter, oai_code::INVALID_REQUEST)
    };
    validate_generate_rmpv_map_keys(parsed, "").map_err(validation_response)?;
    for (key, value) in parsed {
        let rmpv::Value::String(key) = key else {
            unreachable!("top-level generate map keys were validated above")
        };
        validate_generate_rmpv_json_value(
            value,
            key.as_str()
                .expect("top-level generate key UTF-8 was validated above"),
        )
        .map_err(validation_response)?;
    }

    let prompt = match rmpv_map_get(parsed, "prompt") {
        None | Some(rmpv::Value::Nil) => return Ok(None),
        Some(rmpv::Value::String(prompt)) if prompt.as_str().is_some_and(|p| !p.is_empty()) => {
            prompt.as_str().expect("checked above").to_string()
        }
        Some(rmpv::Value::String(_)) => return Ok(None),
        Some(_) => {
            return Err(sampler_bad_request(
                "'prompt' must be a string".to_string(),
                "prompt",
                oai_code::INVALID_REQUEST,
            ))
        }
    };
    let images = parse_native_generate_images(
        rmpv_map_get(parsed, "images")
            .map(rmpv_to_json_owned)
            .as_ref(),
    )?;
    // Mirror the JSON path's granular error attribution so SDKs that
    // branch on `error.param` see the same field name whether the
    // wire format is JSON or msgpack.
    let max_new_tokens = match rmpv_map_get(parsed, "max_new_tokens") {
        None | Some(rmpv::Value::Nil) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    "'max_new_tokens' is required".to_string(),
                    oai_type::INVALID_REQUEST,
                    Some("max_new_tokens"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response())
        }
        Some(v) => match v {
            rmpv::Value::Integer(i) => match i.as_u64().and_then(|n| u32::try_from(n).ok()) {
                Some(n) => n,
                None => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(json_openai_error(
                            "'max_new_tokens' must be a positive integer fitting in u32"
                                .to_string(),
                            oai_type::INVALID_REQUEST,
                            Some("max_new_tokens"),
                            oai_code::INVALID_REQUEST,
                        )),
                    )
                        .into_response())
                }
            },
            _ => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(json_openai_error(
                        "'max_new_tokens' must be an integer".to_string(),
                        oai_type::INVALID_REQUEST,
                        Some("max_new_tokens"),
                        oai_code::INVALID_REQUEST,
                    )),
                )
                    .into_response())
            }
        },
    };
    if max_new_tokens == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json_openai_error(
                "'max_new_tokens' must be > 0".to_string(),
                oai_type::INVALID_REQUEST,
                Some("max_new_tokens"),
                oai_code::INVALID_REQUEST,
            )),
        )
            .into_response());
    }
    let parse_number = |field: &str| -> Result<Option<f64>, Response> {
        match rmpv_map_get(parsed, field) {
            None | Some(rmpv::Value::Nil) => Ok(None),
            Some(rmpv::Value::F32(value)) => Ok(Some(*value as f64)),
            Some(rmpv::Value::F64(value)) => Ok(Some(*value)),
            Some(rmpv::Value::Integer(value)) => value.as_f64().map(Some).ok_or_else(|| {
                sampler_bad_request(
                    format!("'{field}' must be a number"),
                    field,
                    oai_code::INVALID_REQUEST,
                )
            }),
            Some(_) => Err(sampler_bad_request(
                format!("'{field}' must be a number"),
                field,
                oai_code::INVALID_REQUEST,
            )),
        }
    };
    let temperature = match parse_number("temperature")? {
        None => None,
        Some(f) if f.is_finite() && f >= 0.0 && f <= f32::MAX as f64 => Some(f as f32),
        Some(_) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    "'temperature' must be a finite number >= 0",
                    oai_type::INVALID_REQUEST,
                    Some("temperature"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response());
        }
    };
    let top_p = match parse_number("top_p")? {
        None => None,
        Some(f) if f.is_finite() && f > 0.0 && f <= 1.0 => Some(f as f32),
        Some(_) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    "'top_p' must be in (0, 1]",
                    oai_type::INVALID_REQUEST,
                    Some("top_p"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response());
        }
    };
    // Accept ``stop`` as a string OR array of strings and reject a
    // non-string scalar / non-string array entry with the same 400 as
    // the chat + JSON paths. The prior ``rmpv_string_array`` accepted
    // arrays only (silently dropping a string ``stop``) and
    // ``filter_map``-ed non-string entries away silently. Bridge the
    // rmpv branch to ``serde_json`` once (``stop`` is tiny — a handful
    // of short strings — so the conversion cost is negligible) and reuse
    // the shared ``parse_json_stop`` so all three routes stay in
    // lockstep.
    let stop = match rmpv_map_get(parsed, "stop") {
        None | Some(rmpv::Value::Nil) => None,
        Some(v) => parse_json_stop(Some(&rmpv_to_json_owned(v)))?,
    };
    // Penalty parsing mirrors the JSON twin: validate in
    // ``[-2.0, 2.0]``, absent / nil → ``None`` (worker default). Out-
    // of-range surfaces the same 400 envelope as the JSON path.
    #[allow(clippy::result_large_err)]
    fn rmpv_parse_penalty(
        parsed: &[(rmpv::Value, rmpv::Value)],
        field: &'static str,
    ) -> Result<Option<f64>, Response> {
        let val = match rmpv_map_get(parsed, field) {
            None | Some(rmpv::Value::Nil) => return Ok(None),
            Some(v) => v,
        };
        let f = match val {
            rmpv::Value::F32(f) => Some(*f as f64),
            rmpv::Value::F64(f) => Some(*f),
            rmpv::Value::Integer(i) => i.as_f64(),
            _ => None,
        };
        match f {
            Some(f) if f.is_finite() && (-2.0..=2.0).contains(&f) => Ok(Some(f)),
            _ => Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    format!("'{field}' must be a number in [-2.0, 2.0]"),
                    oai_type::INVALID_REQUEST,
                    Some(field),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response()),
        }
    }
    let frequency_penalty = rmpv_parse_penalty(parsed, "frequency_penalty")?;
    let presence_penalty = rmpv_parse_penalty(parsed, "presence_penalty")?;
    // Msgpack twin of the JSON ``grammar`` field. Convert
    // the rmpv branch to ``serde_json`` once so we can reuse the
    // gateway's shared :func:`grammar::parse_grammar` (which is
    // serde_json-shaped — JSON has no binary/ext types so the
    // conversion is lossless for grammar payloads).
    let grammar = match rmpv_map_get(parsed, "grammar") {
        None | Some(rmpv::Value::Nil) => None,
        Some(v) => {
            let as_json = rmpv_to_json_owned(v);
            match super::grammar::parse_grammar(&as_json) {
                super::grammar::GrammarParseResult::Ok(g) => Some(g),
                super::grammar::GrammarParseResult::Err(resp) => return Err(resp),
            }
        }
    };
    let bridge = |field: &str| rmpv_map_get(parsed, field).map(rmpv_to_json_owned);
    let routing_key = parse_optional_string_field(bridge("routing_key").as_ref(), "routing_key")?
        .filter(|value| !value.is_empty());
    let prompt_cache_key =
        parse_optional_string_field(bridge("prompt_cache_key").as_ref(), "prompt_cache_key")?
            .filter(|value| !value.is_empty());
    // See `generate_params_from_json` — same privacy contract for
    // msgpack ingress. We intentionally do **not** thread the field
    // into `WorkParams`/`WorkItem`.
    let safety_identifier =
        parse_optional_string_field(bridge("safety_identifier").as_ref(), "safety_identifier")?;
    if safety_identifier.is_some() {
        tracing::trace!("safety_identifier acknowledged and dropped (msgpack)");
    }

    parse_generate_options_field(bridge("options").as_ref())?;

    // Bridge each msgpack sampler field through to JSON once so
    // we can reuse the pure-function helpers shared with the JSON
    // parser (no per-format duplication of range / cross-field
    // checks). Each field is tiny (scalar / short map) so the
    // conversion cost is negligible.
    let seed = parse_rmpv_seed_field(rmpv_map_get(parsed, "seed"))?;
    let logit_bias = parse_logit_bias_field(bridge("logit_bias").as_ref())?;
    let lora_adapter = parse_lora_adapter_field(bridge("lora_adapter").as_ref())?;

    let stream = match rmpv_map_get(parsed, "stream") {
        None | Some(rmpv::Value::Nil) => false,
        Some(rmpv::Value::Boolean(b)) => *b,
        Some(_) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    "'stream' must be a boolean".to_string(),
                    oai_type::INVALID_REQUEST,
                    Some("stream"),
                    oai_code::INVALID_REQUEST,
                )),
            )
                .into_response());
        }
    };
    // Strict accept-list — map keys are already proven unique UTF-8 strings.
    for (k, _) in parsed.iter() {
        let rmpv::Value::String(key) = k else {
            unreachable!("generate map keys were validated above")
        };
        let key = key
            .as_str()
            .expect("generate map key UTF-8 was validated above");
        if !GENERATE_ACCEPTED_FIELDS.contains(&key) {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json_openai_error(
                    format!("'{key}' is not supported by this endpoint"),
                    oai_type::INVALID_REQUEST,
                    Some(key),
                    oai_code::UNSUPPORTED_FIELD,
                )),
            )
                .into_response());
        }
    }

    for field in ["logprobs", "top_logprobs"] {
        if !stream
            && rmpv_map_get(parsed, field).is_some_and(|value| !matches!(value, rmpv::Value::Nil))
        {
            return Err(sampler_bad_request(
                format!("'{field}' is supported only with 'stream: true' on the native endpoint"),
                field,
                oai_code::UNSUPPORTED_FIELD,
            ));
        }
    }
    let logprobs = parse_logprobs_field(bridge("logprobs").as_ref())?;
    let top_logprobs = parse_top_logprobs_field(bridge("top_logprobs").as_ref())?;
    check_logprobs_consistency(logprobs, top_logprobs)?;

    let input = match images {
        Some(images) => publisher::GenerateInput::Messages {
            messages: vec![publisher::ChatMessage {
                role: "user".to_string(),
                content: prompt,
                tool_calls: None,
                tool_call_id: None,
                images: Some(images),
                content_parts: None,
            }],
        },
        None => publisher::GenerateInput::Prompt { prompt },
    };
    Ok(Some(publisher::GenerateParams {
        input,
        max_new_tokens,
        temperature,
        top_p,
        stop,
        frequency_penalty,
        presence_penalty,
        // top_k / repetition_penalty / min_tokens / chat_template_kwargs
        // are exposed only via the chat route today; SIE-native
        // /v1/generate stays a thin prompt wrapper and ignores them.
        top_k: None,
        repetition_penalty: None,
        min_tokens: None,
        chat_template_kwargs: None,
        grammar,
        routing_key,
        prompt_cache_key,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        seed,
        logit_bias,
        logprobs,
        top_logprobs,
        n: None,
        best_of: None,
        stream,
        lora_adapter,
    }))
}

/// Convert a native JSON item to `rmpv::Value`, preserving the media bytes
/// contract before the item is published to workers as msgpack.
fn json_item_to_rmpv(value: serde_json::Value) -> Result<rmpv::Value, String> {
    let mut value = json_to_rmpv(value);
    decode_native_media_fields(&mut value)?;
    Ok(value)
}

fn decode_native_media_fields(item: &mut rmpv::Value) -> Result<(), String> {
    let rmpv::Value::Map(entries) = item else {
        return Ok(());
    };

    for (key, value) in entries {
        let Some(key) = rmpv_key_str(key) else {
            continue;
        };
        match key {
            "images" | "documents" => decode_native_media_array(value, key)?,
            "audio" | "video" | "document" => decode_native_media_object(value, key)?,
            _ => {}
        }
    }
    Ok(())
}

fn decode_native_media_array(value: &mut rmpv::Value, field: &str) -> Result<(), String> {
    let rmpv::Value::Array(items) = value else {
        return Ok(());
    };

    for (index, item) in items.iter_mut().enumerate() {
        decode_native_media_object_data(item, &format!("{field}[{index}].data"))?;
    }
    Ok(())
}

fn decode_native_media_object(value: &mut rmpv::Value, field: &str) -> Result<(), String> {
    decode_native_media_object_data(value, &format!("{field}.data"))
}

fn decode_native_media_object_data(value: &mut rmpv::Value, path: &str) -> Result<(), String> {
    let rmpv::Value::Map(entries) = value else {
        return Ok(());
    };

    let Some((_, data_value)) = entries
        .iter_mut()
        .find(|(key, _)| rmpv_key_str(key) == Some("data"))
    else {
        return Ok(());
    };

    let rmpv::Value::String(data) = data_value else {
        return Ok(());
    };
    let data = data
        .as_str()
        .ok_or_else(|| format!("{path} must be UTF-8 base64"))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| format!("invalid base64 in {path}: {e}"))?;
    *data_value = rmpv::Value::Binary(bytes);
    Ok(())
}

fn rmpv_key_str(value: &rmpv::Value) -> Option<&str> {
    match value {
        rmpv::Value::String(s) => s.as_str(),
        _ => None,
    }
}

/// One-shot generic conversion from `serde_json::Value` to `rmpv::Value`.
/// Used for the JSON request-body path: workers all speak msgpack, so
/// we normalize to `rmpv` once at ingress and avoid having two item
/// representations flowing through the rest of the publisher.
/// JSON has no binary or ext types, so this generic conversion is lossless and cheap
/// (no per-byte blow-up like `rmpv_to_json` suffered in the opposite
/// direction).
fn json_to_rmpv(value: serde_json::Value) -> rmpv::Value {
    match value {
        serde_json::Value::Null => rmpv::Value::Nil,
        serde_json::Value::Bool(b) => rmpv::Value::Boolean(b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                rmpv::Value::Integer(i.into())
            } else if let Some(u) = n.as_u64() {
                rmpv::Value::Integer(u.into())
            } else if let Some(f) = n.as_f64() {
                rmpv::Value::F64(f)
            } else {
                rmpv::Value::Nil
            }
        }
        serde_json::Value::String(s) => rmpv::Value::String(s.into()),
        serde_json::Value::Array(arr) => {
            rmpv::Value::Array(arr.into_iter().map(json_to_rmpv).collect())
        }
        serde_json::Value::Object(map) => rmpv::Value::Map(
            map.into_iter()
                .map(|(k, v)| (rmpv::Value::String(k.into()), json_to_rmpv(v)))
                .collect(),
        ),
    }
}

fn emit_audit_log(entry: AuditEntry) {
    info!(
        event = %entry.event,
        method = %entry.method,
        endpoint = %entry.endpoint,
        status = entry.status,
        token_id = %entry.token_id,
        model = %entry.model,
        pool = %entry.pool,
        gpu = %entry.gpu,
        worker = %entry.worker,
        latency_ms = entry.latency_ms,
        body_bytes = entry.body_bytes,
        "audit"
    );
}

/// Fuses the `msgpack → serde_json` conversion for response bodies
/// with inline `msgpack_numpy` sentinel decoding, so `bin` / `ext`
/// payloads skip the `Vec<serde_json::Value::Number>` (one number per
/// byte) detour that a generic rmpv-to-json conversion would produce.
///
/// Used on the JSON-response hot path. The `decode_dtype_values` /
/// `reshape_array` helpers below do the actual numeric decoding.
fn rmpv_to_response_json(value: rmpv::Value) -> serde_json::Value {
    match value {
        rmpv::Value::Map(entries) => {
            if let Some(decoded) = try_decode_rmpv_numpy(&entries) {
                return decoded;
            }
            let mut map = serde_json::Map::with_capacity(entries.len());
            for (k, v) in entries {
                let key = match k {
                    rmpv::Value::String(s) => s.into_str().unwrap_or_default().to_string(),
                    rmpv::Value::Binary(b) => String::from_utf8(b).unwrap_or_default(),
                    other => format!("{}", other),
                };
                map.insert(key, rmpv_to_response_json(v));
            }
            serde_json::Value::Object(map)
        }
        rmpv::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(rmpv_to_response_json).collect())
        }
        other => rmpv_to_json(other),
    }
}

/// Inspect a map that might be a `msgpack_numpy` sentinel and, if so,
/// decode the packed bytes straight into a nested JSON array without
/// ever materializing a `Vec<Number>` byte-by-byte.
///
/// Returns `None` if the map lacks any required sentinel key — the
/// caller then walks the map generically.
fn try_decode_rmpv_numpy(entries: &[(rmpv::Value, rmpv::Value)]) -> Option<serde_json::Value> {
    let mut is_nd = false;
    let mut dtype: Option<&str> = None;
    let mut data: Option<&[u8]> = None;
    let mut shape_src: Option<&[rmpv::Value]> = None;

    for (k, v) in entries {
        let key = match k {
            rmpv::Value::String(s) => s.as_str(),
            rmpv::Value::Binary(b) => std::str::from_utf8(b).ok(),
            _ => None,
        };
        let Some(key) = key else { continue };
        match key {
            "nd" => {
                if let rmpv::Value::Boolean(b) = v {
                    is_nd = *b;
                }
            }
            "type" => {
                if let rmpv::Value::String(s) = v {
                    dtype = s.as_str();
                }
            }
            "data" => {
                data = match v {
                    rmpv::Value::Binary(b) => Some(b.as_slice()),
                    // Some msgpack_numpy variants pack the buffer as
                    // an ext-type (code 0x15/0x17 etc.); the payload
                    // bytes are still the raw dtype-packed buffer.
                    rmpv::Value::Ext(_, b) => Some(b.as_slice()),
                    _ => None,
                };
            }
            "shape" => {
                if let rmpv::Value::Array(a) = v {
                    shape_src = Some(a.as_slice());
                }
            }
            _ => {}
        }
    }

    if !is_nd {
        return None;
    }
    let dtype = dtype?;
    let data = data?;
    let shape: Vec<usize> = shape_src
        .map(|arr| {
            arr.iter()
                .filter_map(|v| match v {
                    rmpv::Value::Integer(i) => i.as_u64().map(|n| n as usize),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default();

    let flat_values = decode_dtype_values(dtype, data)?;
    let expected_len = if shape.is_empty() {
        flat_values.len()
    } else {
        shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?
    };
    if expected_len != flat_values.len() {
        warn!(
            dtype = %dtype,
            expected_len,
            actual_len = flat_values.len(),
            shape = ?shape,
            "numpy sentinel shape does not match decoded data length"
        );
        return None;
    }
    Some(reshape_array(&flat_values, &shape))
}

/// Convert an rmpv::Value to serde_json::Value, handling binary data
/// by converting it to a JSON array of byte values.
fn rmpv_to_json(value: rmpv::Value) -> serde_json::Value {
    match value {
        rmpv::Value::Nil => serde_json::Value::Null,
        rmpv::Value::Boolean(b) => serde_json::Value::Bool(b),
        rmpv::Value::Integer(i) => {
            if let Some(n) = i.as_i64() {
                serde_json::Value::Number(n.into())
            } else if let Some(n) = i.as_u64() {
                serde_json::Value::Number(n.into())
            } else {
                serde_json::Value::Null
            }
        }
        rmpv::Value::F32(f) => serde_json::Number::from_f64(f as f64)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        rmpv::Value::F64(f) => serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        rmpv::Value::String(s) => {
            match s.into_str() {
                Some(s) => serde_json::Value::String(s.to_string()),
                None => serde_json::Value::Null, // Invalid UTF-8
            }
        }
        rmpv::Value::Binary(bytes) => {
            // Binary outside a numpy sentinel: keep the legacy
            // "array of byte values" shape so non-numpy payloads
            // that happen to contain `bin` (rare — workers prefer
            // the numpy sentinel even for 1-D tensors) still
            // serialize to something JSON can represent.
            serde_json::Value::Array(
                bytes
                    .into_iter()
                    .map(|b| serde_json::Value::from(b as u64))
                    .collect(),
            )
        }
        rmpv::Value::Array(arr) => {
            serde_json::Value::Array(arr.into_iter().map(rmpv_to_json).collect())
        }
        rmpv::Value::Map(entries) => {
            let mut map = serde_json::Map::new();
            for (k, v) in entries {
                // msgpack map keys can be binary strings from Python
                let key = match k {
                    rmpv::Value::String(s) => s.into_str().unwrap_or_default().to_string(),
                    rmpv::Value::Binary(b) => String::from_utf8(b).unwrap_or_default(),
                    other => format!("{}", other),
                };
                map.insert(key, rmpv_to_json(v));
            }
            serde_json::Value::Object(map)
        }
        rmpv::Value::Ext(_, data) => {
            // Extension types: convert data to byte array like Binary
            serde_json::Value::Array(
                data.into_iter()
                    .map(|b| serde_json::Value::from(b as u64))
                    .collect(),
            )
        }
    }
}

/// Decode raw bytes into a flat array of JSON values based on numpy dtype.
fn decode_dtype_values(dtype: &str, data: &[u8]) -> Option<Vec<serde_json::Value>> {
    match dtype {
        "<f4" => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|chunk| {
                        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        serde_json::Value::from(val as f64)
                    })
                    .collect(),
            )
        }
        "<f8" => {
            if !data.len().is_multiple_of(8) {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|chunk| {
                        let val = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        serde_json::Value::from(val)
                    })
                    .collect(),
            )
        }
        "<f2" => {
            if !data.len().is_multiple_of(2) {
                return None;
            }
            Some(
                data.chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        let val = f16_to_f32(bits);
                        serde_json::Value::from(val as f64)
                    })
                    .collect(),
            )
        }
        "<i4" => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|chunk| {
                        let val = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        serde_json::Value::from(val as i64)
                    })
                    .collect(),
            )
        }
        "<i8" => {
            if !data.len().is_multiple_of(8) {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|chunk| {
                        let val = i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        serde_json::Value::from(val)
                    })
                    .collect(),
            )
        }
        "<i2" => {
            if !data.len().is_multiple_of(2) {
                return None;
            }
            Some(
                data.chunks_exact(2)
                    .map(|chunk| {
                        let val = i16::from_le_bytes([chunk[0], chunk[1]]);
                        serde_json::Value::from(val as i64)
                    })
                    .collect(),
            )
        }
        "<u4" => {
            if !data.len().is_multiple_of(4) {
                return None;
            }
            Some(
                data.chunks_exact(4)
                    .map(|chunk| {
                        let val = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                        serde_json::Value::from(val as u64)
                    })
                    .collect(),
            )
        }
        "<u8" => {
            if !data.len().is_multiple_of(8) {
                return None;
            }
            Some(
                data.chunks_exact(8)
                    .map(|chunk| {
                        let val = u64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        serde_json::Value::from(val)
                    })
                    .collect(),
            )
        }
        "|b1" => Some(
            data.iter()
                .map(|&b| serde_json::Value::Bool(b != 0))
                .collect(),
        ),
        "|u1" => Some(
            data.iter()
                .map(|&b| serde_json::Value::from(b as u64))
                .collect(),
        ),
        "|i1" => Some(
            data.iter()
                .map(|&b| serde_json::Value::from(b as i8 as i64))
                .collect(),
        ),
        _ => {
            warn!(dtype = %dtype, "unsupported numpy dtype in msgpack_numpy conversion");
            None
        }
    }
}

/// Convert IEEE 754 half-precision (f16) bits to f32.
fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half >> 15) & 1) as u32;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal f16 → normalized f32
        let mut m = mantissa;
        let mut e: i32 = -14;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3ff;
        let f32_exp = ((e + 127) as u32) & 0xff;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }

    if exponent == 31 {
        let f32_mantissa = mantissa << 13;
        return f32::from_bits((sign << 31) | (0xff << 23) | f32_mantissa);
    }

    let f32_exp = (exponent as i32 - 15 + 127) as u32;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
}

/// Reshape a flat array of values into nested JSON arrays according to the given shape.
fn reshape_array(flat: &[serde_json::Value], shape: &[usize]) -> serde_json::Value {
    if shape.is_empty() || shape.len() == 1 {
        return serde_json::Value::Array(flat.to_vec());
    }
    reshape_recursive(flat, shape, 0).0
}

fn reshape_recursive(
    flat: &[serde_json::Value],
    shape: &[usize],
    dim: usize,
) -> (serde_json::Value, usize) {
    if dim == shape.len() - 1 {
        let n = shape[dim].min(flat.len());
        let arr: Vec<serde_json::Value> = flat[..n].to_vec();
        return (serde_json::Value::Array(arr), n);
    }

    // Defense-in-depth: even though `try_decode_rmpv_numpy` validates
    // `product(shape) == flat.len()` before calling in, cap the
    // pre-allocation at the remaining flat length so a malformed
    // `shape[dim]` (reached via any future caller that skips that check)
    // can't trigger a multi-GB allocation. The loop below already breaks
    // once `flat` is exhausted, so capping never under-allocates a valid
    // reshape.
    let mut result = Vec::with_capacity(shape[dim].min(flat.len()));
    let mut offset = 0;
    for _ in 0..shape[dim] {
        if offset >= flat.len() {
            break;
        }
        let (sub_arr, consumed) = reshape_recursive(&flat[offset..], shape, dim + 1);
        result.push(sub_arr);
        offset += consumed;
    }
    (serde_json::Value::Array(result), offset)
}

#[derive(Debug, PartialEq, Eq)]
struct OpenAiEmbeddingInput {
    texts: Vec<String>,
    token_count: u64,
}

fn estimate_embedding_tokens(texts: &[String]) -> u64 {
    let total: usize = texts.iter().map(|s| s.len()).sum();
    u64::max(1, (total / 4) as u64)
}

/// `usage.sie_token_source` — whether the emitted `/v1/embeddings` token count
/// is the worker's exact post-tokenization number or the request-time
/// character approximation. Reported because the two cannot be told apart from
/// the numbers alone, and only one of them reconciles with a token bill.
const USAGE_SOURCE_WORKER: &str = "worker";
const USAGE_SOURCE_ESTIMATE: &str = "character_estimate";

/// The authoritative token count the inner `/v1/encode` reply reported, if any.
///
/// Both encode ingresses now carry it: the direct one puts the worker's
/// `input_token_counts` on `EncodeResponse.usage`, and the queue one aggregates
/// the same numbers out of the workers' `UnitCounts`
/// ([`aggregate_result_usage`]). `None` means neither could count, never that
/// the count was zero — a measured `0` (a request that read no text) arrives
/// here as `Some(0)` and is reported as `0`.
fn encode_usage_input_tokens(encode_response: &Value) -> Option<u64> {
    encode_response.get("usage")?.get("input_tokens")?.as_u64()
}

/// The `usage` block `/v1/embeddings` reports, from the encode reply the
/// request actually produced.
///
/// The worker's exact count wins. When the encode reply carried none, the
/// request-time character estimate is emitted rather than dropping `usage`
/// altogether — a successful embedding must not turn into an error over a
/// reporting field, and the `openai` client cannot parse a response without
/// `usage` at all — but it is LABELLED `character_estimate`, so a caller
/// reconciling `usage` against a token bill can see which of the two numbers
/// they hold instead of silently trusting `chars / 4`.
fn embeddings_usage(encode_response: &Value, estimated_tokens: u64) -> Value {
    let (prompt_tokens, token_source) = match encode_usage_input_tokens(encode_response) {
        Some(tokens) => (tokens, USAGE_SOURCE_WORKER),
        None => (estimated_tokens, USAGE_SOURCE_ESTIMATE),
    };
    json!({
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens,
        "sie_token_source": token_source,
    })
}

/// The native `/v1/encode/{model}` body ONE `/v1/embeddings` request rewrites
/// itself into before the in-process re-dispatch.
///
/// `/v1/embeddings` is not dispatched directly: the handler normalizes its
/// `input` into native encode items and re-issues the request against
/// `/v1/encode/{model}`, so the reservation the live path takes is the
/// reservation for THIS body, not for the compat one. The managed
/// cost-estimate dry run (#2435) must therefore price the same rewritten body —
/// which it does by calling this function, the one the handler itself calls.
/// A private second rewrite here is exactly how an estimate would come to price
/// a request shape that never reaches the planner.
pub fn openai_embeddings_encode_body(input: &Value) -> Result<(Value, u64), String> {
    let OpenAiEmbeddingInput { texts, token_count } = openai_embedding_input_to_texts(input)?;
    Ok((
        json!({
            "items": texts.iter().map(|t| json!({"text": t})).collect::<Vec<_>>(),
            "params": {"output_types": ["dense"]},
        }),
        token_count,
    ))
}

fn openai_embedding_input_to_texts(input: &Value) -> Result<OpenAiEmbeddingInput, String> {
    match input {
        Value::String(s) => {
            let texts = vec![s.clone()];
            Ok(OpenAiEmbeddingInput {
                token_count: estimate_embedding_tokens(&texts),
                texts,
            })
        }
        Value::Array(a) if a.is_empty() => Err("input array is empty".to_string()),
        Value::Array(a) if a.iter().all(|x| x.is_string()) => {
            if a.len() > MAX_EMBEDDING_INPUTS {
                return Err(format!(
                    "input must contain at most {MAX_EMBEDDING_INPUTS} texts; split the request into smaller batches"
                ));
            }
            let texts: Vec<String> = a
                .iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect();
            Ok(OpenAiEmbeddingInput {
                token_count: estimate_embedding_tokens(&texts),
                texts,
            })
        }
        Value::Array(a)
            if a.iter().all(|x| x.as_i64().is_some())
                || a.iter().all(|x| {
                    x.as_array()
                        .map(|arr| arr.iter().all(|inner| inner.as_i64().is_some()))
                        .unwrap_or(false)
                }) =>
        {
            Err(
                "token-array embeddings input is not supported by the gateway; use text input"
                    .to_string(),
            )
        }
        _ => Err("input must be a string or array of strings".to_string()),
    }
}

fn extract_dense_embedding_vector(item: &Value) -> Option<Vec<f64>> {
    let dense = item.get("dense")?;
    if let Some(arr) = dense.as_array() {
        return arr.iter().map(Value::as_f64).collect();
    }
    dense
        .get("values")
        .and_then(|v| v.as_array())
        .and_then(|vals| vals.iter().map(Value::as_f64).collect())
}

fn openai_embedding_value(vector: Vec<f64>, encoding_format: &str) -> Value {
    if encoding_format == "base64" {
        let mut bytes = Vec::with_capacity(vector.len() * std::mem::size_of::<f32>());
        for value in vector {
            bytes.extend_from_slice(&(value as f32).to_le_bytes());
        }
        Value::String(base64::engine::general_purpose::STANDARD.encode(bytes))
    } else {
        json!(vector)
    }
}

fn openai_embedding_items_to_data(
    items: &[Value],
    expected_len: usize,
    encoding_format: &str,
) -> Result<Vec<Value>, String> {
    if items.len() != expected_len {
        return Err(format!(
            "encode response item count mismatch: expected {}, got {}",
            expected_len,
            items.len()
        ));
    }

    let mut data = Vec::with_capacity(items.len());
    for (idx, item) in items.iter().enumerate() {
        let Some(vec) = extract_dense_embedding_vector(item) else {
            return Err(format!("item {idx} missing dense embedding"));
        };
        data.push(json!({
            "object": "embedding",
            "embedding": openai_embedding_value(vec, encoding_format),
            "index": idx,
        }));
    }
    Ok(data)
}

#[utoipa::path(
    post,
    path = "/v1/embeddings",
    tag = "inference",
    description = "OpenAI-compatible embeddings proxy. A 200 response contains one embedding per input; partial or truncated internal encode success is treated as a 500 INTERNAL_ERROR instead of returning a partial 200. Every error path returns the OpenAI `{error:{message,type,param,code}}` envelope (inner SIE-native encode failures are translated), so an `openai`-client error handler works unchanged.",
    request_body = crate::openapi::OpenAIEmbeddingRequest,
    params(
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    responses(
        (status = 200, description = "OpenAI-compatible embeddings response", body = crate::openapi::OpenAIEmbeddingsListResponse),
        (status = 400, description = "Invalid request", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 401, description = "Missing or invalid bearer token", body = crate::openapi::StandardApiError),
        (status = 404, description = "Model not found", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 409, description = "Bundle override conflicts with model routing", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 413, description = "Request body too large", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 500, description = "All batch items failed or gateway internal error", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 502, description = "MODEL_LOAD_FAILED", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 503, description = "Provisioning in progress, queue unavailable, GPU not configured, model loading, or capacity exhausted", body = crate::openapi::OpenAIErrorEnvelope),
        (status = 504, description = "Result channel closed", body = crate::openapi::OpenAIErrorEnvelope)
    )
)]
pub async fn proxy_openai_embeddings(State(state): State<Arc<AppState>>, req: Request) -> Response {
    check_sdk_version(req.headers());
    // `/v1/embeddings` input is text (a string or array of strings), not
    // multimodal payloads — so it gets the same text-appropriate cap as
    // the chat path (16 MiB) rather than the legacy 256 MiB, which left a
    // trivial OOM-under-concurrency vector open.
    let max_body = compat_request_body_limit("/v1/embeddings");
    let hdr = req.headers().clone();
    let (parts, body) = req.into_parts();
    let body_bytes = match to_bytes(body, max_body).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(embeddings_error(
                    err_code::PAYLOAD_TOO_LARGE,
                    None,
                    format!("request body: {}", e),
                )),
            )
                .into_response();
        }
    };
    let parsed: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(embeddings_error(
                    err_code::INVALID_REQUEST,
                    None,
                    format!("invalid JSON: {}", e),
                )),
            )
                .into_response();
        }
    };
    let model_str = match parsed.get("model").and_then(|v| v.as_str()) {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(embeddings_error(
                    err_code::INVALID_REQUEST,
                    Some("model"),
                    "field \"model\" is required",
                )),
            )
                .into_response();
        }
    };
    if !is_valid_compat_model_id(&model_str) {
        return (
            StatusCode::BAD_REQUEST,
            Json(embeddings_error(
                err_code::INVALID_REQUEST,
                Some("model"),
                "invalid model id for path",
            )),
        )
            .into_response();
    }
    let enc_fmt = parsed
        .get("encoding_format")
        .and_then(|v| v.as_str())
        .unwrap_or("float");
    if enc_fmt != "float" && enc_fmt != "base64" {
        return (
            StatusCode::BAD_REQUEST,
            Json(embeddings_error(
                err_code::INVALID_REQUEST,
                Some("encoding_format"),
                "encoding_format must be either 'float' or 'base64'",
            )),
        )
            .into_response();
    }
    let input = parsed.get("input").cloned().unwrap_or(Value::Null);
    let (encode_body, token_count) = match openai_embeddings_encode_body(&input) {
        Ok(normalized) => normalized,
        Err(msg) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(embeddings_error(
                    err_code::INVALID_REQUEST,
                    Some("input"),
                    msg,
                )),
            )
                .into_response();
        }
    };
    // One encode item per normalized input text — the count the response
    // assembler pads/validates its `data` array against.
    let input_count = encode_body
        .get("items")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let encode_bytes = match serde_json::to_vec(&encode_body) {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(embeddings_error(
                    err_code::INTERNAL_ERROR,
                    None,
                    format!("encode body: {}", e),
                )),
            )
                .into_response();
        }
    };
    let version = parts.version;
    let extensions = parts.extensions;
    let mut inner_headers = HeaderMap::new();
    for (name, val) in hdr.iter() {
        if is_openai_compat_inner_request_header(name.as_str()) {
            // `append`, not `insert`: W3C `tracestate` may arrive as
            // multiple header fields and `HeaderMap::iter()` yields each
            // separately — `insert` would drop all but the last, so a
            // multi-line `tracestate` would be truncated before forwarding.
            inner_headers.append(name.clone(), val.clone());
        }
    }
    inner_headers.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    inner_headers.insert(
        axum::http::header::ACCEPT,
        HeaderValue::from_static("application/json"),
    );
    let uri = match compatibility_model_uri("encode", &model_str) {
        Ok(u) => u,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(embeddings_error(
                    err_code::INVALID_REQUEST,
                    Some("model"),
                    "invalid model id for path",
                )),
            )
                .into_response();
        }
    };
    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .version(version);
    for (k, v) in inner_headers.iter() {
        builder = builder.header(k, v);
    }
    let mut inner_req = match builder.body(Body::from(encode_bytes)) {
        Ok(r) => r,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(embeddings_error(
                    err_code::INTERNAL_ERROR,
                    None,
                    "failed to build internal encode request",
                )),
            )
                .into_response();
        }
    };
    *inner_req.extensions_mut() = extensions;
    let resp = proxy_request(State(state.clone()), inner_req, "encode").await;
    let inner_status = resp.status();
    if inner_status != StatusCode::OK {
        // Genuine errors (4xx/5xx) from the SIE-native encode path arrive as
        // `{detail:{code,message}}` (or the 502/503 `{error:{...}}` shapes);
        // translate them into the OpenAI envelope so the embeddings surface
        // stays parseable by `openai`-client error handling.
        if inner_status.is_client_error() || inner_status.is_server_error() {
            return translate_inner_compat_error(resp).await;
        }
        return resp;
    }
    // From here the worker has already run and reported its units, so EVERY
    // failure below is the gateway's: a successful encode this handler could not
    // render into the client's shape, with the client receiving none of it.
    // EVERY such arm is marked [`GatewayOwnedFault`] — the two buffering ones
    // and the three translation ones after them — so a metered composition
    // releases the hold instead of debiting for an undelivered result, the same
    // treatment `audio.rs` gives all four of its equivalents. The buffering and
    // translation arms are one class: "the worker delivered, we could not", and
    // marking only the half nearest the read would bill the other half in full.
    let enc_headers = resp.headers().clone();
    // NOT `max_body`. That is the REQUEST cap, and this body is the encode
    // reply — an amplification of the request, not a copy of it. See
    // [`MAX_COMPAT_RESPONSE_BODY`].
    let rb = match buffer_compat_response_body(resp.into_body()).await {
        CompatResponseBody::Buffered(b) => b,
        CompatResponseBody::TooLarge { observed } => {
            // Terminal, and deliberately not 503: the identical request yields
            // the identical oversized reply, so a retryable surface would have
            // SDK retry loops hammering a permanent condition forever. 413 is
            // already this route's documented terminal size verdict and carries
            // no `Retry-After`.
            //
            // The status is terminal AND the fault is gateway-owned; those are
            // orthogonal. The status tells the caller not to retry unchanged;
            // the marker tells billing not to charge for zero embeddings.
            warn!(
                limit = MAX_COMPAT_RESPONSE_BODY,
                observed_bytes = observed,
                "encode response exceeds the compat response cap"
            );
            // Only ONE remedy is named, because only one works. An earlier
            // revision also offered `encoding_format="base64"`; that is inert
            // here — the field is never forwarded to `/v1/encode` (see
            // `openai_embeddings_encode_body`) and base64 is applied afterwards,
            // when the OUTER response is assembled from this very buffer. A
            // caller who followed it paid for a second identical encode and got
            // a byte-identical rejection (#2617).
            let detail = match observed {
                Some(bytes) => format!(
                    "the embedding response for this request is {bytes} bytes, over the \
                     {MAX_COMPAT_RESPONSE_BODY}-byte limit; send fewer inputs per request"
                ),
                None => format!(
                    "the embedding response for this request exceeds the \
                     {MAX_COMPAT_RESPONSE_BODY}-byte limit; send fewer inputs per request"
                ),
            };
            return compat_translation_fault(
                "read_encode_response_too_large",
                StatusCode::PAYLOAD_TOO_LARGE,
                err_code::PAYLOAD_TOO_LARGE,
                Some("input"),
                detail,
            );
        }
        CompatResponseBody::Unreadable => {
            // Pre-existing 500, now marked. It billed in full before this change
            // for exactly the same reason the 413 would have.
            return compat_translation_fault(
                "read_encode_response",
                StatusCode::INTERNAL_SERVER_ERROR,
                err_code::INTERNAL_ERROR,
                None,
                "failed to read encode response body".to_string(),
            );
        }
    };
    let enc: Value = match serde_json::from_slice(&rb) {
        Ok(v) => v,
        Err(e) => {
            return compat_translation_fault(
                "decode_encode_response",
                StatusCode::INTERNAL_SERVER_ERROR,
                err_code::INTERNAL_ERROR,
                None,
                format!("encode JSON: {}", e),
            );
        }
    };
    let Some(items) = enc.get("items").and_then(|i| i.as_array()) else {
        return compat_translation_fault(
            "encode_response_shape",
            StatusCode::INTERNAL_SERVER_ERROR,
            err_code::INTERNAL_ERROR,
            None,
            "encode response missing items".to_string(),
        );
    };
    let data = match openai_embedding_items_to_data(items, input_count, enc_fmt) {
        Ok(data) => data,
        Err(msg) => {
            return compat_translation_fault(
                "format_embeddings",
                StatusCode::INTERNAL_SERVER_ERROR,
                err_code::INTERNAL_ERROR,
                None,
                msg,
            );
        }
    };
    let usage = embeddings_usage(&enc, token_count);
    let out = json!({
        "object": "list",
        "data": data,
        "model": model_str,
        "usage": usage,
    });
    let mut out_resp = (StatusCode::OK, Json(out)).into_response();
    for (k, v) in enc_headers.iter() {
        if is_openai_compat_forwarded_header(k.as_str()) {
            out_resp.headers_mut().insert(k.clone(), v.clone());
        }
    }
    out_resp
}

const MAX_RERANK_DOCUMENTS: usize = 1_000;

#[derive(Clone, Copy, Debug, PartialEq)]
enum RerankCompatibilityVersion {
    V1,
    V2,
}

#[derive(Debug, PartialEq)]
struct NormalizedRerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    top_n: Option<usize>,
    return_documents: bool,
    options: Map<String, Value>,
}

fn normalize_rerank_options(value: Option<&Value>) -> Result<Map<String, Value>, String> {
    let options = match value {
        None | Some(Value::Null) => return Ok(Map::new()),
        Some(Value::Object(value)) => value,
        Some(_) => return Err("'options' must be an object".to_string()),
    };
    const ALLOWED_OPTIONS: &[&str] = &["profile", "max_seq_length"];
    if let Some(unknown) = options
        .keys()
        .find(|key| !ALLOWED_OPTIONS.contains(&key.as_str()))
    {
        return Err(format!("options field '{unknown}' is not supported"));
    }
    if let Some(profile) = options.get("profile").filter(|value| !value.is_null()) {
        profile
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| "options.profile must be a non-blank string or null".to_string())?;
    }
    if let Some(max_seq_length) = options
        .get("max_seq_length")
        .filter(|value| !value.is_null())
    {
        max_seq_length
            .as_u64()
            .filter(|value| *value > 0)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                "options.max_seq_length must be a positive platform-sized integer or null"
                    .to_string()
            })?;
    }
    Ok(options.clone())
}

fn normalize_rerank_request(
    value: &Value,
    version: RerankCompatibilityVersion,
) -> Result<NormalizedRerankRequest, String> {
    let object = value
        .as_object()
        .ok_or_else(|| "request body must be a JSON object".to_string())?;
    const V1_ALLOWED_FIELDS: &[&str] = &[
        "model",
        "query",
        "documents",
        "top_n",
        "return_documents",
        "options",
    ];
    const V2_ALLOWED_FIELDS: &[&str] = &[
        "model",
        "query",
        "documents",
        "top_n",
        "max_tokens_per_doc",
        "priority",
        "options",
    ];
    let allowed_fields = match version {
        RerankCompatibilityVersion::V1 => V1_ALLOWED_FIELDS,
        RerankCompatibilityVersion::V2 => V2_ALLOWED_FIELDS,
    };
    if let Some(unknown) = object
        .keys()
        .find(|key| !allowed_fields.contains(&key.as_str()))
    {
        return Err(format!("field '{unknown}' is not supported"));
    }
    if version == RerankCompatibilityVersion::V2 {
        for field in ["max_tokens_per_doc", "priority"] {
            if object.get(field).is_some_and(|value| !value.is_null()) {
                return Err(format!(
                    "field '{field}' is not supported; omit it or send null"
                ));
            }
        }
    }

    let model = object
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "'model' must be a non-blank string".to_string())?
        .to_string();
    let query = object
        .get("query")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "'query' must be a non-blank string".to_string())?
        .to_string();
    let document_values = object
        .get("documents")
        .and_then(Value::as_array)
        .filter(|values| !values.is_empty())
        .ok_or_else(|| "'documents' must be a non-empty array of strings".to_string())?;
    if document_values.len() > MAX_RERANK_DOCUMENTS {
        return Err(format!(
            "'documents' exceeds the maximum of {MAX_RERANK_DOCUMENTS} per request"
        ));
    }
    let documents = document_values
        .iter()
        .map(|document| {
            document
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
                .ok_or_else(|| "'documents' must contain only non-blank strings".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;

    let top_n = match object.get("top_n") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let value = value
                .as_u64()
                .filter(|value| *value > 0)
                .ok_or_else(|| "'top_n' must be a positive integer".to_string())?;
            Some(
                usize::try_from(value)
                    .map_err(|_| "'top_n' must fit in the platform integer range".to_string())?,
            )
        }
    };
    let return_documents = match (version, object.get("return_documents")) {
        (RerankCompatibilityVersion::V1, None | Some(Value::Null)) => false,
        (RerankCompatibilityVersion::V1, Some(Value::Bool(value))) => *value,
        (RerankCompatibilityVersion::V1, Some(_)) => {
            return Err("'return_documents' must be a boolean".to_string())
        }
        (RerankCompatibilityVersion::V2, _) => false,
    };
    let options = normalize_rerank_options(object.get("options"))?;

    Ok(NormalizedRerankRequest {
        model,
        query,
        documents,
        top_n,
        return_documents,
        options,
    })
}

fn compatibility_model_uri(endpoint: &str, model: &str) -> Result<axum::http::Uri, String> {
    let model = model.trim_start_matches('/');
    if model.is_empty() {
        return Err("'model' must contain a model id".to_string());
    }
    let encoded_model = utf8_percent_encode(model, NON_ALPHANUMERIC);
    format!("/v1/{endpoint}/{encoded_model}")
        .parse()
        .map_err(|_| "invalid model id for path".to_string())
}

fn rerank_response_from_score(
    native: &Value,
    documents: &[String],
    top_n: Option<usize>,
    return_documents: bool,
) -> Result<Value, String> {
    let model = native
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| "score response missing model".to_string())?;
    let scores = native
        .get("scores")
        .and_then(Value::as_array)
        .ok_or_else(|| "score response missing scores".to_string())?;
    if scores.len() != documents.len() {
        return Err(format!(
            "score response item count mismatch: expected {}, got {}",
            documents.len(),
            scores.len()
        ));
    }

    let mut seen = vec![false; documents.len()];
    let mut ranked = Vec::with_capacity(scores.len());
    for entry in scores {
        let item_id = entry
            .get("item_id")
            .and_then(Value::as_str)
            .ok_or_else(|| "score response entry missing item_id".to_string())?;
        let index = item_id
            .parse::<usize>()
            .map_err(|_| "score response item_id is not a document index".to_string())?;
        if index >= documents.len() {
            return Err("score response item_id is out of range".to_string());
        }
        if std::mem::replace(&mut seen[index], true) {
            return Err("score response contains a duplicate item_id".to_string());
        }
        let score = entry
            .get("score")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .ok_or_else(|| "score response entry has an invalid score".to_string())?;
        ranked.push((index, score));
    }
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    if let Some(limit) = top_n {
        ranked.truncate(limit);
    }

    let usage = native
        .get("usage")
        .and_then(Value::as_object)
        .ok_or_else(|| "score response missing authoritative usage".to_string())?;
    let input_tokens = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| "score response usage missing input_tokens".to_string())?;
    let images = usage.get("images").map(|value| {
        value
            .as_u64()
            .ok_or_else(|| "score response usage has an invalid images count".to_string())
    });
    let mut output_usage = Map::new();
    output_usage.insert("input_tokens".to_string(), json!(input_tokens));
    if let Some(images) = images {
        output_usage.insert("images".to_string(), json!(images?));
    }

    let results = ranked
        .into_iter()
        .map(|(index, score)| {
            let mut result = Map::new();
            result.insert("index".to_string(), json!(index));
            result.insert("relevance_score".to_string(), json!(score));
            if return_documents {
                result.insert("document".to_string(), json!({"text": documents[index]}));
            }
            Value::Object(result)
        })
        .collect::<Vec<_>>();

    Ok(json!({
        "model": model,
        "results": results,
        "usage": output_usage,
    }))
}

fn rerank_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({"message": message.into()}))).into_response()
}

/// A `/v1/rerank` 500 for a POST-DISPATCH failure, marked as gateway-owned.
///
/// The `score` counterpart of [`compat_translation_fault`] and `audio.rs`'s
/// `translation_fault`. Once `proxy_request` has returned 200 the worker has run
/// and the meter already holds its units, so a failure to read, decode, or
/// reshape that reply is OURS: the client receives zero rerank results and,
/// without this marker, the managed edge falls through to
/// `settle_planned_request` and debits the hold in full.
///
/// Deliberately NOT folded into [`rerank_error`], which also serves
/// `translate_inner_rerank_error` — that path carries the WORKER's own verdict on
/// a request that never produced billable output through this handler, and
/// marking it would change a settlement this change has no business touching.
fn rerank_translation_fault(stage: &'static str, message: impl Into<String>) -> Response {
    GatewayOwnedFault::new(stage).mark(rerank_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        message.into(),
    ))
}

async fn translate_inner_rerank_error(resp: Response) -> Response {
    // An error envelope, like `translate_inner_compat_error` — see the note
    // there for why this stays at the request-shaped bound.
    const MAX: usize = 16 * 1024 * 1024;
    let status = resp.status();
    let headers = resp.headers().clone();
    let parsed: Value = match to_bytes(resp.into_body(), MAX).await {
        Ok(body) => serde_json::from_slice(&body).unwrap_or(Value::Null),
        Err(_) => Value::Null,
    };
    let message = parsed
        .get("detail")
        .and_then(|detail| detail.get("message"))
        .or_else(|| parsed.get("error").and_then(|error| error.get("message")))
        .and_then(Value::as_str)
        .unwrap_or("internal server error");
    let mut output = rerank_error(status, message);
    for (name, value) in headers.iter() {
        if is_openai_compat_forwarded_header(name.as_str())
            || name.as_str().eq_ignore_ascii_case("retry-after")
            || name.as_str().eq_ignore_ascii_case("x-sie-error-code")
        {
            output.headers_mut().insert(name.clone(), value.clone());
        }
    }
    output
}

#[utoipa::path(
    post,
    path = "/v1/rerank",
    tag = "inference",
    description = "Cohere v1-compatible text-only subset over SIE's native score primitive. Supported request fields are model, query, string documents, top_n, return_documents, and the documented SIE options extension. Unknown or unsupported fields reject with 400. The adapter rejects partial native results and forwards only authoritative worker-emitted usage.",
    request_body = crate::openapi::RerankRequest,
    params(
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    responses(
        (status = 200, description = "Cohere v1-compatible subset response", body = crate::openapi::RerankResponse),
        (status = 400, description = "Invalid request", body = crate::openapi::RerankError),
        (status = 401, description = "Missing or invalid bearer token", body = crate::openapi::RerankError),
        (status = 404, description = "Model not found", body = crate::openapi::RerankError),
        (status = 409, description = "Bundle override conflicts with model routing", body = crate::openapi::RerankError),
        (status = 413, description = "Request body too large", body = crate::openapi::RerankError),
        (status = 500, description = "Malformed or partial native score result", body = crate::openapi::RerankError),
        (status = 502, description = "MODEL_LOAD_FAILED", body = crate::openapi::RerankError),
        (status = 503, description = "Provisioning, model loading, or capacity exhausted", body = crate::openapi::RerankError),
        (status = 504, description = "Result channel closed", body = crate::openapi::RerankError)
    )
)]
pub async fn proxy_rerank(State(state): State<Arc<AppState>>, req: Request) -> Response {
    proxy_rerank_inner(RerankCompatibilityVersion::V1, state, req).await
}

#[utoipa::path(
    post,
    path = "/v2/rerank",
    tag = "inference",
    description = "Cohere v2-compatible text-only subset over SIE's native score primitive. Supported request fields are model, query, string documents, top_n, and the documented SIE options extension. max_tokens_per_doc and priority must be omitted or null; unknown or unsupported fields reject with 400. The adapter rejects partial native results and forwards only authoritative worker-emitted usage.",
    request_body = crate::openapi::RerankV2Request,
    params(
        ("X-SIE-MACHINE-PROFILE" = Option<String>, Header, description = "Preferred GPU or machine profile"),
        ("X-SIE-Pool" = Option<String>, Header, description = "Explicit pool routing override"),
        ("X-SIE-SDK-Version" = Option<String>, Header, description = "Client SDK version for skew warnings")
    ),
    responses(
        (status = 200, description = "Cohere v2-compatible subset response", body = crate::openapi::RerankResponse),
        (status = 400, description = "Invalid or unsupported request", body = crate::openapi::RerankError),
        (status = 401, description = "Missing or invalid bearer token", body = crate::openapi::RerankError),
        (status = 404, description = "Model not found", body = crate::openapi::RerankError),
        (status = 409, description = "Bundle override conflicts with model routing", body = crate::openapi::RerankError),
        (status = 413, description = "Request body too large", body = crate::openapi::RerankError),
        (status = 500, description = "Malformed or partial native score result", body = crate::openapi::RerankError),
        (status = 502, description = "MODEL_LOAD_FAILED", body = crate::openapi::RerankError),
        (status = 503, description = "Provisioning, model loading, or capacity exhausted", body = crate::openapi::RerankError),
        (status = 504, description = "Result channel closed", body = crate::openapi::RerankError)
    )
)]
pub async fn proxy_rerank_v2(State(state): State<Arc<AppState>>, req: Request) -> Response {
    proxy_rerank_inner(RerankCompatibilityVersion::V2, state, req).await
}

async fn proxy_rerank_inner(
    version: RerankCompatibilityVersion,
    state: Arc<AppState>,
    req: Request,
) -> Response {
    check_sdk_version(req.headers());
    // Bounds BOTH the rerank request and the inner `score` reply below, and
    // that symmetry is sound here where it was not for embeddings: a score
    // reply is `{index, relevance_score}` per document, capped at
    // [`MAX_RERANK_DOCUMENTS`] (1000) — tens of KiB, with no per-item vector to
    // amplify. The response is smaller than the request that produced it.
    const MAX: usize = 16 * 1024 * 1024;
    let headers = req.headers().clone();
    let (parts, body) = req.into_parts();
    let body = match to_bytes(body, MAX).await {
        Ok(body) => body,
        Err(error) => {
            return rerank_error(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!("request body: {error}"),
            );
        }
    };
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(value) => value,
        Err(error) => {
            return rerank_error(StatusCode::BAD_REQUEST, format!("invalid JSON: {error}"));
        }
    };
    let normalized = match normalize_rerank_request(&parsed, version) {
        Ok(value) => value,
        Err(message) => return rerank_error(StatusCode::BAD_REQUEST, message),
    };

    let score_body = json!({
        "query": {"text": normalized.query},
        "items": normalized.documents.iter().enumerate().map(|(index, document)| {
            json!({"id": index.to_string(), "text": document})
        }).collect::<Vec<_>>(),
        "options": normalized.options,
    });
    let uri = match compatibility_model_uri("score", &normalized.model) {
        Ok(uri) => uri,
        Err(message) => return rerank_error(StatusCode::BAD_REQUEST, message),
    };
    let score_bytes = match serde_json::to_vec(&score_body) {
        Ok(body) => body,
        Err(error) => {
            return rerank_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("score body: {error}"),
            );
        }
    };

    let mut builder = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .version(parts.version);
    for (name, value) in headers.iter() {
        if is_openai_compat_inner_request_header(name.as_str()) {
            builder = builder.header(name, value);
        }
    }
    builder = builder
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .header(axum::http::header::ACCEPT, "application/json");
    let mut inner = match builder.body(Body::from(score_bytes)) {
        Ok(request) => request,
        Err(_) => {
            return rerank_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to build internal score request",
            );
        }
    };
    *inner.extensions_mut() = parts.extensions;
    let response = proxy_request(State(state), inner, "score").await;
    if response.status() != StatusCode::OK {
        return translate_inner_rerank_error(response).await;
    }

    // From here the worker has already run and reported its units, so every
    // failure below is the gateway's, exactly as on the embeddings and audio
    // surfaces: the client receives zero rerank results and must not be
    // debited for them. Each arm is marked [`GatewayOwnedFault`].
    let response_headers = response.headers().clone();
    let native: Value = match to_bytes(response.into_body(), MAX).await {
        Ok(body) => match serde_json::from_slice(&body) {
            Ok(value) => value,
            Err(error) => {
                return rerank_translation_fault(
                    "decode_score_response",
                    format!("score response is not valid JSON: {error}"),
                );
            }
        },
        Err(_) => {
            return rerank_translation_fault(
                "read_score_response",
                "failed to read score response body",
            );
        }
    };
    let output = match rerank_response_from_score(
        &native,
        &normalized.documents,
        normalized.top_n,
        normalized.return_documents,
    ) {
        Ok(output) => output,
        Err(message) => return rerank_translation_fault("format_rerank", message),
    };
    let mut output = (StatusCode::OK, Json(output)).into_response();
    for (name, value) in response_headers.iter() {
        if is_openai_compat_forwarded_header(name.as_str()) {
            output.headers_mut().insert(name.clone(), value.clone());
        }
    }
    output
}

#[utoipa::path(
    post,
    path = "/v1/moderations",
    tag = "inference",
    description = "OpenAI-compatible moderations endpoint. Not implemented: SIE serves \
                   operator-provided models only and has no moderation model or governance \
                   store yet (Tier 0). The route is registered so the surface is discoverable \
                   and returns an explicit 501 `not_implemented` rather than a 404 — never a \
                   silent 'not flagged', which would be an unsafe lie about content safety.",
    responses(
        (status = 501, description = "Moderations not implemented", body = crate::openapi::OpenAIErrorEnvelope),
    )
)]
pub async fn proxy_moderations() -> Response {
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(json_openai_error(
            "the /v1/moderations endpoint is not implemented; no moderation model is configured.",
            oai_type::SERVER_ERROR,
            None,
            "not_implemented",
        )),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use crate::queue::dispatch::{
        ChunkEnvelope, PendingGenerationSnapshot, PublishTarget, StreamOutcome, WorkParams,
        WorkResult,
    };
    use tokio::sync::{broadcast, oneshot, Notify};

    #[derive(Default)]
    struct AbandonmentProbe {
        claimed: AtomicBool,
        begin_calls: AtomicUsize,
        finish_calls: AtomicUsize,
        finished: Notify,
    }

    #[async_trait::async_trait]
    impl WorkDispatcher for AbandonmentProbe {
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
            unreachable!("abandonment guard test does not publish work")
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
                Arc<Notify>,
                DispatchDurability,
            ),
            String,
        > {
            unreachable!("abandonment guard test does not publish generation")
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
            unreachable!("abandonment guard test does not publish generation")
        }

        async fn publish_cancel(&self, _request_id: &str) {}

        fn begin_work_abandonment(&self, _request_id: &str) -> bool {
            self.begin_calls.fetch_add(1, Ordering::Relaxed);
            !self.claimed.swap(true, Ordering::AcqRel)
        }

        async fn finish_work_abandonment(&self, _request_id: &str) {
            self.finish_calls.fetch_add(1, Ordering::Relaxed);
            self.finished.notify_one();
        }

        async fn republish_to_pool(
            &self,
            _request_id: &str,
            _reason: &'static str,
        ) -> Result<bool, String> {
            unreachable!("abandonment guard test does not republish")
        }

        async fn republish_pending_result_to_pool(
            &self,
            _request_id: &str,
            _reason: &'static str,
        ) -> Result<bool, String> {
            unreachable!("abandonment guard test does not republish")
        }

        fn drop_pending_stream(&self, _request_id: &str) {}

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

    #[test]
    fn dispatcher_defaults_to_no_first_chunk_pool_republish() {
        let dispatcher = AbandonmentProbe::default();
        assert!(!dispatcher.supports_first_chunk_pool_republish());
    }

    #[test]
    fn enabled_first_chunk_capability_preserves_initial_republish_and_pool_timeout_decisions() {
        assert_eq!(
            first_chunk_deadline_action(true, true, false, false),
            FirstChunkDeadlineAction::RepublishToPool
        );
        assert_eq!(
            first_chunk_deadline_action(true, false, false, false),
            FirstChunkDeadlineAction::SurfaceTimeout
        );
    }

    #[test]
    fn enabled_first_chunk_capability_waits_for_overall_on_single_consumer_lane() {
        assert_eq!(
            first_chunk_deadline_action(true, true, false, true),
            FirstChunkDeadlineAction::WaitForOverall
        );
    }

    #[test]
    fn enabled_first_chunk_capability_surfaces_after_republish_was_already_attempted() {
        assert_eq!(
            first_chunk_deadline_action(true, true, true, false),
            FirstChunkDeadlineAction::SurfaceTimeout
        );
    }

    #[test]
    fn no_republish_capability_waits_for_overall_without_cancel_or_first_chunk_timeout() {
        for was_direct_dispatched in [false, true] {
            assert_eq!(
                first_chunk_deadline_action(false, was_direct_dispatched, false, false,),
                FirstChunkDeadlineAction::WaitForOverall
            );
        }
    }

    #[tokio::test]
    async fn pending_work_guard_drop_claims_and_finishes_once() {
        let probe = Arc::new(AbandonmentProbe::default());
        let publisher: Arc<dyn WorkDispatcher> = probe.clone();
        let finished = probe.finished.notified();
        drop(PendingWorkAbandonGuard::new(
            Arc::clone(&publisher),
            "req-1".to_string(),
        ));
        tokio::time::timeout(Duration::from_secs(1), finished)
            .await
            .expect("abandonment cleanup finished");

        drop(PendingWorkAbandonGuard::new(publisher, "req-1".to_string()));
        tokio::task::yield_now().await;
        assert_eq!(probe.begin_calls.load(Ordering::Relaxed), 2);
        assert_eq!(probe.finish_calls.load(Ordering::Relaxed), 1);
    }

    // ── parse_model_spec ───────────────────────────────────────────

    #[test]
    fn test_parse_model_spec_no_bundle() {
        let (bundle, model) = parse_model_spec("BAAI/bge-m3");
        assert_eq!(bundle, "");
        assert_eq!(model, "BAAI/bge-m3");
    }

    // ── resolve_path_model_id ──────────────────────────────────────

    /// Build a `canonicalize`-style closure from a single canonical name:
    /// any case-insensitive match returns the canonical spelling, else
    /// `None` (mirrors `ModelRegistry::resolve_canonical_model_name`).
    fn canon_of(canonical: &'static str) -> impl Fn(&str) -> Option<String> {
        move |id: &str| {
            if id.eq_ignore_ascii_case(canonical) {
                Some(canonical.to_string())
            } else {
                None
            }
        }
    }

    #[test]
    fn test_resolve_path_model_id_passthrough_when_registry_has_path_form() {
        // If the registry happens to store the id with `__` (unusual but
        // possible — e.g. a fake-id fixture in a test config), the resolver
        // must not rewrite it, otherwise it would shadow a real model.
        let registry = canon_of("weird__path__model");
        assert_eq!(
            resolve_path_model_id("weird__path__model", registry),
            "weird__path__model"
        );
    }

    #[test]
    fn test_resolve_path_model_id_rewrites_double_underscore_to_slash() {
        // Common case: SDK encodes `Qwen/Qwen3-4B-Instruct-2507` as
        // `Qwen__Qwen3-4B-Instruct-2507` in the URL path; resolver hands
        // back the registry-canonical slash form.
        let registry = canon_of("Qwen/Qwen3-4B-Instruct-2507");
        assert_eq!(
            resolve_path_model_id("Qwen__Qwen3-4B-Instruct-2507", registry),
            "Qwen/Qwen3-4B-Instruct-2507"
        );
    }

    #[test]
    fn test_canonicalize_with_aliases_passes_through_real_model() {
        // A real model id resolves directly; aliases are not consulted.
        let aliases = std::collections::HashMap::new();
        let out =
            canonicalize_with_aliases(&aliases, "Qwen/Qwen3.5-4B", canon_of("Qwen/Qwen3.5-4B"));
        assert_eq!(out.as_deref(), Some("Qwen/Qwen3.5-4B"));
    }

    #[test]
    fn test_canonicalize_with_aliases_resolves_alias_to_target() {
        // "code" → target, case-insensitive on the alias key; the target is
        // then canonicalised through the registry.
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("code".to_string(), "Qwen/Qwen3.5-4B".to_string());
        let canon = canon_of("Qwen/Qwen3.5-4B");
        assert_eq!(
            canonicalize_with_aliases(&aliases, "code", &canon).as_deref(),
            Some("Qwen/Qwen3.5-4B")
        );
        assert_eq!(
            canonicalize_with_aliases(&aliases, "CODE", &canon).as_deref(),
            Some("Qwen/Qwen3.5-4B")
        );
    }

    #[test]
    fn test_canonicalize_with_aliases_target_verbatim_when_registry_empty() {
        // Alias target not in the registry → return the target verbatim so the
        // 404 references the resolved model, not the alias.
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("code".to_string(), "Qwen/Qwen3.5-4B".to_string());
        assert_eq!(
            canonicalize_with_aliases(&aliases, "code", |_| None).as_deref(),
            Some("Qwen/Qwen3.5-4B")
        );
    }

    #[test]
    fn test_canonicalize_with_aliases_resolves_sie_safe_target() {
        // An alias target stored in the SIE-safe "__" form resolves to the
        // slash form via resolve_path_model_id, not just bare canonicalize.
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("code".to_string(), "Qwen__Qwen3.5-4B".to_string());
        assert_eq!(
            canonicalize_with_aliases(&aliases, "code", canon_of("Qwen/Qwen3.5-4B")).as_deref(),
            Some("Qwen/Qwen3.5-4B")
        );
    }

    #[test]
    fn test_canonicalize_with_aliases_unknown_falls_through() {
        // Neither a real model nor a configured alias → None (normal 404 path).
        let aliases = std::collections::HashMap::new();
        assert_eq!(canonicalize_with_aliases(&aliases, "nope", |_| None), None);
    }

    #[test]
    fn test_resolve_model_spec_real_model_no_bundle() {
        // A real model id resolves directly; aliases not consulted, no bundle.
        let aliases = std::collections::HashMap::new();
        let (bundle, model) = resolve_model_spec_with_aliases(
            &aliases,
            "Qwen/Qwen3.5-4B",
            canon_of("Qwen/Qwen3.5-4B"),
        );
        assert_eq!(bundle, "");
        assert_eq!(model, "Qwen/Qwen3.5-4B");
    }

    #[test]
    fn test_resolve_model_spec_caller_bundle_preserved() {
        // An explicit `bundle:/model` request keeps its bundle.
        let aliases = std::collections::HashMap::new();
        let (bundle, model) = resolve_model_spec_with_aliases(
            &aliases,
            "bf16:/Qwen/Qwen3.5-4B",
            canon_of("Qwen/Qwen3.5-4B"),
        );
        assert_eq!(bundle, "bf16");
        assert_eq!(model, "Qwen/Qwen3.5-4B");
    }

    #[test]
    fn test_resolve_model_spec_alias_carries_bundle() {
        // The headline feature: a `sql` alias whose target carries a bundle
        // routes to that (BF16) bundle while resolving the base model.
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("sql".to_string(), "bf16:/Qwen/Qwen3.6-27B".to_string());
        let (bundle, model) =
            resolve_model_spec_with_aliases(&aliases, "sql", canon_of("Qwen/Qwen3.6-27B"));
        assert_eq!(bundle, "bf16");
        assert_eq!(model, "Qwen/Qwen3.6-27B");
    }

    #[test]
    fn test_resolve_model_spec_alias_with_profile_suffix_still_routes() {
        // A profile-suffixed alias request (`sql:rtx-pro-6000`) must still hit
        // the `sql` alias and adopt its BF16 bundle. The suffix applies to
        // the alias target unless the target already names a concrete profile.
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("sql".to_string(), "bf16:/Qwen/Qwen3.6-27B".to_string());
        let (bundle, model) = resolve_model_spec_with_aliases(
            &aliases,
            "sql:rtx-pro-6000",
            canon_of("Qwen/Qwen3.6-27B:rtx-pro-6000"),
        );
        assert_eq!(bundle, "bf16");
        assert_eq!(model, "Qwen/Qwen3.6-27B:rtx-pro-6000");
    }

    #[test]
    fn test_resolve_model_spec_empty_alias_profile_suffix_does_not_fallback() {
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("sql".to_string(), "bf16:/Qwen/Qwen3.6-27B".to_string());
        let (bundle, model) =
            resolve_model_spec_with_aliases(&aliases, "sql:", canon_of("Qwen/Qwen3.6-27B"));
        assert_eq!(bundle, "");
        assert_eq!(model, "sql:");
    }

    #[test]
    fn test_resolve_model_spec_alias_target_profile_wins_over_request_suffix() {
        let mut aliases = std::collections::HashMap::new();
        aliases.insert(
            "sql".to_string(),
            "bf16:/Qwen/Qwen3.6-27B:bf16-sql".to_string(),
        );
        let (bundle, model) = resolve_model_spec_with_aliases(
            &aliases,
            "sql:rtx-pro-6000",
            canon_of("Qwen/Qwen3.6-27B:bf16-sql"),
        );
        assert_eq!(bundle, "bf16");
        assert_eq!(model, "Qwen/Qwen3.6-27B:bf16-sql");
    }

    #[test]
    fn test_resolve_model_spec_alias_bare_target_has_no_bundle() {
        // An alias with a plain (bundle-less) target resolves the model and
        // leaves the bundle empty — unchanged from the pre-feature behavior.
        let mut aliases = std::collections::HashMap::new();
        aliases.insert(
            "code".to_string(),
            "Qwen/Qwen3-4B-Instruct-2507".to_string(),
        );
        let (bundle, model) = resolve_model_spec_with_aliases(
            &aliases,
            "CODE",
            canon_of("Qwen/Qwen3-4B-Instruct-2507"),
        );
        assert_eq!(bundle, "");
        assert_eq!(model, "Qwen/Qwen3-4B-Instruct-2507");
    }

    #[test]
    fn test_resolve_model_spec_caller_bundle_wins_over_alias_bundle() {
        // An explicit caller bundle on an aliased request overrides the alias's
        // bundle (the model still resolves through the alias).
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("sql".to_string(), "bf16:/Qwen/Qwen3.6-27B".to_string());
        let (bundle, model) =
            resolve_model_spec_with_aliases(&aliases, "fp8:/sql", canon_of("Qwen/Qwen3.6-27B"));
        assert_eq!(bundle, "fp8");
        assert_eq!(model, "Qwen/Qwen3.6-27B");
    }

    #[test]
    fn test_resolve_model_spec_alias_bundle_with_sie_safe_target() {
        // The alias target's model portion may use the SIE-safe `__` form.
        let mut aliases = std::collections::HashMap::new();
        aliases.insert("sql".to_string(), "bf16:/Qwen__Qwen3.6-27B".to_string());
        let (bundle, model) =
            resolve_model_spec_with_aliases(&aliases, "sql", canon_of("Qwen/Qwen3.6-27B"));
        assert_eq!(bundle, "bf16");
        assert_eq!(model, "Qwen/Qwen3.6-27B");
    }

    #[test]
    fn test_resolve_model_spec_unknown_passthrough_no_bundle() {
        // Neither a real model nor an alias → verbatim model, no bundle (404 path).
        let aliases = std::collections::HashMap::new();
        let (bundle, model) = resolve_model_spec_with_aliases(&aliases, "nope", |_| None);
        assert_eq!(bundle, "");
        assert_eq!(model, "nope");
    }

    #[test]
    fn test_resolve_path_model_id_returns_original_when_unknown() {
        // Neither the path id nor the `/`-substituted variant is in the
        // registry — return the original so the 404 path references what
        // the caller actually sent rather than a guess.
        let registry = |_id: &str| None;
        assert_eq!(
            resolve_path_model_id("Org__missing-model", registry),
            "Org__missing-model"
        );
    }

    #[test]
    fn test_resolve_path_model_id_no_double_underscore_returns_original() {
        // Path ids without `__` skip the rewrite branch entirely — even if
        // the registry would happen to match something else.
        let registry = canon_of("my-model");
        assert_eq!(resolve_path_model_id("my-model", registry), "my-model");
    }

    #[test]
    fn test_resolve_path_model_id_folds_case_to_canonical() {
        // H1: case-variant ids must collapse to the single canonical
        // spelling so downstream labels / dispatch keys don't fan out.
        let registry = canon_of("Org/Model");
        assert_eq!(resolve_path_model_id("org/model", &registry), "Org/Model");
        assert_eq!(resolve_path_model_id("ORG/MODEL", &registry), "Org/Model");
        assert_eq!(resolve_path_model_id("Org/Model", &registry), "Org/Model");
        // `__`-encoded case variant also folds via the slash rewrite.
        assert_eq!(resolve_path_model_id("org__model", &registry), "Org/Model");
    }

    #[test]
    fn test_proxy_request_tracing_is_generation_only() {
        assert!(should_trace_proxy_request("generate"));
        for endpoint in crate::endpoint::InferenceEndpoint::NON_GENERATION_QUEUE_LABELS {
            assert!(
                !should_trace_proxy_request(endpoint),
                "{endpoint} must keep the pre-generation proxy hot path"
            );
        }
        assert!(
            !should_trace_proxy_request("unknown"),
            "unknown endpoints must fail closed to the non-generation hot path"
        );
    }

    /// Handler spans must be entered around each future poll, never by a guard
    /// whose lifetime crosses an await. A cross-await guard can be created on
    /// one Tokio worker and dropped on another, leaving the OpenTelemetry
    /// context guard in the first worker's TLS until thread teardown.
    #[test]
    fn async_proxy_spans_are_poll_scoped() {
        let source = include_str!("proxy.rs");
        let forbidden_enter = [".en", "ter()"].concat();

        for (handler_start, handler_end, span_name, guard_parts) in [
            (
                "pub(crate) async fn proxy_request(",
                "\nasync fn proxy_request_inner(",
                "proxy_span",
                ["_proxy_span_", "guard"],
            ),
            (
                "pub async fn proxy_chat(",
                "\nasync fn proxy_chat_inner(",
                "chat_span",
                ["_chat_span_", "guard"],
            ),
        ] {
            let start = source
                .find(handler_start)
                .unwrap_or_else(|| panic!("handler start not found: {handler_start}"));
            let end = source[start..]
                .find(handler_end)
                .map(|offset| start + offset)
                .unwrap_or_else(|| panic!("handler end not found: {handler_end}"));
            let handler = &source[start..end];
            let compact = handler.split_whitespace().collect::<Vec<_>>().join(" ");
            let expected_tail = format!("}} .instrument({span_name}) .await }}");
            let forbidden_guard = guard_parts.concat();

            assert!(
                compact.ends_with(&expected_tail),
                "{handler_start} must instrument the complete handler remainder with {span_name}"
            );
            assert!(
                !handler.contains(&forbidden_enter),
                "{handler_start} must not hold a span entry guard across awaits"
            );
            assert!(
                !handler.contains(&forbidden_guard),
                "{handler_start} retained the old cross-await span guard"
            );
        }
    }

    #[test]
    fn test_resolve_bundle_for_request_empty_registry_falls_back_to_caller() {
        // With an empty registry (no models), the resolver falls back to the
        // caller's bundle override, or "default" when none is given — the same
        // branch the inline `proxy_request` block took before #1543.
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        let registry = ModelRegistry::new(bundles_dir.path(), models_dir.path(), true);
        assert!(!registry.has_any_models());

        let default =
            resolve_bundle_for_request(&registry, "org/model", "org/model", "", None, "encode")
                .expect("empty registry + no override resolves");
        assert_eq!(default, "default");

        let overridden = resolve_bundle_for_request(
            &registry,
            "org/model",
            "org/model",
            "custom-bundle",
            None,
            "encode",
        )
        .expect("empty registry preserves caller bundle");
        assert_eq!(overridden, "custom-bundle");
    }

    #[test]
    fn test_model_revision_header_requires_matching_worker_execution_hash() {
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let execution_hash = "a".repeat(64);
        let matching: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-1",
            "success": true,
            "executed_bundle_config_hash": execution_hash.clone()
        }))
        .unwrap();
        let mut headers = HeaderMap::new();

        insert_model_revision_header(
            &mut headers,
            Some(revision),
            execution_hash.as_str(),
            &[&matching],
        );

        assert_eq!(
            headers
                .get("x-sie-model-revision")
                .and_then(|value| value.to_str().ok()),
            Some(execution_hash.as_str())
        );

        headers.clear();
        insert_model_revision_header(&mut headers, None, execution_hash.as_str(), &[&matching]);
        assert!(headers.get("x-sie-model-revision").is_none());

        let mismatching: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-2",
            "success": true,
            "executed_bundle_config_hash": "b".repeat(64)
        }))
        .unwrap();
        insert_model_revision_header(
            &mut headers,
            Some(revision),
            execution_hash.as_str(),
            &[&matching, &mismatching],
        );
        assert!(headers.get("x-sie-model-revision").is_none());

        insert_model_revision_header(&mut headers, Some(revision), &"b".repeat(64), &[&matching]);
        assert!(headers.get("x-sie-model-revision").is_none());

        let legacy: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-1",
            "success": true
        }))
        .unwrap();
        insert_model_revision_header(
            &mut headers,
            Some(revision),
            execution_hash.as_str(),
            &[&legacy],
        );
        assert!(headers.get("x-sie-model-revision").is_none());

        let mut stream_outcome = crate::queue::streaming::StreamOutcome {
            text: "ok".to_string(),
            finish_reason: "stop".to_string(),
            usage: None,
            attempt_id: "attempt-1".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: Some(execution_hash.clone()),
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        insert_stream_model_revision_header(
            &mut headers,
            Some(revision),
            execution_hash.as_str(),
            &stream_outcome,
        );
        assert_eq!(
            headers
                .get("x-sie-model-revision")
                .and_then(|value| value.to_str().ok()),
            Some(execution_hash.as_str())
        );

        headers.clear();
        stream_outcome.executed_bundle_config_hash = Some("b".repeat(64));
        insert_stream_model_revision_header(
            &mut headers,
            Some(revision),
            execution_hash.as_str(),
            &stream_outcome,
        );
        assert!(headers.get("x-sie-model-revision").is_none());

        stream_outcome.executed_bundle_config_hash = None;
        insert_stream_model_revision_header(
            &mut headers,
            Some(revision),
            execution_hash.as_str(),
            &stream_outcome,
        );
        assert!(headers.get("x-sie-model-revision").is_none());
    }

    #[test]
    fn test_catalog_execution_hash_is_fail_closed_for_every_known_model() {
        assert!(!catalog_execution_hash_is_ready("", true));
        assert!(catalog_execution_hash_is_ready(&"a".repeat(64), true));
        assert!(catalog_execution_hash_is_ready("", false));
    }

    #[test]
    fn test_execution_identity_header_requires_valid_unanimous_worker_proof() {
        let identity = "c".repeat(64);
        let matching: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-1",
            "success": true,
            "execution_identity_sha256": identity.clone()
        }))
        .unwrap();
        let same: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-2",
            "success": true,
            "execution_identity_sha256": identity.clone()
        }))
        .unwrap();
        let mismatching: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-3",
            "success": true,
            "execution_identity_sha256": "d".repeat(64)
        }))
        .unwrap();
        let missing: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-4",
            "success": true
        }))
        .unwrap();
        let malformed: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-5",
            "success": true,
            "execution_identity_sha256": "C".repeat(64)
        }))
        .unwrap();

        let mut headers = HeaderMap::new();
        insert_execution_identity_header(&mut headers, &[&matching, &same]);
        assert_eq!(
            headers
                .get("x-sie-execution-identity-sha256")
                .and_then(|value| value.to_str().ok()),
            Some(identity.as_str())
        );

        for results in [
            vec![&matching, &mismatching],
            vec![&matching, &missing],
            vec![&malformed],
            Vec::new(),
        ] {
            headers.clear();
            insert_execution_identity_header(&mut headers, &results);
            assert!(headers.get("x-sie-execution-identity-sha256").is_none());
        }

        let mut outcome = crate::queue::streaming::StreamOutcome {
            text: "ok".to_string(),
            finish_reason: "stop".to_string(),
            usage: None,
            attempt_id: "attempt-1".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: Some(identity.clone()),
            execution_binding_sha256: None,
        };
        insert_stream_execution_identity_header(&mut headers, &outcome);
        assert_eq!(
            headers
                .get("x-sie-execution-identity-sha256")
                .and_then(|value| value.to_str().ok()),
            Some(identity.as_str())
        );

        headers.clear();
        outcome.execution_identity_sha256 = Some("bad".to_string());
        insert_stream_execution_identity_header(&mut headers, &outcome);
        assert!(headers.get("x-sie-execution-identity-sha256").is_none());
    }

    #[test]
    fn test_execution_binding_header_requires_valid_unanimous_worker_proof() {
        let binding = "e".repeat(64);
        let matching: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-1",
            "success": true,
            "execution_binding_sha256": binding.clone()
        }))
        .unwrap();
        let same: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-2",
            "success": true,
            "execution_binding_sha256": binding.clone()
        }))
        .unwrap();
        let mismatching: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-3",
            "success": true,
            "execution_binding_sha256": "f".repeat(64)
        }))
        .unwrap();
        let missing: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-4",
            "success": true
        }))
        .unwrap();
        let malformed: publisher::WorkResult = serde_json::from_value(json!({
            "request_id": "request-5",
            "success": true,
            "execution_binding_sha256": "E".repeat(64)
        }))
        .unwrap();

        let mut headers = HeaderMap::new();
        insert_execution_binding_header(&mut headers, &[&matching, &same]);
        assert_eq!(
            headers
                .get("x-sie-execution-binding-sha256")
                .and_then(|value| value.to_str().ok()),
            Some(binding.as_str())
        );

        for results in [
            vec![&matching, &mismatching],
            vec![&matching, &missing],
            vec![&malformed],
            Vec::new(),
        ] {
            headers.clear();
            insert_execution_binding_header(&mut headers, &results);
            assert!(headers.get("x-sie-execution-binding-sha256").is_none());
        }

        let mut outcome = crate::queue::streaming::StreamOutcome {
            text: "ok".to_string(),
            finish_reason: "stop".to_string(),
            usage: None,
            attempt_id: "attempt-1".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: Some(binding.clone()),
        };
        insert_stream_execution_binding_header(&mut headers, &outcome);
        assert_eq!(
            headers
                .get("x-sie-execution-binding-sha256")
                .and_then(|value| value.to_str().ok()),
            Some(binding.as_str())
        );

        headers.clear();
        outcome.execution_binding_sha256 = Some("bad".to_string());
        insert_stream_execution_binding_header(&mut headers, &outcome);
        assert!(headers.get("x-sie-execution-binding-sha256").is_none());
    }

    #[tokio::test]
    async fn test_endpoint_error_response_envelope_matches_endpoint() {
        // generate/chat use the OpenAI `{error:{...}}` envelope; native endpoints
        // keep `{detail:{...}}`. The engine-pin errors now route through this
        // helper so a bad X-SIE-Engine on /v1/generate gets the OpenAI shape
        // rather than the native one. See #1567.
        let openai = endpoint_error_response(
            "generate",
            StatusCode::BAD_REQUEST,
            err_code::INVALID_REQUEST,
            oai_type::INVALID_REQUEST,
            oai_code::INVALID_REQUEST,
            None,
            "bad engine",
        );
        let native = endpoint_error_response(
            "encode",
            StatusCode::BAD_REQUEST,
            err_code::INVALID_REQUEST,
            oai_type::INVALID_REQUEST,
            oai_code::INVALID_REQUEST,
            None,
            "bad engine",
        );
        let openai_body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(openai.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let native_body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(native.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(
            openai_body.get("error").is_some(),
            "generate must use the OpenAI envelope"
        );
        assert!(
            native_body.get("detail").is_some(),
            "encode must use the native envelope"
        );
    }

    #[tokio::test]
    async fn dispatch_size_marker_maps_to_typed_payload_too_large() {
        use crate::queue::dispatch::DispatchPayloadTooLarge;

        let error = DispatchPayloadTooLarge::from("managed dispatch frame exceeds 67108864 bytes");
        let response = dispatch_rejection_response("extract", &error.into()).expect("known marker");
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let body: serde_json::Value = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(body["detail"]["code"], err_code::PAYLOAD_TOO_LARGE);
        assert_eq!(
            body["detail"]["message"],
            "managed dispatch frame exceeds 67108864 bytes"
        );
        assert!(
            dispatch_rejection_response(
                "extract",
                &DispatchError::Other("transport failed".to_string()),
            )
            .is_none(),
            "untyped transport failures must remain retryable service errors"
        );
    }

    #[tokio::test]
    async fn dispatch_invalid_input_and_backpressure_have_typed_responses() {
        use crate::queue::dispatch::{DispatchBackpressure, DispatchInvalidInput};

        let invalid = dispatch_rejection_response(
            "extract",
            &DispatchInvalidInput::from("malformed audio").into(),
        )
        .expect("typed invalid input");
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);

        let saturated = dispatch_rejection_response(
            "extract",
            &DispatchBackpressure::from("audio preflight busy").into(),
        )
        .expect("typed backpressure");
        assert_eq!(saturated.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            saturated
                .headers()
                .get("retry-after")
                .and_then(|value| value.to_str().ok()),
            Some(BACKPRESSURE_RETRY_AFTER),
        );
    }

    #[test]
    fn test_retry_after_defaults_pin_wire_values() {
        // These Retry-After wire strings are part of the capacity contract
        // clients/SDKs honor; pin them. See #1574.
        assert_eq!(RetryAfter::DEFAULT.provisioning, "60");
        assert_eq!(RetryAfter::DEFAULT.backpressure, "5");
        assert_eq!(RetryAfter::DEFAULT.gateway_timeout, "5");
        assert_eq!(RetryAfter::DEFAULT.model_loading, "5");
        assert_eq!(RetryAfter::DEFAULT.resource_exhausted, "5");
        assert_eq!(RetryAfter::DEFAULT.lora_loading, "5");
        // The named constants alias the typed home — pin each against its wire
        // literal (not against `RetryAfter::DEFAULT.*`, which would be a
        // tautological `x == x`), so a broken alias is actually caught.
        assert_eq!(PROVISIONING_RETRY_AFTER, "60");
        assert_eq!(BACKPRESSURE_RETRY_AFTER, "5");
        assert_eq!(GATEWAY_TIMEOUT_RETRY_AFTER, "5");
        assert_eq!(MODEL_LOADING_RETRY_AFTER, "5");
        assert_eq!(RESOURCE_EXHAUSTED_RETRY_AFTER, "5");
        assert_eq!(LORA_LOADING_RETRY_AFTER, "5");
    }

    fn admission_test_state(pool_manager: Arc<PoolManager>) -> AppState {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            bundles_dir.path().join("default.yaml"),
            "name: default\nadapters:\n  - module\ndefault: true\n",
        )
        .unwrap();
        let mut gpu_profile_map = std::collections::HashMap::new();
        gpu_profile_map.insert("l4".to_string(), "l4".to_string());
        let configured_physical_lanes = crate::state::demand_tracker::PhysicalLaneCatalog::try_new(
            [
                ("default", "l4", "default"),
                (
                    "test-zero-assigned-native",
                    "l4",
                    "test-zero-assigned-native-bundle",
                ),
                (
                    "test-zero-assigned-openai",
                    "l4",
                    "test-zero-assigned-openai-bundle",
                ),
                ("test-bp-demand-pool", "l4", "test-bp-demand-bundle"),
            ]
            .into_iter()
            .map(|(pool, profile, bundle)| {
                crate::state::demand_tracker::PhysicalLane::try_new(pool, profile, bundle).unwrap()
            }),
        )
        .unwrap();
        let config = crate::config::Config {
            host: "127.0.0.1".to_string(),
            port: 0,
            worker_urls: Vec::new(),
            use_kubernetes: false,
            k8s_namespace: "default".to_string(),
            k8s_service: String::new(),
            k8s_port: 0,
            health_mode: "http".to_string(),
            nats_url: String::new(),
            nats_config_trusted_producers: Vec::new(),
            auth_mode: "none".to_string(),
            auth_tokens: Vec::new(),
            admin_token: String::new(),
            auth_exempt_operational: false,
            log_level: "info".to_string(),
            json_logs: false,
            enable_pools: true,
            hot_reload: false,
            watch_polling: false,
            multi_router: false,
            request_timeout: 30.0,
            max_stream_pending: 1024,
            max_lane_in_flight_items:
                crate::queue::lane_admission::DEFAULT_MAX_LANE_IN_FLIGHT_ITEMS,
            lane_backpressure_enforce: false,
            stream_max_age_s: 300,
            stream_storage: crate::config::StreamStorage::Memory,
            stream_num_replicas: 1,
            configured_gpus: vec!["l4".to_string()],
            gpu_profile_map,
            configured_physical_lanes: configured_physical_lanes.clone(),
            static_queue_pools: Vec::new(),
            model_aliases: std::collections::HashMap::new(),
            published_model_aliases: Default::default(),
            bundles_dir: bundles_dir.path().to_string_lossy().to_string(),
            models_dir: models_dir.path().to_string_lossy().to_string(),
            config_service_url: None,
            config_service_token: None,
            config_modal_proxy_token: None,
            payload_store_url: String::new(),
            public_base_url: None,
        };
        AppState {
            registry: Arc::new(WorkerRegistry::new(Duration::from_secs(30), None)),
            config: Arc::new(config),
            model_registry: Arc::new(crate::state::model_registry::ModelRegistry::new(
                bundles_dir.path(),
                models_dir.path(),
                true,
            )),
            pool_manager,
            work_publisher: None,
            lane_backlog_source: None,
            demand_tracker: Arc::new(crate::state::demand_tracker::DemandTracker::new(
                configured_physical_lanes,
            )),
            config_epoch: crate::state::config_epoch::ConfigEpoch::new(),
            model_access_policy: None,
        }
    }

    struct FixedGenerationRoutePolicy {
        customer_model: &'static str,
        intent: GenerationRequestIntent,
        route: GovernedGenerationRoute,
    }

    struct PermissiveModelAccessPolicy {
        generation: Arc<dyn crate::server::GenerationRoutePolicy>,
    }

    impl crate::server::ModelAccessPolicy for PermissiveModelAccessPolicy {
        fn visible(&self, _resolved_model: &str, _ext: &axum::http::Extensions) -> bool {
            true
        }

        fn generation_route_policy(&self) -> Option<&dyn crate::server::GenerationRoutePolicy> {
            Some(self.generation.as_ref())
        }
    }

    fn install_generation_policy(
        state: &mut AppState,
        policy: Arc<dyn crate::server::GenerationRoutePolicy>,
    ) {
        state.model_access_policy =
            Some(Arc::new(PermissiveModelAccessPolicy { generation: policy }));
    }

    impl crate::server::GenerationRoutePolicy for FixedGenerationRoutePolicy {
        fn resolve(
            &self,
            customer_model: &str,
            intent: GenerationRequestIntent,
        ) -> Option<GovernedGenerationRoute> {
            (customer_model == self.customer_model && intent == self.intent)
                .then(|| self.route.clone())
        }
    }

    fn fixed_generation_route(
        model: &str,
        intent: GenerationRequestIntent,
    ) -> FixedGenerationRoutePolicy {
        FixedGenerationRoutePolicy {
            customer_model: "vendor/model",
            intent,
            route: GovernedGenerationRoute {
                model: model.to_string(),
                bundle: "default".to_string(),
                pool: "default".to_string(),
                machine_profile: "l4".to_string(),
            },
        }
    }

    #[derive(Default)]
    struct RecordingGenerationRoutePolicy {
        calls: std::sync::Mutex<Vec<(String, GenerationRequestIntent)>>,
    }

    impl RecordingGenerationRoutePolicy {
        fn take_calls(&self) -> Vec<(String, GenerationRequestIntent)> {
            std::mem::take(&mut *self.calls.lock().expect("route-policy call lock"))
        }
    }

    impl crate::server::GenerationRoutePolicy for RecordingGenerationRoutePolicy {
        fn resolve(
            &self,
            customer_model: &str,
            intent: GenerationRequestIntent,
        ) -> Option<GovernedGenerationRoute> {
            self.calls
                .lock()
                .expect("route-policy call lock")
                .push((customer_model.to_string(), intent));
            let model = match intent {
                GenerationRequestIntent::Default => customer_model.to_string(),
                GenerationRequestIntent::Grammar => format!("{customer_model}:no-spec"),
            };
            Some(GovernedGenerationRoute {
                model,
                bundle: "default".to_string(),
                pool: "default".to_string(),
                machine_profile: "l4".to_string(),
            })
        }
    }

    async fn recording_generation_state() -> (Arc<AppState>, Arc<RecordingGenerationRoutePolicy>) {
        let pool_manager = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        pool_manager.create_default_pool().await;
        let mut state = admission_test_state(pool_manager);
        state.model_registry = Arc::new(grammar_routed_registry());
        let policy = Arc::new(RecordingGenerationRoutePolicy::default());
        install_generation_policy(&mut state, policy.clone());
        (Arc::new(state), policy)
    }

    /// A policy that is installed but compiles no route at all, and records every
    /// id it was asked about. The recording is what makes the #3441 ordering
    /// observable: an unknown model must 404 without the policy ever being
    /// consulted, and "was consulted" is not visible in the status code alone.
    #[derive(Default)]
    struct EmptyGenerationRoutePolicy {
        asked: std::sync::Mutex<Vec<String>>,
    }

    impl EmptyGenerationRoutePolicy {
        fn asked(&self) -> Vec<String> {
            self.asked.lock().expect("asked lock").clone()
        }
    }

    impl crate::server::GenerationRoutePolicy for EmptyGenerationRoutePolicy {
        fn resolve(
            &self,
            customer_model: &str,
            _intent: GenerationRequestIntent,
        ) -> Option<GovernedGenerationRoute> {
            self.asked
                .lock()
                .expect("asked lock")
                .push(customer_model.to_string());
            None
        }
    }

    struct MappedGenerationRoutePolicy {
        routes:
            std::collections::HashMap<(String, GenerationRequestIntent), GovernedGenerationRoute>,
    }

    impl crate::server::GenerationRoutePolicy for MappedGenerationRoutePolicy {
        fn resolve(
            &self,
            customer_model: &str,
            intent: GenerationRequestIntent,
        ) -> Option<GovernedGenerationRoute> {
            self.routes
                .get(&(customer_model.to_string(), intent))
                .cloned()
        }
    }

    #[derive(Default)]
    struct GenerationTargetProbe {
        targets: std::sync::Mutex<Vec<(String, PublishTarget)>>,
    }

    impl GenerationTargetProbe {
        fn take_target(&self) -> (String, PublishTarget) {
            self.targets
                .lock()
                .expect("target probe lock")
                .pop()
                .expect("generation publish target")
        }

        fn target_count(&self) -> usize {
            self.targets.lock().expect("target probe lock").len()
        }
    }

    #[async_trait::async_trait]
    impl WorkDispatcher for GenerationTargetProbe {
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
            unreachable!("generation target probe only accepts generation")
        }

        async fn publish_generate_streaming(
            &self,
            target: PublishTarget,
            display_model: &str,
            _engine: &str,
            _bundle_config_hash: &str,
            _params: &WorkParams,
            _admission_pool: &str,
        ) -> Result<
            (
                String,
                oneshot::Receiver<StreamOutcome>,
                Arc<Notify>,
                DispatchDurability,
            ),
            String,
        > {
            self.targets
                .lock()
                .expect("target probe lock")
                .push((display_model.to_string(), target));
            let (tx, rx) = oneshot::channel();
            tx.send(StreamOutcome {
                text: "ok".to_string(),
                finish_reason: "stop".to_string(),
                usage: Some(crate::queue::streaming::UsageBlock {
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    total_tokens: 2,
                }),
                attempt_id: "attempt-1".to_string(),
                ttft_ms: None,
                tpot_ms: None,
                error: None,
                origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
                tool_calls: None,
                logprobs: None,
                candidates: Vec::new(),
                executed_bundle_config_hash: None,
                execution_identity_sha256: None,
                execution_binding_sha256: None,
            })
            .expect("target probe outcome receiver");
            Ok((
                "request-1".to_string(),
                rx,
                Arc::new(Notify::new()),
                DispatchDurability::accepted(),
            ))
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
            unreachable!("bounded target proof uses non-streaming requests")
        }

        async fn publish_cancel(&self, _request_id: &str) {}

        fn begin_work_abandonment(&self, _request_id: &str) -> bool {
            true
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

        fn drop_pending_stream(&self, _request_id: &str) {}

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

    fn mixed_governed_registry() -> crate::state::model_registry::ModelRegistry {
        use crate::types::model::{ModelConfig, ProfileConfig};

        let registry = grammar_routed_registry();
        let mut profiles = std::collections::HashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                kv_budget_tokens: None,
                max_output_tokens: None,
                grammar_profile: None,
                chat_template_kwargs: None,
                adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        let tasks: serde_yaml::Value = serde_yaml::from_str(
            "generate:\n  capabilities:\n    grammar: [json_schema]\n    streaming: false\n",
        )
        .expect("generation task");
        registry
            .add_model_config(ModelConfig {
                name: "org/h".to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: Some(tasks),
            })
            .expect("h100 test model");
        registry
    }

    async fn mixed_governed_generation_state(
        reverse_insertion: bool,
    ) -> (Arc<AppState>, Arc<GenerationTargetProbe>) {
        let profiles = vec![
            "l4".to_string(),
            "a100-80gb".to_string(),
            "h100".to_string(),
        ];
        let pool_manager = Arc::new(PoolManager::new(profiles.clone()));
        pool_manager.create_default_pool().await;
        let mut state = admission_test_state(pool_manager);
        state.model_registry = Arc::new(mixed_governed_registry());

        let physical_lanes = crate::state::demand_tracker::PhysicalLaneCatalog::try_from_raw(
            profiles.iter().map(|profile| {
                (
                    "default".to_string(),
                    profile.clone(),
                    "default".to_string(),
                )
            }),
        )
        .expect("mixed physical lanes");
        let config = Arc::make_mut(&mut state.config);
        config.configured_gpus = profiles.clone();
        config.gpu_profile_map = profiles
            .iter()
            .map(|profile| (profile.clone(), profile.clone()))
            .collect();
        config.configured_physical_lanes = physical_lanes.clone();
        state.demand_tracker = Arc::new(crate::state::demand_tracker::DemandTracker::new(
            physical_lanes,
        ));

        let mut routes = vec![
            (
                ("org/g".to_string(), GenerationRequestIntent::Default),
                GovernedGenerationRoute {
                    model: "org/g".to_string(),
                    bundle: "default".to_string(),
                    pool: "default".to_string(),
                    machine_profile: "l4".to_string(),
                },
            ),
            (
                ("org/g".to_string(), GenerationRequestIntent::Grammar),
                GovernedGenerationRoute {
                    model: "org/g:no-spec".to_string(),
                    bundle: "default".to_string(),
                    pool: "default".to_string(),
                    machine_profile: "a100-80gb".to_string(),
                },
            ),
            (
                ("org/h".to_string(), GenerationRequestIntent::Default),
                GovernedGenerationRoute {
                    model: "org/h".to_string(),
                    bundle: "default".to_string(),
                    pool: "default".to_string(),
                    machine_profile: "h100".to_string(),
                },
            ),
        ];
        if reverse_insertion {
            routes.reverse();
        }
        install_generation_policy(
            &mut state,
            Arc::new(MappedGenerationRoutePolicy {
                routes: routes.into_iter().collect(),
            }),
        );

        let (bundle_hash, _, _) = state
            .model_registry
            .bundle_execution_evidence("default", "default", "org/g");
        let mut worker_profiles = profiles;
        if reverse_insertion {
            worker_profiles.reverse();
        }
        for profile in worker_profiles {
            let mut status = worker_msg("default", &profile, "default");
            status.name = format!("worker-{profile}");
            status.bundle_config_hash = bundle_hash.clone();
            state
                .registry
                .update_worker(&format!("http://worker-{profile}:8080"), status)
                .await;
        }

        let probe = Arc::new(GenerationTargetProbe::default());
        state.work_publisher = Some(probe.clone());
        (Arc::new(state), probe)
    }

    fn assert_generation_target(
        target: (String, PublishTarget),
        display_model: &str,
        model: &str,
        profile: &str,
    ) {
        assert_eq!(target.0, display_model);
        let (pool, machine_profile, bundle, dispatch_model) = match target.1 {
            PublishTarget::Worker {
                pool,
                machine_profile,
                bundle,
                model,
                ..
            }
            | PublishTarget::Pool {
                pool,
                machine_profile,
                bundle,
                model,
            } => (pool, machine_profile, bundle, model),
        };
        assert_eq!(
            (
                pool.as_str(),
                machine_profile.as_str(),
                bundle.as_str(),
                dispatch_model.as_str()
            ),
            ("default", profile, "default", model)
        );
    }

    fn json_request(uri: &str, body: serde_json::Value) -> Request {
        Request::builder()
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).expect("request JSON")))
            .expect("request")
    }

    #[tokio::test]
    async fn governed_native_generate_routes_default_and_grammar_before_worker_lookup() {
        for (body, expected_intent) in [
            (
                serde_json::json!({"prompt": "hello", "max_new_tokens": 4}),
                GenerationRequestIntent::Default,
            ),
            (
                serde_json::json!({
                    "prompt": "hello",
                    "max_new_tokens": 4,
                    "grammar": {"json_schema": {"type": "object"}}
                }),
                GenerationRequestIntent::Grammar,
            ),
        ] {
            let (state, policy) = recording_generation_state().await;
            let response = proxy_request(
                State(state),
                json_request("/v1/generate/org%2Fg", body),
                "generate",
            )
            .await;
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(
                policy.take_calls(),
                vec![("org/g".to_string(), expected_intent)]
            );
        }
    }

    #[tokio::test]
    async fn governed_openai_generation_routes_chat_completions_and_responses() {
        let cases = [
            (
                "/v1/chat/completions",
                serde_json::json!({
                    "model": "org/g",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4
                }),
                GenerationRequestIntent::Default,
            ),
            (
                "/v1/chat/completions",
                serde_json::json!({
                    "model": "org/g",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer",
                            "schema": {"type": "object"}
                        }
                    }
                }),
                GenerationRequestIntent::Grammar,
            ),
            (
                "/v1/completions",
                serde_json::json!({
                    "model": "org/g",
                    "prompt": "hello",
                    "max_tokens": 4
                }),
                GenerationRequestIntent::Default,
            ),
            (
                "/v1/responses",
                serde_json::json!({
                    "model": "org/g",
                    "input": "hello",
                    "max_output_tokens": 4
                }),
                GenerationRequestIntent::Default,
            ),
        ];

        for (uri, body, expected_intent) in cases {
            let (state, policy) = recording_generation_state().await;
            let request = json_request(uri, body);
            let response = match uri {
                "/v1/chat/completions" => proxy_chat(State(state), request).await,
                "/v1/completions" => proxy_completions(State(state), request).await,
                "/v1/responses" => proxy_responses(State(state), request).await,
                _ => unreachable!("bounded OpenAI generation test surface"),
            };
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(
                policy.take_calls(),
                vec![("org/g".to_string(), expected_intent)],
                "governed route lookup for {uri}"
            );
        }
    }

    #[tokio::test]
    async fn governed_generation_dispatch_targets_are_exact_and_order_independent() {
        for reverse_insertion in [false, true] {
            let (state, probe) = mixed_governed_generation_state(reverse_insertion).await;

            let response = proxy_request(
                State(Arc::clone(&state)),
                json_request(
                    "/v1/generate/org%2Fg",
                    serde_json::json!({"prompt": "hello", "max_new_tokens": 4}),
                ),
                "generate",
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_generation_target(probe.take_target(), "org/g", "org/g", "l4");

            let response = proxy_request(
                State(Arc::clone(&state)),
                json_request(
                    "/v1/generate/org%2Fg",
                    serde_json::json!({
                        "prompt": "hello",
                        "max_new_tokens": 4,
                        "grammar": {"json_schema": {"type": "object"}}
                    }),
                ),
                "generate",
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_generation_target(probe.take_target(), "org/g", "org/g:no-spec", "a100-80gb");

            let response = proxy_chat(
                State(Arc::clone(&state)),
                json_request(
                    "/v1/chat/completions",
                    serde_json::json!({
                        "model": "org/h",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 4
                    }),
                ),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_generation_target(probe.take_target(), "org/h", "org/h", "h100");

            let response = proxy_chat(
                State(Arc::clone(&state)),
                json_request(
                    "/v1/chat/completions",
                    serde_json::json!({
                        "model": "org/g",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 4,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "answer",
                                "schema": {"type": "object"}
                            }
                        }
                    }),
                ),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_generation_target(probe.take_target(), "org/g", "org/g:no-spec", "a100-80gb");

            let response = proxy_completions(
                State(Arc::clone(&state)),
                json_request(
                    "/v1/completions",
                    serde_json::json!({
                        "model": "org/g",
                        "prompt": "hello",
                        "max_tokens": 4
                    }),
                ),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_generation_target(probe.take_target(), "org/g", "org/g", "l4");

            let response = proxy_responses(
                State(Arc::clone(&state)),
                json_request(
                    "/v1/responses",
                    serde_json::json!({
                        "model": "org/h",
                        "input": "hello",
                        "max_output_tokens": 4
                    }),
                ),
            )
            .await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_generation_target(probe.take_target(), "org/h", "org/h", "h100");
        }
    }

    #[tokio::test]
    async fn non_streaming_model_rejects_all_streaming_ingresses_before_publish() {
        let (state, probe) = mixed_governed_generation_state(false).await;
        let cases = [
            (
                "/v1/generate/org%2Fh",
                serde_json::json!({
                    "prompt": "hello",
                    "max_new_tokens": 4,
                    "stream": true
                }),
                "generate",
            ),
            (
                "/v1/chat/completions",
                serde_json::json!({
                    "model": "org/h",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4,
                    "stream": true
                }),
                "chat",
            ),
            (
                "/v1/completions",
                serde_json::json!({
                    "model": "org/h",
                    "prompt": "hello",
                    "max_tokens": 4,
                    "stream": true
                }),
                "completions",
            ),
        ];

        for (uri, body, endpoint) in cases {
            let response = match endpoint {
                "generate" => {
                    proxy_request(State(Arc::clone(&state)), json_request(uri, body), endpoint)
                        .await
                }
                "chat" => proxy_chat(State(Arc::clone(&state)), json_request(uri, body)).await,
                "completions" => {
                    proxy_completions(State(Arc::clone(&state)), json_request(uri, body)).await
                }
                _ => unreachable!("bounded streaming ingress"),
            };
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{endpoint}");
            let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
                .await
                .expect("error response body");
            let value: serde_json::Value =
                serde_json::from_slice(&body).expect("error response JSON");
            assert_eq!(
                value["error"]["code"],
                oai_code::UNSUPPORTED_FIELD,
                "{endpoint}"
            );
            assert_eq!(value["error"]["param"], "stream", "{endpoint}");
            assert_eq!(probe.target_count(), 0, "{endpoint} published queue work");
        }
    }

    #[tokio::test]
    async fn non_streaming_native_generate_rejects_before_cold_lane_demand() {
        let (state, probe) = mixed_governed_generation_state(false).await;
        let mut state = Arc::try_unwrap(state).unwrap_or_else(|_| panic!("unique test state"));
        state.registry = Arc::new(WorkerRegistry::new(Duration::from_secs(30), None));
        let state = Arc::new(state);

        let response = proxy_request(
            State(Arc::clone(&state)),
            json_request(
                "/v1/generate/org%2Fh",
                serde_json::json!({
                    "prompt": "hello",
                    "max_new_tokens": 4,
                    "stream": true
                }),
            ),
            "generate",
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(state.demand_tracker.active_lanes().is_empty());
        assert_eq!(probe.target_count(), 0);
    }

    #[tokio::test]
    async fn non_streaming_native_generate_rejects_before_zero_cap_admission() {
        let pool_manager = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        pool_manager.create_default_pool().await;
        pool_manager
            .create_pool_with_caps(
                "tenant",
                std::collections::HashMap::from([("l4".to_string(), 0)]),
                std::collections::HashMap::from([("l4".to_string(), 0)]),
                None,
                None,
                0,
                vec![],
            )
            .await
            .expect("zero-cap test pool");
        let mut state = admission_test_state(pool_manager);
        state.model_registry = Arc::new(mixed_governed_registry());
        install_generation_policy(
            &mut state,
            Arc::new(MappedGenerationRoutePolicy {
                routes: [(
                    ("org/h".to_string(), GenerationRequestIntent::Default),
                    GovernedGenerationRoute {
                        model: "org/h".to_string(),
                        bundle: "default".to_string(),
                        pool: "tenant".to_string(),
                        machine_profile: "l4".to_string(),
                    },
                )]
                .into_iter()
                .collect(),
            }),
        );
        let probe = Arc::new(GenerationTargetProbe::default());
        state.work_publisher = Some(probe.clone());
        let state = Arc::new(state);

        let response = proxy_request(
            State(Arc::clone(&state)),
            json_request(
                "/v1/generate/org%2Fh",
                serde_json::json!({
                    "prompt": "hello",
                    "max_new_tokens": 4,
                    "stream": true
                }),
            ),
            "generate",
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(state.demand_tracker.active_lanes().is_empty());
        assert_eq!(probe.target_count(), 0);
    }

    #[tokio::test]
    async fn self_hosted_native_grammar_dispatches_profile_variant_after_one_parse() {
        let (state, probe) = mixed_governed_generation_state(false).await;
        let mut state = Arc::try_unwrap(state).unwrap_or_else(|_| panic!("unique test state"));
        state.model_access_policy = None;
        let state = Arc::new(state);
        let mut request = json_request(
            "/v1/generate/org%2Fg",
            serde_json::json!({
                "prompt": "hello",
                "max_new_tokens": 4,
                "grammar": {"json_schema": {"type": "object"}}
            }),
        );
        request.headers_mut().insert(
            "x-sie-machine-profile",
            HeaderValue::from_static("a100-80gb"),
        );

        let response = proxy_request(State(state), request, "generate").await;

        assert_eq!(response.status(), StatusCode::OK);
        assert_generation_target(probe.take_target(), "org/g", "org/g:no-spec", "a100-80gb");
    }

    #[tokio::test]
    async fn installed_empty_generation_policy_fails_native_and_openai_closed() {
        let pool_manager = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        pool_manager.create_default_pool().await;
        let mut state = admission_test_state(pool_manager);
        state.model_registry = Arc::new(grammar_routed_registry());
        install_generation_policy(&mut state, Arc::new(EmptyGenerationRoutePolicy::default()));
        let state = Arc::new(state);

        let native = proxy_request(
            State(Arc::clone(&state)),
            json_request(
                "/v1/generate/org%2Fg",
                serde_json::json!({"prompt": "hello", "max_new_tokens": 4}),
            ),
            "generate",
        )
        .await;
        assert_eq!(native.status(), StatusCode::SERVICE_UNAVAILABLE);

        let chat = proxy_chat(
            State(state),
            json_request(
                "/v1/chat/completions",
                serde_json::json!({
                    "model": "org/g",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4
                }),
            ),
        )
        .await;
        assert_eq!(chat.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn governed_generation_route_is_exact_and_missing_keys_fail_closed() {
        let pool_manager = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut state = admission_test_state(pool_manager);
        install_generation_policy(
            &mut state,
            Arc::new(fixed_generation_route(
                "vendor/model:no-spec",
                GenerationRequestIntent::Grammar,
            )),
        );

        let route = governed_generation_route(
            &state,
            "vendor/model",
            "vendor/model:no-spec",
            GenerationRequestIntent::Grammar,
        )
        .expect("matching governed route")
        .expect("installed policy route");
        assert_eq!(route.machine_profile, "l4");
        assert_eq!(route.model, "vendor/model:no-spec");

        let missing = governed_generation_route(
            &state,
            "vendor/model",
            "vendor/model",
            GenerationRequestIntent::Default,
        )
        .expect_err("a governed mapping miss must not fall back");
        assert_eq!(missing.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn governed_generation_route_adopts_omitted_headers_and_rejects_wrong_gpu() {
        let pool_manager = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let state = admission_test_state(pool_manager);
        let route = fixed_generation_route("vendor/model", GenerationRequestIntent::Default).route;

        let adopted = governed_profile_and_pool(&state, &HeaderMap::new(), &route)
            .expect("omitted overrides adopt the compiled route");
        assert_eq!(adopted.gpu, "l4");
        assert_eq!(adopted.pool_name, "default");

        let mut headers = HeaderMap::new();
        headers.insert("x-sie-machine-profile", HeaderValue::from_static("h100"));
        let rejected = match governed_profile_and_pool(&state, &headers, &route) {
            Err(response) => response,
            Ok(_) => panic!("a conflicting GPU pin must not select a sibling lane"),
        };
        assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);

        let mut conflicting_pool_headers = HeaderMap::new();
        conflicting_pool_headers.insert(
            "x-sie-machine-profile",
            HeaderValue::from_static("default/l4"),
        );
        conflicting_pool_headers.insert("x-sie-pool", HeaderValue::from_static("other"));
        let rejected = match governed_profile_and_pool(&state, &conflicting_pool_headers, &route) {
            Err(response) => response,
            Ok(_) => panic!("conflicting inline and explicit pool pins must fail"),
        };
        assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);

        for header in ["x-sie-machine-profile", "x-sie-pool"] {
            let mut invalid_utf8 = HeaderMap::new();
            invalid_utf8.insert(
                header,
                HeaderValue::from_bytes(b"\xff").expect("opaque header"),
            );
            let rejected = match governed_profile_and_pool(&state, &invalid_utf8, &route) {
                Err(response) => response,
                Ok(_) => panic!("invalid UTF-8 routing headers must fail"),
            };
            assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);
        }
    }

    #[tokio::test]
    async fn governed_native_route_rejects_incompatible_compiled_bundle() {
        let pool_manager = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut state = admission_test_state(pool_manager);
        let mut policy = fixed_generation_route("vendor/model", GenerationRequestIntent::Default);
        policy.route.bundle = "missing-bundle".to_string();
        install_generation_policy(&mut state, Arc::new(policy));

        let result = resolve_routing(
            &state,
            &HeaderMap::new(),
            &axum::http::Extensions::new(),
            "/v1/generate/vendor%2Fmodel",
            "generate",
            Some(GenerationRequestIntent::Default),
        )
        .await;

        match result {
            Ok(_) => panic!("incompatible compiled bundle must fail closed"),
            Err(response) => assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE),
        }
    }

    /// A governed managed deployment whose registry holds exactly one generation
    /// model (`org/g`, with its `:no-spec` grammar variant) and whose installed
    /// policy compiles no route at all — so every governed lookup that runs is a
    /// miss, and a miss is a 503. Also carries an operator alias onto a model the
    /// data plane does not have, so the alias spelling can be exercised too.
    async fn governed_deployment_with_no_compiled_routes(
    ) -> (Arc<AppState>, Arc<EmptyGenerationRoutePolicy>) {
        let pool_manager = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        pool_manager.create_default_pool().await;
        let mut state = admission_test_state(pool_manager);
        state.model_registry = Arc::new(grammar_routed_registry());
        let mut config = (*state.config).clone();
        config
            .model_aliases
            .insert("retired".to_string(), "vendor/missing".to_string());
        state.config = Arc::new(config);
        let policy = Arc::new(EmptyGenerationRoutePolicy::default());
        install_generation_policy(&mut state, policy.clone());
        (Arc::new(state), policy)
    }

    fn openai_error(body: &serde_json::Value, path: &str) -> (String, String, String) {
        let error = &body["error"];
        for field in ["type", "code", "param"] {
            assert!(
                error[field].is_string(),
                "{path}: OpenAI error envelope is missing `{field}`: {body}"
            );
        }
        (
            error["type"].as_str().expect("type").to_string(),
            error["code"].as_str().expect("code").to_string(),
            error["param"].as_str().expect("param").to_string(),
        )
    }

    /// #3441 ordering lock on the NATIVE PUBLIC ROUTE: a model that does not
    /// resolve in this data plane is answered `404 MODEL_NOT_FOUND` before the
    /// governed `(customer_model, intent)` lookup can report its miss as a 503.
    ///
    /// Driven through `proxy_request`, not `governed_generation_route`: the
    /// helper-level test proves a miss fails closed, which is precisely the
    /// verdict that used to leak onto this route, so it cannot observe the
    /// ordering. `policy.asked()` is the ordering evidence — the status code alone
    /// cannot distinguish "404 first" from "asked, missed, then 404 anyway".
    ///
    /// The spellings cover the ways one absent id can reach the route: the SDK
    /// `__` form (the prod-US 2026-08-14 report), percent-encoded and literal
    /// slashes, case folding, a `:profile` suffix, an explicit `bundle:/` pin, and
    /// an operator alias whose target is absent. Each runs with and without a
    /// grammar block, because grammar is the other intent the governed key carries.
    #[tokio::test]
    async fn native_generate_unknown_model_404s_before_the_governed_lookup() {
        let (state, policy) = governed_deployment_with_no_compiled_routes().await;
        let bodies = [
            serde_json::json!({"prompt": "hello", "max_new_tokens": 4}),
            serde_json::json!({
                "prompt": "hello",
                "max_new_tokens": 4,
                "grammar": {"json_schema": {"type": "object"}}
            }),
        ];

        for path in [
            "/v1/generate/Qwen__Qwen3-4B-Instruct-2507",
            "/v1/generate/vendor%2Fmissing",
            "/v1/generate/vendor/missing",
            "/v1/generate/VENDOR__MISSING",
            "/v1/generate/vendor%2Fmissing%3Ano-spec",
            "/v1/generate/default%3A%2Fvendor%2Fmissing",
            "/v1/generate/retired",
        ] {
            for body in &bodies {
                let response = proxy_request(
                    State(Arc::clone(&state)),
                    json_request(path, body.clone()),
                    "generate",
                )
                .await;
                assert_eq!(
                    response.status(),
                    StatusCode::NOT_FOUND,
                    "{path} must be terminally not-found, not a retryable 503"
                );
                let body = to_bytes(response.into_body(), 64 * 1024)
                    .await
                    .expect("body");
                let body: serde_json::Value = serde_json::from_slice(&body).expect("error JSON");
                assert_eq!(
                    openai_error(&body, path),
                    (
                        "model_not_found".to_string(),
                        "model_not_found".to_string(),
                        "model".to_string()
                    ),
                    "{path}"
                );
            }
        }
        assert!(
            policy.asked().is_empty(),
            "governed lookup ran for models this deployment does not have: {:?}",
            policy.asked()
        );

        // The control, and the half of the contract this fix must NOT widen: a
        // model the registry DOES have, whose governed route was never compiled,
        // is deployment drift. It still fails closed as 503 — and the policy is
        // consulted, which is what proves the 404 above is about absence and not
        // about the governed step having been skipped wholesale.
        let known = proxy_request(
            State(Arc::clone(&state)),
            json_request(
                "/v1/generate/org%2Fg",
                serde_json::json!({"prompt": "hello", "max_new_tokens": 4}),
            ),
            "generate",
        )
        .await;
        assert_eq!(known.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(policy.asked(), vec!["org/g".to_string()]);
    }

    /// #2542 no-oracle rendering, carried onto the #3441 path: the governed
    /// deployment's unknown-model 404 must be **byte-identical** to the one the
    /// same registry gives with no policy installed. If the governed surface grew
    /// its own wording, the response would tell a caller whether this deployment
    /// governs generation at all.
    #[tokio::test]
    async fn governed_unknown_model_404_is_byte_identical_to_the_ungoverned_one() {
        let path = "/v1/generate/Qwen__Qwen3-4B-Instruct-2507";
        let body = serde_json::json!({"prompt": "hello", "max_new_tokens": 4});

        let (governed_state, _policy) = governed_deployment_with_no_compiled_routes().await;
        let governed = proxy_request(
            State(governed_state),
            json_request(path, body.clone()),
            "generate",
        )
        .await;
        assert_eq!(governed.status(), StatusCode::NOT_FOUND);

        let (ungoverned_state, _) = governed_deployment_with_no_compiled_routes().await;
        let mut ungoverned_state =
            Arc::try_unwrap(ungoverned_state).unwrap_or_else(|_| panic!("unique test state"));
        ungoverned_state.model_access_policy = None;
        let ungoverned = proxy_request(
            State(Arc::new(ungoverned_state)),
            json_request(path, body),
            "generate",
        )
        .await;
        assert_eq!(ungoverned.status(), StatusCode::NOT_FOUND);

        let governed = to_bytes(governed.into_body(), 64 * 1024)
            .await
            .expect("body");
        let ungoverned = to_bytes(ungoverned.into_body(), 64 * 1024)
            .await
            .expect("body");
        assert_eq!(
            String::from_utf8(governed.to_vec()).expect("utf8"),
            String::from_utf8(ungoverned.to_vec()).expect("utf8"),
            "governed and ungoverned not-found must be one response"
        );
    }

    /// The OpenAI body-model route was already correct (`resolve_model_and_bundle`
    /// runs before `resolve_generation_route`); this pins that the shared
    /// `unknown_model_response` refactor kept it that way, on the same deployment
    /// that exercises the native path above.
    #[tokio::test]
    async fn openai_body_model_keeps_model_not_found_under_a_governed_policy() {
        let (state, policy) = governed_deployment_with_no_compiled_routes().await;
        let chat = proxy_chat(
            State(Arc::clone(&state)),
            json_request(
                "/v1/chat/completions",
                serde_json::json!({
                    "model": "Qwen__Qwen3-4B-Instruct-2507",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4
                }),
            ),
        )
        .await;
        assert_eq!(chat.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(chat.into_body(), 64 * 1024).await.expect("body");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("error JSON");
        assert_eq!(
            openai_error(&body, "/v1/chat/completions"),
            (
                "model_not_found".to_string(),
                "model_not_found".to_string(),
                "model".to_string()
            )
        );
        assert!(
            policy.asked().is_empty(),
            "governed lookup ran for an unknown OpenAI body model: {:?}",
            policy.asked()
        );
    }

    /// A #1841 test policy: reports `visible` per the flag and marks any
    /// `org-`-prefixed id as sealed. Lets the native resolve path's sealed branch be
    /// exercised (and its ordering behind the visibility gate be locked) without the
    /// managed crate.
    struct SealedTestPolicy {
        visible: bool,
    }
    impl crate::server::ModelAccessPolicy for SealedTestPolicy {
        fn visible(&self, _resolved_model: &str, _ext: &axum::http::Extensions) -> bool {
            self.visible
        }
        fn sealed_route(
            &self,
            resolved_model: &str,
            _ext: &axum::http::Extensions,
        ) -> Option<crate::server::SealedRoute> {
            resolved_model
                .starts_with("org-")
                .then(|| crate::server::SealedRoute {
                    engine: "sealed".to_string(),
                    bundle: "sealed".to_string(),
                    pool: "sealed".to_string(),
                    machine_profile: "sealed".to_string(),
                })
        }
    }

    /// A #2542 test policy: refuses one model outright and records every id it
    /// was asked about, so a test can prove WHICH string the seam consults —
    /// the caller's, or the one the registry resolved.
    struct RefusingTestPolicy {
        refuse: &'static str,
        visible: bool,
        asked: Arc<std::sync::Mutex<Vec<String>>>,
    }
    impl RefusingTestPolicy {
        fn new(refuse: &'static str) -> Self {
            Self {
                refuse,
                visible: true,
                asked: Arc::new(std::sync::Mutex::new(Vec::new())),
            }
        }
        fn asked(&self) -> Vec<String> {
            self.asked.lock().expect("asked lock").clone()
        }
    }
    impl crate::server::ModelAccessPolicy for RefusingTestPolicy {
        fn visible(&self, _resolved_model: &str, _ext: &axum::http::Extensions) -> bool {
            self.visible
        }
        fn serving_refusal(
            &self,
            resolved_model: &str,
            _ext: &axum::http::Extensions,
        ) -> Option<Response> {
            self.asked
                .lock()
                .expect("asked lock")
                .push(resolved_model.to_string());
            (resolved_model == self.refuse)
                .then(|| (StatusCode::FORBIDDEN, "refused").into_response())
        }
        /// Seals every `org-` id, exactly like [`SealedTestPolicy`], so ONE
        /// stub can both refuse and seal the same model — the only way to
        /// observe which of the two branches `resolve_routing` reaches first.
        fn sealed_route(
            &self,
            resolved_model: &str,
            _ext: &axum::http::Extensions,
        ) -> Option<crate::server::SealedRoute> {
            resolved_model
                .starts_with("org-")
                .then(|| crate::server::SealedRoute {
                    engine: "sealed".to_string(),
                    bundle: "sealed".to_string(),
                    pool: "sealed".to_string(),
                    machine_profile: "sealed".to_string(),
                })
        }
    }

    fn state_with_alias(alias: &str, target: &str) -> AppState {
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut state = admission_test_state(pm);
        let mut config = (*state.config).clone();
        config
            .model_aliases
            .insert(alias.to_string(), target.to_string());
        state.config = Arc::new(config);
        state
    }

    /// A registry holding exactly one real model, in temp dirs the CALLER owns.
    ///
    /// `admission_test_state`'s dirs are dropped with it, so its registry is
    /// permanently empty — which silently routes an unknown model to the
    /// empty-registry bundle fallback instead of the `has_any_models` 404. The
    /// #2542 no-oracle tests compare against that 404, so they need a registry
    /// that is populated and stays on disk.
    fn populated_registry() -> (Arc<ModelRegistry>, tempfile::TempDir, tempfile::TempDir) {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            bundles_dir.path().join("default.yaml"),
            "name: default\nadapters:\n  - module\ndefault: true\n",
        )
        .unwrap();
        std::fs::write(
            models_dir.path().join("known.yaml"),
            "sie_id: known/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.sentence_transformer:Adapter\n",
        )
        .unwrap();
        let registry = Arc::new(ModelRegistry::new(
            bundles_dir.path(),
            models_dir.path(),
            true,
        ));
        assert!(registry.has_any_models());
        (registry, bundles_dir, models_dir)
    }

    fn hiding_policy() -> Arc<RefusingTestPolicy> {
        Arc::new(RefusingTestPolicy {
            refuse: "nothing-is-refused-here",
            visible: false,
            asked: Arc::new(std::sync::Mutex::new(Vec::new())),
        })
    }

    /// #2542: the seam asks about the model the ALIAS resolves to, not the string
    /// the caller sent. An edge gate can only ever see `fast`, so this is the
    /// vantage point that makes the refusal authoritative.
    #[tokio::test]
    async fn resolve_routing_refuses_a_model_the_policy_will_not_serve() {
        let mut state = state_with_alias("fast", "vendor/refused");
        let policy = Arc::new(RefusingTestPolicy::new("vendor/refused"));
        state.model_access_policy =
            Some(Arc::clone(&policy) as Arc<dyn crate::server::ModelAccessPolicy>);
        let result = resolve_routing(
            &state,
            &HeaderMap::new(),
            &axum::http::Extensions::new(),
            "/v1/encode/fast",
            "encode",
            None,
        )
        .await;
        match result {
            Ok(_) => panic!("a refused model must never reach a lane"),
            Err(resp) => assert_eq!(resp.status(), StatusCode::FORBIDDEN),
        }
        assert_eq!(policy.asked(), vec!["vendor/refused".to_string()]);
    }

    /// The other direction: an alias onto a model the policy serves resolves and
    /// routes normally, so the verdict is a decision and not a blanket deny.
    #[tokio::test]
    async fn resolve_routing_serves_an_alias_the_policy_allows() {
        let mut state = state_with_alias("quick", "vendor/served");
        let policy = Arc::new(RefusingTestPolicy::new("vendor/refused"));
        state.model_access_policy =
            Some(Arc::clone(&policy) as Arc<dyn crate::server::ModelAccessPolicy>);
        let routing = resolve_routing(
            &state,
            &HeaderMap::new(),
            &axum::http::Extensions::new(),
            "/v1/encode/quick",
            "encode",
            None,
        )
        .await
        .expect("an allowed alias must route");
        assert_eq!(routing.model_name, "vendor/served");
        assert_eq!(policy.asked(), vec!["vendor/served".to_string()]);
    }

    /// Ordering lock (#1841 × #2542): a caller who may not SEE the model is
    /// answered the flat 404 and the serving verdict is never consulted, so a
    /// refusal shape can never become a cross-org existence oracle.
    #[tokio::test]
    async fn resolve_routing_runs_the_visibility_gate_before_the_serving_refusal() {
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut state = admission_test_state(pm);
        let mut policy = RefusingTestPolicy::new("org-7/support-embedder");
        policy.visible = false;
        let policy = Arc::new(policy);
        state.model_access_policy =
            Some(Arc::clone(&policy) as Arc<dyn crate::server::ModelAccessPolicy>);
        let result = resolve_routing(
            &state,
            &HeaderMap::new(),
            &axum::http::Extensions::new(),
            "/v1/encode/org-7/support-embedder",
            "encode",
            None,
        )
        .await;
        match result {
            Ok(_) => panic!("a hidden model must 404"),
            Err(resp) => assert_eq!(resp.status(), StatusCode::NOT_FOUND),
        }
        assert!(
            policy.asked().is_empty(),
            "the serving verdict must not run for a model the caller cannot see"
        );
    }

    /// The OTHER half of that ordering (#2542): the refusal runs BEFORE the
    /// sealed branch, so a refused model cannot reach a lane.
    ///
    /// The sealed branch returns `Ok(RoutingResult)` — a dispatch — so a
    /// reorder would not merely change a status code, it would serve a model
    /// the deployment refuses. One policy that both refuses and seals the same
    /// `org-N/` id is the only way to observe which branch wins; separate
    /// stubs, each exercising its own branch, would pass either way.
    #[tokio::test]
    async fn resolve_routing_refuses_before_it_seals() {
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut state = admission_test_state(pm);
        let policy = Arc::new(RefusingTestPolicy::new("org-7/support-embedder"));
        state.model_access_policy =
            Some(Arc::clone(&policy) as Arc<dyn crate::server::ModelAccessPolicy>);
        let result = resolve_routing(
            &state,
            &HeaderMap::new(),
            &axum::http::Extensions::new(),
            "/v1/encode/org-7/support-embedder",
            "encode",
            None,
        )
        .await;
        match result {
            Ok(routing) => panic!("a refused model reached the {} lane", routing.engine),
            Err(resp) => assert_eq!(resp.status(), StatusCode::FORBIDDEN),
        }

        // The control: the SAME policy still seals an org id it does not
        // refuse, so the assertion above is about ordering and not about the
        // sealed branch having been disabled.
        let sealed = resolve_routing(
            &state,
            &HeaderMap::new(),
            &axum::http::Extensions::new(),
            "/v1/encode/org-7/other-embedder",
            "encode",
            None,
        )
        .await
        .expect("an unrefused org model still seals");
        assert_eq!(sealed.engine, "sealed");
    }

    /// #2542 no-oracle rendering, native surface. An operator alias whose target
    /// sits in another org's namespace must not be readable out of the 404 body,
    /// and the refusal must be **byte-identical** to the one an absent model
    /// gets — otherwise the wording alone tells a tenant that something is there
    /// and hidden. Both halves are asserted, because fixing only the id would
    /// leave the message template as the oracle.
    #[tokio::test]
    async fn resolve_routing_cross_org_404_reveals_nothing_the_caller_did_not_send() {
        let (registry, _bundles, _models) = populated_registry();

        let mut hidden_state = state_with_alias("support", "org-42/support-gen");
        hidden_state.model_registry = Arc::clone(&registry);
        hidden_state.model_access_policy =
            Some(hiding_policy() as Arc<dyn crate::server::ModelAccessPolicy>);
        let hidden = match resolve_routing(
            &hidden_state,
            &HeaderMap::new(),
            &axum::http::Extensions::new(),
            "/v1/encode/support",
            "encode",
            None,
        )
        .await
        {
            Ok(_) => panic!("a hidden model must 404"),
            Err(resp) => *resp,
        };
        assert_eq!(hidden.status(), StatusCode::NOT_FOUND);
        let hidden_body = to_bytes(hidden.into_body(), 64 * 1024).await.expect("body");
        let hidden_body = String::from_utf8(hidden_body.to_vec()).expect("utf8");
        assert!(
            !hidden_body.contains("org-42"),
            "the alias target must not appear in the 404 body: {hidden_body}"
        );
        assert!(hidden_body.contains("support"), "{hidden_body}");

        // The SAME request with no policy installed, against the same populated
        // registry: `support` resolves to a model that simply is not there. The
        // two responses must be one response — otherwise the wording alone tells
        // the caller that something exists and is hidden from them.
        let mut absent_state = state_with_alias("support", "org-42/support-gen");
        absent_state.model_registry = registry;
        absent_state.model_access_policy = None;
        let absent = match resolve_routing(
            &absent_state,
            &HeaderMap::new(),
            &axum::http::Extensions::new(),
            "/v1/encode/support",
            "encode",
            None,
        )
        .await
        {
            Ok(_) => panic!("an absent model must 404 against a populated registry"),
            Err(resp) => *resp,
        };
        assert_eq!(absent.status(), StatusCode::NOT_FOUND);
        let absent_body = to_bytes(absent.into_body(), 64 * 1024).await.expect("body");
        assert_eq!(
            hidden_body,
            String::from_utf8(absent_body.to_vec()).expect("utf8"),
            "hidden and absent must be one response"
        );
    }

    /// The same rule on the OpenAI body seam, and the same
    /// hidden-vs-absent equality — this time against a REAL populated registry,
    /// so the comparison is with the live `has_any_models` 404 rather than a
    /// fallback branch.
    #[tokio::test]
    async fn resolve_model_and_bundle_cross_org_404_is_the_absent_model_404() {
        // A populated registry so the unknown-model arm (not the empty-registry
        // fallback) is what an absent model reaches.
        let (registry, _bundles, _models) = populated_registry();

        let mut state = state_with_alias("support", "org-42/support-gen");
        state.model_registry = Arc::clone(&registry);
        state.model_access_policy =
            Some(hiding_policy() as Arc<dyn crate::server::ModelAccessPolicy>);
        let hidden = resolve_model_and_bundle(&state, "support", &axum::http::Extensions::new())
            .expect_err("a hidden model must 404");
        assert_eq!(hidden.status(), StatusCode::NOT_FOUND);

        let mut open = state_with_alias("support", "org-42/support-gen");
        open.model_registry = registry;
        open.model_access_policy = None;
        let absent = resolve_model_and_bundle(&open, "support", &axum::http::Extensions::new())
            .expect_err("an unresolvable model must 404");
        assert_eq!(absent.status(), StatusCode::NOT_FOUND);

        let hidden = to_bytes(hidden.into_body(), 64 * 1024).await.expect("body");
        let absent = to_bytes(absent.into_body(), 64 * 1024).await.expect("body");
        let hidden = String::from_utf8(hidden.to_vec()).expect("utf8");
        assert!(
            !hidden.contains("org-42"),
            "the alias target must not appear in the 404 body: {hidden}"
        );
        assert_eq!(
            hidden,
            String::from_utf8(absent.to_vec()).expect("utf8"),
            "hidden and absent must be one response"
        );
    }

    /// The OpenAI body-model seam carries the same verdict on the same resolved
    /// id — one hook, both routing entry points, no per-surface copy.
    #[test]
    fn resolve_model_and_bundle_refuses_a_model_the_policy_will_not_serve() {
        let mut state = state_with_alias("fast", "vendor/refused");
        let policy = Arc::new(RefusingTestPolicy::new("vendor/refused"));
        state.model_access_policy =
            Some(Arc::clone(&policy) as Arc<dyn crate::server::ModelAccessPolicy>);
        let err = resolve_model_and_bundle(&state, "fast", &axum::http::Extensions::new())
            .expect_err("a refused model must never resolve to a bundle");
        assert_eq!(err.status(), StatusCode::FORBIDDEN);
        assert_eq!(policy.asked(), vec!["vendor/refused".to_string()]);

        let (model, _bundle) =
            resolve_model_and_bundle(&state, "vendor/served", &axum::http::Extensions::new())
                .expect("an allowed model still resolves");
        assert_eq!(model, "vendor/served");
    }

    #[tokio::test]
    async fn resolve_routing_sends_owned_custom_model_to_the_sealed_lane() {
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut state = admission_test_state(pm);
        state.model_access_policy = Some(Arc::new(SealedTestPolicy { visible: true }));
        let routing = resolve_routing(
            &state,
            &HeaderMap::new(),
            &axum::http::Extensions::new(),
            "/v1/encode/org-7/support-embedder",
            "encode",
            None,
        )
        .await
        .expect("a visible sealed model must route, not error");
        // Bundle-less sealed dispatch: `engine` is the sealed marker and the lane
        // tuple is synthetic. bundle/profile resolution was SKIPPED — the model is
        // not in the registry, so the normal path would have returned MODEL_NOT_FOUND.
        assert_eq!(routing.engine, "sealed");
        assert_eq!(routing.bundle, "sealed");
        assert_eq!(routing.pool_name, "sealed");
        assert_eq!(routing.gpu, "sealed");
        assert_eq!(routing.model_name, "org-7/support-embedder");
    }

    #[tokio::test]
    async fn resolve_routing_runs_visibility_gate_before_the_sealed_branch() {
        // Ordering lock: a non-visible caller is 404'd BEFORE the sealed branch, so a
        // cross-org / anonymous caller can never coerce engine="sealed".
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut state = admission_test_state(pm);
        state.model_access_policy = Some(Arc::new(SealedTestPolicy { visible: false }));
        let result = resolve_routing(
            &state,
            &HeaderMap::new(),
            &axum::http::Extensions::new(),
            "/v1/encode/org-7/support-embedder",
            "encode",
            None,
        )
        .await;
        match result {
            Ok(_) => panic!("a hidden model must 404, never reach the sealed branch"),
            Err(resp) => assert_eq!(resp.status(), StatusCode::NOT_FOUND),
        }
    }

    #[tokio::test]
    async fn test_capped_lane_admission_zero_cap_returns_pool_unavailable() {
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4".to_string(), 0);
        let mut caps = std::collections::HashMap::new();
        caps.insert("l4".to_string(), 0);
        pm.create_pool_with_caps("tenant", gpus, caps, None, None, 0, vec![])
            .await
            .unwrap();
        let state = admission_test_state(Arc::clone(&pm));

        let resp = capped_lane_admission_response(
            &state,
            "tenant",
            DEFAULT_POOL_NAME,
            "l4",
            "default",
            ProvisioningSurface::Native,
        )
        .await
        .expect("zero-cap lane should fail fast");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["detail"]["code"], err_code::POOL_CAPACITY_UNAVAILABLE);
    }

    #[tokio::test]
    async fn test_capped_lane_admission_zero_cap_openai_uses_openai_envelope() {
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4".to_string(), 0);
        let mut caps = std::collections::HashMap::new();
        caps.insert("l4".to_string(), 0);
        pm.create_pool_with_caps("tenant", gpus, caps, None, None, 0, vec![])
            .await
            .unwrap();
        let state = admission_test_state(Arc::clone(&pm));

        let resp = capped_lane_admission_response(
            &state,
            "tenant",
            DEFAULT_POOL_NAME,
            "l4",
            "default",
            ProvisioningSurface::OpenAiCompat,
        )
        .await
        .expect("zero-cap lane should fail fast");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            err_code::POOL_CAPACITY_UNAVAILABLE
        );

        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(value.get("detail").is_none());
        assert_eq!(value["error"]["type"], oai_type::SERVER_ERROR);
        assert_eq!(value["error"]["code"], oai_code::TRANSPORT_FAILURE);
    }

    #[tokio::test]
    async fn test_capped_lane_admission_zero_assigned_returns_provisioning() {
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4".to_string(), 0);
        let mut caps = std::collections::HashMap::new();
        caps.insert("l4".to_string(), 1);
        pm.create_pool_with_caps("tenant", gpus, caps, None, None, 0, vec![])
            .await
            .unwrap();
        let state = admission_test_state(Arc::clone(&pm));
        let demand_pool = "test-zero-assigned-native";
        let bundle = "test-zero-assigned-native-bundle";

        let resp = capped_lane_admission_response(
            &state,
            "tenant",
            demand_pool,
            "l4",
            bundle,
            ProvisioningSurface::Native,
        )
        .await
        .expect("capped lane with no admitted workers should provision");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get("retry-after").unwrap(),
            PROVISIONING_RETRY_AFTER
        );
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            err_code::PROVISIONING
        );
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], err_code::PROVISIONING);
        assert!(value.get("status").is_none());

        let lane = state
            .demand_tracker
            .resolve_lane(demand_pool, "l4", bundle)
            .unwrap();
        assert!(state.demand_tracker.active_lanes().contains(&lane));
        state.demand_tracker.clear(&lane);
        assert!(!state.demand_tracker.active_lanes().contains(&lane));
    }

    #[tokio::test]
    async fn test_capped_lane_admission_zero_assigned_openai_returns_retryable_503() {
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4".to_string(), 0);
        let mut caps = std::collections::HashMap::new();
        caps.insert("l4".to_string(), 1);
        pm.create_pool_with_caps("tenant", gpus, caps, None, None, 0, vec![])
            .await
            .unwrap();
        let state = admission_test_state(Arc::clone(&pm));
        let demand_pool = "test-zero-assigned-openai";
        let bundle = "test-zero-assigned-openai-bundle";

        let resp = capped_lane_admission_response(
            &state,
            "tenant",
            demand_pool,
            "l4",
            bundle,
            ProvisioningSurface::OpenAiCompat,
        )
        .await
        .expect("OpenAI-compatible capped lane should provision");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get("retry-after").unwrap(),
            PROVISIONING_RETRY_AFTER
        );
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            err_code::PROVISIONING
        );

        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(value.get("detail").is_none());
        assert_eq!(value["error"]["type"], oai_type::SERVER_ERROR);
        assert_eq!(value["error"]["code"], oai_code::PROVISIONING);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("Provisioning in progress"));
        let lane = state
            .demand_tracker
            .resolve_lane(demand_pool, "l4", bundle)
            .unwrap();
        state.demand_tracker.clear(&lane);
    }

    #[tokio::test]
    async fn test_record_publish_failure_backpressure_records_demand() {
        // Backpressure (a present-but-saturated worker) must record pending
        // demand so KEDA scales the lane; a generic publish failure must not.
        // See #1568.
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let state = admission_test_state(Arc::clone(&pm));
        let bp_pool = "test-bp-demand-pool";
        let bundle = "test-bp-demand-bundle";
        let lane = state
            .demand_tracker
            .resolve_lane(bp_pool, "l4", bundle)
            .unwrap();

        let retry = record_publish_failure(
            &state,
            &lane,
            "stream publish failed: backpressure (lane saturated)",
        );
        assert_eq!(retry, Some(BACKPRESSURE_RETRY_AFTER));
        assert!(
            state.demand_tracker.active_lanes().contains(&lane),
            "backpressure must record pending demand for KEDA"
        );
        state.demand_tracker.clear(&lane);

        // A full work stream rejects the publish at the broker under
        // `DiscardPolicy::New`; that is backpressure by another name, so it
        // must earn the same Retry-After and KEDA demand signal.
        let retry_full = record_publish_failure(
            &state,
            &lane,
            "failed to publish: maximum messages exceeded",
        );
        assert_eq!(retry_full, Some(BACKPRESSURE_RETRY_AFTER));
        assert!(
            state.demand_tracker.active_lanes().contains(&lane),
            "a stream-full rejection must record pending demand for KEDA"
        );
        state.demand_tracker.clear(&lane);

        let retry_other = record_publish_failure(&state, &lane, "connection reset by peer");
        assert_eq!(retry_other, None);
        assert!(
            state.demand_tracker.active_lanes().is_empty(),
            "a generic publish failure must not record demand"
        );
    }

    #[tokio::test]
    async fn durable_dispatch_clears_only_the_exact_confirmed_lane() {
        let state = admission_test_state(Arc::new(PoolManager::new(vec!["l4".to_string()])));
        let confirmed = state
            .demand_tracker
            .resolve_lane("test-bp-demand-pool", "l4", "test-bp-demand-bundle")
            .unwrap();
        let retained = state
            .demand_tracker
            .resolve_lane(
                "test-zero-assigned-native",
                "l4",
                "test-zero-assigned-native-bundle",
            )
            .unwrap();
        assert!(state.demand_tracker.record(&retained));
        assert!(state.demand_tracker.record(&confirmed));
        let handoff = state
            .demand_tracker
            .begin_dispatch_handoff(&confirmed)
            .unwrap();

        state
            .demand_tracker
            .finish_dispatch_handoff(&confirmed, handoff, true);

        assert_eq!(state.demand_tracker.active_lanes(), vec![retained]);
    }

    #[tokio::test]
    async fn failed_or_unavailable_durability_retains_pending_demand() {
        let state = admission_test_state(Arc::new(PoolManager::new(vec!["l4".to_string()])));
        let lane = state
            .demand_tracker
            .resolve_lane("test-bp-demand-pool", "l4", "test-bp-demand-bundle")
            .unwrap();

        for durability in [
            DispatchDurability::from_result(Err("broker rejected publish".to_string())),
            DispatchDurability::from_result(Err("durability monitor unavailable".to_string())),
        ] {
            let guard = DispatchHandoffGuard::new(Arc::clone(&state.demand_tracker), lane.clone())
                .expect("configured lane handoff");
            guard.finish(durability.wait().await.is_ok());
            assert_eq!(state.demand_tracker.active_lanes(), vec![lane.clone()]);
        }
    }

    #[tokio::test]
    async fn cancelled_durability_task_retains_pending_demand() {
        let state = admission_test_state(Arc::new(PoolManager::new(vec!["l4".to_string()])));
        let lane = state
            .demand_tracker
            .resolve_lane("test-bp-demand-pool", "l4", "test-bp-demand-bundle")
            .unwrap();
        let guard = DispatchHandoffGuard::new(Arc::clone(&state.demand_tracker), lane.clone())
            .expect("configured lane handoff");
        let task = tokio::spawn(async move {
            let _guard = guard;
            std::future::pending::<()>().await;
        });
        tokio::task::yield_now().await;
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert_eq!(state.demand_tracker.active_lanes(), vec![lane]);
    }

    #[tokio::test]
    async fn adversarial_unique_profile_headers_cannot_create_keda_demand() {
        let state = Arc::new(admission_test_state(Arc::new(PoolManager::new(vec![
            "l4".to_string()
        ]))));

        for index in 0..2_048 {
            let request = Request::builder()
                .uri("/v1/encode/org/model")
                .header("x-sie-machine-profile", format!("caller-{index}"))
                .body(Body::empty())
                .unwrap();
            let response = proxy_request(State(Arc::clone(&state)), request, "encode").await;
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
            let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(value["detail"]["code"], err_code::GPU_NOT_CONFIGURED);
        }

        assert!(
            state.demand_tracker.active_lanes().is_empty(),
            "caller-controlled machine-profile headers must never create KEDA lanes"
        );
    }

    #[tokio::test]
    async fn empty_configured_profile_catalog_is_fail_closed_not_allow_all() {
        let mut state = admission_test_state(Arc::new(PoolManager::new(Vec::new())));
        let config = Arc::make_mut(&mut state.config);
        config.configured_gpus.clear();
        config.gpu_profile_map.clear();
        let state = Arc::new(state);
        let request = Request::builder()
            .uri("/v1/encode/org/model")
            .header("x-sie-machine-profile", "caller-selected-gpu")
            .body(Body::empty())
            .unwrap();

        let response = proxy_request(State(Arc::clone(&state)), request, "encode").await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 64 * 1024).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["detail"]["code"], err_code::GPU_NOT_CONFIGURED);
        assert!(state.demand_tracker.active_lanes().is_empty());
    }

    #[tokio::test]
    async fn test_batch_publish_target_directs_capped_logical_pool_to_assigned_cold_worker() {
        let pm = Arc::new(PoolManager::new(vec!["l4".to_string()]));
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4".to_string(), 0);
        let mut caps = std::collections::HashMap::new();
        caps.insert("l4".to_string(), 1);
        pm.create_pool_with_caps("tenant", gpus, caps, None, None, 0, vec![])
            .await
            .unwrap();
        pm.assign_workers(
            "tenant",
            &[(
                "assigned-cold".to_string(),
                "http://assigned-cold:8080".to_string(),
                "l4".to_string(),
                "default".to_string(),
                DEFAULT_POOL_NAME.to_string(),
            )],
        )
        .await;

        let state = admission_test_state(Arc::clone(&pm));
        state
            .registry
            .update_worker(
                "http://assigned-cold:8080",
                crate::types::WorkerStatusMessage {
                    name: "assigned-cold".to_string(),
                    ready: true,
                    gpu_count: 1,
                    total_gpu_slots: Some(1),
                    ready_gpu_slots: Some(1),
                    machine_profile: "l4".to_string(),
                    pool_name: DEFAULT_POOL_NAME.to_string(),
                    bundle: "default".to_string(),
                    bundle_config_hash: "h1".to_string(),
                    loaded_models: Vec::new(),
                    models: Vec::new(),
                    gpus: Vec::new(),
                    queue_depth: None,
                    pending_cost: None,
                    inflight_batches: None,
                    memory_used_bytes: None,
                    memory_total_bytes: None,
                    saturated: false,
                    terminated: false,
                },
            )
            .await;

        let target = batch_publish_target(
            &state,
            DEFAULT_POOL_NAME,
            "tenant",
            "BAAI/bge-m3",
            "l4",
            "default",
            "h1",
        )
        .await
        .expect("assigned capped lane should have a direct target");

        match target {
            publisher::PublishTarget::Worker {
                pool,
                machine_profile,
                bundle,
                model,
                worker_id,
            } => {
                assert_eq!(pool, DEFAULT_POOL_NAME);
                assert_eq!(machine_profile, "l4");
                assert_eq!(bundle, "default");
                assert_eq!(model, "BAAI/bge-m3");
                assert_eq!(worker_id, "assigned-cold");
            }
            publisher::PublishTarget::Pool { .. } => {
                panic!("capped logical batch work must not publish to the shared pool subject")
            }
        }
    }

    #[tokio::test]
    async fn test_openai_provisioning_response_empty_gpu_is_retryable_503() {
        let resp = build_openai_provisioning_response("", "default");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get("retry-after").unwrap(),
            PROVISIONING_RETRY_AFTER
        );
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            err_code::PROVISIONING
        );
        assert!(resp.headers().get("x-sie-version").is_some());
        assert!(resp.headers().get("x-sie-server-version").is_some());

        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(value.get("detail").is_none());
        assert!(value.get("status").is_none());
        assert_eq!(value["error"]["type"], oai_type::SERVER_ERROR);
        assert_eq!(value["error"]["code"], oai_code::PROVISIONING);
        let message = value["error"]["message"].as_str().unwrap_or("");
        assert!(message.contains("bundle 'default'"));
        assert!(message.contains("Provisioning in progress"));
    }

    // ── is_valid_pool_name (pool subject-injection guard) ──────────

    #[test]
    fn test_is_valid_pool_name_accepts_allowlisted() {
        for ok in ["default", "eval-l4", "A-b_C-9", "x"] {
            assert!(is_valid_pool_name(ok), "{ok} should be valid");
        }
    }

    #[test]
    fn test_is_valid_pool_name_rejects_subject_injection() {
        // Any of these would re-tokenise / break the
        // `sie.work.{pool}.{machine_profile}.{bundle}.{model}` subject
        // or inject a wildcard.
        for bad in [
            "",          // empty
            "a b",       // whitespace
            "pool/evil", // slash (subject separator after normalize)
            "pool.1",    // dot (subject separator)
            "pool>evil", // `>` full-wildcard
            "pool*",     // `*` token-wildcard
            "a.>",       // wildcard tail
            "pool\nx",   // control char
            "pää",       // non-ascii
            "_default",  // removed legacy sentinel spelling
        ] {
            assert!(!is_valid_pool_name(bad), "{bad:?} should be rejected");
        }
        // Over-length is rejected too.
        assert!(!is_valid_pool_name(&"a".repeat(129)));
        assert!(is_valid_pool_name(&"a".repeat(128)));
    }

    #[tokio::test]
    async fn test_apply_model_pool_default() {
        let mut omitted = String::new();
        apply_model_pool_default(&mut omitted, Some("customer-a"), None)
            .await
            .unwrap();
        assert_eq!(omitted, "customer-a");

        let mut matching = "customer-a".to_string();
        apply_model_pool_default(&mut matching, Some("customer-a"), None)
            .await
            .unwrap();
        assert_eq!(matching, "customer-a");

        let mut mismatched = "default".to_string();
        assert!(
            apply_model_pool_default(&mut mismatched, Some("customer-a"), None)
                .await
                .is_err()
        );

        let mut unknown_omitted = String::new();
        apply_model_pool_default(&mut unknown_omitted, None, None)
            .await
            .unwrap();
        assert_eq!(unknown_omitted, "default");

        let mut unknown_explicit = "customer-a".to_string();
        apply_model_pool_default(&mut unknown_explicit, None, None)
            .await
            .unwrap();
        assert_eq!(unknown_explicit, "customer-a");

        let mut default_model_explicit_runtime_pool = "customer-a".to_string();
        apply_model_pool_default(
            &mut default_model_explicit_runtime_pool,
            Some("default"),
            None,
        )
        .await
        .unwrap();
        assert_eq!(default_model_explicit_runtime_pool, "customer-a");

        let mut default_model_omitted = String::new();
        apply_model_pool_default(&mut default_model_omitted, Some("default"), None)
            .await
            .unwrap();
        assert_eq!(default_model_omitted, "default");
    }

    #[tokio::test]
    async fn test_apply_model_pool_allows_logical_pool_backed_by_model_pool() {
        let pm = PoolManager::new(vec!["l4".to_string()]);
        sync_static_queue_pool(&pm, "customer-queue", "l4").await;
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4".to_string(), 1);
        pm.create_pool_with_caps_on_queue(
            "tenant-a",
            "customer-queue",
            gpus,
            std::collections::HashMap::new(),
            None,
            None,
            0,
            vec![],
        )
        .await
        .unwrap();

        let mut backed_logical_pool = "tenant-a".to_string();
        apply_model_pool_default(&mut backed_logical_pool, Some("customer-queue"), Some(&pm))
            .await
            .unwrap();
        assert_eq!(backed_logical_pool, "tenant-a");

        let mut unrelated_pool = "tenant-b".to_string();
        assert!(
            apply_model_pool_default(&mut unrelated_pool, Some("customer-queue"), Some(&pm))
                .await
                .is_err()
        );
    }

    #[test]
    fn test_decode_model_path_decodes_percent_encoded_slashes() {
        let model = decode_model_path("premium:%2FBAAI%2Fbge-m3").unwrap();
        assert_eq!(model, "premium:/BAAI/bge-m3");
    }

    #[test]
    fn test_decode_model_path_rejects_invalid_utf8() {
        let err = decode_model_path("BAAI%2Fbad%FFmodel").unwrap_err();
        assert!(err.contains("not valid UTF-8"));
    }

    // ── generate_params_from_json (walking-skeleton + grammar) ─────

    /// Helper: unwrap ``Ok(Some(...))`` from the grammar Result shape.
    fn _expect_generate_ok(body: &serde_json::Value) -> publisher::GenerateParams {
        match generate_params_from_json(body) {
            Ok(Some(p)) => p,
            Ok(None) => panic!("expected Some(params), got Ok(None)"),
            Err(_) => panic!("expected Ok(Some(params)), got Err(response)"),
        }
    }

    /// Helper: assert ``Ok(None)`` (= silently-bad request).
    fn _expect_generate_none(body: &serde_json::Value) {
        match generate_params_from_json(body) {
            Ok(None) => {}
            Ok(Some(_)) => panic!("expected Ok(None)"),
            Err(_) => panic!("expected Ok(None), got Err(response)"),
        }
    }

    /// Helper: drain the body of the 400 returned by an ``Err`` arm.
    async fn _expect_generate_err(body: &serde_json::Value) -> serde_json::Value {
        let resp = match generate_params_from_json(body) {
            Err(r) => r,
            Ok(_) => panic!("expected Err, got Ok"),
        };
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn test_generate_params_from_json_happy_path() {
        let body = serde_json::json!({
            "prompt": "Hello",
            "max_new_tokens": 32,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["</s>"],
        });
        let params = _expect_generate_ok(&body);
        match &params.input {
            publisher::GenerateInput::Prompt { prompt } => assert_eq!(prompt, "Hello"),
            publisher::GenerateInput::Messages { .. } => panic!("expected Prompt variant"),
        }
        assert_eq!(params.max_new_tokens, 32);
        assert_eq!(params.temperature, Some(0.7_f32));
        assert_eq!(params.top_p, Some(0.9_f32));
        assert_eq!(params.stop.as_deref(), Some(&["</s>".to_string()][..]));
        assert!(params.grammar.is_none());
    }

    #[test]
    fn test_generate_params_from_json_maps_native_images_to_worker_messages() {
        for stream in [false, true] {
            let body = serde_json::json!({
                "prompt": "Read the image",
                "images": [{"data": "aGVsbG8=", "format": "PNG"}],
                "max_new_tokens": 8,
                "stream": stream,
            });
            let params = _expect_generate_ok(&body);
            assert_eq!(params.stream, stream);
            assert!(generate_params_have_images(&params));
            let publisher::GenerateInput::Messages { messages } = params.input else {
                panic!("image-native generate must use the worker message path")
            };
            assert_eq!(messages.len(), 1);
            assert_eq!(messages[0].role, "user");
            assert_eq!(messages[0].content, "Read the image");
            let images = messages[0].images.as_ref().expect("one image");
            assert_eq!(images[0].data, "aGVsbG8=");
            assert_eq!(images[0].format.as_deref(), Some("png"));
        }
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_malformed_native_images() {
        let too_many = serde_json::Value::Array(vec![
            serde_json::json!({"data": "aGk="});
            MAX_GENERATE_IMAGES + 1
        ]);
        let cases = [
            (serde_json::json!([]), "images"),
            (serde_json::json!([{"data": "!!!"}]), "images[0].data"),
            (serde_json::json!([{"data": "aGk"}]), "images[0].data"),
            (serde_json::json!([{"data": "__8="}]), "images[0].data"),
            (serde_json::json!([{"data": "AB=="}]), "images[0].data"),
            (
                serde_json::json!([{"data": "aGVsbG8=", "url": "https://example.com/image.png"}]),
                "images[0].url",
            ),
            (
                serde_json::json!([{"data": "aGVsbG8=", "format": "png;bad"}]),
                "images[0].format",
            ),
            (too_many, "images"),
        ];
        for (images, expected_param) in cases {
            let body = serde_json::json!({
                "prompt": "Read",
                "images": images,
                "max_new_tokens": 8,
            });
            let error = _expect_generate_err(&body).await;
            assert_eq!(error["error"]["param"], expected_param);
        }
    }

    #[test]
    fn test_generate_params_from_rmpv_maps_native_images_to_worker_messages() {
        let body = serde_json::json!({
            "prompt": "Read",
            "images": [{"data": "aGVsbG8=", "format": "png"}],
            "max_new_tokens": 8,
        });
        let rmpv::Value::Map(map) = json_to_rmpv(body) else {
            panic!("object must map")
        };
        let params = generate_params_from_rmpv(&map)
            .expect("valid request")
            .expect("generate params");
        assert!(generate_params_have_images(&params));
    }

    #[test]
    fn test_generate_native_image_validation_has_json_msgpack_parity() {
        let body = serde_json::json!({
            "prompt": "Read",
            "images": [{"data": "aGk"}],
            "max_new_tokens": 8,
        });
        assert!(generate_params_from_json(&body).is_err());
        let rmpv::Value::Map(map) = json_to_rmpv(body) else {
            panic!("object must map")
        };
        assert!(generate_params_from_rmpv(&map).is_err());
    }

    #[test]
    fn test_generate_native_images_fail_closed_on_model_capability() {
        let image_body = serde_json::json!({
            "prompt": "Read",
            "images": [{"data": "aGk="}],
            "max_new_tokens": 8,
        });
        let image_params = _expect_generate_ok(&image_body);
        assert!(!native_generate_image_input_allowed(
            Some(&image_params),
            false
        ));
        assert!(native_generate_image_input_allowed(
            Some(&image_params),
            true
        ));

        let text_params = _expect_generate_ok(&serde_json::json!({
            "prompt": "Read",
            "max_new_tokens": 8,
        }));
        assert!(native_generate_image_input_allowed(
            Some(&text_params),
            false
        ));
    }

    #[test]
    fn test_generate_native_image_caps_fit_bounded_aggregate_body() {
        assert!(native_generate_image_encoded_size_allowed(
            MAX_GENERATE_IMAGE_BASE64_CHARS
        ));
        assert!(!native_generate_image_encoded_size_allowed(0));
        assert!(!native_generate_image_encoded_size_allowed(
            MAX_GENERATE_IMAGE_BASE64_CHARS + 1
        ));
    }

    #[test]
    fn test_generate_params_from_json_rejects_empty_prompt() {
        let body = serde_json::json!({"prompt": "", "max_new_tokens": 8});
        _expect_generate_none(&body);
    }

    /// BUG D regression (JSON): ``/v1/generate`` must accept a *string*
    /// ``stop`` (→ single-element vec), matching the chat path. The prior
    /// parser took ``as_array()`` only and silently dropped a string.
    #[test]
    fn test_generate_params_from_json_accepts_string_stop() {
        let body = serde_json::json!({
            "prompt": "Hi",
            "max_new_tokens": 8,
            "stop": "\n\n",
        });
        let params = _expect_generate_ok(&body);
        assert_eq!(params.stop.as_deref(), Some(&["\n\n".to_string()][..]));
    }

    /// BUG D regression (JSON): a non-string array entry must reject with
    /// a 400 (not silently drop the entry via ``filter_map``).
    #[tokio::test]
    async fn test_generate_params_from_json_rejects_non_string_stop_entry() {
        let body = serde_json::json!({
            "prompt": "Hi",
            "max_new_tokens": 8,
            "stop": [123],
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "stop");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    /// BUG D regression (msgpack): the ``/v1/generate`` msgpack twin must
    /// accept a string ``stop`` and reject a non-string array entry the
    /// same way as the JSON path.
    #[test]
    fn test_generate_params_from_rmpv_accepts_string_stop() {
        let body = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (rmpv::Value::from("stop"), rmpv::Value::from("\n\n")),
        ];
        let params = generate_params_from_rmpv(&body)
            .expect("rmpv ok")
            .expect("some params");
        assert_eq!(params.stop.as_deref(), Some(&["\n\n".to_string()][..]));
    }

    #[tokio::test]
    async fn test_generate_params_from_rmpv_rejects_non_string_stop_entry() {
        let body = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (
                rmpv::Value::from("stop"),
                rmpv::Value::Array(vec![rmpv::Value::from(123)]),
            ),
        ];
        let resp = generate_params_from_rmpv(&body).expect_err("expected 400");
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["param"], "stop");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_zero_max_new_tokens() {
        let body = serde_json::json!({"prompt": "x", "max_new_tokens": 0});
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "max_new_tokens");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_missing_max_new_tokens() {
        let body = serde_json::json!({"prompt": "x"});
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "max_new_tokens");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_non_integer_max_new_tokens() {
        let body = serde_json::json!({"prompt": "x", "max_new_tokens": "many"});
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "max_new_tokens");
    }

    // ── generate grammar parsing ───────────────────────────────────

    #[test]
    fn test_generate_params_from_json_accepts_json_schema_grammar() {
        let body = serde_json::json!({
            "prompt": "Hi",
            "max_new_tokens": 16,
            "grammar": {"json_schema": {"type": "object", "properties": {"x": {"type": "integer"}}}},
        });
        let params = _expect_generate_ok(&body);
        match params.grammar {
            Some(publisher::GrammarSpec::JsonSchema { .. }) => {}
            other => panic!("expected JsonSchema grammar, got {other:?}"),
        }
    }

    #[test]
    fn test_generate_params_from_json_accepts_regex_grammar() {
        let body = serde_json::json!({
            "prompt": "Hi",
            "max_new_tokens": 16,
            "grammar": {"regex": r"\d{3}-\d{4}"},
        });
        let params = _expect_generate_ok(&body);
        match params.grammar {
            Some(publisher::GrammarSpec::Regex { value, .. }) => {
                assert_eq!(value, r"\d{3}-\d{4}");
            }
            other => panic!("expected Regex grammar, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_grammar_mutex_violation() {
        let body = serde_json::json!({
            "prompt": "Hi",
            "max_new_tokens": 16,
            "grammar": {"json_schema": {"type": "object"}, "regex": "[A-Z]+"},
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "grammar");
    }

    #[test]
    fn test_generate_params_from_json_dereferences_grammar_dollar_ref() {
        let body = serde_json::json!({
            "prompt": "Hi",
            "max_new_tokens": 16,
            "grammar": {
                "json_schema": {
                    "$defs": {
                        "X": {"type": "object", "properties": {"x": {"type": "string"}}}
                    },
                    "properties": {"item": {"$ref": "#/$defs/X"}}
                }
            },
        });
        let params = _expect_generate_ok(&body);
        match params.grammar {
            Some(publisher::GrammarSpec::JsonSchema { value, .. }) => {
                assert_eq!(
                    value["properties"]["item"]["properties"]["x"]["type"],
                    "string"
                );
                let encoded = serde_json::to_string(&value).unwrap();
                assert!(
                    !encoded.contains("\"$ref\""),
                    "schema should be dereferenced: {encoded}"
                );
            }
            other => panic!("expected JsonSchema grammar, got {other:?}"),
        }
    }

    #[test]
    fn test_generate_params_from_rmpv_carries_grammar() {
        // Build a synthetic rmpv body that mirrors a JSON body
        // ``{prompt, max_new_tokens, grammar: {regex: "[a-z]+"}}``.
        let body = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (
                rmpv::Value::from("grammar"),
                rmpv::Value::Map(vec![(
                    rmpv::Value::from("regex"),
                    rmpv::Value::from("[a-z]+"),
                )]),
            ),
        ];
        let params = generate_params_from_rmpv(&body)
            .expect("rmpv ok")
            .expect("some params");
        match params.grammar {
            Some(publisher::GrammarSpec::Regex { value, .. }) => assert_eq!(value, "[a-z]+"),
            other => panic!("expected Regex grammar, got {other:?}"),
        }
    }

    // ── M8: /v1/generate field-surface parity with chat ───────────
    //
    // These tests guard the contract that ``/v1/generate`` now
    // accepts the same OpenAI sampler knobs the direct Python route
    // accepts (and that the chat route already accepts). Each newly-
    // accepted field gets a forward-path test (the value lands on
    // ``GenerateParams``) and a rejection test (type / range / cross-
    // field). See ADR-0001 for the canonicality argument.

    #[test]
    fn test_generate_params_from_json_forwards_seed() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "seed": 42_i64,
        });
        let params = _expect_generate_ok(&body);
        assert_eq!(params.seed, Some(42));
    }

    #[test]
    fn test_generate_params_from_json_forwards_negative_seed() {
        // Signed seeds must reach the worker unchanged. Reinterpreting ``-1``
        // as u64::MAX overflows pinned SGLang's torch.int64 seed tensor.
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "seed": -1_i64,
        });
        let params = _expect_generate_ok(&body);
        assert_eq!(params.seed, Some(-1));
    }

    #[test]
    fn test_parse_seed_field_preserves_signed_i64_boundaries() {
        for seed in [i64::MIN, -1, 0, i64::MAX] {
            let value = serde_json::json!(seed);
            assert_eq!(
                parse_seed_field(Some(&value)).expect("valid seed"),
                Some(seed)
            );
        }
    }

    #[test]
    fn test_parse_seed_field_rejects_values_outside_signed_i64() {
        let above_max = serde_json::json!(i64::MAX as u64 + 1);
        let below_min: serde_json::Value = serde_json::from_str("-9223372036854775809").unwrap();
        assert!(parse_seed_field(Some(&above_max)).is_err());
        assert!(parse_seed_field(Some(&below_min)).is_err());
    }

    #[test]
    fn test_parse_seed_field_rejects_non_integer_types() {
        for value in [
            serde_json::json!(true),
            serde_json::json!(1.5),
            serde_json::json!("1"),
            serde_json::json!({}),
        ] {
            assert!(parse_seed_field(Some(&value)).is_err());
        }
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_non_integer_seed() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "seed": "forty-two",
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "seed");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[test]
    fn test_generate_params_from_json_forwards_logit_bias() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "logit_bias": {"50256": -25.5, "1234": 10.0},
        });
        let params = _expect_generate_ok(&body);
        let bias = params.logit_bias.expect("logit_bias present");
        assert_eq!(bias.get("50256").copied(), Some(-25.5));
        assert_eq!(bias.get("1234").copied(), Some(10.0));
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_logit_bias_out_of_range() {
        // Mirrors chat: per-value cap of [-100, 100] keeps the sampler
        // safe from runaway biases (E's finding).
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "logit_bias": {"50256": 250.0},
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "logit_bias");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_logit_bias_non_object() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "logit_bias": "not-an-object",
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "logit_bias");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_logit_bias_non_integer_key() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "logit_bias": {"abc": 1.0},
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "logit_bias");
    }

    #[test]
    fn test_generate_params_from_json_forwards_logprobs_true() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "stream": true,
            "logprobs": true,
            "top_logprobs": 5,
        });
        let params = _expect_generate_ok(&body);
        assert_eq!(params.logprobs, Some(true));
        assert_eq!(params.top_logprobs, Some(5));
    }

    #[test]
    fn test_generate_params_from_json_accepts_logprobs_false_without_top_logprobs() {
        // Cross-field rule: ``logprobs: false`` with ``top_logprobs`` absent
        // is valid (OpenAI's default for chat).
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "stream": true,
            "logprobs": false,
        });
        let params = _expect_generate_ok(&body);
        assert_eq!(params.logprobs, Some(false));
        assert_eq!(params.top_logprobs, None);
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_logprobs_non_bool() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "stream": true,
            "logprobs": "yes",
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "logprobs");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_top_logprobs_out_of_range() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "stream": true,
            "logprobs": true,
            "top_logprobs": 50,
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "top_logprobs");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_top_logprobs_without_logprobs() {
        // Cross-field rule: ``top_logprobs > 0`` requires ``logprobs:
        // true`` (mirrors OpenAI).
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "stream": true,
            "top_logprobs": 3,
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "top_logprobs");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_top_logprobs_with_logprobs_false() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "stream": true,
            "logprobs": false,
            "top_logprobs": 3,
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "top_logprobs");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_lossy_native_fields() {
        for (field, value) in [
            ("n", serde_json::json!(2)),
            ("best_of", serde_json::json!(4)),
            ("stream_options", serde_json::json!({"include_usage": true})),
        ] {
            let mut body = serde_json::json!({
                "prompt": "x",
                "max_new_tokens": 4,
                "stream": true,
            });
            body[field] = value;
            let v = _expect_generate_err(&body).await;
            assert_eq!(v["error"]["param"], field);
            assert_eq!(v["error"]["code"], "unsupported_field");
        }
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_blocking_logprobs() {
        for field in ["logprobs", "top_logprobs"] {
            let mut body = serde_json::json!({
                "prompt": "x",
                "max_new_tokens": 4,
            });
            body[field] = if field == "logprobs" {
                serde_json::json!(true)
            } else {
                serde_json::json!(3)
            };
            let v = _expect_generate_err(&body).await;
            assert_eq!(v["error"]["param"], field);
            assert_eq!(v["error"]["code"], "unsupported_field");
        }
    }

    #[test]
    fn test_generate_params_from_json_forwards_lora_adapter() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "lora_adapter": "my-adapter",
        });
        let params = _expect_generate_ok(&body);
        assert_eq!(params.lora_adapter.as_deref(), Some("my-adapter"));
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_empty_lora_adapter() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "lora_adapter": "",
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "lora_adapter");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_lora_adapter_non_string() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "lora_adapter": 123,
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "lora_adapter");
    }

    #[test]
    fn test_generate_params_from_json_forwards_stream() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "stream": true,
        });
        let params = _expect_generate_ok(&body);
        assert!(params.stream);
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_stream_non_bool() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "stream": "yes",
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "stream");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    // ── M8: strict accept-list (A's discipline) ─────────────────────

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_unknown_field() {
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "totally_unknown_knob": 42,
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "totally_unknown_knob");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_chat_only_tools() {
        // Tool-calling stays chat-only; the direct Python route doesn't
        // accept them either and neither should we.
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "tools": [],
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "tools");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_generate_params_from_json_rejects_chat_only_top_k() {
        // ``top_k`` / ``repetition_penalty`` are chat-only today —
        // SIE-native /v1/generate stays a thin prompt wrapper per the
        // existing comment in :func:`generate_params_from_json`.
        let body = serde_json::json!({
            "prompt": "x",
            "max_new_tokens": 4,
            "top_k": 40,
        });
        let v = _expect_generate_err(&body).await;
        assert_eq!(v["error"]["param"], "top_k");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    // ── M8: rmpv twin parity ───────────────────────────────────────

    #[test]
    fn test_generate_params_from_rmpv_forwards_sampler_fields() {
        // Spot-check that the rmpv parser threads the same fields
        // through to ``GenerateParams`` so SDKs on either wire format
        // see identical worker payloads. Individual range/type rules
        // are validated above on the JSON twin via the shared helpers.
        let body = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (rmpv::Value::from("seed"), rmpv::Value::from(-7i64)),
            (rmpv::Value::from("stream"), rmpv::Value::Boolean(true)),
            (rmpv::Value::from("logprobs"), rmpv::Value::Boolean(true)),
            (rmpv::Value::from("top_logprobs"), rmpv::Value::from(3u32)),
            (
                rmpv::Value::from("lora_adapter"),
                rmpv::Value::from("my-adapter"),
            ),
        ];
        let params = generate_params_from_rmpv(&body)
            .expect("rmpv ok")
            .expect("some params");
        assert_eq!(params.seed, Some(-7));
        assert_eq!(params.logprobs, Some(true));
        assert_eq!(params.top_logprobs, Some(3));
        assert_eq!(params.n, None);
        assert_eq!(params.best_of, None);
        assert_eq!(params.lora_adapter.as_deref(), Some("my-adapter"));
    }

    #[test]
    fn test_generate_params_from_rmpv_rejects_non_signed_integer_seed() {
        for seed in [
            rmpv::Value::F64(f64::NAN),
            rmpv::Value::F64(f64::INFINITY),
            rmpv::Value::from(u64::MAX),
            rmpv::Value::Boolean(true),
            rmpv::Value::from("42"),
        ] {
            let body = vec![
                (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
                (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
                (rmpv::Value::from("seed"), seed),
            ];
            assert!(generate_params_from_rmpv(&body).is_err());
        }
    }

    #[tokio::test]
    async fn test_generate_params_from_rmpv_rejects_unknown_field() {
        let body = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (
                rmpv::Value::from("totally_unknown_knob"),
                rmpv::Value::from(42),
            ),
        ];
        let resp = generate_params_from_rmpv(&body).expect_err("expected 400");
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["param"], "totally_unknown_knob");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_generate_params_from_rmpv_rejects_lossy_native_fields() {
        for (field, value) in [
            ("n", rmpv::Value::from(2)),
            ("best_of", rmpv::Value::from(2)),
            (
                "stream_options",
                rmpv::Value::Map(vec![(
                    rmpv::Value::from("include_usage"),
                    rmpv::Value::Boolean(true),
                )]),
            ),
        ] {
            let body = vec![
                (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
                (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
                (rmpv::Value::from(field), value),
            ];
            let resp = generate_params_from_rmpv(&body).expect_err("expected 400");
            let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
                .await
                .unwrap();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(v["error"]["param"], field);
            assert_eq!(v["error"]["code"], "unsupported_field");
        }
    }

    #[tokio::test]
    async fn test_generate_params_from_rmpv_rejects_blocking_logprobs() {
        let body = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (rmpv::Value::from("logprobs"), rmpv::Value::Boolean(true)),
        ];
        let resp = generate_params_from_rmpv(&body).expect_err("expected 400");
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["param"], "logprobs");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_generate_params_from_rmpv_rejects_top_logprobs_without_logprobs() {
        let body = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (rmpv::Value::from("stream"), rmpv::Value::Boolean(true)),
            (rmpv::Value::from("top_logprobs"), rmpv::Value::from(3u32)),
        ];
        let resp = generate_params_from_rmpv(&body).expect_err("expected 400");
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["param"], "top_logprobs");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_generate_json_rejects_present_wrong_types() {
        for (field, value) in [
            ("temperature", serde_json::json!("0.7")),
            ("top_p", serde_json::json!(true)),
            ("routing_key", serde_json::json!(42)),
            ("prompt_cache_key", serde_json::json!([])),
            ("safety_identifier", serde_json::json!({})),
        ] {
            let mut body = serde_json::json!({"prompt": "hi", "max_new_tokens": 8});
            body[field] = value;
            let error = _expect_generate_err(&body).await;
            assert_eq!(error["error"]["param"], field);
            assert_eq!(error["error"]["code"], "invalid_request");
        }
    }

    #[test]
    fn test_generate_json_accepts_schema_nullable_fields() {
        let body = serde_json::json!({
            "prompt": "hi",
            "max_new_tokens": 8,
            "temperature": null,
            "top_p": null,
            "routing_key": null,
            "prompt_cache_key": null,
            "safety_identifier": null,
        });
        let work = work_params_from_json(&body, "generate").expect("valid work params");
        let generate = work.generate.expect("generate params");
        assert_eq!(generate.temperature, None);
        assert_eq!(generate.top_p, None);
        assert_eq!(work.routing_key, None);
        assert_eq!(work.prompt_cache_key, None);
    }

    #[test]
    fn test_generate_json_options_default_profile_is_consumed() {
        let body = serde_json::json!({
            "prompt": "hi",
            "max_new_tokens": 8,
            "options": {"profile": "default", "overall_timeout_s": 20},
        });
        let work = work_params_from_json(&body, "generate").expect("valid work params");
        assert_eq!(
            work.options,
            Some(serde_json::json!({"overall_timeout_s": 20}))
        );
    }

    #[tokio::test]
    async fn test_generate_json_options_fail_closed_before_dispatch() {
        for (options, parameter) in [
            (serde_json::json!({"unknown": true}), "options.unknown"),
            (
                serde_json::json!({"default_sampling": {"temperature": "0.7"}}),
                "options.default_sampling.temperature",
            ),
            (
                serde_json::json!({"overall_timeout_s": null}),
                "options.overall_timeout_s",
            ),
        ] {
            let body = serde_json::json!({"prompt": "hi", "max_new_tokens": 8, "options": options});
            let error = _expect_generate_err(&body).await;
            assert_eq!(error["error"]["param"], parameter);
            assert!(matches!(
                error["error"]["code"].as_str(),
                Some("invalid_request" | "unsupported_field")
            ));
        }
    }

    #[tokio::test]
    async fn test_generate_json_rejects_non_default_options_profile() {
        let body = serde_json::json!({
            "prompt": "hi",
            "max_new_tokens": 8,
            "options": {"profile": "fast"},
        });
        let QueueParseError::PreBuilt(response) =
            work_params_from_json(&body, "generate").unwrap_err()
        else {
            panic!("expected prebuilt response")
        };
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let error: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(error["error"]["param"], "options.profile");
        assert!(error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("model:profile"));
    }

    #[test]
    fn test_generate_rmpv_rejects_wrong_sampler_types() {
        for (field, value) in [
            ("temperature", rmpv::Value::from("0.7")),
            ("top_p", rmpv::Value::Boolean(true)),
        ] {
            let body = vec![
                (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
                (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
                (rmpv::Value::from(field), value),
            ];
            assert!(generate_params_from_rmpv(&body).is_err());
        }
    }

    #[test]
    fn test_generate_rmpv_accepts_schema_nullable_fields() {
        let body = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (rmpv::Value::from("temperature"), rmpv::Value::Nil),
            (rmpv::Value::from("top_p"), rmpv::Value::Nil),
            (rmpv::Value::from("routing_key"), rmpv::Value::Nil),
            (rmpv::Value::from("prompt_cache_key"), rmpv::Value::Nil),
            (rmpv::Value::from("safety_identifier"), rmpv::Value::Nil),
        ];
        let generate = generate_params_from_rmpv(&body)
            .expect("valid msgpack")
            .expect("generate params");
        assert_eq!(generate.temperature, None);
        assert_eq!(generate.top_p, None);
        assert_eq!(generate.routing_key, None);
        assert_eq!(generate.prompt_cache_key, None);
    }

    #[test]
    fn test_generate_rmpv_rejects_non_string_and_duplicate_keys() {
        let non_string = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (
                rmpv::Value::Binary(b"temperature".to_vec()),
                rmpv::Value::from(0.7),
            ),
        ];
        assert!(generate_params_from_rmpv(&non_string).is_err());

        let duplicate = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("prompt"), rmpv::Value::from("bye")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
        ];
        assert!(generate_params_from_rmpv(&duplicate).is_err());
    }

    #[test]
    fn test_generate_rmpv_rejects_nested_duplicate_keys_without_narrowing_schema_keys() {
        let duplicate_nested = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (
                rmpv::Value::from("options"),
                rmpv::Value::Map(vec![
                    (
                        rmpv::Value::from("overall_timeout_s"),
                        rmpv::Value::from(10),
                    ),
                    (
                        rmpv::Value::from("overall_timeout_s"),
                        rmpv::Value::from(20),
                    ),
                ]),
            ),
        ];
        assert!(generate_params_from_rmpv(&duplicate_nested).is_err());

        let arbitrary_schema_key = vec![
            (rmpv::Value::from("prompt"), rmpv::Value::from("hi")),
            (rmpv::Value::from("max_new_tokens"), rmpv::Value::from(8u32)),
            (
                rmpv::Value::from("grammar"),
                rmpv::Value::Map(vec![(
                    rmpv::Value::from("json_schema"),
                    rmpv::Value::Map(vec![
                        (rmpv::Value::from("type"), rmpv::Value::from("object")),
                        (
                            rmpv::Value::from("x-vendor-key"),
                            rmpv::Value::from("preserved"),
                        ),
                    ]),
                )]),
            ),
        ];
        assert!(generate_params_from_rmpv(&arbitrary_schema_key).is_ok());
    }
    // ── build_generate_success_body ───────────────────────────────

    #[test]
    fn test_build_generate_success_body_composes_envelope() {
        // Worker emits msgpack: {text, finish_reason, usage{...}}
        let worker_blob = rmp_serde::to_vec_named(&serde_json::json!({
            "text": "hello world",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }))
        .unwrap();
        let r = publisher::WorkResult {
            work_item_id: "req-1.0".to_string(),
            request_id: "req-1".to_string(),
            item_index: 0,
            success: true,
            result_msgpack: worker_blob,
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let body = build_generate_success_body("Qwen/Qwen3-4B-Instruct", &[&r], false);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["model"], "Qwen/Qwen3-4B-Instruct");
        assert_eq!(value["text"], "hello world");
        assert_eq!(value["finish_reason"], "stop");
        assert_eq!(value["usage"]["completion_tokens"], 2);
    }

    // ── env_seconds_or + timeout metric labels ─────────────────────

    #[test]
    fn test_env_seconds_or_uses_default_when_missing() {
        // Use a unique key per test run to avoid leaking state.
        let key = format!("SIE_TEST_NONEXISTENT_KEY_{}", std::process::id());
        assert_eq!(env_seconds_or(&key, 7.5), 7.5);
    }

    #[test]
    fn test_env_seconds_or_parses_positive_float() {
        let key = format!("SIE_TEST_PARSE_FLOAT_{}", std::process::id());
        // SAFETY: env mutation in unit tests is best-effort isolated;
        // the key is process-id-suffixed.
        std::env::set_var(&key, "12.25");
        assert_eq!(env_seconds_or(&key, 1.0), 12.25);
        std::env::remove_var(&key);
    }

    #[test]
    fn test_env_seconds_or_rejects_zero_and_negative() {
        let key = format!("SIE_TEST_ZERO_{}", std::process::id());
        std::env::set_var(&key, "0");
        assert_eq!(env_seconds_or(&key, 5.0), 5.0);
        std::env::set_var(&key, "-3");
        assert_eq!(env_seconds_or(&key, 5.0), 5.0);
        std::env::remove_var(&key);
    }

    // ── ADR-0003: generation timeout authority + invariant ────────

    #[test]
    fn test_enforce_first_chunk_invariant_passes_through_when_ok() {
        // The model-profile-typical shape: 60s first_chunk, 300s overall.
        // The invariant holds; values are returned unchanged.
        let (fc, ov) = enforce_first_chunk_invariant(60.0, 300.0, "Qwen/Qwen3.5-4B", "a100-40gb");
        assert_eq!(fc, 60.0);
        assert_eq!(ov, 300.0);
    }

    #[test]
    fn test_enforce_first_chunk_invariant_passes_through_when_equal() {
        // Edge of the invariant — equal is allowed (overall >= first_chunk).
        let (fc, ov) = enforce_first_chunk_invariant(30.0, 30.0, "m", "p");
        assert_eq!(fc, 30.0);
        assert_eq!(ov, 30.0);
    }

    #[test]
    fn test_enforce_first_chunk_invariant_clamps_when_violated() {
        // Misconfiguration: first_chunk=60s, overall=30s. The first-chunk
        // policy would never fire because overall expires sooner. We clamp
        // first_chunk down to overall so the request can still complete.
        let (fc, ov) = enforce_first_chunk_invariant(60.0, 30.0, "m", "p");
        assert_eq!(fc, 30.0);
        assert_eq!(ov, 30.0);
    }

    // H7 contract (ADR-0003): generation streaming does NOT clamp
    // `timeout_config.overall` with `state.config.request_timeout`. The
    // legacy ceiling continues to bound encode/score/extract, but the
    // two generation call-sites in `proxy.rs::handle_generation_streaming`
    // and `sse.rs::handle_sse_stream` feed `timeout_config.overall`
    // directly into the driver. A textual-contract test was tried here
    // but self-matched its own assertion strings; the contract is
    // enforced by ADR-0003, the invariant unit tests above, and the
    // gateway integration tests that exercise long-running generations.

    // ── StreamCancelGuard (streaming cancel-on-drop) ──────────────
    //
    // ── build_generate_success_body_v2 ────────────────────────────

    #[test]
    fn test_build_generate_success_body_v2_aggregates_outcome() {
        let outcome = crate::queue::streaming::StreamOutcome {
            text: "Hello world!".to_string(),
            finish_reason: "stop".to_string(),
            usage: Some(crate::queue::streaming::UsageBlock {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            }),
            attempt_id: "att-abc".to_string(),
            ttft_ms: Some(120.5),
            tpot_ms: Some(45.2),
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let body = build_generate_success_body_v2("Qwen/Qwen3-4B-Instruct", &outcome, false);
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["model"], "Qwen/Qwen3-4B-Instruct");
        assert_eq!(value["text"], "Hello world!");
        assert_eq!(value["finish_reason"], "stop");
        assert_eq!(value["usage"]["prompt_tokens"], 5);
        assert_eq!(value["usage"]["completion_tokens"], 3);
        assert_eq!(value["usage"]["total_tokens"], 8);
        assert_eq!(value["attempt_id"], "att-abc");
        assert_eq!(value["ttft_ms"], 120.5);
        assert_eq!(value["tpot_ms"], 45.2);
    }

    // ── build_queue_result_timeout_response ──────────────────────
    //
    // A gateway-side result wait timeout is distinct from worker-emitted
    // MODEL_LOADING. Keep the signal explicit so load tests and dashboards do
    // not misread queue/head-of-line stalls as cold model loads.

    #[test]
    fn test_build_queue_result_timeout_response_is_504() {
        let resp = build_queue_result_timeout_response("BAAI/bge-m3", 30.0);
        assert_eq!(resp.status(), StatusCode::GATEWAY_TIMEOUT);
    }

    #[test]
    fn test_build_queue_result_timeout_response_sets_retry_after_and_version_headers() {
        let resp = build_queue_result_timeout_response("BAAI/bge-m3", 30.0);
        let headers = resp.headers();
        assert_eq!(
            headers.get("retry-after").unwrap(),
            GATEWAY_TIMEOUT_RETRY_AFTER
        );
        assert!(headers.get("x-sie-version").is_some());
        assert!(headers.get("x-sie-server-version").is_some());
    }

    #[test]
    fn test_build_queue_result_timeout_response_sets_error_code_header() {
        let resp = build_queue_result_timeout_response("BAAI/bge-m3", 30.0);
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            err_code::GATEWAY_TIMEOUT
        );
    }

    #[tokio::test]
    async fn test_build_queue_result_timeout_response_body_is_gateway_detail_contract() {
        let resp = build_queue_result_timeout_response("BAAI/bge-m3", 30.0);
        let body_bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("response body is valid JSON");
        assert_eq!(body["detail"]["code"], err_code::GATEWAY_TIMEOUT);
        assert!(body.get("error").is_none());
        let msg = body["detail"]["message"].as_str().unwrap_or("");
        assert!(
            msg.contains("BAAI/bge-m3"),
            "message references the model id: {msg}"
        );
        assert!(
            msg.contains("30"),
            "message references the timeout value: {msg}"
        );
    }

    #[tokio::test]
    async fn test_build_model_load_failed_response_uses_legacy_error_envelope_and_headers() {
        let resp = build_model_load_failed_response();
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            MODEL_LOAD_FAILED_ERROR_CODE
        );
        assert!(resp.headers().get("x-sie-error-version").is_some());

        let body_bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("response body is valid JSON");
        assert_eq!(body["error"]["code"], MODEL_LOAD_FAILED_ERROR_CODE);
        assert_eq!(body["error"]["message"], MODEL_LOAD_FAILED_PUBLIC_MESSAGE);
        assert!(body.get("detail").is_none());
    }

    // ── /v1/embeddings error-envelope unification (roadmap §3 item 1.4) ──
    //
    // The OpenAI-shaped embeddings surface must re-surface inner
    // `/v1/encode` failures in the `{error:{…}}` envelope (not the
    // SIE-native `{detail:{…}}`), preserving status + `Retry-After`.

    #[tokio::test]
    async fn test_translate_inner_compat_error_detail_to_openai_envelope() {
        let mut resp = (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json_detail(err_code::QUEUE_UNAVAILABLE, "pool saturated")),
        )
            .into_response();
        resp.headers_mut()
            .insert("retry-after", HeaderValue::from_static("5"));
        let out = translate_inner_compat_error(resp).await;
        assert_eq!(out.status(), StatusCode::SERVICE_UNAVAILABLE);
        // Retry-After survives the rewrite so SDK auto-retry still fires.
        assert_eq!(out.headers().get("retry-after").unwrap(), "5");
        let body_bytes = axum::body::to_bytes(out.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).expect("valid JSON");
        assert!(body.get("detail").is_none(), "no leaked detail shape");
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(body["error"]["code"], "transport_failure");
        assert_eq!(body["error"]["message"], "pool saturated");
    }

    #[tokio::test]
    async fn test_translate_inner_compat_pool_capacity_error_to_openai_envelope() {
        let resp = (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json_detail(
                err_code::POOL_CAPACITY_UNAVAILABLE,
                "pool admits zero workers",
            )),
        )
            .into_response();
        let out = translate_inner_compat_error(resp).await;
        assert_eq!(out.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            out.headers().get("retry-after").is_none(),
            "hard cap-zero is not retryable provisioning"
        );
        assert_eq!(
            out.headers().get("x-sie-error-code").unwrap(),
            err_code::POOL_CAPACITY_UNAVAILABLE
        );

        let body_bytes = axum::body::to_bytes(out.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).expect("valid JSON");
        assert!(body.get("detail").is_none(), "no leaked detail shape");
        assert_eq!(body["error"]["type"], oai_type::SERVER_ERROR);
        assert_eq!(body["error"]["code"], oai_code::TRANSPORT_FAILURE);
        assert_eq!(body["error"]["message"], "pool admits zero workers");
    }

    #[tokio::test]
    async fn test_translate_inner_compat_provisioning_error_to_openai_503() {
        let mut resp = (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": {
                    "code": err_code::PROVISIONING,
                    "message": "No worker available for GPU type 'l4'. Provisioning in progress.",
                },
            })),
        )
            .into_response();
        resp.headers_mut().insert(
            "retry-after",
            HeaderValue::from_static(PROVISIONING_RETRY_AFTER),
        );

        let out = translate_inner_compat_error(resp).await;
        assert_eq!(out.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            out.headers().get("retry-after").unwrap(),
            PROVISIONING_RETRY_AFTER
        );
        assert_eq!(
            out.headers().get("x-sie-error-code").unwrap(),
            err_code::PROVISIONING
        );

        let body_bytes = axum::body::to_bytes(out.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).expect("valid JSON");
        assert!(body.get("detail").is_none(), "no leaked detail shape");
        assert_eq!(body["error"]["type"], oai_type::SERVER_ERROR);
        assert_eq!(body["error"]["code"], oai_code::PROVISIONING);
        assert!(body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("l4"));
    }

    #[tokio::test]
    async fn test_translate_inner_compat_error_falls_back_to_status_for_error_shape() {
        // 502/503 SDK-stable bodies are `{error:{…}}` with no `detail.code`;
        // the HTTP status drives classification and the message is preserved.
        let resp = (
            StatusCode::NOT_FOUND,
            Json(json!({"error": {"message": "model 'x' not found"}})),
        )
            .into_response();
        let out = translate_inner_compat_error(resp).await;
        assert_eq!(out.status(), StatusCode::NOT_FOUND);
        let body_bytes = axum::body::to_bytes(out.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).expect("valid JSON");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["code"], "model_not_found");
        assert_eq!(body["error"]["message"], "model 'x' not found");
    }

    #[test]
    fn test_sie_code_from_status_maps_error_statuses() {
        assert_eq!(
            sie_code_from_status(StatusCode::BAD_REQUEST),
            "INVALID_REQUEST"
        );
        assert_eq!(
            sie_code_from_status(StatusCode::NOT_FOUND),
            "MODEL_NOT_FOUND"
        );
        assert_eq!(
            sie_code_from_status(StatusCode::PAYLOAD_TOO_LARGE),
            "PAYLOAD_TOO_LARGE"
        );
        assert_eq!(
            sie_code_from_status(StatusCode::TOO_MANY_REQUESTS),
            "RATE_LIMIT"
        );
        assert_eq!(
            sie_code_from_status(StatusCode::SERVICE_UNAVAILABLE),
            "QUEUE_UNAVAILABLE"
        );
        assert_eq!(
            sie_code_from_status(StatusCode::GATEWAY_TIMEOUT),
            "GATEWAY_TIMEOUT"
        );
        assert_eq!(
            sie_code_from_status(StatusCode::INTERNAL_SERVER_ERROR),
            "INTERNAL_SERVER_ERROR"
        );
    }

    // ── retryable error code translation ───────────────────────────
    //
    // When every item in a batch fails with the *same* retryable code
    // (RESOURCE_EXHAUSTED from worker-side OOM recovery exhaustion, or
    // MODEL_LOADING from a worker still warming up), the gateway emits
    // a 503 with the SDK-expected body / headers so auto-retry kicks
    // in. Mixed batches keep going through the legacy 500
    // `all_items_failed` path so callers can inspect per-item codes.

    fn _err_result(code: Option<&str>, msg: &str) -> publisher::WorkResult {
        publisher::WorkResult {
            work_item_id: "req.0".to_string(),
            request_id: "req".to_string(),
            item_index: 0,
            success: false,
            result_msgpack: Vec::new(),
            error: Some(msg.to_string()),
            error_code: code.map(str::to_string),
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        }
    }

    #[tokio::test]
    async fn test_managed_invalid_request_maps_to_native_extract_400_envelope() {
        let result = _err_result(Some("invalid_request"), "unsupported audio container");
        let errors = vec![&result];
        let code = unanimous_client_error_code(&errors).expect("client error should classify");
        assert_eq!(code, "invalid_request");
        let response = endpoint_error_response(
            "extract",
            StatusCode::BAD_REQUEST,
            err_code::INVALID_REQUEST,
            oai_type::INVALID_REQUEST,
            oai_code::INVALID_REQUEST,
            None,
            result.error.as_deref().unwrap(),
        );
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["detail"]["code"], err_code::INVALID_REQUEST);
        assert_eq!(body["detail"]["message"], "unsupported audio container");
    }

    fn _ok_result(item_index: u32, result_msgpack: Vec<u8>) -> publisher::WorkResult {
        publisher::WorkResult {
            work_item_id: format!("req.{item_index}"),
            request_id: "req".to_string(),
            item_index,
            success: true,
            result_msgpack,
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        }
    }

    #[test]
    fn test_result_payload_too_large_is_detected_among_successful_siblings() {
        let ok = _ok_result(0, vec![0x80]);
        let oversized = _err_result(Some("PAYLOAD_TOO_LARGE"), "large");
        let other = _err_result(Some("inference_error"), "other");

        assert_eq!(
            result_payload_too_large_error(&[ok.clone(), oversized.clone()])
                .and_then(|result| result.error_code.as_deref()),
            Some("PAYLOAD_TOO_LARGE")
        );
        assert!(result_payload_too_large_error(&[ok, other]).is_none());
        assert!(result_payload_too_large_error(&[]).is_none());
    }

    #[tokio::test]
    async fn test_result_payload_too_large_response_is_typed_413() {
        let resp = build_result_payload_too_large_response(
            "encode",
            "Encoded result exceeds the transport limit",
        );
        assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            PAYLOAD_TOO_LARGE_ERROR_CODE
        );
        assert!(resp.headers().get("x-sie-version").is_some());
        assert!(resp.headers().get("x-sie-server-version").is_some());

        let body_bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["detail"]["code"], "PAYLOAD_TOO_LARGE");
        assert!(body["detail"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("transport limit"));
    }

    #[test]
    fn test_result_transport_failure_is_detected_among_successful_siblings() {
        let ok = _ok_result(0, vec![0x80]);
        let transport = _err_result(Some("transport_failure"), "untrusted detail");
        let other = _err_result(Some("inference_error"), "other");

        assert_eq!(
            result_transport_failure_error(&[ok.clone(), transport])
                .and_then(|result| result.error_code.as_deref()),
            Some("transport_failure")
        );
        assert!(result_transport_failure_error(&[ok, other]).is_none());
        assert!(result_transport_failure_error(&[]).is_none());
    }

    #[tokio::test]
    async fn test_result_transport_failure_response_is_static_typed_503_without_retry_hint() {
        let resp = build_result_transport_failure_response("encode");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            oai_code::TRANSPORT_FAILURE
        );
        assert!(resp.headers().get("retry-after").is_none());
        assert!(resp.headers().get("x-sie-version").is_some());
        assert!(resp.headers().get("x-sie-server-version").is_some());

        let body_bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap();
        assert_eq!(body["detail"]["code"], "transport_failure");
        assert_eq!(
            body["detail"]["message"],
            "Worker result transport validation failed"
        );
    }

    #[test]
    fn test_unanimous_retryable_error_code_resource_exhausted() {
        let r1 = _err_result(Some("RESOURCE_EXHAUSTED"), "oom 1");
        let r2 = _err_result(Some("RESOURCE_EXHAUSTED"), "oom 2");
        let errors: Vec<&publisher::WorkResult> = vec![&r1, &r2];
        assert_eq!(
            unanimous_retryable_error_code(&errors),
            Some(RESOURCE_EXHAUSTED_ERROR_CODE)
        );
    }

    #[test]
    fn test_unanimous_retryable_error_code_model_loading() {
        let r1 = _err_result(Some("MODEL_LOADING"), "loading");
        let errors: Vec<&publisher::WorkResult> = vec![&r1];
        assert_eq!(
            unanimous_retryable_error_code(&errors),
            Some(MODEL_LOADING_ERROR_CODE)
        );
    }

    #[test]
    fn test_unanimous_retryable_error_code_lora_loading() {
        // LoRA-load-on-demand is also SDK-retryable; gateway must
        // translate it the same way as MODEL_LOADING.
        let r1 = _err_result(Some("LORA_LOADING"), "loading lora adapter");
        let errors: Vec<&publisher::WorkResult> = vec![&r1];
        assert_eq!(
            unanimous_retryable_error_code(&errors),
            Some(LORA_LOADING_ERROR_CODE)
        );
    }

    #[test]
    fn test_build_retryable_error_response_lora_loading_status_and_headers() {
        let resp =
            build_retryable_error_response(LORA_LOADING_ERROR_CODE, "Loading lora adapter 'foo'");
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let headers = resp.headers();
        assert_eq!(
            headers.get("retry-after").unwrap(),
            LORA_LOADING_RETRY_AFTER
        );
        assert_eq!(
            headers.get("x-sie-error-code").unwrap(),
            LORA_LOADING_ERROR_CODE
        );
    }

    #[test]
    fn test_unanimous_retryable_error_code_mixed_batch_returns_none() {
        // Mixed batches (one OOM, one inference_error) must NOT be
        // collapsed into a 503 — caller needs to see per-item details.
        let r1 = _err_result(Some("RESOURCE_EXHAUSTED"), "oom");
        let r2 = _err_result(Some("inference_error"), "shape mismatch");
        let errors: Vec<&publisher::WorkResult> = vec![&r1, &r2];
        assert_eq!(unanimous_retryable_error_code(&errors), None);
    }

    #[test]
    fn test_unanimous_retryable_error_code_unknown_code_returns_none() {
        // An unrecognized code — even if unanimous — does NOT trigger
        // 503 retry; we only opt-in known retryable codes.
        let r1 = _err_result(Some("CUSTOM_BACKEND_ERROR"), "x");
        let r2 = _err_result(Some("CUSTOM_BACKEND_ERROR"), "y");
        let errors: Vec<&publisher::WorkResult> = vec![&r1, &r2];
        assert_eq!(unanimous_retryable_error_code(&errors), None);
    }

    #[test]
    fn test_unanimous_retryable_error_code_missing_code_returns_none() {
        let r1 = _err_result(None, "no code");
        let errors: Vec<&publisher::WorkResult> = vec![&r1];
        assert_eq!(unanimous_retryable_error_code(&errors), None);
    }

    #[test]
    fn test_unanimous_terminal_client_errors_map_to_400_and_413() {
        for (code, status) in [
            (INVALID_INPUT_ERROR_CODE, StatusCode::BAD_REQUEST),
            (PAYLOAD_TOO_LARGE_ERROR_CODE, StatusCode::PAYLOAD_TOO_LARGE),
        ] {
            let first = _err_result(Some(code), "rejected 1");
            let second = _err_result(Some(code), "rejected 2");
            let errors = vec![&first, &second];
            assert_eq!(
                unanimous_terminal_client_error(&errors),
                Some((status, code))
            );
        }

        let invalid = _err_result(Some(INVALID_INPUT_ERROR_CODE), "invalid");
        let oversized = _err_result(Some(PAYLOAD_TOO_LARGE_ERROR_CODE), "large");
        assert_eq!(
            unanimous_terminal_client_error(&[&invalid, &oversized]),
            None
        );
    }

    #[test]
    fn test_unanimous_invalid_input_requires_every_worker_result_to_match() {
        let first = _err_result(Some(INVALID_INPUT_ERROR_CODE), "bad media");
        let second = _err_result(Some(INVALID_INPUT_ERROR_CODE), "bad document");
        let inference = _err_result(Some("inference_error"), "backend failure");

        assert_eq!(
            unanimous_worker_error_message(&[&first, &second], INVALID_INPUT_ERROR_CODE),
            Some("bad media".to_string())
        );
        assert_eq!(
            unanimous_worker_error_message(&[&first, &inference], INVALID_INPUT_ERROR_CODE),
            None
        );
    }

    #[tokio::test]
    async fn test_terminal_client_error_response_preserves_native_contract() {
        let response = build_terminal_client_error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            PAYLOAD_TOO_LARGE_ERROR_CODE,
            "payload exceeds limit",
        );
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            response.headers().get("x-sie-error-code").unwrap(),
            PAYLOAD_TOO_LARGE_ERROR_CODE
        );
        let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["detail"]["code"], PAYLOAD_TOO_LARGE_ERROR_CODE);
        assert_eq!(value["detail"]["message"], "payload exceeds limit");
    }

    #[tokio::test]
    async fn test_invalid_input_is_native_400_and_cohere_error_envelope() {
        let native = build_invalid_input_response("unsupported image payload");
        assert_eq!(native.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            native.headers().get("x-sie-error-code").unwrap(),
            INVALID_INPUT_ERROR_CODE
        );
        let native_body = axum::body::to_bytes(native.into_body(), 16 * 1024)
            .await
            .unwrap();
        let native_json: Value = serde_json::from_slice(&native_body).unwrap();
        assert_eq!(native_json["detail"]["code"], INVALID_INPUT_ERROR_CODE);
        assert_eq!(
            native_json["detail"]["message"],
            "unsupported image payload"
        );

        let compat =
            translate_inner_rerank_error(build_invalid_input_response("unsupported image payload"))
                .await;
        assert_eq!(compat.status(), StatusCode::BAD_REQUEST);
        assert_eq!(
            compat.headers().get("x-sie-error-code").unwrap(),
            INVALID_INPUT_ERROR_CODE
        );
        let compat_body = axum::body::to_bytes(compat.into_body(), 16 * 1024)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<Value>(&compat_body).unwrap(),
            json!({"message": "unsupported image payload"})
        );
    }

    #[test]
    fn test_queue_success_body_json_omits_partial_error_envelope() {
        let payload = rmp_serde::to_vec(&json!({"result": "ok"})).unwrap();
        let ok = _ok_result(0, payload);
        let body = build_queue_success_body("encode", "model-a", &[&ok], false);
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["model"], "model-a");
        assert_eq!(parsed["items"][0]["result"], "ok");
        assert!(parsed.get("errors").is_none());
    }

    #[test]
    fn test_queue_success_body_preserves_partial_extract_data_and_error() {
        let payload = rmp_serde::to_vec(&json!({
            "id": "doc-1",
            "data": {"partial": true},
            "error": {"code": "INFERENCE_ERROR", "message": "export failed"}
        }))
        .unwrap();
        let result = _ok_result(0, payload);
        let body = build_queue_success_body("extract", "docling", &[&result], false);
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["items"][0]["data"], json!({"partial": true}));
        assert_eq!(parsed["items"][0]["error"]["code"], "INFERENCE_ERROR");
        assert_eq!(parsed["items"][0]["error"]["message"], "export failed");
    }

    #[test]
    fn test_queue_success_body_json_preserves_decode_failure_item() {
        let bad = _ok_result(0, vec![0xc1]);
        let body = build_queue_success_body("encode", "model-a", &[&bad], false);
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["model"], "model-a");
        let items = parsed["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["item_index"], 0);
        assert_eq!(items[0]["work_item_id"], "req.0");
        assert_eq!(items[0]["error"]["code"], "RESULT_DECODE_FAILED");
        assert!(items[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("failed to decode result_msgpack"));
    }

    #[test]
    fn test_queue_success_body_msgpack_has_server_envelope_only() {
        let payload = rmp_serde::to_vec(&json!({"result": "ok"})).unwrap();
        let ok = _ok_result(0, payload);
        let body = build_queue_success_body("encode", "model-a", &[&ok], true);
        let parsed: serde_json::Value = rmp_serde::from_slice(&body).unwrap();

        assert_eq!(parsed["model"], "model-a");
        assert_eq!(parsed["items"][0]["result"], "ok");
        assert_eq!(parsed.as_object().unwrap().len(), 2);
    }

    #[test]
    fn test_queue_success_body_msgpack_trusts_worker_item_bytes() {
        let bad = _ok_result(0, vec![0xc1]);
        let body = build_queue_success_body("encode", "model-a", &[&bad], true);

        assert!(rmp_serde::from_slice::<serde_json::Value>(&body).is_err());
        assert_eq!(body.last().copied(), Some(0xc1));
    }

    #[test]
    fn test_queue_success_body_score_msgpack_trusts_worker_item_bytes() {
        let bad = _ok_result(0, vec![0xc1]);
        let body = build_queue_success_body("score", "model-a", &[&bad], true);

        assert!(rmp_serde::from_slice::<serde_json::Value>(&body).is_err());
        assert_eq!(body.last().copied(), Some(0xc1));
    }

    #[test]
    fn test_openai_embeddings_reject_token_id_array() {
        let err = openai_embedding_input_to_texts(&json!([10, 20, 30])).unwrap_err();

        assert!(err.contains("token-array embeddings input is not supported"));
    }

    #[test]
    fn test_openai_embeddings_reject_nested_token_arrays() {
        let err = openai_embedding_input_to_texts(&json!([[10, 20, 30], [40, 50]])).unwrap_err();

        assert!(err.contains("token-array embeddings input is not supported"));
    }

    #[test]
    fn test_openai_embeddings_item_cap_accepts_256_and_rejects_257() {
        let accepted = json!(vec!["text"; MAX_EMBEDDING_INPUTS]);
        let normalized =
            openai_embedding_input_to_texts(&accepted).expect("256 inputs are accepted");
        assert_eq!(normalized.texts.len(), MAX_EMBEDDING_INPUTS);

        let rejected = json!(vec!["text"; MAX_EMBEDDING_INPUTS + 1]);
        let error =
            openai_embedding_input_to_texts(&rejected).expect_err("257 inputs are rejected");
        assert!(error.contains("at most 256 texts"));
        assert!(error.contains("split the request"));
    }

    #[test]
    fn test_openai_embeddings_forwarded_headers_include_encode_timings() {
        for name in [
            "x-sie-request-id",
            "x-sie-version",
            "x-sie-server-version",
            "x-sie-worker",
            "x-queue-publish-time",
            "x-queue-wait-time",
            "x-queue-time",
            "x-sie-execution-identity-sha256",
            "x-sie-execution-binding-sha256",
            "x-inference-time",
            "x-tokenization-time",
            "x-postprocessing-time",
            "x-payload-fetch-time",
        ] {
            assert!(
                is_openai_compat_forwarded_header(name),
                "{name} should be forwarded from /v1/encode to /v1/embeddings"
            );
        }
        assert!(is_openai_compat_forwarded_header("X-SIE-WORKER"));
        assert!(!is_openai_compat_forwarded_header("content-type"));
        assert!(!is_openai_compat_forwarded_header("x-sie-error-code"));
    }

    #[test]
    fn test_openai_compat_model_ids_use_config_service_grammar() {
        for model in [
            "openai/whisper-large-v3-turbo",
            "jinaai/jina-embeddings-v3",
            "model_1.0",
        ] {
            assert!(is_valid_compat_model_id(model), "{model} should be valid");
        }
        for model in [
            "/leading",
            "two..dots",
            "back\\slash",
            "query?x=1",
            "fragment#x",
            "unicode-model-模型",
        ] {
            assert!(
                !is_valid_compat_model_id(model),
                "{model} should be invalid"
            );
        }
    }

    #[test]
    fn test_openai_embeddings_inner_request_forwards_trace_context() {
        // Auth/routing hints plus the W3C trace headers must reach the
        // internal /v1/encode request so the worker span continues the
        // client's trace (otherwise embeddings is the lone disconnected
        // path — every other endpoint injects queue trace context).
        for name in [
            "authorization",
            "x-sie-machine-profile",
            "x-sie-pool",
            "x-sie-engine",
            "x-sie-sdk-version",
            "traceparent",
            "tracestate",
        ] {
            assert!(
                is_openai_compat_inner_request_header(name),
                "{name} should be forwarded onto the internal /v1/encode request"
            );
        }
        assert!(is_openai_compat_inner_request_header("TraceParent"));
        assert!(!is_openai_compat_inner_request_header("cookie"));
        assert!(!is_openai_compat_inner_request_header("x-sie-request-id"));
    }

    #[test]
    fn test_openai_embedding_value_base64_is_little_endian_f32() {
        let encoded = openai_embedding_value(vec![1.0, -2.0], "base64");
        let Value::String(encoded) = encoded else {
            panic!("expected base64 string");
        };

        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        assert_eq!(bytes, vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0xc0]);
    }

    #[test]
    fn test_openai_embedding_items_to_data_rejects_partial_encode_response() {
        let items = vec![json!({"dense": [1.0, 2.0]})];
        let err = openai_embedding_items_to_data(&items, 2, "float").unwrap_err();
        assert!(err.contains("expected 2, got 1"));
    }

    #[test]
    fn test_openai_embedding_items_to_data_rejects_missing_dense_vector() {
        let items = vec![json!({"sparse": {"indices": [1], "values": [0.5]}})];
        let err = openai_embedding_items_to_data(&items, 1, "float").unwrap_err();
        assert_eq!(err, "item 0 missing dense embedding");
    }

    #[test]
    fn test_openai_embedding_items_to_data_rejects_non_numeric_dense_array() {
        let items = vec![json!({"dense": [1.0, "bad", 3.0]})];
        let err = openai_embedding_items_to_data(&items, 1, "float").unwrap_err();
        assert_eq!(err, "item 0 missing dense embedding");
    }

    #[test]
    fn test_openai_embedding_items_to_data_rejects_non_numeric_dense_values() {
        let items = vec![json!({"dense": {"values": [1.0, false, 3.0]}})];
        let err = openai_embedding_items_to_data(&items, 1, "float").unwrap_err();
        assert_eq!(err, "item 0 missing dense embedding");
    }

    #[test]
    fn test_build_retryable_error_response_resource_exhausted_status_and_headers() {
        let resp = build_retryable_error_response(
            RESOURCE_EXHAUSTED_ERROR_CODE,
            "CUDA out of memory after recovery",
        );
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let headers = resp.headers();
        assert_eq!(
            headers.get("retry-after").unwrap(),
            RESOURCE_EXHAUSTED_RETRY_AFTER
        );
        assert_eq!(
            headers.get("x-sie-error-code").unwrap(),
            RESOURCE_EXHAUSTED_ERROR_CODE
        );
        assert!(headers.get("x-sie-version").is_some());
        assert!(headers.get("x-sie-server-version").is_some());
    }

    #[tokio::test]
    async fn test_build_retryable_error_response_resource_exhausted_body() {
        // SDK reads `error.code` to decide whether to retry; body must
        // carry the structured envelope.
        let resp = build_retryable_error_response(
            RESOURCE_EXHAUSTED_ERROR_CODE,
            "CUDA out of memory after recovery",
        );
        let body_bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        let body: serde_json::Value =
            serde_json::from_slice(&body_bytes).expect("response body is valid JSON");
        assert_eq!(body["error"]["code"], RESOURCE_EXHAUSTED_ERROR_CODE);
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("out of memory"),
            "message preserves upstream error text: {body}"
        );
    }

    #[test]
    fn test_batch_direct_fallback_delay_leaves_request_budget() {
        assert_eq!(batch_direct_fallback_delay(0.5), Duration::from_secs(1));
        assert_eq!(batch_direct_fallback_delay(30.0), Duration::from_secs(10));
        assert_eq!(batch_direct_fallback_delay(300.0), Duration::from_secs(10));
        assert_eq!(
            batch_direct_fallback_delay(f64::NAN),
            Duration::from_secs(10)
        );
    }

    #[test]
    fn test_parse_model_spec_with_bundle() {
        let (bundle, model) = parse_model_spec("premium:/BAAI/bge-m3");
        assert_eq!(bundle, "premium");
        assert_eq!(model, "BAAI/bge-m3");
    }

    #[test]
    fn test_parse_model_spec_plain_name() {
        let (bundle, model) = parse_model_spec("my-model");
        assert_eq!(bundle, "");
        assert_eq!(model, "my-model");
    }

    #[test]
    fn test_parse_model_spec_empty() {
        let (bundle, model) = parse_model_spec("");
        assert_eq!(bundle, "");
        assert_eq!(model, "");
    }

    // ── check_sdk_version cache ────────────────────────────────────

    #[test]
    fn test_check_sdk_version_caches_parsed_minor() {
        // Use a deliberately unique version string so concurrent
        // tests don't fight over the cache entry.
        let version = "99.42.7-test-parsed";
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-sie-sdk-version",
            HeaderValue::from_static("99.42.7-test-parsed"),
        );

        assert!(SDK_VERSION_CACHE.get(version).is_none());
        check_sdk_version(&headers);

        let cached = SDK_VERSION_CACHE
            .get(version)
            .map(|v| *v)
            .expect("cache entry should be populated after first hit");
        assert_eq!(cached, Some(42));
    }

    #[test]
    fn test_check_sdk_version_caches_unparseable_as_none() {
        let version = "garbage-no-dots-test";
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-sie-sdk-version",
            HeaderValue::from_static("garbage-no-dots-test"),
        );

        check_sdk_version(&headers);
        // Second call hits the `Some(None)` fast path; no crash,
        // no re-parse, no re-insertion.
        check_sdk_version(&headers);

        let cached = SDK_VERSION_CACHE
            .get(version)
            .map(|v| *v)
            .expect("cache entry should be populated even for garbage");
        assert_eq!(cached, None);
    }

    // ── resolve_machine_profile ────────────────────────────────────

    fn make_gpu_map(gpus: &[&str]) -> std::collections::HashMap<String, String> {
        gpus.iter()
            .map(|g| (g.to_lowercase(), g.to_string()))
            .collect()
    }

    #[test]
    fn test_resolve_machine_profile_exact_match() {
        let m = make_gpu_map(&["l4-spot", "a100-40gb"]);
        assert_eq!(resolve_machine_profile("l4-spot", &m), "l4-spot");
    }

    #[test]
    fn test_resolve_machine_profile_case_insensitive() {
        let m = make_gpu_map(&["L4-Spot"]);
        assert_eq!(resolve_machine_profile("l4-spot", &m), "L4-Spot");
    }

    #[test]
    fn test_resolve_machine_profile_spot_fallback() {
        let m = make_gpu_map(&["l4-spot"]);
        assert_eq!(resolve_machine_profile("l4", &m), "l4-spot");
    }

    #[test]
    fn test_resolve_machine_profile_no_match() {
        let m = make_gpu_map(&["l4-spot"]);
        assert_eq!(resolve_machine_profile("h100", &m), "h100");
    }

    // ── is_hop_by_hop ──────────────────────────────────────────────

    #[test]
    fn test_is_hop_by_hop_true() {
        assert!(is_hop_by_hop("connection"));
        assert!(is_hop_by_hop("Connection"));
        assert!(is_hop_by_hop("Transfer-Encoding"));
        assert!(is_hop_by_hop("host"));
        assert!(is_hop_by_hop("keep-alive"));
    }

    #[test]
    fn test_is_hop_by_hop_false() {
        assert!(!is_hop_by_hop("content-type"));
        assert!(!is_hop_by_hop("authorization"));
        assert!(!is_hop_by_hop("x-custom-header"));
    }

    // ── parse_engine_pin ───────────────────────────────────────────

    fn hdr(raw: &[u8]) -> HeaderMap {
        let mut m = HeaderMap::new();
        m.insert(
            HeaderName::from_static("x-sie-engine"),
            HeaderValue::from_bytes(raw).expect("test value should be valid"),
        );
        m
    }

    #[test]
    fn test_parse_engine_pin_absent() {
        let m = HeaderMap::new();
        assert_eq!(parse_engine_pin(&m), EnginePinParse::None);
    }

    #[test]
    fn test_parse_engine_pin_empty_or_whitespace_is_none() {
        assert_eq!(parse_engine_pin(&hdr(b"")), EnginePinParse::None);
        assert_eq!(parse_engine_pin(&hdr(b"   ")), EnginePinParse::None);
        assert_eq!(parse_engine_pin(&hdr(b"\t")), EnginePinParse::None);
    }

    #[test]
    fn test_parse_engine_pin_lowercase_known_passes_through() {
        assert_eq!(
            parse_engine_pin(&hdr(b"pytorch")),
            EnginePinParse::Some("pytorch".to_string()),
        );
        assert_eq!(
            parse_engine_pin(&hdr(b"candle")),
            EnginePinParse::Some("candle".to_string()),
        );
    }

    #[test]
    fn test_parse_engine_pin_normalises_case_and_whitespace() {
        // Mixed/upper case must round-trip to the lowercase canonical
        // form so downstream registry lookups don't miss-match.
        assert_eq!(
            parse_engine_pin(&hdr(b"PyTorch")),
            EnginePinParse::Some("pytorch".to_string()),
        );
        assert_eq!(
            parse_engine_pin(&hdr(b"  PYTORCH  ")),
            EnginePinParse::Some("pytorch".to_string()),
        );
    }

    #[test]
    fn test_parse_engine_pin_unknown_token_is_unknown() {
        assert_eq!(
            parse_engine_pin(&hdr(b"future-engine")),
            EnginePinParse::Unknown("future-engine".to_string()),
        );
        assert_eq!(
            parse_engine_pin(&hdr(b"jax")),
            EnginePinParse::Unknown("jax".to_string()),
        );
        // Whitespace stripped before reporting the raw token, so the
        // 400 message stays readable.
        assert_eq!(
            parse_engine_pin(&hdr(b"  jax  ")),
            EnginePinParse::Unknown("jax".to_string()),
        );
    }

    #[test]
    fn test_parse_engine_pin_invalid_utf8_is_invalid() {
        // 0xFF is an invalid UTF-8 lead byte.
        let mut m = HeaderMap::new();
        m.insert(
            HeaderName::from_static("x-sie-engine"),
            HeaderValue::from_bytes(&[0xFFu8]).expect("bytes always valid"),
        );
        assert_eq!(parse_engine_pin(&m), EnginePinParse::InvalidUtf8);
    }

    #[test]
    fn test_parse_queue_request_reads_nested_params() {
        let body = serde_json::to_vec(&json!({
            "items": [{"text": "a"}],
            "params": {
                "output_types": ["dense"],
                "instruction": "search",
                "options": {
                    "truncate": true,
                    "is_query": true
                }
            }
        }))
        .unwrap();

        let (items, params) = parse_queue_request(&body, false, "encode").unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(params.output_types, Some(vec!["dense".to_string()]));
        assert_eq!(params.instruction, Some("search".to_string()));
        assert!(params.is_query);
        assert_eq!(params.options.unwrap()["truncate"], true);
    }

    #[test]
    fn test_parse_queue_request_json_encode_carries_output_dtype_without_options() {
        let body = serde_json::to_vec(&json!({
            "items": [{"text": "a"}],
            "params": {"output_dtype": "float32"}
        }))
        .unwrap();

        let (_, params) = parse_queue_request(&body, false, "encode").unwrap();
        assert_eq!(params.options, Some(json!({"output_dtype": "float32"})));
    }

    #[test]
    fn test_parse_queue_request_json_encode_output_dtype_overrides_options() {
        let body = serde_json::to_vec(&json!({
            "items": [{"text": "a"}],
            "params": {
                "output_dtype": "float32",
                "options": {
                    "output_dtype": "float16",
                    "profile": "quantized"
                }
            }
        }))
        .unwrap();

        let (_, params) = parse_queue_request(&body, false, "encode").unwrap();
        assert_eq!(
            params.options,
            Some(json!({
                "output_dtype": "float32",
                "profile": "quantized"
            }))
        );
    }

    fn rmpv_map_get<'a>(value: &'a rmpv::Value, key: &str) -> Option<&'a rmpv::Value> {
        let rmpv::Value::Map(entries) = value else {
            return None;
        };
        entries.iter().find_map(|(entry_key, entry_value)| {
            if matches!(entry_key, rmpv::Value::String(s) if s.as_str() == Some(key)) {
                Some(entry_value)
            } else {
                None
            }
        })
    }

    fn assert_media_data(value: &rmpv::Value, field: &str, expected: &[u8]) {
        let media = rmpv_map_get(value, field).unwrap_or_else(|| panic!("{field} missing"));
        let data = rmpv_map_get(media, "data").unwrap_or_else(|| panic!("{field}.data missing"));
        assert_eq!(data, &rmpv::Value::Binary(expected.to_vec()));
    }

    #[test]
    fn test_parse_queue_request_json_decodes_native_media_data_to_binary() {
        let body = serde_json::to_vec(&json!({
            "items": [{
                "id": "t",
                "images": [{"data": "aGVsbG8=", "format": "png"}],
                "audio": {"data": "YXVkaW8=", "format": "wav"},
                "video": {"data": "dmlkZW8=", "format": "mp4"},
                "document": {"data": "ZG9j", "format": "pdf"},
                "metadata": {"data": "bm90LW1lZGlh"}
            }],
            "params": {"output_types": ["multivector"]}
        }))
        .unwrap();

        let (items, params) = parse_queue_request(&body, false, "encode").unwrap();
        assert_eq!(params.output_types, Some(vec!["multivector".to_string()]));
        let item = items.first().expect("one item");

        let images = rmpv_map_get(item, "images").expect("images missing");
        let rmpv::Value::Array(images) = images else {
            panic!("images must stay an array");
        };
        let image = images.first().expect("one image");
        let data = rmpv_map_get(image, "data").expect("images[0].data missing");
        assert_eq!(data, &rmpv::Value::Binary(b"hello".to_vec()));
        assert!(matches!(
            rmpv_map_get(image, "format"),
            Some(rmpv::Value::String(s)) if s.as_str() == Some("png")
        ));

        assert_media_data(item, "audio", b"audio");
        assert_media_data(item, "video", b"video");
        assert_media_data(item, "document", b"doc");
        assert!(matches!(
            rmpv_map_get(
                rmpv_map_get(item, "metadata").expect("metadata missing"),
                "data"
            ),
            Some(rmpv::Value::String(s)) if s.as_str() == Some("bm90LW1lZGlh")
        ));
    }

    #[test]
    fn test_parse_queue_request_json_rejects_invalid_native_media_base64() {
        let body = serde_json::to_vec(&json!({
            "items": [{"images": [{"data": "!!!", "format": "png"}]}]
        }))
        .unwrap();

        let err = parse_queue_request(&body, false, "encode").unwrap_err();
        match err {
            QueueParseError::Generic(message) => {
                assert!(
                    message.contains("invalid base64 in images[0].data"),
                    "unexpected error: {message}"
                );
            }
            QueueParseError::PreBuilt(_) => panic!("expected generic parse error"),
        }
    }

    #[test]
    fn maximum_audio_fits_extract_ingress_on_json_and_msgpack() {
        const MAX_AUDIO_BYTES: usize = 24 * 1024 * 1024;

        let audio = vec![0x5a; MAX_AUDIO_BYTES];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&audio);
        let body = serde_json::to_vec(&json!({
            "input": {"audio": {"data": encoded, "format": "wav"}}
        }))
        .unwrap();
        assert!(body.len() > 32 * 1024 * 1024);
        assert!(body.len() <= MAX_EXTRACT_BODY);

        let (items, _) = parse_queue_request(&body, false, "extract").unwrap();
        assert_media_data(items.first().expect("one item"), "audio", &audio);
        drop(items);
        drop(body);

        let body_value = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("items"),
                rmpv::Value::Array(vec![rmpv::Value::Map(vec![(
                    rmpv::Value::from("audio"),
                    rmpv::Value::Map(vec![
                        (
                            rmpv::Value::from("data"),
                            rmpv::Value::Binary(audio.clone()),
                        ),
                        (rmpv::Value::from("format"), rmpv::Value::from("wav")),
                    ]),
                )])]),
            ),
            (
                rmpv::Value::from("params"),
                rmpv::Value::Map(vec![
                    (rmpv::Value::from("instruction"), rmpv::Value::Nil),
                    (rmpv::Value::from("options"), rmpv::Value::Map(Vec::new())),
                ]),
            ),
        ]);
        let body = rmp_serde::to_vec(&body_value).unwrap();
        assert!(body.len() <= MAX_AUDIO_BYTES + 1024);
        assert!(body.len() <= MAX_EXTRACT_BODY);

        let (items, _) = parse_queue_request(&body, true, "extract").unwrap();
        assert_media_data(items.first().expect("one item"), "audio", &audio);
    }

    #[test]
    fn test_parse_queue_request_score_keeps_query_and_items() {
        let body = serde_json::to_vec(&json!({
            "query": {"text": "hello"},
            "items": [{"text": "a"}, {"text": "b"}],
            "instruction": "rank"
        }))
        .unwrap();

        let (items, params) = parse_queue_request(&body, false, "score").unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(
            params.query_item,
            Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("hello"),
            )]))
        );
        assert_eq!(params.instruction, Some("rank".to_string()));
    }

    #[test]
    fn test_parse_queue_request_score_instruction_precedence_matches_json_and_msgpack() {
        let cases = [
            (
                "top-level wins",
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "instruction": "top-level",
                    "options": {"instruction": "option", "normalize": true}
                }),
                Some("top-level"),
            ),
            (
                "empty top-level suppresses fallback",
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "instruction": "",
                    "options": {"instruction": "option"}
                }),
                Some(""),
            ),
            (
                "missing top-level promotes option",
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "options": {"instruction": "option", "normalize": true}
                }),
                Some("option"),
            ),
            (
                "null top-level promotes option",
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "instruction": null,
                    "options": {"instruction": "option", "normalize": true}
                }),
                Some("option"),
            ),
            (
                "null option instruction remains absent",
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "instruction": null,
                    "options": {"instruction": null, "normalize": true}
                }),
                None,
            ),
            (
                "empty option is promoted",
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "options": {"instruction": ""}
                }),
                Some(""),
            ),
            (
                "null options remain absent",
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "instruction": null,
                    "options": null
                }),
                None,
            ),
            (
                "no instruction remains absent",
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "options": {"normalize": true}
                }),
                None,
            ),
        ];

        for (name, body, expected) in cases {
            let json_body = serde_json::to_vec(&body).unwrap();
            let msgpack_body = rmp_serde::to_vec_named(&body).unwrap();
            let (_, json_params) = parse_queue_request(&json_body, false, "score").unwrap();
            let (_, msgpack_params) = parse_queue_request(&msgpack_body, true, "score").unwrap();

            for params in [&json_params, &msgpack_params] {
                assert_eq!(params.instruction.as_deref(), expected, "{name}");
                assert_eq!(params.options, body.get("options").cloned(), "{name}");
            }
        }
    }

    #[test]
    fn test_parse_queue_request_score_rejects_invalid_typed_grammar_json_and_msgpack() {
        let invalid_values = [
            ("boolean", json!(true)),
            ("number", json!(7)),
            ("array", json!([])),
            ("object", json!({})),
        ];

        let mut cases = Vec::new();
        for (kind, value) in invalid_values {
            cases.push((
                format!("top-level instruction {kind}"),
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "instruction": value
                }),
                "'instruction' must be a string or null",
            ));
        }
        for (kind, value) in [
            ("boolean", json!(true)),
            ("number", json!(7)),
            ("string", json!("not-an-object")),
            ("array", json!([])),
        ] {
            cases.push((
                format!("options container {kind}"),
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "options": value
                }),
                "'options' must be an object or null",
            ));
        }
        for (kind, value) in [
            ("boolean", json!(false)),
            ("number", json!(7)),
            ("array", json!([])),
            ("object", json!({})),
        ] {
            cases.push((
                format!("nested instruction {kind}"),
                json!({
                    "query": {"text": "q"},
                    "items": [{"text": "d"}],
                    "options": {"instruction": value}
                }),
                "'options.instruction' must be a string or null",
            ));
        }

        for (name, body, expected) in cases {
            let encoded = [
                ("json", serde_json::to_vec(&body).unwrap(), false),
                ("msgpack", rmp_serde::to_vec_named(&body).unwrap(), true),
            ];
            for (wire, bytes, is_msgpack) in encoded {
                match parse_queue_request(&bytes, is_msgpack, "score").unwrap_err() {
                    QueueParseError::Generic(message) => assert!(
                        message.contains(expected),
                        "{name} over {wire}: unexpected error: {message}"
                    ),
                    QueueParseError::PreBuilt(_) => {
                        panic!("{name} over {wire}: expected generic 400-equivalent error")
                    }
                }
            }
        }
    }

    #[test]
    fn test_parse_queue_request_score_rejects_msgpack_binary_nested_instruction() {
        let body_value = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("query"),
                rmpv::Value::Map(vec![(rmpv::Value::from("text"), rmpv::Value::from("q"))]),
            ),
            (
                rmpv::Value::from("items"),
                rmpv::Value::Array(vec![rmpv::Value::Map(vec![(
                    rmpv::Value::from("text"),
                    rmpv::Value::from("d"),
                )])]),
            ),
            (
                rmpv::Value::from("options"),
                rmpv::Value::Map(vec![(
                    rmpv::Value::from("instruction"),
                    rmpv::Value::Binary(b"not-a-msgpack-string".to_vec()),
                )]),
            ),
        ]);
        let body = rmp_serde::to_vec(&body_value).unwrap();

        match parse_queue_request(&body, true, "score").unwrap_err() {
            QueueParseError::Generic(message) => assert!(
                message.contains("'options.instruction' must be a string or null"),
                "unexpected error: {message}"
            ),
            QueueParseError::PreBuilt(_) => panic!("expected generic 400-equivalent error"),
        }
    }

    #[test]
    fn test_parse_queue_request_score_rejects_msgpack_duplicate_and_non_string_keys() {
        let score_body = |extra: Vec<(rmpv::Value, rmpv::Value)>| {
            let mut entries = vec![
                (
                    rmpv::Value::from("query"),
                    rmpv::Value::Map(vec![(rmpv::Value::from("text"), rmpv::Value::from("q"))]),
                ),
                (
                    rmpv::Value::from("items"),
                    rmpv::Value::Array(vec![rmpv::Value::Map(vec![(
                        rmpv::Value::from("text"),
                        rmpv::Value::from("d"),
                    )])]),
                ),
            ];
            entries.extend(extra);
            rmpv::Value::Map(entries)
        };
        let options = |instruction_entries: Vec<(rmpv::Value, rmpv::Value)>| {
            rmpv::Value::Map(instruction_entries)
        };

        let cases = vec![
            (
                "duplicate top-level instruction valid then invalid",
                score_body(vec![
                    (rmpv::Value::from("instruction"), rmpv::Value::from("valid")),
                    (
                        rmpv::Value::from("instruction"),
                        rmpv::Value::Boolean(false),
                    ),
                ]),
                "duplicate score request field 'instruction'",
            ),
            (
                "duplicate top-level instruction invalid then valid",
                score_body(vec![
                    (
                        rmpv::Value::from("instruction"),
                        rmpv::Value::Boolean(false),
                    ),
                    (rmpv::Value::from("instruction"), rmpv::Value::from("valid")),
                ]),
                "duplicate score request field 'instruction'",
            ),
            (
                "duplicate top-level options object then scalar",
                score_body(vec![
                    (
                        rmpv::Value::from("options"),
                        options(vec![(
                            rmpv::Value::from("instruction"),
                            rmpv::Value::from("valid"),
                        )]),
                    ),
                    (rmpv::Value::from("options"), rmpv::Value::Boolean(false)),
                ]),
                "duplicate score request field 'options'",
            ),
            (
                "duplicate top-level options scalar then object",
                score_body(vec![
                    (rmpv::Value::from("options"), rmpv::Value::Boolean(false)),
                    (
                        rmpv::Value::from("options"),
                        options(vec![(
                            rmpv::Value::from("instruction"),
                            rmpv::Value::from("valid"),
                        )]),
                    ),
                ]),
                "duplicate score request field 'options'",
            ),
            (
                "duplicate nested instruction valid then invalid",
                score_body(vec![(
                    rmpv::Value::from("options"),
                    options(vec![
                        (rmpv::Value::from("instruction"), rmpv::Value::from("valid")),
                        (
                            rmpv::Value::from("instruction"),
                            rmpv::Value::Boolean(false),
                        ),
                    ]),
                )]),
                "duplicate score request field 'options.instruction'",
            ),
            (
                "duplicate nested instruction invalid then valid",
                score_body(vec![(
                    rmpv::Value::from("options"),
                    options(vec![
                        (
                            rmpv::Value::from("instruction"),
                            rmpv::Value::Boolean(false),
                        ),
                        (rmpv::Value::from("instruction"), rmpv::Value::from("valid")),
                    ]),
                )]),
                "duplicate score request field 'options.instruction'",
            ),
            (
                "binary top-level key",
                score_body(vec![(
                    rmpv::Value::Binary(b"instruction".to_vec()),
                    rmpv::Value::from("value"),
                )]),
                "score request field names must be MessagePack strings",
            ),
            (
                "binary nested key",
                score_body(vec![(
                    rmpv::Value::from("options"),
                    options(vec![(
                        rmpv::Value::Binary(b"instruction".to_vec()),
                        rmpv::Value::from("value"),
                    )]),
                )]),
                "'options' field names must be MessagePack strings",
            ),
            (
                "integer top-level key",
                score_body(vec![(rmpv::Value::from(7), rmpv::Value::from("value"))]),
                "score request field names must be MessagePack strings",
            ),
            (
                "integer nested key",
                score_body(vec![(
                    rmpv::Value::from("options"),
                    options(vec![(rmpv::Value::from(7), rmpv::Value::from("value"))]),
                )]),
                "'options' field names must be MessagePack strings",
            ),
        ];

        for (name, value, expected) in cases {
            let body = rmp_serde::to_vec(&value).unwrap();
            match parse_queue_request(&body, true, "score").unwrap_err() {
                QueueParseError::Generic(message) => assert!(
                    message.contains(expected),
                    "{name}: unexpected error: {message}"
                ),
                QueueParseError::PreBuilt(_) => {
                    panic!("{name}: expected generic 400-equivalent error")
                }
            }
        }
    }

    #[test]
    fn test_parse_queue_request_score_defaults_missing_query_to_empty_object() {
        let body = serde_json::to_vec(&json!({
            "items": [{"text": "a"}]
        }))
        .unwrap();

        let (items, params) = parse_queue_request(&body, false, "score").unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(params.query_item, Some(rmpv::Value::Map(Vec::new())));
    }

    #[test]
    fn test_parse_queue_request_score_rejects_non_object_query() {
        let body = serde_json::to_vec(&json!({
            "query": "hello",
            "items": [{"text": "a"}]
        }))
        .unwrap();
        let err = parse_queue_request(&body, false, "score").unwrap_err();
        match err {
            QueueParseError::Generic(s) => assert!(
                s.contains("'query' must be an object"),
                "unexpected error: {s}"
            ),
            QueueParseError::PreBuilt(_) => panic!("expected generic parse error"),
        }
    }

    #[test]
    fn test_parse_queue_request_encode_rejects_non_object_item() {
        let body = serde_json::to_vec(&json!({
            "items": ["a bare string is not an item"]
        }))
        .unwrap();
        let err = parse_queue_request(&body, false, "encode").unwrap_err();
        match err {
            QueueParseError::Generic(s) => assert!(
                s.contains("item at index 0 must be an object"),
                "unexpected error: {s}"
            ),
            QueueParseError::PreBuilt(_) => panic!("expected generic parse error"),
        }
    }

    #[test]
    fn test_parse_queue_request_extract_rejects_non_object_input() {
        let body = serde_json::to_vec(&json!({
            "input": 42
        }))
        .unwrap();
        let err = parse_queue_request(&body, false, "extract").unwrap_err();
        match err {
            QueueParseError::Generic(s) => assert!(
                s.contains("item at index 0 must be an object"),
                "unexpected error: {s}"
            ),
            QueueParseError::PreBuilt(_) => panic!("expected generic parse error"),
        }
    }

    #[test]
    fn test_parse_queue_request_msgpack_score_rejects_non_object_item() {
        let body_value = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("query"),
                rmpv::Value::Map(vec![(rmpv::Value::from("text"), rmpv::Value::from("q"))]),
            ),
            (
                rmpv::Value::from("items"),
                rmpv::Value::Array(vec![rmpv::Value::from("a bare string is not an item")]),
            ),
        ]);
        let body = rmp_serde::to_vec(&body_value).unwrap();
        let err = parse_queue_request(&body, true, "score").unwrap_err();
        match err {
            QueueParseError::Generic(s) => assert!(
                s.contains("item at index 0 must be an object"),
                "unexpected error: {s}"
            ),
            QueueParseError::PreBuilt(_) => panic!("expected generic parse error"),
        }
    }

    /// Msgpack-in encode request: tuning fields are read only from the nested
    /// ``params`` map (parity with ``sie_server`` / msgspec).
    #[test]
    fn test_parse_queue_request_msgpack_encode_reads_params() {
        let body_value = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("items"),
                rmpv::Value::Array(vec![rmpv::Value::Map(vec![(
                    rmpv::Value::from("text"),
                    rmpv::Value::from("hello"),
                )])]),
            ),
            (
                rmpv::Value::from("params"),
                rmpv::Value::Map(vec![
                    (
                        rmpv::Value::from("output_types"),
                        rmpv::Value::Array(vec![rmpv::Value::from("dense")]),
                    ),
                    (
                        rmpv::Value::from("instruction"),
                        rmpv::Value::from("search"),
                    ),
                    (
                        rmpv::Value::from("options"),
                        rmpv::Value::Map(vec![(
                            rmpv::Value::from("is_query"),
                            rmpv::Value::Boolean(true),
                        )]),
                    ),
                ]),
            ),
        ]);
        let body = rmp_serde::to_vec(&body_value).unwrap();

        let (items, params) = parse_queue_request(&body, true, "encode").unwrap();
        assert_eq!(items.len(), 1);
        // The per-item value must stay an `rmpv::Value::Map` —
        // the whole point of the passthrough path is that msgpack-in
        // items never round-trip through `serde_json::Value`.
        assert!(matches!(&items[0], rmpv::Value::Map(_)));
        assert_eq!(params.output_types, Some(vec!["dense".to_string()]));
        assert_eq!(params.instruction, Some("search".to_string()));
        assert!(params.is_query);
    }

    #[test]
    fn test_parse_queue_request_msgpack_encode_carries_output_dtype_without_options() {
        let body = rmp_serde::to_vec_named(&json!({
            "items": [{"text": "hello"}],
            "params": {"output_dtype": "float32"}
        }))
        .unwrap();

        let (_, params) = parse_queue_request(&body, true, "encode").unwrap();
        assert_eq!(params.options, Some(json!({"output_dtype": "float32"})));
    }

    #[test]
    fn test_parse_queue_request_msgpack_encode_output_dtype_overrides_options() {
        let body = rmp_serde::to_vec_named(&json!({
            "items": [{"text": "hello"}],
            "params": {
                "output_dtype": "float32",
                "options": {
                    "output_dtype": "float16",
                    "profile": "quantized"
                }
            }
        }))
        .unwrap();

        let (_, params) = parse_queue_request(&body, true, "encode").unwrap();
        assert_eq!(
            params.options,
            Some(json!({
                "output_dtype": "float32",
                "profile": "quantized"
            }))
        );
    }

    /// Regression guard for the rmpv-passthrough correctness fix:
    /// a msgpack-in request carrying a `bin` blob (e.g. a raw
    /// numpy buffer the SDK packs with `msgpack_numpy`) must reach
    /// the publisher as `rmpv::Value::Binary` byte-for-byte. The
    /// old serde_json-intermediate path expanded every byte into a
    /// `Value::Number`, so the wire bytes workers received were
    /// corrupted for any binary-heavy request.
    #[test]
    fn test_parse_queue_request_msgpack_preserves_binary_in_item() {
        let payload: Vec<u8> = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02];
        let body_value = rmpv::Value::Map(vec![(
            rmpv::Value::from("items"),
            rmpv::Value::Array(vec![rmpv::Value::Map(vec![(
                rmpv::Value::from("blob"),
                rmpv::Value::Binary(payload.clone()),
            )])]),
        )]);
        let body = rmp_serde::to_vec(&body_value).unwrap();

        let (items, _) = parse_queue_request(&body, true, "encode").unwrap();
        assert_eq!(items.len(), 1);
        let rmpv::Value::Map(entries) = &items[0] else {
            panic!("expected Map, got {:?}", items[0]);
        };
        let blob = entries
            .iter()
            .find(|(k, _)| matches!(k, rmpv::Value::String(s) if s.as_str() == Some("blob")))
            .map(|(_, v)| v)
            .expect("blob field missing");
        assert_eq!(blob, &rmpv::Value::Binary(payload));
    }

    /// Top-level msgpack bodies must be maps — parity with the
    /// JSON path which rejects non-object bodies with 400. A
    /// top-level array / scalar is almost always a mis-encoded
    /// client request; silently accepting it as a single item
    /// used to let the request fail later in worker-specific
    /// ways instead of at ingress.
    #[test]
    fn test_parse_queue_request_msgpack_rejects_non_map_top_level() {
        let array_body = rmp_serde::to_vec(&rmpv::Value::Array(vec![
            rmpv::Value::from(1),
            rmpv::Value::from(2),
        ]))
        .unwrap();
        let err = parse_queue_request(&array_body, true, "encode").unwrap_err();
        match err {
            QueueParseError::Generic(s) => {
                assert!(s.contains("msgpack map"), "unexpected error: {s}")
            }
            QueueParseError::PreBuilt(_) => panic!("expected generic error"),
        }

        let scalar_body = rmp_serde::to_vec(&rmpv::Value::from(42)).unwrap();
        let err = parse_queue_request(&scalar_body, true, "encode").unwrap_err();
        match err {
            QueueParseError::Generic(s) => {
                assert!(s.contains("msgpack map"), "unexpected error: {s}")
            }
            QueueParseError::PreBuilt(_) => panic!("expected generic error"),
        }
    }

    /// Msgpack-in score request: `query` + `items` live at the
    /// top level, and `params.query_item` must carry the rmpv
    /// representation (not a round-tripped serde_json blob).
    #[test]
    fn test_parse_queue_request_msgpack_score_keeps_query_as_rmpv() {
        let body_value = rmpv::Value::Map(vec![
            (
                rmpv::Value::from("query"),
                rmpv::Value::Map(vec![(rmpv::Value::from("text"), rmpv::Value::from("q"))]),
            ),
            (
                rmpv::Value::from("items"),
                rmpv::Value::Array(vec![
                    rmpv::Value::Map(vec![(rmpv::Value::from("text"), rmpv::Value::from("a"))]),
                    rmpv::Value::Map(vec![(rmpv::Value::from("text"), rmpv::Value::from("b"))]),
                ]),
            ),
        ]);
        let body = rmp_serde::to_vec(&body_value).unwrap();

        let (items, params) = parse_queue_request(&body, true, "score").unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(
            params.query_item,
            Some(rmpv::Value::Map(vec![(
                rmpv::Value::from("text"),
                rmpv::Value::from("q"),
            )]))
        );
    }

    #[test]
    fn test_parse_queue_request_score_ignores_nested_params() {
        let body = serde_json::to_vec(&json!({
            "items": [{"text": "candidate"}],
            "instruction": "top-level",
            "options": {"truncate": false},
            "params": {
                "instruction": "nested",
                "options": {"truncate": true}
            }
        }))
        .unwrap();

        let (items, params) = parse_queue_request(&body, false, "score").unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(params.query_item, Some(rmpv::Value::Map(Vec::new())));
        assert_eq!(params.instruction, Some("top-level".to_string()));
        assert_eq!(params.options, Some(json!({"truncate": false})));
    }

    #[test]
    fn test_parse_queue_request_score_enforces_candidate_bounds() {
        let empty = serde_json::to_vec(&json!({
            "query": {"text": "query"},
            "items": []
        }))
        .unwrap();
        let err = parse_queue_request(&empty, false, "score").unwrap_err();
        match err {
            QueueParseError::Generic(message) => assert!(message.contains("at least one")),
            QueueParseError::PreBuilt(_) => panic!("expected generic parse error"),
        }

        let too_many = serde_json::to_vec(&json!({
            "query": {"text": "query"},
            "items": vec![json!({"text": "candidate"}); MAX_SCORE_ITEMS + 1]
        }))
        .unwrap();
        let err = parse_queue_request(&too_many, false, "score").unwrap_err();
        match err {
            QueueParseError::Generic(message) => assert!(message.contains("at most 1000")),
            QueueParseError::PreBuilt(_) => panic!("expected generic parse error"),
        }
    }

    // ── msgpack_numpy conversion tests ──────────────────────────
    //
    // These exercise the fused `rmpv_to_response_json` path. All
    // fixtures use the exact rmpv shape that Python adapter processes emit
    // via `msgpack_numpy` — a `Map` whose `data` key holds a
    // `Binary` blob — so the tests double as a wire-format guard:
    // if Python-side encoding ever changes, these flip first.

    /// Helper: build a `{"nd": true, "type": ..., "shape": [...],
    /// "data": <binary>}` rmpv sentinel for dtype/shape tests.
    fn numpy_sentinel(dtype: &str, shape: &[usize], bytes: Vec<u8>) -> rmpv::Value {
        rmpv::Value::Map(vec![
            (rmpv::Value::from("nd"), rmpv::Value::Boolean(true)),
            (rmpv::Value::from("type"), rmpv::Value::from(dtype)),
            (
                rmpv::Value::from("shape"),
                rmpv::Value::Array(
                    shape
                        .iter()
                        .map(|&n| rmpv::Value::Integer((n as u64).into()))
                        .collect(),
                ),
            ),
            (rmpv::Value::from("data"), rmpv::Value::Binary(bytes)),
        ])
    }

    #[test]
    fn test_rmpv_to_response_json_f32_array() {
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let value = numpy_sentinel("<f4", &[3], bytes);
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr.len(), 3);
        assert!((arr[0].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((arr[1].as_f64().unwrap() - 2.0).abs() < 1e-6);
        assert!((arr[2].as_f64().unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_rmpv_to_response_json_f64_array() {
        let bytes: Vec<u8> = [1.5f64, 2.5].iter().flat_map(|f| f.to_le_bytes()).collect();
        let value = numpy_sentinel("<f8", &[2], bytes);
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr.len(), 2);
        assert!((arr[0].as_f64().unwrap() - 1.5).abs() < 1e-10);
        assert!((arr[1].as_f64().unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_rmpv_to_response_json_i32_array() {
        let bytes: Vec<u8> = [42i32, -7].iter().flat_map(|i| i.to_le_bytes()).collect();
        let value = numpy_sentinel("<i4", &[2], bytes);
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0].as_i64().unwrap(), 42);
        assert_eq!(arr[1].as_i64().unwrap(), -7);
    }

    #[test]
    fn test_rmpv_to_response_json_bool_array() {
        let value = numpy_sentinel("|b1", &[3], vec![1, 0, 1]);
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr.len(), 3);
        assert!(arr[0].as_bool().unwrap());
        assert!(!arr[1].as_bool().unwrap());
        assert!(arr[2].as_bool().unwrap());
    }

    #[test]
    fn test_rmpv_to_response_json_u8_array() {
        let value = numpy_sentinel("|u1", &[3], vec![0, 128, 255]);
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr[0].as_u64().unwrap(), 0);
        assert_eq!(arr[1].as_u64().unwrap(), 128);
        assert_eq!(arr[2].as_u64().unwrap(), 255);
    }

    #[test]
    fn test_rmpv_to_response_json_2d_shape() {
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let value = numpy_sentinel("<f4", &[2, 3], bytes);
        let outer = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(outer.len(), 2);
        let row0 = outer[0].as_array().unwrap();
        assert_eq!(row0.len(), 3);
        assert!((row0[0].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((row0[2].as_f64().unwrap() - 3.0).abs() < 1e-6);
        let row1 = outer[1].as_array().unwrap();
        assert!((row1[0].as_f64().unwrap() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_rmpv_to_response_json_rejects_mismatched_numpy_shape() {
        let bytes: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let value = numpy_sentinel("<f4", &[2, 3], bytes);
        if let rmpv::Value::Map(entries) = value {
            assert!(
                try_decode_rmpv_numpy(&entries).is_none(),
                "mismatched shape must not silently truncate decoded values"
            );
        } else {
            panic!("numpy_sentinel helper must return a map");
        }
    }

    #[test]
    fn test_rmpv_to_response_json_empty_data() {
        let value = numpy_sentinel("<f4", &[0], Vec::new());
        let arr = rmpv_to_response_json(value).as_array().unwrap().clone();
        assert_eq!(arr.len(), 0);
    }

    /// Non-sentinel maps walk through untouched and keep their
    /// original shape — we must not misdecode user maps that happen
    /// to carry a `"data"` key.
    #[test]
    fn test_rmpv_to_response_json_no_sentinel_unchanged() {
        let value = rmpv::Value::Map(vec![
            (rmpv::Value::from("key"), rmpv::Value::from("value")),
            (
                rmpv::Value::from("nested"),
                rmpv::Value::Map(vec![(
                    rmpv::Value::from("a"),
                    rmpv::Value::Integer(1.into()),
                )]),
            ),
        ]);
        let json_val = rmpv_to_response_json(value);
        assert_eq!(json_val["key"], "value");
        assert_eq!(json_val["nested"]["a"], 1);
    }

    /// Guards that a sentinel-shaped map whose `nd` is `false` is
    /// left alone: the `nd == true` marker is load-bearing, plain
    /// user maps that happen to share the key set must pass through
    /// unchanged.
    #[test]
    fn test_rmpv_to_response_json_ignores_nd_false() {
        let value = rmpv::Value::Map(vec![
            (rmpv::Value::from("nd"), rmpv::Value::Boolean(false)),
            (rmpv::Value::from("type"), rmpv::Value::from("<f4")),
            (
                rmpv::Value::from("shape"),
                rmpv::Value::Array(vec![rmpv::Value::Integer(1.into())]),
            ),
            (rmpv::Value::from("data"), rmpv::Value::Binary(vec![0; 4])),
        ]);
        let json_val = rmpv_to_response_json(value);
        assert_eq!(json_val["nd"], false);
        assert_eq!(json_val["type"], "<f4");
    }

    #[test]
    fn test_f16_to_f32_basic() {
        let result = f16_to_f32(0x3C00); // f16 1.0
        assert!((result - 1.0).abs() < 1e-6);
        let result = f16_to_f32(0x0000); // f16 zero
        assert_eq!(result, 0.0);
        let result = f16_to_f32(0x8000); // f16 negative zero
        assert!(result.is_sign_negative());
        assert_eq!(result, -0.0f32);
    }

    #[test]
    fn test_queue_result_single_item_msgpack_passthrough() {
        let payload = rmp_serde::to_vec(&json!({"embedding": [1.0, 2.0]})).unwrap();
        let result = publisher::WorkResult {
            work_item_id: "r1.0".to_string(),
            request_id: "r1".to_string(),
            item_index: 0,
            success: true,
            result_msgpack: payload.clone(),
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units: None,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let resp_body = result.result_msgpack.clone();
        assert_eq!(resp_body, payload);
    }

    #[test]
    fn test_queue_result_single_item_json_decode() {
        let payload = rmp_serde::to_vec(&json!({"embedding": [1.0, 2.0]})).unwrap();
        let json_val: serde_json::Value = rmp_serde::from_slice(&payload).unwrap();
        let resp_body = serde_json::to_vec(&json_val).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&resp_body).unwrap();
        assert_eq!(parsed["embedding"][0], 1.0);
        assert_eq!(parsed["embedding"][1], 2.0);
    }

    #[test]
    fn test_queue_result_multi_item_msgpack_decode() {
        let payload1 = rmp_serde::to_vec(&json!({"result": "a"})).unwrap();
        let payload2 = rmp_serde::to_vec(&json!({"result": "b"})).unwrap();
        let results = [
            publisher::WorkResult {
                work_item_id: "r1.0".to_string(),
                request_id: "r1".to_string(),
                item_index: 0,
                success: true,
                result_msgpack: payload1,
                error: None,
                error_code: None,
                inference_ms: None,
                queue_ms: None,
                processing_ms: None,
                worker_id: None,
                tokenization_ms: None,
                postprocessing_ms: None,
                payload_fetch_ms: None,
                units: None,
                worker_direct: false,
                executed_bundle_config_hash: None,
                execution_identity_sha256: None,
                execution_binding_sha256: None,
            },
            publisher::WorkResult {
                work_item_id: "r1.1".to_string(),
                request_id: "r1".to_string(),
                item_index: 1,
                success: true,
                result_msgpack: payload2,
                error: None,
                error_code: None,
                inference_ms: None,
                queue_ms: None,
                processing_ms: None,
                worker_id: None,
                tokenization_ms: None,
                postprocessing_ms: None,
                payload_fetch_ms: None,
                units: None,
                worker_direct: false,
                executed_bundle_config_hash: None,
                execution_identity_sha256: None,
                execution_binding_sha256: None,
            },
        ];
        let items: Vec<serde_json::Value> = results
            .iter()
            .map(|r| rmp_serde::from_slice(&r.result_msgpack).unwrap())
            .collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["result"], "a");
        assert_eq!(items[1]["result"], "b");
        let combined = json!({"items": items});
        let body = serde_json::to_vec(&combined).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["items"].as_array().unwrap().len(), 2);
    }

    /// End-to-end guard: build a msgpack payload the way a real
    /// Python adapter process would (sentinel map with `data` as `Binary`),
    /// decode via `rmp_serde::from_slice` → `rmpv_to_response_json`,
    /// and check that bin data is decoded inline without byte-array
    /// inflation. Non-numpy fields (`text`) must survive untouched.
    #[test]
    fn test_rmpv_to_response_json_from_real_msgpack_bytes() {
        use rmpv::Value as MsgValue;

        let f32_bytes: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let payload = MsgValue::Map(vec![
            (
                MsgValue::String("embedding".into()),
                numpy_sentinel("<f4", &[3], f32_bytes),
            ),
            (
                MsgValue::String("text".into()),
                MsgValue::String("hello".into()),
            ),
        ]);

        let msgpack_bytes = rmp_serde::to_vec(&payload).unwrap();
        let decoded: rmpv::Value = rmp_serde::from_slice(&msgpack_bytes).unwrap();
        let json_val = rmpv_to_response_json(decoded);

        let arr = json_val["embedding"].as_array().unwrap();
        assert_eq!(arr.len(), 3);
        assert!((arr[0].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((arr[2].as_f64().unwrap() - 3.0).abs() < 1e-6);
        assert_eq!(json_val["text"], "hello");
    }

    /// The hot path fuses msgpack_numpy decode into the rmpv walk.
    /// This test builds the exact shape Python adapter processes produce — a
    /// map whose `data` field is an `rmpv::Value::Binary` blob —
    /// and confirms the fused function decodes the dtype directly
    /// from the byte slice, without ever materialising a
    /// byte-per-`Number` intermediate.
    #[test]
    fn test_rmpv_to_response_json_decodes_numpy_binary_directly() {
        use rmpv::Value as MsgValue;

        let f32_bytes: Vec<u8> = [1.0f32, 2.0f32, 3.0f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let sentinel = MsgValue::Map(vec![
            (MsgValue::String("nd".into()), MsgValue::Boolean(true)),
            (
                MsgValue::String("type".into()),
                MsgValue::String("<f4".into()),
            ),
            (
                MsgValue::String("shape".into()),
                MsgValue::Array(vec![MsgValue::Integer(3.into())]),
            ),
            (MsgValue::String("data".into()), MsgValue::Binary(f32_bytes)),
        ]);
        let payload = MsgValue::Map(vec![
            (MsgValue::String("embedding".into()), sentinel),
            (
                MsgValue::String("text".into()),
                MsgValue::String("hello".into()),
            ),
        ]);

        let json_val = rmpv_to_response_json(payload);

        let arr = json_val["embedding"]
            .as_array()
            .expect("embedding should be a flat array");
        assert_eq!(arr.len(), 3);
        assert!((arr[0].as_f64().unwrap() - 1.0).abs() < 1e-6);
        assert!((arr[1].as_f64().unwrap() - 2.0).abs() < 1e-6);
        assert!((arr[2].as_f64().unwrap() - 3.0).abs() < 1e-6);
        assert_eq!(json_val["text"], "hello");
    }

    /// Some msgpack_numpy variants pack the dtype buffer as an ext
    /// type instead of plain binary; the fused decode path treats
    /// both identically.
    #[test]
    fn test_rmpv_to_response_json_decodes_numpy_ext_data() {
        use rmpv::Value as MsgValue;

        let f32_bytes: Vec<u8> = [0.25f32, 0.5f32]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let sentinel = MsgValue::Map(vec![
            (MsgValue::String("nd".into()), MsgValue::Boolean(true)),
            (
                MsgValue::String("type".into()),
                MsgValue::String("<f4".into()),
            ),
            (
                MsgValue::String("shape".into()),
                MsgValue::Array(vec![MsgValue::Integer(2.into())]),
            ),
            (
                MsgValue::String("data".into()),
                MsgValue::Ext(0x15, f32_bytes),
            ),
        ]);

        let arr = rmpv_to_response_json(sentinel)
            .as_array()
            .expect("ext data should decode to array")
            .clone();
        assert_eq!(arr.len(), 2);
        assert!((arr[0].as_f64().unwrap() - 0.25).abs() < 1e-6);
        assert!((arr[1].as_f64().unwrap() - 0.5).abs() < 1e-6);
    }

    /// Non-sentinel maps must pass through unchanged — no accidental
    /// decode of user-defined maps that happen to include `"data"`.
    #[test]
    fn test_rmpv_to_response_json_passes_through_non_sentinel_maps() {
        use rmpv::Value as MsgValue;

        let map = MsgValue::Map(vec![
            (
                MsgValue::String("type".into()),
                MsgValue::String("user-data".into()),
            ),
            (
                MsgValue::String("data".into()),
                MsgValue::String("some value".into()),
            ),
        ]);

        let json_val = rmpv_to_response_json(map);
        assert_eq!(json_val["type"], "user-data");
        assert_eq!(json_val["data"], "some value");
    }

    /// Python msgpack sometimes emits binary keys (e.g. `b"nd"`,
    /// `b"type"`) when the encoder is not in `strict_map_key=True`
    /// mode. `rmpv_to_response_json` must still decode both the
    /// sentinel (at the embedding level) and the surrounding map
    /// keys back into string-keyed JSON.
    #[test]
    fn test_rmpv_to_response_json_handles_binary_map_keys() {
        use rmpv::Value as MsgValue;

        let map = MsgValue::Map(vec![(
            MsgValue::Binary(b"key".to_vec()),
            MsgValue::String("value".into()),
        )]);
        let json_val = rmpv_to_response_json(map);
        assert_eq!(json_val["key"], "value");
    }

    // ── resolve_effective_pool (scale-from-zero decision) ──────────
    //
    // These tests guard the contract that route resolution returns
    // `Provisioning` whenever no healthy worker is registered for
    // `(bundle, gpu)` — regardless of whether the caller sent an
    // `X-SIE-MACHINE-PROFILE` header. An earlier regression gated the
    // provisioning branch on `!gpu.is_empty()`, so default-routing cold
    // starts fell through to a `"default"` pool publish and hung.

    use crate::types::worker::{GpuStatus, ModelStatus, WorkerStatusMessage};
    use std::time::Duration as StdDuration;

    fn pool_registry() -> WorkerRegistry {
        WorkerRegistry::new(StdDuration::from_secs(30), None)
    }

    fn worker_msg(bundle: &str, gpu: &str, pool: &str) -> WorkerStatusMessage {
        WorkerStatusMessage {
            name: "worker-1".into(),
            ready: true,
            gpu_count: 1,
            total_gpu_slots: None,
            ready_gpu_slots: None,
            machine_profile: gpu.into(),
            pool_name: pool.into(),
            bundle: bundle.into(),
            bundle_config_hash: "abc".into(),
            loaded_models: vec![],
            models: vec![ModelStatus { queue_depth: 0 }],
            gpus: vec![GpuStatus {
                memory_used_bytes: 0,
                memory_total_bytes: 4000,
            }],
            queue_depth: None,
            pending_cost: None,
            inflight_batches: None,
            memory_used_bytes: None,
            memory_total_bytes: None,
            saturated: false,
            terminated: false,
        }
    }

    async fn sync_static_queue_pool(pm: &PoolManager, name: &str, profile: &str) {
        let mut gpus = std::collections::HashMap::new();
        gpus.insert(profile.to_string(), 0);
        pm.sync_static_pools(&[crate::types::pool::PoolSpec {
            name: name.to_string(),
            queue_pool: name.to_string(),
            bundle: None,
            gpus,
            gpu_caps: std::collections::HashMap::new(),
            ttl_seconds: None,
            minimum_worker_count: 0,
            pinned_models: Vec::new(),
        }])
        .await
        .unwrap();
    }

    fn route(pool: &str, gpu: &str) -> PoolResolution {
        PoolResolution::Route(QueueRoute {
            pool_name: pool.to_string(),
            machine_profile: gpu.to_string(),
        })
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_returns_provisioning_when_empty_and_no_gpu() {
        // No workers at all. Caller sends no `X-SIE-MACHINE-PROFILE`.
        // Before the fix this returned `Pool("default")` and the gateway
        // published to a nonexistent consumer. After the fix we emit
        // `Provisioning` so the caller returns its surface-specific
        // provisioning response and records pending demand for KEDA.
        let reg = pool_registry();
        let out = resolve_effective_pool(&reg, None, "default", "", "", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        // No GPU expressed → exact_gpu_match is definitionally false.
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_returns_provisioning_when_no_gpu_match() {
        // Worker exists but for a different GPU and a different bundle.
        // Nothing matches → provision.
        let reg = pool_registry();
        reg.update_worker("http://w1:8080", worker_msg("default", "a100", "pool-a"))
            .await;
        let out = resolve_effective_pool(&reg, None, "premium", "l4", "", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_ignores_isolated_pool_when_unpinned() {
        // Bare `gpu` traffic belongs to the default pool. A healthy
        // worker in a caller-created isolation pool must not receive
        // unpinned traffic just because it matches `(bundle, gpu)`.
        let reg = pool_registry();
        reg.update_worker(
            "http://w1:8080",
            worker_msg("default", "l4-spot", "eval-l4"),
        )
        .await;

        let out = resolve_effective_pool(&reg, None, "default", "l4-spot", "", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_provisions_when_gpu_mismatch() {
        // Client pinned `l4` but the cluster only has an `a100` worker on
        // the same bundle. Machine profile is now part of the queue lane, so
        // the gateway must return provisioning for the exact cold lane rather
        // than silently routing to a different profile.
        let reg = pool_registry();
        reg.update_worker("http://w1:8080", worker_msg("default", "a100", "default"))
            .await;
        let pm = PoolManager::new(vec!["l4".into(), "a100".into()]);
        let out = resolve_effective_pool(&reg, Some(&pm), "default", "l4", "", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
        assert_eq!(out.pending_demand_profiles, vec!["l4".to_string()]);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_lowercases_explicit_gpu_pending_demand_profiles() {
        // KEDA ScaledObject queries use lowercased machine_profile labels. Keep
        // explicit-GPU cold demand on that same label shape even when callers or
        // profile aliases pass mixed-case values through routing.
        let reg = pool_registry();
        let pm = PoolManager::new(vec!["RTX6000".into()]);

        let default_pool =
            resolve_effective_pool(&reg, Some(&pm), "sglang", "RTX6000", "", "").await;
        assert_eq!(default_pool.resolution, PoolResolution::Provisioning);
        assert_eq!(
            default_pool.pending_demand_profiles,
            vec!["rtx6000".to_string()]
        );

        let mut gpus = std::collections::HashMap::new();
        gpus.insert("RTX6000".to_string(), 0);
        pm.create_pool("tenant-a", gpus, None, None, 0, vec![])
            .await
            .unwrap();
        let pinned_pool =
            resolve_effective_pool(&reg, Some(&pm), "sglang", "RTX6000", "tenant-a", "").await;
        assert_eq!(pinned_pool.resolution, PoolResolution::Provisioning);
        assert_eq!(
            pinned_pool.pending_demand_profiles,
            vec!["rtx6000".to_string()]
        );
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_returns_pool_when_worker_matches_no_gpu() {
        // Common default-routing flow: no GPU header, one worker on the
        // requested bundle → route to its pool directly (no provisioning error).
        let reg = pool_registry();
        reg.update_worker(
            "http://w1:8080",
            worker_msg("default", "l4-spot", "default"),
        )
        .await;
        let out = resolve_effective_pool(&reg, None, "default", "", "", "").await;
        assert_eq!(out.resolution, route("default", "l4-spot"));
        // No GPU preference → no demand tracking applicable.
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_reports_exact_gpu_match_when_tuple_exists() {
        // Exact (bundle, gpu) worker exists → `exact_gpu_match` is
        // `true` and the caller skips the demand-tracking write.
        let reg = pool_registry();
        reg.update_worker(
            "http://w1:8080",
            worker_msg("default", "l4-spot", "default"),
        )
        .await;
        let out = resolve_effective_pool(&reg, None, "default", "l4-spot", "", "").await;
        assert_eq!(out.resolution, route("default", "l4-spot"));
        assert!(out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_honours_explicit_pool_name() {
        // Caller pinned only a pool. The new subject shape also needs a
        // concrete machine-profile token, so an empty registry cannot be
        // published to safely.
        let reg = pool_registry();
        let out = resolve_effective_pool(&reg, None, "default", "", "my-bench", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
        assert!(out.pending_demand_profiles.is_empty());
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_unknown_api_pool_fails_closed() {
        let reg = pool_registry();
        reg.update_worker(
            "http://w1:8080",
            worker_msg("default", "l4-spot", "customer-acme"),
        )
        .await;
        let pm = PoolManager::new(vec!["l4-spot".into()]);

        let out =
            resolve_effective_pool(&reg, Some(&pm), "default", "l4-spot", "customer-acme", "")
                .await;

        assert_eq!(
            out.resolution,
            PoolResolution::PoolNotFound("customer-acme".to_string())
        );
        assert_eq!(out.admission_pool, "customer-acme");
        assert_eq!(out.demand_pool, DEFAULT_POOL_NAME);
        assert!(out.pending_demand_profiles.is_empty());
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_infers_cold_profile_from_single_profile_pool() {
        // A tenant pool with exactly one configured machine profile can scale
        // from a pool-only request: the gateway still returns provisioning, but
        // pending_demand gets the concrete profile label KEDA queries.
        let reg = pool_registry();
        let pm = PoolManager::new(vec!["l4".into()]);
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4".to_string(), 0);
        pm.create_pool("my-bench", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let out = resolve_effective_pool(&reg, Some(&pm), "default", "", "my-bench", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
        assert_eq!(out.pending_demand_profiles, vec!["l4".to_string()]);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_fans_out_demand_across_ambiguous_pool_profiles() {
        // A pool-only cold request cannot name one profile of a multi-profile
        // pool, so demand fans out to every profile (sorted) — each candidate
        // cold lane can then scale from zero and a capable one serves the work.
        let reg = pool_registry();
        let pm = PoolManager::new(vec!["l4".into(), "a100".into()]);
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4".to_string(), 0);
        gpus.insert("a100".to_string(), 0);
        pm.create_pool("my-bench", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let out = resolve_effective_pool(&reg, Some(&pm), "default", "", "my-bench", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
        assert_eq!(
            out.pending_demand_profiles,
            vec!["a100".to_string(), "l4".to_string()]
        );
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_fans_out_default_pool_demand_when_gpu_agnostic() {
        // The scale-from-zero case from issue 1681: an unpinned, gpu-agnostic
        // request for a bundle the multi-profile default pool serves. The
        // default pool is implicit, so its cold lanes are the cluster's
        // configured profiles; with no worker yet, demand fans out to each so
        // every candidate lane can scale from zero and a capable one comes up.
        let reg = pool_registry();
        let pm = PoolManager::new(vec!["RTX6000".into(), "l4".into()]);

        let out = resolve_effective_pool(&reg, Some(&pm), "sglang", "", "", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
        // Sorted + lowercased to match the KEDA machine_profile labels.
        assert_eq!(
            out.pending_demand_profiles,
            vec!["l4".to_string(), "rtx6000".to_string()]
        );
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_provisions_explicit_pool_when_only_other_pool_exists() {
        // Explicit pool + GPU names a cold lane. A default or unrelated
        // isolation worker cannot consume the pinned pool's stream, so the
        // gateway must return a retryable provisioning error instead of
        // publishing to a lane with no active consumer.
        let reg = pool_registry();
        reg.update_worker("http://w1:8080", worker_msg("default", "l4-spot", "pool-a"))
            .await;
        let pm = PoolManager::new(vec!["l4-spot".into()]);
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        pm.create_pool("my-bench", gpus, None, None, 0, vec![])
            .await
            .unwrap();
        let out =
            resolve_effective_pool(&reg, Some(&pm), "default", "l4-spot", "my-bench", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
        assert_eq!(out.pending_demand_profiles, vec!["l4-spot".to_string()]);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_pinned_pool_reports_match_inside_same_pool() {
        let reg = pool_registry();
        reg.update_worker(
            "http://w1:8080",
            worker_msg("default", "l4-spot", "my-bench"),
        )
        .await;
        let out = resolve_effective_pool(&reg, None, "default", "l4-spot", "my-bench", "").await;
        assert_eq!(out.resolution, route("my-bench", "l4-spot"));
        assert!(out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_api_pool_routes_to_default_backing_queue() {
        let reg = pool_registry();
        reg.update_worker(
            "http://w1:8080",
            worker_msg("default", "l4-spot", DEFAULT_POOL_NAME),
        )
        .await;
        let pm = PoolManager::new(vec!["l4-spot".into()]);
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("my-bench", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let out =
            resolve_effective_pool(&reg, Some(&pm), "default", "l4-spot", "my-bench", "").await;
        assert_eq!(out.resolution, route(DEFAULT_POOL_NAME, "l4-spot"));
        assert_eq!(out.admission_pool, "my-bench");
        assert_eq!(out.demand_pool, DEFAULT_POOL_NAME);
        assert!(out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_pool_only_respects_single_profile_api_pool() {
        let reg = pool_registry();
        reg.update_worker(
            "http://w1:8080",
            worker_msg("default", "a100", DEFAULT_POOL_NAME),
        )
        .await;
        let pm = PoolManager::new(vec!["l4".into(), "a100".into()]);
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4".to_string(), 1);
        pm.create_pool("tenant-l4", gpus, None, None, 0, vec![])
            .await
            .unwrap();

        let out = resolve_effective_pool(&reg, Some(&pm), "default", "", "tenant-l4", "").await;

        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert_eq!(out.admission_pool, "tenant-l4");
        assert_eq!(out.demand_pool, DEFAULT_POOL_NAME);
        assert_eq!(out.pending_demand_profiles, vec!["l4".to_string()]);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_api_pool_routes_to_custom_backing_queue() {
        let reg = pool_registry();
        reg.update_worker(
            "http://w1:8080",
            worker_msg("default", "l4-spot", "customer-queue"),
        )
        .await;
        let pm = PoolManager::new(vec!["l4-spot".into()]);
        sync_static_queue_pool(&pm, "customer-queue", "l4-spot").await;
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool_with_caps_on_queue(
            "tenant-a",
            "customer-queue",
            gpus,
            std::collections::HashMap::new(),
            None,
            None,
            0,
            vec![],
        )
        .await
        .unwrap();

        let out =
            resolve_effective_pool(&reg, Some(&pm), "default", "l4-spot", "tenant-a", "").await;
        assert_eq!(out.resolution, route("customer-queue", "l4-spot"));
        assert_eq!(out.admission_pool, "tenant-a");
        assert_eq!(out.demand_pool, "customer-queue");
        assert!(out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_api_pool_uses_custom_backing_queue_for_cold_demand() {
        let reg = pool_registry();
        let pm = PoolManager::new(vec!["l4-spot".into()]);
        sync_static_queue_pool(&pm, "customer-queue", "l4-spot").await;
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool_with_caps_on_queue(
            "tenant-a",
            "customer-queue",
            gpus,
            std::collections::HashMap::new(),
            None,
            None,
            0,
            vec![],
        )
        .await
        .unwrap();

        let out =
            resolve_effective_pool(&reg, Some(&pm), "default", "l4-spot", "tenant-a", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert_eq!(out.admission_pool, "tenant-a");
        assert_eq!(out.demand_pool, "customer-queue");
        assert_eq!(out.pending_demand_profiles, vec!["l4-spot".to_string()]);
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_pinned_pool_with_missing_gpu_tuple_reports_no_match() {
        // Caller pinned a pool AND expressed a GPU preference, but no
        // worker matches the exact tuple. Return provisioning and record
        // demand for KEDA instead of publishing into a cold stream.
        let reg = pool_registry();
        reg.update_worker("http://w1:8080", worker_msg("default", "a100", "pool-a"))
            .await;
        let pm = PoolManager::new(vec!["l4".into(), "a100".into()]);
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4".to_string(), 0);
        pm.create_pool("my-bench", gpus, None, None, 0, vec![])
            .await
            .unwrap();
        let out = resolve_effective_pool(&reg, Some(&pm), "default", "l4", "my-bench", "").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
        assert_eq!(out.pending_demand_profiles, vec!["l4".to_string()]);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_provisions_when_hash_is_stale() {
        let reg = pool_registry();
        reg.update_worker(
            "http://w1:8080",
            worker_msg("default", "l4-spot", "default"),
        )
        .await;
        let out = resolve_effective_pool(&reg, None, "default", "l4-spot", "", "new-hash").await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
    }

    #[tokio::test]
    async fn test_resolve_effective_pool_pinned_pool_provisions_when_hash_mismatch() {
        let reg = pool_registry();
        reg.update_worker(
            "http://w1:8080",
            worker_msg("default", "l4-spot", "default"),
        )
        .await;
        let pm = PoolManager::new(vec!["l4-spot".into()]);
        let mut gpus = std::collections::HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        pm.create_pool("my-bench", gpus, None, None, 0, vec![])
            .await
            .unwrap();
        let out = resolve_effective_pool(
            &reg,
            Some(&pm),
            "default",
            "l4-spot",
            "my-bench",
            "new-hash",
        )
        .await;
        assert_eq!(out.resolution, PoolResolution::Provisioning);
        assert!(!out.exact_gpu_match);
        assert_eq!(out.pending_demand_profiles, vec!["l4-spot".to_string()]);
    }

    // ── grammar routing: dispatch vs display model ─────────────────
    //
    // The chat (`proxy_chat`) and native generate (`queue_mode_proxy`)
    // handlers route a grammar-constrained request to the model's declared
    // `:no-spec` `grammar_profile` variant for DISPATCH (NATS subject + work
    // item) while keeping the requested base id for DISPLAY (response body,
    // `record_generation_success`, audit log). Both read
    // `info_extras.grammar_profile` off the resolved model, then call
    // `route_grammar_to_profile`. These tests exercise that exact composition
    // against a live `ModelRegistry`, one level above the
    // `ModelRegistry::grammar_route_variant` unit test (which checks the
    // registry lookup in isolation). A full handler call cannot observe the
    // dispatch id — it is internal to the NATS publish, which the queue-only
    // gateway skips when `work_publisher` is `None`.

    /// An empty `ModelRegistry` with a `default` bundle advertising the
    /// `sentence_transformer` adapter — callers add their model configs. The
    /// temp dirs are read at construction only, so they can drop here.
    fn empty_registry() -> crate::state::model_registry::ModelRegistry {
        let bundles_dir = tempfile::TempDir::new().unwrap();
        let models_dir = tempfile::TempDir::new().unwrap();
        std::fs::write(
            bundles_dir.path().join("default.yaml"),
            "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
        )
        .unwrap();
        crate::state::model_registry::ModelRegistry::new(
            bundles_dir.path(),
            models_dir.path(),
            true,
        )
    }

    #[tokio::test]
    async fn non_streaming_model_rejects_stream_with_exact_parameter() {
        use crate::types::model::{ModelConfig, ProfileConfig};

        let registry = empty_registry();
        let profiles = [(
            "default".to_string(),
            ProfileConfig {
                kv_budget_tokens: None,
                max_output_tokens: None,
                grammar_profile: None,
                chat_template_kwargs: None,
                adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        )]
        .into_iter()
        .collect();
        let tasks =
            serde_yaml::from_str("generate:\n  capabilities:\n    streaming: false\n").unwrap();
        registry
            .add_model_config(ModelConfig {
                name: "org/buffered".to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: Some(tasks),
            })
            .unwrap();

        let response =
            unsupported_streaming_response(&registry, "org/buffered", "org/buffered", true)
                .expect("streaming must be rejected");

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], oai_code::UNSUPPORTED_FIELD);
        assert_eq!(value["error"]["param"], "stream");
        assert!(
            unsupported_streaming_response(&registry, "org/buffered", "org/buffered", false,)
                .is_none()
        );
    }

    fn grammar_routed_registry() -> crate::state::model_registry::ModelRegistry {
        use crate::types::model::{ModelConfig, ProfileConfig};
        use std::collections::HashMap as StdHashMap;

        let registry = empty_registry();
        let mk = || ProfileConfig {
            kv_budget_tokens: None,
            max_output_tokens: None,
            grammar_profile: None,
            chat_template_kwargs: None,
            adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
            max_batch_tokens: Some(4096),
            compute_precision: None,
            adapter_options: None,
            extends: None,
        };
        let mut profiles = StdHashMap::new();
        profiles.insert("default".to_string(), mk());
        // A non-`default` profile expands into the `org/g:no-spec` variant
        // entry that grammar routing dispatches to.
        let mut no_spec = mk();
        no_spec.adapter_options = Some(serde_json::json!({
            "loadtime": {"speculative": {"enabled": false}},
        }));
        profiles.insert("no-spec".to_string(), no_spec);
        // `tasks.generate.grammar_profile: no-spec` is what surfaces on
        // `info_extras.grammar_profile` — the field the handlers read.
        let tasks: serde_yaml::Value = serde_yaml::from_str(
            "generate:\n  grammar_profile: no-spec\n  capabilities:\n    grammar: [json_schema]\n",
        )
        .unwrap();
        registry
            .add_model_config(ModelConfig {
                name: "org/g".to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: Some(tasks),
            })
            .unwrap();
        registry
    }

    /// Mirror the handler routing step: route the requested model in place.
    /// The variant is resolved off the base model inside
    /// `route_grammar_to_profile`. Returns `(display_id, dispatch_id)`.
    fn route_like_handler(
        registry: &crate::state::model_registry::ModelRegistry,
        requested: &str,
    ) -> (String, String) {
        let display = requested.to_string();
        let mut dispatch = display.clone();
        route_grammar_to_profile(registry, &mut dispatch);
        (display, dispatch)
    }

    #[test]
    fn test_grammar_request_dispatches_variant_display_stays_base() {
        let registry = grammar_routed_registry();
        // Precondition: the declared profile surfaces for the handlers to read.
        assert_eq!(
            registry
                .get_model_info("org/g")
                .and_then(|m| m.info_extras.grammar_profile.clone())
                .as_deref(),
            Some("no-spec"),
        );

        let (display, dispatch) = route_like_handler(&registry, "org/g");
        // DISPATCH id is rewritten to the `:no-spec` variant; the DISPLAY id
        // (response body / metrics / audit) stays the requested base model.
        assert_eq!(dispatch, "org/g:no-spec");
        assert_eq!(display, "org/g");
    }

    #[test]
    fn test_grammar_request_on_variant_id_does_not_double_route() {
        // A request that already names the variant must not re-route to
        // `org/g:no-spec:no-spec`: the variant entry clears its own
        // `grammar_profile` routing hint, so display == dispatch.
        let registry = grammar_routed_registry();
        let (display, dispatch) = route_like_handler(&registry, "org/g:no-spec");
        assert_eq!(dispatch, "org/g:no-spec");
        assert_eq!(display, "org/g:no-spec");
    }

    #[test]
    fn test_non_grammar_model_keeps_base_dispatch() {
        // A model that declares no `grammar_profile` never rewrites: the
        // dispatch id equals the display id.
        use crate::types::model::{ModelConfig, ProfileConfig};
        use std::collections::HashMap as StdHashMap;

        let registry = grammar_routed_registry();
        let mut profiles = StdHashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                kv_budget_tokens: None,
                max_output_tokens: None,
                grammar_profile: None,
                chat_template_kwargs: None,
                adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        registry
            .add_model_config(ModelConfig {
                name: "org/plain".to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: None,
            })
            .unwrap();

        let (display, dispatch) = route_like_handler(&registry, "org/plain");
        assert_eq!(dispatch, "org/plain");
        assert_eq!(display, "org/plain");
    }

    #[test]
    fn test_grammar_gate_validates_against_narrowed_variant_adapters() {
        // The gate-ordering fix routes a grammar request to the `:no-spec`
        // variant BEFORE the LoRA gate runs, so the gate validates against the
        // variant's *narrowed* per-profile allow-list, not the base. Here the
        // `default` profile serves adapter `a1` and `no-spec` serves `a2`. A
        // grammar request dispatches to `org/g2:no-spec`, whose allow-list must
        // REJECT `a1` (served only by the base default profile) and accept `a2`.
        // Gating the base — the pre-fix behaviour — would wrongly accept `a1`.
        use crate::types::model::{ModelConfig, ProfileConfig};
        use std::collections::HashMap as StdHashMap;

        let registry = empty_registry();
        let default_profile = ProfileConfig {
            kv_budget_tokens: None,
            max_output_tokens: None,
            grammar_profile: None,
            chat_template_kwargs: None,
            adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
            max_batch_tokens: Some(4096),
            compute_precision: None,
            adapter_options: Some(serde_json::json!({
                "loadtime": { "lora_paths": { "a1": "org/a1" } }
            })),
            extends: None,
        };
        let nospec_profile = ProfileConfig {
            kv_budget_tokens: None,
            max_output_tokens: None,
            grammar_profile: None,
            chat_template_kwargs: None,
            adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
            max_batch_tokens: Some(4096),
            compute_precision: None,
            adapter_options: Some(serde_json::json!({
                "loadtime": {
                    "lora_paths": { "a2": "org/a2" },
                    "speculative": { "enabled": false }
                }
            })),
            extends: None,
        };
        let mut profiles = StdHashMap::new();
        profiles.insert("default".to_string(), default_profile);
        profiles.insert("no-spec".to_string(), nospec_profile);
        let tasks: serde_yaml::Value =
            serde_yaml::from_str("generate:\n  grammar_profile: no-spec\n").unwrap();
        registry
            .add_model_config(ModelConfig {
                name: "org/g2".to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: Some(tasks),
            })
            .unwrap();

        let (display, dispatch) = route_like_handler(&registry, "org/g2");
        assert_eq!(display, "org/g2");
        assert_eq!(dispatch, "org/g2:no-spec");

        let base_info = registry.get_model_info(&display).unwrap();
        let variant_info = registry.get_model_info(&dispatch).unwrap();
        // Base default profile serves `a1` — gating the base would accept it.
        assert!(matches!(
            validate_lora_for_profile(&base_info, "default", "a1"),
            LoraValidation::Ok
        ));
        // The routed variant's narrowed allow-list rejects `a1`, accepts `a2`.
        assert!(matches!(
            validate_lora_for_profile(&variant_info, "default", "a1"),
            LoraValidation::UnknownAdapter
        ));
        assert!(matches!(
            validate_lora_for_profile(&variant_info, "default", "a2"),
            LoraValidation::Ok
        ));
    }

    #[test]
    fn test_grammar_variant_absent_degrades_to_base() {
        // Safety contract: a declared `grammar_profile` whose variant is not in
        // the registry degrades to the base model (never hang / 5xx). Here
        // `grammar_profile: ghost` names a profile that is not defined, so no
        // `org/g3:ghost` variant entry is minted and routing keeps the base.
        use crate::types::model::{ModelConfig, ProfileConfig};
        use std::collections::HashMap as StdHashMap;

        let registry = empty_registry();
        let mut profiles = StdHashMap::new();
        profiles.insert(
            "default".to_string(),
            ProfileConfig {
                kv_budget_tokens: None,
                max_output_tokens: None,
                grammar_profile: None,
                chat_template_kwargs: None,
                adapter_path: Some("sie_server.adapters.sentence_transformer:Adapter".to_string()),
                max_batch_tokens: Some(4096),
                compute_precision: None,
                adapter_options: None,
                extends: None,
            },
        );
        let tasks: serde_yaml::Value =
            serde_yaml::from_str("generate:\n  grammar_profile: ghost\n").unwrap();
        registry
            .add_model_config(ModelConfig {
                name: "org/g3".to_string(),
                hf_revision: None,
                adapter_module: None,
                default_bundle: None,
                pool: None,
                profiles,
                inputs: None,
                max_sequence_length: None,
                tasks: Some(tasks),
            })
            .unwrap();

        // The declared (but undefined) profile is still surfaced...
        assert_eq!(
            registry
                .get_model_info("org/g3")
                .and_then(|m| m.info_extras.grammar_profile.clone())
                .as_deref(),
            Some("ghost"),
        );
        // ...yet routing degrades to base because `org/g3:ghost` does not exist.
        let (display, dispatch) = route_like_handler(&registry, "org/g3");
        assert_eq!(display, "org/g3");
        assert_eq!(dispatch, "org/g3");
    }

    // ── /v1/chat/completions parsing + body composition ────────────

    fn _chat_body_min(model: &str) -> serde_json::Value {
        serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 32,
        })
    }

    fn _expect_chat_ok(body: serde_json::Value) -> ChatRequestParams {
        match chat_params_from_json(&body) {
            ChatParamsResult::Ok(p) => p,
            ChatParamsResult::Err(resp) => {
                panic!("expected Ok, got error response: {:?}", resp.status())
            }
        }
    }

    async fn _expect_chat_err(body: serde_json::Value) -> serde_json::Value {
        let resp = match chat_params_from_json(&body) {
            ChatParamsResult::Ok(_) => panic!("expected Err, got Ok"),
            ChatParamsResult::Err(r) => r,
        };
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .expect("read body");
        serde_json::from_slice(&body).expect("body is JSON")
    }

    #[test]
    fn test_chat_params_from_json_happy_path() {
        let p = _expect_chat_ok(_chat_body_min("Qwen/Qwen3-4B-Instruct-2507"));
        assert_eq!(p.model, "Qwen/Qwen3-4B-Instruct-2507");
        assert_eq!(p.messages.len(), 1);
        assert_eq!(p.messages[0].role, "user");
        assert_eq!(p.messages[0].content, "Hi");
        assert_eq!(p.max_new_tokens, 32);
    }

    // ── chat response_format regex/EBNF + developer role (roadmap 1.7) ──

    #[test]
    fn test_chat_response_format_regex_builds_regex_grammar() {
        let mut body = _chat_body_min("m");
        body["response_format"] = serde_json::json!({"type": "regex", "regex": "^(yes|no)$"});
        let p = _expect_chat_ok(body);
        let Some(publisher::GrammarSpec::Regex { value, .. }) = p.grammar else {
            panic!("expected Regex grammar");
        };
        assert_eq!(value, "^(yes|no)$");
    }

    #[test]
    fn test_chat_response_format_grammar_builds_ebnf_grammar() {
        let mut body = _chat_body_min("m");
        body["response_format"] = serde_json::json!({
            "type": "grammar",
            "grammar": "root ::= \"red\" | \"green\"",
            "syntax": "ebnf",
        });
        let p = _expect_chat_ok(body);
        let Some(publisher::GrammarSpec::Ebnf { value, .. }) = p.grammar else {
            panic!("expected Ebnf grammar");
        };
        assert!(value.contains("root ::="));
    }

    #[test]
    fn test_chat_response_format_grammar_defaults_syntax_to_ebnf() {
        let mut body = _chat_body_min("m");
        body["response_format"] =
            serde_json::json!({"type": "grammar", "grammar": "root ::= \"x\""});
        let p = _expect_chat_ok(body);
        assert!(matches!(
            p.grammar,
            Some(publisher::GrammarSpec::Ebnf { .. })
        ));
    }

    #[tokio::test]
    async fn test_chat_response_format_regex_requires_regex_field() {
        let mut body = _chat_body_min("m");
        body["response_format"] = serde_json::json!({"type": "regex"});
        let err = _expect_chat_err(body).await;
        assert_eq!(err["error"]["param"], "response_format.regex");
        assert_eq!(err["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_chat_response_format_regex_rejects_overlength() {
        let mut body = _chat_body_min("m");
        let big = "a".repeat(crate::handlers::grammar::MAX_REGEX_LEN + 1);
        body["response_format"] = serde_json::json!({"type": "regex", "regex": big});
        let err = _expect_chat_err(body).await;
        assert_eq!(err["error"]["param"], "grammar.regex");
    }

    #[tokio::test]
    async fn test_chat_response_format_grammar_rejects_bad_syntax() {
        let mut body = _chat_body_min("m");
        body["response_format"] = serde_json::json!({
            "type": "grammar", "grammar": "root ::= \"x\"", "syntax": "peg",
        });
        let err = _expect_chat_err(body).await;
        assert_eq!(err["error"]["param"], "response_format.syntax");
    }

    #[tokio::test]
    async fn test_chat_response_format_unknown_type_still_rejected() {
        let mut body = _chat_body_min("m");
        body["response_format"] = serde_json::json!({"type": "yaml"});
        let err = _expect_chat_err(body).await;
        assert_eq!(err["error"]["code"], "unsupported_field");
    }

    #[test]
    fn test_chat_params_normalizes_developer_role_to_system() {
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "developer", "content": "You are terse."},
            {"role": "user", "content": "Hi"},
        ]);
        let p = _expect_chat_ok(body);
        assert_eq!(p.messages[0].role, "system");
        assert_eq!(p.messages[0].content, "You are terse.");
        assert_eq!(p.messages[1].role, "user");
    }

    #[tokio::test]
    async fn test_chat_params_rejects_unknown_role() {
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([{"role": "wizard", "content": "Hi"}]);
        let err = _expect_chat_err(body).await;
        assert_eq!(err["error"]["code"], "invalid_request");
    }

    /// ``tools`` with the OpenAI shape parses and surfaces on
    /// :class:`ChatRequestParams` so the chat handler can plumb it
    /// into ``GenerateParams``.
    #[test]
    fn test_chat_params_from_json_accepts_tools() {
        let mut body = _chat_body_min("m");
        body["tools"] = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Lookup weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]);
        let p = _expect_chat_ok(body);
        assert!(p.tools.is_some());
        assert_eq!(p.tools.as_ref().unwrap().len(), 1);
    }

    /// ``tools[].function.parameters`` runs through the JSON-Schema
    /// safety walker — a schema with ``$ref`` is rejected with a 400
    /// keyed on the offending tool index.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_invalid_tool_schema() {
        let mut body = _chat_body_min("m");
        body["tools"] = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"$ref": "#/$defs/Foo"}},
                },
            },
        }]);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        let param = v["error"]["param"].as_str().unwrap_or("");
        assert!(
            param.starts_with("tools[0].function.parameters"),
            "expected param path under tools[0].function.parameters, got {param}"
        );
    }

    /// Empty ``tools`` array — OpenAI returns 400 in that case.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_empty_tools() {
        let mut body = _chat_body_min("m");
        body["tools"] = serde_json::json!([]);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "tools");
    }

    /// The legacy ``functions`` / ``function_call`` keys are still
    /// rejected with a deprecation hint pointing at the new ``tools``
    /// / ``tool_choice`` keys.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_legacy_function_call() {
        let mut body = _chat_body_min("m");
        body["functions"] = serde_json::json!([{"name": "f"}]);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "unsupported_field");
        assert_eq!(v["error"]["param"], "functions");
        let msg = v["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("tools"), "msg should mention tools: {msg}");
    }

    /// ``tool_choice`` accepts the three literal strings and the
    /// structured ``{type: function, function: {name}}`` shape; rejects
    /// unknown variants.
    #[test]
    fn test_chat_params_from_json_validates_tool_choice() {
        // Each variant requires ``tools`` to be present.
        let tools = serde_json::json!([{
            "type": "function",
            "function": {"name": "f"},
        }]);
        for choice in ["auto", "none", "required"] {
            let mut body = _chat_body_min("m");
            body["tools"] = tools.clone();
            body["tool_choice"] = serde_json::json!(choice);
            let p = _expect_chat_ok(body);
            assert_eq!(p.tool_choice.as_ref().unwrap(), &serde_json::json!(choice));
        }
        let mut body = _chat_body_min("m");
        body["tools"] = tools.clone();
        body["tool_choice"] = serde_json::json!({
            "type": "function",
            "function": {"name": "f"},
        });
        let p = _expect_chat_ok(body);
        assert!(p.tool_choice.is_some());
    }

    #[tokio::test]
    async fn test_chat_params_from_json_rejects_unknown_tool_choice_string() {
        let mut body = _chat_body_min("m");
        body["tools"] = serde_json::json!([{
            "type": "function",
            "function": {"name": "f"},
        }]);
        body["tool_choice"] = serde_json::json!("sometimes");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "tool_choice");
    }

    #[tokio::test]
    async fn test_chat_params_from_json_rejects_tool_choice_without_tools() {
        let mut body = _chat_body_min("m");
        body["tool_choice"] = serde_json::json!("auto");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "tool_choice");
    }

    #[tokio::test]
    async fn test_chat_params_rejects_forced_tool_choice_with_response_format() {
        // tool_choice="required" and response_format both constrain
        // decoding; the combination is rejected up front.
        let mut body = _chat_body_min("m");
        body["tools"] = serde_json::json!([{ "type": "function", "function": {"name": "f"} }]);
        body["tool_choice"] = serde_json::json!("required");
        body["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}}
        });
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "tool_choice");
    }

    #[tokio::test]
    async fn test_chat_params_rejects_named_tool_choice_with_response_format() {
        let mut body = _chat_body_min("m");
        body["tools"] = serde_json::json!([{ "type": "function", "function": {"name": "f"} }]);
        body["tool_choice"] = serde_json::json!({"type": "function", "function": {"name": "f"}});
        body["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}}
        });
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "tool_choice");
    }

    #[test]
    fn test_chat_params_allows_auto_tool_choice_with_response_format() {
        // "auto"/"none" don't force a grammar, so they're compatible
        // with response_format.
        let mut body = _chat_body_min("m");
        body["tools"] = serde_json::json!([{ "type": "function", "function": {"name": "f"} }]);
        body["tool_choice"] = serde_json::json!("auto");
        body["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}}
        });
        let p = _expect_chat_ok(body);
        assert!(p.tool_choice.is_some());
        assert!(p.grammar.is_some());
    }

    #[test]
    fn test_chat_params_from_json_accepts_parallel_tool_calls() {
        let mut body = _chat_body_min("m");
        body["tools"] = serde_json::json!([{
            "type": "function",
            "function": {"name": "f"},
        }]);
        body["parallel_tool_calls"] = serde_json::json!(false);
        let p = _expect_chat_ok(body);
        assert_eq!(p.parallel_tool_calls, Some(false));
    }

    #[tokio::test]
    async fn test_chat_params_from_json_rejects_non_bool_parallel_tool_calls() {
        let mut body = _chat_body_min("m");
        body["parallel_tool_calls"] = serde_json::json!("yes");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "parallel_tool_calls");
    }

    /// Penalties in range parse and surface on :class:`ChatRequestParams`
    /// so the chat handler can plumb them into ``GenerateParams``.
    #[test]
    fn test_chat_params_from_json_accepts_penalties_in_range() {
        let mut body = _chat_body_min("m");
        body["frequency_penalty"] = serde_json::json!(0.5);
        body["presence_penalty"] = serde_json::json!(-1.5);
        let p = _expect_chat_ok(body);
        assert_eq!(p.frequency_penalty, Some(0.5));
        assert_eq!(p.presence_penalty, Some(-1.5));
    }

    #[test]
    fn test_chat_params_from_json_accepts_zero_penalties() {
        let mut body = _chat_body_min("m");
        body["frequency_penalty"] = serde_json::json!(0);
        body["presence_penalty"] = serde_json::json!(0.0);
        let p = _expect_chat_ok(body);
        assert_eq!(p.frequency_penalty, Some(0.0));
        assert_eq!(p.presence_penalty, Some(0.0));
    }

    /// Out-of-range high — OpenAI's contract is ``[-2.0, 2.0]``; values
    /// above 2.0 surface as 400 invalid_request with the offending param.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_penalty_above_range() {
        let mut body = _chat_body_min("m");
        body["frequency_penalty"] = serde_json::json!(2.5);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "frequency_penalty");
        let msg = v["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("[-2.0, 2.0]"), "msg: {msg}");
    }

    /// Out-of-range low.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_penalty_below_range() {
        let mut body = _chat_body_min("m");
        body["presence_penalty"] = serde_json::json!(-2.1);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "presence_penalty");
    }

    /// Non-numeric penalty rejects with the same param attribution.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_non_numeric_penalty() {
        let mut body = _chat_body_min("m");
        body["frequency_penalty"] = serde_json::json!("nope");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "frequency_penalty");
    }

    /// ``top_k`` (integer ≥ 1) and ``repetition_penalty`` (float in
    /// ``(0.0, 2.0]``) are common non-OpenAI sampling knobs (Together /
    /// Fireworks / vLLM). They parse and surface on
    /// :class:`ChatRequestParams` so the chat handler plumbs them into
    /// ``GenerateParams``.
    #[test]
    fn test_chat_params_from_json_accepts_top_k_and_repetition_penalty() {
        let mut body = _chat_body_min("m");
        body["top_k"] = serde_json::json!(10);
        body["repetition_penalty"] = serde_json::json!(1.1);
        let p = _expect_chat_ok(body);
        assert_eq!(p.top_k, Some(10));
        assert_eq!(p.repetition_penalty, Some(1.1));
    }

    /// ``top_k`` below 1 is meaningless — reject with param attribution.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_top_k_below_one() {
        let mut body = _chat_body_min("m");
        body["top_k"] = serde_json::json!(0);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "top_k");
    }

    /// Non-integer ``top_k`` rejects.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_non_integer_top_k() {
        let mut body = _chat_body_min("m");
        body["top_k"] = serde_json::json!("nope");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "top_k");
    }

    /// ``min_tokens`` non-negative integer parses through.
    #[test]
    fn test_chat_params_from_json_accepts_min_tokens() {
        let mut body = _chat_body_min("m");
        body["min_tokens"] = serde_json::json!(10);
        let p = _expect_chat_ok(body);
        assert_eq!(p.min_tokens, Some(10));
    }

    /// ``min_tokens`` negative rejects with param attribution.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_negative_min_tokens() {
        let mut body = _chat_body_min("m");
        body["min_tokens"] = serde_json::json!(-1);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "min_tokens");
    }

    /// ``chat_template_kwargs`` as an object parses and round-trips.
    #[test]
    fn test_chat_params_from_json_accepts_chat_template_kwargs_object() {
        let mut body = _chat_body_min("m");
        body["chat_template_kwargs"] = serde_json::json!({
            "enable_thinking": false,
            "guardian_config": {"risk_name": "harm"}
        });
        let p = _expect_chat_ok(body);
        assert_eq!(
            p.chat_template_kwargs
                .as_ref()
                .and_then(|v| v.get("enable_thinking")),
            Some(&serde_json::json!(false))
        );
    }

    /// ``chat_template_kwargs`` non-object rejects.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_non_object_chat_template_kwargs() {
        let mut body = _chat_body_min("m");
        body["chat_template_kwargs"] = serde_json::json!("oops");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "chat_template_kwargs");
    }

    /// Arbitrary tokenizer kwargs reject instead of crossing the queue.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_unknown_chat_template_kwarg() {
        let mut body = _chat_body_min("m");
        body["chat_template_kwargs"] = serde_json::json!({"arbitrary_tokenizer_kwarg": true});
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "unsupported_field");
        assert_eq!(
            v["error"]["param"],
            "chat_template_kwargs.arbitrary_tokenizer_kwarg"
        );
    }

    /// Accepted template kwargs keep their declared scalar types.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_non_boolean_enable_thinking() {
        let mut body = _chat_body_min("m");
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": "false"});
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "chat_template_kwargs.enable_thinking");
    }

    /// Granite Guardian config is closed to its one declared risk selector.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_unknown_guardian_config_kwarg() {
        let mut body = _chat_body_min("m");
        body["chat_template_kwargs"] = serde_json::json!({
            "guardian_config": {"risk_name": "harm", "extra": true}
        });
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "unsupported_field");
        assert_eq!(
            v["error"]["param"],
            "chat_template_kwargs.guardian_config.extra"
        );
    }

    /// ``repetition_penalty`` outside ``(0.0, 2.0]`` rejects (both ends).
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_repetition_penalty_out_of_range() {
        let mut body = _chat_body_min("m");
        body["repetition_penalty"] = serde_json::json!(3.0);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "repetition_penalty");

        let mut body0 = _chat_body_min("m");
        body0["repetition_penalty"] = serde_json::json!(0);
        let v0 = _expect_chat_err(body0).await;
        assert_eq!(v0["error"]["param"], "repetition_penalty");
    }

    /// ``logprobs: true`` is accepted (request parses, no field surfaced
    /// on :class:`ChatRequestParams`) — the worker does not emit
    /// logprobs, so the request is silently honoured by ignoring the
    /// knob. Same for ``top_logprobs``.
    #[test]
    fn test_chat_params_from_json_accepts_logprobs_true() {
        let mut body = _chat_body_min("m");
        body["logprobs"] = serde_json::json!(true);
        body["top_logprobs"] = serde_json::json!(5);
        let _p = _expect_chat_ok(body);
    }

    /// ``logprobs: false`` and ``top_logprobs: 0`` (the defaults) also
    /// parse cleanly — guards against an over-zealous future refactor
    /// that requires the field to be absent.
    #[test]
    fn test_chat_params_from_json_accepts_logprobs_false() {
        let mut body = _chat_body_min("m");
        body["logprobs"] = serde_json::json!(false);
        body["top_logprobs"] = serde_json::json!(0);
        let _p = _expect_chat_ok(body);
    }

    // ── response_format translation ────────────────────────────────

    /// Loose JSON mode (``json_object``) is now accepted and
    /// translated to a built-in generic JSON schema. The resulting
    /// grammar is labelled ``"json_object"`` so cache observability
    /// surfaces the loose mode distinctly from caller-supplied
    /// schemas.
    #[test]
    fn test_chat_params_from_json_response_format_json_object_accepts() {
        let mut body = _chat_body_min("m");
        body["response_format"] = serde_json::json!({"type": "json_object"});
        let p = _expect_chat_ok(body);
        match p.grammar {
            Some(publisher::GrammarSpec::JsonSchema {
                ref value,
                ref label,
                strict,
            }) => {
                assert_eq!(label.as_deref(), Some("json_object"));
                assert_eq!(strict, None);
                assert_eq!(value["type"], "object");
                assert_eq!(value["additionalProperties"], true);
            }
            other => panic!("expected JsonSchema grammar with json_object label, got {other:?}"),
        }
    }

    #[test]
    fn test_chat_params_from_json_response_format_json_schema_parses() {
        let mut body = _chat_body_min("m");
        body["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "math_response",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "number"}},
                    "required": ["answer"],
                },
            }
        });
        let p = _expect_chat_ok(body);
        match p.grammar {
            Some(publisher::GrammarSpec::JsonSchema { label, strict, .. }) => {
                assert_eq!(label.as_deref(), Some("math_response"));
                assert_eq!(strict, Some(true));
            }
            other => panic!("expected JsonSchema grammar, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_chat_params_from_json_response_format_missing_schema_returns_400() {
        let mut body = _chat_body_min("m");
        body["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {"name": "x", "strict": true},
        });
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "response_format.json_schema.schema");
    }

    #[tokio::test]
    async fn test_chat_params_from_json_response_format_unknown_type_returns_400() {
        let mut body = _chat_body_min("m");
        body["response_format"] = serde_json::json!({"type": "ebnf"});
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "unsupported_field");
        assert_eq!(v["error"]["param"], "response_format.type");
    }

    /// The shared safety walker applies to the chat translator too, after
    /// internal ``$ref`` entries inside ``json_schema.schema`` are inlined.
    #[test]
    fn test_chat_params_from_json_response_format_dereferences_dollar_ref() {
        let mut body = _chat_body_min("m");
        body["response_format"] = serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "$defs": {
                        "Step": {
                            "type": "object",
                            "properties": {
                                "explanation": {"type": "string"},
                                "output": {"type": "string"}
                            },
                            "required": ["explanation", "output"],
                            "additionalProperties": false
                        }
                    },
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/Step"}
                        },
                        "single": {
                            "$ref": "#/$defs/Step",
                            "description": "one step"
                        },
                        "final_answer": {"type": "string"}
                    },
                    "required": ["steps", "single", "final_answer"],
                    "additionalProperties": false
                },
            }
        });
        let p = _expect_chat_ok(body);
        match p.grammar {
            Some(publisher::GrammarSpec::JsonSchema { value, .. }) => {
                assert_eq!(
                    value["properties"]["steps"]["items"]["properties"]["explanation"]["type"],
                    "string"
                );
                assert_eq!(
                    value["properties"]["steps"]["items"]["additionalProperties"],
                    false
                );
                let single_all_of = value["properties"]["single"]["allOf"]
                    .as_array()
                    .expect("Pydantic ref sibling should be preserved as allOf");
                assert_eq!(
                    single_all_of[0]["properties"]["explanation"]["type"],
                    "string"
                );
                assert_eq!(single_all_of[1]["description"], "one step");
                assert_eq!(value["additionalProperties"], false);
                let encoded = serde_json::to_string(&value).unwrap();
                assert!(
                    !encoded.contains("\"$ref\"") && !encoded.contains("\"$defs\""),
                    "schema should be dereferenced: {encoded}"
                );
            }
            other => panic!("expected JsonSchema grammar, got {other:?}"),
        }
    }

    #[test]
    fn test_chat_params_from_json_max_completion_tokens_takes_precedence() {
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            "max_completion_tokens": 64,
        });
        let p = _expect_chat_ok(body);
        assert_eq!(p.max_new_tokens, 64);
    }

    #[test]
    fn test_chat_params_from_json_falls_back_to_max_tokens() {
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
        });
        let p = _expect_chat_ok(body);
        assert_eq!(p.max_new_tokens, 16);
    }

    #[test]
    fn test_chat_params_from_json_defaults_max_tokens_when_absent() {
        // OpenAI-compat: a request that omits BOTH max_completion_tokens
        // and max_tokens must NOT 400 — it defaults. Generic clients
        // (Open WebUI, the openai SDK with no explicit cap) rely on this.
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "Hi"}],
        });
        let p = _expect_chat_ok(body);
        // Assert the no-token path resolved to the configured default
        // rather than hard-coding the literal, so the test tracks the
        // single source of truth in `default_max_tokens()`.
        assert_eq!(p.max_new_tokens as u64, default_max_tokens());
        assert!(p.max_new_tokens > 0);
    }

    #[tokio::test]
    async fn test_chat_params_from_json_rejects_invalid_role() {
        // ``tool`` is now a valid role (multi-turn tool use); use a
        // genuinely-unknown role to exercise the rejection path.
        let body = serde_json::json!({
            "model": "m",
            "messages": [{"role": "function", "content": "noop"}],
            "max_completion_tokens": 8,
        });
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "messages[0].role");
    }

    #[tokio::test]
    async fn test_chat_params_from_json_accepts_tool_roundtrip() {
        // assistant message with tool_calls (content null) + a tool
        // result message must both parse.
        let body = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "weather in Tokyo?"},
                {"role": "assistant", "content": serde_json::Value::Null,
                 "tool_calls": [{"id": "call_1", "type": "function",
                   "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp_c\":18}"}
            ],
            "max_completion_tokens": 8,
        });
        let p = _expect_chat_ok(body);
        assert_eq!(p.messages.len(), 3);
        assert!(p.messages[1].tool_calls.is_some());
        assert_eq!(p.messages[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[tokio::test]
    async fn test_chat_params_from_json_rejects_unknown_field() {
        let mut body = _chat_body_min("m");
        body["wat"] = serde_json::json!(true);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "unsupported_field");
        assert_eq!(v["error"]["param"], "wat");
    }

    /// Phase: SSE streaming — ``stream: true`` is now accepted and
    /// round-trips into :attr:`ChatRequestParams.stream`. The handler
    /// branches to the SSE response builder on this flag. Regression
    /// guard against a future refactor that drops the field.
    #[test]
    fn test_chat_params_from_json_accepts_stream_true() {
        let mut body = _chat_body_min("m");
        body["stream"] = serde_json::json!(true);
        let p = _expect_chat_ok(body);
        assert!(p.stream);
        // Default for include_usage when stream_options is absent.
        assert!(!p.stream_include_usage);
    }

    /// ``stream: false`` (and absent) keeps the aggregating path.
    #[test]
    fn test_chat_params_from_json_stream_false_default() {
        let p = _expect_chat_ok(_chat_body_min("m"));
        assert!(!p.stream);
        let mut body = _chat_body_min("m");
        body["stream"] = serde_json::json!(false);
        let p = _expect_chat_ok(body);
        assert!(!p.stream);
    }

    /// ``stream`` must be boolean; non-boolean values reject.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_non_bool_stream() {
        let mut body = _chat_body_min("m");
        body["stream"] = serde_json::json!("yes");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "stream");
    }

    /// ``stream_options.include_usage`` round-trips when present.
    #[test]
    fn test_chat_params_from_json_accepts_stream_options_include_usage() {
        let mut body = _chat_body_min("m");
        body["stream"] = serde_json::json!(true);
        body["stream_options"] = serde_json::json!({"include_usage": true});
        let p = _expect_chat_ok(body);
        assert!(p.stream);
        assert!(p.stream_include_usage);
    }

    /// Unknown keys inside ``stream_options`` reject explicitly with
    /// ``param: "stream_options.<key>"``.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_stream_options_unknown_keys() {
        let mut body = _chat_body_min("m");
        body["stream"] = serde_json::json!(true);
        body["stream_options"] = serde_json::json!({"foo": 1});
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "unsupported_field");
        assert_eq!(v["error"]["param"], "stream_options.foo");
    }

    /// Non-boolean ``include_usage`` rejects as invalid_request.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_non_bool_include_usage() {
        let mut body = _chat_body_min("m");
        body["stream_options"] = serde_json::json!({"include_usage": "yes"});
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "stream_options.include_usage");
    }

    #[test]
    fn test_chat_params_from_json_plumbs_routing_and_cache_keys() {
        let mut body = _chat_body_min("m");
        body["routing_key"] = serde_json::json!("user-42");
        body["prompt_cache_key"] = serde_json::json!("cache-abc");
        let p = _expect_chat_ok(body);
        assert_eq!(p.routing_key.as_deref(), Some("user-42"));
        assert_eq!(p.prompt_cache_key.as_deref(), Some("cache-abc"));
    }

    #[tokio::test]
    async fn test_chat_params_from_json_safety_identifier_silent() {
        // ``safety_identifier`` is accepted (not rejected) but not
        // surfaced anywhere we can observe. The test just verifies the
        // request still parses.
        let mut body = _chat_body_min("m");
        body["safety_identifier"] = serde_json::json!("user-anon-1");
        let _p = _expect_chat_ok(body);
    }

    /// ``seed`` round-trips onto :class:`ChatRequestParams.seed` so the
    /// chat handler can plumb it through to the worker's SGLang
    /// ``sampling_params.sampling_seed``.
    #[test]
    fn test_chat_params_from_json_seed_round_trips() {
        let mut body = _chat_body_min("m");
        body["seed"] = serde_json::json!(42);
        let p = _expect_chat_ok(body);
        assert_eq!(p.seed, Some(42));
    }

    /// Negative signed seeds are preserved on the worker wire.
    #[test]
    fn test_chat_params_from_json_seed_negative_accepted() {
        let mut body = _chat_body_min("m");
        body["seed"] = serde_json::json!(-1);
        let p = _expect_chat_ok(body);
        assert_eq!(p.seed, Some(-1));
    }

    /// ``seed: null`` is explicitly allowed by OpenAI's spec.
    #[test]
    fn test_chat_params_from_json_seed_null_accepted() {
        let mut body = _chat_body_min("m");
        body["seed"] = serde_json::Value::Null;
        let p = _expect_chat_ok(body);
        assert_eq!(p.seed, None);
    }

    /// ``logit_bias`` parses to a sorted map and round-trips onto
    /// the params for the worker to forward to SGLang.
    #[test]
    fn test_chat_params_from_json_logit_bias_round_trips() {
        let mut body = _chat_body_min("m");
        body["logit_bias"] = serde_json::json!({"1234": 5.0, "5678": -2.5});
        let p = _expect_chat_ok(body);
        let map = p.logit_bias.expect("logit_bias should be Some");
        assert_eq!(map.get("1234"), Some(&5.0));
        assert_eq!(map.get("5678"), Some(&-2.5));
    }

    /// Out-of-range bias values reject as invalid_request.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_logit_bias_out_of_range() {
        let mut body = _chat_body_min("m");
        body["logit_bias"] = serde_json::json!({"1234": 150.0});
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "logit_bias");
    }

    /// Non-numeric token-id keys reject.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_logit_bias_non_int_keys() {
        let mut body = _chat_body_min("m");
        body["logit_bias"] = serde_json::json!({"hello": 1.0});
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "logit_bias");
    }

    /// ``logprobs: true`` + ``top_logprobs: 5`` round-trip onto the
    /// params so the worker can request per-token alternates from SGLang.
    #[test]
    fn test_chat_params_from_json_logprobs_round_trips() {
        let mut body = _chat_body_min("m");
        body["logprobs"] = serde_json::json!(true);
        body["top_logprobs"] = serde_json::json!(5);
        let p = _expect_chat_ok(body);
        assert_eq!(p.logprobs, Some(true));
        assert_eq!(p.top_logprobs, Some(5));
    }

    /// ``top_logprobs`` without ``logprobs: true`` is OpenAI-illegal —
    /// reject with the offending param.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_top_logprobs_without_logprobs() {
        let mut body = _chat_body_min("m");
        body["top_logprobs"] = serde_json::json!(3);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "top_logprobs");
    }

    /// ``top_logprobs`` above OpenAI's ``20`` cap rejects.
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_top_logprobs_above_cap() {
        let mut body = _chat_body_min("m");
        body["logprobs"] = serde_json::json!(true);
        body["top_logprobs"] = serde_json::json!(21);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "top_logprobs");
    }

    /// ``n=1`` (the only currently-implemented value) parses and round-
    /// trips onto :class:`ChatRequestParams.n`. The wire shape is in
    /// place; gateway-side fan-out for larger values remains deferred.
    #[test]
    fn test_chat_params_from_json_n_equals_one_ok() {
        let mut body = _chat_body_min("m");
        body["n"] = serde_json::json!(1);
        let p = _expect_chat_ok(body);
        assert_eq!(p.n, Some(1));
    }

    /// Non-streaming ``n>1`` is accepted (roadmap 1.5): the value parses and
    /// surfaces on :class:`ChatRequestParams.n` for forwarding to the worker,
    /// which runs the candidates server-side and returns a multi-entry
    /// ``choices`` array.
    #[test]
    fn test_chat_params_from_json_n_gt_one_non_stream_accepted() {
        let mut body = _chat_body_min("m");
        body["n"] = serde_json::json!(3);
        let p = _expect_chat_ok(body);
        assert_eq!(p.n, Some(3));
    }

    #[test]
    fn test_chat_params_accepts_lora_adapter() {
        let mut body = _chat_body_min("m");
        body["lora_adapter"] = serde_json::json!("acme-support");
        let p = _expect_chat_ok(body);
        assert_eq!(p.lora_adapter.as_deref(), Some("acme-support"));
    }

    #[test]
    fn test_chat_params_accepts_text_content_parts() {
        // OpenAI multimodal content as an array of text parts → concatenated.
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ]}
        ]);
        let p = _expect_chat_ok(body);
        assert_eq!(p.messages[0].content, "Hello world");
    }

    #[tokio::test]
    async fn test_chat_params_rejects_remote_image_url() {
        // Remote (http/https) image URLs are not fetched at the gateway —
        // clients must inline images as base64 data URIs.
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            ]}
        ]);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "messages[0].content[1].image_url");
    }

    #[test]
    fn test_chat_params_accepts_image_data_uri() {
        // A base64 ``data:`` URI is decoded into ``ChatMessage.images``; the
        // text part lands in ``content``. (Capability gating happens later in
        // ``proxy_chat`` after model resolution, not here.)
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
            ]}
        ]);
        let p = _expect_chat_ok(body);
        assert_eq!(p.messages[0].content, "what is this?");
        let images = p.messages[0].images.as_ref().expect("images populated");
        assert_eq!(images.len(), 1);
        // ``data`` is the base64 payload string (decodes to b"hello"); kept as
        // a string so it survives the sidecar's serde_json::Value.
        assert_eq!(images[0].data, "aGVsbG8=");
        assert_eq!(images[0].format.as_deref(), Some("png"));
    }

    #[test]
    fn test_chat_params_preserves_interleaved_content_parts() {
        // #1294: text↔image interleaving must survive into ``content_parts`` in
        // the original order, while ``images`` stays a flat in-order list and
        // ``content`` stays the concatenated text (back-compat).
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": [
                {"type": "text", "text": "Page 1:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1nQQ=="}},
                {"type": "text", "text": "Page 2:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1nQg=="}},
                {"type": "text", "text": "which has a cat?"},
            ]}
        ]);
        let p = _expect_chat_ok(body);
        // Flat fields unchanged: concatenated text + in-order image bytes.
        assert_eq!(p.messages[0].content, "Page 1:Page 2:which has a cat?");
        let images = p.messages[0].images.as_ref().expect("images populated");
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].data, "aW1nQQ==");
        assert_eq!(images[1].data, "aW1nQg==");
        // Ordered layout records the interleaving the worker renders in place.
        let parts = p.messages[0]
            .content_parts
            .as_ref()
            .expect("content_parts populated for vision msg");
        assert_eq!(parts.len(), 5);
        assert!(matches!(&parts[0], publisher::ContentPart::Text { text } if text == "Page 1:"));
        assert!(matches!(parts[1], publisher::ContentPart::Image));
        assert!(matches!(&parts[2], publisher::ContentPart::Text { text } if text == "Page 2:"));
        assert!(matches!(parts[3], publisher::ContentPart::Image));
        assert!(
            matches!(&parts[4], publisher::ContentPart::Text { text } if text == "which has a cat?")
        );
    }

    #[test]
    fn test_chat_params_text_only_has_no_content_parts() {
        // Text-only messages must NOT carry content_parts (no wire bloat, no
        // behavior change) — the images-first/legacy render path stays in use.
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([{"role": "user", "content": "hello"}]);
        let p = _expect_chat_ok(body);
        assert!(p.messages[0].content_parts.is_none());
        // A multipart array with NO image also stays content_parts-free.
        let mut body2 = _chat_body_min("m");
        body2["messages"] = serde_json::json!([
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        ]);
        let p2 = _expect_chat_ok(body2);
        assert_eq!(p2.messages[0].content, "ab");
        assert!(p2.messages[0].content_parts.is_none());
    }

    #[tokio::test]
    async fn test_chat_params_rejects_too_many_images() {
        let mut body = _chat_body_min("m");
        let mut parts = vec![serde_json::json!({"type": "text", "text": "describe"})];
        for _ in 0..17 {
            parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,aGk="}
            }));
        }
        body["messages"] = serde_json::json!([{"role": "user", "content": parts}]);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("too many images"));
    }

    #[test]
    fn test_decode_image_data_uri_variants() {
        // Returns the base64 payload string + format. Bare-string url form +
        // object form both work.
        let (data, fmt) =
            decode_image_data_uri(&serde_json::json!("data:image/jpeg;base64,aGk=")).unwrap();
        assert_eq!(data, "aGk="); // decodes to b"hi"
        assert_eq!(fmt.as_deref(), Some("jpeg"));
        let (data2, _) =
            decode_image_data_uri(&serde_json::json!({"url": "data:image/png;base64,aGVsbG8="}))
                .unwrap();
        assert_eq!(data2, "aGVsbG8="); // decodes to b"hello"
                                       // Remote URL, missing base64 marker, and bad base64 all reject.
        assert!(decode_image_data_uri(&serde_json::json!("https://x/y.png")).is_err());
        assert!(decode_image_data_uri(&serde_json::json!("data:image/png,aGk=")).is_err());
        assert!(decode_image_data_uri(&serde_json::json!("data:image/png;base64,!!!")).is_err());
        // Non-image media types reject (must be image/*).
        assert!(decode_image_data_uri(&serde_json::json!("data:text/plain;base64,aGk=")).is_err());
        assert!(decode_image_data_uri(&serde_json::json!("data:;base64,aGk=")).is_err());
    }

    #[test]
    fn test_chat_params_accepts_best_of() {
        let mut body = _chat_body_min("m");
        body["n"] = serde_json::json!(2);
        body["best_of"] = serde_json::json!(5);
        let p = _expect_chat_ok(body);
        assert_eq!(p.best_of, Some(5));
        assert_eq!(p.n, Some(2));
    }

    #[tokio::test]
    async fn test_chat_params_rejects_best_of_less_than_n() {
        let mut body = _chat_body_min("m");
        body["n"] = serde_json::json!(4);
        body["best_of"] = serde_json::json!(2);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "best_of");
    }

    #[tokio::test]
    async fn test_chat_params_rejects_best_of_with_stream() {
        let mut body = _chat_body_min("m");
        body["best_of"] = serde_json::json!(3);
        body["stream"] = serde_json::json!(true);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "unsupported_field");
        assert_eq!(v["error"]["param"], "best_of");
    }

    #[tokio::test]
    async fn test_chat_params_rejects_non_string_lora_adapter() {
        let mut body = _chat_body_min("m");
        body["lora_adapter"] = serde_json::json!(123);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "lora_adapter");
    }

    /// Streaming + n>1 is now accepted: the worker fans candidates out as
    /// per-`choice_index` SSE delta chunks.
    #[test]
    fn test_chat_params_from_json_n_gt_one_with_stream_accepted() {
        let mut body = _chat_body_min("m");
        body["n"] = serde_json::json!(2);
        body["stream"] = serde_json::json!(true);
        let p = _expect_chat_ok(body);
        assert_eq!(p.n, Some(2));
        assert!(p.stream);
    }

    /// `n>1` body builder: a `StreamOutcome` carrying multiple `candidates`
    /// produces a multi-entry `choices` array (one per candidate, each with
    /// its own index/content/finish_reason); `usage` is the aggregate.
    #[test]
    fn test_build_chat_completion_body_emits_multi_candidate_choices() {
        use crate::queue::streaming::{CandidateData, StreamOutcome, UsageBlock};
        let outcome = StreamOutcome {
            text: String::new(),
            finish_reason: "stop".to_string(),
            usage: Some(UsageBlock {
                prompt_tokens: 5,
                completion_tokens: 9,
                total_tokens: 14,
            }),
            attempt_id: "att".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: vec![
                CandidateData {
                    text: "Alpha".to_string(),
                    finish_reason: Some("stop".to_string()),
                    logprobs: None,
                    tool_calls: None,
                },
                CandidateData {
                    text: "Beta".to_string(),
                    finish_reason: Some("length".to_string()),
                    logprobs: None,
                    tool_calls: None,
                },
            ],
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let bytes = build_chat_completion_body("m", "req", &outcome).expect("ok");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let choices = v["choices"].as_array().expect("choices array");
        assert_eq!(choices.len(), 2);
        assert_eq!(choices[0]["index"], 0);
        assert_eq!(choices[0]["message"]["content"], "Alpha");
        assert_eq!(choices[0]["message"]["role"], "assistant");
        assert_eq!(choices[0]["finish_reason"], "stop");
        assert_eq!(choices[1]["index"], 1);
        assert_eq!(choices[1]["message"]["content"], "Beta");
        assert_eq!(choices[1]["finish_reason"], "length");
        // Aggregate usage (worker sums completion across candidates).
        assert_eq!(v["usage"]["completion_tokens"], 9);
        assert_eq!(v["usage"]["prompt_tokens"], 5);
    }

    /// H5 non-streaming: the multi-candidate body builder surfaces per-
    /// candidate ``tool_calls`` on ``choices[i].message`` with
    /// ``content: null``, matching OpenAI's non-streaming spec.
    #[test]
    fn test_build_chat_completion_body_per_candidate_tool_calls() {
        use crate::queue::streaming::{CandidateData, StreamOutcome, UsageBlock};
        let outcome = StreamOutcome {
            text: String::new(),
            finish_reason: "tool_calls".to_string(),
            usage: Some(UsageBlock {
                prompt_tokens: 6,
                completion_tokens: 12,
                total_tokens: 18,
            }),
            attempt_id: "att".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: vec![
                CandidateData {
                    text: String::new(),
                    finish_reason: Some("tool_calls".to_string()),
                    logprobs: None,
                    tool_calls: Some(vec![serde_json::json!({
                        "id": "call_x",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Tokyo\"}"
                        }
                    })]),
                },
                CandidateData {
                    text: "plain answer".to_string(),
                    finish_reason: Some("stop".to_string()),
                    logprobs: None,
                    tool_calls: None,
                },
            ],
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let bytes = build_chat_completion_body("m", "req-tc", &outcome).expect("ok");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let c0 = &v["choices"][0];
        assert!(c0["message"]["content"].is_null());
        assert_eq!(
            c0["message"]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(c0["finish_reason"], "tool_calls");
        let c1 = &v["choices"][1];
        assert_eq!(c1["message"]["content"], "plain answer");
        assert!(c1["message"].get("tool_calls").is_none());
        assert_eq!(c1["finish_reason"], "stop");
    }

    /// M4: per-candidate logprobs on the non-streaming ``n>1`` body — each
    /// ``choices[i].logprobs`` is the OpenAI ``{content, refusal}`` envelope
    /// built from that candidate's own slice (no longer ``null``).
    #[test]
    fn test_build_chat_completion_body_per_candidate_logprobs() {
        use crate::queue::streaming::{CandidateData, StreamOutcome, UsageBlock};
        let cand_lp = vec![serde_json::json!({
            "token": "x",
            "logprob": -1.0,
            "bytes": [120],
            "top_logprobs": [],
        })];
        let outcome = StreamOutcome {
            text: String::new(),
            finish_reason: "stop".to_string(),
            usage: Some(UsageBlock {
                prompt_tokens: 3,
                completion_tokens: 4,
                total_tokens: 7,
            }),
            attempt_id: "att".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: vec![CandidateData {
                text: "x".to_string(),
                finish_reason: Some("stop".to_string()),
                logprobs: Some(cand_lp),
                tool_calls: None,
            }],
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let bytes = build_chat_completion_body("m", "req-lp", &outcome).expect("ok");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let lps = &v["choices"][0]["logprobs"];
        assert!(!lps.is_null(), "candidate logprobs must surface (M4)");
        assert_eq!(lps["content"][0]["token"], "x");
    }

    // ── /v1/chat/completions M1 hardening (sampler + token cap types) ──
    //
    // Present-but-wrong-type sampler / max_tokens values must now 400
    // rather than silently falling back to the default.

    #[tokio::test]
    async fn test_chat_rejects_string_max_completion_tokens() {
        let mut body = _chat_body_min("m");
        body["max_completion_tokens"] = serde_json::json!("16");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["param"], "max_completion_tokens");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_chat_rejects_negative_max_completion_tokens() {
        let mut body = _chat_body_min("m");
        body["max_completion_tokens"] = serde_json::json!(-1);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["param"], "max_completion_tokens");
    }

    #[tokio::test]
    async fn test_chat_rejects_float_max_completion_tokens() {
        let mut body = _chat_body_min("m");
        body["max_completion_tokens"] = serde_json::json!(1.5);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["param"], "max_completion_tokens");
    }

    #[tokio::test]
    async fn test_chat_rejects_string_max_tokens_legacy() {
        let mut body = _chat_body_min("m");
        body["max_tokens"] = serde_json::json!("16");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["param"], "max_tokens");
    }

    #[test]
    fn test_chat_accepts_null_max_completion_tokens_falls_to_default() {
        // Explicit null must not 400 — falls back to the default.
        let mut body = _chat_body_min("m");
        body["max_completion_tokens"] = serde_json::Value::Null;
        let p = _expect_chat_ok(body);
        assert!(p.max_new_tokens > 0);
    }

    #[tokio::test]
    async fn test_chat_rejects_string_temperature() {
        let mut body = _chat_body_min("m");
        body["temperature"] = serde_json::json!("0.5");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["param"], "temperature");
    }

    #[tokio::test]
    async fn test_chat_rejects_bool_temperature() {
        let mut body = _chat_body_min("m");
        body["temperature"] = serde_json::json!(true);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["param"], "temperature");
    }

    #[tokio::test]
    async fn test_chat_rejects_string_top_p() {
        let mut body = _chat_body_min("m");
        body["top_p"] = serde_json::json!("0.9");
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["param"], "top_p");
    }

    #[test]
    fn test_chat_accepts_null_temperature() {
        let mut body = _chat_body_min("m");
        body["temperature"] = serde_json::Value::Null;
        let p = _expect_chat_ok(body);
        assert!(p.temperature.is_none());
    }

    // ── /v1/chat/completions M13 hardening (tool-history shape) ────────
    //
    // Malformed tool_calls entries, misplaced tool_call_id, and lossy
    // content-part validation must now surface as 400.

    fn _chat_assistant_with_tool_calls(tcs: serde_json::Value) -> serde_json::Value {
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": serde_json::Value::Null, "tool_calls": tcs},
        ]);
        body
    }

    #[tokio::test]
    async fn test_chat_rejects_tool_calls_entry_missing_id() {
        let body = _chat_assistant_with_tool_calls(serde_json::json!([
            {"type": "function", "function": {"name": "f", "arguments": "{}"}}
        ]));
        let v = _expect_chat_err(body).await;
        assert!(v["error"]["param"].as_str().unwrap().ends_with(".id"));
    }

    #[tokio::test]
    async fn test_chat_rejects_tool_calls_entry_wrong_type() {
        let body = _chat_assistant_with_tool_calls(serde_json::json!([
            {"id": "abc", "type": "not_a_function", "function": {"name": "f", "arguments": "{}"}}
        ]));
        let v = _expect_chat_err(body).await;
        assert!(v["error"]["param"].as_str().unwrap().ends_with(".type"));
    }

    #[tokio::test]
    async fn test_chat_rejects_tool_calls_entry_missing_function_object() {
        let body = _chat_assistant_with_tool_calls(serde_json::json!([
            {"id": "abc", "type": "function"}
        ]));
        let v = _expect_chat_err(body).await;
        assert!(v["error"]["param"].as_str().unwrap().ends_with(".function"));
    }

    #[tokio::test]
    async fn test_chat_rejects_tool_calls_entry_missing_function_name() {
        let body = _chat_assistant_with_tool_calls(serde_json::json!([
            {"id": "abc", "type": "function", "function": {"arguments": "{}"}}
        ]));
        let v = _expect_chat_err(body).await;
        assert!(v["error"]["param"]
            .as_str()
            .unwrap()
            .ends_with("function.name"));
    }

    #[tokio::test]
    async fn test_chat_rejects_tool_calls_entry_non_string_arguments() {
        // OpenAI ships arguments as a JSON-string; an object must reject.
        let body = _chat_assistant_with_tool_calls(serde_json::json!([
            {"id": "abc", "type": "function", "function": {"name": "f", "arguments": {"k": 1}}}
        ]));
        let v = _expect_chat_err(body).await;
        assert!(v["error"]["param"]
            .as_str()
            .unwrap()
            .ends_with("function.arguments"));
    }

    #[test]
    fn test_chat_accepts_well_formed_tool_calls() {
        let body = _chat_assistant_with_tool_calls(serde_json::json!([
            {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ]));
        let p = _expect_chat_ok(body);
        assert!(p.messages[1].tool_calls.is_some());
    }

    #[tokio::test]
    async fn test_chat_rejects_tool_call_id_on_user_role() {
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": "hi", "tool_call_id": "call_1"},
        ]);
        let v = _expect_chat_err(body).await;
        assert!(v["error"]["param"]
            .as_str()
            .unwrap()
            .ends_with(".tool_call_id"));
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_chat_rejects_missing_tool_call_id_on_tool_role() {
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": serde_json::Value::Null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "result"},
        ]);
        let v = _expect_chat_err(body).await;
        assert!(v["error"]["param"]
            .as_str()
            .unwrap()
            .ends_with(".tool_call_id"));
    }

    #[test]
    fn test_chat_accepts_tool_call_id_on_tool_role() {
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": serde_json::Value::Null, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        ]);
        let p = _expect_chat_ok(body);
        assert_eq!(p.messages[2].tool_call_id.as_deref(), Some("call_1"));
    }

    #[tokio::test]
    async fn test_chat_rejects_text_part_with_missing_text() {
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": [{"type": "text"}]},
        ]);
        let v = _expect_chat_err(body).await;
        assert!(v["error"]["param"].as_str().unwrap().ends_with(".text"));
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_chat_rejects_text_part_with_non_string_text() {
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": [{"type": "text", "text": 42}]},
        ]);
        let v = _expect_chat_err(body).await;
        assert!(v["error"]["param"].as_str().unwrap().ends_with(".text"));
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_chat_rejects_content_part_with_missing_type() {
        let mut body = _chat_body_min("m");
        body["messages"] = serde_json::json!([
            {"role": "user", "content": [{"text": "hi"}]},
        ]);
        let v = _expect_chat_err(body).await;
        assert!(v["error"]["param"].as_str().unwrap().ends_with(".type"));
    }

    // ── /v1/completions ──────────────────────────────────────────────

    fn _completions_body_min(model: &str) -> serde_json::Value {
        serde_json::json!({"model": model, "prompt": "Once upon a time"})
    }

    async fn _completions_err(body: serde_json::Value) -> serde_json::Value {
        let resp = match completions_params_from_json(&body) {
            CompletionsParamsResult::Ok(_) => panic!("expected Err"),
            CompletionsParamsResult::Err(r) => r,
        };
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn test_completions_params_ok_defaults() {
        match completions_params_from_json(&_completions_body_min("m")) {
            CompletionsParamsResult::Ok(p) => {
                assert_eq!(p.model, "m");
                assert_eq!(p.prompt, "Once upon a time");
                assert_eq!(p.max_new_tokens, 16); // OpenAI default
            }
            CompletionsParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    #[test]
    fn test_completions_params_parses_sampling() {
        let mut body = _completions_body_min("m");
        body["max_tokens"] = serde_json::json!(64);
        body["temperature"] = serde_json::json!(0.5);
        body["stop"] = serde_json::json!(["\n\n"]);
        match completions_params_from_json(&body) {
            CompletionsParamsResult::Ok(p) => {
                assert_eq!(p.max_new_tokens, 64);
                assert_eq!(p.temperature, Some(0.5));
                assert_eq!(p.stop.as_deref(), Some(&["\n\n".to_string()][..]));
            }
            CompletionsParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    /// ``stream_options.include_usage`` on /v1/completions. A streamed response
    /// cannot carry usage in headers, so this flag is the only surface on which
    /// a streamed completion reports what it consumed.
    #[test]
    fn test_completions_params_accepts_stream_options_include_usage() {
        let mut body = _completions_body_min("m");
        assert!(matches!(
            completions_params_from_json(&body),
            CompletionsParamsResult::Ok(ref p) if !p.stream_include_usage
        ));
        body["stream"] = serde_json::json!(true);
        body["stream_options"] = serde_json::json!({"include_usage": true});
        match completions_params_from_json(&body) {
            CompletionsParamsResult::Ok(p) => {
                assert!(p.stream);
                assert!(p.stream_include_usage);
            }
            CompletionsParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    #[tokio::test]
    async fn test_completions_rejects_stream_options_unknown_keys_and_types() {
        let mut body = _completions_body_min("m");
        body["stream_options"] = serde_json::json!({"foo": 1});
        let v = _completions_err(body.clone()).await;
        assert_eq!(v["error"]["param"], "stream_options.foo");

        body["stream_options"] = serde_json::json!({"include_usage": "yes"});
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "stream_options.include_usage");
    }

    #[tokio::test]
    async fn test_completions_missing_model() {
        let v = _completions_err(serde_json::json!({"prompt": "hi"})).await;
        assert_eq!(v["error"]["param"], "model");
    }

    #[tokio::test]
    async fn test_completions_missing_prompt() {
        let v = _completions_err(serde_json::json!({"model": "m"})).await;
        assert_eq!(v["error"]["param"], "prompt");
    }

    #[test]
    fn test_completions_accepts_stream() {
        let mut body = _completions_body_min("m");
        body["stream"] = serde_json::json!(true);
        match completions_params_from_json(&body) {
            CompletionsParamsResult::Ok(p) => assert!(p.stream),
            CompletionsParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    #[tokio::test]
    async fn test_completions_rejects_n_gt_one() {
        let mut body = _completions_body_min("m");
        body["n"] = serde_json::json!(2);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "n");
    }

    #[tokio::test]
    async fn test_completions_rejects_batched_array_prompt() {
        let mut body = _completions_body_min("m");
        body["prompt"] = serde_json::json!(["a", "b"]);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "prompt");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[test]
    fn test_build_text_completion_body_shape() {
        let outcome = crate::queue::streaming::StreamOutcome {
            text: "a continuation".to_string(),
            finish_reason: "length".to_string(),
            usage: Some(crate::queue::streaming::UsageBlock {
                prompt_tokens: 4,
                completion_tokens: 16,
                total_tokens: 20,
            }),
            attempt_id: "att".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let bytes = build_text_completion_body("m", "req-1", &outcome).expect("ok");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert!(v["id"].as_str().unwrap().starts_with("cmpl-"));
        assert_eq!(v["choices"][0]["text"], "a continuation");
        assert_eq!(v["choices"][0]["index"], 0);
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        // H3: ``logprobs`` is rejected at the input parser, so the
        // response body no longer carries an always-null ``logprobs``.
        assert!(
            v["choices"][0]
                .as_object()
                .unwrap()
                .get("logprobs")
                .is_none(),
            "'logprobs' should not appear in /v1/completions responses",
        );
        assert_eq!(v["usage"]["total_tokens"], 20);
        assert!(v["system_fingerprint"].is_string());
    }

    // ── /v1/completions H3 hardening ─────────────────────────────────
    //
    // Strict allow-list + present-but-wrong-type rejection. These cases
    // previously silently coerced or dropped — they must now 400.

    #[tokio::test]
    async fn test_completions_rejects_echo() {
        let mut body = _completions_body_min("m");
        body["echo"] = serde_json::json!(true);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "echo");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_completions_rejects_suffix() {
        let mut body = _completions_body_min("m");
        body["suffix"] = serde_json::json!("...");
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "suffix");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_completions_rejects_logprobs() {
        let mut body = _completions_body_min("m");
        body["logprobs"] = serde_json::json!(5);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "logprobs");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_completions_rejects_best_of() {
        let mut body = _completions_body_min("m");
        body["best_of"] = serde_json::json!(3);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "best_of");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_completions_rejects_unknown_top_level_field() {
        let mut body = _completions_body_min("m");
        body["foo_bar"] = serde_json::json!(1);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "foo_bar");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_completions_rejects_non_bool_stream_string() {
        let mut body = _completions_body_min("m");
        body["stream"] = serde_json::json!("true");
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "stream");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_completions_rejects_non_bool_stream_int() {
        let mut body = _completions_body_min("m");
        body["stream"] = serde_json::json!(1);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "stream");
    }

    #[test]
    fn test_completions_accepts_n_one() {
        // n:1 is a no-op accept (it's the implicit default).
        let mut body = _completions_body_min("m");
        body["n"] = serde_json::json!(1);
        match completions_params_from_json(&body) {
            CompletionsParamsResult::Ok(_) => {}
            CompletionsParamsResult::Err(_) => panic!("expected Ok for n=1"),
        }
    }

    #[tokio::test]
    async fn test_completions_rejects_n_zero() {
        let mut body = _completions_body_min("m");
        body["n"] = serde_json::json!(0);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "n");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_completions_rejects_non_integer_n_string() {
        let mut body = _completions_body_min("m");
        body["n"] = serde_json::json!("2");
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "n");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_completions_rejects_non_integer_n_float() {
        let mut body = _completions_body_min("m");
        body["n"] = serde_json::json!(1.5);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "n");
    }

    #[tokio::test]
    async fn test_completions_rejects_negative_max_tokens() {
        let mut body = _completions_body_min("m");
        body["max_tokens"] = serde_json::json!(-1);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "max_tokens");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_completions_rejects_string_max_tokens() {
        let mut body = _completions_body_min("m");
        body["max_tokens"] = serde_json::json!("16");
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "max_tokens");
    }

    #[test]
    fn test_completions_accepts_null_max_tokens() {
        // Explicit null must not 400 — it falls back to the default.
        let mut body = _completions_body_min("m");
        body["max_tokens"] = serde_json::Value::Null;
        match completions_params_from_json(&body) {
            CompletionsParamsResult::Ok(p) => assert_eq!(p.max_new_tokens, 16),
            CompletionsParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    #[tokio::test]
    async fn test_completions_rejects_string_temperature() {
        let mut body = _completions_body_min("m");
        body["temperature"] = serde_json::json!("0.5");
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "temperature");
    }

    #[tokio::test]
    async fn test_completions_rejects_negative_temperature() {
        let mut body = _completions_body_min("m");
        body["temperature"] = serde_json::json!(-0.1);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "temperature");
    }

    #[tokio::test]
    async fn test_completions_rejects_top_p_out_of_range() {
        let mut body = _completions_body_min("m");
        body["top_p"] = serde_json::json!(1.5);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "top_p");
    }

    #[tokio::test]
    async fn test_completions_rejects_string_top_p() {
        let mut body = _completions_body_min("m");
        body["top_p"] = serde_json::json!("0.9");
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "top_p");
    }

    #[tokio::test]
    async fn test_completions_rejects_mixed_stop_array() {
        let mut body = _completions_body_min("m");
        body["stop"] = serde_json::json!(["a", 1]);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "stop");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_completions_rejects_string_frequency_penalty() {
        let mut body = _completions_body_min("m");
        body["frequency_penalty"] = serde_json::json!("0.5");
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "frequency_penalty");
    }

    #[tokio::test]
    async fn test_completions_rejects_out_of_range_presence_penalty() {
        let mut body = _completions_body_min("m");
        body["presence_penalty"] = serde_json::json!(3.0);
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "presence_penalty");
    }

    #[tokio::test]
    async fn test_completions_rejects_string_seed() {
        let mut body = _completions_body_min("m");
        body["seed"] = serde_json::json!("42");
        let v = _completions_err(body).await;
        assert_eq!(v["error"]["param"], "seed");
    }

    #[test]
    fn test_completions_preserves_negative_seed() {
        let mut body = _completions_body_min("m");
        body["seed"] = serde_json::json!(-1);
        match completions_params_from_json(&body) {
            CompletionsParamsResult::Ok(params) => assert_eq!(params.seed, Some(-1)),
            CompletionsParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    // ── /v1/responses ────────────────────────────────────────────────

    fn _responses_body_min(model: &str) -> serde_json::Value {
        serde_json::json!({"model": model, "input": "Tell me a joke"})
    }

    async fn _responses_err(body: serde_json::Value) -> serde_json::Value {
        let resp = match responses_params_from_json(&body) {
            ResponsesParamsResult::Ok(_) => panic!("expected Err"),
            ResponsesParamsResult::Err(r) => r,
        };
        let bytes = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn test_responses_shared_request_conformance_vectors() {
        let raw = include_str!(
            "../../../../conformance/openai_generation/responses_request_vectors.json"
        );
        let vectors: serde_json::Value = serde_json::from_str(raw).expect("valid conformance JSON");

        for case in vectors["accepted"].as_array().expect("accepted vectors") {
            let name = case["name"].as_str().expect("case name");
            let params = match responses_params_from_json(&case["body"]) {
                ResponsesParamsResult::Ok(params) => params,
                ResponsesParamsResult::Err(_) => panic!("accepted case {name:?} was rejected"),
            };
            let (prompt, messages) = match params.input {
                publisher::GenerateInput::Prompt { prompt } => {
                    (Some(prompt), serde_json::Value::Null)
                }
                publisher::GenerateInput::Messages { messages } => (
                    None,
                    serde_json::Value::Array(
                        messages
                            .into_iter()
                            .map(|message| {
                                serde_json::json!({
                                    "role": message.role,
                                    "content": message.content,
                                })
                            })
                            .collect(),
                    ),
                ),
            };
            let mut observed = serde_json::json!({
                "model": params.model,
                "prompt": prompt,
                "messages": messages,
                "max_output_tokens": params.max_new_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "seed": params.seed,
            });
            let mut expected = case["expected"].clone();
            for field in ["temperature", "top_p"] {
                match (observed[field].as_f64(), expected[field].as_f64()) {
                    (Some(left), Some(right)) => assert!(
                        (left - right).abs() <= f32::EPSILON as f64,
                        "accepted case {name:?} field {field:?}: {left} != {right}"
                    ),
                    (None, None) if observed[field].is_null() && expected[field].is_null() => {}
                    _ => panic!("accepted case {name:?} field {field:?} differs"),
                }
                observed
                    .as_object_mut()
                    .expect("observed object")
                    .remove(field);
                expected
                    .as_object_mut()
                    .expect("expected object")
                    .remove(field);
            }
            assert_eq!(observed, expected, "accepted case {name:?}");
        }

        for case in vectors["rejected"].as_array().expect("rejected vectors") {
            let name = case["name"].as_str().expect("case name");
            let error = _responses_err(case["body"].clone()).await;
            assert_eq!(
                error["error"]["param"], case["param"],
                "rejected case {name:?}"
            );
            assert_eq!(
                error["error"]["code"], case["code"],
                "rejected case {name:?}"
            );
        }
    }

    #[test]
    fn test_responses_params_ok_string_input() {
        match responses_params_from_json(&_responses_body_min("m")) {
            ResponsesParamsResult::Ok(p) => {
                assert_eq!(p.model, "m");
                assert_eq!(p.max_new_tokens, 16);
                match p.input {
                    publisher::GenerateInput::Prompt { prompt } => {
                        assert_eq!(prompt, "Tell me a joke")
                    }
                    publisher::GenerateInput::Messages { .. } => panic!("expected Prompt"),
                }
            }
            ResponsesParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    #[tokio::test]
    async fn test_responses_missing_input() {
        let v = _responses_err(serde_json::json!({"model": "m"})).await;
        assert_eq!(v["error"]["param"], "input");
    }

    #[test]
    fn test_responses_accepts_array_input_as_messages() {
        let mut body = _responses_body_min("m");
        body["input"] = serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "hello "}, {"type": "output_text", "text": "there"}]},
        ]);
        match responses_params_from_json(&body) {
            ResponsesParamsResult::Ok(p) => match p.input {
                publisher::GenerateInput::Messages { messages } => {
                    assert_eq!(messages.len(), 2);
                    assert_eq!(messages[0].role, "user");
                    assert_eq!(messages[0].content, "hi");
                    assert_eq!(messages[1].content, "hello there"); // text parts concatenated
                }
                publisher::GenerateInput::Prompt { .. } => panic!("expected Messages"),
            },
            ResponsesParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    #[tokio::test]
    async fn test_responses_rejects_empty_array_input() {
        let mut body = _responses_body_min("m");
        body["input"] = serde_json::json!([]);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "input");
    }

    #[tokio::test]
    async fn test_responses_rejects_stream() {
        let mut body = _responses_body_min("m");
        body["stream"] = serde_json::json!(true);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "stream");
    }

    // ── /v1/responses H2 hardening ───────────────────────────────────
    //
    // Strict allow-list + present-but-wrong-type rejection. These cases
    // previously silently coerced or dropped — they must now 400.

    #[tokio::test]
    async fn test_responses_rejects_previous_response_id() {
        let mut body = _responses_body_min("m");
        body["previous_response_id"] = serde_json::json!("resp-abc");
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "previous_response_id");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_responses_rejects_tools() {
        let mut body = _responses_body_min("m");
        body["tools"] = serde_json::json!([{"type": "function"}]);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "tools");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_responses_rejects_tool_choice() {
        let mut body = _responses_body_min("m");
        body["tool_choice"] = serde_json::json!("auto");
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "tool_choice");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_responses_rejects_reasoning() {
        let mut body = _responses_body_min("m");
        body["reasoning"] = serde_json::json!({"effort": "high"});
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "reasoning");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_responses_rejects_background() {
        let mut body = _responses_body_min("m");
        body["background"] = serde_json::json!(true);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "background");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_responses_rejects_metadata() {
        let mut body = _responses_body_min("m");
        body["metadata"] = serde_json::json!({"k": "v"});
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "metadata");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_responses_rejects_instructions() {
        let mut body = _responses_body_min("m");
        body["instructions"] = serde_json::json!("be terse");
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "instructions");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_responses_rejects_unknown_top_level_field() {
        let mut body = _responses_body_min("m");
        body["foo_bar"] = serde_json::json!(1);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "foo_bar");
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_responses_rejects_non_bool_stream_string() {
        let mut body = _responses_body_min("m");
        body["stream"] = serde_json::json!("true");
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "stream");
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_responses_rejects_non_bool_stream_int() {
        let mut body = _responses_body_min("m");
        body["stream"] = serde_json::json!(1);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "stream");
    }

    #[tokio::test]
    async fn test_responses_rejects_input_image_part() {
        let mut body = _responses_body_min("m");
        body["input"] = serde_json::json!([
            {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,..."}]},
        ]);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_responses_rejects_unknown_content_part_type() {
        let mut body = _responses_body_min("m");
        body["input"] = serde_json::json!([
            {"role": "user", "content": [{"type": "audio", "data": "..."}]},
        ]);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["code"], "unsupported_field");
    }

    #[tokio::test]
    async fn test_responses_rejects_missing_text_on_text_part() {
        let mut body = _responses_body_min("m");
        body["input"] = serde_json::json!([
            {"role": "user", "content": [{"type": "input_text"}]},
        ]);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_responses_rejects_non_string_text_on_text_part() {
        let mut body = _responses_body_min("m");
        body["input"] = serde_json::json!([
            {"role": "user", "content": [{"type": "input_text", "text": 42}]},
        ]);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[tokio::test]
    async fn test_responses_rejects_missing_role_on_input_message() {
        let mut body = _responses_body_min("m");
        body["input"] = serde_json::json!([{"content": "hi"}]);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert!(v["error"]["param"].as_str().unwrap().ends_with(".role"));
    }

    #[tokio::test]
    async fn test_responses_rejects_invalid_role_on_input_message() {
        let mut body = _responses_body_min("m");
        body["input"] = serde_json::json!([{"role": "alien", "content": "hi"}]);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
    }

    #[test]
    fn test_responses_accepts_developer_role_as_system() {
        let mut body = _responses_body_min("m");
        body["input"] = serde_json::json!([{"role": "developer", "content": "be helpful"}]);
        match responses_params_from_json(&body) {
            ResponsesParamsResult::Ok(p) => match p.input {
                publisher::GenerateInput::Messages { messages } => {
                    assert_eq!(messages[0].role, "system");
                }
                _ => panic!("expected Messages"),
            },
            ResponsesParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    #[tokio::test]
    async fn test_responses_rejects_string_temperature() {
        let mut body = _responses_body_min("m");
        body["temperature"] = serde_json::json!("0.5");
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "temperature");
    }

    #[tokio::test]
    async fn test_responses_rejects_top_p_out_of_range() {
        let mut body = _responses_body_min("m");
        body["top_p"] = serde_json::json!(1.5);
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "top_p");
    }

    #[tokio::test]
    async fn test_responses_rejects_string_seed() {
        let mut body = _responses_body_min("m");
        body["seed"] = serde_json::json!("42");
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "seed");
    }

    #[test]
    fn test_responses_preserves_negative_seed() {
        let mut body = _responses_body_min("m");
        body["seed"] = serde_json::json!(-1);
        match responses_params_from_json(&body) {
            ResponsesParamsResult::Ok(params) => assert_eq!(params.seed, Some(-1)),
            ResponsesParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    #[tokio::test]
    async fn test_responses_rejects_string_max_output_tokens() {
        let mut body = _responses_body_min("m");
        body["max_output_tokens"] = serde_json::json!("16");
        let v = _responses_err(body).await;
        assert_eq!(v["error"]["param"], "max_output_tokens");
    }

    #[test]
    fn test_responses_accepts_null_temperature() {
        // Explicit null must not 400 — it is treated as "absent".
        let mut body = _responses_body_min("m");
        body["temperature"] = serde_json::Value::Null;
        match responses_params_from_json(&body) {
            ResponsesParamsResult::Ok(p) => assert!(p.temperature.is_none()),
            ResponsesParamsResult::Err(_) => panic!("expected Ok"),
        }
    }

    #[test]
    fn test_build_responses_body_shape() {
        let outcome = crate::queue::streaming::StreamOutcome {
            text: "a joke".to_string(),
            finish_reason: "stop".to_string(),
            usage: Some(crate::queue::streaming::UsageBlock {
                prompt_tokens: 5,
                completion_tokens: 7,
                total_tokens: 12,
            }),
            attempt_id: "att".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let bytes = build_responses_body("m", "req-1", &outcome).expect("ok");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "response");
        assert_eq!(v["status"], "completed");
        assert!(v["id"].as_str().unwrap().starts_with("resp-"));
        assert_eq!(v["output"][0]["type"], "message");
        assert_eq!(v["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(v["output"][0]["content"][0]["text"], "a joke");
        assert_eq!(v["usage"]["input_tokens"], 5);
        assert_eq!(v["usage"]["output_tokens"], 7);
    }

    /// ``n=0`` rejects as invalid_request (must be positive).
    #[tokio::test]
    async fn test_chat_params_from_json_rejects_n_zero() {
        let mut body = _chat_body_min("m");
        body["n"] = serde_json::json!(0);
        let v = _expect_chat_err(body).await;
        assert_eq!(v["error"]["code"], "invalid_request");
        assert_eq!(v["error"]["param"], "n");
    }

    #[test]
    fn test_chat_params_from_json_accepts_stop_string_or_array() {
        let mut body = _chat_body_min("m");
        body["stop"] = serde_json::json!("</s>");
        let p = _expect_chat_ok(body);
        assert_eq!(p.stop.as_deref(), Some(&["</s>".to_string()][..]));

        let mut body = _chat_body_min("m");
        body["stop"] = serde_json::json!(["a", "b"]);
        let p = _expect_chat_ok(body);
        assert_eq!(p.stop.as_deref().unwrap().len(), 2);
    }

    #[test]
    fn test_build_chat_completion_body_composes_openai_shape() {
        let outcome = crate::queue::streaming::StreamOutcome {
            text: "Hi there!".to_string(),
            finish_reason: "stop".to_string(),
            usage: Some(crate::queue::streaming::UsageBlock {
                prompt_tokens: 5,
                completion_tokens: 3,
                total_tokens: 8,
            }),
            attempt_id: "att-xyz".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let bytes = build_chat_completion_body("Qwen/Qwen3-4B-Instruct-2507", "req-1", &outcome)
            .expect("ok");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["model"], "Qwen/Qwen3-4B-Instruct-2507");
        assert_eq!(v["id"], "chatcmpl-req-1");
        assert_eq!(v["choices"][0]["index"], 0);
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["message"]["content"], "Hi there!");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        // OpenAI always includes `logprobs` (null when not requested).
        assert!(v["choices"][0]["logprobs"].is_null());
        // OpenAI response envelope always carries `system_fingerprint`;
        // it is now a meaningful, non-null per-(model, build) value.
        assert!(
            v.as_object().unwrap().contains_key("system_fingerprint"),
            "system_fingerprint must be present in the blocking body"
        );
        let fp = v["system_fingerprint"]
            .as_str()
            .expect("non-null fingerprint");
        assert!(
            fp.starts_with("fp_"),
            "fingerprint must be the fp_ form, got {fp}"
        );
        assert_eq!(v["usage"]["prompt_tokens"], 5);
        assert_eq!(v["usage"]["completion_tokens"], 3);
        assert_eq!(v["usage"]["total_tokens"], 8);
    }

    #[test]
    fn test_system_fingerprint_is_stable_and_model_sensitive() {
        let a = system_fingerprint("Qwen/Qwen3-4B-Instruct-2507");
        // Stable for a fixed (model, gateway build).
        assert_eq!(a, system_fingerprint("Qwen/Qwen3-4B-Instruct-2507"));
        assert!(a.starts_with("fp_"));
        // Changes when the model changes (different backend config).
        assert_ne!(a, system_fingerprint("BAAI/bge-m3"));
    }

    #[tokio::test]
    async fn test_moderations_returns_501_not_implemented() {
        let resp = proxy_moderations().await;
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = axum::body::to_bytes(resp.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // Explicit OpenAI-shaped 501 — never a silent "not flagged".
        assert_eq!(v["error"]["code"], "not_implemented");
        assert_eq!(v["error"]["type"], "server_error");
    }

    /// ``content_filter`` and ``function_call`` are valid OpenAI finish
    /// reasons emitted by the worker (see
    /// ``queue::streaming::is_known_finish_reason``). They must pass
    /// through unchanged rather than collapse to ``stop`` — otherwise a
    /// safety-stopped completion is silently reported to the client as a
    /// clean stop.
    #[test]
    fn test_map_chat_finish_reason_preserves_content_filter() {
        assert_eq!(map_chat_finish_reason("content_filter"), "content_filter");
        assert_eq!(map_chat_finish_reason("function_call"), "function_call");
    }

    /// When the worker returned per-token logprobs the non-streaming body
    /// surfaces them in the OpenAI ``{content: [...], refusal: null}``
    /// shape on ``choices[0].logprobs`` — previously hardcoded null.
    #[test]
    fn test_build_chat_completion_body_emits_logprobs() {
        let outcome = crate::queue::streaming::StreamOutcome {
            text: "Hi".to_string(),
            finish_reason: "stop".to_string(),
            usage: Some(crate::queue::streaming::UsageBlock {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            }),
            attempt_id: "a".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: Some(vec![serde_json::json!({
                "token": "Hi",
                "logprob": -0.5,
                "bytes": [72, 105],
                "top_logprobs": [],
            })]),
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let bytes = build_chat_completion_body("m", "r", &outcome).expect("ok");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let lp = &v["choices"][0]["logprobs"];
        assert!(lp["content"].is_array(), "logprobs.content must be present");
        assert_eq!(lp["content"][0]["token"], "Hi");
        assert_eq!(lp["content"][0]["logprob"], -0.5);
        assert!(lp["refusal"].is_null());
    }

    /// No logprobs requested → ``choices[0].logprobs`` stays null.
    #[test]
    fn test_build_chat_completion_body_logprobs_null_when_absent() {
        let outcome = crate::queue::streaming::StreamOutcome {
            text: "Hi".to_string(),
            finish_reason: "stop".to_string(),
            usage: Some(crate::queue::streaming::UsageBlock {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            }),
            attempt_id: "a".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let bytes = build_chat_completion_body("m", "r", &outcome).expect("ok");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["choices"][0]["logprobs"].is_null());
    }

    /// When the worker observed tool-call deltas across the stream
    /// the aggregated chat-completion body surfaces them on
    /// ``choices[0].message.tool_calls`` with ``finish_reason:
    /// "tool_calls"``. ``message.content`` is JSON ``null`` (not the
    /// empty string) to match OpenAI's contract.
    #[test]
    fn test_chat_completion_body_aggregates_tool_calls() {
        let outcome = crate::queue::streaming::StreamOutcome {
            text: String::new(),
            finish_reason: "tool_calls".to_string(),
            usage: Some(crate::queue::streaming::UsageBlock {
                prompt_tokens: 7,
                completion_tokens: 11,
                total_tokens: 18,
            }),
            attempt_id: "att".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: Some(vec![crate::queue::streaming::AggregatedToolCall {
                index: 0,
                id: "call_xyz".to_string(),
                kind: "function".to_string(),
                name: "get_weather".to_string(),
                arguments: r#"{"city":"Paris"}"#.to_string(),
            }]),
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let bytes = build_chat_completion_body("m", "req-tc", &outcome).expect("ok");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
        let msg = &v["choices"][0]["message"];
        assert_eq!(msg["role"], "assistant");
        assert!(msg["content"].is_null());
        let calls = msg["tool_calls"].as_array().expect("tool_calls array");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_xyz");
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        assert_eq!(calls[0]["function"]["arguments"], r#"{"city":"Paris"}"#);
    }

    #[tokio::test]
    async fn test_build_chat_completion_body_missing_usage_500() {
        let outcome = crate::queue::streaming::StreamOutcome {
            text: "ok".to_string(),
            finish_reason: "stop".to_string(),
            usage: None,
            attempt_id: "att".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let resp = build_chat_completion_body("m", "req-1", &outcome).expect_err("missing usage");
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(resp.into_body(), 16 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["code"], "malformed_worker_response");
    }

    #[test]
    fn test_map_chat_finish_reason_collapses_to_openai_vocab() {
        assert_eq!(map_chat_finish_reason("stop"), "stop");
        assert_eq!(map_chat_finish_reason("length"), "length");
        // SIE-native ``cancelled`` / ``error`` aren't valid OpenAI
        // finish reasons; they collapse to ``stop`` so SDKs that
        // strictly validate the enum still parse the response. (Error
        // outcomes never reach the success body path.)
        assert_eq!(map_chat_finish_reason("cancelled"), "stop");
        assert_eq!(map_chat_finish_reason("weird-future-value"), "stop");
    }

    // ── Fix 5: 429 rate_limit_error for KV-saturated + pool-full path

    #[tokio::test]
    async fn test_build_streaming_error_response_no_consumers_is_provisioning() {
        let err = StreamingDriverErr::PublishFailed {
            message: "nats: no consumers available".to_string(),
            retry_after: Some(PROVISIONING_RETRY_AFTER),
        };
        let resp = build_streaming_error_response(&err);
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get("retry-after").unwrap(),
            PROVISIONING_RETRY_AFTER
        );
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            err_code::PROVISIONING
        );

        let body = axum::body::to_bytes(resp.into_body(), 16 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], oai_type::SERVER_ERROR);
        assert_eq!(v["error"]["code"], oai_code::PROVISIONING);
    }

    #[tokio::test]
    async fn test_build_streaming_publish_failed_for_sse_no_consumers_is_provisioning() {
        let resp = build_streaming_publish_failed_for_sse(
            "no consumers available for work stream",
            Some(PROVISIONING_RETRY_AFTER),
        );
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers().get("retry-after").unwrap(),
            PROVISIONING_RETRY_AFTER
        );
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            err_code::PROVISIONING
        );

        let body = axum::body::to_bytes(resp.into_body(), 16 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], oai_type::SERVER_ERROR);
        assert_eq!(v["error"]["code"], oai_code::PROVISIONING);
    }

    /// ``rate_limit_exceeded`` worker code maps to 429 with the OpenAI
    /// ``rate_limit_error`` envelope ``type`` and the
    /// ``rate_limit_exceeded`` ``code`` discriminator. The
    /// :func:`build_streaming_error_response` helper also stamps a
    /// ``Retry-After: 1`` header so the SDK retries with bounded
    /// backoff.
    #[tokio::test]
    async fn test_build_streaming_error_response_rate_limit_returns_429() {
        let err = StreamingDriverErr::WorkerError {
            code: "rate_limit_exceeded".to_string(),
            message: "KV cache saturated and pool republish failed".to_string(),
            param: None,
            retry_after_s: None,
            request_id: "req-rl-1".to_string(),
            attempt_id: "att-rl-1".to_string(),
        };
        let resp = build_streaming_error_response(&err);
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        // Retry-After header is mandatory on 429 per the OpenAI contract.
        let retry = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok());
        assert_eq!(retry, Some("1"), "missing Retry-After header on 429");
        let body = axum::body::to_bytes(resp.into_body(), 16 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        // OpenAI envelope: {"error": {"message", "type", "param", "code"}}
        let err_obj = v.get("error").expect("error envelope");
        assert_eq!(err_obj["type"], "rate_limit_error");
        assert_eq!(err_obj["code"], "rate_limit_exceeded");
        // ``param`` is non-field-specific for this failure mode.
        assert!(err_obj["param"].is_null(), "param must be null on 429");
        // SIE-native attempt_id is preserved alongside the envelope so
        // SIE-aware SDKs can correlate retries.
        assert_eq!(err_obj["attempt_id"], "att-rl-1");
    }

    #[tokio::test]
    async fn test_oversized_generate_payload_worker_error_returns_413() {
        let err = StreamingDriverErr::WorkerError {
            code: PAYLOAD_TOO_LARGE_ERROR_CODE.to_string(),
            message: "referenced payload exceeds the worker size limit".to_string(),
            param: None,
            retry_after_s: None,
            request_id: "req-large-1".to_string(),
            attempt_id: "att-large-1".to_string(),
        };
        let response = build_streaming_error_response(&err);

        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            response.headers().get("x-sie-error-code").unwrap(),
            PAYLOAD_TOO_LARGE_ERROR_CODE
        );
        let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["type"], oai_type::INVALID_REQUEST);
        assert_eq!(value["error"]["code"], oai_code::INVALID_REQUEST);
        assert_eq!(value["error"]["attempt_id"], "att-large-1");
    }

    #[tokio::test]
    async fn test_streaming_unsupported_field_preserves_exact_parameter() {
        let err = StreamingDriverErr::WorkerError {
            code: oai_code::UNSUPPORTED_FIELD.to_string(),
            message: "top_k is unavailable".to_string(),
            param: Some("top_k".to_string()),
            retry_after_s: None,
            request_id: "req-unsupported-1".to_string(),
            attempt_id: "att-unsupported-1".to_string(),
        };
        let response = build_streaming_error_response(&err);

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert!(response.headers().get("retry-after").is_none());
        let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], oai_code::UNSUPPORTED_FIELD);
        assert_eq!(value["error"]["param"], "top_k");
        assert_eq!(value["error"]["attempt_id"], "att-unsupported-1");
    }

    /// The cold-start quota refusal must reach the wire with NO Retry-After:
    /// the blanket 429 header block previously stamped `Retry-After: 1` on it,
    /// re-creating the exact auto-retry hot loop the hourly ceiling exists to
    /// prevent (review finding) — and the envelope code must be the typed
    /// refusal, never a generic `inference_error`.
    #[tokio::test]
    async fn test_streaming_cold_start_refusal_is_429_without_retry_after() {
        let err = StreamingDriverErr::WorkerError {
            code: COLD_START_RATE_LIMITED_ERROR_CODE.to_string(),
            message: "org exceeded its sealed cold-starts-per-hour ceiling".to_string(),
            param: None,
            retry_after_s: None,
            request_id: "req-cold-1".to_string(),
            attempt_id: "att-cold-1".to_string(),
        };
        let resp = build_streaming_error_response(&err);

        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(resp.headers().get("retry-after").is_none());
        let body = axum::body::to_bytes(resp.into_body(), 16 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], oai_type::RATE_LIMIT);
        assert_eq!(v["error"]["code"], COLD_START_RATE_LIMITED_ERROR_CODE);
    }

    /// The worker's typed ``empty_model_output`` terminal (#3136) must reach
    /// buffered consumers verbatim — not flattened to the generic
    /// ``inference_error`` fallback — with the same 500 / ``server_error`` /
    /// no-Retry-After convention buffered inference errors already use
    /// (terminal, not a capacity state).
    #[tokio::test]
    async fn test_streaming_empty_model_output_code_preserved_on_buffered_path() {
        let err = StreamingDriverErr::WorkerError {
            code: oai_code::EMPTY_MODEL_OUTPUT.to_string(),
            message: "model produced no visible output text".to_string(),
            param: None,
            retry_after_s: None,
            request_id: "req-empty-1".to_string(),
            attempt_id: "att-empty-1".to_string(),
        };
        let resp = build_streaming_error_response(&err);

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        // Terminal code: never a retry hint.
        assert!(resp.headers().get("retry-after").is_none());
        assert_eq!(
            resp.headers().get("x-sie-request-id").unwrap(),
            "req-empty-1"
        );
        let body = axum::body::to_bytes(resp.into_body(), 16 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], oai_type::SERVER_ERROR);
        assert_eq!(v["error"]["code"], oai_code::EMPTY_MODEL_OUTPUT);
        assert_eq!(v["error"]["attempt_id"], "att-empty-1");
        // The retry classifier must keep treating it as terminal.
        assert_eq!(
            worker_error_retry_after(oai_code::EMPTY_MODEL_OUTPUT, None),
            None
        );
    }

    #[tokio::test]
    async fn test_streaming_model_load_failed_uses_terminal_sdk_contract() {
        let err = StreamingDriverErr::WorkerError {
            code: MODEL_LOAD_FAILED_ERROR_CODE.to_string(),
            message: "SENSITIVE_WORKER_FAILURE_SENTINEL".to_string(),
            param: None,
            retry_after_s: None,
            request_id: "req-load-failed-1".to_string(),
            attempt_id: "att-load-failed-1".to_string(),
        };
        let resp = build_streaming_error_response(&err);

        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        assert_eq!(
            resp.headers().get("x-sie-error-code").unwrap(),
            MODEL_LOAD_FAILED_ERROR_CODE
        );
        assert_eq!(
            resp.headers().get("x-sie-request-id").unwrap(),
            "req-load-failed-1"
        );
        assert!(resp.headers().get("retry-after").is_none());

        let body = axum::body::to_bytes(resp.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], MODEL_LOAD_FAILED_ERROR_CODE);
        assert_eq!(value["error"]["message"], MODEL_LOAD_FAILED_PUBLIC_MESSAGE);
        assert_eq!(value["error"]["error_class"], MODEL_LOAD_FAILED_ERROR_CODE);
        assert_eq!(value["error"]["attempts"], 1);
        assert_eq!(value["error"]["permanent"], true);
        assert!(!String::from_utf8_lossy(&body).contains("SENSITIVE_WORKER_FAILURE_SENTINEL"));
    }

    #[tokio::test]
    async fn test_streaming_unknown_worker_error_fails_closed() {
        let err = StreamingDriverErr::WorkerError {
            code: "SENSITIVE_WORKER_ERROR_CODE_SENTINEL".to_string(),
            message: "SENSITIVE_WORKER_FAILURE_SENTINEL".to_string(),
            param: None,
            retry_after_s: None,
            request_id: "req-unknown-1".to_string(),
            attempt_id: "att-unknown-1".to_string(),
        };
        let resp = build_streaming_error_response(&err);

        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert!(resp.headers().get("retry-after").is_none());
        let body = axum::body::to_bytes(resp.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "inference_error");
        assert_eq!(
            value["error"]["message"],
            crate::queue::streaming::UNKNOWN_WORKER_ERROR_PUBLIC_MESSAGE
        );
        let serialized = String::from_utf8_lossy(&body);
        assert!(!serialized.contains("SENSITIVE_WORKER_ERROR_CODE_SENTINEL"));
        assert!(!serialized.contains("SENSITIVE_WORKER_FAILURE_SENTINEL"));
    }

    #[tokio::test]
    async fn test_buffered_worker_inference_error_sanitizes_untrusted_message() {
        let sentinel = "SENSITIVE_BACKEND_SENTINEL";
        let err = StreamingDriverErr::WorkerError {
            code: "inference_error".to_string(),
            message: sentinel.to_string(),
            param: Some(sentinel.to_string()),
            retry_after_s: None,
            request_id: "req-sensitive-1".to_string(),
            attempt_id: "att-sensitive-1".to_string(),
        };

        let response = build_streaming_error_response(&err);

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            value["error"]["message"],
            "internal error during generation"
        );
        assert!(value["error"]["param"].is_null());
        assert!(!value.to_string().contains(sentinel));
    }

    #[tokio::test]
    async fn test_buffered_unknown_worker_error_sanitizes_untrusted_message() {
        let sentinel = "SENSITIVE_WORKER_ERROR_CODE_SENTINEL";
        let err = StreamingDriverErr::WorkerError {
            code: sentinel.to_string(),
            message: sentinel.to_string(),
            param: Some(sentinel.to_string()),
            retry_after_s: None,
            request_id: "req-future-1".to_string(),
            attempt_id: "att-future-1".to_string(),
        };

        let response = build_streaming_error_response(&err);

        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(value["error"]["code"], "inference_error");
        assert_eq!(
            value["error"]["message"],
            crate::queue::streaming::UNKNOWN_WORKER_ERROR_PUBLIC_MESSAGE
        );
        assert!(value["error"]["param"].is_null());
        assert!(!value.to_string().contains(sentinel));
    }

    async fn assert_streaming_worker_error_is_retryable(
        code: &'static str,
        retry_after: &'static str,
    ) {
        let err = StreamingDriverErr::WorkerError {
            code: code.to_string(),
            message: "Retryable worker error; retry later.".to_string(),
            param: None,
            retry_after_s: None,
            request_id: "req-retry-1".to_string(),
            attempt_id: "att-retry-1".to_string(),
        };
        let resp = build_streaming_error_response(&err);

        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(resp.headers().get("retry-after").unwrap(), retry_after);
        assert_eq!(resp.headers().get("x-sie-error-code").unwrap(), code);
        let body = axum::body::to_bytes(resp.into_body(), 16 * 1024)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let err_obj = v.get("error").expect("error envelope");
        assert_eq!(err_obj["type"], oai_type::SERVER_ERROR);
        assert_eq!(err_obj["code"], code);
        assert_eq!(err_obj["attempt_id"], "att-retry-1");
    }

    #[tokio::test]
    async fn test_build_streaming_error_response_retryable_worker_errors_return_503() {
        assert_streaming_worker_error_is_retryable(
            RESOURCE_EXHAUSTED_ERROR_CODE,
            RESOURCE_EXHAUSTED_RETRY_AFTER,
        )
        .await;
        assert_streaming_worker_error_is_retryable(
            MODEL_LOADING_ERROR_CODE,
            MODEL_LOADING_RETRY_AFTER,
        )
        .await;
        assert_streaming_worker_error_is_retryable(
            LORA_LOADING_ERROR_CODE,
            LORA_LOADING_RETRY_AFTER,
        )
        .await;
    }

    #[tokio::test]
    async fn test_resource_exhausted_uses_worker_configured_retry_hint() {
        let err = StreamingDriverErr::WorkerError {
            code: RESOURCE_EXHAUSTED_ERROR_CODE.to_string(),
            message: "scheduler full".to_string(),
            param: Some("SENSITIVE_CAPACITY_PARAM".to_string()),
            retry_after_s: Some(12),
            request_id: "req-retry-configured".to_string(),
            attempt_id: "att-retry-configured".to_string(),
        };

        let response = build_streaming_error_response(&err);

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers().get("retry-after").unwrap(), "12");
        assert_eq!(
            response.headers().get("x-sie-error-code").unwrap(),
            RESOURCE_EXHAUSTED_ERROR_CODE
        );
        let body = axum::body::to_bytes(response.into_body(), 16 * 1024)
            .await
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(value["error"]["param"].is_null());
    }

    /// HTTP status mapping unit test — guards against an off-by-one
    /// edit that drops the 429 mapping without touching
    /// ``build_streaming_error_response``.
    #[test]
    fn test_worker_error_http_status_rate_limit_is_429() {
        assert_eq!(
            worker_error_http_status("rate_limit_exceeded"),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[test]
    fn test_worker_error_http_status_model_load_failed_is_502() {
        assert_eq!(
            worker_error_http_status(MODEL_LOAD_FAILED_ERROR_CODE),
            StatusCode::BAD_GATEWAY
        );
    }

    /// The sealed cold-start quota refusal (#2426) is a deliberate 429 with
    /// a rate-limit envelope type — never a generic server error — and stays
    /// off the retryable lists (no Retry-After: an hourly quota must not
    /// invite auto-retry storms).
    #[test]
    fn test_cold_start_rate_limited_maps_to_429_not_500() {
        assert_eq!(
            worker_error_http_status(COLD_START_RATE_LIMITED_ERROR_CODE),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            worker_error_openai_type(COLD_START_RATE_LIMITED_ERROR_CODE),
            oai_type::RATE_LIMIT
        );
        assert!(worker_error_retry_after(COLD_START_RATE_LIMITED_ERROR_CODE, None).is_none());
    }

    /// Envelope ``type`` mapping unit test.
    #[test]
    fn test_worker_error_openai_type_rate_limit() {
        assert_eq!(
            worker_error_openai_type("rate_limit_exceeded"),
            oai_type::RATE_LIMIT
        );
    }

    /// The shared retry classifier maps each retryable worker code to its
    /// `Retry-After` + canonical code and returns `None` for terminal codes.
    /// Locks the single source of truth the streaming and unary error paths
    /// share (see `worker_error_retry_after`).
    #[test]
    fn test_worker_error_retry_after_maps_retryable_codes() {
        assert_eq!(
            worker_error_retry_after(RESOURCE_EXHAUSTED_ERROR_CODE, None),
            Some((
                RESOURCE_EXHAUSTED_RETRY_AFTER.to_string(),
                RESOURCE_EXHAUSTED_ERROR_CODE
            ))
        );
        assert_eq!(
            worker_error_retry_after(MODEL_LOADING_ERROR_CODE, None),
            Some((
                MODEL_LOADING_RETRY_AFTER.to_string(),
                MODEL_LOADING_ERROR_CODE
            ))
        );
        assert_eq!(
            worker_error_retry_after(LORA_LOADING_ERROR_CODE, None),
            Some((
                LORA_LOADING_RETRY_AFTER.to_string(),
                LORA_LOADING_ERROR_CODE
            ))
        );
        // Terminal / non-retryable codes carry no retry hint.
        assert_eq!(worker_error_retry_after("invalid_request", None), None);
        assert_eq!(
            worker_error_retry_after("transport_failure", Some(12)),
            None
        );
        assert_eq!(
            worker_error_retry_after(RESOURCE_EXHAUSTED_ERROR_CODE, Some(12)),
            Some(("12".to_string(), RESOURCE_EXHAUSTED_ERROR_CODE))
        );
        for invalid in [0, 61] {
            assert_eq!(
                worker_error_retry_after(RESOURCE_EXHAUSTED_ERROR_CODE, Some(invalid)),
                Some((
                    RESOURCE_EXHAUSTED_RETRY_AFTER.to_string(),
                    RESOURCE_EXHAUSTED_ERROR_CODE
                ))
            );
        }
    }

    /// Regression guard: the non-streaming aggregating path (the
    /// `stream: false` default) still composes the OpenAI
    /// ``chat.completion`` body exactly as before, character for
    /// character — no accidental SSE-style envelope leakage into
    /// the buffered response.
    #[test]
    fn test_stream_false_chat_completion_body_unchanged() {
        let outcome = crate::queue::streaming::StreamOutcome {
            text: "Hi".to_string(),
            finish_reason: "stop".to_string(),
            usage: Some(crate::queue::streaming::UsageBlock {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
            }),
            attempt_id: "att".to_string(),
            ttft_ms: None,
            tpot_ms: None,
            error: None,
            origin: crate::queue::streaming::StreamOutcomeOrigin::WorkerTerminal,
            tool_calls: None,
            logprobs: None,
            candidates: Vec::new(),
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        };
        let bytes = build_chat_completion_body("m", "req-x", &outcome).expect("ok");
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Non-streaming uses `chat.completion`, not `chat.completion.chunk`.
        assert_eq!(v["object"], "chat.completion");
        // And the message is fully assembled (no `delta` field).
        assert_eq!(v["choices"][0]["message"]["content"], "Hi");
        assert!(v["choices"][0].get("delta").is_none());
    }

    // ── M10: per-profile LoRA adapter validation ───────────────────
    //
    // ``validate_lora_for_profile`` is the seam the chat and generate
    // lora_adapter gates share. These tests pin down the per-profile
    // scoping discipline: a request for profile A must reject an
    // adapter that's only configured on profile B, even though the
    // model's union ``info_extras.lora_adapters`` lists both.

    fn entry_with_per_profile_adapters() -> crate::types::model::ModelEntry {
        use crate::types::model::{ModelEntry, ModelInfoExtras};
        let mut per_profile = std::collections::HashMap::new();
        per_profile.insert(
            "default".to_string(),
            vec!["a1".to_string(), "a2".to_string()],
        );
        per_profile.insert("a100".to_string(), vec!["b1".to_string()]);
        let info_extras = ModelInfoExtras {
            inputs: vec!["text".to_string()],
            outputs: vec!["tokens".to_string()],
            dims: std::collections::HashMap::new(),
            max_sequence_length: None,
            revision: None,
            max_output_tokens: None,
            profile_max_output_tokens: std::collections::HashMap::new(),
            grammar_capabilities: None,
            streaming_supported: None,
            grammar_profile: None,
            profile_parents: std::collections::HashMap::new(),
            tools_supported: None,
            code: false,
            sql: false,
            guard: false,
            // Union summary (back-compat). The gate must NOT consult this.
            lora_adapters: Some(vec!["a1".to_string(), "a2".to_string(), "b1".to_string()]),
            profile_lora_adapters: Some(per_profile),
        };
        ModelEntry {
            name: "acme/multi".to_string(),
            canonical_base_model: "acme/multi".to_string(),
            canonical_profile: "default".to_string(),
            pool: None,
            bundles: Vec::new(),
            adapter_modules: std::collections::HashSet::new(),
            profile_names: ["default".to_string(), "a100".to_string()]
                .iter()
                .cloned()
                .collect(),
            profile_configs: std::collections::HashMap::new(),
            info_extras,
        }
    }

    fn successful_item_result(
        payload: Value,
        units: Option<publisher::UnitCounts>,
    ) -> publisher::WorkResult {
        publisher::WorkResult {
            work_item_id: "request.0".to_string(),
            request_id: "request".to_string(),
            item_index: 0,
            success: true,
            result_msgpack: rmp_serde::to_vec_named(&payload).unwrap(),
            error: None,
            error_code: None,
            inference_ms: None,
            queue_ms: None,
            processing_ms: None,
            worker_id: None,
            tokenization_ms: None,
            postprocessing_ms: None,
            payload_fetch_ms: None,
            units,
            worker_direct: false,
            executed_bundle_config_hash: None,
            execution_identity_sha256: None,
            execution_binding_sha256: None,
        }
    }

    #[test]
    fn test_score_success_body_includes_authoritative_usage_json_and_msgpack() {
        let result = successful_item_result(
            json!([{"item_id": "0", "score": 0.75, "rank": 0}]),
            Some(publisher::UnitCounts {
                input_tokens: Some(19),
                pages: None,
                images: Some(2),
                audio_ms: None,
                pairs: None,
                gpu_second: None,
                output_tokens: None,
            }),
        );
        for use_msgpack in [false, true] {
            let body = build_queue_success_body("score", "reranker", &[&result], use_msgpack);
            let response: Value = if use_msgpack {
                rmp_serde::from_slice(&body).unwrap()
            } else {
                serde_json::from_slice(&body).unwrap()
            };
            assert_eq!(response["usage"]["input_tokens"], 19);
            assert_eq!(response["usage"]["images"], 2);
            assert_eq!(response["scores"][0]["item_id"], "0");
        }
    }

    /// The queue ingress reports the worker's own token count on `encode`, so
    /// the two encode ingresses agree and `/v1/embeddings` has an exact number
    /// to render on the managed path too.
    #[test]
    fn test_encode_success_body_includes_authoritative_usage_json_and_msgpack() {
        let result = successful_item_result(
            json!({"id": "0", "dense": {"dims": 2, "dtype": "float32", "values": [0.1, 0.2]}}),
            Some(publisher::UnitCounts {
                input_tokens: Some(19),
                pages: None,
                images: Some(2),
                audio_ms: None,
                pairs: None,
                gpu_second: None,
                output_tokens: None,
            }),
        );
        for use_msgpack in [false, true] {
            let body = build_queue_success_body("encode", "embedder", &[&result], use_msgpack);
            let response: Value = if use_msgpack {
                rmp_serde::from_slice(&body).unwrap()
            } else {
                serde_json::from_slice(&body).unwrap()
            };
            assert_eq!(response["usage"]["input_tokens"], 19);
            assert_eq!(response["usage"]["images"], 2);
            assert_eq!(response["items"][0]["id"], "0");
        }
    }

    /// A worker that authoritatively read no text reports `0`, not a missing
    /// dimension: zero is a measurement, and the caller is entitled to read it.
    #[test]
    fn test_encode_success_body_reports_an_authoritative_zero() {
        let result = successful_item_result(
            json!({"id": "0"}),
            Some(publisher::UnitCounts {
                input_tokens: Some(0),
                pages: None,
                images: Some(3),
                audio_ms: None,
                pairs: None,
                gpu_second: None,
                output_tokens: None,
            }),
        );
        let body = build_queue_success_body("encode", "embedder", &[&result], false);
        let response: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(response["usage"]["input_tokens"], 0);
        assert_eq!(response["usage"]["images"], 3);
    }

    /// No units, no `usage`. Omission is the only honest rendering of "this
    /// path could not count" — the alternative is an invented number.
    #[test]
    fn test_encode_success_body_omits_usage_without_worker_units() {
        let result = successful_item_result(json!({"id": "0"}), None);
        let body = build_queue_success_body("encode", "embedder", &[&result], false);
        let response: Value = serde_json::from_slice(&body).unwrap();
        assert!(response.get("usage").is_none());
        assert_eq!(response["items"][0]["id"], "0");
    }

    /// `/v1/embeddings` reports the worker's count, NOT `chars / 4`.
    ///
    /// The estimate for this input is 12 (48 characters); the worker measured
    /// 7. The endpoint that carries the most traffic must not be the one whose
    /// `usage` disagrees with the bill.
    #[test]
    fn test_embeddings_usage_prefers_the_worker_count_over_the_estimate() {
        let texts = vec!["x".repeat(48)];
        let estimate = estimate_embedding_tokens(&texts);
        assert_eq!(estimate, 12, "guard: the estimate is chars / 4");

        let usage = embeddings_usage(&json!({"usage": {"input_tokens": 7}}), estimate);
        assert_eq!(usage["prompt_tokens"], 7);
        assert_eq!(usage["total_tokens"], 7);
        assert_eq!(usage["sie_token_source"], "worker");
    }

    /// An authoritative zero survives the compat layer as `0` and stays
    /// labelled exact — it must not fall through to the estimate, which can
    /// never be zero (it floors at 1).
    #[test]
    fn test_embeddings_usage_reports_an_authoritative_zero() {
        let usage = embeddings_usage(&json!({"usage": {"input_tokens": 0}}), 12);
        assert_eq!(usage["prompt_tokens"], 0);
        assert_eq!(usage["total_tokens"], 0);
        assert_eq!(usage["sie_token_source"], "worker");
    }

    /// With no authoritative count the estimate is still emitted — dropping
    /// `usage` would break every OpenAI client on a successful embedding — but
    /// it says so.
    #[test]
    fn test_embeddings_usage_labels_the_character_estimate_fallback() {
        for encode_response in [
            json!({"items": []}),
            json!({"usage": {}}),
            json!({"usage": {"input_tokens": "seven"}}),
            json!({"usage": null}),
        ] {
            let usage = embeddings_usage(&encode_response, 12);
            assert_eq!(usage["prompt_tokens"], 12);
            assert_eq!(usage["total_tokens"], 12);
            assert_eq!(
                usage["sie_token_source"], "character_estimate",
                "an estimate presented as exact is the defect being fixed: {encode_response}",
            );
        }
    }

    #[test]
    fn test_score_success_body_omits_unavailable_usage_fields() {
        let result = successful_item_result(
            json!([{"item_id": "0", "score": 0.75, "rank": 0}]),
            Some(publisher::UnitCounts {
                input_tokens: Some(19),
                pages: None,
                images: None,
                audio_ms: None,
                pairs: None,
                gpu_second: None,
                output_tokens: None,
            }),
        );
        let body = build_queue_success_body("score", "reranker", &[&result], false);
        let response: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(response["usage"]["input_tokens"], 19);
        assert!(response["usage"].get("images").is_none());
    }

    #[test]
    fn test_score_usage_aggregation_rejects_partial_worker_counts() {
        let complete = successful_item_result(
            json!([]),
            Some(publisher::UnitCounts {
                input_tokens: Some(19),
                pages: None,
                images: Some(1),
                audio_ms: None,
                pairs: None,
                gpu_second: None,
                output_tokens: None,
            }),
        );
        let missing_tokens = successful_item_result(
            json!([]),
            Some(publisher::UnitCounts {
                input_tokens: None,
                pages: None,
                images: Some(1),
                audio_ms: None,
                pairs: None,
                gpu_second: None,
                output_tokens: None,
            }),
        );
        assert!(aggregate_result_usage(&[&complete, &missing_tokens]).is_none());
    }

    #[test]
    fn test_score_usage_aggregation_rejects_overflow() {
        let maximum = successful_item_result(
            json!([]),
            Some(publisher::UnitCounts {
                input_tokens: Some(u64::MAX),
                pages: None,
                images: None,
                audio_ms: None,
                pairs: None,
                gpu_second: None,
                output_tokens: None,
            }),
        );
        let one = successful_item_result(
            json!([]),
            Some(publisher::UnitCounts {
                input_tokens: Some(1),
                pages: None,
                images: None,
                audio_ms: None,
                pairs: None,
                gpu_second: None,
                output_tokens: None,
            }),
        );
        assert!(aggregate_result_usage(&[&maximum, &one]).is_none());
    }

    #[test]
    fn test_normalize_rerank_request_accepts_v1_and_v2_subsets() {
        let v1 = normalize_rerank_request(
            &json!({
                "model": "Qwen__Qwen3-Reranker-0.6B",
                "query": "query",
                "documents": ["first", "second"],
                "top_n": 1,
                "return_documents": true,
                "options": {"profile": "long", "max_seq_length": 512},
            }),
            RerankCompatibilityVersion::V1,
        )
        .unwrap();
        assert_eq!(v1.documents.len(), 2);
        assert_eq!(v1.top_n, Some(1));
        assert!(v1.return_documents);
        assert_eq!(v1.options["profile"], "long");
        assert_eq!(v1.options["max_seq_length"], 512);

        let v2 = normalize_rerank_request(
            &json!({
                "model": "Qwen__Qwen3-Reranker-0.6B",
                "query": "query",
                "documents": ["first", "second"],
                "top_n": 2,
                "max_tokens_per_doc": null,
                "priority": null,
                "options": {"profile": null, "max_seq_length": null},
            }),
            RerankCompatibilityVersion::V2,
        )
        .unwrap();
        assert_eq!(v2.documents.len(), 2);
        assert_eq!(v2.top_n, Some(2));
        assert!(!v2.return_documents);
        assert!(v2.options["profile"].is_null());
        assert!(v2.options["max_seq_length"].is_null());
    }

    #[test]
    fn test_normalize_rerank_request_rejects_unknown_and_adversarial_fields() {
        for (version, body, expected) in [
            (
                RerankCompatibilityVersion::V1,
                json!({"model": "m", "query": "q", "documents": ["d"], "priority": 1}),
                "field 'priority' is not supported",
            ),
            (
                RerankCompatibilityVersion::V2,
                json!({"model": "m", "query": "q", "documents": ["d"], "priority": 1}),
                "field 'priority' is not supported; omit it or send null",
            ),
            (
                RerankCompatibilityVersion::V2,
                json!({"model": "m", "query": "q", "documents": ["d"], "max_tokens_per_doc": 512}),
                "field 'max_tokens_per_doc' is not supported; omit it or send null",
            ),
            (
                RerankCompatibilityVersion::V2,
                json!({"model": "m", "query": "q", "documents": ["d"], "return_documents": true}),
                "field 'return_documents' is not supported",
            ),
            (
                RerankCompatibilityVersion::V1,
                json!({"model": "m", "query": "q", "documents": ["d"], "top_n": true}),
                "'top_n' must be a positive integer",
            ),
            (
                RerankCompatibilityVersion::V1,
                json!({"model": "m", "query": "q", "documents": ["d"], "options": []}),
                "'options' must be an object",
            ),
            (
                RerankCompatibilityVersion::V1,
                json!({"model": "m", "query": "q", "documents": ["d"], "options": {"max_length": 512}}),
                "options field 'max_length' is not supported",
            ),
            (
                RerankCompatibilityVersion::V1,
                json!({"model": "m", "query": " ", "documents": ["d"]}),
                "'query' must be a non-blank string",
            ),
            (
                RerankCompatibilityVersion::V1,
                json!({"model": "m", "query": "q", "documents": [" \t"]}),
                "'documents' must contain only non-blank strings",
            ),
            (
                RerankCompatibilityVersion::V1,
                json!({"model": "m", "query": "q", "documents": ["d"], "options": {"profile": " "}}),
                "options.profile must be a non-blank string or null",
            ),
            (
                RerankCompatibilityVersion::V1,
                json!({"model": "m", "query": "q", "documents": ["d"], "options": {"max_seq_length": 0}}),
                "options.max_seq_length must be a positive platform-sized integer or null",
            ),
        ] {
            assert_eq!(
                normalize_rerank_request(&body, version).unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn test_compatibility_model_uri_encodes_untrusted_model_delimiters() {
        for endpoint in ["encode", "score"] {
            let uri = compatibility_model_uri(endpoint, "/Qwen/model?pool=other#fragment").unwrap();
            assert!(uri.query().is_none());
            let prefix = format!("/v1/{endpoint}/");
            let raw_model = uri.path().strip_prefix(&prefix).unwrap();
            assert_eq!(
                decode_model_path(raw_model).unwrap(),
                "Qwen/model?pool=other#fragment"
            );
        }
        assert_eq!(
            compatibility_model_uri("score", "///").unwrap_err(),
            "'model' must contain a model id"
        );
    }

    #[test]
    fn test_normalize_rerank_request_rejects_oversized_candidates() {
        let documents = vec![Value::String("d".to_string()); MAX_RERANK_DOCUMENTS + 1];
        let error = normalize_rerank_request(
            &json!({"model": "m", "query": "q", "documents": documents}),
            RerankCompatibilityVersion::V1,
        )
        .unwrap_err();
        assert!(error.contains("exceeds the maximum of 1000"));
    }

    #[test]
    fn test_rerank_response_projects_top_n_documents_and_partial_usage() {
        let response = rerank_response_from_score(
            &json!({
                "model": "reranker",
                "scores": [
                    {"item_id": "1", "score": 0.9, "rank": 0},
                    {"item_id": "0", "score": 0.2, "rank": 1}
                ],
                "usage": {"input_tokens": 42}
            }),
            &["first".to_string(), "second".to_string()],
            Some(1),
            true,
        )
        .unwrap();
        assert_eq!(response["results"].as_array().unwrap().len(), 1);
        assert_eq!(response["results"][0]["index"], 1);
        assert_eq!(response["results"][0]["document"]["text"], "second");
        assert_eq!(response["usage"]["input_tokens"], 42);
        assert!(response["usage"].get("images").is_none());
    }

    #[test]
    fn test_rerank_response_rejects_partial_or_duplicate_scores() {
        let documents = ["first".to_string(), "second".to_string()];
        for (native, expected) in [
            (
                json!({"model": "m", "scores": [{"item_id": "0", "score": 0.5, "rank": 0}], "usage": {"input_tokens": 4}}),
                "item count mismatch",
            ),
            (
                json!({"model": "m", "scores": [{"item_id": "0", "score": 0.5, "rank": 0}, {"item_id": "0", "score": 0.4, "rank": 1}], "usage": {"input_tokens": 4}}),
                "duplicate item_id",
            ),
        ] {
            let error = rerank_response_from_score(&native, &documents, None, false).unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn test_rerank_response_rejects_missing_or_malformed_usage() {
        let scores = json!([{"item_id": "0", "score": 0.5, "rank": 0}]);
        for (usage, expected) in [
            (Value::Null, "missing authoritative usage"),
            (json!({"images": 1}), "missing input_tokens"),
            (
                json!({"input_tokens": 4, "images": -1}),
                "invalid images count",
            ),
        ] {
            let native = json!({"model": "m", "scores": scores.clone(), "usage": usage});
            let error =
                rerank_response_from_score(&native, &["doc".to_string()], None, false).unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn test_chat_lora_adapter_rejected_when_not_in_selected_profile() {
        // Chat path: the gate always asks for profile ``"default"`` (chat
        // has no profile param; variants are addressed via the model
        // spec and arrive narrowed). Requesting ``b1`` — which is only
        // configured on the ``a100`` profile of the base entry — must
        // reject as ``UnknownAdapter`` even though the model's union
        // ``lora_adapters`` lists it. M10 regression.
        let entry = entry_with_per_profile_adapters();
        assert_eq!(
            validate_lora_for_profile(&entry, "default", "b1"),
            LoraValidation::UnknownAdapter,
        );
        // Sanity: an adapter actually configured on the default profile
        // passes.
        assert_eq!(
            validate_lora_for_profile(&entry, "default", "a1"),
            LoraValidation::Ok,
        );
    }

    #[test]
    fn test_generate_lora_adapter_rejected_when_not_in_selected_profile() {
        // /v1/generate path: profile name comes from
        // ``params.options.profile``. Requesting ``a1`` — which is
        // configured only on the ``default`` profile — under
        // ``profile=a100`` must reject as ``UnknownAdapter``. The union
        // would have let it through; per-profile scoping must not.
        let entry = entry_with_per_profile_adapters();
        assert_eq!(
            validate_lora_for_profile(&entry, "a100", "a1"),
            LoraValidation::UnknownAdapter,
        );
        // Cross-direction sanity: ``b1`` on ``a100`` is fine.
        assert_eq!(
            validate_lora_for_profile(&entry, "a100", "b1"),
            LoraValidation::Ok,
        );
    }

    #[test]
    fn test_lora_adapter_gate_rejects_unknown_profile_distinctly() {
        // Requests targeting a non-existent profile must surface as
        // ``UnknownProfile`` (translated to ``invalid_request`` with
        // ``param: "profile"`` upstream), not collapsed into
        // ``unknown_lora_adapter``. That distinction matters so SDKs
        // can tell the user "this profile doesn't exist" vs "this
        // adapter doesn't exist on this profile".
        let entry = entry_with_per_profile_adapters();
        assert_eq!(
            validate_lora_for_profile(&entry, "nonexistent", "a1"),
            LoraValidation::UnknownProfile,
        );
    }

    #[test]
    fn test_lora_adapter_gate_allows_default_even_when_undeclared() {
        // ``"default"`` is always considered a valid profile name even
        // when the model didn't explicitly declare it — workers
        // synthesize a default profile at load time, matching the
        // behavior of ``generation_timeout_config``. With no adapters
        // declared anywhere, the gate falls through to
        // ``UnknownAdapter`` (request had a ``lora_adapter`` but the
        // model advertises none).
        use crate::types::model::{ModelEntry, ModelInfoExtras};
        let entry = ModelEntry {
            name: "acme/bare".to_string(),
            canonical_base_model: "acme/bare".to_string(),
            canonical_profile: "default".to_string(),
            pool: None,
            bundles: Vec::new(),
            adapter_modules: std::collections::HashSet::new(),
            profile_names: std::collections::HashSet::new(),
            profile_configs: std::collections::HashMap::new(),
            info_extras: ModelInfoExtras::default(),
        };
        assert_eq!(
            validate_lora_for_profile(&entry, "default", "anything"),
            LoraValidation::UnknownAdapter,
        );
    }

    // ---- `/v1/embeddings` response cap (#2617) --------------------------------
    //
    // These exist because the FIRST version of this cap shipped a test that
    // "certified the mechanism while never exercising it": it probed only `CAP`
    // and `CAP + 1`, and `CAP + 1` happened to be the single size at which the
    // over-cap branch could fire. Everything larger — i.e. every real oversized
    // reply — fell into the error arm and returned the opaque 500 the change
    // existed to remove, with the suite green throughout.
    //
    // So the rule for this block: never probe only the boundary. Every test here
    // covers `CAP + 2` and a realistic multiple, and the classification tests are
    // written so that reverting `buffer_compat_response_body` to the old
    // `to_bytes(body, CAP + 1)` shape makes them fail.

    /// A realistic `/v1/encode` dense reply, serialized the way the inner
    /// handler serializes it. `dims` floats per item, values in the range real
    /// normalized embeddings occupy so `serde_json`'s shortest-roundtrip
    /// formatting produces realistic byte counts rather than `0.0`.
    fn encode_reply_bytes(items: usize, dims: usize) -> Vec<u8> {
        let vector: Vec<f64> = (0..dims)
            .map(|i| {
                // f32-representable, sign-alternating, ~1e-2 magnitude — the
                // shape a normalized embedding coordinate actually has.
                let raw = ((i % 977) as f32) / 30011.0;
                let signed = if i % 2 == 0 { raw } else { -raw };
                signed as f64
            })
            .collect();
        let reply = json!({
            "items": (0..items)
                .map(|_| json!({ "dense": vector }))
                .collect::<Vec<_>>(),
        });
        serde_json::to_vec(&reply).expect("serialize encode reply")
    }

    #[tokio::test]
    async fn compat_response_cap_admits_a_realistic_large_encode_reply() {
        // 700 inputs at 1024 dims (bge-m3's dense width): a real bulk-embedding
        // batch, far inside `MAX_QUEUE_REQUEST_ITEMS` (4096), that must keep
        // being served. This is the "realistic large response still passes" case.
        let reply = encode_reply_bytes(700, 1024);
        assert!(
            reply.len() > 8 * 1024 * 1024,
            "fixture is meant to be genuinely large; got {} bytes",
            reply.len(),
        );
        assert!(
            reply.len() < MAX_COMPAT_RESPONSE_BODY,
            "a realistic batch must still fit; got {} bytes",
            reply.len(),
        );

        match buffer_compat_response_body(Body::from(reply.clone())).await {
            CompatResponseBody::Buffered(bytes) => assert_eq!(bytes.len(), reply.len()),
            other => panic!("a realistic large reply must pass, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn compat_response_cap_classifies_sizes_past_the_boundary() {
        // THE regression test for the vacuous-boundary defect. `CAP` and
        // `CAP + 1` alone were not enough: the broken implementation passed both
        // and mis-classified everything else. `CAP + 2` is the smallest size that
        // exposed it, and the two larger sizes are what production actually
        // produces.
        match buffer_compat_response_body(Body::from(vec![b'x'; MAX_COMPAT_RESPONSE_BODY])).await {
            CompatResponseBody::Buffered(bytes) => {
                assert_eq!(bytes.len(), MAX_COMPAT_RESPONSE_BODY)
            }
            other => panic!("a body exactly at the cap must pass, got {other:?}"),
        }
        // Classification first, across every size — so a regression reports
        // WHICH size it mis-classified rather than tripping on a detail of the
        // first one. Under the old implementation this names cap+2 directly.
        let mut misclassified = Vec::new();
        let mut unannounced = Vec::new();
        for over in [1usize, 2, 4096, MAX_COMPAT_RESPONSE_BODY] {
            let len = MAX_COMPAT_RESPONSE_BODY + over;
            match buffer_compat_response_body(Body::from(vec![b'x'; len])).await {
                CompatResponseBody::TooLarge { observed } => {
                    // The size hint is exact for an in-process body, so the
                    // caller-facing message can quote a real number.
                    if observed != Some(len as u64) {
                        unannounced.push(over);
                    }
                }
                other => misclassified.push(format!("cap+{over} -> {other:?}")),
            }
        }
        assert!(
            misclassified.is_empty(),
            "every oversized body must be TooLarge; these were not: {misclassified:?}",
        );
        assert!(
            unannounced.is_empty(),
            "an in-process body announces its length; these did not report it: {unannounced:?}",
        );
    }

    #[tokio::test]
    async fn compat_response_cap_refuses_a_realistic_oversized_encode_reply() {
        // The concrete production case the gate raised: a max-size batch at the
        // queue's own item ceiling. This is the shape that returned an opaque
        // 500 under the first implementation.
        let reply = encode_reply_bytes(1400, 1024);
        assert!(
            reply.len() > MAX_COMPAT_RESPONSE_BODY + 1,
            "fixture must be genuinely oversized, not boundary-sized; got {} bytes",
            reply.len(),
        );
        match buffer_compat_response_body(Body::from(reply)).await {
            CompatResponseBody::TooLarge { observed } => assert!(observed.is_some()),
            other => panic!("a realistic oversized reply must be TooLarge, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn compat_response_cap_refuses_an_oversized_body_that_announces_no_length() {
        // A body with no size hint cannot be pre-checked, so it must be caught by
        // the `LengthLimitError` downcast instead. Without that second stage this
        // case would be reported as an unreadable stream — a 500 for what is
        // really a size condition. `lower()` is 0 here, which is what makes the
        // downcast load-bearing rather than belt-and-braces.
        let chunk = vec![b'y'; 1024 * 1024];
        let chunks = (MAX_COMPAT_RESPONSE_BODY / chunk.len()) + 4;
        let stream = futures_util::stream::iter(
            (0..chunks).map(move |_| Ok::<_, std::io::Error>(vec![b'y'; 1024 * 1024])),
        );
        let body = Body::from_stream(stream);
        {
            use hyper::body::Body as _;
            assert_eq!(
                body.size_hint().lower(),
                0,
                "the streamed fixture must be un-pre-checkable for this test to mean anything",
            );
        }
        match buffer_compat_response_body(body).await {
            CompatResponseBody::TooLarge { observed } => {
                assert_eq!(observed, None, "an unannounced body has no length to quote")
            }
            other => panic!("an oversized unannounced body must be TooLarge, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_broken_stream_stays_unreadable_rather_than_too_large() {
        // The other half of the classification: the two failures must not be
        // collapsed back together. A stream that ERRORS below the cap is a
        // plumbing fault, not a size verdict, and must keep the 500.
        let stream = futures_util::stream::iter(vec![
            Ok::<_, std::io::Error>(vec![b'z'; 16]),
            Err(std::io::Error::other("upstream went away")),
        ]);
        match buffer_compat_response_body(Body::from_stream(stream)).await {
            CompatResponseBody::Unreadable => {}
            other => panic!("a broken stream must stay Unreadable, got {other:?}"),
        }
    }

    /// Every post-dispatch terminal arm must carry [`GatewayOwnedFault`].
    ///
    /// The worker has already run and the meter already holds its units by the
    /// time any of them fires, so an unmarked response is billed in full for zero
    /// delivered embeddings (#2641's defect class).
    ///
    /// This drives the two SHARED constructors the arms return through, which is
    /// what makes marking a property of the constructor rather than of each call
    /// site. It does NOT drive the handlers end to end — that needs a live queue
    /// — so the guarantee that every arm uses these constructors rather than
    /// building a bare response is held by
    /// `no_post_dispatch_arm_builds_an_unmarked_response` below, not by this.
    #[test]
    fn every_post_dispatch_fault_is_marked_gateway_owned() {
        let too_large = compat_translation_fault(
            "read_encode_response_too_large",
            StatusCode::PAYLOAD_TOO_LARGE,
            err_code::PAYLOAD_TOO_LARGE,
            Some("input"),
            "oversized".to_string(),
        );
        assert_eq!(too_large.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            too_large.extensions().get::<GatewayOwnedFault>(),
            Some(&GatewayOwnedFault::new("read_encode_response_too_large")),
            "an over-cap reply must not bill: the customer receives no embeddings",
        );
        // Terminal means no retry hint — a retryable surface here would have SDK
        // loops hammering a permanent condition (#2443).
        assert!(
            too_large.headers().get("retry-after").is_none(),
            "the over-cap verdict is terminal and must carry no Retry-After",
        );

        let unreadable = compat_translation_fault(
            "read_encode_response",
            StatusCode::INTERNAL_SERVER_ERROR,
            err_code::INTERNAL_ERROR,
            None,
            "failed".to_string(),
        );
        assert_eq!(
            unreadable.extensions().get::<GatewayOwnedFault>(),
            Some(&GatewayOwnedFault::new("read_encode_response")),
            "the pre-existing unreadable 500 billed in full too; it is marked now",
        );

        // The three TRANSLATION arms after the buffering pair. They are the same
        // class — worker delivered, we could not render it — and were billing in
        // full until they were routed through this constructor.
        for stage in [
            "decode_encode_response",
            "encode_response_shape",
            "format_embeddings",
        ] {
            let response = compat_translation_fault(
                stage,
                StatusCode::INTERNAL_SERVER_ERROR,
                err_code::INTERNAL_ERROR,
                None,
                "translation failed".to_string(),
            );
            assert_eq!(
                response.extensions().get::<GatewayOwnedFault>(),
                Some(&GatewayOwnedFault::new(stage)),
                "{stage}: zero embeddings delivered must mean nothing billed",
            );
        }

        // `/v1/rerank` runs the identical sequence against `score` and had NO
        // marked arm at all before this change.
        for stage in [
            "read_score_response",
            "decode_score_response",
            "format_rerank",
        ] {
            let response = rerank_translation_fault(stage, "translation failed");
            assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(
                response.extensions().get::<GatewayOwnedFault>(),
                Some(&GatewayOwnedFault::new(stage)),
                "{stage}: zero rerank results delivered must mean nothing billed",
            );
        }
    }

    /// No post-dispatch arm builds a bare response instead of going through a
    /// marked constructor.
    ///
    /// The constructors above are only load-bearing if every arm actually uses
    /// one. This reads the two handlers' own source between the point the worker
    /// reply is in hand and the success return, and fails on any early return
    /// that is not routed through its marking constructor — which is exactly how
    /// the three embeddings arms and all three rerank arms went unmarked while
    /// their siblings were fixed.
    #[test]
    fn no_post_dispatch_arm_builds_an_unmarked_response() {
        let source = include_str!("proxy.rs");
        for (marker, end, marked_constructor) in [
            (
                "// From here the worker has already run and reported its units, so EVERY",
                "let usage = embeddings_usage(&enc, token_count);",
                "return compat_translation_fault(",
            ),
            (
                "// From here the worker has already run and reported its units, so every",
                "let mut output = (StatusCode::OK, Json(output)).into_response();",
                "return rerank_translation_fault(",
            ),
        ] {
            let start = source
                .find(marker)
                .unwrap_or_else(|| panic!("post-dispatch region marker not found: {marker}"));
            let stop = source[start..]
                .find(end)
                .unwrap_or_else(|| panic!("post-dispatch region end not found: {end}"));
            let region = &source[start..start + stop];
            let unmarked: Vec<&str> = region
                .lines()
                .filter(|line| line.trim_start().starts_with("return "))
                .filter(|line| !line.contains(marked_constructor))
                .collect();
            assert!(
                unmarked.is_empty(),
                "post-dispatch arm builds a response outside a marking constructor, so it \
                 bills in full for an undelivered result: {unmarked:?}",
            );
        }
    }

    /// The over-cap message must name only remedies that actually work.
    ///
    /// `encoding_format="base64"` was named in the first revision and is inert:
    /// it is never forwarded to `/v1/encode`, and base64 is applied afterwards to
    /// the OUTER response assembled from the very buffer that was refused. A
    /// caller who followed it paid for a second identical encode and received a
    /// byte-identical rejection.
    #[test]
    fn the_over_cap_message_does_not_name_the_inert_base64_remedy() {
        let body = embeddings_error(
            err_code::PAYLOAD_TOO_LARGE,
            Some("input"),
            format!(
                "the embedding response for this request is {} bytes, over the \
                 {MAX_COMPAT_RESPONSE_BODY}-byte limit; send fewer inputs per request",
                MAX_COMPAT_RESPONSE_BODY + 1
            ),
        );
        let rendered = serde_json::to_string(&body).expect("serialize");
        assert!(
            !rendered.contains("base64"),
            "base64 cannot shrink the capped body; naming it sends the caller \
             into a paid loop: {rendered}",
        );
        assert!(rendered.contains("send fewer inputs per request"));
    }

    /// `encoding_format` provably cannot change the body the cap measures.
    ///
    /// This is the structural reason the remedy above was removed, pinned so a
    /// future change that re-adds the advice has to make it true first: the inner
    /// encode body is built from `input` alone.
    #[test]
    fn encoding_format_never_reaches_the_inner_encode_request() {
        let input = json!(["alpha", "beta"]);
        let (float_body, _) = openai_embeddings_encode_body(&input).expect("encode body");
        // There is no second builder that takes the format — the same call is
        // the only path for both `float` and `base64` callers, so the inner
        // reply, and therefore the capped buffer, is byte-identical.
        let (again, _) = openai_embeddings_encode_body(&input).expect("encode body");
        assert_eq!(float_body, again);
        let rendered = serde_json::to_string(&float_body).expect("serialize");
        assert!(
            !rendered.contains("encoding_format") && !rendered.contains("base64"),
            "the inner encode request carries no encoding hint: {rendered}",
        );
    }
}
