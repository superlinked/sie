//! Bootstrap the in-memory `ModelRegistry` from the `sie-config` control plane.
//!
//! At startup, after any optional filesystem seed in `SIE_BUNDLES_DIR` /
//! `SIE_MODELS_DIR` has been loaded, the gateway asks `sie-config` for the
//! authoritative state in two phases:
//!
//! 1. `GET /v1/configs/bundles` (+ per-bundle `GET /v1/configs/bundles/{id}`)
//!    — fetches the bundle/adapter surface and installs it via
//!    `ModelRegistry::install_bundles`. This MUST happen before the model
//!    fetch because `add_model_config` rejects any adapter not in a known
//!    bundle. Historically the gateway image baked these bundles in, which
//!    drifted from `sie-config`'s view and made every published model fail
//!    validation with `Adapter(s) not in any known bundle`. Pulling them
//!    over HTTP makes `sie-config` the single source of truth.
//! 2. `GET /v1/configs/export` (admin-auth) — fetches every persisted model
//!    config and replaces the registry's model set so the gateway's served
//!    view matches `sie-config`'s view, including removals.
//!
//! Live updates after bootstrap arrive via NATS deltas
//! (`sie.config.models.*`), which the `NatsManager` feeds into
//! `ModelRegistry::add_model_config` (append-only: a profile that already
//! exists with a different `CanonicalProfile` is rejected, so single-epoch
//! duplicate publishes are caught). The bootstrap and epoch-poller
//! catch-up paths here use `replace_model_configs_authoritative` instead,
//! which treats the export as the whole current model surface. This is
//! required so same-epoch filesystem-baseline drift cannot leave stale
//! models in the gateway after the new control-plane hash is stored.
//!
//! Failure handling:
//!
//! - The gateway does **not** block startup on a successful bootstrap. It
//!   serves traffic immediately using whatever the filesystem seed produced.
//! - A background task retries `GET /v1/configs/export` with exponential
//!   backoff (capped) until it succeeds. Every successful fetch is applied
//!   into the shared `ModelRegistry` and stored as the current `ConfigEpoch`.
//! - While bootstrap has not yet succeeded, API-added models from
//!   `sie-config` are missing. `GET /readyz` is process readiness only: once
//!   the gateway listener is serving it returns **200** + plain text `ok`, even
//!   with zero workers, so the first inference request can reach the gateway and
//!   trigger scale-from-zero via a surface-specific provisioning response.
//!   Bootstrap catch-up is visible separately via
//!   `GET /v1/configs/models/{id}/status` (`config_epoch` on that payload), the
//!   canonical `sie.gateway.config.applied_epoch` /
//!   `sie.gateway.config.bootstrap.degraded` telemetry, and gateway logs — not
//!   via `/readyz` flipping on export completion.
//! - `state::config_poller` runs in parallel, periodically reconciling
//!   against `GET /v1/configs/epoch` so any missed NATS deltas after the
//!   initial bootstrap are caught within one poll interval.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use serde::Deserialize;
use tracing::{info, warn};

// RFC 3986 `path` segment safe set: encode everything in `CONTROLS` plus the
// reserved characters that have meaning inside a URL. This is the same mask
// the `url` crate uses for its `PATH_SEGMENT` fragment.
const PATH_SEGMENT: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'`')
    .add(b'{')
    .add(b'}')
    .add(b'/')
    .add(b'%');

use crate::config::ModalProxyToken;
use crate::observability::metrics::{self as telemetry, ConfigOperation, ConfigOutcome};
use crate::state::bundle_config_hashes_hash::BundleConfigHashesHash;
use crate::state::bundles_hash::BundlesHash;
use crate::state::config_epoch::ConfigEpoch;
use crate::state::model_registry::ModelRegistry;
use crate::types::bundle::{BundleInfo, DEFAULT_ENGINE, KNOWN_ENGINES};
use crate::types::model::ModelConfig;

const BOOTSTRAP_TIMEOUT: Duration = Duration::from_secs(10);
const BACKOFF_INITIAL: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(60);

#[derive(Debug, Deserialize)]
struct ExportSnapshot {
    #[serde(default)]
    epoch: u64,
    #[serde(default)]
    bundle_config_hashes: HashMap<String, String>,
    #[serde(default)]
    bundle_pool_config_hashes: HashMap<String, HashMap<String, String>>,
    #[serde(default)]
    models: Vec<ExportedModel>,
}

/// `GET /v1/configs/epoch` carries independent drift signals:
///
/// - `epoch` — monotonic counter of model-config writes. Drives the
///   gateway's model-snapshot re-fetch path (existing behavior).
/// - `bundles_hash` — sha256 fingerprint over `sie-config`'s loaded bundle
///   set. Bundles are filesystem artifacts inside the `sie-config` image, so
///   their "version" is effectively redeploy-time and the model epoch
///   doesn't observe them. Without this signal a `sie-config` redeploy that
///   adds a bundle would not propagate to the gateway until either the
///   gateway pod restarts or a coincidental model write bumps the epoch.
/// - `bundle_config_hashes_hash` — sha256 fingerprint over the per-bundle
///   config hashes exported by `sie-config`. This catches filesystem-baseline
///   or no-store config drift where the model epoch stays unchanged but the
///   hashes workers advertise no longer match the stale gateway view.
///
/// Empty string is the documented "nothing to sync" sentinel and is what
/// `sie-config` returns when its registry is unavailable (degraded state).
#[derive(Debug, Deserialize)]
struct EpochResponse {
    #[serde(default)]
    epoch: u64,
    #[serde(default)]
    bundles_hash: String,
    #[serde(default)]
    bundle_config_hashes_hash: String,
}

/// What `fetch_epoch` returns to the caller. Public so the poller can pass
/// the hash through to `BundlesHash::store_if_changed` without having to
/// re-deserialize.
#[derive(Debug, Clone)]
pub struct EpochSnapshot {
    pub epoch: u64,
    pub bundles_hash: String,
    pub bundle_config_hashes_hash: String,
}

#[derive(Debug, Deserialize)]
struct ExportedModel {
    #[serde(default)]
    model_id: String,
    /// Raw YAML as stored by `sie-config`. Preferred source for replay because
    /// it preserves sie_id/name aliasing and any field ordering we might care
    /// about. We fall back to the structured `model_config` block if missing.
    #[serde(default)]
    raw_yaml: Option<String>,
    #[serde(default)]
    model_config: Option<serde_json::Value>,
}

/// Shape of `GET /v1/configs/bundles`. Only `bundle_id` is consumed — the
/// other fields (`priority`, `adapter_count`, `source`) are summary metadata
/// for human consumers; the authoritative view comes from the per-bundle
/// `GET /v1/configs/bundles/{id}` YAML which carries the full adapter list.
#[derive(Debug, Deserialize)]
struct BundleListResponse {
    #[serde(default)]
    bundles: Vec<BundleListEntry>,
}

#[derive(Debug, Deserialize)]
struct BundleListEntry {
    #[serde(default)]
    bundle_id: String,
}

/// Shape of `GET /v1/configs/bundles/{id}` (YAML). Mirrors `sie-config`'s
/// `BundleInfo` serialization: `name`, `priority`, `adapters[]`. We
/// intentionally ignore the `source` field — it's purely informational.
#[derive(Debug, Deserialize)]
struct BundleSpec {
    name: String,
    #[serde(default = "BundleSpec::default_priority")]
    priority: i32,
    #[serde(default)]
    adapters: Vec<String>,
    #[serde(default = "BundleSpec::default_engine")]
    engine: String,
}

impl BundleSpec {
    fn default_priority() -> i32 {
        100
    }

    fn default_engine() -> String {
        DEFAULT_ENGINE.to_string()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BootstrapError {
    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("unexpected status {status} from {url}: {body}")]
    BadStatus {
        status: u16,
        url: String,
        body: String,
    },
    /// The snapshot was fetched but one or more model entries failed to
    /// parse or apply. `ConfigEpoch` was **not** advanced, so the poller will
    /// retry on the next tick.
    #[error("partial apply: {applied}/{total} models applied, {failed} failed (epoch {epoch} not advanced)")]
    PartialApply {
        epoch: u64,
        applied: usize,
        failed: usize,
        total: usize,
    },
}

pub(crate) fn telemetry_outcome(error: &BootstrapError) -> ConfigOutcome {
    match error {
        BootstrapError::PartialApply { .. } => ConfigOutcome::PartialApply,
        BootstrapError::Http(_) | BootstrapError::BadStatus { .. } => ConfigOutcome::FetchError,
    }
}

pub struct BootstrapClient {
    base_url: String,
    admin_token: Option<String>,
    /// Optional Modal platform proxy-auth token (#1740). When set, every
    /// request also carries `Modal-Key` / `Modal-Secret` so it clears the
    /// Modal edge before app code sees it. Absent on self-host / dev.
    modal_proxy_token: Option<ModalProxyToken>,
    managed_proxy_auth: bool,
    http: reqwest::Client,
}

fn managed_proxy_auth_enabled() -> bool {
    std::env::var("SIE_MODAL_PROXY_AUTH")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

fn canonical_modal_config_origin(raw: &str) -> Option<String> {
    let parsed = reqwest::Url::parse(raw).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    if parsed.scheme() != "https"
        || !host.ends_with(".modal.run")
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.port_or_known_default() != Some(443)
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || parsed.path() != "/"
    {
        return None;
    }
    Some(format!("https://{host}"))
}

fn validate_managed_config_origin(
    base_url: &str,
    allowed_origin: Option<&str>,
) -> Result<(), String> {
    let allowed = allowed_origin
        .and_then(canonical_modal_config_origin)
        .ok_or_else(|| "managed config proxy-auth origin is missing or untrusted".to_string())?;
    let actual = canonical_modal_config_origin(base_url)
        .ok_or_else(|| "managed config-service URL is untrusted".to_string())?;
    if actual != allowed {
        return Err(
            "managed config-service URL does not match its provisioned proxy-auth origin"
                .to_string(),
        );
    }
    Ok(())
}

fn build_config_http_client(managed_proxy_auth: bool) -> Result<reqwest::Client, String> {
    let mut builder = reqwest::Client::builder().timeout(BOOTSTRAP_TIMEOUT);
    if managed_proxy_auth {
        // reqwest strips Authorization on cross-origin redirects but not
        // custom Modal headers. Config endpoints never require redirects.
        builder = builder.redirect(reqwest::redirect::Policy::none());
    }
    builder
        .build()
        .map_err(|error| format!("build config HTTP client: {error}"))
}

/// Outcome of a single `BootstrapClient::bootstrap` call.
///
/// `epoch` is whatever `sie-config` reported in the snapshot envelope.
/// `applied` counts configs the registry accepted into the authoritative
/// replacement snapshot. The shared meaning is "this export entry is
/// reconciled against the current registry state", which is what the
/// epoch-advance gate needs.
/// `failed` counts configs that could not be parsed or whose apply
/// returned a registry-level error. `total` is the size of the incoming
/// `models` list. `bootstrap_once` refuses to advance `ConfigEpoch` when
/// `failed > 0`, so a partially-successful apply leaves the poller
/// eligible to retry on the next tick instead of wedging silently at a
/// misleading epoch.
#[derive(Debug, Clone)]
pub struct BootstrapOutcome {
    pub epoch: u64,
    /// sha256 fingerprint of `sie-config`'s bundle set as observed on
    /// `GET /v1/configs/epoch` **before** this bootstrap's `fetch_bundles`
    /// call. Recorded so `state::config_poller` knows what hash we're
    /// "caught up to". The pre-fetch ordering is load-bearing: if a bundle
    /// is added on `sie-config` between our epoch read and our bundle list
    /// read we'll install a newer bundle set than this hash reflects, and
    /// the next poll tick sees `remote != stored` and re-fetches. The
    /// inverse — storing a hash that's newer than the bundles we actually
    /// installed — would silently wedge the gateway on a stale registry.
    /// May be empty if `sie-config` reported no hash (registry-degraded
    /// state); an empty stored hash will always mismatch any real remote
    /// value and trigger a re-fetch on the next tick, which is the
    /// intended fail-safe.
    pub bundles_hash: String,
    /// sha256 fingerprint of the per-bundle config-hash map as observed on
    /// `GET /v1/configs/epoch` before this bootstrap. See `bundles_hash` for
    /// why storing the pre-fetch value is the conservative ordering.
    pub bundle_config_hashes_hash: String,
    pub applied: usize,
    pub failed: usize,
    pub total: usize,
}

impl BootstrapOutcome {
    /// True iff every model in the snapshot was parsed and applied (or
    /// was a no-op replay). A zero-model snapshot is considered clean.
    pub fn is_complete(&self) -> bool {
        self.failed == 0
    }
}

impl BootstrapClient {
    pub fn new(base_url: String, admin_token: Option<String>) -> Result<Self, String> {
        let managed_proxy_auth = managed_proxy_auth_enabled();
        if managed_proxy_auth {
            validate_managed_config_origin(
                &base_url,
                std::env::var("SIE_CONFIG_PROXY_AUTH_ORIGIN")
                    .ok()
                    .as_deref(),
            )?;
        }
        let http = build_config_http_client(managed_proxy_auth)?;
        Ok(Self {
            base_url,
            admin_token,
            modal_proxy_token: None,
            managed_proxy_auth,
            http,
        })
    }

    /// Attach an optional Modal platform proxy-auth token (#1740). Kept as a
    /// builder so `new`'s two-arg signature (and its many call sites) stays
    /// stable; the config client sets this from `Config::config_modal_proxy_token`.
    pub fn with_modal_proxy_token(mut self, token: Option<ModalProxyToken>) -> Self {
        // A token present outside the explicit managed posture is ignored. This
        // keeps arbitrary OSS config URLs working without turning them into a
        // destination for the Modal credential.
        self.modal_proxy_token = self.managed_proxy_auth.then_some(token).flatten();
        self
    }

    /// Apply both auth layers to a request builder: the in-app admin bearer
    /// (`config_service_token`) and, when configured, the Modal platform
    /// proxy-auth headers (#1740). The two are independent — the bearer is
    /// checked by `sie_config`, the `Modal-Key` / `Modal-Secret` pair by the
    /// Modal edge — so both are sent when present.
    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut req = req;
        if let Some(token) = &self.admin_token {
            req = req.bearer_auth(token);
        }
        if let Some(proxy) = &self.modal_proxy_token {
            req = req
                .header("Modal-Key", &proxy.key)
                .header("Modal-Secret", &proxy.secret);
        }
        req
    }

    /// Fetch the current epoch + drift fingerprints from `GET /v1/configs/epoch`.
    /// Used by `state::config_poller` for bounded-staleness detection
    /// without pulling a full snapshot every tick.
    pub async fn fetch_epoch(&self) -> Result<EpochSnapshot, BootstrapError> {
        let url = format!("{}/v1/configs/epoch", self.base_url.trim_end_matches('/'));
        let req = self.apply_auth(self.http.get(&url));
        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(BootstrapError::BadStatus {
                status: status.as_u16(),
                url,
                body,
            });
        }
        let payload: EpochResponse = resp.json().await?;
        Ok(EpochSnapshot {
            epoch: payload.epoch,
            bundles_hash: payload.bundles_hash,
            bundle_config_hashes_hash: payload.bundle_config_hashes_hash,
        })
    }

    /// Fetch the bundle/adapter surface from `sie-config` via
    /// `GET /v1/configs/bundles` followed by per-bundle
    /// `GET /v1/configs/bundles/{id}`. Returns the parsed `BundleInfo` list
    /// ready for `ModelRegistry::install_bundles`.
    ///
    /// We do two HTTP calls per bundle (one list + one YAML each) instead of
    /// extending `/v1/configs/export` because the existing list/get endpoints
    /// already emit the right shape and bundle counts are tiny (typically
    /// 3–5). Failure of any call (list or any per-bundle GET) bubbles up as
    /// `BootstrapError`; partial bundle installs would leave the registry in
    /// a worse state than the previous attempt's snapshot, so we treat bundle
    /// fetches as all-or-nothing.
    pub async fn fetch_bundles(&self) -> Result<Vec<BundleInfo>, BootstrapError> {
        let base = self.base_url.trim_end_matches('/');
        let list_url = format!("{}/v1/configs/bundles", base);
        let req = self.apply_auth(self.http.get(&list_url));
        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(BootstrapError::BadStatus {
                status: status.as_u16(),
                url: list_url,
                body,
            });
        }
        let listing: BundleListResponse = resp.json().await?;

        let mut out = Vec::with_capacity(listing.bundles.len());
        for entry in listing.bundles {
            if entry.bundle_id.is_empty() {
                continue;
            }
            // `reqwest`'s `Client::get(&str)` does NOT percent-encode path
            // segments — it parses the full URL and leaves reserved
            // characters where they are. Today bundle IDs are filesystem
            // names (`default`, `sglang`, `florence2`) and would round-trip
            // fine, but a future namespaced ID could contain `/`, `:`, or
            // whitespace and silently reach a wrong endpoint. Encode the
            // segment explicitly so the invariant doesn't depend on the
            // naming convention.
            let encoded_id = utf8_percent_encode(&entry.bundle_id, PATH_SEGMENT).to_string();
            let yaml_url = format!("{}/v1/configs/bundles/{}", base, encoded_id);
            let req = self.apply_auth(self.http.get(&yaml_url));
            let resp = req.send().await?;
            let status = resp.status();
            if !status.is_success() {
                let body = resp.text().await.unwrap_or_default();
                return Err(BootstrapError::BadStatus {
                    status: status.as_u16(),
                    url: yaml_url,
                    body,
                });
            }
            let yaml = resp.text().await?;
            let spec: BundleSpec =
                serde_yaml::from_str(&yaml).map_err(|e| BootstrapError::BadStatus {
                    status: 200,
                    url: yaml_url.clone(),
                    body: format!("malformed bundle YAML: {}", e),
                })?;
            if !KNOWN_ENGINES.contains(&spec.engine.as_str()) {
                return Err(BootstrapError::BadStatus {
                    status: 200,
                    url: yaml_url,
                    body: format!(
                        "unknown bundle engine {:?}; allowed={:?}",
                        spec.engine, KNOWN_ENGINES
                    ),
                });
            }
            out.push(BundleInfo {
                name: spec.name,
                priority: spec.priority,
                adapters: spec.adapters,
                engine: spec.engine,
            });
        }

        info!(bundle_count = out.len(), "fetched bundles from sie-config");
        Ok(out)
    }

    /// Fetch bundles + the model snapshot, install both into `registry`.
    /// Bundles MUST be installed first so the model-apply loop can validate
    /// each profile's `adapter_path` against the freshly-fetched adapter set.
    pub async fn bootstrap(
        &self,
        registry: &ModelRegistry,
    ) -> Result<BootstrapOutcome, BootstrapError> {
        // Snapshot the bundles hash BEFORE fetching bundles. See the
        // `BootstrapOutcome::bundles_hash` doc comment for why the pre-fetch
        // ordering matters — storing a hash that's NEWER than the bundle
        // set we actually installed would let the poller believe it's
        // caught up on a stale registry.
        let pre_bootstrap = self.fetch_epoch().await?;
        let bundles = self.fetch_bundles().await?;
        registry.install_bundles(bundles);

        let url = format!("{}/v1/configs/export", self.base_url.trim_end_matches('/'));
        let req = self.apply_auth(self.http.get(&url));

        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(BootstrapError::BadStatus {
                status: status.as_u16(),
                url,
                body,
            });
        }

        let snapshot: ExportSnapshot = resp.json().await?;
        let total = snapshot.models.len();
        info!(
            epoch = snapshot.epoch,
            model_count = total,
            "received config export from sie-config"
        );

        let mut failed = 0usize;
        let mut configs = Vec::new();
        for model in snapshot.models {
            match parse_exported_model(&model) {
                Ok(Some(config)) => configs.push(config),
                Ok(None) => {
                    tracing::debug!(
                        model = %model.model_id,
                        "export entry has no config body; skipping",
                    );
                }
                Err(e) => {
                    warn!(
                        model = %model.model_id,
                        error = %e,
                        "failed to parse exported model config"
                    );
                    failed += 1;
                }
            }
        }

        let mut applied = 0usize;
        if failed == 0 {
            match registry.replace_model_configs_authoritative(configs) {
                Ok(count) => {
                    applied = count;
                    registry.install_bundle_config_hashes(snapshot.bundle_config_hashes);
                    registry.install_bundle_pool_config_hashes(snapshot.bundle_pool_config_hashes);
                }
                Err(e) => {
                    warn!(error = %e, "failed to install authoritative config export");
                    failed += 1;
                }
            }
        }

        Ok(BootstrapOutcome {
            epoch: snapshot.epoch,
            bundles_hash: pre_bootstrap.bundles_hash,
            bundle_config_hashes_hash: pre_bootstrap.bundle_config_hashes_hash,
            applied,
            failed,
            total,
        })
    }
}

fn parse_exported_model(model: &ExportedModel) -> Result<Option<ModelConfig>, String> {
    if let Some(yaml) = model.raw_yaml.as_deref() {
        let trimmed = yaml.trim();
        if !trimmed.is_empty() {
            return serde_yaml::from_str::<ModelConfig>(trimmed)
                .map(Some)
                .map_err(|e| e.to_string());
        }
    }

    if let Some(value) = &model.model_config {
        if value.is_null() {
            return Ok(None);
        }
        return serde_json::from_value::<ModelConfig>(value.clone())
            .map(Some)
            .map_err(|e| e.to_string());
    }

    Ok(None)
}

/// Run one bootstrap attempt. Pure helper used by both the background
/// retry loop and the epoch poller's catch-up path. Exposed for tests
/// that want to drive a single attempt against a mock `sie-config`.
///
/// Epoch-advance policy: `ConfigEpoch::set_max` is called **only when
/// every model in the snapshot was parsed and applied cleanly**
/// (`outcome.failed == 0`). A partial-success result returns
/// `BootstrapError::PartialApply` and leaves the epoch untouched, so the
/// poller keeps detecting drift on the next tick instead of silently
/// settling at a misleading epoch with an incomplete registry.
pub async fn bootstrap_once(
    client: &BootstrapClient,
    registry: &ModelRegistry,
    config_epoch: &ConfigEpoch,
    bundles_hash: &BundlesHash,
    bundle_config_hashes_hash: &BundleConfigHashesHash,
) -> Result<BootstrapOutcome, BootstrapError> {
    let outcome = client.bootstrap(registry).await?;
    if !outcome.is_complete() {
        warn!(
            epoch = outcome.epoch,
            applied = outcome.applied,
            failed = outcome.failed,
            total = outcome.total,
            "bootstrap applied only partially; epoch NOT advanced (poller will retry)"
        );
        return Err(BootstrapError::PartialApply {
            epoch: outcome.epoch,
            applied: outcome.applied,
            failed: outcome.failed,
            total: outcome.total,
        });
    }
    let advanced = config_epoch.set_max(outcome.epoch);
    // Record the bundle fingerprint AFTER the bootstrap is known complete.
    // Same "partial apply → don't advance" philosophy as the epoch: if the
    // bundle install or model apply failed we must keep the stored hash
    // stale so the poller re-enters the re-fetch branch on the next tick.
    let bundles_hash_changed = bundles_hash.store(outcome.bundles_hash.clone());
    let bundle_config_hashes_hash_changed =
        bundle_config_hashes_hash.store(outcome.bundle_config_hashes_hash.clone());
    info!(
        epoch = outcome.epoch,
        applied = outcome.applied,
        total = outcome.total,
        epoch_advanced = advanced,
        bundles_hash_changed,
        bundle_config_hashes_hash_changed,
        "bootstrap from sie-config complete",
    );
    Ok(outcome)
}

/// After this many failed attempts (or this much wall-clock elapsed,
/// whichever comes first) the retry loop raises its log level from
/// `warn!` to `error!` and flips the canonical
/// `sie.gateway.config.bootstrap.degraded` gauge to 1. This is the SRE-visible
/// signal that the gateway has been
/// serving filesystem-seed traffic for long enough that it's probably
/// diverged from the control plane. The loop keeps retrying — we never
/// give up — but a paging alert on `degraded == 1` is the right escape
/// hatch for a misconfigured `SIE_CONFIG_SERVICE_URL` or a prolonged
/// `sie-config` outage.
const DEGRADED_AFTER_ATTEMPTS: u32 = 10;
const DEGRADED_AFTER: std::time::Duration = std::time::Duration::from_secs(5 * 60);

/// Spawn a background task that retries `GET /v1/configs/export` with
/// exponential backoff until it succeeds. Returns immediately so `main` can
/// proceed to bind the HTTP listener — the gateway serves filesystem-seed
/// traffic while the task catches up.
///
/// Behavior:
///
/// - No-op (logs at info) when `base_url` is unset. Single-process tests and
///   demos don't need to stand up `sie-config`.
/// - On each failure, records the bounded outcome in
///   `sie.gateway.config.operations{operation="bootstrap", outcome=...}`,
///   logs a warning, and sleeps `backoff`, doubled up to `BACKOFF_MAX`.
/// - After `DEGRADED_AFTER_ATTEMPTS` attempts or `DEGRADED_AFTER` of
///   failed retries (whichever happens first), sets
///   `sie.gateway.config.bootstrap.degraded` to 1 and escalates logs to
///   `error!`. Emitted at most once per "cycle" — flapping is fine, a
///   sustained 1-minute+ burst of these is the paging signal.
/// - After the first success, the task clears the degraded gauge and
///   exits. Further staleness detection is handled by
///   `state::config_poller`.
pub fn spawn_bootstrap_retry(
    base_url: Option<&str>,
    admin_token: Option<&str>,
    modal_proxy_token: Option<&ModalProxyToken>,
    registry: Arc<ModelRegistry>,
    config_epoch: ConfigEpoch,
    bundles_hash: BundlesHash,
    bundle_config_hashes_hash: BundleConfigHashesHash,
) -> tokio::task::JoinHandle<()> {
    let base_url = base_url.map(str::to_string);
    let admin_token = admin_token.map(str::to_string);
    let modal_proxy_token = modal_proxy_token.cloned();
    tokio::spawn(async move {
        telemetry::set_config_bootstrap_degraded(false);
        let Some(base) = base_url else {
            info!("SIE_CONFIG_SERVICE_URL not set; skipping config bootstrap");
            return;
        };
        let client = match BootstrapClient::new(base, admin_token)
            .map(|c| c.with_modal_proxy_token(modal_proxy_token))
        {
            Ok(c) => c,
            Err(e) => {
                telemetry::record_config_operation(
                    ConfigOperation::Bootstrap,
                    ConfigOutcome::ClientError,
                );
                telemetry::set_config_bootstrap_degraded(true);
                warn!(error = %e, "failed to build bootstrap HTTP client; giving up");
                return;
            }
        };
        let mut backoff = BACKOFF_INITIAL;
        let mut attempts: u32 = 0;
        let started = std::time::Instant::now();
        let mut degraded = false;
        loop {
            match bootstrap_once(
                &client,
                registry.as_ref(),
                &config_epoch,
                &bundles_hash,
                &bundle_config_hashes_hash,
            )
            .await
            {
                Ok(_) => {
                    telemetry::record_config_operation(
                        ConfigOperation::Bootstrap,
                        ConfigOutcome::Success,
                    );
                    telemetry::set_config_bootstrap_degraded(false);
                    if degraded {
                        info!(
                            attempts,
                            elapsed_secs = started.elapsed().as_secs(),
                            "bootstrap recovered after sustained failure; clearing degraded state"
                        );
                    }
                    return;
                }
                Err(e) => {
                    let outcome = telemetry_outcome(&e);
                    telemetry::record_config_operation(ConfigOperation::Bootstrap, outcome);
                    attempts = attempts.saturating_add(1);
                    let elapsed = started.elapsed();
                    let should_degrade = !degraded
                        && (attempts >= DEGRADED_AFTER_ATTEMPTS || elapsed >= DEGRADED_AFTER);
                    if should_degrade {
                        degraded = true;
                        telemetry::set_config_bootstrap_degraded(true);
                        tracing::error!(
                            error = %e,
                            attempts,
                            elapsed_secs = elapsed.as_secs(),
                            retry_in_secs = backoff.as_secs(),
                            "bootstrap from sie-config DEGRADED: sustained failures, gateway is serving filesystem-seed only"
                        );
                    } else {
                        warn!(
                            error = %e,
                            attempts,
                            retry_in_secs = backoff.as_secs(),
                            "bootstrap from sie-config failed; serving filesystem seed only and retrying"
                        );
                    }
                    tokio::time::sleep(backoff).await;
                    backoff = std::cmp::min(backoff * 2, BACKOFF_MAX);
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use tempfile::TempDir;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn bootstrap_errors_map_to_bounded_telemetry_outcomes() {
        assert_eq!(
            telemetry_outcome(&BootstrapError::PartialApply {
                epoch: 1,
                applied: 1,
                failed: 1,
                total: 2,
            }),
            ConfigOutcome::PartialApply
        );
        assert_eq!(
            telemetry_outcome(&BootstrapError::BadStatus {
                status: 503,
                url: "http://config.invalid/export".to_string(),
                body: "unavailable".to_string(),
            }),
            ConfigOutcome::FetchError
        );
    }

    /// Build a registry against empty filesystem dirs and pre-install the
    /// `default` bundle directly via `install_bundles` — the same path
    /// `BootstrapClient::bootstrap` would take in production after fetching
    /// `/v1/configs/bundles`. Returning the `TempDir` keeps the empty seed
    /// dirs alive for the registry's lifetime, which keeps the path-existence
    /// branches in `ModelRegistry::reload` exercising the "dir exists, no
    /// files" code path rather than the "dir missing" one. Both branches are
    /// tolerated, but the existing tests were written against the former.
    fn make_registry() -> (Arc<ModelRegistry>, TempDir) {
        let temp = TempDir::new().unwrap();
        let bundles = temp.path().join("bundles");
        let models = temp.path().join("models");
        std::fs::create_dir_all(&bundles).unwrap();
        std::fs::create_dir_all(&models).unwrap();
        let registry = Arc::new(ModelRegistry::new(&bundles, &models, true));
        registry.install_bundles(vec![BundleInfo {
            name: "default".to_string(),
            priority: 10,
            adapters: vec!["sie_server.adapters.sentence_transformer".to_string()],
            engine: DEFAULT_ENGINE.to_string(),
        }]);
        (registry, temp)
    }

    fn model_config(name: &str) -> ModelConfig {
        ModelConfig {
            name: name.to_string(),
            hf_revision: None,
            adapter_module: None,
            default_bundle: None,
            pool: None,
            profiles: HashMap::from([(
                "default".to_string(),
                crate::types::model::ProfileConfig {
                    adapter_path: Some(
                        "sie_server.adapters.sentence_transformer:Adapter".to_string(),
                    ),
                    max_batch_tokens: None,
                    compute_precision: None,
                    adapter_options: None,
                    extends: None,
                },
            )]),
            inputs: None,
            max_sequence_length: None,
            tasks: None,
        }
    }

    /// Mount minimal `/v1/configs/bundles*` endpoints on `server` returning a
    /// single `default` bundle that covers
    /// `sie_server.adapters.sentence_transformer`. Tests that exercise the
    /// existing `bootstrap()` flow rely on the bundle fetch succeeding before
    /// the model-apply loop runs; this helper keeps each test's body focused
    /// on its model-shape assertions.
    async fn mount_default_bundles(server: &MockServer) {
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "bundles": [{
                    "bundle_id": "default",
                    "priority": 10,
                    "adapter_count": 1,
                    "source": "filesystem",
                }],
            })))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles/default"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
            ))
            .mount(server)
            .await;
    }

    /// Mount a canned `/v1/configs/epoch` response for tests that go
    /// through `BootstrapClient::bootstrap` — that method calls
    /// `fetch_epoch` before fetching bundles so it can record the current
    /// bundle hash into `BootstrapOutcome`. Tests that don't set up this
    /// mock would otherwise get a 404 and their bootstrap call would fail
    /// with a `BadStatus` error long before the model-apply path runs.
    async fn mount_default_epoch(server: &MockServer, epoch: u64, bundles_hash: &str) {
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": epoch,
                "bundles_hash": bundles_hash,
                "bundle_config_hashes_hash": "bundle_config_hashes_hash",
            })))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn bootstrap_applies_models_from_snapshot() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        let body = serde_json::json!({
            "snapshot_version": 1,
            "epoch": 42,
            "generated_at": "2026-04-17T00:00:00Z",
            "bundle_config_hashes": {"default": "control-plane-hash"},
            "bundle_pool_config_hashes": {"default": {"default": "control-plane-pool-hash"}},
            "models": [
                {
                    "model_id": "test/model",
                    "raw_yaml": "sie_id: test/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.sentence_transformer:Adapter\n",
                    "affected_bundles": ["default"],
                }
            ],
        });

        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .and(header("authorization", "Bearer admin-secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), Some("admin-secret".into())).unwrap();
        let outcome = client.bootstrap(registry.as_ref()).await.unwrap();

        assert_eq!(outcome.epoch, 42);
        assert_eq!(outcome.applied, 1);
        assert_eq!(outcome.failed, 0);
        assert_eq!(outcome.total, 1);
        assert!(outcome.is_complete());
        assert_eq!(
            registry.compute_bundle_config_hash("default"),
            "control-plane-hash"
        );
        assert_eq!(
            registry.compute_bundle_config_hash_for_pool("default", "default"),
            "control-plane-pool-hash"
        );
        assert_eq!(
            registry.get_model_profile_names("test/model"),
            vec!["default".to_string()],
        );
    }

    #[tokio::test]
    async fn bootstrap_replaces_authoritative_model_set_and_drops_stale_models() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        registry
            .add_model_config(model_config("stale/model"))
            .expect("seed stale gateway-local model");
        assert!(registry.get_model_info("stale/model").is_some());

        let body = serde_json::json!({
            "snapshot_version": 1,
            "epoch": 43,
            "generated_at": "2026-04-17T00:00:00Z",
            "bundle_config_hashes": {"default": "control-plane-hash"},
            "models": [
                {
                    "model_id": "kept/model",
                    "raw_yaml": "sie_id: kept/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.sentence_transformer:Adapter\n",
                    "affected_bundles": ["default"],
                }
            ],
        });

        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let epoch = ConfigEpoch::new();
        let bundles_hash = BundlesHash::new();
        let bundle_config_hashes_hash = BundleConfigHashesHash::new();
        bootstrap_once(
            &client,
            registry.as_ref(),
            &epoch,
            &bundles_hash,
            &bundle_config_hashes_hash,
        )
        .await
        .unwrap();

        assert!(registry.get_model_info("kept/model").is_some());
        assert!(
            registry.get_model_info("stale/model").is_none(),
            "authoritative export replacement must remove models omitted from sie-config"
        );
    }

    #[tokio::test]
    async fn bootstrap_soft_fails_on_5xx() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(503).set_body_string("maintenance"))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let err = client.bootstrap(registry.as_ref()).await.unwrap_err();
        match err {
            BootstrapError::BadStatus { status, .. } => assert_eq!(status, 503),
            other => panic!("expected BadStatus, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn spawn_bootstrap_retry_noops_without_url() {
        // No URL configured → task exits immediately, never touching the
        // registry. We `await` the join handle to confirm it terminated.
        let (registry, _tmp) = make_registry();
        let handle = spawn_bootstrap_retry(
            None,
            None,
            None,
            registry,
            ConfigEpoch::new(),
            BundlesHash::new(),
            BundleConfigHashesHash::new(),
        );
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("retry task should exit immediately when url is None")
            .unwrap();
    }

    #[tokio::test]
    async fn bootstrap_once_updates_config_epoch() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        let body = serde_json::json!({
            "snapshot_version": 1,
            "epoch": 77,
            "generated_at": "2026-04-17T00:00:00Z",
            "models": [],
        });
        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let epoch = ConfigEpoch::new();
        let bundles_hash = BundlesHash::new();
        let bundle_config_hashes_hash = BundleConfigHashesHash::new();
        let client = BootstrapClient::new(server.uri(), None).unwrap();
        bootstrap_once(
            &client,
            registry.as_ref(),
            &epoch,
            &bundles_hash,
            &bundle_config_hashes_hash,
        )
        .await
        .unwrap();
        assert_eq!(epoch.get(), 77);
        // BootstrapOutcome carries the pre-bootstrap bundles_hash; confirm
        // the bridge into BundlesHash actually stored it.
        assert_eq!(bundles_hash.get(), "deadbeef");
        assert_eq!(bundle_config_hashes_hash.get(), "bundle_config_hashes_hash");

        // Re-run against a stale snapshot: epoch must not move backward.
        let stale_body = serde_json::json!({
            "snapshot_version": 1,
            "epoch": 5,
            "generated_at": "2026-04-17T00:00:00Z",
            "models": [],
        });
        let stale_server = MockServer::start().await;
        mount_default_bundles(&stale_server).await;
        mount_default_epoch(&stale_server, 0, "deadbeef").await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&stale_body))
            .mount(&stale_server)
            .await;
        let stale_client = BootstrapClient::new(stale_server.uri(), None).unwrap();
        bootstrap_once(
            &stale_client,
            registry.as_ref(),
            &epoch,
            &bundles_hash,
            &bundle_config_hashes_hash,
        )
        .await
        .unwrap();
        assert_eq!(epoch.get(), 77);
    }

    #[tokio::test]
    async fn fetch_epoch_returns_parsed_value() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": 42,
                "bundles_hash": "cafebabe",
                "bundle_config_hashes_hash": "facefeed",
            })))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let snapshot = client.fetch_epoch().await.unwrap();
        assert_eq!(snapshot.epoch, 42);
        assert_eq!(snapshot.bundles_hash, "cafebabe");
        assert_eq!(snapshot.bundle_config_hashes_hash, "facefeed");
    }

    #[tokio::test]
    async fn config_client_drops_modal_token_outside_managed_posture() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .and(header("authorization", "Bearer admin-secret"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": 5,
            })))
            .mount(&server)
            .await;

        let client = BootstrapClient {
            base_url: server.uri(),
            admin_token: Some("admin-secret".into()),
            modal_proxy_token: None,
            managed_proxy_auth: false,
            http: reqwest::Client::new(),
        }
        .with_modal_proxy_token(Some(ModalProxyToken {
            key: "wk-abc".into(),
            secret: "ws-xyz".into(),
        }));
        let snapshot = client.fetch_epoch().await.unwrap();
        assert_eq!(snapshot.epoch, 5);

        let requests = server.received_requests().await.unwrap();
        let request = requests.first().unwrap();
        assert!(!request.headers.contains_key("modal-key"));
        assert!(!request.headers.contains_key("modal-secret"));
    }

    #[tokio::test]
    async fn managed_config_client_sends_both_auth_layers_without_following_redirects() {
        let destination = MockServer::start().await;
        let origin = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .and(header("authorization", "Bearer admin-secret"))
            .and(header("modal-key", "wk-abc"))
            .and(header("modal-secret", "ws-xyz"))
            .respond_with(
                ResponseTemplate::new(302)
                    .append_header("location", format!("{}/capture", destination.uri())),
            )
            .mount(&origin)
            .await;

        let client = BootstrapClient {
            base_url: origin.uri(),
            admin_token: Some("admin-secret".into()),
            modal_proxy_token: Some(ModalProxyToken {
                key: "wk-abc".into(),
                secret: "ws-xyz".into(),
            }),
            managed_proxy_auth: true,
            http: build_config_http_client(true).unwrap(),
        };
        let error = client
            .fetch_epoch()
            .await
            .expect_err("302 must not be followed");
        assert!(matches!(
            error,
            BootstrapError::BadStatus { status: 302, .. }
        ));
        assert_eq!(origin.received_requests().await.unwrap().len(), 1);
        assert!(destination.received_requests().await.unwrap().is_empty());
    }

    #[test]
    fn managed_config_origin_requires_exact_canonical_modal_origin() {
        assert!(validate_managed_config_origin(
            "https://workspace--config.modal.run",
            Some("https://workspace--config.modal.run"),
        )
        .is_ok());
        for base in [
            "https://attacker.example",
            "https://sibling--config.modal.run",
            "https://workspace--config.modal.run.evil.example",
            "https://user@workspace--config.modal.run",
            "http://workspace--config.modal.run",
            "https://workspace--config.modal.run:8443",
            "https://workspace--config.modal.run/path",
            "not a URL",
        ] {
            assert!(validate_managed_config_origin(
                base,
                Some("https://workspace--config.modal.run"),
            )
            .is_err());
        }
    }

    /// Without a proxy token, the client sends NO Modal headers — the
    /// self-host / dev path must not leak an empty or partial credential. The
    /// in-app bearer is unaffected (it rides independently).
    #[tokio::test]
    async fn config_client_omits_modal_headers_when_token_absent() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": 9,
            })))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), Some("admin-secret".into())).unwrap();
        let snapshot = client.fetch_epoch().await.unwrap();
        assert_eq!(snapshot.epoch, 9);

        let requests = server
            .received_requests()
            .await
            .expect("wiremock records requests by default");
        let req = requests.first().expect("exactly one epoch request");
        assert!(
            !req.headers.contains_key("modal-key"),
            "no proxy token -> no Modal-Key header"
        );
        assert!(!req.headers.contains_key("modal-secret"));
        assert!(
            req.headers.contains_key("authorization"),
            "the in-app bearer still rides every request"
        );
    }

    #[tokio::test]
    async fn fetch_epoch_tolerates_missing_bundles_hash_field() {
        // Older sie-config returns only
        // `{"epoch": N}`. The field carries `#[serde(default)]` so we
        // deserialize it as the empty string — the documented
        // "nothing to sync" sentinel that makes the poller skip the
        // hash-mismatch branch instead of thrashing.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/epoch"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"epoch": 7})))
            .mount(&server)
            .await;
        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let snapshot = client.fetch_epoch().await.unwrap();
        assert_eq!(snapshot.epoch, 7);
        assert_eq!(snapshot.bundles_hash, "");
        assert_eq!(snapshot.bundle_config_hashes_hash, "");
    }

    #[tokio::test]
    async fn bootstrap_handles_empty_snapshot() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        let body = serde_json::json!({
            "snapshot_version": 1,
            "epoch": 0,
            "generated_at": "2026-04-17T00:00:00Z",
            "models": [],
        });

        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let outcome = client.bootstrap(registry.as_ref()).await.unwrap();
        assert_eq!(outcome.epoch, 0);
        assert_eq!(outcome.applied, 0);
        assert_eq!(outcome.failed, 0);
        assert!(outcome.is_complete());
    }

    #[tokio::test]
    async fn bootstrap_surfaces_401_with_body() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(
                ResponseTemplate::new(401).set_body_string("{\"error\":\"invalid_admin_token\"}"),
            )
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), Some("wrong".into())).unwrap();
        let err = client.bootstrap(registry.as_ref()).await.unwrap_err();
        match err {
            BootstrapError::BadStatus { status, body, .. } => {
                assert_eq!(status, 401);
                assert!(body.contains("invalid_admin_token"));
            }
            other => panic!("expected BadStatus, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn bootstrap_rejects_malformed_json_body() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not-json-at-all"))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let err = client.bootstrap(registry.as_ref()).await.unwrap_err();
        match err {
            BootstrapError::Http(e) => {
                assert!(e.is_decode(), "expected decode error, got {:?}", e);
            }
            other => panic!("expected Http decode error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn bootstrap_accepts_model_config_json_fallback() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        let body = serde_json::json!({
            "snapshot_version": 1,
            "epoch": 7,
            "generated_at": "2026-04-17T00:00:00Z",
            "models": [
                {
                    "model_id": "json/model",
                    "raw_yaml": null,
                    "model_config": {
                        "sie_id": "json/model",
                        "pool": "customer-a",
                        "profiles": {
                            "default": {
                                "adapter_path": "sie_server.adapters.sentence_transformer:Adapter",
                                "max_batch_tokens": 2048,
                            }
                        }
                    },
                    "affected_bundles": ["default"],
                }
            ],
        });

        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let outcome = client.bootstrap(registry.as_ref()).await.unwrap();
        assert_eq!(outcome.epoch, 7);
        assert_eq!(outcome.applied, 1);
        assert_eq!(outcome.failed, 0);
        assert_eq!(
            registry.get_model_profile_names("json/model"),
            vec!["default".to_string()],
        );
    }

    #[test]
    fn parse_exported_model_preserves_pool_from_json_fallback() {
        let model = ExportedModel {
            model_id: "json/model".to_string(),
            raw_yaml: None,
            model_config: Some(serde_json::json!({
                "sie_id": "json/model",
                "pool": "customer-a",
                "profiles": {
                    "default": {
                        "adapter_path": "sie_server.adapters.sentence_transformer:Adapter",
                    }
                }
            })),
        };

        let config = parse_exported_model(&model).unwrap().unwrap();
        assert_eq!(config.pool, Some("customer-a".to_string()));
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert!(yaml.contains("pool: customer-a"));
    }

    #[tokio::test]
    async fn bootstrap_skips_entry_without_config_body() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        let body = serde_json::json!({
            "snapshot_version": 1,
            "epoch": 1,
            "generated_at": "2026-04-17T00:00:00Z",
            "models": [
                {
                    "model_id": "placeholder/model",
                    "raw_yaml": "   \n",
                    "model_config": null,
                    "affected_bundles": [],
                }
            ],
        });

        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let outcome = client.bootstrap(registry.as_ref()).await.unwrap();
        assert_eq!(outcome.epoch, 1);
        assert_eq!(outcome.applied, 0);
        assert_eq!(outcome.failed, 0);
        assert!(registry
            .get_model_profile_names("placeholder/model")
            .is_empty());
    }

    #[tokio::test]
    async fn bootstrap_refuses_partial_model_install_on_malformed_entry() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        let body = serde_json::json!({
            "snapshot_version": 1,
            "epoch": 9,
            "generated_at": "2026-04-17T00:00:00Z",
            "models": [
                {
                    "model_id": "broken/model",
                    "raw_yaml": "::: not yaml :::",
                    "affected_bundles": ["default"],
                },
                {
                    "model_id": "good/model",
                    "raw_yaml": "sie_id: good/model\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.sentence_transformer:Adapter\n",
                    "affected_bundles": ["default"],
                }
            ],
        });

        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let outcome = client.bootstrap(registry.as_ref()).await.unwrap();
        assert_eq!(outcome.applied, 0);
        assert_eq!(outcome.failed, 1);
        assert_eq!(outcome.total, 2);
        assert!(!outcome.is_complete());
        assert!(registry.get_model_profile_names("broken/model").is_empty());
        assert!(
            registry.get_model_profile_names("good/model").is_empty(),
            "authoritative export replacement must not partially install the valid rows when one row is malformed"
        );
    }

    #[tokio::test]
    async fn bootstrap_once_refuses_to_advance_epoch_on_partial_apply() {
        // If any model in the snapshot fails to parse or apply,
        // `bootstrap_once` must return `PartialApply` and leave
        // `ConfigEpoch` untouched so the poller keeps retrying. Advancing
        // the epoch here would wedge the replica silently with an empty
        // registry that believes it's caught up.
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_bundles(&server).await;
        mount_default_epoch(&server, 0, "deadbeef").await;

        let body = serde_json::json!({
            "snapshot_version": 1,
            "epoch": 99,
            "generated_at": "2026-04-17T00:00:00Z",
            "models": [
                {
                    "model_id": "broken/model",
                    "raw_yaml": "::: not yaml :::",
                    "affected_bundles": ["default"],
                }
            ],
        });
        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let epoch = ConfigEpoch::new();
        let bundles_hash = BundlesHash::new();
        let bundle_config_hashes_hash = BundleConfigHashesHash::new();
        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let err = bootstrap_once(
            &client,
            registry.as_ref(),
            &epoch,
            &bundles_hash,
            &bundle_config_hashes_hash,
        )
        .await
        .unwrap_err();
        match err {
            BootstrapError::PartialApply {
                epoch: snap_epoch,
                applied,
                failed,
                total,
            } => {
                assert_eq!(snap_epoch, 99);
                assert_eq!(applied, 0);
                assert_eq!(failed, 1);
                assert_eq!(total, 1);
            }
            other => panic!("expected PartialApply, got {:?}", other),
        }
        // Epoch must still be 0 — we haven't caught up yet.
        assert_eq!(epoch.get(), 0);
        // Same philosophy for the bundles_hash: a partial apply leaves
        // it stale so the poller re-enters the re-fetch branch on the
        // next tick. Storing the fresh hash here would let the poller
        // believe the registry is caught up despite the failure.
        assert_eq!(bundles_hash.get(), "");
        assert_eq!(bundle_config_hashes_hash.get(), "");
    }

    /// Two-phase fetch: list endpoint enumerates IDs, per-bundle endpoint
    /// returns the YAML body. This is the happy-path covering the wire
    /// contract; a regression here means the gateway will refuse every model
    /// with `Adapter(s) not in any known bundle`, which is exactly the
    /// failure mode this whole change is removing.
    #[tokio::test]
    async fn fetch_bundles_parses_two_bundle_listing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "bundles": [
                    {"bundle_id": "default", "priority": 10, "adapter_count": 1, "source": "filesystem"},
                    {"bundle_id": "transformers5", "priority": 20, "adapter_count": 2, "source": "filesystem"},
                ],
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles/default"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
            ))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles/transformers5"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "name: transformers5\npriority: 20\nadapters:\n  - sie_server.adapters.t5\n  - sie_server.adapters.qwen3_vl\n",
            ))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let bundles = client.fetch_bundles().await.unwrap();
        assert_eq!(bundles.len(), 2);
        let by_name: std::collections::HashMap<_, _> =
            bundles.iter().map(|b| (b.name.as_str(), b)).collect();
        assert_eq!(by_name["default"].priority, 10);
        assert_eq!(by_name["default"].adapters.len(), 1);
        assert_eq!(by_name["transformers5"].adapters.len(), 2);
    }

    /// `fetch_bundles` is all-or-nothing: a single 404 on a per-bundle GET
    /// fails the entire fetch. The alternative — partial install — would
    /// silently drop adapters and recreate the original drift bug under a
    /// different cause (missing instead of stale).
    #[tokio::test]
    async fn fetch_bundles_fails_when_per_bundle_get_404s() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "bundles": [{"bundle_id": "ghost", "priority": 10, "adapter_count": 0, "source": "filesystem"}],
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles/ghost"))
            .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let err = client.fetch_bundles().await.unwrap_err();
        match err {
            BootstrapError::BadStatus { status, url, .. } => {
                assert_eq!(status, 404);
                assert!(url.ends_with("/v1/configs/bundles/ghost"));
            }
            other => panic!("expected BadStatus, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn fetch_bundles_rejects_unknown_engine() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "bundles": [{"bundle_id": "bad-engine", "priority": 10, "adapter_count": 1, "source": "filesystem"}],
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles/bad-engine"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "name: bad-engine\npriority: 10\nengine: future-engine\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
            ))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let err = client.fetch_bundles().await.unwrap_err();
        match err {
            BootstrapError::BadStatus { status, body, .. } => {
                assert_eq!(status, 200);
                assert!(body.contains("unknown bundle engine"));
                assert!(body.contains("future-engine"));
            }
            other => panic!("expected BadStatus, got {:?}", other),
        }
    }

    /// Bundles fetched from `sie-config` must materialize into the registry
    /// before the model-apply loop runs. We start from a registry with the
    /// `default` bundle already installed (mimicking a previous successful
    /// bootstrap) and then run a fresh `bootstrap` against a new bundle set
    /// that adds `transformers5` and a model that depends on it. If
    /// `install_bundles` weren't called as part of `bootstrap`, the model
    /// would be rejected with `Adapter(s) not in any known bundle`.
    #[tokio::test]
    async fn bootstrap_installs_bundles_before_applying_models() {
        let server = MockServer::start().await;
        let (registry, _tmp) = make_registry();
        mount_default_epoch(&server, 12, "hash_after_t5_added").await;

        // Bundle list now includes a new t5 bundle.
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "bundles": [
                    {"bundle_id": "default", "priority": 10, "adapter_count": 1, "source": "filesystem"},
                    {"bundle_id": "transformers5", "priority": 20, "adapter_count": 1, "source": "filesystem"},
                ],
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles/default"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "name: default\npriority: 10\nadapters:\n  - sie_server.adapters.sentence_transformer\n",
            ))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/configs/bundles/transformers5"))
            .respond_with(ResponseTemplate::new(200).set_body_string(
                "name: transformers5\npriority: 20\nadapters:\n  - sie_server.adapters.t5\n",
            ))
            .mount(&server)
            .await;

        let body = serde_json::json!({
            "snapshot_version": 1,
            "epoch": 12,
            "generated_at": "2026-04-22T00:00:00Z",
            "models": [{
                "model_id": "google/t5-small",
                "raw_yaml": "sie_id: google/t5-small\nprofiles:\n  default:\n    adapter_path: sie_server.adapters.t5:Adapter\n",
                "affected_bundles": ["transformers5"],
            }],
        });
        Mock::given(method("GET"))
            .and(path("/v1/configs/export"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&body))
            .mount(&server)
            .await;

        let client = BootstrapClient::new(server.uri(), None).unwrap();
        let outcome = client.bootstrap(registry.as_ref()).await.unwrap();
        assert_eq!(outcome.applied, 1);
        assert_eq!(outcome.failed, 0);
        // BootstrapOutcome carries the hash we saw BEFORE fetching bundles.
        // Ordering matters: see the struct doc on `bundles_hash`.
        assert_eq!(outcome.bundles_hash, "hash_after_t5_added");
        assert_eq!(
            registry.get_model_bundles("google/t5-small"),
            vec!["transformers5".to_string()],
        );
        // The previous bundle is gone now (install_bundles is replace, not
        // merge): the registry view exactly matches sie-config's view.
        let mut listed = registry.list_bundles();
        listed.sort();
        assert_eq!(
            listed,
            vec!["default".to_string(), "transformers5".to_string()]
        );
    }

    /// `install_bundles` rebuilds the per-model `bundles` association list,
    /// so a model that was previously routable can become unroutable when
    /// `sie-config` removes the bundle that backed it. This is the desired
    /// behavior — the gateway must not keep serving against a stale routing
    /// table after the control plane has retired a bundle.
    #[test]
    fn install_bundles_replaces_existing_bundle_set() {
        let temp = TempDir::new().unwrap();
        let bundles = temp.path().join("bundles");
        let models = temp.path().join("models");
        std::fs::create_dir_all(&bundles).unwrap();
        std::fs::create_dir_all(&models).unwrap();

        let registry = ModelRegistry::new(&bundles, &models, true);
        registry.install_bundles(vec![
            BundleInfo {
                name: "a".to_string(),
                priority: 10,
                adapters: vec!["mod.a".to_string()],
                engine: DEFAULT_ENGINE.to_string(),
            },
            BundleInfo {
                name: "b".to_string(),
                priority: 20,
                adapters: vec!["mod.b".to_string()],
                engine: DEFAULT_ENGINE.to_string(),
            },
        ]);
        let mut listed = registry.list_bundles();
        listed.sort();
        assert_eq!(listed, vec!["a".to_string(), "b".to_string()]);

        // Replace with a strict subset; the old bundle must vanish.
        registry.install_bundles(vec![BundleInfo {
            name: "a".to_string(),
            priority: 10,
            adapters: vec!["mod.a".to_string()],
            engine: DEFAULT_ENGINE.to_string(),
        }]);
        assert_eq!(registry.list_bundles(), vec!["a".to_string()]);
        assert!(registry.get_bundle_info("b").is_none());
    }
}
